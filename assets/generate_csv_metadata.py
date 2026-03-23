"""
generate_csv_metadata.py
===============
Parse one or several NeMo-style YAML configs referencing JSONL manifests
and produce a single metadata CSV.

Usage
-----
# Single YAML
python generate_csv_metadata.py config.yaml --output_dir ./out --data_root /data

# Multiple YAMLs (results are concatenated, a 'yaml_source' column is added)
python generate_csv_metadata.py train.yaml eval.yaml --output_dir ./out --data_root /data

# Skip manifests that don't exist on disk (useful for dry-runs)
python generate_csv_metadata.py config.yaml --skip_missing
"""

import os
import sys
import json
import math
import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def seconds_to_dhms(total_seconds: float) -> str:
    total_seconds = int(total_seconds)
    d, rem = divmod(total_seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s   = divmod(rem, 60)
    return f"{d}d {h:02d}h {m:02d}m {s:02d}s"


def word_count(text) -> int:
    if not text or not isinstance(text, str):
        return 0
    return len(text.split())


def _infer_lang_from_path(path_str: str) -> str:
    if not path_str:
        return ""
    codes = {"en", "fr", "ar", "de", "es", "it", "nl", "pt", "ru", "zh"}
    for part in str(path_str).split("/"):
        if part.strip().lower() in codes:
            return part.strip().lower()
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# YAML flattening
# ──────────────────────────────────────────────────────────────────────────────

def flatten_manifests(yaml_path: str) -> list:
    """
    Walk the YAML tree and return one dict per multimodal_conversation entry.
    Preserves the full tag hierarchy (task, lang, source_lang, target_lang).
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    entries = []

    def count_conversations(cfg_list: list) -> int:
        return sum(
            1 for c in cfg_list
            if isinstance(c, dict) and c.get("type") == "multimodal_conversation"
        )

    def recurse(node,
                inherited_task="", inherited_lang="", inherited_subtask="",
                inherited_weight=1.0, group_weight=None,
                inherited_src_lang="", inherited_tgt_lang=""):
        if not isinstance(node, dict):
            return

        tags = node.get("tags", {}) or {}
        if isinstance(tags, str):
            tags = {}

        task     = tags.get("task",        inherited_task)
        lang     = tags.get("lang",        inherited_lang)
        if not lang:
            lang = _infer_lang_from_path(node.get("manifest_filepath", ""))
        sub_task = tags.get("sub_task",    inherited_subtask)
        src_lang = tags.get("source_lang", inherited_src_lang)
        tgt_lang = tags.get("target_lang", inherited_tgt_lang)
        weight   = float(node.get("weight", inherited_weight) or inherited_weight)
        ntype    = node.get("type", "")

        if ntype == "multimodal_conversation":
            entries.append({
                "manifest_filepath": node.get("manifest_filepath", ""),
                "raw_manifest_path": node.get("manifest_filepath", ""),
                "task_type":         task,
                "sub_task":          sub_task,
                "language":          lang,
                "source_lang":       src_lang,
                "target_lang":       tgt_lang,
                "weight_dataset":    None,
                "weight_group":      group_weight if group_weight is not None else weight,
            })
            return

        if ntype == "group":
            children   = node.get("input_cfg", [])
            n_datasets = count_conversations(children)
            start_idx  = len(entries)
            for child in children:
                recurse(child, task, lang, sub_task, weight,
                        group_weight=weight,
                        inherited_src_lang=src_lang,
                        inherited_tgt_lang=tgt_lang)
            if n_datasets > 0:
                for entry in entries[start_idx:]:
                    if entry["weight_dataset"] is None:
                        entry["weight_dataset"] = round(1.0 / n_datasets, 6)

    for top in cfg.get("input_cfg", []):
        recurse(top)

    for entry in entries:
        if entry["weight_dataset"] is None:
            entry["weight_dataset"] = 1.0

    return entries


# ──────────────────────────────────────────────────────────────────────────────
# JSONL parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_manifest(manifest_paths: list,
                   task_type: str, sub_task: str,
                   language: str, source_lang: str, target_lang: str,
                   dataset_name: str, split: str, note: str) -> dict | None:

    durations, instruction_wc, response_wc = [], [], []
    speaker_ids, sampling_rates, channels_list = set(), [], []
    num_samples  = 0
    num_segments = 0
    paths_used   = []

    for mpath in manifest_paths:
        p = Path(mpath)
        if not p.exists():
            continue
        paths_used.append(mpath)

        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                num_samples += 1

                # duration — check multiple locations
                dur = rec.get("duration") or rec.get("audio_duration") or rec.get("dur")

                if dur is None:
                    for turn in rec.get("conversations", rec.get("conversation", [])):
                        if not isinstance(turn, dict):
                            continue
                        if str(turn.get("type", "")).lower() == "audio":
                            dur = turn.get("duration") or turn.get("audio_duration")
                            if dur is not None:
                                break
                        val = str(turn.get("value", ""))
                        if any(val.endswith(e) for e in (".wav",".mp3",".flac",".ogg",".opus",".m4a")):
                            dur = turn.get("duration") or turn.get("audio_duration")
                            if dur is not None:
                                break

                for loc in [
                    rec.get("context") or {},
                    (rec.get("input_values") or {}).get("audio") or {},
                    rec.get("audio") or {},
                ]:
                    if dur is None and isinstance(loc, dict):
                        dur = loc.get("duration") or loc.get("audio_duration")

                if dur is not None:
                    try:
                        durations.append(float(dur))
                        num_segments += 1
                    except (ValueError, TypeError):
                        pass

                # sample rate / channels / speaker
                ctx = rec.get("context") or {}
                sr  = (rec.get("sample_rate") or rec.get("sampling_rate")
                       or (ctx.get("sample_rate") if isinstance(ctx, dict) else None))
                if sr:
                    try: sampling_rates.append(int(sr))
                    except (ValueError, TypeError): pass

                ch = (rec.get("num_channels") or rec.get("channels")
                      or (ctx.get("num_channels") if isinstance(ctx, dict) else None))
                if ch:
                    try: channels_list.append(int(ch))
                    except (ValueError, TypeError): pass

                spk = (rec.get("speaker") or rec.get("speaker_id")
                       or rec.get("speaker_id_str")
                       or (ctx.get("speaker_id") if isinstance(ctx, dict) else None))
                if spk:
                    speaker_ids.add(str(spk))

                # instruction / response word counts
                conversations = rec.get("conversations", rec.get("conversation", []))
                if isinstance(conversations, list) and conversations:
                    for turn in conversations:
                        role  = turn.get("role", turn.get("from", "")).lower()
                        value = turn.get("value", turn.get("content", turn.get("text", "")))
                        wc    = word_count(value)
                        if role in ("user", "human", "instruction", "input"):
                            instruction_wc.append(wc)
                        elif role in ("assistant", "gpt", "system", "output", "response", "bot"):
                            response_wc.append(wc)
                else:
                    for key in ("question", "instruction", "input"):
                        v = rec.get(key)
                        if v:
                            instruction_wc.append(word_count(str(v))); break
                    for key in ("answer", "response", "output", "text"):
                        v = rec.get(key)
                        if v:
                            response_wc.append(word_count(str(v))); break

    if num_samples == 0:
        return None

    def ss(arr, fn):
        return round(fn(arr), 3) if arr else None

    return {
        "dataset_name":                dataset_name,
        "split":                       split,
        "note":                        note,
        "task_type":                   task_type,
        "sub_task":                    sub_task,
        "language":                    language,
        "source_lang":                 source_lang,
        "target_lang":                 target_lang,
        "num_audio_segments":          num_segments,
        "num_samples":                 num_samples,
        "total_duration_sec":          round(sum(durations), 2) if durations else None,
        "total_duration_dhms":         seconds_to_dhms(sum(durations)) if durations else None,
        "min_segment_duration_sec":    ss(durations, min),
        "max_segment_duration_sec":    ss(durations, max),
        "avg_segment_duration_sec":    ss(durations, lambda x: sum(x)/len(x)),
        "median_segment_duration_sec": ss(durations, lambda x: float(np.median(x))),
        "std_segment_duration_sec":    ss(durations, lambda x: float(np.std(x))),
        "avg_instruction_words":       ss(instruction_wc, lambda x: sum(x)/len(x)),
        "min_instruction_words":       ss(instruction_wc, min),
        "max_instruction_words":       ss(instruction_wc, max),
        "avg_response_words":          ss(response_wc, lambda x: sum(x)/len(x)),
        "min_response_words":          ss(response_wc, min),
        "max_response_words":          ss(response_wc, max),
        "num_unique_speakers":         len(speaker_ids) if speaker_ids else None,
        "avg_audio_sampling_rate":     round(sum(sampling_rates)/len(sampling_rates)) if sampling_rates else None,
        "avg_audio_channels":          round(sum(channels_list)/len(channels_list), 2) if channels_list else None,
        "path":                        " | ".join(paths_used),
    }


def _worker(args_tuple):
    key, paths, e_first = args_tuple
    stats = parse_manifest(
        paths, key[3], key[4], key[5], key[6], key[7], key[0], key[1], key[2]
    )
    base = {
        "dataset_name":    key[0], "split":        key[1], "note":         key[2],
        "task_type":       key[3], "sub_task":      key[4], "language":     key[5],
        "source_lang":     key[6], "target_lang":   key[7],
        "weight_dataset":  e_first["weight_dataset"] or 1.0,
        "weight_group":    e_first.get("weight_group", e_first["weight_dataset"]) or 1.0,
        "raw_manifest_path": key[8],
        "yaml_source":     key[9],
    }
    if stats is None:
        base.update({k: None for k in [
            "num_audio_segments","num_samples","total_duration_sec","total_duration_dhms",
            "min_segment_duration_sec","max_segment_duration_sec","avg_segment_duration_sec",
            "median_segment_duration_sec","std_segment_duration_sec",
            "avg_instruction_words","min_instruction_words","max_instruction_words",
            "avg_response_words","min_response_words","max_response_words",
            "num_unique_speakers","avg_audio_sampling_rate","avg_audio_channels",
        ]})
        base["path"] = " | ".join(paths)
        base["file_exists"] = False
    else:
        stats.update(base)
        stats["file_exists"] = True
        base = stats
    return base, bool(stats), paths


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

COL_ORDER = [
    "yaml_source",
    "dataset_name","split","note","task_type","sub_task","language",
    "source_lang","target_lang","num_audio_segments","num_samples",
    "total_duration_sec","total_duration_dhms",
    "min_segment_duration_sec","max_segment_duration_sec",
    "avg_segment_duration_sec","median_segment_duration_sec","std_segment_duration_sec",
    "avg_instruction_words","min_instruction_words","max_instruction_words",
    "avg_response_words","min_response_words","max_response_words",
    "num_unique_speakers","avg_audio_sampling_rate","avg_audio_channels",
    "weight_group","weight_dataset","file_exists","raw_manifest_path","path",
]


def main():
    parser = argparse.ArgumentParser(
        description="Parse NeMo YAML(s) → metadata CSV.")
    parser.add_argument("yaml_paths", nargs="+",
                        help="One or more YAML config files.")
    parser.add_argument("--output_dir",   default=None,
                        help="Output directory (default: next to first YAML).")
    parser.add_argument("--output_csv",   default="metadata.csv",
                        help="Output CSV filename (default: metadata.csv).")
    parser.add_argument("--data_root",    default="",
                        help="Root path to replace ${oc.env:DATA_FOLDER}.")
    parser.add_argument("--skip_missing", action="store_true",
                        help="Ignore manifests that don't exist on disk.")
    parser.add_argument("--workers",      type=int, default=8,
                        help="Parallel workers (default: 8).")
    args = parser.parse_args()

    first_yaml = Path(args.yaml_paths[0])
    out_dir    = Path(args.output_dir) if args.output_dir else first_yaml.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    DATA_FOLDER = args.data_root or os.environ.get("DATA_FOLDER", "")

    all_tasks = []   # (key, paths, e_first)

    for yaml_path in args.yaml_paths:
        yaml_name = Path(yaml_path).stem
        print(f"\n📂 Parsing YAML: {yaml_path}")
        entries = flatten_manifests(yaml_path)
        print(f"   Found {len(entries)} manifest entries.")

        groups: dict = {}
        for e in entries:
            path = e["manifest_filepath"]
            if DATA_FOLDER:
                path = path.replace("${oc.env:DATA_FOLDER}", DATA_FOLDER)
            p    = Path(path)
            name = p.parent.name
            stem = p.stem
            split = "train" if stem.startswith("train") else stem
            note  = []
            if "recasepunc" in stem: note.append("with punctuations")
            if "max30"      in stem: note.append("max duration is 30s")
            e.update({
                "path":             path,
                "dataset_name":     name,
                "split":            split,
                "note":             ", ".join(note),
                "raw_manifest_path": e.get("raw_manifest_path", path),
                "yaml_source":      yaml_name,
            })
            key = (
                name, split, ", ".join(note),
                e["task_type"], e["sub_task"],
                e["language"], e["source_lang"], e["target_lang"],
                e.get("raw_manifest_path", ""), yaml_name,
            )
            groups.setdefault(key, []).append(e)

        print(f"   Aggregated into {len(groups)} distinct groups.")
        for key, group_entries in groups.items():
            paths   = [ge["path"] for ge in group_entries]
            e_first = group_entries[0]
            all_tasks.append((key, paths, e_first))

    print(f"\n   Total groups to parse: {len(all_tasks)} (workers={args.workers})")

    parsed_rows = []
    missing     = 0
    total       = len(all_tasks)

    try:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_worker, t): t for t in all_tasks}
            for i, future in enumerate(as_completed(futures)):
                base, success, paths = future.result()
                if not success:
                    missing += len(paths)
                    if not args.skip_missing:
                        key = futures[future][0]
                        raise RuntimeError(
                            f"Missing manifests for {key[0]} {key[1]}: {paths}"
                        )
                parsed_rows.append(base)
                pct = (i + 1) / total * 100
                print(f"\r  [{i+1}/{total}] {pct:3.0f}% | {base['dataset_name'][:35]:<35}",
                      end="", flush=True)
    except KeyboardInterrupt:
        print("\n\n❗ Interrupted."); sys.exit(1)

    print(f"\n\n   Found: {total - missing}  |  Missing: {missing}")

    df = pd.DataFrame(parsed_rows)
    df = df[[c for c in COL_ORDER if c in df.columns]]

    csv_path = out_dir / args.output_csv
    df.to_csv(csv_path, index=False)

    # Quick summary
    print("\n" + "="*60)
    total_h = df["total_duration_sec"].sum(skipna=True) / 3600
    print(f"  Rows  : {len(df)}")
    print(f"  YAMLs : {len(args.yaml_paths)}")
    print(f"  Hours : {total_h:,.1f} h")
    for task, sub in df.groupby("task_type"):
        print(f"    {task:<20} {int(sub['num_samples'].sum()):>10,} samples  "
              f"{sub['total_duration_sec'].sum(skipna=True)/3600:>8.1f} h")
    print("="*60)
    print(f"\n✅ CSV → {csv_path}\n")


if __name__ == "__main__":
    main()