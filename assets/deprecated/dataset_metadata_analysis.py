"""
Dataset Metadata Extractor & Visualizer
Parses a NeMo-style YAML config referencing JSONL manifests,
computes per-dataset statistics, produces a Markdown README report and plots.

Weight philosophy (updated)
---------------------------
Weights are **data-driven** (natural weights): each dataset's sampling
probability is proportional to its relative size (duration or sample count)
within its group, with optional temperature smoothing to avoid extreme
imbalances.

Three-level hierarchy mirroring the NeMo YAML structure:

  Level 1 — Task weight      sums to 1.0 globally
  Level 2 — Language weight  sums to 1.0 within each task
  Level 3 — Dataset weight   sums to 1.0 within each (task, language) group

Effective probability = task_w × lang_w × dataset_w

Temperature T controls smoothing:
  T = 1.0  → fully proportional to size (large datasets dominate)
  T = 2.0  → moderate levelling (default)
  T → ∞   → uniform within each group (ignores size)
"""

import os
import sys
import re
import logging
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import seaborn as sns

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Visual encoding
# ──────────────────────────────────────────────────────────────────────────────

LANG_COLORS: dict = {
    "fr":      "#002395",
    "en":      "#C8102E",
    "es":      "#AA151B",
    "de":      "#444444",
    "it":      "#009246",
    "nl":      "#FF6600",
    "pt":      "#006600",
    "ar":      "#007A3D",
    "mixed":   "#AAAAAA",
    "unknown": "#CCCCCC",
    "":        "#CCCCCC",
}

TASK_HATCHES: dict = {
    "asr":              "",
    "ast":              "///",
    "qa":               "...",
    "aqa":              "|||",
    "other":            "ooo",
    "audio_captioning": "---",
    "music_captioning": "***",
    "mqa":              "xxx",
}


def lang_color(lang: str) -> str:
    return LANG_COLORS.get(str(lang).lower(), "#888888")


def task_hatch(task: str) -> str:
    return TASK_HATCHES.get(str(task).lower(), "")


# ──────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────────────────────────────────

def seconds_to_dhms(total_seconds: float) -> str:
    total_seconds = int(total_seconds)
    d, rem = divmod(total_seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s   = divmod(rem, 60)
    return f"{d}d {h:02d}h {m:02d}m {s:02d}s"


def fmt_num(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    v = int(v)
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v / 1_000:.1f}K"
    return str(v)


def fmt_hours(secs) -> str:
    if secs is None or (isinstance(secs, float) and math.isnan(secs)):
        return "—"
    h = secs / 3600
    return f"{h:,.0f} h" if h >= 1000 else f"{h:.1f} h"


def word_count(text) -> int:
    if not text or not isinstance(text, str):
        return 0
    return len(text.split())


# YAML flattening
# ──────────────────────────────────────────────────────────────────────────────

def _infer_lang_from_path(path_str):
    """Fallback: try to extract a common language code from the path if not tagged."""
    if not path_str: return ""
    # Common codes we expect to see as standalone path segments
    codes = {"en", "fr", "ar", "de", "es", "it", "nl", "pt", "ru", "zh"}
    parts = str(path_str).split("/")
    for p in parts:
        p_clean = p.strip().lower()
        if p_clean in codes:
            return p_clean
    return ""

def flatten_manifests(yaml_path: str) -> list:
    """
    Walk the YAML tree and return one dict per multimodal_conversation entry.

    weight_group   = weight of the nearest parent group node
    weight_dataset = 1/N  (N = number of conversation siblings in that group)
                     → equal share for each dataset within the group,
                       which is exactly NeMo's default behaviour.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if isinstance(cfg, list):
        cfg = {"input_cfg": cfg}

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
            # Fallback to path inference if no tag is found
            mf_path = node.get("manifest_filepath", "")
            lang = _infer_lang_from_path(mf_path)

        sub_task = tags.get("sub_task",    inherited_subtask)
        src_lang = tags.get("source_lang", inherited_src_lang)
        tgt_lang = tags.get("target_lang", inherited_tgt_lang)
        weight   = float(node.get("weight", inherited_weight) or inherited_weight)

        ntype = node.get("type", "")

        if ntype == "multimodal_conversation":
            entries.append({
                "manifest_filepath": node.get("manifest_filepath", ""),
                "raw_manifest_path": node.get("manifest_filepath", ""),  # Preserve original
                "task_type":         task,
                "sub_task":          sub_task,
                "language":          lang,
                "source_lang":       src_lang,
                "target_lang":       tgt_lang,
                "weight_dataset":    None,          # filled in by parent group
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

            # assign 1/N to every direct conversation child just added
            if n_datasets > 0:
                for entry in entries[start_idx:]:
                    if entry["weight_dataset"] is None:
                        entry["weight_dataset"] = round(1.0 / n_datasets, 3)

    for top in cfg.get("input_cfg", []):
        recurse(top)

    # top-level conversations without a group parent (default to 1.0)
    for entry in entries:
        if entry["weight_dataset"] is None:
            entry["weight_dataset"] = 1.0

    return entries


# ──────────────────────────────────────────────────────────────────────────────
# JSONL parsing
# ──────────────────────────────────────────────────────────────────────────────

def _worker_parse_group(args_tuple):
    """
    Worker function for parallel manifest parsing.
    args_tuple: (key, paths, e_first)
    """
    key, paths, e_first = args_tuple
    # parse_manifest(paths, task, sub_task, lang, src_lang, tgt_lang, name, split, note)
    stats = parse_manifest(paths, key[3], key[4], key[5], key[6], key[7],
                           key[0], key[1], key[2])

    base = {
        "dataset_name": key[0], "split": key[1], "note": key[2],
        "task_type": key[3],    "sub_task": key[4], "language": key[5],
        "source_lang": key[6],  "target_lang": key[7],
        "weight_dataset": e_first["weight_dataset"] or 1.0,
        "weight_group":   e_first.get("weight_group", e_first["weight_dataset"]) or 1.0,
        "raw_manifest_path": key[8],
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
        base["path"] = " | ".join(paths); base["file_exists"] = False
    else:
        stats.update(base); stats["file_exists"] = True
        base = stats
    return base, bool(stats), paths


def parse_manifest(manifest_paths: list,
                   task_type: str, sub_task: str,
                   language: str, source_lang: str, target_lang: str,
                   dataset_name: str, split: str, note: str):
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

                # ── duration lookup (multiple possible locations) ─────────────
                dur = rec.get("duration") or rec.get("audio_duration") or rec.get("dur")

                if dur is None:
                    convs = rec.get("conversations") or rec.get("conversation") or []
                    if isinstance(convs, list):
                        for turn in convs:
                            if not isinstance(turn, dict):
                                continue
                            if str(turn.get("type", "")).lower() == "audio":
                                d = turn.get("duration") or turn.get("audio_duration")
                                if d is not None:
                                    dur = d; break
                            val = str(turn.get("value", ""))
                            if any(val.endswith(e) for e in (".wav",".mp3",".flac",".ogg",".opus",".m4a")):
                                d = turn.get("duration") or turn.get("audio_duration")
                                if d is not None:
                                    dur = d; break

                if dur is None:
                    ctx = rec.get("context") or {}
                    if isinstance(ctx, dict):
                        dur = ctx.get("duration") or ctx.get("audio_duration")

                if dur is None:
                    iv = rec.get("input_values") or {}
                    if isinstance(iv, dict):
                        ao = iv.get("audio") or {}
                        if isinstance(ao, dict):
                            dur = ao.get("duration")

                if dur is None:
                    ao = rec.get("audio") or {}
                    if isinstance(ao, dict):
                        dur = ao.get("duration") or ao.get("audio_duration")

                if dur is not None:
                    try:
                        durations.append(float(dur)); num_segments += 1
                    except (ValueError, TypeError):
                        pass

                # ── sample rate, channels, speaker ────────────────────────────
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

                # ── instruction / response word counts ────────────────────────
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
        "total_duration_sec":          round(sum(durations), 2)    if durations else None,
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


# ──────────────────────────────────────────────────────────────────────────────
# Markdown table helper
# ──────────────────────────────────────────────────────────────────────────────

def _md_table(headers, rows) -> str:
    if not rows:
        return "_No data._"
    widths = [
        max(len(str(h)), max(len(str(r[i])) for r in rows))
        for i, h in enumerate(headers)
    ]
    sep  = "| " + " | ".join("-" * w for w in widths) + " |"
    head = "| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    body = "\n".join(
        "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    )
    return "\n".join([head, sep, body])


# ──────────────────────────────────────────────────────────────────────────────
# Per-task detail section
# ──────────────────────────────────────────────────────────────────────────────

def _task_section(df: pd.DataFrame, task: str,
                  total_samples: int, total_dur_sec: float) -> list:
    sub = df[df["task_type"] == task].copy()

    def _lang(row):
        lang = str(row.get("language", "") or "").strip()
        if task == "ast":
            src = str(row.get("source_lang", "") or "").strip()
            tgt = str(row.get("target_lang", "") or "").strip()
            if src and tgt:
                return f"{src}→{tgt}"
            return src or tgt or "unknown"
        return lang or "unknown"

    sub["_lang"] = sub.apply(_lang, axis=1)
    sub = sub.sort_values(["_lang", "dataset_name", "split"])

    rows = []
    for _, r in sub.iterrows():
        n_samp  = r["num_samples"]
        dur_sec = r["total_duration_sec"]
        pct     = f"{100 * n_samp / total_samples:.2f}%" if (n_samp and total_samples) else "—"
        avg_seg = r.get("avg_segment_duration_sec")
        rows.append([
            r["dataset_name"], r["split"], r.get("note", ""), r["_lang"],
            fmt_num(n_samp), pct, fmt_hours(dur_sec),
            f"{avg_seg:.1f} s" if avg_seg else "—",
        ])

    t_samp = sub["num_samples"].sum()
    t_dur  = sub["total_duration_sec"].sum(skipna=True)
    t_pct  = f"{100 * t_samp / total_samples:.2f}%" if total_samples else "—"
    rows.append([f"**TOTAL ({task.upper()})**", "", "", "",
                 f"**{fmt_num(t_samp)}**", f"**{t_pct}**",
                 f"**{fmt_hours(t_dur)}**", ""])

    lines = [
        f"### {task.upper()}", "",
        f"> {len(sub)} manifests &nbsp;·&nbsp; "
        f"{sub['dataset_name'].nunique()} unique datasets &nbsp;·&nbsp; "
        f"{sub['_lang'].nunique()} language(s)", "",
    ]
    lines.append(_md_table(
        ["Dataset", "Split", "Note", "Language", "Samples", "% Total", "Duration", "Avg Seg"],
        rows,
    ))
    lines.append("")
    return lines


# ──────────────────────────────────────────────────────────────────────────────
# Natural weight engine  (THE core function)
# ──────────────────────────────────────────────────────────────────────────────

_EPSILON     = 1e-9
_TEMPERATURE = 2.0


def _lang_key(row) -> str:
    """Consistent language key used across all weight functions."""
    task = str(row.get("task_type", "") or "").strip()
    lang = str(row.get("language",  "") or "").strip()
    if task == "ast":
        src = str(row.get("source_lang", "") or "").strip()
        tgt = str(row.get("target_lang", "") or "").strip()
        if src and tgt:
            return f"{src}→{tgt}"
        return src or tgt or "unknown"
    return lang or "unknown"


def compute_natural_weights(
    df: pd.DataFrame,
    metric: str = "duration",
    temperature: float = _TEMPERATURE,
) -> tuple:
    """
    Compute data-driven sampling weights at all three YAML hierarchy levels.

    Parameters
    ----------
    df          : DataFrame produced by the main parsing loop.
    metric      : "duration" → weight ∝ total_duration_sec
                  "samples"  → weight ∝ num_samples
                  If the primary metric is missing/zero, falls back to the other.
    temperature : Smoothing T > 0.
                  score = size ^ (1/T)
                  T=1 → proportional | T=2 → moderate | T→∞ → uniform

    Returns
    -------
    (df_w, tl, task_agg, task_w_map)
      df_w      : input df with weight columns added (one row per dataset)
      tl        : (task, lang) aggregation table
      task_agg  : task-level aggregation table
      task_w_map: {task_type: task_w}  dict
    """
    if metric not in ("duration", "samples"):
        raise ValueError(f"metric must be 'duration' or 'samples', got {metric!r}")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    df = df.copy()
    df["_lkey"] = df.apply(_lang_key, axis=1)

    # ── Size metric with fallback ─────────────────────────────────────────────
    def _size(row):
        dur = row.get("total_duration_sec")
        n   = row.get("num_samples")
        dur_ok = dur is not None and not (isinstance(dur, float) and math.isnan(dur)) and float(dur) > 0
        n_ok   = n   is not None and not (isinstance(n,   float) and math.isnan(n))   and float(n)   > 0
        if metric == "duration":
            return float(dur) if dur_ok else (float(n) if n_ok else _EPSILON)
        else:
            return float(n) if n_ok else (float(dur) if dur_ok else _EPSILON)

    df["size_metric"] = df.apply(_size, axis=1)
    df["raw_score"]   = df["size_metric"].apply(lambda x: x ** (1.0 / temperature))

    # ── Level 3: Dataset weight  (within task × lang) ─────────────────────────
    grp_sum       = df.groupby(["task_type", "_lkey"])["raw_score"].transform("sum")
    df["dataset_w"] = df["raw_score"] / grp_sum.replace(0, _EPSILON)

    # ── Aggregate to (task, lang) ─────────────────────────────────────────────
    tl = (
        df.groupby(["task_type", "_lkey"])
          .agg(
              n_datasets    = ("dataset_name",      "nunique"),
              total_dur_h   = ("total_duration_sec", lambda x: x.sum(skipna=True) / 3600),
              total_samples = ("num_samples",        "sum"),
              group_score   = ("raw_score",          "sum"),
          )
          .reset_index()
    )

    # ── Level 2: Language weight  (within task) ───────────────────────────────
    task_score_sum = tl.groupby("task_type")["group_score"].transform("sum")
    tl["lang_w"]   = tl["group_score"] / task_score_sum.replace(0, _EPSILON)

    # ── Level 1: Task weight  (global) ───────────────────────────────────────
    task_agg = (
        tl.groupby("task_type")["group_score"]
          .sum().reset_index()
          .rename(columns={"group_score": "task_score"})
    )
    grand_total        = task_agg["task_score"].sum()
    task_agg["task_w"] = task_agg["task_score"] / max(grand_total, _EPSILON)
    task_w_map         = dict(zip(task_agg["task_type"], task_agg["task_w"]))

    tl["task_w"] = tl["task_type"].map(task_w_map)

    # ── Merge back ────────────────────────────────────────────────────────────
    df = df.merge(
        tl[["task_type", "_lkey", "lang_w", "task_w"]],
        on=["task_type", "_lkey"],
        how="left",
    )

    # ── Effective probability ─────────────────────────────────────────────────
    df["effective_prob"]     = df["task_w"] * df["lang_w"] * df["dataset_w"]
    df["effective_prob_pct"] = df["effective_prob"] * 100.0

    # Aliases used by downstream reporting functions
    df["sampling_prob"]     = df["effective_prob"]
    df["sampling_prob_pct"] = df["effective_prob_pct"]

    # ── Est. passes per epoch ─────────────────────────────────────────────────
    total_n = float(df["num_samples"].sum(skipna=True)) or 1.0
    df["expected_passes_per_epoch"] = (
        (df["sampling_prob"] * total_n) / df["num_samples"].clip(lower=1)
    ).round(2)

    # Compatibility aliases for older reporting code
    df["weight_group"]        = df["task_w"] * df["lang_w"]
    df["weight_dataset"]      = df["dataset_w"]
    df["raw_group_weight"]    = df["weight_group"]
    df["effective_weight"]    = df["effective_prob"]
    df["dataset_effective_w"] = df["effective_prob"]
    df["_dl"]                 = df["_lkey"]

    return df, tl, task_agg, task_w_map


def compute_sampling_weights(
    df: pd.DataFrame,
    metric: str = "duration",
    temperature: float = _TEMPERATURE,
    min_weight: float = 0.0,
) -> pd.DataFrame:
    """Convenience wrapper — returns only the annotated df."""
    df_w, _, _, _ = compute_natural_weights(df, metric=metric, temperature=temperature)
    if min_weight > 0 and "effective_prob" in df_w.columns:
        df_w["effective_prob"] = df_w["effective_prob"].clip(lower=min_weight)
        df_w["effective_prob"] = df_w["effective_prob"] / df_w["effective_prob"].sum()
        df_w["sampling_prob_pct"] = df_w["effective_prob"] * 100.0
    return df_w


# ──────────────────────────────────────────────────────────────────────────────
# Section 6.5 — Actual sampling weights report
# ──────────────────────────────────────────────────────────────────────────────

def _sampling_weight_section(
    df_w: pd.DataFrame,
    metric: str = "duration",
    temperature: float = _TEMPERATURE,
) -> list:
    lines = [
        "## 6.5 Actual Sampling Weights",
        "",
        "> Probabilities computed from **natural (data-driven) weights**: "
        f"`size ^ (1/T)` with metric=**{metric}**, T=**{temperature:.1f}**.",
        "",
        "### Methodology",
        "",
        "1. **Size metric** per dataset: "
        + ("total duration (seconds)" if metric == "duration" else "number of samples")
        + " — fallback to the other if missing.",
        f"2. **Score** = `size ^ (1/{temperature:.1f})` (temperature smoothing).",
        "3. **Dataset weight (L3)** = `score_i / Σ scores` within the (task, lang) group.",
        "4. **Language weight (L2)** = `Σ scores in lang / Σ scores in task`.",
        "5. **Task weight (L1)** = `Σ scores in task / Σ all scores`.",
        "6. **Effective probability** = `task_w × lang_w × dataset_w` → sums to 100%.",
        "7. **Est. passes/epoch** = `(prob × total_samples) / dataset_samples`.",
        "",
        "---",
        "",
    ]

    df_w = df_w.copy()

    # Summary table
    agg = (
        df_w.groupby(["task_type", "_dl"])
            .agg(
                n_datasets   = ("dataset_name",      "nunique"),
                samples      = ("num_samples",        "sum"),
                duration_h   = ("total_duration_sec", lambda x: x.sum(skipna=True) / 3600),
                task_w       = ("task_w",             "first"),
                lang_w       = ("lang_w",             "first"),
                sampling_pct = ("sampling_prob_pct",  "sum"),
            )
            .reset_index()
            .sort_values(["task_type", "sampling_pct"], ascending=[True, False])
    )

    lines += ["### Summary: sampling probability by Task × Language", "",
              "> `Sampling %` sums to 100% across the whole table.", ""]

    agg_rows = []
    for _, r in agg.iterrows():
        agg_rows.append([
            r["task_type"], r["_dl"], str(int(r["n_datasets"])),
            fmt_num(r["samples"]), f"{r['duration_h']:.1f} h",
            f"{r['task_w']:.4f}", f"{r['lang_w']:.4f}",
            f"{r['sampling_pct']:.3f}%",
        ])
    agg_rows.append([
        "**TOTAL**", "", "",
        f"**{fmt_num(int(df_w['num_samples'].sum()))}**",
        f"**{fmt_hours(df_w['total_duration_sec'].sum(skipna=True))}**",
        "", "",
        f"**{df_w['sampling_prob_pct'].sum():.1f}%**",
    ])
    lines.append(_md_table(
        ["Task", "Language", "Datasets", "Samples", "Duration",
         "Task w (L1)", "Lang w (L2)", "Sampling %"],
        agg_rows,
    ))
    lines += ["", "---", "", "### Per-task detail", ""]

    size_col = "Duration" if metric == "duration" else "Samples"

    for task in sorted(df_w["task_type"].dropna().unique()):
        sub = df_w[df_w["task_type"] == task].copy()
        sub = sub.sort_values(["_dl", "sampling_prob_pct"], ascending=[True, False])
        t_prob = sub["sampling_prob_pct"].sum()

        lines += [f"#### {task.upper()}", "",
                  f"> {len(sub)} datasets &nbsp;·&nbsp; share: **{t_prob:.3f}%**", ""]

        detail_rows = []
        for _, r in sub.iterrows():
            size_val = (fmt_hours(r.get("total_duration_sec"))
                        if metric == "duration" else fmt_num(r.get("num_samples")))
            ep   = r.get("expected_passes_per_epoch", "—")
            flag = (" ⚠ over-sampled"   if isinstance(ep, (int, float)) and ep > 5
                    else " ⚑ under-sampled" if isinstance(ep, (int, float)) and ep < 0.1
                    else "")
            detail_rows.append([
                r["dataset_name"] + flag, r["_dl"],
                fmt_num(r.get("num_samples")), size_val,
                f"{r['task_w']:.4f}", f"{r['lang_w']:.4f}", f"{r['dataset_w']:.6f}",
                f"{float(r['sampling_prob_pct']):.4f}%",
                f"{ep}×" if ep != "—" else "—",
            ])
        detail_rows.append([
            f"**TOTAL {task.upper()}**", "", "", "", "", "", "",
            f"**{t_prob:.3f}%**", "",
        ])
        lines.append(_md_table(
            ["Dataset", "Language", "Samples", size_col,
             "Task w (L1)", "Lang w (L2)", "Dataset w (L3)",
             "Sampling %", "Est. passes"],
            detail_rows,
        ))
        lines.append("")

    # Balance observations
    lines += ["---", "", "### Balance observations", ""]
    obs = []

    over = df_w[df_w["expected_passes_per_epoch"] > 5].sort_values(
        "expected_passes_per_epoch", ascending=False)
    if not over.empty:
        names = ", ".join(
            f"**{r['dataset_name']}** ({r['_dl']}, {r['expected_passes_per_epoch']:.1f}×)"
            for _, r in over.head(5).iterrows()
        )
        obs.append(f"⚠️  **Over-sampled** (> 5 passes/epoch — overfitting risk): {names}.")

    under = df_w[df_w["expected_passes_per_epoch"] < 0.1]
    if not under.empty:
        names = ", ".join(
            f"**{r['dataset_name']}** ({r['_dl']}, {r['expected_passes_per_epoch']:.3f}×)"
            for _, r in under.head(5).iterrows()
        )
        obs.append(f"⚑  **Under-sampled** (< 0.1 passes/epoch — wasted data): {names}.")

    top1 = df_w.nlargest(1, "sampling_prob_pct").iloc[0]
    obs.append(
        f"📌  Most sampled: **{top1['dataset_name']}** ({top1['_dl']}) "
        f"at **{top1['sampling_prob_pct']:.3f}%** of steps."
    )

    for note in obs:
        lines.append(f"- {note}")
    lines.append("")
    return lines


# ──────────────────────────────────────────────────────────────────────────────
# Section 8 — Suggested weights (same engine, different T / metric possible)
# ──────────────────────────────────────────────────────────────────────────────

def suggest_weights_section(df: pd.DataFrame,
                            metric: str = "duration",
                            temperature: float = _TEMPERATURE,
                            min_weight: float = 0.0) -> list:
    """
    Section 8: produce the recommended YAML weights at all three levels.
    Uses compute_natural_weights() — same formula as Section 6.5.
    """
    df_w = compute_sampling_weights(df, metric=metric, temperature=temperature,
                                    min_weight=min_weight)

    tl = df_w.groupby(["task_type","_lkey"]).agg(
        n_datasets=("dataset_name","count"),
        total_samples=("num_samples","sum"),
        total_dur_h=("total_duration_sec", lambda x: x.sum()/3600),
        task_w=("task_w","first"),
        lang_w=("lang_w","first"),
    ).reset_index()

    task_agg = df_w.groupby("task_type").agg(task_w=("task_w","first")).reset_index()
    task_w_map = dict(zip(task_agg["task_type"], task_agg["task_w"]))

    metric_label = "total duration (seconds)" if metric == "duration" else "number of samples"
    size_col     = "Duration" if metric == "duration" else "Samples"
    tasks        = sorted(tl["task_type"].dropna().unique())
    tl_sorted    = tl.sort_values(["task_type", "lang_w"], ascending=[True, False])

    lines = [
        "## 8. Suggested Sampling Weights",
        "",
        f"> Weights are **data-driven**, proportional to **{metric_label}** "
        f"with temperature smoothing **T={temperature:.1f}**.",
        "",
        "### Weighting formula",
        "",
        "| Level | Scope | Put on node | Formula |",
        "|-------|-------|-------------|---------|",
        "| **L1 — Task**     | global           | task `group`     | `Σsize_task / Σsize_all` |",
        "| **L2 — Language** | within task      | language `group` | `Σsize_lang / Σsize_task` |",
        "| **L3 — Dataset**  | within task×lang | `multimodal_conversation` | `size_ds / Σsize_group` |",
        "",
        f"where `size = {metric_label} ^ (1/{temperature:.1f})`",
        "",
        "**Temperature T:**",
        "- T=1 → fully proportional (large datasets dominate)",
        f"- T={temperature:.1f} → moderate levelling ← current",
        "- T→∞ → uniform (all datasets equally likely)",
        "",
        "**Effective probability** of any dataset = `task_w × lang_w × dataset_w` → sums to 100%",
        "",
        "---",
        "",
    ]

    # Level 1
    lines += ["### Level 1 — Task weights", "",
              "> Sum = 1.0 across all tasks.", ""]
    t1_rows = []
    for _, r in task_agg.sort_values("task_w", ascending=False).iterrows():
        t_sub = tl[tl["task_type"] == r["task_type"]]
        size_val = (f"{t_sub['total_dur_h'].sum():.1f} h" if metric == "duration"
                    else fmt_num(int(t_sub["total_samples"].sum())))
        t1_rows.append([r["task_type"], str(int(t_sub["n_datasets"].sum())),
                        size_val, f"{r['task_w']:.6f}"])
    t1_rows.append(["**TOTAL**", "", "", f"**{task_agg['task_w'].sum():.4f}**"])
    lines.append(_md_table(["Task", "Datasets", size_col, "Task weight (L1)"], t1_rows))
    lines += ["", "---", ""]

    # Level 2
    lines += ["### Level 2 — Language weights (within each task)", "",
              "> Sum = 1.0 within each task.", ""]
    t2_rows = []
    for _, r in tl_sorted.iterrows():
        size_val = (f"{r['total_dur_h']:.1f} h" if metric == "duration"
                    else fmt_num(int(r["total_samples"])))
        t2_rows.append([
            r["task_type"], r["_lkey"], str(int(r["n_datasets"])), size_val,
            f"{r['task_w']:.4f}", f"{r['lang_w']:.6f}",
            f"{r['task_w'] * r['lang_w'] * 100:.3f}%",
        ])
    lines.append(_md_table(
        ["Task", "Language", "Datasets", size_col,
         "Task w (L1)", "Lang w (L2) ← YAML", "Effective share"],
        t2_rows,
    ))
    lines += ["", "> ✅ **Check**: `Σ effective_share` = 100%.", "", "---", ""]

    # Level 3 (per task)
    lines += ["### Level 3 — Dataset weights (within each Task × Language group)", "",
              "> Sum = 1.0 within each language group.", ""]

    for task in tasks:
        sub = df_w[df_w["task_type"] == task].sort_values(
            ["_lkey", "dataset_w"], ascending=[True, False])
        lines += [f"#### {task.upper()}", ""]
        ds_rows = []
        for _, r in sub.iterrows():
            size_val = (fmt_hours(r.get("total_duration_sec")) if metric == "duration"
                        else fmt_num(r.get("num_samples")))
            ep   = r.get("expected_passes_per_epoch", "—")
            flag = (" ⚠" if isinstance(ep, (int, float)) and ep > 5
                    else " ⚑" if isinstance(ep, (int, float)) and ep < 0.1 else "")
            ds_rows.append([
                r["dataset_name"] + flag, r["_lkey"], size_val,
                f"{r['task_w']:.4f}", f"{r['lang_w']:.4f}",
                f"{r['dataset_w']:.6f}", f"{r['effective_prob_pct']:.4f}%",
                f"{ep}×" if ep != "—" else "—",
            ])
        lines.append(_md_table(
            ["Dataset", "Language", size_col,
             "Task w (L1)", "Lang w (L2)", "Dataset w (L3) ← YAML",
             "Effective prob", "Est. passes/epoch"],
            ds_rows,
        ))
        lines.append("")

    # YAML skeleton
    lines += ["---", "",
              "### YAML skeleton — paste into `input_cfg_train.yaml`", "",
              "> Replace `<path>` with actual manifest paths.", "",
              "```yaml", "input_cfg:"]

    for task in tasks:
        t_rows = tl_sorted[tl_sorted["task_type"] == task]
        tw = task_w_map.get(task, 0.0)
        tw_label = f"{tw:.4f}"
        t_size = (f"{t_rows['total_dur_h'].sum():.1f} h" if metric == "duration"
                  else fmt_num(int(t_rows["total_samples"].sum())))
        lines += [
            f"  - type: group",
            f"    weight: {tw_label}   # L1 — {task.upper()} ({t_size})",
            f"    tags: {{task: {task}}}",
            f"    input_cfg:",
        ]
        for _, lr in t_rows.iterrows():
            lang  = lr["_lkey"]
            lw    = lr["lang_w"]
            lsize = (f"{lr['total_dur_h']:.0f} h" if metric == "duration"
                     else fmt_num(int(lr["total_samples"])))
            lw_label = f"{lw:.4f}"

            # ── Build the tags string correctly for AST vs other tasks ──────
            if task == "ast" and "→" in lang:
                src, tgt = lang.split("→", 1)
                tags_str = f"{{task: {task}, source_lang: {src}, target_lang: {tgt}}}"
            else:
                tags_str = f"{{task: {task}, lang: {lang}}}"

            lines += [
                f"      - type: group",
                f"        weight: {lw_label}   # L2 — {lang} ({lsize})",
                f"        tags: {tags_str}",
                f"        input_cfg:",
            ]
            ds_sub = (df_w[(df_w["task_type"] == task) & (df_w["_lkey"] == lang)]
                      .sort_values("dataset_w", ascending=False))
            for _, dr in ds_sub.iterrows():
                ds_size = (fmt_hours(dr.get("total_duration_sec")) if metric == "duration"
                           else fmt_num(dr.get("num_samples")))
                # Use raw_manifest_path if available, fallback to <path>
                m_path = dr.get("raw_manifest_path", "<path>")
                dw_label = f"{dr['dataset_w']:.4f}" # Changed to 4 decimal places
                if len(ds_sub) > 1:
                    lines += [
                        f"          - type: multimodal_conversation",
                        f"            weight: {dw_label}"
                        f"   # L3 — {dr['dataset_name']} ({ds_size})",
                        f"            manifest_filepath: {m_path}",
                    ]
                else:
                    lines += [
                        f"          - type: multimodal_conversation",
                        f"            # {dr['dataset_name']} ({ds_size})",
                        f"            manifest_filepath: {m_path}",
                    ]
        lines.append("")

    lines += ["```", ""]
    return lines


# ──────────────────────────────────────────────────────────────────────────────
# Section 9 — max_steps recommendation
# ──────────────────────────────────────────────────────────────────────────────

def suggest_max_steps_section(df: pd.DataFrame,
                              batch_size: int = 32,
                              grad_accum: int = 1,
                              target_epochs: float = 1.0) -> list:
    total_samples   = int(df["num_samples"].sum(skipna=True))
    effective_bs    = max(batch_size * grad_accum, 1)
    steps_per_epoch = math.ceil(total_samples / effective_bs)
    target_steps    = math.ceil(steps_per_epoch * target_epochs)

    lines = [
        "## 9. Suggested `max_steps` for Training", "",
        "> `steps_per_epoch` = ⌈total_samples / effective_batch_size⌉.", "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Total training samples | **{total_samples:,}** |",
        f"| Batch size | **{batch_size}** |",
        f"| Gradient accumulation | **{grad_accum}** |",
        f"| Effective batch size | **{effective_bs:,}** |",
        f"| Steps per epoch | **{steps_per_epoch:,}** |",
        f"| Target epochs | **{target_epochs:.1f}** |",
        f"| **Suggested max_steps** | **{target_steps:,}** |",
        "", "---", "", "### Presets", "",
    ]

    preset_rows = []
    for label, ep in [("Conservative", 0.5), ("Recommended", 1.0),
                       ("Extended", 3.0), ("Aggressive", 5.0)]:
        steps  = math.ceil(steps_per_epoch * ep)
        marker = " ← **recommended**" if ep == 1.0 else ""
        preset_rows.append([label, f"{ep:.1f}", f"{steps:,}", marker])
    lines.append(_md_table(["Preset", "Epochs", "max_steps", "Note"], preset_rows))
    lines.append("")

    task_grp = (df.groupby("task_type")
                  .agg(samples=("num_samples", "sum"))
                  .reset_index()
                  .sort_values("samples", ascending=False))
    task_grp["pct"]          = (100 * task_grp["samples"] / max(total_samples, 1)).round(1)
    task_grp["steps_at_1ep"] = (task_grp["samples"] / effective_bs).apply(math.ceil)

    lines += ["### Per-task breakdown", ""]
    tb_rows = [[r["task_type"], fmt_num(r["samples"]),
                f"{r['pct']:.1f}%", f"{r['steps_at_1ep']:,}"]
               for _, r in task_grp.iterrows()]
    tb_rows.append(["**TOTAL**", f"**{fmt_num(total_samples)}**",
                    "**100.0%**", f"**{steps_per_epoch:,}**"])
    lines.append(_md_table(["Task", "Samples", "% of total", "Steps @ 1 epoch"], tb_rows))
    lines.append("")
    return lines


# ──────────────────────────────────────────────────────────────────────────────
# Section 7 — Weight comparison table (Original vs T=1 vs T=2)
# ──────────────────────────────────────────────────────────────────────────────

def _weight_comparison_section(df: pd.DataFrame,
                                metric: str = "duration") -> list:
    """
    Build a Markdown section that shows, for every dataset, three columns
    side-by-side so the user can decide which weighting strategy to use:

      Original  — current YAML weights  (uniform 1/N within each group)
      T=1       — pure proportional to size  (weight ∝ duration or samples)
      T=2       — sqrt-smoothed  (weight ∝ √size, moderate levelling)

    All three produce effective probabilities that sum to 100%.
    """

    def _compute(df_in: pd.DataFrame, T: float) -> pd.Series:
        """Return effective_prob_pct Series for temperature T."""
        d = df_in.copy()
        d["_lkey"] = d.apply(_lang_key, axis=1)

        def _size(row):
            dur = row.get("total_duration_sec")
            n   = row.get("num_samples")
            dur_ok = dur is not None and not (isinstance(dur, float) and math.isnan(dur)) and float(dur) > 0
            n_ok   = n   is not None and not (isinstance(n,   float) and math.isnan(n))   and float(n)   > 0
            if metric == "duration":
                return float(dur) if dur_ok else (float(n) if n_ok else _EPSILON)
            return float(n) if n_ok else (float(dur) if dur_ok else _EPSILON)

        d["_score"] = d.apply(_size, axis=1).apply(lambda x: x ** (1.0 / T))

        # L3 — dataset weight within (task, lang)
        gs = d.groupby(["task_type", "_lkey"])["_score"].transform("sum")
        d["_ds_w"] = d["_score"] / gs.replace(0, _EPSILON)

        # L2 — language weight within task
        tl = d.groupby(["task_type", "_lkey"])["_score"].sum().reset_index(name="gs")
        ts = tl.groupby("task_type")["gs"].transform("sum")
        tl["_lw"] = tl["gs"] / ts.replace(0, _EPSILON)

        # L1 — task weight globally
        ta = tl.groupby("task_type")["gs"].sum().reset_index(name="ts")
        ta["_tw"] = ta["ts"] / max(ta["ts"].sum(), _EPSILON)
        tl = tl.merge(ta[["task_type", "_tw"]], on="task_type", how="left")

        d = d.merge(tl[["task_type", "_lkey", "_lw", "_tw"]],
                    on=["task_type", "_lkey"], how="left")
        d["_eff"] = d["_tw"] * d["_lw"] * d["_ds_w"]
        return (d["_eff"] * 100).round(4)

    def _orig_pct(df_in: pd.DataFrame) -> pd.Series:
        """Effective probability from the raw YAML weight_group × weight_dataset."""
        eff = df_in["weight_group"] * df_in["weight_dataset"]
        return (eff / eff.sum() * 100).round(4)

    df = df.copy()
    df["_lkey"]   = df.apply(_lang_key, axis=1)
    df["orig_pct"] = _orig_pct(df)
    df["t1_pct"]   = _compute(df, T=1.0).values
    df["t2_pct"]   = _compute(df, T=2.0).values

    total_n = float(df["num_samples"].sum(skipna=True)) or 1.0
    for col, pcol in [("orig_pct","orig_passes"), ("t1_pct","t1_passes"), ("t2_pct","t2_passes")]:
        df[pcol] = ((df[col] / 100 * total_n) / df["num_samples"].clip(lower=1)).round(2)

    size_col = "Duration" if metric == "duration" else "Samples"

    lines = [
        "## 7. Weight Strategy Comparison",
        "",
        "> Side-by-side comparison of three sampling strategies for every dataset.",
        "> Use this table to decide which weights to put in your YAML.",
        "",
        "| Strategy | Formula | Effect |",
        "|----------|---------|--------|",
        "| **Original** | uniform `1/N` within each group (current YAML) | All datasets in a group are equally likely |",
        f"| **T=1** | weight ∝ {('duration' if metric=='duration' else 'samples')} | Larger datasets sampled proportionally more |",
        f"| **T=2** | weight ∝ √{('duration' if metric=='duration' else 'samples')} | Moderate levelling — recommended when sizes differ a lot |",
        "",
        "> **Columns**",
        "> - `Orig% / T=1% / T=2%` — effective sampling probability (all three sum to 100%)",
        "> - `Δ T=1 / Δ T=2` — change vs Original (+positive = more sampling)",
        "> - `Orig× / T=1× / T=2×` — estimated full passes through the dataset per training epoch",
        ">   - ⚠ `> 5×` → over-sampled (overfitting risk)",
        ">   - ⚑ `< 0.1×` → under-sampled (wasted data)",
        "",
        "---",
        "",
    ]

    def _fp(v):
        return f"{v:.3f}%" if (v is not None and not (isinstance(v, float) and math.isnan(v))) else "—"

    def _fd(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.3f}%"

    def _fx(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
        flag = " ⚠" if v > 5 else (" ⚑" if v < 0.1 else "")
        return f"{v:.2f}×{flag}"

    def _size_val(row):
        if metric == "duration":
            return fmt_hours(row.get("total_duration_sec"))
        return fmt_num(row.get("num_samples"))

    for task in sorted(df["task_type"].dropna().unique()):
        t_sub = df[df["task_type"] == task]
        lines += [f"### {task.upper()}", ""]

        for lang in sorted(t_sub["_lkey"].unique()):
            l_sub = t_sub[t_sub["_lkey"] == lang].sort_values("orig_pct", ascending=False)
            lines += [f"#### {lang}", ""]

            rows = []
            for _, r in l_sub.iterrows():
                d_t1 = r["t1_pct"] - r["orig_pct"]
                d_t2 = r["t2_pct"] - r["orig_pct"]
                rows.append([
                    r["dataset_name"],
                    _size_val(r),
                    _fp(r["orig_pct"]),
                    _fp(r["t1_pct"]),
                    _fp(r["t2_pct"]),
                    _fd(d_t1),
                    _fd(d_t2),
                    _fx(r["orig_passes"]),
                    _fx(r["t1_passes"]),
                    _fx(r["t2_passes"]),
                ])

            # Subtotal row
            rows.append([
                f"**TOTAL {lang}**", "",
                f"**{_fp(l_sub['orig_pct'].sum())}**",
                f"**{_fp(l_sub['t1_pct'].sum())}**",
                f"**{_fp(l_sub['t2_pct'].sum())}**",
                "", "", "", "", "",
            ])

            lines.append(_md_table(
                ["Dataset", size_col,
                 "Orig%", "T=1%", "T=2%",
                 "Δ T=1", "Δ T=2",
                 "Orig×", "T=1×", "T=2×"],
                rows,
            ))
            lines.append("")

    # Grand total
    lines += [
        "---", "",
        "### Grand total",
        "",
        _md_table(
            ["Strategy", "Sum of effective %"],
            [
                ["Original", f"{df['orig_pct'].sum():.2f}%"],
                ["T=1",      f"{df['t1_pct'].sum():.2f}%"],
                ["T=2",      f"{df['t2_pct'].sum():.2f}%"],
            ],
        ),
        "",
    ]

    # Quick decision guide
    n_over = {
        "Original": len(df[df["orig_passes"] > 5]),
        "T=1":      len(df[df["t1_passes"]   > 5]),
        "T=2":      len(df[df["t2_passes"]   > 5]),
    }
    n_under = {
        "Original": len(df[df["orig_passes"] < 0.1]),
        "T=1":      len(df[df["t1_passes"]   < 0.1]),
        "T=2":      len(df[df["t2_passes"]   < 0.1]),
    }
    best_over = min(n_over, key=n_over.get)

    lines += [
        "### Which strategy should I use?",
        "",
        _md_table(
            ["Strategy", "Over-sampled datasets (>5×)", "Under-sampled datasets (<0.1×)"],
            [[s, str(n_over[s]), str(n_under[s])] for s in ["Original", "T=1", "T=2"]],
        ),
        "",
        f"> 💡 **Fewest over-sampled datasets**: **{best_over}**",
        ">",
        "> - Datasets roughly equal in size → **Original** (1/N) is fine",
        "> - Large size differences within a group → **T=2** smooths the extremes",
        "> - You trust size reflects quality → **T=1** (fully proportional)",
        "> - You want maximum diversity → **T=2** or higher",
        "",
    ]

    return lines


# ──────────────────────────────────────────────────────────────────────────────
# Main README writer
# ──────────────────────────────────────────────────────────────────────────────

def write_summary(df: pd.DataFrame, out_path: Path,
                  metric: str = "duration",
                  temperature: float = _TEMPERATURE,
                  min_weight: float = 0.0,
                  suggest_weights: bool = False,
                  show_actual_weights: bool = False,
                  show_weight_comparison: bool = False,
                  suggest_steps: bool = False,
                  show_details: bool = False,
                  show_cross_table: bool = False,
                  batch_size: int = 32,
                  grad_accum: int = 1,
                  target_epochs: float = 1.0) -> Path:
    df = df.copy()

    df_w = compute_sampling_weights(df, metric=metric, temperature=temperature, min_weight=min_weight)

    df["display_lang"] = df.apply(_lang_key, axis=1)
    df_w["display_lang"] = df_w["_lkey"]
    df_w["_dl"] = df_w["_lkey"]

    total_samples   = int(df["num_samples"].sum())
    total_dur_sec   = float(df["total_duration_sec"].sum(skipna=True))
    total_dur_hr    = total_dur_sec / 3600
    total_manifests = len(df)
    total_datasets  = df["dataset_name"].nunique()
    total_langs     = df["display_lang"].nunique()
    total_tasks     = df["task_type"].nunique()

    task_grp = (df.groupby("task_type")
                  .agg(manifests=("dataset_name","count"),
                       datasets=("dataset_name","nunique"),
                       samples=("num_samples","sum"),
                       hours=("total_duration_sec", lambda x: x.sum(skipna=True)/3600))
                  .sort_values("samples", ascending=False).reset_index())
    task_grp["samples_%"] = (100 * task_grp["samples"] / total_samples).round(1)
    task_grp["hours_%"]   = (100 * task_grp["hours"]   / total_dur_hr).round(1)

    lang_grp = (df.groupby("display_lang")
                  .agg(manifests=("dataset_name","count"),
                       samples=("num_samples","sum"),
                       hours=("total_duration_sec", lambda x: x.sum(skipna=True)/3600))
                  .sort_values("samples", ascending=False).reset_index())
    lang_grp["samples_%"] = (100 * lang_grp["samples"] / total_samples).round(1)
    lang_grp["hours_%"]   = (100 * lang_grp["hours"]   / total_dur_hr).round(1)

    now   = datetime.now().strftime("%Y-%m-%d %H:%M")
    tasks = sorted(df["task_type"].dropna().unique())

    # Build structure and TOC dynamically
    struct = [{"id": "global", "title": "Global Statistics & Visuals", "show": True}]
    if suggest_weights:
        struct.append({"id": "suggest", "title": "Suggested Sampling Weights (YAML)", "show": True})
    if show_weight_comparison:
        struct.append({"id": "compare", "title": "Weight Strategy Comparison", "show": True})
    if show_details:
        struct.append({"id": "details", "title": "Detailed Dataset Tables", "show": True})
    if show_actual_weights:
        struct.append({"id": "actual", "title": "Actual Sampling Weights", "show": True})
    if suggest_steps:
        struct.append({"id": "steps", "title": "Suggested max_steps", "show": True})
    struct.append({"id": "notes", "title": "Balance Notes", "show": True})

    toc = ["## 📌 Table of Contents", ""]
    s_map = {}
    for i, s in enumerate(struct):
        n = i + 1
        s_map[s["id"]] = n
        slug = s["title"].lower().replace(" ", "-").replace("&", "").replace("(", "").replace(")", "").replace("--", "-")
        toc.append(f"{n}. [{s['title']}](#{n}-{slug})")

    lines = [
        "# 📊 Dataset Composition Report", "",
        f"> Generated: {now}", "", "---", "",
    ] + toc + [
        "", "---", "",
        f"## {s_map['global']}. Global Statistics & Visuals", "",
        "### 📈 Summary Metrics", "",
        "| Metric | Value |", "|--------|-------|",
        f"| Total manifests          | **{total_manifests:,}** |",
        f"| Unique datasets          | **{total_datasets:,}** |",
        f"| Unique languages / pairs | **{total_langs}** |",
        f"| Total samples            | **{fmt_num(total_samples)}** |",
        f"| Total audio duration     | **{total_dur_hr:,.1f} h** |",
        f"| Weight metric            | **{metric}** (T={temperature:.1f}) |",
        "",
        "### 🍩 Composition Visuals", "",
        "| Raw Size Distribution | Suggested Weights (T={:.1f}) |".format(temperature),
        "|:---:|:---:|",
        "| ![Raw](00_global_donut_raw.png) | ![Weighted](00_global_donut_weighted.png) |",
        "",
        "### 📊 Breakdown by Task", "",
    ]

    lines.append(_md_table(
        ["Task", "Datasets", "Samples", "Samples %", "Hours %"],
        [[r["task_type"], str(int(r["datasets"])),
          fmt_num(r["samples"]), f"{r['samples_%']:.1f}%", f"{r['hours_%']:.1f}%"]
         for _, r in task_grp.iterrows()],
    ))
    lines += ["", "### 🌐 Breakdown by Language", ""]
    lines.append(_md_table(
        ["Language / Pair", "Samples", "Samples %", "Hours %"],
        [[r["display_lang"], fmt_num(r["samples"]),
          f"{r['samples_%']:.1f}%", f"{r['hours_%']:.1f}%"]
         for _, r in lang_grp.iterrows()],
    ))

    if show_cross_table:
        cross = (df.groupby(["task_type","display_lang"])["num_samples"]
                   .sum().reset_index()
                   .pivot_table(index="task_type", columns="display_lang",
                                values="num_samples", aggfunc="sum", fill_value=0))
        lines += ["", "### ⚔️ Task × Language Cross-Table", ""]
        lines.append(_md_table(
            ["Task"] + [str(c) for c in cross.columns.tolist()],
            [[str(t)] + [fmt_num(int(v)) for v in row.values] for t, row in cross.iterrows()],
        ))

    if show_details:
        lines += ["", "---", "", f"## {s_map['details']}. Detailed Dataset Tables", "",
                  "> Sorted by language then dataset name. Last row = task totals.", ""]
        for task in tasks:
            lines += _task_section(df, task, total_samples, total_dur_sec)

    if show_weight_comparison:
        lines += ["---", "", f"## {s_map['compare']}. Weight Strategy Comparison", ""]
        lines += _weight_comparison_section(df, metric=metric)

    if show_actual_weights:
        lines += ["---", "", f"## {s_map['actual']}. Actual Sampling Weights", ""]
        lines += _sampling_weight_section(df_w, metric=metric, temperature=temperature)

    if suggest_weights:
        lines += ["---", "", f"## {s_map['suggest']}. Suggested Sampling Weights (YAML)", ""]
        lines += suggest_weights_section(df, metric=metric, temperature=temperature,
                                         min_weight=min_weight)

    if suggest_steps:
        lines += ["---", "", f"## {s_map['steps']}. Suggested max_steps", ""]
        lines += suggest_max_steps_section(df_w, batch_size=batch_size,
                                           grad_accum=grad_accum, target_epochs=target_epochs)

    # Balance Notes (at the very end)
    notes = []
    dom_task = task_grp.iloc[0]["task_type"]
    dom_pct  = float(task_grp.iloc[0]["samples_%"])
    notes.append(
        f"{'⚠️' if dom_pct > 50 else '✅'}  Task **{dom_task}** is the largest "
        f"at **{dom_pct:.1f}%** of all samples."
    )
    for lang in ["en", "fr"]:
        row = lang_grp[lang_grp["display_lang"] == lang]
        pct = float(row["samples_%"].values[0]) if len(row) else 0.0
        notes.append(f"{'🟡' if pct > 40 else '✅'}  Language **{lang}**: **{pct:.1f}%** of total.")
    top3     = lang_grp.head(3)["display_lang"].tolist()
    top3_pct = float(lang_grp.head(3)["samples_%"].sum())
    notes.append(f"📌  Top-3 languages ({', '.join(top3)}) → **{top3_pct:.1f}%** of samples.")
    low_cov = lang_grp[lang_grp["manifests"] == 1]
    if not low_cov.empty:
        notes.append(f"⚠️  Single-manifest languages: {', '.join(low_cov['display_lang'].tolist())}.")

    lines += ["", "---", "", f"## {s_map['notes']}. Balance Notes", ""]
    for note in notes:
        lines.append(f"- {note}")
    lines += ["", "---", "", "_Report generated by `dataset_metadata_analysis.py`_"]

    md_path = out_path.with_suffix(".md") if out_path.suffix != ".md" else out_path
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  ✓  README → {md_path}")

    # Compact stdout summary
    print("\n" + "=" * 70)
    print("  DATASET BALANCE SUMMARY")
    print("=" * 70)
    print(f"\n  Metric: {metric}  |  Temperature: {temperature}")
    print(f"  Manifests: {total_manifests}  |  Datasets: {total_datasets}  |"
          f"  Duration: {total_dur_hr:,.1f} h\n")
    print(f"  {'Task':<22} {'Samples':>10}  {'%':>6}  {'Hours':>9}")
    print("  " + "-" * 54)
    for _, r in task_grp.iterrows():
        print(f"  {r['task_type']:<22} {int(r['samples']):>10,}  "
              f"{r['samples_%']:>5.1f}%  {r['hours']:>8.1f}h")
    if show_actual_weights:
        print(f"\n  {'Task':<8} {'Lang':<10} {'Dataset':<28} "
              f"{'ds_w':>8} {'samp%':>7}  {'passes':>7}")
        print("  " + "-" * 70)
        for _, r in df_w.sort_values(["task_type","_dl","sampling_prob_pct"],
                                     ascending=[True,True,False]).iterrows():
            ep   = r.get("expected_passes_per_epoch", "?")
            flag = " ⚠" if (isinstance(ep,(int,float)) and ep > 5) else \
                   " ⚑" if (isinstance(ep,(int,float)) and ep < 0.1) else ""
            print(f"  {str(r['task_type']):<8} {str(r['_dl']):<10} "
                  f"{str(r['dataset_name'])[:27]:<28} "
                  f"{r['dataset_w']:>8.4f} {r['sampling_prob_pct']:>6.3f}%"
                  f"  {ep:>6}×{flag}")
        print(f"  {'':->70}")
        print(f"  {'TOTAL':>49} {df_w['sampling_prob_pct'].sum():>6.1f}%")
    print("=" * 70 + "\n")
    return md_path


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

PALETTE = "Set2"


def set_style():
    sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.1)
    plt.rcParams.update({"figure.dpi": 130, "axes.titlesize": 13,
                          "axes.labelsize": 11, "hatch.linewidth": 0.9})


def plot_global_donut(df: pd.DataFrame, out: Path, metric: str,
                      temperature: float = 2.0, weight_col: str = None,
                      split_label: str = "train"):
    metric_label = "duration" if metric == "duration" else "samples"
    size_col = weight_col if weight_col else ("total_duration_sec" if metric == "duration" else "num_samples")

    if weight_col and weight_col in df.columns:
        use_weights = True
    else:
        use_weights = False

    if df.empty or df[size_col].sum() == 0:
        print(f"  ⚠  No data for donut ({size_col})."); return
    def make_label(row):
        task = str(row["task_type"]).strip()
        lang = str(row.get("language", "") or "").strip()
        src  = str(row.get("source_lang", "") or "").strip()
        tgt  = str(row.get("target_lang", "") or "").strip()
        grp  = task.upper() if task in ("asr","ast","qa","mqa","aqa") \
               else task.replace("_"," ").title()
        if task == "ast":
            lbl = (f"AST ({src}→{tgt})" if src and tgt else f"AST ({src}→?)" if src else "AST")
            return grp, lbl, src or lang or "unknown", tgt or lang or "unknown", task
        langs = (lang, lang) if lang else ("unknown", "unknown")
        for t, prefix in [("asr","ASR"),("qa","QA"),("aqa","AQA"),("mqa","Music QA")]:
            if task == t:
                grp2 = "Music QA" if t == "mqa" else grp
                return grp2, f"{prefix} ({lang})" if lang else prefix, *langs, task
        if task in ("audio_captioning","music_captioning"):
            lbl = task.replace("_"," ").title()
            return lbl, f"{lbl} ({lang})" if lang else lbl, *langs, task
        lbl = task.replace("_"," ").title()
        return grp, f"{lbl} ({lang})" if lang else lbl, *langs, task

    df = df.copy()
    info = df.apply(make_label, axis=1)
    df["_grp"]  = [x[0] for x in info]
    df["_lbl"]  = [x[1] for x in info]
    df["_lsrc"] = [x[2] for x in info]
    df["_ltgt"] = [x[3] for x in info]
    df["_tsk"]  = [x[4] for x in info]

    lbl_map = {row["_lbl"]: {"group": row["_grp"], "l1": row["_lsrc"],
                               "l2": row["_ltgt"], "task": row["_tsk"]}
               for _, row in df.iterrows()}

    if weight_col and weight_col in df.columns:
        df["_ew"] = df[weight_col] / 100.0 if "pct" in weight_col else df[weight_col]
        use_weights = True
    elif use_weights:
        df_w = compute_sampling_weights(df, metric=metric, temperature=temperature)
        df["_ew"] = df_w["effective_prob"]

    dur_sum      = df.groupby("_lbl")["total_duration_sec"].sum().fillna(0)
    use_duration = dur_sum.sum() > 0

    if use_weights:
        N = df["num_samples"].sum()
        df["_es"] = df["_ew"] * N
        if use_duration:
            df["_ad"] = df["total_duration_sec"] / df["num_samples"].clip(lower=1)
            df["_wv"] = df["_es"] * df["_ad"]
            grp = df.groupby("_lbl")["_wv"].sum().fillna(0)
            scale = dur_sum.sum() / grp.sum() if grp.sum() > 0 else 1.0
            grp   = grp * scale
        else:
            grp = df.groupby("_lbl")["_es"].sum().fillna(0)
    else:
        grp = dur_sum if use_duration else df.groupby("_lbl")["num_samples"].sum().fillna(0)

    grp = grp[grp > 0].sort_values(ascending=False)
    if grp.empty:
        print("  ⚠  No data for donut."); return

    if use_duration:
        metric_label  = "audio hours"
        centre_suffix = "hr total"
        total_display = grp.sum() / 3600
        fmt_val = lambda v: f"{v:,.0f} hr"
    else:
        metric_label  = "samples"
        centre_suffix = "samples total"
        total_display = grp.sum()
        fmt_val = lambda v: f"{v:,.0f}"

    total_val = grp.sum()
    pcts      = 100 * grp / float(total_val)
    main_grp  = grp[pcts >= 0.5].copy()
    small_grp = grp[pcts < 0.5]
    if not small_grp.empty:
        other_lbl           = "Other (< 0.5% each)"
        main_grp[other_lbl] = small_grp.sum()
        lbl_map[other_lbl]  = {"group":"Other","l1":"mixed","l2":"mixed","task":"other"}

    main_disp = main_grp / 3600 if use_duration else main_grp
    main_pcts = 100 * main_grp / float(total_val)

    fig    = plt.figure(figsize=(20, 12))
    ax_d   = fig.add_axes([0.02, 0.06, 0.46, 0.86])
    ax_leg = fig.add_axes([0.50, 0.05, 0.48, 0.92])
    ax_d.set_aspect("equal"); ax_d.set_xlim(-1.25,1.25)
    ax_d.set_ylim(-1.30,1.25); ax_d.axis("off")

    theta = 90.0
    for lbl, val in zip(main_grp.index, main_grp.values):
        d_theta = 360 * val / float(main_grp.sum())
        t1, t2  = theta - d_theta, theta
        m       = lbl_map[lbl]
        c1, c2, ht = lang_color(m["l1"]), lang_color(m["l2"]), task_hatch(m["task"])
        if m["l1"] == m["l2"]:
            ax_d.add_patch(Wedge((0,0), 1.0, t1, t2, width=0.42,
                facecolor=c1, edgecolor="white", linewidth=1.6, hatch=ht, alpha=0.93))
        else:
            ax_d.add_patch(Wedge((0,0), 1.00, t1, t2, width=0.21,
                facecolor=c2, edgecolor="white", linewidth=1.4, hatch=ht, alpha=0.93))
            ax_d.add_patch(Wedge((0,0), 0.79, t1, t2, width=0.21,
                facecolor=c1, edgecolor="white", linewidth=1.4, hatch=ht, alpha=0.93))
        theta = t1

    wt_tag = f"\n(Weighted, metric={metric}, T={temperature:.1f})" if use_weights else f" (Raw {metric_label})"
    ax_d.text(0,  0.13, f"Split: {split_label}",
              ha="center", va="center", fontsize=15, fontweight="bold", color="#1a1a1a")
    ax_d.text(0, -0.13, f"{fmt_val(total_display)} {centre_suffix}{wt_tag}",
              ha="center", va="center", fontsize=11, color="#555555")
    ax_d.text(0, -1.22,
              "AST pairs — Inner ring: source lang  |  Outer ring: target lang",
              ha="center", va="bottom", fontsize=7.5, color="#666666", style="italic")

    ax_leg.axis("off")
    leg_data = [{"group": lbl_map[lbl]["group"], "label": lbl, "pct": p, "val": dv,
                 "c1": lang_color(lbl_map[lbl]["l1"]), "c2": lang_color(lbl_map[lbl]["l2"]),
                 "ht": task_hatch(lbl_map[lbl]["task"])}
                for lbl, p, dv in zip(main_grp.index, main_pcts.values, main_disp.values)]
    grp_totals = {}
    for item in leg_data:
        grp_totals[item["group"]] = grp_totals.get(item["group"], 0) + item["val"]
    leg_data.sort(key=lambda x: (-grp_totals[x["group"]], x["group"], -x["val"]))

    n_rows = len(leg_data) + len(grp_totals) + 6
    row_h  = 1.0 / n_rows
    col_xs = [0.00, 0.055, 0.66, 0.84]
    ax_leg.text(col_xs[1], 1.0, "Dataset Group", fontsize=10, fontweight="bold",
                va="top", transform=ax_leg.transAxes)
    ax_leg.text(col_xs[2], 1.0, "%", fontsize=10, fontweight="bold",
                va="top", ha="right", transform=ax_leg.transAxes)
    ax_leg.text(col_xs[3], 1.0, metric_label.title(), fontsize=10, fontweight="bold",
                va="top", ha="right", transform=ax_leg.transAxes)

    y_pos = 1.0 - 2 * row_h
    current_group = None
    for item in leg_data:
        if item["group"] != current_group:
            current_group = item["group"]
            y_pos -= row_h * 0.35
            ax_leg.text(0.0, y_pos + row_h * 0.15, current_group, fontsize=9.5,
                        fontweight="bold", color="#111111", va="center",
                        transform=ax_leg.transAxes)
            y_pos -= row_h
        sw = 0.038
        if item["c1"] == item["c2"]:
            ax_leg.add_patch(plt.Rectangle((col_xs[0], y_pos - row_h*0.3), sw, row_h*0.7,
                transform=ax_leg.transAxes, clip_on=False,
                facecolor=item["c1"], hatch=item["ht"], edgecolor="white", linewidth=0.4))
        else:
            ax_leg.add_patch(plt.Rectangle((col_xs[0]+sw/2, y_pos-row_h*0.3), sw/2, row_h*0.7,
                transform=ax_leg.transAxes, clip_on=False,
                facecolor=item["c1"], hatch=item["ht"], edgecolor="white", linewidth=0.4))
            ax_leg.add_patch(plt.Rectangle((col_xs[0]+sw/2, y_pos-row_h*0.3), sw/2, row_h*0.7,
                transform=ax_leg.transAxes, clip_on=False,
                facecolor=item["c2"], hatch=item["ht"], edgecolor="white", linewidth=0.4))
        lbl_str = item["label"][:46] + "…" if len(item["label"]) > 48 else item["label"]
        ax_leg.text(col_xs[1], y_pos+row_h*0.15, lbl_str, fontsize=8.5, va="center",
                    transform=ax_leg.transAxes, clip_on=False)
        ax_leg.text(col_xs[2], y_pos+row_h*0.15, f"{item['pct']:.1f}%", fontsize=8.5,
                    va="center", ha="right", transform=ax_leg.transAxes, clip_on=False)
        ax_leg.text(col_xs[3], y_pos+row_h*0.15, fmt_val(item["val"]), fontsize=8.5,
                    va="center", ha="right", transform=ax_leg.transAxes, clip_on=False)
        y_pos -= row_h

    suffix = f" (Suggested Weights, T={temperature:.1f})" if weight_col else f" (Raw {metric_label})"
    fig.suptitle(f"Dataset composition — {suffix}\n"
                 "Color = language  |  Hatch = task",
                 fontsize=13, fontweight="bold", y=0.995)

    fname = "00_global_donut_weighted.png" if weight_col else "00_global_donut_raw.png"
    plt.savefig(out / fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  ✓  {out / fname}")


def plot_instruction_response_lengths(df: pd.DataFrame, out: Path):
    instr = df["avg_instruction_words"].dropna()
    resp  = df["avg_response_words"].dropna()
    if instr.empty and resp.empty:
        print("  ⚠  No word-count data — skipping."); return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if not instr.empty:
        axes[0].hist(instr, bins=30, color=sns.color_palette(PALETTE)[1], edgecolor="white")
        axes[0].set_xlabel("Avg instruction words"); axes[0].set_ylabel("# datasets")
        axes[0].set_title("Instruction length distribution")
    if not resp.empty:
        axes[1].hist(resp, bins=30, color=sns.color_palette(PALETTE)[2], edgecolor="white")
        axes[1].set_xlabel("Avg response words"); axes[1].set_ylabel("# datasets")
        axes[1].set_title("Response length distribution")
    plt.tight_layout()
    out_p = out / "01_instruction_response_lengths.png"
    fig.savefig(out_p, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓  {out_p}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Generate dataset metadata CSV + Markdown report from NeMo YAML.")
    parser.add_argument("yaml_path")
    parser.add_argument("--output_dir",   default=None)
    parser.add_argument("--data_root",    default="")
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--metric",       default="duration",
                        choices=["duration", "samples"],
                        help="Size metric for weight computation: "
                             "'duration' (default) or 'samples'.")
    parser.add_argument("--temperature",  type=float, default=_TEMPERATURE,
                        help=f"Smoothing temperature T (default {_TEMPERATURE}). "
                             "T=1→proportional, T=2→moderate, T→∞→uniform.")
    parser.add_argument("--min_weight",   type=float, default=0.0001,
                        help="Minimum sampling weight floor for datasets (default: 0.0001). "
                             "Use >0 to ensure small datasets are sampled.")
    parser.add_argument("--suggest_weights",  action="store_true",
                        help="Add Section 8: data-driven YAML weight suggestions.")
    parser.add_argument("--actual_weights",   action="store_true",
                        help="Add Section 6.5: actual sampling probabilities.")
    parser.add_argument("--weight_comparison", action="store_true",
                        help="Add Section 7: side-by-side Original vs T=1 vs T=2 comparison table.")
    parser.add_argument("--suggest_steps",    action="store_true",
                        help="Add Section 9: recommended max_steps.")
    parser.add_argument("--show_details",      action="store_true",
                        help="Show detailed dataset tables (default: False).")
    parser.add_argument("--show_cross_table", action="store_true",
                        help="Show Task x Language cross-table (default: False).")
    parser.add_argument("--output_yaml",      default=None,
                        help="Save suggested YAML config to this path.")
    parser.add_argument("--workers",      type=int,   default=8,
                        help="Number of parallel workers (default: 8).")
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--grad_accum",   type=int,   default=1)
    parser.add_argument("--target_epochs",type=float, default=1.0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else script_dir / "dataset_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 YAML        : {args.yaml_path}")
    print(f"📁 Output      : {out_dir}")
    print(f"⚖️  Metric      : {args.metric}  |  Temperature: {args.temperature}")
    if args.min_weight > 0:
        print(f"⚖️  Min Weight  : {args.min_weight}")
    print()

    entries = flatten_manifests(args.yaml_path)
    print(f"   Found {len(entries)} manifest entries.\n")

    DATA_FOLDER = args.data_root or os.environ.get("DATA_FOLDER", "")

    processed_entries = []
    for e in entries:
        path = e["manifest_filepath"]
        if DATA_FOLDER:
            path = path.replace("${oc.env:DATA_FOLDER}", DATA_FOLDER)
        p            = Path(path)
        dataset_name = p.parent.name
        stem         = p.stem
        split        = "train" if stem.startswith("train") else stem
        note         = []
        if "recasepunc" in stem: note.append("with punctuations")
        if "max30"      in stem: note.append("max duration is 30s")
        e.update({"path": path, "dataset_name": dataset_name,
                  "split": split, "note": ", ".join(note),
                  "raw_manifest_path": e.get("raw_manifest_path", path)})
        processed_entries.append(e)

    groups = {}
    for e in processed_entries:
        key = (e["dataset_name"], e["split"], e["note"], e["task_type"], e["sub_task"],
               e["language"], e["source_lang"], e["target_lang"], e.get("raw_manifest_path", ""))
        groups.setdefault(key, []).append(e)

    print(f"   Aggregated into {len(groups)} distinct dataset groups.\n")

    rows, missing = 0, 0
    total = len(groups)
    print(f"   Processing {total} groups with {args.workers} workers...")

    # Prepare worker tasks
    tasks_data = []
    for key, group_entries in groups.items():
        paths = [e["path"] for e in group_entries]
        e_first = group_entries[0]
        tasks_data.append((key, paths, e_first))

    parsed_rows = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_worker_parse_group, t): t for t in tasks_data}

            for i, future in enumerate(as_completed(futures)):
                base, success, paths = future.result()
                if not success:
                    missing += len(paths)
                    if not args.skip_missing:
                        key = futures[future][0]
                        raise RuntimeError(f"Missing manifests for {key[0]} {key[1]}: {paths}")

                parsed_rows.append(base)

                # Update progress bar
                percent = (i + 1) / total * 100
                ds_name = base["dataset_name"]
                print(f"\r  [{i+1}/{total}] {percent:3.0f}% | {ds_name[:30]:<30}", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\n❗ Interrupted by user. Exiting...")
        sys.exit(1)

    print(f"\n   Parsed: {len(entries)-missing} found, {missing} missing.\n")

    df = pd.DataFrame(parsed_rows)
    col_order = [
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
    df = df[[c for c in col_order if c in df.columns]]
    csv_path = out_dir / "dataset_metadata.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV  → {csv_path}  ({len(df)} rows)\n")

    df_vis = df[df.get("file_exists", pd.Series(True, index=df.index)) == True].copy()
    df_vis = df_vis[df_vis["num_samples"].notna()]
    if df_vis.empty:
        print("⚠  No manifest data found on disk — using YAML metadata.")
        df_vis = df.copy()
        df_vis["num_samples"] = df_vis["num_samples"].fillna(0)

    print("🎨 Generating plots …")
    set_style()
    split_label = (df_vis["split"].mode()[0]
                   if "split" in df_vis.columns and not df_vis.empty else "train")
    df_w = compute_sampling_weights(df_vis, metric=args.metric, temperature=args.temperature, min_weight=args.min_weight)
    plot_global_donut(df_vis, out_dir, args.metric, args.temperature, weight_col=None, split_label=split_label)
    plot_global_donut(df_w, out_dir, args.metric, args.temperature, weight_col="sampling_prob_pct", split_label=split_label)
    plot_instruction_response_lengths(df_vis, out_dir)

    print("\n📝 Writing README …")
    write_summary(
        df_vis, out_dir / "README.md",
        metric=args.metric,
        temperature=args.temperature,
        min_weight=args.min_weight,
        suggest_weights=args.suggest_weights,
        show_actual_weights=args.actual_weights,
        show_weight_comparison=args.weight_comparison,
        suggest_steps=args.suggest_steps,
        show_details=args.show_details,
        show_cross_table=args.show_cross_table,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        target_epochs=args.target_epochs,
    )

    if args.output_yaml:
        yaml_lines = suggest_weights_section(
            df_vis, metric=args.metric, temperature=args.temperature,
            min_weight=args.min_weight
        )
        # Extract content between ```yaml and ```
        try:
            start_idx = -1
            for i, line in enumerate(yaml_lines):
                if "```yaml" in line: start_idx = i + 1; break
            if start_idx != -1:
                end_idx = -1
                for i in range(start_idx, len(yaml_lines)):
                    if "```" in yaml_lines[i]: end_idx = i; break
                if end_idx != -1:
                    with open(args.output_yaml, "w", encoding="utf-8") as f:
                        f.write("\n".join(yaml_lines[start_idx:end_idx]))
                    print(f"✅ YAML → {args.output_yaml}")
        except Exception as e:
            print(f"  ⚠ Failed to write YAML snippet: {e}")
    print(f"\n✅ Done!  Outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()