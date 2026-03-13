"""
Dataset Metadata Extractor & Visualizer
Parses a NeMo-style YAML config referencing JSONL manifests,
computes per-dataset statistics, produces a Markdown README report and plots.

Changes vs previous version:
  - Output directory defaults to <script_dir>/dataset_analysis  (same level as script)
  - Donut chart hatches are now PER-TASK (consistent across all charts)
  - Removed: plot_task_distribution, plot_duration_by_lang
  - Added:   per-task detailed dataset tables in the README
"""

import os
import json
import math
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
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge
import seaborn as sns

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Visual encoding: language colors  +  task hatches
# ──────────────────────────────────────────────────────────────────────────────

LANG_COLORS: dict = {
    "fr":      "#002395",   # Blue        — French flag
    "en":      "#C8102E",   # Red         — UK flag
    "es":      "#AA151B",   # Deep red    — Spain
    "de":      "#444444",   # Dark grey   — Germany (black renders poorly)
    "it":      "#009246",   # Green       — Italy
    "nl":      "#FF6600",   # Orange      — Netherlands
    "pt":      "#006600",   # Dark green  — Portugal
    "ar":      "#007A3D",   # Green       — Arab League
    "mixed":   "#AAAAAA",
    "unknown": "#CCCCCC",
    "":        "#CCCCCC",
}

# Hatch patterns are TASK-based so the same task always looks the same
TASK_HATCHES: dict = {
    "asr":              "",       # solid            — most common task
    "ast":              "///",    # forward diagonals
    "qa":               "...",    # dots
    "aqa":              "|||",    # vertical lines
    "other":            "ooo",    # circles
    "audio_captioning": "---",    # horizontal lines
    "music_captioning": "***",    # stars
    "mqa":              "xxx",    # cross-hatch
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


# ──────────────────────────────────────────────────────────────────────────────
# YAML flattening
# ──────────────────────────────────────────────────────────────────────────────

def flatten_manifests(yaml_path: str) -> list:
    """Walk the YAML tree and yield one dict per multimodal_conversation entry."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    entries = []

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
        sub_task = tags.get("sub_task",    inherited_subtask)
        src_lang = tags.get("source_lang", inherited_src_lang)
        tgt_lang = tags.get("target_lang", inherited_tgt_lang)
        weight   = float(node.get("weight", inherited_weight) or inherited_weight)

        ntype = node.get("type", "")

        if ntype == "multimodal_conversation":
            entries.append({
                "manifest_filepath": node.get("manifest_filepath", ""),
                "task_type":         task,
                "sub_task":          sub_task,
                "language":          lang,
                "source_lang":       src_lang,
                "target_lang":       tgt_lang,
                "weight_dataset":    weight,
                "weight_group":      group_weight if group_weight is not None else weight,
            })
            return

        if ntype == "group":
            for child in node.get("input_cfg", []):
                recurse(child, task, lang, sub_task, weight,
                        group_weight=weight,
                        inherited_src_lang=src_lang,
                        inherited_tgt_lang=tgt_lang)

    for top in cfg.get("input_cfg", []):
        recurse(top)

    return entries


# ──────────────────────────────────────────────────────────────────────────────
# JSONL parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_manifest(manifest_path: str,
                   task_type: str, sub_task: str,
                   language: str, source_lang: str, target_lang: str):
    p = Path(manifest_path)
    if not p.exists():
        return None

    durations, instruction_wc, response_wc = [], [], []
    speaker_ids, sampling_rates, channels_list = set(), [], []
    num_samples = 0
    num_segments = 0

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

            # Duration — multi-priority chain
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
        "dataset_name":                p.parent.name,
        "split":                       p.stem,
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
        "path":                        str(p),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Markdown helpers
# ──────────────────────────────────────────────────────────────────────────────

def _md_table(headers, rows) -> str:
    """Render a padded Markdown table."""
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
# Per-task dataset tables
# ──────────────────────────────────────────────────────────────────────────────

def _task_section(df: pd.DataFrame, task: str,
                  total_samples: int, total_dur_sec: float) -> list:
    """
    Return Markdown lines for one task's dataset table.

    Columns: Dataset | Split | Language | Samples | % Total | Duration | Avg Seg (s)
    Rows sorted by language then dataset name; totals row appended.
    """
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
            r["dataset_name"],
            r["split"],
            r["_lang"],
            fmt_num(n_samp),
            pct,
            fmt_hours(dur_sec),
            f"{avg_seg:.1f} s" if avg_seg else "—",
        ])

    # Totals row
    t_samp = sub["num_samples"].sum()
    t_dur  = sub["total_duration_sec"].sum(skipna=True)
    t_pct  = f"{100 * t_samp / total_samples:.2f}%" if total_samples else "—"
    rows.append([
        f"**TOTAL ({task.upper()})**", "", "",
        f"**{fmt_num(t_samp)}**",
        f"**{t_pct}**",
        f"**{fmt_hours(t_dur)}**",
        "",
    ])

    n_manifests = len(sub)
    n_datasets  = sub["dataset_name"].nunique()
    n_langs     = sub["_lang"].nunique()

    lines = [
        f"### {task.upper()}",
        "",
        f"> {n_manifests} manifests &nbsp;·&nbsp; "
        f"{n_datasets} unique datasets &nbsp;·&nbsp; "
        f"{n_langs} language(s)",
        "",
    ]
    lines.append(_md_table(
        ["Dataset", "Split", "Language", "Samples", "% Total", "Duration", "Avg Seg"],
        rows,
    ))
    lines.append("")
    return lines


# ──────────────────────────────────────────────────────────────────────────────
# Main README writer
# ──────────────────────────────────────────────────────────────────────────────

def write_summary(df: pd.DataFrame, out_path: Path) -> Path:
    """Write a README-style Markdown report and print a compact summary to stdout."""
    df = df.copy()

    def _dlang(row):
        task = str(row.get("task_type", "")).strip()
        lang = str(row.get("language", "") or "").strip()
        if task == "ast":
            src = str(row.get("source_lang", "") or "").strip()
            tgt = str(row.get("target_lang", "") or "").strip()
            if src and tgt:
                return f"{src}→{tgt}"
            return src or tgt or "unknown"
        return lang or "unknown"

    df["display_lang"] = df.apply(_dlang, axis=1)

    # ── Aggregates ────────────────────────────────────────────────────────────
    total_samples   = int(df["num_samples"].sum())
    total_dur_sec   = float(df["total_duration_sec"].sum(skipna=True))
    total_dur_hr    = total_dur_sec / 3600
    total_manifests = len(df)
    total_datasets  = df["dataset_name"].nunique()
    total_langs     = df["display_lang"].nunique()
    total_tasks     = df["task_type"].nunique()

    task_grp = (df.groupby("task_type")
                  .agg(manifests=("dataset_name", "count"),
                       datasets=("dataset_name", "nunique"),
                       samples=("num_samples", "sum"),
                       hours=("total_duration_sec", lambda x: x.sum(skipna=True) / 3600))
                  .sort_values("samples", ascending=False)
                  .reset_index())
    task_grp["samples_%"] = (100 * task_grp["samples"] / total_samples).round(1)
    task_grp["hours_%"]   = (100 * task_grp["hours"]   / total_dur_hr).round(1)

    lang_grp = (df.groupby("display_lang")
                  .agg(manifests=("dataset_name", "count"),
                       samples=("num_samples", "sum"),
                       hours=("total_duration_sec", lambda x: x.sum(skipna=True) / 3600))
                  .sort_values("samples", ascending=False)
                  .reset_index())
    lang_grp["samples_%"] = (100 * lang_grp["samples"] / total_samples).round(1)
    lang_grp["hours_%"]   = (100 * lang_grp["hours"]   / total_dur_hr).round(1)

    cross = (df.groupby(["task_type", "display_lang"])["num_samples"]
               .sum()
               .reset_index()
               .pivot_table(index="task_type", columns="display_lang",
                            values="num_samples", aggfunc="sum", fill_value=0))

    dur_df = df[df["avg_segment_duration_sec"].notna()]

    # Balance notes
    notes = []
    dom_task = task_grp.iloc[0]["task_type"]
    dom_pct  = float(task_grp.iloc[0]["samples_%"])
    if dom_pct > 50:
        notes.append(f"⚠️  Task **{dom_task}** dominates with **{dom_pct:.1f}%** of all samples.")
    else:
        notes.append(f"✅  No single task exceeds 50%. Most dominant: **{dom_task}** at **{dom_pct:.1f}%**.")

    for lang in ["en", "fr"]:
        row = lang_grp[lang_grp["display_lang"] == lang]
        pct = float(row["samples_%"].values[0]) if len(row) else 0.0
        flag = "🟡" if pct > 40 else "✅"
        notes.append(f"{flag}  Language **{lang}**: **{pct:.1f}%** of total samples.")

    top3 = lang_grp.head(3)["display_lang"].tolist()
    top3_pct = float(lang_grp.head(3)["samples_%"].sum())
    notes.append(f"📌  Top-3 languages ({', '.join(top3)}) account for **{top3_pct:.1f}%** of samples.")

    low_cov = lang_grp[lang_grp["manifests"] == 1]
    if not low_cov.empty:
        notes.append(f"⚠️  Low-coverage (1 manifest only): {', '.join(low_cov['display_lang'].tolist())}.")

    # ── Build Markdown ────────────────────────────────────────────────────────
    now   = datetime.now().strftime("%Y-%m-%d %H:%M")
    tasks = sorted(df["task_type"].dropna().unique())

    toc_tasks = "\n".join(f"   - [{t.upper()}](#{t.lower()})" for t in tasks)

    lines = [
        "# 📊 Dataset Composition Report",
        "",
        f"> Generated: {now}",
        "",
        "---",
        "",
        "## 📌 Table of Contents",
        "",
        "1. [Global Statistics](#1-global-statistics)",
        "2. [Distribution by Task](#2-distribution-by-task)",
        "3. [Distribution by Language](#3-distribution-by-language)",
        "4. [Task × Language Cross-Table](#4-task--language-cross-table)",
        "5. [Audio Duration Statistics](#5-audio-duration-statistics)",
        "6. [Detailed Dataset Tables per Task](#6-detailed-dataset-tables-per-task)",
        toc_tasks,
        "7. [Balance Notes](#7-balance-notes)",
        "",
        "---",
        "",
        "## 1. Global Statistics",
        "",
        "| Metric                   | Value |",
        "|--------------------------|-------|",
        f"| Total manifests          | **{total_manifests:,}** |",
        f"| Unique datasets          | **{total_datasets:,}** |",
        f"| Unique languages / pairs | **{total_langs}** |",
        f"| Unique tasks             | **{total_tasks}** |",
        f"| Total samples            | **{fmt_num(total_samples)}** ({total_samples:,}) |",
        f"| Total audio duration     | **{total_dur_hr:,.1f} h** ({seconds_to_dhms(total_dur_sec)}) |",
        "",
        "---",
        "",
        "## 2. Distribution by Task",
        "",
    ]

    task_rows = [
        [r["task_type"], str(int(r["manifests"])), str(int(r["datasets"])),
         fmt_num(r["samples"]), f"{r['samples_%']:.1f}%",
         f"{r['hours']:.1f} h", f"{r['hours_%']:.1f}%"]
        for _, r in task_grp.iterrows()
    ]
    lines.append(_md_table(
        ["Task", "Manifests", "Datasets", "Samples", "Samples %", "Hours", "Hours %"],
        task_rows,
    ))

    lines += ["", "---", "", "## 3. Distribution by Language", ""]
    lang_rows = [
        [r["display_lang"], str(int(r["manifests"])), fmt_num(r["samples"]),
         f"{r['samples_%']:.1f}%", f"{r['hours']:.1f} h", f"{r['hours_%']:.1f}%"]
        for _, r in lang_grp.iterrows()
    ]
    lines.append(_md_table(
        ["Language / Pair", "Manifests", "Samples", "Samples %", "Hours", "Hours %"],
        lang_rows,
    ))

    lines += ["", "---", "", "## 4. Task × Language Cross-Table", "",
              "> Cell values = samples.  Rows = tasks, columns = languages/pairs.", ""]
    cross_headers = ["Task"] + [str(c) for c in cross.columns.tolist()]
    cross_rows    = [
        [str(t)] + [fmt_num(int(v)) for v in row.values]
        for t, row in cross.iterrows()
    ]
    lines.append(_md_table(cross_headers, cross_rows))

    lines += ["", "---", "", "## 5. Audio Duration Statistics", ""]
    if not dur_df.empty:
        dur_task = (dur_df.groupby("task_type")
                          .agg(avg_seg=("avg_segment_duration_sec", "mean"),
                               min_seg=("min_segment_duration_sec", "min"),
                               max_seg=("max_segment_duration_sec", "max"))
                          .reset_index())
        dur_rows = [
            [r["task_type"],
             f"{r['avg_seg']:.2f} s" if r["avg_seg"] else "—",
             f"{r['min_seg']:.2f} s" if r["min_seg"] else "—",
             f"{r['max_seg']:.2f} s" if r["max_seg"] else "—"]
            for _, r in dur_task.iterrows()
        ]
        lines.append(_md_table(
            ["Task", "Avg Segment", "Min Segment", "Max Segment"], dur_rows))
    else:
        lines.append("_No duration data available._")

    # ── Section 6: per-task tables ────────────────────────────────────────────
    lines += [
        "", "---", "",
        "## 6. Detailed Dataset Tables per Task", "",
        "> One table per task. Columns: Dataset · Split · Language · "
        "Samples · % of total · Duration · Avg segment duration.",
        "> Sorted by language then dataset name. Last row = task totals.",
        "",
    ]
    for task in tasks:
        lines += _task_section(df, task, total_samples, total_dur_sec)

    # ── Section 7 ─────────────────────────────────────────────────────────────
    lines += ["---", "", "## 7. Balance Notes", ""]
    for note in notes:
        lines.append(f"- {note}")

    lines += ["", "---", "", "_Report generated by `dataset_analysis.py`_"]

    # Write
    md_path = out_path.with_suffix(".md") if out_path.suffix != ".md" else out_path
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  ✓  README → {md_path}")

    # Stdout
    print("\n" + "=" * 70)
    print("  DATASET BALANCE SUMMARY")
    print("=" * 70)
    print(f"\n  Manifests : {total_manifests}  |  Datasets : {total_datasets}")
    print(f"  Samples   : {fmt_num(total_samples)}  |  Duration : {total_dur_hr:,.1f} h\n")
    print(f"  {'Task':<22} {'Samples':>10}  {'%':>6}  {'Hours':>9}")
    print("  " + "-" * 54)
    for _, r in task_grp.iterrows():
        print(f"  {r['task_type']:<22} {int(r['samples']):>10,}  "
              f"{r['samples_%']:>5.1f}%  {r['hours']:>8.1f}h")
    print(f"\n  {'Language':<14} {'Samples':>10}  {'%':>6}  {'Hours':>9}")
    print("  " + "-" * 44)
    for _, r in lang_grp.iterrows():
        print(f"  {str(r['display_lang']):<14} {int(r['samples']):>10,}  "
              f"{r['samples_%']:>5.1f}%  {r['hours']:>8.1f}h")
    print("\n  Balance Notes:")
    for note in notes:
        clean = (note.replace("⚠️","[!]").replace("✅","[ok]")
                     .replace("🟡","[~]").replace("📌","[*]"))
        print(f"    {clean}")
    print("=" * 70 + "\n")

    return md_path


# ──────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ──────────────────────────────────────────────────────────────────────────────

PALETTE = "Set2"


def set_style():
    sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi":     130,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "hatch.linewidth": 0.9,
    })


def _save(fig, path: Path, name: str):
    out = path / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Donut chart  (color = language,  hatch = task)
# ──────────────────────────────────────────────────────────────────────────────

def plot_global_donut(df: pd.DataFrame, out: Path,
                      split_label: str = "train",
                      use_weights: bool = False):
    """
    Donut chart:
      - Wedge COLOR  = source language (national flag palette)
      - Wedge HATCH  = task (same task → same hatch in every chart)
      - AST bilingual pairs: outer half = target lang color, inner = source lang color
                             both halves share the same task hatch (///)
      - Slices < 0.5% merged into "Other"
      - Legend includes a language-color key AND a task-hatch key
    """

    def make_label(row):
        task = str(row["task_type"]).strip()
        lang = str(row.get("language", "") or "").strip()
        src  = str(row.get("source_lang", "") or "").strip()
        tgt  = str(row.get("target_lang", "") or "").strip()

        grp = task.upper() if task in ("asr","ast","qa","mqa","aqa") \
              else task.replace("_", " ").title()

        if task == "ast":
            lbl = (f"AST ({src}→{tgt})" if src and tgt
                   else f"AST ({src}→?)" if src else "AST")
            return grp, lbl, src or lang or "unknown", tgt or lang or "unknown", task

        langs = (lang, lang) if lang else ("unknown", "unknown")
        if task == "asr":
            return grp, f"ASR ({lang})" if lang else "ASR", *langs, task
        if task == "qa":
            return grp, f"QA ({lang})"  if lang else "QA",  *langs, task
        if task == "aqa":
            return grp, f"AQA ({lang})" if lang else "AQA", *langs, task
        if task == "mqa":
            return "Music QA", f"Music QA ({lang})" if lang else "Music QA", *langs, task
        if task in ("audio_captioning", "music_captioning"):
            lbl = task.replace("_", " ").title()
            return lbl, f"{lbl} ({lang})" if lang else lbl, *langs, task
        lbl = task.replace("_", " ").title()
        return grp, f"{lbl} ({lang})" if lang else lbl, *langs, task

    df = df.copy()
    info = df.apply(make_label, axis=1)
    df["_grp"]  = [x[0] for x in info]
    df["_lbl"]  = [x[1] for x in info]
    df["_lsrc"] = [x[2] for x in info]
    df["_ltgt"] = [x[3] for x in info]
    df["_tsk"]  = [x[4] for x in info]

    lbl_map = {}
    for _, row in df.iterrows():
        lbl_map[row["_lbl"]] = {
            "group": row["_grp"],
            "l1":    row["_lsrc"],
            "l2":    row["_ltgt"],
            "task":  row["_tsk"],
        }

    # ── Weights ───────────────────────────────────────────────────────────────
    if use_weights:
        gk = df.groupby(["task_type","language","weight_group"])["weight_dataset"].transform("sum")
        df["_wn"] = df["weight_dataset"] / gk.replace(0, 1)
        df["_ew"] = df["weight_group"] * df["_wn"]

    # ── Metric ────────────────────────────────────────────────────────────────
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
        fmt_val       = lambda v: f"{v:,.0f} hr"
        disp_vals     = grp / 3600
    else:
        metric_label  = "samples"
        centre_suffix = "samples total"
        total_display = grp.sum()
        fmt_val       = lambda v: f"{v:,.0f}"
        disp_vals     = grp

    total_val = grp.sum()
    pcts      = 100 * grp / float(total_val)

    # Merge small slices
    main_grp  = grp[pcts >= 0.5].copy()
    small_grp = grp[pcts < 0.5]
    if not small_grp.empty:
        other_lbl = "Other (< 0.5% each)"
        main_grp[other_lbl] = small_grp.sum()
        lbl_map[other_lbl]  = {"group":"Other","l1":"mixed","l2":"mixed","task":"other"}

    main_disp = main_grp / 3600 if use_duration else main_grp
    main_pcts = 100 * main_grp / float(total_val)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig    = plt.figure(figsize=(20, 12))
    ax_d   = fig.add_axes([0.02, 0.06, 0.46, 0.86])
    ax_leg = fig.add_axes([0.50, 0.05, 0.48, 0.92])

    ax_d.set_aspect("equal")
    ax_d.set_xlim(-1.25, 1.25)
    ax_d.set_ylim(-1.30, 1.25)
    ax_d.axis("off")

    # ── Wedges ────────────────────────────────────────────────────────────────
    theta      = 90.0
    total_main = float(main_grp.sum())

    for lbl, val in zip(main_grp.index, main_grp.values):
        d_theta = 360 * val / total_main
        t1, t2  = theta - d_theta, theta
        m       = lbl_map[lbl]
        c1, c2  = lang_color(m["l1"]), lang_color(m["l2"])
        ht      = task_hatch(m["task"])   # hatch = task, not language

        if m["l1"] == m["l2"]:
            ax_d.add_patch(Wedge(
                (0,0), 1.0, t1, t2, width=0.42,
                facecolor=c1, edgecolor="white", linewidth=1.6,
                hatch=ht, alpha=0.93))
        else:
            # bilingual: outer=target lang, inner=source lang, same task hatch
            ax_d.add_patch(Wedge(
                (0,0), 1.00, t1, t2, width=0.21,
                facecolor=c2, edgecolor="white", linewidth=1.4,
                hatch=ht, alpha=0.93))
            ax_d.add_patch(Wedge(
                (0,0), 0.79, t1, t2, width=0.21,
                facecolor=c1, edgecolor="white", linewidth=1.4,
                hatch=ht, alpha=0.93))
        theta = t1

    wt_tag = "\n(Weighted)" if use_weights else ""
    ax_d.text(0,  0.13, f"Split: {split_label}",
              ha="center", va="center", fontsize=15, fontweight="bold", color="#1a1a1a")
    ax_d.text(0, -0.13, f"{fmt_val(total_display)} {centre_suffix}{wt_tag}",
              ha="center", va="center", fontsize=11, color="#555555")
    ax_d.text(0, -1.22,
              "AST pairs — Inner ring: source lang  |  Outer ring: target lang",
              ha="center", va="bottom", fontsize=7.5, color="#666666", style="italic")

    # ── Legend ────────────────────────────────────────────────────────────────
    ax_leg.axis("off")

    leg_data = []
    for lbl, p, dv in zip(main_grp.index, main_pcts.values, main_disp.values):
        m = lbl_map[lbl]
        leg_data.append({
            "group": m["group"], "label": lbl,
            "pct": p, "val": dv,
            "c1": lang_color(m["l1"]), "c2": lang_color(m["l2"]),
            "ht": task_hatch(m["task"]),
        })

    grp_totals = {}
    for item in leg_data:
        grp_totals[item["group"]] = grp_totals.get(item["group"], 0) + item["val"]
    leg_data.sort(key=lambda x: (-grp_totals[x["group"]], x["group"], -x["val"]))

    n_rows = len(leg_data) + len(grp_totals) + 6   # +6 for key rows at bottom
    row_h  = 1.0 / n_rows
    col_xs = [0.00, 0.055, 0.66, 0.84]

    ax_leg.text(col_xs[1], 1.0, "Dataset Group",
                fontsize=10, fontweight="bold", va="top", transform=ax_leg.transAxes)
    ax_leg.text(col_xs[2], 1.0, "%",
                fontsize=10, fontweight="bold", va="top", ha="right", transform=ax_leg.transAxes)
    ax_leg.text(col_xs[3], 1.0, metric_label.title(),
                fontsize=10, fontweight="bold", va="top", ha="right", transform=ax_leg.transAxes)

    y_pos = 1.0 - 2 * row_h
    current_group = None

    for item in leg_data:
        if item["group"] != current_group:
            current_group = item["group"]
            y_pos -= row_h * 0.35
            ax_leg.text(0.0, y_pos + row_h * 0.15, current_group,
                        fontsize=9.5, fontweight="bold", color="#111111",
                        va="center", transform=ax_leg.transAxes)
            y_pos -= row_h

        sw = 0.038
        if item["c1"] == item["c2"]:
            ax_leg.add_patch(plt.Rectangle(
                (col_xs[0], y_pos - row_h * 0.3), sw, row_h * 0.7,
                transform=ax_leg.transAxes, clip_on=False,
                facecolor=item["c1"], hatch=item["ht"], edgecolor="white", linewidth=0.4))
        else:
            ax_leg.add_patch(plt.Rectangle(
                (col_xs[0],      y_pos - row_h * 0.3), sw/2, row_h * 0.7,
                transform=ax_leg.transAxes, clip_on=False,
                facecolor=item["c1"], hatch=item["ht"], edgecolor="white", linewidth=0.4))
            ax_leg.add_patch(plt.Rectangle(
                (col_xs[0]+sw/2, y_pos - row_h * 0.3), sw/2, row_h * 0.7,
                transform=ax_leg.transAxes, clip_on=False,
                facecolor=item["c2"], hatch=item["ht"], edgecolor="white", linewidth=0.4))

        lbl_str = item["label"] if len(item["label"]) <= 48 else item["label"][:46] + "…"
        ax_leg.text(col_xs[1], y_pos + row_h * 0.15, lbl_str,
                    fontsize=8.5, va="center", transform=ax_leg.transAxes, clip_on=False)
        ax_leg.text(col_xs[2], y_pos + row_h * 0.15, f"{item['pct']:.1f}%",
                    fontsize=8.5, va="center", ha="right", transform=ax_leg.transAxes, clip_on=False)
        ax_leg.text(col_xs[3], y_pos + row_h * 0.15, fmt_val(item["val"]),
                    fontsize=8.5, va="center", ha="right", transform=ax_leg.transAxes, clip_on=False)
        y_pos -= row_h

    # ── Key: language colors ──────────────────────────────────────────────────
    y_pos -= row_h * 0.7
    ax_leg.text(0.0, y_pos, "Language colors:",
                fontsize=8, fontweight="bold", va="top", transform=ax_leg.transAxes)
    y_pos -= row_h * 0.5
    langs_seen = sorted(
        {lbl_map[k]["l1"] for k in lbl_map}
        | {lbl_map[k]["l2"] for k in lbl_map}
        - {"mixed", "unknown", ""}
    )
    x_cur = 0.0
    for lng in langs_seen:
        if x_cur > 0.92: break
        ax_leg.add_patch(plt.Rectangle(
            (x_cur, y_pos - row_h * 0.5), 0.024, row_h * 0.45,
            transform=ax_leg.transAxes, clip_on=False,
            facecolor=lang_color(lng), edgecolor="#333", linewidth=0.3))
        ax_leg.text(x_cur + 0.027, y_pos - row_h * 0.25, lng,
                    fontsize=7.5, va="center", transform=ax_leg.transAxes, clip_on=False)
        x_cur += 0.105
    y_pos -= row_h

    # ── Key: task hatches ─────────────────────────────────────────────────────
    y_pos -= row_h * 0.6
    ax_leg.text(0.0, y_pos, "Task hatches:",
                fontsize=8, fontweight="bold", va="top", transform=ax_leg.transAxes)
    y_pos -= row_h * 0.5
    tasks_seen = sorted({lbl_map[k]["task"] for k in lbl_map})
    x_cur = 0.0
    for tsk in tasks_seen:
        if x_cur > 0.92: break
        ax_leg.add_patch(plt.Rectangle(
            (x_cur, y_pos - row_h * 0.5), 0.024, row_h * 0.45,
            transform=ax_leg.transAxes, clip_on=False,
            facecolor="#bbbbbb", hatch=task_hatch(tsk),
            edgecolor="#333", linewidth=0.3))
        ax_leg.text(x_cur + 0.027, y_pos - row_h * 0.25, tsk,
                    fontsize=7.5, va="center", transform=ax_leg.transAxes, clip_on=False)
        x_cur += 0.105

    suffix = " (Weighted)" if use_weights else " (Raw)"
    fig.suptitle(
        f"Dataset composition — {metric_label}{suffix}\n"
        "Color = language  |  Hatch pattern = task",
        fontsize=13, fontweight="bold", y=0.995,
    )

    fname = "00_global_donut_weighted.png" if use_weights else "00_global_donut_raw.png"
    plt.savefig(out / fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  ✓  {out / fname}")


# ──────────────────────────────────────────────────────────────────────────────
# Instruction / response length histogram
# ──────────────────────────────────────────────────────────────────────────────

def plot_instruction_response_lengths(df: pd.DataFrame, out: Path):
    instr = df["avg_instruction_words"].dropna()
    resp  = df["avg_response_words"].dropna()
    if instr.empty and resp.empty:
        print("  ⚠  No word-count data — skipping length plot.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if not instr.empty:
        axes[0].hist(instr, bins=30, color=sns.color_palette(PALETTE)[1], edgecolor="white")
        axes[0].set_xlabel("Avg instruction words")
        axes[0].set_ylabel("# datasets")
        axes[0].set_title("Instruction length distribution")
    if not resp.empty:
        axes[1].hist(resp, bins=30, color=sns.color_palette(PALETTE)[2], edgecolor="white")
        axes[1].set_xlabel("Avg response words")
        axes[1].set_ylabel("# datasets")
        axes[1].set_title("Response length distribution")
    plt.tight_layout()
    _save(fig, out, "01_instruction_response_lengths.png")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Output folder sits NEXT TO this script, not next to the input YAML
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Generate dataset metadata CSV + Markdown report from NeMo YAML.")
    parser.add_argument("yaml_path",
                        help="Path to the input YAML config file.")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: <script_dir>/dataset_analysis).")
    parser.add_argument("--data_root",  default="",
                        help="Substitute for ${oc.env:DATA_FOLDER}.")
    parser.add_argument("--skip_missing", action="store_true",
                        help="Silently skip manifests not found on disk.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else script_dir / "dataset_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 YAML      : {args.yaml_path}")
    print(f"📁 Output    : {out_dir}\n")

    entries = flatten_manifests(args.yaml_path)
    print(f"   Found {len(entries)} manifest entries.\n")

    DATA_FOLDER = args.data_root or os.environ.get("DATA_FOLDER", "")

    rows    = []
    missing = 0

    for i, e in enumerate(entries):
        if i % 5 == 0:
            print(f"  [{i+1}/{len(entries)}] …", end="\r")

        path = e["manifest_filepath"]
        if DATA_FOLDER:
            path = path.replace("${oc.env:DATA_FOLDER}", DATA_FOLDER)
            path = path.replace("${oc.env:DATA_FOLDER}/", DATA_FOLDER.rstrip("/") + "/")

        stats = parse_manifest(
            path,
            task_type   = e["task_type"],
            sub_task    = e["sub_task"],
            language    = e["language"],
            source_lang = e["source_lang"],
            target_lang = e["target_lang"],
        )

        base = {
            "dataset_name":   Path(path).parent.name,
            "split":          Path(path).stem,
            "task_type":      e["task_type"],
            "sub_task":       e["sub_task"],
            "language":       e["language"],
            "source_lang":    e["source_lang"],
            "target_lang":    e["target_lang"],
            "weight_dataset": e["weight_dataset"],
            "weight_group":   e.get("weight_group", e["weight_dataset"]),
        }

        if stats is None:
            missing += 1
            if not args.skip_missing:
                base.update({k: None for k in [
                    "num_audio_segments","num_samples",
                    "total_duration_sec","total_duration_dhms",
                    "min_segment_duration_sec","max_segment_duration_sec",
                    "avg_segment_duration_sec","median_segment_duration_sec",
                    "std_segment_duration_sec",
                    "avg_instruction_words","min_instruction_words","max_instruction_words",
                    "avg_response_words","min_response_words","max_response_words",
                    "num_unique_speakers","avg_audio_sampling_rate","avg_audio_channels",
                ]})
                base["path"]        = path
                base["file_exists"] = False
                rows.append(base)
        else:
            stats.update(base)
            stats["file_exists"] = True
            rows.append(stats)

    print(f"\n   Parsed: {len(entries)-missing} found, {missing} missing.\n")

    df = pd.DataFrame(rows)

    col_order = [
        "dataset_name","split","task_type","sub_task","language",
        "source_lang","target_lang",
        "num_audio_segments","num_samples",
        "total_duration_sec","total_duration_dhms",
        "min_segment_duration_sec","max_segment_duration_sec",
        "avg_segment_duration_sec","median_segment_duration_sec","std_segment_duration_sec",
        "avg_instruction_words","min_instruction_words","max_instruction_words",
        "avg_response_words","min_response_words","max_response_words",
        "num_unique_speakers","avg_audio_sampling_rate","avg_audio_channels",
        "weight_group","weight_dataset","file_exists","path",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    csv_path = out_dir / "dataset_metadata.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV  → {csv_path}  ({len(df)} rows)\n")

    df_vis = df[df.get("file_exists", pd.Series(True, index=df.index)) == True].copy()
    df_vis = df_vis[df_vis["num_samples"].notna()]

    if df_vis.empty:
        print("⚠  No manifest data found on disk. Building report from YAML metadata.")
        df_vis = df.copy()
        df_vis["num_samples"] = df_vis["num_samples"].fillna(0)

    print("🎨 Generating plots …")
    set_style()
    split_label = (df_vis["split"].mode()[0]
                   if "split" in df_vis.columns and not df_vis.empty else "train")

    # Only: 2 donut charts + word-length histogram
    plot_global_donut(df_vis, out_dir, split_label=split_label, use_weights=False)
    plot_global_donut(df_vis, out_dir, split_label=split_label, use_weights=True)
    plot_instruction_response_lengths(df_vis, out_dir)

    print("\n📝 Writing README …")
    write_summary(df_vis, out_dir / "README.md")

    print(f"\n✅ Done!  Outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()