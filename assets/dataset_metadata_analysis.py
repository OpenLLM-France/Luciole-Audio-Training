"""
Dataset Metadata Extractor & Visualizer
Parses a NeMo-style YAML config referencing JSONL manifests,
computes per-dataset statistics, produces a Markdown README report and plots.

Changes vs previous version:
  - Output directory defaults to <script_dir>/dataset_analysis  (same level as script)
  - Donut chart hatches are now PER-TASK (consistent across all charts)
  - Removed: plot_task_distribution, plot_duration_by_lang
  - Added:   per-task detailed dataset tables in the README
  - Added:   Section 6.5 — Actual sampling weights (hierarchical, temperature-smoothed)
             respecting Task → Language → Dataset hierarchy from YAML config weights.
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

def parse_manifest(manifest_paths: list,
                   task_type: str, sub_task: str,
                   language: str, source_lang: str, target_lang: str,
                   dataset_name: str, split: str, note: str):
    durations, instruction_wc, response_wc = [], [], []
    speaker_ids, sampling_rates, channels_list = set(), [], []
    num_samples = 0
    num_segments = 0

    paths_used = []
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
        note    = r.get("note", "")
        rows.append([
            r["dataset_name"],
            r["split"],
            note,
            r["_lang"],
            fmt_num(n_samp),
            pct,
            fmt_hours(dur_sec),
            f"{avg_seg:.1f} s" if avg_seg else "—",
        ])

    t_samp = sub["num_samples"].sum()
    t_dur  = sub["total_duration_sec"].sum(skipna=True)
    t_pct  = f"{100 * t_samp / total_samples:.2f}%" if total_samples else "—"
    rows.append([
        f"**TOTAL ({task.upper()})**", "", "", "",
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
        ["Dataset", "Split", "Note", "Language", "Samples", "% Total", "Duration", "Avg Seg"],
        rows,
    ))
    lines.append("")
    return lines


# ──────────────────────────────────────────────────────────────────────────────
# Sampling weight engine
# ──────────────────────────────────────────────────────────────────────────────

_EPSILON     = 1e-9
_TEMPERATURE = 2.0   # within-language smoothing temperature (T=2 by default)
                      # T=1 → proportional to sqrt(duration)
                      # T=2 → moderate levelling (recommended)
                      # T→∞ → uniform within lang group


def _base_metric(row) -> float:
    """sqrt(duration_sec) if available, else sqrt(num_samples). Never zero."""
    dur = row.get("total_duration_sec")
    if dur is not None and not (isinstance(dur, float) and math.isnan(dur)) and float(dur) > 0:
        return math.sqrt(float(dur))
    n = row.get("num_samples")
    if n is not None and not (isinstance(n, float) and math.isnan(n)) and float(n) > 0:
        return math.sqrt(float(n))
    return _EPSILON


def compute_sampling_weights(df: pd.DataFrame,
                              temperature: float = _TEMPERATURE) -> pd.DataFrame:
    """
    Compute true sampling probabilities from the YAML group weights +
    within-language temperature balancing.

    Hierarchy respected:  Task group weight  →  Language group weight  →  Dataset weight
    Within-language balancing uses:  w_i = sqrt(duration_i) ^ (1/T),  normalised.

    New columns added
    -----------------
    raw_group_weight          float  weight_group value from YAML
    intra_lang_raw_w          float  sqrt(duration) or sqrt(samples)
    intra_lang_temp_w         float  intra_lang_raw_w ^ (1/T)
    intra_lang_norm_w         float  normalised within (task, lang) group  → sums to 1
    dataset_effective_w       float  raw_group_weight × intra_lang_norm_w
    sampling_prob             float  dataset_effective_w / Σ(all dataset_effective_w)
    sampling_prob_pct         float  sampling_prob × 100
    expected_passes_per_epoch float  how many full passes through this dataset per epoch
    """
    df = df.copy()

    for col in ("weight_group", "weight_dataset", "num_samples",
                "total_duration_sec", "task_type", "language",
                "source_lang", "target_lang"):
        if col not in df.columns:
            df[col] = None

    df["raw_group_weight"] = pd.to_numeric(df["weight_group"], errors="coerce").fillna(0.0)

    # Language key consistent with display_lang
    def _lang_key(row):
        task = str(row.get("task_type", "") or "").strip()
        lang = str(row.get("language",  "") or "").strip()
        if task == "ast":
            src = str(row.get("source_lang", "") or "").strip()
            tgt = str(row.get("target_lang", "") or "").strip()
            if src and tgt:
                return f"{src}→{tgt}"
            return src or tgt or "unknown"
        return lang or "unknown"

    df["_lang_key"] = df.apply(_lang_key, axis=1)

    # Base metric and temperature weight
    df["intra_lang_raw_w"]  = df.apply(_base_metric, axis=1)
    df["intra_lang_temp_w"] = df["intra_lang_raw_w"].apply(
        lambda x: x ** (1.0 / temperature)
    )

    # Normalise within each (task, language) group
    grp_sum = df.groupby(["task_type", "_lang_key"])["intra_lang_temp_w"].transform("sum")
    df["intra_lang_norm_w"] = df["intra_lang_temp_w"] / grp_sum.replace(0, _EPSILON)

    # Effective weight = YAML group weight × within-lang normalised weight
    df["dataset_effective_w"] = df["raw_group_weight"] * df["intra_lang_norm_w"]

    # Normalise to true training sampling probability
    total_ew = df["dataset_effective_w"].sum()
    df["sampling_prob"]     = df["dataset_effective_w"] / max(total_ew, _EPSILON)
    df["sampling_prob_pct"] = df["sampling_prob"] * 100.0

    # Estimated passes per epoch:
    #   If the sampler drew total_samples × sampling_prob samples from this dataset,
    #   the dataset would be traversed that many / its own sample count times.
    total_samples_all = float(df["num_samples"].sum(skipna=True)) or 1.0
    df["expected_passes_per_epoch"] = (
        (df["sampling_prob"] * total_samples_all)
        / df["num_samples"].clip(lower=1)
    ).round(2)

    df.drop(columns=["_lang_key"], inplace=True, errors="ignore")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Sampling-weight summary section (Section 6.5)
# ──────────────────────────────────────────────────────────────────────────────

def _sampling_weight_section(df_w: pd.DataFrame,
                              temperature: float = _TEMPERATURE) -> list:
    """
    Generate Markdown lines for Section 6.5.

    One sub-table per task, sorted by language then descending sampling_prob.
    Columns: Dataset | Language | Samples | Duration | Group weight |
             Intra-lang norm weight | Effective weight | Sampling prob % | Est. passes/epoch
    Includes a top-level summary table (task × language aggregate),
    a methodology note, and balance observations.
    """
    lines: list = []

    lines += [
        "## 6.5 Actual Sampling Weights",
        "",
        "> These weights reflect **how often each dataset is actually drawn** during "
        "training, accounting for the full Task → Language → Dataset hierarchy from "
        "the YAML config and within-language balancing.",
        "",
        "### Methodology",
        "",
        f"1. **Group weight** (`weight_group` from YAML) sets the probability budget "
        f"for each Task and Language level.",
        f"2. **Within-language balancing** uses `sqrt(total_duration_sec)` "
        f"(or `sqrt(num_samples)` if duration unavailable) as the raw score, "
        f"then applies **temperature smoothing T={temperature:.1f}**: "
        f"`score_i ^ (1/T)`, normalised within the language group.",
        f"   - T=1 → proportional to √duration (larger datasets dominate).",
        f"   - T=2 (current) → moderate levelling of size differences.",
        f"   - T→∞ → uniform sampling within the language group.",
        f"3. **Effective weight** = group_weight × intra_lang_norm_weight.",
        f"4. **Sampling probability** = effective_weight / Σ(all effective_weights) "
        f"→ sums to 100% across the full training pipeline.",
        f"5. **Est. passes/epoch** ≈ (sampling_prob × total_samples) / dataset_samples "
        f"— how many full sweeps the model makes through this dataset per training epoch.",
        "",
        "---",
        "",
    ]

    # ── Aggregate summary table: Task × Language ──────────────────────────
    def _dlang(row):
        task = str(row.get("task_type", "") or "").strip()
        lang = str(row.get("language",  "") or "").strip()
        if task == "ast":
            src = str(row.get("source_lang", "") or "").strip()
            tgt = str(row.get("target_lang", "") or "").strip()
            if src and tgt: return f"{src}→{tgt}"
            return src or tgt or "unknown"
        return lang or "unknown"

    df_w = df_w.copy()
    df_w["_dl"] = df_w.apply(_dlang, axis=1)

    agg = (df_w.groupby(["task_type", "_dl"])
               .agg(
                   n_datasets=("dataset_name", "nunique"),
                   samples=("num_samples", "sum"),
                   duration_h=("total_duration_sec",
                                lambda x: x.sum(skipna=True) / 3600),
                   group_w=("raw_group_weight", "first"),
                   sampling_pct=("sampling_prob_pct", "sum"),
               )
               .reset_index()
               .sort_values(["task_type", "sampling_pct"], ascending=[True, False]))

    lines.append("### Summary: sampling probability by Task × Language")
    lines.append("")
    lines.append("> Each row = one language group within a task. "
                 "`Sampling %` sums to 100% across the whole table.")
    lines.append("")

    agg_rows = []
    for _, r in agg.iterrows():
        agg_rows.append([
            r["task_type"],
            r["_dl"],
            str(int(r["n_datasets"])),
            fmt_num(r["samples"]),
            f"{r['duration_h']:.1f} h",
            f"{r['group_w']:.4f}",
            f"{r['sampling_pct']:.3f}%",
        ])
    # totals
    agg_rows.append([
        "**TOTAL**", "", "",
        f"**{fmt_num(int(df_w['num_samples'].sum()))}**",
        f"**{fmt_hours(df_w['total_duration_sec'].sum(skipna=True))}**",
        "",
        f"**{df_w['sampling_prob_pct'].sum():.1f}%**",
    ])

    lines.append(_md_table(
        ["Task", "Language", "Datasets", "Samples", "Duration",
         "Group weight", "Sampling %"],
        agg_rows,
    ))
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Per-task detailed tables ───────────────────────────────────────────
    lines.append("### Per-task sampling weight detail")
    lines.append("")

    tasks = sorted(df_w["task_type"].dropna().unique())

    for task in tasks:
        sub = df_w[df_w["task_type"] == task].copy()
        sub["_lang"] = sub.apply(_dlang, axis=1)
        sub = sub.sort_values(["_lang", "sampling_prob_pct"], ascending=[True, False])

        n_ds   = len(sub)
        t_prob = sub["sampling_prob_pct"].sum()

        lines.append(f"#### {task.upper()}")
        lines.append("")
        lines.append(f"> {n_ds} datasets &nbsp;·&nbsp; "
                     f"total sampling share: **{t_prob:.3f}%**")
        lines.append("")

        detail_rows = []
        for _, r in sub.iterrows():
            dur = r.get("total_duration_sec")
            passes = r.get("expected_passes_per_epoch", "—")

            # Flag potential issues
            flag = ""
            sp = float(r["sampling_prob_pct"])
            ep = float(passes) if passes != "—" else None
            if ep is not None and ep > 5:
                flag = " ⚠ over-sampled"
            elif ep is not None and ep < 0.05:
                flag = " ⚑ under-sampled"

            detail_rows.append([
                r["dataset_name"] + flag,
                r["_lang"],
                fmt_num(r.get("num_samples")),
                fmt_hours(dur),
                f"{r['raw_group_weight']:.4f}",
                f"{r['intra_lang_norm_w']:.4f}",
                f"{r['dataset_effective_w']:.5f}",
                f"{sp:.4f}%",
                f"{passes}×" if passes != "—" else "—",
            ])

        # Subtotals per language within this task
        detail_rows.append([
            f"**TOTAL {task.upper()}**", "", "", "",
            "", "", "",
            f"**{t_prob:.3f}%**", "",
        ])

        lines.append(_md_table(
            ["Dataset", "Language", "Samples", "Duration",
             "Group wt", "Intra-lang wt", "Eff. weight",
             "Sampling %", "Est. passes"],
            detail_rows,
        ))
        lines.append("")

    # ── Balance observations ───────────────────────────────────────────────
    lines += ["---", "", "### Balance observations", ""]

    obs = []

    # Over-sampled (passes > 5)
    over = df_w[df_w["expected_passes_per_epoch"] > 5].copy()
    if not over.empty:
        over = over.sort_values("expected_passes_per_epoch", ascending=False)
        names = ", ".join(
            f"**{r['dataset_name']}** ({r['_dl']}, {r['expected_passes_per_epoch']:.1f}×)"
            for _, r in over.head(5).iterrows()
        )
        obs.append(
            f"⚠️  **Over-sampled datasets** (est. passes/epoch > 5 — overfitting risk): "
            f"{names}."
        )

    # Under-sampled (passes < 0.05)
    under = df_w[df_w["expected_passes_per_epoch"] < 0.05].copy()
    if not under.empty:
        names = ", ".join(
            f"**{r['dataset_name']}** ({r['_dl']}, {r['expected_passes_per_epoch']:.3f}×)"
            for _, r in under.head(5).iterrows()
        )
        obs.append(
            f"⚑  **Under-sampled datasets** (est. passes/epoch < 0.05 — wasted data): "
            f"{names}."
        )

    # Most dominant single dataset
    top1 = df_w.nlargest(1, "sampling_prob_pct").iloc[0]
    obs.append(
        f"📌  Most sampled dataset: **{top1['dataset_name']}** "
        f"({top1['_dl']}) at **{top1['sampling_prob_pct']:.3f}%** "
        f"of total training steps."
    )

    # Lang group vs raw share
    lang_cfg_pct  = agg.groupby("_dl")["sampling_pct"].sum()
    raw_lang_pct  = (df_w.groupby("_dl")["num_samples"].sum()
                     / df_w["num_samples"].sum() * 100)
    for lang in lang_cfg_pct.index:
        cfg  = lang_cfg_pct.get(lang, 0.0)
        raw  = raw_lang_pct.get(lang, 0.0)
        diff = cfg - raw
        if abs(diff) > 5:
            direction = "↑ boosted" if diff > 0 else "↓ reduced"
            obs.append(
                f"{'🟢' if diff > 0 else '🟠'}  Language **{lang}**: "
                f"configured at **{cfg:.1f}%** vs raw data share **{raw:.1f}%** "
                f"({direction} by {abs(diff):.1f} pp)."
            )

    if not obs:
        obs.append("✅  No significant balance issues detected.")

    for note in obs:
        lines.append(f"- {note}")

    lines.append("")
    return lines


# ──────────────────────────────────────────────────────────────────────────────
# Suggested weight section
# ──────────────────────────────────────────────────────────────────────────────

def suggest_weights_section(df: pd.DataFrame,
                            temperature: float = _TEMPERATURE) -> list:
    """
    Compute *suggested* YAML weights at all three hierarchy levels:

      Level 1 – Task weight       : relative weight of each task group
                                    (sum to 1.0 across all tasks)
      Level 2 – Language weight   : weight of each language inside its task
                                    (sum to 1.0 *within* each task)
      Level 3 – Dataset weight    : weight of each dataset inside its (task, lang)
                                    (sum to 1.0 *within* each language group)

    Metric: sqrt(total_duration_sec) when available, else sqrt(num_samples).
    Temperature smoothing (T) is applied at every level.

    Returns Markdown lines for a "Section 8: Suggested Weights" block.
    """
    lines: list = []

    lines += [
        "## 8. Suggested Sampling Weights",
        "",
        "> Weights are **automatically computed** at all three YAML hierarchy levels "
        "using `sqrt(total_duration_sec)` (or `sqrt(num_samples)` as fallback), "
        f"with temperature smoothing T={temperature:.1f}.",
        "",
        "> **How to read this section:**",
        "> - **Level 1 (Task weight)** → put on the task-level `group` node.",
        "> - **Level 2 (Language weight)** → put on each language `group` node *inside* the task.",
        "> - **Level 3 (Dataset weight)** → put on each `multimodal_conversation` node "
        "*inside* the language group.",
        "> - Effective probability of a dataset = Task_w × Lang_w × Dataset_w.",
        "",
        "---",
        "",
    ]

    df = df.copy()

    # ── Language key ──────────────────────────────────────────────────────────
    def _lkey(row):
        task = str(row.get("task_type", "") or "").strip()
        lang = str(row.get("language",  "") or "").strip()
        if task == "ast":
            src = str(row.get("source_lang", "") or "").strip()
            tgt = str(row.get("target_lang", "") or "").strip()
            if src and tgt:
                return f"{src}→{tgt}"
            return src or tgt or "unknown"
        return lang or "unknown"

    df["_lkey"] = df.apply(_lkey, axis=1)

    # ── Base metric + temperature smoothing ───────────────────────────────────
    def _base(row):
        dur = row.get("total_duration_sec")
        if dur is not None and not (isinstance(dur, float) and math.isnan(dur)) and float(dur) > 0:
            return math.sqrt(float(dur))
        n = row.get("num_samples")
        if n is not None and not (isinstance(n, float) and math.isnan(n)) and float(n) > 0:
            return math.sqrt(float(n))
        return _EPSILON

    df["_base"]   = df.apply(_base, axis=1)
    df["_temp_w"] = df["_base"].apply(lambda x: x ** (1.0 / temperature))

    # ── Level 3: Dataset weight  (within each task × lang group) ─────────────
    grp_sum = df.groupby(["task_type", "_lkey"])["_temp_w"].transform("sum")
    df["dataset_w"] = df["_temp_w"] / grp_sum.replace(0, _EPSILON)

    # ── Aggregate to (task, lang) ─────────────────────────────────────────────
    tl = (df.groupby(["task_type", "_lkey"])
             .agg(
                 n_datasets   = ("dataset_name",      "nunique"),
                 total_dur_h  = ("total_duration_sec", lambda x: x.sum(skipna=True) / 3600),
                 total_samples= ("num_samples",        "sum"),
                 group_temp_w = ("_temp_w",            "sum"),
             )
             .reset_index())

    # ── Level 2: Language weight (within each task, sums to 1.0 per task) ────
    task_sum_tl = tl.groupby("task_type")["group_temp_w"].transform("sum")
    tl["lang_w"] = tl["group_temp_w"] / task_sum_tl.replace(0, _EPSILON)

    # ── Level 1: Task weight (across all tasks, sums to 1.0 globally) ─────────
    task_agg = (tl.groupby("task_type")["group_temp_w"]
                  .sum()
                  .reset_index()
                  .rename(columns={"group_temp_w": "task_total_w"}))
    grand_total = task_agg["task_total_w"].sum()
    task_agg["task_w"] = task_agg["task_total_w"] / max(grand_total, _EPSILON)
    task_w_map = dict(zip(task_agg["task_type"], task_agg["task_w"]))
    tl["task_w"] = tl["task_type"].map(task_w_map)
    tl["eff_group_prob"] = tl["task_w"] * tl["lang_w"]

    tasks     = sorted(tl["task_type"].dropna().unique())
    tl_sorted = tl.sort_values(["task_type", "lang_w"], ascending=[True, False])

    # ── Level 1 table ─────────────────────────────────────────────────────────
    lines.append("### Level 1 — Task weights")
    lines.append("")
    lines.append("> Put these on the **task-level group nodes**. Sum = 1.0 across all tasks.")
    lines.append("")
    t1_rows = []
    for _, r in task_agg.sort_values("task_w", ascending=False).iterrows():
        t_sub = tl[tl["task_type"] == r["task_type"]]
        t1_rows.append([
            r["task_type"],
            str(int(t_sub["n_datasets"].sum())),
            f"{t_sub['total_dur_h'].sum():.1f} h",
            fmt_num(t_sub["total_samples"].sum()),
            f"{r['task_w']:.6f}",
        ])
    t1_rows.append(["**TOTAL**", "", "", "", f"**{task_agg['task_w'].sum():.4f}**"])
    lines.append(_md_table(
        ["Task", "Datasets", "Duration", "Samples", "Task weight (L1)"],
        t1_rows,
    ))
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Level 2 table ─────────────────────────────────────────────────────────
    lines.append("### Level 2 — Language weights (within each task)")
    lines.append("")
    lines.append("> Put these on **language-level group nodes inside each task**. "
                 "Sum = 1.0 *within* each task.")
    lines.append("")
    t2_rows = []
    for _, r in tl_sorted.iterrows():
        t2_rows.append([
            r["task_type"], r["_lkey"],
            str(int(r["n_datasets"])),
            f"{r['total_dur_h']:.1f} h",
            fmt_num(r["total_samples"]),
            f"{r['task_w']:.4f}",
            f"{r['lang_w']:.6f}",
            f"{r['eff_group_prob'] * 100:.3f}%",
        ])
    lines.append(_md_table(
        ["Task", "Language", "Datasets", "Duration", "Samples",
         "Task w (L1)", "Lang w (L2) ← YAML", "Effective share"],
        t2_rows,
    ))
    lines.append("")
    lines.append("> **Check**: Effective share = Task_w × Lang_w.  All rows sum to 100%.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Level 3 tables (per task) ─────────────────────────────────────────────
    lines.append("### Level 3 — Dataset weights (within each Task × Language group)")
    lines.append("")
    lines.append("> Put these on **individual dataset nodes** (`multimodal_conversation`). "
                 "Sum = 1.0 *within* each language group.")
    lines.append("")
    for task in tasks:
        sub_df = df[df["task_type"] == task].copy()
        sub_df = sub_df.sort_values(["_lkey", "dataset_w"], ascending=[True, False])
        sub_tl = tl[tl["task_type"] == task][["_lkey", "task_w", "lang_w"]]
        sub_df = sub_df.merge(sub_tl, on="_lkey", how="left")
        sub_df["eff_prob_pct"] = (sub_df["task_w"] * sub_df["lang_w"]
                                  * sub_df["dataset_w"] * 100)
        lines.append(f"#### {task.upper()}")
        lines.append("")
        ds_rows = []
        for _, r in sub_df.iterrows():
            ds_rows.append([
                r["dataset_name"], r["_lkey"],
                fmt_num(r.get("num_samples")),
                fmt_hours(r.get("total_duration_sec")),
                f"{r['task_w']:.4f}",
                f"{r['lang_w']:.4f}",
                f"{r['dataset_w']:.6f}",
                f"{r['eff_prob_pct']:.4f}%",
            ])
        lines.append(_md_table(
            ["Dataset", "Language", "Samples", "Duration",
             "Task w (L1)", "Lang w (L2)", "Dataset w (L3) ← YAML", "Effective prob"],
            ds_rows,
        ))
        lines.append("")

    # ── YAML skeleton ─────────────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("### YAML skeleton — full three-level hierarchy")
    lines.append("")
    lines.append("> Copy-paste skeleton. Replace `<path>` with actual manifest paths.")
    lines.append("")
    lines.append("```yaml")
    lines.append("input_cfg:")
    for task in tasks:
        t_rows = tl_sorted[tl_sorted["task_type"] == task]
        tw = task_w_map.get(task, 0.0)
        lines.append(f"  - type: group")
        lines.append(f"    weight: {tw:.6f}   # L1 — {task.upper()}")
        lines.append(f"    tags: {{task: {task}}}")
        lines.append(f"    input_cfg:")
        for _, r in t_rows.iterrows():
            lines.append(f"      - type: group")
            lines.append(f"        weight: {r['lang_w']:.6f}   # L2 — {r['_lkey']} within {task.upper()}")
            lines.append(f"        tags: {{task: {task}, lang: {r['_lkey']}}}")
            lines.append(f"        input_cfg:")
            ds_sub = df[(df["task_type"] == task) & (df["_lkey"] == r["_lkey"])].sort_values(
                "dataset_w", ascending=False)
            for _, dr in ds_sub.iterrows():
                lines.append(f"          - type: multimodal_conversation")
                lines.append(f"            weight: {dr['dataset_w']:.6f}   # L3 — {dr['dataset_name']}")
                lines.append(f"            manifest_filepath: <path>")
        lines.append("")
    lines.append("```")
    lines.append("")

    return lines


# ──────────────────────────────────────────────────────────────────────────────
# Suggested max_steps section
# ──────────────────────────────────────────────────────────────────────────────

def suggest_max_steps_section(df: pd.DataFrame,
                              batch_size: int = 32,
                              grad_accum: int = 1,
                              target_epochs: float = 1.0) -> list:
    """
    Propose a range of max_steps for training based on:
      - total number of training samples
      - effective batch size  = batch_size × grad_accum
      - a configurable target number of epochs (default = 1)
    Also prints conservative (0.5 epoch), recommended (1 epoch), extended (3 epochs)
    and aggressive (5 epochs) estimates.
    """
    lines: list = []

    total_samples = int(df["num_samples"].sum(skipna=True))
    effective_bs  = batch_size * grad_accum
    if effective_bs <= 0:
        effective_bs = 1

    steps_per_epoch = math.ceil(total_samples / effective_bs)

    presets = [
        ("Conservative",   0.5),
        ("Recommended",    1.0),
        ("Extended",       3.0),
        ("Aggressive",     5.0),
    ]

    target_steps = math.ceil(steps_per_epoch * target_epochs)

    lines += [
        "## 9. Suggested `max_steps` for Training",
        "",
        "> Estimates assume a **weighted sampler** (no hard epochs). "
        "`steps_per_epoch` = ⌈total_samples / effective_batch_size⌉.",
        "",
        f"| Parameter             | Value |",
        f"|----------------------|------|",
        f"| Total training samples | **{total_samples:,}** |",
        f"| Batch size (`batch_size`) | **{batch_size}** |",
        f"| Gradient accumulation (`grad_accum`) | **{grad_accum}** |",
        f"| Effective batch size   | **{effective_bs:,}** |",
        f"| Steps per epoch        | **{steps_per_epoch:,}** |",
        f"| Target epochs          | **{target_epochs:.1f}** |",
        f"| **Suggested max_steps** | **{target_steps:,}** |",
        "",
        "---",
        "",
        "### Presets",
        "",
    ]

    preset_rows = []
    for label, ep in presets:
        steps = math.ceil(steps_per_epoch * ep)
        marker = " ← **recommended**" if ep == 1.0 else ""
        preset_rows.append([
            label,
            f"{ep:.1f}",
            f"{steps:,}",
            marker,
        ])
    lines.append(_md_table(
        ["Preset", "Epochs", "max_steps", "Note"],
        preset_rows,
    ))
    lines.append("")

    # Per-task breakdown
    lines.append("### Per-task breakdown")
    lines.append("")
    lines.append("> How many steps are \"effectively spent\" on each task at the "
                 "recommended epoch count (based on raw sample counts, not weights).")
    lines.append("")

    task_grp = (df.groupby("task_type")
                  .agg(samples=("num_samples", "sum"))
                  .reset_index()
                  .sort_values("samples", ascending=False))
    task_grp["pct"] = (100 * task_grp["samples"] / max(total_samples, 1)).round(1)
    task_grp["steps_at_1ep"] = (task_grp["samples"] / effective_bs).apply(math.ceil)

    tb_rows = []
    for _, r in task_grp.iterrows():
        tb_rows.append([
            r["task_type"],
            fmt_num(r["samples"]),
            f"{r['pct']:.1f}%",
            f"{r['steps_at_1ep']:,}",
        ])
    tb_rows.append([
        "**TOTAL**",
        f"**{fmt_num(total_samples)}**",
        "**100.0%**",
        f"**{steps_per_epoch:,}**",
    ])
    lines.append(_md_table(
        ["Task", "Samples", "% of total", "Steps @ 1 epoch"],
        tb_rows,
    ))
    lines.append("")

    lines.append("> **Tip**: When using a weighted sampler, the actual \"effective epochs\" "
                 "per dataset will differ from the raw counts above — see "
                 "Section 6.5 (Est. passes/epoch) for the per-dataset breakdown.")
    lines.append("")

    return lines


# ──────────────────────────────────────────────────────────────────────────────
# Main README writer
# ──────────────────────────────────────────────────────────────────────────────

def write_summary(df: pd.DataFrame, out_path: Path,
                  temperature: float = _TEMPERATURE,
                  suggest_weights: bool = False,
                  show_actual_weights: bool = False,
                  suggest_steps: bool = False,
                  batch_size: int = 32,
                  grad_accum: int = 1,
                  target_epochs: float = 1.0) -> Path:
    """
    Write a README-style Markdown report and print a compact summary to stdout.

    Section 6.5 (Actual Sampling Weights) is computed here using compute_sampling_weights().
    """
    df = df.copy()

    # ── Compute sampling weights ──────────────────────────────────────────
    df_w = compute_sampling_weights(df, temperature=temperature)

    def _dlang(row):
        task = str(row.get("task_type", "") or "").strip()
        lang = str(row.get("language", "") or "").strip()
        if task == "ast":
            src = str(row.get("source_lang", "") or "").strip()
            tgt = str(row.get("target_lang", "") or "").strip()
            if src and tgt:
                return f"{src}→{tgt}"
            return src or tgt or "unknown"
        return lang or "unknown"

    df["display_lang"]   = df.apply(_dlang, axis=1)
    df_w["display_lang"] = df_w.apply(_dlang, axis=1)
    df_w["_dl"]          = df_w["display_lang"]

    # ── Aggregates ────────────────────────────────────────────────────────
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

    # ── Build Markdown ────────────────────────────────────────────────────
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
        *(["6.5. [Actual Sampling Weights](#65-actual-sampling-weights)"] if show_actual_weights else []),
        *(["8. [Suggested Sampling Weights](#8-suggested-sampling-weights)"] if suggest_weights else []),
        *(["9. [Suggested max_steps for Training](#9-suggested-max_steps-for-training)"] if suggest_steps else []),
        "10. [Balance Notes](#10-balance-notes)",
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

    # Section 6
    lines += [
        "", "---", "",
        "## 6. Detailed Dataset Tables per Task", "",
        "> One table per task. Columns: Dataset · Split · Note · Language · "
        "Samples · % of total · Duration · Avg segment duration.",
        "> Sorted by language then dataset name. Last row = task totals.",
        "",
    ]
    for task in tasks:
        lines += _task_section(df, task, total_samples, total_dur_sec)

    # ── Section 6.5 — ACTUAL SAMPLING WEIGHTS (Optional) ──────────────────
    if show_actual_weights:
        lines += ["---", ""]
        lines += _sampling_weight_section(df_w, temperature=temperature)

    # ── Section 8 — Suggested weights (Optional) ────────────────────────
    if suggest_weights:
        lines += ["---", ""]
        lines += suggest_weights_section(df_w, temperature=temperature)

    # ── Section 9 — max_steps recommendation (Optional) ──────────────────
    if suggest_steps:
        lines += ["---", ""]
        lines += suggest_max_steps_section(
            df_w,
            batch_size=batch_size,
            grad_accum=grad_accum,
            target_epochs=target_epochs,
        )

    # Section 10 — Balance Notes (Always last)
    lines += ["---", "", "## 10. Balance Notes", ""]
    if notes:
        for note in notes:
            lines.append(f"- {note}")
    else:
        lines.append("_No notes calculated._")

    lines += ["", "---", "", "_Report generated by `dataset_analysis.py`_"]

    # Write
    md_path = out_path.with_suffix(".md") if out_path.suffix != ".md" else out_path
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  ✓  README → {md_path}")

    # ── Stdout compact summary ─────────────────────────────────────────────
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

    # Sampling weight compact table (optional stdout)
    if show_actual_weights:
        print(f"\n  {'':=<70}")
        print(f"  ACTUAL SAMPLING PROBABILITIES  (T={temperature:.1f}, sqrt-duration balancing)")
        print(f"  {'':=<70}")
        print(f"  {'Task':<8} {'Lang':<8} {'Dataset':<28} {'Samp%':>7}  {'Passes/ep':>9}")
        print("  " + "-" * 65)
        df_w_sorted = df_w.sort_values(
            ["task_type", "_dl", "sampling_prob_pct"], ascending=[True, True, False]
        )
        for _, r in df_w_sorted.iterrows():
            ep = r.get("expected_passes_per_epoch", "?")
            flag = " ⚠" if (isinstance(ep, (int, float)) and ep > 5) else \
                   " ⚑" if (isinstance(ep, (int, float)) and ep < 0.05) else ""
            print(f"  {str(r['task_type']):<8} {str(r['_dl']):<8} "
                  f"{str(r['dataset_name'])[:27]:<28} "
                  f"{r['sampling_prob_pct']:>6.3f}%  {ep:>8}×{flag}")
        print(f"  {'':->65}")
        print(f"  {'TOTAL':>46} {df_w['sampling_prob_pct'].sum():>6.1f}%")

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

    if use_weights:
        gk = df.groupby(["task_type","language","weight_group"])["weight_dataset"].transform("sum")
        df["_wn"] = df["weight_dataset"] / gk.replace(0, 1)
        df["_ew"] = df["weight_group"] * df["_wn"]

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

    main_grp  = grp[pcts >= 0.5].copy()
    small_grp = grp[pcts < 0.5]
    if not small_grp.empty:
        other_lbl = "Other (< 0.5% each)"
        main_grp[other_lbl] = small_grp.sum()
        lbl_map[other_lbl]  = {"group":"Other","l1":"mixed","l2":"mixed","task":"other"}

    main_disp = main_grp / 3600 if use_duration else main_grp
    main_pcts = 100 * main_grp / float(total_val)

    fig    = plt.figure(figsize=(20, 12))
    ax_d   = fig.add_axes([0.02, 0.06, 0.46, 0.86])
    ax_leg = fig.add_axes([0.50, 0.05, 0.48, 0.92])

    ax_d.set_aspect("equal")
    ax_d.set_xlim(-1.25, 1.25)
    ax_d.set_ylim(-1.30, 1.25)
    ax_d.axis("off")

    theta      = 90.0
    total_main = float(main_grp.sum())

    for lbl, val in zip(main_grp.index, main_grp.values):
        d_theta = 360 * val / total_main
        t1, t2  = theta - d_theta, theta
        m       = lbl_map[lbl]
        c1, c2  = lang_color(m["l1"]), lang_color(m["l2"])
        ht      = task_hatch(m["task"])

        if m["l1"] == m["l2"]:
            ax_d.add_patch(Wedge(
                (0,0), 1.0, t1, t2, width=0.42,
                facecolor=c1, edgecolor="white", linewidth=1.6,
                hatch=ht, alpha=0.93))
        else:
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

    n_rows = len(leg_data) + len(grp_totals) + 6
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
    parser.add_argument("--temperature", type=float, default=_TEMPERATURE,
                        help=f"Within-language sampling temperature (default: {_TEMPERATURE}). "
                             "T=1 → proportional to sqrt(duration). T→∞ → uniform.")
    parser.add_argument("--suggest_weights", action="store_true",
                        help="Add Section 8 to the README: auto-computed balanced YAML weights "
                             "for each task/language/dataset group.")
    parser.add_argument("--actual_weights", action="store_true",
                        help="Add Section 6.5 to the README: the actual sampling probabilities "
                             "based on the current weights in your YAML.")
    parser.add_argument("--suggest_steps", action="store_true",
                        help="Add Section 9 to the README: recommended max_steps for training.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Per-GPU batch size used to estimate max_steps (default: 32).")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps (default: 1). "
                             "effective_bs = batch_size × grad_accum.")
    parser.add_argument("--target_epochs", type=float, default=1.0,
                        help="Target number of epochs for max_steps suggestion (default: 1.0).")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else script_dir / "dataset_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 YAML      : {args.yaml_path}")
    print(f"📁 Output    : {out_dir}")
    print(f"🌡  Temperature: {args.temperature}")
    if args.suggest_weights:
        print(f"⚖️  Suggest weights: enabled")
    if args.suggest_steps:
        print(f"📈 Suggest max_steps: enabled  "
              f"(batch={args.batch_size}, grad_accum={args.grad_accum}, "
              f"target_epochs={args.target_epochs})")
    if args.actual_weights:
        print(f"⚖️  Show actual weights: enabled")
    print()

    entries = flatten_manifests(args.yaml_path)
    print(f"   Found {len(entries)} manifest entries.\n")

    DATA_FOLDER = args.data_root or os.environ.get("DATA_FOLDER", "")

    processed_entries = []
    for e in entries:
        path = e["manifest_filepath"]
        if DATA_FOLDER:
            path = path.replace("${oc.env:DATA_FOLDER}", DATA_FOLDER)
            path = path.replace("${oc.env:DATA_FOLDER}/", DATA_FOLDER.rstrip("/") + "/")
        p = Path(path)
        dataset_name = p.parent.name
        stem = p.stem

        split = stem
        note = []
        if stem.startswith("train"):
            split = "train"
            if "recasepunc" in stem:
                note.append("with punctuations")
            if "max30" in stem:
                note.append("max duration is 30s")

        e["path"] = path
        e["dataset_name"] = dataset_name
        e["split"] = split
        e["note"] = ", ".join(note)
        processed_entries.append(e)

    groups = {}
    for e in processed_entries:
        key = (e["dataset_name"], e["split"], e["note"], e["task_type"], e["sub_task"],
               e["language"], e["source_lang"], e["target_lang"])
        if key not in groups:
            groups[key] = []
        groups[key].append(e)

    rows    = []
    missing = 0

    print(f"   Aggregated into {len(groups)} distinct dataset groups.\n")

    for i, (key, group_entries) in enumerate(groups.items()):
        if i % 5 == 0:
            print(f"  [{i+1}/{len(groups)}] …", end="\r")

        paths   = [e["path"] for e in group_entries]
        e_first = group_entries[0]

        stats = parse_manifest(
            manifest_paths = paths,
            task_type      = key[3],
            sub_task       = key[4],
            language       = key[5],
            source_lang    = key[6],
            target_lang    = key[7],
            dataset_name   = key[0],
            split          = key[1],
            note           = key[2]
        )

        base = {
            "dataset_name":   key[0],
            "split":          key[1],
            "note":           key[2],
            "task_type":      key[3],
            "sub_task":       key[4],
            "language":       key[5],
            "source_lang":    key[6],
            "target_lang":    key[7],
            "weight_dataset": e_first["weight_dataset"],
            "weight_group":   e_first.get("weight_group", e_first["weight_dataset"]),
        }

        if stats is None:
            missing += len(paths)
            if not args.skip_missing:
                raise RuntimeError(
                    f"Could not find valid manifests for {key[0]} {key[1]}: {paths}")
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
            base["path"]        = " | ".join(paths)
            base["file_exists"] = False
            rows.append(base)
        else:
            stats.update(base)
            stats["file_exists"] = True
            rows.append(stats)

    print(f"\n   Parsed: {len(entries)-missing} found, {missing} missing.\n")

    df = pd.DataFrame(rows)

    col_order = [
        "dataset_name","split","note","task_type","sub_task","language",
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

    plot_global_donut(df_vis, out_dir, split_label=split_label, use_weights=False)
    plot_global_donut(df_vis, out_dir, split_label=split_label, use_weights=True)
    plot_instruction_response_lengths(df_vis, out_dir)

    print("\n📝 Writing README …")
    write_summary(
        df_vis, out_dir / "README.md",
        temperature=args.temperature,
        suggest_weights=args.suggest_weights,
        suggest_steps=args.suggest_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        target_epochs=args.target_epochs,
    )

    print(f"\n✅ Done!  Outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()