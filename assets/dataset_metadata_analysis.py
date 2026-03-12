"""
Dataset Metadata Extractor & Visualizer
Parses a NeMo-style YAML config referencing JSONL manifests,
computes per-dataset statistics, produces a CSV summary and plots.
"""

import os
import sys
import json
import math
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
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


def flatten_manifests(yaml_path: str):
    """
    Walk the YAML tree and yield dicts:
      { manifest_filepath, task_type, sub_task, language, weight_group, weight_dataset }
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    entries = []

    def recurse(node, inherited_task="", inherited_lang="", inherited_subtask="",
                 inherited_weight=1.0, group_weight=None,
                 inherited_src_lang="", inherited_tgt_lang=""):
        if not isinstance(node, dict):
            return

        # Merge tags — dataset-level tags override group-level (inherited) values
        tags = node.get("tags", {}) or {}
        if isinstance(tags, str):
            tags = {}
        task      = tags.get("task",        inherited_task)
        lang      = tags.get("lang",        inherited_lang)
        sub_task  = tags.get("sub_task",    inherited_subtask)
        # source_lang / target_lang: inherit from parent, override with own tags if present
        src_lang  = tags.get("source_lang", inherited_src_lang)
        tgt_lang  = tags.get("target_lang", inherited_tgt_lang)

        weight = float(node.get("weight", inherited_weight) or inherited_weight)

        ntype = node.get("type", "")
        if ntype == "multimodal_conversation":
            manifest = node.get("manifest_filepath", "")
            entries.append({
                "manifest_filepath": manifest,
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
            children = node.get("input_cfg", [])
            for child in children:
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

def parse_manifest(manifest_path: str, task_type: str, sub_task: str,
                   language: str, source_lang: str, target_lang: str) -> dict | None:
    """
    Read one JSONL manifest and return a stats dict.
    Returns None if the file does not exist.
    """
    p = Path(manifest_path)
    if not p.exists():
        return None

    durations        = []
    instruction_wc   = []
    response_wc      = []
    speaker_ids      = set()
    sampling_rates   = []
    channels_list    = []

    num_samples  = 0
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

            # ── Duration ──────────────────────────────────────────────────
            # Priority 1: top-level fields
            dur = rec.get("duration") or rec.get("audio_duration") or rec.get("dur")

            # Priority 2: scan conversation turns for type="audio" turns
            #   e.g. {"from": "User", "value": "/path/audio.wav", "type": "audio", "duration": 34.475}
            if dur is None:
                convs = rec.get("conversations") or rec.get("conversation") or []
                if isinstance(convs, list):
                    for turn in convs:
                        if not isinstance(turn, dict):
                            continue
                        turn_type = str(turn.get("type", "")).lower()
                        if turn_type == "audio":
                            d = turn.get("duration") or turn.get("audio_duration")
                            if d is not None:
                                dur = d
                                break
                        # also check if value looks like an audio path and duration is on the turn
                        val = str(turn.get("value", ""))
                        if any(val.endswith(ext) for ext in (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a")):
                            d = turn.get("duration") or turn.get("audio_duration")
                            if d is not None:
                                dur = d
                                break

            # Priority 3: nested context block (NeMo multimodal format)
            if dur is None:
                ctx = rec.get("context") or {}
                if isinstance(ctx, dict):
                    dur = ctx.get("duration") or ctx.get("audio_duration")

            # Priority 4: input_values.audio.duration (HuggingFace-style)
            if dur is None:
                iv = rec.get("input_values") or {}
                if isinstance(iv, dict):
                    audio_obj = iv.get("audio") or {}
                    if isinstance(audio_obj, dict):
                        dur = audio_obj.get("duration")

            # Priority 5: rec.audio.duration
            if dur is None:
                audio_obj = rec.get("audio") or {}
                if isinstance(audio_obj, dict):
                    dur = audio_obj.get("duration") or audio_obj.get("audio_duration")

            if dur is not None:
                try:
                    durations.append(float(dur))
                    num_segments += 1
                except (ValueError, TypeError):
                    pass

            # ── Audio meta ────────────────────────────────────────────────
            ctx = rec.get("context") or {}
            sr = (rec.get("sample_rate") or rec.get("sampling_rate")
                  or (ctx.get("sample_rate") if isinstance(ctx, dict) else None))
            if sr:
                try:
                    sampling_rates.append(int(sr))
                except (ValueError, TypeError):
                    pass
            ch = (rec.get("num_channels") or rec.get("channels")
                  or (ctx.get("num_channels") if isinstance(ctx, dict) else None))
            if ch:
                try:
                    channels_list.append(int(ch))
                except (ValueError, TypeError):
                    pass

            # ── Speaker ───────────────────────────────────────────────────
            spk = (rec.get("speaker") or rec.get("speaker_id")
                   or rec.get("speaker_id_str")
                   or (ctx.get("speaker_id") if isinstance(ctx, dict) else None))
            if spk:
                speaker_ids.add(str(spk))

            # ── Conversations / instructions / responses ──────────────────
            conversations = rec.get("conversations", rec.get("conversation", []))
            if isinstance(conversations, list) and conversations:
                for turn in conversations:
                    role  = turn.get("role",  turn.get("from", "")).lower()
                    value = turn.get("value", turn.get("content", turn.get("text", "")))
                    wc    = word_count(value)
                    if role in ("user", "human", "instruction", "input"):
                        instruction_wc.append(wc)
                    elif role in ("assistant", "gpt", "system", "output", "response", "bot"):
                        response_wc.append(wc)
            else:
                # Flat format
                for key in ("question", "instruction", "input"):
                    v = rec.get(key)
                    if v:
                        instruction_wc.append(word_count(str(v)))
                        break
                for key in ("answer", "response", "output", "text"):
                    v = rec.get(key)
                    if v:
                        response_wc.append(word_count(str(v)))
                        break

    if num_samples == 0:
        return None

    def safe_stat(arr, fn):
        return round(fn(arr), 3) if arr else None

    def median(arr):
        return float(np.median(arr))

    def std(arr):
        return float(np.std(arr))

    dataset_name = p.parent.name   # e.g. "FLEURS", "CommonVoice"
    split        = p.stem           # e.g. "train", "train_recasepunc"

    return {
        "dataset_name":               dataset_name,
        "split":                      split,
        "task_type":                  task_type,
        "sub_task":                   sub_task,
        "language":                   language,
        "source_lang":                source_lang,
        "target_lang":                target_lang,
        "num_audio_segments":         num_segments,
        "num_samples":                num_samples,
        "total_duration_sec":         round(sum(durations), 2)    if durations else None,
        "total_duration_dhms":        seconds_to_dhms(sum(durations)) if durations else None,
        "min_segment_duration_sec":   safe_stat(durations, min),
        "max_segment_duration_sec":   safe_stat(durations, max),
        "avg_segment_duration_sec":   safe_stat(durations, lambda x: sum(x)/len(x)),
        "median_segment_duration_sec":safe_stat(durations, median),
        "std_segment_duration_sec":   safe_stat(durations, std),
        "avg_instruction_words":      safe_stat(instruction_wc, lambda x: sum(x)/len(x)),
        "min_instruction_words":      safe_stat(instruction_wc, min),
        "max_instruction_words":      safe_stat(instruction_wc, max),
        "avg_response_words":         safe_stat(response_wc, lambda x: sum(x)/len(x)),
        "min_response_words":         safe_stat(response_wc, min),
        "max_response_words":         safe_stat(response_wc, max),
        "num_unique_speakers":        len(speaker_ids) if speaker_ids else None,
        "avg_audio_sampling_rate":    round(sum(sampling_rates)/len(sampling_rates)) if sampling_rates else None,
        "avg_audio_channels":         round(sum(channels_list)/len(channels_list), 2) if channels_list else None,
        "path":                       str(p),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

PALETTE = "Set2"

def set_style():
    sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 130,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })


def save(fig, path: Path, name: str):
    out = path / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out}")


def plot_task_distribution(df: pd.DataFrame, out: Path):
    counts = df.groupby("task_type")["num_samples"].sum().sort_values(ascending=False)
    
    # Split into low (< 1M) and high (>= 1M)
    counts_low  = counts[counts < 1000000]
    counts_high = counts[counts >= 1000000]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    def format_val(v):
        if v >= 1000000:
            return f"{v/1000000:.1f}M"
        if v >= 1000:
            return f"{v/1000:.0f}K"
        return str(int(v))

    # Plot low counts
    if not counts_low.empty:
        bars = axes[0].barh(counts_low.index, counts_low.values,
                           color=sns.color_palette(PALETTE, len(counts_low)))
        axes[0].bar_label(bars, labels=[format_val(v) for v in counts_low.values], padding=4)
        axes[0].set_xlabel("Total samples")
        axes[0].set_title("Task distribution (< 1M samples)")
        axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format_val(x)))
        axes[0].invert_yaxis()
    else:
        axes[0].text(0.5, 0.5, "No tasks < 1M samples", ha="center", va="center")

    # Plot high counts
    if not counts_high.empty:
        bars = axes[1].barh(counts_high.index, counts_high.values,
                           color=sns.color_palette(PALETTE, len(counts_high)))
        axes[1].bar_label(bars, labels=[format_val(v) for v in counts_high.values], padding=4)
        axes[1].set_xlabel("Total samples")
        axes[1].set_title("Task distribution (>= 1M samples)")
        axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format_val(x)))
        axes[1].invert_yaxis()
    else:
        axes[1].text(0.5, 0.5, "No tasks >= 1M samples", ha="center", va="center")

    plt.tight_layout()
    save(fig, out, "01_task_distribution.png")


# def plot_dataset_per_task(df: pd.DataFrame, out: Path):
#     """
#     One subplot per task. Each subplot shows a horizontal bar per dataset
#     (sorted by samples desc) so individual contributions are clearly readable.
#     """
#     tasks = sorted(df["task_type"].dropna().unique())
#     if not tasks:
#         return

#     ncols   = 2
#     nrows   = math.ceil(len(tasks) / ncols)

#     # Per-task dataset counts to size subplot heights proportionally
#     task_ds_counts = {t: df[df["task_type"] == t]["dataset_name"].nunique() for t in tasks}
#     row_unit   = 0.40   # inches per bar
#     min_h      = 2.0

#     subplot_heights = []
#     fQAor ri in range(nrows):
#         h = min_h
#         for ci in range(ncols):
#             idx = ri * ncols + ci
#             if idx < len(tasks):
#                 h = max(h, task_ds_counts[tasks[idx]] * row_unit + 1.5)
#         subplot_heights.append(h)

#     total_h = sum(subplot_heights) + 1.5
#     fig = plt.figure(figsize=(22, total_h))
#     from matplotlib.gridspec import GridSpec
#     gs = GridSpec(nrows, ncols, figure=fig,
#                   height_ratios=subplot_heights, hspace=0.55, wspace=0.45)

#     # Build a stable per-dataset color map across all subplots
#     all_ds = sorted(df["dataset_name"].dropna().unique())
#     palette = (sns.color_palette("tab20", 20) +
#                sns.color_palette("tab20b", 20) +
#                sns.color_palette("tab20c", 20))
#     ds_colors = {ds: palette[i % len(palette)] for i, ds in enumerate(all_ds)}

#     for idx, task in enumerate(tasks):
#         ri = idx // ncols
#         ci = idx  % ncols
#         ax = fig.add_subplot(gs[ri, ci])

#         sub = (df[df["task_type"] == task]
#                .groupby("dataset_name")["num_samples"]
#                .sum()
#                .dropna()
#                .sort_values(ascending=True))   # ascending → biggest at top for barh

#         if sub.empty:
#             ax.set_visible(False)
#             continue

#         colors = [ds_colors[ds] for ds in sub.index]
#         bars   = ax.barh(sub.index, sub.values, color=colors, edgecolor="white", linewidth=0.4)

#         # Value label at end of each bar
#         x_max = sub.values.max()
#         for bar, val in zip(bars, sub.values):
#             ax.text(val + x_max * 0.01, bar.get_y() + bar.get_height() / 2,
#                     f"{val:,.0f}", va="center", ha="left", fontsize=7.5, color="#333333")

#         ax.set_title(f"Task: {task}", fontsize=11, fontweight="bold", pad=6)
#         ax.set_xlabel("Samples", fontsize=9)
#         ax.set_xlim(right=x_max * 1.22)
#         ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
#         ax.xaxis.grid(True, linestyle="--", alpha=0.45)
#         ax.set_axisbelow(True)
#         ax.tick_params(axis="y", labelsize=8)

#     for idx in range(len(tasks), nrows * ncols):
#         fig.add_subplot(gs[idx // ncols, idx % ncols]).set_visible(False)

#     fig.suptitle("Dataset contribution per task", fontsize=15, fontweight="bold", y=1.01)
#     plt.savefig(out / "02_dataset_per_task.png", bbox_inches="tight", dpi=130)
#     plt.close(fig)
#     print(f"  ✓  {out / '02_dataset_per_task.png'}")


def plot_language_per_dataset(df: pd.DataFrame, out: Path):
    counts = df.groupby(["dataset_name", "language"])["num_samples"].sum().reset_index()
    # Limit to top-30 datasets by total samples for readability
    top = (df.groupby("dataset_name")["num_samples"].sum()
             .nlargest(30).index.tolist())
    counts = counts[counts["dataset_name"].isin(top)]
    pivot  = counts.pivot_table(index="dataset_name", columns="language",
                                 values="num_samples", aggfunc="sum", fill_value=0)
    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.45)))
    pivot.plot(kind="barh", stacked=True, ax=ax, colormap=PALETTE)
    ax.set_xlabel("Samples")
    ax.set_title("Language distribution per dataset (top 30 by samples)")
    ax.legend(title="Language", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.invert_yaxis()
    save(fig, out, "04_language_per_dataset.png")

def plot_instruction_response_lengths(df: pd.DataFrame, out: Path):
    instr = df["avg_instruction_words"].dropna()
    resp  = df["avg_response_words"].dropna()
    if instr.empty and resp.empty:
        print("  ⚠  No word-count data – skipping length plot.")
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
    save(fig, out, "03_instruction_response_lengths.png")


def plot_global_donut(df: pd.DataFrame, out: Path, split_label: str = "train", use_weights: bool = False):
    import matplotlib.patches as mpatches
    """
    Donut chart with clean, non-overlapping outside labels.
    Handles extreme data imbalance by grouping slices < 0.5% into "Other".
    Assigns one color per language. For pairs (e.g. AST en->fr), draws two colored rings.
    If use_weights is True, it scales the sizes by effective dataset weights.
    """
    def make_label(row):
        task = str(row["task_type"]).strip()
        lang = str(row.get("language", "") or "").strip()
        src  = str(row.get("source_lang", "") or "").strip()
        tgt  = str(row.get("target_lang", "") or "").strip()
        
        task_group = task.upper() if task in ("asr", "ast", "qa", "mqa", "aqa") else task.replace("_", " ").title()
        
        if task == "ast":
            if src and tgt:
                return task_group, f"AST ({src}→{tgt})", src, tgt
            elif src:
                return task_group, f"AST ({src}→?)", src, "unknown"
            elif lang:
                return task_group, f"AST ({lang})", lang, lang
            return task_group, "AST", "unknown", "unknown"
        
        langs = (lang, lang) if lang else ("unknown", "unknown")
        
        if task == "asr":
            return task_group, f"ASR ({lang})" if lang else "ASR", *langs
        if task == "qa":
            return task_group, f"QA ({lang})" if lang else "QA", *langs
        if task =="aqa":
            return task_group, f"AQA ({lang})" if lang else "AQA", *langs
        if task == "mqa":
            return "Music QA", f"Music QA ({lang})" if lang else "Music QA", *langs
        if task == "audio_captioning":
            return "Audio Captioning", f"Audio Captioning ({lang})" if lang else "Audio Captioning", *langs
        if task == "music_captioning":
            return "Music Captioning", "Music Captioning", *langs
        
        label = task.replace("_", " ").title()
        return task_group, f"{label} ({lang})" if lang else label, *langs

    df = df.copy()
    labels_info = df.apply(make_label, axis=1)
    df["group"] = [x[0] for x in labels_info]
    df["label"] = [x[1] for x in labels_info]
    df["lang_source"] = [x[2] for x in labels_info]
    df["lang_target"] = [x[3] for x in labels_info]

    # Calculate effective weights if requested
    if use_weights:
        group_key = df.groupby(["task_type", "language", "weight_group"])["weight_dataset"].transform("sum")
        df["weight_dataset_norm"] = df["weight_dataset"] / group_key.replace(0, 1)
        df["effective_weight"] = df["weight_group"] * df["weight_dataset_norm"]

    # Extract single language mapping for each unique label
    label_info_map = {}
    for idx, row in df.iterrows():
        label_info_map[row["label"]] = {
            "group": row["group"],
            "l1": row["lang_source"],
            "l2": row["lang_target"]
        }

    # ── Choose metric ────────────────────────────────────────────────────────
    dur_sum = df.groupby("label")["total_duration_sec"].sum().fillna(0)
    use_duration = dur_sum.sum() > 0

    if use_weights:
        # Effective weight represents the probability of a drawing a single sample from this dataset in the dataloader.
        # To find the equivalent total samples, we imagine drawing N total samples (where N = raw total samples).
        total_samples_all = df["num_samples"].sum()
        df["equiv_samples"] = df["effective_weight"] * total_samples_all
        
        if use_duration:
            # Expected duration per sample
            df["avg_dur"] = df["total_duration_sec"] / df["num_samples"].clip(lower=1)
            # The equivalent total hours = (equivalent samples) * (average duration per sample)
            df["weighted_val"] = df["equiv_samples"] * df["avg_dur"]
            
            grp = df.groupby("label")["weighted_val"].sum().fillna(0)
            
            # Rescale the sum to explicitly match the raw total hours so the numbers look comparable
            total_raw = dur_sum.sum()
            total_eff = grp.sum()
            scale = total_raw / total_eff if total_eff > 0 else 1.0
            grp = grp * scale 
        else:
            df["weighted_val"] = df["equiv_samples"]
            grp = df.groupby("label")["weighted_val"].sum().fillna(0)
    else:
        if use_duration:
            grp = dur_sum
        else:
            grp = df.groupby("label")["num_samples"].sum().fillna(0)

    grp = grp[grp > 0].sort_values(ascending=False)

    if use_duration:
        metric_label = "audio hours"
        centre_suffix = "hr total"
        display_vals = grp / 3600
        total_display = grp.sum() / 3600
        fmt_val = lambda v: f"{v:,.0f} hr"
    else:
        metric_label = "samples"
        centre_suffix = "samples total"
        display_vals = grp
        total_display = grp.sum()
        fmt_val = lambda v: f"{v:,.0f}"

    if grp.empty or grp.sum() == 0:
        print("  ⚠  No data (duration or samples) for donut chart.")
        return

    total_val = grp.sum()
    pcts = 100 * grp / float(total_val)
    
    # ── Grouping < 0.5% ──────────────────────────────────────────────────────
    threshold = 0.5
    main_grp = grp[pcts >= threshold].copy()
    small_grp = grp[pcts < threshold]
    has_small = not small_grp.empty
    
    if has_small:
        other_lbl = "Other (< 0.5% each)"
        other_val = small_grp.sum()
        main_grp[other_lbl] = other_val
        label_info_map[other_lbl] = {"group": "Other", "l1": "mixed", "l2": "mixed"}
    
    main_display_vals = main_grp / 3600 if use_duration else main_grp
    main_pcts = 100 * main_grp / float(total_val)

    # ── Assign language colors ───────────────────────────────────────────────
    unique_langs = set()
    for info in label_info_map.values():
        unique_langs.add(info["l1"])
        unique_langs.add(info["l2"])
    
    # Exclude special "mixed" and "unknown" from the palette, assign them explicitly
    special_langs = {"mixed", "unknown", ""}
    lang_list = sorted([l for l in unique_langs if l not in special_langs])
    
    palette = sns.color_palette("tab20", 20) + sns.color_palette("tab20b", 20) + sns.color_palette("tab20c", 20)
    lang_colors = {lang: palette[i % len(palette)] for i, lang in enumerate(lang_list)}
    lang_colors["mixed"] = (0.7, 0.7, 0.7) # Gray
    lang_colors["unknown"] = (0.8, 0.8, 0.8)
    lang_colors[""] = (0.8, 0.8, 0.8)

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 11))
    ax_donut = fig.add_axes([0.02, 0.05, 0.45, 0.88])
    ax_leg = fig.add_axes([0.5, 0.05, 0.45, 0.9])

    # ── Main Donut (Nested for bilingual) ────────────────────────────────────
    ax_donut.set_aspect("equal")
    ax_donut.set_xlim(-1.2, 1.2)
    ax_donut.set_ylim(-1.2, 1.2)
    ax_donut.axis("off")

    theta = 90  # start angle (counterclock=False, so we subtract)
    total_val_main = main_grp.sum()

    for lbl, val in zip(main_grp.index, main_grp.values):
        d_theta = 360 * val / total_val_main
        theta1 = theta - d_theta
        theta2 = theta
        
        info = label_info_map[lbl]
        c1 = lang_colors[info["l1"]]
        c2 = lang_colors[info["l2"]]
        
        if c1 == c2:
            # Single solid wedge width 0.4
            w = mpatches.Wedge((0, 0), 1.0, theta1, theta2, width=0.4, 
                               facecolor=c1, edgecolor="white", linewidth=2.0)
            ax_donut.add_patch(w)
        else:
            # Bilingual: dual nested wedges (Outer=target, Inner=source)
            w_out = mpatches.Wedge((0, 0), 1.0, theta1, theta2, width=0.2, 
                                   facecolor=c2, edgecolor="white", linewidth=2.0)
            w_in  = mpatches.Wedge((0, 0), 0.8, theta1, theta2, width=0.2, 
                                   facecolor=c1, edgecolor="white", linewidth=2.0)
            ax_donut.add_patch(w_out)
            ax_donut.add_patch(w_in)
            
        theta = theta1

    ax_donut.text(0,  0.11, f"Split: {split_label}",
                  ha="center", va="center", fontsize=15, fontweight="bold", color="#1a1a1a")
    ax_donut.text(0, -0.12, f"{fmt_val(total_display)} {centre_suffix}" + ("\n(Scaled)" if use_weights else ""),
                  ha="center", va="center", fontsize=12, color="#555555")

    # ── Hierarchical Legend ──────────────────────────────────────────────────
    ax_leg.axis("off")
    
    # Organize data for legend
    leg_data = []
    for lbl, p, dv in zip(main_grp.index, main_pcts.values, main_display_vals.values):
        info = label_info_map[lbl]
        c1 = lang_colors[info["l1"]]
        c2 = lang_colors[info["l2"]]
        leg_data.append({"group": info["group"], "label": lbl, "pct": p, "val": dv, "c1": c1, "c2": c2})
        
    group_sizes = {}
    for item in leg_data:
        group_sizes[item["group"]] = group_sizes.get(item["group"], 0) + item["val"]
        
    leg_data.sort(key=lambda x: (-group_sizes[x["group"]], x["group"], -x["val"]))

    # Render Legend
    row_h  = 1.0 / (len(leg_data) + len(group_sizes) + 2)
    col_xs = [0.0, 0.06, 0.65, 0.82]   # swatch | label | pct | value

    ax_leg.text(col_xs[1], 1.0, "Group",  fontsize=11, fontweight="bold", va="top", transform=ax_leg.transAxes)
    ax_leg.text(col_xs[2], 1.0, "%",      fontsize=11, fontweight="bold", va="top", ha="right", transform=ax_leg.transAxes)
    ax_leg.text(col_xs[3], 1.0, metric_label.title(), fontsize=11, fontweight="bold", va="top", ha="right", transform=ax_leg.transAxes)

    y_pos = 1.0 - 2 * row_h
    current_group = None
    
    for item in leg_data:
        # Print Group Header
        if item["group"] != current_group:
            current_group = item["group"]
            y_pos -= row_h * 0.5
            ax_leg.text(0.0, y_pos + row_h*0.15, current_group, fontsize=10, fontweight="bold", color="#1a1a1a", va="center", transform=ax_leg.transAxes)
            y_pos -= row_h
            
        # Draw Swatch
        if item["c1"] == item["c2"]:
            # One color swatch
            rect = plt.Rectangle((col_xs[0], y_pos - row_h * 0.3),
                                   0.035, row_h * 0.7,
                                   transform=ax_leg.transAxes,
                                   color=item["c1"], clip_on=False)
            ax_leg.add_patch(rect)
        else:
            # Two color split swatch
            rect_inner = plt.Rectangle((col_xs[0], y_pos - row_h * 0.3),
                                   0.0175, row_h * 0.7,
                                   transform=ax_leg.transAxes,
                                   color=item["c1"], clip_on=False)
            rect_outer = plt.Rectangle((col_xs[0] + 0.0175, y_pos - row_h * 0.3),
                                   0.0175, row_h * 0.7,
                                   transform=ax_leg.transAxes,
                                   color=item["c2"], clip_on=False)
            ax_leg.add_patch(rect_inner)
            ax_leg.add_patch(rect_outer)
            
        # Label
        lbl_str = item["label"] if len(item["label"]) <= 45 else item["label"][:43] + "…"
        ax_leg.text(col_xs[1], y_pos + row_h * 0.15, lbl_str,
                    fontsize=8.5, va="center", transform=ax_leg.transAxes, clip_on=False)
        # Pct
        ax_leg.text(col_xs[2], y_pos + row_h * 0.15, f"{item['pct']:.1f}%",
                    fontsize=8.5, va="center", ha="right", transform=ax_leg.transAxes, clip_on=False)
        # Value
        ax_leg.text(col_xs[3], y_pos + row_h * 0.15, fmt_val(item["val"]),
                    fontsize=8.5, va="center", ha="right", transform=ax_leg.transAxes, clip_on=False)
        
        y_pos -= row_h

    title_suffix = " (Weighted)" if use_weights else " (Raw)"
    fig.suptitle(f"Dataset composition by task & language \u2014 {metric_label}{title_suffix}",
                 fontsize=14, fontweight="bold", y=0.99)
    
    out_filename = "00_global_donut_weighted.png" if use_weights else "00_global_donut_raw.png"
    plt.savefig(out / out_filename, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  ✓  {out / out_filename}")


def plot_duration_by_lang(df: pd.DataFrame, out: Path):
    df = df.copy()
    def get_lang_label(row):
        task = str(row["task_type"]).strip()
        if task == "ast":
            src = str(row.get("source_lang", "") or "").strip()
            tgt = str(row.get("target_lang", "") or "").strip()
            if src and tgt:
                return f"{src} \u2192 {tgt}"
            return src or tgt or "unknown"
        return str(row.get("language", "unknown"))

    df["display_lang"] = df.apply(get_lang_label, axis=1)
    grp = df.groupby(["display_lang", "task_type"])["total_duration_sec"].sum().reset_index()
    grp["total_duration_hr"] = grp["total_duration_sec"] / 3600
    if grp.empty:
        return
    pivot = grp.pivot_table(index="display_lang", columns="task_type",
                             values="total_duration_hr", aggfunc="sum", fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind="bar", ax=ax, colormap="tab10", width=0.75)
    ax.set_xlabel("Language")
    ax.set_ylabel("Hours")
    ax.set_title("Total audio duration (hours) by language/pair and task")
    ax.legend(title="Task", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.xticks(rotation=45, ha="right")
    save(fig, out, "02_duration_by_language_task.png")


def plot_weighted_vs_raw_samples(df: pd.DataFrame, out: Path):
    """
    One subplot per task_type. Each subplot shows paired horizontal bars
    (raw vs weighted) for every dataset belonging to that task.
    Datasets are sorted by raw samples descending within each subplot.
    """
    df = df[df["num_samples"].notna()].copy()
    if df.empty:
        print("  \u26a0  No sample data for weighted vs raw plot.")
        return

    # ── Effective weight: group_weight * (ds_weight / sum_in_group) ────────
    group_key = df.groupby(["task_type", "language", "weight_group"])["weight_dataset"].transform("sum")
    df["weight_dataset_norm"] = df["weight_dataset"] / group_key.replace(0, 1)
    df["effective_weight"]    = df["weight_group"] * df["weight_dataset_norm"]

    # Global scale: weighted totals == raw totals (comparable axis)
    total_raw = df["num_samples"].sum()
    total_eff = df["effective_weight"].sum()
    scale     = total_raw / total_eff if total_eff > 0 else 1.0
    df["weighted_samples"] = df["effective_weight"] * scale

    # Readable dataset label = name + lang tag
    def row_label(r):
        lang = str(r.get("language", "") or "").strip()
        name = str(r["dataset_name"])
        return f"{name} ({lang})" if lang else name

    df["label"] = df.apply(row_label, axis=1)

    tasks      = sorted(df["task_type"].dropna().unique())
    n_tasks    = len(tasks)
    if n_tasks == 0:
        return

    colors   = sns.color_palette("tab10")
    c_raw    = colors[0]   # blue
    c_wgt    = colors[1]   # orange

    # ── Layout: 2 columns of subplots ──────────────────────────────────────
    ncols   = 2
    nrows   = math.ceil(n_tasks / ncols)

    # Pre-compute per-task dataset counts to size row heights proportionally
    task_counts = {t: len(df[df["task_type"] == t]) for t in tasks}
    max_count   = max(task_counts.values(), default=1)

    # Each subplot height = proportional to its dataset count (min 2 rows)
    row_unit   = 0.55          # inches per dataset row
    min_height = 2.0
    subplot_heights = []
    for row_i in range(nrows):
        h = min_height
        for col_i in range(ncols):
            idx = row_i * ncols + col_i
            if idx < n_tasks:
                h = max(h, task_counts[tasks[idx]] * row_unit + 1.2)
        subplot_heights.append(h)

    total_height = sum(subplot_heights) + 1.5   # title space
    fig = plt.figure(figsize=(22, total_height))

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(nrows, ncols, figure=fig,
                  height_ratios=subplot_heights,
                  hspace=0.55, wspace=0.35)

    for idx, task in enumerate(tasks):
        row_i = idx // ncols
        col_i = idx  % ncols
        ax    = fig.add_subplot(gs[row_i, col_i])

        sub = df[df["task_type"] == task].sort_values("num_samples", ascending=True).copy()
        n   = len(sub)
        if n == 0:
            ax.set_visible(False)
            continue

        y       = np.arange(n)
        bar_h   = 0.36
        raw_vals = sub["num_samples"].values
        wgt_vals = sub["weighted_samples"].values
        x_max_val = max(raw_vals.max(), wgt_vals.max())

        ax.barh(y + bar_h / 2, raw_vals, height=bar_h,
                color=c_raw, alpha=0.88, label="Raw")
        ax.barh(y - bar_h / 2, wgt_vals, height=bar_h,
                color=c_wgt, alpha=0.88, label="Weighted")

        # Value labels + ratio badge
        for i, (rv, wv) in enumerate(zip(raw_vals, wgt_vals)):
            pad = x_max_val * 0.012
            ax.text(rv + pad, y[i] + bar_h / 2,
                    f"{rv:,.0f}", va="center", ha="left", fontsize=7.5, color="#222222")
            ax.text(wv + pad, y[i] - bar_h / 2,
                    f"{wv:,.0f}", va="center", ha="left", fontsize=7.5, color="#444444")

            ratio  = wv / rv if rv > 0 else 1.0
            diff   = ratio - 1.0
            badge_color  = "#1a7a1a" if diff >= 0 else "#c0392b"
            badge_symbol = "\u25b2" if diff >= 0 else "\u25bc"
            ax.text(x_max_val * 1.02 + pad * 6, y[i],
                    f"{badge_symbol}{abs(diff)*100:.0f}%",
                    va="center", ha="left", fontsize=7, color=badge_color, fontweight="bold")

        ax.set_yticks(y)
        ax.set_yticklabels(sub["label"].tolist(), fontsize=8)
        ax.set_xlabel("Samples", fontsize=9)
        ax.set_title(f"Task: {task}", fontsize=11, fontweight="bold", pad=6)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.xaxis.grid(True, linestyle="--", alpha=0.45)
        ax.set_axisbelow(True)
        ax.set_xlim(right=x_max_val * 1.25)

        if idx == 0:
            ax.legend(fontsize=8.5, loc="lower right",
                      handles=[
                          plt.Rectangle((0,0),1,1, color=c_raw, alpha=0.88, label="Raw samples"),
                          plt.Rectangle((0,0),1,1, color=c_wgt, alpha=0.88, label="Weighted samples"),
                      ])

    # Hide unused subplots
    for idx in range(n_tasks, nrows * ncols):
        fig.add_subplot(gs[idx // ncols, idx % ncols]).set_visible(False)

    fig.suptitle(
        "Raw vs Weighted samples per dataset — by task\n"
        "(weighted = raw \u00d7 effective_weight, globally scaled to same total)",
        fontsize=14, fontweight="bold", y=1.01
    )

    plt.savefig(out / "02_weighted_vs_raw_samples.png",
                bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"  \u2713  {out / '03_weighted_vs_raw_samples.png'}")

# ──────────────────────────────────────────────────────────────────────────────
# Summary report
# ──────────────────────────────────────────────────────────────────────────────

def write_summary(df: pd.DataFrame, out_path: Path):
    lines = []
    lines.append("=" * 70)
    lines.append("  DATASET BALANCE SUMMARY REPORT")
    lines.append("=" * 70)

    total_samples  = df["num_samples"].sum()
    total_dur_hr   = df["total_duration_sec"].sum(skipna=True) / 3600
    total_datasets = len(df)
    lines.append(f"\nTotal manifests parsed : {total_datasets}")
    lines.append(f"Total samples          : {total_samples:,}")
    lines.append(f"Total audio duration   : {total_dur_hr:,.1f} hours")

    # ── Create display language for AST ───────────────────────────────────────
    def get_display_lang(row):
        task = str(row.get("task_type", "")).strip()
        lang = str(row.get("language", "") or "").strip()
        if task == "ast":
            src = str(row.get("source_lang", "") or "").strip()
            tgt = str(row.get("target_lang", "") or "").strip()
            if src and tgt:
                return f"{src}→{tgt}"
            return src or tgt or "unknown"
        return lang if lang else "unknown"

    df = df.copy()
    df["display_lang"] = df.apply(get_display_lang, axis=1)

    lines.append("\n── Task distribution ──────────────────────────────────────────")
    task_grp = df.groupby("task_type").agg(
        datasets=("dataset_name", "count"),
        samples=("num_samples", "sum"),
        hours=("total_duration_sec", lambda x: x.sum(skipna=True)/3600)
    ).sort_values("samples", ascending=False)
    for task, row in task_grp.iterrows():
        pct = 100 * row.samples / total_samples if total_samples else 0
        lines.append(f"  {task:<40} {row.samples:>9,} samples "
                     f"({pct:5.1f}%)  {row.hours:8.1f} h  [{int(row.datasets)} manifests]")

    lines.append("\n── Language distribution ───────────────────────────────────────")
    lang_grp = df.groupby("display_lang").agg(
        samples=("num_samples", "sum"),
        hours=("total_duration_sec", lambda x: x.sum(skipna=True)/3600)
    ).sort_values("samples", ascending=False)
    for lang, row in lang_grp.iterrows():
        pct = 100 * row.samples / total_samples if total_samples else 0
        lines.append(f"  {str(lang):<10}  {row.samples:>9,} samples ({pct:5.1f}%)  {row.hours:8.1f} h")

    lines.append("\n── Top-10 datasets by samples ──────────────────────────────────")
    top10 = df.nlargest(10, "num_samples")[
        ["dataset_name", "task_type", "display_lang", "num_samples", "total_duration_sec"]]
    for _, r in top10.iterrows():
        hours = (r.total_duration_sec or 0) / 3600
        lines.append(f"  {r.dataset_name:<35} {r.task_type:<20} "
                     f"lang={str(r.display_lang):<5} {r.num_samples:>9,} samples  {hours:7.1f} h")

    lines.append("\n── Balance notes ───────────────────────────────────────────────")
    if total_samples:
        dominant = task_grp.index[0]
        dominant_pct = 100 * task_grp.loc[dominant, "samples"] / total_samples
        if dominant_pct > 50:
            lines.append(f"  ⚠  Task '{dominant}' dominates ({dominant_pct:.1f}% of samples).")
        else:
            lines.append("  ✓  No single task exceeds 50% of samples.")

        en_pct = 100 * lang_grp.get("samples", pd.Series()).get("en", 0) / total_samples
        fr_pct = 100 * lang_grp.get("samples", pd.Series()).get("fr", 0) / total_samples
        lines.append(f"  EN share: {en_pct:.1f}%   FR share: {fr_pct:.1f}%")

    lines.append("\n" + "=" * 70)

    report = "\n".join(lines)
    print(report)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  ✓  Summary written → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    script_dir = Path(__file__).resolve().parent
    
    parser = argparse.ArgumentParser(
        description="Generate dataset metadata CSV + visualizations from NeMo YAML.")
    parser.add_argument("yaml_path",        help="Path to the input YAML config file.")
    parser.add_argument("--output_dir",     default=str(script_dir / "dataset_analysis"),
                        help=f"Directory for output files (default: {script_dir / 'dataset_analysis'}).")
    parser.add_argument("--data_root",      default="",
                        help="Optional root to prepend to manifest paths if they use env vars.")
    parser.add_argument("--skip_missing",   action="store_true",
                        help="Silently skip manifests that do not exist.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 Parsing YAML: {args.yaml_path}")
    entries = flatten_manifests(args.yaml_path)
    print(f"   Found {len(entries)} manifest entries.\n")

    # Optionally resolve env-var placeholders
    DATA_FOLDER = args.data_root or os.environ.get("DATA_FOLDER", "")

    rows = []
    missing = 0
    for i, e in enumerate(entries):
        if i % 5 == 0: print(f"  [{i+1}/{len(entries)}] ...", end="\r")
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
        if stats is None:
            missing += 1
            if not args.skip_missing:
                # Still add a placeholder row with just metadata from YAML
                rows.append({
                    "dataset_name":    Path(path).parent.name,
                    "split":           Path(path).stem,
                    "task_type":       e["task_type"],
                    "sub_task":        e["sub_task"],
                    "language":        e["language"],
                    "source_lang":     e["source_lang"],
                    "target_lang":     e["target_lang"],
                    "num_audio_segments": None,
                    "num_samples":     None,
                    "total_duration_sec": None,
                    "total_duration_dhms": None,
                    "min_segment_duration_sec": None,
                    "max_segment_duration_sec": None,
                    "avg_segment_duration_sec": None,
                    "median_segment_duration_sec": None,
                    "std_segment_duration_sec": None,
                    "avg_instruction_words": None,
                    "min_instruction_words": None,
                    "max_instruction_words": None,
                    "avg_response_words": None,
                    "min_response_words": None,
                    "max_response_words": None,
                    "num_unique_speakers": None,
                    "avg_audio_sampling_rate": None,
                    "avg_audio_channels": None,
                    "path": path,
                    "weight_dataset": e["weight_dataset"],
                    "weight_group": e.get("weight_group", e["weight_dataset"]),
                    "file_exists": False,
                })
        else:
            stats["weight_dataset"] = e["weight_dataset"]
            stats["weight_group"]   = e.get("weight_group", e["weight_dataset"])
            stats["file_exists"]    = True
            rows.append(stats)

    print(f"\n   Parsed: {len(entries) - missing} found, {missing} missing.\n")

    df = pd.DataFrame(rows)

    # Canonical column order
    col_order = [
        "dataset_name", "split", "task_type", "sub_task", "language",
        "source_lang", "target_lang",
        "num_audio_segments", "num_samples",
        "total_duration_sec", "total_duration_dhms",
        "min_segment_duration_sec", "max_segment_duration_sec",
        "avg_segment_duration_sec", "median_segment_duration_sec",
        "std_segment_duration_sec",
        "avg_instruction_words", "min_instruction_words", "max_instruction_words",
        "avg_response_words", "min_response_words", "max_response_words",
        "num_unique_speakers", "avg_audio_sampling_rate", "avg_audio_channels",
        "weight_group", "weight_dataset", "file_exists", "path",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    csv_path = out_dir / "dataset_metadata.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV saved → {csv_path}  ({len(df)} rows)\n")

    # ── Visualizations ────────────────────────────────────────────────────────
    df_vis = df[df["file_exists"] == True].copy() if "file_exists" in df.columns else df.copy()
    df_vis = df_vis[df_vis["num_samples"].notna()]

    if df_vis.empty:
        print("⚠  No data available for plotting (all manifests missing?).")
    else:
        print("🎨 Generating plots …")
        set_style()
        split_label = df_vis["split"].mode()[0] if "split" in df_vis.columns and not df_vis.empty else "train"
        plot_global_donut(df_vis, out_dir, split_label=split_label, use_weights=False)
        plot_global_donut(df_vis, out_dir, split_label=split_label, use_weights=True)
        # plot_task_distribution(df_vis, out_dir)
        # plot_weighted_vs_raw_samples(df_vis, out_dir)
        plot_instruction_response_lengths(df_vis, out_dir)
    # ── Summary report ────────────────────────────────────────────────────────
    print("\n📝 Writing summary report …")
    if not df_vis.empty:
        write_summary(df_vis, out_dir / "summary_report.txt")
    else:
        # Write a stub summary from YAML metadata alone
        print("  (based on YAML metadata only — no manifest data available)")
        with open(out_dir / "summary_report.txt", "w") as f:
            f.write("No manifest files found. Summary based on YAML config only.\n\n")
            f.write(f"Total manifest entries in YAML: {len(df)}\n")
            task_counts = df["task_type"].value_counts()
            f.write("\nTask counts (manifests):\n")
            for t, c in task_counts.items():
                f.write(f"  {t:<40} {c} manifests\n")
            lang_counts = df["language"].value_counts()
            f.write("\nLanguage counts (manifests):\n")
            for l, c in lang_counts.items():
                f.write(f"  {l:<10} {c} manifests\n")

    print("\n✅ Done! All outputs in:", out_dir.resolve())


if __name__ == "__main__":
    main()