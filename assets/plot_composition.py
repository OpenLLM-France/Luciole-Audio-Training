"""
plot_composition.py
===================
Generate dataset composition plots from a metadata CSV.
Optionally takes a YAML to overlay its explicit weights instead of
computing data-driven weights from scratch.

Outputs
-------
  [prefix]overall_<metric>.png                         — before/after resampling side-by-side donuts (all tasks)
  [prefix]subtasks/[prefix]subtasks_<task>_<metric>.png            — subtask breakdown within each task
  [prefix]datasets/[prefix]datasets_<task>[_<subtask>]_<metric>.png   — dataset breakdown within each (sub)task
  [prefix]languages/[prefix]languages_<task>[_<subtask>]_<metric>.png — language breakdown within each (sub)task
  [prefix]length_distrib.png                           — instruction / response word-count histograms

Usage
-----
# Basic: compute weights from CSV with default settings
python plot_composition.py metadata.csv --output_dir ./plots

# With a specific YAML to read weights from
python plot_composition.py metadata.csv --yaml suggested.yaml --output_dir ./plots

# Custom temperature / metric
python plot_composition.py metadata.csv --metric duration --temperature 3.0

"""

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import pandas as pd
import seaborn as sns

from analyze_yaml_cfg import (
    _EPSILON,
    _lang_key,
    fmt_hours,
    fmt_num,
    read_yaml_weights,
)
from suggest_yaml_cfg import compute_weights

warnings.filterwarnings("ignore")

_TEMPERATURE = 1.0
PALETTE = "Set2"

# ──────────────────────────────────────────────────────────────────────────────
# Visual encoding
# ──────────────────────────────────────────────────────────────────────────────

LANG_COLORS = {
    "fr": "#002395",  # French flag blue
    "en": "#C8102E",  # UK/US flag red
    "es": "#F1BF00",  # Spanish flag gold (was red — clashed with English)
    "de": "#1A1A1A",  # German flag black
    "it": "#009246",  # Italian flag green
    "nl": "#FF6600",  # Dutch House of Orange
    "pt": "#6F2DA8",  # Port-wine purple (was green — clashed with Italian/Arabic)
    "ar": "#8B4513",  # Saddle brown / desert (was green — clashed)
    "mixed": "#AAAAAA",
    "unknown": "#CCCCCC",
    "": "#CCCCCC",
}
TASK_HATCHES = {
    "asr": "",
    "ast": "///",
    "qa": "xxx",
    "aqa": "|||",
    "sound": "+++",
    "music": "\\\\\\",
    "other": "ooo",
    "audio_captioning": "---",
    "music_captioning": "***",
    "mqa": "...",
    "task_switching": "x+x",  # multi-task mix — busy crosshatch
}

# Hatch styles used to dynamically encode subtasks (sorted from larger to smaller).
HATCH_CYCLE = [
    "",
    "///",
    "xxx",
    "|||",
    "+++",
    "\\\\\\",
    "ooo",
    "---",
    "***",
    "...",
    "//x",
    "++x",
    "**.",
    "//\\\\",
    "x+x",
]


def lang_color(lang: str) -> str:
    return LANG_COLORS.get(str(lang).lower(), "#888888")


def task_hatch(task: str, hatch_map: dict | None = None) -> str:
    return (hatch_map if hatch_map is not None else TASK_HATCHES).get(str(task).lower(), "")


# ──────────────────────────────────────────────────────────────────────────────
# Label builder
# ──────────────────────────────────────────────────────────────────────────────


def _normalize_dataset_name(name: str) -> str:
    """Collapse closely related dataset variants under a single display name."""
    if name.lower().startswith("commonvoice"):
        return "CommonVoice"
    return name


def _safe_str(v) -> str:
    """Convert a possibly-NaN/None value to a stripped string, '' on missing."""
    if v is None:
        return ""
    if isinstance(v, float) and v != v:  # NaN
        return ""
    return str(v).strip()


def make_label(row, group_field: str = "task_type"):
    """Return (group, unique_label, display_label, lang_src, lang_tgt, hatch_key).

    *group_field* selects the column used to form the legend's grouping (the
    hatch dimension). Default is "task_type" (overall plot); pass "sub_task"
    (or any other column) to group by that field instead.
    """
    task = _safe_str(row.get("task_type", "")).lower()
    lang = _safe_str(row.get("language", ""))
    src = _safe_str(row.get("source_lang", ""))
    tgt = _safe_str(row.get("target_lang", ""))

    if group_field == "task_type":
        hatch_key = task
        grp = (
            task.upper()
            if task in ("asr", "ast", "qa", "mqa", "aqa")
            else task.capitalize()
            if task in ("sound", "music")
            else task.replace("_", " ").title()
        )
    else:
        grp_val = _safe_str(row.get(group_field, "")) or "(none)"
        if group_field == "dataset_name" and grp_val != "(none)":
            grp_val = _normalize_dataset_name(grp_val)
        hatch_key = grp_val.lower()
        if grp_val == "(none)":
            grp = f"(no {group_field.replace('_', ' ')})"
        elif group_field == "language":
            grp = grp_val.replace("-", "→") if "-" in grp_val else grp_val
        elif grp_val.replace("_", "").islower():
            grp = grp_val.replace("_", " ").title()
        else:
            grp = grp_val.replace("_", " ")

    if task == "ast":
        disp = f"{src}→{tgt}" if src and tgt else f"{src}→?" if src else "?"
        return grp, f"{grp}:{disp}", disp, src or lang or "unknown", tgt or lang or "unknown", hatch_key

    disp = lang or "?"
    langs = (lang or "unknown", lang or "unknown")
    return grp, f"{grp}:{disp}", disp, *langs, hatch_key


# ──────────────────────────────────────────────────────────────────────────────
# Main donut plot
# ──────────────────────────────────────────────────────────────────────────────


def plot_composition(
    df: pd.DataFrame,
    out: Path,
    metric: str,
    temperature: float,
    split_label: str = "train",
    prefix: str = "",
    no_title: bool = False,
    group_field: str = "task_type",
    hatch_map: dict | None = None,
    filename_base: str = "overall",
    title_extra: str = "",
) -> None:
    """
    Two donuts side by side:
      Left  — raw audio duration (before resampling)
      Right — sampling probability (after resampling, from df["effective_prob"])
    Shared legend with columns: Raw % | Raw hrs | Sampling %
    Green/red colouring highlights up/down-sampled datasets.

    *group_field* selects the legend's grouping/hatch dimension ("task_type"
    by default, or e.g. "sub_task" for per-task subtask plots). *hatch_map*
    overrides TASK_HATCHES; when None and group_field != "task_type", it is
    built dynamically (one hatch per distinct value of group_field, ordered
    by total size).
    """

    # ── Consistency check on the grouping set ────────────────────────────────
    def _key(v):
        s = str(v).strip()
        if group_field == "dataset_name" and s:
            s = _normalize_dataset_name(s)
        return s.lower() or "(none)"

    keys_in_data = {_key(t) for t in df[group_field].fillna("").unique()}
    if group_field == "task_type":
        has_aqa = "aqa" in keys_in_data
        has_sound_or_music = bool(keys_in_data & {"sound", "music"})
        if has_aqa and has_sound_or_music:
            raise ValueError(
                f"Inconsistent task set: 'aqa' coexists with 'sound'/'music' "
                f"({sorted(keys_in_data)}). Harmonise the YAML/CSV to use either "
                "'aqa' OR 'sound'/'music', not both."
            )

    # ── Resolve hatch_map ────────────────────────────────────────────────────
    if hatch_map is None:
        if group_field == "task_type":
            hatch_map = TASK_HATCHES
            missing_hatches = keys_in_data - set(hatch_map.keys())
            if missing_hatches:
                raise ValueError(
                    f"No visual encoding (hatch) defined for task(s) {sorted(missing_hatches)}. "
                    f"Add an entry to TASK_HATCHES in plot_composition.py."
                )
        else:
            size_col = "total_duration_sec" if metric == "duration" else "num_samples"
            tmp_key = df[group_field].fillna("").astype(str).map(_key)
            sizes = df[size_col].fillna(0).groupby(tmp_key).sum().sort_values(ascending=False)
            sizes = sizes[sizes > 0]
            hatch_map = {str(k): HATCH_CYCLE[i % len(HATCH_CYCLE)] for i, k in enumerate(sizes.index)}

    title_hatch_label = "task" if group_field == "task_type" else group_field.replace("_", " ")

    # ── Build display labels ─────────────────────────────────────────────────
    info = df.apply(lambda r: make_label(r, group_field), axis=1)
    df = df.copy()
    df["_grp"] = [x[0] for x in info]
    df["_lbl"] = [x[1] for x in info]  # unique key (e.g. "ASR:fr")
    df["_disp"] = [x[2] for x in info]  # display label (e.g. "fr")
    df["_lsrc"] = [x[3] for x in info]
    df["_ltgt"] = [x[4] for x in info]
    df["_tsk"] = [x[5] for x in info]

    lbl_map = {}
    for _, row in df.iterrows():
        lbl = row["_lbl"]
        if lbl not in lbl_map:
            lbl_map[lbl] = {
                "group": row["_grp"],
                "disp": row["_disp"],
                "l1": row["_lsrc"],
                "l2": row["_ltgt"],
                "task": row["_tsk"],
            }

    # ── LEFT: raw distribution ───────────────────────────────────────────────
    raw_col = "total_duration_sec" if metric == "duration" else "num_samples"
    raw_grp = df.groupby("_lbl")[raw_col].sum().fillna(0)
    raw_grp = raw_grp[raw_grp > 0].sort_values(ascending=False)
    raw_total = raw_grp.sum()

    # ── RIGHT: expected distribution after resampling ─────────────────────────
    if metric == "duration":
        # effective_prob is a per-sample weight; multiply by avg duration to get
        # the expected duration contribution of each dataset.
        avg_dur = df["total_duration_sec"].fillna(0) / df["num_samples"].clip(lower=1)
        df["_resampled"] = df["effective_prob"] * avg_dur
    else:
        # effective_prob is already the expected sample share
        df["_resampled"] = df["effective_prob"]
    prob_grp = df.groupby("_lbl")["_resampled"].sum().fillna(0)
    prob_grp = prob_grp[prob_grp > 0].sort_values(ascending=False)
    prob_total = prob_grp.sum()

    raw_main = raw_grp.copy()
    prob_main = prob_grp.copy()

    # Union of labels, sorted by raw num_samples (before resampling) desc
    all_labels_set = set(raw_main.index) | set(prob_main.index)
    samples_grp = df.groupby("_lbl")["num_samples"].sum().fillna(0)
    task_samples_total = {}
    for lbl in all_labels_set:
        task = lbl_map[lbl]["task"]
        task_samples_total[task] = task_samples_total.get(task, 0) + samples_grp.get(lbl, 0)
    all_labels = sorted(
        all_labels_set,
        key=lambda lbl: (-task_samples_total[lbl_map[lbl]["task"]], -samples_grp.get(lbl, 0)),
    )

    # Reindex series to follow the sorted order
    raw_main = raw_main.reindex(all_labels).dropna()
    prob_main = prob_main.reindex(all_labels).dropna()

    # ── Coverage check: every group_field value in the data must appear ──────
    plotted_keys = {lbl_map[lbl]["task"].strip().lower() for lbl in all_labels}
    missing_keys = keys_in_data - plotted_keys
    if missing_keys:
        raise ValueError(
            f"{group_field} value(s) {sorted(missing_keys)} present in the data but not "
            f"represented in the plot (no raw size and no effective probability)."
        )

    if metric == "duration":
        raw_abs = {lbl: raw_grp.get(lbl, 0) / 3600 for lbl in all_labels}
    else:
        raw_abs = {lbl: int(raw_grp.get(lbl, 0)) for lbl in all_labels}
    raw_pct = {lbl: 100 * raw_grp.get(lbl, 0) / max(raw_total, _EPSILON) for lbl in all_labels}
    prob_pct = {lbl: 100 * prob_grp.get(lbl, 0) / max(prob_total, _EPSILON) for lbl in all_labels}

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(30, 13))
    ax_l = fig.add_axes([0.00, 0.06, 0.29, 0.86])
    ax_r = fig.add_axes([0.26, 0.06, 0.29, 0.86])
    ax_leg = fig.add_axes([0.56, 0.04, 0.44, 0.92])

    for ax in (ax_l, ax_r):
        ax.set_aspect("equal")
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.30, 1.25)
        ax.axis("off")
    ax_leg.axis("off")

    def _draw_donut(ax, grp_series, centre_top, centre_bot):
        theta = 90.0
        total = grp_series.sum()
        for lbl, val in grp_series.items():
            d_theta = 360 * val / float(total)
            t1, t2 = theta - d_theta, theta
            m = lbl_map[lbl]
            c1, c2 = lang_color(m["l1"]), lang_color(m["l2"])
            ht = task_hatch(m["task"], hatch_map)
            if m["l1"] == m["l2"]:
                ax.add_patch(
                    Wedge(
                        (0, 0),
                        1.0,
                        t1,
                        t2,
                        width=0.42,
                        facecolor=c1,
                        edgecolor="white",
                        linewidth=1.6,
                        hatch=ht,
                        alpha=0.93,
                    )
                )
            else:
                ax.add_patch(
                    Wedge(
                        (0, 0),
                        1.00,
                        t1,
                        t2,
                        width=0.21,
                        facecolor=c2,
                        edgecolor="white",
                        linewidth=1.4,
                        hatch=ht,
                        alpha=0.93,
                    )
                )
                ax.add_patch(
                    Wedge(
                        (0, 0),
                        0.79,
                        t1,
                        t2,
                        width=0.21,
                        facecolor=c1,
                        edgecolor="white",
                        linewidth=1.4,
                        hatch=ht,
                        alpha=0.93,
                    )
                )
            theta = t1
        ax.text(0, 0.13, centre_top, ha="center", va="center", fontsize=17, fontweight="bold", color="#1a1a1a")
        ax.text(0, -0.13, centre_bot, ha="center", va="center", fontsize=14, color="#555555")

    t_str = f", T={temperature:.1f}" if temperature != 1.0 else ""
    if metric == "duration":
        raw_centre = f"{raw_total / 3600:,.0f} hr total"
        raw_title = "Actual audio hours\nbefore resampling"
        prob_centre = f"Expected duration"
        prob_title = f"Expected duration share\nafter resampling{t_str})"
    else:
        raw_centre = f"{fmt_num(int(raw_total))} samples"
        raw_title = "Actual sample counts\nbefore resampling"
        prob_centre = f"Expected samples"
        prob_title = f"Expected sample share\nafter resampling{t_str})"

    _draw_donut(ax_l, raw_main, centre_top=f"Split: {split_label}", centre_bot=raw_centre)
    ax_l.set_title(raw_title, fontsize=16, fontweight="bold", pad=8, y=1.01)

    _draw_donut(ax_r, prob_main, centre_top=f"Split: {split_label}", centre_bot=prob_centre)
    ax_r.set_title(prob_title, fontsize=16, fontweight="bold", pad=8, y=1.01)

    # ── Legend ────────────────────────────────────────────────────────────────
    leg_items = []
    for lbl in all_labels:
        m = lbl_map[lbl]
        leg_items.append(
            {
                "group": m["group"],
                "label": m["disp"],
                "raw_abs": raw_abs.get(lbl, 0),
                "raw_pct": raw_pct.get(lbl, 0),
                "raw_samples": samples_grp.get(lbl, 0),
                "prob_pct": prob_pct.get(lbl, 0),
                "c1": lang_color(m["l1"]),
                "c2": lang_color(m["l2"]),
                "ht": task_hatch(m["task"], hatch_map),
            }
        )

    grp_totals = {}
    for it in leg_items:
        g = it["group"]
        if g not in grp_totals:
            grp_totals[g] = {"raw_pct": 0, "raw_abs": 0, "raw_samples": 0, "prob_pct": 0}
        grp_totals[g]["raw_pct"] += it["raw_pct"]
        grp_totals[g]["raw_abs"] += it["raw_abs"]
        grp_totals[g]["raw_samples"] += it["raw_samples"]
        grp_totals[g]["prob_pct"] += it["prob_pct"]
    leg_items.sort(key=lambda x: (-grp_totals[x["group"]]["raw_samples"], -x["raw_samples"]))

    n_rows = len(leg_items) + len(grp_totals) + 6
    row_h = 1.0 / max(n_rows, 1)
    col_xs = [0.00, 0.055, 0.28, 0.43, 0.62]

    abs_header = "Raw hrs" if metric == "duration" else "Raw count"
    for col, txt, ha in [
        (col_xs[0], "Dataset Group", "left"),
        (col_xs[2], "Raw %", "right"),
        (col_xs[3], abs_header, "right"),
        (col_xs[4], "Resampled %", "right"),
    ]:
        ax_leg.text(col, 1.00, txt, fontsize=13, fontweight="bold", va="top", ha=ha, transform=ax_leg.transAxes)

    ax_leg.plot(
        [0, 1],
        [1.0 - 1.6 * row_h, 1.0 - 1.6 * row_h],
        color="#cccccc",
        linewidth=0.6,
        transform=ax_leg.transAxes,
        clip_on=False,
    )

    y_pos = 1.0 - 2 * row_h
    current_group = None

    for item in leg_items:
        if item["group"] != current_group:
            current_group = item["group"]
            gt = grp_totals[current_group]
            y_pos -= row_h * 0.35
            ty_grp = y_pos + row_h * 0.15
            ax_leg.text(
                0.0,
                ty_grp,
                current_group,
                fontsize=13,
                fontweight="bold",
                color="#111111",
                va="center",
                transform=ax_leg.transAxes,
            )
            ax_leg.text(
                col_xs[2],
                ty_grp,
                f"{gt['raw_pct']:.1f}%",
                fontsize=12.5,
                fontweight="bold",
                va="center",
                ha="right",
                color="#111111",
                transform=ax_leg.transAxes,
                clip_on=False,
            )
            abs_txt = f"{gt['raw_abs']:,.0f}" if metric == "duration" else fmt_num(gt["raw_abs"])
            ax_leg.text(
                col_xs[3],
                ty_grp,
                abs_txt,
                fontsize=12.5,
                fontweight="bold",
                va="center",
                ha="right",
                color="#111111",
                transform=ax_leg.transAxes,
                clip_on=False,
            )
            delta_grp = gt["prob_pct"] - gt["raw_pct"]
            clr_grp = "#c0392b" if delta_grp < -1 else "#27ae60" if delta_grp > 1 else "#111111"
            ax_leg.text(
                col_xs[4],
                ty_grp,
                f"{gt['prob_pct']:.1f}%",
                fontsize=12.5,
                fontweight="bold",
                va="center",
                ha="right",
                color=clr_grp,
                transform=ax_leg.transAxes,
                clip_on=False,
            )
            y_pos -= row_h

        sw = 0.036
        ry = y_pos - row_h * 0.3
        rh = row_h * 0.7
        if item["c1"] == item["c2"]:
            ax_leg.add_patch(
                plt.Rectangle(
                    (col_xs[0], ry),
                    sw,
                    rh,
                    transform=ax_leg.transAxes,
                    clip_on=False,
                    facecolor=item["c1"],
                    hatch=item["ht"],
                    edgecolor="white",
                    linewidth=0.4,
                )
            )
        else:
            ax_leg.add_patch(
                plt.Rectangle(
                    (col_xs[0], ry),
                    sw / 2,
                    rh,
                    transform=ax_leg.transAxes,
                    clip_on=False,
                    facecolor=item["c1"],
                    hatch=item["ht"],
                    edgecolor="white",
                    linewidth=0.4,
                )
            )
            ax_leg.add_patch(
                plt.Rectangle(
                    (col_xs[0] + sw / 2, ry),
                    sw / 2,
                    rh,
                    transform=ax_leg.transAxes,
                    clip_on=False,
                    facecolor=item["c2"],
                    hatch=item["ht"],
                    edgecolor="white",
                    linewidth=0.4,
                )
            )

        lbl_str = item["label"][:44] + "…" if len(item["label"]) > 46 else item["label"]
        ty = y_pos + row_h * 0.15

        ax_leg.text(col_xs[1], ty, lbl_str, fontsize=12, va="center", transform=ax_leg.transAxes, clip_on=False)
        ax_leg.text(
            col_xs[2],
            ty,
            f"{item['raw_pct']:.1f}%",
            fontsize=12,
            va="center",
            ha="right",
            transform=ax_leg.transAxes,
            clip_on=False,
        )
        abs_txt = f"{item['raw_abs']:,.0f}" if metric == "duration" else fmt_num(item["raw_abs"])
        ax_leg.text(
            col_xs[3], ty, abs_txt, fontsize=12, va="center", ha="right", transform=ax_leg.transAxes, clip_on=False
        )

        delta = item["prob_pct"] - item["raw_pct"]
        clr = "#c0392b" if delta < -1 else "#27ae60" if delta > 1 else "#333333"
        ax_leg.text(
            col_xs[4],
            ty,
            f"{item['prob_pct']:.1f}%",
            fontsize=12,
            va="center",
            ha="right",
            color=clr,
            transform=ax_leg.transAxes,
            clip_on=False,
        )

        y_pos -= row_h

    if not no_title:
        title = f"Dataset composition — Before vs After resampling   (metric={metric}{t_str})"
        if title_extra:
            title += f"   [{title_extra}]"
        title += f"\nColor = language  |  Hatch = {title_hatch_label}"
        fig.suptitle(title, fontsize=17, fontweight="bold", y=1.00)

    category = filename_base.split("_", 1)[0]
    if category in ("subtasks", "datasets", "languages"):
        subdir = out / f"{prefix}{category}"
        subdir.mkdir(parents=True, exist_ok=True)
        out_path = subdir / f"{prefix}{filename_base}_{metric}.png"
    else:
        out_path = out / f"{prefix}{filename_base}_{metric}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  ✓  {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Word-length histogram
# ──────────────────────────────────────────────────────────────────────────────


def plot_word_lengths(df: pd.DataFrame, out: Path, prefix: str = "") -> None:
    instr = df["avg_instruction_words"].dropna()
    resp = df["avg_response_words"].dropna()
    if instr.empty and resp.empty:
        print("  ⚠  No word-count data — skipping word-length plot.")
        return

    sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pal = sns.color_palette(PALETTE)

    if not instr.empty:
        axes[0].hist(instr, bins=30, color=pal[1], edgecolor="white")
        axes[0].set_xlabel("Avg instruction words")
        axes[0].set_ylabel("# datasets")
        axes[0].set_title("Instruction length distribution")

    if not resp.empty:
        axes[1].hist(resp, bins=30, color=pal[2], edgecolor="white")
        axes[1].set_xlabel("Avg response words")
        axes[1].set_ylabel("# datasets")
        axes[1].set_title("Response length distribution")

    plt.tight_layout()
    out_path = out / f"{prefix}length_distrib.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="metadata CSV → composition plots (before/after resampling).")
    parser.add_argument("csv_path", help="metadata.csv produced by generate_csv.py")
    parser.add_argument(
        "--yaml",
        default=None,
        help="Optional: YAML file whose weights override computed ones. "
        "Useful to plot the final suggested YAML directly.",
    )
    parser.add_argument("--output_dir", default=None, help="Output directory (default: same folder as CSV).")
    parser.add_argument(
        "--metric",
        default="duration",
        choices=["duration", "samples"],
        help="Size metric for weight computation (default: duration).",
    )
    parser.add_argument(
        "--temperature", type=float, default=_TEMPERATURE, help=f"Smoothing T>0 (default {_TEMPERATURE})."
    )
    parser.add_argument(
        "--min_weight", type=float, default=0.0001, help="Minimum effective probability floor (default: 0.0001)."
    )
    parser.add_argument("--skip_missing", action="store_true", help="Drop rows where file_exists=False.")
    parser.add_argument(
        "--ignore_missing_manifest",
        action="store_true",
        help="Only warn (instead of error) when YAML manifests are missing from CSV.",
    )
    parser.add_argument(
        "--no_title", action="store_true", help="Remove the figure title and save with no white borders."
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    out_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📊 Loading CSV  : {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Loaded {len(df)} rows.")

    if args.skip_missing and "file_exists" in df.columns:
        before = len(df)
        df = df[df["file_exists"] == True].copy()
        print(f"   Dropped {before - len(df)} missing-file rows.")

    df = df[df["num_samples"].notna()].copy()

    if args.yaml:
        # Read effective weights directly from the YAML
        print(f"\n📂 Reading weights from YAML: {args.yaml}")
        yaml_weights = read_yaml_weights(args.yaml)
        # Filter CSV to only datasets present in the YAML
        yaml_paths = set(yaml_weights.keys())
        csv_paths = set(df["raw_manifest_path"].dropna())
        missing = yaml_paths - csv_paths
        if missing:
            msg = f"{len(missing)} dataset(s) in YAML but missing from CSV:\n"
            msg += "\n".join(f"      {m}" for m in sorted(missing))
            if args.ignore_missing_manifest:
                print(f"\n⚠️  {msg}")
            else:
                raise SystemExit(f"\n❌ {msg}\n   Use --ignore_missing_manifest to skip this error.")
        before = len(df)
        df = df[df["raw_manifest_path"].isin(yaml_paths)].copy()
        if len(df) < before:
            print(f"   Dropped {before - len(df)} CSV rows not in YAML ({len(df)} remaining).")

        # Map weights onto the dataframe via raw_manifest_path
        def _lookup_prob(row):
            path = row.get("raw_manifest_path", "")
            return yaml_weights.get(path, 0.0)

        df["effective_prob"] = df.apply(_lookup_prob, axis=1)
        total_prob = df["effective_prob"].sum()
        if total_prob > 0:
            df["effective_prob"] = df["effective_prob"] / total_prob
        df["effective_prob_pct"] = df["effective_prob"] * 100.0
        weight_source = f"YAML: {Path(args.yaml).name}"
        temperature = args.temperature  # used only for label
    else:
        print(f"\n⚖️  Computing weights: metric={args.metric}, T={args.temperature}")
        df = compute_weights(df, metric=args.metric, temperature=args.temperature, min_weight=args.min_weight)
        weight_source = f"computed (metric={args.metric}, T={args.temperature})"
        temperature = args.temperature

    print(f"   Weight source : {weight_source}")
    print(f"   Total prob sum: {df['effective_prob'].sum():.4f}")

    split_label = df["split"].mode()[0] if "split" in df.columns and not df.empty else "train"

    prefix = ""
    if args.yaml:
        prefix = Path(args.yaml).stem + "_"

    print("\n🎨 Generating plots …")
    plot_composition(
        df, out_dir, args.metric, temperature, split_label=split_label, prefix=prefix, no_title=args.no_title
    )

    # Per-task subtask plots
    size_col = "total_duration_sec" if args.metric == "duration" else "num_samples"
    task_order = (
        df.assign(_t=df["task_type"].fillna("").astype(str).str.strip().str.lower())
        .groupby("_t")[size_col]
        .sum()
        .sort_values(ascending=False)
    )
    for task in task_order.index:
        if not task:
            continue
        sub = df[df["task_type"].fillna("").astype(str).str.strip().str.lower() == task].copy()
        if sub.empty:
            continue
        n_subtasks = sub["sub_task"].fillna("").astype(str).str.strip().replace("", "(none)").nunique()
        if n_subtasks <= 1:
            print(f"   (skipping subtask plot for task='{task}': only {n_subtasks} subtask)")
            continue
        plot_composition(
            sub,
            out_dir,
            args.metric,
            temperature,
            split_label=split_label,
            prefix=prefix,
            no_title=args.no_title,
            group_field="sub_task",
            filename_base=f"subtasks_{task}",
            title_extra=f"task={task}",
        )

    # Per-(sub)task dataset and language distribution plots
    for task in task_order.index:
        if not task:
            continue
        sub_t = df[df["task_type"].fillna("").astype(str).str.strip().str.lower() == task].copy()
        if sub_t.empty:
            continue
        sub_t["_subtask_norm"] = sub_t["sub_task"].fillna("").astype(str).str.strip().replace("", "(none)")
        subtask_sizes = sub_t.groupby("_subtask_norm")[size_col].sum().sort_values(ascending=False)
        for st in subtask_sizes.index:
            df_st = sub_t[sub_t["_subtask_norm"] == st].copy()
            if df_st.empty:
                continue
            if st == "(none)":
                fname_ds = f"datasets_{task}"
                fname_lg = f"languages_{task}"
                extra = f"task={task}"
            else:
                fname_ds = f"datasets_{task}_{st.lower()}"
                fname_lg = f"languages_{task}_{st.lower()}"
                extra = f"task={task}, sub_task={st}"

            n_datasets = df_st["dataset_name"].fillna("").astype(str).str.strip().nunique()
            if n_datasets <= 1:
                print(f"   (skipping dataset plot for task='{task}', sub_task='{st}': only {n_datasets} dataset)")
            else:
                plot_composition(
                    df_st,
                    out_dir,
                    args.metric,
                    temperature,
                    split_label=split_label,
                    prefix=prefix,
                    no_title=args.no_title,
                    group_field="dataset_name",
                    filename_base=fname_ds,
                    title_extra=extra,
                )

            n_langs = df_st["language"].fillna("").astype(str).str.strip().nunique()
            if n_langs <= 1:
                print(f"   (skipping language plot for task='{task}', sub_task='{st}': only {n_langs} language)")
            else:
                plot_composition(
                    df_st,
                    out_dir,
                    args.metric,
                    temperature,
                    split_label=split_label,
                    prefix=prefix,
                    no_title=args.no_title,
                    group_field="language",
                    filename_base=fname_lg,
                    title_extra=extra,
                )

    # plot_word_lengths(df, out_dir, prefix=prefix)
    print(f"\n✅ Plots saved to: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
