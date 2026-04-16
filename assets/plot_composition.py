"""
plot_composition.py
===================
Generate dataset composition plots from a metadata CSV.
Optionally takes a YAML to overlay its explicit weights instead of
computing data-driven weights from scratch.

Outputs
-------
  [prefix]dataset_distrib_pies_<metric>.png  — before/after resampling side-by-side donuts
  [prefix]length_distrib.png                — instruction / response word-count histograms

Usage
-----
# Basic: compute weights from CSV with default settings
python plot_composition.py metadata.csv --output_dir ./plots

# With a specific YAML to read weights from
python plot_composition.py metadata.csv --yaml suggested.yaml --output_dir ./plots

# Custom temperature / metric
python plot_composition.py metadata.csv --metric duration --temperature 3.0

# Filter to one yaml_source if CSV has multiple
python plot_composition.py metadata.csv --yaml_source train
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
    _EPSILON, _lang_key, fmt_hours, fmt_num, read_yaml_weights,
)
from suggest_yaml_cfg import compute_weights

warnings.filterwarnings("ignore")

_TEMPERATURE = 1.0
PALETTE      = "Set2"

# ──────────────────────────────────────────────────────────────────────────────
# Visual encoding
# ──────────────────────────────────────────────────────────────────────────────

LANG_COLORS = {
    "fr": "#002395", "en": "#C8102E", "es": "#AA151B", "de": "#444444",
    "it": "#009246", "nl": "#FF6600", "pt": "#006600", "ar": "#007A3D",
    "mixed": "#AAAAAA", "unknown": "#CCCCCC", "": "#CCCCCC",
}
TASK_HATCHES = {
    "asr": "", "ast": "///", "qa": "xxx", "aqa": "|||",
    "other": "ooo", "audio_captioning": "---",
    "music_captioning": "***", "mqa": "...",
}

def lang_color(lang: str) -> str:
    return LANG_COLORS.get(str(lang).lower(), "#888888")

def task_hatch(task: str) -> str:
    return TASK_HATCHES.get(str(task).lower(), "")


# ──────────────────────────────────────────────────────────────────────────────
# Label builder
# ──────────────────────────────────────────────────────────────────────────────

def make_label(row):
    task = str(row["task_type"]).strip()
    lang = str(row.get("language", "") or "").strip()
    src  = str(row.get("source_lang", "") or "").strip()
    tgt  = str(row.get("target_lang", "") or "").strip()

    if task == "ast":
        lbl = f"AST ({src}→{tgt})" if src and tgt else f"AST ({src}→?)" if src else "AST"
        return "AST", lbl, src or lang or "unknown", tgt or lang or "unknown", task

    if task == "other":
        lbl = f"Other ({lang})" if lang else "Other"
        return ("Other", lbl, lang or "unknown", lang or "unknown", task)

    grp   = task.upper() if task in ("asr","ast","qa","mqa","aqa") \
            else task.replace("_"," ").title()
    langs = (lang, lang) if lang else ("unknown","unknown")

    for t, prefix in [("asr","ASR"),("qa","QA"),("aqa","Audio QA"),("mqa","Music QA")]:
        if task == t:
            return grp, f"{prefix} ({lang})" if lang else prefix, *langs, task

    if task in ("audio_captioning","music_captioning"):
        lbl = task.replace("_"," ").title()
        return lbl, f"{lbl} ({lang})" if lang else lbl, *langs, task

    lbl = task.replace("_"," ").title()
    return grp, f"{lbl} ({lang})" if lang else lbl, *langs, task


# ──────────────────────────────────────────────────────────────────────────────
# Main donut plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_composition(df: pd.DataFrame,
                     out: Path,
                     metric: str,
                     temperature: float,
                     split_label: str = "train",
                     prefix: str = "") -> None:
    """
    Two donuts side by side:
      Left  — raw audio duration (before resampling)
      Right — sampling probability (after resampling, from df["effective_prob"])
    Shared legend with columns: Raw % | Raw hrs | Sampling %
    Green/red colouring highlights up/down-sampled datasets.
    """

    # ── Build display labels ─────────────────────────────────────────────────
    info        = df.apply(make_label, axis=1)
    df = df.copy()
    df["_grp"]  = [x[0] for x in info]
    df["_lbl"]  = [x[1] for x in info]
    df["_lsrc"] = [x[2] for x in info]
    df["_ltgt"] = [x[3] for x in info]
    df["_tsk"]  = [x[4] for x in info]

    lbl_map = {}
    for _, row in df.iterrows():
        lbl = row["_lbl"]
        if lbl not in lbl_map:
            lbl_map[lbl] = {"group": row["_grp"], "l1": row["_lsrc"],
                            "l2": row["_ltgt"],   "task": row["_tsk"]}



    # ── LEFT: raw distribution ───────────────────────────────────────────────
    raw_col   = "total_duration_sec" if metric == "duration" else "num_samples"
    raw_grp   = df.groupby("_lbl")[raw_col].sum().fillna(0)
    raw_grp   = raw_grp[raw_grp > 0].sort_values(ascending=False)
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
    prob_grp   = df.groupby("_lbl")["_resampled"].sum().fillna(0)
    prob_grp   = prob_grp[prob_grp > 0].sort_values(ascending=False)
    prob_total = prob_grp.sum()

    def _collapse(series, total):
        pcts  = 100 * series / max(float(total), _EPSILON)
        # Drop slices <0.2% entirely — no catch-all bucket
        return series[pcts >= 0.2].copy()

    raw_main  = _collapse(raw_grp,  raw_total)
    prob_main = _collapse(prob_grp, prob_total)

    # Union of labels, sorted by task sampling weight desc, then by entry weight desc
    all_labels_set = set(raw_main.index) | set(prob_main.index)
    task_prob_total = {}
    for lbl in all_labels_set:
        task = lbl_map[lbl]["task"]
        task_prob_total[task] = task_prob_total.get(task, 0) + prob_grp.get(lbl, 0)
    all_labels = sorted(
        all_labels_set,
        key=lambda lbl: (-task_prob_total[lbl_map[lbl]["task"]], -prob_grp.get(lbl, 0)),
    )

    # Reindex series to follow the sorted order
    raw_main  = raw_main.reindex(all_labels).dropna()
    prob_main = prob_main.reindex(all_labels).dropna()

    if metric == "duration":
        raw_abs = {lbl: raw_grp.get(lbl, 0) / 3600 for lbl in all_labels}
    else:
        raw_abs = {lbl: int(raw_grp.get(lbl, 0))    for lbl in all_labels}
    raw_pct    = {lbl: 100 * raw_grp.get(lbl, 0) / max(raw_total, _EPSILON)
                  for lbl in all_labels}
    prob_pct   = {lbl: 100 * prob_grp.get(lbl, 0) / max(prob_total, _EPSILON)
                  for lbl in all_labels}

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig    = plt.figure(figsize=(30, 13))
    ax_l   = fig.add_axes([0.01, 0.06, 0.29, 0.86])
    ax_r   = fig.add_axes([0.32, 0.06, 0.29, 0.86])
    ax_leg = fig.add_axes([0.63, 0.04, 0.36, 0.92])

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
            t1, t2  = theta - d_theta, theta
            m       = lbl_map[lbl]
            c1, c2  = lang_color(m["l1"]), lang_color(m["l2"])
            ht      = task_hatch(m["task"])
            if m["l1"] == m["l2"]:
                ax.add_patch(Wedge((0,0), 1.0, t1, t2, width=0.42,
                    facecolor=c1, edgecolor="white", linewidth=1.6, hatch=ht, alpha=0.93))
            else:
                ax.add_patch(Wedge((0,0), 1.00, t1, t2, width=0.21,
                    facecolor=c2, edgecolor="white", linewidth=1.4, hatch=ht, alpha=0.93))
                ax.add_patch(Wedge((0,0), 0.79, t1, t2, width=0.21,
                    facecolor=c1, edgecolor="white", linewidth=1.4, hatch=ht, alpha=0.93))
            theta = t1
        ax.text(0,  0.13, centre_top, ha="center", va="center",
                fontsize=13, fontweight="bold", color="#1a1a1a")
        ax.text(0, -0.13, centre_bot, ha="center", va="center",
                fontsize=10, color="#555555")
        ax.text(0, -1.22,
                "AST — Inner ring: source lang  |  Outer ring: target lang",
                ha="center", va="bottom", fontsize=7, color="#666666", style="italic")

    if metric == "duration":
        raw_centre = f"{raw_total/3600:,.0f} hr total\n(Raw duration)"
        raw_title  = "Before resampling\n(actual audio hours)"
        prob_centre = f"Expected duration\n(Weighted, T={temperature:.1f})"
        prob_title  = f"After resampling\n(expected duration share, T={temperature:.1f})"
    else:
        raw_centre = f"{fmt_num(int(raw_total))} samples\n(Raw counts)"
        raw_title  = "Before resampling\n(actual sample counts)"
        prob_centre = f"Expected samples\n(Weighted, T={temperature:.1f})"
        prob_title  = f"After resampling\n(expected sample share, T={temperature:.1f})"

    _draw_donut(ax_l, raw_main,
                centre_top=f"Split: {split_label}",
                centre_bot=raw_centre)
    ax_l.set_title(raw_title, fontsize=12, fontweight="bold", pad=8, y=1.01)

    _draw_donut(ax_r, prob_main,
                centre_top=f"Split: {split_label}",
                centre_bot=prob_centre)
    ax_r.set_title(prob_title, fontsize=12, fontweight="bold", pad=8, y=1.01)

    # ── Legend ────────────────────────────────────────────────────────────────
    leg_items = []
    for lbl in all_labels:
        m = lbl_map[lbl]
        leg_items.append({
            "group":    m["group"],
            "label":    lbl,
            "raw_abs":  raw_abs.get(lbl, 0),
            "raw_pct":  raw_pct.get(lbl, 0),
            "prob_pct": prob_pct.get(lbl, 0),
            "c1": lang_color(m["l1"]),
            "c2": lang_color(m["l2"]),
            "ht": task_hatch(m["task"]),
        })

    grp_totals = {}
    for it in leg_items:
        g = it["group"]
        if g not in grp_totals:
            grp_totals[g] = {"raw_pct": 0, "raw_abs": 0, "prob_pct": 0}
        grp_totals[g]["raw_pct"]  += it["raw_pct"]
        grp_totals[g]["raw_abs"]  += it["raw_abs"]
        grp_totals[g]["prob_pct"] += it["prob_pct"]
    leg_items.sort(key=lambda x: (-grp_totals[x["group"]]["prob_pct"], -x["prob_pct"]))

    n_rows = len(leg_items) + len(grp_totals) + 6
    row_h  = 1.0 / max(n_rows, 1)
    col_xs = [0.00, 0.055, 0.60, 0.73, 0.90]

    abs_header = "Raw hrs" if metric == "duration" else "Raw count"
    for col, txt, ha in [
        (col_xs[1], "Dataset Group",  "left"),
        (col_xs[2], "Raw %",          "right"),
        (col_xs[3], abs_header,       "right"),
        (col_xs[4], "Resampled %",    "right"),
    ]:
        ax_leg.text(col, 1.00, txt, fontsize=9, fontweight="bold",
                    va="top", ha=ha, transform=ax_leg.transAxes)

    ax_leg.plot([0, 1], [1.0 - 1.6*row_h, 1.0 - 1.6*row_h],
                color="#cccccc", linewidth=0.6,
                transform=ax_leg.transAxes, clip_on=False)

    y_pos         = 1.0 - 2 * row_h
    current_group = None

    for item in leg_items:
        if item["group"] != current_group:
            current_group = item["group"]
            gt = grp_totals[current_group]
            y_pos -= row_h * 0.35
            ty_grp = y_pos + row_h * 0.15
            ax_leg.text(0.0, ty_grp, current_group,
                        fontsize=9, fontweight="bold", color="#111111",
                        va="center", transform=ax_leg.transAxes)
            ax_leg.text(col_xs[2], ty_grp, f"{gt['raw_pct']:.1f}%", fontsize=8.5,
                        fontweight="bold", va="center", ha="right", color="#111111",
                        transform=ax_leg.transAxes, clip_on=False)
            abs_txt = f"{gt['raw_abs']:,.0f}" if metric == "duration" else fmt_num(gt['raw_abs'])
            ax_leg.text(col_xs[3], ty_grp, abs_txt, fontsize=8.5,
                        fontweight="bold", va="center", ha="right", color="#111111",
                        transform=ax_leg.transAxes, clip_on=False)
            delta_grp = gt["prob_pct"] - gt["raw_pct"]
            clr_grp   = "#c0392b" if delta_grp < -1 else "#27ae60" if delta_grp > 1 else "#111111"
            ax_leg.text(col_xs[4], ty_grp, f"{gt['prob_pct']:.1f}%", fontsize=8.5,
                        fontweight="bold", va="center", ha="right", color=clr_grp,
                        transform=ax_leg.transAxes, clip_on=False)
            y_pos -= row_h

        sw = 0.036
        ry = y_pos - row_h * 0.3
        rh = row_h * 0.7
        if item["c1"] == item["c2"]:
            ax_leg.add_patch(plt.Rectangle(
                (col_xs[0], ry), sw, rh,
                transform=ax_leg.transAxes, clip_on=False,
                facecolor=item["c1"], hatch=item["ht"],
                edgecolor="white", linewidth=0.4))
        else:
            ax_leg.add_patch(plt.Rectangle(
                (col_xs[0], ry), sw/2, rh,
                transform=ax_leg.transAxes, clip_on=False,
                facecolor=item["c1"], hatch=item["ht"],
                edgecolor="white", linewidth=0.4))
            ax_leg.add_patch(plt.Rectangle(
                (col_xs[0]+sw/2, ry), sw/2, rh,
                transform=ax_leg.transAxes, clip_on=False,
                facecolor=item["c2"], hatch=item["ht"],
                edgecolor="white", linewidth=0.4))

        lbl_str = item["label"][:44] + "…" if len(item["label"]) > 46 else item["label"]
        ty = y_pos + row_h * 0.15

        ax_leg.text(col_xs[1], ty, lbl_str, fontsize=8, va="center",
                    transform=ax_leg.transAxes, clip_on=False)
        ax_leg.text(col_xs[2], ty, f"{item['raw_pct']:.1f}%", fontsize=8,
                    va="center", ha="right", transform=ax_leg.transAxes, clip_on=False)
        abs_txt = f"{item['raw_abs']:,.0f}" if metric == "duration" else fmt_num(item['raw_abs'])
        ax_leg.text(col_xs[3], ty, abs_txt, fontsize=8,
                    va="center", ha="right", transform=ax_leg.transAxes, clip_on=False)

        delta = item["prob_pct"] - item["raw_pct"]
        clr   = "#c0392b" if delta < -1 else "#27ae60" if delta > 1 else "#333333"
        ax_leg.text(col_xs[4], ty, f"{item['prob_pct']:.1f}%", fontsize=8,
                    va="center", ha="right", color=clr,
                    transform=ax_leg.transAxes, clip_on=False)

        y_pos -= row_h

    ax_leg.text(0.0, max(y_pos - row_h, 0.01),
                "Resampled % colour:  🟢 up-sampled vs raw   🔴 down-sampled vs raw",
                fontsize=7.5, color="#555555", style="italic",
                va="bottom", transform=ax_leg.transAxes)

    fig.suptitle(
        f"Dataset composition — Before vs After resampling"
        f"   (metric={metric}, T={temperature:.1f})\n"
        "Color = language  |  Hatch = task",
        fontsize=13, fontweight="bold", y=1.00,
    )

    out_path = out / f"{prefix}dataset_distrib_pies_{metric}.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  ✓  {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Word-length histogram
# ──────────────────────────────────────────────────────────────────────────────

def plot_word_lengths(df: pd.DataFrame, out: Path, prefix: str = "") -> None:
    instr = df["avg_instruction_words"].dropna()
    resp  = df["avg_response_words"].dropna()
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
    parser = argparse.ArgumentParser(
        description="metadata CSV → composition plots (before/after resampling).")
    parser.add_argument("csv_path",
                        help="metadata.csv produced by generate_csv.py")
    parser.add_argument("--yaml",        default=None,
                        help="Optional: YAML file whose weights override computed ones. "
                             "Useful to plot the final suggested YAML directly.")
    parser.add_argument("--output_dir",  default=None,
                        help="Output directory (default: same folder as CSV).")
    parser.add_argument("--metric",      default="duration",
                        choices=["duration","samples"],
                        help="Size metric for weight computation (default: duration).")
    parser.add_argument("--temperature", type=float, default=_TEMPERATURE,
                        help=f"Smoothing T>0 (default {_TEMPERATURE}).")
    parser.add_argument("--min_weight",  type=float, default=0.0001,
                        help="Minimum effective probability floor (default: 0.0001).")
    parser.add_argument("--yaml_source", default=None,
                        help="Filter CSV to this yaml_source value.")
    parser.add_argument("--skip_missing", action="store_true",
                        help="Drop rows where file_exists=False.")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    out_dir  = Path(args.output_dir) if args.output_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📊 Loading CSV  : {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Loaded {len(df)} rows.")

    if args.yaml_source:
        df = df[df["yaml_source"] == args.yaml_source].copy()
        print(f"   Filtered to yaml_source='{args.yaml_source}': {len(df)} rows.")

    if args.skip_missing and "file_exists" in df.columns:
        before = len(df)
        df = df[df["file_exists"] == True].copy()
        print(f"   Dropped {before - len(df)} missing-file rows.")

    df = df[df["num_samples"].notna()].copy()

    if args.yaml:
        # Read effective weights directly from the YAML
        print(f"\n📂 Reading weights from YAML: {args.yaml}")
        yaml_weights = read_yaml_weights(args.yaml)
        # Map them onto the dataframe via raw_manifest_path
        def _lookup_prob(row):
            path = row.get("raw_manifest_path", "")
            return yaml_weights.get(path, 0.0)
        df["effective_prob"] = df.apply(_lookup_prob, axis=1)
        total_prob = df["effective_prob"].sum()
        if total_prob > 0:
            df["effective_prob"] = df["effective_prob"] / total_prob
        df["effective_prob_pct"] = df["effective_prob"] * 100.0
        weight_source = f"YAML: {Path(args.yaml).name}"
        temperature   = args.temperature  # used only for label
    else:
        print(f"\n⚖️  Computing weights: metric={args.metric}, T={args.temperature}")
        df = compute_weights(df, metric=args.metric,
                             temperature=args.temperature,
                             min_weight=args.min_weight)
        weight_source = f"computed (metric={args.metric}, T={args.temperature})"
        temperature   = args.temperature

    print(f"   Weight source : {weight_source}")
    print(f"   Total prob sum: {df['effective_prob'].sum():.4f}")

    split_label = (df["split"].mode()[0]
                   if "split" in df.columns and not df.empty else "train")

    prefix = ""
    if args.yaml:
        prefix = Path(args.yaml).stem + "_"

    print("\n🎨 Generating plots …")
    plot_composition(df, out_dir, args.metric, temperature, split_label=split_label, prefix=prefix)
    # plot_word_lengths(df, out_dir, prefix=prefix)
    print(f"\n✅ Plots saved to: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()