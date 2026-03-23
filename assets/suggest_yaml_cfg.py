"""
suggest_yaml_cfg.py
===============
Given a metadata CSV (from generate_csv_metadata.py) and an original YAML,
compute data-driven sampling weights and write a suggested YAML.

Usage
-----
python suggest_yaml_cfg.py metadata.csv original.yaml \\
    --output suggested.yaml \\
    --metric duration \\
    --temperature 2.0 \\
    --min_weight 0.0001
"""

import argparse
import math
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

_EPSILON     = 1e-9
_TEMPERATURE = 2.0


# ──────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────────────────────────────────

def fmt_num(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    v = int(v)
    if v >= 1_000_000: return f"{v/1_000_000:.2f}M"
    if v >= 1_000:     return f"{v/1_000:.1f}K"
    return str(v)


def fmt_hours(secs) -> str:
    if secs is None or (isinstance(secs, float) and math.isnan(secs)):
        return "—"
    h = secs / 3600
    return f"{h:,.0f} h" if h >= 1000 else f"{h:.1f} h"


# ──────────────────────────────────────────────────────────────────────────────
# Language key
# ──────────────────────────────────────────────────────────────────────────────

def _lang_key(row) -> str:
    task = str(row.get("task_type", "") or "").strip()
    lang = str(row.get("language",  "") or "").strip()
    if task == "ast":
        src = str(row.get("source_lang", "") or "").strip()
        tgt = str(row.get("target_lang", "") or "").strip()
        if src and tgt:
            return f"{src}→{tgt}"
        return src or tgt or "unknown"
    return lang or "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Weight engine
# ──────────────────────────────────────────────────────────────────────────────

def compute_weights(df: pd.DataFrame,
                    metric: str = "duration",
                    temperature: float = _TEMPERATURE,
                    min_weight: float = 0.0) -> pd.DataFrame:
    """
    Three-level data-driven weights:
      L1 — task weight      (global)
      L2 — language weight  (within task)
      L3 — dataset weight   (within task × language)

    score = size ^ (1/T)
    """
    df = df.copy()
    df["_lkey"] = df.apply(_lang_key, axis=1)

    def _size(row):
        dur = row.get("total_duration_sec")
        n   = row.get("num_samples")
        dur_ok = dur is not None and not (isinstance(dur, float) and math.isnan(dur)) and float(dur) > 0
        n_ok   = n   is not None and not (isinstance(n,   float) and math.isnan(n))   and float(n)   > 0
        if metric == "duration":
            return float(dur) if dur_ok else (float(n) if n_ok else _EPSILON)
        return float(n) if n_ok else (float(dur) if dur_ok else _EPSILON)

    df["_size"]  = df.apply(_size, axis=1)
    df["_score"] = df["_size"].apply(lambda x: x ** (1.0 / temperature))

    # L3
    gs = df.groupby(["task_type", "_lkey"])["_score"].transform("sum")
    df["dataset_w"] = df["_score"] / gs.replace(0, _EPSILON)

    # L2
    tl = (df.groupby(["task_type", "_lkey"])["_score"].sum()
            .reset_index(name="group_score"))
    ts = tl.groupby("task_type")["group_score"].transform("sum")
    tl["lang_w"] = tl["group_score"] / ts.replace(0, _EPSILON)

    # L1
    ta = tl.groupby("task_type")["group_score"].sum().reset_index(name="task_score")
    grand = ta["task_score"].sum()
    ta["task_w"] = ta["task_score"] / max(grand, _EPSILON)
    tl = tl.merge(ta[["task_type","task_w"]], on="task_type", how="left")

    df = df.merge(tl[["task_type","_lkey","lang_w","task_w"]],
                  on=["task_type","_lkey"], how="left")

    df["effective_prob"]     = df["task_w"] * df["lang_w"] * df["dataset_w"]
    df["effective_prob_pct"] = df["effective_prob"] * 100.0

    if min_weight > 0:
        df["effective_prob"] = df["effective_prob"].clip(lower=min_weight)
        df["effective_prob"] = df["effective_prob"] / df["effective_prob"].sum()
        df["effective_prob_pct"] = df["effective_prob"] * 100.0
        # Recompute task_w / lang_w / dataset_w from adjusted probs for YAML output
        # (keep them as-is; YAML will use the raw L1/L2/L3 decomposition)

    total_n = float(df["num_samples"].sum(skipna=True)) or 1.0
    df["expected_passes"] = (
        (df["effective_prob"] * total_n) / df["num_samples"].clip(lower=1)
    ).round(2)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# YAML writer
# ──────────────────────────────────────────────────────────────────────────────

def write_yaml(df: pd.DataFrame,
               metric: str,
               temperature: float,
               output_path: Path) -> None:
    """Write a NeMo-compatible input_cfg YAML from the weighted DataFrame."""

    tl = (df.groupby(["task_type","_lkey"])
            .agg(
                total_dur_h   = ("total_duration_sec", lambda x: x.sum(skipna=True)/3600),
                total_samples = ("num_samples",        "sum"),
                lang_w        = ("lang_w",             "first"),
                task_w        = ("task_w",             "first"),
            )
            .reset_index()
            .sort_values(["task_type","lang_w"], ascending=[True, False]))

    task_agg = (df.groupby("task_type")
                  .agg(task_w=("task_w","first"))
                  .reset_index()
                  .sort_values("task_w", ascending=False))

    size_label = lambda h, n: (f"{h:.0f} h" if metric == "duration" else fmt_num(int(n)))

    lines = ["input_cfg:"]

    for _, tr in task_agg.iterrows():
        task = tr["task_type"]
        tw   = tr["task_w"]
        t_sub = tl[tl["task_type"] == task]
        t_h   = t_sub["total_dur_h"].sum()
        t_n   = t_sub["total_samples"].sum()

        lines += [
            f"  - type: group",
            f"    weight: {tw:.4f}   # L1 — {task.upper()} ({size_label(t_h, t_n)})",
            f"    tags: {{task: {task}}}",
            f"    input_cfg:",
        ]

        for _, lr in t_sub.iterrows():
            lang  = lr["_lkey"]
            lw    = lr["lang_w"]
            l_h   = lr["total_dur_h"]
            l_n   = lr["total_samples"]

            if task == "ast" and "→" in lang:
                src, tgt = lang.split("→", 1)
                tags_str = f"{{task: {task}, source_lang: {src}, target_lang: {tgt}}}"
            else:
                tags_str = f"{{task: {task}, lang: {lang}}}"

            lines += [
                f"      - type: group",
                f"        weight: {lw:.4f}   # L2 — {lang} ({size_label(l_h, l_n)})",
                f"        tags: {tags_str}",
                f"        input_cfg:",
            ]

            ds_sub = (df[(df["task_type"] == task) & (df["_lkey"] == lang)]
                      .sort_values("dataset_w", ascending=False))

            for _, dr in ds_sub.iterrows():
                dw     = dr["dataset_w"]
                d_h    = dr.get("total_duration_sec", 0) or 0
                d_n    = dr.get("num_samples", 0) or 0
                d_size = fmt_hours(d_h) if metric == "duration" else fmt_num(d_n)
                m_path = dr.get("raw_manifest_path", "<path>")

                if len(ds_sub) > 1:
                    lines += [
                        f"          - type: multimodal_conversation",
                        f"            weight: {dw:.4f}   # L3 — {dr['dataset_name']} ({d_size})",
                        f"            manifest_filepath: {m_path}",
                    ]
                else:
                    lines += [
                        f"          - type: multimodal_conversation",
                        f"            # {dr['dataset_name']} ({d_size})",
                        f"            manifest_filepath: {m_path}",
                    ]

        lines.append("")  # blank line between tasks

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ YAML → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Summary printer
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, metric: str, temperature: float) -> None:
    print("\n" + "="*72)
    print("  SAMPLING WEIGHT SUMMARY")
    print(f"  metric={metric}  T={temperature}")
    print("="*72)
    print(f"\n  {'Task':<8} {'Lang':<12} {'Dataset':<30} "
          f"{'ds_w':>8} {'samp%':>7} {'passes':>7}")
    print("  " + "-"*72)

    for _, r in df.sort_values(["task_type","_lkey","effective_prob_pct"],
                               ascending=[True,True,False]).iterrows():
        ep   = r.get("expected_passes", "?")
        flag = (" ⚠" if isinstance(ep,(int,float)) and ep > 5
                else " ⚑" if isinstance(ep,(int,float)) and ep < 0.1 else "")
        print(f"  {str(r['task_type']):<8} {str(r['_lkey']):<12} "
              f"{str(r['dataset_name'])[:29]:<30} "
              f"{r['dataset_w']:>8.4f} {r['effective_prob_pct']:>6.3f}%"
              f"  {ep:>5}×{flag}")

    print("  " + "-"*72)
    print(f"  {'TOTAL':>53} {df['effective_prob_pct'].sum():>6.1f}%")

    over  = df[df["expected_passes"] > 5]
    under = df[df["expected_passes"] < 0.1]
    if not over.empty:
        names = ", ".join(f"{r['dataset_name']} ({r['expected_passes']:.1f}×)"
                          for _, r in over.head(5).iterrows())
        print(f"\n  ⚠  Over-sampled  (>5×): {names}")
    if not under.empty:
        names = ", ".join(f"{r['dataset_name']} ({r['expected_passes']:.3f}×)"
                          for _, r in under.head(5).iterrows())
        print(f"  ⚑  Under-sampled (<0.1×): {names}")
    print("="*72 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="metadata CSV → suggested NeMo sampling YAML.")
    parser.add_argument("csv_path",
                        help="metadata.csv produced by generate_csv.py")
    parser.add_argument("--output",      default="suggested.yaml",
                        help="Output YAML path (default: suggested.yaml).")
    parser.add_argument("--metric",      default="duration",
                        choices=["duration","samples"],
                        help="Size metric for weighting (default: duration).")
    parser.add_argument("--temperature", type=float, default=_TEMPERATURE,
                        help=f"Smoothing temperature T>0 (default {_TEMPERATURE}). "
                             "T=1→proportional, T=2→moderate, T→∞→uniform.")
    parser.add_argument("--min_weight",  type=float, default=0.0001,
                        help="Minimum effective probability floor (default: 0.0001).")
    parser.add_argument("--yaml_source", default=None,
                        help="If CSV contains multiple YAMLs, filter to this one.")
    parser.add_argument("--skip_missing", action="store_true",
                        help="Drop rows where file_exists=False before computing weights.")
    args = parser.parse_args()

    print(f"\n📊 Loading CSV : {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    print(f"   Loaded {len(df)} rows.")

    if args.yaml_source:
        df = df[df["yaml_source"] == args.yaml_source].copy()
        print(f"   Filtered to yaml_source='{args.yaml_source}': {len(df)} rows.")

    if args.skip_missing and "file_exists" in df.columns:
        before = len(df)
        df = df[df["file_exists"] == True].copy()
        print(f"   Dropped {before - len(df)} missing-file rows.")

    df = df[df["num_samples"].notna()].copy()

    print(f"\n⚖️  metric={args.metric}  T={args.temperature}  min_weight={args.min_weight}")
    df = compute_weights(df, metric=args.metric,
                         temperature=args.temperature,
                         min_weight=args.min_weight)

    print_summary(df, args.metric, args.temperature)
    write_yaml(df, args.metric, args.temperature, Path(args.output))


if __name__ == "__main__":
    main()