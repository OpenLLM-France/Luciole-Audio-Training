"""
analyze_yaml_cfg.py
===================
Analyze a NeMo sampling YAML against a metadata CSV.
Reads the weights from the YAML, matches them to dataset statistics
from the CSV, and prints a sampling summary with outlier detection.

Usage
-----
python analyze_yaml_cfg.py out/metadata.csv out/suggested.yaml

# With a specific yaml_source filter
python analyze_yaml_cfg.py out/metadata.csv out/suggested.yaml --yaml_source input_cfg_train
"""

import argparse
import math
import warnings
from pathlib import Path

import pandas as pd
import yaml

warnings.filterwarnings("ignore")

_EPSILON      = 1e-9
_OVER_FACTOR  = 10.0   # flag if expected_passes > median × this factor
_UNDER_FACTOR = 0.1    # flag if expected_passes < median × this factor


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
# YAML weight reader
# ──────────────────────────────────────────────────────────────────────────────

def read_yaml_weights(yaml_path: str) -> dict:
    """
    Walk a NeMo input_cfg YAML and return a dict mapping
    manifest_filepath → effective weight (product of all ancestor weights).
    The returned weights are normalised to sum to 1.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    weights = {}

    def recurse(node, parent_weight=1.0):
        if not isinstance(node, dict):
            return
        w     = float(node.get("weight", 1.0) or 1.0)
        ntype = node.get("type", "")
        if ntype == "multimodal_conversation":
            path = node.get("manifest_filepath", "")
            weights[path] = parent_weight * w
            return
        if ntype == "group":
            for child in node.get("input_cfg", []):
                recurse(child, parent_weight=parent_weight * w)

    for top in cfg.get("input_cfg", []):
        recurse(top, parent_weight=1.0)

    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    return weights


# ──────────────────────────────────────────────────────────────────────────────
# Summary printer
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, title: str = "") -> None:
    print("\n" + "="*72)
    print("  SAMPLING WEIGHT SUMMARY")
    if title:
        print(f"  {title}")
    print("="*72)

    median_ep = float(df["expected_passes"].median()) if not df.empty else 1.0
    over_thr  = median_ep * _OVER_FACTOR
    under_thr = median_ep * _UNDER_FACTOR

    print(f"\n  median passes: {median_ep:.2f}×  "
          f"(⚠ >{over_thr:.2f}×  ⚑ <{under_thr:.3f}×)")
    print(f"\n  {'Task':<8} {'Lang':<12} {'Dataset':<30} "
          f"{'ds_w':>8} {'samp%':>7} {'passes':>7}")
    print("  " + "-"*72)

    for _, r in df.sort_values(["task_type","_lkey","effective_prob_pct"],
                               ascending=[True,True,True]).iterrows():
        ep   = r.get("expected_passes", "?")
        flag = (" ⚠" if isinstance(ep,(int,float)) and ep > over_thr
                else " ⚑" if isinstance(ep,(int,float)) and ep < under_thr else "")
        print(f"  {str(r['task_type']):<8} {str(r['_lkey']):<12} "
              f"{str(r['dataset_name'])[:29]:<30} "
              f"{r['dataset_w']:>8.4f} {r['effective_prob_pct']:>6.3f}%"
              f"  {ep:>5}×{flag}")

    print("  " + "-"*72)
    print(f"  {'TOTAL':>53} {df['effective_prob_pct'].sum():>6.1f}%")

    over  = df[df["expected_passes"] > over_thr]
    under = df[df["expected_passes"] < under_thr]
    if not over.empty:
        names = ", ".join(f"{r['dataset_name']} ({r['expected_passes']:.1f}×)"
                          for _, r in over.sort_values("expected_passes", ascending=False).head(5).iterrows())
        print(f"\n  ⚠  Over-sampled  (>{over_thr:.1f}×): {names}")
    if not under.empty:
        names = ", ".join(f"{r['dataset_name']} ({r['expected_passes']:.3f}×)"
                          for _, r in under.sort_values("expected_passes").head(5).iterrows())
        print(f"  ⚑  Under-sampled (<{under_thr:.3f}×): {names}")
    print("="*72 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze a NeMo sampling YAML against a metadata CSV.")
    parser.add_argument("csv_path",
                        help="metadata.csv produced by generate_csv_metadata.py")
    parser.add_argument("yaml_path",
                        help="NeMo input_cfg YAML to analyze (e.g. suggested.yaml).")
    parser.add_argument("--yaml_source", default=None,
                        help="Filter CSV to this yaml_source value.")
    parser.add_argument("--skip_missing", action="store_true",
                        help="Drop rows where file_exists=False.")
    args = parser.parse_args()

    print(f"\n📊 Loading CSV  : {args.csv_path}")
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

    print(f"\n📂 Reading YAML : {args.yaml_path}")
    yaml_weights = read_yaml_weights(args.yaml_path)
    print(f"   Found {len(yaml_weights)} manifest entries.")

    # Map weights onto the dataframe via raw_manifest_path
    df["_lkey"] = df.apply(_lang_key, axis=1)
    df["effective_prob"] = df["raw_manifest_path"].map(yaml_weights).fillna(0)
    total_prob = df["effective_prob"].sum()
    if total_prob > 0:
        df["effective_prob"] = df["effective_prob"] / total_prob
    df["effective_prob_pct"] = df["effective_prob"] * 100.0

    # dataset_w: weight within task+lang group
    gs = df.groupby(["task_type", "_lkey"])["effective_prob"].transform("sum")
    df["dataset_w"] = df["effective_prob"] / gs.replace(0, _EPSILON)

    # expected_passes
    total_n = float(df["num_samples"].sum(skipna=True)) or 1.0
    df["expected_passes"] = (
        (df["effective_prob"] * total_n) / df["num_samples"].clip(lower=1)
    ).round(2)

    matched   = (df["effective_prob"] > 0).sum()
    unmatched = (df["effective_prob"] == 0).sum()
    if unmatched > 0:
        print(f"   ⚠ {unmatched} CSV rows had no matching entry in the YAML.")
    print(f"   Matched {matched} / {len(df)} rows.")

    print_summary(df, title=f"YAML: {args.yaml_path}")


if __name__ == "__main__":
    main()
