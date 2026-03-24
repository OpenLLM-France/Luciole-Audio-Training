"""
analyze_yaml_cfg.py
===================
Analyze a NeMo sampling YAML against a metadata CSV.
Reads the weights from the YAML, matches them to dataset statistics
from the CSV, and estimates how many passes each dataset will see
after a given number of training steps.

Usage
-----
python analyze_yaml_cfg.py out/metadata.csv out/suggested.yaml

# With steps and samples per step
python analyze_yaml_cfg.py out/metadata.csv out/suggested.yaml \
    --steps 1000000 --samples_per_step 32

# With a specific yaml_source filter
python analyze_yaml_cfg.py out/metadata.csv out/suggested.yaml --yaml_source input_cfg_train

# Save report to file
python analyze_yaml_cfg.py out/metadata.csv out/suggested.yaml --output notes.txt
"""

import argparse
import math
import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml

# Allow importing from assets/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_csv_metadata import flatten_manifests

warnings.filterwarnings("ignore")

_EPSILON            = 1e-9
_WARN_THRESHOLD     = 5.0       # ⚠ if passes > this
_CRITICAL_THRESHOLD = 10.0      # ⚠⚠ if passes > this
_UNDER_THRESHOLD    = 0.25      # ⚑ if passes < this
_CRITICAL_UNDER     = 0.1       # ⚑⚑ if passes < this


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


def _task_level_weights(yaml_path: str) -> dict:
    """Extract task-level weights from the YAML (top-level groups)."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    task_w = {}
    for node in cfg.get("input_cfg", []):
        if not isinstance(node, dict):
            continue
        tags = node.get("tags", {}) or {}
        task = tags.get("task", "unknown")
        w = float(node.get("weight", 1.0) or 1.0)
        task_w[task] = w
    return task_w


# ──────────────────────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, title: str = "") -> None:
    """Legacy wrapper used by suggest_yaml_cfg.py."""
    _task_order = {"asr": 0, "ast": 1, "qa": 2, "aqa": 3}
    _lang_order = {"en": 0, "fr": 1}
    df["_task_sort"] = df["task_type"].map(lambda t: _task_order.get(t, 99))
    df["_lang_sort"] = df["_lkey"].map(lambda l: _lang_order.get(l, 99))

    print("\n" + "=" * 80)
    print("  SAMPLING WEIGHT SUMMARY")
    if title:
        print(f"  {title}")
    print("=" * 80)

    col = "passes" if "passes" in df.columns else "expected_passes"

    print(f"\n  (⚑⚑ <{_CRITICAL_UNDER} passes  ⚑ <{_UNDER_THRESHOLD} passes"
          f"  ⚠ >{_WARN_THRESHOLD:.0f} passes  ⚠⚠ >{_CRITICAL_THRESHOLD:.0f} passes)")
    print(f"\n  {'Task':<8} {'Lang':<12} {'Dataset':<35} "
          f"{'w/group':>8} {'prob%':>7} {'passes':>8}")
    print("  " + "-" * 80)

    for _, r in df.sort_values(
        ["_task_sort", "task_type", "_lang_sort", "_lkey", "num_samples"],
        ascending=[True, True, True, True, False],
    ).iterrows():
        ep = r.get(col, "?")
        flag = (" ⚠⚠" if isinstance(ep, (int, float)) and ep > _CRITICAL_THRESHOLD
                else " ⚠" if isinstance(ep, (int, float)) and ep > _WARN_THRESHOLD
                else " ⚑⚑" if isinstance(ep, (int, float)) and ep < _CRITICAL_UNDER
                else " ⚑" if isinstance(ep, (int, float)) and ep < _UNDER_THRESHOLD else "")
        print(f"  {str(r['task_type']):<8} {str(r['_lkey']):<12} "
              f"{str(r['dataset_name'])[:34]:<35} "
              f"{r['dataset_w']:>8.4f} {r['effective_prob_pct']:>6.3f}%"
              f"  {ep:>7}×{flag}")

    print("  " + "-" * 80)
    print(f"  {'TOTAL':>58} {df['effective_prob_pct'].sum():>6.1f}%")
    print("=" * 80 + "\n")


def generate_report(df: pd.DataFrame, yaml_path: str,
                    steps: int, samples_per_step: int) -> str:
    lines = []
    w = lines.append

    task_weights = _task_level_weights(yaml_path)
    total_samples_drawn = steps * samples_per_step

    w("")
    w(f"📂 Reading weights: {yaml_path}")
    w(f"   Tasks:  {task_weights}")
    lang_keys = df["_lkey"].nunique()
    w(f"   Langs:  {lang_keys} entries")
    w("")
    w(f"⚙️  steps={steps:,}  samples_per_step={samples_per_step}"
      f"  total_samples={total_samples_drawn:,}")

    w("")
    w("=" * 80)
    w("  ESTIMATED PASSES PER DATASET")
    w(f"  {steps:,} steps × {samples_per_step} samples/step"
      f" = {total_samples_drawn:,} total samples drawn")
    w("=" * 80)

    w(f"\n  (⚑⚑ <{_CRITICAL_UNDER} passes  ⚑ <{_UNDER_THRESHOLD} passes"
      f"  ⚠ >{_WARN_THRESHOLD:.0f} passes  ⚠⚠ >{_CRITICAL_THRESHOLD:.0f} passes)")

    w(f"\n  {'Task':<8} {'Lang':<12} {'Dataset':<35} "
      f"{'w/group':>8} {'prob%':>7} {'passes':>8}")
    w("  " + "-" * 80)

    _task_order = {"asr": 0, "ast": 1, "qa": 2, "aqa": 3}
    _lang_order = {"en": 0, "fr": 1}
    df["_task_sort"] = df["task_type"].map(lambda t: _task_order.get(t, 99))
    df["_lang_sort"] = df["_lkey"].map(lambda l: _lang_order.get(l, 99))
    for _, r in df.sort_values(
        ["_task_sort", "task_type", "_lang_sort", "_lkey", "num_samples"],
        ascending=[True, True, True, True, False],
    ).iterrows():
        ep = r["passes"]
        flag = (" ⚠⚠" if ep > _CRITICAL_THRESHOLD
                else " ⚠" if ep > _WARN_THRESHOLD
                else " ⚑⚑" if ep < _CRITICAL_UNDER
                else " ⚑" if ep < _UNDER_THRESHOLD else "")
        w(f"  {str(r['task_type']):<8} {str(r['_lkey']):<12} "
          f"{str(r['dataset_name'])[:34]:<35} "
          f"{r['dataset_w']:>8.4f} {r['effective_prob_pct']:>6.3f}%"
          f"  {ep:>7.2f}×{flag}")

    w("  " + "-" * 80)
    w(f"  {'TOTAL':>58} {df['effective_prob_pct'].sum():>6.1f}%")

    critical_over = df[df["passes"] > _CRITICAL_THRESHOLD]
    warn_over = df[(df["passes"] > _WARN_THRESHOLD) & (df["passes"] <= _CRITICAL_THRESHOLD)]
    critical_under = df[df["passes"] < _CRITICAL_UNDER]
    warn_under = df[(df["passes"] < _UNDER_THRESHOLD) & (df["passes"] >= _CRITICAL_UNDER)]
    if not critical_over.empty:
        names = ", ".join(
            f"{r['dataset_name']} ({r['passes']:.1f}×)"
            for _, r in critical_over.sort_values("passes", ascending=False).head(5).iterrows()
        )
        w(f"\n  ⚠⚠ Critical over-sampled (>{_CRITICAL_THRESHOLD:.0f}×): {names}")
    if not warn_over.empty:
        names = ", ".join(
            f"{r['dataset_name']} ({r['passes']:.1f}×)"
            for _, r in warn_over.sort_values("passes", ascending=False).head(5).iterrows()
        )
        w(f"  ⚠  Over-sampled (>{_WARN_THRESHOLD:.0f}×): {names}")
    if not critical_under.empty:
        names = ", ".join(
            f"{r['dataset_name']} ({r['passes']:.3f}×)"
            for _, r in critical_under.sort_values("passes").head(5).iterrows()
        )
        w(f"  ⚑⚑ Critical under-sampled (<{_CRITICAL_UNDER}×): {names}")
    if not warn_under.empty:
        names = ", ".join(
            f"{r['dataset_name']} ({r['passes']:.3f}×)"
            for _, r in warn_under.sort_values("passes").head(5).iterrows()
        )
        w(f"  ⚑  Under-sampled (<{_UNDER_THRESHOLD}×): {names}")
    w("=" * 80)

    # Over-sampled detail table
    all_over = df[df["passes"] > _WARN_THRESHOLD]
    if not all_over.empty:
        w("")
        w("=" * 80)
        w("  OVER-SAMPLED DATASETS (details)")
        w("=" * 80)
        w(f"\n  {'Task':<8} {'Lang':<12} {'Dataset':<35} "
          f"{'samples':>10} {'passes':>8}")
        w("  " + "-" * 80)
        for _, r in all_over.sort_values("passes", ascending=False).iterrows():
            ep = r["passes"]
            flag = " ⚠⚠" if ep > _CRITICAL_THRESHOLD else " ⚠"
            w(f"  {str(r['task_type']):<8} {str(r['_lkey']):<12} "
              f"{str(r['dataset_name'])[:34]:<35} "
              f"{int(r['num_samples']):>10,} {ep:>7.2f}×{flag}")
        w("  " + "-" * 80)
        w("=" * 80)

    w("")

    return "\n".join(lines)


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
    parser.add_argument("--steps", type=int, default=1_000_000,
                        help="Number of training steps (default: 1_000_000).")
    parser.add_argument("--samples_per_step", type=int, default=32,
                        help="Samples consumed per step (default: 32).")
    parser.add_argument("--yaml_source", default=None,
                        help="Filter CSV to this yaml_source value.")
    parser.add_argument("--skip_missing", action="store_true",
                        help="Drop rows where file_exists=False.")
    parser.add_argument("--output", default=None,
                        help="Write report to this file (default: stdout).")
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

    # Estimate passes
    total_samples_drawn = args.steps * args.samples_per_step
    df["passes"] = (
        (df["effective_prob"] * total_samples_drawn)
        / df["num_samples"].clip(lower=1)
    ).round(2)

    matched   = (df["effective_prob"] > 0).sum()
    unmatched = (df["effective_prob"] == 0).sum()
    if unmatched > 0:
        print(f"   ⚠ {unmatched} CSV rows had no matching entry in the YAML:")
        for _, r in df[df["effective_prob"] == 0].iterrows():
            print(f"      - {r['dataset_name']} ({r.get('raw_manifest_path', '?')})")
    print(f"   Matched {matched} / {len(df)} rows.")

    # Generate report
    report = generate_report(df, args.yaml_path, args.steps, args.samples_per_step)

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"\n✅ Report → {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
