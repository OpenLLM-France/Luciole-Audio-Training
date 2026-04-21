"""
suggest_yaml_cfg.py
===============
Given a metadata CSV (from generate_csv_metadata.py),
compute data-driven sampling weights and write a suggested YAML.

Usage
-----
python suggest_yaml_cfg.py metadata.csv \\
    --output suggested.yaml \\
    --metric duration \\
    --temperature 2.0 \\
    --min_weight 0.0001

# Guide output with an existing YAML's task/language weights and dataset list
python suggest_yaml_cfg.py metadata.csv \\
    --yaml input_cfg_train.yaml \\
    --input_weights_mode hard
"""

import argparse
import math
import warnings
from pathlib import Path

import pandas as pd
import yaml

from analyze_yaml_cfg import (
    _EPSILON, _lang_key, fmt_hours, fmt_num, print_summary,
)

warnings.filterwarnings("ignore")

_TEMPERATURE = 2.0


# ──────────────────────────────────────────────────────────────────────────────
# Input weight parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_input_weights(yaml_path: str) -> tuple:
    """
    Parse a NeMo YAML and extract the hierarchical weights.

    Returns
    -------
    task_weights : dict[str, float]
        {task_name: weight} as written in the YAML (not normalised).
    lang_weights : dict[tuple[str, str], float]
        {(task_name, lang_key): weight} for language sub-groups.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if isinstance(cfg, list):
        cfg = {"input_cfg": cfg}

    task_weights = {}
    lang_weights = {}

    def _validate_tags(tags, context=""):
        for k, v in tags.items():
            if v is None:
                raise ValueError(
                    f"YAML tag '{k}' has no value (parsed as None){context}. "
                    f"Add a space after the colon: '{k}: <value>' instead of '{k}<value>'"
                )

    for top in cfg.get("input_cfg", []):
        if not isinstance(top, dict) or top.get("type") != "group":
            continue
        tags = top.get("tags", {}) or {}
        _validate_tags(tags)
        task = tags.get("task", "")
        if not task:
            continue
        task_weights[task] = float(top.get("weight", 1.0) or 1.0)

        for child in top.get("input_cfg", []):
            if not isinstance(child, dict) or child.get("type") != "group":
                continue
            ctags = child.get("tags", {}) or {}
            _validate_tags(ctags, context=f" in task '{task}'")
            lang = ctags.get("lang", "")
            src  = ctags.get("source_lang", "")
            tgt  = ctags.get("target_lang", "")
            if src and tgt:
                lkey = f"{src}→{tgt}"
            elif lang:
                lkey = lang
            else:
                continue
            lang_weights[(task, lkey)] = float(child.get("weight", 1.0) or 1.0)

    return task_weights, lang_weights


def parse_yaml_manifests(yaml_path: str) -> set:
    """Extract all manifest_filepath values from a NeMo YAML (recursive)."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = set()

    def _walk(node):
        if isinstance(node, dict):
            mf = node.get("manifest_filepath")
            if mf:
                paths.add(mf)
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(cfg)
    return paths


# ──────────────────────────────────────────────────────────────────────────────
# Weight engine
# ──────────────────────────────────────────────────────────────────────────────

def compute_weights(df: pd.DataFrame,
                    metric: str = "duration",
                    temperature: float = _TEMPERATURE,
                    min_weight: float = 0.0,
                    max_passes: float = 0.0,
                    input_task_w: dict = None,
                    input_lang_w: dict = None,
                    input_weights_mode: str = "hard") -> pd.DataFrame:
    """
    Three-level data-driven weights:
      L1 — task weight      (global)
      L2 — language weight  (within task)
      L3 — dataset weight   (within task × language)

    score = size ^ (1/T)

    When input weights are provided (--input_weights), the mode controls how
    they interact with data-driven weights:
      hard      — L1/L2 are taken directly from the YAML; L3 stays data-driven.
      soft      — L1/L2 = product(yaml_w, data_w), renormalised.
      weighting — each dataset score is multiplied by its YAML task×lang
                   weight *before* computing all three levels.
    """
    df = df.copy()
    df["_lkey"] = df.apply(_lang_key, axis=1)

    def _size(row):
        if metric == "duration":
            val = row.get("total_duration_sec")
            col = "total_duration_sec"
        else:
            val = row.get("num_samples")
            col = "num_samples"
        if val is None or (isinstance(val, float) and math.isnan(val)) or float(val) <= 0:
            raise ValueError(
                f"Dataset '{row.get('dataset_name', '?')}': "
                f"invalid {col}={val!r} (metric={metric})"
            )
        return float(val)

    df["_size"]  = df.apply(_size, axis=1)
    df["_score"] = df["_size"].apply(lambda x: x ** (1.0 / temperature))

    # When metric is duration, convert from duration-proportional scores to
    # sampling-proportional scores.  The YAML weights are per-sample
    # probabilities, so to achieve a target duration share p_i for dataset i
    # we need:  w_i ∝ p_i / avg_duration_i = D^(1/T) × n / D = n × D^(1/T−1)
    if metric == "duration":
        avg_dur = df["_size"] / df["num_samples"].clip(lower=1)
        df["_score"] = df["_score"] / avg_dur.replace(0, _EPSILON)

    # ── weighting mode: fold YAML weights into scores before computing levels
    if input_weights_mode == "weighting" and (input_task_w or input_lang_w):
        def _yaml_mult(row):
            tw = (input_task_w or {}).get(row["task_type"], 1.0)
            lw = (input_lang_w or {}).get((row["task_type"], row["_lkey"]), 1.0)
            return tw * lw
        df["_score"] = df["_score"] * df.apply(_yaml_mult, axis=1)

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

    # ── hard / soft: override or blend L1 and L2 with YAML weights
    if input_weights_mode in ("hard", "soft") and (input_task_w or input_lang_w):
        _apply_input_weights(ta, tl, input_task_w, input_lang_w, input_weights_mode)

    tl = tl.merge(ta[["task_type","task_w"]], on="task_type", how="left")

    df = df.merge(tl[["task_type","_lkey","lang_w","task_w"]],
                  on=["task_type","_lkey"], how="left")

    df["effective_prob"]     = df["task_w"] * df["lang_w"] * df["dataset_w"]
    df["effective_prob_pct"] = df["effective_prob"] * 100.0

    if min_weight > 0:
        df["effective_prob"] = df["effective_prob"].clip(lower=min_weight)
        df["effective_prob"] = df["effective_prob"] / df["effective_prob"].sum()
        df["effective_prob_pct"] = df["effective_prob"] * 100.0

    total_n = float(df["num_samples"].sum(skipna=True)) or 1.0

    # ── max_passes: cap L3 weights so no dataset exceeds the threshold
    if max_passes > 0:
        df = _cap_passes(df, total_n, max_passes)

    df["expected_passes"] = (
        (df["effective_prob"] * total_n) / df["num_samples"].clip(lower=1)
    ).round(2)

    return df


def _cap_passes(df, total_n, max_passes, max_iter=50):
    """Reduce dataset_w for datasets exceeding max_passes, redistributing
    the freed weight to uncapped siblings in the same (task, lang) group.
    Iterates because capping one dataset raises the effective_prob of others."""
    for _ in range(max_iter):
        df["effective_prob"] = df["task_w"] * df["lang_w"] * df["dataset_w"]
        df["effective_prob_pct"] = df["effective_prob"] * 100.0
        passes = (df["effective_prob"] * total_n) / df["num_samples"].clip(lower=1)
        over = passes > max_passes
        if not over.any():
            break
        for (task, lkey), grp in df[over].groupby(["task_type", "_lkey"]):
            mask = (df["task_type"] == task) & (df["_lkey"] == lkey)
            group = df.loc[mask]
            group_passes = (group["effective_prob"] * total_n) / group["num_samples"].clip(lower=1)
            capped = group_passes > max_passes
            if capped.all():
                # All exceed — just normalise evenly
                df.loc[mask, "dataset_w"] = 1.0 / len(group)
                continue
            # Set capped datasets to exactly max_passes worth of dataset_w
            tw = group["task_w"].iloc[0]
            lw = group["lang_w"].iloc[0]
            group_prob = tw * lw
            if group_prob <= 0:
                continue
            for idx in group.index[capped]:
                target_prob = max_passes * df.loc[idx, "num_samples"] / total_n
                df.loc[idx, "dataset_w"] = target_prob / group_prob
            # Redistribute remaining weight to uncapped datasets proportionally
            used = df.loc[mask & capped, "dataset_w"].sum()
            remaining = max(1.0 - used, 0.0)
            uncapped_mask = mask & ~capped
            uncapped_sum = df.loc[uncapped_mask, "dataset_w"].sum()
            if uncapped_sum > 0 and remaining > 0:
                df.loc[uncapped_mask, "dataset_w"] *= remaining / uncapped_sum
    # Refresh effective_prob after capping
    df["effective_prob"] = df["task_w"] * df["lang_w"] * df["dataset_w"]
    df["effective_prob_pct"] = df["effective_prob"] * 100.0
    return df


def _apply_input_weights(ta, tl, input_task_w, input_lang_w, mode):
    """Modify ta['task_w'] and tl['lang_w'] in-place for hard/soft modes."""

    # ── L1: task weights ──────────────────────────────────────────────────
    if input_task_w:
        yaml_total = sum(input_task_w.values())
        if mode == "hard":
            for task, w in input_task_w.items():
                mask = ta["task_type"] == task
                if mask.any():
                    ta.loc[mask, "task_w"] = w / yaml_total
        else:  # soft
            for task, w in input_task_w.items():
                mask = ta["task_type"] == task
                if mask.any():
                    ta.loc[mask, "task_w"] *= w / yaml_total
            s = ta["task_w"].sum()
            if s > 0:
                ta["task_w"] /= s

    # ── L2: language weights (per task) ───────────────────────────────────
    if input_lang_w:
        for task in tl["task_type"].unique():
            yaml_langs = {lk: w for (t, lk), w in input_lang_w.items() if t == task}
            if not yaml_langs:
                continue
            yaml_total = sum(yaml_langs.values())
            mask = tl["task_type"] == task
            if mode == "hard":
                for lkey, w in yaml_langs.items():
                    lmask = mask & (tl["_lkey"] == lkey)
                    if lmask.any():
                        tl.loc[lmask, "lang_w"] = w / yaml_total
            else:  # soft
                for lkey, w in yaml_langs.items():
                    lmask = mask & (tl["_lkey"] == lkey)
                    if lmask.any():
                        tl.loc[lmask, "lang_w"] *= w / yaml_total
                s = tl.loc[mask, "lang_w"].sum()
                if s > 0:
                    tl.loc[mask, "lang_w"] /= s


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
            .sort_values(["task_w","lang_w"], ascending=[False, False]))

    task_agg = (df.groupby("task_type")
                  .agg(task_w=("task_w","first"))
                  .reset_index()
                  .sort_values("task_w", ascending=False))

    def size_label(dur_h, n_samples):
        return f"{fmt_hours(dur_h * 3600)}, {fmt_num(n_samples)} samples"

    lines = []

    for _, tr in task_agg.iterrows():
        task = tr["task_type"]
        tw   = tr["task_w"]
        t_sub = tl[tl["task_type"] == task]
        t_h   = t_sub["total_dur_h"].sum()
        t_n   = t_sub["total_samples"].sum()

        lines += [
            f"- type: group",
            f"  weight: {tw:.4f}   # L1 — {task.upper()} ({size_label(t_h, t_n)})",
            f"  tags: {{task: {task}}}",
            f"  input_cfg:",
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
                f"    - type: group",
                f"      weight: {lw:.4f}   # L2 — {lang} ({size_label(l_h, l_n)})",
                f"      tags: {tags_str}",
                f"      input_cfg:",
            ]

            ds_sub = (df[(df["task_type"] == task) & (df["_lkey"] == lang)]
                      .sort_values("dataset_w", ascending=False))

            for _, dr in ds_sub.iterrows():
                dw     = dr["dataset_w"]
                d_h    = (dr.get("total_duration_sec", 0) or 0) / 3600
                d_n    = dr.get("num_samples", 0) or 0
                d_size = size_label(d_h, d_n)
                m_path = dr.get("raw_manifest_path", "<path>")

                if len(ds_sub) > 1:
                    lines += [
                        f"        - type: multimodal_conversation",
                        f"          weight: {dw:.4f}   # L3 — {dr['dataset_name']} ({d_size})",
                        f"          manifest_filepath: {m_path}",
                    ]
                else:
                    lines += [
                        f"        - type: multimodal_conversation",
                        f"          # {dr['dataset_name']} ({d_size})",
                        f"          manifest_filepath: {m_path}",
                    ]

        lines.append("")  # blank line between tasks

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ YAML → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="metadata CSV → suggested NeMo sampling YAML.")
    parser.add_argument("csv_path",
                        help="metadata.csv produced by generate_csv.py")
    parser.add_argument("--output",      default=None,
                        help="Output YAML path. If omitted, only print the summary.")
    parser.add_argument("--metric",      default="duration",
                        choices=["duration","samples"],
                        help="Size metric for weighting (default: duration).")
    parser.add_argument("--temperature", type=float, default=_TEMPERATURE,
                        help=f"Smoothing temperature T>0 (default {_TEMPERATURE}). "
                             "T=1→proportional, T=2→moderate, T→∞→uniform.")
    parser.add_argument("--min_weight",  type=float, default=0.0001,
                        help="Minimum effective probability floor (default: 0.0001).")
    parser.add_argument("--max_passes",  type=float, default=0.0,
                        help="Maximum expected passes per dataset (default: 0 = unlimited). "
                             "Caps L3 weights and redistributes excess within each group.")
    parser.add_argument("--skip_missing", action="store_true",
                        help="Drop rows where file_exists=False before computing weights.")
    parser.add_argument("--yaml", "--input_weights", default=None, dest="yaml",
                        help="Optional NeMo YAML: used to extract task/language weights "
                             "AND to filter the CSV to only datasets present in the YAML.")
    parser.add_argument("--input_weights_mode", default="hard",
                        choices=["hard", "soft", "weighting"],
                        help="How --yaml weights are applied (default: hard). "
                             "hard: impose YAML weights for L1/L2, L3 stays data-driven. "
                             "soft: blend YAML and data-driven weights (product, renormalised). "
                             "weighting: multiply each dataset score by its YAML task×lang weight.")
    parser.add_argument("--ignore_missing_manifest", action="store_true",
                        help="Only warn (instead of error) when YAML manifests are missing from CSV.")
    args = parser.parse_args()

    print(f"\n📊 Loading CSV : {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    print(f"   Loaded {len(df)} rows.")

    if args.skip_missing and "file_exists" in df.columns:
        before = len(df)
        df = df[df["file_exists"] == True].copy()
        print(f"   Dropped {before - len(df)} missing-file rows.")

    df = df[df["num_samples"].notna()].copy()

    input_task_w, input_lang_w = None, None
    if args.yaml:
        print(f"\n📂 Reading YAML: {args.yaml}  (mode={args.input_weights_mode})")
        input_task_w, input_lang_w = parse_input_weights(args.yaml)
        print(f"   Tasks:  {input_task_w}")
        print(f"   Langs:  {len(input_lang_w)} entries")

        # Filter CSV to only datasets present in the YAML
        yaml_manifests = parse_yaml_manifests(args.yaml)
        csv_manifests = set(df["raw_manifest_path"].dropna())
        missing = yaml_manifests - csv_manifests
        if missing:
            msg = f"{len(missing)} dataset(s) in YAML but missing from CSV:\n"
            msg += "\n".join(f"      {m}" for m in sorted(missing))
            if args.ignore_missing_manifest:
                print(f"\n⚠️  {msg}")
            else:
                raise SystemExit(f"\n❌ {msg}\n   Use --ignore_missing_manifest to skip this error.")
        extra = csv_manifests - yaml_manifests
        if extra:
            before = len(df)
            df = df[df["raw_manifest_path"].isin(yaml_manifests)].copy()
            print(f"   Dropped {before - len(df)} CSV rows not in YAML ({len(df)} remaining).")

    max_p = f"  max_passes={args.max_passes}" if args.max_passes > 0 else ""
    print(f"\n⚖️  metric={args.metric}  T={args.temperature}  min_weight={args.min_weight}{max_p}")
    df = compute_weights(df, metric=args.metric,
                         temperature=args.temperature,
                         min_weight=args.min_weight,
                         max_passes=args.max_passes,
                         input_task_w=input_task_w,
                         input_lang_w=input_lang_w,
                         input_weights_mode=args.input_weights_mode)

    print_summary(df, title=f"metric={args.metric}  T={args.temperature}")
    if args.output:
        write_yaml(df, args.metric, args.temperature, Path(args.output))


if __name__ == "__main__":
    main()