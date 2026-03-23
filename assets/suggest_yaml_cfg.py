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

# Guide output with an existing YAML's task/language weights
python suggest_yaml_cfg.py metadata.csv \\
    --input_weights input_cfg_train.yaml \\
    --input_weights_mode hard
"""

import argparse
import math
import warnings
from pathlib import Path

import pandas as pd
import yaml

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

    task_weights = {}
    lang_weights = {}

    for top in cfg.get("input_cfg", []):
        if not isinstance(top, dict) or top.get("type") != "group":
            continue
        tags = top.get("tags", {}) or {}
        task = tags.get("task", "")
        if not task:
            continue
        task_weights[task] = float(top.get("weight", 1.0) or 1.0)

        for child in top.get("input_cfg", []):
            if not isinstance(child, dict) or child.get("type") != "group":
                continue
            ctags = child.get("tags", {}) or {}
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


# ──────────────────────────────────────────────────────────────────────────────
# Weight engine
# ──────────────────────────────────────────────────────────────────────────────

def compute_weights(df: pd.DataFrame,
                    metric: str = "duration",
                    temperature: float = _TEMPERATURE,
                    min_weight: float = 0.0,
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
        dur = row.get("total_duration_sec")
        n   = row.get("num_samples")
        dur_ok = dur is not None and not (isinstance(dur, float) and math.isnan(dur)) and float(dur) > 0
        n_ok   = n   is not None and not (isinstance(n,   float) and math.isnan(n))   and float(n)   > 0
        if metric == "duration":
            return float(dur) if dur_ok else (float(n) if n_ok else _EPSILON)
        return float(n) if n_ok else (float(dur) if dur_ok else _EPSILON)

    df["_size"]  = df.apply(_size, axis=1)
    df["_score"] = df["_size"].apply(lambda x: x ** (1.0 / temperature))

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
    df["expected_passes"] = (
        (df["effective_prob"] * total_n) / df["num_samples"].clip(lower=1)
    ).round(2)

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
                               ascending=[True,True,True]).iterrows():
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
    parser.add_argument("--input_weights", default=None,
                        help="Optional NeMo YAML whose task/language weights are used "
                             "to guide the output (e.g. input_cfg_train.yaml).")
    parser.add_argument("--input_weights_mode", default="hard",
                        choices=["hard", "soft", "weighting"],
                        help="How --input_weights are applied (default: hard). "
                             "hard: impose YAML weights for L1/L2, L3 stays data-driven. "
                             "soft: blend YAML and data-driven weights (product, renormalised). "
                             "weighting: multiply each dataset score by its YAML task×lang weight.")
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

    input_task_w, input_lang_w = None, None
    if args.input_weights:
        print(f"\n📂 Reading input weights: {args.input_weights}  (mode={args.input_weights_mode})")
        input_task_w, input_lang_w = parse_input_weights(args.input_weights)
        print(f"   Tasks:  {input_task_w}")
        print(f"   Langs:  {len(input_lang_w)} entries")

    print(f"\n⚖️  metric={args.metric}  T={args.temperature}  min_weight={args.min_weight}")
    df = compute_weights(df, metric=args.metric,
                         temperature=args.temperature,
                         min_weight=args.min_weight,
                         input_task_w=input_task_w,
                         input_lang_w=input_lang_w,
                         input_weights_mode=args.input_weights_mode)

    print_summary(df, args.metric, args.temperature)
    write_yaml(df, args.metric, args.temperature, Path(args.output))


if __name__ == "__main__":
    main()