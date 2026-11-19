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
import shlex
import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml

from analyze_yaml_cfg import (
    _EPSILON,
    _lang_key,
    fmt_hours,
    fmt_num,
    print_summary,
)

warnings.filterwarnings("ignore")

_TEMPERATURE = 2.0


# ──────────────────────────────────────────────────────────────────────────────
# Input weight parsing
# ──────────────────────────────────────────────────────────────────────────────


def parse_input_weights(yaml_path: str) -> tuple:
    """
    Walk the YAML hierarchy fully and compute per-(task, lang) target
    probabilities. At each level, sibling weights are normalised to sum to 1
    and the cumulative product is taken down to each leaf manifest. Each leaf
    is then bucketed by its LEAF-resolved (task, lang) tags (with overrides
    applied).

    This correctly handles:
      - Duplicate (task, lang) sub-groups at intermediate levels — their
        contributions are summed.
      - Task or language overrides at the leaf level — the leaf's *resolved*
        task/lang determines its destination bucket.

    Returns
    -------
    task_weights : dict[str, float]
        {task_name: cumulative_probability}, summing to ~1.0 across all tasks
        whose leaves appear in the YAML.
    lang_weights : dict[tuple[str, str], float]
        {(task_name, lang_key): cumulative_probability}. Within a task, these
        sum to that task's total probability.
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

    def walk(nodes, prefix_prob, task, lang, src_lang, tgt_lang, ctx=""):
        if not nodes:
            return
        weights = [float(n.get("weight", 1.0) or 1.0) if isinstance(n, dict) else 1.0 for n in nodes]
        total = sum(weights) or 1.0
        for node, w in zip(nodes, weights):
            if not isinstance(node, dict):
                continue
            tags = node.get("tags") or {}
            if isinstance(tags, str):
                tags = {}
            _validate_tags(tags, context=ctx)
            t2 = tags.get("task", task)
            l2 = tags.get("lang", lang)
            s2 = tags.get("source_lang", src_lang)
            tg2 = tags.get("target_lang", tgt_lang)
            p = prefix_prob * (w / total)
            if node.get("type") == "multimodal_conversation":
                if t2 == "ast" and s2 and tg2:
                    lkey = f"{s2}→{tg2}"
                elif l2:
                    lkey = l2
                else:
                    lkey = "unknown"
                if t2:
                    task_weights[t2] = task_weights.get(t2, 0.0) + p
                    lang_weights[(t2, lkey)] = lang_weights.get((t2, lkey), 0.0) + p
            elif node.get("input_cfg"):
                walk(node["input_cfg"], p, t2, l2, s2, tg2, ctx=f" in task '{t2}'")

    walk(cfg.get("input_cfg", []), 1.0, "", "", "", "")

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


def compute_weights(
    df: pd.DataFrame,
    metric: str = "duration",
    temperature: float = _TEMPERATURE,
    min_weight: float = 0.0,
    max_passes: float = 0.0,
    total_samples: float = 0.0,
    input_task_w: dict = None,
    input_lang_w: dict = None,
    input_weights_mode: str = "hard",
) -> pd.DataFrame:
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
            raise ValueError(f"Dataset '{row.get('dataset_name', '?')}': invalid {col}={val!r} (metric={metric})")
        return float(val)

    df["_size"] = df.apply(_size, axis=1)
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
    tl = df.groupby(["task_type", "_lkey"])["_score"].sum().reset_index(name="group_score")
    ts = tl.groupby("task_type")["group_score"].transform("sum")
    tl["lang_w"] = tl["group_score"] / ts.replace(0, _EPSILON)

    # L1
    ta = tl.groupby("task_type")["group_score"].sum().reset_index(name="task_score")
    grand = ta["task_score"].sum()
    ta["task_w"] = ta["task_score"] / max(grand, _EPSILON)

    # ── hard / soft: override or blend L1 and L2 with YAML weights
    if input_weights_mode in ("hard", "soft") and (input_task_w or input_lang_w):
        _apply_input_weights(ta, tl, input_task_w, input_lang_w, input_weights_mode)

    tl = tl.merge(ta[["task_type", "task_w"]], on="task_type", how="left")

    df = df.merge(tl[["task_type", "_lkey", "lang_w", "task_w"]], on=["task_type", "_lkey"], how="left")

    df["effective_prob"] = df["task_w"] * df["lang_w"] * df["dataset_w"]
    df["effective_prob_pct"] = df["effective_prob"] * 100.0

    if min_weight > 0:
        df["effective_prob"] = df["effective_prob"].clip(lower=min_weight)
        df["effective_prob"] = df["effective_prob"] / df["effective_prob"].sum()
        df["effective_prob_pct"] = df["effective_prob"] * 100.0

    total_n = total_samples or float(df["num_samples"].sum(skipna=True)) or 1.0

    # ── max_passes: cap L3 weights so no dataset exceeds the threshold
    if max_passes > 0:
        df = _cap_passes(df, total_n, max_passes)

    df["expected_passes"] = ((df["effective_prob"] * total_n) / df["num_samples"].clip(lower=1)).round(2)

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


def _base_size(node, df_map, size_field):
    """For a dataset group (unweighted group of manifests where the variants carry a
    ``tags.sub_task``): the size of its base manifest, the one without ``sub_task``.
    None for any other node."""
    children = [c for c in node.get("input_cfg", []) if isinstance(c, dict)]
    if node.get("type") != "group" or node.get("weight") is not None or not children:
        return None
    if any(c.get("type") != "multimodal_conversation" for c in children):
        return None
    base = [c for c in children if "sub_task" not in (c.get("tags") or {})]
    if len(base) != 1 or len(children) == 1:
        return None
    return float(df_map.get(base[0].get("manifest_filepath", ""), {}).get(size_field, 0) or 0)


def _variant_samples(nodes, df_map):
    """Samples of the variants of the dataset groups (same audio as their base)."""
    extra = 0.0
    for n in nodes:
        if not isinstance(n, dict) or n.get("type") != "group":
            continue
        base = _base_size(n, df_map, "num_samples")
        if base is None:
            extra += _variant_samples(n.get("input_cfg", []), df_map)
        else:
            extra += sum(float(df_map.get(c.get("manifest_filepath", ""), {}).get("num_samples", 0) or 0)
                         for c in n["input_cfg"] if isinstance(c, dict)) - base
    return extra


def _compute_tree_leaf_probs(nodes, prefix_prob, df_map, size_field, temperature):
    """Walk a YAML sub-tree and assign an effective probability to each leaf.

    Within every sibling group:
      - A sibling with an explicit ``weight:`` is taken as its **fraction** of
        the sibling pool (so ``weight: 0.18`` literally means 18% of the
        parent's budget — same intuition as the top-level weights).
      - Siblings *without* an explicit weight share the remaining
        ``1 − sum(explicit)`` data-driven: leaves by ``size**(1/T)``, dataset
        groups (base + variants tagged ``sub_task``) like their base manifest,
        other plain groups by a default score of 1.0.
      - If every sibling is explicit, their values are renormalised to sum to 1.

    Returns a list of dicts: {node, effective_prob, path, samples,
    dataset_name, duration_sec}.
    """
    leaves = []
    siblings = [n for n in nodes if isinstance(n, dict)]
    if not siblings:
        return leaves

    is_explicit = [n.get("weight") is not None for n in siblings]
    explicit_w = [float(n.get("weight") or 0) if e else 0.0 for n, e in zip(siblings, is_explicit)]
    explicit_sum = sum(explicit_w)

    default_score = []
    for n, e in zip(siblings, is_explicit):
        if e:
            default_score.append(0.0)
        elif n.get("type") == "multimodal_conversation":
            d = df_map.get(n.get("manifest_filepath", ""), {})
            size = float(d.get(size_field, 0) or 0)
            default_score.append(size ** (1.0 / temperature) if size > 0 else 0.0)
        else:
            # a dataset group (base + variants) competes like its base manifest
            size = _base_size(n, df_map, size_field)
            default_score.append(size ** (1.0 / temperature) if size else 1.0)
    default_sum = sum(default_score)

    weights = [0.0] * len(siblings)
    has_defaults = default_sum > 0
    if has_defaults:
        # Explicit slots take their stated fraction; defaults split the rest.
        remaining = max(0.0, 1.0 - explicit_sum)
        for i, e in enumerate(is_explicit):
            if e:
                weights[i] = explicit_w[i]
            else:
                weights[i] = (default_score[i] / default_sum) * remaining
    else:
        # No data-driven siblings — renormalise explicit weights to sum to 1.
        if explicit_sum > 0:
            weights = [w / explicit_sum for w in explicit_w]

    total = sum(weights) or 1.0
    for n, w in zip(siblings, weights):
        child_prefix = prefix_prob * (w / total)
        if n.get("type") == "group":
            leaves.extend(
                _compute_tree_leaf_probs(n.get("input_cfg", []), child_prefix, df_map, size_field, temperature)
            )
        elif n.get("type") == "multimodal_conversation":
            path = n.get("manifest_filepath", "")
            d = df_map.get(path, {})
            leaves.append(
                {
                    "node": n,
                    "effective_prob": child_prefix,
                    "path": path,
                    "samples": float(d.get("num_samples", 0) or 0),
                    "dataset_name": d.get("dataset_name", "?"),
                    "duration_sec": float(d.get("total_duration_sec", 0) or 0),
                }
            )

    return leaves


def _apply_max_passes_tree(leaves, total_samples, max_passes, max_iter=50):
    """Cap effective_prob so no leaf exceeds *max_passes* expected passes.
    Excess budget is redistributed proportionally to remaining uncapped leaves."""
    if max_passes <= 0 or not leaves:
        return leaves
    for _ in range(max_iter):
        capped_any = False
        for leaf in leaves:
            if leaf["samples"] <= 0:
                continue
            passes = leaf["effective_prob"] * total_samples / leaf["samples"]
            if passes > max_passes:
                target = max_passes * leaf["samples"] / total_samples
                delta = leaf["effective_prob"] - target
                if delta > 1e-12:
                    leaf["effective_prob"] = target
                    uncapped = [
                        l
                        for l in leaves
                        if l is not leaf
                        and l["samples"] > 0
                        and (l["effective_prob"] * total_samples / l["samples"]) < max_passes
                    ]
                    pool = sum(l["effective_prob"] for l in uncapped)
                    if pool > 0:
                        for l in uncapped:
                            l["effective_prob"] += delta * l["effective_prob"] / pool
                        capped_any = True
        if not capped_any:
            break
    return leaves


def _format_tags(tags: dict) -> str:
    return "{" + ", ".join(f"{k}: {v}" for k, v in tags.items()) + "}"


def write_yaml_preserving_structure(
    input_yaml_path: str,
    df: pd.DataFrame,
    output_path: Path,
    metric: str,
    temperature: float,
    max_passes: float = 0.0,
    total_samples: float = 0.0,
) -> None:
    """Mirror the input YAML tree, copying every group's weight and tags
    verbatim, and filling in per-leaf weights computed as data-driven shares
    within each sibling group (size^(1/T)). Optionally caps per-leaf passes
    via max_passes (redistributing excess to uncapped siblings), passes being
    counted over *total_samples* draws (0 = one pass over the YAML's data)."""

    with open(input_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if isinstance(cfg, list):
        cfg = {"input_cfg": cfg}

    df_map = {}
    for _, row in df.iterrows():
        p = row.get("raw_manifest_path")
        if not p:
            continue
        df_map[p] = {
            "num_samples": float(row.get("num_samples", 0) or 0),
            "total_duration_sec": float(row.get("total_duration_sec", 0) or 0),
            "dataset_name": row.get("dataset_name", "?"),
        }

    size_field = "total_duration_sec" if metric == "duration" else "num_samples"
    # max_passes counts each dataset once: its variants (same audio) do not grow the total
    total_samples = total_samples or (
        sum(d["num_samples"] for d in df_map.values()) - _variant_samples(cfg.get("input_cfg", []), df_map) or 1.0
    )

    leaves = _compute_tree_leaf_probs(cfg.get("input_cfg", []), 1.0, df_map, size_field, temperature)
    if max_passes > 0:
        _apply_max_passes_tree(leaves, total_samples, max_passes)
    path_eff = {l["path"]: l["effective_prob"] for l in leaves}

    def _size_label(dur_sec, n_samples):
        return f"{fmt_hours(dur_sec)}, {fmt_num(int(n_samples))} samples"

    def _group_size(node):
        """Sum (duration_sec, num_samples) over all leaves under this node."""
        dur_total = 0.0
        n_total = 0
        if node.get("type") == "multimodal_conversation":
            d = df_map.get(node.get("manifest_filepath", ""), {})
            return float(d.get("total_duration_sec", 0) or 0), int(d.get("num_samples", 0) or 0)
        for ch in node.get("input_cfg", []):
            if isinstance(ch, dict):
                d, n = _group_size(ch)
                dur_total += d
                n_total += n
        return dur_total, n_total

    def _subtree_prob(node):
        """Sum effective_prob over all leaves under this node."""
        if not isinstance(node, dict):
            return 0.0
        if node.get("type") == "multimodal_conversation":
            return path_eff.get(node.get("manifest_filepath", ""), 0.0)
        return sum(_subtree_prob(ch) for ch in node.get("input_cfg", []))

    lines = []

    def emit(nodes, indent, depth):
        ind = " " * indent
        siblings = [n for n in nodes if isinstance(n, dict)]
        sib_probs = [_subtree_prob(n) for n in siblings]
        pool = sum(sib_probs) or 1.0
        sib_shares = [p / pool for p in sib_probs]
        # We only emit a `weight:` line when the parent has >1 children that
        # carry mass (otherwise the single child is implicitly weight=1).
        n_active_sibs = sum(1 for p in sib_probs if p > 0)
        share_by_node = dict(zip(map(id, siblings), sib_shares))

        for node in nodes:
            if not isinstance(node, dict):
                continue
            ntype = node.get("type")
            tags = node.get("tags") or {}
            if isinstance(tags, str):
                tags = {}
            tags_str = _format_tags(tags) if tags else None
            share = share_by_node[id(node)]

            if ntype == "group":
                children = node.get("input_cfg", [])
                dur, n = _group_size(node)
                comment = f"   # {_size_label(dur, n)}" if (dur or n) else ""
                lines.append(f"{ind}- type: group")
                if n_active_sibs > 1:
                    lines.append(f"{ind}  weight: {share:.4f}{comment}")
                elif comment:
                    lines.append(f"{ind}  {comment.strip()}")
                if tags_str:
                    lines.append(f"{ind}  tags: {tags_str}")
                lines.append(f"{ind}  input_cfg:")
                emit(children, indent + 4, depth + 1)

            elif ntype == "multimodal_conversation":
                path = node.get("manifest_filepath", "")
                d = df_map.get(path, {})
                name = d.get("dataset_name", "?")
                ns = int(d.get("num_samples", 0))
                dur = float(d.get("total_duration_sec", 0))
                label = _size_label(dur, ns)
                lines.append(f"{ind}- type: multimodal_conversation")
                if n_active_sibs > 1:
                    lines.append(f"{ind}  weight: {share:.4f}   # {name} ({label})")
                else:
                    lines.append(f"{ind}  # {name} ({label})")
                if tags_str:
                    lines.append(f"{ind}  tags: {tags_str}")
                lines.append(f"{ind}  manifest_filepath: {path}")

        if depth == 0:
            lines.append("")  # blank line between top-level groups

    emit(cfg.get("input_cfg", []), 0, 0)

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✅ YAML (tree-preserving) → {output_path}")


def write_yaml_fill_missing_weights(
    input_yaml_path: str,
    df: pd.DataFrame,
    output_path: Path,
    metric: str,
    temperature: float,
    max_passes: float = 0.0,
    total_samples: float = 0.0,
) -> None:
    """Copy the input YAML text as is (weights, comments, layout) and only insert
    a ``weight:`` on each node that has siblings but no weight.

    Same rule as ``_compute_tree_leaf_probs``: the unweighted siblings share
    ``1 − sum(explicit weights)`` by ``size**(1/T)``, a dataset group (base +
    variants tagged ``sub_task``) counting as its base manifest.

    With *max_passes*, a node may not be drawn more than *max_passes* times over
    *total_samples* draws (0 = one pass over the YAML's data); for a dataset
    group, passes = its effective probability × total / base rows, since every
    variant reuses the base audio. The excess only goes to the uncapped new
    weights of the same level: existing weights, hence task / language shares,
    never change."""
    import re

    text = Path(input_yaml_path).read_text(encoding="utf-8").split("\n")
    cfg = yaml.safe_load("\n".join(text))
    nodes = cfg if isinstance(cfg, list) else cfg.get("input_cfg", [])

    df_map = {
        row["raw_manifest_path"]: {
            "num_samples": float(row.get("num_samples", 0) or 0),
            "total_duration_sec": float(row.get("total_duration_sec", 0) or 0),
        }
        for _, row in df.iterrows()
        if row.get("raw_manifest_path")
    }
    size_field = "total_duration_sec" if metric == "duration" else "num_samples"
    total_samples = total_samples or (
        sum(d["num_samples"] for d in df_map.values()) - _variant_samples(nodes, df_map) or 1.0
    )

    # manifest path -> (line index, indent of "manifest_filepath:")
    line_of = {}
    for i, l in enumerate(text):
        m = re.match(r"^(\s*)manifest_filepath:\s*(\S+)", l)
        if m:
            if m.group(2) in line_of:
                raise SystemExit(f"❌ {m.group(2)} appears twice in {input_yaml_path}")
            line_of[m.group(2)] = (i, len(m.group(1)))

    def _type_line(path, indent):
        i = line_of[path][0]
        while not re.match(r"^ {%d}- type: " % indent, text[i]):
            i -= 1
        return i

    def _stats(path):
        d = df_map.get(path)
        if not d or not d["num_samples"]:
            raise SystemExit(f"❌ {path} missing from the CSV (needed to weight its dataset)")
        return d

    inserts = {}  # index of the node's "- type:" line -> (indent, weight, comment)
    report = []

    def walk(nodes, prefix):
        sib = [n for n in nodes if isinstance(n, dict)]
        explicit = sum(float(n["weight"]) for n in sib if n.get("weight") is not None)
        new = {}  # id(node) -> (path whose size counts, indent of the node's "- type:")
        for n in sib:
            if len(sib) < 2 or n.get("weight") is not None:
                continue
            if n.get("type") == "multimodal_conversation":
                path = n["manifest_filepath"]
                new[id(n)] = (path, line_of[path][1] - 2)
            else:
                base = [c for c in n.get("input_cfg", []) if isinstance(c, dict) and "sub_task" not in (c.get("tags") or {})]
                if len(base) != 1 or base[0].get("type") != "multimodal_conversation":
                    raise SystemExit(f"❌ unweighted group with no single base manifest: {n}")
                path = base[0]["manifest_filepath"]
                new[id(n)] = (path, line_of[path][1] - 6)

        w = {}
        if new:
            score = {k: float(_stats(p)[size_field]) ** (1.0 / temperature) for k, (p, _) in new.items()}
            remaining, free = max(0.0, 1.0 - explicit), set(new)
            while free:  # water-filling: cap, give the rest to the uncapped ones
                tot = sum(score[k] for k in free)
                for k in free:
                    w[k] = score[k] / tot * remaining
                over = {
                    k for k in free
                    if max_passes > 0 and prefix * w[k] * total_samples / _stats(new[k][0])["num_samples"] > max_passes
                }
                if not over:
                    break
                for k in over:
                    w[k] = max_passes * _stats(new[k][0])["num_samples"] / (prefix * total_samples)
                    remaining -= w[k]
                free -= over
            if not free and remaining > 1e-9:
                print(f"⚠️  every new weight capped next to {new[next(iter(new))][0]}: "
                      f"{remaining:.4f} of the level left to the existing weights")
            for k, (path, indent) in new.items():
                rows = _stats(path)["num_samples"]
                passes = prefix * w[k] * total_samples / rows
                report.append((passes, path))
                comment = f"{int(rows):,} rows in base manifest"
                if max_passes > 0:
                    comment += f", {passes:.1f} passes"
                inserts[_type_line(path, indent)] = (indent, w[k], comment)

        allw = [w.get(id(n), float(n.get("weight", 1.0) or 1.0)) for n in sib]
        s = sum(allw) or 1.0
        for n, x in zip(sib, allw):
            if n.get("input_cfg"):
                walk(n["input_cfg"], prefix * x / s)

    walk(nodes, 1.0)

    out = []
    for i, l in enumerate(text):
        out.append(l)
        if i in inserts:
            indent, wt, comment = inserts[i]
            out.append(" " * (indent + 2) + f"weight: {wt:.4f}   # {comment}")
    output_path.write_text("\n".join(out), encoding="utf-8")

    capped = sum(1 for p, _ in report if max_passes > 0 and p >= max_passes - 1e-6)
    print(f"\n{len(inserts)} weights inserted ({capped} capped at {max_passes:g} passes, "
          f"total={total_samples:,.0f} samples)")
    for passes, path in sorted(report, reverse=True)[:10]:
        print(f"   {passes:6.1f} passes  {path}")
    print(f"✅ YAML (missing weights filled) → {output_path}")


def write_yaml(df: pd.DataFrame, metric: str, temperature: float, output_path: Path) -> None:
    """Write a NeMo-compatible input_cfg YAML from the weighted DataFrame."""

    tl = (
        df.groupby(["task_type", "_lkey"])
        .agg(
            total_dur_h=("total_duration_sec", lambda x: x.sum(skipna=True) / 3600),
            total_samples=("num_samples", "sum"),
            lang_w=("lang_w", "first"),
            task_w=("task_w", "first"),
        )
        .reset_index()
        .sort_values(["task_w", "lang_w"], ascending=[False, False])
    )

    task_agg = (
        df.groupby("task_type").agg(task_w=("task_w", "first")).reset_index().sort_values("task_w", ascending=False)
    )

    def size_label(dur_h, n_samples):
        return f"{fmt_hours(dur_h * 3600)}, {fmt_num(n_samples)} samples"

    lines = []

    for _, tr in task_agg.iterrows():
        task = tr["task_type"]
        tw = tr["task_w"]
        t_sub = tl[tl["task_type"] == task]
        t_h = t_sub["total_dur_h"].sum()
        t_n = t_sub["total_samples"].sum()

        lines += [
            f"- type: group",
            f"  weight: {tw:.4f}   # L1 — {task.upper()} ({size_label(t_h, t_n)})",
            f"  tags: {{task: {task}}}",
            f"  input_cfg:",
        ]

        for _, lr in t_sub.iterrows():
            lang = lr["_lkey"]
            lw = lr["lang_w"]
            l_h = lr["total_dur_h"]
            l_n = lr["total_samples"]

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

            ds_sub = df[(df["task_type"] == task) & (df["_lkey"] == lang)].sort_values("dataset_w", ascending=False)

            for _, dr in ds_sub.iterrows():
                dw = dr["dataset_w"]
                d_h = (dr.get("total_duration_sec", 0) or 0) / 3600
                d_n = dr.get("num_samples", 0) or 0
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
    parser = argparse.ArgumentParser(description="metadata CSV → suggested NeMo sampling YAML.")
    parser.add_argument("csv_path", help="metadata.csv produced by generate_csv.py")
    parser.add_argument("--output", default=None, help="Output YAML path. If omitted, only print the summary.")
    parser.add_argument(
        "--metric",
        default="duration",
        choices=["duration", "samples"],
        help="Size metric for weighting (default: duration).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=_TEMPERATURE,
        help=f"Smoothing temperature T>0 (default {_TEMPERATURE}). T=1→proportional, T=2→moderate, T→∞→uniform.",
    )
    parser.add_argument(
        "--min_weight", type=float, default=0.0001, help="Minimum effective probability floor (default: 0.0001)."
    )
    parser.add_argument(
        "--max_passes",
        type=float,
        default=0.0,
        help="Maximum expected passes per dataset (default: 0 = unlimited). "
        "Caps L3 weights and redistributes excess within each group.",
    )
    parser.add_argument(
        "--total_samples",
        type=float,
        default=0.0,
        help="Number of samples drawn during training, used to count --max_passes "
        "(default: 0 = one pass over the data, i.e. the sum of num_samples).",
    )
    parser.add_argument(
        "--fill_missing_weights",
        action="store_true",
        help="With --yaml: keep the input YAML as is (weights, comments) and only add a "
        "weight to each node without one; --max_passes then caps only those new weights.",
    )
    parser.add_argument(
        "--skip_missing", action="store_true", help="Drop rows where file_exists=False before computing weights."
    )
    parser.add_argument(
        "--yaml",
        "--input_weights",
        default=None,
        dest="yaml",
        help="Optional NeMo YAML: used to extract task/language weights "
        "AND to filter the CSV to only datasets present in the YAML.",
    )
    parser.add_argument(
        "--input_weights_mode",
        default="hard",
        choices=["hard", "soft", "weighting"],
        help="How --yaml weights are applied (default: hard). "
        "hard: impose YAML weights for L1/L2, L3 stays data-driven. "
        "soft: blend YAML and data-driven weights (product, renormalised). "
        "weighting: multiply each dataset score by its YAML task×lang weight.",
    )
    parser.add_argument(
        "--ignore_missing_manifest",
        action="store_true",
        help="Only warn (instead of error) when YAML manifests are missing from CSV.",
    )
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
    df = compute_weights(
        df,
        metric=args.metric,
        temperature=args.temperature,
        min_weight=args.min_weight,
        max_passes=args.max_passes,
        total_samples=args.total_samples,
        input_task_w=input_task_w,
        input_lang_w=input_lang_w,
        input_weights_mode=args.input_weights_mode,
    )

    if not args.fill_missing_weights:
        # flat task × lang × dataset model: does not reflect the tree kept by --fill_missing_weights
        print_summary(df, title=f"metric={args.metric}  T={args.temperature}")
    if args.output:
        if args.yaml:
            write = write_yaml_fill_missing_weights if args.fill_missing_weights else write_yaml_preserving_structure
            write(
                args.yaml,
                df,
                Path(args.output),
                metric=args.metric,
                temperature=args.temperature,
                max_passes=args.max_passes,
                total_samples=args.total_samples,
            )
        else:
            write_yaml(df, args.metric, args.temperature, Path(args.output))
        prepend_command_header(Path(args.output), args)


def prepend_command_header(output_path: Path, args: argparse.Namespace) -> None:
    """Write at the top of the output how it was made: the command line as typed and every
    argument with its value, defaults included, so the file can be regenerated as is."""
    width = max(len(k) for k in vars(args))
    lines = [
        "# ============================================================",
        "# Generated by assets/suggest_yaml_cfg.py: do not edit the weights by hand,",
        "# edit the --yaml file and rerun the command below.",
        f"#   python {shlex.join(['suggest_yaml_cfg.py'] + sys.argv[1:])}",
        "# Arguments (defaults included):",
        *(f"#   {k:<{width}} = {v!r}" for k, v in vars(args).items()),
        "# ============================================================",
        "",
    ]
    output_path.write_text("\n".join(lines) + "\n" + output_path.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    main()
