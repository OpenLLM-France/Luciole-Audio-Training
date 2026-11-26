"""
dataloader_state_report.py
==========================
Summarise what a resumable (indexed + StatefulDataLoader) Lhotse training has
actually consumed, per dataset / task / sub-task / language, and compare it to
the weights requested in the training ``input_cfg``.

How it works
------------
Each checkpoint ``<exp>/checkpoints/step=XXXX.ckpt/meta.pt`` stores, for every
(data-parallel rank x dataloader worker), the full multiplexer state: for each
leaf of the ``input_cfg`` a ``position`` inside its shard (``shard_id`` /
``num_shards``) and an ``epoch`` counter (number of completed passes over that
shard). Samples drawn from a leaf are therefore

    sum over (rank, worker) of  epoch * shard_len + position

where ``shard_len`` is derived from the manifest length read from its ``.idx``
sidecar under ``indexes_root``. Counts are cuts *drawn* from each source, i.e.
before the global duration/token filter and including the bucketing buffer
(compare ``total drawn`` with the sampler's ``kept_cuts`` in the summary).

Weights are compared as per-sample probabilities (product of the normalised
weights along the YAML tree), which is what the Lhotse multiplexer samples.

Distinct samples: within one epoch a shard never repeats a line, so the number of
distinct lines seen from a shard is ``shard_len`` once ``epoch >= 1``, else
``position`` (max over the workers reading that same shard).

Outputs (in --output_dir)
-------------------------
weights_check/   did the dataloader follow the weights of the input_cfg?
  summary.txt                  — global numbers + largest deviations
  datasets.csv                 — one row per input_cfg leaf (target vs actual share)
  tasks.csv / subtasks.csv / languages.csv / task_languages.csv
  evolution.csv                — per checkpoint x dataset shares (with --all_checkpoints)
  plots/*.png                  — unless --no_plots
data_usage/      how much of each corpus was seen, and how many times?
  summary.txt                  — lines / hours available, seen, distinct, repeated; passes buckets
  datasets.csv                 — one row per leaf: size, seen, distinct, coverage, passes, hours
  tasks.csv / subtasks.csv / languages.csv / task_languages.csv
  workers.csv                  — per (rank, worker) sampler diagnostics (drawn, kept, discarded)
  evolution.csv                — per checkpoint x dataset seen / distinct / passes (with --all_checkpoints)
  plots/*.png                  — unless --no_plots

Usage
-----
# Latest checkpoint of an experiment (input_cfg / indexes_root read from exp_config.yaml)
python dataloader_state_report.py \
    $ALL_CCFRSCRATCH/audio/training/speechlm2_experiments/Luciole-8B/luciole8b_v4_singlephase \
    --output_dir $SCRATCH/speechlm/dataloader_reports/luciole8b_v4_singlephase

# A given checkpoint, plus the evolution over every checkpoint
python dataloader_state_report.py <exp_dir> --checkpoint 50000 --all_checkpoints

Only needs torch, pandas, pyyaml and matplotlib. Loading a ``meta.pt`` is light
(tens of MB), so it can run on a login node.
"""

import argparse
import math
import os
import re
from pathlib import Path

import pandas as pd
import torch
import yaml

_TOP_DEVIATIONS = 15
_TOP_PASSES = 40


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exp_dir", help="Experiment directory (containing checkpoints/ and exp_config.yaml).")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint dir name, path, or step number. Default: the checkpoint with the highest step.",
    )
    parser.add_argument(
        "--all_checkpoints", action="store_true", help="Also compute the evolution over every checkpoint."
    )
    parser.add_argument("--input_cfg", default=None, help="Override data.train_ds.input_cfg from exp_config.yaml.")
    parser.add_argument(
        "--indexes_root", default=None, help="Override data.train_ds.indexes_root from exp_config.yaml."
    )
    parser.add_argument(
        "--data_folder",
        default=None,
        help="Value for ${oc.env:DATA_FOLDER}. Default: $DATA_FOLDER, else $ALL_CCFRSCRATCH/audio.",
    )
    parser.add_argument(
        "--metadata_csv",
        default=None,
        help="metadata.csv (from generate_csv_metadata.py) used to estimate hours. "
        "Default: metadata.csv next to the input_cfg, if present.",
    )
    parser.add_argument("--output_dir", default=None, help="Default: ./dataloader_reports/<exp_name>")
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir).resolve()
    exp_cfg = _read_exp_config(exp_dir)
    train_ds = (exp_cfg.get("data") or {}).get("train_ds") or {}

    input_cfg = args.input_cfg or train_ds.get("input_cfg")
    if not isinstance(input_cfg, str):
        raise SystemExit("Could not find a train input_cfg path in exp_config.yaml; pass --input_cfg.")
    indexes_root = args.indexes_root or train_ds.get("indexes_root")
    data_folder = args.data_folder or os.environ.get("DATA_FOLDER")
    if data_folder is None and os.environ.get("ALL_CCFRSCRATCH"):
        data_folder = os.path.join(os.environ["ALL_CCFRSCRATCH"], "audio")
    metadata_csv = args.metadata_csv or (Path(input_cfg).parent / "metadata.csv")
    output_dir = Path(args.output_dir or Path("dataloader_reports") / exp_dir.name)
    weights_dir, usage_dir = output_dir / "weights_check", output_dir / "data_usage"
    for d in (weights_dir, usage_dir):
        d.mkdir(parents=True, exist_ok=True)

    ctx = _Context(data_folder=data_folder, indexes_root=indexes_root)
    cfg = yaml.safe_load(open(input_cfg, encoding="utf-8"))
    leaves = _collect_leaves(cfg, ctx)
    avg_dur = _read_avg_durations(metadata_csv, ctx) if Path(metadata_csv).exists() else {}
    for leaf in leaves:
        leaf["avg_dur_sec"] = avg_dur.get(_canonical_manifest(leaf["manifest"], ctx))
        leaf["size"] = _manifest_size(leaf, ctx)

    checkpoints = _list_checkpoints(exp_dir)
    selected = _select_checkpoint(checkpoints, args.checkpoint)
    print(f"Experiment : {exp_dir}")
    print(f"Checkpoint : {selected[1]}")
    print(f"input_cfg  : {input_cfg}  ({len(leaves)} leaves)")

    state = _consumption_from_meta(_meta_path(selected[1]), cfg, leaves, ctx)
    datasets = _datasets_table(leaves, state)
    groups = {
        "tasks": ["task"],
        "subtasks": ["task", "sub_task"],
        "languages": ["lang"],
        "task_languages": ["task", "lang"],
    }
    tables = {name: _group_table(datasets, keys) for name, keys in groups.items()}
    usage_tables = {name: _usage_group_table(datasets, keys) for name, keys in groups.items()}

    datasets[_WEIGHT_COLS].to_csv(weights_dir / "datasets.csv", index=False)
    datasets[_USAGE_COLS].sort_values("passes", ascending=False).to_csv(usage_dir / "datasets.csv", index=False)
    for name in groups:
        tables[name].to_csv(weights_dir / f"{name}.csv", index=False)
        usage_tables[name].to_csv(usage_dir / f"{name}.csv", index=False)
    pd.DataFrame(state["workers"]).to_csv(usage_dir / "workers.csv", index=False)

    summary = _summary_text(exp_dir, selected[1], input_cfg, state, datasets, tables)
    (weights_dir / "summary.txt").write_text(summary)
    usage_summary = _usage_summary_text(exp_dir, selected[1], input_cfg, state, datasets, usage_tables)
    (usage_dir / "summary.txt").write_text(usage_summary)
    print(summary)
    print(usage_summary)

    evolution = None
    if args.all_checkpoints:
        rows = []
        for step, ckpt in checkpoints:
            print(f"  evolution: {ckpt.name}")
            st = _consumption_from_meta(_meta_path(ckpt), cfg, leaves, ctx)
            total = sum(st["seen"])
            for leaf, seen, distinct in zip(leaves, st["seen"], st["distinct"], strict=True):
                size = leaf.get("size")
                rows.append(
                    dict(
                        step=st["global_step"],
                        checkpoint=ckpt.name,
                        dataset=leaf["dataset"],
                        task=leaf["task"],
                        sub_task=leaf["sub_task"],
                        lang=leaf["lang"],
                        target_pct=100 * leaf["prob"],
                        seen=seen,
                        actual_pct=100 * seen / total if total else float("nan"),
                        size=size,
                        distinct=distinct,
                        coverage_pct=100 * distinct / size if size else float("nan"),
                        passes=seen / size if size else float("nan"),
                    )
                )
        evolution = pd.DataFrame(rows)
        keys = ["step", "checkpoint", "dataset", "task", "sub_task", "lang"]
        evolution[keys + ["target_pct", "seen", "actual_pct"]].to_csv(weights_dir / "evolution.csv", index=False)
        evolution[keys + ["size", "seen", "distinct", "coverage_pct", "passes"]].to_csv(
            usage_dir / "evolution.csv", index=False
        )

    if not args.no_plots:
        title = f"{exp_dir.name} @ step {state['global_step']}"
        _make_plots(datasets, tables, evolution, weights_dir / "plots", title=title)
        _make_usage_plots(datasets, usage_tables, evolution, usage_dir / "plots", title=title)
    print(f"\nWrote report to {weights_dir} and {usage_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Config / checkpoint discovery
# ──────────────────────────────────────────────────────────────────────────────


class _Context:
    def __init__(self, data_folder, indexes_root):
        self.data_folder = data_folder
        self.indexes_root = indexes_root
        self.size_cache = {}


def _read_exp_config(exp_dir: Path) -> dict:
    path = exp_dir / "exp_config.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(open(path, encoding="utf-8")) or {}


def _list_checkpoints(exp_dir: Path) -> list:
    """Return [(step, path)] sorted by step; for equal steps the non '-last' copy wins."""
    ckpts = {}
    for p in sorted((exp_dir / "checkpoints").glob("*.ckpt")):
        if not _meta_path(p).exists():
            continue
        m = re.search(r"step=(\d+)", p.name)
        if not m:
            continue
        step = int(m.group(1))
        if step not in ckpts or ckpts[step].name.endswith("-last.ckpt"):
            ckpts[step] = p
    if not ckpts:
        raise SystemExit(f"No checkpoint (single file or dir with meta.pt) found under {exp_dir / 'checkpoints'}.")
    return sorted(ckpts.items())


def _meta_path(ckpt: Path) -> Path:
    """Distributed checkpoints are dirs holding meta.pt; DDP ones (e.g. the 1B) are one .ckpt file."""
    return ckpt if ckpt.is_file() else ckpt / "meta.pt"


def _select_checkpoint(checkpoints: list, which):
    if which is None:
        return checkpoints[-1]
    if Path(which).exists():
        p = Path(which).resolve()
        p = p.parent if p.name == "meta.pt" else p
        return (-1, p)
    for step, p in checkpoints:
        if which == p.name or (which.isdigit() and int(which) == step):
            return (step, p)
    raise SystemExit(f"Checkpoint {which!r} not found. Available: {[p.name for _, p in checkpoints]}")


# ──────────────────────────────────────────────────────────────────────────────
# input_cfg tree
# ──────────────────────────────────────────────────────────────────────────────


def _collect_leaves(cfg, ctx: _Context) -> list:
    """Flatten the input_cfg tree; each leaf carries its per-sample target probability and inherited tags."""
    leaves = []

    def walk(entries, prob, tags):
        weights = [float(e.get("weight", 1.0) or 1.0) for e in entries]
        total = sum(weights)
        for entry, w in zip(entries, weights):
            p = prob * w / total
            t = {**tags, **(entry.get("tags") or {})}
            if "input_cfg" in entry:
                walk(entry["input_cfg"], p, t)
            else:
                manifest = entry.get("manifest_filepath")
                if isinstance(manifest, list):
                    manifest = manifest[0]
                leaves.append(dict(manifest=str(manifest), prob=p, tags=t))

    walk(cfg if isinstance(cfg, list) else cfg["input_cfg"], 1.0, {})

    for leaf in leaves:
        path_parts = _relative_parts(leaf["manifest"], ctx)
        t = leaf.pop("tags")
        leaf["task"] = t.get("task") or (path_parts[0] if path_parts else "unknown")
        leaf["sub_task"] = t.get("sub_task", "")
        leaf["source_lang"] = t.get("source_lang", "")
        leaf["target_lang"] = t.get("target_lang", "")
        leaf["lang"] = t.get("lang") or t.get("target_lang") or _lang_from_path(path_parts)
        leaf["dataset"] = _dataset_label(path_parts)
    counts = pd.Series([l["dataset"] for l in leaves]).value_counts()
    for leaf in leaves:
        if counts[leaf["dataset"]] > 1:
            leaf["dataset"] += ":" + Path(leaf["manifest"]).stem
    return leaves


def _relative_parts(manifest: str, ctx: _Context) -> list:
    path = _resolve_env(manifest, ctx)
    if ctx.data_folder and path.startswith(ctx.data_folder.rstrip("/") + "/"):
        path = path[len(ctx.data_folder.rstrip("/")) + 1 :]
    parts = [p for p in path.split("/") if p]
    if parts and parts[0] == "nemo":
        parts = parts[1:]
    return parts


def _dataset_label(parts: list) -> str:
    """asr/fr/context/FLEURS/train_randomorder.jsonl -> asr/fr/FLEURS (non-train file stems are kept)."""
    if not parts:
        return "unknown"
    *dirs, fname = parts
    dirs = [d for d in dirs if d != "context" and not d.endswith("_shards")]
    stem = re.sub(r"(_randomorder)?(__OP_.*_CL_)?\.jsonl$", "", fname)
    if not stem.startswith("train"):
        dirs.append(stem)
    return "/".join(dirs)


def _lang_from_path(parts: list) -> str:
    """First path component that looks like a language (fr, en, multilang, ...) or a pair (fr-en -> en)."""
    for part in parts[1:4]:
        if part in ("multilang", "mixed"):
            return "mixed"
        m = re.fullmatch(r"([a-z]{2})(?:-([a-z]{2}))?", part)
        if m:
            return m.group(2) or m.group(1)
    return "unknown"


def _resolve_env(path: str, ctx: _Context) -> str:
    if "${oc.env:DATA_FOLDER}" in path:
        if not ctx.data_folder:
            raise SystemExit("input_cfg uses ${oc.env:DATA_FOLDER}: set $DATA_FOLDER or pass --data_folder.")
        path = path.replace("${oc.env:DATA_FOLDER}", ctx.data_folder)
    return path


def _expand_shards(path: str) -> list:
    """NeMo brace-expansion markers: foo__OP_0..51_CL_.jsonl -> foo_0.jsonl ... foo_51.jsonl."""
    m = re.search(r"_OP_(\d+)\.\.(\d+)_CL_", path)
    if not m:
        return [path]
    return [path[: m.start()] + str(i) + path[m.end() :] for i in range(int(m.group(1)), int(m.group(2)) + 1)]


def _manifest_size(leaf: dict, ctx: _Context):
    """Number of records in a (possibly sharded) manifest, read from its .idx sidecars (uint64 offsets + end)."""
    key = leaf["manifest"]
    if key not in ctx.size_cache:
        total = 0
        try:
            for f in _expand_shards(_resolve_env(key, ctx)):
                f = os.path.abspath(f)
                idx = (ctx.indexes_root.rstrip("/") + f if ctx.indexes_root else f) + ".idx"
                total += os.path.getsize(idx) // 8 - 1
        except OSError:
            total = None
        ctx.size_cache[key] = total
    return ctx.size_cache[key]


def _canonical_manifest(path: str, ctx: _Context) -> str:
    """Map a randomorder / sharded manifest path back to the original manifest listed in metadata.csv."""
    path = _resolve_env(path, ctx) if ctx.data_folder else path
    # .../train_contextual_biasing_shards/train_contextual_biasing__OP_0..54_CL_.jsonl
    #   -> .../train_contextual_biasing.jsonl
    path = re.sub(r"(_randomorder)?_shards/[^/]*$", ".jsonl", path)
    return re.sub(r"_randomorder\.jsonl$", ".jsonl", path)


def _read_avg_durations(metadata_csv, ctx: _Context) -> dict:
    meta = pd.read_csv(metadata_csv)
    out = {}
    for path, dur in zip(meta["raw_manifest_path"], meta["avg_segment_duration_sec"]):
        if isinstance(path, str) and not pd.isna(dur):
            out[_canonical_manifest(path, ctx)] = float(dur)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Dataloader state
# ──────────────────────────────────────────────────────────────────────────────


def _consumption_from_meta(meta_path: Path, cfg, leaves: list, ctx: _Context) -> dict:
    # mmap: a single-file DDP .ckpt also holds the weights (GBs), only the loop state is read.
    meta = torch.load(meta_path, map_location="cpu", weights_only=False, mmap=True)
    try:
        per_rank = meta["loops"]["fit_loop"]["state_dict"]["combined_loader"][0]["train_dataloader_per_rank"]
    except (KeyError, IndexError, TypeError):
        raise SystemExit(f"{meta_path} has no per-rank StatefulDataLoader state (was use_stateful_dataloader on?).")

    entries = cfg if isinstance(cfg, list) else cfg["input_cfg"]
    seen = [0] * len(leaves)
    max_epoch = [0] * len(leaves)
    # distinct lines per (leaf, shard): workers reading the same shard share the same line order.
    shard_distinct = [dict() for _ in leaves]
    workers = []
    for rank_entry in per_rank:
        snapshots = rank_entry["state"]["_snapshot"]["_worker_snapshots"]
        for worker_name, snap in sorted(snapshots.items()):
            sampler_state = snap["fetcher_state"]["dataset_iter_state"]["sampler_state"]
            leaf_states = []
            _walk_state(entries, sampler_state["cuts_state"], leaf_states)
            if len(leaf_states) != len(leaves):
                raise SystemExit(
                    f"State has {len(leaf_states)} leaves but input_cfg has {len(leaves)}: "
                    "is this the input_cfg the run was trained with?"
                )
            drawn = 0
            for i, ls in enumerate(leaf_states):
                epoch, position = int(ls.get("epoch", 0) or 0), int(ls["position"])
                shard_len = _shard_len(leaves[i], ls, ctx) if epoch else None
                n = epoch * shard_len + position if epoch else position
                seen[i] += n
                drawn += n
                max_epoch[i] = max(max_epoch[i], epoch)
                shard = (int(ls.get("shard_id") or 0), int(ls.get("num_shards") or 1))
                d = shard_len if epoch else position
                shard_distinct[i][shard] = max(shard_distinct[i].get(shard, 0), d)
            diag = _sum_diagnostics(sampler_state.get("diagnostics", {}))
            workers.append(dict(dp_rank=rank_entry["dp_rank"], worker=worker_name, drawn=drawn, **diag))
    distinct = [sum(d.values()) for d in shard_distinct]
    return dict(global_step=meta.get("global_step"), seen=seen, max_epoch=max_epoch, distinct=distinct, workers=workers)


def _walk_state(entries: list, state, out: list) -> None:
    """Walk the multiplexer state tree in lockstep with the input_cfg tree, collecting leaf iterator states."""
    if len(entries) == 1:  # single-child groups are not wrapped in a multiplexer
        entry = entries[0]
        if "input_cfg" in entry:
            _walk_state(entry["input_cfg"], state, out)
        else:
            out.append(_leaf_state(state))
        return
    mux = _descend(state, "inner_states")
    if not isinstance(mux, dict) or "inner_states" not in mux:
        raise SystemExit("Could not find a multiplexer state matching the input_cfg tree.")
    inner = mux["inner_states"]
    if len(inner) != len(entries):
        raise SystemExit(f"Multiplexer has {len(inner)} sources but input_cfg group has {len(entries)} entries.")
    for entry, sub in zip(entries, inner):
        if "input_cfg" in entry:
            _walk_state(entry["input_cfg"], sub, out)
        else:
            out.append(_leaf_state(sub))


def _descend(state, key):
    """Follow single-child wrappers (filters, maps, repeats) until a dict holding `key` or a leaf position."""
    while True:
        if isinstance(state, list) and len(state) == 1:
            state = state[0]
        elif isinstance(state, dict) and key not in state and "position" not in state:
            if "_state" in state:
                state = state["_state"]
            elif "source" in state:
                state = state["source"]
            else:
                return state
        else:
            return state


def _leaf_state(state) -> dict:
    leaf = _descend(state, "position")
    if not isinstance(leaf, dict) or "position" not in leaf:
        raise SystemExit("Leaf iterator state has no 'position' (non-indexed source?).")
    return leaf


def _shard_len(leaf: dict, ls: dict, ctx: _Context) -> int:
    n = _manifest_size(leaf, ctx)
    if n is None:
        raise SystemExit(f"Missing .idx for {leaf['manifest']} (needed because it wrapped around); check --indexes_root.")
    sid, ns = int(ls.get("shard_id") or 0), int(ls.get("num_shards") or 1)
    return (n - sid + ns - 1) // ns if n > sid else 0


def _sum_diagnostics(diag: dict) -> dict:
    keys = ("kept_cuts", "discarded_cuts", "kept_batches", "discarded_batches")
    out = dict.fromkeys(keys, 0)
    for stats in (diag.get("stats_per_epoch") or {}).values():
        for k in keys:
            out[k] += int(stats.get(k, 0) or 0)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Tables
# ──────────────────────────────────────────────────────────────────────────────


_WEIGHT_COLS = [
    "dataset", "task", "sub_task", "lang", "source_lang", "target_lang",
    "target_pct", "actual_pct", "ratio", "seen", "size", "passes", "max_epoch", "hours_est", "manifest",
]  # fmt: skip
_USAGE_COLS = [
    "dataset", "task", "sub_task", "lang", "source_lang", "target_lang",
    "size", "seen", "distinct", "repeated", "coverage_pct", "passes", "max_epoch",
    "avg_dur_sec", "hours_available", "hours_seen", "hours_distinct", "manifest",
]  # fmt: skip
_PASS_BUCKETS = [0, 0.1, 0.5, 1, 2, 5, 10, float("inf")]


def _datasets_table(leaves: list, state: dict) -> pd.DataFrame:
    seen, max_epoch, distinct = state["seen"], state["max_epoch"], state["distinct"]
    total = sum(seen)
    rows = []
    for leaf, s, ep, dist in zip(leaves, seen, max_epoch, distinct, strict=True):
        size = leaf.get("size")
        actual = s / total if total else float("nan")
        dur = leaf.get("avg_dur_sec")
        rows.append(
            dict(
                dataset=leaf["dataset"],
                task=leaf["task"],
                sub_task=leaf["sub_task"],
                lang=leaf["lang"],
                source_lang=leaf["source_lang"],
                target_lang=leaf["target_lang"],
                target_pct=100 * leaf["prob"],
                actual_pct=100 * actual,
                ratio=actual / leaf["prob"] if leaf["prob"] else float("nan"),
                seen=s,
                size=size,
                passes=s / size if size else float("nan"),
                max_epoch=ep,
                hours_est=s * dur / 3600 if dur else float("nan"),
                distinct=dist,
                repeated=s - dist,
                coverage_pct=100 * dist / size if size else float("nan"),
                avg_dur_sec=dur if dur else float("nan"),
                hours_available=size * dur / 3600 if dur and size else float("nan"),
                hours_seen=s * dur / 3600 if dur else float("nan"),
                hours_distinct=dist * dur / 3600 if dur else float("nan"),
                manifest=leaf["manifest"],
            )
        )
    return pd.DataFrame(rows)


def _group_table(datasets: pd.DataFrame, keys: list) -> pd.DataFrame:
    g = (
        datasets.groupby(keys, dropna=False)
        .agg(
            n_datasets=("dataset", "count"),
            target_pct=("target_pct", "sum"),
            actual_pct=("actual_pct", "sum"),
            seen=("seen", "sum"),
            hours_est=("hours_est", lambda x: x.sum(min_count=1)),
        )
        .reset_index()
    )
    g["ratio"] = g["actual_pct"] / g["target_pct"]
    return g.sort_values("target_pct", ascending=False)


def _summary_text(exp_dir, ckpt, input_cfg, state, datasets: pd.DataFrame, tables: dict) -> str:
    w = pd.DataFrame(state["workers"])
    total = int(datasets["seen"].sum())
    kept = int(w["kept_cuts"].sum())
    tv = 0.5 * (datasets["actual_pct"] - datasets["target_pct"]).abs().sum()
    # Workers share the multiplexer seed by design, so sampling noise is that of one stream of ~total/n_workers draws.
    n_streams = len(w)
    expected = datasets["target_pct"] / 100 * total
    z = (datasets["seen"] - expected) / (expected * (1 - datasets["target_pct"] / 100) * n_streams).pow(0.5)

    lines = [
        f"experiment        : {exp_dir}",
        f"checkpoint        : {ckpt.name}",
        f"input_cfg         : {input_cfg}",
        f"global_step       : {state['global_step']}",
        f"ranks x workers   : {w['dp_rank'].nunique()} x {len(w) // max(w['dp_rank'].nunique(), 1)}",
        f"cuts drawn        : {total:,}",
        f"sampler kept_cuts : {kept:,}  (drawn - kept = {total - kept:,}: filtered + bucketing buffer)",
        f"sampler batches   : {int(w['kept_batches'].sum()):,}",
        f"est. hours seen   : {datasets['hours_est'].sum():,.0f}",
        f"TV distance       : {tv:.3f} %  (0.5 * sum |actual - target|)",
        f"ratio actual/target: min {datasets['ratio'].min():.3f}  median {datasets['ratio'].median():.3f}  "
        f"max {datasets['ratio'].max():.3f}",
        f"|z| > 3 (single-stream noise model): {int((z.abs() > 3).sum())} / {len(datasets)}",
        f"datasets with passes > 1 / > 5 / > 10: {int((datasets['passes'] > 1).sum())} / "
        f"{int((datasets['passes'] > 5).sum())} / {int((datasets['passes'] > 10).sum())}",
        "",
        "By task:",
        _fmt_group(tables["tasks"], ["task"]),
        "",
        "By language:",
        _fmt_group(tables["languages"], ["lang"]),
        "",
        f"Largest relative deviations (top {_TOP_DEVIATIONS}):",
        _fmt_datasets(datasets.reindex((datasets["ratio"] - 1).abs().sort_values(ascending=False).index)),
        "",
        f"Largest absolute deviations (top {_TOP_DEVIATIONS}):",
        _fmt_datasets(
            datasets.reindex((datasets["actual_pct"] - datasets["target_pct"]).abs().sort_values(ascending=False).index)
        ),
        "",
        f"Most repeated datasets (top {_TOP_DEVIATIONS}):",
        _fmt_datasets(datasets.sort_values("passes", ascending=False)),
    ]
    return "\n".join(lines) + "\n"


def _usage_group_table(datasets: pd.DataFrame, keys: list) -> pd.DataFrame:
    def hsum(x):
        return x.sum(min_count=1)

    g = (
        datasets.assign(
            fully_seen=datasets["coverage_pct"] >= 100,
            over_5=datasets["passes"] > 5,
        )
        .groupby(keys, dropna=False)
        .agg(
            n_datasets=("dataset", "count"),
            size=("size", "sum"),
            seen=("seen", "sum"),
            distinct=("distinct", "sum"),
            repeated=("repeated", "sum"),
            n_fully_seen=("fully_seen", "sum"),
            n_over_5_passes=("over_5", "sum"),
            min_passes=("passes", "min"),
            max_passes=("passes", "max"),
            hours_available=("hours_available", hsum),
            hours_seen=("hours_seen", hsum),
            hours_distinct=("hours_distinct", hsum),
        )
        .reset_index()
    )
    g["passes"] = g["seen"] / g["size"]
    g["coverage_pct"] = 100 * g["distinct"] / g["size"]
    g["repeated_pct_of_seen"] = 100 * g["repeated"] / g["seen"]
    g["share_of_seen_pct"] = 100 * g["seen"] / datasets["seen"].sum()
    cols = keys + [
        "n_datasets", "size", "seen", "share_of_seen_pct", "distinct", "coverage_pct", "repeated",
        "repeated_pct_of_seen", "passes", "min_passes", "max_passes", "n_fully_seen", "n_over_5_passes",
        "hours_available", "hours_seen", "hours_distinct",
    ]  # fmt: skip
    return g[cols].sort_values("seen", ascending=False)


def _usage_summary_text(exp_dir, ckpt, input_cfg, state, datasets: pd.DataFrame, tables: dict) -> str:
    w = pd.DataFrame(state["workers"])
    size, seen, distinct = int(datasets["size"].sum()), int(datasets["seen"].sum()), int(datasets["distinct"].sum())
    kept = int(w["kept_cuts"].sum())
    h_av, h_seen, h_dist = (datasets[c].sum() for c in ("hours_available", "hours_seen", "hours_distinct"))

    buckets = pd.cut(datasets["passes"], _PASS_BUCKETS, right=False)
    by_bucket = (
        datasets.assign(bucket=buckets)
        .groupby("bucket", observed=False)
        .agg(n_datasets=("dataset", "count"), seen=("seen", "sum"), hours_seen=("hours_seen", "sum"))
        .reset_index()
    )
    by_bucket["share_of_seen_pct"] = 100 * by_bucket["seen"] / seen
    by_bucket["bucket"] = by_bucket["bucket"].astype(str).str.replace("inf", "∞")

    lines = [
        f"experiment        : {exp_dir}",
        f"checkpoint        : {ckpt.name}",
        f"input_cfg         : {input_cfg}",
        f"global_step       : {state['global_step']}",
        "",
        f"lines available   : {size:,}  ({h_av:,.0f} h est.)  in {len(datasets)} manifests",
        f"cuts drawn        : {seen:,}  ({h_seen:,.0f} h est.)  = {seen / size:.2f} passes over the whole mix",
        f"sampler kept_cuts : {kept:,}  (drawn - kept = {seen - kept:,}: filtered + bucketing buffer)",
        f"distinct lines    : {distinct:,}  ({h_dist:,.0f} h est.)  = {100 * distinct / size:.1f} % of the lines",
        f"never seen        : {size - distinct:,}  ({h_av - h_dist:,.0f} h est.)",
        f"repeated draws    : {seen - distinct:,}  = {100 * (seen - distinct) / seen:.1f} % of what was drawn",
        f"fully seen (>= 1 pass on every shard): {int((datasets['coverage_pct'] >= 100).sum())} / {len(datasets)}",
        "",
        "Datasets by number of passes:",
        by_bucket.to_string(index=False, float_format=lambda v: f"{v:.1f}"),
        "",
        "By task:",
        _fmt_usage(tables["tasks"], ["task"]),
        "",
        "By sub-task:",
        _fmt_usage(tables["subtasks"], ["task", "sub_task"]),
        "",
        "By language:",
        _fmt_usage(tables["languages"], ["lang"]),
        "",
        f"Most repeated datasets (top {_TOP_DEVIATIONS}):",
        _fmt_usage_datasets(datasets.sort_values("passes", ascending=False)),
        "",
        f"Least covered datasets (top {_TOP_DEVIATIONS}):",
        _fmt_usage_datasets(datasets.sort_values("coverage_pct")),
        "",
        f"Most unseen hours (top {_TOP_DEVIATIONS}):",
        _fmt_usage_datasets(
            datasets.reindex(
                (datasets["hours_available"] - datasets["hours_distinct"]).sort_values(ascending=False).index
            )
        ),
    ]
    return "\n".join(lines) + "\n"


def _fmt_usage(df: pd.DataFrame, keys: list) -> str:
    cols = keys + [
        "n_datasets", "share_of_seen_pct", "passes", "min_passes", "max_passes", "coverage_pct",
        "repeated_pct_of_seen", "hours_available", "hours_seen", "hours_distinct",
    ]  # fmt: skip
    return df[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}")


def _fmt_usage_datasets(df: pd.DataFrame) -> str:
    cols = ["dataset", "size", "seen", "distinct", "coverage_pct", "passes", "hours_available", "hours_distinct"]
    return df[cols].head(_TOP_DEVIATIONS).to_string(index=False, float_format=lambda v: f"{v:.2f}")


def _fmt_group(df: pd.DataFrame, keys: list) -> str:
    cols = keys + ["n_datasets", "target_pct", "actual_pct", "ratio", "seen", "hours_est"]
    return df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}")


def _fmt_datasets(df: pd.DataFrame) -> str:
    cols = ["dataset", "target_pct", "actual_pct", "ratio", "seen", "size", "passes"]
    return df[cols].head(_TOP_DEVIATIONS).to_string(index=False, float_format=lambda v: f"{v:.3f}")


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────


def _make_plots(datasets: pd.DataFrame, tables: dict, evolution, out_dir: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = list(tables["tasks"]["task"])
    cmap = plt.get_cmap("tab20")
    task_color = {t: cmap(i % 20) for i, t in enumerate(tasks)}
    colors = datasets["task"].map(task_color)

    # 1. target vs actual share per dataset (log-log)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(datasets["target_pct"], datasets["actual_pct"], c=list(colors), s=18, alpha=0.85)
    lo = min(datasets["target_pct"].min(), datasets["actual_pct"][datasets["actual_pct"] > 0].min()) / 1.5
    hi = max(datasets["target_pct"].max(), datasets["actual_pct"].max()) * 1.5
    ax.plot([lo, hi], [lo, hi], color="grey", lw=1, ls="--")
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi), xlabel="target share (%)", ylabel="actual share (%)")
    _task_legend(ax, task_color)
    ax.set_title(f"Per-dataset share — {title}")
    _save(fig, out_dir / "target_vs_actual.png")

    # 2. ratio vs target share (noise funnel)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(datasets["target_pct"], datasets["ratio"], c=list(colors), s=18, alpha=0.85)
    ax.axhline(1.0, color="grey", lw=1, ls="--")
    for _, r in datasets.reindex((datasets["ratio"] - 1).abs().sort_values(ascending=False).index).head(8).iterrows():
        ax.annotate(r["dataset"], (r["target_pct"], r["ratio"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set(xscale="log", xlabel="target share (%)", ylabel="actual / target")
    _task_legend(ax, task_color)
    ax.set_title(f"Deviation from requested weights — {title}")
    _save(fig, out_dir / "ratio_vs_target.png")

    # 3. grouped target / actual bars for tasks, languages and sub-tasks
    for name, key in (("tasks", "task"), ("languages", "lang"), ("subtasks", "sub_task")):
        df = tables[name]
        labels = df[key].astype(str) if name != "subtasks" else df["task"] + "/" + df["sub_task"].replace("", "-")
        _bar_target_actual(plt, labels, df, out_dir / f"{name}.png", f"Share by {key} — {title}")

    if evolution is not None and evolution["step"].nunique() > 1:
        _plot_evolution(plt, evolution, task_color, out_dir)


def _make_usage_plots(datasets: pd.DataFrame, tables: dict, evolution, out_dir: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = list(tables["tasks"].sort_values("task")["task"])
    cmap = plt.get_cmap("tab20")
    task_color = {t: cmap(i % 20) for i, t in enumerate(tasks)}

    # 1. most repeated datasets
    top = datasets.sort_values("passes", ascending=False).head(_TOP_PASSES).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.25 * len(top) + 1.5))
    ax.barh(top["dataset"], top["passes"], color=list(top["task"].map(task_color)))
    ax.set(xlabel="passes over the dataset (seen / size)")
    ax.tick_params(axis="y", labelsize=7)
    _task_legend(ax, task_color, loc="lower right")
    ax.set_title(f"Most repeated datasets — {title}")
    _save(fig, out_dir / "passes_top.png")

    # 5. passes distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    passes = datasets["passes"].dropna()
    passes = passes[passes > 0]
    bins = [10**x for x in _linspace(math.log10(passes.min()), math.log10(passes.max()), 30)]
    ax.hist(passes, bins=bins, color="#4C72B0")
    ax.axvline(1.0, color="grey", lw=1, ls="--")
    ax.set(xscale="log", xlabel="passes over the dataset (log scale)", ylabel="# datasets")
    _plain_log_ticks(ax.xaxis)
    ax.set_title(f"Passes per dataset — {title}")
    _save(fig, out_dir / "passes_hist.png")

    # 3. passes vs dataset size: big corpora barely seen, small ones repeated
    fig, ax = plt.subplots(figsize=(9, 6))
    _scatter_passes(ax, datasets["size"], datasets["passes"], datasets["task"].map(task_color))
    _task_legend(ax, task_color)
    ax.set_title(f"Passes vs dataset size — {title}")
    _save(fig, out_dir / "passes_vs_size.png")

    # 4. hours available / distinct seen / seen, per task, sub-task and language
    for name, key in (("tasks", "task"), ("subtasks", "sub_task"), ("languages", "lang")):
        df = tables[name].sort_values("hours_available", ascending=False)
        labels = df[key].astype(str) if name != "subtasks" else df["task"] + "/" + df["sub_task"].replace("", "-")
        _bar_hours(plt, labels, df, out_dir / f"hours_{name}.png", f"Hours by {key} — {title}")

    # 5. passes over training, per task
    if evolution is not None and evolution["step"].nunique() > 1:
        by_task = evolution.groupby(["step", "task"])[["seen", "distinct", "size"]].sum().reset_index()
        for col, ylabel, fname in (
            ("passes", "passes (seen / size)", "evolution_task_passes.png"),
            ("coverage_pct", "distinct lines seen (% of size)", "evolution_task_coverage.png"),
        ):
            fig, ax = plt.subplots(figsize=(10, 6))
            for task, df in by_task.groupby("task"):
                y = df["seen"] / df["size"] if col == "passes" else 100 * df["distinct"] / df["size"]
                ax.plot(df["step"], y, marker="o", ms=3, label=task, color=task_color.get(task))
            if col == "passes":
                ax.axhline(1.0, color="grey", lw=1, ls="--")
            if col == "passes":
                ax.set_yscale("log")
                _plain_log_ticks(ax.yaxis)
            ax.set(
                xlabel="step", ylabel=ylabel + (" (log scale)" if col == "passes" else ""),
                ylim=(None if col == "passes" else 0, None),
            )  # fmt: skip
            ax.legend(fontsize=7, ncol=2)
            ax.set_title(f"Per-task {ylabel} over training")
            _save(fig, out_dir / fname)

    _make_detailed_usage_plots(plt, datasets, evolution, out_dir, title)


def _make_detailed_usage_plots(plt, datasets: pd.DataFrame, evolution, out_dir: Path, title: str) -> None:
    """by_task/<task>/, by_task/<task>/<lang>/ and by_language/<lang>/: hours, passes and coverage in detail."""
    datasets = datasets.assign(corpus=datasets["dataset"].map(_corpus_label))
    if evolution is not None:
        evolution = evolution.assign(corpus=evolution["dataset"].map(_corpus_label))
        if evolution["step"].nunique() < 2:
            evolution = None

    def evo(mask_fn):
        return None if evolution is None else evolution[mask_fn(evolution)]

    for task, df in datasets.groupby("task"):
        d = out_dir / "by_task" / _safe_name(task)
        name = f"{task} — {title}"
        _detail_folder(plt, df, evo(lambda e: e["task"] == task), d, name, ("lang", "sub_task", "corpus"), "lang")
        if df["lang"].nunique() < 2:
            continue
        for lang, dfl in df.groupby("lang"):
            e = evo(lambda e: (e["task"] == task) & (e["lang"] == lang))
            _detail_folder(
                plt, dfl, e, d / _safe_name(lang), f"{task} {lang} — {title}", ("sub_task", "corpus"), "corpus",
                with_datasets=True,
            )  # fmt: skip

    for lang, df in datasets.groupby("lang"):
        d = out_dir / "by_language" / _safe_name(lang)
        e = evo(lambda e: e["lang"] == lang)
        _detail_folder(plt, df, e, d, f"{lang} — {title}", ("task", "sub_task", "corpus"), "task")


def _detail_folder(plt, df, evolution, d: Path, name: str, group_keys, evo_key, with_datasets=False) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for key in group_keys:
        if key == "corpus":
            base, all_variants = _corpus_tables(df)
            views = ((base, "base", "base manifests only"), (all_variants, "all_variants", _ALL_VARIANTS))
            for g, suffix, what in views:
                g = g.sort_values("hours_available", ascending=False)
                _bar_hours(plt, g["corpus"], g, d / f"hours_corpora_{suffix}.png", f"Hours by corpus, {what} — {name}")
            continue
        g = _usage_group_table(df, [key] if key != "sub_task" else ["task", "sub_task"])
        if len(g) < 2:
            continue
        labels = _group_labels(g, key)
        g = g.assign(_label=list(labels)).sort_values("hours_available", ascending=False)
        _bar_hours(plt, g["_label"], g, d / f"hours_{_PLURAL[key]}.png", f"Hours by {key} — {name}")
    if with_datasets:
        g = df.sort_values("hours_available", ascending=False)
        _bar_hours(plt, g["dataset"].map(_short_label), g, d / "hours_datasets.png", f"Hours by dataset — {name}")

    if not ((df["size"] > 0) & (df["passes"] > 0)).any():
        return
    color_key = "sub_task" if with_datasets else group_keys[0]
    values = sorted(df[color_key].astype(str).unique())
    cmap = plt.get_cmap("tab20")
    color = {v: cmap(i % 20) for i, v in enumerate(values)}
    fig, ax = plt.subplots(figsize=(9, 6))
    _scatter_passes(ax, df["size"], df["passes"], df[color_key].astype(str).map(color))
    _task_legend(ax, {(v or "-"): c for v, c in color.items()})
    ax.set_title(f"Passes vs dataset size (colour = {color_key}) — {name}", fontsize=9)
    _save(fig, d / "passes_vs_size.png")

    if evolution is None:
        return
    if evo_key != "corpus":
        by = evolution.groupby(["step", evo_key])[["seen", "distinct", "size"]].sum().reset_index()
        _plot_evolution_groups(plt, by, evo_key, d, name, _PLURAL[evo_key], coverage=True)
        return
    base_rows = _base_rows(evolution)
    base = evolution[base_rows].groupby(["step", "corpus"])[["seen", "distinct", "size"]].sum()
    _plot_evolution_groups(plt, base.reset_index(), "corpus", d, f"base manifests only — {name}", "corpora_base")
    seen = evolution.groupby(["step", "corpus"])["seen"].sum()
    all_variants = base[["size"]].join(seen.rename("seen"), how="inner").reset_index()
    _plot_evolution_groups(
        plt, all_variants, "corpus", d, f"all variants — {name}", "corpora_all_variants", coverage=False,
        passes_label="passes per base line (all variants)",
    )  # fmt: skip


def _plot_evolution_groups(
    plt, by, evo_key, d: Path, name: str, suffix: str, coverage=True, passes_label="passes (seen / size)"
) -> None:
    groups = sorted(by[evo_key].astype(str).unique())
    cmap = plt.get_cmap("tab20")
    styles = ["-", "--", ":", "-."]
    plots = [("passes", passes_label, f"evolution_passes_{suffix}.png")]
    if coverage:
        plots.append(("coverage_pct", "distinct lines seen (% of size)", f"evolution_coverage_{suffix}.png"))
    for col, ylabel, fname in plots:
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for i, (key, g) in enumerate(by.groupby(evo_key)):
            y = g["seen"] / g["size"] if col == "passes" else 100 * g["distinct"] / g["size"]
            ax.plot(
                g["step"], y, marker="o", ms=2, label=str(key) or "-", color=cmap(i % 20), ls=styles[(i // 20) % 4]
            )
        if col == "passes":
            ax.axhline(1.0, color="grey", lw=1, ls="--")
            ax.set_yscale("log")
            _plain_log_ticks(ax.yaxis)
        ax.set(
                xlabel="step", ylabel=ylabel + (" (log scale)" if col == "passes" else ""),
                ylim=(None if col == "passes" else 0, None),
            )  # fmt: skip
        ax.legend(
            fontsize=6, ncol=1 + len(groups) // 25, loc="center left", bbox_to_anchor=(1.0, 0.5), handlelength=4
        )
        ax.set_title(f"{ylabel} over training, per {evo_key} — {name}", fontsize=9)
        _save(fig, d / fname)


_ALL_VARIANTS = "all variants, passes per base line"


def _base_rows(df: pd.DataFrame) -> pd.Series:
    """Base manifests (no sub_task); every row of a corpus that has no base manifest."""
    base = df["sub_task"].fillna("") == ""
    has_base = base.groupby([df["task"], df["lang"], df["corpus"]]).transform("any")
    return base | ~has_base


def _corpus_tables(df: pd.DataFrame):
    """Per corpus: base manifests only, and all variants with draws counted against the base size."""
    base = _usage_group_table(df[_base_rows(df)], ["corpus"])
    all_variants = _usage_group_table(df, ["corpus"]).set_index("corpus")
    b = base.set_index("corpus")
    all_variants["size"] = b["size"]
    all_variants["hours_available"] = b["hours_available"]
    all_variants["passes"] = all_variants["seen"] / all_variants["size"]
    all_variants["coverage_pct"] = float("nan")
    all_variants["hours_distinct"] = float("nan")
    return base, all_variants.reset_index()


_PLURAL = {"task": "tasks", "lang": "languages", "sub_task": "subtasks", "corpus": "corpora"}


def _group_labels(g: pd.DataFrame, key: str):
    if key == "sub_task":
        return g["task"] + "/" + g["sub_task"].replace("", "-")
    return g[key].astype(str).replace("", "-")


def _corpus_label(dataset: str) -> str:
    """asr/fr/clean/CFPB/events:train_x -> CFPB; ast/es-fr/clean/CoVoST:train -> es-fr/CoVoST."""
    parts = dataset.split("/")
    pair = parts[1] if len(parts) > 1 and "-" in parts[1] else None
    rest = [p for p in parts[2:] if p != "clean"]
    if not rest:
        return dataset
    corpus = rest[0].split(":")[0]
    return f"{pair}/{corpus}" if pair else corpus


def _short_label(dataset: str) -> str:
    """Drops the task/lang prefix and clean/: asr/fr/clean/CFPB/events:train -> CFPB/events:train."""
    parts = dataset.split("/")
    pair = parts[1] if len(parts) > 1 and "-" in parts[1] else None
    rest = [p for p in parts[2:] if p != "clean"]
    return "/".join(([pair] if pair else []) + rest) if rest else dataset


def _safe_name(value) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)) or "none"


def _scatter_passes(ax, size, passes, colors) -> None:
    ax.scatter(size, passes, c=list(colors), s=18, alpha=0.85)
    ax.axhline(1.0, color="grey", lw=1, ls="--")
    ax.set(xscale="log", yscale="log", xlabel="lines in the manifest (log scale)", ylabel="passes (log scale)")
    _plain_log_ticks(ax.xaxis)
    _plain_log_ticks(ax.yaxis)


_LOG_TICKS = [m * 10**e for e in range(-3, 8) for m in (1, 2, 5)]


def _plain_log_ticks(axis) -> None:
    """Log axis labelled 0.1, 0.2, 0.5, 1, 2, 5, 10, 20k, 1M instead of 10^-1, 10^0, 10^1."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    def fmt(v, _):
        for div, suffix in ((1e6, "M"), (1e3, "k")):
            if v >= div:
                return f"{v / div:g}{suffix}"
        return f"{v:g}"

    axis.set_major_locator(FixedLocator(_LOG_TICKS))
    axis.set_major_formatter(FuncFormatter(fmt))
    axis.set_minor_formatter(NullFormatter())


def _bar_hours(plt, labels, df: pd.DataFrame, path: Path, title: str) -> None:
    if not (df["hours_available"] > 0).any():  # text-only data has no duration
        return
    n = len(df)
    fig, ax = plt.subplots(figsize=(11, 0.4 * n + 1.5))
    y = list(range(n))[::-1]
    h = 0.27
    ax.barh([v + h for v in y], df["hours_available"], height=h, label="available", color="#BBBBBB")
    if df["hours_distinct"].notna().any():
        ax.barh(y, df["hours_distinct"], height=h, label="distinct seen", color="#55A868")
    ax.barh([v - h for v in y], df["hours_seen"], height=h, label="seen (with repeats)", color="#4C72B0")
    for v, s, p, c in zip(y, df["hours_seen"], df["passes"], df["coverage_pct"], strict=True):
        text = f" {p:.2f} passes per base line" if pd.isna(c) else f" {p:.2f} passes, {c:.0f}% seen"
        ax.text(s, v - h, text, va="center", fontsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels(list(labels), fontsize=7)
    ax.set(xscale="log", xlabel="hours (est., log scale)")
    _plain_log_ticks(ax.xaxis)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title(title)
    _save(fig, path)


def _plot_evolution(plt, evolution: pd.DataFrame, task_color: dict, out_dir: Path) -> None:
    evo = evolution.assign(absdev=(evolution["actual_pct"] - evolution["target_pct"]).abs())
    tv = evo.groupby("step")["absdev"].sum() / 2
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tv.index, tv.values, marker="o")
    ax.set(xlabel="step", ylabel="TV distance to target mix (%)", ylim=(0, None))
    ax.set_title("Cumulative mix deviation over training")
    _save(fig, out_dir / "evolution_tv_distance.png")

    by_task = evolution.groupby(["step", "task"])[["actual_pct", "target_pct"]].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    for task, df in by_task.groupby("task"):
        ratio = df["actual_pct"] / df["target_pct"]
        ax.plot(df["step"], ratio, marker="o", ms=3, label=task, color=task_color.get(task))
    ax.axhline(1.0, color="grey", lw=1, ls="--")
    ax.set(xlabel="step", ylabel="cumulative actual / target share")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Per-task deviation over training")
    _save(fig, out_dir / "evolution_task_ratio.png")


def _bar_target_actual(plt, labels, df: pd.DataFrame, path: Path, title: str) -> None:
    n = len(df)
    fig, ax = plt.subplots(figsize=(9, 0.32 * n + 1.5))
    y = list(range(n))[::-1]
    ax.barh([v + 0.2 for v in y], df["target_pct"], height=0.4, label="target", color="#BBBBBB")
    ax.barh([v - 0.2 for v in y], df["actual_pct"], height=0.4, label="actual", color="#4C72B0")
    for v, a, r in zip(y, df["actual_pct"], df["ratio"]):
        ax.text(a, v - 0.2, f" {r:.3f}", va="center", fontsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels(list(labels), fontsize=7)
    ax.set(xlabel="share of drawn samples (%)")
    ax.legend(loc="lower right")
    ax.set_title(title)
    _save(fig, path)


def _task_legend(ax, task_color: dict, loc="best") -> None:
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], marker="o", ls="", color=c, label=t) for t, c in task_color.items()]
    ax.legend(handles=handles, fontsize=7, loc=loc, ncol=2)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    import matplotlib.pyplot as plt

    plt.close(fig)


def _linspace(a: float, b: float, n: int) -> list:
    return [a + (b - a) * i / (n - 1) for i in range(n)] if n > 1 and b > a else [a, a + 1e-9]


if __name__ == "__main__":
    main()
