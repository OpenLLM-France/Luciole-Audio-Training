"""What a duplex run's training dataloader has consumed, per dataset: ../assets/dataloader_state_report.py
adapted to the duplex checkpoints, without changing it.

Differences with the SALM runs that script was written for:
  - checkpoints are single DDP files `step-NNNNNN.ckpt` (CHECKPOINT_EQUALS_CHAR "-"), not `step=N.ckpt/meta.pt`
    directories: loaded with mmap (4.5 GB each, only the loop state is read);
  - manifests live under audio/duplex/nemo: that prefix is dropped from the dataset labels;
  - no metadata.csv: the hours are estimated from the mean audio duration (sum of the user audio turns) of
    --sample random lines per dataset, read through the .idx offsets.

  python dataloader_report.py $ALL_CCFRSCRATCH/audio/training/duplex/duplex_2heads_25k_4gpu \\
      --output_dir $SCRATCH/speechlm/dataloader_reports/duplex_2heads_25k_4gpu --all_checkpoints
"""

import json
import os
import random
import re
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets"))
import dataloader_state_report as report  # noqa: E402

_SAMPLE = 500


def _list_checkpoints(exp_dir: Path) -> list:
    """[(step, file)] of step-NNNNNN.ckpt / step=NNNNNN.ckpt files; for equal steps the non '-last' copy wins."""
    ckpts = {}
    for p in sorted((exp_dir / "checkpoints").glob("*.ckpt")):
        m = re.search(r"step[=-](\d+)", p.name)
        if not m or not p.is_file():
            continue
        step = int(m.group(1))
        if step not in ckpts or ckpts[step].name.endswith("-last.ckpt"):
            ckpts[step] = p
    if not ckpts:
        raise SystemExit(f"No step-NNNNNN.ckpt under {exp_dir / 'checkpoints'}.")
    return sorted(ckpts.items())


def _load(path, **kwargs):
    """report calls torch.load(<ckpt>/meta.pt): load the checkpoint file itself, memory-mapped."""
    path = Path(path)
    if path.name == "meta.pt" and path.parent.is_file():
        path = path.parent
    kwargs.pop("map_location", None)
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)


_relative_parts_orig = report._relative_parts


def _relative_parts(manifest, ctx):
    parts = _relative_parts_orig(manifest, ctx)
    return parts[2:] if parts[:2] == ["duplex", "nemo"] else parts


def _mean_audio_duration(leaf: dict, ctx, n: int, rng: random.Random):
    """Mean summed duration of the audio turns over n random lines (offsets from the .idx sidecars)."""
    durs = []
    shards = report._expand_shards(report._resolve_env(leaf["manifest"], ctx))
    offsets = []
    for f in shards:
        f = os.path.abspath(f)
        idx = (ctx.indexes_root.rstrip("/") + f if ctx.indexes_root else f) + ".idx"
        try:
            offsets.append((f, np.fromfile(idx, dtype=np.uint64)))
        except OSError:
            return None
    total = sum(len(o) - 1 for _, o in offsets)
    for k in sorted(rng.sample(range(total), min(n, total))):
        for f, o in offsets:
            if k < len(o) - 1:
                with open(f, "rb") as fh:
                    fh.seek(int(o[k]))
                    d = json.loads(fh.read(int(o[k + 1] - o[k])))
                durs.append(sum(t.get("duration", 0.0) for t in d["conversations"] if t.get("type") == "audio"))
                break
            k -= len(o) - 1
    return sum(durs) / len(durs) if durs else None


def main():
    report._list_checkpoints = _list_checkpoints
    report.torch = types.SimpleNamespace(load=_load)
    report._relative_parts = _relative_parts

    # Hours: report reads metadata.csv into {canonical manifest: mean duration}; feed it sampled means instead.
    n = int(os.environ.get("DUPLEX_REPORT_SAMPLE", _SAMPLE))
    collect = report._collect_leaves

    def _collect_leaves(cfg, ctx):
        leaves = collect(cfg, ctx)
        rng = random.Random(0)
        means = {}
        for leaf in leaves:
            means[report._canonical_manifest(leaf["manifest"], ctx)] = _mean_audio_duration(leaf, ctx, n, rng)
            print(f"  mean audio duration {leaf['dataset']}: {means[report._canonical_manifest(leaf['manifest'], ctx)]}")
        report._read_avg_durations = lambda *_: means
        return leaves

    report._collect_leaves = _collect_leaves
    if "--metadata_csv" not in sys.argv:
        sys.argv += ["--metadata_csv", os.devnull]  # any existing path: report then calls _read_avg_durations
    report.main()


if __name__ == "__main__":
    main()
