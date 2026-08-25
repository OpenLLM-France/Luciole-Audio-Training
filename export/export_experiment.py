"""Batch-export checkpoints from a training experiment folder to HuggingFace format.

Walks `<exp_dir>/checkpoints/`, selects first + last + every-N-step checkpoints,
runs the same conversion as `to_hf.py` per checkpoint, then rewrites the exported
`config.json` to reference portable model names instead of local training paths.
"""
import argparse
import gc
import json
import re
import sys
import traceback
from pathlib import Path

from to_hf import export_checkpoint

LLM_MATCH = "Luciole-1B-SFT-1.1"
LLM_REPLACE = "OpenLLM-France/Luciole-1B-SFT-1.1"
ASR_MATCH = "canary-1b-v2.nemo"
ASR_REPLACE = "nvidia/canary-1b-v2"

SALM_CLASS = "nemo.collections.speechlm2.models.SALM"
SALM_AUTOMODEL_CLASS = "nemo.collections.speechlm2.models.SALMAutomodel"


def resolve_class_path(exp_config: Path, override: str | None) -> str:
    """Pick the model class for export.

    Honors an explicit --class_path override; otherwise auto-detects from the
    experiment's own exp_config.yaml so the export backend can never drift from
    the training backend: SALMAutomodel when model.use_nemo_automodel is set,
    else SALM. exp_config is already resolved (no interpolations) by salm_train.py.
    """
    if override:
        return override
    from omegaconf import OmegaConf

    model_cfg = OmegaConf.load(exp_config).get("model", {})
    use_automodel = bool(model_cfg.get("use_nemo_automodel", False))
    return SALM_AUTOMODEL_CLASS if use_automodel else SALM_CLASS


def parse_step(entry: Path) -> int | None:
    """Extract the training step from a checkpoint name.

    Handles bare digits ('002500'[.ckpt]) and Lightning's default format
    ('step=002500.ckpt', 'step=015000-last.ckpt'). Returns None if no step found.
    """
    stem = entry.name[:-5] if entry.name.endswith(".ckpt") else entry.name
    m = re.search(r"step=(\d+)", stem) or re.fullmatch(r"(\d+)", stem)
    return int(m.group(1)) if m else None


def list_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    # Dedup by step, preferring the plain checkpoint over a '-last' duplicate
    # (the final numbered checkpoint already covers that step).
    by_step: dict[int, Path] = {}
    for entry in ckpt_dir.iterdir():
        if not (entry.name.endswith(".ckpt") or (entry.is_dir() and parse_step(entry) is not None)):
            continue
        step = parse_step(entry)
        if step is None:
            continue
        if step in by_step and "-last" in entry.name:
            continue
        by_step[step] = entry
    return sorted(by_step.items())


def select(ckpts: list[tuple[int, Path]], every: int | None,
           include_steps: list[int] | None = None,
           include_first: bool = True, include_last: bool = True) -> list[tuple[int, Path]]:
    if not ckpts:
        return []
    include = set(include_steps or [])
    # The "last" checkpoint is the one Lightning marks with '-last' in its name;
    # fall back to the highest step if no such marker survived dedup.
    last_marked = [s for s, p in ckpts if "-last" in p.name]
    last_step = last_marked[-1] if last_marked else ckpts[-1][0]

    if every is None or every <= 0:
        keep_steps = {s for s, _ in ckpts} if not include else set(include)
        if not include:
            return ckpts
    else:
        keep_steps = set(include)
        keep_steps.update(step for step, _ in ckpts if step % every == 0)
    if include_first:
        keep_steps.add(ckpts[0][0])
    if include_last:
        keep_steps.add(last_step)
    return [(s, p) for s, p in ckpts if s in keep_steps]


def patch_value(value):
    if not isinstance(value, str):
        return value, False
    if LLM_MATCH in value and value != LLM_REPLACE:
        return LLM_REPLACE, True
    if ASR_MATCH in value and value != ASR_REPLACE:
        return ASR_REPLACE, True
    return value, False


def patch_config_json(config_path: Path) -> bool:
    with config_path.open() as f:
        cfg = json.load(f)

    changed = False
    for container in (cfg, cfg.get("model") if isinstance(cfg, dict) else None):
        if not isinstance(container, dict):
            continue
        for key in ("pretrained_llm", "pretrained_asr"):
            if key in container:
                new_val, did_change = patch_value(container[key])
                if did_change:
                    print(f"    {key}: {container[key]!r} -> {new_val!r}")
                    container[key] = new_val
                    changed = True

    if changed:
        with config_path.open("w") as f:
            json.dump(cfg, f, indent=2)
    return changed


def free_gpu_memory() -> None:
    """Release the GPU memory held by the previously exported checkpoint.

    Each model is moved onto the GPU in ``to_hf.load_model``; dropping the Python
    reference only returns the blocks to torch's CUDA caching allocator, which keeps
    them reserved for the process. Without an explicit ``empty_cache`` the exports pile
    up and the 2nd/3rd checkpoint OOMs even though any single one fits on the GPU. A GC
    pass first drops the (possibly cycle-referenced, or exception-traceback-pinned)
    model so its storage is actually free before we hand the blocks back.
    """
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def export_one(ckpt_path: Path, class_path: str, exp_config: Path, output_dir: Path, force: bool) -> bool:
    config_json = output_dir / "config.json"
    if output_dir.exists() and config_json.exists() and not force:
        print(f"  skip (exists): {output_dir}")
        return True

    output_dir.mkdir(parents=True, exist_ok=True)
    model = export_checkpoint(str(ckpt_path), class_path, str(exp_config), str(output_dir))
    del model

    if config_json.exists():
        if not patch_config_json(config_json):
            print(f"  config.json: no matching keys to patch")
    else:
        print(f"  warning: {config_json} not produced by save_pretrained")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp_dir", required=True, type=Path)
    ap.add_argument("--every", type=int, default=None,
                    help="Keep first+last plus steps divisible by this. Omit to export all.")
    ap.add_argument("--steps", type=int, nargs="+", default=[10000, 15000],
                    help="Explicit step(s) to always include, in addition to --every selection. "
                         "Steps with no matching checkpoint are silently ignored. "
                         "Default: 10000 15000 (pass --steps with other values to override).")
    ap.add_argument("--include-first", action=argparse.BooleanOptionalAction, default=True,
                    help="Always include the first (smallest-step) checkpoint. Default: on "
                         "(disable with --no-include-first).")
    ap.add_argument("--include-last", action=argparse.BooleanOptionalAction, default=True,
                    help="Always include the last ('-last' in name, else highest-step) checkpoint. "
                         "Default: on (disable with --no-include-last).")
    ap.add_argument("--class_path", default=None,
                    help="Model class for export. Default: auto-detect from exp_config.yaml "
                         "(SALMAutomodel if model.use_nemo_automodel, else SALM).")
    ap.add_argument("--output_dir", type=Path, default=None,
                    help="Where to write exports. Defaults to <exp_dir>/hf_checkpoints.")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Re-export even if output_dir/config.json already exists.")
    args = ap.parse_args()

    exp_config = args.exp_dir / "exp_config.yaml"
    ckpt_dir = args.exp_dir / "checkpoints"
    hf_root = args.output_dir if args.output_dir is not None else args.exp_dir / "hf_checkpoints"
    if not exp_config.is_file():
        sys.exit(f"Missing {exp_config}")
    if not ckpt_dir.is_dir():
        sys.exit(f"Missing {ckpt_dir}")

    class_path = resolve_class_path(exp_config, args.class_path)
    print(f"Model class: {class_path}"
          f"{' (from --class_path)' if args.class_path else ' (auto-detected from exp_config.yaml)'}")

    ckpts = list_checkpoints(ckpt_dir)
    selected = select(ckpts, args.every, args.steps, args.include_first, args.include_last)
    if args.steps:
        available = {step for step, _ in ckpts}
        missing = [s for s in args.steps if s not in available]
        if missing:
            print(f"Warning: requested --steps with no checkpoint: {sorted(missing)}")
    if not selected:
        sys.exit(f"No checkpoints matching NNNNNN.ckpt found in {ckpt_dir}")

    print(f"Found {len(ckpts)} checkpoints, selected {len(selected)}:")
    for step, path in selected:
        print(f"  step={step:>6}  {path.name}")
    if args.dry_run:
        return

    failures = []
    for step, ckpt_path in selected:
        # Start every export from a clean allocator: free the previous checkpoint's GPU
        # memory here, at the loop boundary, where the prior iteration's exception context
        # (which would otherwise pin a failed export's partial model) has already cleared.
        free_gpu_memory()
        output_dir = hf_root / ckpt_path.stem.replace("=", "_")
        print(f"\n[step={step}] exporting {ckpt_path.name} -> {output_dir}")
        try:
            export_one(ckpt_path, class_path, exp_config, output_dir, args.force)
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failures.append((step, str(e)))

    print("\n=== Summary ===")
    print(f"exported: {len(selected) - len(failures)} / {len(selected)}")
    if failures:
        for step, msg in failures:
            print(f"  step={step}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
