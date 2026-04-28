"""Batch-export checkpoints from a training experiment folder to HuggingFace format.

Walks `<exp_dir>/checkpoints/`, selects first + last + every-N-step checkpoints,
runs the same conversion as `to_hf.py` per checkpoint, then rewrites the exported
`config.json` to reference portable model names instead of local training paths.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

from to_hf import load_model

LLM_MATCH = "Luciole-1B-SFT-1.1"
LLM_REPLACE = "Luciole-1B-SFT-1.1"
ASR_MATCH = "canary-1b-v2.nemo"
ASR_REPLACE = "nvidia/canary-1b-v2"


def parse_step(entry: Path) -> int | None:
    stem = entry.name[:-5] if entry.name.endswith(".ckpt") else entry.name
    try:
        return int(stem)
    except ValueError:
        return None


def list_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for entry in ckpt_dir.iterdir():
        if entry.name.endswith(".ckpt") or (entry.is_dir() and entry.name.isdigit()):
            step = parse_step(entry)
            if step is not None:
                out.append((step, entry))
    out.sort(key=lambda x: x[0])
    return out


def select(ckpts: list[tuple[int, Path]], every: int | None) -> list[tuple[int, Path]]:
    if not ckpts:
        return []
    if every is None or every <= 0:
        return ckpts
    keep_steps = {ckpts[0][0], ckpts[-1][0]}
    keep_steps.update(step for step, _ in ckpts if step % every == 0)
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


def export_one(ckpt_path: Path, class_path: str, exp_config: Path, output_dir: Path, force: bool) -> bool:
    config_json = output_dir / "config.json"
    if output_dir.exists() and config_json.exists() and not force:
        print(f"  skip (exists): {output_dir}")
        return True

    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(str(ckpt_path), class_path, str(exp_config))
    model.save_pretrained(str(output_dir))
    if hasattr(model, "tokenizer") and model.tokenizer is not None:
        model.tokenizer.save_pretrained(str(output_dir))
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
    ap.add_argument("--class_path", default="nemo.collections.speechlm2.models.SALM")
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

    ckpts = list_checkpoints(ckpt_dir)
    selected = select(ckpts, args.every)
    if not selected:
        sys.exit(f"No checkpoints matching NNNNNN.ckpt found in {ckpt_dir}")

    print(f"Found {len(ckpts)} checkpoints, selected {len(selected)}:")
    for step, path in selected:
        print(f"  step={step:>6}  {path.name}")
    if args.dry_run:
        return

    failures = []
    for step, ckpt_path in selected:
        output_dir = hf_root / ckpt_path.stem
        print(f"\n[step={step}] exporting {ckpt_path.name} -> {output_dir}")
        try:
            export_one(ckpt_path, args.class_path, exp_config, output_dir, args.force)
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
