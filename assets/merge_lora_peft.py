import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE_LAYER_RE = re.compile(r"^(.*)\.base_layer\.weight$")
AUTOMODEL_LORA_A_RE = re.compile(r"^(.*)\.lora_A\.weight$")


def merge_peft(keys, get_tensor, lora_cfg):
    """HF-PEFT style (used by the 1B): tensor names '...base_layer.weight' /
    '...lora_A.default.weight' / '...lora_B.default.weight'. PEFT's get_peft_model also wraps
    *every* LLM tensor (not just the LoRA-targeted ones) with a 'base_model.model.' prefix."""
    scale = lora_cfg["lora_alpha"] / lora_cfg["r"]
    print(f"[PEFT] LoRA scale (alpha/r) = {scale}")

    merged = {}
    skip = set()
    n_merged = 0
    for key in list(keys):
        m = BASE_LAYER_RE.match(key)
        if not m:
            continue
        prefix = m.group(1)
        lora_a_key = f"{prefix}.lora_A.default.weight"
        lora_b_key = f"{prefix}.lora_B.default.weight"
        if lora_a_key not in keys or lora_b_key not in keys:
            raise ValueError(f"Found {key} but missing matching {lora_a_key} / {lora_b_key}")

        base = get_tensor(key)
        lora_a = get_tensor(lora_a_key).to(torch.float32)
        lora_b = get_tensor(lora_b_key).to(torch.float32)
        delta = (lora_b @ lora_a) * scale
        new_weight = (base.to(torch.float32) + delta).to(base.dtype)

        new_key = f"{prefix}.weight"
        merged[new_key] = new_weight
        skip |= {key, lora_a_key, lora_b_key}
        n_merged += 1
        print(f"  merged {new_key}  (base_layer {key}, delta scale {scale})")

    for key in keys - skip:
        merged[key] = get_tensor(key)

    print(f"Merged {n_merged} LoRA adapter pairs into base weights ({len(merged)} tensors total)")

    # PEFT's get_peft_model wraps the *whole* module, so every LLM tensor (not just the
    # LoRA-targeted ones) is prefixed "base_model.model." in the state dict. Strip it so the
    # keys match a plain (non-PEFT) model, matching the now-lora-less config.json.
    unwrapped = {}
    for key, tensor in merged.items():
        new_key = key.replace(".base_model.model.", ".", 1) if ".base_model.model." in key else key
        unwrapped[new_key] = tensor
    n_unwrapped = sum(1 for k in merged if ".base_model.model." in k)
    print(f"Stripped PEFT 'base_model.model.' wrapper from {n_unwrapped} tensors")
    return unwrapped


def merge_automodel(keys, get_tensor, lora_cfg):
    """NeMo-Automodel-native style (used by the 8B/23B): tensor names '<name>.weight' /
    '<name>.lora_A.weight' / '<name>.lora_B.weight'. The base weight keeps its original name
    (no wrapper prefix, no adapter name), so merging is a plain in-place update:
    merged = weight + (alpha/dim) * (lora_B.weight @ lora_A.weight)
    (formula and scale taken from nemo_automodel.components._peft.lora.LoRALinear)."""
    scale = lora_cfg["alpha"] / lora_cfg["dim"]
    print(f"[Automodel] LoRA scale (alpha/dim) = {scale}")

    merged = {}
    skip = set()
    n_merged = 0
    for key in list(keys):
        m = AUTOMODEL_LORA_A_RE.match(key)
        if not m:
            continue
        prefix = m.group(1)
        lora_a_key = key
        lora_b_key = f"{prefix}.lora_B.weight"
        base_key = f"{prefix}.weight"
        if lora_b_key not in keys or base_key not in keys:
            raise ValueError(f"Found {lora_a_key} but missing matching {lora_b_key} / {base_key}")

        base = get_tensor(base_key)
        lora_a = get_tensor(lora_a_key).to(torch.float32)
        lora_b = get_tensor(lora_b_key).to(torch.float32)
        delta = (lora_b @ lora_a) * scale
        new_weight = (base.to(torch.float32) + delta).to(base.dtype)

        merged[base_key] = new_weight
        skip |= {base_key, lora_a_key, lora_b_key}
        n_merged += 1
        print(f"  merged {base_key}  (delta scale {scale})")

    for key in keys - skip:
        merged[key] = get_tensor(key)

    print(f"Merged {n_merged} LoRA adapter pairs into base weights ({len(merged)} tensors total)")
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters into the base weights of a NeMo speechlm2 SALM checkpoint. "
        "Supports both the HF-PEFT style used by the 1B ('...base_layer.weight' / "
        "'...lora_A.default.weight') and the NeMo-Automodel-native style used by the 8B/23B "
        "('<name>.weight' + '<name>.lora_A.weight', keyed off config.json's 'use_nemo_automodel')."
    )
    parser.add_argument("checkpoint_dir", type=Path, help="Path to the checkpoint folder, e.g. .../hf_checkpoints/step_100000")
    parser.add_argument("output_dir", type=Path, help="Output folder to create (must not exist, must differ from checkpoint_dir)")
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir
    out_dir = args.output_dir
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    if out_dir.resolve() == ckpt_dir.resolve():
        raise ValueError("output_dir must be different from checkpoint_dir")
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists, refusing to overwrite")

    config_path = ckpt_dir / "config.json"
    config = json.loads(config_path.read_text())
    lora_cfg = config.get("lora")
    if not lora_cfg:
        raise ValueError(f"{config_path} has no 'lora' section, nothing to merge")

    is_automodel = bool(config.get("use_nemo_automodel"))
    if is_automodel:
        if "dim" not in lora_cfg or "alpha" not in lora_cfg:
            raise ValueError(f"use_nemo_automodel=true but lora config is missing 'dim'/'alpha': {lora_cfg}")
    else:
        if "r" not in lora_cfg or "lora_alpha" not in lora_cfg:
            raise ValueError(f"Expected HF-PEFT lora config with 'r'/'lora_alpha', got: {lora_cfg}")
    print(f"Detected LoRA style: {'NeMo-Automodel-native (8B/23B)' if is_automodel else 'HF-PEFT (1B)'}")

    safetensors_path = ckpt_dir / "model.safetensors"
    if not safetensors_path.is_file():
        raise FileNotFoundError(f"{safetensors_path} not found (sharded checkpoints not supported by this script)")

    with safe_open(str(safetensors_path), framework="pt") as f:
        keys = set(f.keys())
        get_tensor = f.get_tensor
        if is_automodel:
            merged = merge_automodel(keys, get_tensor, lora_cfg)
        else:
            merged = merge_peft(keys, get_tensor, lora_cfg)

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ckpt_dir, out_dir, ignore=shutil.ignore_patterns("model.safetensors"))

    new_config = {k: v for k, v in config.items() if k not in ("lora", "use_nemo_automodel")}
    (out_dir / "config.json").write_text(json.dumps(new_config, indent=2) + "\n")

    save_file(merged, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    print(f"Wrote {out_dir / 'model.safetensors'}")
    print(f"Wrote {out_dir / 'config.json'} (lora / use_nemo_automodel removed)")


if __name__ == "__main__":
    main()
