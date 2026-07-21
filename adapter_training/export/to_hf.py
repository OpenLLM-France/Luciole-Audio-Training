# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_file

from nemo.collections.speechlm2.parts.hf_hub import LLM_BACKBONE_DIR
from nemo.core.config import hydra_runner
from nemo.utils.dtype import str_to_dtype
from nemo.utils.model_utils import import_class_by_path


@dataclass
class HfExportConfig:
    # Name of the model class to be imported, e.g. nemo.collections.speechlm2.models.SALM
    class_path: str

    # Path to PyTorch Lightning checkpoint file (normal ckpt) or directory (distributed ckpt)
    ckpt_path: str

    # Path to the experiment's config, used to instantiate the model class.
    ckpt_config: str

    # Path where we should save the HuggingFace Hub compatible checkpoint
    output_dir: str

    # Dtype used for stored parameters
    dtype: str = "bfloat16"


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    if Path(checkpoint_path).is_dir():
        from torch.distributed.checkpoint import load

        state_dict = {"state_dict": model.state_dict()}
        load(state_dict, checkpoint_id=checkpoint_path)
        model.load_state_dict(state_dict["state_dict"])
    else:
        ckpt_data = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt_data["state_dict"])


def setup_distributed_from_config(strategy_cfg: dict) -> Any:
    """Initialize torch.distributed and create a device mesh from a Hydra strategy config.

    Instantiates the strategy from the trainer config dict (as found in the
    experiment YAML), initializes the process group, resolves automodel
    configs, and calls :meth:`strategy.create_device_mesh`.

    Returns:
        An :class:`AutomodelParallelStrategy` with device_mesh ready.
    """
    import hydra

    from nemo.utils.trainer_utils import _resolve_automodel_configs

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    strategy = hydra.utils.instantiate(strategy_cfg)
    _resolve_automodel_configs(strategy)
    strategy.create_device_mesh()
    return strategy


def consolidate_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Gather a full (non-sharded) state dict from a model with DTensor parameters."""
    from torch.distributed.tensor import DTensor

    consolidated = {}
    for key, value in model.state_dict().items():
        if isinstance(value, DTensor):
            consolidated[key] = value.full_tensor().cpu()
        else:
            consolidated[key] = value.cpu()
    return consolidated


def _canonical_torch_dtype_name(dtype: str | torch.dtype) -> str:
    """Return the PyTorch dtype name accepted by Transformers configs."""
    return str(str_to_dtype(dtype)).replace("torch.", "")


# Nemotron-H is reachable through two implementations that differ ONLY in the name of the
# trunk submodule: the bundled remote code shipped with Luciole-8B calls it `backbone`,
# transformers' built-in nemotron_h calls it `model`. Training builds the bundled one (the
# nemo_automodel parallelizer wants `.backbone`), so the trained state dict comes out
# `llm.backbone.*` -- but nothing could then EVALUATE it:
#
#   * The built-in cannot load it: `llm.backbone.*` matches none of its `llm.model.*`
#     params, and hf_hub._load_state_dict_with_dtensors keeps only keys present in BOTH, so
#     every LLM tensor was dropped SILENTLY -> a randomly initialised 8B LLM emitting
#     fluent, confident, completely unrelated text (no missing-key warning anywhere).
#   * The bundled one cannot generate: its modeling_nemotron_h.py is a transformers-4.x-era
#     copy whose prepare_inputs_for_generation assumes generate() has not pre-created the
#     cache. transformers>=5 inverts that, so it unconditionally slices `input_ids`, which
#     is None on SALM's inputs_embeds-only (audio-conditioned) path -> AttributeError:
#     'NoneType' object has no attribute 'shape'. And nemo_automodel 0.4.0 hard-pins
#     transformers==5.5.0, so downgrading is not on the table.
#
# So the export normalises onto the BUILT-IN naming. That direction is canonical, not a
# preference: vLLM's own nemotron_h.py does the same remap when loading Nemotron-H weights
# (`name = name.replace("backbone", "model")`, models/nemotron_h.py:585) and calls its trunk
# `self.model`. Since that replace is a no-op once tensors are already `model.*`, exporting
# in built-in naming keeps the vLLM path working too.
LLM_TRUNK_OLD = "llm.backbone."
LLM_TRUNK_NEW = "llm.model."


def _remap_llm_trunk(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename the LLM trunk from the bundled `backbone` to the built-in `model`.

    A pure prefix rename -- layer ordering and every name below the trunk are identical
    between the two implementations. `perception.*` and `llm.lm_head.weight` are untouched.
    No-op for a state dict that is already in built-in naming (or a non-Nemotron-H backbone).
    """
    n = sum(1 for k in state_dict if k.startswith(LLM_TRUNK_OLD))
    if n == 0:
        return state_dict
    print(f"[to_hf] remapped {n} LLM tensors {LLM_TRUNK_OLD}* -> {LLM_TRUNK_NEW}* "
          "(transformers built-in naming)")
    return {
        (LLM_TRUNK_NEW + k[len(LLM_TRUNK_OLD):] if k.startswith(LLM_TRUNK_OLD) else k): v
        for k, v in state_dict.items()
    }


def _hf_export_config(model: torch.nn.Module, dtype: str | torch.dtype) -> dict[str, Any]:
    """Build the exported root config without mutating the training config."""
    config = OmegaConf.to_container(model.cfg) if isinstance(model.cfg, DictConfig) else deepcopy(model.cfg)
    dtype_name = _canonical_torch_dtype_name(dtype)
    config["dtype"] = dtype_name
    config["torch_dtype"] = dtype_name
    # Training-only weight init: the exported checkpoint already holds all trained
    # weights, so re-seeding from this (often now-missing) training checkpoint at
    # load time is both unnecessary and wrong. Drop it from the exported config.
    config.pop("init_from_checkpoint", None)
    # The weights are exported in built-in naming (see _remap_llm_trunk), so the loader must
    # build the BUILT-IN class to match them. trust_remote_code is a training-time need (the
    # parallelizer wants the bundled `.backbone`); carrying it into the exported checkpoint
    # would rebuild the bundled class at eval, whose generate() is broken under
    # transformers>=5. force_hf goes with it: it only selects between nemo_automodel's custom
    # impl and the HF one, and is meaningless once no remote code ships.
    config["trust_remote_code"] = False
    return config


def save_hf_checkpoint(model: torch.nn.Module, state_dict: dict, cfg: HfExportConfig) -> None:
    """Save a consolidated state dict and model config in HuggingFace Hub format."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_dtype = str_to_dtype(cfg.dtype)
    state_dict = {k: v.to(target_dtype) for k, v in state_dict.items()}
    state_dict = _remap_llm_trunk(state_dict)

    save_file(state_dict, output_dir / "model.safetensors")

    config = _hf_export_config(model, cfg.dtype)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    save_llm_backbone_config(model, output_dir)


def save_llm_backbone_config(model: torch.nn.Module, output_dir: str | Path) -> None:
    """Save the original LLM config separately from the NeMo wrapper config."""
    llm_config = getattr(getattr(model, "llm", None), "config", None)
    if llm_config is None:
        return

    llm_backbone_dir = Path(output_dir) / LLM_BACKBONE_DIR
    llm_backbone_dir.mkdir(parents=True, exist_ok=True)
    llm_config.save_pretrained(str(llm_backbone_dir))

    # save_pretrained() copies the config's OWN dynamic module (configuration_nemotron_h.py)
    # and writes an auto_map naming modeling_nemotron_h.NemotronHForCausalLM -- but it never
    # copies the modeling file, because a config object does not know about it. The exported
    # llm_backbone/ therefore had a DANGLING auto_map: trust_remote_code=True would resolve
    # the config remotely and then silently fall back to the built-in for the MODEL, which is
    # how a checkpoint whose weights were `backbone`-named ended up being loaded into a
    # `model`-named class and losing every LLM tensor.
    #
    # We export in built-in naming (see _remap_llm_trunk) and set trust_remote_code=False, so
    # no remote code should be consulted at all. Strip the auto_map and drop the copied .py so
    # the artifact is self-consistent and there is nothing to fall back to. Leaving them would
    # re-arm exactly the failure above for anyone who flips trust_remote_code back on.
    cfg_path = llm_backbone_dir / "config.json"
    if cfg_path.is_file():
        llm_cfg = json.loads(cfg_path.read_text())
        if llm_cfg.pop("auto_map", None) is not None:
            cfg_path.write_text(json.dumps(llm_cfg, indent=2))
            print("[to_hf] stripped dangling auto_map from llm_backbone/config.json")
    for stray in llm_backbone_dir.glob("*.py"):
        stray.unlink()
        print(f"[to_hf] removed unused bundled module {stray.name} (built-in naming exported)")


def _detect_vllm_architecture(model_cfg: dict) -> str:
    """Determine the vLLM plugin model class for the checkpoint.

    The SALM plugin registers a single architecture name and selects between
    transformer and hybrid backends at instantiation time, so this function
    just verifies the backbone config is reachable and returns the unified
    name; the hybrid-vs-transformer split is handled inside the plugin.

    Raises:
        ValueError: if the HF config can't be loaded or has no 'architectures'.
    """
    pretrained_llm = model_cfg.get("pretrained_llm", "")
    try:
        from transformers import AutoConfig

        llm_cfg = AutoConfig.from_pretrained(pretrained_llm, trust_remote_code=True)
    except Exception as e:
        raise ValueError(
            f"Could not load HF config for pretrained_llm={pretrained_llm!r}: {e}. "
            f"Fix the 'pretrained_llm' field or ensure HF access during conversion."
        ) from e

    archs = getattr(llm_cfg, "architectures", [])
    if not archs:
        raise ValueError(f"HF config for {pretrained_llm!r} has empty 'architectures'.")

    return "NeMoSpeechLMForConditionalGeneration"


def prepare_for_vllm(output_dir: str, model_cfg: dict) -> None:
    """Patch a saved checkpoint to be vLLM-ready.

    Adds tokenizer (with audio token and chat template), patches config.json
    with model_type/architectures, and writes generation_config.json.

    Args:
        output_dir: Path to the HuggingFace checkpoint directory.
        model_cfg: Model config dict (from experiment YAML).

    Raises:
        ValueError: If ``pretrained_llm`` or ``audio_locator_tag`` is missing.
    """
    from transformers import AutoTokenizer

    from nemo.utils import logging as LOG

    output_dir = Path(output_dir)
    pretrained_llm = model_cfg.get("pretrained_llm", "")
    if not pretrained_llm:
        raise ValueError("model config has no 'pretrained_llm'; cannot load tokenizer for vLLM")

    # ``model.audio_locator_tag`` is the SoT for the audio placeholder;
    # fail loud rather than default, since a mismatch is silent at inference.
    audio_token = model_cfg.get("audio_locator_tag")
    if not audio_token:
        raise ValueError("model config has no 'audio_locator_tag' (set it in the training YAML).")

    # 1. Patch config.json (arch, model_type, audio_locator_tag for vLLM plugin).
    arch_model_cfg = dict(model_cfg)
    llm_backbone_dir = output_dir / LLM_BACKBONE_DIR
    if (llm_backbone_dir / "config.json").exists():
        arch_model_cfg["pretrained_llm"] = str(llm_backbone_dir)
    arch = _detect_vllm_architecture(arch_model_cfg)
    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["model_type"] = "nemo_speechlm"
    config["architectures"] = [arch]
    config["audio_locator_tag"] = audio_token
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    # 2. Save tokenizer (backbone chat_template carries over via save_pretrained)
    existing = [
        f.name
        for f in output_dir.iterdir()
        if f.name in ("tokenizer_config.json", "tokenizer.json", "generation_config.json")
    ]
    if existing:
        LOG.info("Overwriting existing files in %s: %s", output_dir, existing)
    tok = AutoTokenizer.from_pretrained(pretrained_llm, trust_remote_code=True)
    if audio_token not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [audio_token]})
    tok.save_pretrained(str(output_dir))
    # Newer transformers splits long chat_template into a separate
    # ``chat_template.jinja`` file; inline it back and drop the file.
    tok_cfg_path = output_dir / "tokenizer_config.json"
    tok_cfg = json.loads(tok_cfg_path.read_text())
    jinja_file = output_dir / "chat_template.jinja"
    if jinja_file.exists():
        jinja_from_file = jinja_file.read_text()
        if jinja_from_file.strip():
            tok_cfg["chat_template"] = jinja_from_file
        jinja_file.unlink()
    # Normalize to dict form; transformers writes a list which HF loaders reject.
    tok_cfg["extra_special_tokens"] = {"audio_token": audio_token}
    # Some NeMo containers save a proprietary ``TokenizersBackend`` class
    # unknown to HF; the underlying tokenizer.json is standard, so force
    # the universal base class.
    tok_cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    tok_cfg_path.write_text(json.dumps(tok_cfg, indent=2) + "\n")

    # 4. Minimal generation_config.json (EOS only; sampling params belong on
    #    the server, not baked into the checkpoint).
    gen_cfg = {"eos_token_id": [tok.eos_token_id]}
    (output_dir / "generation_config.json").write_text(json.dumps(gen_cfg, indent=2) + "\n")


def _try_prepare_for_vllm(output_dir: str, model_cfg: dict) -> None:
    """Run vLLM prep; on ``ValueError``, warn and keep the HF-only output.

    Backward compat for callers that never needed vLLM (e.g., NeMo SALM).
    """
    from nemo.utils import logging as LOG

    try:
        prepare_for_vllm(output_dir, model_cfg)
    except ValueError as e:
        LOG.warning(
            "Checkpoint saved as HF-only; vLLM prep skipped: %s. "
            "The checkpoint is still loadable by NeMo SALM and plain HF, but "
            "is NOT vLLM-ready until prep succeeds.",
            e,
        )


def _uses_automodel_parallel(strategy_cfg: dict) -> bool:
    """Check if the strategy config targets AutomodelParallelStrategy."""
    target = strategy_cfg.get("_target_", "")
    return "AutomodelParallelStrategy" in target


def load_model(ckpt_path: str, class_path: str, ckpt_config: str, dtype: str = "bfloat16"):
    """Build the model from ``ckpt_config`` and load a single-file Lightning checkpoint.

    Mirrors the non-distributed branch of ``main`` and returns the model ready for
    ``save_pretrained``. For distributed (directory) checkpoints, use ``main`` via
    ``torchrun`` instead — this helper only handles single-file ``.ckpt`` files.
    """
    if not Path(ckpt_path).exists():
        raise RuntimeError(f"No such file or directory: {ckpt_path}")

    full_cfg = OmegaConf.to_container(OmegaConf.load(ckpt_config), resolve=True)
    model_cfg = full_cfg["model"]
    model_cfg["torch_dtype"] = _canonical_torch_dtype_name(dtype)
    cls = import_class_by_path(class_path)

    model_cfg["init_configure_model"] = True
    model = cls(model_cfg)
    load_checkpoint(model, ckpt_path)
    model = model.to(str_to_dtype(dtype))
    model_cfg["pretrained_weights"] = False
    return model


def export_checkpoint(ckpt_path: str, class_path: str, ckpt_config: str, output_dir: str, dtype: str = "bfloat16"):
    """Export a single-file (non-distributed) checkpoint to HuggingFace format.

    Loads the model and writes the full HF export: weights + root config, the separate
    LLM backbone config, the tokenizer, and (best-effort) vLLM artifacts. This is the
    shared single-checkpoint path used by both ``main`` and the batch exporter
    (``export_experiment.py``). For distributed (directory) checkpoints, use ``main``
    via ``torchrun`` instead. Returns the loaded model.
    """
    model = load_model(ckpt_path, class_path, ckpt_config, dtype)
    export_cfg = _hf_export_config(model, dtype)
    model.save_pretrained(str(output_dir), config=export_cfg)
    save_llm_backbone_config(model, output_dir)
    if getattr(model, "tokenizer", None) is not None:
        model.tokenizer.save_pretrained(str(output_dir))
    _try_prepare_for_vllm(str(output_dir), export_cfg)
    return model


@hydra_runner(config_name="HfExportConfig", schema=HfExportConfig)
def main(cfg: HfExportConfig) -> None:
    """
    Read PyTorch Lightning checkpoint and export the model to HuggingFace Hub format.
    The resulting model can be then initialized via ModelClass.from_pretrained(path).

    Also supports distributed checkpoints for models trained with FSDP2/TP
    via AutomodelParallelStrategy.  Parallelism sizes (tp_size, pp_size, etc.)
    are read automatically from the ``trainer.strategy`` section of the
    experiment config (``ckpt_config``).

    When the checkpoint is a distributed checkpoint (a directory), launch this
    script via ``torchrun`` with the same number of GPUs used for training.

    Examples:
        # Single-file checkpoint — original SALM (HF Transformers backend):
        python to_hf.py \\
            class_path=nemo.collections.speechlm2.models.SALM \\
            ckpt_path=/path/to/checkpoint.ckpt \\
            ckpt_config=/path/to/config.yaml \\
            output_dir=/path/to/hf_output

        # Single-file checkpoint — SALMAutomodel (NeMo Automodel backend):
        python to_hf.py \\
            class_path=nemo.collections.speechlm2.models.SALMAutomodel \\
            ckpt_path=/path/to/checkpoint.ckpt \\
            ckpt_config=/path/to/config.yaml \\
            output_dir=/path/to/hf_output

        # Distributed checkpoint (parallelism read from config automatically):
        torchrun --nproc-per-node=8 to_hf.py \\
            class_path=nemo.collections.speechlm2.models.SALMAutomodel \\
            ckpt_path=/path/to/distributed_ckpt_dir \\
            ckpt_config=/path/to/config.yaml \\
            output_dir=/path/to/hf_output
    """
    if not Path(cfg.ckpt_path).exists():
        raise RuntimeError(f"No such file or directory: {cfg.ckpt_path}")

    full_cfg = OmegaConf.to_container(OmegaConf.load(cfg.ckpt_config), resolve=True)
    model_cfg = full_cfg["model"]
    model_cfg["torch_dtype"] = _canonical_torch_dtype_name(cfg.dtype)
    cls = import_class_by_path(cfg.class_path)

    strategy_cfg = full_cfg.get("trainer", {}).get("strategy", {})

    _is_torchrun = "RANK" in os.environ
    if _is_torchrun and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    is_distributed = (
        _is_torchrun
        and Path(cfg.ckpt_path).is_dir()
        and _uses_automodel_parallel(strategy_cfg)
        and dist.get_world_size() > 1
    )

    if is_distributed:
        strategy = setup_distributed_from_config(strategy_cfg)

        # Don't call configure_model() inside __init__ — we set device_mesh first.
        model_cfg["init_configure_model"] = False
        model = cls(model_cfg)
        model.configure_model(
            device_mesh=strategy.device_mesh,
            distributed_config=strategy.distributed_config,
            moe_config=strategy.moe_config,
            moe_mesh=strategy.moe_mesh,
        )
        model_cfg["pretrained_weights"] = False

        load_checkpoint(model, cfg.ckpt_path)

        # Consolidate DTensors to regular tensors and save on rank 0.
        consolidated = consolidate_state_dict(model)
        if dist.get_rank() == 0:
            save_hf_checkpoint(model, consolidated, cfg)
            _try_prepare_for_vllm(cfg.output_dir, model_cfg)

        dist.barrier()
        dist.destroy_process_group()
    else:
        export_checkpoint(cfg.ckpt_path, cfg.class_path, cfg.ckpt_config, cfg.output_dir, cfg.dtype)


if __name__ == "__main__":
    main()
