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
import os

import torch
from lightning.pytorch import Trainer, seed_everything
from omegaconf import OmegaConf

from nemo.collections.speechlm2 import SALM, DataModule, SALMDataset
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

if torch.cuda.is_available():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def _normalize_for_automodel(cfg):
    """Rewrite DDP/HF-PEFT-shaped config blocks into NeMo-Automodel shapes.

    base.yaml is built for SALM (HuggingFace backend + DDPStrategy). Hydra
    deep-merges dicts and cannot delete keys, so an overlay can't strip the
    HF-PEFT lora keys or the DDP-only strategy kwargs — they would leak into
    SALMAutomodel's PeftConfig / AutomodelParallelStrategy and raise TypeError.
    Normalize them here, once, when use_nemo_automodel is set.
    """
    OmegaConf.set_struct(cfg, False)
    # LoRA: HF-PEFT keys -> Automodel PeftConfig keys (make_peft_config forwards
    # every key verbatim, so r/lora_alpha/lora_dropout/task_type must be remapped).
    lora = cfg.model.get("lora")
    if lora:
        rename = {"r": "dim", "lora_alpha": "alpha", "lora_dropout": "dropout"}
        cfg.model.lora = {rename.get(k, k): v for k, v in lora.items() if k != "task_type"}
    # Strategy: replace DDPStrategy wholesale. AutomodelParallelStrategy calls the
    # model's configure_model() with the device mesh (so leave init_configure_model
    # at its default false). ep_size=1 for dense LLMs (no MoE).
    #
    # Default = pure FSDP2 (params/grads/optim sharded across all DP ranks, ZeRO-3).
    # An optional `model.automodel_parallel` block overrides any strategy kwarg, e.g.
    # to run DDP-equivalent (replicate, no shard) or HSDP. It is popped here so it
    # never reaches the SALMAutomodel ctor (which would reject the unknown key).
    strategy = {
        "_target_": "nemo.collections.speechlm2.parts.parallel.AutomodelParallelStrategy",
        "dp_size": None,
        "tp_size": 1,
        "pp_size": 1,
        "cp_size": 1,
        "ep_size": 1,
    }
    parallel_override = cfg.model.get("automodel_parallel")
    if parallel_override is not None:
        override = OmegaConf.to_container(parallel_override, resolve=True)
        # Convenience sentinel: dp_replicate_size: "world" -> replicate across ALL
        # ranks so dp_shard collapses to 1 (DDP-equivalent: full params per GPU,
        # only a gradient all-reduce, no param all-gather). WORLD_SIZE is set by
        # torchrun before this runs, so it resolves regardless of --gpus.
        if override.get("dp_replicate_size") == "world":
            override["dp_replicate_size"] = int(os.environ.get("WORLD_SIZE", 1))
        strategy.update(override)
        del cfg.model["automodel_parallel"]
    cfg.trainer.strategy = strategy


@hydra_runner(config_path="conf", config_name="config_canary-1b-v2_linear")
def train(cfg):
    OmegaConf.resolve(cfg)
    # `lora: null` in a config overlay means "disable LoRA". base.yaml always
    # defines model.lora and Hydra can't delete the key (it deep-merges dicts), so
    # it arrives as None and SALM.maybe_install_lora would call LoraConfig(**None).
    # Strip it here so LoRA is simply not installed (e.g. curriculum stages 1-2).
    if cfg.model.get("lora", "x") is None:
        OmegaConf.set_struct(cfg, False)
        del cfg.model["lora"]
    if cfg.model.get("use_nemo_automodel", False):
        _normalize_for_automodel(cfg)
    if torch.cuda.is_available():
        torch.distributed.init_process_group(backend="nccl")
    seed_everything(cfg.data.train_ds.seed)
    torch.set_float32_matmul_precision("medium")
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    model_cls = SALM
    if cfg.model.get("use_nemo_automodel", False):
        from nemo.collections.speechlm2 import SALMAutomodel

        model_cls = SALMAutomodel

    with trainer.init_module():
        model = model_cls(OmegaConf.to_container(cfg.model, resolve=True))

    # [attn-check] Log the self-attention the encoder *actually instantiated* (reads the
    # live nn.Module, not the yaml): rel_pos -> RelPositionMultiHeadAttention (full),
    # rel_pos_local_attn -> RelPositionMultiHeadAttentionLongformer (local). Per-run proof
    # for the attention A/B; harmless for every other run.
    # SALMAutomodel defers building perception/llm to configure_model() (called later by
    # AutomodelParallelStrategy with the device mesh), so model.perception is None here — skip.
    if model.perception is not None:
        _enc = model.perception.encoder
        logging.info(
            f"[attn-check] self_attention_model={_enc.self_attention_model} "
            f"att_context_size={_enc.att_context_size} "
            f"attn_class={type(_enc.layers[0].self_attn).__name__}"
        )
    else:
        logging.info("[attn-check] perception not built yet (Automodel deferred init); skipping.")

    dataset = SALMDataset(tokenizer=model.tokenizer)
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
