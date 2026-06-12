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


@hydra_runner(config_path="conf", config_name="config_canary-1b-v2_linear")
def train(cfg):
    OmegaConf.resolve(cfg)
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
    _enc = model.perception.encoder
    logging.info(
        f"[attn-check] self_attention_model={_enc.self_attention_model} "
        f"att_context_size={_enc.att_context_size} "
        f"attn_class={type(_enc.layers[0].self_attn).__name__}"
    )

    dataset = SALMDataset(tokenizer=model.tokenizer)
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
