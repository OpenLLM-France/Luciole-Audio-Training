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
"""NeMo's examples/speechlm2/stt_duplex_train.py, plus the SALM validation features of ../training/salm_train.py:
`validate_before_fit` (baseline validation at step 0) and `validation_schedule` (sparse, front-loaded
validation steps).

Also moves `torch.cuda.set_device` under the main guard. With the 'spawn' start method every DataLoader
worker re-imports this module: at module level, that call opened a ~520 MiB CUDA context per worker on
its rank's GPU.
"""
import multiprocessing as mp
import os
import sys

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from agent_timing import apply_agent_timing
from lookahead import RandomLookahead
from nemo.collections.speechlm2 import DataModule, DuplexSTTDataset, DuplexSTTModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

# Set multiprocessing start method to 'spawn' for CUDA compatibility with DataLoader workers
# This prevents "Cannot re-initialize CUDA in forked subprocess" errors
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Start method already set


def _validate_and_log(trainer, model, datamodule, step, tag, ckpt_path=None):
    """trainer.validate() and log its epoch metrics at `step` (validate() doesn't persist them)."""
    # Lightning's own logging of these metrics uses the fit loop's step counter (0 outside fit): without
    # this, every checkpoint validation also landed at step 0, over the baseline.
    trainer.fit_loop.epoch_loop._batches_that_stepped = step
    results = trainer.validate(model, datamodule, ckpt_path=ckpt_path)
    if results:
        scalars = {}
        for key, value in results[0].items():
            try:
                scalars[key] = float(value)
            except (TypeError, ValueError):
                continue  # non-scalar entries aren't plottable
        for lg in trainer.loggers:
            lg.log_metrics(scalars, step=step)
            lg.save()
        logging.info(f"[{tag}] logged {len(scalars)} scalars at step {step}: {sorted(scalars)}")


@hydra_runner(config_path="conf", config_name="duplex_qa")
def train(cfg):
    # data.agent_timing (+ lag, wait_words) into every conversation_as_duplex entry, and the derived
    # data.agent_text_timestamps. Loads the train input_cfg and the validation sets (conf/data/val_cfg_<model>.yaml),
    # both passed as paths, before resolve so their ${oc.env:...} interpolations are resolved with the rest.
    apply_agent_timing(cfg)
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = True
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    # avoid using `=` in the checkpoint name
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            callback.CHECKPOINT_EQUALS_CHAR = "-"

    with trainer.init_module():
        model = DuplexSTTModel(OmegaConf.to_container(cfg.model, resolve=True))

    dataset = DuplexSTTDataset(
        tokenizer=model.tokenizer,
        frame_length=cfg.data.frame_length,
        source_sample_rate=cfg.data.source_sample_rate,
        input_roles=cfg.data.input_roles,
        output_roles=cfg.data.output_roles,
        aug_by_swap_role=cfg.data.get("aug_by_swap_role", False),
        cfg=cfg.data,
        model_cfg=cfg.model,
    )
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)

    # Encoder att_context_size given as a list: one lookahead drawn per training step (lookahead.py).
    if RandomLookahead.enabled(model):
        trainer.callbacks.append(RandomLookahead(seed=cfg.data.train_ds.get("seed", 0)))
        enc = model.perception.encoder
        logging.info(f"[lookahead] training on {enc.att_context_size_all} (probs {enc.att_context_probs}), "
                     f"validation on {enc.att_context_size_all[0]}")

    if (sched := cfg.get("validation_schedule", None)) and trainer.limit_val_batches:
        # Imported here, not at module level: salm_train sets the CUDA device on import, which the
        # spawned DataLoader workers (they re-import this module) must not do.
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
        from salm_train import ValidationSchedule

        class ResumeSafeValidationSchedule(ValidationSchedule):
            """Lightning's candidates follow the batch index, not global_step: after a resume from a
            checkpoint taken at an epoch's last batch, that batch is stepped again and global_step runs one
            ahead (candidates at 7501, 10001, ...). ValidationSchedule matches global_step exactly and then
            cancelled every later validation (duplex_2heads_delay1_4gpu, 2026-10-14): match the nearest
            multiple of the interval instead."""

            def wants(self, step: int) -> bool:
                if isinstance(trainer.val_check_batch, int) and trainer.val_check_batch > 0:
                    step = round(step / trainer.val_check_batch) * trainer.val_check_batch
                return super().wants(step)

        cb = ResumeSafeValidationSchedule(sched.steps, sched.get("then_every", None))
        trainer.callbacks.append(cb)
        logging.info(
            f"[val-schedule] validating at steps {cb.steps}"
            + (f" then every {cb.then_every}" if cb.then_every else "")
            + f" (val_check_interval={trainer.val_check_interval} gates the candidates)"
        )

    # Baseline validation at step 0, fresh runs only: same logic and caveats as ../training/salm_train.py
    # (Lightning's sanity check doesn't log; trainer.validate() doesn't persist its epoch metrics).
    baseline_val = bool(cfg.get("validate_before_fit", False)) and bool(trainer.limit_val_batches)
    if baseline_val and trainer.ckpt_path:
        logging.info(f"[baseline-val] skipped: resuming from {trainer.ckpt_path}")
        baseline_val = False

    if baseline_val:
        logging.info("[baseline-val] running validation before training")
        _validate_and_log(trainer, model, datamodule, step=0, tag="baseline-val")

    # Validation only, of existing checkpoints (++validate_ckpts=[path, ...]), each logged at its
    # global_step: catches up validations a run skipped, then exits without training.
    if ckpts := cfg.get("validate_ckpts", None):
        for path in ckpts:
            step = int(torch.load(path, map_location="cpu", mmap=True, weights_only=False)["global_step"])
            logging.info(f"[validate-ckpt] {path} (global_step {step})")
            _validate_and_log(trainer, model, datamodule, step=step, tag="validate-ckpt", ckpt_path=path)
        return

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    train()
