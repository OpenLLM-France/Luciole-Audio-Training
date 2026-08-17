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
from lightning.pytorch import Callback, Trainer, seed_everything
from omegaconf import OmegaConf

from nemo.collections.speechlm2 import SALM, DataModule, SALMDataset
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

if torch.cuda.is_available():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


class ValidationSchedule(Callback):
    """Validate on a SPARSE, front-loaded schedule instead of a fixed interval.

    Motivation: the interesting movement happens early. A fixed
    `val_check_interval` either wastes time late in a long run or is too coarse
    to catch the first few thousand steps. This gives e.g.
    500, 1000, 2500, 5000, 10000, 15000, 20000, then every 10k.

    How it works — and why it looks the way it does. Lightning decides *whether*
    to validate with a modulo (`training_epoch_loop._should_check_val_fx`:
    `(current_iteration + 1) % trainer.val_check_batch == 0`), so an arbitrary
    milestone list cannot be expressed by tuning `val_check_interval` alone.
    So: set `val_check_interval` to the GCD of the milestones (every milestone
    must be a multiple of it) and cancel the unwanted checks here.

    Cancelling is done via `val_loop._max_batches`, NOT `trainer.limit_val_batches`:
    `_EvaluationLoop.setup_data` early-returns once `_combined_loader` exists
    (evaluation_loop.py:147), so mutating `limit_val_batches` has no effect after
    the first validation. `_EvaluationLoop.run` checks `skip` (== `sum(max_batches) == 0`)
    *before* `on_run_start`, so a zeroed loop costs nothing and fires no hooks —
    importantly it never reaches `on_validation_epoch_end`, which would raise on
    an empty `_partial_val_losses`.

    `_max_batches` is Lightning-internal. It is stable in the pinned 2.x used
    here, but this is the part to check first if a Lightning upgrade breaks
    validation timing.
    """

    def __init__(self, steps, then_every=None):
        self.steps = sorted({int(s) for s in steps})
        self.then_every = int(then_every) if then_every else None
        self._full = None  # real per-dataloader batch counts, captured on first use

    def wants(self, step: int) -> bool:
        if step in self.steps:
            return True
        if self.then_every and self.steps and step > self.steps[-1]:
            return step % self.then_every == 0
        return False

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        # Runs before `on_advance_end`, where Lightning decides on validation.
        loop = trainer.fit_loop.epoch_loop.val_loop
        max_batches = getattr(loop, "_max_batches", None)
        if not max_batches:
            return  # dataloaders not set up yet; let the first validation through
        if self._full is None or any(b > 0 for b in max_batches):
            self._full = list(max_batches)  # remember the real sizes before zeroing
        run = self.wants(trainer.global_step)
        loop._max_batches = list(self._full) if run else [0] * len(self._full)


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
    # to run HSDP (dp_replicate_size=2). It is popped here so it never reaches the
    # SALMAutomodel ctor (which would reject the unknown key).
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
        if override.get("dp_replicate_size") == "world":
            raise ValueError(
                "dp_replicate_size: 'world' (DDP-equivalent) is not supported by FSDP2 -- "
                "create_device_mesh requires dp_replicate_size < dp_size. Use an integer "
                "< the GPU count for HSDP, or 1 for full sharding."
            )
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
    # Forces Nemotron-H's Mamba2 mixers onto the NON-fused (slower) path so a LoRA on
    # out_proj is actually trained: use_mem_eff_path is hard-coded True in
    # modeling_nemotron_h.py (not a config field) and the fused kernel consumes
    # out_proj.weight directly, so the LoRA gets zero gradient and stays inert. Patch
    # the CLASS, since the Automodel LLM is only built later in configure_model. The
    # import is lazy so non-Nemotron runs never touch it.
    if cfg.model.get("mamba_force_torch_path", None) is not None:
        OmegaConf.set_struct(cfg, False)
        _force_torch = bool(cfg.model["mamba_force_torch_path"])
        del cfg.model["mamba_force_torch_path"]
        if _force_torch:
            from transformers.models.nemotron_h import modeling_nemotron_h as _mnh
            _orig_mixer_init = _mnh.NemotronHMamba2Mixer.__init__

            def _mixer_init_no_memeff(self, *args, **kwargs):
                _orig_mixer_init(self, *args, **kwargs)
                self.use_mem_eff_path = False

            _mnh.NemotronHMamba2Mixer.__init__ = _mixer_init_no_memeff
            logging.info(
                "[mamba] use_mem_eff_path forced False on NemotronHMamba2Mixer -- "
                "non-fused path so a LoRA on out_proj is actually trained"
            )
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

    if (sched := cfg.get("validation_schedule", None)) and trainer.limit_val_batches:
        cb = ValidationSchedule(sched.steps, sched.get("then_every", None))
        trainer.callbacks.append(cb)
        logging.info(
            f"[val-schedule] validating at steps {cb.steps}"
            + (f" then every {cb.then_every}" if cb.then_every else "")
            + f" (val_check_interval={trainer.val_check_interval} gates the candidates)"
        )

    # Baseline validation at step 0, before any weight update. Gives every
    # val_loss_*/val_acc_* curve its true starting point (the warm-start
    # checkpoint's state) — otherwise the first point is at val_check_interval
    # and an improvement is indistinguishable from a regression.
    # Lightning's sanity check can't do this: it computes validation but
    # logger_connector skips logging while trainer.sanity_checking is True.
    # Fresh runs only. On a resume it restates the previous leg's last validation, but
    # still lands at step 0, so TensorBoard draws a flat line across the whole new leg.
    # Key off trainer.ckpt_path, which exp_manager sets only when it really found a
    # checkpoint — stricter than the launcher's --resume, since resume_if_exists=true
    # means a plain relaunch resumes without that flag.
    baseline_val = bool(cfg.get("validate_before_fit", False)) and bool(trainer.limit_val_batches)
    if baseline_val and trainer.ckpt_path:
        logging.info(
            f"[baseline-val] skipped: resuming from {trainer.ckpt_path}, so a step-0 "
            f"baseline would only restate the previous leg's last validation"
        )
        baseline_val = False

    if baseline_val:
        logging.info("[baseline-val] running validation before training")
        results = trainer.validate(model, datamodule)

        # `trainer.validate()` PRINTS its results table but does not persist the
        # epoch-level metrics: the TensorBoard file it opens ends up holding only
        # per-step scalars (validation_step_timing), so every val_loss_*/val_acc_*
        # curve would still start at the first val_check_interval. Push the
        # returned metrics in explicitly at step 0. They land in this stage's
        # event file, and TensorBoard merges all event files in the run dir, so
        # the step-0 point joins the curve that `fit` writes afterwards.
        if results:
            baseline = {}
            for key, value in results[0].items():
                try:
                    baseline[key] = float(value)
                except (TypeError, ValueError):
                    continue  # non-scalar entries aren't plottable
            for lg in trainer.loggers:
                lg.log_metrics(baseline, step=0)
                lg.save()
            logging.info(f"[baseline-val] logged {len(baseline)} scalars at step 0: {sorted(baseline)}")

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
