"""Multi-lookahead training of the duplex model on the frozen cache-aware encoder.

`model.perception.encoder.att_context_size` given as a list of lookaheads, e.g. [[56, 3], [56, 0], [56, 6], [56, 13]]
(optional `att_context_probs`, summing to 1): every training step uses one of them, drawn at random, so that
the projection and the LoRA learn the encoder outputs of every lookahead and the latency can be chosen at
inference (inference/stream_duplex.py --att-context-size). The FIRST one is used for validation and inference.

ConformerEncoder draws a lookahead per forward only in training mode, and DuplexSTTModel.training_step puts
its frozen modules (the encoder included) back in eval() at every step: left alone, the list would only ever
train its first lookahead. This callback does the draw instead, setting the encoder's current lookahead before
each training batch (in eval mode, the encoder uses `att_context_size`) and restoring the first one after.
One draw per step and per rank (seeded by rank): the same lookahead for a whole batch, as in NeMo's own
multi-lookahead training.
"""
import random

from lightning.pytorch import Callback


class RandomLookahead(Callback):
    def __init__(self, seed: int = 0):
        self.seed = seed

    @staticmethod
    def enabled(model) -> bool:
        return len(model.perception.encoder.att_context_size_all) > 1

    def on_fit_start(self, trainer, pl_module):
        enc = pl_module.perception.encoder
        self.enc = enc
        self.choices = [list(c) for c in enc.att_context_size_all]
        self.probs = list(enc.att_context_probs)
        self.default = list(self.choices[0])
        self.rng = random.Random(self.seed * 1000 + trainer.global_rank)
        enc.att_context_size = self.default

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        lookahead = self.rng.choices(self.choices, weights=self.probs)[0]
        self.enc.att_context_size = lookahead
        # Right context in frames, to split the training loss by lookahead in the logs.
        pl_module.log("train_lookahead", float(lookahead[1]), on_step=True, on_epoch=False, rank_zero_only=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.enc.att_context_size = self.default

    def on_validation_start(self, trainer, pl_module):
        if hasattr(self, "enc"):  # trainer.validate() before fit (baseline-val) never ran on_fit_start
            self.enc.att_context_size = self.default
