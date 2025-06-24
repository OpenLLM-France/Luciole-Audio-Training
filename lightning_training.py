import os
import argparse
import logging
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from data.audio_chuncks_dataset import get_dataset
from models.lucas_setup import model_factory
from configs import ModelConfig, TrainConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_ppl(loss):
    return torch.exp(loss)


class LightningLucAS(pl.LightningModule):
    def __init__(self, model, tokenizer, train_config: TrainConfig, model_config: ModelConfig):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.train_config = train_config
        self.model_config = model_config
        self.save_hyperparameters(ignore=['model', 'tokenizer'])

        if self.train_config.train_projector_only:
            self.model.train_projector_only()

        trainable_params = model.get_trainable_parameters()
        logger.info("Trainable parameters summary:")
        for component, num_params in trainable_params.items():
            logger.info(f"  {component}: {num_params:,}")
        self.model_train_params = (
            self.model.encoder_projector.parameters()
            if self.train_config.train_projector_only else self.model.parameters()
        )

    def forward(self, **batch):
        outputs, acc = self.model(**batch)
        return outputs, acc

    def training_step(self, batch, batch_idx):
        outputs, acc = self.model(**batch)
        loss = outputs.loss
        ppl = compute_ppl(loss)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_ppl', ppl, prog_bar=False, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model_train_params,
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay
        )
        num_steps = self.train_config.num_epochs * self.trainer.estimated_stepping_batches
        scheduler = {
            'scheduler': LambdaLR(
                optimizer,
                lr_lambda=lambda step: (
                    min(1.0, step / self.train_config.warmup_step)
                    if step < self.train_config.warmup_step
                    else max(0.0, 1 - (step - self.train_config.warmup_step) / max(1, num_steps - self.train_config.warmup_step))
                )
            ),
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]


class SpeechDataModule(pl.LightningDataModule):
    def __init__(self, dataset_dirs, tokenizer, model_config: ModelConfig, train_config: TrainConfig):
        super().__init__()
        self.dataset_dirs = dataset_dirs
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.train_config = train_config
        self.num_workers = self.model_config.num_workers

    def setup(self, stage=None):
        self.train_ds = get_dataset(
            dataset_dir=self.dataset_dirs,
            tokenizer=self.tokenizer,
            model_config=self.model_config,
            train_config=self.train_config,
            split='train'
        )
        self.val_ds = None

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.train_config.batch_size_training,
            shuffle=False,
            num_workers=self.model_config.num_workers,
            collate_fn=self.train_ds.data_collator
        )

    def val_dataloader(self):
        return None


class ProjectorSaveCallback(Callback):
    def __init__(self, save_dir: str, every_n_steps: int = 100):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.every_n_steps = every_n_steps
        self.initial_saved = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.every_n_steps == 0 and trainer.global_step > 0:
            # Save only the projector
            torch.save(pl_module.model.encoder_projector.state_dict(), self.save_dir / "pytorch_model.bin")
            pl_module.log("saved_projector_step", float(trainer.global_step), prog_bar=True)

            # First-time full save
            if not self.initial_saved:
                self.initial_saved = True
                self.save_all_once(trainer, pl_module)

    def save_all_once(self, trainer, pl_module):
        logger.info("Saving full model assets on first checkpoint...")

        if hasattr(pl_module.tokenizer, "save_pretrained"):
            pl_module.tokenizer.save_pretrained(self.save_dir)

        pl_module.model_config.save(self.save_dir / "model_config.json")
        pl_module.train_config.save(self.save_dir / "train_config.json")

        training_state = {
            "epoch": trainer.current_epoch,
            "step": trainer.global_step
        }
        torch.save(training_state, self.save_dir / "training_states.pt")


class FinalSaveCallback(Callback):
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = Path(output_dir)

    def on_train_end(self, trainer, pl_module):
        logger.info("Saving final projector weights and training state only...")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save projector weights
        projector_path = self.output_dir / "pytorch_model.bin"
        torch.save(pl_module.model.encoder_projector.state_dict(), projector_path)

        # Save final training state
        training_state = {
            "epoch": trainer.current_epoch,
            "step": trainer.global_step
        }
        torch.save(training_state, self.output_dir / "training_states.pt")

        logger.info(f"Final save done at: {projector_path}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, default="LucAS_multitasks")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--save_every_n_steps", type=int, default=100)
    args = parser.parse_args()

    pl.seed_everything(1234)

    model_config = ModelConfig()
    train_config = TrainConfig()
    train_config.batch_size_training = args.batch_size
    train_config.num_epochs = args.epochs
    train_config.output_dir = args.output_dir
    train_config.use_fp16 = True
    train_config.run_validation = False

    os.makedirs(train_config.output_dir, exist_ok=True)
    model_config.save(os.path.join(train_config.output_dir, "model_config.json"))
    train_config.save(os.path.join(train_config.output_dir, "train_config.json"))

    model, tokenizer = model_factory(train_config, model_config, metric='acc')
    datamodule = SpeechDataModule(args.dataset_dirs, tokenizer, model_config, train_config)
    lightning_model = LightningLucAS(model, tokenizer, train_config, model_config)

    logger_tb = TensorBoardLogger(save_dir=train_config.output_dir, name="runs")

    callbacks = [
        ProjectorSaveCallback(
            save_dir=train_config.output_dir,
            every_n_steps=args.save_every_n_steps
        ),
        FinalSaveCallback(train_config.output_dir)
    ]
    if args.devices > 1:
        strategy = "ddp"
    else:
        strategy = "auto"
    trainer = pl.Trainer(
        logger=logger_tb,
        default_root_dir=train_config.output_dir,
        max_epochs=train_config.num_epochs,
        precision="16-mixed" if train_config.use_fp16 else 32,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=strategy,
        callbacks=callbacks,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        log_every_n_steps=50
    )

    trainer.fit(lightning_model, datamodule=datamodule)


if __name__ == '__main__':
    main()
