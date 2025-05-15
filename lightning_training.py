import os
import argparse
import logging
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
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
        if not loss.requires_grad:
            loss.requires_grad_(True)
        ppl = compute_ppl(loss)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_ppl', ppl, prog_bar=False, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs, acc = self.model(**batch)
        loss = outputs.loss
        ppl = compute_ppl(loss)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_ppl', ppl, prog_bar=False, on_step=False, on_epoch=True)
        return {'val_loss': loss, 'val_acc': acc}
        
    def on_validation_epoch_end(self):
        # Log the current validation metrics
        if self.trainer.is_global_zero:
            metrics = self.trainer.callback_metrics
            val_loss = metrics.get('val_loss', torch.tensor(0.0)).item()
            val_acc = metrics.get('val_acc', torch.tensor(0.0)).item()
            current_epoch = self.trainer.current_epoch
            current_global_step = self.trainer.global_step
            
            logger.info(f"Validation completed - Epoch: {current_epoch}, Step: {current_global_step}, "
                       f"Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

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

    def setup(self, stage=None):
        self.train_ds = get_dataset(
            dataset_dir=self.dataset_dirs,
            tokenizer=self.tokenizer,
            model_config=self.model_config,
            train_config=self.train_config,
            split='train'
        )
        if self.train_config.run_validation:
            self.val_ds = get_dataset(
                dataset_dir=self.dataset_dirs,
                tokenizer=self.tokenizer,
                model_config=self.model_config,
                train_config=self.train_config,
                split='test'
            )
        else:
            self.val_ds = None

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.train_config.batch_size_training,
            shuffle=True,
            num_workers=self.model_config.num_workers,
            collate_fn=self.train_ds.data_collator
        )

    def val_dataloader(self):
        if self.val_ds is None:
            return None
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.train_config.batch_size_training,
            shuffle=False,
            num_workers=1,
            collate_fn=self.val_ds.data_collator
        )

class DualCheckpointCallback(pl.Callback):
    """Saves both PEFT and full checkpoints, handles final model save"""
    def __init__(self, save_dir, peft_save_dir=None, merge_final_lora=True):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.peft_save_dir = Path(peft_save_dir) if peft_save_dir else self.save_dir/"peft"
        self.merge_final_lora = merge_final_lora
        self.is_peft = None  # Will detect automatically
        
        # Create directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.peft_save_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_end(self, trainer, pl_module):
        """Save both checkpoint types after validation"""
        if trainer.is_global_zero:
            self._save_current_checkpoint(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        """Final save after training completes"""
        if trainer.is_global_zero:
            logger.info("\n=== Final model saving ===")
            self._save_final_model(trainer, pl_module)

    def _save_current_checkpoint(self, trainer, pl_module):
        """Save both PEFT and full checkpoints"""
        # Detect PEFT status once
        if self.is_peft is None:
            self.is_peft = hasattr(pl_module.model.llm, "peft_config")

        # Save full checkpoint
        full_checkpoint_path = self.save_dir/f"checkpoint_{trainer.global_step}.pt"
        self._save_full_checkpoint(pl_module, full_checkpoint_path, trainer)
        
        # Save PEFT-specific components if using PEFT
        if self.is_peft:
            peft_checkpoint_path = self.peft_save_dir/f"peft_{trainer.global_step}"
            self._save_peft_checkpoint(pl_module, peft_checkpoint_path)

    def _save_full_checkpoint(self, pl_module, path, trainer):
        """Save full model state similar to your original function"""
        torch.save({
            "encoder": pl_module.model.encoder.state_dict(),
            "projector": pl_module.model.encoder_projector.state_dict(),
            "llm": pl_module.model.llm.state_dict(),
            "epoch": trainer.current_epoch,
            "step": trainer.global_step,
            "optimizer": pl_module.optimizers().state_dict(),
            "scheduler": pl_module.lr_schedulers()[0].state_dict(),
        }, path)
        logger.info(f"Full checkpoint saved to {path}")

    def _save_peft_checkpoint(self, pl_module, path):
        """Save PEFT-specific components"""
        if isinstance(pl_module.model.llm, PeftModel):
            pl_module.model.llm.save_pretrained(path)
            logger.info(f"PEFT adapter saved to {path}")
            
            # Save base model components
            base_path = path/"base_components.bin"
            torch.save({
                "encoder": pl_module.model.encoder.state_dict(),
                "projector": pl_module.model.encoder_projector.state_dict()
            }, base_path)

    def _save_final_model(self, trainer, pl_module):
        """Final model save with optional LoRA merging"""
        final_path = self.save_dir/"final_model"
        final_path.mkdir(exist_ok=True)
        
        if self.is_peft and self.merge_final_lora:
            logger.info("Merging LoRA weights into final model...")
            merged_model = pl_module.model.llm.merge_and_unload()
            
            # Save merged model
            torch.save({
                "encoder": pl_module.model.encoder.state_dict(),
                "projector": pl_module.model.encoder_projector.state_dict(),
                "llm": merged_model.state_dict()
            }, final_path/"merged_model.bin")
        else:
            # Save regular final model
            self._save_full_checkpoint(pl_module, final_path/"final_model.bin", trainer)
        
        logger.info(f"Final model saved to {final_path}")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Add PEFT status to checkpoint metadata"""
        return {
            "is_peft": self.is_peft,
            "merge_final_lora": self.merge_final_lora
        }
def load_checkpoint(path, model):
    checkpoint = torch.load(path)
    
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.projector.load_state_dict(checkpoint['projector'])
    
    if checkpoint.get('peft_config', None):
        # Load PEFT model
        model.llm = PeftModel.from_pretrained(
            model.llm, 
            path/"peft_adapter",
            is_trainable=True
        )
    else:
        model.llm.load_state_dict(checkpoint['llm'])
        
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LucAS Training Script")
    parser.add_argument("--dataset_dirs", nargs="+", required=True, help="List of dataset directories.")
    parser.add_argument("--output_dir", type=str, default="ckpts", help="Directory to store outputs and checkpoints.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to resume checkpoint.")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--precision", type=str, default="16", choices=["16", "32"], help="Floating point precision.")
    parser.add_argument("--eval_only", action="store_true", help="Only run validation without training.")
    parser.add_argument("--eval_step", type=int, default=1000, help="Run validation every N steps.")
    parser.add_argument("--save_after_eval", action="store_true", help="Save model after each validation.")
    parser.add_argument("--gradient_clip_val", type=float, default=None, help="Gradient clipping value.")
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator type (cpu, gpu, tpu, etc).")
    parser.add_argument("--devices", type=str, default="auto", help="Number of devices to use.")
    parser.add_argument("--strategy", type=str, default="auto", help="Training strategy.")
    parser.add_argument("--log_every_n_steps", type=int, default=50, help="Log every N steps.")

    args = parser.parse_args()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    pl.seed_everything(1234)

    model_config = ModelConfig()
    train_config = TrainConfig()

    train_config.batch_size_training = args.batch_size
    train_config.num_epochs = args.epochs
    train_config.output_dir = args.output_dir
    train_config.use_fp16 = args.precision == "16"
    train_config.validation_step = args.eval_step

    # Create model directories
    os.makedirs(train_config.output_dir, exist_ok=True)
    model_save_dir = os.path.join(train_config.output_dir, "model_checkpoints")
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Save configurations
    model_config.save(os.path.join(train_config.output_dir, "model_config.json"))
    train_config.save(os.path.join(train_config.output_dir, "train_config.json"))

    model, tokenizer = model_factory(train_config, model_config, metric='acc')

    if train_config.use_gradient_checkpointing:
        if hasattr(model, "encoder") and hasattr(model.encoder, "gradient_checkpointing_enable"):
            model.encoder.gradient_checkpointing_enable(dict(use_reentrant=False))
        if hasattr(model, "llm") and hasattr(model.llm, "gradient_checkpointing_enable"):
            model.llm.gradient_checkpointing_enable(dict(use_reentrant=False))

    datamodule = SpeechDataModule(
        dataset_dirs=args.dataset_dirs,
        tokenizer=tokenizer,
        model_config=model_config,
        train_config=train_config
    )

    # Setup callbacks
    callbacks = []
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=train_config.output_dir,
        filename='best-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=train_config.save_top_k_checkpoints,
        save_on_train_epoch_end=False,
        save_last=True,
        every_n_epochs=1
    )
    callbacks.append(checkpoint_callback)
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=train_config.patience,
        mode='min',
        check_finite=True
    )
    callbacks.append(early_stop)
    
    # Enhanced model save callback
    if args.save_after_eval:
        checkpoint_callback = DualCheckpointCallback(
            save_dir=model_save_dir,
            peft_save_dir=os.path.join(model_save_dir, "peft_checkpoints"),
            merge_final_lora=True  # Set False to keep LoRA separate
        )
        callbacks.append(checkpoint_callback)

    logger_tb = TensorBoardLogger(save_dir=train_config.output_dir, name="logs")

    # Calculate validation interval (steps or time-based)
    val_check_interval = args.eval_step if args.eval_step > 0 else 1.0
    
    trainer = pl.Trainer(
        logger=logger_tb,
        default_root_dir=train_config.output_dir,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        precision=args.precision,
        max_epochs=train_config.num_epochs,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=train_config.gradient_accumulation_steps,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=None if args.eval_step > 0 else 1,
        num_sanity_val_steps=0,  # Disable sanity validation to save time
        callbacks=callbacks,
        enable_model_summary=True,
        enable_progress_bar=True,
        enable_checkpointing=True,
        use_distributed_sampler=False,
        log_every_n_steps=args.log_every_n_steps
    )

    lightning_model = LightningLucAS(model, tokenizer, train_config, model_config)

    if args.checkpoint_path:
        logger.info(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        lightning_model.load_state_dict(checkpoint["state_dict"])

    if args.eval_only:
        logger.info("Running evaluation only...")
        trainer.validate(
            lightning_model, 
            datamodule=datamodule, 
            ckpt_path=args.checkpoint_path if args.checkpoint_path else None
        )
    else:
        logger.info("Starting training...")
        trainer.fit(
            lightning_model, 
            datamodule=datamodule,
            ckpt_path=args.checkpoint_path if args.checkpoint_path else None
        )
        logger.info(f"Best model saved at: {checkpoint_callback.best_model_path}")