import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import AdamW
from transformers import get_scheduler
from torch.utils.data import DistributedSampler

from data.audio_chuncks_dataset import get_dataset
from models.lucas_setup import model_factory
from configs import ModelConfig, TrainConfig
from peft import PeftConfig


# ---------------- Logging Setup ---------------- #
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

@pl.utilities.rank_zero_only
def log_info(message): logger.info(message)

@pl.utilities.rank_zero_only
def log_warning(message): logger.warning(message)


# ---------------- Utility Functions ---------------- #
def compute_ppl(loss):
    return torch.exp(loss)

def load_checkpoint_safely(checkpoint_path, map_location='cpu'):
    """Safely load checkpoint with proper error handling"""
    torch.serialization.add_safe_globals([PeftConfig, TrainConfig, ModelConfig])
    try:
        if 'weights_only' in torch.load.__code__.co_varnames:
            checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
        log_info(f"Successfully loaded checkpoint from {checkpoint_path}")
        return checkpoint
    except Exception as e:
        log_warning(f"Failed to load checkpoint {checkpoint_path}: {e}")
        raise


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint in a directory"""
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    # Look for final checkpoint first
    final_ckpt = checkpoint_dir / "pytorch_model_final.ckpt"
    if final_ckpt.exists():
        log_info(f"Found final checkpoint: {final_ckpt}")
        return str(final_ckpt)
    
    # Find latest step checkpoint
    step_ckpts = list(checkpoint_dir.glob("pytorch_model_step*.ckpt"))
    if step_ckpts:
        latest_ckpt = max(step_ckpts, key=lambda p: int(p.stem.split("step")[-1]))
        log_info(f"Found latest step checkpoint: {latest_ckpt}")
        return str(latest_ckpt)
    
    return None


# ---------------- Lightning Module ---------------- #
class LightningLucAS(pl.LightningModule):
    def __init__(self, model, tokenizer, train_config: TrainConfig, model_config: ModelConfig):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.train_config = train_config
        self.model_config = model_config
        self.save_hyperparameters(ignore=['model', 'tokenizer'])

        # Checkpoint restoration state
        self.restored_epoch = 0
        self.restored_global_step = 0
        self.restored_total_batch_idx = 0
        self.checkpoint_loaded = False

        if getattr(self.train_config, 'train_projector_only', False):
            self.model.train_projector_only()

        self.model_train_params = (
            self.model.encoder_projector.parameters()
            if getattr(self.train_config, 'train_projector_only', False)
            else self.model.parameters()
        )

    def on_fit_start(self):
        """Called when fit starts - after optimizers are configured"""
        if self.trainer.is_global_zero:
            if hasattr(self.model, "get_trainable_parameters"):
                for comp, count in self.model.get_trainable_parameters().items():
                    log_info(f"  {comp}: {count:,}")
            else:
                log_info(f"Total trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, _):
        outputs, acc = self.model(**batch)
        loss = outputs.loss
        self.log_dict({
            'train_loss': loss,
            'train_acc': acc,
            'train_ppl': compute_ppl(loss),
        }, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model_train_params, 
            lr=self.train_config.learning_rate, 
            weight_decay=self.train_config.weight_decay
        )
        
        # # Calculate total steps
        num_warmup_steps = getattr(self.train_config, "warmup_steps", 500)
        num_training_steps = getattr(self.train_config, "max_steps", 11000)

        scheduler = get_scheduler(
            name=getattr(self.train_config, "lr_scheduler_type", "linear"),  # "linear", "cosine", etc.
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1, 'monitor': 'train_loss'}]

    def load_checkpoint_state(self, checkpoint_path: str):
        """Load checkpoint state into the model"""
        checkpoint = load_checkpoint_safely(checkpoint_path)
        
        # Load model state
        if "model_state_dict" in checkpoint:
            model_state = checkpoint["model_state_dict"]
            
            # Handle different model structures
            if hasattr(self.model, 'encoder_projector'):
                try:
                    self.model.encoder_projector.load_state_dict(model_state, strict=False)
                    log_info("Loaded encoder_projector state")
                except Exception as e:
                    log_warning(f"Failed to load encoder_projector state: {e}")
            else:
                try:
                    self.model.load_state_dict(model_state, strict=False)
                    log_info("Loaded full model state")
                except Exception as e:
                    log_warning(f"Failed to load model state: {e}")
        
        # Store training info for later restoration
        training_info = checkpoint.get("training_info", {})
        self.restored_epoch = training_info.get("epoch", 0)
        self.restored_global_step = training_info.get("global_step", 0)
        self.restored_total_batch_idx = training_info.get("total_batch_idx", 0)
        self.checkpoint_loaded = True
        
        log_info(f"Checkpoint loaded - will resume from epoch {self.restored_epoch}, step {self.restored_global_step}")
        
        return checkpoint

    def load_state_dict(self, state_dict, strict=True):
        """Override to handle state dict loading gracefully"""
        try:
            super().load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            if "Unexpected key(s)" in str(e) or "Missing key(s)" in str(e):
                log_warning(f"State dict mismatch: {e}")
                log_warning("Loading with strict=False")
                super().load_state_dict(state_dict, strict=False)
            else:
                raise


# ---------------- DataModule ---------------- #
class SpeechDataModule(pl.LightningDataModule):
    def __init__(self, dataset_dirs, tokenizer, model_config, train_config):
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
            split='train',
        )

    def train_dataloader(self):
        # sampler = DistributedSampler(self.train_ds) if self.trainer.num_devices > 1 else None
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.train_config.batch_size_training,
            shuffle=False,#(sampler is None),
            # sampler=sampler,
            num_workers=self.model_config.num_workers,
            collate_fn=self.train_ds.data_collator,
            persistent_workers=False,
            pin_memory=self.train_config.use_fp16
        )

    def val_dataloader(self):
        return []


# ---------------- Checkpoint Restoration Callback ---------------- #
class CheckpointRestorationCallback(Callback):
    """Callback to restore optimizer and scheduler states after they're initialized"""
    
    def __init__(self, checkpoint_data: Dict[str, Any]):
        self.checkpoint_data = checkpoint_data
        self.restored = False

    def on_fit_start(self, trainer, pl_module):
        """Restore optimizer, scheduler, and progress tracking states"""
        if self.restored:
            return
        
        # Restore optimizer states
        optimizer_states = self.checkpoint_data.get("optimizer_states", [])
        if optimizer_states and len(trainer.optimizers) == len(optimizer_states):
            for opt, state in zip(trainer.optimizers, optimizer_states):
                try:
                    opt.load_state_dict(state)
                    log_info("Restored optimizer state")
                except Exception as e:
                    log_warning(f"Failed to restore optimizer state: {e}")
        
        # Restore scheduler states
        lr_scheduler_states = self.checkpoint_data.get("lr_scheduler_states", [])
        lr_scheduler_configs = getattr(trainer, 'lr_scheduler_configs', [])

        if lr_scheduler_states and len(lr_scheduler_configs) == len(lr_scheduler_states):
            for sch_cfg, state in zip(lr_scheduler_configs, lr_scheduler_states):
                scheduler = getattr(sch_cfg, 'scheduler', None)
                if scheduler is not None:
                    try:
                        scheduler.load_state_dict(state)
                        log_info("Restored scheduler state")
                    except Exception as e:
                        log_warning(f"Failed to restore scheduler state: {e}")
        
        # Restore training progress
        training_info = self.checkpoint_data.get("training_info", {})
        if training_info:
            restored_epoch = training_info.get("epoch", 0)
            restored_step = training_info.get("global_step", 0)
            restored_total_batch_idx = training_info.get("total_batch_idx", 0)
            restored_batch_idx = training_info.get("batch_idx", 0)
            
            fit_loop = trainer.fit_loop
            fit_loop.epoch_progress.current.completed = restored_epoch
            fit_loop.epoch_progress.current.processed = restored_epoch

            fit_loop.epoch_loop.manual_optimization.optim_step_progress.total.completed =(restored_step)
            fit_loop.epoch_loop.automatic_optimization.optim_progress.optimizer.step.total.completed = (restored_step)
           
            log_info(f"Training progress restored: epoch={restored_epoch}, global_step={restored_step}, total_batch_idx={restored_batch_idx}")
        
        self.restored = True
    

# ---------------- Checkpoint Saving Callback ---------------- #
class CombinedCheckpointSaveCallback(Callback):
    def __init__(self, save_dir: str, every_n_steps: int = 100, save_full_model: bool = False):
        super().__init__()
        self.save_dir = Path(save_dir).resolve()
        self.every_n_steps = every_n_steps
        self.save_full_model = save_full_model
        self.initial_saved = False
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.is_global_zero and trainer.global_step > 0 and trainer.global_step % self.every_n_steps == 0:
            self._save_combined_checkpoint(trainer, pl_module, step=trainer.global_step)
            pl_module.log("saved_checkpoint_step", float(trainer.global_step), prog_bar=True)

            if not self.initial_saved:
                self.initial_saved = True
                self._save_model_assets_once(pl_module)

    def on_train_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            log_info("Saving final combined checkpoint and training state...")
            self._save_combined_checkpoint(trainer, pl_module, final=True)
            log_info("Final combined checkpoint saved.")
    
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            log_info(f"Epoch {trainer.current_epoch} ended. Saving checkpoint...")
            self._save_combined_checkpoint(
                trainer,
                pl_module,
                step=trainer.global_step,
                final=False
            )

    def _save_combined_checkpoint(self, trainer, pl_module, step: int = None, final: bool = False, keep_only=2):
        model_state = (
            pl_module.model.state_dict() if self.save_full_model
            else pl_module.model.encoder_projector.state_dict()
        )

        # Move optimizer state dicts to CPU
        optimizer_states = [
            {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in opt.state_dict().items()}
            for opt in trainer.optimizers
        ]

        # Move scheduler state dicts to CPU
        lr_scheduler_states = []
        for sch_cfg in getattr(trainer, 'lr_scheduler_configs', []):
            scheduler = getattr(sch_cfg, 'scheduler', None)
            if scheduler is not None:
                state = scheduler.state_dict()
                state_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in state.items()}
                lr_scheduler_states.append(state_cpu)

        training_info = {
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "total_batch_idx": trainer.fit_loop.total_batch_idx,
            "batch_idx": trainer.fit_loop.batch_idx,
            "finished": final
        }

        combined_checkpoint = {
            "model_state_dict": model_state,
            "optimizer_states": optimizer_states,
            "lr_scheduler_states": lr_scheduler_states,
            "training_info": training_info
        }

        filename = "pytorch_model_final.ckpt" if final else f"pytorch_model_step{step}.ckpt"
        save_path = self.save_dir / filename

        try:
            torch.save(combined_checkpoint, save_path)
            log_info(f"Checkpoint saved at: {save_path}")
        except Exception as e:
            log_warning(f"Failed to save checkpoint at {save_path}: {e}")
            return

        if not final:
            all_ckpts = sorted(
                self.save_dir.glob("pytorch_model_step*.ckpt"),
                key=lambda p: int(p.stem.split("step")[-1]),
                reverse=True
            )
            for old_ckpt in all_ckpts[keep_only:]:
                try:
                    old_ckpt.unlink()
                    log_info(f"Deleted old checkpoint: {old_ckpt}")
                except Exception as e:
                    log_warning(f"Failed to delete {old_ckpt}: {e}")

    def _save_model_assets_once(self, pl_module):
        # Save tokenizer if possible
        if hasattr(pl_module.tokenizer, "save_pretrained"):
            try:
                pl_module.tokenizer.save_pretrained(self.save_dir)
                log_info("Tokenizer saved.")
            except Exception as e:
                log_warning(f"Failed to save tokenizer: {e}")

        # Save model and training config files
        try:
            pl_module.model_config.save(self.save_dir / "model_config.json")
            pl_module.train_config.save(self.save_dir / "train_config.json")
            log_info("Model and training configs saved.")
        except Exception as e:
            log_warning(f"Failed to save config files: {e}")


class EmptyCacheCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log_info("Emptied CUDA cache after epoch.")


# ---------------- Main Function ---------------- #
def main():
    torch.serialization.add_safe_globals([PeftConfig, TrainConfig, ModelConfig])

    parser = argparse.ArgumentParser(description="Train LucAS model with checkpoint resumption")
    parser.add_argument("--dataset_dirs", nargs="+", required=True, help="Dataset directories")
    parser.add_argument("--output_dir", type=str, default="LucAS_multitasks", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--use_quantization", action='store_true', help="Use quantization")
    parser.add_argument("--accelerator", type=str, default="gpu", help="Accelerator type")
    parser.add_argument("--strategy", type=str, default="auto", help="Training strategy")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint directory or specific checkpoint file")
    parser.add_argument("--precision", type=str, default="16-mixed", choices=["32", "16-mixed", "bf16-mixed"], help="Training precision")
    parser.add_argument("--save_every_n_steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--auto_resume", action='store_true', help="Automatically resume from latest checkpoint in output_dir")
    
    args = parser.parse_args()

    # Set up reproducibility
    pl.seed_everything(1234)
    torch.set_float32_matmul_precision('high')

    # Determine checkpoint to resume from
    resume_checkpoint = None
    if args.auto_resume:
        resume_checkpoint = find_latest_checkpoint(args.output_dir)
        if resume_checkpoint:
            log_info(f"Auto-resuming from: {resume_checkpoint}")
    elif args.resume_from:
        if Path(args.resume_from).is_file():
            resume_checkpoint = args.resume_from
        else:
            resume_checkpoint = find_latest_checkpoint(args.resume_from)
        
        if resume_checkpoint:
            log_info(f"Resuming from: {resume_checkpoint}")
        else:
            log_warning(f"No checkpoint found at: {args.resume_from}")

    # Load or create configs
    config_dir = Path(args.output_dir)
    if resume_checkpoint:
        # Try to load configs from checkpoint directory
        checkpoint_dir = Path(resume_checkpoint).parent
        model_config_path = checkpoint_dir / "model_config.json"
        train_config_path = checkpoint_dir / "train_config.json"
        
        if model_config_path.exists() and train_config_path.exists():
            model_config = ModelConfig.load(model_config_path)
            train_config = TrainConfig.load(train_config_path)
            log_info("Loaded configs from checkpoint directory")
        else:
            model_config = ModelConfig()
            train_config = TrainConfig()
            log_info("Using default configs (checkpoint configs not found)")
    else:
        model_config = ModelConfig()
        train_config = TrainConfig()
        log_info("Using default configs (no checkpoint to resume)")

    # Update configs with command line arguments
    train_config.batch_size_training = args.batch_size
    train_config.num_epochs = args.epochs
    train_config.output_dir = args.output_dir
    train_config.learning_rate = args.learning_rate
    train_config.use_fp16 = args.precision in ["16-mixed", "bf16-mixed"]
    train_config.quantization = args.use_quantization
    model_config.num_workers = args.num_workers

    # Create output directory and save configs
    config_dir.mkdir(parents=True, exist_ok=True)
    model_config.save(config_dir / "model_config.json")
    train_config.save(config_dir / "train_config.json")

    # Create model and lightning module
    rank = getattr(pl.utilities.rank_zero_only, 'rank', 0)
    model, tokenizer = model_factory(train_config, model_config, rank=rank, metric='acc')
    lightning_model = LightningLucAS(model, tokenizer, train_config, model_config)
    
    # Set up callbacks
    callbacks = [
        CombinedCheckpointSaveCallback(
            args.output_dir, 
            args.save_every_n_steps, 
            save_full_model=False
        ),
        EmptyCacheCallback()
    ]
    
    # Load checkpoint if resuming
    checkpoint_data = None
    if resume_checkpoint:
        try:
            checkpoint_data = lightning_model.load_checkpoint_state(resume_checkpoint)
            log_info("Checkpoint state loaded successfully")
            callbacks.append(CheckpointRestorationCallback(checkpoint_data))

        except Exception as e:
            log_warning(f"Failed to load checkpoint: {e}")
            log_info("Starting training from scratch")
            checkpoint_data = None
    if checkpoint_data:
        log_info(f"Resuming from epoch={lightning_model.restored_epoch}, step={lightning_model.restored_global_step}")
    
    # Set up logger
    logger_tb = TensorBoardLogger(save_dir=train_config.output_dir, name="runs")

    # Create trainer
    trainer = pl.Trainer(
        logger=logger_tb,
        default_root_dir=train_config.output_dir,
        max_epochs=train_config.num_epochs,
        precision=args.precision,
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy="ddp" if args.devices > 1 else args.strategy,
        callbacks=callbacks,
        enable_checkpointing=False,  # We handle checkpointing ourselves
        num_sanity_val_steps=0,
        log_every_n_steps=50,
    )

    # Create datamodule
    datamodule = SpeechDataModule(args.dataset_dirs, tokenizer, model_config, train_config)

    # Start training
    if resume_checkpoint:
        log_info(f"Resuming training from checkpoint: {resume_checkpoint}")
    else:
        log_info("Starting training from scratch")
    
    log_info(f"Training with config: {trainer.fit_loop.epoch_progress}")
    
    trainer.fit(lightning_model, datamodule=datamodule)
    log_info("Training completed!")


if __name__ == '__main__':
    main()