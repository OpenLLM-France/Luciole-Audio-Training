import os
import shutil
import pickle
import logging
import torch
from peft import PeftModel, PeftConfig
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add safe globals at module level
torch.serialization.add_safe_globals([PeftConfig])

def safe_torch_load(path: str) -> Dict[str, Any]:
    """Safe checkpoint loading with fallback mechanism"""
    try:
        # First try with weights_only=True (secure)
        return torch.load(path, map_location='cpu', weights_only=True)
    except (pickle.UnpicklingError, RuntimeError) as e:
        logger.warning(f"Secure loading failed ({str(e)}), falling back to unsafe mode")
        return torch.load(path, map_location='cpu', weights_only=False)


def save_model_checkpoint_peft(
    model: torch.nn.Module,
    tokenizer: Any,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    scaler: Optional[torch.cuda.amp.GradScaler],
    train_config: Any,
    epoch: int,
    step_count: int,
    merge_lora: bool = False
) -> None:
    """Save model checkpoint with PEFT adapter handling"""
    try:
        logger.info("--> Saving model checkpoint ...")
        save_dir = train_config.output_dir
        os.makedirs(save_dir, exist_ok=True)

        # Handle LoRA merging if requested
        adapter_merged = False
        if merge_lora and isinstance(model.llm, PeftModel):
            with torch.no_grad():
                model.llm = model.llm.merge_and_unload()
            logger.info("LoRA adapters merged into base model")
            adapter_merged = True

        # Save model components
        checkpoint = {
            "encoder": model.encoder.state_dict(),
            "projector": model.encoder_projector.state_dict(),
            "llm": model.llm.state_dict(),
            "epoch": epoch,
            "step": step_count,
            "config": {
                "model_config": getattr(model, "config", {}),
                "train_config": train_config.__dict__
            }
        }
        
        checkpoint_path = os.path.join(save_dir, "pytorch_model.bin")
        torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=True)

        # Save training states
        training_state = {
            'optimizer': optimizer.state_dict() if optimizer else None,
            'scheduler': scheduler.state_dict() if scheduler else None,
            'scaler': scaler.state_dict() if scaler else None
        }
        torch.save(training_state, os.path.join(save_dir, "training_states.pt"))

        # Save tokenizer and configs
        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)
        
        if hasattr(model.llm, "config"):
            model.llm.config.save_pretrained(save_dir)

        # Save PEFT adapter if not merged
        if isinstance(model.llm, PeftModel) and not adapter_merged:
            adapter_dir = os.path.join(save_dir, "peft_adapter")
            model.llm.save_pretrained(adapter_dir, safe_serialization=True)
            logger.info(f"PEFT adapter saved to {adapter_dir}")

        logger.info(f"Checkpoint saved to {save_dir}")
    except Exception as e:
        logger.error(f"Error saving checkpoint: {str(e)}")
        raise

def load_optimizer_scheduler_scaler(
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    scaler: Optional[torch.cuda.amp.GradScaler],
    load_dir: str,
    device: torch.device
) -> Tuple[Optional[torch.optim.Optimizer], Optional[Any], Optional[torch.cuda.amp.GradScaler]]:
    """Load training states safely"""
    training_states_path = os.path.join(load_dir, "training_states.pt")
    if not os.path.exists(training_states_path):
        logger.error(f"Training states not found at {training_states_path}")
        return optimizer, scheduler, scaler

    try:
        training_state = safe_torch_load(training_states_path)
        
        if optimizer is not None and "optimizer" in training_state:
            optimizer.load_state_dict(training_state["optimizer"])
        
        if scheduler is not None and "scheduler" in training_state:
            scheduler.load_state_dict(training_state["scheduler"])
        
        if scaler is not None and "scaler" in training_state:
            scaler.load_state_dict(training_state["scaler"])
        
        logger.info(f"Training states loaded from {training_states_path}")
        return optimizer, scheduler, scaler
    except Exception as e:
        logger.error(f"Error loading training states: {str(e)}")
        return optimizer, scheduler, scaler

def load_model_checkpoint_peft(
    model: torch.nn.Module,
    load_dir: str
) -> Tuple[torch.nn.Module, int, int, Any, Any]:
    """Load model checkpoint with PEFT adapter handling"""
    from adapter_training.utils.config_utils import load_all_configs  # Local import to avoid circular dependencies

    try:
        logger.info(f"--> Loading model checkpoint from {load_dir}")
        
        # Load configs
        model_config, train_config = load_all_configs(
            os.path.join(load_dir, "model_config.json"),
            os.path.join(load_dir, "train_config.json")
        )

        # Load model weights
        checkpoint_path = os.path.join(load_dir, "pytorch_model.bin")
        checkpoint = safe_torch_load(checkpoint_path)
        
        # Load model components
        model.encoder.load_state_dict(checkpoint["encoder"], strict=False)
        model.encoder_projector.load_state_dict(checkpoint["projector"], strict=False)
        
        ckpt_vocab_size = checkpoint["llm"]["model.embed_tokens.weight"].shape[0]
        model.llm.resize_token_embeddings(ckpt_vocab_size)
        model.llm.load_state_dict(checkpoint["llm"], strict=False)
        
        epoch = checkpoint.get("epoch", 0)
        step = checkpoint.get("step", 0)
        logger.info(f"Loaded model weights from {checkpoint_path}")
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        raise RuntimeError(f"Checkpoint loading failed: {str(e)}") from e