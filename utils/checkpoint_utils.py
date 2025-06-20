import os
import shutil
import torch
from peft import PeftModel

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def limit_checkpoints(checkpoint_dir, max_checkpoints=3):
    """
    Limits the number of checkpoint folders to `max_checkpoints`.
    Deletes older checkpoints based on modification time.
    """
    checkpoint_dirs = [
        f for f in os.listdir(checkpoint_dir)
        if os.path.isdir(os.path.join(checkpoint_dir, f))
    ]
    checkpoint_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    excess_count = len(checkpoint_dirs) - max_checkpoints
    if excess_count > 0:
        logger.info(f"Deleting {excess_count} old checkpoint directories...")
        for old_checkpoint in checkpoint_dirs[:excess_count]:
            old_checkpoint_path = os.path.join(checkpoint_dir, old_checkpoint)
            logger.info(f"Deleting old checkpoint: {old_checkpoint_path}")
            shutil.rmtree(old_checkpoint_path)
            logger.info(f"Deleted checkpoint: {old_checkpoint_path}")

def save_model_checkpoint_peft(model, tokenizer, optimizer, scheduler, scaler, train_config, epoch, step_count, merge_lora=False):
    logger.info("--> Saving model checkpoint ...")
    save_dir = train_config.output_dir
    os.makedirs(save_dir, exist_ok=True)

    adapter_merged = False
    if merge_lora and isinstance(model.llm, PeftModel):
        model.llm = model.llm.merge_and_unload()
        logger.info("LoRA adapters merged into the base model.")
        adapter_merged = True

    # Save model weights
    checkpoint_path = os.path.join(save_dir, "pytorch_model.bin")
    torch.save({
        "encoder": model.encoder.state_dict(),
        "projector": model.encoder_projector.state_dict(),
        "llm": model.llm.state_dict(),
        "epoch": epoch,
        "step": step_count
    }, checkpoint_path)

    # Save optimizer/scheduler states
    training_state_path = os.path.join(save_dir, "training_states.pt")
    torch.save({
        'optimizer': optimizer.state_dict() if optimizer else None,
        'scheduler': scheduler.state_dict() if scheduler else None,
        'scaler': scaler.state_dict() if scaler else None
    }, training_state_path)

    # save tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(save_dir)
        logger.info(f"Tokenizer and vocab saved to {save_dir}")
    
    # save LLM Configs
    if hasattr(model.llm, "config"):
        model.llm.config.save_pretrained(save_dir)
        logger.info(f"Model config saved to {save_dir}")
        
    logger.info(f"Model weights saved to {checkpoint_path}")
    logger.info(f"Training states saved to {training_state_path}")

    # Save PEFT adapter config only if not merged
    if isinstance(model.llm, PeftModel) and not adapter_merged:
        adapter_save_dir = os.path.join(save_dir, "peft_adapter")
        os.makedirs(adapter_save_dir, exist_ok=True)
        model.llm.save_pretrained(adapter_save_dir, save_embedding_layers=True)
        logger.info(f"PEFT adapter saved to {adapter_save_dir}")
    elif adapter_merged:
        logger.info("LoRA was merged, skipping PEFT adapter save.")
    else:
        logger.warning("model.llm is not a PeftModel; no adapter to save.")

    logger.info("--> PEFT model checkpoint save completed.")


def load_optimizer_scheduler_scaler(optimizer, scheduler, scaler, load_dir, device):
    
    training_states_path = os.path.join(load_dir, "training_states.pt")
    
    if os.path.exists(training_states_path):
        try:
            training_state = torch.load(training_states_path, map_location="cpu")
            optimizer.load_state_dict(training_state["optimizer"])
            scheduler.load_state_dict(training_state["scheduler"])
            scaler.load_state_dict(training_state["scaler"])
            # move_optimizer_state(optimizer, device)
        except Exception as e:
            logger.warning(f"Error loading training states ['Optimizer', 'scheduler', 'scaler']: {e}")
    else:
        logger.error('Path not Found')
        
    logger.info(f"Optimizer, scheduler, and scaler states loaded from {load_dir}")
    return optimizer, scheduler, scaler

def load_model_checkpoint_peft(model, load_dir):
    
    from utils.config_utils import load_all_configs

    logger.info(f"--> Loading model checkpoint from {load_dir} ...")
    model_config_file = os.path.join(load_dir, "model_config.json")
    train_config_file = os.path.join(load_dir, "train_config.json")

    if os.path.exists(model_config_file) and os.path.exists(train_config_file):
        model_config, train_config = load_all_configs(model_config_file, train_config_file)
    else:
        raise FileNotFoundError(f"No model_config.json or train_config.json found in {load_dir}")
        
    checkpoint_path = os.path.join(load_dir, "pytorch_model.bin")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.encoder.load_state_dict(checkpoint["encoder"], strict=False)
        model.encoder_projector.load_state_dict(checkpoint["projector"], strict=False)
        
        ckpt_vocab_size = checkpoint["llm"]["model.embed_tokens.weight"].shape[0]
        model.llm.resize_token_embeddings(ckpt_vocab_size)
        model.llm.load_state_dict(checkpoint["llm"], strict=False)
        
        epoch = checkpoint.get("epoch", 0)
        step = checkpoint.get("step", 0)
        logger.info(f"Loaded model weights from {checkpoint_path}")
    except Exception as e:
        raise RuntimeError(
            f"Error loading model checkpoint from {checkpoint_path}: {str(e)}"
        ) from e

    # Load PEFT adapter if available
    adapter_save_dir = os.path.join(load_dir, "peft_adapter")
    if train_config.use_peft and os.path.exists(adapter_save_dir):
        try:
            logger.info(f"Attempting to load PEFT adapter from {adapter_save_dir}")
            model.llm = PeftModel.from_pretrained(model.llm, adapter_save_dir, is_trainable=True)
            logger.info(f"PEFT adapter loaded from {adapter_save_dir}")
        except Exception as e:
            logger.warning(f"Failed to load PEFT adapter: {e}")
    else:
        if not train_config.use_peft:
            logger.info("train_config.use_peft is False; skipping PEFT adapter load.")
        else:
            logger.warning(f"No PEFT adapter found at {adapter_save_dir}; model may have been merged.")

    logger.info("--> Model checkpoint load completed.")
    return model, epoch, step, train_config, model_config
