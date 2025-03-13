import os
import shutil
import torch
from peft import PeftModel

from utils.config_utils import load_config 

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

def save_optimizer_scheduler_scaler(optimizer, scheduler, scaler, save_dir):
    optimizer_path = os.path.join(save_dir, "optimizer.pt")
    scheduler_path = os.path.join(save_dir, "scheduler.pt")
    scaler_path = os.path.join(save_dir, "scaler.pt")

    torch.save({"optimizer_state_dict": optimizer.state_dict()}, optimizer_path)
    torch.save({"scheduler_state_dict": scheduler.state_dict()}, scheduler_path)
    torch.save({"scaler_state_dict": scaler.state_dict()}, scaler_path)

    logger.info(f"Optimizer, scheduler, and scaler states saved in {save_dir}")

def move_optimizer_state(optimizer, device):
    for param in optimizer.state:
        state = optimizer.state[param]
        for key, value in state.items():
            if torch.is_tensor(value):
                new_value = value.to(device)
                if key != 'step':
                    new_value = new_value.float()
                state[key] = new_value

def load_optimizer_scheduler_scaler(optimizer, scheduler, scaler, load_dir, device):
    optimizer_path = os.path.join(load_dir, "optimizer.pt")
    scheduler_path = os.path.join(load_dir, "scheduler.pt")
    scaler_path = os.path.join(load_dir, "scaler.pt")

    if optimizer and os.path.exists(optimizer_path):
        try:
            optimizer_state = torch.load(optimizer_path, map_location="cpu")
            optimizer.load_state_dict(optimizer_state["optimizer_state_dict"])
            move_optimizer_state(optimizer, device)
        except Exception as e:
            logger.warning(f"Error loading optimizer state: {e}")

    if scheduler and os.path.exists(scheduler_path):
        try:
            scheduler_state = torch.load(scheduler_path, map_location="cpu")
            scheduler.load_state_dict(scheduler_state["scheduler_state_dict"])
        except Exception as e:
            logger.warning(f"Error loading scheduler state: {e}")

    if scaler and os.path.exists(scaler_path):
        try:
            scaler_state = torch.load(scaler_path, map_location="cpu")
            scaler.load_state_dict(scaler_state["scaler_state_dict"])
        except Exception as e:
            logger.warning(f"Error loading scaler state: {e}")

    logger.info(f"Optimizer, scheduler, and scaler states loaded from {load_dir}")
    return optimizer, scheduler, scaler

def save_model_checkpoint_peft(model, train_config, epoch, step_count, checkpoint_name="checkpoint", save_trainable_only=True, merge_lora=False):
    logger.info("--> Saving PEFT model checkpoint ...")
    
    save_dir = os.path.join(train_config.output_dir, checkpoint_name)
    os.makedirs(save_dir, exist_ok=True)
    
    if merge_lora and isinstance(model.llm, PeftModel):
        model.llm = model.llm.merge_and_unload()
        logger.info("LoRA adapters merged into the base model.")
    
    def get_state_dict(component):
        full_state = component.state_dict()
        if save_trainable_only:
            return {name: param for name, param in component.named_parameters() if param.requires_grad}
        return full_state

    checkpoint_path = os.path.join(save_dir, "model.pt")
    torch.save({
        "encoder_state": get_state_dict(model.encoder),
        "projector_state": get_state_dict(model.encoder_projector),
        "llm_state": get_state_dict(model.llm),
        "epoch": epoch,
        "step": step_count
    }, checkpoint_path)

    logger.info(f"Model state dictionaries saved at {checkpoint_path}")
    
    if isinstance(model.llm, PeftModel):
        adapter_save_dir = os.path.join(save_dir, "peft_adapter")
        os.makedirs(adapter_save_dir, exist_ok=True)
        model.llm.save_pretrained(adapter_save_dir)
        logger.info(f"PEFT adapter configuration saved at {adapter_save_dir}")
    else:
        logger.warning("Model.llm is not a PeftModel; skipping adapter configuration save.")
    
    limit_checkpoints(train_config.output_dir, max_checkpoints=3)
    logger.info("--> PEFT model checkpoint save completed.")

def load_model_checkpoint_peft(model, train_config, model_config, checkpoint_name="checkpoint"):
    logger.info("--> Loading PEFT model checkpoint ...")
    load_dir = os.path.join(train_config.output_dir, checkpoint_name)
    output_dir = train_config.output_dir

    model_config_file = os.path.join(output_dir, "model_config.json")
    model_config = type(model_config)(**load_config(model_config_file))
    train_config_file = os.path.join(output_dir, "train_config.json")
    train_config = type(train_config)(**load_config(train_config_file))

    checkpoint_path = os.path.join(load_dir, "model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
        model.encoder.load_state_dict(checkpoint["encoder_state"], strict=False)
        model.llm.load_state_dict(checkpoint["llm_state"], strict=False)
        model.encoder_projector.load_state_dict(checkpoint["projector_state"], strict=False)
        logger.info(f"Model state dictionaries loaded from {checkpoint_path}")
    except Exception as e:
        logger.error(f"Error loading model checkpoint: {e}")
        raise 

    if train_config.use_peft:
        adapter_save_dir = os.path.join(load_dir, "peft_adapter")
        if os.path.exists(adapter_save_dir):
            try:
                from peft import PeftModel
                model.llm = PeftModel.from_pretrained(model.llm, adapter_save_dir, strict=False)
                logger.info(f"--> PEFT adapter configuration loaded from {adapter_save_dir}")
            except Exception as e:
                logger.warning(f"Error loading PEFT adapter: {e}")
        else:
            logger.warning(f"PEFT adapter configuration not found at {adapter_save_dir}.")

    logger.info("--> Model checkpoint load completed.")
    return model, train_config, model_config
