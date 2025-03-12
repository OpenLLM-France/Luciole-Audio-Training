from dataclasses import asdict
import os
import json
import torch
import logging
import shutil
import psutil
from peft import PeftModel
from configs import ModelConfig, TrainConfig
# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_config(config_path):
    with open(config_path, "r") as file:
        config = json.load(file)
    return config

def check_frozen_layers_peft_model(model):
     for i, layer in enumerate(model.base_model.model.model.layers):
            for name, param in layer.named_parameters():
                logger.info(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")
                
def freeze_transformer_layers(model, num_layer):
   for i, layer in enumerate(model.model.layers):
            if i < num_layer:
                for param in layer.parameters():
                    param.requires_grad = False

def limit_checkpoints(checkpoint_dir, max_checkpoints=3):
    """
    Limits the number of checkpoint folders in the directory to `max_checkpoints`.
    Older checkpoints are deleted, keeping the most recent ones based on creation time.
    """
    # List and filter only directories, and get their paths and modification times
    checkpoint_dirs = [
        f for f in os.listdir(checkpoint_dir)
        if os.path.isdir(os.path.join(checkpoint_dir, f))
    ]
    
    # Sort directories by creation time (oldest first)
    checkpoint_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    print(checkpoint_dirs)
    # If the number of directories exceeds the max, delete the oldest ones
    excess_count = len(checkpoint_dirs) - max_checkpoints
    if excess_count > 0:
        print(f"Deleting {excess_count} old checkpoint directories...")
        for old_checkpoint in checkpoint_dirs[:excess_count]:
            old_checkpoint_path = os.path.join(checkpoint_dir, old_checkpoint)
            print(f"Deleting old checkpoint: {old_checkpoint_path}")
            shutil.rmtree(old_checkpoint_path)  # Efficiently remove directory and its contents
            print(f"Deleted checkpoint: {old_checkpoint_path}")

# save the optimizer, the scheduler and the scaler to load the model to finetune it
def save_optimizer_scheduler_scaler(optimizer, scheduler, scaler, save_dir):
    """
    Saves the optimizer, scheduler, and scaler states.
    """
    optimizer_path = os.path.join(save_dir, "optimizer.pt")
    scheduler_path = os.path.join(save_dir, "scheduler.pt")
    scaler_path = os.path.join(save_dir, "scaler.pt")

    torch.save({"optimizer_state_dict": optimizer.state_dict()}, optimizer_path)
    torch.save({"scheduler_state_dict": scheduler.state_dict()}, scheduler_path)
    torch.save({"scaler_state_dict": scaler.state_dict()}, scaler_path)

    logger.info(f"Optimizer, scheduler, and scaler states saved in {save_dir}")


def move_optimizer_state(optimizer, device):
    """
    Moves all tensors in the optimizer's state to the specified device.
    Also casts non-'step' tensors to float32 for consistency.
    """
    for param in optimizer.state:
        state = optimizer.state[param]
        for key, value in state.items():
            if torch.is_tensor(value):
                new_value = value.to(device)
                # For non-'step' keys, enforce float32 dtype for consistency.
                if key != 'step':
                    new_value = new_value.float()
                state[key] = new_value

def load_optimizer_scheduler_scaler(optimizer, scheduler, scaler, load_dir, device):
    """
    Loads the optimizer, scheduler, and scaler states if available,
    and moves optimizer state tensors to the specified device.
    """
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



# save the model Checkpoint               
def save_model_checkpoint_peft(model, train_config, epoch,step_count, checkpoint_name="checkpoint", save_trainable_only=True, merge_lora=False):
    """
    Saves a model checkpoint with options for saving only trainable parameters for specific components,
    and ensures that the PEFT configuration is saved correctly.

    Args:
        model (object): The model containing encoder, LLM, and projector components.
        optimizer (object): The optimizer used during training.
        scheduler (object): The learning rate scheduler.
        train_config (object): The training configuration containing output directory details.
        checkpoint_name (str): Subdirectory name for the checkpoint.
        save_trainable_only (bool): Whether to save only trainable parameters for each component.
        merge_lora (bool): Whether to merge LoRA adapters into the base model before saving.
    """
    logger.info("--> Saving PEFT model checkpoint ...")
    
    # Define the save directory and create it if it doesn't exist
    save_dir = os.path.join(train_config.output_dir, checkpoint_name)
    os.makedirs(save_dir, exist_ok=True)
    
    if merge_lora and isinstance(model.llm, PeftModel):
        model.llm == model.llm.merge_and_unload()
        logger.info("LoRA adapters merged into the base model.")
    
    # Prepare state dictionaries for each component
    def get_state_dict(component):
        full_state = component.state_dict()
        if save_trainable_only:
            return {name: param for name, param in component.named_parameters() if param.requires_grad}
        return full_state
    
    # Save encoder and projector state dictionaries
    checkpoint_path = os.path.join(save_dir, "model.pt")
   
    torch.save({
        "encoder_state": get_state_dict(model.encoder),
        "projector_state": get_state_dict(model.encoder_projector),
        "llm_state": get_state_dict(model.llm),
        "epoch": epoch,
        "step": step_count
    }, checkpoint_path)

    logger.info(f"Model state dictionaries saved at {checkpoint_path}")
    
    # Save PEFT adapter configuration
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

    # Reload configuration files
    model_config_file = os.path.join(output_dir, "model_config.json")
    model_config = ModelConfig(**load_config(model_config_file))
    train_config_file = os.path.join(output_dir, "train_config.json")
    train_config = TrainConfig(**load_config(train_config_file))

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
                # Wrap the base model with the PEFT adapter
                model.llm = PeftModel.from_pretrained(model.llm, adapter_save_dir, strict=False)
                logger.info(f"--> PEFT adapter configuration loaded from {adapter_save_dir}")
            except Exception as e:
                logger.warning(f"Error loading PEFT adapter: {e}")
        else:
            logger.warning(f"PEFT adapter configuration not found at {adapter_save_dir}.")

    logger.info("--> Model checkpoint load completed.")
    return model, train_config, model_config



def save_metrics_to_json(metrics, filepath="train_log.json"):
    """
    Save training and validation metrics to a JSON file.

    Args:
        metrics: Dictionary containing metrics to log.
        filepath: Path to the JSON file.
    """
    # Ensure the file exists or initialize it
    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            json.dump([], f)  # Create an empty JSON array

    # Convert tensors to native Python types
    def convert_tensors(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, (int, float)):
            return obj  # Leave native int or float untouched
        elif isinstance(obj, dict):
            return {k: convert_tensors(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tensors(i) for i in obj]
        return obj

    metrics = convert_tensors(metrics)  # Convert metrics recursively

    # Load existing logs and append new metrics
    with open(filepath, "r") as f:
        logs = json.load(f)
    
    logs.append(metrics)

    # Write updated logs back to the file
    with open(filepath, "w") as f:
        json.dump(logs, f, indent=4)

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 ** 2  # Memory in MB

def setup_directories(train_config):
    if not os.path.exists(train_config.output_dir):
        os.makedirs(train_config.output_dir, exist_ok=True)
    
    model_config_file = os.path.join(train_config.output_dir, "model_config.json")
    train_config_file = os.path.join(train_config.output_dir, "train_config.json")
    
    return model_config_file, train_config_file

def save_config_files(model_config, train_config, model_config_file, train_config_file):
    with open(model_config_file, "w") as json_file:
        json.dump(asdict(model_config), json_file, indent=4)
    
    with open(train_config_file, "w") as json_file:
        data = asdict(train_config)
        json.dump(data, json_file, indent=4)