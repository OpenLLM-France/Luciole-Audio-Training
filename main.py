import os
import json
from typing import Dict, Any

from configs import ModelConfig, TrainConfig
from train import train
from data.audio_chat_to_dataset import get_dataset

from utils.checkpoint_utils import load_model_checkpoint_peft, load_optimizer_scheduler_scaler
from utils.config_utils import setup_directories, save_config_files
from utils.model_utils import freeze_transformer_layers, check_frozen_layers_peft_model

from models.lucas_setup import model_factory
from utils.env import auto_device

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler

import argparse

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

torch.manual_seed(1234)  # Ensure reproducibility
def setup_logging(output_dir):
    """Initialize logging to file and console."""
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] - %(message)s')
    file_handler = logging.FileHandler(os.path.join(output_dir, "log.txt"), mode='w')
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
def parse_args():
    parser = argparse.ArgumentParser(description="Configure data source paths.")
    parser.add_argument(
        "--dataset_dirs",
        nargs="+",
        required=True,
        help="Paths for training and validation directories (these dirs should contain train/test folders with *.parquet files)."
    )
    parser.add_argument(
        "--gpus",
        type=int,
        help="Whether to use a GPU (specify GPU index). Default is CPU if not set."
    )
    return parser.parse_args()

def load_data(dataset_dirs, model_config, train_config, tokenizer, is_validation=False):
    """Load dataset and create DataLoader."""
    dataset = get_dataset(
        dataset_dir=dataset_dirs,
        tokenizer = tokenizer,
        model_config=model_config,
        train_config=train_config,
        split="test" if is_validation else "train",
    )
    return DataLoader(
        dataset,
        batch_size=train_config.batch_size_training,
        shuffle=not is_validation,
        num_workers=1 if is_validation else model_config.num_workers,
        collate_fn=dataset.data_collator,
    ), dataset
    
def load_last_epoch_and_step(train_log_path):
    """
    Load the last epoch and step from the train_log.json file.
    """
    training_log_path = os.path.join(train_log_path, "train_log.json")
    if os.path.exists(training_log_path):
        with open(training_log_path, 'r') as f:
            try:
                train_log = json.load(f)
                # Assuming each entry in the log contains 'epoch' and 'step'
                last_log = train_log[-1]  # Last entry in the log
                last_epoch = last_log.get('epoch', None)
                last_step = last_log.get('step', None)
                return last_epoch, last_step
            except Exception as e:
                logger.error(f"Error loading the train log: {e}")
                return None, None
    else:
        logger.warning(f"Train log file not found: {train_log_path}")
        return None, None
    
def load_latest_checkpoint(model, train_config, model_config):
    """Load the latest model checkpoint if available."""
    output_dir = train_config.output_dir
    if not os.path.isdir(output_dir) or not os.listdir(output_dir):
        logger.info("No checkpoint found. Training from scratch.")
        return model, train_config, model_config
    
    logger.info("--> Loading the latest checkpoint.")
    
    checkpoint_dirs = sorted(
        (d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d)) and d.startswith(train_config.model_name)),
        key=lambda d: int(d.split('_')[-1]) if d.split('_')[-1].isdigit() else -1,
        reverse=True
    )
    latest_epoch_folder = checkpoint_dirs[0] if checkpoint_dirs else None
    
    if latest_epoch_folder:
        logger.info(f'--> Latest checkpoint found: {latest_epoch_folder}')        
        return load_model_checkpoint_peft(model, train_config, model_config, checkpoint_name=latest_epoch_folder)
    
    logger.info("No valid checkpoint found.")
    return model, train_config, model_config

def compare_configs(old_config: Dict[str, Any], new_config: Dict[str, Any]) -> bool:
    """Compare two config dictionaries, handling non-serializable objects."""
    
    # helper to properly handle PeftConfig and similar objects
    def complicated_config(config):
        complicated = {}
        for k, v in config.items():
            if hasattr(v, '__dict__'):  # Handle objects with __dict__ attribute
                complicated[k] = complicated_config(v.__dict__)
            elif isinstance(v, (list, tuple)):
                complicated[k] = [complicated_config(i.__dict__) if hasattr(i, '__dict__') else i for i in v]
            elif isinstance(v, dict):
                complicated[k] = complicated_config(v)
            else:
                complicated[k] = v
        return complicated
    
    try:
        return json.dumps(complicated_config(old_config), sort_keys=True) != \
               json.dumps(complicated_config(new_config), sort_keys=True)
    except TypeError:
        # Fallback to simple dict comparison if complicated config func fails
        return old_config != new_config

def update_train_config_if_changed(
    train_config: 'TrainConfig',
    output_dir: str,
    force_update: bool = False
) -> None:
    """Check if training config changed and update if needed."""
    train_config_path = os.path.join(output_dir, "train_config.json")
    new_train_config = train_config.__dict__
    
    # Check if update is needed
    needs_update = force_update or not os.path.exists(train_config_path)
    
    if not needs_update:
        try:
            with open(train_config_path, 'r') as f:
                old_train_config = json.load(f)
            needs_update = compare_configs(old_train_config, new_train_config)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error reading existing config: {e}")
            needs_update = True
    
    # Update if needed
    if needs_update:
        os.makedirs(output_dir, exist_ok=True)
        try:
            # Convert PeftConfig and other complex objects to dicts
            complicated_config = {}
            for k, v in new_train_config.items():
                if hasattr(v, '__dict__'):
                    complicated_config[k] = v.__dict__
                else:
                    complicated_config[k] = v
            
            with open(train_config_path, 'w') as f:
                json.dump(complicated_config, f, indent=4)
            logger.info("Training configuration updated.")
        except Exception as e:
            logger.error(f"Failed to update training config: {e}")
            raise
    else:
        logger.info("Training configuration unchanged - using existing file.")

def enable_gradient_checkpointing(model, train_config):
    """Enable gradient checkpointing if applicable."""
    if train_config.use_gradient_checkpointing:
        if hasattr(model, "encoder") and hasattr(model.encoder, "gradient_checkpointing_enable"):
            model.encoder.gradient_checkpointing_enable(dict(use_reentrant=False))
        if hasattr(model, "llm") and hasattr(model.llm, "gradient_checkpointing_enable"):
            model.llm.gradient_checkpointing_enable(dict(use_reentrant=False))

def main():
    args = parse_args()
    model_config, train_config = ModelConfig(), TrainConfig()
    
    # Initialize directories and handle config updates
    os.makedirs(train_config.output_dir, exist_ok=True)
    setup_logging(train_config.output_dir)
    
    # Update configs if changed (only training config in your case)
    update_train_config_if_changed(train_config, train_config.output_dir)
    
    # Load model and tokenizer
    model, tokenizer = model_factory(train_config, model_config, metric="acc")
    device = auto_device()
    model.to(device)
    
    # Handle checkpoint loading
    model, train_config, model_config = load_latest_checkpoint(model, train_config, model_config)
    
    # Handle training resumption
    start_epoch, start_step = 1, 0
    if os.path.exists(os.path.join(train_config.output_dir, "train_log.json")):
        last_epoch, last_step = load_last_epoch_and_step(train_config.output_dir)
        if last_epoch is not None and last_step is not None:
            start_epoch, start_step = last_epoch, last_step
    
    # Data loading
    dataset_dirs = args.dataset_dirs
    logger.info(f"Data Source Configuration: {dataset_dirs}")
    
    train_loader, dataset_train = load_data(dataset_dirs, model_config, train_config, tokenizer)
    validate_loader, dataset_val = (load_data(dataset_dirs, model_config, train_config, tokenizer, is_validation=True)
                                  if train_config.run_validation else (None, None))
    
    logger.info(f"--> Training Set Length = {len(dataset_train)}")
    logger.info(f"--> Validation Set Length = {len(dataset_val) if validate_loader else 0}")
    
    # Optimizer setup
    optimizer = AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    num_steps = train_config.num_epochs * len(train_loader)
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: (step / train_config.warmup_step if step < train_config.warmup_step
                              else max(0.0, 1 - (step - train_config.warmup_step) / max(1, num_steps - train_config.warmup_step)))
    )
    scaler = torch.amp.GradScaler() if train_config.use_fp16 else None
    
    # Load optimizer state if available
    if os.path.isdir(train_config.output_dir):
        optimizer, scheduler, scaler = load_optimizer_scheduler_scaler(optimizer, scheduler, scaler, train_config.output_dir, device)
    
    # Model configuration
    enable_gradient_checkpointing(model, train_config)
    
    if not train_config.use_peft and train_config.freeze_layers:
        freeze_transformer_layers(model, train_config.num_freeze_layers)
        check_frozen_layers_peft_model(model)
    
    if model_config.using_llm_type == "unsloth":
        model = torch.compile(model)
    
    # Training
    results = train(model, train_loader, validate_loader, optimizer, scheduler, scaler, train_config, device)
    for k, v in results.items():
        logger.info(f'Key: {k}, Value: {v}')

if __name__ == "__main__":
    main()