import os
import json

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
    
    if not os.path.exists(os.path.join(train_config.output_dir, "model_config.json")) or not os.path.exists(os.path.join(train_config.output_dir, "train_config.json")):
        model_config_file, train_config_file = setup_directories(train_config)
        save_config_files(model_config, train_config, model_config_file, train_config_file)
        
    os.makedirs(train_config.output_dir, exist_ok=True)
    setup_logging(train_config.output_dir)
    start_epoch, start_step = 1, 0
   
    model, tokenizer = model_factory(train_config, model_config, metric="acc")
    device = auto_device()
    model.to(device)
   
    model, train_config, model_config = load_latest_checkpoint(model, train_config, model_config)

    if os.path.exists(os.path.join(train_config.output_dir, "train_log.json")):
        last_epoch, last_step = load_last_epoch_and_step(train_config.output_dir)
        if last_epoch is not None and last_step is not None:
            start_epoch, start_step = last_epoch, last_step
    
    
    dataset_dirs = args.dataset_dirs
    
    print("Data Source Configuration:")
    print(dataset_dirs)
    
    train_loader, dataset_train = load_data(dataset_dirs, model_config, train_config, tokenizer)
    validate_loader, dataset_val = (load_data(dataset_dirs, model_config, train_config, tokenizer, is_validation=True)
                                    if train_config.run_validation else (None, None))
    
    logger.info(f"--> Training Set Length = {len(dataset_train)}")
    logger.info(f"--> Validation Set Length = {len(dataset_val) if validate_loader else 0}")
    
    
    optimizer = AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    num_steps = train_config.num_epochs * len(train_loader)
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: (step / train_config.warmup_step if step < train_config.warmup_step
                                else max(0.0, 1 - (step - train_config.warmup_step) / max(1, num_steps - train_config.warmup_step)))
    )
    scaler = torch.amp.GradScaler() if train_config.use_fp16 else None
    
    if  os.path.isdir(train_config.output_dir):
        optimizer, scheduler, scaler = load_optimizer_scheduler_scaler(optimizer, scheduler, scaler, train_config.output_dir, device)
        
    enable_gradient_checkpointing(model, train_config)
    
    if not train_config.use_peft and train_config.freeze_layers:
        freeze_transformer_layers(model, train_config.num_freeze_layers)
        check_frozen_layers_peft_model(model)
    
    if model_config.using_llm_type == "unsloth":
        model = torch.compile(model)
    
    results = train(model, train_loader, validate_loader, optimizer, scheduler, scaler, train_config, device)
    for k, v in results.items():
        logger.info(f'Key: {k}, Value: {v}')

if __name__ == "__main__":
    main()