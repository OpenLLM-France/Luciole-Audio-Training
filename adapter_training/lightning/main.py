import os
import json
from typing import Dict, Any

from configs import ModelConfig, TrainConfig
from train import train
from data.audio_chuncks_dataset import get_dataset

from utils.checkpoint_utils import load_model_checkpoint_peft, load_optimizer_scheduler_scaler
from utils.model_utils import freeze_transformer_layers, check_frozen_layers_peft_model

from models.lucas_setup import model_factory
from utils.env import auto_device

import torch

import argparse
from utils.model_utils import freeze_component
import logging

logger = logging.getLogger(__name__)

def setup_logging(output_dir: str) -> None:
    """Configure logging to both file and console without duplicates"""
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create handlers
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] - %(message)s')
    
    file_handler = logging.FileHandler(os.path.join(output_dir, "training.log"))
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Configure module logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent propagation to root logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training pipeline configuration")
    parser.add_argument("-d","--dataset_dirs", nargs="+", required=True)
    parser.add_argument("-bst", "--batch_size_train", type=int)
    parser.add_argument("-bsv", "--batch_size_val", type=int)
    parser.add_argument("-s", "--runing_steps", type=int)
    parser.add_argument("-e", "--num_epochs", type=int)
    parser.add_argument("-r", "--restart_train", action="store_true")
    parser.add_argument("-o", "--output_dir", type=str)
    parser.add_argument("-l", "--load_checkpoint", type=str)
    parser.add_argument("-t", "--train_projector_only", action="store_true")
    parser.add_argument("-n", "--num_workers", type=int)
    parser.add_argument("-lr","--learning_rate", type=float)
    parser.add_argument("-f16","--use_fp16", action="store_true")
    parser.add_argument("-g","--use_gradient_checkpointing", action="store_true")
    parser.add_argument("--gpus", type=int)
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
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=train_config.batch_size_training,
        shuffle=False,
        num_workers=0 if is_validation else model_config.num_workers,
        collate_fn=dataset.data_collator,
        pin_memory=False,persistent_workers=False,   
    ), dataset
    

def compare_configs(old_config: Dict[str, Any], new_config: Dict[str, Any]) -> bool:
    def normalize(cfg):
        if hasattr(cfg, '__dict__'):
            return normalize(vars(cfg))
        if isinstance(cfg, dict):
            return {k: normalize(v) for k, v in cfg.items()}
        if isinstance(cfg, (list, tuple)):
            return [normalize(x) for x in cfg]
        return cfg
    return json.dumps(normalize(old_config), sort_keys=True) != json.dumps(normalize(new_config), sort_keys=True)

def update_train_config_if_changed(train_config: TrainConfig, output_dir: str, force: bool = False) -> None:
    path = os.path.join(output_dir, "train_config.json")
    needs_update = force or not os.path.exists(path)
    try:
        if not needs_update:
            with open(path, 'r') as f:
                old = json.load(f)
                needs_update = compare_configs(old, train_config.__dict__)
    except Exception as e:
        logger.warning(f"Compare failed, forcing update: {e}")
        needs_update = True
    if needs_update:
        os.makedirs(output_dir, exist_ok=True)

        def serialize(obj):
            if hasattr(obj, '__dict__'):
                return serialize(obj.__dict__)
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [serialize(i) for i in obj]
            return obj

        with open(path, 'w') as f:
            json.dump(serialize(train_config.__dict__), f, indent=4)
        logger.info("Train config updated.")

def enable_gradient_checkpointing(model, train_config):
    if not train_config.use_gradient_checkpointing:
        return
    for attr in ["encoder", "llm"]:
        mod = getattr(model, attr, None)
        if hasattr(mod, "gradient_checkpointing_enable"):
            mod.gradient_checkpointing_enable(dict(use_reentrant=False))


def update_config_from_args(config, args, mapping: Dict[str, str]):
    for arg_key, config_key in mapping.items():
        value = getattr(args, arg_key, None)
        if value is not None:
            setattr(config, config_key, value)

def main():
    args = parse_args()
    model_config, train_config = ModelConfig(), TrainConfig()
    
    # Initialize directories FIRST
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure logging EARLY
    setup_logging(args.output_dir)  # Moved this up
    
    # Then proceed with other operations
    TRAIN_ARG_MAPPING = {
        "batch_size_train": "batch_size_training",
        "batch_size_val": "batch_size_validation",
        "runing_steps": "runing_steps",
        "num_epochs": "num_epochs",
        "output_dir": "output_dir",
        "restart_train": "restart_train",
        "train_projector_only": "train_projector_only",
        "learning_rate": "learning_rate",
        "use_fp16": "use_fp16",
        "use_gradient_checkpointing": "use_gradient_checkpointing",
    } 
    update_config_from_args(
        train_config,
        args,
        mapping=TRAIN_ARG_MAPPING
    )
    update_train_config_if_changed(train_config, args.output_dir)
    
    # Load model and tokenizer
    model, tokenizer = model_factory(train_config, model_config, metric="acc")
    device = auto_device()
    model.to(device)
    model.train()
    
    # Handle checkpoint loading
    start_epoch, start_step = 1, 0
    if args.load_checkpoint:
        logger.info(f'--> Latest checkpoint found: {args.load_checkpoint}')
        model, start_epoch, start_step, train_config, model_config = load_model_checkpoint_peft(
            model, args.load_checkpoint
        )
        update_config_from_args(
            train_config,
            args,
            mapping=TRAIN_ARG_MAPPING
            )

    if args.restart_train:
        logger.info("Resetting training progress")
        start_epoch = 0
        start_step = 0
    
    if not os.path.exists(os.path.join(train_config.output_dir, "model_config.json")):
        model_config_file = os.path.join(train_config.output_dir, "model_config.json")
        model_config.save(model_config_file)
        
    if args.train_projector_only or train_config.train_projector_only:
        model.train_projector_only()          
          
    # Data loading
    model_config.num_workers = args.num_workers if args.num_workers is not None else model_config.num_workers 
    dataset_dirs = args.dataset_dirs
    logger.info(f"Data Source Configuration: {dataset_dirs}")
    train_loader, dataset_train = load_data(dataset_dirs, model_config, train_config, tokenizer)
    validate_loader, dataset_val = (load_data(dataset_dirs, model_config, train_config, tokenizer, is_validation=True)
                                  if train_config.run_validation else (None, None))
    # Optimizer 
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=train_config.learning_rate) # , weight_decay=train_config.weight_decay
            
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)
    scaler = torch.amp.GradScaler() if train_config.use_fp16 else None
    
    # Load optimizer state if available
    if args.load_checkpoint and os.path.isdir(args.output_dir):
        optimizer, scheduler, scaler = load_optimizer_scheduler_scaler(optimizer, scheduler, scaler, args.output_dir, device)
    
    # enable gradient checkpointing if it's True
    enable_gradient_checkpointing(model, train_config)
    
    if model_config.using_llm_type == "unsloth":
        try:
            model = torch.compile(model)
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
       
    trainable_params = model.get_trainable_parameters()
    logger.info(f"Trainable parameters summary:")
    for component, num_params in trainable_params.items():
        logger.info(f"  {component}: {num_params:,}")
    
    
    if args.train_projector_only or train_config.train_projector_only:    
        assert trainable_params['encoder'] == 0, "Encoder should be frozen"
        assert trainable_params['llm'] == 0, "LLM should be frozen"
        assert trainable_params['projector'] > 0, "Projector should be trainable"
        assert trainable_params['total'] == trainable_params['projector'], "Only projector should be trainable"
    
    if train_config.freeze_llm:
        assert trainable_params['llm'] == 0, "LLM should be frozen"
        
    if train_config.freeze_encoder:
        assert trainable_params['encoder'] == 0, "Encoder should be frozen"
         
    try:
        logger.info("Starting training...")
        results = train(
            model, tokenizer, train_loader, validate_loader, optimizer, scheduler, 
            scaler, train_config, start_epoch, start_step,  device
        )
        # Log final results
        logger.info("Training completed successfully!")
        logger.info(f"Final metrics: {results}")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise e
    for k, v in results.items():
        logger.info(f'Key: {k}, Value: {v}')

if __name__ == "__main__":
    main()
