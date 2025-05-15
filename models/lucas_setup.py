import os
import json
import torch
import logging
from dataclasses import is_dataclass, asdict
from functools import lru_cache
from typing import Tuple, Dict, Any, Optional, Union
from peft import (
    PeftModel,
    LoraConfig, 
    AdaptionPromptConfig, 
    PrefixTuningConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@lru_cache(maxsize=2)
def set_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.add_tokens(["<|start_of_audio|>", "<|end_of_audio|>"], special_tokens=True)
    return tokenizer

def set_peft_config(train_config: Any, save_path: str = None) -> Any:
    peft_configs = {
        "lora": LoraConfig,
        "llama_adapter": AdaptionPromptConfig,
        "prefix": PrefixTuningConfig,
    }
    
    # Extract configuration
    config = train_config.peft_config
    if is_dataclass(config):
        params = asdict(config)
    else:
        params = config.copy()
    
    # Get PEFT method and remove from params
    peft_method = params.pop("peft_method", "lora")
    
    # Validate PEFT method
    if peft_method not in peft_configs:
        raise ValueError(f"Unsupported PEFT method: {peft_method}. Choose from: {list(peft_configs.keys())}")
    
    # Create config object
    config_obj = peft_configs[peft_method](**params)
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(params, f)
        
    return config_obj

def load_llm_with_unsloth(
    model_name: str, 
    max_seq_length: int, 
    train_config: Any
) -> Tuple[Any, AutoTokenizer]:
    try:
        from unsloth import FastLanguageModel
        
        # Load model with Unsloth
        llm, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        
        # Apply PEFT
        llm = FastLanguageModel.get_peft_model(
            llm,
            r=train_config.peft_config.r,
            target_modules=train_config.peft_config.target_modules,
            lora_alpha=train_config.peft_config.lora_alpha,
            lora_dropout=train_config.peft_config.lora_dropout,
            bias=train_config.peft_config.bias,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        return llm, tokenizer
    except ImportError:
        raise ImportError("The 'unsloth' library is not installed. Please install it or set using_llm_type to None.")

def load_llm_standard(
    model_name: str, 
    train_config: Any
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a model with standard HuggingFace approach.
    
    Args:
        model_name: HuggingFace model name
        train_config: Training configuration
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load model with quantization 
    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    # Get tokenizer
    tokenizer = set_tokenizer(model_name)
    
    # Apply PEFT adapter if requested
    if getattr(train_config, "use_peft", False) and getattr(train_config, "peft_config", None) is not None:
        if isinstance(llm, PeftModel):
            logger.warning("Model already has a PEFT adapter. Skipping re-initialization.")
        else:
            peft_config = set_peft_config(train_config, save_path=None)
            llm = prepare_model_for_kbit_training(llm)
            llm = get_peft_model(llm, peft_config)
            logger.info("Applied PEFT model configuration:")
            llm.print_trainable_parameters()

    return llm, tokenizer

def set_llm(
    model_conf: Any, 
    train_config: Any, 
    inference_mode: Optional[bool] = None
) -> Tuple[Any, AutoTokenizer]:
    """
    Set up the language model based on configuration.
    
    Args:
        model_conf: Model configuration
        train_config: Training configuration
        inference_mode: If True, use inference mode
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Load the appropriate model
    if model_conf.using_llm_type == "unsloth" and inference_mode is None:
        llm, tokenizer = load_llm_with_unsloth(
            model_conf.llm_name_hf, 
            model_conf.llm_dim, 
            train_config
        )
    else:
        llm, tokenizer = load_llm_standard(
            model_conf.llm_name_hf, 
            train_config
        )
    
    if train_config.freeze_llm:
        from utils.model_utils import freeze_component
        freeze_component(llm, 'LLM')
        llm.eval()
        
    return llm, tokenizer


def model_factory(
    train_config: Any, 
    model_config: Any, 
    **kwargs: Dict[str, Any]
) -> Tuple[Any, AutoTokenizer]:
    """
    Create the model based on configurations.
    
    Args:
        train_config: Training configuration
        model_config: Model configuration
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Set inference mode if provided
    inference_mode = kwargs.get('inference_mode', False)
    if inference_mode:
        model_config.using_llm_type = None
        logger.info("Using inference mode, disabling unsloth")

    # Set up LLM
    llm, tokenizer = set_llm(model_config, train_config, inference_mode)
    
    # Resize token embeddings to match tokenizer
    llm.resize_token_embeddings(len(tokenizer))

    # Import and set up encoder
    from models.encoder import set_encoder
    encoder = set_encoder(model_config, train_config)

    # Import and set up encoder adaptor (projector)
    from models.encoder_adaptor import set_encoder_adaptor
    encoder_projector = set_encoder_adaptor(model_config)
    
    # Import and create LucAS model
    from models.lucas import LucAS
    model = LucAS(encoder, llm, encoder_projector, tokenizer, train_config, model_config, **kwargs)
    
    # Log trainable parameter information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    
    return model, tokenizer