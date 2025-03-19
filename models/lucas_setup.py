import os
import json
import torch
import logging
from dataclasses import is_dataclass, asdict
from peft import LoraConfig, AdaptionPromptConfig, PrefixTuningConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def set_tokenizer(model_conf):
    tokenizer = AutoTokenizer.from_pretrained(model_conf.llm_name_hf)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.add_tokens(["<|start_of_audio|>", "<|end_of_audio|>"], special_tokens=True)
    return tokenizer

def set_peft_config(train_config, save_path):
    peft_configs = {
        "lora": LoraConfig,
        "llama_adapter": AdaptionPromptConfig,
        "prefix": PrefixTuningConfig,
    }
    config = train_config.peft_config  # Could be a dataclass or dict
    if is_dataclass(config):
        params = asdict(config)
    else:
        params = config.copy()
    peft_method = params.get("peft_method", "lora")
    params.pop("peft_method", None)
    if peft_method not in peft_configs:
        raise ValueError(f"Unsupported PEFT method: {peft_method}")
    config_obj = peft_configs[peft_method](**params)
    with open(save_path, "w") as f:
        json.dump(params, f)
    return config_obj

def set_llm(model_conf, train_config, inference_mode=None):
    try:
        if model_conf.using_llm_type == "unsloth" and inference_mode is None:
            from unsloth import FastLanguageModel
            llm, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_conf.llm_name_hf,
                max_seq_length=model_conf.llm_dim,
                dtype=None,
                load_in_4bit=True,
            )
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
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            llm = AutoModelForCausalLM.from_pretrained(
                model_conf.llm_name_hf,
                quantization_config=quantization_config,
                device_map="auto",
                local_files_only=True
            )
            tokenizer = set_tokenizer(model_conf)
            if train_config.use_peft:
                if hasattr(train_config, 'peft_config') and train_config.peft_config is not None:
                    peft_config = set_peft_config(
                        train_config, 
                        save_path=os.path.join(train_config.output_dir, "adapter_config.json")
                    )
                    llm = prepare_model_for_kbit_training(llm)
                    llm = get_peft_model(llm, peft_config)
                    llm.print_trainable_parameters()
    except ImportError:
        raise ImportError("The 'unsloth' library is not installed. Please install it or ensure it's available.")

    if train_config.freeze_llm:
        logger.info("LLM layers frozen for training.")
        for _, param in llm.named_parameters():
            param.requires_grad = False
        llm.eval()
    return llm, tokenizer

def model_factory(train_config, model_config, **kwargs):
    inference_mode = kwargs.get('inference_mode')
    if inference_mode:
        model_config.using_llm_type = None

    llm, tokenizer = set_llm(model_config, train_config, inference_mode)
    llm.resize_token_embeddings(len(tokenizer))

    from models.encoder import set_encoder  # Assuming you have these modules.
    encoder = set_encoder(model_config)

    if train_config.freeze_encoder:
        logger.info("Encoder layers frozen for training.")
        for _, param in encoder.named_parameters():
            param.requires_grad = False
        encoder.eval()

    from models.encoder_adaptor import set_encoder_adaptor  # Also assumed.
    encoder_projector = set_encoder_adaptor(model_config)

    from models.lucas import LucAS
    model = LucAS(encoder, llm, encoder_projector, tokenizer, train_config, model_config, **kwargs)
    return model, tokenizer
