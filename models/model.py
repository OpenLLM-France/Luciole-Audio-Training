from typing import Optional, List
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

from linastt.utils.env import *
from models.encoder import *
from models.encoder_adaptor import *
from utils import load_model_checkpoint_peft

from peft import LoraConfig, AdaptionPromptConfig, PrefixTuningConfig
from dataclasses import is_dataclass, asdict
import json
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from torch.utils.checkpoint import checkpoint

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_embeddings(model, input_ids):
    if hasattr(model, "embed_tokens"):
        return model.embed_tokens(input_ids)
    elif hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens(input_ids)
    elif hasattr(model.model.model, "embed_tokens"):
        return model.model.model.embed_tokens(input_ids)
    elif hasattr(model.model.model.model, "embed_tokens"):
        return model.model.model.model.embed_tokens(input_ids)
    raise ValueError("embed_tokens method not found in the LLM model structure.")

class LucAS(nn.Module):
    def __init__(
        self, 
        encoder: nn.Module,
        llm: nn.Module, 
        encoder_projector: nn.Module, 
        tokenizer, 
        train_config, 
        model_config, 
        **kwargs):
        super().__init__()
        self.encoder = encoder
        self.llm = llm
        self.encoder_projector = encoder_projector
        self.tokenizer = tokenizer
        self.train_config = train_config
        self.model_config = model_config
        self.metric = self.train_config.metric

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        audio_mel = kwargs.get("audio", None)
        modality_mask = kwargs.get("modality_mask", None)
        encoder_outs = None

        # Process audio input
        if audio_mel is not None:
            # if self.train_config.freeze_encoder:
            #     logger.info("Encoder layers frozen for training.")
            #     for _, param in self.encoder.named_parameters(): 
            #         param.requires_grad = False
            #     self.encoder.eval()
                
            if audio_mel.dim() == 2:  # If single audio sample
                audio_mel = audio_mel.unsqueeze(0)
            encoder_outs = self.encoder.extract_variable_length_features(audio_mel.permute(0, 2, 1))
            encoder_outs = self.encoder_projector(encoder_outs)

        # Handle tokenized input
        if input_ids is not None:
            input_ids[input_ids == -1] = 0
            if input_ids.dtype != torch.long:
                input_ids = input_ids.long()
            inputs_embeds = get_embeddings(self.llm, input_ids)

        if inputs_embeds is None:
            raise ValueError("`inputs_embeds` cannot be None. Ensure `input_ids` or precomputed embeddings are provided.")

        # Handle modality masking
        if modality_mask is not None:
            modality_mask_start_indices = (modality_mask == True).float().argmax(dim=1)
            modality_lengths = torch.clamp(modality_mask.sum(dim=1), max=encoder_outs.shape[1]).tolist()
            encoder_outs_pad = torch.zeros_like(inputs_embeds)
            for i in range(encoder_outs.shape[0]):
                start_idx = modality_mask_start_indices[i]
                length = modality_lengths[i]
                encoder_outs_pad[i, start_idx:start_idx + length] = encoder_outs[i][:length]
            inputs_embeds = encoder_outs_pad + inputs_embeds * (~modality_mask[:, :, None])
                
        # Pass through the language model
        model_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            use_reentrant=False,
        )
        
        preds = torch.argmax(model_outputs.logits, -1)
        acc = -1 if labels is None else compute_accuracy(preds[:, :-1], labels[:, 1:], ignore_label=-100)
        mask = labels[:, 1:] != -100
        
        # print(f'preds = {self.tokenizer.decode(preds[:, :-1].masked_select(mask).tolist())}')
        # print(f"labels = {self.tokenizer.decode(labels[:, 1:].masked_select(mask).tolist())}")
        # print('=====')
        return model_outputs, acc
    
    @torch.no_grad()
    def generate(self, input_ids: Optional[torch.LongTensor] = None, inputs_embeds: Optional[torch.FloatTensor] = None, attention_mask: Optional[torch.Tensor] = None, use_cache: bool = True, **kwargs):
        if inputs_embeds is None and input_ids is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")

        model_outputs = self.llm.generate(
            input_ids=input_ids, 
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask, 
            max_new_tokens=self.model_config.max_new_tokens,
            num_beams=self.model_config.num_beams,
            do_sample=self.model_config.do_sample,
            min_length=self.model_config.min_length,
            top_p=self.model_config.top_p,
            repetition_penalty=self.model_config.repetition_penalty,
            length_penalty=self.model_config.length_penalty,
            temperature=self.model_config.temperature,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs,
        )
        return model_outputs

def compute_accuracy(pad_outputs, pad_targets, ignore_label=-100):
    mask = pad_targets != ignore_label
    numerator = torch.sum(pad_outputs.masked_select(mask) == pad_targets.masked_select(mask))
    denominator = torch.sum(mask)
    return (numerator.float() / denominator.float())

def set_peft_config(train_config, save_path):
    peft_configs = {"lora": LoraConfig, "llama_adapter": AdaptionPromptConfig, "prefix": PrefixTuningConfig}
    config = train_config.peft_config  # This could be a dataclass instance or dict.
    
    # If config is a dataclass instance, convert it to dict; otherwise, assume it's a dict.
    if is_dataclass(config):
        params = asdict(config)
    else:
        params = config.copy()  # copy to avoid modifying original
    
    # Extract the peft_method and then remove it from params so it doesn't get passed to the constructor.
    peft_method = params.get("peft_method", "lora")
    params.pop("peft_method", None)  # Remove the key if present.
    
    if peft_method not in peft_configs:
        raise ValueError(f"Unsupported PEFT method: {peft_method}")
    
    # Instantiate the config without the unexpected keyword.
    config_obj = peft_configs[peft_method](**params)
    
    # Save the parameters to a JSON file for reference.
    with open(save_path, "w") as f:
        json.dump(params, f)
    
    return config_obj

def set_llm(model_conf, train_config, inference_mode=None):
    # Try importing unsloth
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
            llm = AutoModelForCausalLM.from_pretrained(model_conf.llm_name_hf, quantization_config=quantization_config, device_map="auto")

            tokenizer = set_tokenizer(model_conf)

            if train_config.use_peft : # and not inference_mode
                if hasattr(train_config, 'peft_config') and train_config.peft_config is not None:
                    peft_config = set_peft_config(train_config, save_path=os.path.join(train_config.output_dir, "adapter_config.json"))
                    llm = prepare_model_for_kbit_training(llm)
                    llm = get_peft_model(llm, peft_config)
                    llm.print_trainable_parameters()

    except ImportError:
        raise ImportError("The 'unsloth' library is not installed. Please install it or ensure it's available in your environment.")

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
    encoder = set_encoder(model_config)
    
    if train_config.freeze_encoder:
        logger.info("Encoder layers frozen for training.")
        for _, param in encoder.named_parameters(): 
            param.requires_grad = False
        encoder.eval()
        
    encoder_projector = set_encoder_adaptor(model_config)
    model = LucAS(encoder, llm, encoder_projector, tokenizer, train_config, model_config, **kwargs)
    return model, tokenizer

def set_tokenizer(model_conf):
    tokenizer = AutoTokenizer.from_pretrained(model_conf.llm_name_hf)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.add_tokens(["<|start_of_audio|>", "<|end_of_audio|>"],  special_tokens=True)
    return tokenizer

if __name__ == '__main__':
    from configs import TrainConfig, ModelConfig
    model_config = ModelConfig()
    train_config = TrainConfig()
    llm, tokenizer = model_factory(train_config, model_config)
    embeddings = llm.llm.get_input_embeddings()
    print(llm.llm.model.model.embed_tokens)
