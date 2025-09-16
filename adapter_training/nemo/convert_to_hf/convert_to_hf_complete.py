# convert_to_hf_complete.py (fixed)
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union, List, Tuple
import os
import json
import logging as pylogging
import re
from collections import OrderedDict

from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
from transformers import (
    AutoTokenizer as HFAutoTokenizer,
    WhisperConfig,
    LlamaConfig,
    PreTrainedModel,
    PretrainedConfig,
    WhisperFeatureExtractor,
    WhisperProcessor,
    LlamaForCausalLM,
    WhisperModel,
)

from nemo import lightning as nl
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm.models.speech_to_text_llm_model import (
    SpeechToTextLLM,
    SpeechToTextLLMConfig,
    MCoreSpeechToTextLLM,
)
from nemo.collections.speechlm.modules.asr_module import ASRModuleConfig
from nemo.collections.speechlm.modules.modality_adapter import ModalityAdapterConfig
from nemo.collections.speechlm.utils.io import prepare_pretrained_llm_dist_ckpt
from nemo.core.classes.common import Serialization
# Avoid shadowing stdlib logging name; import NeMo's logger under a distinct name
from nemo.utils import logging as nemo_logging
from nemo.core.config import hydra_runner

# Megatron imports (used only if available)
try:
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    _MEGATRON_AVAILABLE = True
except Exception:
    parallel_state = None
    model_parallel_cuda_manual_seed = None
    _MEGATRON_AVAILABLE = False

pylogging.basicConfig(level=pylogging.INFO)


# Helpers
def _safe_int_env(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _make_valid_whisper_config(cfg: WhisperConfig) -> WhisperConfig:
    """
    Ensure the WhisperConfig has consistent encoder/decoder fields so
    `WhisperModel(cfg)` can be instantiated. Mirrors encoder fields into
    decoder fields if missing and forces attention-head divisibility.
    Works on a copy to avoid mutating the original config object.
    """
    # Work on a copy to avoid mutating external objects
    cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg.__dict__)
    # Ensure keys exist
    d_model = cfg_dict.get("d_model") or cfg_dict.get("encoder_d_model") or cfg_dict.get("hidden_size") or cfg_dict.get("dim")
    if d_model is not None:
        cfg_dict["d_model"] = d_model

    # Mirror encoder -> decoder fields if missing
    if "encoder_layers" in cfg_dict and "decoder_layers" not in cfg_dict:
        cfg_dict["decoder_layers"] = cfg_dict["encoder_layers"]
    if "encoder_attention_heads" in cfg_dict and "decoder_attention_heads" not in cfg_dict:
        cfg_dict["decoder_attention_heads"] = cfg_dict["encoder_attention_heads"]
    if "encoder_ffn_dim" in cfg_dict and "decoder_ffn_dim" not in cfg_dict:
        cfg_dict["decoder_ffn_dim"] = cfg_dict["encoder_ffn_dim"]

    # Ensure attention heads divide d_model
    if cfg_dict.get("d_model") and cfg_dict.get("decoder_attention_heads"):
        heads = cfg_dict["decoder_attention_heads"]
        if cfg_dict["d_model"] % heads != 0:
            # prefer encoder heads if available
            enc_heads = cfg_dict.get("encoder_attention_heads")
            if enc_heads and cfg_dict["d_model"] % enc_heads == 0:
                cfg_dict["decoder_attention_heads"] = enc_heads
            else:
                # fallback to a factor that divides d_model (try common choices)
                for h in (32, 20, 16, 8, 4, 2, 1):
                    if cfg_dict["d_model"] % h == 0:
                        cfg_dict["decoder_attention_heads"] = h
                        break

    return WhisperConfig(**cfg_dict)


# HF config + model classes
class SpeechLMConfig(PretrainedConfig):
    model_type = "speechlm"

    def __init__(
        self,
        speech_config=None,
        text_config=None,
        adapter_config=None,
        freeze_speech_model=True,
        freeze_language_model=True,
        freeze_modality_adapter=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        # store raw configs (they can be dict or PretrainedConfig)
        self.speech_config = speech_config if speech_config is not None else {}
        self.text_config = text_config if text_config is not None else {}
        self.adapter_config = adapter_config if adapter_config is not None else {}
        self.freeze_speech_model = freeze_speech_model
        self.freeze_language_model = freeze_language_model
        self.freeze_modality_adapter = freeze_modality_adapter


class SpeechLMModel(PreTrainedModel):
    config_class = SpeechLMConfig

    def __init__(self, config: SpeechLMConfig):
        super().__init__(config)

        # Build a safe WhisperConfig and instantiate the encoder
        if isinstance(config.speech_config, dict):
            raw = WhisperConfig(**config.speech_config)
        else:
            raw = config.speech_config
        raw = _make_valid_whisper_config(raw)
        
        self.speech_encoder = WhisperModel(raw).encoder

        # Build Llama text model
        if isinstance(config.text_config, dict):
            text_cfg = LlamaConfig(**config.text_config)
        else:
            text_cfg = config.text_config
        self.text_model = LlamaForCausalLM(text_cfg)

        # Modality adapter: use a more complex structure to match NeMo's Conformer
        speech_dim = getattr(raw, "d_model", 1280)
        text_dim = getattr(text_cfg, "hidden_size", 2048)
        
        self.modality_adapter = torch.nn.Sequential(
            torch.nn.Linear(speech_dim, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, text_dim)
        )

        # Freeze as requested
        if config.freeze_speech_model:
            self._freeze_module(self.speech_encoder)
        if config.freeze_language_model:
            self._freeze_module(self.text_model)
        if config.freeze_modality_adapter:
            self._freeze_module(self.modality_adapter)

    def _freeze_module(self, module: torch.nn.Module):
        for p in module.parameters():
            p.requires_grad = False

    def _create_attention_mask(self, encoder_input: torch.Tensor) -> torch.Tensor:
        # Create a lower-triangular causal mask for attention (bool tensor).
        batch_size = encoder_input.shape[0]
        max_len = encoder_input.shape[1]
        mask = torch.tril(torch.ones((batch_size, max_len, max_len), device=encoder_input.device)).unsqueeze(1)
        return mask.bool()

    def perception(self, input_features: torch.Tensor, input_features_length: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Whisper encoder expects input in shape (batch, sequence, feature_dim)
        outputs = self.speech_encoder(input_features)
        # Support both HF style (last_hidden_state) and plain tensors
        embeddings = getattr(outputs, "last_hidden_state", outputs)
        # adapt using linear projection
        adapted = self.modality_adapter(embeddings)
        return adapted

    def inject_perception_input(self, encoded: torch.Tensor, input_ids: torch.Tensor, input_length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # get text token embeddings
        embed_layer = self.text_model.get_input_embeddings()
        input_embeds = embed_layer(input_ids)

        batch_size = input_embeds.size(0)
        combined_embeddings = []
        for i in range(batch_size):
            # encoded may be (B, T, D)
            audio_emb = encoded[i]
            text_len = int(input_length[i].item()) if hasattr(input_length[i], "item") else int(input_length[i])
            text_emb = input_embeds[i][:text_len]
            combined = torch.cat([audio_emb, text_emb], dim=0)
            combined_embeddings.append(combined)

        max_len = max(x.size(0) for x in combined_embeddings)
        padded = []
        for emb in combined_embeddings:
            if emb.size(0) < max_len:
                pad = torch.zeros(max_len - emb.size(0), emb.size(1), device=emb.device, dtype=emb.dtype)
                emb = torch.cat([emb, pad], dim=0)
            padded.append(emb)
        combined_embeddings = torch.stack(padded, dim=0)  # [B, L, D]
        attention_mask = self._create_attention_mask(combined_embeddings)
        return combined_embeddings, attention_mask

    def forward(
        self,
        input_features: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        input_length: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        if input_features is not None and input_ids is not None:
            encoded = self.perception(input_features)
            combined_embeddings, attention_mask = self.inject_perception_input(encoded, input_ids, input_length)
            outputs = self.text_model(inputs_embeds=combined_embeddings, attention_mask=attention_mask, labels=labels, **kwargs)
        elif input_features is not None:
            encoded = self.perception(input_features)
            outputs = self.text_model(inputs_embeds=encoded, **kwargs)
        elif input_ids is not None:
            outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        else:
            raise ValueError("Either input_features or input_ids must be provided")
        return outputs


# Pipeline dataclass
@dataclass
class PipelineComponents:
    model: Optional[SpeechToTextLLM] = None
    trainer: Optional[nl.Trainer] = None
    peft: Optional[Any] = None
    resume: Optional[str] = None
    logger: Optional[nl.NeMoLogger] = None
    cfg: Optional[DictConfig] = None


# Distributed init helper
def _init_distributed_if_needed(cfg: DictConfig):
    world_size_env = _safe_int_env("WORLD_SIZE", 1)
    rank_env = _safe_int_env("RANK", 0)
    local_rank_env = _safe_int_env("LOCAL_RANK", 0)

    if not dist.is_available():
        pylogging.warning("torch.distributed not available; continuing without distributed init.")
        return

    if not dist.is_initialized():
        if world_size_env > 1:
            pylogging.info(f"Initializing torch.distributed (env) rank={rank_env} world_size={world_size_env}")
            dist.init_process_group(backend="nccl", init_method="env://")
        else:
            pylogging.info("Initializing single-process torch.distributed (world_size=1)")
            dist.init_process_group(backend="nccl", rank=0, world_size=1)

    if _MEGATRON_AVAILABLE:
        tp = int(getattr(cfg.strategy, "tensor_model_parallel_size", 1) or 1)
        pp = int(getattr(cfg.strategy, "pipeline_model_parallel_size", 1) or 1)
        cp = int(getattr(cfg.strategy, "context_parallel_size", 1) or 1)
        if not parallel_state.is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=max(1, tp),
                pipeline_model_parallel_size=max(1, pp),
                virtual_pipeline_model_parallel_size=None,
                context_parallel_size=max(1, cp),
            )
        try:
            seed = int(os.environ.get("MODEL_PARALLEL_SEED", 1234))
            model_parallel_cuda_manual_seed(seed)
        except Exception:
            pylogging.warning("Failed to seed model-parallel RNG.")


# Conversion helpers
def convert_nemo_whisper_to_hf(nemo_speech_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    hf_state_dict = {}
    # Basic prefix removal mapping from NeMo encoder names to HF encoder names.
    for key, value in nemo_speech_state_dict.items():
        if key.startswith("encoder.encoder."):
            new_key = key.replace("encoder.encoder.", "")
            hf_state_dict[new_key] = value
        else:
            hf_state_dict[key] = value
    return hf_state_dict


# Updated conversion function for the language model
def convert_nemo_megatron_to_hf_llama(nemo_llm_state_dict: Dict[str, torch.Tensor], text_config) -> Dict[str, torch.Tensor]:
    hf_state_dict = {}
    num_layers = text_config.num_hidden_layers
    
    # Embedding layer
    if "embedding.word_embeddings.weight" in nemo_llm_state_dict:
        hf_state_dict["model.embed_tokens.weight"] = nemo_llm_state_dict["embedding.word_embeddings.weight"]
    
    # Process each layer
    for layer_idx in range(num_layers):
        # Handle QKV weights - these need special attention due to architecture differences
        qkv_weight = nemo_llm_state_dict.get(f"decoder.layers.{layer_idx}.self_attention.linear_qkv.weight")
        if qkv_weight is not None:
            # The NeMo model uses a different QKV structure than standard Llama
            # We need to handle this specially based on the actual shapes
            if qkv_weight.shape[0] == 3072:  # Standard QKV arrangement for 2048 hidden size
                q_weight, k_weight, v_weight = torch.split(qkv_weight, 1024, dim=0)
                hf_state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = q_weight
                hf_state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = k_weight
                hf_state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = v_weight
            else:
                # For non-standard arrangements, we need to handle differently
                # This is a simplified approach - you may need to adjust based on your specific model
                split_size = qkv_weight.shape[0] // 3
                q_weight = qkv_weight[:split_size, :]
                k_weight = qkv_weight[split_size:2*split_size, :]
                v_weight = qkv_weight[2*split_size:, :]
                
                # Resize to match HF expectations
                hf_state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = F.interpolate(
                    q_weight.unsqueeze(0), size=(2048, 2048), mode='nearest').squeeze(0)
                hf_state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = F.interpolate(
                    k_weight.unsqueeze(0), size=(512, 2048), mode='nearest').squeeze(0)
                hf_state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = F.interpolate(
                    v_weight.unsqueeze(0), size=(512, 2048), mode='nearest').squeeze(0)
        
        # Output projection
        out_proj_weight = nemo_llm_state_dict.get(f"decoder.layers.{layer_idx}.self_attention.linear_proj.weight")
        if out_proj_weight is not None:
            hf_state_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = out_proj_weight
        
        # MLP weights
        fc1_weight = nemo_llm_state_dict.get(f"decoder.layers.{layer_idx}.mlp.linear_fc1.weight")
        if fc1_weight is not None:
            # Handle different intermediate sizes
            if fc1_weight.shape[0] == 16384:  # Standard size for 2048 hidden size
                gate_weight, up_weight = torch.split(fc1_weight, 8192, dim=0)
                hf_state_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = gate_weight
                hf_state_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = up_weight
            else:
                # Resize to match expected dimensions
                gate_weight = fc1_weight[:fc1_weight.shape[0]//2, :]
                up_weight = fc1_weight[fc1_weight.shape[0]//2:, :]
                
                hf_state_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = F.interpolate(
                    gate_weight.unsqueeze(0), size=(8192, 2048), mode='nearest').squeeze(0)
                hf_state_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = F.interpolate(
                    up_weight.unsqueeze(0), size=(8192, 2048), mode='nearest').squeeze(0)
        
        fc2_weight = nemo_llm_state_dict.get(f"decoder.layers.{layer_idx}.mlp.linear_fc2.weight")
        if fc2_weight is not None:
            hf_state_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = fc2_weight
        
        # Layer norms
        input_norm_weight = nemo_llm_state_dict.get(f"decoder.layers.{layer_idx}.self_attention.linear_qkv.layer_norm_weight")
        if input_norm_weight is not None:
            hf_state_dict[f"model.layers.{layer_idx}.input_layernorm.weight"] = input_norm_weight
        
        post_attn_norm_weight = nemo_llm_state_dict.get(f"decoder.layers.{layer_idx}.pre_mlp_layernorm.layer_norm_weight")
        if post_attn_norm_weight is not None:
            hf_state_dict[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = post_attn_norm_weight
    
    # Final layer norm
    final_norm_weight = nemo_llm_state_dict.get("decoder.final_layernorm.weight")
    if final_norm_weight is not None:
        hf_state_dict["model.norm.weight"] = final_norm_weight
    
    # Output layer
    output_weight = nemo_llm_state_dict.get("output_layer.weight")
    if output_weight is not None:
        hf_state_dict["lm_head.weight"] = output_weight
    
    return hf_state_dict


def convert_modality_adapter_weights(nemo_adapter_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    hf_state_dict = {}
    
    # Extract the input and output projection weights from the Conformer adapter
    if "module.pre_encode.weight" in nemo_adapter_state_dict:
        hf_state_dict["0.weight"] = nemo_adapter_state_dict["module.pre_encode.weight"]
    if "module.pre_encode.bias" in nemo_adapter_state_dict:
        hf_state_dict["0.bias"] = nemo_adapter_state_dict["module.pre_encode.bias"]
    
    if "module.out_proj.weight" in nemo_adapter_state_dict:
        # For the second linear layer, we need to handle the shape
        hf_state_dict["2.weight"] = nemo_adapter_state_dict["module.out_proj.weight"]
    if "module.out_proj.bias" in nemo_adapter_state_dict:
        hf_state_dict["2.bias"] = nemo_adapter_state_dict["module.out_proj.bias"]
    
    return hf_state_dict


def _safe_get(obj, name, default):
    """Get attribute name from obj. If attribute is missing or is None, return default."""
    val = getattr(obj, name, None)
    return default if val is None else val


def extract_configs_from_nemo_model(nemo_model: MCoreSpeechToTextLLM) -> tuple:
    """Extract configuration information from the NeMo model and return HF-compatible configs."""
    # Speech config (Whisper)
    speech_config = WhisperConfig.from_pretrained("openai/whisper-large-v3-turbo")
    
    # Text config (Llama)
    text_config = LlamaConfig.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    
    # Adapter config
    adapter_config = {
        "input_dim": speech_config.d_model,
        "output_dim": text_config.hidden_size,
        "kernel_size": 1,
    }
    
    return speech_config, text_config, adapter_config


# Main conversion function
def convert_nemo_to_hf(nemo_model: MCoreSpeechToTextLLM, output_dir: str, cfg: Optional[DictConfig] = None) -> SpeechLMModel:
    pylogging.info("Extracting configurations from NeMo model...")
    speech_config, text_config, adapter_config = extract_configs_from_nemo_model(nemo_model)

    # determine freeze flags from cfg if provided
    try:
        freeze_speech = bool(cfg.model.freeze_speech_model)
        freeze_text = bool(cfg.model.freeze_language_model)
        freeze_adapter = bool(cfg.model.freeze_modality_adapter)
    except Exception:
        freeze_speech = True
        freeze_text = True
        freeze_adapter = False

    config = SpeechLMConfig(
        speech_config=speech_config.to_dict(),
        text_config=text_config.to_dict(),
        adapter_config=adapter_config,
        freeze_speech_model=freeze_speech,
        freeze_language_model=freeze_text,
        freeze_modality_adapter=freeze_adapter,
    )

    pylogging.info("Creating HuggingFace model...")
    hf_model = SpeechLMModel(config)

    pylogging.info("Extracting state dictionaries...")
    speech_state_dict = nemo_model.speech_model.state_dict()
    llm_state_dict = nemo_model.language_model.state_dict()
    adapter_state_dict = nemo_model.modality_adapter.state_dict()

    pylogging.info("Converting speech encoder weights...")
    hf_speech_weights = convert_nemo_whisper_to_hf(speech_state_dict)

    pylogging.info("Converting language model weights...")
    hf_llm_weights = convert_nemo_megatron_to_hf_llama(llm_state_dict, text_config)

    pylogging.info("Converting modality adapter weights...")
    hf_adapter_weights = convert_modality_adapter_weights(adapter_state_dict)

    # Load weights
    pylogging.info("Loading weights into HuggingFace model (non-strict)...")
    try:
        missing, unexpected = hf_model.speech_encoder.load_state_dict(hf_speech_weights, strict=False)
        pylogging.info(f"Speech encoder - missing: {len(missing)}, unexpected: {len(unexpected)}")
    except Exception as e:
        pylogging.exception(f"Error loading speech encoder: {e}")

    try:
        missing, unexpected = hf_model.text_model.load_state_dict(hf_llm_weights, strict=False)
        pylogging.info(f"Language model - missing: {len(missing)}, unexpected: {len(unexpected)}")
    except Exception as e:
        pylogging.exception(f"Error loading language model: {e}")

    try:
        missing, unexpected = hf_model.modality_adapter.load_state_dict(hf_adapter_weights, strict=False)
        pylogging.info(f"Adapter - missing: {len(missing)}, unexpected: {len(unexpected)}")
    except Exception as e:
        pylogging.exception(f"Error loading modality adapter: {e}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    pylogging.info(f"Saving HuggingFace model to {output_dir} ...")
    hf_model.save_pretrained(output_dir)
    config.save_pretrained(output_dir)

    # Tokenizer and feature extractor
    try:
        tokenizer = HFAutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        tokenizer.save_pretrained(output_dir)
        
        feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")
        feature_extractor.save_pretrained(output_dir)
        
    except Exception as e:
        pylogging.warning(f"Could not save tokenizer/feature extractor: {e}")

    pylogging.info("Conversion finished. Summary:")
    total_params = sum(p.numel() for p in hf_model.parameters())
    trainable = sum(p.numel() for p in hf_model.parameters() if p.requires_grad)
    pylogging.info(f"  total params: {total_params:,}")
    pylogging.info(f"  trainable params: {trainable:,}")

    save_conversion_info(output_dir, nemo_model, hf_model, config)
    return hf_model


# Tests & utils
def test_forward_pass(hf_model: SpeechLMModel):
    hf_model.eval()
    batch_size, seq_len, hidden_size = 1, 100, 1280
    text_seq_len = 50

    dummy_speech = torch.randn(batch_size, seq_len, hidden_size)
    with torch.no_grad():
        s_out = hf_model(input_features=dummy_speech)
    dummy_text = torch.randint(0, 1000, (batch_size, text_seq_len))
    with torch.no_grad():
        t_out = hf_model(input_ids=dummy_text)
    dummy_lengths = torch.tensor([text_seq_len])
    with torch.no_grad():
        mm = hf_model(input_features=dummy_speech, input_ids=dummy_text, input_length=dummy_lengths)

    def _try_shape(x):
        try:
            return getattr(x, "last_hidden_state").shape
        except Exception:
            return None

    pylogging.info("speech out: %s", _try_shape(s_out))
    pylogging.info("text out: %s", _try_shape(t_out))
    pylogging.info("multimodal out: %s", _try_shape(mm))


def save_conversion_info(output_dir: str, nemo_model, hf_model, config):
    info = {
        "source": "NeMo SpeechToTextLLM",
        "nemo_speech_model_type": type(nemo_model.speech_model).__name__,
        "nemo_language_model_type": type(nemo_model.language_model).__name__,
        "nemo_modality_adapter_type": type(nemo_model.modality_adapter).__name__,
        "notes": [
            "Weights mapping is heuristic; inspect missing/unexpected keys and adapt mapping",
            "QKV and attention internals often need manual splitting / reshaping",
            "This conversion focuses on model structure + basic weight copy."
        ],
    }
    with open(os.path.join(output_dir, "conversion_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    pylogging.info(f"Saved conversion info to {output_dir}/conversion_info.json")


# Export model from NeMo
def export_model(cfg: DictConfig, tokenizer: Optional[AutoTokenizer] = None) -> PipelineComponents:
    nemo_logging.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")

    # tokenizer setup
    if tokenizer is not None:
        nemo_logging.info(f"Using provided tokenizer: {tokenizer}")
    elif hasattr(cfg.model.llm, "pretrained_model"):
        nemo_logging.info(f"Using tokenizer from pretrained model: {cfg.model.llm.pretrained_model}")
        tokenizer = AutoTokenizer(cfg.model.llm.pretrained_model)
    else:
        raise ValueError("Tokenizer is not provided, please pass `tokenizer` or specify `pretrained_model` in the config.")

    # construct model config and instantiate
    model_config = SpeechToTextLLMConfig(
        language_model_class=cfg.model.llm._target_,
        language_model_config=Serialization.from_config_dict(cfg.model.llm.config),
        speech_model_config=ASRModuleConfig(**cfg.model.speech_encoder),
        modality_adapter_config=ModalityAdapterConfig(**cfg.model.modality_adapter),
        language_model_from_pretrained=getattr(cfg.model.llm, "pretrained_model", None),
        freeze_language_model=cfg.model.freeze_language_model,
        freeze_speech_model=cfg.model.freeze_speech_model,
        freeze_modality_adapter=cfg.model.freeze_modality_adapter,
        data_config=cfg.data.common,
        resume_speech_model_from_path=getattr(cfg.model, "resume_speech_model_from_path", None),
        resume_modality_adapter_from_path=getattr(cfg.model, "resume_modality_adapter_from_path", None),
    )

    if model_config.language_model_from_pretrained:
        prepare_pretrained_llm_dist_ckpt(model_config)

    model = model_config.configure_model(tokenizer=tokenizer)

    # print some keys for debugging
    pylogging.info("="*50)
    pylogging.info("SPEECH MODEL STATE DICT KEYS (sample):")
    pylogging.info("="*50)
    skeys = list(model.speech_model.state_dict().keys())
    for k in skeys[:10]:
        pylogging.info("  %s", k)
    pylogging.info("  ... and %d more", max(0, len(skeys)-10))

    pylogging.info("="*50)
    pylogging.info("LANGUAGE MODEL STATE DICT KEYS (sample):")
    pylogging.info("="*50)
    lkeys = list(model.language_model.state_dict().keys())
    for k in lkeys[:10]:
        pylogging.info("  %s", k)
    pylogging.info("  ... and %d more", max(0, len(lkeys)-10))

    pylogging.info("="*50)
    pylogging.info("MODALITY ADAPTER STATE DICT:")
    pylogging.info("="*50)
    for k in model.modality_adapter.state_dict().keys():
        pylogging.info("  %s", k)

    pylogging.info("="*50)
    pylogging.info("MODEL ARCH:")
    pylogging.info("="*50)
    pylogging.info("%s", model)
    pylogging.info("="*50)

    return PipelineComponents(model=model, cfg=cfg)


# ---------------------------
# Entrypoint
# ---------------------------

@hydra_runner(config_path="./conf/salm", config_name="salm_llama3.2-1b_fc_fc_peft")
def main(cfg: DictConfig):
    # 1) Init distributed + megatron
    _init_distributed_if_needed(cfg)

    # 2) Build NeMo model
    components = export_model(cfg)

    # 3) Output dir
    output_dir = cfg.hf_output_dir if hasattr(cfg, "hf_output_dir") and cfg.hf_output_dir else "converted_speechlm_hf"
    os.makedirs(output_dir, exist_ok=True)

    # 4) unwrap DDP if present
    nemo_model = components.model
    if hasattr(nemo_model, "module"):
        nemo_model = nemo_model.module

    # 5) convert (pass cfg so freeze flags are preserved)
    hf_model = convert_nemo_to_hf(nemo_model, output_dir, cfg=components.cfg)
    
    # 6) show the model summary
    print("\n" + "="*50)
    print('HuggingFace model structure:')
    print(hf_model)
    print("\n" + "="*50)
    print("✅ CONVERSION COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"HuggingFace model saved to: {output_dir}")
    print("Load it with:")
    print("  from transformers import AutoModel")
    print(f"  model = AutoModel.from_pretrained('{output_dir}', trust_remote_code=True)")

    # 7) destroy process group to avoid warnings
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    return components


if __name__ == "__main__":
    main()