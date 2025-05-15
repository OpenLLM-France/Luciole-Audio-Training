import os
import json
import argparse
import torch
import logging
from utils.env import auto_device
from utils.audio import load_audio
from configs import TrainConfig, ModelConfig
from models.lucas_setup import model_factory
from peft import PeftModel
import whisper
import gc
from pathlib import Path
import numpy as np
# ---------------------------------------------------------------------------- #
#                                    Logging                                    #
# ---------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO,  # Changed to INFO from DEBUG to reduce logging overhead
                    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)

def discover_and_load_checkpoint(model, train_cfg: TrainConfig, ckpt_root: str) -> torch.nn.Module:
    """
    Look in ckpt_root for model.pt or folders named <model_name>_<epoch>,
    pick the highest epoch, load its `model.pt` and optional PEFT.
    """
    if not os.path.isdir(ckpt_root):
        logger.warning(f"No checkpoint directory at {ckpt_root}")
        return model

    # Check if model.pt is directly in ckpt_root
    model_pt = os.path.join(ckpt_root, "model.pt")
    if os.path.isfile(model_pt):
        logger.info(f"Loading checkpoint directly from {model_pt}")
        # Load with map_location to control device placement
        dd = torch.load(model_pt, map_location="cpu")
        
        # Load state dictionaries with memory optimization
        model.encoder.load_state_dict(dd["encoder_state"], strict=False)
        del dd["encoder_state"]
        gc.collect()
        
        model.llm.load_state_dict(dd["llm_state"], strict=False)
        del dd["llm_state"]
        gc.collect()
        
        model.encoder_projector.load_state_dict(dd["projector_state"], strict=False)
        del dd["projector_state"]
        del dd
        gc.collect()
        
        logger.info("Model weights loaded from direct checkpoint.")

        # Load PEFT adapter if present
        peft_dir = os.path.join(ckpt_root, "peft_adapter")
        if os.path.isdir(peft_dir):
            logger.info(f"Found PEFT adapter directory: {peft_dir}")
            try:
                # Use device_map="auto" for optimal memory distribution
                model.llm = PeftModel.from_pretrained(
                    model.llm, 
                    peft_dir,
                    is_trainable=False,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    offload_folder="offload_folder"  # Enable disk offloading for large models
                )
                logger.info("✅ PEFT adapter loaded successfully.")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load PEFT adapter: {e}")
        else:
            logger.info("No PEFT adapter directory found in checkpoint root.")
        return model

    # Otherwise, look for subdirectories with epoch numbers
    candidates = [
        d for d in os.listdir(ckpt_root)
        if d.startswith(train_cfg.model_name + "_")
           and d.split("_")[-1].isdigit()
    ]
    if not candidates:
        logger.info("No matching checkpoints found.")
        return model

    latest = sorted(candidates, key=lambda d: int(d.split("_")[-1]))[-1]
    ckpt_dir = os.path.join(ckpt_root, latest)
    model_pt = os.path.join(ckpt_dir, "model.pt")
    if not os.path.isfile(model_pt):
        logger.error(f"Expected model.pt under {ckpt_dir}, but none found.")
        return model

    logger.info(f"Loading checkpoint from: {ckpt_dir}")
    dd = torch.load(model_pt, map_location="cpu")
    
    # Load state dictionaries with memory optimization
    model.encoder.load_state_dict(dd["encoder_state"], strict=False)
    del dd["encoder_state"]
    gc.collect()
    
    model.llm.load_state_dict(dd["llm_state"], strict=False)
    del dd["llm_state"]
    gc.collect()
    
    model.encoder_projector.load_state_dict(dd["projector_state"], strict=False)
    del dd["projector_state"]
    del dd
    gc.collect()
    
    logger.info("Model weights loaded.")

    peft_dir = os.path.join(ckpt_dir, "peft_adapter")
    if os.path.isdir(peft_dir):
        logger.info(f"Found PEFT adapter directory: {peft_dir}")
        try:
            model.llm = PeftModel.from_pretrained(
                model.llm, 
                peft_dir,
                is_trainable=False,
                torch_dtype=torch.float16,
                device_map="auto",
                offload_folder="offload_folder"  # Enable disk offloading for large models
            )
            logger.info("✅ PEFT adapter loaded successfully.")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load PEFT adapter: {e}")
    else:
        logger.info("No PEFT adapter directory found.")

    return model


def load_model_and_tokenizer(model_dir: str, device: torch.device, use_fp16: bool = True):
    mc_path = os.path.join(model_dir, "model_config.json")
    tc_path = os.path.join(model_dir, "train_config.json")
    mc,tc = load_all_configs(mc_path, tc_path)
    
    logger.info("Building model...")
    # Set inference_mode=True to optimize memory usage
    model, tokenizer = model_factory(tc, mc, inference_mode=True)
    
    # Configure model based on device and dtype
    dtype = torch.float16 if use_fp16 and device.type == 'cuda' else torch.float32
    logger.info(f"Using model dtype: {dtype}")
    
    # Use gradient checkpointing to save memory
    if hasattr(model.llm, "gradient_checkpointing_enable"):
        model.llm.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    # Create device map for better memory distribution
    if device.type == 'cuda':
        # Move encoder to CPU initially to save GPU memory
        model.encoder = model.encoder.to('cpu')
        model.encoder_projector = model.encoder_projector.to('cpu')
        
        # Move LLM to specified device with appropriate dtype
        model.llm = model.llm.to(device=device, dtype=dtype)
    else:
        model = model.to(device)

    ck_root = os.path.join(model_dir, "asllama_ckp_step_12000")
    model = discover_and_load_checkpoint(model, tc, ck_root)
    model.eval()  # Set to evaluation mode

    # Log PEFT usage
    if "peft" in str(type(model.llm)).lower():
        logger.info(f"✅ PEFT is enabled. LLM type: {type(model.llm)}")
    else:
        logger.warning(f"🚫 PEFT is NOT enabled. LLM type: {type(model.llm)}")

    return model, tokenizer, mc, tc


def process_audio_in_chunks(model, tokenizer, audio_file, device, model_cfg, start=None, end=None, chunk_size=30):
    """Process longer audio in manageable chunks to reduce memory usage"""
    
    if isinstance(audio_file, (str, Path)) and Path(audio_file).exists():
        wav = load_audio(audio_file, start=start, end=end)
    elif isinstance(audio_file, np.ndarray):
        wav = audio_file
    else:
        print("UNK Format")
    wav = whisper.pad_or_trim(wav)
    
    # Create mel spectrogram on CPU first
    mel = whisper.log_mel_spectrogram(wav, n_mels=model_cfg.mel_size).to('cpu')
    audio_mel = mel.unsqueeze(0)
    logger.info(f"Audio mel shape: {audio_mel.shape}")
    
    # Move encoder to device for processing
    model.encoder = model.encoder.to(device)
    model.encoder_projector = model.encoder_projector.to(device)
    
    # Feature extraction with memory optimization
    with torch.no_grad(), torch.amp.autocast(device_type=device.type if device.type=='cuda' else 'cpu'):
        audio_mel = audio_mel.to(device)
        feats = model.encoder.extract_variable_length_features(audio_mel)
        feats = model.encoder_projector(feats)
        
        # Clean up to free memory
        del audio_mel
        torch.cuda.empty_cache() if device.type=='cuda' else None
        
        # Convert features to LLM dtype
        feats = feats.to(device=device, dtype=model.llm.dtype)
        logger.info(f"Encoded features shape: {feats.shape}")
    
    # Move encoder back to CPU to save GPU memory during generation
    if device.type == 'cuda':
        model.encoder = model.encoder.to('cpu')
        model.encoder_projector = model.encoder_projector.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()
    
    return feats


def generate_with_memory_efficiency(model, tokenizer, feats, prompt, device):
    """Run generation with memory optimizations"""
    # Encode prompt
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Get embeddings without storing computational graph
    with torch.no_grad():
        text_emb = model.llm.get_input_embeddings()(ids)
        text_emb = text_emb.to(device=device, dtype=model.llm.dtype)
        logger.info(f"Prompt embedding shape: {text_emb.shape}")
    
    # Clear unnecessary tensors
    del ids
    torch.cuda.empty_cache() if device.type=='cuda' else None
    
    # Prepare for generation
    inputs_embeds = torch.cat([feats, text_emb], dim=1)
    attn_mask = torch.ones(inputs_embeds.size()[:2], dtype=torch.long, device=device)
    
    # Clear more unnecessary tensors
    del text_emb, feats
    torch.cuda.empty_cache() if device.type=='cuda' else None
    
    logger.info("Running generation...")
    
    # Configure generation parameters for memory efficiency
    gen_kwargs = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attn_mask,
        "use_cache": True,
    }
    
    # Run generation with reduced memory footprint
    with torch.no_grad(), torch.amp.autocast(device_type=device.type if device.type=='cuda' else 'cpu'):
        # Check if model has default generation parameters and avoid passing conflicting params
        if isinstance(model.llm, PeftModel):
            # For PEFT models, we need to be careful with generation params
            out = model.generate(**gen_kwargs)
        else:
            # For regular models, we can add max_new_tokens
            out = model.generate(**gen_kwargs, max_new_tokens=512)
    
    # Clear tensors to free memory
    del inputs_embeds, attn_mask, gen_kwargs
    torch.cuda.empty_cache() if device.type=='cuda' else None
    
    raw = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    del out
    torch.cuda.empty_cache() if device.type=='cuda' else None
    
    return raw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model_dir", required=True,
                   help="Path to your model folder (containing model_config.json, train_config.json, checkpoints/)")
    p.add_argument("-a", "--audio", required=True, help="Path to one audio file")
    p.add_argument("--gpus", default="0", help="CUDA_VISIBLE_DEVICES")
    p.add_argument("--start", type=float, default=None, help="strat duration of the audio")
    p.add_argument("--end", type=float, default=None, help="end duration of the audio")
    p.add_argument("--fp16", action="store_true", help="Use FP16 precision to save memory")
    p.add_argument("--cpu_offload", action="store_true", help="Offload some model parts to CPU")
    p.add_argument("--memory_efficient", action="store_true", help="Use more aggressive memory optimization")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    device = auto_device()
    
    # Configure PyTorch for memory efficiency
    if device.type == 'cuda':
        # Set PyTorch to release memory faster
        torch.cuda.empty_cache()
        # Enable memory efficient attention if available
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    logger.info(f"Using device: {device}")
    
    # Force garbage collection
    gc.collect()
    
    logger.info("Loading model and tokenizer...")
    model, tokenizer, model_cfg, train_cfg = load_model_and_tokenizer(
        args.model_dir, 
        device, 
        use_fp16=args.fp16
    )

    if not os.path.isfile(args.audio):
        raise FileNotFoundError(f"Audio not found: {args.audio}")
    
    # Process audio with optimized memory usage
    feats = process_audio_in_chunks(
        model, 
        tokenizer, 
        args.audio, 
        device, 
        model_cfg, 
        start=args.start, 
        end=args.end
    )
    
    # Generate with memory-efficient settings
    prompt = "Transcribe the speech from the fr language into text, then Translate it into en:"
    raw = generate_with_memory_efficiency(model, tokenizer, feats, prompt, device)
    
    # Clean up GPU memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    print("\n" + "#"*40)
    i0 = raw.find("**")
    result = raw[i0:].strip() if i0 != -1 else raw.strip()
    print(result)
    print("#"*40 + "\n")


if __name__ == "__main__":
    main()