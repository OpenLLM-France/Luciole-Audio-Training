import os
import sys
import json
import logging
import csv
from textwrap import dedent

import torch
from torch.utils.data import DataLoader

import whisper
from peft import PeftModel

from configs import TrainConfig, ModelConfig
from utils.env import auto_device
from models.lucas_setup import model_factory
from models.lucas import get_embeddings

from data.audio_chuncks_dataset_new import get_dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_model_and_tokenizer(model_dir: str, device: torch.device):
    """Load model and tokenizer from the specified directory."""
    from utils.config_utils import load_all_configs
    mc_path = os.path.join(model_dir, "model_config.json")
    tc_path = os.path.join(model_dir, "train_config.json")
    
    mc, tc = load_all_configs(mc_path, tc_path)
    
    logger.info("Building model...")
    model, tokenizer = model_factory(tc, mc, inference_mode=True)
    
    if device.type == 'cuda':
        model.encoder = model.encoder.to('cpu')
        model.encoder_projector = model.encoder_projector.to('cpu')
        model.llm = model.llm.to(device)
    else:
        model = model.to(device)
    
    logger.info(f"Loading checkpoint from {model_dir}")
    from utils.checkpoint_utils import load_model_checkpoint_peft
    model, _, _, _, _ = load_model_checkpoint_peft(model, model_dir)
    model.eval()
    
    if "peft" in str(type(model.llm)).lower():
        logger.info(f"✅ PEFT enabled. LLM type: {type(model.llm)}")
    else:
        logger.warning(f"🚫 PEFT NOT enabled. LLM type: {type(model.llm)}")
    
    return model, tokenizer, mc, tc

def load_data(dataset_dirs, model_config, train_config, tokenizer, is_validation=False):
    logger.info(f"Loading test dataset from: {dataset_dirs}")
    
    dataset = get_dataset(
        dataset_dir=dataset_dirs,
        tokenizer=tokenizer,
        model_config=model_config,
        train_config=train_config,
        split="test",
        inference_mode=True
    )
    
    logger.info(f"Dataset size: {len(dataset)} examples")
    
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=dataset.data_collator,
        pin_memory=False
    ), dataset

def prepare_inputs(model, sample, device, encoder_dtype):
    if 'audio' in sample:
        sample['audio'] = sample['audio'].to(device=device, dtype=encoder_dtype)
    
    for key in ['input_ids', 'attention_mask', 'labels']:
        if key in sample:
            sample[key] = sample[key].to(device)
    
    return sample

def split_audio_tensor(audio_tensor, chunk_size=3000):
    """
    Splits an audio tensor along the time dimension into fixed-size chunks.
    
    Args:
        audio_tensor: Audio tensor with shape [batch, time, mel]
        chunk_size: Maximum chunk size for the time dimension
        
    Returns:
        List of tensors with shape [batch, time_chunk, mel]
    """
    batch_size, time_length, mel_dim = audio_tensor.shape
    splits = []

    # Process in chunk_size increments
    for start in range(0, time_length, chunk_size):
        end = min(start + chunk_size, time_length)
        chunk = audio_tensor[:, start:end, :]
        splits.append(chunk)

    return splits

def process_audio_batch(encoder, audio, chunk_processing=True, max_chunk_size=3000):
    """
    Process an audio batch through the encoder, handling chunking if needed.
    
    Args:
        encoder: Audio encoder model
        audio: Audio tensor [batch_size, num_chunks, frames_per_chunk, n_mels]
        chunk_processing: Whether to process in chunks even with single chunk
        max_chunk_size: Maximum chunk size when processing large inputs
        
    Returns:
        Encoder features [batch_size, seq_len, hidden_size]
    """
    bsz, num_chunks, frames_per_chunk, n_mels = audio.shape
    
    # Case 1: Multiple chunks in input
    if num_chunks > 1:
        logger.debug(f"Processing {num_chunks} audio chunks separately")
        # Process each chunk individually and concatenate
        encoder_feats = torch.cat([
            encoder.extract_variable_length_features(audio[:, i].permute(0, 2, 1))
            for i in range(num_chunks)
        ], dim=1)
        
    # Case 2: Single chunk but we should process in smaller pieces
    elif chunk_processing and frames_per_chunk > max_chunk_size:
        logger.debug(f"Single large chunk detected, processing in {frames_per_chunk//max_chunk_size+1} parts")
        # Flatten the audio tensor
        audio_flat = audio.reshape(bsz, -1, n_mels).permute(0, 2, 1)
        # Split into smaller chunks
        chunks = split_audio_tensor(audio_flat.permute(0, 2, 1), chunk_size=max_chunk_size)
        # Process each chunk and concatenate
        encoder_feats = torch.cat([
            encoder.extract_variable_length_features(chunk.permute(0, 2, 1))
            for chunk in chunks
        ], dim=1)
    
    # Case 3: Single chunk, process directly
    else:
        logger.debug("Processing single audio chunk")
        audio_flat = audio.reshape(bsz, -1, n_mels).permute(0, 2, 1)
        encoder_feats = encoder.extract_variable_length_features(audio_flat)
    
    return encoder_feats

if __name__ == '__main__':
    from tqdm import tqdm
    if len(sys.argv) < 3:
        print("Usage: python inference_from_dataloader.py <model_path> <dataset_dir>")
        sys.exit(1)

    model_path = sys.argv[1]
    dataset_dir = sys.argv[2]

    device = auto_device()
    model, tokenizer, model_config, train_config = load_model_and_tokenizer(model_path, device)
    model.to(device)
    model.eval()
    torch.manual_seed(train_config.seed)

    test_loader, _ = load_data(
        dataset_dirs=dataset_dir,
        model_config=model_config,
        train_config=train_config,
        tokenizer=tokenizer,
        is_validation=True
    )

    logger.info("Starting batch inference on test dataset...")
    data_name = os.path.basename(os.path.normpath(dataset_dir))
    results_dir = os.path.join(model_path, "results")
    os.makedirs(results_dir, exist_ok=True)

    tsv_path = os.path.join(results_dir, f"predictions_{data_name}.tsv")
    write_header = not os.path.exists(tsv_path)  # Only write header if file doesn't exist

    with open(tsv_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "prediction", "reference"], delimiter="\t")
        if write_header:
            writer.writeheader()
        
        with torch.inference_mode():
            for idx, sample in enumerate(tqdm(test_loader, desc="Running inference")):
                try:
                    sample = prepare_inputs(model, sample, device, encoder_dtype=torch.float32)
                    audio = sample["audio"]
                    with torch.no_grad():
                        encoder_outs = process_audio_batch(model.encoder, audio)

                    if encoder_outs.ndim == 4 and encoder_outs.shape[1] == 1:
                        encoder_outs = encoder_outs.squeeze(1)  # now shape is [B, T, D]

                    encoder_outs = model.encoder_projector(encoder_outs)
                    encoder_outs = encoder_outs.to(dtype=torch.float32)

                    user_text="Fournissez une transcription propre de cette conversation en français."
                    prompt_template = (
                        "<|start_header_id|>user<|end_header_id|>\n"
                        "{user_message}\n<|eot_id|>\n\n"
                        "<|start_header_id|>assistant<|end_header_id|>\n"
                    )
                    formatted_prompt = prompt_template.format(
                                            user_message=user_text.strip()
                                        )
                        
                    prompt_ids = tokenizer.encode(formatted_prompt)
                    prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64, device=device)

                    inputs_embeds = get_embeddings(model.llm, prompt_ids)
                    inputs_embeds = torch.cat((encoder_outs, inputs_embeds[None, :, :]), dim=1).to(dtype=model.llm.dtype)
                    # print("inputs_embeds: ",inputs_embeds)
                    attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long).to(inputs_embeds.device)
                    # print("attention_mask: ",attention_mask)
                    with torch.no_grad():
                        model_outputs = model.generate(
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            use_cache=True,
                            max_new_tokens=125,
                        )
                    # print("Model Output: ",model_outputs)
                    pred_text = tokenizer.batch_decode(model_outputs, skip_special_tokens=True)[0]

                    # print("#########################################################################")
                    # print(f"[{idx}] PRED: {pred_text.strip()}")
                    if "labels" in sample:
                        labels = sample["labels"]
                        if labels.ndim > 1:
                            labels = labels[0]
                        label_text = tokenizer.decode(labels[labels != -100].tolist(), skip_special_tokens=True)
                        # print(f"[{idx}] REF:  {label_text}")
                    else:
                        label_text = ""
                    # print("#########################################################################")

                    # ✅ Write single row to file and flush
                    writer.writerow({
                        "index": idx,
                        "prediction": pred_text.strip(),
                        "reference": label_text,
                    })
                    f.flush()

                except Exception as e:
                    logger.error(f"❌ Error processing sample {idx}: {e}")
