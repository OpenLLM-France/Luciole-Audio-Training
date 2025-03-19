import os
import sys
import json
from textwrap import dedent

from configs import TrainConfig, ModelConfig

from utils.env import *
from utils.audio import load_audio
from utils.checkpoint_utils import load_model_checkpoint_peft

from models.lucas_setup import model_factory
from models.lucas import get_embeddings

from data.dataset import *

import whisper
from peft import PeftModel

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_model_and_checkpoint(checkpoint_path, model_config, train_config, device):
    # Initialize the model
    logging.info("Initializing model...")
    kwargs = {"inference_mode": True}
    model, tokenizer = model_factory(train_config, model_config, **kwargs)
    model.to(device)

    if os.path.isdir(checkpoint_path) and os.listdir(checkpoint_path):
        logger.info("--> Loading the latest checkpoint.")
        output_dir = checkpoint_path

        # Filter for directories matching the naming convention
        directories = [
            d for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d)) and d.startswith(train_config.model_name)
        ]

        # Sort directories by their numeric suffix in descending order
        try:
            directories.sort(key=lambda d: int(d.split('_')[-1]), reverse=True)
        except ValueError as e:
            logger.error(f"Error sorting directories. Ensure proper naming convention (e.g., '{train_config.model_name}').")
            directories = []

        # Select the latest checkpoint folder
        latest_epoch_folder = directories[0] if directories else None
        logger.info(f'--> The last checkpoint done was {latest_epoch_folder}')

        if latest_epoch_folder:
            model, train_config, model_config = load_model_checkpoint_peft(
                model, train_config, model_config, checkpoint_name=latest_epoch_folder
            )
        else:
            logger.info("No valid checkpoint found.")
    else:
        logger.info("Output directory is empty or does not exist.")
    
    model.eval()
    
    return model, tokenizer

if __name__ == '__main__':
    
    model_path = sys.argv[1] 
    wave_path = sys.argv[2] 
    
    model_config, train_config = ModelConfig(), TrainConfig()
    
    device = auto_device()
    torch.manual_seed(train_config.seed)

    model, tokenizer = load_model_and_checkpoint(model_path, model_config, train_config, device)
    model.to(device)
    model.eval()
    
    # Validate wave_path
    if not wave_path or not os.path.exists(wave_path):
        raise ValueError(f"Invalid wave_path: {wave_path}. Please provide a valid audio file path.")

    # Load and preprocess audio
    logging.info("Loading and preprocessing audio...")
    audio_raw = load_audio(wave_path, )
    audio_raw = whisper.pad_or_trim(audio_raw)

    mel_size = model_config.mel_size
    audio_mel = whisper.log_mel_spectrogram(audio_raw, n_mels=mel_size).permute(1, 0)
    
    
    audio_mel = audio_mel.unsqueeze(0).permute(0, 2, 1).to(device)  # Shape: (1, 80, 3000)
     
    logging.info("Generating transcription...")
    with torch.no_grad():
        # Extract features and project them
        logging.info("Extracting features from audio...")
        encoder_outs = model.encoder.extract_variable_length_features(audio_mel)
        encoder_outs = model.encoder_projector(encoder_outs)

        # Ensure dtype consistency
        encoder_outs = encoder_outs.to(dtype=torch.float32)

        prompt = f"Transcribe the speech from the fr language into text, and can you analyse the text pls"
            
        prompt_ids = tokenizer.encode(prompt)
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64).to(device)

        inputs_embeds = get_embeddings(model.llm, prompt_ids)
        logging.info(f"Debug: inputs_embeds.shape: {inputs_embeds.shape if inputs_embeds is not None else 'None'}")
        # print(inputs_embeds)

        inputs_embeds = torch.cat((encoder_outs, inputs_embeds[None, :, :]), dim=1).to(dtype=model.llm.dtype)
        # Attention mask
        attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long).to(inputs_embeds.device)

        # Ensure dimensions match
        assert attention_mask.shape[1] == inputs_embeds.shape[1], "Attention mask and inputs_embeds dimensions mismatch"

        try:
            model_outputs = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache = True,
            )
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise

    Preds = tokenizer.batch_decode(model_outputs, skip_special_tokens=True)[0]
    
    print('#########################################################################')
    print(Preds)
    print('#########################################################################')