import os
import sys
import json
import logging
import torch
from textwrap import dedent

from configs import TrainConfig, ModelConfig
from utils.env import auto_device
from utils.audio import load_audio
from utils.checkpoint_utils import load_model_checkpoint_peft
from models.lucas_setup import model_factory
from models.lucas import get_embeddings
from utils.config_utils import load_all_configs
from data.dataset import *
import whisper

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_model_and_tokenizer(model_dir: str, device: torch.device):
    """Load model and tokenizer from the specified directory."""
    # Load configurations
    mc_path = os.path.join(model_dir, "model_config.json")
    tc_path = os.path.join(model_dir, "train_config.json")
    
    model_config, train_config = load_all_configs(mc_path, tc_path)

    logger.info("Building model...")
    model, tokenizer = model_factory(train_config, model_config, inference_mode=True)
    # set tokenizer vocab size
    
    # Device placement
    if device.type == 'cuda':
        model.encoder = model.encoder.to('cpu')
        model.encoder_projector = model.encoder_projector.to('cpu')
        model.llm = model.llm.to(device)
    else:
        model = model.to(device)

    logger.info(f"Loading checkpoint from {model_dir}")
    model, _, _, _, _ = load_model_checkpoint_peft(model, model_dir)
    model.eval()

    if "peft" in str(type(model.llm)).lower():
        logger.info(f"✅ PEFT enabled. LLM type: {type(model.llm)}")
    else:
        logger.warning(f"🚫 PEFT NOT enabled. LLM type: {type(model.llm)}")

    return model, tokenizer, model_config, train_config


if __name__ == '__main__':
    model_path = sys.argv[1]
    wave_path = sys.argv[2]
    
    device = auto_device()
    model, tokenizer, model_config, train_config = load_model_and_tokenizer(model_path, device)

    model.to(device)
    model.eval()
    torch.manual_seed(train_config.seed)

    # Validate wave_path
    if not wave_path or not os.path.exists(wave_path):
        raise ValueError(f"Invalid wave_path: {wave_path}. Please provide a valid audio file path.")

    logger.info("Loading and preprocessing audio...")
    audio_raw = load_audio(wave_path)
    audio_raw = whisper.pad_or_trim(audio_raw)

    mel_size = model_config.mel_size
    audio_mel = whisper.log_mel_spectrogram(audio_raw, n_mels=mel_size).permute(1, 0)
    audio_mel = audio_mel.unsqueeze(0).permute(0, 2, 1).to(device)

    total_frames = audio_mel.shape[0]
    from data.audio_chuncks_dataset import _make_pseudo_tokens
    tokens = _make_pseudo_tokens(total_frames, tokenizer, -100)

    logger.info("Generating transcription...")
    with torch.no_grad():
        logger.info("Extracting features from audio...")
        encoder_outs = model.encoder.extract_variable_length_features(audio_mel)
        encoder_outs = model.encoder_projector(encoder_outs)
        encoder_outs = encoder_outs.to(dtype=torch.float32)

        user_text = "Transcribe this Audio."
        prompt_template = (
            "<|start_header_id|>user<|end_header_id|>\n"
            "{user_message}\n<|eot_id|>\n\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
        formatted_prompt = prompt_template.format(user_message=user_text.strip())

        prompt_ids = tokenizer.encode(formatted_prompt)
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64).to(device)
        inputs_embeds = get_embeddings(model.llm, prompt_ids)

        logger.info(f"Debug: inputs_embeds.shape: {inputs_embeds.shape if inputs_embeds is not None else 'None'}")
        inputs_embeds = torch.cat((encoder_outs, inputs_embeds[None, :, :]), dim=1).to(dtype=model.llm.dtype)

        attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long).to(inputs_embeds.device)
        assert attention_mask.shape[1] == inputs_embeds.shape[1], "Attention mask and inputs_embeds dimensions mismatch"

        try:
            model_outputs = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                max_new_tokens=100
            )
            print("Model output:", model_outputs)
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise

    Preds = tokenizer.batch_decode(model_outputs, skip_special_tokens=True)[0]
    print('#########################################################################')
    print(Preds)
    print('#########################################################################')
