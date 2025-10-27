import sys
import torch
import whisper
import os
from utils.env import auto_device
from utils.audio import load_audio
from utils.checkpoint_utils import load_model_checkpoint_peft
from models.lucas_setup import model_factory
from utils.config_utils import load_all_configs
from models.lucas import get_embeddings
from data.audio_chuncks_dataset import _make_pseudo_tokens

@torch.inference_mode()
def simple_infer(model_path, audio_path, prompt=""):
    device = auto_device()
    # Load config + model
    mc_path, tc_path = f"{model_path}/model_config.json", f"{model_path}/train_config.json"
    model_config, train_config = load_all_configs(mc_path, tc_path)
    model, tokenizer = model_factory(train_config, model_config, inference_mode=True)
    model, *_ = load_model_checkpoint_peft(model, model_path)
    model = model.to(device).eval()

    # Load + process audio
    raw_audio = whisper.pad_or_trim(load_audio(audio_path))
    mel = whisper.log_mel_spectrogram(raw_audio, n_mels=model_config.mel_size).permute(1, 0)
    mel = mel.unsqueeze(0).permute(0, 2, 1).to(device)

    # Encode audio
    encoder_outs = model.encoder.extract_variable_length_features(mel)
    encoder_outs = model.encoder_projector(encoder_outs).to(dtype=torch.float32)

    # Prompt
    prompt = f"<|start_header_id|>user<|end_header_id|>\n{prompt}\n<|eot_id|>\n\n<|start_header_id|>assistant<|end_header_id|>\n"
    prompt_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).to(device)
    prompt_embed = get_embeddings(model.llm, prompt_ids)[None, :, :]  # [1, seq_len, dim]

    # Join encoder outputs with prompt
    inputs_embeds = torch.cat((encoder_outs, prompt_embed), dim=1).to(dtype=model.llm.dtype)
    attention_mask = torch.ones(inputs_embeds.shape[:-1], dtype=torch.long).to(device)

    # Generate output
    output = model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=100)
    text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    print("🗣️  result:", text)


if __name__ == "__main__":
    simple_infer(sys.argv[1], sys.argv[2], sys.argv[3])
