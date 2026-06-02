import random

import torch
import torchaudio
import numpy as np

from adapter_training.data.audio import SpeechAugment

from ssak.utils.text import numbers_and_symbols_to_letters

global _noiser
_noiser = None


def text_to_speech(
    text,
    device=None,
    voice=None,
    repo_id="kyutai/tts-1.6b-en_fr",
    language="fr",
    model_sampling_rate=24_000,
    sampling_rate=16_000,
    add_noise=False,
):

    # # Check if there are numbers in the text and format them
    if any(char.isdigit() for char in text):
        if language is None:
            raise ValueError("Language must be specified when text contains numbers")
        text = numbers_and_symbols_to_letters(text, lang=language)
        print(f"Formatted text: {text}")


    # Load processor and model from Hugging Face, with caching in (V)RAM
    tts_model = get_tts_model(device=device, repo_id=repo_id)
    if voice is None:
        voice = random.choice(get_tts_voices())

    entries = tts_model.prepare_script([text], padding_between=1)
    voice_path = tts_model.get_voice_path(voice)
    condition_attributes = tts_model.make_condition_attributes(
        [voice_path], cfg_coef=2.0
    )

    result = tts_model.generate([entries], [condition_attributes])
    with tts_model.mimi.streaming(1), torch.no_grad():
        pcms = []
        for frame in result.frames[tts_model.delay_steps :]:
            pcm = tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
            pcms.append(np.clip(pcm[0, 0], -1, 1))
        audio_tensor = np.concatenate(pcms, axis=-1)

    if sampling_rate != model_sampling_rate:
        audio_tensor = torch.from_numpy(audio_tensor)
        audio_tensor = torchaudio.transforms.Resample(model_sampling_rate, sampling_rate)(audio_tensor)
        audio_tensor = audio_tensor.numpy()

    if add_noise:
        global _noiser
        if _noiser is None:
            _noiser = SpeechAugment()
        audio_tensor = _noiser(audio_tensor, sampling_rate)

    return audio_tensor

global tts_model, tts_voices
tts_model = None
tts_voices = None

def get_tts_model(device=None, repo_id="kyutai/tts-1.6b-en_fr"):
    from moshi.models.loaders import CheckpointInfo
    from moshi.models.tts import TTSModel
    global tts_model

    # Set up device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if tts_model is None:
        print("Loading TTS model...")
        checkpoint_info = CheckpointInfo.from_hf_repo(repo_id)
        tts_model = TTSModel.from_checkpoint_info(
            checkpoint_info, n_q=32, temp=0.6, device=device
        )
    return tts_model

def get_tts_voices(repo_id="kyutai/tts-voices"):
    global tts_voices
    if tts_voices is None:
        print("Loading TTS voices...")
        from huggingface_hub import list_repo_files
        tts_voices = [f for f in list_repo_files(repo_id) if f.endswith(".wav")]
    return tts_voices

def preload_all_voices():
    for voice in get_tts_voices():
        get_tts_model().get_voice_path(voice)

if __name__ == "__main__":

    import argparse
    import os

    from adapter_training.data.audio import save_audio
    parser = argparse.ArgumentParser()
    parser.add_argument("words", type=str, nargs="+", help="Text to convert to speech")
    parser.add_argument("--device", type=str, default=None, help="Device to use for inference")
    parser.add_argument("--language", type=str, default="fr", help="Language of the text, e.g. 'fr' for French, 'en' for English, etc. ")
    parser.add_argument("--output", type=str, default="out", help="Output folder name")
    parser.add_argument("--num", type=int, default=10, help="Number of generations")
    parser.add_argument("--preload", action="store_true", help="Preload all voices")
    args = parser.parse_args()

    if len(args.words) == 1 and os.path.isfile(args.words[0]):
        with open(args.words[0], "r") as f:
            text = f.read().strip()
    else:
        text = " ".join(args.words)

    if args.preload:
        preload_all_voices()

    for i in range(args.num):
        audio_tensor = text_to_speech(text, device=args.device, add_noise=True)
        os.makedirs(args.output, exist_ok=True)
        save_audio(os.path.join(args.output, f"audio_{i:03d}.wav"), audio_tensor)

