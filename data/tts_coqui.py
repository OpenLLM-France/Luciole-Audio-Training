
import torch
import random
from TTS.api import TTS
import torchaudio
from audio import SpeechAugment
import numpy as np

# Load a multilingual / French model with multi-speaker support
# XTTS v2 is multilingual (supports French and many speakers)
# You could also try "tts_models/fr/mai/tacotron2-DDC" (single-speaker, but French only)
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"

tts = TTS(model_name, gpu=True)

# List available speakers
# speakers = tts.speaker_manager.speakers
speakers = list(tts.synthesizer.tts_model.speaker_manager.speakers.keys())
# print("Available speakers:", speakers)


global _tts_model
_tts_model = {}

global _speakers
_speakers = {}

global _noiser
_noiser = None

def text_to_speech(
    text,
    device=None,
    repo_id="tts_models/multilingual/multi-dataset/xtts_v2",
    language="fr",
    model_sampling_rate=24_000,
    sampling_rate=16_000,
    add_noise=False,
):
    global _tts_model, _speakers, _noiser
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if repo_id not in _tts_model:
        _tts_model[repo_id] = TTS(repo_id, gpu=device.startswith("cuda"))
    if repo_id not in _speakers:
        _speakers[repo_id] = list(_tts_model[repo_id].synthesizer.tts_model.speaker_manager.speakers.keys())
    tts = _tts_model[repo_id]
    speaker = random.choice(_speakers[repo_id])

    audio_tensor = np.array(tts.tts(text=text, speaker=speaker, language=language), dtype="float32")

    if sampling_rate != model_sampling_rate:
        audio_tensor = torch.from_numpy(audio_tensor)
        audio_tensor = torchaudio.transforms.Resample(model_sampling_rate, sampling_rate)(audio_tensor)
        audio_tensor = audio_tensor.numpy()

    if add_noise:
        global _noiser
        if _noiser is None:
            _noiser = SpeechAugment()
        audio_tensor = _noiser(audio_tensor, 16000)

    return audio_tensor
    

if __name__ == "__main__":

    import argparse
    import os
    random.seed(1234)

    from audio import save_audio
    parser = argparse.ArgumentParser()
    parser.add_argument("words", type=str, nargs="+", help="Text to convert to speech")
    parser.add_argument("--device", type=str, default=None, help="Device to use for inference")
    parser.add_argument("--language", type=str, default="fr", help="Language of the text, e.g. 'fr' for French, 'en' for English, etc. ")
    parser.add_argument("--output", type=str, default="out", help="Output folder name")
    parser.add_argument("--num", type=int, default=10, help="Number of generations")
    args = parser.parse_args()

    if len(args.words) == 1 and os.path.isfile(args.words[0]):
        with open(args.words[0], "r") as f:
            text = f.read().strip()
    else:
        text = " ".join(args.words)

    for i in range(args.num):
        audio_tensor = text_to_speech(text, device=args.device, add_noise=True)
        os.makedirs(args.output, exist_ok=True)
        save_audio(os.path.join(args.output, f"audio_{i:03d}.wav"), audio_tensor)