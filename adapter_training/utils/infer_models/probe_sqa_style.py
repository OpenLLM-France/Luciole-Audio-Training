"""Probe how concurrent multimodal speech+text models answer SLU SQA questions.

Goal: eyeball whether SOTA models respond with full sentences
(e.g. "The word \"fox\" is spoken at 5.42 seconds.") or bare values
(e.g. "5.42"), so we can decide which answer style to use in our
synthetic SQA data (see data/synthetic/generate_slu_variants.py).

Example:
    python utils/infer_models/probe_sqa_style.py \\
        --audio /path/to/clip.wav \\
        --question 'At what time in the audio is the word "fox" spoken?' \\
        --model qwen2-audio --model phi4-mm --model qwen2.5-omni
"""

import argparse
import gc
import os
import sys

import librosa
import numpy as np
import torch


MODEL_CHOICES = ("qwen2-audio", "phi4-mm", "qwen2.5-omni")


def load_audio_16k(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=16000, mono=True)
    return np.ascontiguousarray(audio.astype(np.float32))


def run_qwen2_audio(audio: np.ndarray, question: str, device) -> str:
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2AudioForConditionalGeneration,
    )

    model_id = "Qwen/Qwen2-Audio-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map={"": device},
    )
    model.eval()

    conversation = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio},
            {"type": "text", "text": question},
        ],
    }]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    with torch.no_grad():
        inputs = processor(
            text=text,
            audio=[audio],
            return_tensors="pt",
            sampling_rate=16000,
            padding=True,
        ).to(device)
        generated = model.generate(**inputs, max_new_tokens=150)
        new_tokens = generated[:, inputs.input_ids.size(1):]
    response = processor.batch_decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    del model, processor
    return response


def run_phi4_mm(audio: np.ndarray, question: str, device) -> str:
    from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

    model_id = "microsoft/Phi-4-multimodal-instruct"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        _attn_implementation="eager",
    ).to(device)
    model.eval()

    try:
        gen_config = GenerationConfig.from_pretrained(model_id)
    except Exception:
        gen_config = GenerationConfig()

    prompt = f"<|user|><|audio_1|>{question}<|end|><|assistant|>"
    with torch.no_grad():
        inputs = processor(
            text=prompt,
            audios=[(audio, 16000)],
            return_tensors="pt",
        ).to(device)
        generated = model.generate(
            **inputs,
            max_new_tokens=200,
            generation_config=gen_config,
        )
        new_tokens = generated[:, inputs.input_ids.size(1):]
    response = processor.batch_decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    del model, processor
    return response


def run_qwen25_omni(audio: np.ndarray, question: str, device) -> str:
    try:
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    except ImportError:
        return ("[skipped] transformers is too old for Qwen2.5-Omni. "
                "Upgrade with: pip install -U 'transformers>=4.52'")

    model_id = "Qwen/Qwen2.5-Omni-7B"
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": device},
    )
    model.eval()

    conversation = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio},
            {"type": "text", "text": question},
        ],
    }]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    with torch.no_grad():
        inputs = processor(
            text=text,
            audio=[audio],
            return_tensors="pt",
            sampling_rate=16000,
            padding=True,
        ).to(device)
        try:
            generated = model.generate(
                **inputs, max_new_tokens=200, return_audio=False
            )
        except TypeError:
            generated = model.generate(**inputs, max_new_tokens=200)
        if isinstance(generated, tuple):
            generated = generated[0]
        new_tokens = generated[:, inputs.input_ids.size(1):]
    response = processor.batch_decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    del model, processor
    return response


RUNNERS = {
    "qwen2-audio": run_qwen2_audio,
    "phi4-mm": run_phi4_mm,
    "qwen2.5-omni": run_qwen25_omni,
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--audio", required=True, help="Path to a .wav/.mp3/.flac file.")
    parser.add_argument("--question", required=True, help="The SQA question to ask.")
    parser.add_argument(
        "--model",
        action="append",
        choices=MODEL_CHOICES,
        default=None,
        help="Model to test; pass multiple times to test several (default: qwen2-audio).",
    )
    parser.add_argument("--gpus", type=int, default=0, help="CUDA device index.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = args.model or ["qwen2-audio"]
    audio = load_audio_16k(args.audio)

    print(f"# audio: {args.audio}  ({len(audio)/16000:.2f}s)")
    print(f"# question: {args.question}\n")

    for name in models:
        print(f"=== {name} ===", flush=True)
        try:
            response = RUNNERS[name](audio, args.question, device)
            print(response)
        except Exception as e:
            print(f"[error] {type(e).__name__}: {e}", file=sys.stderr)
        print()
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
