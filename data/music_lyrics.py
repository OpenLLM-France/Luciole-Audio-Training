import os
import re
import random

# Relative import
import sys

import datasets
import pandas

sys.path.append(os.path.dirname(__file__))
from audio import (
    combine_audios_factory,
    conform_audio,
    cut_audio,
    load_audio,
    reverberation_factory,
)
from tts import text_to_speech
from transcribe import transcribe_with_cache

# This compiles several Musical Genre Classification datasets, building formatted instructions
# - https://huggingface.co/datasets/gmenon/slt-lyrics-audio

def music_lyrics_instruct_data_iterator(
    streaming=True,
    system_prompt=None,
    debug_folder=None,
    language="en",
    proba_vocal=0.5,
    **kwargs):

    instruction_file = os.path.join(
        os.path.dirname(__file__),
        "assets",
        f"instruction_music_lyrics_{language}_written.txt"
    )
    assert os.path.exists(instruction_file), f"File not found: {instruction_file}"
    with open(instruction_file) as f:
        instructions_written = [line.strip() for line in f.read().split("\n") if line.strip()]

    instruction_file = os.path.join(
        os.path.dirname(__file__),
        "assets",
        f"instruction_music_lyrics_{language}_spoken.txt"
    )
    assert os.path.exists(instruction_file), f"File not found: {instruction_file}"
    with open(instruction_file) as f:
        instructions_spoken = [line.strip() for line in f.read().split("\n") if line.strip()]

    REPOS = [
        (
            "DynamicSuperb/MusicGenreClassification_FMA",
            ["test"]
        ),
        (
            "lewtun/music_genres",
            ["train", "test"]
        ),
        (
            "mteb/music-genre",
            ["test"]
        ),
        (
            "MahiA/GT-Music-Genre",
            ["train", "test"]
        ),
    ]
    for lyrics in [
        "Punk", "Jazz", "Country", "Soul-RnB"
    ]:
        REPOS.append(
            (
                f"ylacombe/music_lyricss_{lyrics}",
                ["train"]
            )
        )

    for repo, splits in REPOS:
        for split in splits:

            yield repo + "--" + split

            ds = datasets.load_dataset(
                repo,
                streaming=streaming,
                split=split,
                **kwargs
            )
            for sample in ds:
                vocal = random.random() < proba_vocal
                instruction = random.choice(instructions_written) if not vocal else random.choice(instructions_spoken)
                data = make_data_instruct(repo,
                    instruction,
                    sample["audio"] if "audio" in sample else sample,
                    system_prompt=system_prompt,
                    language=language,
                    vocal=vocal,
                    debug_folder=debug_folder,
                )
                if data is not None:
                    yield data


global _add_reverb_to_clip
_add_reverb_to_clip = None
global _add_reverb_to_recording
_add_reverb_to_recording = None
global _audios_combiner
_audios_combiner = None

def make_data_instruct(
    repo,
    instruction,
    audio,
    sampling_rate=16_000,
    system_prompt=None,
    language="en",
    vocal=False,
    debug_folder=None,
    ):
    global _add_reverb_to_clip, _add_reverb_to_recording, _audios_combiner

    # Compile audio
    assert isinstance(audio, dict)
    if isinstance(audio, dict) and "array" in audio:
        # Example: https://huggingface.co/datasets/DynamicSuperb/MusicGenreClassification_FMA
        music_clip = audio["array"]
        audio_path = audio["path"]
        actual_sampling_rate = audio.get("sampling_rate", sampling_rate)
        if sampling_rate and actual_sampling_rate != sampling_rate:
            music_clip = conform_audio(music_clip, actual_sampling_rate, sampling_rate=sampling_rate)
            actual_sampling_rate = sampling_rate
        sampling_rate = actual_sampling_rate
    else:
        assert "path" in audio, f"Unexpected audio format: {audio} of type {type(audio)}"
        # Example: https://huggingface.co/datasets/MahiA/GT-Music-Genre
        audio_path = audio["path"]
        pulled_folder = os.path.basename(repo)
        if not os.path.isdir(pulled_folder):
            cmd = f"git clone https://huggingface.co/{repo}"
            raise NotImplementedError(f"You must run:\n{cmd}")
        sampling_rate = 16_000
        music_clip = load_audio(os.path.join(pulled_folder, audio_path), sampling_rate=sampling_rate)
    assert sampling_rate
    assert len(music_clip)

    music_clip = cut_audio(music_clip, sampling_rate, duration=(10, 90))

    transcript = transcribe_with_cache(debug_folder, music_clip).strip()

    # Remove all non word characters
    transcript_norm = re.sub(r"[^a-zA-Z0-9\s]", "", transcript)
    transcript_norm = re.sub(r"\s+", " ", transcript_norm).strip()

    num_words = len(transcript_norm.split())
    num_lines = len(transcript_norm.split("\n"))
    if num_lines == 1 and (
        not transcript_norm
        or num_words < 3
        or any(expr in transcript.lower() for expr in ["thank", "music", "right back", "production"])):
        # Skip short transcripts that are probably not relevant (hallucinations of Whisper ASR)
        print(f"Skipping short transcript: '{transcript}'")
        return None

    if _add_reverb_to_clip is None:
        _add_reverb_to_clip = reverberation_factory(
            rir_scale_factor = (0.5, 1.0),
            gain_scaling_factor = (-20, 6),
        )
    if _add_reverb_to_recording is None:
        _add_reverb_to_recording = reverberation_factory(
            rir_scale_factor = (0.7, 1.0),
        )
    if _audios_combiner is None:
        _audios_combiner = combine_audios_factory()

    if vocal:
        description = {
            "en": [
                # Note: the Parler-TTS models were trained on description prompts mentioning "male/female speaker"
                #       see https://huggingface.co/datasets/parler-tts/mls-eng-speaker-descriptions
                "A female speaker asks a question.",
                "A male speaker asks a question.",
                "A woman is asking a question.",
                "A man is asking a question.",
                "A girl is asking a question.",
                "A boy is asking a question.",
                "A child is giving an instruction",
                "A girl is providing instructions",
            ],
            "fr": [
                "Une femme pose une question.",
                "Un homme pose une question.",
                "Une femme donne une instruction.",
                "Un homme donne une instruction.",
                "Une fille pose une question.",
                "Un garçon pose une question.",
                "Un enfant pose une question.",
            ],
        }[language]
        description = random.choice(description)
        audio_instruction = text_to_speech(instruction, description)
        music_clip = _add_reverb_to_clip(music_clip, sampling_rate)
        _audios_combiner.music_clip = music_clip
        final_waveform = _audios_combiner(audio_instruction, sampling_rate)
        add_reverb = _add_reverb_to_recording

    else:
        final_waveform = music_clip
        add_reverb = _add_reverb_to_clip

    final_waveform = add_reverb(final_waveform, sampling_rate)

    ########################################################
    # Here is defined the output format (list of messages)

    audio_data = {"type": "audio", "array": final_waveform, "sampling_rate": sampling_rate}
    if audio_path:
        audio_data["path"] = audio_path

    instruction_before = random.random() < 0.5 if not vocal else None
    full_instruction = []
    if instruction_before is True:
        full_instruction.append({"type": "text", "text": instruction})
    full_instruction.append(audio_data)
    if instruction_before is False:
        full_instruction.append({"type": "text", "text": instruction})

    return (
        [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}] if system_prompt else []
    ) + [
        {"role": "user", "content": full_instruction},
        {"role": "assistant","content": [{"type": "text", "text": transcript}]}
    ]

    ########################################################

def string_to_integer(s: str) -> int:
    return abs(hash(s))


def main_dump_parquet():
    import argparse

    parser = argparse.ArgumentParser(description="Dump Music Genre Classification datasets")
    parser.add_argument("--output", default="out", help="Output folder")
    parser.add_argument("--language", default="en", type=str, help="Language")
    parser.add_argument("--max_docs", default=None, type=int, help="Maximum number of documents")
    parser.add_argument("--num_docs_per_parquet", default=200, type=int, help="Number of documents per parquet")
    parser.add_argument("--debug_folder", default=None, help="Debug folder")
    args = parser.parse_args()

    global messages, first_idx, last_idx, dataset_name, explicit_subname
    messages = []
    first_idx = last_idx = 0
    dataset_name = "data"
    explicit_subname = False
    def _flush():
        global messages, first_idx, last_idx, dataset_name, explicit_subname
        if messages:
            last_idx += len(messages)
            output_filename = os.path.join(
                args.output,
                f"MusicGenre--{dataset_name}--{first_idx:06d}-{last_idx:06d}.parquet"
            )
            print(f"Dumping {output_filename} ...")
            os.makedirs(args.output, exist_ok=True)
            pandas.DataFrame({"messages": messages}).to_parquet(output_filename)
            first_idx = last_idx
            messages = []

    for i, data in enumerate(
        music_lyrics_instruct_data_iterator(
            language=args.language,
            debug_folder=args.debug_folder
        )):
        if args.max_docs and i >= args.max_docs:
            break
        if isinstance(data, str):
            explicit_subname = True
            new_dataset_name = data.replace("/", "--")
            if new_dataset_name != dataset_name:
                _flush()
                dataset_name = new_dataset_name
                # Reset indices
                first_idx = last_idx = 0
                # Pseudo-deterministic randomness for each dataset
                random.seed(string_to_integer(dataset_name))
            continue
        messages.append(data)
        if len(messages) >= args.num_docs_per_parquet:
            _flush()
    _flush()


if __name__ == "__main__":
    main_dump_parquet()
