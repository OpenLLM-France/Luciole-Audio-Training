import os
import random
import datasets
import pandas
import slugify

# Relative import
import sys
sys.path.append(os.path.dirname(__file__))
from audio import (
    load_audio,
    conform_audio,
    reverberation_factory,
    combine_audios_factory,
    cut_audio,
    save_audio,
)
from tts import text_to_speech

# This compiles several Musical Genre Classification datasets, building formatted instructions
# - https://huggingface.co/datasets/DynamicSuperb/MusicGenreClassification_FMA
# - https://huggingface.co/datasets/lewtun/music_genres
# - https://huggingface.co/datasets/mteb/music-genre
# - https://huggingface.co/datasets/MahiA/GT-Music-Genre
# - https://huggingface.co/datasets/ylacombe/music_genres_XXX (where XXX is Jazz, Punk, Country, ...)

def music_genre_instruct_data_iterator(
    streaming=True,
    system_prompt=None,
    debug_folder=None,
    proba_vocal=0.7,
    **kwargs):

    instruction_file = os.path.join(
        os.path.dirname(__file__),
        "assets",
        "instruction_music_genre_en_written.txt"
    )
    assert os.path.exists(instruction_file), f"File not found: {instruction_file}"
    with open(instruction_file) as f:
        instructions_written = [line.strip() for line in f.read().split("\n") if line.strip()]

    instruction_file = os.path.join(
        os.path.dirname(__file__),
        "assets",
        "instruction_music_genre_en_spoken.txt"
    )
    assert os.path.exists(instruction_file), f"File not found: {instruction_file}"
    with open(instruction_file) as f:
        instructions_spoken = [line.strip() for line in f.read().split("\n") if line.strip()]

    REPOS_INSTRUCTS = [
        (
            "DynamicSuperb/MusicGenreClassification_FMA",
            ["test"], "label",
        )
    ]

    # Already prepared dataset
    for repo, splits, label in REPOS_INSTRUCTS:
        for split in splits:
            ds = datasets.load_dataset(
                repo,
                streaming=streaming,
                split="test",
                **kwargs
            )

            yield repo + "--" + split

            for sample in ds:
                yield make_data_instruct(repo,
                    sample["instruction"],
                    sample[label],
                    sample["audio"],
                    system_prompt=system_prompt,
                    vocal=False,
                    debug_folder=debug_folder,
               )

    REPOS_CLASSIFICATION_TASK = [
        (
            "lewtun/music_genres",
            ["train", "test"], "genre", None
        ),
        (
            "mteb/music-genre",
            ["test"], "label", {
                0: "Alternative",
                1: "Blues",
                2: "Electronic",
                3: "Folk / Country",
                4: "Soul / R&B", # Funk / ...?
                5: "Jazz",
                6: "Pop",
                7: "Rap / Hip-Hop",
                8: "Rock",
            }
        ),
        (
            "MahiA/GT-Music-Genre",
            ["train", "test"], "classname", None
        ),
    ]
    for genre in [
        "Punk", "Jazz", "Country", "Soul-RnB"
    ]:
        REPOS_CLASSIFICATION_TASK.append(
            (
                f"ylacombe/music_genres_{genre}",
                ["train"], "genre", None
            )
        )

    for repo, splits, label, label_dict in REPOS_CLASSIFICATION_TASK:
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
                yield make_data_instruct(repo,
                    instruction,
                    format_genre(sample[label], label_dict=label_dict),
                    sample["audio"] if "audio" in sample else sample,
                    system_prompt=system_prompt,
                    vocal=vocal,
                    debug_folder=debug_folder,
                )


global _add_reverb_to_clip
_add_reverb_to_clip = None
global _add_reverb_to_recording
_add_reverb_to_recording = None
global _audios_combiner
_audios_combiner = None

def make_data_instruct(
    repo,
    instruction,
    answer,
    audio,
    sampling_rate=16_000,
    system_prompt=None,
    vocal=False,
    label_dict=None,
    debug_folder=None,
    ):

    answer = format_genre(answer, label_dict=label_dict, repo=repo)

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


    global _add_reverb_to_clip
    if _add_reverb_to_clip is None:
        _add_reverb_to_clip = reverberation_factory(
            rir_scale_factor = (0.5, 1.0),
            gain_scaling_factor = (-20, 6),
        )
    global _add_reverb_to_recording
    if _add_reverb_to_recording is None:
        _add_reverb_to_recording = reverberation_factory(
            rir_scale_factor = (0.7, 1.0),
        )
    global _audios_combiner
    if _audios_combiner is None:
        _audios_combiner = combine_audios_factory()

    if vocal:
        audio_instruction = text_to_speech(
            instruction,
            [
                "A female asks a question.",
                "A male asks a question.",
            ]
        )
        music_clip = cut_audio(music_clip, sampling_rate=sampling_rate, duration=(4 + len(audio_instruction), 12 + len(audio_instruction)))
        music_clip = _add_reverb_to_clip(music_clip, sampling_rate)
        add_reverb = _add_reverb_to_recording
        _audios_combiner.music_clip = music_clip
        final_waveform = _audios_combiner(audio_instruction, sampling_rate)

    else:
        final_waveform = cut_audio(music_clip, sampling_rate, duration=(4, 12))
        add_reverb = _add_reverb_to_clip

    if debug_folder:
        # Dump audio for manual inspection (debug, check, ...)
        if vocal:
            debug_folder = os.path.join(debug_folder, "vocal")
        else:
            debug_folder = os.path.join(debug_folder, "textual")
        os.makedirs(debug_folder, exist_ok=True)
        audio_filename = os.path.join(debug_folder, f"{slugify.slugify(answer)}_{slugify.slugify(instruction)}.wav")
        save_audio(audio_filename, final_waveform, sampling_rate=sampling_rate)

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
        {"role": "assistant","content": [{"type": "text", "text": answer}]}
    ]

    ########################################################


def format_genre(genre, proba_this_is=0, label_dict=None, repo="UNK"):
    if isinstance(genre, int):
        # https://huggingface.co/datasets/mteb/music-genre
        assert label_dict, f"Missing label dictionary for {repo}"
        genre_name = label_dict.get(genre)
        assert genre_name, f"Unknown genre: {genre}"
        genre = genre_name

    genre = genre.replace("Soul-RnB", "Soul / R&B")
    if " / " in genre:
        genres = genre.split(" / ")
        if len(genres) == 2:
            genre1, genre2 = genres
            if random.random() <= 0.5:
                genre1, genre2 = genre2, genre1
            genres.append(f"{genre1} (or {genre2})")
        genre = random.choice(genres)
    if proba_this_is and random.random() <= proba_this_is:
        # We could do this, but it looks like a false good idea
        # Having a LLM that produces short answers is ... good :D
        return "This is " + genre.lower()
    else: # Capitalize
        genre = genre[0].upper() + genre[1:]
    return genre




def string_to_integer(s: str) -> int:
    return abs(hash(s))


def main_dump_parquet():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="out", help="Output folder")
    parser.add_argument("--max_docs", default=200, type=int, help="Number of documents per parquet")
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

    for data in music_genre_instruct_data_iterator(debug_folder=args.debug_folder):
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
        if len(messages) >= args.max_docs:
            _flush()
    _flush()


if __name__ == "__main__":
    main_dump_parquet()
