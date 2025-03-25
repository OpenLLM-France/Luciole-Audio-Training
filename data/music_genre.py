import os
import random
import datasets
import pandas

# Relative import
import sys
sys.path.append(os.path.dirname(__file__))
from audio import load_audio, conform_audio

# This compiles several Musical Genre Classification datasets, building formatted instructions
# - https://huggingface.co/datasets/DynamicSuperb/MusicGenreClassification_FMA
# - https://huggingface.co/datasets/lewtun/music_genres
# - https://huggingface.co/datasets/mteb/music-genre
# - https://huggingface.co/datasets/MahiA/GT-Music-Genre
# - https://huggingface.co/datasets/ylacombe/music_genres_XXX (where XXX is Jazz, Punk, Country, ...)

def iterate_data(
    streaming=True,
    system_prompt=None,
    **kwargs):

    instruction_file = os.path.join(
        os.path.dirname(__file__),
        "assets",
        "instruction_music_genre.txt"
    )
    assert os.path.exists(instruction_file), f"File not found: {instruction_file}"
    with open(instruction_file) as f:
        instructions = [line.strip() for line in f.read().split("\n") if line.strip()]

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
                yield format_data(repo,
                    sample["instruction"],
                    sample[label],
                    sample["audio"],
                    system_prompt=system_prompt
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
                yield format_data(repo,
                    random.choice(instructions),
                    format_genre(sample[label], label_dict=label_dict),
                    sample["audio"] if "audio" in sample else sample,
                    system_prompt=system_prompt,
                )


def format_data(repo, instruction, answer, audio, sampling_rate=16_000, system_prompt=None, label_dict=None):
    answer = format_genre(answer, label_dict=label_dict, repo=repo)
    assert isinstance(audio, dict)
    if isinstance(audio, dict) and "array" in audio:
        # Example: https://huggingface.co/datasets/DynamicSuperb/MusicGenreClassification_FMA
        audio_array = audio["array"]
        audio_path = audio["path"]
        actual_sampling_rate = audio.get("sampling_rate", sampling_rate)
        if sampling_rate and actual_sampling_rate != sampling_rate:
            audio_array = conform_audio(audio_array, actual_sampling_rate, sampling_rate=sampling_rate)
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
        audio_array = load_audio(os.path.join(pulled_folder, audio_path), sampling_rate=sampling_rate)
    assert sampling_rate
    assert len(audio_array)
    audio_data = {"type": "audio", "array": audio_array, "sampling_rate": sampling_rate}
    if audio_path:
        audio_data["path"] = audio_path
    return ([
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }] if system_prompt else []) + [
            {
            "role": "user",
            "content": [
                    audio_data,
                    {"type": "text", "text": instruction},
                ],
            },
            {
            "role": "assistant",
            "content": [
                    {"type": "text", "text": answer},
                ],
            }
        ]

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

    for data in iterate_data():
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
