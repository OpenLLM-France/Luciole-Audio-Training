import os
import sys
import random
import datasets
import pandas

# Load audios
import librosa
import numpy as np
import soxbindings as sox
import torch
import torchaudio

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


AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".opus"]

def load_audio(path, start = None, end = None, sampling_rate = 16_000, mono = True, return_format = 'array'):
    """
    Load an audio file and return the data.

    Parameters
    ----------
    path: str
        path to the audio file
    start: float
        start time in seconds. If None, the file will be loaded from the beginning.
    end: float
        end time in seconds. If None the file will be loaded until the end.
    sampling_rate: int
        destination sampling rate in Hz
    mono: bool
        if True, convert to mono
    return_format: str (default: 'array')
        'array': numpy.array
        'torch': torch.Tensor
        'bytes': bytes

    verbose: bool
        if True, print the steps
    """
    assert return_format in ['array', 'torch', 'bytes']
    if not os.path.isfile(path):
        # Because soxbindings does not indicate the filename if the file does not exist
        raise RuntimeError(f"File not found: {path}")
    # Test if we have read permission on the file
    elif not os.access(path, os.R_OK):
        # os.system("chmod a+r %s" % path)
        raise RuntimeError(f"Missing reading permission for: {path}")

    must_cut = start or end

    if return_format == 'torch' and not must_cut:
        if must_cut: # This path is super slow and has been disabled
            start = float(start if start else 0)
            sr = torchaudio.info(path).sampling_rate
            offset = int(start * sr)
            num_frames = -1
            if end:
                end = float(end)
                num_frames = int((end - start) * sr)
            audio, sr = torchaudio.load(path, frame_offset=offset, num_frames=num_frames)
        else:
            audio, sr = torchaudio.load(path)

    else:

        with suppress_stderr():
            # stderr could print these harmless warnings:
            # 1/ Could occur with sox.read
            #   mp3: MAD lost sync
            #   mp3: recoverable MAD error
            # 2/ Could occur with sox.get_info
            #   wav: wave header missing extended part of fmt chunk
            if must_cut: # is not None:
                start = float(start if start else 0)
                sr = sox.get_info(path)[0].rate
                offset = int(start * sr)
                nframes = 0
                if end: # is not None:
                    end = float(end)
                    nframes = int((end - start) * sr)
                audio, sr = sox.read(path, offset = offset, nframes = nframes)
            else:
                audio, sr = sox.read(path)

        audio = np.float32(audio)

    audio = conform_audio(audio, sr, sampling_rate=sampling_rate, mono=mono, return_format=return_format)

    if sampling_rate is None:
        return (audio, sr)
    return audio


class suppress_stderr:
    """
    A context manager for doing a "deep suppression" of stdout and stderr in Python,
    i.e. will suppress all print, even if the print originates in a compiled C/Fortran sub-function.
    """
    def __enter__(self):
        self.errnull_file = open(os.devnull, 'w')
        self.old_stderr_fileno_undup = sys.stderr.fileno()
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())
        self.old_stderr = sys.stderr
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stderr = self.old_stderr
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)
        os.close(self.old_stderr_fileno)
        self.errnull_file.close()


def conform_audio(audio, sr, sampling_rate=16_000, mono=True, return_format='array'):
    """
    Conform the audio to the desired format (mono channel, fixed frequency -- 16kHz)
    """
    if mono:
        if len(audio.shape) == 1:
            pass
        elif len(audio.shape) > 2:
            raise RuntimeError("Audio with more than 2 dimensions not supported")
        elif min(audio.shape) == 1:
            audio = audio.reshape(audio.shape[0] * audio.shape[1])
        else:
            if isinstance(audio, torch.Tensor):
                audio = audio.numpy()
            else:
                audio = audio.transpose()
            audio = librosa.to_mono(audio)
    if sampling_rate is not None and sr != sampling_rate:
        if not isinstance(audio, torch.Tensor):
            audio = torch.Tensor(audio)

        # # We don't use librosa here because there is a problem with multi-threading
        # audio = librosa.resample(audio, orig_sr = sr, target_sr = sampling_rate)

        audio = torchaudio.transforms.Resample(sr, sampling_rate)(torch.Tensor(audio))

    if return_format == "torch" and not isinstance(audio, torch.Tensor):
        audio = torch.Tensor(audio)
    elif return_format != "torch":
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        elif isinstance(audio, list):
            audio = np.array(audio, dtype=np.float32)
        if return_format == "bytes":
            audio = array_to_bytes(audio)

    return audio


def array_to_bytes(audio):
    return (audio * 32768).astype(np.int16).tobytes()


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
                # Pseudo-deterministic for each dataset
                random.seed(string_to_integer(dataset_name))
            continue
        messages.append(data)
        if len(messages) >= args.max_docs:
            _flush()
    _flush()


if __name__ == "__main__":
    main_dump_parquet()
