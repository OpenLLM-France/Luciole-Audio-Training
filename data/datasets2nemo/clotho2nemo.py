
import argparse
import csv
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from ssak.utils.nemo_dataset import NemoDataset, NemoDatasetRow, NemoTurn

random.seed(42)
np.random.seed(42)

# Sound-captioning instructions (paraphrases). One is picked at random per row by
# ``set_context_if_none`` to build the context version of the manifest.
prompts = [
    "Please help me generate an audio caption for the audio clip.",
    "Describe the sounds you hear in a single sentence.",
    "Write a short caption describing this audio.",
    "What sounds can you hear in this recording?",
    "Provide a short summary of the sounds.",
    "Caption the audio: what is happening?",
    "Describe the audio clip in a few words.",
    "Tell me what this sound is.",
    "Give a brief description of the acoustic scene.",
    "Summarize the sound events present in the clip.",
]

# Clotho split -> (captions csv basename, 7z audio archive basename, output split name).
# The extracted audio folder is named after the split (development/, validation/, evaluation/).
_SPLITS = {
    "development": ("clotho_captions_development.csv", "clotho_audio_development.7z", "train"),
    "validation":  ("clotho_captions_validation.csv",  "clotho_audio_validation.7z",  "dev"),
    "evaluation":  ("clotho_captions_evaluation.csv",   "clotho_audio_evaluation.7z",   "test"),
}

_CAPTION_COLUMNS = ["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"]


def _ensure_extracted(archive: Path, audio_dir: Path, raw_input: Path):
    """Make sure the split's wavs are on disk, extracting the .7z if needed."""
    if audio_dir.is_dir() and any(audio_dir.glob("*.wav")):
        return
    if not archive.exists():
        raise FileNotFoundError(f"Neither extracted folder {audio_dir} nor archive {archive} found")
    print(f"Extracting {archive.name} -> {raw_input} (system 7z)...")
    subprocess.run(["7z", "x", "-y", f"-o{raw_input}", str(archive)], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Clotho (sound captioning) to NeMo format")
    parser.add_argument("--raw_input", type=str, default=None,
                        help="Clotho folder (contains clotho_captions_*.csv and the audio archives / folders).")
    parser.add_argument("--resampled_dir", type=str, default=None,
                        help="Root where converted 16 kHz mono FLACs are written (raw sub-structure is preserved under it).")
    parser.add_argument("--nemo_no_context", type=str, default=None,
                        help="Output folder for the no-context NeMo manifests.")
    parser.add_argument("--nemo_context", type=str, default=None,
                        help="Output folder for the context NeMo manifests.")
    parser.add_argument("--no_resample", action="store_true",
                        help="Reference the original wavs as-is instead of normalizing to 16 kHz mono FLAC.")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Parallel workers for audio normalization.")
    parser.add_argument("--one_caption", action="store_true",
                        help="Keep only the first caption per audio instead of emitting one row per caption (5x fewer rows).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process this many audios per split (debug).")
    args = parser.parse_args()

    data_folder = os.environ.get("DATA_FOLDER", "/data-server/datasets/audio")
    if args.raw_input is None:
        args.raw_input = f"{data_folder}/raw/sounds/Clotho"
    if args.resampled_dir is None:
        args.resampled_dir = f"{data_folder}/converted_audios/sounds/Clotho"
    if args.nemo_no_context is None:
        args.nemo_no_context = f"{data_folder}/nemo/sounds/audio-captioning/en/nocontext/Clotho"
    if args.nemo_context is None:
        args.nemo_context = f"{data_folder}/nemo/sounds/audio-captioning/en/context/Clotho"

    raw_input = Path(args.raw_input)
    resampled_root = Path(args.resampled_dir)
    nemo_no_context = Path(args.nemo_no_context)
    nemo_context = Path(args.nemo_context)
    for d in (nemo_no_context, nemo_context):
        d.mkdir(parents=True, exist_ok=True)

    for split_folder, (csv_name, archive_name, split_name) in _SPLITS.items():
        audio_dir = raw_input / split_folder
        _ensure_extracted(raw_input / archive_name, audio_dir, raw_input)

        dataset = NemoDataset(name="Clotho")
        n_audio = n_missing = n_rows = 0

        with open(raw_input / csv_name, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if args.limit is not None and n_audio >= args.limit:
                    break
                n_audio += 1

                file_name = row["file_name"]
                src = audio_dir / file_name
                if not src.exists():
                    n_missing += 1
                    continue

                stem = Path(file_name).stem
                # Duration from the source stays valid after normalization.
                info = sf.info(str(src))
                duration = round(info.frames / info.samplerate, 3)

                captions = [row[c] for c in _CAPTION_COLUMNS if row.get(c)]
                if args.one_caption:
                    captions = captions[:1]

                for c_idx, caption in enumerate(captions, start=1):
                    dataset.append(NemoDatasetRow(
                        id=f"{stem}_c{c_idx}",
                        turns=[
                            NemoTurn(role="User", value=str(src), turn_type="audio",
                                     duration=duration),
                            NemoTurn(role="Assistant", value=caption, turn_type="text"),
                        ],
                        dataset_name="Clotho",
                        language="en",
                        split=split_name,
                        custom_metadata={"file_name": file_name, "caption_index": c_idx},
                    ))
                    n_rows += 1

                if n_audio % 500 == 0:
                    print(f"[{split_name}] processed {n_audio} audios ({n_rows} rows, {n_missing} missing)...")

        # Normalize to 16 kHz mono FLAC, keeping already-good wav/flac files untouched.
        # (rows sharing an audio path are all updated; dedup is handled internally.)
        if not args.no_resample:
            dataset.normalize_audios(
                str(resampled_root), target_sample_rate=16000, target_extension="flac",
                accepted_extensions=["wav", "flac"], num_workers=args.num_workers,
                relative_to=str(raw_input))

        # No-context manifest, then a context version with a random prompt per row.
        dataset.save(nemo_no_context / f"{split_name}.jsonl")
        dataset.set_context_if_none(prompts)
        dataset.save(nemo_context / f"{split_name}.jsonl")

        cover = 100 * (n_audio - n_missing) / n_audio if n_audio else 0
        print(f"[{split_name}] {n_audio} audios -> {n_rows} rows "
              f"({cover:.1f}% audio coverage, {n_missing} missing) -> {nemo_context / (split_name + '.jsonl')}")

    print("Done")
