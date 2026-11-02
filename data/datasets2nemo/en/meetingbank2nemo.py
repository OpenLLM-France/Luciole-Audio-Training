import argparse
import logging
import os
import random
import re
from pathlib import Path

import datasets
import soundfile as sf
from tqdm import tqdm

from ssak.utils.nemo_dataset import NemoDataset, NemoDatasetRow, NemoTurn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_NAME = "MeetingBank"
SPLITS = ["train", "validation", "test"]

SUMMARY_PROMPTS = [
    "Summarize this meeting.",
    "Provide a summary of what was discussed.",
    "Give an overview of this part of the conversation.",
    "What are the main points covered in this recording?",
    "Give me a summary of the audio.",
]

# uid pattern observed in huuuyeah/meetingbank: "<City>CityCouncil_MMDDYYYY_<itemId>"
UID_RE = re.compile(r"^(?P<city>[A-Za-z]+?)(?:CityCouncil|Council)_(?P<date>\d{6,8})_(?P<item>.+)$")


def parse_uid(uid: str) -> dict | None:
    m = UID_RE.match(uid)
    if not m:
        return None
    return {"city": m.group("city"), "date": m.group("date"), "item": m.group("item")}


def find_audio(audio_dir: Path, uid: str, cache: dict) -> Path | None:
    """Locate the full-meeting mp3 for a given uid.

    Looks under audio_dir/<City>/mp3/ for a file whose name contains the meeting date.
    Result is cached per (city, date) since many uids share a meeting.
    """
    parsed = parse_uid(uid)
    if parsed is None:
        return None
    key = (parsed["city"], parsed["date"])
    if key in cache:
        return cache[key]

    city_dir = audio_dir / parsed["city"] / "mp3"
    if not city_dir.is_dir():
        cache[key] = None
        return None

    candidates = list(city_dir.rglob(f"*{parsed['date']}*.mp3"))
    cache[key] = candidates[0] if candidates else None
    return cache[key]


def audio_duration(path: Path) -> float | None:
    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception as e:
        logger.warning(f"Could not read duration for {path}: {e}")
        return None


def load_source_dataset(cache_dir: str | None):
    logger.info("Loading huuuyeah/meetingbank"
                + (f" from cache_dir={cache_dir}" if cache_dir else ""))
    return datasets.load_dataset("huuuyeah/meetingbank", cache_dir=cache_dir)


def main():
    parser = argparse.ArgumentParser(description="Convert MeetingBank (huuuyeah/meetingbank + MeetingBank_Audio) to NeMo manifest")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="HF datasets cache dir for huuuyeah/meetingbank.")
    parser.add_argument("--audio-dir", type=str, default=None,
                        help="Root of the extracted MeetingBank_Audio (containing <City>/mp3/*.mp3). "
                             f"Default: {{DATA_FOLDER}}/raw/summary/en/MeetingBank/audios")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing manifest .jsonl files instead of skipping them.")
    parser.add_argument("--raw-manifest-path", type=str, default=None,
                        help="Override raw manifest output folder (with custom_metadata).")
    parser.add_argument("--manifest-path", type=str, default=None,
                        help="Override the metadata-free manifest output folder.")
    args = parser.parse_args()

    if args.audio_dir is None:
        args.audio_dir = f"{os.environ['DATA_FOLDER']}/raw/summary/en/MeetingBank/audios"
    if args.raw_manifest_path is None:
        args.raw_manifest_path = f"{os.environ['DATA_FOLDER']}/raw/summary/en/MeetingBank"
    if args.manifest_path is None:
        args.manifest_path = f"{os.environ['DATA_FOLDER']}/nemo/summary/en/MeetingBank"

    AUDIO_DIR = Path(args.audio_dir)
    RAW_MANIFEST_PATH = Path(args.raw_manifest_path)
    MANIFEST_PATH = Path(args.manifest_path)

    if not AUDIO_DIR.is_dir():
        logger.warning(f"Audio dir does not exist: {AUDIO_DIR}. "
                       "Download huuuyeah/MeetingBank_Audio and extract the per-city mp3 zips there.")

    RAW_MANIFEST_PATH.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.mkdir(parents=True, exist_ok=True)

    ds = load_source_dataset(args.cache_dir)

    audio_cache: dict = {}
    duration_cache: dict = {}

    for split in SPLITS:
        if split not in ds:
            logger.info(f"Split {split!r} not in dataset, skipping")
            continue

        manifest_file_raw = RAW_MANIFEST_PATH / f"{split}.jsonl"
        manifest_file = MANIFEST_PATH / f"{split}.jsonl"
        if manifest_file_raw.exists() and manifest_file.exists() and not args.force:
            logger.info(f"[{split}] manifests already exist, skipping: {manifest_file_raw}, {manifest_file}")
            continue

        subset = ds[split]
        logger.info(f"[{split}] total={len(subset)}")

        out_raw = NemoDataset(name=DATASET_NAME)
        out = NemoDataset(name=DATASET_NAME)

        n_missing_audio = 0
        for row in tqdm(subset, desc=split):
            uid = row["uid"]
            summary = row.get("summary")
            transcript = row.get("transcript")
            if not summary:
                logger.warning(f"[{split}] {uid}: no summary, skipping")
                continue

            audio_path = find_audio(AUDIO_DIR, uid, audio_cache)
            if audio_path is None:
                n_missing_audio += 1
                continue

            if audio_path not in duration_cache:
                duration_cache[audio_path] = audio_duration(audio_path)
            duration = duration_cache[audio_path]
            if duration is None:
                continue

            parsed = parse_uid(uid)

            audio_turn = NemoTurn(role="User", value=str(audio_path), turn_type="audio",
                                  duration=round(duration, 3))
            summary_turn = NemoTurn(role="Assistant", value=summary, turn_type="text")

            base_kwargs = dict(
                id=str(uid),
                dataset_name=DATASET_NAME,
                split=split,
                language="en",
                turns=[audio_turn, summary_turn],
            )
            out_raw.append(NemoDatasetRow(
                **base_kwargs,
                custom_metadata={
                    "uid":        uid,
                    "row_id":     row.get("id"),
                    "city":       parsed["city"] if parsed else None,
                    "date":       parsed["date"] if parsed else None,
                    "item":       parsed["item"] if parsed else None,
                    "summary":    summary,
                    "transcript": transcript,
                },
            ))
            prompt_turn = NemoTurn(role="User", value=random.choice(SUMMARY_PROMPTS), turn_type="text")
            lean_kwargs = {k: v for k, v in base_kwargs.items() if k != "turns"}
            out.append(NemoDatasetRow(
                **lean_kwargs,
                turns=[prompt_turn, audio_turn, summary_turn],
            ))

        if n_missing_audio:
            logger.warning(f"[{split}] skipped {n_missing_audio} rows with no matching mp3 under {AUDIO_DIR}")

        out_raw.save(manifest_file_raw)
        out.save(manifest_file)
        logger.info(f"[{split}] wrote {len(out_raw)} rows → {manifest_file_raw}")
        logger.info(f"[{split}] wrote {len(out)} rows → {manifest_file}")


if __name__ == "__main__":
    main()
