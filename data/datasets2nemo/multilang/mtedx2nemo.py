"""MultilingualTEDx ASR text from the raw corpus on data-server, for clean_asr.py (jz/mtedx2nemo.py: the JZ source, through ssak).

Writes clean/Multilingual_TEDx/intermediate/{train,dev,test}.jsonl with the raw lines as they are: the bracket removal
of the old conversion left holes ("les fous, les cavaliers") or a foreign text without its translation. Lines with
brackets are dropped, and in French those with a language marker ("(En anglais)", "(Créole)"), whose text is a French
translation of foreign speech. Event parentheses ("(Rires)") are kept as they are.
"""
import argparse
import json
import logging
import os
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LANGS = (r"anglais|arabe|néerlandais|flamand|créole|allemand|espagnol|italien|grec|chinois|portugais|russe|"
         r"tibétain|japonais|latin|wolof|lingala|bambara|kabyle|berbère|hébreu|turc|swahili|hindi|polonais|"
         r"breton|occitan|basque|corse|alsacien|catalan|roumain|persan|coréen|vietnamien|langue des signes")
FOREIGN_FR = re.compile(r"\((?![^)]*accent)[^)]{0,40}\b(?:%s)\b[^)]{0,40}\)" % LANGS, re.IGNORECASE)


def audio_path(wav_folder, audio_id):
    """The converted wav when there is one (data-server), else the raw flac."""
    converted = str(wav_folder).split("/raw/")
    if len(converted) == 2:
        language = wav_folder.parts[wav_folder.parts.index("transcript") + 1]
        wav = f"{converted[0]}/converted_audios/transcript/{language}/Multilingual_TEDx/{audio_id}.wav"
        if os.path.exists(wav):
            return wav
    return str(wav_folder / f"{audio_id}.flac")


def write_raw_text(input_dataset, output_path, language, force=False):
    output_path.mkdir(parents=True, exist_ok=True)
    for split, name in (("train", "train"), ("valid", "dev"), ("test", "test")):
        out_file = output_path / f"{name}.jsonl"
        if out_file.exists() and not force:
            logger.info(f"{out_file} exists, skipping (use --force)")
            continue
        txt_folder = input_dataset / split / "txt"
        with open(txt_folder / "segments", encoding="utf-8") as f:
            segments = [line.split() for line in f]
        with open(txt_folder / f"{split}.{language}", encoding="utf-8") as f:
            texts = [line.rstrip("\n") for line in f]
        assert len(segments) == len(texts), f"{split}: {len(segments)} segments, {len(texts)} lines"
        stats = {"lines": 0, "brackets": 0, "foreign": 0, "empty": 0}
        with open(f"{out_file}.tmp", "w", encoding="utf-8") as f:
            for (seg_id, audio_id, start, end), text in zip(segments, texts):
                text = " ".join(text.split())
                if "[" in text or "]" in text:
                    stats["brackets"] += 1
                elif language == "fr" and FOREIGN_FR.search(text):
                    stats["foreign"] += 1
                elif not text:
                    stats["empty"] += 1
                else:
                    stats["lines"] += 1
                    f.write(json.dumps({"id": seg_id, "conversations": [
                        {"from": "User", "value": "", "type": "text"},
                        {"from": "User", "value": audio_path(input_dataset / split / "wav", audio_id), "type": "audio",
                         "duration": float(end) - float(start), "offset": float(start)},
                        {"from": "Assistant", "value": text, "type": "text"}]}, ensure_ascii=False) + "\n")
        os.replace(f"{out_file}.tmp", out_file)
        logger.info(f"{out_file}: {stats}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MultilingualTEDx raw text to NeMo manifests")
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--language", type=str, default="fr", choices=["es", "fr", "it", "de", "pt", "ar"])
    parser.add_argument("--root", default=None,
                        help="Folder with {train,valid,test}/{txt,wav}. "
                             "Default: $DATA_FOLDER/raw/transcript/<language>/TEDX/<language>-<language>/data")
    parser.add_argument("--output", default=None,
                        help="Default: $DATA_FOLDER/nemo/asr/<language>/clean/Multilingual_TEDx/intermediate")
    args = parser.parse_args()
    data_folder = os.getenv("DATA_FOLDER")
    write_raw_text(Path(args.root or f"{data_folder}/raw/transcript/{args.language}/TEDX/{args.language}-{args.language}/data"),
                   Path(args.output or f"{data_folder}/nemo/asr/{args.language}/clean/Multilingual_TEDx/intermediate"),
                   args.language, args.force)
