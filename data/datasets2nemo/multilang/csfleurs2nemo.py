"""Convert CS-FLEURS (byan/cs-fleurs, code-switched speech) to NeMo manifests with ASR prompts.

CS-FLEURS rewrites FLEURS sentences with code-switching (a matrix language with words of an
embedded language) and gets them spoken by people (read) or by TTS (xtts, mms). Only the pairs
whose two languages are both Luciole languages are kept, and the MMS subset (concatenative TTS) is
skipped. One file per subset, all the pairs in it, in $DATA_FOLDER/nemo/misc/codeswitching/CS-FLEURS/:

    xtts/train  X-eng, generative TTS       -> train.jsonl
    xtts/test1  X-eng, generative TTS       -> test_xtts1.jsonl
    xtts/test2  {ara,spa}-X, generative TTS -> test_xtts2.jsonl
    read/test   X-eng, read by people       -> test_read.jsonl

The text of xtts/train marks the embedded words with **...** ("اشتعلت النيران في **prison**"): the
markers are removed, the transcription keeps only the words (the marked text stays in
custom_metadata.marked_text). The other subsets have no marker. ’ is written '.

Each line gets an ASR prompt before the audio, composed by data/contexts/compose_asr_prompts.py for
the matrix language, with no constraint (the text is cased and punctuated, as asked by default).
The prompt never names one language, the audio has two: it does not mention the language
(LANGUAGE_MENTION) or asks for the source language; a prompt in another language than the matrix
one always asks for the source language.

The raw data (licence CC-BY-NC-4.0) is downloaded whole beforehand:
    hf download byan/cs-fleurs --repo-type dataset --local-dir $DATA_FOLDER/raw/transcript/multilang/CS-FLEURS

Usage:
    python data/datasets2nemo/multilang/csfleurs2nemo.py
"""
import argparse
import json
import logging
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "data" / "contexts"))
sys.path.insert(0, str(REPO_ROOT / "data" / "processing" / "clean_asr"))
from compose_asr_prompts import DEFAULT_DIR, PromptComposer  # noqa: E402
from rules import rules as language_rules  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ISO 639-3 codes of CS-FLEURS -> Luciole languages
LANGS = {"fra": "fr", "eng": "en", "deu": "de", "spa": "es", "ita": "it", "por": "pt", "nld": "nl", "ara": "ar"}
# subset directory in the repo -> output file name
SUBSETS = {"xtts/train": "train", "xtts/test1": "test_xtts1", "xtts/test2": "test_xtts2", "read/test": "test_read"}
# language mention of a prompt in the matrix language: never explicit (the audio has two languages)
LANGUAGE_MENTION = {"none": 0.8, "source": 0.2}


def kept(pair):
    return all(lang in LANGS for lang in pair.split("-"))


def read_metadata(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clean_text(text):
    """Remove the **...** markers of the embedded words and the spaces they leave; ’ -> '."""
    text = text.replace("**", "").replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


class Prompts:
    """A composer per matrix language (hesitations from conf/rules.yaml of clean_asr)."""

    def __init__(self):
        self.composers = {}

    def compose(self, rng, lang):
        if lang not in self.composers:
            hesitations = list(language_rules(lang).get("disfluencies", {}).get("hesitations") or {})
            self.composers[lang] = PromptComposer(DEFAULT_DIR, lang, hesitations, language_mention=LANGUAGE_MENTION)
        composer = self.composers[lang]
        weights = composer.instruction_languages
        instruction = rng.choices(list(weights), weights=list(weights.values()))[0]
        mention = "source" if instruction != lang else None
        return composer.compose(rng, instruction_language=instruction, mention=mention)


def convert(raw, output_dir, check_audio=True, seed=0):
    rng = random.Random(seed)
    prompts = Prompts()
    output_dir.mkdir(parents=True, exist_ok=True)
    for subset, name in SUBSETS.items():
        lines, stats = [], Counter()
        for row in read_metadata(raw / subset / "metadata.jsonl"):
            pair = row["language"]
            if not kept(pair):
                continue
            audio = (raw / subset / os.path.normpath(row["file_name"])).resolve()
            if check_audio and not audio.exists():
                stats["missing_audio"] += 1
                continue
            text = clean_text(row["text"])
            if not text:
                stats["empty"] += 1
                continue
            matrix, embedded = (LANGS[lang] for lang in pair.split("-"))
            metadata = {"language": pair, "matrix_language": matrix, "embedded_language": embedded,
                        "speaker": row["speaker"], "subset": subset}
            if "**" in row["text"]:
                metadata["marked_text"] = row["text"]
            elif subset == "xtts/train":
                metadata["no_switch"] = True      # 3 % of the train has no embedded word
            if row.get("fluency") is not None:
                metadata["fluency"] = row["fluency"]
            lines.append({
                "id": f"csfleurs_{name}_{row['id']}",
                "dataset_name": "cs-fleurs",
                "conversations": [
                    {"from": "User", "value": prompts.compose(rng, matrix), "type": "text"},
                    {"from": "User", "value": str(audio), "type": "audio", "duration": round(row["duration"], 3)},
                    {"from": "Assistant", "value": text, "type": "text"},
                ],
                "custom_metadata": metadata,
            })
            stats[pair] += 1
        out = output_dir / f"{name}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        hours = sum(line["conversations"][1]["duration"] for line in lines) / 3600
        pairs = {k: v for k, v in sorted(stats.items()) if k not in ("missing_audio", "empty")}
        logger.info(f"{subset}: {out} {len(lines)} lines {hours:.1f} h {pairs}")
        if stats["missing_audio"] or stats["empty"]:
            logger.warning(f"{subset}: {stats['missing_audio']} missing audio, {stats['empty']} empty texts skipped")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=None,
                        help="the downloaded repo (default: $DATA_FOLDER/raw/transcript/multilang/CS-FLEURS)")
    parser.add_argument("--output", default=None,
                        help="output directory (default: $DATA_FOLDER/nemo/misc/codeswitching/CS-FLEURS)")
    parser.add_argument("--no-check-audio", action="store_true", help="do not skip lines whose audio is missing")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    raw = Path(args.input or f"{os.environ['DATA_FOLDER']}/raw/transcript/multilang/CS-FLEURS")
    output_dir = Path(args.output or f"{os.environ['DATA_FOLDER']}/nemo/misc/codeswitching/CS-FLEURS")
    convert(raw, output_dir, check_audio=not args.no_check_audio, seed=args.seed)


if __name__ == "__main__":
    main()
