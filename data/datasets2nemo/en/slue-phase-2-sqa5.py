import argparse
import json
import logging
import os
import re
from pathlib import Path

import datasets
import numpy as np
import soundfile as sf
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_NAME = "slue-phase-2 - SQA5"
SPLITS = ["train", "validation", "test", "verified_test"]


def sanitize(s: str) -> str:
    return re.sub(r"[^\w\-]", "_", s)


def write_flac_if_missing(path: Path, hf_audio: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(hf_audio["array"], dtype=np.float32)
    if arr.ndim > 1:
        arr = arr[:, 0]
    sr = int(hf_audio["sampling_rate"])
    if not path.exists():
        sf.write(path, arr, sr)
    return {"path": path, "duration": len(arr) / sr}


def recover_cased_answer(answer_spans: dict, word2time: dict) -> str:
    """Recover original casing of the answer by locating it in word2time.

    word2time uses start_second == end_second == -1 for punctuation tokens,
    which are skipped when matching the answer span boundaries.
    """
    answer = answer_spans["answer"][0]
    try:
        words  = list(word2time["word"])
        starts = list(word2time["start_second"])
        ends   = list(word2time["end_second"])
        a_start = answer_spans["start_second"][0]
        a_end   = answer_spans["end_second"][0]
        i0 = next(i for i, s in enumerate(starts) if s != -1 and s == a_start)
        i1 = next(i for i, e in enumerate(ends)   if e != -1 and e == a_end)
        cased = " ".join(
            w for w, s in zip(words[i0:i1 + 1], starts[i0:i1 + 1]) if s != -1
        )
        norm = lambda s: re.sub(r"[^\w]", "", s).lower()
        if cased and norm(cased) == norm(answer):
            return cased
        logger.warning(f"Casing recovery mismatch: {cased!r} vs {answer!r}")
    except (StopIteration, KeyError, IndexError):
        logger.warning(f"Casing recovery failed for answer {answer!r}")
    return answer


CSV_QUOTED_RE = re.compile(r'^"(.*)"$', re.S)
OPEN_CURLY_RE = re.compile(r"([“‘])\s+")
CLOSE_CURLY_RE = re.compile(r"\s+([”’])")


# Wikipedia annotations in brackets, not read: "[ citation needed ]", "[ 1 ]", "[ a ]", "[ . . . ]"
WIKI_NOTE_RE = re.compile(r"^(?:\d+|[a-z]|[. ]+|.*\b(?:citation|needed|sic|who|whom|when|which|where|clarification"
                          r"|dubious|unsupported|verification|verify|page|according|original research|source|discuss"
                          r"|disputed|update|year|full|specify|vague|why|how|failed|not in)\b.*)$", re.I)


def drop_wiki_notes(words: list, starts: list):
    """Remove the bracketed Wikipedia annotations ("[ citation needed ]", footnotes "[ 1 ]") and keep
    the words of the other brackets without them ("[ writ for the son ]": an editorial insertion,
    read). None when brackets are unpaired or curly braces are there."""
    out_w, out_s, i = [], [], 0
    while i < len(words):
        if words[i] in "{}" or words[i] == "]":
            return None
        if words[i] == "[":
            try:
                j = words.index("]", i + 1)
            except ValueError:
                return None
            inner = words[i + 1:j]
            if "[" in inner:
                return None
            if not WIKI_NOTE_RE.match(" ".join(inner)):
                out_w += inner
                out_s += starts[i + 1:j]
            i = j + 1
            continue
        out_w.append(words[i])
        out_s.append(starts[i])
        i += 1
    return out_w, out_s


HUNDRED_N_RE = re.compile(r"\b(\d{1,2}) hundred (\d{1,2})(s?)\b")      # "19 hundred 69" -> "1969", "19 hundred 70s"
HUNDRED_RE = re.compile(r"\b(\d{1,2}) hundred(s?)\b(?! (?:and )?\d)")   # "18 hundred" -> "1800"
POINT_RE = re.compile(r"\b\d+(?: point \d+)+\b")                        # "Section 5 point 3 point 1" -> "5.3.1"


def written_numbers(text: str) -> str:
    """The SLUE text spells numbers as read ("19 hundred 69", "Section 5 point 3"): written back."""
    text = HUNDRED_N_RE.sub(lambda m: f"{int(m[1]) * 100 + int(m[2])}{m[3]}", text)
    text = HUNDRED_RE.sub(lambda m: f"{int(m[1]) * 100}{m[2]}", text)
    return POINT_RE.sub(lambda m: m[0].replace(" point ", "."), text)


def document_text(word2time: dict, detok) -> str | None:
    """The read text of a document chunk, detokenized ("America ( a club ) ." -> "America (a club).",
    leading punctuation of a chunk cut mid-sentence removed), without the Wikipedia annotations in
    brackets (drop_wiki_notes); None when part of it was likely not read (a word with no time:
    Chinese, Greek, Syriac... left out by the aligner) or brackets cannot be resolved."""
    kept = drop_wiki_notes(list(word2time.get("word") or []), list(word2time.get("start_second") or []))
    if not kept or not kept[0]:
        return None
    words, starts = kept
    if any(s == -1 and re.search(r"\w", w) for w, s in zip(words, starts)):
        return None
    text = detok.detokenize(words)
    text = CLOSE_CURLY_RE.sub(r"\1", OPEN_CURLY_RE.sub(r"\1", text))
    text = written_numbers(re.sub(r"^[\s.,;:!?)\]”’]+", "", text))
    return text.strip() or None


def question_text(raw: str) -> str:
    """The question as read. A CSV-quoted one (the whole text in quotes, inner quotes doubled) is
    unquoted; a quote left unpaired is removed."""
    text = raw.strip()
    m = CSV_QUOTED_RE.match(text)
    if m and '""' in text:
        text = m[1].replace('""', '"')
    if text.count('"') % 2:
        text = text.replace('"', "")
    return re.sub(r"\s+", " ", text).strip()


def write_asr(raw_dir: Path, out_dir: Path, splits: list[str]):
    """ASR intermediates from the raw manifests (the "full" ones of this script), no prompt yet:
    <split>.jsonl = the document chunks (one line per distinct audio, whole 40 s file, cut
    mid-sentence), <split>_questions.jsonl = the spoken questions. clean_asr.py conf/en.yaml
    (SLUE-SQA5, SLUE-SQA5_questions) cleans them into clean/SLUE-SQA5/."""
    from sacremoses import MosesDetokenizer
    detok = MosesDetokenizer(lang="en")
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        src = raw_dir / f"{split}.jsonl"
        if not src.exists():
            logger.info(f"[asr {split}] no {src}, skipping")
            continue
        docs, questions, dropped = {}, [], 0
        with open(src, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                meta = row["custom_metadata"]
                doc = next(t for t in row["conversations"] if t.get("type") == "audio")
                if doc["value"] not in docs:
                    text = document_text(meta.get("word2time") or {}, detok)
                    dropped += text is None
                    docs[doc["value"]] = (meta["document_id"], doc["duration"], text)
                questions.append((row["id"], meta["question_audio_filepath"], meta["question_audio_duration"],
                                  question_text(meta["raw_question_text"])))

        def line_of(uid, path, duration, text):
            return json.dumps({"id": uid, "dataset_name": "SLUE-SQA5", "conversations": [
                {"from": "User", "value": path, "type": "audio", "duration": duration},
                {"from": "Assistant", "value": text, "type": "text"}]}, ensure_ascii=False) + "\n"

        n_docs = 0
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as w:
            for path, (doc_id, duration, text) in docs.items():
                if text:
                    w.write(line_of(doc_id, path, duration, text))
                    n_docs += 1
        with open(out_dir / f"{split}_questions.jsonl", "w", encoding="utf-8") as w:
            for uid, path, duration, text in questions:
                w.write(line_of(uid, path, duration, text))
        logger.info(f"[asr {split}] {n_docs} documents ({dropped} dropped: words not read), "
                    f"{len(questions)} questions -> {out_dir}")


def load_source_dataset(cache_dir: str | None):
    logger.info(f"Loading asapp/slue-phase-2 (sqa5)"
                + (f" from cache_dir={cache_dir}" if cache_dir else ""))
    return datasets.load_dataset("asapp/slue-phase-2", "sqa5", cache_dir=cache_dir)


def main():
    parser = argparse.ArgumentParser(description="Convert slue-phase-2 sqa5 to NeMo manifest")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="HF datasets cache dir (reuses files already downloaded via load_dataset).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing manifest .jsonl files instead of skipping them.")
    parser.add_argument("--manifest-path", type=str, default=None,
                        help="Override raw manifest output folder.")
    parser.add_argument("--manifest-audioq-textc", type=str, default=None,
                        help="Override NeMo manifest folder for audio-question + text-context.")
    parser.add_argument("--manifest-audioq-audioc", type=str, default=None,
                        help="Override NeMo manifest folder for audio-question + audio-context.")
    parser.add_argument("--manifest-textq-audioc", type=str, default=None,
                        help="Override NeMo manifest folder for text-question + audio-context.")
    parser.add_argument("--audio-path", type=str, default=None,
                        help="Override raw audio output folder.")
    parser.add_argument("--asr", action="store_true",
                        help="Only write the ASR intermediates (nemo/asr/en/clean/SLUE-SQA5/intermediate/) "
                             "from the raw manifests already written, no download.")
    parser.add_argument("--asr-splits", nargs="+", default=["train"],
                        help="Splits for --asr (validation/test stay out: SLUE is an evaluation set).")
    args = parser.parse_args()

    if args.asr:
        raw_dir = Path(args.manifest_path or f"{os.environ['DATA_FOLDER']}/raw/misc/en/slu-phase-2-sqa5")
        write_asr(raw_dir, Path(f"{os.environ['DATA_FOLDER']}/nemo/asr/en/clean/SLUE-SQA5/intermediate"),
                  args.asr_splits)
        return

    from ssak.utils.nemo_dataset import NemoDataset, NemoDatasetRow, NemoTurn  # not needed by --asr

    if args.manifest_path is None:
        args.manifest_path = f"{os.environ['DATA_FOLDER']}/raw/misc/en/context/slue-phase-2-sqa5"
    if args.manifest_audioq_textc is None:
        args.manifest_audioq_textc = f"{os.environ['DATA_FOLDER']}/nemo/question-answering/qa_audio-question-text-context/en/slue-phase-2-sqa5"
    if args.manifest_audioq_audioc is None:
        args.manifest_audioq_audioc = f"{os.environ['DATA_FOLDER']}/nemo/question-answering/qa_audio-question-audio-context/en/slue-phase-2-sqa5"
    if args.manifest_textq_audioc is None:
        args.manifest_textq_audioc = f"{os.environ['DATA_FOLDER']}/nemo/question-answering/qa_audio-context-text-question/en/slue-phase-2-sqa5"
    if args.audio_path is None:
        args.audio_path = f"{os.environ['DATA_FOLDER']}/raw/misc/en/slue-phase-2-sqa5/audios"

    MANIFEST_PATH = Path(args.manifest_path)
    MANIFEST_PATH_AUDIOQ_TEXTC = Path(args.manifest_audioq_textc)
    MANIFEST_PATH_AUDIOQ_AUDIOC = Path(args.manifest_audioq_audioc)
    MANIFEST_PATH_TEXTQ_AUDIOC = Path(args.manifest_textq_audioc)
    AUDIO_PATH = Path(args.audio_path)

    AUDIO_PATH.mkdir(parents=True, exist_ok=True)
    for p in (MANIFEST_PATH, MANIFEST_PATH_AUDIOQ_TEXTC, MANIFEST_PATH_AUDIOQ_AUDIOC, MANIFEST_PATH_TEXTQ_AUDIOC):
        p.mkdir(parents=True, exist_ok=True)

    ds = load_source_dataset(args.cache_dir)

    for split in SPLITS:
        if split not in ds:
            logger.info(f"Split {split!r} not in dataset, skipping")
            continue

        targets = {
            "full":           MANIFEST_PATH               / f"{split}.jsonl",
            "audioQ_textC":   MANIFEST_PATH_AUDIOQ_TEXTC  / f"{split}.jsonl",
            "audioQ_audioC":  MANIFEST_PATH_AUDIOQ_AUDIOC / f"{split}.jsonl",
            "textQ_audioC":   MANIFEST_PATH_TEXTQ_AUDIOC  / f"{split}.jsonl",
        }
        pending = {k: v for k, v in targets.items() if args.force or not v.exists()}
        if not pending:
            logger.info(f"[{split}] all manifests already exist, skipping")
            continue
        for k, v in targets.items():
            if k not in pending:
                logger.info(f"[{split}] {k} already exists, will skip save: {v}")

        subset = ds[split]
        logger.info(f"[{split}] total={len(subset)}")

        outs = {k: NemoDataset(name=DATASET_NAME) for k in pending}

        for row in tqdm(subset, desc=split):
            uid = f"{row['question_id']}-{row['question_speaker_id']}-{row['document_speaker_id']}"

            doc_audio = write_flac_if_missing(
                AUDIO_PATH / "documents" / f"{sanitize(row['document_id'])}.flac",
                row["document_audio"],
            )
            q_audio = write_flac_if_missing(
                AUDIO_PATH / "questions" / f"{sanitize(uid)}.flac",
                row["question_audio"],
            )
            answer = recover_cased_answer(row["answer_spans"], row.get("word2time") or {})

            doc_audio_turn = NemoTurn(role="User", value=str(doc_audio["path"]), turn_type="audio",
                                      duration=round(doc_audio["duration"], 3))
            q_audio_turn   = NemoTurn(role="User", value=str(q_audio["path"]),   turn_type="audio",
                                      duration=round(q_audio["duration"], 3))
            q_text_turn    = NemoTurn(role="User",      value=row["raw_question_text"], turn_type="text")
            answer_turn    = NemoTurn(role="Assistant", value=answer,                    turn_type="text")

            if "full" in outs:
                outs["full"].append(NemoDatasetRow(
                    id=uid,
                    dataset_name=DATASET_NAME,
                    split=split,
                    language="en",
                    turns=[q_text_turn, doc_audio_turn, answer_turn],
                    custom_metadata={
                        "question_id":              row["question_id"],
                        "document_id":              row["document_id"],
                        "question-source":          row["question_id"].split("-")[0],
                        "question_speaker_id":      row["question_speaker_id"],
                        "document_speaker_id":      row["document_speaker_id"],
                        "raw_question_text":        row["raw_question_text"],
                        "normalized_question_text": row.get("normalized_question_text"),
                        "raw_document_text":        row.get("raw_document_text"),
                        "normalized_document_text": row.get("normalized_document_text"),
                        "question_audio_filepath":  str(q_audio["path"]),
                        "question_audio_duration":  round(q_audio["duration"], 3),
                        "answer_spans":             {k: list(v) for k, v in row["answer_spans"].items()},
                        "word2time":                {k: list(v) for k, v in (row.get("word2time") or {}).items()},
                    },
                ))

            if "audioQ_textC" in outs:
                outs["audioQ_textC"].append(NemoDatasetRow(
                    id=uid,
                    dataset_name=DATASET_NAME,
                    split=split,
                    language="en",
                    turns=[
                        q_audio_turn,
                        NemoTurn(role="User", value=row.get("raw_document_text"), turn_type="text"),
                        answer_turn,
                    ],
                ))

            if "audioQ_audioC" in outs:
                outs["audioQ_audioC"].append(NemoDatasetRow(
                    id=uid,
                    dataset_name=DATASET_NAME,
                    split=split,
                    language="en",
                    turns=[q_audio_turn, doc_audio_turn, answer_turn],
                ))

            if "textQ_audioC" in outs:
                outs["textQ_audioC"].append(NemoDatasetRow(
                    id=uid,
                    dataset_name=DATASET_NAME,
                    split=split,
                    language="en",
                    turns=[q_text_turn, doc_audio_turn, answer_turn],
                ))

        for key, out in outs.items():
            manifest_file = targets[key]
            out.save(manifest_file)
            logger.info(f"[{split}] {key}: wrote {len(out)} rows → {manifest_file}")


if __name__ == "__main__":
    main()
