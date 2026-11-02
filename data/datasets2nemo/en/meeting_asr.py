"""ASR helpers shared by ami2nemo.py and icsi2nemo.py (the `--asr` family).

Word records come from each converter's `load_words(..., markers=True)`:
    {"text": str|None, "start": float|None, "end": float|None,
     "join": None|"left"|"right"|"both", "lex": bool}
`join` glues a token to its neighbour ("left": punctuation, "'s"; "right": an opening
quote; "both": the hyphen of "Mm-hmm"). `lex` marks real words, as opposed to the
raw event markers ("<vocalsound:laugh>", "<gap>", ...) that are kept in the text for
data/processing/clean_asr/clean_asr.py to map to [tag]s.
"""

import bisect
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

ASR_SPLIT_NAMES = {"validation": "dev"}   # same file names as the other ASR manifests

# contraction suffixes tokenised apart ("I 'm", "it 's"); not "'cause" / "'em"
CONTRACTIONS = {"'s", "'m", "'re", "'ve", "'ll", "'d", "n't"}


def asr_text(records: list) -> str:
    """Join word records into text, applying their `join` and dropping repeated markers."""
    out = []
    glue_next = False
    for r in records:
        t = r.get("text")
        if not t:
            continue
        if out and not r.get("lex") and t.startswith("<") and t == out[-1]:
            continue    # "@ @" -> one <unintelligible>
        if out and (glue_next or r.get("join") in ("left", "both") or t.lower() in CONTRACTIONS):
            out[-1] += t
        else:
            out.append(t)
        glue_next = r.get("join") in ("right", "both")
    return " ".join(out).strip()


def split_long(records: list, start: float, end: float, max_duration: float,
               min_piece: float = 1.0) -> list[tuple[float, float, list]]:
    """Cut a segment longer than max_duration at its longest pause between two timed
    words (cut in the middle of the pause), recursively. Pieces shorter than min_piece
    are not cut off. Returns [(start, end, records)]; a segment that cannot be cut stays whole."""
    if end - start <= max_duration:
        return [(start, end, records)]
    best = None
    last_end = None
    for i, r in enumerate(records):
        if r.get("start") is None or r.get("end") is None:
            continue
        if last_end is not None:
            cut = (last_end + r["start"]) / 2
            gap = r["start"] - last_end
            if cut - start >= min_piece and end - cut >= min_piece and (best is None or gap > best[0]):
                best = (gap, i, cut)
        last_end = r["end"]
    if best is None:
        return [(start, end, records)]
    _, i, cut = best
    return (split_long(records[:i], start, cut, max_duration, min_piece)
            + split_long(records[i:], cut, end, max_duration, min_piece))


class OtherSpeakers:
    """Time spoken by the other speakers of a meeting inside [a, b] (lexical words only)."""

    def __init__(self, words_by_speaker: dict):
        self.words = sorted(
            (w["start"], w["end"], spk)
            for spk, records in words_by_speaker.items() for w in records
            if w.get("lex") and w.get("start") is not None and w.get("end") is not None
        )
        self.starts = [w[0] for w in self.words]
        self.max_len = max((e - s for s, e, _ in self.words), default=0.0)

    def time(self, speaker, a: float, b: float) -> float:
        t = 0.0
        for s, e, spk in self.words[bisect.bisect_left(self.starts, a - self.max_len):]:
            if s >= b:
                break
            if spk != speaker:
                t += max(0.0, min(e, b) - max(s, a))
        return t


def dump_asr(buckets: dict, raw_dir: Path, out_dir: Path, force: bool):
    """Write {split}.jsonl (train/dev/test) to raw_dir (with metadata) and out_dir (lean)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, (raw, lean) in buckets.items():
        if not len(raw):
            continue
        name = ASR_SPLIT_NAMES.get(split, split)
        raw_file, lean_file = raw_dir / f"{name}.jsonl", out_dir / f"{name}.jsonl"
        if raw_file.exists() and lean_file.exists() and not force:
            logger.info(f"[asr/{name}] exists, skipping (use --force)")
            continue
        raw.save(raw_file)
        lean.save(lean_file)
        logger.info(f"[asr/{name}] wrote {len(raw)} rows → {raw_file} and {lean_file}")
