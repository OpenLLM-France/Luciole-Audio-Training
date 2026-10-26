"""Rebuild the text of SUMM-RE train with the case and punctuation of the whisper transcripts.

The train text of SUMM-RE (automatic IPU transcripts, kaldi-style "c' est", no case nor punctuation)
comes from the same whisper-timestamped run as
raw/transcript/fr/SUMM-RE/full_corpus/individual/archive/transcript/automatic/whisper/unfiltered/v1/json/train/<file>.v1.words.json
whose words keep case and punctuation ("Ah", "voilà.", "clap?").

Per recording, the whisper words ("[*]" disfluency marks removed) are aligned with the concatenation
of all its segments (in time order), then each segment gets transfer_format(segment, aligned whisper
words): words of the segment, case/punctuation of whisper; below ratio 0.8 the segment stays lowercase.
Segments of the recordings that were never transcribed ("sil w_1 w_2 ..." placeholders, which the
old conversion turned into "sil w un w deux") are dropped (reason "filler"), and the old
conversion's "monsieur'" (expansion of "m'") is put back to "m'".

Usage: python3 data/processing/clean_asr/rebuild/summre.py
"""
import difflib
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONTEXT, assistant_text, audio_turn, norm_word, rebuild_manifest, transfer_format  # noqa: E402

DATASET = "SUMM-RE"
WHISPER = ("/data-server/datasets/audio/raw/transcript/fr/SUMM-RE/full_corpus/individual/archive/transcript/"
           "automatic/whisper/unfiltered/v1/json/train")
NUMBERS = (r"un|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|"
           r"vingt|trente|quarante|cinquante|soixante|cent")
FILLER_RE = re.compile(rf"\bsil\b|\bw (?:{NUMBERS})")


def tokens(text):
    """'c'est' / 'c' est' -> ["c'", 'est'] for alignment."""
    return [norm_word(t) for t in re.sub(r"'(?=\S)", "' ", text).split()]


def whisper_words(name):
    d = json.load(open(os.path.join(WHISPER, name + ".v1.words.json"), encoding="utf-8"))
    words = [w.get("text", w.get("word", "")) for s in d["segments"] for w in s["words"]]
    # "as-tu" -> "as tu": transfer_format would leave a truncation dash on "as" when the segment says "as tu"
    return [re.sub(r"(?<=\w)-(?=\w)", " ", w) for w in words if w and w != "[*]"]


def formatted_by_segment(segs, words):
    """segs: [(id, text)] in time order -> {id: whisper words aligned to the segment}."""
    wtok, wown = [], []
    for i, w in enumerate(words):
        for t in tokens(w):
            wtok.append(t)
            wown.append(i)
    stok, sown = [], []
    for sid, text in segs:
        for t in tokens(text):
            stok.append(t)
            sown.append(sid)
    span = {}
    for a, b, size in difflib.SequenceMatcher(None, stok, wtok, autojunk=False).get_matching_blocks():
        for k in range(size):
            lo, hi = span.get(sown[a + k], (wown[b + k], wown[b + k]))
            span[sown[a + k]] = (min(lo, wown[b + k]), max(hi, wown[b + k]))
    return {sid: " ".join(words[lo:hi + 1]) for sid, (lo, hi) in span.items()}


def main():
    name = "train.jsonl"
    by_file = defaultdict(list)
    for line in open(os.path.join(CONTEXT, DATASET, name), encoding="utf-8"):
        rec = json.loads(line)
        text = assistant_text(rec)["value"]
        if not FILLER_RE.search(text):
            by_file[rec["id"].rsplit("_", 1)[0]].append((audio_turn(rec).get("offset", 0.0), rec["id"], text))
    fmt = {}
    for f, segs in by_file.items():
        segs.sort()
        fmt.update(formatted_by_segment([(sid, t) for _, sid, t in segs], whisper_words(f)))
    stats = Counter()

    def fn(rec):
        text = assistant_text(rec)["value"]
        if FILLER_RE.search(text):
            return None, "filler"
        raw = re.sub(r"(\w)' (?=\w)", r"\1'", re.sub(r"\bmonsieur'", "m'", text))
        if rec["id"] not in fmt:
            stats["no whisper words"] += 1
            return raw
        new, ratio = transfer_format(raw, fmt[rec["id"]])
        stats["transferred" if ratio >= 0.8 else "fallback (ratio < 0.8)"] += 1
        return new
    rebuild_manifest(DATASET, name, fn)
    print("  ", dict(stats))


if __name__ == "__main__":
    main()
