"""List the French words that are lowercase in the middle of a sentence (not proper nouns).

Counts, over the cased transcriptions of clean/<Dataset>/intermediate/ (+ the read-speech
manifests of context/), how often each word is written lowercase vs capitalised when it
follows a word with no punctuation. A word is "common" when seen MIN_COUNT times there and
lowercase at least 4 times out of 5.
clean_asr.py (fix_punctuation, subtitle_labels) uses the list to tell a sentence start that
lost its punctuation ("très bien Bon") from a proper noun ("à Paris").
    python data/processing/clean_asr/build_common_words.py
"""
import glob
import json
import os
import re
from collections import Counter

CLEAN = "/data-server/datasets/audio/nemo/asr/fr/clean"
CONTEXT = "/data-server/datasets/audio/nemo/asr/fr/context"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf", "fr_common_words.txt")
WORD_RE = re.compile(r"(\S+)\s+([^\W\d_][\w'’-]*)")
MIN_COUNT, MIN_LOWER_RATIO = 5, 0.8   # seen 5+ times after a lowercase word, 80 %+ lowercase


def texts():
    paths = glob.glob(f"{CLEAN}/*/intermediate/*.jsonl") + [
        f"{CONTEXT}/CommonVoice/train.jsonl",
        f"{CONTEXT}/Multilingual_LibriSpeech_recasepunc/train_recasepunc.jsonl"]
    for path in sorted(p for p in paths if "max30" not in p):
        print(path)
        with open(path, encoding="utf-8") as f:
            for line in f:
                text = json.loads(line)["conversations"][-1]["value"]
                if re.search(r"[A-ZÀ-Þ]", text) and re.search(r"[.,?]", text):   # cased lines only
                    yield text


def main():
    lower, upper = Counter(), Counter()
    for text in texts():
        for m in WORD_RE.finditer(text):
            prev, word = m[1], m[2]
            if not prev[-1].isalpha() or not prev[-1].islower():
                continue
            key = word.lower().replace("’", "'")
            if word[0].islower():
                lower[key] += 1
            elif word[1:] == word[1:].lower():   # "Paris", not "SNCF"
                upper[key] += 1
    # not the cut-off words of the transcripts ("a-", "jour-"): not words
    words = sorted(w for w in lower if lower[w] + upper[w] >= MIN_COUNT
                   and lower[w] / (lower[w] + upper[w]) >= MIN_LOWER_RATIO and not w.endswith("-"))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(words) + "\n")
    print(f"{len(words)} words -> {OUT}")


if __name__ == "__main__":
    main()
