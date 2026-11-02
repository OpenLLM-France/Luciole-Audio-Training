"""List the words of a language that are lowercase in the middle of a sentence (not proper nouns).

Counts, over the cased transcriptions of clean/<Dataset>/intermediate/ (fr) or of context/ (en,
which has no clean/), how often each word is written lowercase vs capitalised when it
follows a word with no punctuation. A word is "common" when seen MIN_COUNT times there and
lowercase at least 4 times out of 5.
clean_asr.py (fix_punctuation, subtitle_labels) uses the list to tell a sentence start that
lost its punctuation ("très bien Bon") from a proper noun ("à Paris").
    python data/processing/clean_asr/build_common_words.py [--lang en]
"""
import argparse
import glob
import json
import os
import re
from collections import Counter

ASR = f"{os.environ['DATA_FOLDER']}/nemo/asr"
CONF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf")
WORD_RE = re.compile(r"(\S+)\s+([^\W\d_][\w'’-]*)")
MIN_COUNT, MIN_LOWER_RATIO = 5, 0.8   # seen 5+ times after a lowercase word, 80 %+ lowercase


def inputs(lang):
    if lang == "fr":
        return glob.glob(f"{ASR}/fr/clean/*/intermediate/*.jsonl")
    return [f"{ASR}/en/context/{d}/train.jsonl" for d in ("CommonVoice", "FLEURS", "VoxPopuli", "Yodas")]


def texts(lang):
    paths = inputs(lang)
    for path in sorted(p for p in paths if "max30" not in p):
        print(path)
        with open(path, encoding="utf-8") as f:
            for line in f:
                text = json.loads(line)["conversations"][-1]["value"]
                if re.search(r"[A-ZÀ-Þ]", text) and re.search(r"[.,?]", text):   # cased lines only
                    yield text


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", choices=["fr", "en"], default="fr")
    lang = p.parse_args().lang
    out = os.path.join(CONF, f"{lang}_common_words.txt")
    lower, upper = Counter(), Counter()
    for text in texts(lang):
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
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(words) + "\n")
    print(f"{len(words)} words -> {out}")


if __name__ == "__main__":
    main()
