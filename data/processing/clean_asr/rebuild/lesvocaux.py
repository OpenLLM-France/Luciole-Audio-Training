"""LesVocaux: text rebuilt from raw/transcript/fr/LesVocaux/lesvocaux/2/txt/<id>.txt (one file per message).

<nib> (name masked by a loud pure tone that replaces the word in the audio) -> [beep];
<usb> (other speaker masked by a tone, or breath/laughter: the corpus mixes both) ->
[noise], [laughter] when followed by "rires". Truncations keep their dash ("ho-",
not "monsieur"). The raw text has no punctuation (only "…" pauses, removed): case
and punctuation come from the current re-punctuated text (transfer_format, the tags
aligned on its "Beep" words so their punctuation is kept). The
max30 cuts get their slice of the file text by word alignment (cuts.py).
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import transfer_format  # noqa: E402
from cuts import run  # noqa: E402

RAW = "/data-server/datasets/audio/raw/transcript/fr/LesVocaux/lesvocaux/2/txt"
TAGS = {"[beep]", "[noise]", "[laughter]", "[inaudible]"}


def clean(t):
    t = re.sub(r"Config\(\).*$", "", t.strip(), flags=re.S)            # 442_07: pasted code
    t = re.sub(r"#\w+\s*(?=<nib>)|(?<=<nib>)\s*#\w+", " ", t)          # "#name <nib>": same name
    t = re.sub(r"#\w+", " <nib> ", t)                                  # "#city" alone
    t = re.sub(r"<usb>\s*rires\b", " [laughter] ", t)
    t = t.replace("<nib>", " [beep] ").replace("<usb>", " [noise] ")
    t = re.sub(r"\[\.\.\.\]", " [inaudible] ", t)
    t = re.sub(r"<\s*p\s*[:;]>|[<>]", " ", t)                          # "<p :>" pause, stray "<"
    t = t.replace("’", "'").replace("‘", "'")
    t = re.sub(r"(\w)' +(?=\w)", r"\1'", t)                          # "j' ai" -> "j'ai"
    t = re.sub(r"…|\.\.\.|(?<!\w)[.,;:/]|[()]|(?<!\S)-(?=\w)", " ", t)   # pauses, "-là"
    return " ".join(t.split())


def fmt(raw, cur):
    """transfer_format with the tags aligned on the "Beep" words of the current text (keeps its punctuation)."""
    tags = [t for t in raw.split() if t in TAGS]
    words = " ".join("beep" if t in TAGS else t for t in raw.split())
    text, ratio = transfer_format(words, cur)
    out, k = [], 0
    for tok in text.split():
        m = re.fullmatch(r"([^\w]*)beep([^\w]*)", tok, re.I)
        if m and k < len(tags):
            tok = tags[k] + m[2]
            k += 1
        out.append(tok)
    return " ".join(out), ratio


def main():
    raw = {}
    for f in os.listdir(RAW):
        if f.endswith(".txt"):
            raw[f[:-4]] = clean(open(os.path.join(RAW, f), encoding="utf-8-sig").read())

    def raw_of(seg_id):
        return raw.get(seg_id, (None, "no_raw"))

    return {name: run("LesVocaux", name, raw_of, fmt=fmt)
            for name in ["train_recasepunc.jsonl", "train_recasepunc_max30.jsonl"]}


if __name__ == "__main__":
    main()
