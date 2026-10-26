"""Rebuild Multilingual_TEDx text from the raw mTEDx fr-fr transcripts.

Text is the raw line of raw/transcript/fr/TEDX/fr-fr/data/<split>/txt/<split>.fr
(matched by segment id through txt/segments). The old conversion removed the
[bracketed] parts, leaving holes ("les fous, les cavaliers") or a foreign text
without its translation: lines whose raw text has brackets are dropped, as are
lines with a language marker ("(En anglais)", "(Créole)"), whose text is a French
translation of foreign speech. Event parentheses ("(Rires)") are kept as they are.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import rebuild_manifest  # noqa: E402

RAW = "/data-server/datasets/audio/raw/transcript/fr/TEDX/fr-fr/data"
LANGS = (r"anglais|arabe|néerlandais|flamand|créole|allemand|espagnol|italien|grec|chinois|portugais|russe|"
         r"tibétain|japonais|latin|wolof|lingala|bambara|kabyle|berbère|hébreu|turc|swahili|hindi|polonais|"
         r"breton|occitan|basque|corse|alsacien|catalan|roumain|persan|coréen|vietnamien|langue des signes")
FOREIGN = re.compile(r"\((?![^)]*accent)[^)]{0,40}\b(?:%s)\b[^)]{0,40}\)" % LANGS, re.IGNORECASE)


def load_raw(split):
    d = os.path.join(RAW, split, "txt")
    with open(os.path.join(d, "segments"), encoding="utf-8") as f:
        ids = [line.split()[0] for line in f]
    with open(os.path.join(d, f"{split}.fr"), encoding="utf-8") as f:
        texts = [line.rstrip("\n") for line in f]
    assert len(ids) == len(texts)
    return dict(zip(ids, texts))


def make_fn(raw):
    def fn(rec):
        text = raw.get(rec["id"])
        if text is None:
            return None, "unmatched"
        if "[" in text or "]" in text:
            return None, "brackets"
        if FOREIGN.search(text):
            return None, "foreign"
        text = re.sub(r"\s+", " ", text).strip()
        return text if text else (None, "empty")
    return fn


if __name__ == "__main__":
    rebuild_manifest("Multilingual_TEDx", "train.jsonl", make_fn(load_raw("train")))
