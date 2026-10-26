"""Rebuild CFPP2000 text from raw/transcript/fr/CFPP2000/CFPP2000/record-N/*.trs.

Each segment's text is every Sync chunk inside its window, all speakers of
overlapping turns included. The kaldi conversion turned the pause events into
"<noise>" (the word "noise" in 14 % of the current lines) and the alternatives
"/a, b/" into "<unk>". Here: pause events and comments removed; rire -> [laughter],
"suite de syllabes incompréhensibles" (event or comment) -> [inaudible], other noises
-> tags; the "mm" / "mm mm" / "ouais" events and the "un autre locuteur dit (mm)"
comments are speech, written as words; the "?" event and the "question" comment
become "?". "/a, b/" -> a, "(n')" / "au(x)" -> "" / "au", "*Name*" -> Name,
"s?ur" -> "sœur". Truncations "particuliè-" kept. Case and punctuation via
transfer_format.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transcriber import alternatives, noise_tag, render, trs_for_audio, window_items  # noqa: E402
from windows import run  # noqa: E402

SPEECH_EVENTS = {"mm", "mm mm", "mm mm mm", "ouais", "hm", "euh"}


def clean_txt(t):
    t = alternatives(t)
    t = re.sub(r"(?<=\w)\?(?=\w)", "œ", t)
    t = re.sub(r"\([^)\s]*\)", "", t)
    t = re.sub(r"[*|\"«»()+]", " ", t)
    t = re.sub(r"(^|\s)[.,](?=\s|$)", " ", t)
    return t


def event(type_, desc, extent):
    d = desc.strip().lower()
    if extent == "end" or type_ in {"pronounce", "lexical", "language"}:
        return None
    if d in SPEECH_EVENTS:
        return d
    if d in {"?", "!"}:
        return d
    if d in {'"', "«", "»"}:
        return None
    return noise_tag(d)


def comment(desc):
    d = desc.strip().lower()
    m = re.match(r"un autre locuteur (?:dit|prononce) \(?([^)]*)\)?$", d)
    if m:
        return m[1]
    if d == "question":
        return "?"
    if re.search(r"incompr[ée]hensible|rire|reniflement|r[aâ]clement", d):
        return noise_tag(d)
    return None


def clean_current(t):
    """Drop the "noise" / "unk" left by the kaldi conversion before the transfer."""
    t = re.sub(r"\bnoise\b|\bunk|unk\b", " ", t)
    t = re.sub(r"\s+([,.;:!?])", r"\1", " ".join(t.split()))
    return re.sub(r"([,.;:!?])[,.;:]+", r"\1", t).lstrip(",.;: ")


def raw_text(audio, start, end, seg_id):
    trs = trs_for_audio(audio)
    if trs is None:
        return None, "no_trs"
    items = window_items(trs, start, end)
    if items is None:
        return None, "no_raw"
    return render(items, clean_txt, event, comment) or (None, "empty")


if __name__ == "__main__":
    run("CFPP2000", raw_text, files=["train_recasepunc.jsonl", "train_recasepunc_max30.jsonl"],
        clean_current=clean_current)
