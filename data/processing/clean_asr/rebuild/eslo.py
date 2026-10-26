"""Rebuild ESLO text from the Transcriber files of raw/transcript/fr/ESLO/ESLO_800H/2.

Each segment's text is every Sync chunk of <REC>/<REC>_C.trs inside its window, all
speakers of overlapping turns included (the kaldi conversion kept one speaker).
Kept from raw: words with their case (proper nouns), "?", truncations "é-",
hesitations; events become tags (rire -> [laughter], tx -> [cough], i/e/r/pf/n ->
[breath], musique -> [music], chante -> [singing], pi/pif -> [inaudible], other
noises such as bb, b, conv, mic -> [noise]). Removed: comments, "&" (non-standard
form marker), "<ph_.../>", "{...}", trailing "*". Case and punctuation from the
current text via transfer_format.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transcriber import default_event, noise_tag, render, trs_for_audio, window_items  # noqa: E402
from windows import run  # noqa: E402


def clean_txt(t):
    t = re.sub(r"<[^>]*>|\{[^}]*\}", " ", t)
    t = re.sub(r"\[([^\]]*)\]", lambda m: f" {noise_tag(m[1])} ", t)
    t = t.replace("&", "").replace("??", "?")
    t = re.sub(r"(?<=\w)\*", "", t)
    t = re.sub(r"(^|\s)(\+|\.\.\.|\.|,)(?=\s|$)", " ", t)
    return t


def comment(desc):
    return "[laughter]" if re.search(r"\brires?\b", desc.lower()) else None


def raw_text(audio, start, end, seg_id):
    trs = trs_for_audio(audio)
    if trs is None:
        return None, "no_trs"
    items = window_items(trs, start, end)
    if items is None:
        return None, "no_raw"
    return render(items, clean_txt, default_event, comment) or (None, "empty")


if __name__ == "__main__":
    run("ESLO", raw_text)
