"""ACSYNT: text rebuilt from the "Orthographe" tier of raw/transcript/fr/ACSYNT/FinalAcsynt/*/*/<REC>.TextGrid.

Each segment is the interval starting at its offset (segment windows from the context/
manifests); Praat special characters are decoded ("\\c,a" -> "ça") and the speaker label
that starts an interval ("L: ", "E : ") is removed. The few segments cut inside an interval
by the first conversion (long read texts) keep their context/ text.

The raw text has the corpus case and punctuation, used as is. Bracketed events
-> tags (the old conversion kept their description as words), "(xxx)" ->
[inaudible], interviewer interjections "<E: hum hum>" removed (the speaker's own "<L: ...>"
inside an interviewer turn is kept, label removed),
dotted acronyms "C.P." -> "CP" (not "C point P"). Parentheses of the read texts
("(Lille)", "(et non la seule Française)") are read aloud by the speakers (speech
rate check), so their words are kept. Segments with a C/V-skeleton anonymisation
("VCVC") are dropped: the name is not masked in the audio.
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONTEXT, assistant_text, first_tier, interval_at, manifest_windows, praat_chars  # noqa: E402
from cuts import run  # noqa: E402

RAW = "/data-server/datasets/audio/raw/transcript/fr/ACSYNT/FinalAcsynt"
FILES = ["train.jsonl"]  # max30 only for LesVocaux, LINAGORA_Meetings, CFPP2000
LABEL_RE = re.compile(r"^\s*[A-Z]\w{0,3}\s*:\s*")      # "L: Ah je fais", "E : Et vos grand-parents ?"

EVENTS = [
    (r"(sifflement\+)?rires?|sourire.*|sourit", "[laughter]"),
    (r"(silence et )?toux.*", "[cough]"),
    (r"inspiration|aspiration|expiration|respiration", "[breath]"),
    (r"soupir.*", "[sigh]"),
    (r"fond musical", "[music]"),
    (r"x+|hésitations inaudibles", "[inaudible]"),
]


def map_event(m):
    inner = " ".join(m[1].lower().split())
    for pat, tag in EVENTS:
        if re.fullmatch(pat, inner):
            return f" {tag} "
    return " [noise] "                            # claquement de langue, reniflement, bruit..., aboiements


def map_paren(m):
    inner = " ".join(m[1].split())
    if re.fullmatch(r"x+", inner.lower()):
        return " [inaudible] "
    if inner in {"...", "s"}:
        return ""
    if re.fullmatch(r"claquement de langue|aspiration", inner):
        return map_event(m)
    return f" {inner} "                           # read text: spoken


def acronym(m):
    rest = m.string[m.end():]
    end = m[0].endswith(".") and re.match(r"\s*$|\s+[A-ZÀ-Ý]", rest)
    return m[0].replace(".", "") + (". " if end else " ")


def clean(t):
    t = t.replace("\x85", "…").replace("`", "'")
    t = re.sub(r"<\s*E\b[^>]*>?", " ", t)                    # interviewer interjections: dropped
    t = re.sub(r"<\s*[A-Z]\w*\s*:?\s*([^>]*)>?", r" \1 ", t)    # main speaker inside the interviewer's turn: kept
    t = re.sub(r"\[([^]]*)\]", map_event, t)
    t = re.sub(r"\(([^)]*)\)", map_paren, t)
    t = re.sub(r"\((rires?|aboiements|bruit sourd)\b\.?", map_event, t)    # unclosed "(rires."
    # "C.P. au C.M.1" -> "CP au CM1"; keep a final dot when it ends a sentence
    t = re.sub(r"\b(?:[A-Z]\.)+[A-Z0-9]*\.?", acronym, t)
    t = re.sub(r'["_+^<>]', " ", t)
    t = re.sub(r"(\[[a-z]+\])|[][]", lambda m: m[1] or " ", t)    # stray brackets
    t = re.sub(r"\s+([,.?!])", r"\1", t)
    t = re.sub(r"([.?!])(?=[A-ZÀ-Ý])", r"\1 ", t)
    t = re.sub(r"^[\s,.]+", "", " ".join(t.split()))
    return " ".join(t.split())


def main():
    grids = {os.path.basename(p)[:-9]: first_tier(p) for p in glob.glob(f"{RAW}/*/*/*.TextGrid")}
    windows = manifest_windows("ACSYNT", FILES)
    current = {}                                  # context/ text, for segments cut inside an interval
    for name in FILES:
        with open(os.path.join(CONTEXT, "ACSYNT", name), encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                current[rec["id"]] = assistant_text(rec)["value"]

    def raw_of(seg_id):
        if seg_id not in windows:
            return None, "no_raw"
        rec, start, _ = windows[seg_id]
        t = interval_at(grids.get(rec, {}), start)
        t = current.get(seg_id) if t is None else LABEL_RE.sub("", praat_chars(t))
        if t is None:
            return None, "no_raw"
        if re.search(r"\b[CV]{3,}\b", t):
            return None, "anonymised_name_audible"
        return clean(t)

    return {name: run("ACSYNT", name, raw_of, transfer=False) for name in FILES}


if __name__ == "__main__":
    main()
