"""CFPB: text rebuilt from the Transcriber files raw/transcript/fr/CFPB/data/*.trs.

Segment windows come from the context/ manifests; the text is every Sync piece of the trs, all speakers and overlap
turns included, lying in the window. "(?)" -> "?", events -> tags, "(n')" (not
pronounced) dropped, "***" -> [inaudible], truncations kept with their dash.
Case and punctuation come from the current re-punctuated text (transfer_format).
"""
import glob
import html
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import manifest_windows  # noqa: E402
from cuts import run  # noqa: E402

RAW = "/data-server/datasets/audio/raw/transcript/fr/CFPB/data"
FILES = ["train_recasepunc.jsonl"]  # max30 only for LesVocaux, LINAGORA_Meetings, CFPP2000

EVENTS = [
    (r"rires?|rire \w+|sourire", "[laughter]"),
    (r"toux", "[cough]"),
    (r"inspiration|respiration|aspiration", "[breath]"),
    (r"syll?abes incompréhensibles", "[inaudible]"),
    (r"bruits?.*|reniflement|cliquetis|siffle(ment.*)?|sonnerie.*|bic.*|cri|on frappe.*|bisou|"
     r"circulation.*|ambulance.*|fort bruit.*", "[noise]"),
]


def map_paren(m):
    inner = m[1].strip()
    if inner in {"?", "?!"}:
        return " ? "
    for pat, tag in EVENTS:
        if re.fullmatch(pat, inner):
            return f" {tag} "
    return " "                                    # (n'), (à part), (voix basse): editorial


def clean(text):
    text = html.unescape(re.sub(r"<[^>]*>", " ", text))
    text = re.sub(r"\$<([^>]*)>\$", r" \1 ", text)          # foreign words: spoken, kept
    text = re.sub(r"\*{3,}|\[[^]]*\]", " [inaudible] ", text)   # *** and phonetic [vədɛ]
    text = re.sub(r"\(([^)]*)\)", map_paren, text)
    text = re.sub(r"(?<=\w)~", "-", text)                      # "s~ je" truncation
    text = re.sub(r"[\^«»\"|<>$]|/{2,}", " ", text)
    text = re.sub(r"(?<=\w)\?", " ?", text)
    return " ".join(t for t in text.split() if t != "-")


def load_pieces():
    pieces = {}                                    # file -> [(start, end, text)] in raw order
    for p in glob.glob(f"{RAW}/*.trs"):
        f = os.path.basename(p)[:-4].lower()
        s = open(p, encoding="utf-8").read()
        out = []
        for m in re.finditer(r"<Turn ([^>]*)>(.*?)</Turn>", s, re.S):
            end = float(re.search(r'endTime="([\d.]+)"', m[1])[1])
            parts = re.split(r'<Sync time="([\d.]+)"/>', m[2])
            times = [float(t) for t in parts[1::2]] + [end]
            for i in range(len(times) - 1):
                out.append((times[i], times[i + 1], parts[2 * i + 2]))
        pieces[f] = out
    return pieces


def main():
    pieces = load_pieces()
    windows = {u: (f.lower(), a, b) for u, (f, a, b) in manifest_windows("CFPB", FILES).items()}

    def raw_of(seg_id):
        if seg_id not in windows:
            return None, "no_raw"
        f, a, b = windows[seg_id]
        texts = [t for s, e, t in pieces[f]
                 if abs(s - a) < 1e-3 or (e > s and min(b, e) - max(a, s) > 0.5 * (e - s))]
        return clean(" ".join(texts))

    return {name: run("CFPB", name, raw_of) for name in FILES}


if __name__ == "__main__":
    main()
