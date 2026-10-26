"""Shared helpers for corpora transcribed with Transcriber (.trs): ESLO, TCOF, CFPP2000.

`trs_chunks` reads a .trs into Sync-delimited chunks (start, end, items), items being
("txt", text), ("ev", type, desc, extent), ("com", desc) or ("who", speaker), in file
order, all speakers of an overlapping turn included. `window_items` returns the
items of the chunks inside a manifest segment window.

Segments longer than 30 s were cut into "<id>_cutNN" pieces in the max30 manifests,
at points that do not fall on Sync times: `split_by_cuts` builds the raw text of the
whole parent window and splits it between the pieces by aligning it with their
current texts.
"""
import difflib
import functools
import glob
import os
import re
import xml.etree.ElementTree as ET

from common import TAG_RE, audio_turn, norm_word

EPS = 0.01
CUT_RE = re.compile(r"^(.*)_cut(\d+)$")

# Transcriber noise codes and free descriptions -> canonical tags (None: removed)
NOISE_CODES = {
    "r": "[breath]", "i": "[breath]", "e": "[breath]", "pf": "[breath]", "n": "[breath]",
    "tx": "[cough]", "bg": "[cough]", "bb": "[noise]", "b": "[noise]", "sif": "[noise]",
    "conv": "[noise]", "mic": "[noise]", "pap": "[noise]", "shh": "[noise]",
    "pi": "[inaudible]", "pif": "[inaudible]", "pron=pi": "[inaudible]", "nontrans": "[inaudible]",
}
NOISE_WORDS = [
    (r"pause", None),
    (r"rire|rires|riant", "[laughter]"),
    (r"applau", "[applause]"),
    (r"musique", "[music]"),
    (r"chant", "[singing]"),
    (r"soupir", "[sigh]"),
    (r"tou(x|sse)|r[aâ]clement", "[cough]"),
    (r"respiration|inspiration|expiration|aspiration|souffl|reniflement|hhh|^pff?$", "[breath]"),
    (r"incompr[ée]hensible|inaudible|inintelligible", "[inaudible]"),
    (r"^bip", "[beep]"),
]


def noise_tag(desc):
    d = desc.strip().lower()
    if d in NOISE_CODES:
        return NOISE_CODES[d]
    for pat, tag in NOISE_WORDS:
        if re.search(pat, d):
            return tag
    return "[noise]"


@functools.lru_cache(maxsize=8)
def trs_chunks(path):
    root = ET.parse(path).getroot()
    out = []
    for turn in root.iter("Turn"):
        spks = (turn.get("speaker") or "").split()
        cur, items = None, []
        for el in turn.iter():
            if el is turn:
                if el.text and el.text.strip():
                    items.append(("txt", el.text.strip()))
                continue
            if el.tag == "Sync":
                if cur is not None:
                    out.append((cur, float(el.get("time")), items))
                cur, items = float(el.get("time")), []
            elif el.tag == "Who":
                nb = int(el.get("nb") or 1) - 1
                items.append(("who", spks[nb] if nb < len(spks) else "?"))
            elif el.tag == "Event":
                items.append(("ev", el.get("type") or "noise", el.get("desc") or "", el.get("extent") or "instantaneous"))
            elif el.tag == "Comment":
                items.append(("com", el.get("desc") or ""))
            if el.tail and el.tail.strip():
                items.append(("txt", el.tail.strip()))
        if cur is not None:
            out.append((cur, float(turn.get("endTime")), items))
    out.sort(key=lambda c: c[0])
    return out


@functools.lru_cache(maxsize=4096)
def trs_for_audio(audio_path):
    d = os.path.dirname(audio_path.replace("/converted_audios/", "/raw/"))
    trs = sorted(glob.glob(os.path.join(glob.escape(d), "*.trs")))
    return trs[0] if trs else None


def window_items(trs, start, end):
    """Items of the chunks of `trs` lying inside [start, end], or None if none does."""
    chunks = [c for c in trs_chunks(trs) if c[0] >= start - EPS and c[1] <= end + EPS and c[1] > c[0]]
    if not chunks:
        return None
    return [it for c in chunks for it in c[2]]


def segment_window(rec):
    a = audio_turn(rec)
    s = a.get("offset", 0.0)
    return a["value"], s, s + a["duration"]


def split_by_cuts(raw, pieces):
    """Split the raw text of a parent window between its cut pieces.

    pieces: current texts of the pieces in order. Each raw token goes to the piece of
    the current word it aligns with; unaligned tokens follow the previous raw token.
    """
    rw = raw.split()
    fw, owner = [], []
    for k, t in enumerate(pieces):
        for w in t.split():
            fw.append(norm_word(w))
            owner.append(k)
    rkey = [None if TAG_RE.match(t) or t in {"?", "!", "."} else norm_word(t) for t in rw]
    sm = difflib.SequenceMatcher(None, rkey, fw, autojunk=False)
    piece_of = [None] * len(rw)
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            piece_of[a + k] = owner[b + k]
    first = next((p for p in piece_of if p is not None), 0)
    cur = first
    for i, p in enumerate(piece_of):
        if p is None or p < cur:
            piece_of[i] = cur
        else:
            cur = p
    out = [[] for _ in pieces]
    for t, p in zip(rw, piece_of):
        out[p].append(t)
    return [" ".join(o) for o in out]


def cut_parent(rec_id):
    m = CUT_RE.match(rec_id)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def render(items, clean_txt, event, comment):
    """Raw text of a window: cleaned text, event/comment tokens (None: removed), speakers dropped."""
    out = []
    for it in items:
        if it[0] == "txt":
            tok = clean_txt(it[1])
        elif it[0] == "ev":
            tok = event(*it[1:])
        elif it[0] == "com":
            tok = comment(it[1])
        else:
            tok = None
        if tok:
            out.append(tok)
    text = " ".join(" ".join(out).split())
    text = re.sub(r"(\[[a-z]+\])( \1)+", r"\1", text)     # "[noise] [noise]" -> "[noise]"
    return re.sub(r"\s+([?!])", r" \1", text)


def alternatives(text):
    """TCOF/CFPP hesitations between transcriptions "/ça, chat/" -> "ça" ("0" = nothing)."""
    text = re.sub(r"/([^/,]*),[^/]*/", lambda m: "" if m[1].strip() == "0" else m[1], text)
    return text.replace("/", " ")


def default_event(type_, desc, extent):
    """Noise events -> tags ("begin"/"end" pairs tagged once); pronounce/lexical/language removed,
    except an unintelligible pronunciation (pi, pif, *) standing alone."""
    if extent == "end":
        return None
    if type_ == "noise":
        return noise_tag(desc)
    if type_ == "pronounce" and extent == "instantaneous" and desc.strip() in {"pi", "pif", "*", "pron=pi"}:
        return "[inaudible]"
    return None
