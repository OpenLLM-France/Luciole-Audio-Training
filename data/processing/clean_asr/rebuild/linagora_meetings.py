"""Rebuild the text of LINAGORA_Meetings from the Transcriber files.

Source: raw/transcript/fr/LINAGORA/Corpus_linagora_interne/Corrections-v3_utf8/*.trs
(same content as Corrections-v3, which the kaldi dir was built from).

Each manifest line is a <Sync> window of a <Turn> (id: ..._Turn-NNN_seg-NNNNNNN, offset = Sync time).
Its text is the whole window as transcribed, punctuation kept, with:
    <Event> rire -> [laughter], tx -> [cough], b/pf -> [breath], conv/bg/mic/bb -> [noise],
    pronounce pi -> [inaudible] (other descs dropped, "end" extents dropped),
    (inaudible) / (incompréhensible) / () -> [inaudible], (uncertain words) -> words,
    R2_13_35 conventions: &euh -> euh, [bc-] [-ps] ... removed, [bc=ouais] -> ouais, [pron=...] removed,
    "enreg()" -> "enreg-".
Overlapping speech (<Who nb=...>) is in one window: the first manifest line of the window gets all
speakers' text, the others are dropped (duplicate_window). The max30 "_cutNN" pieces of a long
window get the raw words aligned to their old text. The one file without punctuation
(R2_13_35) gets case/punctuation from the current text via transfer_format.

Usage: python3 data/processing/clean_asr/rebuild/linagora_meetings.py
"""
import difflib
import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONTEXT, TAGS, assistant_text, audio_turn, norm_word, rebuild_manifest, transfer_format  # noqa: E402

DATASET = "LINAGORA_Meetings"
TRS = "/data-server/datasets/audio/raw/transcript/fr/LINAGORA/Corpus_linagora_interne/Corrections-v3_utf8"
EVENTS = {"rire": "laughter", "rire en fond": "laughter", "tx": "cough", "b": "breath", "pf": "breath",
          "conv": "noise", "bg": "noise", "mic": "noise", "bb": "noise", "pi": "inaudible"}
ID_RE = re.compile(r"_Turn-(\d+)_seg-\d+(?:_cut\d+)?$")
STATS = Counter()


def clean(text):
    t = text.replace("&", "")
    t = re.sub(r"\+?\[pr(?:on)?=[^\]]*\]", " ", t)
    t = re.sub(r"\[-?(?:bc|ps)-?(?:=|\s+)([^\]]*)\]", r" \1 ", t)          # backchannel of another speaker
    t = re.sub(r"\[-?(?:bc|ps)-?\]", " ", t)
    t = re.sub(r"\[(?!(?:%s)\])([^\]]*)\]" % "|".join(TAGS), r" \1 ", t)          # [division of labor]
    t = re.sub(r"\(\s*(?:inaudible|incompréhensible)?\s*\)", " [inaudible] ", t, flags=re.I)
    t = re.sub(r"(\w)\(\)", r"\1-", t)
    t = re.sub(r"\(([^()]*)\)", r" \1 ", t)
    t = re.sub(r'[*"]', " ", t)
    t = re.sub(r"\s+([?!;:,.…])", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def parse_trs(path):
    """-> {turn number: [(start, end, text), ...]} (one entry per Sync window, all speakers)."""
    root = ET.parse(path).getroot()
    turns, n = {}, 0
    for section in root.iter("Section"):
        for turn in section.iter("Turn"):
            n += 1
            windows, cur, start = [], [], float(turn.get("startTime"))

            def add(piece):
                if piece and piece.strip():
                    cur.append(piece.strip())
            add(turn.text)
            for el in turn:
                if el.tag == "Sync":
                    t = float(el.get("time"))
                    if t > start or cur:
                        windows.append([start, t, cur])
                    cur, start = [], t
                elif el.tag == "Event" and el.get("extent") != "end":
                    tag = EVENTS.get(el.get("desc"))
                    if tag and not (el.get("type") == "pronounce" and el.get("desc") != "pi"):
                        cur.append(f"[{tag}]")
                    STATS[f"event {el.get('type')}:{el.get('desc')}->{tag}"] += 1
                add(el.tail)
            windows.append([start, float(turn.get("endTime")), cur])
            turns[n] = [(s, e, clean(" ".join(c))) for s, e, c in windows]
    return turns


def has_punct(turns):
    texts = [t for ws in turns.values() for _, _, t in ws if t]
    return sum(bool(re.search(r"[.,?!]", t)) for t in texts) > 0.3 * len(texts)


def split_by_cuts(raw, cuts):
    """Distribute the raw words over the max30 pieces [(offset, old text)] by aligning with their old texts."""
    rw = raw.split()
    old, owner = [], []
    for k, (_, txt) in enumerate(cuts):
        for w in txt.split():
            old.append(norm_word(w))
            owner.append(k)
    rkey = [norm_word(w) for w in rw]
    assign = [None] * len(rw)
    for a, b, size in difflib.SequenceMatcher(None, rkey, old, autojunk=False).get_matching_blocks():
        for i in range(size):
            assign[a + i] = owner[b + i]
    last = 0
    for i in range(len(rw)):          # unaligned words (tags, lost words) follow the previous word
        assign[i] = last = assign[i] if assign[i] is not None else last
    return [" ".join(w for w, k in zip(rw, assign) if k == j) for j in range(len(cuts))]


def main():
    files = {os.path.basename(p)[:-4].lower(): parse_trs(p) for p in glob.glob(os.path.join(TRS, "*.trs"))}
    punct = {k: has_punct(v) for k, v in files.items()}
    print("files without punctuation:", [k for k, v in punct.items() if not v])
    for name in ["train.jsonl", "train_max30.jsonl"]:
        recs = [json.loads(l) for l in open(os.path.join(CONTEXT, DATASET, name), encoding="utf-8")]
        # window of each line, and old texts of the cut pieces of each window
        window, pieces = {}, defaultdict(list)
        for r in recs:
            a = audio_turn(r)
            key = os.path.basename(a["value"])[:-4].lower()
            turn = int(ID_RE.search(r["id"])[1])
            off = a.get("offset", 0.0)
            ws = files[key][turn]
            w = min(range(len(ws)), key=lambda i: (not ws[i][0] - 0.01 <= off <= ws[i][1] + 0.01, abs(ws[i][0] - off)))
            if not ws[w][0] - 0.01 <= off <= ws[w][1] + 0.01:
                STATS["window not found"] += 1
            window[r["id"]] = (key, turn, w)
            if "_cut" in r["id"]:
                pieces[(key, turn, w, r["id"].split("_Turn-")[0])].append((off, assistant_text(r)["value"], r["id"]))
        cut_text = {}
        for (key, turn, w, _), cuts in pieces.items():
            cuts.sort()
            for (_, _, rid), t in zip(cuts, split_by_cuts(files[key][turn][w][2], [(o, t) for o, t, _ in cuts])):
                cut_text[rid] = t
        seen = set()
        fallback = Counter()

        def fn(rec):
            key, turn, w = window[rec["id"]]
            a = audio_turn(rec)
            win = (a["value"], round(a.get("offset", 0.0), 3), round(a["duration"], 3))
            if win in seen:
                return None, "duplicate_window"
            seen.add(win)
            raw = cut_text[rec["id"]] if rec["id"] in cut_text else files[key][turn][w][2]
            if not re.sub(r"\[\w+\]|[\s.,?!…]", "", raw) and not re.search(r"\[\w+\]", raw):
                return None, "empty"
            if not punct[key]:
                raw, ratio = transfer_format(raw, assistant_text(rec)["value"])
                fallback["fallback" if ratio < 0.8 else "transferred"] += 1
            return raw
        rebuild_manifest(DATASET, name, fn)
        print("  transfer_format (unpunctuated file):", dict(fallback))
    for k, v in sorted(STATS.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
