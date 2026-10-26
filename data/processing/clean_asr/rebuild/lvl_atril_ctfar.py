"""Rebuild the text of LVL-Atril-CTFAR from the delivered transcripts.

Source: raw/transcript/fr/LVL-Atril-CTFAR/kaldi/complete_raw/{text,segments,extra_utt.csv}, the
Atril srt cues (Livraison20241031/<call>/<call>.srt) put in kaldi format by Linagora, ids identical to
the manifest. The text is human-written with case and punctuation; only the tags of the guidelines
(FrenchGuidelines.odt) change:
    [euh] [hum] [uh] [um] [ah] (filler words) -> the word, [répétition] [hésitation] -> removed,
    [rire] -> [laughter], [bruit] -> [noise], [inaudible] kept,
    [nom de l'entreprise] (anonymisation placeholder) -> line dropped.
Digits stay as written. The audio is the stereo call downmixed: cues of the other channel lying
entirely inside the window are inserted in time order; partial overlaps are only counted.

Usage: python3 data/processing/clean_asr/rebuild/lvl_atril_ctfar.py
"""
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import audio_turn, rebuild_manifest  # noqa: E402

DATASET = "LVL-Atril-CTFAR"
KALDI = "/data-server/datasets/audio/raw/transcript/fr/LVL-Atril-CTFAR/kaldi/complete_raw"
TAGS = {"rire": "[laughter]", "bruit": "[noise]", "inaudible": "[inaudible]",
        "répétition": "", "répétitions": "", "repetition": "", "hésitation": "", "hesitation": ""}
FILLERS = {"euh", "eu", "hum", "uh", "um", "ah"}
STATS = Counter()


def convert(text):
    def tag(m):
        inner = m[1].strip()
        key = inner.lower()
        STATS[f"[{key}]"] += 1
        if key in FILLERS:
            return inner
        if key in TAGS:
            return TAGS[key]
        return "\x00"                     # [nom de l'entreprise]
    t = re.sub(r"\[\[?([^\[\]]+)\]\]?", tag, text)
    if "\x00" in t:
        return None
    t = re.sub(r"\b0ui\b", "Oui", t)
    t = re.sub(r"\s+([?!;:,.…])", r"\1", t)
    t = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", t)
    return re.sub(r"\s+", " ", t).strip().lstrip(", ")


def main():
    text = dict((l.rstrip("\n").split(" ", 1) + [""])[:2] for l in open(os.path.join(KALDI, "text"), encoding="utf-8"))
    chan = dict(l.strip().split(",") for l in list(open(os.path.join(KALDI, "extra_utt.csv")))[1:])
    seg, by_call = {}, defaultdict(list)
    for l in open(os.path.join(KALDI, "segments")):
        u, w, s, e = l.split()
        seg[u] = (float(s), float(e))
        by_call[w].append(u)

    for name in ["train.jsonl"]:
        overlap = Counter()

        def fn(rec):
            u = rec["id"]
            if u not in text:
                return None, "not_in_raw"
            a = audio_turn(rec)
            off, dur = a.get("offset", 0.0), a["duration"]
            inside, partial = [], False
            for v in by_call["ctf-ar_" + u.split("_")[2]]:
                s, e = seg[v]
                if chan.get(v) == chan.get(u) or e <= off or s >= off + dur:
                    continue
                if s >= off - 0.02 and e <= off + dur + 0.02:
                    inside.append((s, v))
                elif min(e, off + dur) - max(s, off) > 0.2:
                    partial = True
            overlap["with_inserted_cues"] += bool(inside)
            overlap["partial_overlap"] += partial
            parts = []
            for _, v in sorted(inside + [(seg[u][0], u)]):
                t = convert(text[v])
                if t is None:
                    return None, "anonymisation" if v == u else "anonymisation_in_inserted_cue"
                parts.append(t)
            out = " ".join(p for p in parts if p)
            return out if re.search(r"\w|\[", out) else (None, "empty")
        rebuild_manifest(DATASET, name, fn)
        print("  ", dict(overlap))
    print("  markers:", dict(STATS))


if __name__ == "__main__":
    main()
