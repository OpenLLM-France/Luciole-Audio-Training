"""Rebuild the text of LVL-Atril-CTFNN1 from the delivered srt files.

Source: raw/transcript/fr/LVL-Atril-CTFNN1/Livraison1.2/<call>/<call>[_transcribed].srt
("Speaker k: text" cues). Audio stays the mixed wav (Livraison1.2_mixed). The last field of a
manifest id is lvl-atril-ctfnn1_<spk>_<call>_<n>.

The text is the cue as written (mixed case as delivered, digits, truncation dashes), with
    [laugh] -> [laughter], [cough]/[caugh] -> [cough], [breath], [noise], [inaudible] kept,
    [no-speech] / [other-speech] removed,
    [beep] kept only where a tone was found in the audio (see BEEP_TONE), else the line is dropped.
Cues of the other speaker lying entirely inside the window are inserted in time order (the mixed
audio contains them); partial overlaps cannot be split and are only counted.
The id's cue number is usually the srt number + 1 but not in every file, so the cue is the one
starting at the segment offset (nearest number on ties).

Usage: python3 data/processing/clean_asr/rebuild/lvl_atril_ctfnn1.py
"""
import glob
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import audio_turn, rebuild_manifest  # noqa: E402

DATASET = "LVL-Atril-CTFNN1"
SRT = "/data-server/datasets/audio/raw/transcript/fr/LVL-Atril-CTFNN1/Livraison1.2"
TAGS = {"laugh": "[laughter]", "cough": "[cough]", "caugh": "[cough]", "breath": "[breath]", "noise": "[noise]",
        "inaudible": "[inaudible]", "no-speech": "", "no-speeche": "", "other-speech": ""}
# [beep] cues where a steady tone (> 350 Hz, > 80 ms) is in the role channel: kept as [beep];
# the two others (160613 cue 287, 165996 cue 198, srt numbering) show only speech and are dropped.
BEEP_TONE = {("165840", 63), ("165986", 287), ("152708", 259)}
CUE_RE = re.compile(r"(\d+)\s*\n(\d+):(\d+):(\d+),(\d+) --> (\d+):(\d+):(\d+),(\d+)\s*\n(Speaker \d+): ?(.*?)(?:\n\s*\n|\Z)", re.S)
STATS = Counter()


def sec(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_srt(path):
    """-> {cue number: (start, end, speaker, text)}"""
    txt = open(path, encoding="utf-8-sig").read().replace("\r", "")
    cues = {}
    for m in CUE_RE.finditer(txt):
        g = m.groups()
        cues[int(g[0])] = (sec(*g[1:5]), sec(*g[5:9]), g[9], " ".join(g[10].split()))
    return cues


def convert(text, call, cue):
    def tag(m):
        name = m[1].lower()
        if name == "beep":
            return "[beep]" if (call, cue) in BEEP_TONE else "\x00"
        STATS[f"[{name}]"] += 1
        return TAGS.get(name, m[0])
    t = re.sub(r"\[([^\]]+)\]", tag, text)
    if "\x00" in t:
        return None
    t = re.sub(r"\s+([?!;:,.])", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def main():
    calls = {}
    for path in glob.glob(os.path.join(SRT, "*", "*.srt")):
        calls[os.path.basename(os.path.dirname(path))] = parse_srt(path)

    for name in ["train.jsonl"]:
        overlap = Counter()

        def fn(rec):
            _, _, call, n = rec["id"].rsplit("_", 3)
            cues = calls[call]
            a = audio_turn(rec)
            off, dur = a.get("offset", 0.0), a["duration"]
            cands = [k for k, c in cues.items() if abs(c[0] - off) < 0.011]
            if not cands:
                return None, "cue_not_found"
            cue = min(cands, key=lambda k: abs(k - (int(n) - 1)))
            start, end, spk, _ = cues[cue]
            inside, partial = [], 0
            for k, (s, e, sp, _) in cues.items():
                if sp == spk or e <= off or s >= off + dur:
                    continue
                if s >= off - 0.02 and e <= off + dur + 0.02:
                    inside.append((s, k))
                elif min(e, off + dur) - max(s, off) > 0.2:
                    partial += 1
            overlap["with_inserted_cues"] += bool(inside)
            overlap["partial_overlap"] += bool(partial)
            parts = []
            for s, k in sorted(inside + [(start, cue)]):
                t = convert(cues[k][3], call, k)
                if t is None:
                    return None, "beep_name_audible" if k == cue else "beep_in_inserted_cue"
                parts.append(t)
            text = " ".join(p for p in parts if p)
            return text if text else (None, "empty")
        rebuild_manifest(DATASET, name, fn)
        print("  ", dict(overlap))
    print("  markers:", dict(STATS))


if __name__ == "__main__":
    main()
