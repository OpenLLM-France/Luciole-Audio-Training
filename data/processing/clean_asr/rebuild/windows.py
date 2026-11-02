"""Run a rebuild where the raw text is looked up from each segment's audio window.

`run(dataset, raw_text)` rebuilds the train manifests of `dataset`;
`raw_text(audio, start, end, seg_id)` returns the raw text of a window (canonical
tags, "?" tokens, truncations with "-") or (None, reason); `clean_current` can strip
artefacts of the current text before the case/punctuation transfer. Case and punctuation
come from the current text through `trs_asr.transfer_format`.

- "<id>_cutNN" pieces (max30 manifests) get their share of the parent window's
  raw text (`trs_asr.split_by_cuts`).
- A window that matches no raw span keeps its current text.
- A dash that transfer_format took from a compound of the current text onto a raw
  word that is not a truncation is removed ("dix- sept" -> "dix sept").
- Punctuation transferred onto the word before a tag is moved after the tag.
- Lines sharing the same audio window keep only the first one (lowest id, or
  lowest `order(id)`), the others are dropped as "duplicate_window"; their current
  texts joined serve as the case/punctuation reference of the kept one.
"""
import collections
import json
import os

from common import CONTEXT, assistant_text, audio_turn, rebuild_manifest
from trs_asr import CUT_RE, _sort_key, casepunc_text, in_raw_order, split_by_cuts

FILES = ["train_recasepunc.jsonl"]  # max30 is rebuilt only for LesVocaux, LINAGORA_Meetings (their scripts list it)


def cut_parent(rec_id):
    m = CUT_RE.match(rec_id)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def segment_window(rec):
    a = audio_turn(rec)
    s = a.get("offset", 0.0)
    return a["value"], s, s + a["duration"]


def _index(path):
    segs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            segs[rec["id"]] = (*segment_window(rec), assistant_text(rec)["value"])
    return segs


def run(dataset, raw_text, files=FILES, clean_current=None, order=_sort_key):
    all_stats = {}
    for name in files:
        segs = _index(os.path.join(CONTEXT, dataset, name))
        by_window = collections.defaultdict(list)
        for i in sorted(segs, key=order):
            a, s, e, _ = segs[i]
            by_window[(a, round(s, 3), round(e, 3))].append(i)
        groups = collections.defaultdict(list)
        for i in segs:
            parent, k = cut_parent(i)
            if parent is not None:
                groups[parent].append((k, i))
        piece_raw = {}
        for parent, members in groups.items():
            members.sort()
            ids = [i for _, i in members]
            a = segs[ids[0]][0]
            s = min(segs[i][1] for i in ids)
            e = max(segs[i][2] for i in ids)
            raw = raw_text(a, s, e, parent)
            if isinstance(raw, tuple) or raw is None:
                for i in ids:
                    piece_raw[i] = raw
                continue
            for i, r in zip(ids, split_by_cuts(raw, [segs[i][3] for i in ids])):
                piece_raw[i] = r if r.strip() else (None, "empty")
        stats = collections.Counter()

        def fn(rec):
            a, s, e, cur = segs[rec["id"]]
            same = by_window[(a, round(s, 3), round(e, 3))]
            if same[0] != rec["id"]:
                return None, "duplicate_window"
            if len(same) > 1:                  # the partial texts of the window, as case/punctuation reference
                cur = None
            raw = piece_raw[rec["id"]] if rec["id"] in piece_raw else raw_text(a, s, e, rec["id"])
            if raw is None or raw == (None, "no_raw"):
                stats["no_raw_kept_current"] += 1      # window not on Sync/anchor times
                return clean_current(cur) if clean_current else cur
            if isinstance(raw, tuple):
                return raw
            raw = " ".join(raw.split())
            if not raw:
                return None, "empty"
            if cur is None:
                cur = " ".join(in_raw_order(raw, [segs[i][3] for i in same]))
            text, transferred = casepunc_text(raw, cur, clean_current)
            stats["transferred" if transferred else "fallback"] += 1
            return text
        st = rebuild_manifest(dataset, name, fn)
        st.update(stats)
        print(f"  {dict(stats)}")
        all_stats[name] = st
    return all_stats
