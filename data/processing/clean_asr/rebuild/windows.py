"""Run a rebuild where the raw text is looked up from each segment's audio window.

`run(dataset, raw_text)` rebuilds the train manifests of `dataset`;
`raw_text(audio, start, end, seg_id)` returns the raw text of a window (canonical
tags, "?" tokens, truncations with "-") or (None, reason); `clean_current` can strip
artefacts of the current text before the case/punctuation transfer. Case and punctuation
come from the current text through `common.transfer_format`.

- "<id>_cutNN" pieces (max30 manifests) get their share of the parent window's
  raw text (`transcriber.split_by_cuts`).
- A window that matches no raw span keeps its current text.
- A dash that transfer_format took from a compound of the current text onto a raw
  word that is not a truncation is removed ("dix- sept" -> "dix sept").
- Punctuation transferred onto the word before a tag is moved after the tag.
- Lines sharing the same audio window keep only the first one (lowest id, or
  lowest `order(id)`), the others are dropped as "duplicate_window"; their current
  texts joined serve as the case/punctuation reference of the kept one.
"""
import collections
import difflib
import json
import os
import re

from common import CONTEXT, assistant_text, norm_word, rebuild_manifest, transfer_format
from transcriber import cut_parent, segment_window, split_by_cuts

FILES = ["train_recasepunc.jsonl"]  # max30 only for LesVocaux, LINAGORA_Meetings, CFPP2000


def _index(path):
    segs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            segs[rec["id"]] = (*segment_window(rec), assistant_text(rec)["value"])
    return segs


def _sort_key(seg_id):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", seg_id)]


def in_raw_order(raw, texts):
    """Order partial texts by where their words start in the raw text."""
    rkey = [norm_word(t) for t in raw.split()]

    def pos(t):
        sm = difflib.SequenceMatcher(None, rkey, [norm_word(w) for w in t.split()], autojunk=False)
        blocks = [b for b in sm.get_matching_blocks() if b.size]
        return max(blocks, key=lambda b: b.size).a - max(blocks, key=lambda b: b.size).b if blocks else len(rkey)
    return sorted(texts, key=pos)


def fix_dashes(raw, text):
    """transfer_format keeps the dash of a compound of the current text ("dix-sept") on a raw
    word written apart ("dix sept" -> "dix- sept"): only raw truncations keep a final dash."""
    rw = [t for t in raw.split() if t not in {"?", "!", "."}]
    tw = text.split()
    if len(rw) != len(tw):
        return text
    return " ".join(re.sub(r"-(?=[,.;:!?…»\"”)]*$)", "", t) if "-" in t and not r.rstrip("?!.,").endswith("-") else t
                    for r, t in zip(rw, tw))


def punct_after_tags(text):
    """"comme. [beep] tu" -> "comme [beep]. Tu": the punctuation transfer_format put on the
    word before a tag goes after the tag."""
    text = re.sub(r"([,.;:!?…]+)((?: \[[a-z]+\])+)(?= |$)", r"\2\1", text)
    return re.sub(r"([.?!…] )([a-zà-ÿ])", lambda m: m[1] + m[2].upper(), text)


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
            text, ratio = transfer_format(raw, clean_current(cur) if clean_current else cur)
            stats["fallback" if ratio < 0.8 else "transferred"] += 1
            if ratio < 0.8:                    # raw words as they are, "?" / "." tokens glued back
                return re.sub(r"\s+([?!.])(?=\s|$)", r"\1", text)
            return punct_after_tags(fix_dashes(raw, text))
        st = rebuild_manifest(dataset, name, fn)
        st.update(stats)
        print(f"  {dict(stats)}")
        all_stats[name] = st
    return all_stats
