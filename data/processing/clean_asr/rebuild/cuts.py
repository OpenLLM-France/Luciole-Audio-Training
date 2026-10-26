"""Split a raw transcript over the `_cutNN` pieces of a segment (max30 manifests).

The max30 manifests cut long segments into `<id>_cut01`, `<id>_cut02`... whose
current texts, concatenated, give the text of the whole segment. The raw words
are aligned on that concatenation and each raw word goes to the cut of the
current word it matches; unmatched raw words (events, truncations lost by the
old conversion) follow the previous matched word.
"""
import difflib
import re

from common import TAG_RE, _words, norm_word

CUT_RE = re.compile(r"^(.*)_cut(\d+)$")


def parent_id(seg_id):
    m = CUT_RE.match(seg_id)
    return (m[1], int(m[2])) if m else (seg_id, None)


def _keys(tokens):
    keys, owner = [], []
    for i, t in enumerate(tokens):
        if TAG_RE.match(t) or t in {"?", "!", "."}:
            continue
        for p in re.split(r"(?<=\w)-(?=\w)", t):
            if norm_word(p):
                keys.append(norm_word(p))
                owner.append(i)
    return keys, owner


def split_raw(raw, cuts):
    """raw text + [(cut_number, current_text)] sorted -> {cut_number: raw slice}."""
    rw = raw.split()
    rkeys, rown = _keys(rw)
    fkeys, fcut = [], []
    for n, text in cuts:
        k, _ = _keys(_words(text)[0])
        fkeys += k
        fcut += [n] * len(k)
    assign = [None] * len(rw)
    sm = difflib.SequenceMatcher(None, rkeys, fkeys, autojunk=False)
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            if assign[rown[a + k]] is None:
                assign[rown[a + k]] = fcut[b + k]
    numbers = [n for n, _ in cuts]
    out = {n: [] for n in numbers}
    prev = None
    for i, t in enumerate(rw):
        c = assign[i]
        if c is None:
            nxt = next((x for x in assign[i + 1:] if x is not None), None)
            if prev is None:
                c = nxt if nxt == 1 else None     # words before cut01 is known only if cut01 exists
            elif nxt is not None and nxt - prev > 1 and (prev + 1) not in out:
                c = None                  # words of a cut missing from the manifest
            else:
                c = prev
        if c is not None and prev is not None and c < prev:
            c = prev
        if c is not None:
            out[c].append(t)
            prev = c
    return {n: " ".join(v) for n, v in out.items()}


def run(dataset, name, raw_of, transfer=True, pre=None, fmt=None):
    """rebuild_manifest with cut handling, fallback/tag/digit stats and before/after examples.

    raw_of(seg_id) -> raw text, or (None, reason) to drop; pre(rec) -> (None, reason) or None
    runs first (e.g. duplicate windows). With transfer=False the raw text is used as is;
    fmt(raw, current) -> (text, ratio) replaces transfer_format.
    """
    import collections
    import json
    import os
    from common import CONTEXT, assistant_text, rebuild_manifest, transfer_format

    groups = collections.defaultdict(list)
    with open(os.path.join(CONTEXT, dataset, name), encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pid, n = parent_id(rec["id"])
            if n is not None:
                groups[pid].append((n, assistant_text(rec)["value"]))
    slices = {}
    for pid, cuts in groups.items():
        raw = raw_of(pid)
        if isinstance(raw, tuple):
            continue
        for n, s in split_raw(raw, sorted(cuts)).items():
            slices[f"{pid}_cut{n:02d}"] = s
    st = collections.Counter()
    tags = collections.Counter()
    examples = []

    def fn(rec):
        if pre:
            r = pre(rec)
            if r is not None:
                return r
        pid, n = parent_id(rec["id"])
        raw = raw_of(rec["id"]) if n is None else slices.get(rec["id"], (None, "cut_unaligned"))
        if isinstance(raw, tuple):
            return raw
        cur = assistant_text(rec)["value"]
        text = raw
        if transfer:
            text, ratio = (fmt or transfer_format)(raw, cur)
            st["fallback"] += ratio < 0.8
        text = " ".join(text.split())
        if transfer:                             # "[breath] mais donc." -> "[breath] Mais donc."
            start = r"^" if cur[:1].isupper() else r"(?!)"
            text = re.sub(r"(%s|[.?!…] )((?:\[[a-z]+\] )+)(\w)" % start, lambda m: m[1] + m[2] + m[3].upper(), text)
        if not text:
            return None, "empty"
        if not re.sub(r"\[[a-z]+\]|[\s.,?!…]", "", text):
            st["tag_only"] += 1
        st["digits"] += bool(re.search(r"\d", text))
        tags.update(re.findall(r"\[[a-z]+\]", text))
        if text != cur:
            examples.append((rec["id"], cur, text))
        return text

    stats = rebuild_manifest(dataset, name, fn)
    print(f"  fallback(ratio<0.8)={st['fallback']} digits_lines={st['digits']} tag_only={st['tag_only']} tags={dict(tags)}")
    return stats, st, tags, examples
