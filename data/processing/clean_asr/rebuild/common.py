"""Shared helpers to rebuild the text of ASR manifests from raw corpora.

A rebuild keeps every segment of a manifest (id, audio, offset, duration) and
only replaces the Assistant text, with text taken back from the raw corpus.
Outputs go to clean/<Dataset>/intermediate/<same file name>, and clean_asr.py
reads them from there.

`transfer_format` (datasets2nemo/fr/trs_asr.py) takes the words from the raw text and the
case/punctuation from the current text wherever the words align.
"""
import json
import os
import re
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "datasets2nemo", "fr"))
from trs_asr import TAG_RE, TAGS, _words, noise_tag, norm_word, transfer_format  # noqa: E402,F401

CLEAN = f"{os.environ['DATA_FOLDER']}/nemo/asr/fr/clean"
CONTEXT = f"{os.environ['DATA_FOLDER']}/nemo/asr/fr/context"

def assistant_text(rec):
    return next(t for t in reversed(rec["conversations"]) if t.get("from") == "Assistant" and t.get("type") == "text")


def audio_turn(rec):
    return next(t for t in rec["conversations"] if t.get("type") == "audio")


def manifest_windows(dataset, names, context_dir=CONTEXT):
    """id -> (audio file stem, start, end) of the segments of context/<dataset>/<names>.

    The parent "<id>" of "<id>_cutNN" pieces (max30 manifests) gets the window spanning its
    pieces when it is not itself in the manifests. Segment windows are read from the
    manifests instead of the kaldi dirs they were built from, so a rebuild only needs raw/
    and context/.
    """
    out, parents = {}, {}
    for name in names:
        with open(os.path.join(context_dir, dataset, name), encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                a = audio_turn(rec)
                stem = os.path.splitext(os.path.basename(a["value"]))[0]
                s = a.get("offset", 0.0)
                out[rec["id"]] = (stem, s, s + a["duration"])
                m = re.match(r"^(.*)_cut\d+$", rec["id"])
                if m:
                    p = parents.get(m[1])
                    parents[m[1]] = (stem, s, s + a["duration"]) if p is None else (stem, min(p[1], s), max(p[2], s + a["duration"]))
    for k, v in parents.items():
        out.setdefault(k, v)
    return out


def read_textgrid_bytes(b):
    """TextGrid bytes -> str: utf-16 with or without a (mangled) BOM, utf-8, cp1252."""
    if b[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return b.decode("utf-16")
    i = b.find(b"\x00F\x00i\x00l\x00e")       # utf-16 behind a mangled BOM
    if 0 <= i < 16:                               # \x00 + ascii, raw utf-8 for accents
        return b[i:].replace(b"\x00", b"").decode("utf-8", errors="replace")
    i = b.find(b"F\x00i\x00l\x00e\x00")
    if 0 <= i < 16:
        return b[i:].decode("utf-16-le", errors="replace")
    try:
        return b.decode("utf-8-sig")
    except UnicodeDecodeError:
        return b.decode("cp1252", errors="replace")


def first_tier(path):
    """{interval start rounded to 1 ms: text} of the first tier of a TextGrid (long or short format)."""
    s = read_textgrid_bytes(open(path, "rb").read()).replace('""', "\x00")
    s = s.split("item [2]")[0]
    if "item [" in s:
        iv = re.findall(r'xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "([^"]*)"', s)
    else:                                         # short format: xmin, xmax, "text" triples after the header
        vals = re.findall(r'^\s*("[^"]*"|[\d.]+)\s*$', s, re.M)
        start = next(i for i, v in enumerate(vals) if v == '"IntervalTier"')
        vals = vals[start + 5:]
        iv = [(vals[i], vals[i + 1], vals[i + 2].strip('"')) for i in range(0, len(vals) - 2, 3)]
    return {round(float(a), 3): t.replace("\x00", '"') for a, b, t in iv}


def interval_at(tier, start, eps=0.002):
    """Text of the interval of `tier` (first_tier output) starting at `start` (± eps), or None."""
    for k in (round(start, 3), round(start - 0.001, 3), round(start + 0.001, 3), round(start - eps, 3), round(start + eps, 3)):
        if k in tier:
            return tier[k]
    return None


# Praat special characters: "\c,a" -> "ça", "m\e^me" -> "même"
PRAAT = {"e'": "é", "e`": "è", "e^": "ê", 'e"': "ë", "E'": "É", "E^": "Ê", "a`": "à", "a^": "â", "A`": "À",
         "u`": "ù", "u^": "û", 'u"': "ü", "u¨": "ü", "u'": "ú", "i^": "î", 'i"': "ï", "i'": "í", "o^": "ô", 'o"': "ö",
         "c,": "ç", "C,": "Ç"}


def praat_chars(t):
    return re.sub(r"\\(\w[`'^\",¨])", lambda m: PRAAT.get(m[1], m[1][0]), t)


def rebuild_manifest(dataset, name, fn, context_dir=CONTEXT, out_dir=CLEAN):
    """Apply fn(rec) -> new text or None (drop) to context/<dataset>/<name>, write clean/<dataset>/intermediate/<name>.

    fn may also return (text, reason) with text None to count drop reasons.
    """
    src = os.path.join(context_dir, dataset, name)
    dst = os.path.join(out_dir, dataset, "intermediate", name)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    stats = {"lines": 0, "kept": 0, "changed": 0}
    with open(src, encoding="utf-8") as fin, open(dst + ".tmp", "w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            stats["lines"] += 1
            turn = assistant_text(rec)
            res = fn(rec)
            text, reason = res if isinstance(res, tuple) else (res, "dropped")
            if text is None:
                stats[reason] = stats.get(reason, 0) + 1
                continue
            stats["kept"] += 1
            stats["changed"] += text != turn["value"]
            turn["value"] = text
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(dst + ".tmp", dst)
    print(f"{dataset}/{name} -> {dst}\n  {stats}")
    return stats
