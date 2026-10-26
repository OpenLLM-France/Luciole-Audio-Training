"""Shared helpers to rebuild the text of ASR manifests from raw corpora.

A rebuild keeps every segment of a manifest (id, audio, offset, duration) and
only replaces the Assistant text, with text taken back from the raw corpus.
Outputs go to clean/<Dataset>/intermediate/<same file name>, and clean_asr.py
reads them from there.

Most raw transcripts of spontaneous speech have no punctuation, while the
current text (re-cased and re-punctuated by a model) lost words, events or
truncations. `transfer_format` takes the words from the raw text and the
case/punctuation from the current text wherever the words align.

Raw text conventions expected by `transfer_format`:
    canonical event tags in brackets ([laughter], [inaudible], [beep], ...),
    truncated words ending with "-" ("ho-"), a "?" token or suffix for questions,
    proper nouns and acronyms with their case.
"""
import difflib
import json
import os
import re
import unicodedata

CLEAN = "/data-server/datasets/audio/nemo/asr/fr/clean"
CONTEXT = "/data-server/datasets/audio/nemo/asr/fr/context"

TAGS = {"laughter", "applause", "music", "noise", "cough", "breath", "sigh", "singing", "inaudible", "beep"}
TAG_RE = re.compile(r"^\[[a-z]+\]$")
PREFIX_RE = re.compile(r"^[«\"“(¿¡']+")
SUFFIX_RE = re.compile(r"[»\"”),.;:!?…]+$")


def norm_word(w):
    """Comparison key: lowercase, no surrounding punctuation, ’ -> ', no trailing truncation dash."""
    w = unicodedata.normalize("NFC", w).replace("’", "'").lower()
    w = PREFIX_RE.sub("", SUFFIX_RE.sub("", w))
    return w.rstrip("-")


def _split(token):
    """'«Bonjour,' -> ('«', 'Bonjour', ',')"""
    m1, m2 = PREFIX_RE.match(token), SUFFIX_RE.search(token)
    i = m1.end() if m1 else 0
    j = m2.start() if m2 and m2.start() >= i else len(token)
    return token[:i], token[i:j], token[j:]


def _words(text):
    """Split into word tokens; hyphenated compounds are split so that "quatre-vingts" aligns with "quatre vingts".

    Returns the tokens and the indices of the compound pieces whose trailing "-" is the compound's.
    """
    out, joined = [], set()
    for tok in text.split():
        if TAG_RE.match(tok):
            out.append(tok)
            continue
        pre, core, suf = _split(tok)
        parts = [p for p in re.split(r"(?<=\w)-(?=\w)", core)] if core else [""]
        for k, p in enumerate(parts):
            if k < len(parts) - 1:
                joined.add(len(out))
            out.append((pre if k == 0 else "") + p + ("-" if k < len(parts) - 1 else "") + (suf if k == len(parts) - 1 else ""))
    return out, joined


def transfer_format(raw, formatted, min_ratio=0.8):
    """Words of `raw`, case and punctuation of `formatted` where the words align.

    Returns (text, ratio) where ratio is the fraction of raw words found in
    `formatted`; below `min_ratio`, returns (raw, ratio) unchanged.
    Raw-only words (lost by the old conversion) are inserted as written in raw,
    formatted-only words (artefacts like "noise", "monsieur") are dropped.
    """
    rw = [t for t in raw.split() if t]
    fw, joined = _words(formatted)
    rkey = [None if TAG_RE.match(t) or t in {"?", "!", "."} else norm_word(t) for t in rw]
    fkey = [norm_word(t) if not TAG_RE.match(t) else None for t in fw]
    # compare raw words to formatted words, the raw hyphenated compounds split like `_words`
    rsplit, rmap = [], []
    for i, t in enumerate(rw):
        if rkey[i] is None:
            continue
        for p in re.split(r"(?<=\w)-(?=\w)", t):
            rsplit.append(norm_word(p))
            rmap.append(i)
    n_words = len({i for i in rmap})
    if not n_words:
        return raw, 1.0
    sm = difflib.SequenceMatcher(None, rsplit, fkey, autojunk=False)
    match, first_f = {}, {}
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            match.setdefault(rmap[a + k], []).append(fw[b + k])
            first_f.setdefault(rmap[a + k], b + k)
    # how much of the formatted text the raw text explains: raw-only words (lost
    # overlaps, events) are expected, formatted-only words mean the texts differ
    n_f = sum(k is not None and k != "" for k in fkey)
    ratio = sum(len(v) for v in match.values()) / max(n_f, 1)
    if ratio < min_ratio:
        return raw, ratio

    out = []
    for i, t in enumerate(rw):
        if rkey[i] is None:              # tag or punctuation token of the raw text
            if t in {"?", "!", "."} and out:
                out[-1] = SUFFIX_RE.sub("", out[-1]) + t
            elif t not in {"?", "!", "."}:
                out.append(t)
            continue
        if i not in match:
            out.append(t)
            continue
        parts = match[i]
        pre, core, suf = _split(parts[0])
        if len(parts) > 1:
            core = core + "".join(_split(p)[1] for p in parts[1:])
            suf = _split(parts[-1])[2]
            # the formatted text joined the compound with "-" ("quatre-vingts"): keep the raw spelling
            core = _split(t)[1] if norm_word(t).replace("-", "") == norm_word(core).replace("-", "") else core
        rpre, rcore, rsuf = _split(t)
        if len(parts) == 1 and first_f[i] in joined and not rcore.endswith("-"):
            core = core[:-1]             # raw "dix sept" against formatted "dix-sept": no "dix- sept"
        if any(c.isupper() for c in rcore[1:]) or (rcore[:1].isupper() and not core[:1].isupper()):
            core = rcore                 # proper noun / acronym case from raw
        elif core[:1].isupper() and not rcore[:1].isupper():
            # a capital the formatted text put at its own sentence start is wrong when our
            # text does not start a sentence there; a mid-sentence capital is a proper noun
            b = first_f[i]
            f_sentence_start = b == 0 or re.search(r"[.?!…]$", fw[b - 1])
            after_truncation = out and out[-1].endswith("-")   # "Ch, Chaque" -> "Ch- chaque"
            if (f_sentence_start or after_truncation) and out and not re.search(r"[.?!…]$", out[-1]):
                core = core[:1].lower() + core[1:]
        if rcore.endswith("-"):
            core = core[:1] + rcore[1:] if core else rcore   # truncation keeps its dash
            suf = ""                     # "ve-, vers": the model's comma after a truncation
        if "?" in rsuf:
            suf = SUFFIX_RE.sub("", suf) + "?"
        out.append(pre + core + suf)
    text = " ".join(out)
    # sentence starts, also after leading tags ("[laughter] Je vais")
    text = re.sub(r"([.?!…]\s+(?:\[[a-z]+\]\s+)*)([a-zà-ÿ])", lambda m: m[1] + m[2].upper(), text)
    if fw and _split(fw[0])[1][:1].isupper():       # first word: capital as in the formatted text
        text = re.sub(r"^((?:\[[a-z]+\]\s+)*[^\w\[]*)(\w)", lambda m: m[1] + m[2].upper(), text)
    return text, ratio


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
