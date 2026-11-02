"""ASR text from the raw corpora, shared by the --asr of the *2nemo.py converters (and clean_asr/rebuild/).

Transcriber (.trs) corpora: `trs_chunks` reads a .trs into Sync-delimited chunks (start, end, items), items being
("txt", text), ("ev", type, desc, extent), ("com", desc) or ("who", speaker), in file order, all speakers
of an overlapping turn included; `render` writes the raw text of some items, with canonical [tags].
`dump_trs_asr` writes the ASR manifest of a corpus from these chunks.

Spontaneous speech has no punctuation in its source. `transfer_format` takes the words from the raw text and
the case/punctuation from a recased text wherever the words align: the manifests of nemo/asr/<lang>/context,
which went through a recasepunc model (linto-punctuation) and stand for its output; it is not run again here.

Raw text conventions expected by `transfer_format`:
    canonical event tags in brackets ([laughter], [inaudible], [beep], ...),
    truncated words ending with "-" ("ho-"), a "?" token or suffix for questions,
    proper nouns and acronyms with their case.
"""
import collections
import difflib
import functools
import json
import logging
import os
import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

logger = logging.getLogger(__name__)

EPS = 0.01
CUT_RE = re.compile(r"^(.*)_cut(\d+)$")

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


# TCOF vocabulary, to restore the accents of "ï¿½" (TCOF and CLAPI); tcof2nemo.py sets TCOF_CORPUS from --root
TCOF_CORPUS = None    # default: $DATA_FOLDER/raw/transcript/fr/TCOF/tcof/12/Corpus
BAD = "ï¿½"
ACCENTS = "àâäçéèêëîïôöùûüœÀÂÇÉÈÊÎÔ"
_vocab = None


def vocab():
    global _vocab
    if _vocab is None:
        _vocab = collections.Counter()
        for f in Path(TCOF_CORPUS or f"{os.environ['DATA_FOLDER']}/raw/transcript/fr/TCOF/tcof/12/Corpus").rglob("*.trs"):
            if "@eaDir" in f.parts:
                continue
            for _, _, items in trs_chunks(str(f)):
                for it in items:
                    if it[0] == "txt" and BAD not in it[1]:
                        _vocab.update(re.findall(r"[\w'-]+", it[1]))
            trs_chunks.cache_clear()
    return _vocab


def _pattern(word):
    return re.compile("^" + "".join(f"[{ACCENTS}]" if p == BAD else re.escape(p) for p in re.split(f"({BAD})", word)) + "$")


def repair(word, words=None):
    """Most frequent word of `words`, then of the TCOF vocabulary, matching `word` with each "ï¿½" an accent."""
    if words is not None:
        cands = [(n, w) for w, n in words.items() if _pattern(word).match(w)]
        if cands:
            return max(cands)[1]
    cands = [(n, w) for w, n in vocab().items() if _pattern(word).match(w)]
    return max(cands)[1] if cands else word.replace(BAD, "é")      # unknown word: the most common accent


def compounds():
    """Hyphenated words of the TCOF vocabulary (not stutters like "i-il")."""
    return {w.lower() for w, n in vocab().items()
            if "-" in w.strip("-") and n >= 3 and not w.split("-")[1].lower().startswith(w.split("-")[0].lower())}


def casepunc_text(raw, current, clean_current=None):
    """The words of `raw` with the case and punctuation of the recased text `current`.

    Returns (text, transferred): below the alignment threshold, the raw words as they are.
    """
    text, ratio = transfer_format(raw, clean_current(current) if clean_current else current)
    if ratio < 0.8:                    # raw words as they are, "?" / "." tokens glued back
        return re.sub(r"\s+([?!.])(?=\s|$)", r"\1", text), False
    return punct_after_tags(fix_dashes(raw, text)), True


def _sort_key(seg_id):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", seg_id)]


def _load_casepunc(path):
    """Recased manifest -> {folder of the raw audio: [(id, record, audio turn, text)]}."""
    out = collections.defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            audio = next(t for t in rec["conversations"] if t.get("type") == "audio")
            text = next(t for t in reversed(rec["conversations"]) if t.get("from") == "Assistant" and t.get("type") == "text")
            folder = os.path.dirname(audio["value"].replace("/converted_audios/", "/raw/"))
            out[folder].append((rec["id"], rec, audio, text["value"]))
    return out


def _uncased_row(rec_id, audio_path, start, end, text):
    return {"id": f"{rec_id}_{round(start * 1000)}_{round(end * 1000)}",
            "conversations": [{"from": "User", "value": "", "type": "text"},
                              {"from": "User", "value": audio_path, "type": "audio", "duration": end - start, "offset": start},
                              {"from": "Assistant", "value": text, "type": "text"}]}


def _converted(audio):
    path = os.path.splitext(str(audio).replace("/raw/", "/converted_audios/", 1))[0] + ".wav"
    return path if os.path.exists(path) else str(audio)


def trs_rows(recording, recased, raw_text, clean_current, stats, keep_uncased=False):
    """Rows of one recording (rec_id, audio, trs) from its Sync chunks, in time order.

    A chunk becomes a line, with the case and punctuation of the recased line holding it; several
    chunks held by one recased line stay one line, as do the lines of an audio window; "<id>_cutNN"
    recased lines share out the raw text of their chunks. Chunks no recased line holds (most of them:
    overlapping speech) have no case nor punctuation: they are written only with `keep_uncased`.
    """
    rec_id, audio, trs = recording
    chunks = [c for c in trs_chunks(str(trs)) if c[1] > c[0]]
    used, rows = set(), []

    def raw_of(start, end):
        inside = [i for i, c in enumerate(chunks) if c[0] >= start - EPS and c[1] <= end + EPS]
        used.update(inside)
        return render([it for i in inside for it in chunks[i][2]], *raw_text), bool(inside)

    groups, cuts = {}, collections.defaultdict(list)
    for seg_id, rec, turn, text in sorted(recased, key=lambda r: _sort_key(r[0])):
        s = turn.get("offset", 0.0)
        cut = CUT_RE.match(seg_id)
        if cut:
            cuts[cut[1]].append((int(cut[2]), seg_id, rec, s, s + turn["duration"], text))
        else:
            groups.setdefault((round(s, 3), round(s + turn["duration"], 3)), []).append((rec, text))

    for parent, pieces in cuts.items():
        pieces.sort()
        start, end = min(p[3] for p in pieces), max(p[4] for p in pieces)
        raw, found = raw_of(start, end)
        if not found:
            stats["no_source_chunk"] += len(pieces)
            continue
        for piece, r in zip(pieces, split_by_cuts(raw, [p[5] for p in pieces])):
            if not r.strip():
                stats["empty"] += 1
                continue
            text, transferred = casepunc_text(" ".join(r.split()), piece[5], clean_current)
            stats["transferred" if transferred else "fallback"] += 1
            rows.append((piece[3], _with_text(piece[2], text)))

    for (start, end), lines in groups.items():
        stats["duplicate_window"] += len(lines) - 1
        raw, found = raw_of(start, end)
        if not found:
            stats["no_source_chunk"] += 1
            continue
        raw = " ".join(raw.split())
        if not raw:
            stats["empty"] += 1
            continue
        texts = [t for _, t in lines]
        current = texts[0] if len(texts) == 1 else " ".join(in_raw_order(raw, texts))
        text, transferred = casepunc_text(raw, current, clean_current)
        stats["transferred" if transferred else "fallback"] += 1
        rows.append((start, _with_text(lines[0][0], text)))

    for i, (start, end, items) in enumerate(chunks):
        if i in used:
            continue
        raw = " ".join(render(items, *raw_text).split())
        if not raw:
            continue
        stats["uncased_written" if keep_uncased else "uncased_skipped"] += 1
        if keep_uncased:
            text = re.sub(r"\s+([?!.])(?=\s|$)", r"\1", raw)
            rows.append((start, _uncased_row(rec_id, _converted(audio), start, end, text)))
    rows.sort(key=lambda r: r[0])
    return [r for _, r in rows]


def _with_text(rec, text):
    for t in reversed(rec["conversations"]):
        if t.get("from") == "Assistant" and t.get("type") == "text":
            t["value"] = text
            return rec
    raise ValueError(f"no Assistant text in {rec['id']}")


def dump_trs_asr(name, recordings, raw_text, casepunc_path, output_path, clean_current=None, keep_uncased=False, force=False):
    """Write the ASR manifest of `recordings` [(rec_id, audio, trs)] to `output_path`.

    raw_text: the (clean_txt, event, comment) of `render` for this corpus.
    casepunc_path: recased manifest giving the case and punctuation (it also sets which recordings
    belong to the split: without `keep_uncased`, a recording it does not hold is left out).
    """
    if os.path.exists(output_path) and not force:
        logger.info(f"[asr/{name}] {output_path} exists, skipping (use --force)")
        return
    recased = _load_casepunc(casepunc_path) if casepunc_path else {}
    stats = collections.Counter()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path + ".tmp", "w", encoding="utf-8") as f:
        for recording in recordings:
            folder = str(recording[2].parent)
            if folder not in recased and not keep_uncased:
                stats["recording_not_in_casepunc"] += 1
                continue
            for row in trs_rows(recording, recased.get(folder, []), raw_text, clean_current, stats, keep_uncased):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["lines"] += 1
            trs_chunks.cache_clear()
    os.replace(output_path + ".tmp", output_path)
    logger.info(f"[asr/{name}] {output_path}: {dict(stats)}")
