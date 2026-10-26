"""Rebuild TCOF_Adultes / TCOF_Enfants text from raw/transcript/fr/TCOF/tcof/12/Corpus/*/<REC>/<REC>.trs.

Each segment's text is every Sync chunk inside its window, read again from the .trs
(the kaldi conversion lost the text between two <Event> of a chunk). TCOF conventions
(TCOFConventions2017.pdf): no punctuation except "?" in the adult-child corpora,
"*" / "***" unintelligible -> [inaudible], "###" untranscribed -> [inaudible],
anonymised names "¤P1¤" (a tone in the audio) -> [beep], "/a, b/" -> a,
"leur(s)" / "(n')" -> "leur" / "", pauses "+" "///", overlap "<" ">", liaisons
"=z=", cuts "$$$" removed. Pronounce events keep the orthographic word, an
unintelligible one alone -> [inaudible]; noise events -> tags. Some Enfants files
carry "ï¿½" for lost accents: the word is restored from the corpus vocabulary,
else "ï¿½" -> "é". Case and punctuation via transfer_format.
"""
import collections
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transcriber import NOISE_WORDS, alternatives, noise_tag, render, trs_chunks, trs_for_audio, window_items  # noqa: E402
from windows import run  # noqa: E402

CORPUS = "/data-server/datasets/audio/raw/transcript/fr/TCOF/tcof/12/Corpus"
BAD = "ï¿½"
ACCENTS = "àâäçéèêëîïôöùûüœÀÂÇÉÈÊÎÔ"
_vocab = None


def vocab():
    global _vocab
    if _vocab is None:
        _vocab = collections.Counter()
        for f in glob.glob(CORPUS + "/**/*.trs", recursive=True):
            if "@eaDir" in f:
                continue
            for _, _, items in trs_chunks(f):
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


def inline_tag(desc):
    """An event typed in the text ("[rire]"); others (SAMPA "[E]", "[sic]") removed."""
    d = desc.strip().lower()
    return (noise_tag(d) or "") if any(re.search(p, d) for p, _ in NOISE_WORDS) else ""


def compounds():
    """Hyphenated words of the TCOF vocabulary (not stutters like "i-il")."""
    return {w.lower() for w, n in vocab().items()
            if "-" in w.strip("-") and n >= 3 and not w.split("-")[1].lower().startswith(w.split("-")[0].lower())}


def clean_txt(t):
    if BAD in t:
        words = []
        for w in t.split():
            if BAD in w:
                pre, core, suf = re.match(r"^(\W*)(.*?)([?\W]*)$", w).groups()
                core = repair(core) if BAD in core else core
                w = pre + core + suf
            words.append(w)
        t = " ".join(words)
    t = re.sub(r"\[([^\]]*)\]", lambda m: f" {inline_tag(m[1])} ", t)
    t = re.sub(r"\{[^}]*\}", " ", t)
    t = re.sub(r"¤[A-Za-z]+\d*¤?|\*[A-Z][a-zA-Z]*\d*\*", " [beep] ", t).replace("¤", "")
    t = alternatives(t)
    t = re.sub(r"\*+|###", " [inaudible] ", t)
    t = re.sub(r"=\w+=|\$\$\$|///|\++|[<>\"«»]|(?<!\S)[#=]+(?!\S)", " ", t)
    t = re.sub(r"\([^)\s]*\)", "", t)
    t = re.sub(r"(^|\s)[.,](?=\s|$)", " ", t)
    return t


def event(type_, desc, extent):
    if extent == "end":
        return None
    if type_ == "noise":
        return noise_tag(desc)
    if type_ == "pronounce" and extent == "instantaneous":
        return "[inaudible]"          # a pronunciation with no orthographic word
    return None


def comment(desc):
    return "[laughter]" if re.search(r"\brires?\b", desc.lower()) else None


def raw_text(audio, start, end, seg_id):
    trs = trs_for_audio(audio)
    if trs is None:
        return None, "no_trs"
    items = window_items(trs, start, end)
    if items is None:
        return None, "no_raw"
    return render(items, clean_txt, event, comment) or (None, "empty")


if __name__ == "__main__":
    for ds in sys.argv[1:] or ["TCOF_Adultes", "TCOF_Enfants"]:
        run(ds, raw_text)
