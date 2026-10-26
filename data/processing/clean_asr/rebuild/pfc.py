"""PFC: text rebuilt from the punctuated transcript tier of raw/transcript/fr/PFC/2/*/*_anon.TextGrid.

The raw tier already has case and punctuation, so it is used as is (no transfer_format).
Segment windows come from the context/ manifests. The old conversion split an
interval with an inline overlap "<E1: ...>" into several lines with the same window:
the first keeps the whole interval text, the others are dropped (duplicate_window).
Speaker labels are removed, overlap content is kept, events -> tags, Npers/Nlieu
(buzzed in the audio) -> [beep], truncations "t/" -> "t-", PFC liaison and schwa codes
("on11nVN", "une0412") removed. Digits are kept as in raw
(word lists "1 roc"). Capitals typed without their accent are accented when the corpus writes
the accented word more often ("Ca" -> "Ça", "Etats-Unis" -> "États-Unis", "A partir" -> "À
partir"), and the "Ù"/"Ë" of a mis-encoded file are "ô"/"è" ("cÙte" -> "côte").
"""
import collections
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import first_tier, interval_at, manifest_windows, praat_chars  # noqa: E402
from cuts import parent_id, run  # noqa: E402

RAW = "/data-server/datasets/audio/raw/transcript/fr/PFC/2"
FILES = ["train.jsonl"]  # max30 only for LesVocaux, LINAGORA_Meetings, CFPP2000

EVENTS = [
    (r"rires?.*|rit|sourire|en riant|rire de \w+", "[laughter]"),
    (r"toux|tousse", "[cough]"),
    (r"soupir", "[sigh]"),
    (r"inspiration|respiration|souffle", "[breath]"),
    (r"x+|inaudible|ianudible|incompréhensible|\?+", "[inaudible]"),
    (r"bruits?.*|.*bruit|raclement.*|râclement.*|reniflement|claquement.*|clic|klaxon|pluie|orage|tonnerre|"
     r"aboiements?|bébé pleure|voix extérieures|paroles au fond.*|sa fille babille", "[noise]"),
    (r"oui|non|euh|mm\.?|onomatopée[^:]*: *(mm)\.?", r"\1"),     # whispered answers, "mm": spoken
]


def map_paren(m):
    inner = m[1].strip().lower()
    for pat, tag in EVENTS:
        mm = re.fullmatch(pat, inner)
        if mm:
            if tag == r"\1":
                return f" {mm[1] or inner.rstrip('.')} "
            return f" {tag} "
    return " "                                    # (pause), editorial notes


# unaccented capital -> accented letters tried, the most frequent word in the corpus wins
ACCENTS = {"a": "àâ", "e": "éèê", "c": "ç", "i": "î", "o": "ô", "u": "û"}
# never written accented in PFC, so not learnt
_accented = {"etats": "états", "etats-unis": "états-unis", "etatsuniens": "étatsuniens", "eglise": "église",
             "ecosse": "écosse", "egypte": "égypte", "etretat": "étretat"}


def accent_table(texts, table):
    """Adds to `table` the words whose accented form is 5+ times more frequent in the corpus (any
    case) than the unaccented one ("ca" -> "ça", "ecoutez" -> "écoutez"); not "ou" / "où" (both frequent)."""
    counts = collections.Counter(w.lower() for t in texts for w in re.findall(r"[^\W\d_][\w'’-]*", praat_chars(t)))
    for w, n in counts.items():
        variants = [c + w[1:] for c in ACCENTS.get(w[0], "")]
        best = max(variants, key=lambda v: counts.get(v, 0), default=None)
        if best and counts.get(best, 0) > 5 * n and w not in table:
            table[w] = best
    return table


def accent_capitals(t):
    def repl(m):
        acc = _accented.get(m[0].lower())
        return acc[0].upper() + m[0][1:] if acc else m[0]
    t = re.sub(r"(?<=[a-zà-ÿ])Ù(?=[a-zà-ÿ])", "ô", re.sub(r"(?<=[a-zà-ÿ])Ë(?=[a-zà-ÿ])", "è", t))
    t = re.sub(r"(?<![\w'’-])A(?= [a-zà-ÿA-ZÀ-Þ])", "À", t)     # "A partir", "A Domfront"; not "plan A."
    return re.sub(r"(?<![\w'’-])[AECIOU][\w'’-]+", repl, t)


def clean(t):
    t = accent_capitals(praat_chars(t))
    # PFC phonological coding left in some transcripts: schwa "une0412", "Minis0452tre",
    # liaison "on11nVN", "les11z", "pas10", stray "autre1", "excuse0-moi" (not "CM2", "TV5")
    t = re.sub(r"(?<=[a-zà-ÿ'’])\d{4}", "", t)
    t = re.sub(r"(?<=[a-zà-ÿ'’])\d\d(?:[a-z]{1,2}(?:V[NO])?)?(?!\w)", "", t)
    t = re.sub(r"(?<=[a-zà-ÿ])\d(?=[\s,.?!-]|$)", "", t)
    t = t.replace("\\", " ").replace("´", "'")
    t = re.sub(r"^JL\s+", "", t)                    # interviewer's initials with no ":" (54bjl1, cyajl1)
    # "[lEspREs]", "[afuE uka denie]": SAMPA of what was said, next to the written word: removed
    t = re.sub(r"\[[^]]*(?:[a-z][A-Z]|[A-Z][a-z]*[A-Z]|~)[^]]*\]", " ", t)
    t = re.sub(r"\[([^]]*)\]", r"\1", t)             # "[stupide]": uncertain word, kept
    t = re.sub(r"\b(?:[A-Z][A-Z0-9]{0,8}|[A-Z][a-z])\s?:", " ", t)   # speaker labels "AA1:", "BM :", "Ma:", also inside <...>
    t = re.sub(r"[<>]", " ", t)                    # overlap brackets, content kept
    t = re.sub(r"\(([^)]*)\)", map_paren, t)
    t = re.sub(r"\b(xx+|XX+)\b", " [inaudible] ", t)
    t = re.sub(r"(?<![A-Z])N(pers|personne|lieu|prénom|prenom|nom)\w*\b", " [beep] ", t)   # also glued: "suisNpers"
    t = re.sub(r"(\w)/", r"\1- ", t)               # truncation "t/" -> "t-"
    t = re.sub(r"[/_`~|@\x00-\x1f]", " ", t)          # restart marks "il / il", stray symbols
    t = re.sub(r"\s+([,.?!])", r"\1", t)
    t = re.sub(r"([,.?!])(?=\w)", r"\1 ", t)
    t = re.sub(r"([.?!])[.,;]+|(-),", r"\1\2", t)        # "mais.<X: , tous" -> "mais. tous", "o-,"
    t = re.sub(r",([.?!])", r"\1", t)
    t = re.sub(r"^[\s,.]+", "", " ".join(t.split()))
    t = re.sub(r"([.?!] )([a-zà-ÿ])", lambda m: m[1] + m[2].upper(), t)
    return " ".join(t.split())


def main():
    windows, by_window = {}, collections.defaultdict(list)
    for u, (f, a, b) in manifest_windows("PFC", FILES).items():
        windows[u] = (f, round(a, 3))
        if parent_id(u)[1] is None:
            by_window[(f, round(a, 3), round(b, 3))].append(u)
    order = lambda u: int(u.rsplit("_", 1)[1])
    dup = {u: min(us, key=order) for us in by_window.values() if len(us) > 1 for u in us}
    grids = {os.path.basename(p)[:-9]: first_tier(p) for p in glob.glob(f"{RAW}/*/*_anon.TextGrid")}
    accent_table((t for g in grids.values() for t in g.values()), _accented)

    def raw_of(seg_id):
        if seg_id not in windows:
            return None, "no_raw"
        f, a = windows[seg_id]
        t = interval_at(grids.get(f, {}), a)
        return (None, "no_raw") if t is None else clean(t)

    def pre(rec):
        pid, _ = parent_id(rec["id"])
        if pid in dup and dup[pid] != pid:
            return None, "duplicate_window"
        return None

    return {name: run("PFC", name, raw_of, transfer=False, pre=pre) for name in FILES}


if __name__ == "__main__":
    main()
