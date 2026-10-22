"""Pronunciation-preserving respelling of hotwords (e.g. "Chirac" -> "Chirack"/"Chyrak").

Two engines, selectable:

  * "g2p": a data-driven grapheme-to-phoneme / phoneme-to-grapheme model. Epitran aligns each
    word to its phonemes; a phoneme -> grapheme-chunk table is built from the corpus, and a
    variant is produced by swapping some chunks for other chunks that share the same phoneme.
    (Epitran: Mortensen et al., LREC 2018. Falls back to "rules" for languages Epitran cannot
    handle here, e.g. English.)
  * "rules": hand-written, language-dependent letter-substitution rules, as in Huang et al.,
    "Improving Neural Biasing for Contextual Speech Recognition by Early Context Injection and
    Text Perturbation", Interspeech 2024 (arXiv:2407.10303).

Casing variants (e.g. "Top" -> "TOP") are added for both engines and are language-independent.

A curated lexicon (data/contexts/respell_lexicon.json) overrides both engines for words they
would mangle: "ok" -> "oq" is wrong, so "ok" is mapped to its real alternative spellings
("OK", "okay"). Words shorter than three letters are also left untouched by the generic rules.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

WORD_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*", re.UNICODE)
# Plausible orthography per script (the part after the "-" in an Epitran code): letters of the
# language's own writing system, plus hyphen/apostrophe. This rejects IPA symbols (ɛ, ʒ, ɑ,
# combining marks) that Epitran sometimes leaks into its orthographic slot.
ORTHO_BY_SCRIPT = {
    "Latn": re.compile(r"^[A-Za-zÀ-ÿ'’\-]+$"),  # Latin + Latin-1 accents
    "Arab": re.compile(r"^[ء-ي٠-٩'’\-]+$"),  # Arabic letters and digits
}
DEFAULT_ORTHO_RE = ORTHO_BY_SCRIPT["Latn"]

# Epitran language codes for the languages we support.
EPITRAN_CODES = {
    "fr": "fra-Latn",
    "en": "eng-Latn",
    "es": "spa-Latn",
    "de": "deu-Latn",
    "it": "ita-Latn",
    "pt": "por-Latn",
    "nl": "nld-Latn",
    "ar": "ara-Arab",
}


def load_rules(contexts_dir: Path, lang: str):
    """Load [regex, replacement] substitution rules for ``lang`` from respell_rules.json."""
    path = contexts_dir / "respell_rules.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [(re.compile(pattern), repl) for pattern, repl in data.get(lang, [])]


def load_lexicon(contexts_dir: Path, lang: str) -> dict[str, list[str]]:
    """Curated alternative spellings for ``lang`` from respell_lexicon.json (merges the '*' section).

    These are genuine, pronunciation-preserving spellings (e.g. "ok" -> "OK"/"okay") for words the
    generic letter rules would otherwise mangle ("ok" -> "oq"). The generic respelling is skipped
    for any word listed here. Keys are lower-cased.
    """
    path = contexts_dir / "respell_lexicon.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    lexicon: dict[str, list[str]] = {}
    for section in (data.get("*"), data.get(lang)):
        if isinstance(section, dict):
            for word, alts in section.items():
                if isinstance(alts, list):
                    lexicon[word.lower()] = list(alts)
    return lexicon


def _apply_case(template: str, replacement: str) -> str:
    """Case ``replacement`` after the ``template`` span it replaces. An all-caps span (an acronym
    letter, e.g. the C of COVID) stays all-caps so no lower-case is introduced ("COVID" -> "KOVID",
    never "KoVID"); otherwise only the leading capital is matched (in-word edits)."""
    if template.isupper():
        return replacement.upper()
    if template[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _collapse_doubles(s: str) -> str:
    out: list[str] = []
    for ch in s:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


def _only_doubling(a: str, b: str) -> bool:
    """True when a and b differ only by doubled letters (t/tt, l/ll, ss/s): avoid those edits."""
    return a != b and _collapse_doubles(a.lower()) == _collapse_doubles(b.lower())


def _th_t_swap(a: str, b: str) -> bool:
    """True when a and b differ only by t <-> th (so g2p never respells 't' as 'th' or vice versa:
    'aboutir' -> 'abouthir', 'Thomas' -> 'Tomas')."""
    return a != b and a.lower().replace("th", "t") == b.lower().replace("th", "t")


def match_leading_case(template: str, replacement: str) -> str:
    """Match ``replacement``'s first-letter case to ``template``'s (both directions)."""
    if not replacement:
        return replacement
    if template[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    if template[:1].islower():
        return replacement[:1].lower() + replacement[1:]
    return replacement


class RespellEngine:
    """Generate pronunciation-preserving alternative spellings for a word or phrase."""

    # Words shorter than this are left alone by the letter/G2P rules: too little signal, and the
    # output tends to be noise (e.g. "ok" -> "oq"). Curated lexicon entries override this.
    MIN_RESPELL_LEN = 3

    def __init__(
        self,
        method: str,
        lang: str,
        rng,
        corpus_words,
        contexts_dir: Path,
        casing_max_len: int = 6,
        stopwords: set[str] | None = None,
    ):
        self.rng = rng
        self.lang = lang
        self.casing_max_len = casing_max_len
        self.stopwords = stopwords or set()  # no all-caps casing variant for a stopword ("de" -> "DE")
        self.rules = load_rules(contexts_dir, lang)
        self.lexicon = load_lexicon(contexts_dir, lang)
        self.epi = None
        self.phon2graph: dict[str, Counter] = {}
        self._align_cache: dict[str, list[tuple[str, str]]] = {}
        self.method = method
        if method == "g2p" and not self._init_g2p(corpus_words):
            self.method = "rules"  # Epitran unavailable for this language -> fall back

    # -- engine setup --------------------------------------------------------------------
    def _init_g2p(self, corpus_words) -> bool:
        code = EPITRAN_CODES.get(self.lang)
        if not code:
            return False
        # Epitran's English backend shells out to flite's "lex_lookup"; without it, every word
        # logs "lex_lookup (from flite) is not installed" and the G2P is useless. Skip to rules.
        if code == "eng-Latn" and shutil.which("lex_lookup") is None:
            return False
        try:
            import warnings

            # Epitran logs (not warns) on missing back-ends; quiet its logger to avoid spam.
            logging.getLogger("epitran").setLevel(logging.ERROR)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import epitran

                self.epi = epitran.Epitran(code)
            ortho_re = ORTHO_BY_SCRIPT.get(code.split("-")[-1], DEFAULT_ORTHO_RE)
            table: dict[str, Counter] = defaultdict(Counter)
            for word in corpus_words:
                for grapheme, phoneme in self._align(word):
                    if phoneme and ortho_re.match(grapheme):
                        table[phoneme][grapheme.lower()] += 1
            self.phon2graph = dict(table)
            return bool(self.phon2graph)
        except Exception:
            self.epi = None
            return False

    def _align(self, word: str) -> list[tuple[str, str]]:
        if word not in self._align_cache:
            try:
                self._align_cache[word] = [(t[2], t[3]) for t in self.epi.word_to_tuples(word)]
            except Exception:
                self._align_cache[word] = []
        return self._align_cache[word]

    # -- single-token respelling ---------------------------------------------------------
    def _g2p_token(self, token: str) -> str | None:
        align = self._align(token)
        # Only trust an alignment that reconstructs the original word exactly (Epitran sometimes
        # emits phonemes in its orthography slot, e.g. "Toulouse" -> [..., ('z', 'z')]).
        if not align or "".join(g for g, _ in align).lower() != token.lower():
            return None
        # Change exactly one chunk, for a single-edit variant; avoid mere consonant doubling.
        options = []
        for idx, (grapheme, phoneme) in enumerate(align):
            alts = self.phon2graph.get(phoneme)
            candidates = (
                [
                    g
                    for g in alts
                    if g != grapheme.lower() and not _only_doubling(grapheme, g) and not _th_t_swap(grapheme, g)
                ]
                if alts
                else []
            )
            # Inside an acronym (an all-caps letter), only allow a single-letter swap, so the result
            # stays a clean acronym (COVID -> KOVID), never a mixed/expanded form (COVID -> QuOVID).
            if grapheme.isupper() and len(grapheme) == 1:
                candidates = [g for g in candidates if len(g) == 1]
            if candidates:
                options.append((idx, candidates, alts))
        if not options:
            return None
        idx, candidates, alts = self.rng.choice(options)
        pick = self.rng.choices(candidates, weights=[alts[g] for g in candidates], k=1)[0]
        out = [g for g, _ in align]
        out[idx] = _apply_case(align[idx][0], pick)
        variant = "".join(out)
        return variant if variant != token else None

    def _rules_token(self, token: str) -> str | None:
        if not self.rules:
            return None
        low = token.lower()
        # All possible single substitutions (match span + replacement), then apply exactly one.
        # In an acronym region (all-caps span), only keep single-letter replacements, so the result
        # stays a clean acronym (COVID -> KOVID), never a mixed/expanded form (COVID -> QuOVID/QUOVID).
        matches = [
            (m.start(), m.end(), repl)
            for pattern, repl in self.rules
            for m in pattern.finditer(low)
            if not (token[m.start() : m.end()].isupper() and len(repl) != 1)
        ]
        if not matches:
            return None
        start, end, repl = self.rng.choice(matches)
        variant = token[:start] + _apply_case(token[start:end], repl) + token[end:]
        return variant if variant != token else None

    def _respell_token(self, token: str) -> str | None:
        # Leave curated and very short words to the lexicon / casing variants: the generic rules
        # produce noise on them ("ok" -> "oq").
        if len(token) < self.MIN_RESPELL_LEN or token.lower() in self.lexicon:
            return None
        if self.method == "rules":
            return self._rules_token(token)
        # g2p: mix g2p and rules per attempt, for more diverse variants and a clean fall back where
        # Epitran cannot align the word.
        if self.rng.random() < 0.5:
            return self._g2p_token(token) or self._rules_token(token)
        return self._rules_token(token) or self._g2p_token(token)

    def _respell_surface(self, surface: str) -> str | None:
        """Respell a word, or one random word of a multi-word phrase."""
        if " " not in surface:
            return self._respell_token(surface)
        words = surface.split(" ")
        order = list(range(len(words)))
        self.rng.shuffle(order)
        for idx in order:
            variant = self._respell_token(words[idx])
            if variant:
                words[idx] = variant
                return " ".join(words)
        return None

    # -- public API ----------------------------------------------------------------------
    def variants(self, surface: str, max_variants: int) -> list[str]:
        """Up to ``max_variants`` distinct alternative spellings (canonical not included)."""
        out: list[str] = []
        seen = {surface}
        # Curated single words: use the vetted alternatives verbatim, skip the generic rules.
        if " " not in surface and surface.lower() in self.lexicon:
            for alt in self.lexicon[surface.lower()]:
                if alt not in seen:
                    seen.add(alt)
                    out.append(alt)
            return out[:max_variants]
        attempts = 0
        while len(out) < max_variants and attempts < max_variants * 8:
            attempts += 1
            variant = self._respell_surface(surface)
            if variant and variant not in seen:
                seen.add(variant)
                out.append(variant)
        # Casing variants.
        if " " not in surface:
            if surface.isupper() and len(surface) >= 2:
                casings = [surface.capitalize(), surface.lower()]  # ZEP -> Zep, zep
            elif len(surface) <= self.casing_max_len and surface.lower() not in self.stopwords:
                casings = [surface.upper()]  # Top -> TOP (but never shout a stopword: "de" -> "DE")
            else:
                casings = []
            for form in casings:
                if form != surface and form not in seen:
                    seen.add(form)
                    out.append(form)
        return out[:max_variants]


def iter_token_spans(text: str):
    """Yield (lower_token, start, end) for each word token of ``text``."""
    return [(m.group(0).lower(), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def _rewrite_occurrence(occ: str, spelling: str) -> str:
    """Put ``spelling`` where ``occ`` was: a casing variant (same letters) verbatim so its exact
    case is kept ("voyage" -> "VOYAGE", never "vOYAGE"); an orthographic variant leading-cased to
    the occurrence ("chirac" -> "chyrack", "Chirac" -> "Chyrack")."""
    if spelling.lower() == occ.lower():
        return spelling
    return match_leading_case(occ, spelling)


def _closest_spelling(occ: str, candidates: list[str]) -> str:
    """The listed spelling closest to the real occurrence: an exact match first (so a word already
    spelled as the transcript is left untouched), otherwise the most similar one."""
    return min(
        candidates,
        key=lambda s: (s != occ, 0 if s.lower() == occ.lower() else 1, -SequenceMatcher(None, s, occ).ratio(), s),
    )


def apply_respellings(text: str, replacements: dict[str, list[str]]) -> str:
    """Rewrite each present hotword in ``text`` (keyed by its space-joined lower-case token
    sequence) to a listed spelling. ``replacements`` maps a key to the spelling(s) that appear for
    it in the hotword list; each occurrence is rewritten to the closest one (so when both ``Union``
    and ``UNION`` are listed, a transcript ``Union`` stays ``Union``)."""
    if not replacements:
        return text
    toks = iter_token_spans(text)
    low = [t[0] for t in toks]
    spans: list[tuple[int, int, list[str]]] = []
    for key, variants in replacements.items():
        seq = key.split(" ")
        n = len(seq)
        for i in range(len(low) - n + 1):
            if low[i : i + n] == seq:
                spans.append((toks[i][1], toks[i + n - 1][2], variants))
    if not spans:
        return text
    spans.sort()
    out = []
    cursor = 0
    for start, end, variants in spans:
        if start < cursor:
            continue
        occ = text[start:end]
        out.append(text[cursor:start])
        out.append(_rewrite_occurrence(occ, _closest_spelling(occ, variants)))
        cursor = end
    out.append(text[cursor:])
    return "".join(out)
