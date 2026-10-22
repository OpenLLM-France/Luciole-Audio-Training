#!/usr/bin/env python3
"""Generate contextual-biasing (hotword) training manifests from ASR manifests.

This implements the *extraction-based* recipe for contextual-biasing data: for each
utterance, a list of hotwords is appended to the transcription instruction. The list
mixes

  * "positive" hotwords that are actually spoken in the utterance, and
  * "distractor" hotwords that are not spoken,

so that the model learns to *use* the context when a listed term is present and to
*ignore* it otherwise. A tunable fraction of utterances get a list in which no listed
term is spoken at all (hard negatives), to avoid teaching the model that it should
always emit one of the hotwords.

"hotwords" here means the bias list of a contextual-biasing / ASR-customization system, not a
wake word such as "Hey Siri". Common alternative names for the same thing:

  * bias terms / biasing words / bias list (most explicit for "contextual biasing")
  * boosting words / word boosting (NVIDIA Riva, NeMo)
  * keyterms (AssemblyAI)
  * phrase hints / speech adaptation (Google)
  * custom vocabulary / custom dictionary (Speechmatics, Rev)
  * phrase list (Azure)

Candidate hotwords are computed over *all* input manifests and include both single words
and multi-word expressions (named-entity phrases such as "Sciences du Langage", and strong
statistical collocations). A word is eligible when it is a proper noun, an acronym, or
simply rare; proper nouns, acronyms and phrases are favoured. Rarity is measured on
case-insensitive words, but hotwords keep their case-sensitive surface form ("Tomi",
"NASA"). The rarity cutoff is inferred so the natural share of utterances with no pool
hotword (word or entity phrase) is close to --target-no-positive.

The hotword-prompt language is chosen per utterance: with --lang auto (default) it is
detected from the *instruction* with fastText (so a French transcription whose instruction
is in English gets an English prompt label); otherwise --lang is a fixed language code. A
single data/contexts/biasing_contexts.json holds a short list of labels per language.
fastText is unreliable on short instructions ("Transcris la parole" scores higher for
Italian than French): detection first looks the instruction up among the known ASR-context
templates (data/contexts/<LAN>_asr_contexts.json), which gives an exact language, and only
falls back to fastText biased toward the corpus-majority language via an empirical prior:
the per-prompt label is the argmax of p(lang|text) * prior[lang] (see --lang-prior /
--no-lang-prior and --lang-prior-strength).

Examples whose transcription carries an anonymisation marker (e.g. PFC "Npers") are dropped by
default (--discard-anonymized, patterns in data/contexts/anonymization_patterns.json), and ordinary
common words of the audio language are kept out of the pool (--exclude-common-words).

Input manifests use the conversation format::

    {"id": ..., "conversations": [
        {"from": "User", "value": "<instruction>", "type": "text"},
        {"from": "User", "value": "<audio path>", "type": "audio", ...},
        {"from": "Assistant", "value": "<transcription>", "type": "text"}
    ]}

For every input ``<name>.jsonl`` two files are written next to the output:
``<name>_contextual_biasing.jsonl`` (the manifest) and
``<name>_contextual_biasing_README.txt`` (command line, parameters, detected-language
ratios, and the hotword list with per-hotword usage counts).

Example::

    python data/generation/generate_contextual_biasing.py \\
        /data-server/datasets/audio/nemo/asr/fr/context/ACSYNT/train.jsonl \\
        --lid-model lid.176.ftz

Language identification uses a fastText model. To enable it::

    pip install fasttext            # or: pip install fasttext-wheel  (prebuilt wheels)

    # fastText language-ID model "lid.176" (download once):
    #   small, ~0.9 MB (recommended):
    wget -O lid.176.ftz https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
    #   full, ~126 MB (slightly more accurate):
    wget -O lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

The model path is taken from --lid-model, else $FASTTEXT_LID_MODEL, else
<contexts-dir>/lid.176.ftz (or .bin), else ./lid.176.ftz (or .bin).

Phonetic respelling (--respell-method g2p) uses Epitran for grapheme<->phoneme alignment::

    pip install epitran

This covers ar, de, es, fr, it, nl and pt out of the box. English (en) additionally needs
flite's "lex_lookup" binary; without it the script logs nothing and falls back to the rule-based
respelling for English only. To enable English G2P, build flite from source (the Debian/Ubuntu
"flite" package does NOT ship lex_lookup)::

    git clone https://github.com/festvox/flite.git
    cd flite && ./configure && make && sudo make install
    cd testsuite && make lex_lookup && sudo cp lex_lookup /usr/local/bin/

lex_lookup must be on $PATH (check with ``which lex_lookup``). No install is needed for the other
languages, nor for --respell-method rules.

Common-word filtering (--exclude-common-words, on by default) keeps ordinary high-frequency words
of the audio language out of the hotword pool. It uses bundled stopword lists plus word-frequency
data from wordfreq, which is required when the option is on::

    pip install wordfreq

Run with --no-exclude-common-words to skip this and avoid the dependency.

References (what each one informs):

  * Huang, Yarmohammadi, Khudanpur, Povey, "Improving Neural Biasing for Contextual Speech
    Recognition by Early Context Injection and Text Perturbation", Interspeech 2024
    (arXiv:2407.10303). Perturbs rare words with similar-sounding alternative spellings (in the
    reference AND the context) so the model relies on context. -> the --respell-* options, the
    "rules" respelling engine (respell.py), and the reference-perturbation in build_list_variant.
  * Mortensen, Dalmia, Littell, "Epitran: Precision G2P for Many Languages", LREC 2018.
    -> the grapheme<->phoneme alignment behind --respell-method g2p (respell.py).
  * Pundak, Sainath, Prabhavalkar, Kannan, Zhao, "Deep Context: End-to-End Contextual Speech
    Recognition" (CLAS), SLT 2018 (arXiv:1808.02480). Bias lists built from reference phrases
    plus distractors plus a no-bias option. -> the positives/distractors/no-hotword recipe in
    build_list_variant, --target-no-positive, and --paired.
  * Joulin, Grave, Bojanowski, Mikolov, "Bag of Tricks for Efficient Text Classification", 2016
    (fastText lid.176 model). -> language identification for --lang auto (LanguageIdentifier).
  * Speer et al., "wordfreq" (word-frequency data for many languages).
    -> common-word filtering for --exclude-common-words (make_common_word_predicate).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import sys
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from itertools import accumulate
from pathlib import Path

from respell import RespellEngine, apply_respellings, match_leading_case

# A "word" is a run of unicode letters, keeping internal hyphens so hyphenated compounds
# stay whole ("mother-in-law", "Anglo-saxons"). Apostrophes still split off clitics
# ("l'", "d'", "qu'").
WORD_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*", re.UNICODE)
# Sentence boundaries, used to tell a sentence-initial capital from a proper noun.
STRONG_END_RE = re.compile(r"[!?…]")  # always end a sentence
DECIMAL_DOT_RE = re.compile(r"(?<=\d)\.(?=\d)")  # 3.14 -> not a sentence end
# Abbreviations whose trailing dot is not a sentence end (title/reference abbreviations, so the
# following capitalised word, e.g. "M. Dupont", is a real proper noun, not a sentence start).
ABBREVIATIONS = {
    "m",
    "mm",
    "mme",
    "mmes",
    "mlle",
    "mlles",
    "mr",
    "dr",
    "pr",
    "me",
    "mgr",
    "st",
    "ste",
    "sts",
    "cf",
    "p",
    "pp",
    "fig",
    "vol",
    "no",
    "nos",
    "art",
    "chap",
    "éd",
    "env",
    "cie",
    "ca",
}


def is_sentence_break(gap: str, prev_surface: str) -> bool:
    """Whether the text between two word tokens ends a sentence (ignoring abbreviation dots like
    "M." and decimal dots like "3.14")."""
    if STRONG_END_RE.search(gap):
        return True
    if "." not in gap:
        return False
    if prev_surface and (prev_surface.lower() in ABBREVIATIONS or (len(prev_surface) == 1 and prev_surface.isupper())):
        return False  # abbreviation / initial, e.g. "M. Dupont", "J. S. Bach"
    return "." in DECIMAL_DOT_RE.sub("", gap)


# Lower-case function words allowed to bridge the capitalised tokens of a named entity
# (e.g. "Sciences du Langage", "Conseil d'Etat", "New York Times", "Marks and Spencer").
CONNECTORS = {
    "de",
    "du",
    "des",
    "d",
    "l",
    "le",
    "la",
    "les",
    "et",
    "of",
    "the",
    "and",
    "von",
    "van",
    "der",
    "den",
    "di",
    "da",
    "del",
    "della",
    "do",
    "dos",
    "e",
    "y",
    "und",
    "el",
    "lo",
}

# Hesitation / filler tokens kept out of hotwords and multi-word expressions (item: avoid
# disfluencies). Transcription-convention spellings across the supported languages.
FILLERS = {
    "euh",
    "heu",
    "hum",
    "heum",
    "hein",
    "ben",
    "bah",
    "bè",
    "beh",
    "mmh",
    "mh",
    "hmm",
    "hm",
    "ah",
    "oh",
    "eh",
    "ouais",
    "bof",
    "pff",
    "pfff",
    "enfin",
    "quoi",
    "bon",
    "uh",
    "uhm",
    "um",
    "umm",
    "erm",
    "er",
    "hmmm",
    "mm",
    "mmm",
    "yeah",
    "well",
    "like",
    "este",
    "pues",
    "bueno",
    "ehm",
    "mah",
    "boh",
    "cioè",
    "äh",
    "ähm",
    "öh",
}

# A lone trailing capital after a lower-case word ("frèresL", "étudesL", "petyteE") is almost always
# a glued annotation marker (laughter "L", overlap codes), not a real word.
ARTIFACT_TAIL_RE = re.compile(r"[a-zà-öø-ÿ][A-ZÀ-ÖØ-Þ]$")


def is_annotation_artifact(surface: str) -> bool:
    """True for a candidate surface that looks like a transcription-annotation artifact rather than a
    real word (a word with a single glued trailing capital, e.g. "étudesL", "demi-frèresL")."""
    return bool(ARTIFACT_TAIL_RE.search(surface))


def is_anonymized(text: str, patterns: list[re.Pattern]) -> bool:
    """True when the transcription carries an anonymisation marker (e.g. PFC "Npers", ACSYNT CVC)."""
    return any(p.search(text) for p in patterns)


@dataclass
class HotwordEntry:
    """One candidate hotword (single word or multi-word expression) in the pool."""

    key: str  # lower-cased form (words: "nasa"; phrases: "sciences du langage")
    surface: str  # case-sensitive form shown in the prompt
    weight: float  # sampling weight (rarity x category boost)
    freq: int  # corpus frequency (case-insensitive count)
    is_proper_noun: bool = False
    is_acronym: bool = False
    is_ngram: bool = False
    ngram_kind: str = ""  # "entity" or "colloc" for multi-word expressions


@dataclass
class _Stats:
    """Accumulator for a short end-of-run report."""

    n_augmented: int = 0
    n_skipped: int = 0
    n_anonymized: int = 0
    n_hard_negative: int = 0
    n_twins: int = 0
    list_sizes: list[int] = field(default_factory=list)
    lang_counts: Counter = field(default_factory=Counter)
    used_present: Counter = field(default_factory=Counter)  # times a hotword was a spoken positive
    used_distractor: Counter = field(default_factory=Counter)  # times a hotword was an absent distractor
    # Per-spelling usage, keyed by (hotword key, exact spelling): present vs absent.
    present_spell: Counter = field(default_factory=Counter)
    absent_spell: Counter = field(default_factory=Counter)
    lang_prompts: dict = field(default_factory=lambda: defaultdict(set))  # detected lang -> distinct prompts

    @property
    def mean_list_size(self) -> float:
        return sum(self.list_sizes) / len(self.list_sizes) if self.list_sizes else 0.0

    @property
    def hard_negative_ratio(self) -> float:
        return self.n_hard_negative / self.n_augmented if self.n_augmented else 0.0


# --------------------------------------------------------------------------------------
# Manifest helpers
# --------------------------------------------------------------------------------------
def iter_jsonl(path: Path):
    """Yield (line_number, parsed_record) for each non-empty line of a jsonl file."""
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield i, json.loads(line)


def get_transcription(record: dict) -> str | None:
    """Return the last Assistant text turn (the reference transcription), or None."""
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        return None
    for turn in reversed(conversations):
        if turn.get("from") == "Assistant" and turn.get("type", "text") == "text":
            value = turn.get("value")
            return value if isinstance(value, str) else None
    return None


def get_instruction_index(record: dict) -> int | None:
    """Return the index of the first User text turn (the instruction), or None."""
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        return None
    for idx, turn in enumerate(conversations):
        if turn.get("from") == "User" and turn.get("type", "text") == "text":
            return idx
    return None


def set_transcription(record: dict, text: str) -> None:
    """Overwrite the last Assistant text turn (used for reference-perturbation respelling)."""
    for turn in reversed(record.get("conversations", [])):
        if turn.get("from") == "Assistant" and turn.get("type", "text") == "text":
            turn["value"] = text
            return


# --------------------------------------------------------------------------------------
# Tokenisation and proper-noun / acronym detection
# --------------------------------------------------------------------------------------
def iter_tokens(text: str):
    """Yield (surface, start, end, is_sentence_start) for each word token of ``text``."""
    prev_end = 0
    prev_surface = ""
    first = True
    for match in WORD_RE.finditer(text):
        gap = text[prev_end : match.start()]
        is_start = first or is_sentence_break(gap, prev_surface)
        yield match.group(0), match.start(), match.end(), is_start
        prev_end = match.end()
        prev_surface = match.group(0)
        first = False


def tokenize_with_positions(text: str) -> list[tuple[str, bool]]:
    """Split ``text`` into (surface, is_sentence_start) word tokens."""
    return [(surface, is_start) for surface, _, _, is_start in iter_tokens(text)]


def lower_tokens(text: str) -> list[str]:
    """Ordered lower-cased word tokens of ``text``."""
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def is_acronym_surface(surface: str) -> bool:
    """True for an all-caps token of length >= 2 (e.g. ``NASA``, ``SNCF``)."""
    return len(surface) >= 2 and surface.isupper()


def starts_upper(surface: str) -> bool:
    """True when the first character is an upper-case letter."""
    return bool(surface) and surface[:1] != surface[:1].lower()


def is_entity_token(surface: str, is_start: bool) -> bool:
    """A token that looks like part of a named entity: capitalised mid-sentence, or acronym.

    Single-letter capitals (initials, speaker/annotation markers like "L", "E") are excluded.
    """
    if is_acronym_surface(surface):
        return True
    return len(surface) >= 2 and starts_upper(surface) and not is_start


# A gap between two tokens of the same expression may hold only whitespace, hyphens or
# apostrophes (so "Saint-Sernin" and "Conseil d'Etat" survive, but a comma breaks the run).
CLEAN_GAP_RE = re.compile(r"^[\s'’‑–-]*$")


def clean_gap(text: str, a_end: int, b_start: int) -> bool:
    """True when the text between two tokens contains no expression-breaking punctuation."""
    return bool(CLEAN_GAP_RE.match(text[a_end:b_start]))


def has_phrase(toks: list[str], phrase_by_len: dict[int, set]) -> bool:
    """True when any phrase in ``phrase_by_len`` occurs as a contiguous run of ``toks``."""
    for size, keyset in phrase_by_len.items():
        for i in range(len(toks) - size + 1):
            if tuple(toks[i : i + size]) in keyset:
                return True
    return False


# --------------------------------------------------------------------------------------
# Multi-word expression detection
# --------------------------------------------------------------------------------------
def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b, strict=False):
        if ca != cb:
            break
        n += 1
    return n


def has_internal_stutter(tokens: tuple[str, ...]) -> bool:
    """True when two tokens are (near-)duplicates: identical ("et et", "Mazamet ... Mazamet"),
    one a prefix of the other, or sharing a long common prefix ("anglo-sas"/"anglo-saxons")."""
    for a in range(len(tokens)):
        for b in range(a + 1, len(tokens)):
            x, y = tokens[a], tokens[b]
            if x == y:
                return True
            cpl = _common_prefix_len(x, y)
            if cpl >= 4 and cpl >= 0.7 * min(len(x), len(y)):
                return True
    return False


def is_hyphen_stutter(key: str) -> bool:
    """True for a hyphenated token whose parts repeat or near-repeat ("au-au", "nord-amé-nord")."""
    if "-" not in key:
        return False
    parts = key.split("-")
    return len(set(parts)) < len(parts) or has_internal_stutter(tuple(parts))


def iter_entity_phrases(text: str, max_n: int):
    """Yield (token_tuple_lower, surface) for named-entity phrases in ``text``.

    A phrase is a maximal run of entity tokens (capitalised mid-sentence, or acronym) that
    may be bridged by lower-case connectors, trimmed to its entity tokens, with >= 2 of them.
    """
    toks = list(iter_tokens(text))
    n = len(toks)
    i = 0
    while i < n:
        surface_i, _, _, start_i = toks[i]
        if not is_entity_token(surface_i, start_i):
            i += 1
            continue
        last_entity = i
        k = i + 1
        while k < n and (k - i) < max_n:
            if not clean_gap(text, toks[k - 1][2], toks[k][1]):
                break  # punctuation (e.g. a comma) ends the entity run
            surface_k, _, _, start_k = toks[k]
            if is_entity_token(surface_k, start_k):
                last_entity = k
                k += 1
            elif surface_k.lower() in CONNECTORS:
                k += 1
            else:
                break
        n_entities = sum(1 for t in range(i, last_entity + 1) if is_entity_token(toks[t][0], toks[t][3]))
        if last_entity > i and n_entities >= 2:
            key = tuple(toks[t][0].lower() for t in range(i, last_entity + 1))
            if not has_internal_stutter(key) and not any(t in FILLERS for t in key):
                surface = text[toks[i][1] : toks[last_entity][2]]
                yield key, surface
        i = last_entity + 1


def split_sentences(toks: list[tuple[str, int, int, bool]]):
    """Split a token list (from iter_tokens) into per-sentence sublists."""
    sentence: list[tuple[str, int, int, bool]] = []
    for tok in toks:
        if tok[3] and sentence:  # is_sentence_start
            yield sentence
            sentence = []
        sentence.append(tok)
    if sentence:
        yield sentence


# --------------------------------------------------------------------------------------
# Corpus scan and pool construction
# --------------------------------------------------------------------------------------
def scan_corpus(paths: list[Path], args: argparse.Namespace):
    """First pass: single-word statistics, and named-entity phrase counts."""
    freq: Counter[str] = Counter()
    n_nonstart: Counter[str] = Counter()
    n_nonstart_cap: Counter[str] = Counter()
    n_acronym: Counter[str] = Counter()
    surfaces: dict[str, Counter[str]] = defaultdict(Counter)
    nonstart_surfaces: dict[str, Counter[str]] = defaultdict(Counter)  # casing away from sentence starts
    doc_freq: Counter[str] = Counter()  # number of transcriptions containing the word
    repair: Counter[str] = Counter()  # token is a strict prefix of the next token (disfluency)
    entity_freq: Counter[tuple] = Counter()
    entity_surf: dict[tuple, Counter[str]] = defaultdict(Counter)
    collect_entities = args.ngrams != "none"

    for path in paths:
        for _, record in iter_jsonl(path):
            text = get_transcription(record)
            if not text or is_anonymized(text, args.anon_patterns):
                continue
            toks = tokenize_with_positions(text)
            seen: set[str] = set()
            for i, (surface, is_start) in enumerate(toks):
                key = surface.lower()
                if len(key) < args.min_word_len:
                    continue
                freq[key] += 1
                seen.add(key)
                surfaces[key][surface] += 1
                if is_acronym_surface(surface):
                    n_acronym[key] += 1
                if not is_start:
                    n_nonstart[key] += 1
                    nonstart_surfaces[key][surface] += 1
                    if starts_upper(surface):
                        n_nonstart_cap[key] += 1
                if i + 1 < len(toks):
                    nxt = toks[i + 1][0].lower()
                    if len(nxt) > len(key) and nxt.startswith(key):
                        repair[key] += 1
            for key in seen:
                doc_freq[key] += 1
            if collect_entities:
                for key, surface in iter_entity_phrases(text, args.ngram_max_n):
                    entity_freq[key] += 1
                    entity_surf[key][surface] += 1
    return (
        freq,
        n_nonstart,
        n_nonstart_cap,
        n_acronym,
        surfaces,
        nonstart_surfaces,
        doc_freq,
        repair,
        entity_freq,
        entity_surf,
    )


def classify_keys(freq, n_nonstart, n_nonstart_cap, n_acronym, surfaces, audio_lang, args):
    """Return {key: (is_proper_noun, is_acronym)} for eligible keys, plus their set.

    Proper nouns are told from sentence-initial capitals by looking at mid-sentence (non-start)
    capitalisation only; a word seen *only* at sentence starts but always capitalised is taken as a
    proper noun too. In German every noun is capitalised, so capitalisation is not a proper-noun
    signal there and is ignored (eligibility then rests on rarity and acronyms).
    """
    german = audio_lang == "de"
    classification: dict[str, tuple[bool, bool]] = {}
    proper_or_acronym: set[str] = set()
    for key, count in freq.items():
        if count < args.min_count:
            continue
        is_acronym = n_acronym[key] / count >= 0.5
        if german:
            is_proper = False
        elif n_nonstart[key] > 0:
            is_proper = n_nonstart_cap[key] / n_nonstart[key] >= args.proper_ratio
        else:
            # Only ever at sentence starts: proper noun if always capitalised (never seen lower-case).
            is_proper = all(starts_upper(s) for s in surfaces[key])
        classification[key] = (is_proper, is_acronym)
        if is_proper or is_acronym:
            proper_or_acronym.add(key)
    return classification, proper_or_acronym


def collect_empty_stats(paths, proper_or_acronym, rank_of, eligible_keys, entity_by_len, anon_patterns):
    """Second pass: per augmentable utterance that has no proper noun/acronym and no entity
    phrase, record the rank of its rarest eligible word. Returns (N, rmax_list).
    """
    n_total = 0
    pnless_rmax: list[int] = []
    for path in paths:
        for _, record in iter_jsonl(path):
            text = get_transcription(record)
            if text is None or get_instruction_index(record) is None:
                continue
            if is_anonymized(text, anon_patterns):
                continue
            n_total += 1
            toks = lower_tokens(text)
            keyset = set(toks)
            if keyset & proper_or_acronym:
                continue
            if entity_by_len and has_phrase(toks, entity_by_len):
                continue
            ranks = [rank_of[k] for k in keyset if k in eligible_keys]
            pnless_rmax.append(max(ranks) if ranks else -1)
    return n_total, pnless_rmax


def prefix_disfluency_keys(freq: Counter, max_count: int = 2, factor: float = 3.0, min_completion: int = 3) -> set[str]:
    """Rare words that are a strict prefix of a much more frequent word (e.g. 'racont' ->
    'racontent', 'ain' -> 'ainsi'): likely interrupted-word disfluencies."""
    vocab = sorted(freq)
    out: set[str] = set()
    for w in vocab:
        count = freq[w]
        if count > max_count:
            continue
        threshold = max(min_completion, factor * count)
        k = bisect_left(vocab, w) + 1  # words with prefix w sort right after w itself
        while k < len(vocab) and vocab[k].startswith(w):
            if len(vocab[k]) > len(w) and freq[vocab[k]] >= threshold:
                out.add(w)
                break
            k += 1
    return out


def choose_cutoff(pnless_rmax: list[int], n_total: int, target: float) -> tuple[int, float]:
    """Infer the rarity cutoff rank K so the natural no-hotword rate approaches ``target``.

    A word is "rare" when its frequency rank is >= K; raising K shrinks the pool and raises
    the natural rate (monotonic). Returns (K, natural_rate).
    """
    if n_total == 0 or not pnless_rmax:
        return 0, 0.0
    sorted_r = sorted(pnless_rmax)
    m = len(sorted_r)
    desired = int(target * n_total)
    cutoff = sorted_r[-1] + 1 if desired >= m else sorted_r[desired]
    cutoff = max(cutoff, 0)
    natural = bisect_left(sorted_r, cutoff) / n_total
    return cutoff, natural


def choose_surface(nonstart_counts: Counter[str], all_counts: Counter[str]) -> str:
    """Canonical case-sensitive form: the most common casing away from sentence starts (so a
    sentence-initial capital does not win), falling back to the overall most common form."""
    counts = nonstart_counts or all_counts
    return counts.most_common(1)[0][0]


def build_word_pool(
    freq,
    surfaces,
    nonstart_surfaces,
    classification,
    rank_of,
    cutoff,
    total_tokens,
    disfluency_keys,
    args,
    doc_freq=None,
    n_docs=0,
    stopwords=frozenset(),
    is_common=None,
):
    """Single-word pool: proper nouns, acronyms, and words rarer than the cutoff.

    Excludes words that are too frequent to be useful bias terms (a doc-frequency cap, which drops
    ubiquitous capitalised words like "Monsieur"/"Commission"), and ordinary common words / stopwords
    of the audio language (kept only when they are proper nouns or acronyms)."""
    pool: list[HotwordEntry] = []
    max_df = n_docs * args.max_docfreq_ratio if (doc_freq is not None and n_docs) else None
    for key, count in freq.items():
        if count < args.min_count or key in disfluency_keys:
            continue
        is_proper, is_acronym = classification[key]
        is_rare = rank_of[key] >= cutoff
        if not (is_proper or is_acronym or is_rare):
            continue
        if max_df is not None and doc_freq.get(key, 0) > max_df:
            continue  # ubiquitous word: no biasing value
        if not (is_proper or is_acronym) and (key in stopwords or (is_common is not None and is_common(key))):
            continue  # ordinary common word / stopword of the audio language
        surface = choose_surface(nonstart_surfaces[key], surfaces[key])
        idf = math.log((total_tokens + 1.0) / (count + 1.0)) + 1e-3
        if is_acronym:
            boost = args.acronym_boost
        elif is_proper:
            boost = args.proper_noun_boost
        else:
            boost = 1.0
        pool.append(HotwordEntry(key, surface, idf * boost, count, is_proper, is_acronym))
    return pool


def build_entity_pool(entity_freq, entity_surf, total_tokens, args) -> list[HotwordEntry]:
    """Pool entries for named-entity phrases."""
    pool: list[HotwordEntry] = []
    for key, count in entity_freq.items():
        surface = entity_surf[key].most_common(1)[0][0]
        idf = math.log((total_tokens + 1.0) / (count + 1.0)) + 1e-3
        pool.append(
            HotwordEntry(" ".join(key), surface, idf * args.ngram_boost, count, is_ngram=True, ngram_kind="entity")
        )
    return pool


def extract_collocations(
    paths, freq, word_pool_keys, proper_or_acronym, entity_keys, total_tokens, args
) -> list[HotwordEntry]:
    """Third pass: strong statistical collocations of common words anchored on a rare word.

    A collocation is a *contiguous* run of content tokens (only whitespace between them, so it
    never crosses a comma or other punctuation), none of which is a proper noun or acronym
    (those are left to entity detection), with at least one rare pool word to anchor it. The
    anchor guarantees a collocation never adds coverage beyond the words, so it does not
    perturb the inferred no-hotword rate.
    """
    colloc_freq: Counter[tuple] = Counter()
    colloc_surf: dict[tuple, Counter[str]] = defaultdict(Counter)
    max_n = min(args.ngram_max_n, 3)

    for path in paths:
        for _, record in iter_jsonl(path):
            text = get_transcription(record)
            if not text or is_anonymized(text, args.anon_patterns):
                continue
            for sentence in split_sentences(list(iter_tokens(text))):
                for size in range(2, max_n + 1):
                    for j in range(len(sentence) - size + 1):
                        window = sentence[j : j + size]
                        if any(len(w[0]) < args.min_word_len for w in window):
                            continue
                        if any(not clean_gap(text, a[2], b[1]) for a, b in zip(window, window[1:], strict=False)):
                            continue  # punctuation between tokens: not a contiguous expression
                        keys = tuple(w[0].lower() for w in window)
                        if keys in entity_keys or any(k in proper_or_acronym for k in keys):
                            continue
                        if any(k in FILLERS for k in keys):
                            continue  # a filler in the run -> disfluent expression, not a collocation
                        if has_internal_stutter(keys) or not any(k in word_pool_keys for k in keys):
                            continue
                        colloc_freq[keys] += 1
                        colloc_surf[keys][text[window[0][1] : window[-1][2]]] += 1

    pool: list[HotwordEntry] = []
    for key, count in colloc_freq.items():
        if count < args.ngram_min_count:
            continue
        denom = 1.0
        for tok in key:
            denom *= freq.get(tok, 1)
        pmi = math.log2((count * total_tokens ** (len(key) - 1)) / denom)
        if pmi < args.ngram_min_pmi:
            continue
        surface = colloc_surf[key].most_common(1)[0][0]
        idf = math.log((total_tokens + 1.0) / (count + 1.0)) + 1e-3
        pool.append(
            HotwordEntry(" ".join(key), surface, idf * args.ngram_boost, count, is_ngram=True, ngram_kind="colloc")
        )
    return pool


def count_phrase_frequencies(paths, phrase_by_len: dict[int, set], anon_patterns) -> tuple[Counter, Counter, dict]:
    """Count case-insensitive occurrences of each phrase, how many transcriptions hold it, and the
    actual surfaces it appears with (to pick a canonical spelling that matches real casing)."""
    counts: Counter[str] = Counter()
    doc_counts: Counter[str] = Counter()
    surfaces: dict[str, Counter] = defaultdict(Counter)
    for path in paths:
        for _, record in iter_jsonl(path):
            text = get_transcription(record)
            if not text or is_anonymized(text, anon_patterns):
                continue
            spans = list(iter_tokens(text))
            toks = [s.lower() for s, _, _, _ in spans]
            seen: set[str] = set()
            for size, keyset in phrase_by_len.items():
                for i in range(len(toks) - size + 1):
                    window = tuple(toks[i : i + size])
                    if window in keyset:
                        key = " ".join(window)
                        surfaces[key][text[spans[i][1] : spans[i + size - 1][2]]] += 1
                        counts[key] += 1
                        seen.add(key)
            for key in seen:
                doc_counts[key] += 1
    return counts, doc_counts, surfaces


def present_pool_keys(text: str, pool_by_key: dict[str, HotwordEntry], phrase_by_len: dict[int, set]) -> set[str]:
    """Pool hotwords (words and phrases) spoken in ``text`` (case-insensitive)."""
    toks = lower_tokens(text)
    present = set(toks) & pool_by_key.keys()
    for size, keyset in phrase_by_len.items():
        for i in range(len(toks) - size + 1):
            window = tuple(toks[i : i + size])
            if window in keyset:
                present.add(" ".join(window))
    return present


# --------------------------------------------------------------------------------------
# Sampling utilities
# --------------------------------------------------------------------------------------
class WeightedSampler:
    """Weighted sampling without replacement over a fixed pool, via rejection."""

    def __init__(self, pool: list[HotwordEntry], rng: random.Random):
        self.pool = pool
        self.rng = rng
        self.cumulative = list(accumulate(e.weight for e in pool))
        self.total = self.cumulative[-1] if self.cumulative else 0.0

    def _draw_index(self) -> int:
        r = self.rng.random() * self.total
        return bisect_right(self.cumulative, r)

    def sample(self, k: int, exclude: set[str]) -> list[HotwordEntry]:
        """Sample up to k distinct entries whose key is not in ``exclude``."""
        if k <= 0 or self.total <= 0.0:
            return []
        chosen: list[HotwordEntry] = []
        seen: set[str] = set(exclude)
        attempts = 0
        max_attempts = 40 * k + 100
        while len(chosen) < k and attempts < max_attempts:
            attempts += 1
            idx = self._draw_index()
            if idx >= len(self.pool):
                continue
            entry = self.pool[idx]
            if entry.key in seen:
                continue
            seen.add(entry.key)
            chosen.append(entry)
        return chosen


def weighted_sample_subset(entries: list[HotwordEntry], k: int, rng: random.Random) -> list[HotwordEntry]:
    """Weighted sampling without replacement from a small list (Efraimidis-Spirakis)."""
    if k <= 0:
        return []
    if k >= len(entries):
        return list(entries)
    keyed = [(rng.random() ** (1.0 / max(e.weight, 1e-9)), e) for e in entries]
    keyed.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in keyed[:k]]


def sample_list_size(rng: random.Random, lo: int, hi: int, dist: str, mean: float) -> int:
    """Sample the hotword-list size in [lo, hi].

    ``exponential`` (default) and ``geometric`` concentrate on small lists with a decaying
    tail controlled by ``mean``; ``loguniform`` and ``uniform`` ignore ``mean``.
    """
    lo = max(1, lo)
    hi = max(lo, hi)
    if lo == hi:
        return lo
    if dist == "uniform":
        return rng.randint(lo, hi)
    if dist == "loguniform":
        value = int(math.exp(rng.uniform(math.log(lo), math.log(hi + 1))))
        return min(max(value, lo), hi)
    scale = max(mean - lo, 1e-9)
    if dist == "geometric":
        p = 1.0 / (scale + 1.0)
        offset = 0 if p >= 1.0 else int(math.log(1.0 - rng.random()) / math.log(1.0 - p))
    else:  # truncated exponential on [0, hi - lo]
        z = 1.0 - math.exp(-(hi - lo) / scale)
        offset = int(round(-scale * math.log(1.0 - rng.random() * z)))
    return min(max(lo + offset, lo), hi)


def sample_count_decaying(rng: random.Random, lo: int, hi: int, decay: float) -> int:
    """Sample an integer in [lo, hi] with geometric decay toward ``lo``."""
    if hi <= lo:
        return lo
    value = lo
    while value < hi and rng.random() < decay:
        value += 1
    return value


# --------------------------------------------------------------------------------------
# Language identification (fastText) and prompt templates
# --------------------------------------------------------------------------------------
class LanguageIdentifier:
    """Thin wrapper over a fastText lid.176 model, robust to the numpy-2 predict wrapper."""

    def __init__(self, model_path: Path):
        import fasttext

        self.model = fasttext.load_model(str(model_path))

    def predict(self, text: str) -> str:
        pairs = self.predict_proba(text, 1)
        return pairs[0][0] if pairs else "unknown"

    def predict_proba(self, text: str, k: int = 5) -> list[tuple[str, float]]:
        """Top-``k`` (language, probability) pairs, most probable first."""
        text = " ".join(text.split())
        if not text:
            return []
        try:
            labels, probs = self.model.predict(text, k)
            pairs = [(lbl.replace("__label__", ""), float(p)) for lbl, p in zip(labels, probs, strict=False)]
        except Exception:
            # numpy-2 wrapper path: f.predict yields a list of (prob, "__label__xx") tuples.
            result = self.model.f.predict(text, k, 0.0, "strict")
            pairs = [(lbl.replace("__label__", ""), float(p)) for p, lbl in (result or [])]
        return pairs


def resolve_lid_model(args: argparse.Namespace) -> Path:
    """Find the fastText model from --lid-model, the environment, or known locations."""
    candidates: list[Path] = []
    if args.lid_model:
        candidates.append(Path(args.lid_model))
    env = os.environ.get("FASTTEXT_LID_MODEL")
    if env:
        candidates.append(Path(env))
    for name in ("lid.176.ftz", "lid.176.bin"):
        candidates.append(args.contexts_dir / name)
        candidates.append(Path(name))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "fastText language-ID model not found. Download lid.176.ftz (see the install comments "
        "at the top of this script) and pass --lid-model PATH or set FASTTEXT_LID_MODEL."
    )


def detect_majority_language(paths, langid: LanguageIdentifier, sample: int = 500) -> str:
    """Majority fastText-detected language over a sample of transcriptions (the audio language)."""
    counts: Counter[str] = Counter()
    for path in paths:
        for _, record in iter_jsonl(path):
            text = get_transcription(record)
            if text:
                counts[langid.predict(text)] += 1
            if sum(counts.values()) >= sample:
                break
        if sum(counts.values()) >= sample:
            break
    return counts.most_common(1)[0][0] if counts else "en"


def resolve_audio_lang(args: argparse.Namespace, langid, inputs) -> str:
    """The effective audio/transcription language: --audio-lang, or the majority detected one."""
    if args.audio_lang != "auto":
        return args.audio_lang
    if langid is None:
        try:
            langid = LanguageIdentifier(resolve_lid_model(args))
        except FileNotFoundError:
            return args.lang if args.lang != "auto" else "unknown"
    return detect_majority_language(inputs, langid)


def build_prompt_language_prior(paths, langid: LanguageIdentifier, template_langs, sample: int, alpha: float):
    """Empirical prior over the instruction (prompt) language, from top-1 fastText detections.

    fastText is unreliable on short instructions: "Transcris la parole" scores higher for Italian
    than French, yet the vast majority of prompts in a dataset are usually one language. Counting
    the top-1 label of each prompt (instance-weighted) gives a prior that later pulls such ambiguous
    prompts back to the corpus-majority language. Returns ``(prior, floor)`` where ``prior`` maps a
    language to its (Laplace-smoothed) probability and ``floor`` is the probability of a language not
    seen in the sample. Smoothing spans the observed languages and the supported template languages,
    so no supported language gets a zero prior.
    """
    counts: Counter[str] = Counter()
    seen = 0
    for path in paths:
        for _, record in iter_jsonl(path):
            idx = get_instruction_index(record)
            if idx is not None:
                instruction = record["conversations"][idx].get("value")
                if isinstance(instruction, str) and instruction.strip():
                    counts[langid.predict(instruction)] += 1
                    seen += 1
            if seen >= sample:
                break
        if seen >= sample:
            break
    langs = set(counts) | set(template_langs)
    if not langs:
        return {}, 1.0
    total = sum(counts.values()) + alpha * len(langs)
    prior = {lang: (counts.get(lang, 0) + alpha) / total for lang in langs}
    floor = alpha / total
    return prior, floor


def build_prompt_language_map(contexts_dir: Path) -> dict[str, str]:
    """Map each known ASR-context instruction to its language, from data/contexts/<LAN>_asr_contexts.json.

    Each file groups instructions by the probability that they belong to that language; the >= 0.5
    bucket (the "0.95" lists) holds the instructions authored *in* that language. Matching a prompt
    against this map gives an exact, reliable language, avoiding fastText misreads on short prompts.
    """
    mapping: dict[str, str] = {}
    for path in sorted(contexts_dir.glob("*_asr_contexts.json")):
        lang = path.name.split("_", 1)[0]
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for category in data.values():
            if not isinstance(category, dict):
                continue
            for prob, instructions in category.items():
                try:
                    if float(prob) < 0.5:
                        continue
                except (TypeError, ValueError):
                    continue
                for instruction in instructions:
                    mapping.setdefault(" ".join(str(instruction).split()), lang)
    return mapping


def load_stopwords(contexts_dir: Path, lang: str) -> set[str]:
    """Lower-cased stopwords (function words) for ``lang`` from stopwords.json; empty if absent."""
    path = contexts_dir / "stopwords.json"
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    words = data.get(lang)
    return {w.lower() for w in words} if isinstance(words, list) else set()


def make_common_word_predicate(lang: str, zipf_threshold: float):
    """Return ``is_common(word) -> bool`` backed by wordfreq (word frequency across the whole
    language). A word is common when its Zipf frequency (1=rare .. 7=extremely frequent) is at least
    ``zipf_threshold``; this catches ordinary content words ("vrai", "façon") that a small corpus
    wrongly treats as rare and that a stopword list misses. wordfreq is required here (see the
    install note at the top of the file); install it or pass --no-exclude-common-words."""
    try:
        from wordfreq import zipf_frequency
    except ImportError as exc:
        raise RuntimeError(
            "--exclude-common-words needs the 'wordfreq' package (pip install wordfreq), or pass "
            "--no-exclude-common-words to disable common-word filtering. See the top-of-file install notes."
        ) from exc

    def is_common(word: str) -> bool:
        return zipf_frequency(word, lang) >= zipf_threshold

    return is_common


def load_anonymization_patterns(contexts_dir: Path, extra: list[str]) -> list[re.Pattern]:
    """Compile the default safe anonymisation regexes (anonymization_patterns.json) plus any given
    with --anonymization-pattern. These mark a transcription as anonymised (to be discarded)."""
    patterns: list[str] = []
    path = contexts_dir / "anonymization_patterns.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data.get("patterns"), list):
            patterns.extend(data["patterns"])
    patterns.extend(extra or [])
    return [re.compile(p, re.UNICODE) for p in patterns]


class PromptLanguageResolver:
    """Detect a prompt's language, biased toward the corpus-majority language via a prior.

    Combines fastText's per-language probability with an empirical corpus prior as a Bayesian
    posterior ``p(lang|text) * prior[lang] ** strength`` over the top candidates, then takes the
    argmax. A confidently-detected foreign prompt still wins (its probability dwarfs the prior gap),
    but a short, ambiguous prompt is resolved toward the majority language. With an empty prior or a
    strength of 0 this is plain top-1 detection. Results are cached per distinct instruction.
    """

    def __init__(
        self,
        langid: LanguageIdentifier,
        prior: dict,
        floor: float,
        strength: float,
        topk: int = 7,
        template_lang: dict[str, str] | None = None,
    ):
        self.langid = langid
        self.prior = prior or {}
        self.floor = floor
        self.strength = strength
        self.topk = topk
        self.template_lang = template_lang or {}  # exact instruction -> language (authoritative)
        self._cache: dict[str, str] = {}
        self.raw_counts: Counter[str] = Counter()
        self.corrections: Counter[tuple[str, str]] = Counter()  # (raw top-1, chosen) -> instances
        self.n_template_hits = 0

    def detect(self, text: str) -> str:
        key = " ".join(text.split())
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        # Authoritative: the instruction is a known ASR-context template of a known language.
        templated = self.template_lang.get(key)
        if templated is not None:
            self.n_template_hits += 1
            self._cache[key] = templated
            return templated
        pairs = self.langid.predict_proba(key, self.topk)
        if not pairs:
            self._cache[key] = "unknown"
            return "unknown"
        raw = pairs[0][0]
        self.raw_counts[raw] += 1
        if not self.prior or self.strength <= 0:
            chosen = raw
        else:
            chosen = max(pairs, key=lambda lp: lp[1] * (self.prior.get(lp[0], self.floor) ** self.strength))[0]
            if chosen != raw:
                self.corrections[(raw, chosen)] += 1
        self._cache[key] = chosen
        return chosen


def load_templates(contexts_dir: Path) -> dict[str, list[str]]:
    """Load biasing_contexts.json as a {language: [label, ...]} dict."""
    path = contexts_dir / "biasing_contexts.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing hotword-label file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    templates = {k: v for k, v in data.items() if isinstance(v, list) and v}
    if "en" not in templates:
        raise ValueError(f"{path} must contain an 'en' entry (used as fallback).")
    return templates


def pick_template(lang: str, templates: dict[str, list[str]], rng: random.Random, english_prob: float) -> str:
    """Pick a label for ``lang``; sometimes use the English label for non-English prompts."""
    if lang != "en" and rng.random() < english_prob:
        use = "en"
    elif lang in templates:
        use = lang
    else:
        use = "en"
    return rng.choice(templates[use])


def format_hotword_prompt(hotwords: list[str], template: str, separator: str) -> str:
    joined = separator.join(hotwords)
    if "{hotwords}" in template:
        return template.replace("{hotwords}", joined)
    return f"{template} {joined}"


# --------------------------------------------------------------------------------------
# Per-record generation
# --------------------------------------------------------------------------------------
def present_surface(text: str, key: str, canonical: str) -> str:
    """Surface of the hotword's first occurrence in ``text`` (so the list matches the transcript).

    A sentence-initial capital is normalised toward the canonical casing (the only casing
    difference we allow: a normally lower-case word capitalised because it follows a dot).
    """
    seq = key.split(" ")
    n = len(seq)
    spans = list(iter_tokens(text))
    low = [s.lower() for s, _, _, _ in spans]
    for i in range(len(low) - n + 1):
        if low[i : i + n] == seq:
            surface = text[spans[i][1] : spans[i + n - 1][2]]
            # Normalise a sentence-initial capital only when the word is otherwise a normal
            # Capitalised word; never touch an all-caps or mixed-case form ("TOP" must stay "TOP").
            first = surface.split(" ", 1)[0]
            if spans[i][3] and canonical[:1].islower() and first[:1].isupper() and first[1:] == first[1:].lower():
                surface = surface[:1].lower() + surface[1:]
            return surface
    return canonical


def _choose_spelling(entry: HotwordEntry, spellings, rng: random.Random, respell_prob: float) -> str:
    """The spelling to list: canonical, or a random alternative spelling."""
    alts = spellings.get(entry.key)
    if respell_prob > 0 and alts and len(alts) > 1 and rng.random() < respell_prob:
        return rng.choice(alts[1:])
    return alts[0] if alts else entry.surface


def build_list_variant(
    present_keys: set[str],
    pool_size: int,
    pool_by_key: dict[str, HotwordEntry],
    sampler: WeightedSampler,
    rng: random.Random,
    args: argparse.Namespace,
    include_positive: bool,
    text: str,
    spellings,
):
    """Return (shuffled surfaces, positives, distractors, transcription), or None.

    Listed hotwords may be respelled (--respell-prob). When a *spoken* hotword is respelled, the
    returned transcription is rewritten to the same spelling (reference perturbation). When
    ``include_positive`` is False the list holds only distractors (a no-hotword variant).
    """
    list_size = sample_list_size(rng, args.min_hotwords, args.max_hotwords, args.size_dist, args.size_mean)
    list_size = min(list_size, pool_size)
    if list_size <= 0:
        return None

    available = [pool_by_key[k] for k in present_keys]
    if include_positive and available:
        cap = min(len(available), args.max_positives, list_size)
        n_positive = sample_count_decaying(rng, 1, cap, args.positive_decay)
    else:
        n_positive = 0

    positives = weighted_sample_subset(available, n_positive, rng)
    positive_keys = {e.key for e in positives}
    n_distractor = list_size - len(positives)
    distractors = sampler.sample(n_distractor, exclude=present_keys | positive_keys)

    surfaces: list[str] = []
    listed_present: dict[str, list[str]] = defaultdict(list)  # present key -> spelling(s) shown in the list
    used_spellings: list[tuple[str, str, bool]] = []  # (key, spelling, is_present)
    for entry in positives:
        base = present_surface(text, entry.key, entry.surface)  # match the transcript casing
        spelling = _choose_spelling(entry, spellings, rng, args.respell_prob)
        canonical = spellings.get(entry.key, [entry.surface])[0]
        if spelling == canonical:
            listed = base  # list it exactly as spoken in the transcript
        elif spelling.lower() == canonical.lower():
            listed = spelling  # casing variant (same letters): list that casing verbatim
        else:
            listed = match_leading_case(base, spelling)  # orthographic respelling: sync leading case
        surfaces.append(listed)
        listed_present[entry.key].append(listed)
        used_spellings.append((entry.key, spelling, True))
    for entry in distractors:
        spelling = _choose_spelling(entry, spellings, rng, args.respell_prob)
        surfaces.append(spelling)
        used_spellings.append((entry.key, spelling, False))
    if not surfaces:
        return None
    rng.shuffle(surfaces)
    # Reference perturbation: rewrite each spoken hotword to the spelling it is listed with. When a
    # word is listed in several casings (e.g. "Union" and "UNION"), the transcript keeps the one
    # closest to how it was really spoken (apply_respellings picks per occurrence).
    for s in surfaces:
        k = " ".join(m.group(0).lower() for m in WORD_RE.finditer(s))
        if k in listed_present and s not in listed_present[k]:
            listed_present[k].append(s)
    new_text = apply_respellings(text, listed_present) if listed_present else text
    return surfaces, positives, distractors, new_text, used_spellings


def process_file(
    path: Path,
    out_path: Path,
    pool: list[HotwordEntry],
    pool_by_key: dict[str, HotwordEntry],
    phrase_by_len: dict[int, set],
    templates: dict[str, list[str]],
    prompt_lang: PromptLanguageResolver | None,
    rng: random.Random,
    args: argparse.Namespace,
    force_no_positive: float,
    spellings: dict,
) -> _Stats:
    """Write the contextual-biasing manifest for a single input file."""
    sampler = WeightedSampler(pool, rng)
    stats = _Stats()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out:
        for _, record in iter_jsonl(path):
            text = get_transcription(record)
            instr_idx = get_instruction_index(record)
            if text is None or instr_idx is None:
                stats.n_skipped += 1
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue
            if is_anonymized(text, args.anon_patterns):
                stats.n_anonymized += 1  # drop anonymised examples entirely (not written out)
                continue

            present_keys = present_pool_keys(text, pool_by_key, phrase_by_len)
            instruction = record["conversations"][instr_idx]["value"]
            lang = prompt_lang.detect(instruction) if prompt_lang is not None else args.lang
            if len(stats.lang_prompts[lang]) < 2000:
                stats.lang_prompts[lang].add(" ".join(instruction.split()))

            # In paired mode every utterance with a spoken candidate is emitted twice: once
            # with the candidate in the list, once with distractors only. Otherwise a single
            # line is emitted (positive kept with probability 1 - force_no_positive).
            if present_keys and args.paired:
                include_flags = [True, False]
            elif present_keys:
                include_flags = [rng.random() >= force_no_positive]
            else:
                include_flags = [False]

            for variant_index, include in enumerate(include_flags):
                result = build_list_variant(
                    present_keys, len(pool), pool_by_key, sampler, rng, args, include, text, spellings
                )
                if result is None:
                    continue
                surfaces, positives, distractors, new_text, used_spellings = result
                rec = copy.deepcopy(record)
                if variant_index > 0 and rec.get("id") is not None:
                    rec["id"] = f"{rec['id']}{args.pair_suffix}"
                if new_text != text:
                    set_transcription(rec, new_text)
                template = pick_template(lang, templates, rng, args.english_keyword_prob)
                prompt = format_hotword_prompt(surfaces, template, args.hotword_separator)
                rec["conversations"][instr_idx]["value"] = f"{instruction.rstrip()}{args.instruction_separator}{prompt}"

                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats.n_augmented += 1
                stats.list_sizes.append(len(surfaces))
                stats.lang_counts[lang] += 1
                for key, spelling, is_present in used_spellings:
                    if is_present:
                        stats.used_present[key] += 1
                        stats.present_spell[(key, spelling)] += 1
                    else:
                        stats.used_distractor[key] += 1
                        stats.absent_spell[(key, spelling)] += 1
                if not positives:
                    stats.n_hard_negative += 1
                if variant_index > 0:
                    stats.n_twins += 1
    return stats


def spelling_label(stats, key: str, spelling: str) -> str:
    """A spelling followed by its (times listed as present, times listed as absent) counts."""
    return f"{spelling} ({stats.present_spell.get((key, spelling), 0)}, {stats.absent_spell.get((key, spelling), 0)})"


def write_readme(
    readme_path, in_path, out_path, args, command, derived, stats, pool_sorted, doc_freq, spellings
) -> None:
    """Write a human-readable report next to the generated manifest."""
    lines: list[str] = []
    lines.append("Contextual-biasing manifest - generation report")
    lines.append("=" * 48)
    lines.append(f"Generated : {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Output    : {out_path.name}")
    lines.append(f"Input     : {in_path}")
    lines.append(f"Audio lang: {derived['audio_lang']}" + ("  (auto-detected)" if derived["audio_lang_auto"] else ""))
    lines.append(f"Common-word filter: {derived['common_word_filter']}")
    lines.append("")
    lines.append("Command line:")
    lines.append(f"  {command}")
    lines.append("")
    lines.append("Parameters (explicit values and defaults):")
    for key, value in sorted(vars(args).items()):
        if isinstance(value, list):
            value = [str(v) for v in value]
        elif isinstance(value, Path):
            value = str(value)
        lines.append(f"  {key} = {value!r}")
    lines.append("")
    lines.append("Derived (global, across all inputs):")
    lines.append(f"  rarity_cutoff_rank (most frequent words excluded from 'rare') = {derived['cutoff']}")
    lines.append(f"  hotword_pool_size = {derived['pool_size']}")
    lines.append(
        f"  single_words = {derived['n_words']} (proper_nouns {derived['n_proper']}, "
        f"acronyms {derived['n_acronym']}, both {derived['n_both']}, other_rare {derived['n_other']})"
    )
    lines.append(
        f"  phrases = {derived['n_phrase']} (entities {derived['n_entity']}, collocations {derived['n_colloc']})"
    )
    lines.append(f"  natural_no_hotword_rate = {derived['natural']:.4f}")
    lines.append(f"  forcing_probability = {derived['force']:.4f}")
    lines.append("")
    lines.append("This file:")
    lines.append(
        f"  output_lines = {stats.n_augmented} (incl. {stats.n_twins} no-hotword twins), "
        f"unchanged_lines = {stats.n_skipped}, anonymized_dropped = {stats.n_anonymized}"
    )
    lines.append(f"  effective_no_hotword_ratio = {stats.hard_negative_ratio:.4f} (target {args.target_no_positive})")
    if args.paired:
        lines.append("    (paired mode: target is the candidate-less utterance rate; each candidate utterance")
        lines.append("     also gets a no-hotword twin, so the output ratio is higher by design)")
    lines.append(f"  mean_list_size = {stats.mean_list_size:.2f}")
    lines.append("")
    lines.append("Prompt languages (auto = fastText detection, ratio over augmented lines):")
    total = sum(stats.lang_counts.values()) or 1
    for lang, count in stats.lang_counts.most_common():
        lines.append(f"  {lang:<8} {count / total:.4f}  ({count})")
    lines.append("")
    lines.append(f"Hotwords ({len(pool_sorted)} total), sorted by decreasing number of occurrences.")
    lines.append(
        "Columns: freq=transcriptions containing it; tag; occurrences=total times spoken in all "
        "transcriptions; target=times listed as a spoken hotword; distractor=times listed as an absent one; "
        "spellings=real spelling first, then generated alternatives. Each spelling is followed by "
        "(times listed as present in the transcript, times listed as absent)."
    )
    lines.append(f"  {'freq':>7}  {'tag':<6}  {'occurrences':>11}  {'target':>7}  {'distractor':>10}  spellings")
    for entry in pool_sorted:
        if entry.is_ngram:
            tag = "NE" if entry.ngram_kind == "entity" else "COL"
        elif entry.is_acronym:
            tag = "ACR"
        elif entry.is_proper_noun:
            tag = "PN"
        else:
            tag = "rare"
        df = doc_freq.get(entry.key, 0)
        target = stats.used_present.get(entry.key, 0)
        distractor = stats.used_distractor.get(entry.key, 0)
        forms = spellings.get(entry.key, [entry.surface])
        labels = [spelling_label(stats, entry.key, s) for s in forms]
        shown = labels[0] if len(labels) == 1 else f"{labels[0]}  ~  {', '.join(labels[1:])}"
        lines.append(f"  {df:>7}  {tag:<6}  {entry.freq:>11}  {target:>7}  {distractor:>10}  {shown}")

    lines.append("")
    lines.append("Prompt language detection (fastText) - distinct prompts per detected language (max 50 shown):")
    for lang, _ in stats.lang_counts.most_common():
        prompts = sorted(stats.lang_prompts.get(lang, set()))
        lines.append(f"  [{lang}] {len(prompts)} distinct prompt(s):")
        for prompt in prompts[:50]:
            lines.append(f"    - {prompt}")
        if len(prompts) > 50:
            lines.append("    ...")
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    default_contexts = Path(__file__).resolve().parent.parent / "contexts"
    parser = argparse.ArgumentParser(
        description="Append contextual-biasing hotword prompts to ASR manifests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Input jsonl manifest(s).")
    parser.add_argument(
        "--contexts-dir", type=Path, default=default_contexts, help="Directory holding biasing_contexts.json."
    )
    parser.add_argument(
        "--lang",
        default="auto",
        help="Prompt language: 'auto' detects it from the instruction with fastText, or give a "
        "language code (e.g. fr, en) present in biasing_contexts.json to use it for every line.",
    )
    parser.add_argument(
        "--lid-model",
        default=None,
        help="Path to the fastText lid.176 model (.ftz/.bin), used only when --lang auto. See top-of-file comments.",
    )
    parser.add_argument(
        "--lang-prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --lang auto, bias per-prompt detection toward the corpus-majority language "
        "(fastText misreads short prompts like 'Transcris la parole' as Italian). Use --no-lang-prior to disable.",
    )
    parser.add_argument(
        "--lang-prior-strength",
        type=float,
        default=1.0,
        help="Exponent on the corpus prior in the posterior p(lang|text) * prior[lang]**strength "
        "(0 disables; higher trusts the majority language more).",
    )
    parser.add_argument(
        "--lang-prior-sample",
        type=int,
        default=20000,
        help="How many instructions to sample when estimating the prompt-language prior.",
    )
    parser.add_argument(
        "--lang-prior-alpha",
        type=float,
        default=1.0,
        help="Laplace smoothing count for the prompt-language prior (keeps rare languages non-zero).",
    )
    parser.add_argument(
        "--english-keyword-prob",
        type=float,
        default=0.1,
        help="Probability of using the English label even when the instruction is in another language.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Where to write outputs (default: alongside each input)."
    )
    parser.add_argument(
        "--output-suffix", default="_contextual_biasing", help="Suffix inserted before the .jsonl extension."
    )

    # List composition.
    parser.add_argument("--min-hotwords", type=int, default=1, help="Minimum hotword-list size.")
    parser.add_argument("--max-hotwords", type=int, default=5000, help="Maximum hotword-list size.")
    parser.add_argument(
        "--size-dist",
        choices=["exponential", "geometric", "loguniform", "uniform"],
        default="exponential",
        help="Distribution of the hotword-list size (exponential/geometric favour small lists).",
    )
    parser.add_argument(
        "--size-mean", type=float, default=24.0, help="Mean hotword-list size for exponential/geometric."
    )
    parser.add_argument(
        "--target-no-positive",
        type=float,
        default=0.5,
        help="Target (approximate) fraction of utterances with no spoken candidate hotword. The rarity "
        "cutoff is inferred to approach it; without --paired a forcing probability tops it up.",
    )
    parser.add_argument(
        "--paired",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit each utterance that has a spoken candidate twice: once with the candidate in the "
        "list and once with distractors only, so the model is not forced to rely on the hotwords.",
    )
    parser.add_argument(
        "--pair-suffix", default="-nobias", help="Suffix appended to the id of the no-hotword twin in paired mode."
    )
    parser.add_argument(
        "--max-positives", type=int, default=5, help="Maximum number of spoken hotwords listed per utterance."
    )
    parser.add_argument(
        "--positive-decay", type=float, default=0.45, help="Geometric decay for the number of spoken hotwords."
    )

    # Pool construction (single words).
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Minimum corpus count for a word to be eligible (2 drops one-off typos/mis-transcriptions; "
        "use 1 for clean read speech where a name said once is a real target).",
    )
    parser.add_argument("--min-word-len", type=int, default=2, help="Minimum word length.")
    parser.add_argument(
        "--max-docfreq-ratio",
        type=float,
        default=0.05,
        help="Exclude a single word from the pool when it occurs in more than this fraction of "
        "transcriptions (ubiquitous words like 'Monsieur'/'Commission' are not useful bias terms).",
    )
    parser.add_argument(
        "--exclude-common-words",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep ordinary common words and stopwords of the audio language out of the pool (unless "
        "they are proper nouns or acronyms). Uses stopwords.json; richer filtering needs wordfreq.",
    )
    parser.add_argument(
        "--common-word-zipf",
        type=float,
        default=4.5,
        help="A non-proper word at least this frequent (wordfreq Zipf, 1=rare..7=very frequent) in the "
        "audio language is treated as common and excluded. Proper nouns and acronyms are exempt.",
    )
    parser.add_argument(
        "--discard-anonymized",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop examples whose transcription carries an anonymisation marker (default patterns in "
        "anonymization_patterns.json, e.g. PFC 'Npers', so placeholders never become hotwords).",
    )
    parser.add_argument(
        "--anonymization-pattern",
        action="append",
        default=[],
        metavar="REGEX",
        help="Extra regex marking a transcription as anonymised (repeatable), added to the defaults.",
    )
    parser.add_argument(
        "--filter-disfluencies",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop likely disfluency fragments (a token that is mostly a strict prefix of the next token).",
    )
    parser.add_argument(
        "--disfluency-ratio",
        type=float,
        default=0.5,
        help="A word is a disfluency when this share of its occurrences precede a longer token starting with it.",
    )
    parser.add_argument(
        "--proper-ratio",
        type=float,
        default=0.5,
        help="Min share of capitalised non-sentence-start occurrences for a proper noun.",
    )
    parser.add_argument(
        "--proper-noun-boost", type=float, default=2.0, help="Sampling-weight multiplier for proper nouns."
    )
    parser.add_argument("--acronym-boost", type=float, default=3.0, help="Sampling-weight multiplier for acronyms.")

    # Multi-word expressions.
    parser.add_argument(
        "--ngrams",
        choices=["none", "entity", "all"],
        default="all",
        help="Multi-word hotwords: none, entity phrases only, or entity phrases + collocations.",
    )
    parser.add_argument("--ngram-max-n", type=int, default=4, help="Maximum entity-phrase length in tokens.")
    parser.add_argument("--ngram-min-count", type=int, default=3, help="Minimum corpus count for a collocation.")
    parser.add_argument("--ngram-min-pmi", type=float, default=7.0, help="Minimum PMI (bits) for a collocation.")
    parser.add_argument("--ngram-boost", type=float, default=2.5, help="Sampling-weight multiplier for phrases.")

    # Pronunciation-preserving respelling of hotwords (Huang et al., Interspeech 2024).
    parser.add_argument(
        "--respell-prob",
        type=float,
        default=0.6,
        help="Probability a listed hotword (that has alternatives) is written with a similar-sounding "
        "alternative spelling "
        "(0 disables respelling). When a spoken hotword is respelled, the reference transcription is "
        "rewritten to the same spelling (so the model relies on the context).",
    )
    parser.add_argument(
        "--respell-method",
        choices=["g2p", "rules"],
        default="g2p",
        help="g2p: data-driven phoneme model (Epitran); rules: hand-written letter substitutions. "
        "g2p falls back to rules for languages Epitran cannot handle here (e.g. English).",
    )
    parser.add_argument(
        "--audio-lang",
        default="auto",
        help="Language of the audio/transcription (selects respelling rules); 'auto' detects the "
        "majority transcription language with fastText. Distinct from the prompt language --lang.",
    )
    parser.add_argument(
        "--respell-max-variants", type=int, default=8, help="Max alternative spellings generated per hotword."
    )

    # Formatting.
    parser.add_argument("--hotword-separator", default=", ", help="Separator between hotwords.")
    parser.add_argument(
        "--instruction-separator", default="\n\n", help="Separator between the instruction and the hotword prompt."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    command = "python " + " ".join(sys.argv)

    inputs = list(args.inputs)
    for p in inputs:
        if not p.exists():
            raise FileNotFoundError(f"Input manifest not found: {p}")

    templates = load_templates(args.contexts_dir)
    # Compiled anonymisation markers (used from the first scan on, so markers never enter the pool).
    args.anon_patterns = (
        load_anonymization_patterns(args.contexts_dir, args.anonymization_pattern) if args.discard_anonymized else []
    )
    if args.lang == "auto":
        langid = LanguageIdentifier(resolve_lid_model(args))
    else:
        if args.lang not in templates:
            raise ValueError(f"--lang '{args.lang}' has no labels in biasing_contexts.json ({sorted(templates)}).")
        langid = None

    # Per-prompt language detection: an exact match against the known ASR-context templates is
    # authoritative; otherwise fastText biased toward the corpus-majority language (prior).
    prompt_lang: PromptLanguageResolver | None = None
    template_lang = build_prompt_language_map(args.contexts_dir)
    if langid is not None:
        if args.lang_prior:
            prior, floor = build_prompt_language_prior(
                inputs, langid, templates.keys(), args.lang_prior_sample, args.lang_prior_alpha
            )
        else:
            prior, floor = {}, 1.0
        prompt_lang = PromptLanguageResolver(
            langid, prior, floor, args.lang_prior_strength, template_lang=template_lang
        )

    # Pass 1: single-word statistics and named-entity phrases.
    print(f"Scanning {len(inputs)} manifest(s)...")
    scanned = scan_corpus(inputs, args)
    (
        freq,
        n_nonstart,
        n_nonstart_cap,
        n_acronym,
        surfaces,
        nonstart_surfaces,
        doc_freq,
        repair,
        entity_freq,
        entity_surf,
    ) = scanned
    if not freq:
        raise RuntimeError("No eligible words found in the inputs.")
    ranked = [k for k, _ in freq.most_common()]
    rank_of = {k: i for i, k in enumerate(ranked)}
    total_tokens = sum(freq.values())
    # Effective audio/transcription language (used for German handling, respelling, and the README).
    audio_lang = resolve_audio_lang(args, langid, inputs)
    # Common-word filtering resources for the audio language (items: keep common words/stopwords out).
    stopwords = load_stopwords(args.contexts_dir, audio_lang) if args.exclude_common_words else set()
    is_common = make_common_word_predicate(audio_lang, args.common_word_zipf) if args.exclude_common_words else None
    classification, proper_or_acronym = classify_keys(
        freq, n_nonstart, n_nonstart_cap, n_acronym, surfaces, audio_lang, args
    )

    # Drop likely disfluency fragments: tokens that are mostly a strict prefix of the next
    # token ("ain" before "ainsi", a stutter), kept out of the pool and of coverage.
    disfluency_keys: set[str] = set()
    if args.filter_disfluencies:
        disfluency_keys = {k for k, c in freq.items() if repair[k] / c >= args.disfluency_ratio}
        disfluency_keys |= prefix_disfluency_keys(freq)
        disfluency_keys |= {k for k in freq if is_hyphen_stutter(k)}  # "au-au", "nord-amé-nord-..."
    # Fillers and glued annotation markers are never good hotwords, independent of --filter-disfluencies.
    disfluency_keys |= {k for k in freq if k in FILLERS}
    disfluency_keys |= {k for k in freq if is_annotation_artifact(choose_surface(nonstart_surfaces[k], surfaces[k]))}
    proper_or_acronym -= disfluency_keys
    eligible_keys = {k for k, c in freq.items() if c >= args.min_count and k not in disfluency_keys}

    entity_by_len: dict[int, set] = defaultdict(set)
    for key in entity_freq:
        entity_by_len[len(key)].add(key)

    # Pass 2: infer the rarity cutoff that approaches the target no-hotword rate.
    n_total, pnless_rmax = collect_empty_stats(
        inputs, proper_or_acronym, rank_of, eligible_keys, entity_by_len, args.anon_patterns
    )
    target = args.target_no_positive
    cutoff, natural_rate = choose_cutoff(pnless_rmax, n_total, target)
    force_no_positive = 0.0 if natural_rate >= target else (target - natural_rate) / (1.0 - natural_rate)
    force_no_positive = min(max(force_no_positive, 0.0), 1.0)

    # Build the pool: single words (+ entity phrases + collocations).
    word_pool = build_word_pool(
        freq,
        surfaces,
        nonstart_surfaces,
        classification,
        rank_of,
        cutoff,
        total_tokens,
        disfluency_keys,
        args,
        doc_freq=doc_freq,
        n_docs=n_total,
        stopwords=stopwords,
        is_common=is_common,
    )
    word_pool_keys = {e.key for e in word_pool}
    entity_pool = build_entity_pool(entity_freq, entity_surf, total_tokens, args) if args.ngrams != "none" else []
    colloc_pool = (
        extract_collocations(inputs, freq, word_pool_keys, proper_or_acronym, set(entity_freq), total_tokens, args)
        if args.ngrams == "all"
        else []
    )
    pool = word_pool + entity_pool + colloc_pool
    if not pool:
        raise RuntimeError("Empty hotword pool.")
    pool_by_key = {e.key: e for e in pool}
    phrase_by_len: dict[int, set] = defaultdict(set)
    for entry in entity_pool + colloc_pool:
        tokens = tuple(entry.key.split(" "))
        phrase_by_len[len(tokens)].add(tokens)
    phrase_doc_freq: Counter = Counter()
    if phrase_by_len:
        phrase_freq, phrase_doc_freq, phrase_surfaces = count_phrase_frequencies(
            inputs, phrase_by_len, args.anon_patterns
        )
        for entry in entity_pool + colloc_pool:
            entry.freq = phrase_freq.get(entry.key, entry.freq)
            if entry.key in phrase_surfaces:  # canonical spelling = most common actual casing
                entry.surface = phrase_surfaces[entry.key].most_common(1)[0][0]
            idf = math.log((total_tokens + 1.0) / (entry.freq + 1.0)) + 1e-3
            entry.weight = idf * args.ngram_boost
    # Document frequency (number of transcriptions containing it) for the README.
    doc_freq_display = {e.key: doc_freq.get(e.key, 0) for e in word_pool}
    doc_freq_display.update({e.key: phrase_doc_freq.get(e.key, 0) for e in entity_pool + colloc_pool})
    pool_sorted = sorted(pool, key=lambda e: (-e.freq, e.surface))

    # Pre-generate alternative spellings for every hotword (canonical first).
    spellings = {e.key: [e.surface] for e in pool}
    if args.respell_prob > 0:
        corpus_words = [e.surface for e in pool]
        corpus_words += ranked[: max(0, 3000 - len(corpus_words))]
        engine = RespellEngine(
            args.respell_method, audio_lang, rng, corpus_words, args.contexts_dir, stopwords=stopwords
        )
        for entry in pool:
            variants = engine.variants(entry.surface, args.respell_max_variants)
            if variants:
                spellings[entry.key] = [entry.surface, *variants]
        n_with_variants = sum(1 for v in spellings.values() if len(v) > 1)
        print(
            f"  respelling: method {engine.method} (requested {args.respell_method}), audio_lang {audio_lang}, "
            f"prob {args.respell_prob}; {n_with_variants}/{len(pool)} hotwords got variants"
        )

    n_proper = sum(e.is_proper_noun for e in word_pool)
    n_acr = sum(e.is_acronym for e in word_pool)
    n_both = sum(1 for e in word_pool if e.is_proper_noun and e.is_acronym)
    n_other = sum(1 for e in word_pool if not e.is_proper_noun and not e.is_acronym)
    derived = {
        "cutoff": cutoff,
        "pool_size": len(pool),
        "n_words": len(word_pool),
        "n_proper": n_proper,
        "n_acronym": n_acr,
        "n_both": n_both,  # counted in both n_proper and n_acronym (a capitalised acronym)
        "n_other": n_other,
        "n_phrase": len(entity_pool) + len(colloc_pool),
        "n_entity": len(entity_pool),
        "n_colloc": len(colloc_pool),
        "natural": natural_rate,
        "force": force_no_positive,
        "audio_lang": audio_lang,
        "audio_lang_auto": args.audio_lang == "auto",
        "lang_prior": prompt_lang.prior if prompt_lang is not None else {},
        "common_word_filter": "wordfreq" if is_common is not None else ("stopwords" if stopwords else "off"),
    }

    print(
        f"  pool: {len(pool)} ({len(word_pool)} words [{n_proper} PN, {n_acr} ACR], "
        f"{len(entity_pool)} entity, {len(colloc_pool)} colloc); {len(disfluency_keys)} disfluencies dropped"
    )
    mode = "paired (candidate utterances emitted with + without their hotword)" if args.paired else "single-line"
    print(
        f"  rarity cutoff: top {cutoff} words excluded | natural no-hotword rate {natural_rate:.1%} "
        f"| target {target:.1%} | mode: {mode}"
    )

    total_augmented = 0
    total_hard_negative = 0
    for path in inputs:
        base_dir = args.output_dir if args.output_dir is not None else path.parent
        out_path = base_dir / f"{path.stem}{args.output_suffix}.jsonl"
        readme_path = out_path.with_name(f"{out_path.stem}_README.txt")

        stats = process_file(
            path,
            out_path,
            pool,
            pool_by_key,
            phrase_by_len,
            templates,
            prompt_lang,
            rng,
            args,
            force_no_positive,
            spellings,
        )
        write_readme(
            readme_path, path, out_path, args, command, derived, stats, pool_sorted, doc_freq_display, spellings
        )
        total_augmented += stats.n_augmented
        total_hard_negative += stats.n_hard_negative
        anon = f", {stats.n_anonymized} anonymized dropped" if stats.n_anonymized else ""
        print(
            f"{path.name} -> {out_path.name} (+ {readme_path.name}): "
            f"{stats.n_augmented} output lines ({stats.n_twins} twins), {stats.n_skipped} unchanged{anon} | "
            f"no-hotword {stats.n_hard_negative} ({stats.hard_negative_ratio:.1%}), "
            f"mean list size {stats.mean_list_size:.1f}"
        )

    overall = total_hard_negative / total_augmented if total_augmented else 0.0
    print(f"Effective no-hotword ratio: {total_hard_negative}/{total_augmented} ({overall:.1%}), target {target:.1%}")
    if prompt_lang is not None:
        print(
            f"Prompt language: {prompt_lang.n_template_hits} exact template matches"
            + (" (fastText for the rest)" if prompt_lang.n_template_hits else "")
        )
    if prompt_lang is not None and prompt_lang.corrections:
        n_corr = sum(prompt_lang.corrections.values())
        detail = ", ".join(f"{raw}->{chosen} x{c}" for (raw, chosen), c in prompt_lang.corrections.most_common(8))
        print(f"Prompt-language prior reassigned {n_corr} prompt(s): {detail}")


if __name__ == "__main__":
    main()
