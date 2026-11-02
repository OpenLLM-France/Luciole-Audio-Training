"""Clean the text of ASR manifests with a list of filters configured in YAML.

Each dataset of the config (conf/fr.yaml, its header says how to write one) has:
  input       manifest(s), globs relative to `root`
  output      the directory of the cleaned files, which keep the input file names
  outputs     the files written per input: base (event tags and hesitations removed),
              events, disfluencies, events_disfluencies (kept, in the subdirectory of that name)
  nocasepunc  which lines get a prompt asking for no case nor punctuation
  lowercase   true: the lines with punctuation and no uppercase get a prompt asking for lowercase with punctuation
  output_lines  the lines of each output but base, drawn at random (default: all)
  filters     filters and blocks (named lists of filters, `blocks` of the config), in order
A filter rewrites the text of the Assistant turn or drops the line; lines left with no letter
nor digit (only "euh", "[laughter]" in an output without tags, a [silence] line) are dropped, but
for `empty_fraction` of them (config), kept with an empty text: nothing to transcribe. Then the prompt (first User text turn) is chosen from the final text
(ChoosePrompt): one asking for the markers the output keeps, or for none of them. `prompts`
(config, or --prompts) says how: fixed (drawn from the lists of `contexts`) or compositional
(composed by data/contexts/compose_asr_prompts.py, as the config's `compositional` says).

The words and spellings the filters use, per language, are in conf/rules.yaml, not in the code.

Non-speech events are first mapped to a closed set of bracketed English tags
(`[laughter]`, `[noise]`, ...) by `map_tags`; the `events` filter removes them or keeps them
(the output sets its mode). Hesitations (euh, hum...) work the same way with `disfluencies`.

    context/CFPP2000/train_recasepunc.jsonl -> clean/CFPP2000/train_recasepunc.jsonl
                                                        clean/CFPP2000/events/train_recasepunc.jsonl

Usage:
    python data/processing/clean_asr/clean_asr.py data/processing/clean_asr/conf/fr.yaml --dry-run --limit 100000
    python data/processing/clean_asr/clean_asr.py data/processing/clean_asr/conf/fr.yaml --datasets CFPP2000 Multilingual_TEDx
"""
import argparse
import collections
import difflib
import functools
import glob
import hashlib
import itertools
import html
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import yaml
from tqdm import tqdm

from fix_spurious_caps import fix_text as fix_spurious_caps_text
from rules import as_words, rules as language_rules, words


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CONF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf")
TAG_RE = re.compile(r"\[[^\[\]]+\]")
# a word with a [tag], "[inaudible].", "l'[inaudible]": kept out of disfluency_model's input
TAG_WORD_RE = re.compile(r"\S*\[[^\[\]]+\]\S*")
TRAILING_TAGS_RE = re.compile(r"(?:\s*\[[^\[\]]+\])+\s*$")
LETTER_RE = re.compile(r"[^\W\d_]")
WORD_CHAR_RE = re.compile(r"[^\W_]")      # a letter or a digit: digits are kept as written
PUNCT_RE = re.compile(r"[.,;:!?…]")
CHUNK_SIZE = 1024          # lines per batch of the filters with a `prepare` (models)


class Sample:
    """One manifest line: the Assistant text, the prompt turn and the record."""

    def __init__(self, rec, rng, nocasepunc=False, lowercase=False):
        self.rec = rec
        self.rng = rng
        self.nocasepunc = nocasepunc      # the dataset's `nocasepunc` (see ChoosePrompt)
        self.lowercase = lowercase        # the dataset's `lowercase` (see ChoosePrompt)
        turns = rec.get("conversations")
        if turns is None:
            self._text_holder, self._prompt_turn = rec, None
            self._text_key = "text"
        else:
            self._text_holder = next(
                (t for t in reversed(turns) if t.get("from") == "Assistant" and t.get("type") == "text"), None)
            self._prompt_turn = next(
                (t for t in turns if t.get("from") == "User" and t.get("type") == "text"), None)
            if self._prompt_turn is not None and turns[0] is not self._prompt_turn:
                raise ValueError(f"{rec.get('id')}: the prompt must come before the audio (no randomorder input)")
            self._text_key = "value"

    @property
    def text(self):
        return self._text_holder.get(self._text_key, "") if self._text_holder else ""

    @text.setter
    def text(self, value):
        self._text_holder[self._text_key] = value

    @property
    def prompt(self):
        return self._prompt_turn["value"] if self._prompt_turn else None

    @prompt.setter
    def prompt(self, value):
        if self._prompt_turn is None and self.rec.get("conversations") is not None:
            # an input with no prompt yet (AMI, ICSI: built from the raw transcripts)
            self._prompt_turn = {"from": "User", "value": value, "type": "text"}
            self.rec["conversations"].insert(0, self._prompt_turn)
        elif self._prompt_turn is not None:
            self._prompt_turn["value"] = value


class Filter:
    """Base class: __call__ returns False to drop the line."""
    name = None

    def __init__(self, ctx, **params):
        self.ctx = ctx

    def __call__(self, s):
        raise NotImplementedError


FILTERS = {}


def spec_name(spec):
    """A filter spec is a name or {name: params}."""
    return spec if isinstance(spec, str) else next(iter(spec))


def split_trailing_tags(text):
    """(text before the trailing tags, stripped; the tags) of a line."""
    m = TRAILING_TAGS_RE.search(text)
    core, tail = (text[:m.start()], text[m.start():]) if m else (text, "")
    return core.rstrip(), tail


def count_words(text):
    """Words with a letter or digit, tags left out."""
    return sum(bool(WORD_CHAR_RE.search(w)) for w in TAG_RE.sub(" ", text).split())


@functools.lru_cache(maxsize=None)
def load_common_words(lang="fr"):
    """The words lowercase mid-sentence of a language (`common_words` of conf/rules.yaml,
    built by build_common_words.py), read once per run; empty when the language has none."""
    path = language_rules(lang).get("common_words")
    if not path:
        return frozenset()
    with open(os.path.join(CONF_DIR, path), encoding="utf-8") as f:
        return frozenset(w.strip() for w in f if w.strip())


def register(cls):
    FILTERS[cls.name] = cls
    return cls


def _compile(pattern, ignore_case=False):
    return re.compile(pattern, re.IGNORECASE if ignore_case else 0)


@register
class NormalizeUnicode(Filter):
    """Unicode NFC: an accent written as a combining mark after its letter ("e" + U+0301) becomes
    the accented letter ("é"), as in the rest of the text and in the rules (CommonVoice: 1 %)."""
    name = "normalize_unicode"

    def __call__(self, s):
        s.text = unicodedata.normalize("NFC", s.text)
        return True


@register
class StripHtml(Filter):
    """Decode HTML entities ("&amp;", "&lt;i&gt;", twice for "&amp;gt;") and remove inline
    HTML tags like <i>...</i> or <font color="..."> (keeps their content)."""
    name = "strip_html"
    RE = re.compile(r"</?[a-zA-Z]{1,10}(?:\s+[^<>]*)?\s*/?>")

    def __call__(self, s):
        s.text = self.RE.sub(" ", html.unescape(html.unescape(s.text)))
        return True


@register
class MapTags(Filter):
    """Map dataset-specific event markers to canonical [tag]s.

    `tables` names entries of the config's `tag_tables`; each entry is
    {pattern, tags, ignore_case}. `tags: []` removes the marker.
    """
    name = "map_tags"

    def __init__(self, ctx, tables, **params):
        super().__init__(ctx)
        self.rules = []
        for table in tables:
            for rule in ctx["tag_tables"][table]:
                tags = rule.get("tags", [])
                unknown = set(tags) - set(ctx["tags"])
                if unknown:
                    raise ValueError(f"tag table {table}: unknown tags {unknown}, add them to `tags`")
                repl = " " + " ".join(f"[{t}]" for t in tags) + " "
                self.rules.append((_compile(rule["pattern"], rule.get("ignore_case", False)), repl))

    def __call__(self, s):
        text = s.text
        for pattern, repl in self.rules:
            text = pattern.sub(repl, text)
        s.text = text
        return True


@register
class DropIfMatch(Filter):
    """Drop the line when any pattern matches."""
    name = "drop_if_match"

    def __init__(self, ctx, patterns, ignore_case=False, **params):
        super().__init__(ctx)
        self.patterns = [_compile(p, ignore_case) for p in patterns]

    def __call__(self, s):
        return not any(p.search(s.text) for p in self.patterns)


@register
class UncasedNoPunctuation(Filter):
    """A line with no capital at all loses its punctuation, so that it gets a prompt asking for no
    case nor punctuation (ChoosePrompt, `nocasepunc`): VoxPopuli lines with no raw text are the
    lowercase normalised one with the recase model's punctuation ("ryanair vient encore
    d'annoncer l'annulation..."), proper nouns lowercase too."""
    name = "uncased_no_punctuation"
    RE = re.compile(r"\s*[.,;:!?…]+(?=\s|$)")

    def __call__(self, s):
        if not has_case(s.text):
            s.text = " ".join(self.RE.sub(" ", s.text).split())
        return True


@register
class DropOcr(Filter):
    """Drop the lines of books damaged by OCR: an elided letter before a consonant ("l'ide",
    "d'p", "i'cn'b") with more than `max_unknown` of the words outside `common_words` (accents
    lost: "prparer", "rvlateur"). Colloquial dialogue has the first sign but known words and stays
    ("j'savais", "m'sieu", "v'là les gendarmes", "l'cageux").
    """
    name = "drop_ocr"
    RE = re.compile(r"\b[a-zàâäéèêëîïôöùûüç]['’][bcdfgjklmnpqrstvwxzç]", re.IGNORECASE)

    def __init__(self, ctx, max_unknown=0.1, **params):
        super().__init__(ctx)
        self.max_unknown, self.common = max_unknown, load_common_words("fr")

    def _known(self, w):
        return w in self.common or bool(ELISION_RE.match(w)) and ELISION_RE.sub("", w) in self.common

    def __call__(self, s):
        if not self.RE.search(s.text):
            return True
        words = [w for w in map(_bare, s.text.split()) if LETTER_RE.search(w)]
        unknown = sum(not self._known(w) for w in words)
        return unknown <= self.max_unknown * len(words)


@register
class DropUnknownBrackets(Filter):
    """Drop the line when a [ or ] remains outside the canonical [tag]s (speaker labels, comments, cut-off brackets)."""
    name = "drop_unknown_brackets"

    def __init__(self, ctx, **params):
        super().__init__(ctx)
        self.tag_re = re.compile(r"\[(?:%s)\]" % "|".join(map(re.escape, ctx["tags"])))

    def __call__(self, s):
        return not re.search(r"[\[\]]", self.tag_re.sub(" ", s.text))


@register
class Replace(Filter):
    """Regex substitutions: rules = [{pattern, repl, ignore_case, keep_case}], or the name of a
    list of `replacements` in conf/rules.yaml (`lang`): its `words` ({phrase: replacement}, whole
    words, any case) then its `patterns`.

    keep_case capitalises repl when the match starts with an uppercase letter.
    """
    name = "replace"

    def __init__(self, ctx, rules, lang="fr", **params):
        super().__init__(ctx)
        if isinstance(rules, str):
            rules = self.named(lang, rules)
        self.rules = [(_compile(r["pattern"], r.get("ignore_case", False)), r["repl"], r.get("keep_case", False))
                      for r in rules]

    @staticmethod
    def named(lang, name):
        """The rules of `replacements: {name: ...}` of conf/rules.yaml."""
        sets = language_rules(lang).get("replacements") or {}
        if name not in sets:
            raise ValueError(f"replace: no replacements {name!r} for lang {lang!r} in conf/rules.yaml: {sorted(sets)}")
        out = []
        for phrase, to in (sets[name].get("words") or {}).items():
            to, not_after = (to["to"], to.get("not_after")) if isinstance(to, dict) else (to, None)
            # "peut être" -> "peut-être", not after "ça": (?<!\bça )
            after = "".join(rf"(?<!\b{re.escape(w)} )" for w in sorted(as_words(not_after)))
            pattern = after + r"(?<![\w'’-])" + r"\s+".join(map(re.escape, phrase.split())) + r"(?![\w'’-])"
            out.append({"pattern": pattern, "repl": to.replace("\\", r"\\"), "ignore_case": True, "keep_case": True})
        return out + list(sets[name].get("patterns") or [])

    def __call__(self, s):
        text = s.text
        for pattern, repl, keep_case in self.rules:
            if keep_case:
                text = pattern.sub(
                    lambda m, r=repl: m.expand(r)[:1].upper() + m.expand(r)[1:] if m.group(0)[:1].isupper()
                    else m.expand(r), text)
            else:
                text = pattern.sub(repl, text)
        s.text = text
        return True


@register
class RestoreElisions(Filter):
    """Put back the apostrophe of an elided word written apart from the next one: "j ai" ->
    "j'ai", "qu il" -> "qu'il", "L univers" -> "L'univers" (`restore_elisions` of conf/rules.yaml).
    Left as they are: spelled letters ("N I C O", a one-letter word before or after, a capital
    letter mid-sentence: "le jour J en fait"), a letter
    after a number ("trois d", "2 d") or after a word of `not_after` ("un t"), and the next words
    of `not_before` ("a t il").
    """
    name = "restore_elisions"
    NEXT_RE = re.compile(r"[aàâäeéèêëiîïoôöuûùüyÿhœæ]", re.IGNORECASE)
    ONE_LETTER_WORDS = frozenset("à a y".split())           # "à l époque" is not a spelled word

    def __init__(self, ctx, lang="fr", **params):
        super().__init__(ctx)
        r = language_rules(lang)["restore_elisions"]
        self.words, self.not_after = as_words(r["words"]), as_words(r.get("not_after"))
        self.not_before = {w: as_words(v) for w, v in (r.get("not_before") or {}).items()}

    def __call__(self, s):
        tokens = s.text.split()
        out = []
        for i, tok in enumerate(tokens):
            low = tok.lower()
            nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
            bare_next = _bare(nxt)
            prev = _bare(out[-1]) if out else ""
            spelled = len(prev) == 1 and prev not in self.ONE_LETTER_WORDS \
                or tok.isupper() and len(tok) == 1 and out and not out[-1].endswith((".", "?", "!", "…", '"', "«"))
            if low in self.words and self.NEXT_RE.match(nxt) and len(bare_next) > 1 and not spelled \
                    and not prev.isdigit() and prev not in self.not_after \
                    and bare_next not in self.not_before.get(low, ()):
                tokens[i + 1] = tok + "'" + nxt
                continue
            out.append(tok)
        s.text = " ".join(out)
        return True


@register
class FixSpuriousCaps(Filter):
    """Lowercase capitalised function words with no sentence boundary before (see fix_spurious_caps.py)."""
    name = "fix_spurious_caps"

    def __call__(self, s):
        s.text = fix_spurious_caps_text(s.text)
        return True


@register
class FixMixedCase(Filter):
    """Normalise words with an uppercase letter after a lowercase one ("AciCLOvir" -> "Aciclovir")."""
    name = "fix_mixed_case"
    RE = re.compile(r"\b[^\W\d_]*[a-zà-ÿ][^\W\d_]*[A-ZÀ-Þ][^\W\d_]*\b")

    def __init__(self, ctx, exceptions=(), **params):
        super().__init__(ctx)
        self.exceptions = set(exceptions)

    def __call__(self, s):
        def repl(m):
            w = m.group(0)
            if w in self.exceptions:
                return w
            return w[0] + w[1:].lower() if w[0].isupper() else w.lower()
        s.text = self.RE.sub(repl, s.text)
        return True


@register
class AllCaps(Filter):
    """Runs of >= min_words ALL-CAPS words: lowercase them (action: lowercase) or drop the line (action: drop)."""
    name = "all_caps"

    def __init__(self, ctx, min_words=3, action="lowercase", **params):
        super().__init__(ctx)
        word = r"[A-ZÀ-Þ][A-ZÀ-Þ'’-]*[A-ZÀ-Þ]"
        self.re = re.compile(rf"\b{word}(?:[\s,]+{word}){{{min_words - 1},}}\b")
        self.action = action

    def __call__(self, s):
        m = self.re.search(s.text)
        if not m:
            return True
        if self.action == "drop":
            return False

        def repl(m):
            run = m.group(0).lower()
            before = s.text[:m.start()].rstrip()
            if not before or before[-1] in ".?!…:«\"":
                run = run[0].upper() + run[1:]
            return run
        s.text = self.re.sub(repl, s.text)
        return True


@register
class NumbersToWords(Filter):
    """Write numbers in French words (num2words). Units listed in `units` right after a number are spelled out.

    Numbers glued to letters or symbols are written as they are read and stay untouched
    ("CO2", "3D", "COVID-19", "2.0", "1/2", "2+2=4"): the model can learn to write them.
    """
    name = "numbers_to_words"
    THOUSANDS_RE = re.compile(r"(?<![\d,.])(\d{1,3})((?:[ \u00a0\u202f]\d{3})+|(?:\.\d{3})+)(?!\d|[.,]\d)")
    HOURS_RE = re.compile(r"(?<![\w+=/,.])(\d{1,2}) ?h(?: ?(\d{2}))?(?![\w/])")
    ORDINAL_RE = re.compile(r"(?<![\w+])(\d+)(er|re|ère|e|ème|eme)(s?)\b")
    DECIMAL_RE = re.compile(r"(?<![\w+=/,.])(?<![^\W\d_]-)(\d+),(\d+)(?!\w|,\d|[+=/])")
    INT_RE = re.compile(r"(?<![\w+=/,.])(?<![^\W\d_]-)(\d+)(?!\d|[\w,./]\d|[^\W\d_]|[+=/])")

    def __init__(self, ctx, lang="fr", units=None, **params):
        super().__init__(ctx)
        try:
            from num2words import num2words
        except ImportError as e:
            raise ImportError("numbers_to_words needs num2words: `uv pip install num2words`") from e
        self.n2w = lambda n, **kw: num2words(n, lang=lang, **kw)
        units = units or {}
        self.units = {k: v for k, v in units.items()}
        if self.units:
            alt = "|".join(re.escape(u) for u in sorted(self.units, key=len, reverse=True))
            self.unit_re = re.compile(rf"(?<![\w+=/,.])(\d+(?:,\d+)?)\s?({alt})(?![^\W\d_])")
        else:
            self.unit_re = None

    def _digits(self, digits):
        lead = len(digits) - len(digits.lstrip("0"))
        rest = digits.lstrip("0")
        words = ["zéro"] * lead
        if rest:
            words.append(self.n2w(int(rest)))
        return " ".join(words)

    def __call__(self, s):
        text = self.THOUSANDS_RE.sub(lambda m: m.group(1) + re.sub(r"\D", "", m.group(2)), s.text)
        text = re.sub(r"(?<!\w)[nN]°\s*(?=\d)", "numéro ", text)
        if self.unit_re is not None:
            text = self.unit_re.sub(lambda m: f"{m.group(1)} {self.units[m.group(2)]}", text)

        def ordinal(m):
            n = int(m.group(1))
            if n == 1 and m.group(2) in ("er", "re", "ère"):
                word = "premier" if m.group(2) == "er" else "première"
            else:
                word = self.n2w(n, to="ordinal")
            return word + m.group(3)
        text = self.HOURS_RE.sub(
            lambda m: ("une heure" if int(m.group(1)) == 1 else self.n2w(int(m.group(1))) + " heures")
            + (" " + self.n2w(int(m.group(2))) if m.group(2) and int(m.group(2)) else ""), text)
        text = self.ORDINAL_RE.sub(ordinal, text)
        text = self.DECIMAL_RE.sub(lambda m: f"{self.n2w(int(m.group(1)))} virgule {self._digits(m.group(2))}", text)
        text = self.INT_RE.sub(lambda m: self.n2w(int(m.group(1))), text)
        s.text = text
        return True


@register
class AddFinalPeriod(Filter):
    """Add "." when the text has no final . ? ! … (before trailing [tags]).

    Only for datasets where every line is a full sentence (CommonVoice, FLEURS).
    replace_trailing: a trailing , ; : is replaced by "." instead.
    """
    name = "add_final_period"
    END_RE = re.compile(r"[.?!…][\"»”)']*$")

    def __init__(self, ctx, replace_trailing=True, **params):
        super().__init__(ctx)
        self.replace_trailing = replace_trailing

    def __call__(self, s):
        core, tail = split_trailing_tags(s.text)
        if core and not self.END_RE.search(core):
            if self.replace_trailing and core[-1] in ",;:":
                core = core[:-1].rstrip()
            core += "."
            s.text = core + tail
        return True


def has_case(text):
    """True when the text has an uppercase letter outside its [tags]."""
    return any(c.isupper() for c in TAG_RE.sub(" ", text))


def is_uncased(s):
    """True when the line will get a prompt asking for no case nor punctuation (ChoosePrompt):
    the dataset's `nocasepunc` is on and the line has no uppercase letter nor punctuation.
    The filters that fix case leave such a line as it is."""
    text = TAG_RE.sub(" ", s.text)
    return bool(s.nocasepunc) and not PUNCT_RE.search(text) and not has_case(text)


def stays_lowercase(s):
    """True when no capital must be added to the line: its prompt asks for lowercase."""
    return is_uncased(s) or (s.lowercase and not has_case(s.text))


def is_lowercase_punctuated(text):
    """True when the line has punctuation and no uppercase letter (ChoosePrompt, `lowercase`)."""
    text = TAG_RE.sub(" ", text)
    return bool(PUNCT_RE.search(text)) and not has_case(text)


def is_long_unpunctuated(s):
    """True when the line has no punctuation and at least `nocasepunc` words (a number):
    ChoosePrompt lowercases it and gives it a prompt asking for no case nor punctuation."""
    n = s.nocasepunc
    return n is not True and bool(n) and not PUNCT_RE.search(TAG_RE.sub(" ", s.text)) and count_words(s.text) >= n


class _MarkerFilter(Filter):
    """Keep or strip a kind of marker in the text.

    mode: strip -- markers removed (`strip`; it returns False to drop the line).
    mode: keep  -- markers kept (`keep`).
    The output sets the mode (`outputs` in the config) and the prompt says which (ChoosePrompt).
    """

    def __init__(self, ctx, mode="strip", **params):
        super().__init__(ctx)
        if mode not in ("strip", "keep"):
            raise ValueError(f"{self.name}: mode must be strip or keep, got {mode!r}")
        self.mode = mode

    def strip(self, s):
        """Remove the markers from s.text; return False to drop the line."""
        raise NotImplementedError

    def keep(self, s):
        return True

    def __call__(self, s):
        return self.keep(s) if self.mode == "keep" else self.strip(s)


@register
class Events(_MarkerFilter):
    """Non-speech event [tags] (see _MarkerFilter for the modes).

    In strip mode, a line with a tag in drop_if_stripped (e.g. inaudible) is
    dropped, since removing it would leave a missing word.
    """
    name = "events"

    def __init__(self, ctx, drop_if_stripped=("inaudible",), **params):
        super().__init__(ctx, **params)
        self.drop_re = re.compile(r"\[(?:%s)\]" % "|".join(map(re.escape, drop_if_stripped))) \
            if drop_if_stripped else None

    def strip(self, s):
        if self.drop_re is not None and self.drop_re.search(s.text):
            return False
        s.text = TAG_RE.sub(" ", s.text)
        return True


# French word lists of conf/rules.yaml (see the comments there)
NO_END_WORDS = words("fr", "no_end_words")            # "avec le. Les deux": the "." is a pause
ELISION_RE = re.compile(r"^(?:%s)['’]" % "|".join(sorted(words("fr", "elided"), key=len, reverse=True)),
                        re.IGNORECASE)


def _bare(word):
    """Lowercase word without surrounding quotes/brackets/punctuation."""
    return word.strip("«»\"“”()[].,;:?!…").lower().replace("’", "'")


# imperative / inverted pronoun: "rassure-moi", "dit-il", "va-t-en" (the verb is the common word)
PRONOUN_SUFFIX_RE = re.compile(r"(?:-t)?-(?:%s)$" % "|".join(sorted(words("fr", "inverted_pronouns"))))


def _is_common(word, common, no_end=NO_END_WORDS):
    bare = _bare(word)
    return bare in common or bare in no_end or PRONOUN_SUFFIX_RE.sub("", bare) in common


@functools.lru_cache(maxsize=None)
def elision_re(lang):
    """The elided words of a language with their apostrophe ("l'", "qu'"); en has none: never matches."""
    elided = words(lang, "elided")
    return re.compile(r"^(?:%s)['’]" % "|".join(sorted(elided, key=len, reverse=True)) if elided else r"(?!)",
                      re.IGNORECASE)


def _lower_first(word):
    i = next((k for k, c in enumerate(word) if c.isalpha()), None)
    return word if i is None else word[:i] + word[i].lower() + word[i + 1:]



@register
class Disfluencies(_MarkerFilter):
    """Hesitations (see _MarkerFilter for the modes).

    keep:  normalised to one spelling per sound (`hesitations` of conf/rules.yaml, per `lang`):
           fr: "euh" (euh, heu, euhm, eum, ehum, uh) and "hmm" (hum, hm, hmm, mh, mm, mmm, mmh, mhm,
               mm-hum, and the MH/MMh/MMM/HM of the recase model);
           en: "uh" (uh, uhh, uuh), "um" (um, umm, uhm), "er" (er, erm) and "hmm" (the forms of fr but "hum", a word).
    strip: removed. Discourse words (fr: ben, bah, hein, ah, oh; en: eh, oh, ah, huh, uh-oh) are
           words and stay. In en, "mm-hmm", "uh-huh", "uh-uh", "mm-mm" are hesitations too: their
           yes / no cannot be told from a mere hesitation.
           Cut-off words ("L- la fin") go too (`truncations`), and repeated function words
           are collapsed ("je je" -> "je", `repetitions`); other repetitions are not touched. A comma or "..." right after
           the hesitation goes with it; a sentence end after it (". ? !") moves to the
           previous word, a "." as a comma when the line goes on (`period_after_hesitation`:
           in transcripts it marks the pause, not a sentence end: "J'ai appelé euh. L'éditrice"
           -> "J'ai appelé, l'éditrice"), unless that word cannot end a sentence (`cut_after_no_end`):
           "drop": the end is removed and the next word lowercased when it is a common word
           ("Mais euh. Rassure-moi" -> "Mais rassure-moi", "pour euh. L'Ascension" -> "pour l'Ascension");
           "ellipsis": the cut is written "..." ("Mais... rassure-moi"; a "." there would be taken
           by fix_punctuation for a recase error). The next word is capitalised when a
           capitalised hesitation is removed.
    """
    name = "disfluencies"
    # cut-off word / false start: "L- la fin", "l'année proch- c'est" (not "peut-être", nor " - ")
    TRUNCATION = r"[^\W\d_][\w'’]*-(?=[\s.,;:?!…]|$)"
    SENTINEL_RE = re.compile(r"\x00[\s\x00,;:]*([«\"“(]\s*)?([^\W\d_])")
    # \x01 + punctuation: a sentence end that followed a removed hesitation
    MOVED_END_RE = re.compile(r"(?:([^\s\x00\x01]+)([\s\x00]*))?\x01([.?!]+)")
    LOST_START_RE = re.compile(r"\x02[\s\x00]*([^\s\x00\x02]+)")

    def __init__(self, ctx, lang="fr", truncations=True, repetitions=True, cut_after_no_end="drop",
                 period_after_hesitation="comma", **params):
        """lang: a language of conf/rules.yaml (hesitation spellings, function words never repeated:
        `disfluencies`, words that cannot end a sentence: `no_end_words`).
        truncations: also strip cut-off words "j- je" (off for subtitles, where "anti- " is a split compound).
        repetitions: also collapse repeated function words "je je" -> "je"."""
        super().__init__(ctx, **params)
        forms = self.forms(lang)
        any_form = "|".join(r for _, r in forms)
        alt = rf"(?:{any_form})(?![\w'’-])|{self.TRUNCATION}" if truncations else rf"(?:{any_form})(?![\w'’-])"
        self.strip_re = re.compile(rf"(?:,\s*)?(?<![\w'’-])(?:{alt})(?:\s*(\.\.\.|…|,|[.?!]+))?")
        self.word_re = re.compile(r"(?<![\w'’-])(?:%s)(?![\w'’-])" % "|".join(f"({r})" for _, r in forms))
        self.word_forms = [f for f, _ in forms]
        self.no_end = words(lang, "no_end_words")
        self.elision_re = elision_re(lang)
        if cut_after_no_end not in ("drop", "ellipsis"):
            raise ValueError(f"disfluencies: cut_after_no_end must be drop or ellipsis, got {cut_after_no_end!r}")
        self.cut_after_no_end = cut_after_no_end
        if period_after_hesitation not in ("comma", "keep"):
            raise ValueError(f"disfluencies: period_after_hesitation must be comma or keep, "
                             f"got {period_after_hesitation!r}")
        self.period_comma = period_after_hesitation == "comma"
        self.common = load_common_words(lang)
        self.repetitions = repetitions
        self.repetition_re = re.compile(
            r"(?<![\w'’-])(%s)((?:\s*(?:\.\.\.|…|[.,])?\s+\1)+)(?![\w'’-])(,(?=\s))?"
            % "|".join(sorted(words(lang, "disfluencies", "repeatable_not"))), re.IGNORECASE)

    @staticmethod
    def forms(lang):
        """[(written form, regex of its spellings)] of the hesitations of a language."""
        hesitations = language_rules(lang).get("disfluencies", {}).get("hesitations")
        if not hesitations:
            raise ValueError(f"disfluencies: no hesitations for lang {lang!r} in conf/rules.yaml")
        return [(form, "|".join(spellings)) for form, spellings in hesitations.items()]

    @classmethod
    def _collapse(cls, m):
        if m[1].lower() == "la" and len(m[2].split()) >= 2:   # "la la la": singing
            return m[0]
        # "des des, des, Australiennes" -> "des Australiennes": the comma after a run with commas is a pause
        return m[1] + ("" if "," in m[2] else m[3] or "")

    @staticmethod
    def _replace(m):
        out = " \x01" + m[1] + " " if m[1] and m[1][0] in ".?!" and m[1] != "..." else " "
        # commas around a hesitation go with it ("il a, euh, mangé"); a removed
        # capital leaves a sentinel so that the next word gets one
        return out + ("\x00" if m[0].lstrip(", ")[0].isupper() else "")

    def _move_end(self, m):
        word, space, end = m[1], m[2] or "", m[3]
        if word and word[-1].isalpha() and _bare(word) not in self.no_end:
            # "\x02": the next word is lowercased, as a sentence start that was not one
            if self.period_comma and end == "." and m.string[m.end():].lstrip(" .\x00\x01")[:1].isalpha():
                return word + "," + space + "\x02"
            return word + end + space
        if word and word[-1].isalpha() and end == ".":
            # "\x02": the next word lost its sentence start
            return word + ("..." if self.cut_after_no_end == "ellipsis" else "") + space + "\x02"
        return (word or "") + space

    def _lower_next(self, m):
        word = m[1]
        if word.isupper() and sum(c.isalpha() for c in word) > 1:   # "ONU", "ILS SONT"; not "À."
            return word
        if self.elision_re.match(word) or _is_common(word, self.common, self.no_end):
            return _lower_first(word)
        return word

    def normalize(self, s):
        def repl(m):
            form = self.word_forms[m.lastindex - 1]
            before = TAG_RE.sub(" ", s.text[:m.start()]).rstrip()   # "? [silence] Euh": a sentence start
            starts = not before or before[-1] in ".?!…\"«“"
            return form.capitalize() if starts and cased else form
        cased = not stays_lowercase(s)
        s.text = self.word_re.sub(repl, s.text)

    def keep(self, s):
        self.normalize(s)
        return True

    def strip(self, s):
        cased = not stays_lowercase(s)
        text = self.strip_re.sub(self._replace, s.text)
        if "\x01" in text:
            text = self.MOVED_END_RE.sub(self._move_end, text)
        if "\x00" in text:
            text = self.SENTINEL_RE.sub(lambda m: " " + (m[1] or "") + m[2].upper(), text).replace("\x00", " ")
        if "\x02" in text:
            text = self.LOST_START_RE.sub(lambda m: " " + self._lower_next(m) if cased else " " + m[1],
                                          text).replace("\x02", " ")
        if self.repetitions:
            text = self.repetition_re.sub(self._collapse, text)
        s.text = text
        return True


@register
class DisfluencyModel(Filter):
    """Disfluencies removed by linagora/flaubert_disfluency_detector and the code of its
    repository (src/detect_disfluencies.py), with no rule of ours on top: hesitations,
    repetitions, false starts ("je vais je vais", "les de le") and cut-off words ("bonj-").

    mode: strip -- the model's removal: hmm / pff and cut-off words first (`strip_fillers` of the
                   repository), then the words the model flags (euh, repeats, false starts), a
                   whole word at a time, and the hesitations it left ("goodies euh...": an output
                   without disfluencies has none); the repository renders the text (spacing, punctuation and
                   capital left by a removed word). The discourse words (ben, bah, hein...) are
                   added to the repository's protect list: never removed, unless repeated.
          drop  -- lines with a word flagged are dropped.
          keep  -- the hesitations are normalised to one spelling (`Disfluencies` keep), the model
                   is not loaded (outputs that keep the disfluencies).
    In every mode the hesitations are first written "euh" / "hmm" (`hesitations` of
    conf/rules.yaml), the spellings the repository knows. [tags] are left out of the model input
    and stay where they are. The lines of a chunk are classified in one batch (`prepare`).
    """
    name = "disfluency_model"
    DISCOURSE_WORDS = words("fr", "disfluency_model", "discourse_words")
    # names of the repository code used here, checked at load: a new revision may change them
    REPO_API = ("annotate", "strip_fillers", "strip_disfluency_tags", "_FILLER_WORD_RE", "_PROTECT_WORDS")

    def __init__(self, ctx, mode="strip", model="linagora/flaubert_disfluency_detector",
                 revision="4307de6b425f153b7b82fe181eef190d05443197", threshold=0.55, truncations=True,
                 device="cpu", threads=32, batch_size=64, **params):
        """model: Hugging Face id or local copy of linagora/flaubert_disfluency_detector.
        revision: commit of the repository (weights and code), so that an update does not change
        the outputs unnoticed.
        threshold: probability over which a token is disfluent (0.55: 0.5 removes "pour cent" of
        "cent pour cent", "Suisse" of "Suisse Allemande").
        truncations: also remove the cut-off words "bonj-" (off for subtitles, where "anti- " is a
        split compound)."""
        super().__init__(ctx)
        if mode not in ("keep", "strip", "drop"):
            raise ValueError(f"disfluency_model: mode must be keep, strip or drop, got {mode!r}")
        self.mode, self.threshold, self.truncations, self.batch_size = mode, threshold, truncations, batch_size
        self.hesitations = Disfluencies(ctx, lang="fr", mode="keep")
        # a hesitation word, as normalize() writes it, with its punctuation: "euh,"
        forms = "|".join(re.escape(form) for form, _ in Disfluencies.forms("fr"))
        self.hesitation_re = re.compile(rf"(\W*)({forms})\W*", re.IGNORECASE)
        if mode != "keep":
            self.detector, self.model = self._load(model, revision, device, threads)
            self.detector._PROTECT_WORDS = self.detector._PROTECT_WORDS | self.DISCOURSE_WORDS
            self.gpu = int(device.split(":")[1]) if device.startswith("cuda:") else 0 if device == "cuda" else None

    @classmethod
    @functools.lru_cache(maxsize=None)
    def _load(cls, model_id, revision, device, threads):
        """(detect_disfluencies module of the model repository, loaded model), once per run."""
        import importlib
        import sys
        import torch
        torch.set_num_threads(threads)
        path = model_id
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download
            path = snapshot_download(model_id, revision=revision)
        sys.path.insert(0, os.path.join(path, "src"))
        detector = importlib.import_module("detect_disfluencies")
        missing = [a for a in cls.REPO_API if not hasattr(detector, a)]
        if missing:
            raise RuntimeError(f"disfluency_model: {path} has no {', '.join(missing)} (another revision?)")
        return detector, detector.load_model(path, device=device)

    def strip_fillers(self, text):
        """hmm / pff (and the cut-off words with `truncations`) removed, by the repository code."""
        if self.truncations:
            return self.detector.strip_fillers(text)
        return self.detector.strip_disfluency_tags(
            self.detector._FILLER_WORD_RE.sub(lambda m: "<dsf>" + m[0] + "</dsf>", text))

    @staticmethod
    def model_input(text):
        return " ".join(w for w in text.split() if not TAG_WORD_RE.fullmatch(w))

    def model_text(self, s):
        """s.text with its hesitations normalised and its fillers removed (s left unchanged)."""
        text = s.text
        self.hesitations.normalize(s)
        s.text, out = text, self.strip_fillers(s.text)
        # the repository capitalises what follows a removed filler, a [tag] too ("hmm [noise]." ->
        # "[Noise]."): the tags are lowercase
        return TAG_RE.sub(lambda m: m[0].lower(), out)

    def prepare(self, samples):
        """The chunk's lines classified in one batch. The annotations of a dataset are cached by
        text without its [tags] (the model never sees them): the outputs and the max30 files
        share most lines."""
        if self.mode == "keep":
            return
        cache = self.ctx.setdefault("disfluency_model_cache", {})
        texts = [self.model_text(s) for s in samples]
        keys = [self.model_input(t) for t in texts]
        todo = sorted({k for k in keys if k and k not in cache})
        annotations = self.detector.annotate(self.model, todo, threshold=self.threshold, batch_size=self.batch_size,
                                             gpu=self.gpu, progress=False, word_level=True, protect=True)
        for k, a in zip(todo, annotations):
            cache[k] = a
        for s, t, k in zip(samples, texts, keys):
            s.disfluency_annotation = (s.text, t, cache.get(k))

    @staticmethod
    def flags(text, a):
        """Per word of the model input (text without its [tags]): its flagged characters."""
        out, pos = [], 0
        for w in text.split():
            if TAG_WORD_RE.fullmatch(w):
                continue
            out.append([a.covered[p] and a.char_label[p] == 1 for p in range(pos, pos + len(w))])
            pos += len(w) + 1
        return out

    def flag_hesitations(self, text, flags):
        """flags with the hesitations the model left flagged too (their letters, not their
        punctuation); None when there are no flags and no hesitation."""
        plain = [w for w in text.split() if not TAG_WORD_RE.fullmatch(w)]
        out = None
        for i, w in enumerate(plain):
            m = self.hesitation_re.fullmatch(w)
            if m is None:
                continue
            if out is None:
                out = [list(f) for f in flags] if flags is not None else [[False] * len(x) for x in plain]
            out[i][m.start(2):m.end(2)] = [True] * (m.end(2) - m.start(2))
        return flags if out is None else out

    def render(self, text, flags):
        """text with the flagged characters removed by the repository code. It renders the text
        without its [tags] (a tag is no sentence start for it: "âge ? [silence] Euh, là" would
        keep "là"), then each tag goes back after the last word kept before it."""
        words = text.split()
        plain = [w for w in words if not TAG_WORD_RE.fullmatch(w)]
        marked = []
        for w, f in zip(plain, flags):
            word, k = "", 0
            while k < len(w):
                j = k
                while j < len(w) and f[j] == f[k]:
                    j += 1
                word += f"<dsf>{w[k:j]}</dsf>" if f[k] else w[k:j]
                k = j
            marked.append(word)
        out = self.detector.strip_disfluency_tags(" ".join(marked)).split()
        if len(plain) == len(words):
            return " ".join(out)
        # plain word i -> its word in out, matched on the letters it keeps
        key = lambda w: "".join(c for c in w.lower() if c.isalnum())
        kept = [key("".join(c for c, x in zip(w, f) if not x)) for w, f in zip(plain, flags)]
        to_out = {}
        for i, j, n in difflib.SequenceMatcher(None, kept, [key(o) for o in out], autojunk=False).get_matching_blocks():
            to_out.update((i + d, j + d) for d in range(n))
        after = collections.defaultdict(list)               # out index (-1: line start) -> its tags
        last, i = -1, 0
        for w in words:
            if TAG_WORD_RE.fullmatch(w):
                after[last].append(w)
            else:
                last = to_out.get(i, last)
                i += 1
        res = after[-1][:]
        for j, o in enumerate(out):
            res += [o] + after[j]
        return " ".join(res)

    def __call__(self, s):
        if self.mode == "keep":
            self.hesitations.normalize(s)
            return True
        before, text, a = getattr(s, "disfluency_annotation", (None, None, None))
        if before != s.text:                                # not prepared: one line alone
            self.prepare([s])
            before, text, a = s.disfluency_annotation
        flags = self.flags(text, a) if a is not None and a.ok else None   # not ok: tokens not aligned on the text
        flags = self.flag_hesitations(text, flags)
        flagged = flags is not None and any(map(any, flags))
        if self.mode == "drop":
            return not flagged and text == before
        s.text = self.render(text, flags) if flagged else text
        return True                         # a line left empty: process() keeps a few


@register
class NormalizePunctuation(Filter):
    """Punctuation debris, safe on any text (hand-punctuated ones too):
      - byte-order marks / zero-width characters removed;
      - leading punctuation removed (".elles étaient", "? C'est"), ".." -> ".", "...." -> "...",
        "?." / ".?" -> "?", ". ," / ", ." -> "." (a tag removed between them), "quoi ??" -> "quoi ?".
    """
    name = "normalize_punctuation"
    INVISIBLE_RE = re.compile("[\ufeff\u200b\u200c\u200d\u2060]")
    RULES = [
        (re.compile(r"^[\s.,;:?!…]+"), ""),
        (re.compile(r"\.{4,}"), "..."),
        (re.compile(r"(?<!\.)\.\.(?!\.)"), "."),
        (re.compile(r"([?!])\1+"), r"\1"),
        (re.compile(r"([?!])\s*\.(?!\.)"), r"\1"),
        (re.compile(r"(?<![.])\.\s*(?=[?!])"), ""),
        # a removed tag between two marks: "la mafia. [noise], parce que" -> "la mafia., parce que"
        (re.compile(r"([.?!…])\s*[,;](?=\s|$)"), r"\1"),
        (re.compile(r"\s*,\s*(?=[.?!](?!\.))"), ""),
    ]

    def __call__(self, s):
        text = self.INVISIBLE_RE.sub("", s.text)
        for pattern, repl in self.RULES:
            text = pattern.sub(repl, text)
        s.text = text
        return True


@register
class FixPunctuation(Filter):
    """Punctuation and case errors of the recase model and of subtitles, after normalize_punctuation.

    On cased lines only (not when the prompt asks for no case nor punctuation):
      - "." after a word that cannot end a sentence is removed, and the next word lowercased
        when it is a common word ("avec le. Les deux" -> "avec le les deux",
        "de la. Parentification" -> "de la parentification", but "dans le. Gange" -> "dans le Gange");
      - a sentence start that lost its punctuation gets a "." (`insert_periods`): a subject
        pronoun or a sentence connector capitalised after a lowercase word ("très bien Bon" ->
        "très bien. Bon", "il faut Il faut" -> "il faut. Il faut"), or another cue start
        followed by a lowercase word ("absurde Si ça" -> "absurde. Si ça", not "de Le Pen");
        after a word that cannot end a sentence it is lowercased instead ("les Les zoos" -> "les les zoos"),
        and after a comma too ("l'infrarouge, Donc, qui" -> "l'infrarouge, donc, qui");
      - any other determiner, pronoun, conjunction or preposition capitalised after a lowercase
        word or a comma is lowercased ("des tablettes Qui sont" -> "des tablettes qui sont"),
        unless the next word is capitalised too (proper noun: "à La Réunion", "Le Pen");
      - a lowercase word after ". ? !" is capitalised ("le sens. d'accord" -> "le sens. D'accord").
    `common_words`: the list of words lowercase mid-sentence (build_common_words.py), used to
    tell common words from proper nouns. The words are those of `lang` in conf/rules.yaml.
    """
    name = "fix_punctuation"

    def __init__(self, ctx, lang="fr", insert_periods=True, **params):
        super().__init__(ctx)
        self.common = load_common_words(lang)
        self.insert_periods = insert_periods
        self.no_end = words(lang, "no_end_words")
        self.function_words = self.no_end | words(lang, "function_words")
        self.elision_re = elision_re(lang)
        # the word lists of `fix_punctuation` in conf/rules.yaml
        r = language_rules(lang)["fix_punctuation"]
        self.starters = as_words(r["starters"])           # "très bien Bon" -> "très bien. Bon"
        self.interjections = as_words(r["interjections"])
        self.cue_starts = as_words(r["cue_starts"])       # "est absurde Si ça arrive"
        self.cue_elided = as_words(r.get("cue_elided"))
        # a "." after these is removed: no_end_words, or the list of the language
        self.no_period_after = as_words(r["no_period_after"]) if "no_period_after" in r else self.no_end
        self.title_case = r.get("title_case", False)
        # no capital after an abbreviation's period: "356 avant J.-C. lors", "Piquet Jr. a été", "M. Dupont"
        self.abbrev_re = re.compile(r"^\(?(?:[A-ZÀ-Þ]\.-?)+$|^\(?(?i:%s)\.$"
                                    % "|".join(map(re.escape, sorted(as_words(r["abbreviations"])))))

    def _is_common(self, word):
        return _is_common(word, self.common, self.no_end) or _bare(word) in self.starters

    _lower_first = staticmethod(_lower_first)

    @staticmethod
    def _upper_first(word):
        i = next((k for k, c in enumerate(word) if c.isalpha()), None)
        return word if i is None else word[:i] + word[i].upper() + word[i + 1:]

    @staticmethod
    def _capitalised(word):
        letters = [c for c in word if c.isalpha()]
        return len(letters) > 1 and letters[0].isupper() and any(c.islower() for c in letters[1:])

    def _starter(self, word):
        bare = _bare(word)
        if self.elision_re.match(bare):
            return bare.split("'")[0] + "'" in self.starters or bare == "d'accord"
        return bare in self.starters

    def _in_title(self, prev, nxt):
        """Title case ("What Is It", "It Girl"; `title_case`, en): no sentence start, whatever the
        word. Not before the pronoun "I", capitalised anyway ("now So I went")."""
        return self.title_case and (prev[:1].isupper() or nxt[:1].isupper() and not re.match(r"I\b", nxt))

    def __call__(self, s):
        if not is_uncased(s) and not is_long_unpunctuated(s):   # those stay as they are, lowercased at the end
            s.text = self._fix_case(s.text)
        return True

    def _cue_start(self, word):
        if not word[:1].isalpha():                      # ':«Maintenant', '"Pour une'
            return False
        bare = _bare(word)
        elided = self.elision_re.match(bare)
        if elided:
            return bare[:elided.end()] in self.cue_elided and word[elided.end():][:1].islower()
        return bare in self.cue_starts

    def _fix_case(self, text):
        tokens = text.split()
        for i in range(1, len(tokens)):
            prev, cur = tokens[i - 1], tokens[i]
            if cur.startswith("["):
                continue
            first = next((c for c in cur if c.isalpha()), "")
            # "avec le. Les deux" -> "avec le les deux"
            if prev.endswith(".") and not prev.endswith("..") and prev[:-1].isalpha() \
                    and prev[:-1].lower() in self.no_period_after:
                tokens[i - 1] = prev = prev[:-1]
                if first.isupper() and self._is_common(cur):
                    tokens[i] = cur = self._lower_first(cur)
                continue
            last = prev[-1]
            nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
            capitalised = first.isupper() and (self._capitalised(cur) or self.elision_re.match(cur) or cur == "À")
            if last == "," and capitalised and (self._starter(cur) or self._cue_start(cur) and nxt[:1].islower()
                                                or _bare(cur) in self.function_words and not nxt[:1].isupper()):
                tokens[i] = self._lower_first(cur)     # "l'infrarouge, Donc, qui" -> "l'infrarouge, donc, qui"
                continue
            if last.isalpha() and last.islower():
                if not capitalised:
                    continue
                if self._in_title(prev, nxt):
                    continue
                if _bare(prev) in self.no_end or _bare(prev) in self.interjections:
                    # "les Les zoos", "que Je", "pour L'ascension"; not "à La Rascasse" (next word
                    # capitalised) nor "des Maires" (nouns are left to the source)
                    elided = self.elision_re.match(cur)
                    if elided and cur[elided.end():][:1].islower() \
                            or not elided and (_bare(cur) in self.function_words or self._starter(cur)) \
                            and not nxt[:1].isupper():
                        tokens[i] = self._lower_first(cur)
                elif self.insert_periods and (self._starter(cur) or self._cue_start(cur) and nxt[:1].islower()
                                              and len(_bare(prev)) > 1):
                    tokens[i - 1] = prev + "."
                elif _bare(cur) in self.function_words and not nxt[:1].isupper():
                    tokens[i] = self._lower_first(cur)     # "des tablettes Qui sont" -> "qui sont"
            elif last in ".?!" and not prev.endswith("...") and first.islower() \
                    and not self.abbrev_re.match(prev):
                tokens[i] = self._upper_first(cur)
        return " ".join(tokens)


@register
class SpeakerLabels(Filter):
    """Remove subtitle speaker labels at a sentence start: "Fossoyeur : Ne tirez pas",
    "Public : Paris !", "L: Ah je fais...". A label is one to three capitalised words that
    are not common words (proper nouns), an ALL-CAPS name of up to 3 letters, or one of
    `label_words` ("Public", "Participants"); "Résultat : on a gagné" is speech and stays.
    """
    name = "speaker_labels"
    LABEL = r"(?:[A-ZÀ-Þ](?:\.[A-ZÀ-Þ])+\.?|[A-ZÀ-Þ]\.(?= ?:)|(?:[A-ZÀ-Þ]\. )?[A-ZÀ-Þ][\w'’-]*(?: [A-ZÀ-Þ][\w'’-]*){0,2})"   # "L.L", "N.", "E. Macron"
    # at a sentence start or after a tag ("[laughter] CB : Certainement"), before a word or a quote
    RE = re.compile(rf"(?:^|(?<=[.?!…»\]] ))(?<!\b[A-ZÀ-Þ]\. )({LABEL}) ?: (?=[^\W\d_]|[«\"“])")
    # inside a sentence ("pour le fun B : Ou si"): a proper noun or initials before a capitalised word
    # not after a preposition or determiner: "Section de Wavefolding : Les", "selon Saint Matthieu : Heureux"
    MID_RE = re.compile(rf"([\w'’-]*[^\W\d_]) ({LABEL}) ?: (?=[A-ZÀ-Þ])")
    # inside a sentence only initials ("MHL", "B", "N."): a name there is mostly speech ("comme
    # dirait Kant : Le beau", "Isaac Newton : J'ai vu", "les anciens Romains : Pacta")
    MID_LABEL_RE = re.compile(r"[A-ZÀ-Þ]{1,3}|[A-ZÀ-Þ](?:\.[A-ZÀ-Þ])*\.?")
    NOT_BEFORE_LABEL = NO_END_WORDS | words("fr", "speaker_labels", "not_before_label")

    def __init__(self, ctx, label_words=None, **params):
        """label_words: the words of a label (default: `speaker_labels` of conf/rules.yaml)."""
        super().__init__(ctx)
        self.common = load_common_words("fr")
        self.label_words = frozenset(label_words) if label_words else words("fr", "speaker_labels", "label_words")

    def _is_label(self, label):
        words = [w for w in label.split() if not re.fullmatch(r"[A-ZÀ-Þ]\.", w)]   # initial "E."
        if len(words) == 1 and re.fullmatch(r"[A-ZÀ-Þ]{1,3}|[A-ZÀ-Þ](?:\.[A-ZÀ-Þ])+\.?", label):
            return True
        if len(words) == 1 and _bare(label) in self.label_words:
            return True
        return all(_bare(w) not in self.common for w in words)

    def __call__(self, s):
        text = self.RE.sub(lambda m: "" if self._is_label(m[1]) else m[0], s.text.strip())
        s.text = self.MID_RE.sub(
            lambda m: m[1] + ". " if ELISION_RE.sub("", _bare(m[1])) not in self.NOT_BEFORE_LABEL   # "d'un Français : Ferdinand"
            and self.MID_LABEL_RE.fullmatch(m[2]) and self._is_label(m[2]) else m[0],
            text)
        return True


@register
class SpeechRate(Filter):
    """Drop lines whose text cannot match the audio: audio shorter than `min_duration`, more
    than `max_wps` words per second ("très bien" in 0.16 s), or less than `min_wps` on the audio the words must cover, when it
    is longer than `min_wps_duration` ("C'est enregistré." on 20 s: speech left out). An event
    tag stands for at most `max_tag_seconds` of audio ("[applause] Merci." on 190 s is dropped).
    With `median_ratio` (null: off), the low bound is also the dataset's median rate / `median_ratio`
    (Yodas: 2.76 / 4 = 0.69 w/s), so a fast dataset drops more than one read slowly (CommonVoice 1.93).
    With `max_ratio` (null: off), the high bound is the median * `max_ratio` instead of `max_wps`.
    More than `max_lps` letters per second (null: off) is dropped too, whatever the words: the
    text cannot fit the audio ("aura des aides financières pour continuer à m'aider" on 0.98 s).
    The median is `median_wps`, or computed at start on `median_lines` lines spread evenly over the
    dataset's inputs when not given (not the first lines: the rebuilt inputs are sorted by id).
    """
    name = "speech_rate"

    def __init__(self, ctx, max_wps=8.0, min_wps=0.3, min_wps_duration=5.0, max_tag_seconds=10.0,
                 median_ratio=4, max_ratio=4, median_wps=None, median_lines=25000, min_duration=0.1,
                 max_lps=30, **params):
        super().__init__(ctx)
        self.max_lps = max_lps
        self.max_wps, self.min_wps, self.min_wps_duration = max_wps, min_wps, min_wps_duration
        self.max_tag_seconds, self.min_duration = max_tag_seconds, min_duration
        if (median_ratio or max_ratio) and median_wps is None:
            median_wps = self._dataset_median(median_lines)
        if median_ratio:
            self.min_wps = max(min_wps, median_wps / median_ratio)
        if max_ratio:
            self.max_wps = median_wps * max_ratio
        if median_ratio or max_ratio:
            print(f"[{ctx.get('dataset')}] speech_rate: median {median_wps:.2f} w/s "
                  f"-> min_wps {self.min_wps:.2f}, max_wps {self.max_wps:.2f}")

    def _rates(self, s):
        """(words / duration, audio not covered by tags, words / that audio, duration), None without a duration."""
        turns = s.rec.get("conversations") or []
        duration = next((t.get("duration") for t in turns if t.get("type") == "audio"), None) \
            or s.rec.get("duration")
        if not duration:
            return None
        words = count_words(s.text)
        spoken = duration - self.max_tag_seconds * len(TAG_RE.findall(s.text))
        return words / duration, spoken, words / spoken if spoken > 0 else float("inf"), duration

    def _dataset_median(self, n_lines):
        # once per dataset (the outputs share it), on the input text: close enough for a median
        cache = self.ctx.setdefault("median_wps", {})
        if self.ctx["dataset"] not in cache:
            rates, per_file = [], -(-n_lines // max(len(self.ctx["inputs"]), 1))
            for path in self.ctx["inputs"]:
                with open(path, "rb") as f:
                    step = max(sum(1 for _ in f) // per_file, 1)
                with open(path, encoding="utf-8") as f:
                    for line in itertools.islice(f, 0, None, step):
                        r = self._rates(Sample(json.loads(line), None))
                        if r and r[1] > 1:
                            rates.append(r[2])
            rates.sort()
            cache[self.ctx["dataset"]] = rates[len(rates) // 2] if rates else 0.0
        return cache[self.ctx["dataset"]]

    def __call__(self, s):
        r = self._rates(s)
        if r is None:
            return True
        rate, spoken, spoken_rate, duration = r
        if duration < self.min_duration:
            return False
        if rate > self.max_wps:
            return False
        if self.max_lps and len(LETTER_RE.findall(TAG_RE.sub(" ", s.text))) > self.max_lps * duration:
            return False
        if spoken > self.min_wps_duration and spoken_rate < self.min_wps:
            return False
        return True


@register
class JoinSplitCompounds(Filter):
    """Join compounds split by a subtitle line break: "peut- être" -> "peut-être",
    "c'est- à- dire" -> "c'est-à-dire", when the joined word is in `common_words`.
    Run before `disfluencies`, which would take "peut-" for a cut-off word.
    Not joined, a cut-off word said again: the fragment starts the next word ("re- regarde",
    "il est- il est", "je l'ai- je l'aime"; not "installez- vous, installez- vous"), or it is one
    letter ("on t- on trouve"; not "e- commerce", "a- t- elle").
    """
    name = "join_split_compounds"

    def __init__(self, ctx, **params):
        super().__init__(ctx)
        self.common = frozenset(w for w in load_common_words("fr") if "-" in w)

    @staticmethod
    def _cut_off(tokens, i):
        """tokens[i] ("x-") is a cut-off word: one letter, or said again in one of the next 2 words."""
        frag = _bare(tokens[i])[:-1]
        if len(frag) == 1:                                # not "e- commerce", "a- t- elle", "Y- a-t-il"
            return frag != "e" and not (i + 1 < len(tokens) and "-" in tokens[i + 1])
        for w in tokens[i + 1:i + 3]:
            if _bare(w).startswith(frag) and "-" not in w:
                return True
            if w[-1:] in ".,;:?!":                        # "installez- vous, installez- vous"
                return False
        return False

    def __call__(self, s):
        if "- " not in s.text:
            return True
        tokens = s.text.split()
        out, i = [], 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.endswith("-") and tok[:-1][-1:].isalpha() and not self._cut_off(tokens, i):
                for j in (1, 2):
                    joined = tok + "".join(tokens[i + 1:i + j + 1])
                    if i + j < len(tokens) and not joined.endswith("-") and _bare(joined) in self.common:
                        out.append(joined)
                        i += j + 1
                        break
                else:
                    out.append(tok)
                    i += 1
                continue
            out.append(tok)
            i += 1
        s.text = " ".join(out)
        return True


@register
class NormalizeInterjections(Filter):
    """One spelling per interjection (`normalize_interjections` of conf/rules.yaml): beh/bé/bâ/ba
    -> bah, bein/bin -> ben, hin -> hein, ahh/aah -> ah, ohh/ooh -> oh, pf/pfff -> pff. They are
    words: kept in every variant.
    """
    name = "normalize_interjections"
    FORMS = list(language_rules("fr")["normalize_interjections"]["forms"].items())
    RE = re.compile(r"(?<![\w'’-])(?:%s)(?![\w'’-])" % "|".join(f"(?P<g{i}>{p})" for i, (_, p) in enumerate(FORMS)),
                    re.IGNORECASE)

    def __call__(self, s):
        def repl(m):
            form = self.FORMS[int(m.lastgroup[1:])][0]
            word = m[0]
            if word.isupper() and len(word) > 1:
                return form.upper()
            return form.capitalize() if word[0].isupper() else form
        s.text = self.RE.sub(repl, s.text)
        return True


@register
class NormalizeSpacing(Filter):
    """Collapse spaces and the punctuation debris left by removed markers."""
    name = "normalize_spacing"
    RULES = [
        (re.compile(r"\s+"), " "),
        (re.compile(r"([.?!…])[,;:]+"), r"\1"),            # "sert., Et" -> "sert. Et"
        (re.compile(r"([,;:])(?:\s*[,;:])+"), r"\1"),       # ", ," -> ","
        (re.compile(r"[,;:]\s*(?=[.?!…])"), ""),            # ",." -> "."
        (re.compile(r"(?<!\.)([.?!…])\s+\.(?!\.)"), r"\1"),  # ". ." / "? ." -> "." / "?" (keeps "...")
        (re.compile(r"\s+(,|\.(?![^\W_]))"), r"\1"),      # "pense ." -> "pense."; keeps "with .NET"
        (re.compile(r"\(\s*\)"), ""),                       # empty parentheses
        (re.compile(r"^(?:[\s,;:]|\.(?!\w))+"), ""),         # leading debris (keeps ".eu")
        (re.compile(r"\s+$"), ""),
    ]

    def __call__(self, s):
        text = s.text
        for pattern, repl in self.RULES:
            text = pattern.sub(repl, text)
        s.text = text
        return True


@register
class PeriodIfNextSentence(Filter):
    """Subtitle cues with no final mark get a "." when the next cue of the same recording starts
    a sentence: a common word capitalised ("...les clous" + "Voilà votre tête de lit" -> "...les
    clous."). A next cue starting with a proper noun ("Guillaume", "Heidegger") or a lowercase
    word, a cue ending with a word that cannot end a sentence ("dans", "le") and lines with no
    case or that will get a nocasepunc prompt (is_long_unpunctuated) are left as they are. The next cues come from the dataset's
    inputs (`_silence` files excluded), ordered by offset within each recording.
    """
    name = "period_if_next_sentence"
    END_RE = re.compile(r"[.?!…,;:\"»”)'-]$")

    def __init__(self, ctx, **params):
        super().__init__(ctx)
        if "sentence_before_next" not in ctx:               # ctx is per dataset, shared by its outputs
            ctx["sentence_before_next"] = self._build(ctx["inputs"], load_common_words("fr"))
        self.keys = ctx["sentence_before_next"]

    @staticmethod
    def _key(rec):
        audio = next(t for t in rec["conversations"] if t.get("type") == "audio")
        return audio["value"], round(audio.get("offset") or 0.0, 3)

    @classmethod
    def _build(cls, inputs, common):
        """{wav: offsets of the cues followed by a cue starting with a capitalised common word}"""
        by_wav = defaultdict(list)
        for path in inputs:
            if "_silence" in os.path.basename(path):
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    wav, start = cls._key(rec)
                    word = (Sample(rec, None).text.split(None, 1) or [""])[0]
                    by_wav[wav].append((start, word[:1].isupper() and not word.isupper()
                                        and _is_common(word.lower(), common)))
        keys = {}
        for wav, cues in by_wav.items():
            cues.sort(key=lambda c: c[0])
            keys[wav] = {start for (start, _), (_, sentence) in zip(cues, cues[1:]) if sentence}
        return keys

    def __call__(self, s):
        core, tail = split_trailing_tags(s.text)
        if not core or self.END_RE.search(core) or not has_case(core) or is_long_unpunctuated(s) \
                or _bare(core.rsplit(None, 1)[-1]) in NO_END_WORDS:
            return True
        wav, start = self._key(s.rec)
        if start in self.keys.get(wav, ()):
            s.text = core + "." + tail
        return True


@register
class FrenchTypography(Filter):
    """One typography for every dataset: "'" for every apostrophe (’ ‘ ʼ), '"' for “ ” „, "..." for "…",
    and the French spacing: a space before ; : ! ? (»), after « and after ; : ! ? before a letter
    ("Quoi?Rien" -> "Quoi ? Rien", "«oui»" -> "« oui »"). No space is added inside "?!",
    nor in "10:30", "http://" (the mark must be followed by a space, the end or a closing quote).
    Plain spaces only: the non-breaking ones are replaced. Doubled apostrophes are single ("l''on"),
    and subtitle dialogue dashes are removed ("deux mondes... - Oui, c'est sûr").
    """
    name = "french_typography"
    RULES = [
        (re.compile("[’‘ʼ']+"), "'"),
        (re.compile("[“”„‟]"), '"'),
        (re.compile(r"(?:^|(?<=[.?!»:]))\s*[-–—]\s+(?=[^\W\d_])"), " "),
        (re.compile("…"), "..."),
        (re.compile("[\u00a0\u202f\u2009]"), " "),
        (re.compile(r"(?<=[^\s(\[«])\s*([;:!?]+)(?=\s|$|[»\"”])"), r" \1"),
        (re.compile(r"(?<=[^\W\d_])\s*([;:!?]+)(?=[^\W\d_])"), r" \1 "),
        (re.compile(r"«\s*"), "« "),
        (re.compile(r"\s*»"), " »"),
        (re.compile(r"\s+"), " "),
    ]

    def __call__(self, s):
        text = s.text
        for pattern, repl in self.RULES:
            text = pattern.sub(repl, text)
        s.text = text.strip()
        return True


@register
class EnglishTypography(Filter):
    """The typography of french_typography with the English spacing: "'" for every apostrophe, '"'
    for “ ” „, "..." for "…", plain spaces, and no space before ; : ! ? ("The species ?" -> "The species?").
    """
    name = "english_typography"
    RULES = [
        (re.compile("[’‘ʼ]"), "'"),
        (re.compile("[“”„‟]"), '"'),
        (re.compile("…"), "..."),
        (re.compile("[\u00a0\u202f\u2009]"), " "),
        (re.compile(r"\s+([;:!?]+)(?=\s|$)"), r"\1"),
        (re.compile(r"\s+"), " "),
    ]

    def __call__(self, s):
        text = s.text
        for pattern, repl in self.RULES:
            text = pattern.sub(repl, text)
        s.text = text.strip()
        return True


class PromptPool:
    """The prompts of a context category, drawn with the category's group weights."""

    def __init__(self, ctx, category):
        if category not in ctx["contexts"]:
            raise ValueError(f"context category {category!r} not in {ctx['contexts_path']}")
        groups = ctx["contexts"][category]
        if isinstance(groups, list):
            groups = {"1.0": groups}
        self.weights = [float(w) for w in groups]
        self.groups = list(groups.values())

    def draw(self, rng):
        return rng.choice(rng.choices(self.groups, weights=self.weights)[0])


MARKERS = ("events", "disfluencies")      # the kinds of marker an output keeps or strips


def category_constraints(category):
    """The constraints a context category asks for, as compose_asr_prompts.py takes them:
    no_events_nocasepunc -> {nocasepunc: True, events: False}."""
    found = {marker: not neg for neg, marker in re.findall(r"(no_)?(events|disfluencies)", category)}
    return {"nocasepunc": category == "nocasepunc" or category.endswith("_nocasepunc") or None, **found}


class ChoosePrompt:
    """Last step of every output: the prompt of each line, chosen from its final text.

    keep: the markers the output keeps. Every line gets a prompt asking for them
        (events, disfluencies or events_disfluencies).
    explicit: {default, with_output}: the fraction of the lines whose prompt states explicitly
        what the text does not have, as one combination of these blocks, all possible ones equally
        likely but twice as likely when they hold a `favored` block:
            no_events, no_disfluencies  the markers the output drops (compositional prompts; in
                                        fixed ones, only in an output keeping no marker);
            casepunc                    compositional prompts, cased lines with some uppercase
                                        letter or punctuation: case and punctuation.
        with_output when some block is favored, default otherwise.
    favored: the blocks worth stating more often: the markers another output of the dataset keeps,
        casepunc when the dataset has nocasepunc lines.
    nocasepunc: which lines get the _nocasepunc version of their category (asks for no case
        nor punctuation; the base one is nocasepunc instead of default_contexts):
        false -- none: lowercase fragments of a cased corpus keep the usual prompts;
        true  -- lines with no uppercase letter and no punctuation;
        N     -- those, and lines of N words or more with no punctuation but some capitals ("ces quatre dernières années la Guyane n'a pas connu..."):
                 unpunctuated subtitles whose proper nouns kept their case. They are lowercased.
    lowercase: true -- the lines with punctuation and no uppercase letter (a corpus with no case to
        recover) get a prompt asking for lowercase with punctuation. Compositional prompts only.
    Fixed prompts: a line whose category has no _nocasepunc version (events_disfluencies) is
    dropped. Compositional prompts (ctx["composer"]) exist for every category: none is dropped.
    """
    label = "prompt"

    def __init__(self, ctx, keep=(), explicit=None, favored=(), nocasepunc=False, lowercase=False):
        if lowercase and not ctx.get("composer"):
            raise ValueError("lowercase needs compositional prompts: the fixed lists have no such prompt")
        self.lowercase = lowercase
        if nocasepunc is not False and nocasepunc is not True and not isinstance(nocasepunc, int):
            raise ValueError(f"nocasepunc must be true, false or a number of words, got {nocasepunc!r}")
        self.keep, self.nocasepunc = keep, nocasepunc
        self.explicit, self.favored = explicit or {"default": 0.0, "with_output": 0.0}, set(favored)
        self.composer = ctx.get("composer")
        self.pools = {} if self.composer else {name: PromptPool(ctx, name) for name in ctx["contexts"]}

    def _nocase(self, s):
        if not self.nocasepunc:
            return False
        if is_uncased(s):
            return True
        if is_long_unpunctuated(s):
            s.text = s.text.lower()
            return True
        return False

    def blocks(self, s, nocase):
        """The explicit blocks of the line's prompt: a combination of no_events, no_disfluencies
        and casepunc, or none."""
        dropped = [m for m in MARKERS if m not in self.keep] if self.composer or not self.keep else []
        text = TAG_RE.sub(" ", s.text)
        cased = self.composer and not nocase and (has_case(text) or PUNCT_RE.search(text)) \
            and not (self.lowercase and is_lowercase_punctuated(s.text))
        possible = dropped + (["casepunc"] if cased else [])
        combos = [c for n in range(1, len(possible) + 1) for c in itertools.combinations(possible, n)]
        rate = self.explicit["with_output" if self.favored & set(possible) else "default"]
        if not combos or s.rng.random() >= rate:
            return ()
        return s.rng.choices(combos, weights=[2 if self.favored & set(c) else 1 for c in combos])[0]

    def category(self, s, nocase, blocks):
        """events, no_events, events_no_disfluencies..."""
        parts = [m for m in MARKERS if m in self.keep] + [f"no_{m}" for m in MARKERS if m in blocks]
        return "_".join(parts) or ("nocasepunc" if nocase else "default_contexts")

    def __call__(self, s):
        nocase = self._nocase(s)
        blocks = self.blocks(s, nocase)
        category = self.category(s, nocase, blocks)
        if nocase and category != "nocasepunc":
            category += "_nocasepunc"
        if self.composer:
            constraints = category_constraints(category)
            if "casepunc" in blocks:
                constraints["nocasepunc"] = False
            elif self.lowercase and is_lowercase_punctuated(s.text):
                constraints["nocasepunc"] = "lowercase"
            s.prompt = self.composer.compose(s.rng, **constraints)
            return True
        pool = self.pools.get(category)
        if pool is None:
            if not self.keep:              # no_events_no_disfluencies_nocasepunc: the no_events one
                pool = self.pools[category.replace("_no_disfluencies", "")]
            else:
                return False
        s.prompt = pool.draw(s.rng)
        return True


def resolve_filters(specs, blocks, seen=()):
    """The filter specs of a list of filter and block names, the blocks spliced in."""
    for spec in specs or []:
        name = spec_name(spec)
        if name in blocks:
            if not isinstance(spec, str):
                raise ValueError(f"block {name!r} takes no params (write its filters instead)")
            if name in seen:
                raise ValueError(f"block {name!r} includes itself")
            yield from resolve_filters(blocks[name], blocks, seen + (name,))
        elif name in FILTERS:
            yield spec
        else:
            raise ValueError(f"{name!r} is neither a filter nor a block; filters: {sorted(FILTERS)}, "
                             f"blocks: {sorted(blocks)}")


def build_filters(specs, ctx, overrides=None):
    """overrides: {filter name: params} merged into that filter's params (the markers an output keeps)."""
    filters = []
    seen = Counter()
    for spec in specs:
        name, params = spec_name(spec), (None if isinstance(spec, str) else next(iter(spec.values())))
        if name not in FILTERS:
            raise ValueError(f"unknown filter {name!r}, available: {sorted(FILTERS)}")
        params = {**(params or {}), **((overrides or {}).get(name) or {})}
        f = FILTERS[name](ctx, **params)
        seen[name] += 1
        f.label = name if seen[name] == 1 else f"{name}#{seen[name]}"
        filters.append(f)
    return filters


def output_path(path, output_dir, subdir=""):
    """<dataset output dir>/[<output subdir>/]<input name>."""
    return os.path.join(output_dir, subdir, os.path.basename(path))


def record_rng(seed, rec, lineno):
    key = f"{seed}:{rec.get('id', lineno)}".encode()
    return random.Random(int.from_bytes(hashlib.md5(key).digest()[:8], "little"))


def process(path, out, filters, prompt, seed, dry_run, limit, n_examples, empty_fraction=0.0, select=None):
    """empty_fraction: the lines left with no letter nor digit kept, with an empty text.
    select: the numbers of the only input lines read."""
    stats = defaultdict(Counter)
    examples = defaultdict(list)
    n = n_kept = 0
    fout = None
    if not dry_run:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fout = open(out + ".tmp", "w", encoding="utf-8")
    with open(path, encoding="utf-8") as fin:
        lines = enumerate(tqdm(fin, desc=os.path.basename(path), unit=" lines", leave=False))
        if select:
            lines = (x for x in lines if x[0] in select)
        if limit:
            lines = itertools.islice(lines, limit)
        # chunks of lines go through the filters one filter at a time, so that a filter with a
        # `prepare` (a model) sees the whole chunk at once
        while chunk := list(itertools.islice(lines, CHUNK_SIZE)):
            n += len(chunk)
            alive = []
            for lineno, line in chunk:
                rec = json.loads(line)
                alive.append(Sample(rec, record_rng(seed, rec, lineno), prompt.nocasepunc, prompt.lowercase))
            for f in filters:
                if hasattr(f, "prepare"):
                    f.prepare(alive)
                kept = []
                for s in alive:
                    before, before_prompt = s.text, s.prompt
                    if not f(s):
                        stats[f.label]["dropped"] += 1
                        if len(examples[f.label]) < n_examples:
                            examples[f.label].append((before, None))
                        continue
                    kept.append(s)
                    if s.text != before or s.prompt != before_prompt:
                        stats[f.label]["changed"] += 1
                        if len(examples[f.label]) < n_examples:
                            examples[f.label].append((before, s.text if s.prompt == before_prompt
                                                      else f"{s.text}   <prompt: {s.prompt}>"))
                alive = kept
            for s in alive:
                if not WORD_CHAR_RE.search(s.text):     # a line of tags only stays where tags are kept
                    if s.rng.random() >= empty_fraction:
                        stats["(empty)"]["dropped"] += 1
                        continue
                    stats["(empty)"]["changed"] += 1     # kept: nothing to transcribe
                    s.text = ""
                if not prompt(s):
                    stats[prompt.label]["dropped"] += 1
                    continue
                n_kept += 1
                if fout is not None:
                    fout.write(json.dumps(s.rec, ensure_ascii=False) + "\n")
    if fout is not None:
        fout.close()
        os.replace(out + ".tmp", out)

    print(f"{path}\n  -> {out}{' (dry-run, not written)' if dry_run else ''}")
    print(f"  kept {n_kept}/{n} ({n_kept / max(n, 1):.2%})")
    for label in [f.label for f in filters] + ["(empty)", prompt.label]:
        c = stats.get(label)
        if not c:
            continue
        print(f"  {label}: changed {c['changed']} ({c['changed'] / max(n, 1):.2%}), "
              f"dropped {c['dropped']} ({c['dropped'] / max(n, 1):.2%})")
        for before, after in examples.get(label, []):
            print(f"      {before[:150]!r}\n    → {'DROP' if after is None else repr(after[:150])}")


# the outputs a dataset can list: the markers each keeps (base: none) and its subdirectory
OUTPUTS = {"base": (), "events": ("events",), "disfluencies": ("disfluencies",),
           "events_disfluencies": ("events", "disfluencies")}
# filters whose mode an output sets: the marker they handle
MARKER_FILTERS = {"events": "events", "disfluencies": "disfluencies", "disfluency_model": "disfluencies"}
DATASET_KEYS = {"input", "output", "outputs", "output_lines", "nocasepunc", "lowercase", "filters"}


def load_config(path):
    """The YAML config, with the files of its `include` (relative to it) merged under it."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    merged = {}
    for inc in cfg.pop("include", None) or []:
        merged.update(load_config(os.path.join(os.path.dirname(path), inc)))
    return {**merged, **cfg}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="YAML config (see data/processing/clean_asr/conf/fr.yaml)")
    p.add_argument("--datasets", nargs="+", help="only these datasets from the config")
    p.add_argument("--dry-run", action="store_true", help="print stats and examples, write nothing")
    p.add_argument("--limit", type=int, default=0, help="only the first N lines of each file")
    p.add_argument("--examples", type=int, default=3, help="examples printed per filter and file")
    p.add_argument("--output-dir", help="write <dir>/<dataset>/ instead of each dataset's output (e.g. to test in a scratch dir)")
    p.add_argument("--overwrite", action="store_true", help="overwrite existing outputs")
    p.add_argument("--root", help="override the config's root (e.g. to test on a local copy)")
    p.add_argument("--prompts", choices=["fixed", "compositional"], help="override the config's prompts")
    args = p.parse_args()

    cfg = load_config(args.config)
    ctx = {"tag_tables": cfg.get("tag_tables", {}), "tags": cfg.get("tags", [])}
    prompts = args.prompts or cfg.get("prompts", "fixed")
    if prompts == "compositional":
        sys.path.insert(0, os.path.join(REPO_ROOT, "data", "contexts"))
        from compose_asr_prompts import PromptComposer
        comp = dict(cfg["compositional"])
        hesitations = [form for form, _ in Disfluencies.forms(comp["audio_language"])]   # conf/rules.yaml
        ctx["composer"] = PromptComposer(os.path.join(REPO_ROOT, comp.pop("dir")), hesitations=hesitations, **comp)
    elif prompts == "fixed":
        ctx["contexts_path"] = os.path.join(REPO_ROOT, cfg["contexts"])
        with open(ctx["contexts_path"], encoding="utf-8") as f:
            ctx["contexts"] = json.load(f)
    else:
        raise SystemExit(f"prompts: {prompts!r}, expected fixed or compositional")
    root, seed, blocks = os.path.expandvars(args.root or cfg["root"]), cfg.get("seed", 0), cfg.get("blocks") or {}
    if "$" in root:
        raise SystemExit(f"root {root!r}: set the variable")
    if set(blocks) & set(FILTERS):
        raise SystemExit(f"blocks named like a filter: {sorted(set(blocks) & set(FILTERS))}")
    explicit = cfg.get("explicit_fraction", {"default": 0.05, "with_output": 0.15})
    empty_fraction = cfg.get("empty_fraction", 0.0)

    datasets = cfg["datasets"]
    if args.datasets:
        missing = set(args.datasets) - set(datasets)
        if missing:
            raise SystemExit(f"not in config: {sorted(missing)}")
        datasets = {k: datasets[k] for k in args.datasets}

    for name, dcfg in datasets.items():
        if set(dcfg) - DATASET_KEYS:
            raise SystemExit(f"{name}: unknown keys {sorted(set(dcfg) - DATASET_KEYS)}, expected {sorted(DATASET_KEYS)}")
        outputs = dcfg.get("outputs") or ["base"]
        if set(outputs) - set(OUTPUTS):
            raise SystemExit(f"{name}: unknown outputs {sorted(set(outputs) - set(OUTPUTS))}, expected {list(OUTPUTS)}")
        specs = list(resolve_filters(dcfg["filters"], blocks))
        output_dir = os.path.join(args.output_dir, name) if args.output_dir else os.path.join(root, dcfg["output"])
        inputs = dcfg["input"] if isinstance(dcfg["input"], list) else [dcfg["input"]]
        paths = sorted({p for pattern in inputs for p in glob.glob(os.path.join(root, pattern))})
        if not paths:
            print(f"[{name}] no input matches {inputs}")
        dctx = {**ctx, "dataset": name, "inputs": paths}   # per-dataset state shared by the outputs
        kept_somewhere = {m for o in outputs for m in OUTPUTS[o]}
        for output in outputs:
            keep = OUTPUTS[output]
            overrides = {f: {"mode": "keep"} for f, m in MARKER_FILTERS.items() if m in keep}
            filters = build_filters(specs, dctx, overrides)
            # a few prompts state what the text does not have (no markers, case and punctuation),
            # more when another output keeps the markers or some lines are lowercase unpunctuated
            favored = kept_somewhere | ({"casepunc"} if dcfg.get("nocasepunc", False) is not False else set())
            prompt = ChoosePrompt(dctx, keep, explicit=explicit, favored=favored,
                                  nocasepunc=dcfg.get("nocasepunc", False), lowercase=dcfg.get("lowercase", False))
            # output_lines: an output but base takes that many lines at random, other ones than the
            # other outputs while there are enough: the outputs of read speech, only there to show
            # that a prompt asking for markers may get none (the filters may drop a few)
            lines = dcfg.get("output_lines") if output != "base" else None
            for path in paths:
                select = None
                if lines:
                    with open(path, "rb") as f:
                        order = list(range(sum(1 for _ in f)))
                    random.Random(f"{seed}:{path}").shuffle(order)
                    rank = [o for o in outputs if o != "base"].index(output)
                    start = rank * lines if (rank + 1) * lines <= len(order) else 0
                    select = set(order[start:start + lines])
                out = output_path(path, output_dir, "" if output == "base" else output)
                if os.path.abspath(out) == os.path.abspath(path):
                    raise SystemExit(f"{name}: output {out} would overwrite its input")
                if not args.dry_run and os.path.exists(out) and not args.overwrite:
                    print(f"skip {path}: {out} exists (use --overwrite)")
                    continue
                process(path, out, filters, prompt, seed, args.dry_run, args.limit, args.examples, empty_fraction,
                        select)


if __name__ == "__main__":
    main()
