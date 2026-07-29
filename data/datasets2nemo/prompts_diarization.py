"""Shared diarization prompt/format machinery for the meeting-corpus converters.

This module is the single source of truth for the diarization output variants,
their per-format renderers, the prompts (per language), the backchannel handling,
and the lean-row prompt selection. Both ami2nemo.py and icsi2nemo.py (and any
other meeting converter) import from here, so a prompt/format edit applies
everywhere.

`seg` fields used by the renderers: n (speaker number, 1..), s/e (start/end
seconds, relative to the clip), text (words).

-------------------------------------------------------------------------------
ADDING A NEW LANGUAGE
-------------------------------------------------------------------------------
Prompts are keyed by language and live in `data/contexts/diarization_prompts.json`
(loaded into `_DIAR_PROMPTS` at import). The output *formats* themselves (the
render functions and their keys) are language-independent and live in
`_DIAR_FORMATS`; only the natural-language instruction text differs per
language. To add, say, French:

    1. Add a "fr" entry to the JSON file, mirroring the "en" structure:
         "fr": {
             "generic": { "asr": [...], "timestamps": [...], "timestamps_asr": [...] },
             "formats": { variant: { fmt: [...] for every fmt }, ... },
             "backchannel_suffixes": [...],
         }
    2. That's it. Rows whose `language == "fr"` will draw French prompts; any
       prompt list a language is missing falls back to "en" automatically
       (see `_prompts_for` / `_backchannel_suffixes_for`).

The format keys a language provides must be a subset of `_DIAR_FORMATS[variant]`;
`_validate_prompts()` checks at import time that "en" (the fallback) is complete.
"""

import json
import random
import re

from pathlib import Path

from ssak.utils.nemo_dataset import NemoDatasetRow, NemoTurn

DIAR_VARIANTS = ("asr", "timestamps", "timestamps_asr")

# Fallback language: every other language falls back to this one for any prompt
# list it does not define, and this one must be complete.
DEFAULT_LANGUAGE = "en"

# Base seed mixed into the per-row prompt rng (see `make_diar_lean_row`). Fixed so
# the prompt wording is reproducible run-to-run; change it to reshuffle all prompts.
PROMPT_SEED = 42

# Each diarization variant can be rendered in several output formats. A row picks
# one format, renders its target in that format, and is paired with a prompt drawn
# from that same format's prompt list (in the row's language) — so the prompt
# teaches the model which format to produce. The first format of each variant is
# the "default": generic, format-agnostic prompts map to it.

# Speaker-label styles shared across all three variants. `n` is 1-based.
#   letter -> 'A', 'B', ...    s_num -> 'S1', 'S2', ...    upper -> 'SPEAKER_00', ...
def _lbl_letter(n): return chr(ord("A") + n - 1)
def _lbl_s(n): return f"S{n}"
def _lbl_upper(n): return f"SPEAKER_{n - 1:02d}"

def _r_asr_speaker_colon(segs, meeting):
    return "\n".join(f"Speaker {s['n']}: {s['text']}" for s in segs if s["text"])

def _r_asr_bare_letter(segs, meeting):
    return "\n".join(f"{_lbl_letter(s['n'])}: {s['text']}" for s in segs if s["text"])

def _r_asr_s_num(segs, meeting):
    return "\n".join(f"{_lbl_s(s['n'])}: {s['text']}" for s in segs if s["text"])

def _r_asr_speaker_upper(segs, meeting):
    return "\n".join(f"{_lbl_upper(s['n'])}: {s['text']}" for s in segs if s["text"])

def _r_asr_dash(segs, meeting):
    return "\n".join(f"Speaker {s['n']} - {s['text']}" for s in segs if s["text"])

def _r_asr_letter(segs, meeting):
    return "\n".join(f"Speaker {chr(ord('A') + s['n'] - 1)}: {s['text']}"
                     for s in segs if s["text"])

def _r_ts_plain(segs, meeting):
    return "\n".join(f"Speaker {s['n']} {s['s']:.2f} {s['e']:.2f}" for s in segs)

def _r_ts_rttm(segs, meeting):
    # Classic NIST RTTM: SPEAKER <uri> <chan> <onset> <dur> <NA> <NA> <spk> <NA> <NA>
    return "\n".join(
        f"SPEAKER {meeting} 1 {s['s']:.2f} {s['e'] - s['s']:.2f} <NA> <NA> Speaker_{s['n']} <NA> <NA>"
        for s in segs)

def _r_ts_bracket(segs, meeting):
    return "\n".join(f"[{s['s']:.2f} - {s['e']:.2f}] Speaker {s['n']}" for s in segs)

def _r_ts_bare_letter(segs, meeting):
    return "\n".join(f"[{s['s']:.2f} - {s['e']:.2f}] {_lbl_letter(s['n'])}" for s in segs)

def _r_ts_s_num(segs, meeting):
    return "\n".join(f"[{s['s']:.2f} - {s['e']:.2f}] {_lbl_s(s['n'])}" for s in segs)

def _r_ts_speaker_upper(segs, meeting):
    return "\n".join(f"[{s['s']:.2f} - {s['e']:.2f}] {_lbl_upper(s['n'])}" for s in segs)

def _r_tsa_bracket_colon(segs, meeting):
    return "\n".join(f"[{s['s']:.2f}-{s['e']:.2f}] Speaker {s['n']}: {s['text']}"
                     for s in segs if s["text"])

def _r_tsa_bare_letter(segs, meeting):
    return "\n".join(f"[{s['s']:.2f}-{s['e']:.2f}] {_lbl_letter(s['n'])}: {s['text']}"
                     for s in segs if s["text"])

def _r_tsa_s_num(segs, meeting):
    return "\n".join(f"[{s['s']:.2f}-{s['e']:.2f}] {_lbl_s(s['n'])}: {s['text']}"
                     for s in segs if s["text"])

def _r_tsa_speaker_upper(segs, meeting):
    return "\n".join(f"[{s['s']:.2f}-{s['e']:.2f}] {_lbl_upper(s['n'])}: {s['text']}"
                     for s in segs if s["text"])

def _r_tsa_inline(segs, meeting):
    return "\n".join(f"Speaker {s['n']} ({s['s']:.2f}-{s['e']:.2f}): {s['text']}"
                     for s in segs if s["text"])

def _r_tsa_arrow(segs, meeting):
    return "\n".join(f"{s['s']:.2f} --> {s['e']:.2f}  Speaker {s['n']}: {s['text']}"
                     for s in segs if s["text"])

def _hms(t, sep):
    """Seconds -> 'HH:MM:SS<sep>mmm' (sep is ',' for SRT, '.' for WebVTT)."""
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"

def _r_tsa_srt(segs, meeting):
    blocks, idx = [], 1
    for s in segs:
        if not s["text"]:
            continue
        blocks.append(f"{idx}\n{_hms(s['s'], ',')} --> {_hms(s['e'], ',')}\n"
                      f"Speaker {s['n']}: {s['text']}")
        idx += 1
    return "\n\n".join(blocks)

def _r_tsa_vtt(segs, meeting):
    cues = [f"{_hms(s['s'], '.')} --> {_hms(s['e'], '.')}\nSpeaker {s['n']}: {s['text']}"
            for s in segs if s["text"]]
    return "WEBVTT\n\n" + "\n\n".join(cues)


# JSON: a structured, schema-heavy output (kept to a small slice of the rows, see
# JSON_FORMAT_RATIO). Speakers are identified as 'spk<N>'. Only fields derivable
# from the source segments (n/s/e/text) are emitted: there is no per-word timing,
# no confidence, no raw-vs-processed distinction and no per-segment language in the
# pipeline, so those schema fields are deliberately omitted rather than fabricated.
def _spk(n): return f"spk{n}"

def _r_asr_json(segs, meeting):
    # ASR + speaker, no timestamps.
    result = "\n".join(f"{_spk(s['n'])}: {s['text']}" for s in segs if s["text"])
    segments = [{"segment": s["text"], "spk_id": _spk(s["n"])}
                for s in segs if s["text"]]
    return json.dumps({"transcription_result": result, "segments": segments},
                      ensure_ascii=False, indent=2)

def _r_ts_json(segs, meeting):
    # Speaker + timestamps, no transcript: a per-speaker summary plus the segments.
    speakers = {}
    for s in segs:
        spk = _spk(s["n"])
        agg = speakers.setdefault(spk, {"spk_id": spk, "duration": 0.0, "nbr_seg": 0})
        agg["duration"] = round(agg["duration"] + (s["e"] - s["s"]), 2)
        agg["nbr_seg"] += 1
    segments = [{"seg_id": i, "spk_id": _spk(s["n"]),
                 "seg_begin": round(s["s"], 2), "seg_end": round(s["e"], 2)}
                for i, s in enumerate(segs, 1)]
    return json.dumps({"speakers": list(speakers.values()), "segments": segments},
                      ensure_ascii=False, indent=2)

def _r_tsa_json(segs, meeting):
    # ASR + speaker + timestamps.
    result = "\n".join(f"{_spk(s['n'])}: {s['text']}" for s in segs if s["text"])
    segments = [{"segment": s["text"],
                 "start": round(s["s"], 2), "end": round(s["e"], 2),
                 "duration": round(s["e"] - s["s"], 2), "spk_id": _spk(s["n"])}
                for s in segs if s["text"]]
    return json.dumps({"transcription_result": result, "segments": segments},
                      ensure_ascii=False, indent=2)


# Whole-word label-casing styles. French (Locuteur/LOCUTEUR/locuteur) back the
# French-only formats in `_LANG_EXTRA_FORMATS["fr"]`; English SPEAKER/speaker are
# base formats (distinct from the zero-padded 'SPEAKER_00' style).
def _r_asr_locuteur_colon(segs, meeting):
    return "\n".join(f"Locuteur {s['n']}: {s['text']}" for s in segs if s["text"])

def _r_asr_locuteur_dash(segs, meeting):
    return "\n".join(f"Locuteur {s['n']} - {s['text']}" for s in segs if s["text"])

def _r_asr_locuteur_caps(segs, meeting):
    return "\n".join(f"LOCUTEUR {s['n']}: {s['text']}" for s in segs if s["text"])

def _r_asr_locuteur_lower(segs, meeting):
    return "\n".join(f"locuteur {s['n']}: {s['text']}" for s in segs if s["text"])

def _r_asr_speaker_caps(segs, meeting):
    return "\n".join(f"SPEAKER {s['n']}: {s['text']}" for s in segs if s["text"])

def _r_asr_speaker_lower(segs, meeting):
    return "\n".join(f"speaker {s['n']}: {s['text']}" for s in segs if s["text"])

def _r_ts_locuteur_bracket(segs, meeting):
    return "\n".join(f"[{s['s']:.2f} - {s['e']:.2f}] Locuteur {s['n']}" for s in segs)

def _r_ts_locuteur_caps(segs, meeting):
    return "\n".join(f"[{s['s']:.2f} - {s['e']:.2f}] LOCUTEUR {s['n']}" for s in segs)

def _r_ts_locuteur_lower(segs, meeting):
    return "\n".join(f"[{s['s']:.2f} - {s['e']:.2f}] locuteur {s['n']}" for s in segs)

def _r_ts_speaker_caps(segs, meeting):
    return "\n".join(f"[{s['s']:.2f} - {s['e']:.2f}] SPEAKER {s['n']}" for s in segs)

def _r_ts_speaker_lower(segs, meeting):
    return "\n".join(f"[{s['s']:.2f} - {s['e']:.2f}] speaker {s['n']}" for s in segs)

def _r_tsa_locuteur_bracket_colon(segs, meeting):
    return "\n".join(f"[{s['s']:.2f}-{s['e']:.2f}] Locuteur {s['n']}: {s['text']}"
                     for s in segs if s["text"])

def _r_tsa_locuteur_inline(segs, meeting):
    return "\n".join(f"Locuteur {s['n']} ({s['s']:.2f}-{s['e']:.2f}): {s['text']}"
                     for s in segs if s["text"])

def _r_tsa_locuteur_caps(segs, meeting):
    return "\n".join(f"[{s['s']:.2f}-{s['e']:.2f}] LOCUTEUR {s['n']}: {s['text']}"
                     for s in segs if s["text"])

def _r_tsa_locuteur_lower(segs, meeting):
    return "\n".join(f"[{s['s']:.2f}-{s['e']:.2f}] locuteur {s['n']}: {s['text']}"
                     for s in segs if s["text"])

def _r_tsa_speaker_caps(segs, meeting):
    return "\n".join(f"[{s['s']:.2f}-{s['e']:.2f}] SPEAKER {s['n']}: {s['text']}"
                     for s in segs if s["text"])

def _r_tsa_speaker_lower(segs, meeting):
    return "\n".join(f"[{s['s']:.2f}-{s['e']:.2f}] speaker {s['n']}: {s['text']}"
                     for s in segs if s["text"])


# --------------------------------------------------------------------------- #
# Output formats: language-independent. Maps variant -> format key -> renderer. #
# The first format key of each variant is its default (used by generic prompts).#
# --------------------------------------------------------------------------- #
_DIAR_FORMATS = {
    "asr": {
        "speaker_colon": {"render": _r_asr_speaker_colon},
        "bare_letter":   {"render": _r_asr_bare_letter},
        "s_num":         {"render": _r_asr_s_num},
        "speaker_upper": {"render": _r_asr_speaker_upper},
        "dash":          {"render": _r_asr_dash},
        "letter":        {"render": _r_asr_letter},
        "speaker_caps":  {"render": _r_asr_speaker_caps},
        "speaker_lower": {"render": _r_asr_speaker_lower},
        "json":          {"render": _r_asr_json},
    },
    "timestamps": {
        "plain":         {"render": _r_ts_plain},
        "rttm":          {"render": _r_ts_rttm},
        "bracket":       {"render": _r_ts_bracket},
        "bare_letter":   {"render": _r_ts_bare_letter},
        "s_num":         {"render": _r_ts_s_num},
        "speaker_upper": {"render": _r_ts_speaker_upper},
        "speaker_caps":  {"render": _r_ts_speaker_caps},
        "speaker_lower": {"render": _r_ts_speaker_lower},
        "json":          {"render": _r_ts_json},
    },
    "timestamps_asr": {
        "bracket_colon": {"render": _r_tsa_bracket_colon},
        "bare_letter":   {"render": _r_tsa_bare_letter},
        "s_num":         {"render": _r_tsa_s_num},
        "speaker_upper": {"render": _r_tsa_speaker_upper},
        "inline":        {"render": _r_tsa_inline},
        "arrow":         {"render": _r_tsa_arrow},
        "srt":           {"render": _r_tsa_srt},
        "vtt":           {"render": _r_tsa_vtt},
        "speaker_caps":  {"render": _r_tsa_speaker_caps},
        "speaker_lower": {"render": _r_tsa_speaker_lower},
        "json":          {"render": _r_tsa_json},
    },
}

# First key of each variant is its default format (used by generic prompts).
DIAR_DEFAULT_FORMAT = {v: next(iter(fmts)) for v, fmts in _DIAR_FORMATS.items()}

# JSON is a structured, schema-heavy format; keep it a small slice of all rows.
JSON_FORMAT_RATIO = 0.05


def choose_format(variant, formats, rng, format_variety=True,
                  generic_ratio=0.4, json_ratio=JSON_FORMAT_RATIO):
    """Pick (format_key, prompt_style) for one diarization row.

    - Without `format_variety`: always the variant's default format + a generic
      (no-format) prompt.
    - 'json' is emitted for a fixed small fraction (`json_ratio`) of rows, always
      paired with an explicit, format-specifying prompt. It is never the default
      format and is never drawn by the generic/uniform path, so its overall share
      is exactly `json_ratio`.
    - Otherwise: with probability `generic_ratio` a generic prompt on the default
      format, else an explicit prompt on a uniformly-random non-json format.

    `rng` is a `random.Random`; `formats` is the variant's format dict (typically
    `_DIAR_FORMATS[variant]` or `formats_for(language, variant)`)."""
    if not format_variety:
        return DIAR_DEFAULT_FORMAT[variant], "generic"
    if "json" in formats and rng.random() < json_ratio:
        return "json", "explicit"
    if rng.random() < generic_ratio:
        return DIAR_DEFAULT_FORMAT[variant], "generic"
    non_json = [k for k in formats if k != "json"]
    return rng.choice(non_json), "explicit"


# --------------------------------------------------------------------------- #
# Language-specific EXTRA formats, merged on top of `_DIAR_FORMATS` only for    #
# rows in that language (see `formats_for`). These are NOT in the base table,   #
# so they never appear in other languages' output and are not subject to the    #
# English-completeness check; instead `_validate_prompts` checks each extra     #
# format has prompts in its own language. Format keys must not collide with the #
# base ones. The default format (generic prompts) stays the base default.       #
# --------------------------------------------------------------------------- #
_LANG_EXTRA_FORMATS = {
    "fr": {
        "asr": {
            "locuteur_colon": {"render": _r_asr_locuteur_colon},
            "locuteur_dash":  {"render": _r_asr_locuteur_dash},
            "locuteur_caps":  {"render": _r_asr_locuteur_caps},
            "locuteur_lower": {"render": _r_asr_locuteur_lower},
        },
        "timestamps": {
            "locuteur_bracket": {"render": _r_ts_locuteur_bracket},
            "locuteur_caps":    {"render": _r_ts_locuteur_caps},
            "locuteur_lower":   {"render": _r_ts_locuteur_lower},
        },
        "timestamps_asr": {
            "locuteur_bracket_colon": {"render": _r_tsa_locuteur_bracket_colon},
            "locuteur_inline":        {"render": _r_tsa_locuteur_inline},
            "locuteur_caps":          {"render": _r_tsa_locuteur_caps},
            "locuteur_lower":         {"render": _r_tsa_locuteur_lower},
        },
    },
}


def formats_for(language, variant):
    """Return the format dict for (language, variant): the base formats plus any
    language-specific extras. Use this instead of `_DIAR_FORMATS[variant]` when
    picking/rendering a format so French rows can also draw 'Locuteur' formats."""
    base = dict(_DIAR_FORMATS[variant])
    extra = _LANG_EXTRA_FORMATS.get(language, {}).get(variant, {})
    base.update(extra)
    return base


# --------------------------------------------------------------------------- #
# Prompts: keyed by language, stored in data/contexts/diarization_prompts.json  #
# so they can be edited (or translated) without touching this module. Add a new #
# language by adding a top-level entry mirroring that structure (see the module #
# docstring). Missing prompt lists fall back to DEFAULT_LANGUAGE.               #
#                                                                               #
# json.load preserves key and list order, which PROMPT_SEED-based sampling      #
# depends on: reordering the file changes which prompt a given row gets.        #
# --------------------------------------------------------------------------- #
DIAR_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "contexts" / "diarization_prompts.json"

with open(DIAR_PROMPTS_PATH, encoding="utf-8") as _f:
    _DIAR_PROMPTS = json.load(_f)


def _validate_prompts():
    """At import time, ensure the fallback language is complete (every variant and
    every format has a non-empty prompt list). Catches typos when editing prompts."""
    base = _DIAR_PROMPTS[DEFAULT_LANGUAGE]
    for variant, fmts in _DIAR_FORMATS.items():
        assert base["generic"].get(variant), f"missing generic prompts for {variant!r}"
        for fmt in fmts:
            assert base["formats"].get(variant, {}).get(fmt), \
                f"missing {DEFAULT_LANGUAGE!r} prompts for {variant!r}/{fmt!r}"
    for lang, block in _DIAR_PROMPTS.items():
        suffixes = block.get("backchannel_suffixes")
        assert isinstance(suffixes, dict), f"{lang!r} backchannel_suffixes must be a dict"
        for key in ("transcribed", "timestamps"):
            assert suffixes.get(key), f"{lang!r} missing {key!r} backchannel suffixes"

    # Each language's EXTRA formats must have prompts in that same language (they
    # have no English fallback, being language-native), and must not shadow a base
    # format key.
    for lang, by_variant in _LANG_EXTRA_FORMATS.items():
        lang_block = _DIAR_PROMPTS.get(lang, {})
        for variant, fmts in by_variant.items():
            for fmt in fmts:
                assert fmt not in _DIAR_FORMATS[variant], \
                    f"extra format {lang!r}/{variant!r}/{fmt!r} collides with a base format"
                assert lang_block.get("formats", {}).get(variant, {}).get(fmt), \
                    f"missing {lang!r} prompts for extra format {variant!r}/{fmt!r}"


_validate_prompts()


def _prompts_for(language, variant, fmt, style):
    """Prompt list for (language, variant, format, style), falling back to
    DEFAULT_LANGUAGE for any list the requested language does not define."""
    block = _DIAR_PROMPTS.get(language) or _DIAR_PROMPTS[DEFAULT_LANGUAGE]
    if style == "generic":
        prompts = block.get("generic", {}).get(variant)
    else:
        prompts = block.get("formats", {}).get(variant, {}).get(fmt)
    if not prompts and language != DEFAULT_LANGUAGE:
        return _prompts_for(DEFAULT_LANGUAGE, variant, fmt, style)
    return prompts


def _backchannel_suffixes_for(language, variant):
    """Backchannel-version suffix list for a language/variant. The 'timestamps'
    variant emits no words, so it uses segment-oriented wording; 'asr' and
    'timestamps_asr' use transcription-oriented wording."""
    block = _DIAR_PROMPTS.get(language) or _DIAR_PROMPTS[DEFAULT_LANGUAGE]
    suffixes = block.get("backchannel_suffixes") or _DIAR_PROMPTS[DEFAULT_LANGUAGE]["backchannel_suffixes"]
    key = "timestamps" if variant == "timestamps" else "transcribed"
    return suffixes.get(key) or _DIAR_PROMPTS[DEFAULT_LANGUAGE]["backchannel_suffixes"][key]


# --------------------------- backchannel detection --------------------------- #
# Backchannels / acknowledgements / fillers. A turn whose every lexical token is
# one of these is treated as a backchannel and dropped from the "clean" version.
# Keyed by language; the language-specific set is unioned with the universal
# vocalic core (mm/hmm/ah/oh/...). English is the default.
_BACKCHANNEL_CORE = {
    "mm", "mmm", "mhm", "mm-hmm", "mmhmm", "mm-hm", "hmm", "hm", "hmmm",
    "ah", "aha", "oh", "ooh", "huh", "mm-mm", "hm-mm", "mm-mmm", "mh",
}
_BACKCHANNEL_WORDS_BY_LANG = {
    "en": {
        "uh-huh", "uhhuh", "uh", "uhh", "uhm", "um", "umm", "er", "erm",
        "yeah", "yep", "yup", "nah", "kay", "okay", "ok", "right", "sure",
    },
    "fr": {
        # acquiescements / régulateurs / hésitations courants à l'oral
        "ouais", "oui", "ouaip", "non", "nan", "voilà", "voila",
        "d'accord", "daccord", "ok", "okay", "hein", "bah", "ben", "beh",
        "euh", "heu", "hum", "humhum", "mhmh", "mmh", "mouais", "ouf",
        "ah-ouais", "ah-oui", "ah-bon", "bon-ben",
    },
}
_BC_EDGE = re.compile(r"^[^\w']+|[^\w']+$")


def _backchannel_words_for(language):
    lang = language if language in _BACKCHANNEL_WORDS_BY_LANG else DEFAULT_LANGUAGE
    return _BACKCHANNEL_CORE | _BACKCHANNEL_WORDS_BY_LANG[lang]


def _is_backchannel(text: str, language: str = DEFAULT_LANGUAGE) -> bool:
    """True if every lexical token in `text` is a backchannel/acknowledgement/filler
    in the given language (falling back to the default language's set)."""
    words = _backchannel_words_for(language)
    toks = text.split()
    if not toks or len(toks) > 4:
        return False
    has_word = False
    for t in toks:
        w = _BC_EDGE.sub("", t).lower().strip("'")
        if not w:  # pure punctuation token
            continue
        has_word = True
        if w not in words:
            return False
    return has_word


# Non-lexical hesitation/filler sounds. Unlike the acknowledgement words above
# (oui/non/voilà/...), these carry no meaning and are safe to remove *anywhere* in
# a turn — so the "clean" (no-backchannel) version strips them in-line, while the
# "full" version keeps them. Lexical acknowledgements are only dropped when they
# form a whole turn (see _is_backchannel), never mid-sentence.
_FILLER_WORDS = {
    # English
    "uh", "uhh", "uhm", "um", "umm", "er", "erm", "uh-huh", "uhhuh",
    # French
    "euh", "heu", "heum", "euhm", "heuh", "ben-euh",
    # cross-lingual vocalic hesitations
    "hum", "humhum", "hmm", "hmmm", "hm", "mh", "mm", "mmm", "mmh", "mhm", "mhmh",
}


def strip_fillers(text: str, language: str = DEFAULT_LANGUAGE) -> str:
    """Remove in-line filler/hesitation tokens (euh, uh, hum, mh, ...) from `text`,
    leaving real words — including lexical acknowledgements like oui/non/voilà —
    untouched. Collapses the whitespace left behind."""
    out = []
    for t in text.split():
        w = _BC_EDGE.sub("", t).lower().strip("'")
        if w in _FILLER_WORDS:
            continue
        out.append(t)
    return " ".join(out)


def clean_window_pieces(window, language: str = DEFAULT_LANGUAGE):
    """Build the 'clean' (no-backchannel) view of a window: drop turns that are
    entirely backchannels/fillers, and strip in-line fillers from the rest. Returns
    new piece dicts (the originals, used by the verbatim 'full' view, are
    untouched). Turns that become empty after filler removal are dropped."""
    out = []
    for p in window:
        text = p.get("text", "")
        if _is_backchannel(text, language):
            continue
        stripped = strip_fillers(text, language)
        if not stripped.strip():
            continue
        out.append({**p, "text": stripped})
    return out


def _defines_prompt(language, variant, fmt, style) -> bool:
    """True if `language` defines a prompt for (variant, fmt, style) itself, without
    falling back to DEFAULT_LANGUAGE."""
    block = _DIAR_PROMPTS.get(language)
    if not block:
        return False
    if style == "generic":
        return bool(block.get("generic", {}).get(variant))
    return bool(block.get("formats", {}).get(variant, {}).get(fmt))


def make_diar_lean_row(r: NemoDatasetRow, cross_lingual_ratio: float = 0.0) -> NemoDatasetRow:
    """Lean training row: prepend a prompt matching the row's prompt_style/format,
    in the row's language (falling back to DEFAULT_LANGUAGE), and — for the
    backchannel-included version — asking to keep them; drop metadata.

    With probability `cross_lingual_ratio` the prompt is instead drawn from a
    different language (prompt-language augmentation: a French prompt on English
    audio and vice versa). Only languages that define a prompt for the row's exact
    (variant, format, style) are eligible — so French-only formats (e.g. the
    'Locuteur' styles) never get an English prompt. The row's `language` field is
    left unchanged; only the instruction wording differs."""
    md = r.custom_metadata or {}
    variant, fmt, style = md.get("variant"), md.get("format"), md.get("prompt_style")
    language = r.language or DEFAULT_LANGUAGE

    # Deterministic per-row rng so the prompt wording (language draw, phrasing,
    # backchannel suffix) is reproducible across regenerations, like the structural
    # format choice in `choose_format`. Seeded from the row id plus the
    # variant/format/style so sibling rows sharing an id still draw independently.
    rng = random.Random(f"{PROMPT_SEED}.{r.id}.{variant}.{fmt}.{style}.prompt")

    prompt_language = language
    if cross_lingual_ratio and rng.random() < cross_lingual_ratio:
        others = [lang for lang in _DIAR_PROMPTS
                  if lang != language and _defines_prompt(lang, variant, fmt, style)]
        if others:
            prompt_language = rng.choice(others)

    turns = list(r.turns)
    prompts = _prompts_for(prompt_language, variant, fmt, style)
    if prompts:
        prompt = rng.choice(prompts)
        if md.get("backchannels"):
            prompt = prompt + " " + rng.choice(_backchannel_suffixes_for(prompt_language, variant))
        turns = [NemoTurn(role="User", value=prompt, turn_type="text")] + turns
    return NemoDatasetRow(
        id=r.id, dataset_name=r.dataset_name,
        split=r.split, language=r.language, turns=turns,
    )
