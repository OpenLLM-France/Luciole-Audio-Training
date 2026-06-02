"""Generate per-word timestamped transcription training data.

Reads SLU jsonl files (same shape as generate_slu_variants.py expects) and emits
one (question, audio, answer) triple per record where the answer is the full
audio transcription with each word annotated with its start timestamp.

Three prompt/answer modes, chosen per record with weighted randomness:

    * JSON mode — the prompt asks for JSON, the answer is a JSON array of
      {"word": ..., "time": ...} entries.
    * Format-specified mode — the prompt shows an example of a specific inline
      format; the answer follows that exact format.
    * Default mode — the prompt doesn't specify a format; the answer uses a
      single canonical default format (`word (time)`).

Unlike generate_slu_variants.py, this task does not depend on `answer_spans` —
every record with `custom_metadata.word2time` and an audio turn produces one
variant.

Prompts come in English (default) or French (--language fr); the spoken content
and the timestamped answer are unchanged either way, so the same audio yields a
cross-lingual instruction pair.

Run:
    python data/synthetic/generate_timestamped_transcription.py \\
        /path/to/dir_with_train_dev_test --output_dir out --language fr
"""

import argparse
import json
import random
from pathlib import Path


# ---------- Format specs ----------------------------------------------------

# Inline format renderers take (word, start, end). Start-only formats ignore
# `end` (prefixed `_e`); range formats use both.
def _fmt_paren(w, s, _e):        return f"{w} ({s:.1f}s)"
def _fmt_bracket(w, s, _e):      return f"{w} [{s:.1f}]"
def _fmt_line(w, s, _e):         return f"{s:.1f}s: {w}"
def _fmt_paren_range(w, s, e):   return f"{w} ({s:.1f}s-{e:.1f}s)"
def _fmt_bracket_range(w, s, e): return f"{w} [{s:.1f}-{e:.1f}]"
def _fmt_line_range(w, s, e):    return f"{s:.1f}s-{e:.1f}s: {w}"
def _fmt_t_range(w, s, e):       return f"<t>{s:.1f}s-{e:.1f}</t> {w}"

# `example` is language-keyed: only the demo words differ ("Hello world" /
# "Bonjour le monde") — the timestamp format itself is identical across langs.
FORMAT_SPECS = {
    # Word-level formats
    "paren":              {"level": "word",     "render": _fmt_paren,         "joiner": " ",  "example": {"en": "Hello (0.0s) world (0.5s)",                      "fr": "Bonjour (0.0s) monde (0.5s)"}},
    "line":               {"level": "word",     "render": _fmt_line,          "joiner": "\n", "example": {"en": "0.0s: Hello\n0.5s: world",                       "fr": "0.0s: Bonjour\n0.5s: monde"}},
    "bracket_range":      {"level": "word",     "render": _fmt_bracket_range, "joiner": " ",  "example": {"en": "Hello [0.0-0.5] world [0.5-1.0]",                "fr": "Bonjour [0.0-0.5] monde [0.5-1.0]"}},
    "line_range":         {"level": "word",     "render": _fmt_line_range,    "joiner": "\n", "example": {"en": "0.0s-0.5s: Hello\n0.5s-1.0s: world",            "fr": "0.0s-0.5s: Bonjour\n0.5s-1.0s: monde"}},
    "t_range":            {"level": "word",     "render": _fmt_t_range,       "joiner": "\n", "example": {"en": "<t>0.0s-0.5</t>: Hello\n<t>0.5s-1.0</t>: world", "fr": "<t>0.0s-0.5</t>: Bonjour\n<t>0.5s-1.0</t>: monde"}},
    # Sentence-level formats (same renderers; the "token" passed in is a full sentence)
    "bracket_sent":       {"level": "sentence", "render": _fmt_bracket,       "joiner": " ",  "example": {"en": "Hello world! [0.0] How are you? [0.5]",          "fr": "Bonjour le monde ! [0.0] Comment ça va ? [0.5]"}},
    "line_sent":          {"level": "sentence", "render": _fmt_line,          "joiner": "\n", "example": {"en": "0.0s: Hello world!\n0.5s: How are you?",        "fr": "0.0s: Bonjour le monde !\n0.5s: Comment ça va ?"}},
    "paren_range_sent":   {"level": "sentence", "render": _fmt_paren_range,   "joiner": " ",  "example": {"en": "Hello world! (0.0s-0.5s) How are you? (0.5s-2.0s)", "fr": "Bonjour le monde ! (0.0s-0.5s) Comment ça va ? (0.5s-2.0s)"}},
    "line_range_sent":    {"level": "sentence", "render": _fmt_line_range,    "joiner": "\n", "example": {"en": "0.0s-0.5s: Hello world!\n0.5s-2.0s: How are you?", "fr": "0.0s-0.5s: Bonjour le monde !\n0.5s-2.0s: Comment ça va ?"}},
    "t_range_sent":       {"level": "sentence", "render": _fmt_t_range,       "joiner": "\n", "example": {"en": "<t>0.0s-0.5</t>: Hello world!\n<t>0.5s-2.0</t>: How are you?", "fr": "<t>0.0s-0.5</t>: Bonjour le monde !\n<t>0.5s-2.0</t>: Comment ça va ?"}},
}

# Default format used when the prompt doesn't specify one. Kept fixed so the
# model learns a canonical behavior for unconstrained prompts.
DEFAULT_FORMAT = "paren"
NON_DEFAULT_FORMATS = [f for f in FORMAT_SPECS if f != DEFAULT_FORMAT]


# JSON shapes. Each has an `example` string shown to the model in schema-
# specified prompts, and a renderer that produces the actual JSON answer.
# All renderers accept `pairs = [(word, start, end), ...]`.
def _render_list_of_objects(pairs):
    return json.dumps(
        [{"word": w, "time": round(s, 1)} for w, s, _e in pairs],
        ensure_ascii=False,
    )


def _render_list_of_objects_startend(pairs):
    return json.dumps(
        [{"word": w, "start": round(s, 1), "end": round(e, 1)} for w, s, e in pairs],
        ensure_ascii=False,
    )


def _render_list_of_objects_timestamp(pairs):
    return json.dumps(
        [{"word": w, "timestamp": round(s, 1)} for w, s, _e in pairs],
        ensure_ascii=False,
    )


def _render_list_of_sentences(sentences):
    return json.dumps(
        [
            {"sentence": text, "start": round(st, 1), "end": round(en, 1)}
            for text, st, en in sentences
        ],
        ensure_ascii=False,
    )


# `example` is language-keyed (demo values only); JSON field names are part of
# the output schema and stay identical across languages.
JSON_SCHEMAS = {
    "list_of_objects": {
        "level": "word",
        "render": _render_list_of_objects,
        "example": {
            "en": '[{"word": "Hello", "time": 0.0}, {"word": "world", "time": 0.5}]',
            "fr": '[{"word": "Bonjour", "time": 0.0}, {"word": "monde", "time": 0.5}]',
        },
    },
    "list_of_objects_startend": {
        "level": "word",
        "render": _render_list_of_objects_startend,
        "example": {
            "en": '[{"word": "Hello", "start": 0.0, "end": 0.5}, {"word": "world", "start": 0.5, "end": 1.0}]',
            "fr": '[{"word": "Bonjour", "start": 0.0, "end": 0.5}, {"word": "monde", "start": 0.5, "end": 1.0}]',
        },
    },
    "list_of_objects_timestamp": {
        "level": "word",
        "render": _render_list_of_objects_timestamp,
        "example": {
            "en": '[{"word": "Hello", "timestamp": 0.0}, {"word": "world", "timestamp": 0.5}]',
            "fr": '[{"word": "Bonjour", "timestamp": 0.0}, {"word": "monde", "timestamp": 0.5}]',
        },
    },
    "sentence_level": {
        "level": "sentence",
        "render": _render_list_of_sentences,
        "example": {
            "en": '[{"sentence": "Hello world!", "start": 0.0, "end": 0.5}, {"sentence": "How are you?", "start": 0.5, "end": 2.0}]',
            "fr": '[{"sentence": "Bonjour le monde !", "start": 0.0, "end": 0.5}, {"sentence": "Comment ça va ?", "start": 0.5, "end": 2.0}]',
        },
    },
}

DEFAULT_JSON_SCHEMA = "list_of_objects"


# ---------- Prompts ---------------------------------------------------------

# Prompt languages supported. English prompts over French (or any) audio are a
# deliberate cross-lingual instruction-following setup; pass --language fr to
# emit French-language prompts instead (the spoken content is unchanged).
LANGUAGES = ("en", "fr")

# Prompts used with DEFAULT_FORMAT. Entries containing `{example}` get the
# canonical-format example substituted in at render time. Keyed by language.
DEFAULT_PROMPTS = {
    "en": [
        "Transcribe the audio and include a timestamp for each word.",
        "Give me a word-level transcription of the audio with timestamps, for instance `{example}`.",
        "Transcribe the recording word by word, marking when each word starts.",
        "Provide a timestamped transcription of the audio, e.g. `{example}`.",
        "Write out the audio transcript and attach a timestamp to every word.",
        "I need a transcription annotated with per-word start times - something like {example}.",
        "Transcribe the clip and tell me when each word is pronounced.",
        "Produce a word-by-word transcript, with the second at which each word begins.",
        "Give me the transcript of the audio with per-word timestamps, for example {example}.",
        "Transcribe the audio with the time at which each word is spoken.",
    ],
    "fr": [
        "Transcris l'audio en incluant un horodatage pour chaque mot.",
        "Donne-moi une transcription mot à mot de l'audio avec des horodatages, par exemple `{example}`.",
        "Transcris l'enregistrement mot par mot en indiquant le début de chaque mot.",
        "Fournis une transcription horodatée de l'audio, par exemple `{example}`.",
        "Écris la transcription de l'audio et associe un horodatage à chaque mot.",
        "J'ai besoin d'une transcription annotée avec l'instant de début de chaque mot — quelque chose comme {example}.",
        "Transcris le clip et indique-moi quand chaque mot est prononcé.",
        "Produis une transcription mot par mot, avec la seconde à laquelle chaque mot commence.",
        "Donne-moi la transcription de l'audio avec un horodatage par mot, par exemple {example}.",
        "Transcris l'audio avec l'instant auquel chaque mot est prononcé.",
    ],
}

# Prompts that pin a specific format. {example} is filled from the chosen spec.
FORMAT_PROMPTS = {
    "en": [
        "Transcribe the audio with per-word timestamps. Use this format: `{example}`.",
        "Give me a timestamped word-level transcription following the pattern: `{example}`.",
        "Transcribe the audio. Format each word like this: {example}.",
        "Produce a word-level transcript using the format `{example}`.",
        "I need a per-word timestamped transcription written as: {example}.",
        "Transcribe the audio with timestamps. Follow this example: {example}.",
        "Write out a timestamped transcription using this format: {example}.",
        "Transcribe the recording; each word should appear as `{example}`.",
    ],
    "fr": [
        "Transcris l'audio avec un horodatage par mot. Utilise ce format : `{example}`.",
        "Donne-moi une transcription horodatée au niveau du mot suivant le modèle : `{example}`.",
        "Transcris l'audio. Formate chaque mot ainsi : {example}.",
        "Produis une transcription au niveau du mot en utilisant le format `{example}`.",
        "J'ai besoin d'une transcription horodatée par mot écrite ainsi : {example}.",
        "Transcris l'audio avec des horodatages. Suis cet exemple : {example}.",
        "Écris une transcription horodatée en utilisant ce format : {example}.",
        "Transcris l'enregistrement ; chaque mot doit apparaître comme `{example}`.",
    ],
}

# Sentence-level counterpart. Used whenever a sentence-level format is picked,
# so the question matches the granularity of the answer.
FORMAT_SENTENCE_PROMPTS = {
    "en": [
        "Transcribe the audio with per-sentence timestamps. Use this format: `{example}`.",
        "Give me a sentence-level timestamped transcription following the pattern: `{example}`.",
        "Transcribe the audio. Format each sentence like this: `{example}`.",
        "Produce a sentence-level transcript using the format `{example}`.",
        "I need a per-sentence timestamped transcription written as: `{example}`.",
        "Transcribe the audio with sentence-level timestamps. Follow this example: `{example}`.",
        "Write out a timestamped transcription split by sentence, using this format: `{example}`.",
        "Transcribe the recording; each sentence should appear as `{example}`.",
    ],
    "fr": [
        "Transcris l'audio avec un horodatage par phrase. Utilise ce format : `{example}`.",
        "Donne-moi une transcription horodatée au niveau de la phrase suivant le modèle : `{example}`.",
        "Transcris l'audio. Formate chaque phrase ainsi : `{example}`.",
        "Produis une transcription au niveau de la phrase en utilisant le format `{example}`.",
        "J'ai besoin d'une transcription horodatée par phrase écrite ainsi : `{example}`.",
        "Transcris l'audio avec des horodatages par phrase. Suis cet exemple : `{example}`.",
        "Écris une transcription horodatée découpée par phrase, en utilisant ce format : `{example}`.",
        "Transcris l'enregistrement ; chaque phrase doit apparaître comme `{example}`.",
    ],
}

# JSON prompts used with DEFAULT_JSON_SCHEMA. Entries containing `{example}`
# get the canonical-schema example substituted in at render time.
JSON_DEFAULT_PROMPTS = {
    "en": [
        "Transcribe the audio with per-word timestamps and return the result as JSON.",
        "Give me a timestamped word-level transcription in JSON format, for instance `{example}`.",
        "Transcribe the audio and return the transcription as JSON.",
        "Transcribe the recording and format the output as JSON, e.g. `{example}`.",
        "Give me the transcript as JSON, with a timestamp attached to every word.",
        "Transcribe the clip and output the result as JSON — something like `{example}`.",
        "Produce a JSON transcription of the audio with per-word start times.",
        "Transcribe the audio with timestamps; format your answer as JSON.",
    ],
    "fr": [
        "Transcris l'audio avec un horodatage par mot et renvoie le résultat en JSON.",
        "Donne-moi une transcription horodatée au niveau du mot au format JSON, par exemple `{example}`.",
        "Transcris l'audio et renvoie la transcription en JSON.",
        "Transcris l'enregistrement et formate la sortie en JSON, par exemple `{example}`.",
        "Donne-moi la transcription en JSON, avec un horodatage associé à chaque mot.",
        "Transcris le clip et renvoie le résultat en JSON — quelque chose comme `{example}`.",
        "Produis une transcription JSON de l'audio avec l'instant de début de chaque mot.",
        "Transcris l'audio avec des horodatages ; formate ta réponse en JSON.",
    ],
}

# JSON prompts that pin a specific shape. {example} is filled from the schema.
JSON_SCHEMA_PROMPTS = {
    "en": [
        "Transcribe the audio with per-word timestamps. Return JSON in this shape: `{example}`.",
        "Give me the transcription as JSON following this exact schema: `{example}`.",
        "Transcribe the audio and output JSON like: `{example}`.",
        "Produce a JSON transcription matching this structure: `{example}`.",
        "Output a JSON transcription; the result should look like: `{example}`.",
        "Transcribe the recording as JSON in the following format: `{example}`.",
        "Return the transcript as JSON, using this pattern: `{example}`.",
    ],
    "fr": [
        "Transcris l'audio avec un horodatage par mot. Renvoie du JSON selon cette forme : `{example}`.",
        "Donne-moi la transcription en JSON suivant exactement ce schéma : `{example}`.",
        "Transcris l'audio et produis du JSON comme : `{example}`.",
        "Produis une transcription JSON correspondant à cette structure : `{example}`.",
        "Génère une transcription JSON ; le résultat doit ressembler à : `{example}`.",
        "Transcris l'enregistrement en JSON au format suivant : `{example}`.",
        "Renvoie la transcription en JSON, en utilisant ce modèle : `{example}`.",
    ],
}

# Sentence-level JSON prompts. Used when a sentence-level schema is picked, so
# the question text matches the granularity of the answer.
JSON_SENTENCE_PROMPTS = {
    "en": [
        "Transcribe the audio with per-sentence timestamps. Return JSON in this shape: `{example}`.",
        "Give me a sentence-level transcription as JSON following this schema: `{example}`.",
        "Transcribe the audio and output one JSON entry per sentence, like: `{example}`.",
        "Produce a JSON transcription split by sentence matching this structure: `{example}`.",
        "Transcribe the recording as a JSON list of sentences with start/end times: `{example}`.",
        "Return the transcript as JSON, grouped into sentences: `{example}`.",
    ],
    "fr": [
        "Transcris l'audio avec un horodatage par phrase. Renvoie du JSON selon cette forme : `{example}`.",
        "Donne-moi une transcription au niveau de la phrase en JSON suivant ce schéma : `{example}`.",
        "Transcris l'audio et produis une entrée JSON par phrase, comme : `{example}`.",
        "Produis une transcription JSON découpée par phrase correspondant à cette structure : `{example}`.",
        "Transcris l'enregistrement sous forme de liste JSON de phrases avec des temps de début/fin : `{example}`.",
        "Renvoie la transcription en JSON, regroupée par phrases : `{example}`.",
    ],
}

# Mix presets: each maps the four modes to a weight. `no_json` excludes JSON
# modes entirely, `full_json` emits only JSON, `mixed` is 15% JSON overall.
# Weights per preset must sum to > 0; modes with weight 0 are never chosen.
MIX_PRESETS = {
    "no_json": {
        "default":          0.55,
        "format_specified": 0.45,
        "json_default":     0.0,
        "json_specified":   0.0,
    },
    "mixed": {
        "default":          0.50,
        "format_specified": 0.35,
        "json_default":     0.05,
        "json_specified":   0.10,
    },
    "full_json": {
        "default":          0.0,
        "format_specified": 0.0,
        "json_default":     0.40,
        "json_specified":   0.60,
    },
}


# ---------- Rendering ------------------------------------------------------

def _clean_pairs(w2t):
    """(word, start_second, end_second) triples. Pure-punctuation tokens (and
    tokens with invalid negative timestamps) are attached to the preceding
    alnum word so the rendered transcript keeps its punctuation, e.g.
    ["Congress", "."] -> [("Congress.", ...)]."""
    ends = w2t.get("end_second") or w2t["start_second"]
    out = []
    for w, s, e in zip(w2t["word"], w2t["start_second"], ends):
        w_stripped = (w or "").strip()
        if not w_stripped:
            continue
        has_alnum = any(c.isalnum() for c in w_stripped)
        if has_alnum and s >= 0:
            out.append([w_stripped, s, e])
        elif not has_alnum and out:
            out[-1][0] = out[-1][0] + w_stripped
    return [tuple(x) for x in out]


# Fallback sentence-segmentation thresholds, used when terminal punctuation
# (`.`, `?`, `!`) is missing or sparse in the source `word2time["word"]` list
# (common for Wikipedia-derived transcripts). A gap of >= PAUSE_GAP_S between
# two consecutive words is treated as a soft sentence boundary, but only once
# the current sentence already has MIN_WORDS_BEFORE_PAUSE_SPLIT alnum words
# (so we don't break off a single "The" when there's a hesitation). Sentences
# are also force-split once they grow past MAX_WORDS_PER_SENT alnum tokens or
# MAX_SENT_DURATION_S seconds, so we never emit a single 30s "sentence".
PAUSE_GAP_S = 1.0
MIN_WORDS_BEFORE_PAUSE_SPLIT = 5
MAX_WORDS_PER_SENT = 35
MAX_SENT_DURATION_S = 18.0


def _build_sentences(w2t):
    """Aggregate word-level timings into (sentence_text, start, end) triples.

    Primary boundary cue: tokens whose last char is `.`, `!`, or `?` — whether
    attached to a word (`world.`) or a standalone punctuation token. When the
    transcript lacks such markers, we fall back to splitting on long pauses
    between words and on max-length caps (see PAUSE_GAP_S /
    MAX_WORDS_PER_SENT / MAX_SENT_DURATION_S).
    """
    words = w2t.get("word", [])
    starts = w2t.get("start_second", [])
    ends = w2t.get("end_second") or starts

    sentences = []
    text = ""
    sent_start = None
    sent_end = None
    word_count = 0
    prev_end = None
    for w, s, e in zip(words, starts, ends):
        w_stripped = (w or "").strip()
        if not w_stripped:
            continue
        has_alnum = any(c.isalnum() for c in w_stripped)
        # Pure-punctuation tokens (e.g. standalone "." or ",") usually carry a
        # sentinel start_second=-1; we still need to keep them in the rendered
        # text and let terminal `.`/`?`/`!` flush the current sentence.
        if not has_alnum:
            if sent_start is not None:
                text = text + w_stripped  # attach with no leading space
                if w_stripped[-1] in ".!?":
                    sentences.append((text, sent_start, sent_end))
                    text, sent_start, sent_end, word_count = "", None, None, 0
            continue
        if s is None or s < 0:
            continue
        # Soft pause-based split: a long silence between the previous word's
        # end and this word's start is a likely sentence boundary, but require
        # the current sentence to already carry enough content so a single
        # hesitation word ("The") doesn't end up as its own sentence.
        if (
            sent_start is not None
            and has_alnum
            and prev_end is not None
            and (s - prev_end) >= PAUSE_GAP_S
            and word_count >= MIN_WORDS_BEFORE_PAUSE_SPLIT
        ):
            sentences.append((text, sent_start, sent_end))
            text, sent_start, sent_end, word_count = "", None, None, 0
        if sent_start is None and has_alnum:
            sent_start = s
        if sent_start is not None:
            text = f"{text} {w_stripped}" if text and has_alnum else text + w_stripped
            sent_end = e
            if has_alnum:
                word_count += 1
                prev_end = e
        # Hard cap: avoid 30+ word "sentences" when no boundaries are detected.
        too_long = word_count >= MAX_WORDS_PER_SENT or (
            sent_start is not None and (sent_end - sent_start) >= MAX_SENT_DURATION_S
        )
        if sent_start is not None and (w_stripped[-1] in ".!?" or too_long):
            sentences.append((text, sent_start, sent_end))
            text, sent_start, sent_end, word_count = "", None, None, 0

    if text and sent_start is not None:
        sentences.append((text, sent_start, sent_end))
    return sentences


def render_format(pairs, fmt_kind, w2t=None):
    spec = FORMAT_SPECS[fmt_kind]
    items = _build_sentences(w2t or {}) if spec.get("level") == "sentence" else pairs
    return spec["joiner"].join(spec["render"](text, s, e) for text, s, e in items)


def render_json(pairs, schema, w2t=None):
    spec = JSON_SCHEMAS[schema]
    if spec.get("level") == "sentence":
        return spec["render"](_build_sentences(w2t or {}))
    return spec["render"](pairs)


# ---------- Variant builder ------------------------------------------------

def _pick_mode(weights):
    modes, ws = zip(*weights.items())
    return random.choices(modes, weights=ws, k=1)[0]


def build_variant(record, weights=None, lang="en"):
    """Return a jsonl record with a timestamped transcription, or None if unsupported.

    `weights` is a dict mapping each mode (default, format_specified,
    json_default, json_specified) to a selection weight. If None, uses the
    `mixed` preset. `lang` selects the prompt language ("en" or "fr"); the
    transcription answer is unaffected (it comes from the audio's alignment).
    """
    if weights is None:
        weights = MIX_PRESETS["mixed"]

    cm = record.get("custom_metadata", {})
    w2t = cm.get("word2time", {})
    if not w2t.get("word"):
        return None

    document_audio = None
    duration = ""
    for turn in record.get("conversations", []):
        if turn.get("type") == "audio":
            document_audio = turn["value"]
            duration = turn.get("duration", "")
            break
    if document_audio is None:
        return None

    pairs = _clean_pairs(w2t)
    if not pairs:
        return None

    def _fill(template, example):
        return template.format(example=example) if "{example}" in template else template

    mode = _pick_mode(weights)
    if mode == "json_default":
        example = JSON_SCHEMAS[DEFAULT_JSON_SCHEMA]["example"][lang]
        question = _fill(random.choice(JSON_DEFAULT_PROMPTS[lang]), example)
        transcription = render_json(pairs, DEFAULT_JSON_SCHEMA, w2t)
    elif mode == "json_specified":
        schema = random.choice(list(JSON_SCHEMAS))
        spec = JSON_SCHEMAS[schema]
        example = spec["example"][lang]
        prompts = JSON_SENTENCE_PROMPTS if spec.get("level") == "sentence" else JSON_SCHEMA_PROMPTS
        question = random.choice(prompts[lang]).format(example=example)
        transcription = render_json(pairs, schema, w2t)
    elif mode == "format_specified":
        fmt_kind = random.choice(NON_DEFAULT_FORMATS)
        spec = FORMAT_SPECS[fmt_kind]
        example = spec["example"][lang]
        prompts = FORMAT_SENTENCE_PROMPTS if spec.get("level") == "sentence" else FORMAT_PROMPTS
        question = random.choice(prompts[lang]).format(example=example)
        transcription = render_format(pairs, fmt_kind, w2t)
    else:  # default
        example = FORMAT_SPECS[DEFAULT_FORMAT]["example"][lang]
        question = _fill(random.choice(DEFAULT_PROMPTS[lang]), example)
        transcription = render_format(pairs, DEFAULT_FORMAT, w2t)

    base_id = record.get("id") or record.get("_id") or ""
    return {
        "id": f"{base_id}_transcribe_timestamped" if base_id else None,
        "conversations": [
            {"from": "User", "value": question, "type": "text"},
            {"from": "User", "value": document_audio, "type": "audio", "duration": duration},
            {"from": "Assistant", "value": transcription, "type": "text"},
        ],
    }


# Records whose last word ends more than this many seconds before the audio
# duration are skipped: the timestamped transcript would be misleading because
# a large tail of the recording has no token-level annotation.
TRAILING_GAP_TOLERANCE_S = 1.0


def _trailing_gap(record):
    """Return (gap_seconds, audio_duration, last_word_end) for a record, or
    None if it can't be computed (no audio duration, or no usable timestamps)."""
    cm = record.get("custom_metadata", {})
    w2t = cm.get("word2time", {})
    ends = w2t.get("end_second") or w2t.get("start_second") or []
    valid_ends = [t for t in ends if t is not None and t >= 0]
    if not valid_ends:
        return None
    last_end = max(valid_ends)
    duration = None
    for turn in record.get("conversations", []):
        if turn.get("type") == "audio":
            try:
                duration = float(turn.get("duration", ""))
            except (TypeError, ValueError):
                duration = None
            break
    if duration is None or duration <= 0:
        return None
    return duration - last_end, duration, last_end


def process_split(input_file: Path, output_file: Path, weights, lang="en") -> tuple[int, int, int]:
    kept = dropped = trailing_skipped = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with input_file.open("r", encoding="utf-8") as fin, output_file.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            gap = _trailing_gap(record)
            if gap is not None and gap[0] > TRAILING_GAP_TOLERANCE_S:
                # Per-record warning suppressed -- count is reported per split
                # and totalled at the end.
                # rec_id = record.get("id") or record.get("_id") or "<no-id>"
                # print(
                #     f"  WARNING: skipping {rec_id} -- last word ends at "
                #     f"{gap[2]:.1f}s but audio duration is {gap[1]:.1f}s "
                #     f"(trailing gap {gap[0]:.1f}s > {TRAILING_GAP_TOLERANCE_S}s)"
                # )
                trailing_skipped += 1
                continue
            out = build_variant(record, weights, lang)
            if out is None:
                dropped += 1
                continue
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            kept += 1
    return kept, dropped, trailing_skipped


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="Directory with train/validation/test jsonl files")
    parser.add_argument("--output_dir", type=str, default="output_timestamped_transcription")
    parser.add_argument(
        "--mix",
        nargs="+",
        choices=list(MIX_PRESETS),
        default=list(MIX_PRESETS),
        help="Which mix preset(s) to emit. One subfolder per preset under --output_dir.",
    )
    parser.add_argument(
        "--language", type=str, choices=LANGUAGES, default="en",
        help="Language of the prompts (the spoken/transcribed content is unchanged).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    input_dir = Path(args.data_path)
    split_files = sorted(input_dir.glob("*.jsonl"))
    if not split_files:
        raise SystemExit(f"No *.jsonl files found in {input_dir}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Splits detected: {[f.stem for f in split_files]}")
    print(f"Mixes:           {args.mix}")
    print(f"Language:        {args.language}")

    total_trailing_skipped = 0
    for mix_name in args.mix:
        weights = MIX_PRESETS[mix_name]
        mix_dir = output_root / mix_name
        mix_dir.mkdir(parents=True, exist_ok=True)
        for split_file in split_files:
            split = split_file.stem
            out_file = mix_dir / f"{split}.jsonl"
            kept, dropped, trailing_skipped = process_split(split_file, out_file, weights, args.language)
            total_trailing_skipped += trailing_skipped
            print(
                f"[{mix_name}/{split}] kept={kept} dropped={dropped} "
                f"trailing_skipped={trailing_skipped} -> {out_file}"
            )

    print(
        f"Total rows discarded due to trailing gap > {TRAILING_GAP_TOLERANCE_S}s: "
        f"{total_trailing_skipped}"
    )
    (output_root / "completed.txt").touch()
