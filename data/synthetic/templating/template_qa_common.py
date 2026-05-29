"""Generic, task-agnostic helpers for timestamped audio QA template generators.

Owns:
- Multilingual joiners (`_OXFORD_AND`, `_oxford_join`, multi-substitute helpers).
- Sentence / word / span extraction against a `word2time` table.
- Negative-sample plumbing (OOV-word pick, out-of-audio timestamp pick).
- The `process_split` driver — parametrized with `build_positive` / `build_negative`
  callables supplied by the task-specific module (e.g. SLU variants).
- Shared constants: `RANGE_MODE_PROB`, `TRAILING_GAP_TOLERANCE_S`,
  pure-negative tokens, JSON answer schemas.

The SLU-specific templates and dispatch live in `template_slu_variants.py`.
"""
import json
import random
import re
from pathlib import Path


LANGUAGES = ("en", "fr")

# Probability that a word2time / word2time_first sample swaps to range mode,
# where the question explicitly asks for both start AND end timestamps and the
# answer reports the time interval. Range mode only fires when there is a
# single occurrence (or, for word2time_first, on the chosen first occurrence)
# so we don't have to render multiple ranges per answer.
RANGE_MODE_PROB = 0.3

# Records whose last word ends more than this many seconds before the audio
# duration are skipped: a large unannotated tail produces misleading negative
# time samples (`bad_time = duration + ...`) and risks OOV-word negatives that
# are actually present in the un-transcribed portion of the audio.
TRAILING_GAP_TOLERANCE_S = 2.0

# Pure-format negative answers: only the literal expected token, no prose.
PURE_NEG_NOT_FOUND = "not_found"
PURE_NEG_INVALID_TIME = "invalid_timestamp"

# Per-language word used by `oxford_join` (final coordinator in a list).
_OXFORD_AND = {"en": "and", "fr": "et"}


# JSON schemas available for any `answer_json`-style task. Each entry supplies
# a `render(q, a, t, s)` function that produces the answer JSON string and a
# literal `example` shown inline in shape-pinning prompts.
def _render_answer_json_flat(q, a, t, s):
    return json.dumps(
        {"question": q, "answer": a, "timestamp": t, "full_sentence": s},
        ensure_ascii=False,
    )


def _render_answer_json_nested(q, a, t, s):
    return json.dumps(
        {"question": q, "answer": a, "source": {"timestamp": t, "sentence": s}},
        ensure_ascii=False,
    )


ANSWER_JSON_SCHEMAS = {
    "flat": {
        "render": _render_answer_json_flat,
        "example": '{"question": "...", "answer": "...", "timestamp": 12.3, "full_sentence": "..."}',
    },
    "nested": {
        "render": _render_answer_json_nested,
        "example": '{"question": "...", "answer": "...", "source": {"timestamp": 12.3, "sentence": "..."}}',
    },
}


def choose_template(templates, single_token):
    """Pick a template. When the answer is multi-token, restrict to phrase-safe ones."""
    pool = [t for t, kind in templates if single_token or kind == "any"]
    return random.choice(pool)


def is_alnum_token(w):
    return any(c.isalnum() for c in w)


def find_all_occurrences(phrase_lower_tokens, w2t):
    """Find all case-insensitive matches of the tokenized phrase in w2t["word"],
    skipping pure-punctuation tokens. Returns list of (start_word_idx, start_s)."""
    if not phrase_lower_tokens:
        return []
    filt = [(i, w.lower()) for i, w in enumerate(w2t["word"]) if is_alnum_token(w)]
    starts = w2t["start_second"]
    n = len(phrase_lower_tokens)
    out = []
    for k in range(len(filt) - n + 1):
        window = filt[k:k + n]
        if [w for _, w in window] == phrase_lower_tokens:
            idx = window[0][0]
            s = starts[idx] if idx < len(starts) else 0.0
            out.append((idx, s))
    return out


def phrase_end_time(w2t, start_idx, n_alnum_tokens):
    """Walk forward from start_idx in w2t['word'], skipping pure-punctuation
    tokens, and return the end_second of the n-th alnum token. Used to derive
    the end timestamp of a multi-word phrase when only its start index/time
    are known. Returns None if unreachable."""
    words = w2t.get("word", [])
    ends = w2t.get("end_second") or w2t.get("start_second") or []
    seen = 0
    for i in range(start_idx, len(words)):
        if is_alnum_token(words[i]):
            seen += 1
            if seen == n_alnum_tokens:
                return ends[i] if i < len(ends) else None
    return None


def oxford_join(parts, lang):
    coord = _OXFORD_AND[lang]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} {coord} {parts[1]}"
    return ", ".join(parts[:-1]) + f", {coord} {parts[-1]}"


def escape_braces(s):
    """Escape `{` and `}` so the string survives a subsequent .format() call
    without being interpreted as a format-placeholder."""
    return s.replace("{", "{{").replace("}", "}}")


def substitute_multi_time(template, times, lang):
    """Replace `{time:.1f}s` and `{time:.1f}` placeholders with joined
    multi-time strings, so existing single-time templates render naturally for
    multi-occurrence answers (e.g. "at 9.1s, 11.3s, and 24.8s")."""
    with_s = escape_braces(oxford_join([f"{t:.1f}s" for t in times], lang))
    no_s = escape_braces(oxford_join([f"{t:.1f}" for t in times], lang))
    # Order matters: replace the more specific pattern (with trailing 's') first.
    template = template.replace("{time:.1f}s", with_s)
    template = template.replace("{time:.1f}", no_s)
    return template


def substitute_multi_sentence(template, sentences, lang):
    """Replace the `{sentence}` placeholder with a quote-split joined
    string so existing templates like `\"{sentence}\"` render as e.g.
    `"S1", "S2", and "S3"` — the outer quotes from the template wrap the
    first and last sentence naturally. Only called for len(sentences) >= 2."""
    coord = _OXFORD_AND[lang]
    if len(sentences) == 2:
        joined = f'{sentences[0]}" {coord} "{sentences[1]}'
    else:
        middle = '", "'.join(sentences[:-1])
        joined = f'{middle}", {coord} "{sentences[-1]}'
    # Source sentences can contain `{` or `}` characters (some records have
    # bracketed asides); escape them so the subsequent .format() call on the
    # template doesn't try to interpret them as placeholders.
    return template.replace("{sentence}", escape_braces(joined))


def dedup_sentences_for_occurrences(w2t, occurrences):
    """Walk each occurrence's enclosing sentence, deduplicated by sentence
    start index so two hits in the same sentence yield one entry."""
    seen = set()
    out = []
    words = w2t["word"]
    for i, _ in occurrences:
        s, start_idx = extract_sentence(words, i)
        if start_idx in seen:
            continue
        seen.add(start_idx)
        out.append(s)
    return out


def extract_sentence(words, word_idx):
    """Return (sentence_text, start_word_idx) for the sentence surrounding
    words[word_idx], using '.' / '?' / '!' as sentence boundaries."""
    boundaries = {".", "?", "!"}
    start = 0
    for i in range(word_idx - 1, -1, -1):
        if words[i] in boundaries:
            start = i + 1
            break
    end = len(words)
    for i in range(word_idx, len(words)):
        if words[i] in boundaries:
            end = i
            break
    sentence = " ".join(words[start:end]).strip()
    sentence = re.sub(r"\s+([,;:])", r"\1", sentence)
    # Some transcripts emit `"` as its own token, which after a naive space-join
    # produces sentences with stray quote tokens floating between words (and
    # makes the JSON-encoded output even noisier with escapes). Drop those and
    # collapse the resulting double spaces.
    sentence = sentence.replace('"', "")
    # Drop bracketed asides like `[ also attested as ' into heaven ' ]` —
    # they're transcript annotations, not spoken content.
    sentence = re.sub(r"\[[^\[\]]*\]", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence, start


def extract_qa_text(conversations):
    """Pull the original textual question (first User text turn) and answer
    (first Assistant text turn) from the source conversation. Either may be None."""
    question = answer = None
    for turn in conversations:
        if turn.get("type") != "text":
            continue
        who = str(turn.get("from", "")).lower()
        if question is None and who in ("user", "human"):
            question = turn.get("value")
        elif answer is None and who in ("assistant", "gpt", "ai"):
            answer = turn.get("value")
    return question, answer


def locate_answer_span(word2time, answer_start, answer_end, tol=1e-3):
    """Return (start_idx, end_idx_inclusive) of the answer in word2time's word list,
    or None if the start time doesn't line up with any word."""
    starts = word2time["start_second"]
    start_idx = next((i for i, s in enumerate(starts) if abs(s - answer_start) < tol), None)
    if start_idx is None:
        return None
    end_idx = start_idx
    for i in range(start_idx + 1, len(starts)):
        if starts[i] < answer_end - tol:
            end_idx = i
        else:
            break
    return start_idx, end_idx


def cased_answer_phrase(w2t, start_idx, end_idx):
    """Join tokens [start_idx..end_idx], dropping pure-punctuation tokens so
    things like 'private ,' don't leak into the answer."""
    tokens = [w for w in w2t["word"][start_idx:end_idx + 1] if any(c.isalnum() for c in w)]
    return " ".join(tokens)


def get_document_audio(conversations):
    for turn in conversations:
        if turn.get("type") == "audio":
            return turn["value"], turn.get("duration", "")
    return None, ""


def trailing_gap(record):
    """Return (gap_seconds, audio_duration, last_word_end) or None if the
    record lacks a usable audio `duration` or any positive end timestamps."""
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


def audio_duration(w2t, audio_duration_value):
    try:
        d = float(audio_duration_value)
        if d > 0:
            return d
    except (TypeError, ValueError):
        pass
    starts = w2t.get("start_second") or []
    ends = w2t.get("end_second") or []
    if ends:
        try:
            return float(max(ends))
        except (TypeError, ValueError):
            pass
    if starts:
        try:
            return float(max(starts)) + 0.5
        except (TypeError, ValueError):
            pass
    return None


def pick_oov_word(w2t, candidates):
    vocab = {w.lower() for w in w2t.get("word", []) if is_alnum_token(w)}
    pool = [w for w in candidates if w.lower() not in vocab]
    return random.choice(pool) if pool else None


def wrap_record(record, task, document_audio, duration, question, answer, suffix=""):
    base_id = record.get("id") or record.get("_id") or ""
    return {
        "id": f"{base_id}_{task}{suffix}" if base_id else None,
        "conversations": [
            {"from": "User", "value": question, "type": "text"},
            {"from": "User", "value": document_audio, "type": "audio", "duration": duration},
            {"from": "Assistant", "value": answer, "type": "text"},
        ],
    }


def pure_times(times):
    return ",".join(f"{t:.1f}s" for t in times)


def process_split(
    input_file: Path,
    output_file: Path,
    task: str,
    build_positive,
    build_negative=None,
    negative_eligible_tasks=(),
    negative_rate: float = 0.0,
    **builder_kwargs,
) -> tuple[int, int, int, int]:
    """Stream `input_file` → `output_file`, applying a task-specific builder.

    `build_positive(record, task, **builder_kwargs)` and (optional)
    `build_negative(record, task, **builder_kwargs)` are callables returning
    a record dict or None. Records whose last-aligned word ends more than
    TRAILING_GAP_TOLERANCE_S before the audio's reported duration are skipped
    upstream of the builder."""
    kept = dropped = neg = trailing_skipped = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with input_file.open("r", encoding="utf-8") as fin, output_file.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            gap = trailing_gap(record)
            if gap is not None and gap[0] > TRAILING_GAP_TOLERANCE_S:
                trailing_skipped += 1
                continue
            out = build_positive(record, task, **builder_kwargs)
            if out is None:
                dropped += 1
                continue
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            kept += 1
            if (
                negative_rate > 0
                and build_negative is not None
                and task in negative_eligible_tasks
                and random.random() < negative_rate
            ):
                neg_out = build_negative(record, task, **builder_kwargs)
                if neg_out is not None:
                    fout.write(json.dumps(neg_out, ensure_ascii=False) + "\n")
                    neg += 1
    return kept, dropped, neg, trailing_skipped
