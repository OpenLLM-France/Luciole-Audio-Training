"""Generate timestamp-grounded (question, audio, answer) triples from any
aligned-transcript jsonl, sampling the target word/time directly from a
`custom_metadata.word2time` table (so it works on plain transcription corpora —
e.g. aligned MLS — that have no `answer_spans` and no Q&A).

Four umbrella tasks are exposed via --tasks. Each is a POOL of weighted
sub-variants (positive and negative); generating a row for a task draws one
sub-variant from its pool. See SUBTASK_WEIGHTS below for the full taxonomy:

    word2time      word→time     all / first            (+ negatives, OOV word)
    time2word      time→word     at / starts_at         (+ negatives)
    time2sentence  time→sentence at / starts_at         (+ negatives)
    word2sentence  word→sentence pos                    (+ negative, OOV word)

`time2word` and `time2sentence` share the same two probe shapes at different
granularity:
    * at         — what word/sentence is spoken AT t (t = the unit's midpoint).
    * starts_at  — which word/sentence STARTS at t (t = the unit's exact onset).

Negatives are FIRST-CLASS sub-variants (suffix `_neg`): OOV-word queries for the
word-keyed tasks, out-of-range timestamp queries for the time-keyed tasks. Their
low selection weight (0.05) keeps them at ~5% of generated rows.

Each source record yields ROWS_PER_INPUT output rows (default 1; a coefficient,
overridable with --rows_per_input). Each row draws one sub-variant from the
task's pool, weighted by SUBTASK_WEIGHTS and without replacement (so multiple
rows from one record never repeat a sub-variant).

Run:
    python template_timestamp_tasks.py /path/to/dir_with_train_dev_test \\
        --output_dir out --language fr --rows_per_input 1 --pure_splits test
"""
import argparse
import json
import random
import re
from pathlib import Path

from template_qa_common import (
    LANGUAGES,
    PURE_NEG_INVALID_TIME,
    PURE_NEG_NOT_FOUND,
    RANGE_MODE_PROB,
    TRAILING_GAP_TOLERANCE_S,
    audio_duration,
    choose_template,
    get_document_audio,
    is_alnum_token,
    pick_oov_word,
    pure_times,
    substitute_multi_sentence,
    substitute_multi_time,
    trailing_gap,
    wrap_record,
)
from template_slu_variants import (
    ANSWER_TEMPLATES,
    ANSWER_TEMPLATES_RANGE,
    NEGATIVE_ANSWER_TEMPLATES,
    NEGATIVE_QUESTION_TEMPLATES,
    NEGATIVE_WORD_CANDIDATES,
    QUESTION_TEMPLATES,
    QUESTION_TEMPLATES_RANGE,
)


# ===========================================================================
# Configuration — all tunable knobs live here.
# ===========================================================================

# The umbrella tasks selectable via --tasks.
TASKS = ("word2time", "time2word", "time2sentence", "word2sentence")

# Per-sub-variant SELECTION WEIGHT, grouped by umbrella task. Every output row
# draws one sub-variant from its task's pool with probability proportional to
# these weights; 0 disables a sub-variant. Positives are 1.0 and negatives
# (suffix `_neg`) are 0.05, so negatives end up ~5% of generated rows
# (0.05 / (1.0 + 0.05) per pair). Bump the `_neg` values to get more negatives.
SUBTASK_WEIGHTS = {
    "word2time": {
        "all":       1.0,
        "all_neg":   0.05,
        "first":     1.0,
        "first_neg": 0.05,
    },
    "time2word": {
        "at":            1.0,
        "at_neg":        0.05,
        "starts_at":     1.0,
        "starts_at_neg": 0.05,
    },
    "time2sentence": {
        "at":            1.0,
        "at_neg":        0.05,
        "starts_at":     1.0,
        "starts_at_neg": 0.05,
    },
    "word2sentence": {
        "pos":     1.0,
        "pos_neg": 0.05,
    },
}

# Number of output rows generated per input record (a coefficient — raise it
# later to oversample). Each row independently draws one sub-variant weighted by
# SUBTASK_WEIGHTS. Floats are rounded stochastically per record, and the value
# is capped at the number of available sub-variants (rows never repeat one).
# Overridable on the CLI with --rows_per_input.
ROWS_PER_INPUT = 1.0

# Maps the `at` probe / its negative to the imported SLU template key per
# granularity (the `at` shape reuses the original time2word/time2sentence
# wording; starts_at uses the local PROBE_* templates below).
_AT_KEY = {"word": "time2word", "sentence": "time2sentence"}


# ===========================================================================
# Local probe templates — starts_at, per granularity. Each entry is
# (text, kind) for choose_template; all are kind "any". Placeholders: {time}
# and {word}/{sentence} (both bound to the same unit text).
# ===========================================================================

PROBE_QUESTION_TEMPLATES = {
    "word": {
        "starts_at": {
            "en": [
                ("Which word starts at {time:.1f} seconds?", "any"),
                ("What word begins at {time:.1f}s?", "any"),
                ("Tell me the word whose onset is at {time:.1f}s.", "any"),
                ("At {time:.1f} seconds, which word begins?", "any"),
                ("Which word starts being spoken at the {time:.1f}-second mark?", "any"),
                ("Give me the word that starts at {time:.1f}s.", "any"),
            ],
            "fr": [
                ("Quel mot commence à {time:.1f} secondes ?", "any"),
                ("Quel mot débute à {time:.1f}s ?", "any"),
                ("Dis-moi le mot dont le début est à {time:.1f}s.", "any"),
                ("À {time:.1f} secondes, quel mot commence ?", "any"),
                ("Quel mot commence à être prononcé au repère {time:.1f} secondes ?", "any"),
                ("Donne-moi le mot qui commence à {time:.1f}s.", "any"),
            ],
        },
    },
    "sentence": {
        "starts_at": {
            "en": [
                ("Which sentence starts at {time:.1f} seconds?", "any"),
                ("What sentence begins at {time:.1f}s?", "any"),
                ("Tell me the sentence that starts at {time:.1f}s.", "any"),
                ("At {time:.1f} seconds, which sentence begins?", "any"),
                ("Which sentence starts being spoken at the {time:.1f}-second mark?", "any"),
                ("Give me the sentence beginning at {time:.1f}s.", "any"),
            ],
            "fr": [
                ("Quelle phrase commence à {time:.1f} secondes ?", "any"),
                ("Quelle phrase débute à {time:.1f}s ?", "any"),
                ("Dis-moi la phrase qui commence à {time:.1f}s.", "any"),
                ("À {time:.1f} secondes, quelle phrase commence ?", "any"),
                ("Quelle phrase commence à être prononcée au repère {time:.1f} secondes ?", "any"),
                ("Donne-moi la phrase qui commence à {time:.1f}s.", "any"),
            ],
        },
    },
}

PROBE_ANSWER_TEMPLATES = {
    "word": {
        "starts_at": {
            "en": [
                ("The word that starts at {time:.1f}s is \"{word}\".", "any"),
                ("\"{word}\" begins at {time:.1f} seconds.", "any"),
                ("It is the word {word}.", "any"),
                ("{word} starts at that moment.", "any"),
                ("\"{word}\"", "any"),
                ("The word starting at {time:.1f}s is {word}.", "any"),
            ],
            "fr": [
                ("Le mot qui commence à {time:.1f}s est \"{word}\".", "any"),
                ("\"{word}\" commence à {time:.1f} secondes.", "any"),
                ("Il s'agit du mot {word}.", "any"),
                ("{word} commence à ce moment.", "any"),
                ("\"{word}\"", "any"),
                ("Le mot commençant à {time:.1f}s est {word}.", "any"),
            ],
        },
    },
    "sentence": {
        "starts_at": {
            "en": [
                ("The sentence that starts at {time:.1f}s is: \"{sentence}\".", "any"),
                ("\"{sentence}\" begins at {time:.1f} seconds.", "any"),
                ("It is: {sentence}", "any"),
                ("The sentence starting at {time:.1f}s is: {sentence}", "any"),
                ("\"{sentence}\"", "any"),
                ("Sentence: {sentence}", "any"),
            ],
            "fr": [
                ("La phrase qui commence à {time:.1f}s est : \"{sentence}\".", "any"),
                ("\"{sentence}\" commence à {time:.1f} secondes.", "any"),
                ("C'est : {sentence}", "any"),
                ("La phrase commençant à {time:.1f}s est : {sentence}", "any"),
                ("\"{sentence}\"", "any"),
                ("Phrase : {sentence}", "any"),
            ],
        },
    },
}


# ===========================================================================
# Text / alignment helpers.
# ===========================================================================

def clean_edges(w):
    """Strip leading/trailing punctuation while keeping word-internal apostrophes
    and hyphens (so French tokens like "m'avez", "Dites-moi", "Qu'est-ce" and
    "cerveau." -> "cerveau" survive correctly). `\\w` matches accented letters."""
    return re.sub(r"^[^\w]+|[^\w]+$", "", w, flags=re.UNICODE)


def _clean_join(tokens):
    """Join transcript tokens into a readable sentence, tucking punctuation back
    against the preceding word and collapsing whitespace."""
    text = " ".join(t.strip() for t in tokens if t and t.strip())
    text = re.sub(r"\s+([,;:.!?])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def pick_word_index(w2t):
    """Pick a random alnum word index with a valid (non-negative) start time.
    Prefers distinctive tokens (>= 4 alnum chars) so timestamp queries don't all
    target stopwords like "le"/"de"; falls back to any alnum token."""
    starts = w2t.get("start_second", [])
    words = w2t.get("word", [])
    cands = [
        i for i, w in enumerate(words)
        if is_alnum_token(w) and i < len(starts)
        and starts[i] is not None and starts[i] >= 0
    ]
    if not cands:
        return None
    distinctive = [i for i in cands if sum(c.isalnum() for c in words[i]) >= 4]
    return random.choice(distinctive or cands)


def _is_boundary(w):
    """True if token `w` carries sentence-terminal `.`/`?`/`!`."""
    s = (w or "").strip()
    return bool(s) and s[-1] in ".!?"


def sentence_spans(words):
    """List of (start_idx, end_idx_exclusive) sentence spans, split on terminal
    `.`/`?`/`!` attached to a token."""
    spans = []
    start = 0
    for i, w in enumerate(words):
        if _is_boundary(w):
            spans.append((start, i + 1))
            start = i + 1
    if start < len(words):
        spans.append((start, len(words)))
    return spans


def find_occurrences(w2t, target_lower):
    """All (idx, start_second) where the edge-cleaned, lowercased token equals
    `target_lower`."""
    starts = w2t.get("start_second", [])
    out = []
    for i, w in enumerate(w2t.get("word", [])):
        if not is_alnum_token(w):
            continue
        if clean_edges(w).lower() == target_lower:
            s = starts[i] if i < len(starts) and starts[i] is not None else 0.0
            out.append((i, s))
    return out


def extract_sentence_attached(words, word_idx):
    """Return (sentence_text, start_word_idx) for the sentence containing
    words[word_idx], using the same terminal-punctuation boundaries as
    `sentence_spans`."""
    for start, end in sentence_spans(words):
        if start <= word_idx < end:
            return _clean_join(words[start:end]), start
    return _clean_join(words), 0


def dedup_sentences(w2t, occurrences):
    """Enclosing sentence per occurrence, deduplicated by sentence start index."""
    seen = set()
    out = []
    words = w2t.get("word", [])
    for i, _ in occurrences:
        s, start_idx = extract_sentence_attached(words, i)
        if start_idx in seen:
            continue
        seen.add(start_idx)
        out.append(s)
    return out


def sentence_units(w2t):
    """All sentences as (text, start_second, end_second) triples, dropping
    sentences with no validly-timed alnum word."""
    words = w2t.get("word", [])
    starts = w2t.get("start_second", [])
    ends = w2t.get("end_second") or starts
    out = []
    for s, e in sentence_spans(words):
        alnum = [
            i for i in range(s, e)
            if is_alnum_token(words[i]) and i < len(starts)
            and starts[i] is not None and starts[i] >= 0
        ]
        if not alnum:
            continue
        last = alnum[-1]
        end_t = ends[last] if last < len(ends) and ends[last] is not None else starts[alnum[0]]
        out.append((_clean_join(words[s:e]), starts[alnum[0]], end_t))
    return out


# ----- Unit sampling (the granularity-specific part of probe tasks) ---------

def pick_word_unit(w2t):
    """Sample one distinctive word as (text, start, end), or None."""
    idx = pick_word_index(w2t)
    if idx is None:
        return None
    starts = w2t["start_second"]
    ends = w2t.get("end_second") or starts
    text = clean_edges(w2t["word"][idx])
    if not text:
        return None
    start = starts[idx]
    end = ends[idx] if idx < len(ends) and ends[idx] is not None else start
    return text, start, end


def pick_sentence_unit(w2t):
    """Sample one sentence as (text, start, end), or None."""
    units = sentence_units(w2t)
    return random.choice(units) if units else None


_GRANULARITY = {
    "word": {"unit": pick_word_unit},
    "sentence": {"unit": pick_sentence_unit},
}


def _probe_fmt(text=None, time=None):
    """Format kwargs covering the placeholders a probe template may use. The
    unit text is bound to word/sentence alike; unreferenced keys are ignored by
    str.format."""
    return {"word": text, "sentence": text, "time": time}


# ===========================================================================
# Sub-variant builders. Each returns (question, answer, pure_answer) or None.
# ===========================================================================

def _build_word2time(sub, w2t, lang):
    qt, at = QUESTION_TEMPLATES[lang], ANSWER_TEMPLATES[lang]
    qtr, atr = QUESTION_TEMPLATES_RANGE[lang], ANSWER_TEMPLATES_RANGE[lang]

    if sub.endswith("_neg"):
        key = "word2time_first" if sub.startswith("first") else "word2time"
        # word2time refers to "le mot {word}", so collapse multi-word candidates
        # (e.g. "raton laveur", "appel vidéo") to a single token; pick_oov_word
        # still re-checks the collapsed word is out of the clip's vocab.
        oov = pick_oov_word(w2t, [w.split()[0] for w in NEGATIVE_WORD_CANDIDATES[lang]])
        if oov is None:
            return None
        question = random.choice(NEGATIVE_QUESTION_TEMPLATES[lang][key]).format(word=oov)
        answer = random.choice(NEGATIVE_ANSWER_TEMPLATES[lang][key]).format(word=oov)
        return question, answer, PURE_NEG_NOT_FOUND

    idx = pick_word_index(w2t)
    if idx is None:
        return None
    starts = w2t["start_second"]
    ends = w2t.get("end_second") or starts
    phrase = clean_edges(w2t["word"][idx])
    if not phrase:
        return None
    answer_start = starts[idx]
    answer_end = ends[idx] if idx < len(ends) and ends[idx] is not None else answer_start
    occ = find_occurrences(w2t, phrase.lower())

    if sub == "first":
        if not occ:
            return None
        first_idx, first_time = min(occ, key=lambda io: io[1])
        if random.random() < RANGE_MODE_PROB:
            first_end = ends[first_idx] if first_idx < len(ends) and ends[first_idx] is not None else first_time
            fmt = {"word": phrase, "time": first_time, "end_time": first_end}
            question = choose_template(qtr["word2time_first"], True).format(**fmt)
            answer = choose_template(atr["word2time_first"], True).format(**fmt)
            return question, answer, f"{first_time:.1f}s-{first_end:.1f}s"
        fmt = {"word": phrase, "time": first_time}
        question = choose_template(qt["word2time_first"], True).format(**fmt)
        answer = choose_template(at["word2time_first"], True).format(**fmt)
        return question, answer, f"{first_time:.1f}s"

    # sub == "all": any/all occurrences, with optional range mode.
    if len(occ) <= 1 and random.random() < RANGE_MODE_PROB:
        fmt = {"word": phrase, "time": answer_start, "end_time": answer_end}
        question = choose_template(qtr["word2time"], True).format(**fmt)
        answer = choose_template(atr["word2time"], True).format(**fmt)
        return question, answer, f"{answer_start:.1f}s-{answer_end:.1f}s"
    q_tpl = choose_template(qt["word2time"], True)
    a_tpl = choose_template(at["word2time"], True)
    if len(occ) >= 2:
        times = [s for _, s in occ]
        a_tpl = substitute_multi_time(a_tpl, times, lang)
        return q_tpl.format(word=phrase), a_tpl.format(word=phrase), pure_times(times)
    sentence, _ = extract_sentence_attached(w2t["word"], idx)
    fmt = {"word": phrase, "time": answer_start, "sentence": sentence}
    return q_tpl.format(**fmt), a_tpl.format(**fmt), f"{answer_start:.1f}s"


def _build_word2sentence(sub, w2t, lang):
    qt, at = QUESTION_TEMPLATES[lang], ANSWER_TEMPLATES[lang]

    if sub.endswith("_neg"):
        oov = pick_oov_word(w2t, NEGATIVE_WORD_CANDIDATES[lang])
        if oov is None:
            return None
        question = random.choice(NEGATIVE_QUESTION_TEMPLATES[lang]["word2sentence"]).format(word=oov)
        answer = random.choice(NEGATIVE_ANSWER_TEMPLATES[lang]["word2sentence"]).format(word=oov)
        return question, answer, PURE_NEG_NOT_FOUND

    idx = pick_word_index(w2t)
    if idx is None:
        return None
    phrase = clean_edges(w2t["word"][idx])
    if not phrase:
        return None
    occ = find_occurrences(w2t, phrase.lower())
    sentence, _ = extract_sentence_attached(w2t["word"], idx)
    q_tpl = choose_template(qt["word2sentence"], True)
    uniq = dedup_sentences(w2t, occ) if len(occ) >= 2 else []
    if len(uniq) >= 2:
        multi_safe = [t for t in at["word2sentence"] if '"{sentence}"' in t[0]]
        a_tpl = substitute_multi_sentence(choose_template(multi_safe, True), uniq, lang)
        return q_tpl.format(word=phrase), a_tpl.format(word=phrase), "|".join(uniq)
    a_tpl = choose_template(at["word2sentence"], True)
    fmt = {"word": phrase, "sentence": sentence}
    return q_tpl.format(**fmt), a_tpl.format(**fmt), sentence


def _build_probe(gran, sub, w2t, duration, lang):
    """Build one of the time→unit probe sub-variants (at / starts_at) and their
    negatives, for granularity `gran` in {word, sentence}.

    `at` queries the unit's midpoint and reuses the imported time2word/
    time2sentence wording; `starts_at` queries the exact onset and uses the
    local PROBE_* templates."""
    neg = sub.endswith("_neg")
    variant = sub[:-4] if neg else sub

    if not neg:
        unit = _GRANULARITY[gran]["unit"](w2t)
        if unit is None:
            return None
        text, start, end = unit
        if variant == "at":
            fmt = _probe_fmt(text=text, time=(start + end) / 2.0)
            question = choose_template(QUESTION_TEMPLATES[lang][_AT_KEY[gran]], True).format(**fmt)
            answer = choose_template(ANSWER_TEMPLATES[lang][_AT_KEY[gran]], True).format(**fmt)
            return question, answer, text
        # starts_at
        fmt = _probe_fmt(text=text, time=start)
        question = choose_template(PROBE_QUESTION_TEMPLATES[gran][variant][lang], True).format(**fmt)
        answer = choose_template(PROBE_ANSWER_TEMPLATES[gran][variant][lang], True).format(**fmt)
        return question, answer, text

    # Negative: query a timestamp past the end of the audio.
    dur = audio_duration(w2t, duration)
    if dur is None:
        return None
    t = dur + random.uniform(0.5, 10.0)
    if variant == "at":
        question = random.choice(NEGATIVE_QUESTION_TEMPLATES[lang][_AT_KEY[gran]]).format(time=t)
    else:  # starts_at
        question = choose_template(PROBE_QUESTION_TEMPLATES[gran][variant][lang], True).format(time=t)
    answer = random.choice(NEGATIVE_ANSWER_TEMPLATES[lang][_AT_KEY[gran]]).format(time=t)
    return question, answer, PURE_NEG_INVALID_TIME


def build_subtask(task, sub, w2t, duration, lang):
    """Dispatch one sub-variant to its builder. Returns (q, a, pure) or None."""
    if task == "word2time":
        return _build_word2time(sub, w2t, lang)
    if task == "word2sentence":
        return _build_word2sentence(sub, w2t, lang)
    if task == "time2word":
        return _build_probe("word", sub, w2t, duration, lang)
    if task == "time2sentence":
        return _build_probe("sentence", sub, w2t, duration, lang)
    return None


# ===========================================================================
# Row assembly — ROWS_PER_INPUT weighted draws per source record.
# ===========================================================================

def _stochastic_round(x):
    base = int(x)
    return base + (1 if random.random() < (x - base) else 0)


def _weighted_sample_without_replacement(items, weights, k):
    """Draw up to k distinct items with probability proportional to weights."""
    items, weights = list(items), list(weights)
    out = []
    for _ in range(min(k, len(items))):
        r = random.uniform(0, sum(weights))
        upto = 0.0
        for i, w in enumerate(weights):
            upto += w
            if upto >= r:
                out.append(items.pop(i))
                weights.pop(i)
                break
    return out


def build_rows(record, task, lang, style, rows_per_input):
    """Build the list of output rows for one source record: `rows_per_input`
    (stochastically rounded) sub-variants of `task`, drawn without replacement
    and weighted by SUBTASK_WEIGHTS. Returns (rows, n_negative)."""
    cm = record.get("custom_metadata", {})
    w2t = cm.get("word2time", {})
    if not w2t.get("word") or not w2t.get("start_second"):
        return [], 0
    document_audio, duration = get_document_audio(record.get("conversations", []))
    if document_audio is None:
        return [], 0

    pool = [(s, w) for s, w in SUBTASK_WEIGHTS[task].items() if w > 0]
    if not pool:
        return [], 0
    n_rows = min(_stochastic_round(rows_per_input), len(pool))
    if n_rows <= 0:
        return [], 0
    chosen = _weighted_sample_without_replacement(
        [s for s, _ in pool], [w for _, w in pool], n_rows
    )

    rows, n_neg = [], 0
    for sub in chosen:
        built = build_subtask(task, sub, w2t, duration, lang)
        if built is None:
            continue
        question, answer, pure = built
        suffix = f"_{sub}" + ("_pure" if style == "pure" else "")
        rows.append(wrap_record(
            record, task, document_audio, duration,
            question, pure if style == "pure" else answer, suffix=suffix,
        ))
        if sub.endswith("_neg"):
            n_neg += 1
    return rows, n_neg


def generate_split(input_file: Path, output_file: Path, task, lang, style, rows_per_input):
    """Stream `input_file` -> `output_file`, emitting `rows_per_input` rows per
    source record. Records whose last-aligned word ends more than
    TRAILING_GAP_TOLERANCE_S before the audio's reported duration are skipped."""
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
            rows, n_neg = build_rows(record, task, lang, style, rows_per_input)
            if not rows:
                dropped += 1
                continue
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += len(rows)
            neg += n_neg
    return kept, dropped, neg, trailing_skipped


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="Directory with train/dev/test jsonl files")
    parser.add_argument("--output_dir", type=str, default="output_timestamp_tasks")
    parser.add_argument(
        "--tasks", type=str, nargs="+", choices=TASKS, default=list(TASKS),
    )
    parser.add_argument(
        "--language", type=str, choices=LANGUAGES, default="en",
        help="Which language's prompt templates to use.",
    )
    parser.add_argument(
        "--rows_per_input", type=float, default=ROWS_PER_INPUT,
        help="Output rows generated per input record (rounded stochastically, "
             "capped at the number of available sub-variants).",
    )
    parser.add_argument(
        "--pure_splits", type=str, nargs="*", default=["all"],
        help="Split stems that also get a pure-answer version in the `<task>` "
             "folder (chat-style answers always go to `<task>_chat`). Use 'all' "
             "(the default) for every detected split, or list specific stems.",
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
    print(f"Tasks:           {args.tasks}")
    print(f"Language:        {args.language}")
    print(f"Rows per input:  {args.rows_per_input}")

    pure_all = "all" in args.pure_splits

    total_trailing_skipped = 0
    for task in args.tasks:
        for split_file in split_files:
            split = split_file.stem
            # Chat-style (natural-language) answers live in a `<task>_chat`
            # folder; pure short-form answers live in `<task>`.
            chat_file = output_root / f"{task}_chat" / f"{split}.jsonl"
            kept, dropped, neg, trailing_skipped = generate_split(
                split_file, chat_file, task, args.language, "chat", args.rows_per_input
            )
            total_trailing_skipped += trailing_skipped
            print(
                f"[{task}_chat/{split}] kept={kept} negatives={neg} dropped={dropped} "
                f"trailing_skipped={trailing_skipped} -> {chat_file}"
            )
            if pure_all or split in args.pure_splits:
                pure_file = output_root / task / f"{split}.jsonl"
                kept_p, dropped_p, neg_p, trailing_p = generate_split(
                    split_file, pure_file, task, args.language, "pure", args.rows_per_input
                )
                print(
                    f"[{task}/{split}] kept={kept_p} negatives={neg_p} dropped={dropped_p} "
                    f"trailing_skipped={trailing_p} -> {pure_file}"
                )

    print(
        f"Total rows discarded due to trailing gap > {TRAILING_GAP_TOLERANCE_S}s: "
        f"{total_trailing_skipped}"
    )
    (output_root / "completed.txt").touch()
