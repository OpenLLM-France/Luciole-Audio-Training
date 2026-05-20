import argparse
import json
import random
import re
from pathlib import Path


TASKS = ("word2time", "word2time_first", "time2sentence", "time2word", "word2sentence", "answer_with_source", "answer_with_time", "format_json_answer")

# Each template is (text, kind). kind="word" templates use the noun "word"
# and are only chosen when the answer is a single token. kind="any" works
# for both single words and multi-word phrases.
QUESTION_TEMPLATES = {
    "word2time": [
        ("At what time in the audio is the word \"{word}\" spoken?", "word"),
        ("When does the speaker talk about {word}?", "any"),
        ("Find the word {word} in the audio and tell me its timestamp.", "word"),
        ("Locate \"{word}\" in the recording. When does it occur?", "any"),
        ("At which second is {word} pronounced?", "any"),
        ("Give me the timestamp where '{word}' appears.", "any"),
        ("Where in the clip can I hear \"{word}\"?", "any"),
        ("Tell me when they talk about {word} in the audio.", "any"),
        ("I want the start time of the word {word}. What is it?", "word"),
        ("Pinpoint the moment when ({word}) is uttered.", "any"),
        ("Can you tell me at what second \"{word}\" is said?", "any"),
        ("Target word: {word}. When is it spoken?", "word"),
    ],
    "word2time_first": [
        ("What is the first timestamp at which the word \"{word}\" is spoken?", "word"),
        ("When is {word} first said in the audio?", "any"),
        ("Find the first occurrence of {word} in the recording.", "any"),
        ("At what time does {word} first appear?", "any"),
        ("Give me the first timestamp at which '{word}' appears.", "any"),
        ("When is the word {word} first pronounced?", "word"),
        ("What is the earliest second at which \"{word}\" is uttered?", "any"),
        ("Pinpoint the first moment {word} is spoken.", "any"),
        ("At which second does the speaker first say {word}?", "any"),
        ("Target word: {word}. When does it first appear?", "word"),
    ],
    "time2sentence": [
        ("What sentence is being spoken around {time:.1f} seconds?", "any"),
        ("Around {time:.1f}s, what sentence do you hear?", "any"),
        ("Transcribe the full sentence containing the {time:.1f}s mark.", "any"),
        ("Give me the sentence that spans {time:.1f} seconds.", "any"),
        ("Listen near {time:.1f}s, what full sentence is the speaker saying?", "any"),
        ("Report the sentence that includes the {time:.1f}-second mark.", "any"),
        ("At roughly {time:.1f}s, give me the entire sentence being spoken.", "any"),
        ("Which sentence contains the {time:.1f}s moment?", "any"),
        ("Around {time:.1f}s, can you transcribe the full sentence?", "any"),
        ("What sentence spans time {time:.1f}s in the recording?", "any"),
        ("The {time:.1f}s mark falls within a sentence, transcri it fully.", "any"),
        ("Give me the complete sentence that covers {time:.1f}s.", "any"),
    ],
    "time2word": [
        ("What word is spoken at {time:.1f} seconds?", "word"),
        ("Which word does the speaker say at {time:.1f}s?", "word"),
        ("At {time:.1f} seconds, what word can you hear?", "word"),
        ("Tell me the word uttered at the {time:.1f}-second mark.", "word"),
        ("Which word corresponds to {time:.1f}s in the recording?", "word"),
        ("At around {time:.1f}s, what is the speaker saying (single word)?", "word"),
        ("Give me the word pronounced at {time:.1f} seconds.", "word"),
        ("What word is at timestamp {time:.1f}?", "word"),
        ("Listen at {time:.1f}s and report the word you hear.", "word"),
        ("What is being said at {time:.1f} seconds?", "any"),
        ("What do you hear at {time:.1f}s?", "any"),
        ("Report what the speaker utters at {time:.1f} seconds.", "any"),
        ("What is pronounced at {time:.1f}s?", "any"),
    ],
    "word2sentence": [
        ("Which sentence in the audio contains the word \"{word}\"?", "word"),
        ("Find the sentence where {word} is spoken and transcribe it.", "any"),
        ("Give me the full sentence that includes {word}.", "any"),
        ("In which sentence does the speaker use \"{word}\"?", "any"),
        ("Transcribe the sentence containing '{word}'.", "any"),
        ("What sentence does {word} appear in?", "any"),
        ("Report the sentence in which {word} is said.", "any"),
        ("\"{word}\" is somewhere in the audio, give me the sentence it's part of.", "any"),
        ("Locate {word} and transcribe the surrounding sentence.", "any"),
        ("Target word: {word}. Give me the sentence it appears in.", "word"),
    ],
    "answer_with_source": [
        ("Answer the following question from the audio, and cite the source sentence with its timestamp: \"{question}\"", "any"),
        ("Listen to the audio and answer: {question}. Also give me the sentence it comes from and when it's spoken.", "any"),
        ("\"{question}\" Can you answer this using the recording and tell me the supporting sentence and its start time.", "any"),
        ("From the audio, answer this and attribute your answer to a specific sentence plus timestamp. Question: {question}", "any"),
        ("Give me the answer to \"{question}\" along with the sentence in the audio that contains it and the time at which that sentence begins.", "any"),
        ("Answer the question based on the recording. Include the source sentence and when it is said. Question: {question}", "any"),
        ("Please answer '{question}' and back it up with the source sentence from the audio and its timestamp.", "any"),
        ("Using the audio, answer the question: {question}. Cite the sentence the answer comes from and the time it starts.", "any"),
    ],
    "answer_with_time": [
        ("Answer the following question from the audio and give me the timestamp where the answer is spoken: \"{question}\"", "any"),
        ("Listen to the audio and answer: {question}. When is the answer said?", "any"),
        ("Question: {question}. Answer this and tell me when the answer is spoken in the audio.", "any"),
        ("From the audio, answer \"{question}\" and report the timestamp of the answer.", "any"),
        ("Give me the answer to this question along with the time at which it is spoken: {question}", "any"),
        ("Answer the question based on the recording. Include the time the answer is said. Question: {question}", "any"),
        ("Please answer '{question}' and tell me when the answer appears in the audio.", "any"),
        ("Using the audio, answer this and give the timestamp of the answer: {question}", "any"),
    ],
    # Every prompt embeds {example} — the literal JSON shape picked from
    # ANSWER_JSON_SCHEMAS — so the model always sees the expected format.
    "format_json_answer": [
        ("Answer the following question using the audio and return your response as JSON in this shape: {example}. Question: \"{question}\"", "any"),
        ("Listen to the audio and answer the question: {question}. Return the result as a JSON object like: {example}.", "any"),
        ("Answer \"{question}\" based on the audio. Output JSON like: {example}.", "any"),
        ("From the audio, answer this and return a JSON object like: {example}. Question: {question}", "any"),
        ("Using the recording, answer the question: {question}. Format your response as JSON matching: {example}.", "any"),
        ("Please answer '{question}' from the audio and output JSON: {example}.", "any"),
        ("Answer this using the recording and return JSON matching this shape: {example}. Question: {question}", "any"),
        ("Question: {question}. Answer this using the audio and return JSON in this exact shape: {example}.", "any"),
    ],
}

ANSWER_TEMPLATES = {
    "word2time": [
        ("The word \"{word}\" is spoken at {time:.1f} seconds.", "word"),
        ("You can hear {word} at around {time:.1f}s.", "any"),
        ("\"{word}\" appears at {time:.1f} seconds.", "any"),
        ("The speaker says {word} at {time:.1f}s.", "any"),
        ("At {time:.1f} seconds, the word {word} is pronounced.", "word"),
        ("'{word}' is said at {time:.1f}s in the audio.", "any"),
        ("It occurs at {time:.1f} seconds.", "any"),
        ("{time:.1f}s.", "any"),
        ("Around {time:.1f} seconds.", "any"),
        ("The word starts at {time:.1f} seconds.", "word"),
        ("In the audio, \"{word}\" starts at {time:.1f}s.", "any"),
        ("Word: {word}. Time: {time:.1f}s.", "word"),
    ],
    "word2time_first": [
        ("The word \"{word}\" is first spoken at {time:.1f} seconds.", "word"),
        ("{word} first appears at {time:.1f}s.", "any"),
        ("The first occurrence of {word} is at {time:.1f} seconds.", "any"),
        ("First said at {time:.1f}s.", "any"),
        ("\"{word}\" is first uttered at {time:.1f}s.", "any"),
        ("The first timestamp is {time:.1f}s.", "any"),
        ("{time:.1f}s.", "any"),
        ("Around {time:.1f} seconds.", "any"),
        ("The first time the speaker says {word} is at {time:.1f}s.", "any"),
        ("First occurrence: {time:.1f}s.", "any"),
    ],
    "time2sentence": [
        ("Around that moment, the speaker is saying: {sentence}", "any"),
        ("The full sentence being spoken is: \"{sentence}\".", "any"),
        ("Around {time:.1f}s, you can hear: {sentence}", "any"),
        ("The sentence containing the {time:.1f}s mark is: \"{sentence}\".", "any"),
        ("The speaker utters: {sentence}", "any"),
        ("\"{sentence}\"", "any"),
        ("The sentence that spans {time:.1f}s is: {sentence}", "any"),
        ("At roughly {time:.1f}s, the full sentence being said is: \"{sentence}\".", "any"),
        ("Sentence: {sentence}", "any"),
    ],
    "time2word": [
        ("The word spoken at {time:.1f}s is \"{word}\".", "word"),
        ("At that moment, the speaker says {word}.", "any"),
        ("You can hear the word {word}.", "word"),
        ("It is the word \"{word}\".", "word"),
        ("Around {time:.1f} seconds, the word is {word}.", "word"),
        ("\"{word}\"", "any"),
        ("The speaker pronounces {word} at {time:.1f}s.", "any"),
        ("The speaker says '{word}'.", "any"),
        ("You can hear {word}.", "any"),
        ("Word: {word}.", "word"),
    ],
    "word2sentence": [
        ("The sentence containing \"{word}\" is: \"{sentence}\".", "any"),
        ("{word} appears in this sentence: \"{sentence}\".", "any"),
        ("The speaker uses {word} in: \"{sentence}\".", "any"),
        ("It is found in the following sentence: \"{sentence}\".", "any"),
        ("You can hear \"{word}\" in: \"{sentence}\".", "any"),
        ("\"{sentence}\"", "any"),
        ("'{word}' is part of: \"{sentence}\".", "any"),
        ("Sentence containing {word}: \"{sentence}\".", "any"),
    ],
    "answer_with_time": [
        ("The answer is \"{answer}\", spoken at {time:.1f}s.", "any"),
        ("'{answer}', said at {time:.1f}s.", "any"),
        ("Answer: {answer} (at {time:.1f}s).", "any"),
        ("{answer}, which is said at {time:.1f} seconds.", "any"),
        ("The answer is: {answer}. It is spoken at {time:.1f}s in the audio.", "any"),
        ("{answer}, spoken at {time:.1f}s.", "any"),
        ("You can hear the answer \"{answer}\" at {time:.1f}s.", "any"),
        ("\"{answer}\" is the answer; it appears at {time:.1f} seconds.", "any"),
        ("Answer: {answer}. Time: {time:.1f}s.", "any"),
    ],
    "answer_with_source": [
        ("The answer is {answer}. It comes from the sentence \"{sentence}\", spoken starting at {time:.1f}s.", "any"),
        ("\"{answer}\". This is stated in the sentence \"{sentence}\", which begins at {time:.1f}s.", "any"),
        ("Answer: {answer}. Source: {sentence}. Starts around: {time:.1f}s.", "any"),
        ("The audio says the answer is \"{answer}\", specifically in the sentence \"{sentence}\" beginning at {time:.1f}s.", "any"),
        ("The answer is: {answer}. The supporting sentence \"{sentence}\" is spoken at {time:.1f}s.", "any"),
        ("The response {answer} (found in the sentence: {sentence}), which starts around {time:.1f}s in the audio.", "any"),
        ("\"{answer}\" is the answer, which comes from the sentence \"{sentence}\" at {time:.1f}s.", "any"),
    ],
}


# Probability that a word2time / word2time_first sample swaps to range mode,
# where the question explicitly asks for both start AND end timestamps and the
# answer reports the time interval. Range mode only fires when there is a
# single occurrence (or, for word2time_first, on the chosen first occurrence)
# so we don't have to render multiple ranges per answer.
RANGE_MODE_PROB = 0.3

QUESTION_TEMPLATES_RANGE = {
    "word2time": [
        ("Give me both the start and end timestamps of the word \"{word}\".", "word"),
        ("From what second to what second is {word} spoken?", "any"),
        ("What are the start and end times of \"{word}\" in the audio?", "any"),
        ("Locate {word}: I need its timestamps.", "any"),
        ("When does {word} start and end?", "any"),
        ("Give me the time range during which '{word}' is uttered.", "any"),
        ("Tell me when {word} begins and ends in the recording.", "any"),
        ("What is the start and end second of the word {word}?", "word"),
        ("I need the full interval (start and end) for \"{word}\".", "any"),
    ],
    "word2time_first": [
        ("What are the start and end timestamps of the first occurrence of \"{word}\"?", "word"),
        ("When does {word} first start and end in the audio?", "any"),
        ("Give me the start and end time of the first time {word} is spoken.", "any"),
        ("From what second to what second does {word} first appear?", "any"),
        ("First occurrence of {word}: I need both start and end times.", "any"),
        ("What is the start and end second of the first '{word}'?", "any"),
        ("Tell me when {word} first occurs in the recording.", "any"),
        ("Give me the full interval of the first occurrence of {word}.", "any"),
    ],
}

ANSWER_TEMPLATES_RANGE = {
    "word2time": [
        ("The word \"{word}\" is spoken from {time:.1f}s to {end_time:.1f}s.", "word"),
        ("{word} runs from {time:.1f}s to {end_time:.1f}s.", "any"),
        ("Start: {time:.1f}s. End: {end_time:.1f}s.", "any"),
        ("From {time:.1f}s to {end_time:.1f}s.", "any"),
        ("\"{word}\" begins at {time:.1f}s and ends at {end_time:.1f}s.", "any"),
        ("{time:.1f}s - {end_time:.1f}s.", "any"),
        ("It is uttered between {time:.1f}s and {end_time:.1f}s.", "any"),
        ("The interval is {time:.1f}s to {end_time:.1f}s.", "any"),
    ],
    "word2time_first": [
        ("The word \"{word}\" is first spoken from {time:.1f}s to {end_time:.1f}s.", "word"),
        ("{word} first runs from {time:.1f}s to {end_time:.1f}s.", "any"),
        ("Start: {time:.1f}s. End: {end_time:.1f}s.", "any"),
        ("From {time:.1f}s to {end_time:.1f}s.", "any"),
        ("First occurrence: {time:.1f}s - {end_time:.1f}s.", "any"),
        ("\"{word}\" first begins at {time:.1f}s and ends at {end_time:.1f}s.", "any"),
        ("The first occurrence runs from {time:.1f}s to {end_time:.1f}s.", "any"),
        ("First interval: {time:.1f}s to {end_time:.1f}s.", "any"),
    ],
}


# JSON schemas available for the `answer_json` task. Each entry supplies
# a `render(q, a, t, s)` function that produces the answer JSON string, a
# literal `example` shown inline in shape-pinning prompts, and a `fields`
# prose description used in schema-describing prompts.
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


# Candidate words unlikely to appear in speech transcripts — filtered at
# runtime against the actual word list to guarantee a true OOV for the clip.
NEGATIVE_WORD_CANDIDATES = [
    "xylophone", "quokka", "zeppelin", "kumquat", "platypus", "obsidian",
    "saxophone", "marshmallow", "parsnip", "jellyfish", "labyrinth",
    "hippopotamus", "kaleidoscope", "rhubarb", "tangerine", "walrus",
    "narwhal", "avocado", "cinnamon", "dandelion", "eucalyptus",
    "flamingo", "gazelle", "igloo", "mongoose", "nutmeg", "ostrich",
    "pineapple", "raccoon", "trombone", "vulcanize", "zucchini",
    "aardvark", "bumblebee", "chinchilla", "geyser", "pomegranate",
    "document", "time", "restaurant", "meeting", "appointment",
    "file", "text", "email", "message", "plane", "taxi", "doctor",
    "next week", "important", "phone conversation",
    "armadillo", "python", "pelican", "octopus", "barnacle",
    "wombat", "puffin", "cassowary", "manatee", "lemur",
    "saffron", "paprika", "cardamom", "lavender", "dish",
    "didgeridoo", "harpsichord", "accordion", "ukulele", "tambourine",
    "stalactite", "tundra", "fjord", "monsoon", "station",
    "telescope", "code", "helicopter", "bulldozer", "lighthouse",
    "passport", "umbrella", "treadmill", "cathedral", "scaffolding",
    "yesterday", "tomorrow", "midnight", "weekend", "holiday",
    "hospital", "library", "stadium", "warehouse", "embassy",
    "lawyer", "plumber", "astronaut", "engineer", "journalist",
    "spreadsheet", "podcast", "newsletter", "invoice", "receipt",
    "bicycle", "motorcycle", "scooter", "ferry", "subway",
    "next month", "last summer", "very urgent", "completely broken",
    "video call", "team meeting", "annual report", "credit card",
]

_NEGATIVE_TIME_ANSWERS = [
    "{time:.1f}s is past the end of the audio, so it is not a valid timestamp.",
    "The timestamp {time:.1f}s is beyond the duration of the recording. Not a valid timestamp.",
    "The audio ends before {time:.1f}s, so this is not a valid timestamp.",
    "That is not a valid timestamp: the audio is shorter than {time:.1f} seconds.",
    "Not a valid timestamp ({time:.1f}s is after the end of the audio).",
    "There is nothing at {time:.1f}s because the recording ends earlier.",
    "{time:.1f} seconds falls outside the audio, so the timestamp is invalid.",
    "Invalid timestamp: {time:.1f}s.",
]

NEGATIVE_QUESTION_TEMPLATES = {
    "word2time": [
        "At what time in the audio is the word \"{word}\" spoken?",
        "When does the speaker say {word}?",
        "Find the word {word} in the audio and tell me its timestamp.",
        "At which second is {word} pronounced?",
        "Where in the clip can I hear '{word}'?",
        "Give me the timestamp where {word} appears.",
        "Tell me when \"{word}\" is said in the audio.",
        "Can you locate {word} in the recording?",
        "Pinpoint the moment when ({word}) is uttered.",
        "Target word: {word}. When is it spoken?",
    ],
    "word2time_first": [
        "What is the first timestamp at which the word \"{word}\" is spoken?",
        "When is {word} first said in the audio?",
        "Find the first occurrence of {word} in the recording.",
        "At what time does {word} first appear?",
        "Give me the first timestamp at which '{word}' appears.",
        "When is the word {word} first pronounced?",
        "What is the earliest second at which \"{word}\" is uttered?",
        "Pinpoint the first moment {word} is spoken.",
        "Target word: {word}. When does it first appear?",
    ],
    "word2sentence": [
        "Which sentence in the audio contains the word \"{word}\"?",
        "Find the sentence where {word} is spoken and transcribe it.",
        "Give me the full sentence that includes {word}.",
        "In which sentence does the speaker use \"{word}\"?",
        "Transcribe the sentence containing '{word}'.",
        "What sentence does {word} appear in?",
        "Report the sentence in which {word} is said.",
        "Locate \"{word}\" and transcribe the surrounding sentence.",
        "Target word: {word}. Give me the sentence it appears in.",
    ],
    "time2word": [
        "What word is spoken at {time:.1f} seconds?",
        "Which word does the speaker say at {time:.1f}s?",
        "At {time:.1f} seconds, what word can you hear?",
        "Tell me the word uttered at the {time:.1f}-second mark.",
        "Give me the word pronounced at {time:.1f} seconds.",
        "What is said at {time:.1f}s?",
        "Report the word at timestamp {time:.1f}.",
    ],
    "time2sentence": [
        "What sentence is being spoken around {time:.1f} seconds?",
        "Around {time:.1f}s, what sentence do you hear?",
        "Transcribe the sentence at {time:.1f} seconds.",
        "Which sentence contains the {time:.1f}s moment?",
        "Give me the sentence that spans {time:.1f}s.",
        "Listen near {time:.1f}s. What full sentence is being said?",
    ],
}

NEGATIVE_ANSWER_TEMPLATES = {
    "word2time": [
        "The word \"{word}\" is not spoken in the audio.",
        "I cannot find {word} in the recording.",
        "'{word}' does not appear in the audio.",
        "The word {word} is not present in this clip.",
        "\"{word}\" is not said anywhere in the audio.",
        "The speaker never says {word} in this recording.",
        "There is no occurrence of {word} in the audio.",
        "Not found: {word}.",
    ],
    "word2time_first": [
        "The word \"{word}\" is never spoken in the audio, so there is no first timestamp.",
        "I cannot find any occurrence of {word} in the recording.",
        "'{word}' does not appear in the audio at all.",
        "There is no first occurrence of {word} -- the word is not in this clip.",
        "\"{word}\" is not said anywhere in the audio, so it has no first timestamp.",
        "The speaker never says {word}; no first occurrence to report.",
        "{word} is absent from the recording.",
        "Not found: {word}. No first occurrence.",
    ],
    "word2sentence": [
        "The word \"{word}\" is not spoken in the audio, so there is no sentence containing it.",
        "I cannot find {word} in the recording; no sentence contains it.",
        "'{word}' does not appear in the audio, so no sentence can be reported.",
        "The word {word} is not present in this clip, so there is no matching sentence.",
        "\"{word}\" is not said anywhere in the audio; no sentence contains it.",
        "The speaker never says {word}, so no sentence in the audio includes it.",
        "There is no occurrence of {word} in the audio, so no sentence can be returned.",
        "Not found in the audio: {word}. No sentence to report.",
    ],
    "time2word": _NEGATIVE_TIME_ANSWERS,
    "time2sentence": _NEGATIVE_TIME_ANSWERS,
}


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


def _is_alnum_token(w):
    return any(c.isalnum() for c in w)


def find_all_occurrences(phrase_lower_tokens, w2t):
    """Find all case-insensitive matches of the tokenized phrase in w2t["word"],
    skipping pure-punctuation tokens. Returns list of (start_word_idx, start_s)."""
    if not phrase_lower_tokens:
        return []
    filt = [(i, w.lower()) for i, w in enumerate(w2t["word"]) if _is_alnum_token(w)]
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


def _phrase_end_time(w2t, start_idx, n_alnum_tokens):
    """Walk forward from start_idx in w2t['word'], skipping pure-punctuation
    tokens, and return the end_second of the n-th alnum token. Used to derive
    the end timestamp of a multi-word phrase when only its start index/time
    are known. Returns None if unreachable."""
    words = w2t.get("word", [])
    ends = w2t.get("end_second") or w2t.get("start_second") or []
    seen = 0
    for i in range(start_idx, len(words)):
        if _is_alnum_token(words[i]):
            seen += 1
            if seen == n_alnum_tokens:
                return ends[i] if i < len(ends) else None
    return None


def _oxford_join(parts):
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _escape_braces(s):
    """Escape `{` and `}` so the string survives a subsequent .format() call
    without being interpreted as a format-placeholder."""
    return s.replace("{", "{{").replace("}", "}}")


def _substitute_multi_time(template, times):
    """Replace `{time:.1f}s` and `{time:.1f}` placeholders with Oxford-joined
    multi-time strings, so existing single-time templates render naturally for
    multi-occurrence answers (e.g. "at 9.1s, 11.3s, and 24.8s")."""
    with_s = _escape_braces(_oxford_join([f"{t:.1f}s" for t in times]))
    no_s = _escape_braces(_oxford_join([f"{t:.1f}" for t in times]))
    # Order matters: replace the more specific pattern (with trailing 's') first.
    template = template.replace("{time:.1f}s", with_s)
    template = template.replace("{time:.1f}", no_s)
    return template


def _substitute_multi_sentence(template, sentences):
    """Replace the `{sentence}` placeholder with a quote-split Oxford-joined
    string so existing templates like `\"{sentence}\"` render as e.g.
    `"S1", "S2", and "S3"` — the outer quotes from the template wrap the
    first and last sentence naturally. Only called for len(sentences) >= 2."""
    if len(sentences) == 2:
        joined = f'{sentences[0]}" and "{sentences[1]}'
    else:
        middle = '", "'.join(sentences[:-1])
        joined = f'{middle}", and "{sentences[-1]}'
    # Source sentences can contain `{` or `}` characters (some records have
    # bracketed asides); escape them so the subsequent .format() call on the
    # template doesn't try to interpret them as placeholders.
    return template.replace("{sentence}", _escape_braces(joined))


def _dedup_sentences_for_occurrences(w2t, occurrences):
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


# Records whose last word ends more than this many seconds before the audio
# duration are skipped: a large unannotated tail produces misleading negative
# time samples (`bad_time = duration + ...`) and risks OOV-word negatives that
# are actually present in the un-transcribed portion of the audio.
TRAILING_GAP_TOLERANCE_S = 2.0


def _get_document_audio(conversations):
    for turn in conversations:
        if turn.get("type") == "audio":
            return turn["value"], turn.get("duration", "")
    return None, ""


def _trailing_gap(record):
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


def _audio_duration(w2t, audio_duration):
    try:
        d = float(audio_duration)
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


def _pick_oov_word(w2t):
    vocab = {w.lower() for w in w2t.get("word", []) if _is_alnum_token(w)}
    pool = [w for w in NEGATIVE_WORD_CANDIDATES if w.lower() not in vocab]
    return random.choice(pool) if pool else None


def _wrap_record(record, task, document_audio, duration, question, answer, suffix=""):
    base_id = record.get("id") or record.get("_id") or ""
    return {
        "id": f"{base_id}_{task}{suffix}" if base_id else None,
        "conversations": [
            {"from": "User", "value": question, "type": "text"},
            {"from": "User", "value": document_audio, "type": "audio", "duration": duration},
            {"from": "Assistant", "value": answer, "type": "text"},
        ],
    }


# Pure-format negative answers: only the literal expected token, no prose.
_PURE_NEG_NOT_FOUND = "not_found"
_PURE_NEG_INVALID_TIME = "invalid_timestamp"


def build_negative_variant(record, task, style="chat"):
    """Return a `_neg` record for `task`, or None if a negative cannot be built
    (task not eligible, no OOV word available, no usable duration, or missing audio).

    With style="pure", the answer is reduced to a literal token (`not_found` or
    `invalid_timestamp`) instead of a natural-language sentence."""
    if task not in NEGATIVE_QUESTION_TEMPLATES:
        return None
    cm = record.get("custom_metadata", {})
    w2t = cm.get("word2time", {})
    document_audio, duration = _get_document_audio(record.get("conversations", []))
    if document_audio is None:
        return None
    if task in ("word2time", "word2time_first", "word2sentence"):
        oov = _pick_oov_word(w2t)
        if oov is None:
            return None
        fmt = {"word": oov}
        pure_answer = _PURE_NEG_NOT_FOUND
    else:  # time2word, time2sentence
        dur = _audio_duration(w2t, duration)
        if dur is None:
            return None
        fmt = {"time": dur + random.uniform(0.5, 10.0)}
        pure_answer = _PURE_NEG_INVALID_TIME
    question = random.choice(NEGATIVE_QUESTION_TEMPLATES[task]).format(**fmt)
    answer = random.choice(NEGATIVE_ANSWER_TEMPLATES[task]).format(**fmt)
    if style == "pure":
        answer = pure_answer
        suffix = "_neg_pure"
    else:
        suffix = "_neg"
    return _wrap_record(record, task, document_audio, duration, question, answer, suffix=suffix)


def _pure_times(times):
    return ",".join(f"{t:.1f}s" for t in times)


def build_variant(record, task, style="chat"):
    """Return a new jsonl record for `task`, or None if the source record can't support it.

    With style="pure", the answer is the literal expected value only (e.g. "10.2s,25.6s"
    or just the target sentence/word) instead of a natural-language reply. Used for
    verified_test so eval can compare against a canonical short-form answer."""
    cm = record.get("custom_metadata", {})
    spans = cm.get("answer_spans", {})
    w2t = cm.get("word2time", {})
    if not spans.get("answer") or not w2t.get("word"):
        return None

    answer_start = spans["start_second"][0]
    answer_end = spans.get("end_second", [answer_start])[0]
    span = locate_answer_span(w2t, answer_start, answer_end)
    if span is None:
        return None
    start_idx, end_idx = span

    phrase = cased_answer_phrase(w2t, start_idx, end_idx) or spans["answer"][0]
    single_token = " " not in phrase

    sentence, sent_start_idx = extract_sentence(w2t["word"], start_idx)
    starts = w2t["start_second"]
    sentence_start_time = (
        starts[sent_start_idx] if sent_start_idx < len(starts) else answer_start
    )

    conversations = record.get("conversations", [])
    document_audio, duration = _get_document_audio(conversations)
    if document_audio is None:
        return None

    if task == "answer_with_source":
        src_question, src_answer = extract_qa_text(conversations)
        if not src_question or not src_answer:
            return None
        fmt = {
            "question": src_question,
            "answer": src_answer,
            "sentence": sentence,
            "time": sentence_start_time,
        }
        question = choose_template(QUESTION_TEMPLATES[task], single_token).format(**fmt)
        answer = choose_template(ANSWER_TEMPLATES[task], single_token).format(**fmt)
        pure_answer = f"{src_answer}|{sentence_start_time:.1f}s|{sentence}"
    elif task == "answer_with_time":
        src_question, src_answer = extract_qa_text(conversations)
        if not src_question or not src_answer:
            return None
        lookup_tokens = [t.lower() for t in phrase.split() if _is_alnum_token(t)]
        occurrences = find_all_occurrences(lookup_tokens, w2t)
        q_tpl = choose_template(QUESTION_TEMPLATES[task], single_token)
        a_tpl = choose_template(ANSWER_TEMPLATES[task], single_token)
        question = q_tpl.format(question=src_question)
        if len(occurrences) >= 2:
            times = [s for _, s in occurrences]
            a_tpl = _substitute_multi_time(a_tpl, times)
            answer = a_tpl.format(answer=src_answer)
            pure_answer = f"{src_answer}|{_pure_times(times)}"
        else:
            answer = a_tpl.format(answer=src_answer, time=answer_start)
            pure_answer = f"{src_answer}|{answer_start:.1f}s"
    elif task == "format_json_answer":
        src_question, src_answer = extract_qa_text(conversations)
        if not src_question or not src_answer:
            return None
        schema_name = random.choice(list(ANSWER_JSON_SCHEMAS))
        schema = ANSWER_JSON_SCHEMAS[schema_name]
        q_tpl = choose_template(QUESTION_TEMPLATES[task], single_token)
        question = q_tpl.format(
            question=src_question,
            example=schema["example"],
        )
        answer = schema["render"](
            src_question, src_answer, round(answer_start, 1), sentence
        )
        # JSON output is already the canonical short form.
        pure_answer = answer
    elif task == "word2time":
        lookup_tokens = [t.lower() for t in phrase.split() if _is_alnum_token(t)]
        occurrences = find_all_occurrences(lookup_tokens, w2t)
        # Range mode: the question explicitly asks for both start and end
        # timestamps. Limited to single-occurrence cases (rendering multiple
        # ranges per answer would require a separate joiner).
        if len(occurrences) <= 1 and random.random() < RANGE_MODE_PROB:
            fmt = {"word": phrase, "time": answer_start, "end_time": answer_end}
            question = choose_template(QUESTION_TEMPLATES_RANGE[task], single_token).format(**fmt)
            answer = choose_template(ANSWER_TEMPLATES_RANGE[task], single_token).format(**fmt)
            pure_answer = f"{answer_start:.1f}s-{answer_end:.1f}s"
        else:
            q_tpl = choose_template(QUESTION_TEMPLATES[task], single_token)
            a_tpl = choose_template(ANSWER_TEMPLATES[task], single_token)
            if len(occurrences) >= 2:
                # Reuse the chosen templates; swap the {time:.1f} placeholder for a
                # joined multi-time string so the answer reads e.g. "at 9.1s, 11.3s,
                # and 24.8s" while the rest of the template wording is unchanged.
                times = [s for _, s in occurrences]
                a_tpl = _substitute_multi_time(a_tpl, times)
                question = q_tpl.format(word=phrase)  # q templates don't use {time}
                answer = a_tpl.format(word=phrase)
                pure_answer = _pure_times(times)
            else:
                fmt = {"word": phrase, "time": answer_start, "sentence": sentence}
                question = q_tpl.format(**fmt)
                answer = a_tpl.format(**fmt)
                pure_answer = f"{answer_start:.1f}s"
    elif task == "word2time_first":
        lookup_tokens = [t.lower() for t in phrase.split() if _is_alnum_token(t)]
        occurrences = find_all_occurrences(lookup_tokens, w2t)
        if not occurrences:
            return None
        first_idx, first_time = min(occurrences, key=lambda io: io[1])
        if random.random() < RANGE_MODE_PROB:
            first_end = _phrase_end_time(w2t, first_idx, len(lookup_tokens))
            if first_end is None:
                first_end = first_time
            fmt = {"word": phrase, "time": first_time, "end_time": first_end}
            question = choose_template(QUESTION_TEMPLATES_RANGE[task], single_token).format(**fmt)
            answer = choose_template(ANSWER_TEMPLATES_RANGE[task], single_token).format(**fmt)
            pure_answer = f"{first_time:.1f}s-{first_end:.1f}s"
        else:
            fmt = {"word": phrase, "time": first_time}
            question = choose_template(QUESTION_TEMPLATES[task], single_token).format(**fmt)
            answer = choose_template(ANSWER_TEMPLATES[task], single_token).format(**fmt)
            pure_answer = f"{first_time:.1f}s"
    elif task == "word2sentence":
        lookup_tokens = [t.lower() for t in phrase.split() if _is_alnum_token(t)]
        occurrences = find_all_occurrences(lookup_tokens, w2t)
        q_tpl = choose_template(QUESTION_TEMPLATES[task], single_token)
        unique_sentences = _dedup_sentences_for_occurrences(w2t, occurrences) if len(occurrences) >= 2 else []
        if len(unique_sentences) >= 2:
            # Multi-substitution relies on the surrounding `"..."` to render
            # `"S1", "S2", and "S3"`, so restrict to answer templates that wrap
            # `{sentence}` in double quotes.
            multi_safe = [t for t in ANSWER_TEMPLATES[task] if '"{sentence}"' in t[0]]
            a_tpl = choose_template(multi_safe, single_token)
            a_tpl = _substitute_multi_sentence(a_tpl, unique_sentences)
            question = q_tpl.format(word=phrase)  # q templates don't use {sentence}
            answer = a_tpl.format(word=phrase)
            pure_answer = "|".join(unique_sentences)
        else:
            a_tpl = choose_template(ANSWER_TEMPLATES[task], single_token)
            # Single occurrence OR multiple hits inside one sentence -> normal flow.
            fmt = {"word": phrase, "sentence": sentence}
            question = q_tpl.format(**fmt)
            answer = a_tpl.format(**fmt)
            pure_answer = sentence
    elif task == "time2word":
        fmt = {"word": phrase, "time": answer_start, "sentence": sentence}
        question = choose_template(QUESTION_TEMPLATES[task], single_token).format(**fmt)
        answer = choose_template(ANSWER_TEMPLATES[task], single_token).format(**fmt)
        pure_answer = phrase
    elif task == "time2sentence":
        fmt = {"word": phrase, "time": answer_start, "sentence": sentence}
        question = choose_template(QUESTION_TEMPLATES[task], single_token).format(**fmt)
        answer = choose_template(ANSWER_TEMPLATES[task], single_token).format(**fmt)
        pure_answer = sentence
    else:
        fmt = {"word": phrase, "time": answer_start, "sentence": sentence}
        question = choose_template(QUESTION_TEMPLATES[task], single_token).format(**fmt)
        answer = choose_template(ANSWER_TEMPLATES[task], single_token).format(**fmt)
        pure_answer = answer

    if style == "pure":
        return _wrap_record(record, task, document_audio, duration, question, pure_answer, suffix="_pure")
    return _wrap_record(record, task, document_audio, duration, question, answer)


def process_split(input_file: Path, output_file: Path, task: str, negative_rate: float = 0.0, style: str = "chat") -> tuple[int, int, int, int]:
    kept = dropped = neg = trailing_skipped = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with input_file.open("r", encoding="utf-8") as fin, output_file.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            gap = _trailing_gap(record)
            if gap is not None and gap[0] > TRAILING_GAP_TOLERANCE_S:
                trailing_skipped += 1
                continue
            out = build_variant(record, task, style=style)
            if out is None:
                dropped += 1
                continue
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            kept += 1
            if negative_rate > 0 and task in NEGATIVE_QUESTION_TEMPLATES \
                    and random.random() < negative_rate:
                neg_out = build_negative_variant(record, task, style=style)
                if neg_out is not None:
                    fout.write(json.dumps(neg_out, ensure_ascii=False) + "\n")
                    neg += 1
    return kept, dropped, neg, trailing_skipped


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="Directory with train/validation/test jsonl files")
    parser.add_argument("--output_dir", type=str, default="output_slu_variants")
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        choices=TASKS,
        default=list(TASKS),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--negative_rate",
        type=float,
        default=0.1,
        help="For word2time/word2sentence/time2word/time2sentence, fraction of "
             "kept positive samples that also get an additional negative sample "
             "appended (OOV word or out-of-range timestamp). Negatives are added, "
             "not substituted.",
    )
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

    total_trailing_skipped = 0
    for task in args.tasks:
        for split_file in split_files:
            split = split_file.stem
            out_file = output_root / task / f"{split}.jsonl"
            kept, dropped, neg, trailing_skipped = process_split(split_file, out_file, task, negative_rate=args.negative_rate)
            total_trailing_skipped += trailing_skipped
            print(
                f"[{task}/{split}] kept={kept} negatives={neg} dropped={dropped} "
                f"trailing_skipped={trailing_skipped} -> {out_file}"
            )
            # verified_test gets a second "pure" file with literal short-form
            # answers (e.g. "10.2s,25.6s") for canonical eval comparison.
            if split == "verified_test":
                pure_file = output_root / task / f"{split}_pure.jsonl"
                kept_p, dropped_p, neg_p, trailing_p = process_split(
                    split_file, pure_file, task, negative_rate=args.negative_rate, style="pure"
                )
                print(
                    f"[{task}/{split}_pure] kept={kept_p} negatives={neg_p} dropped={dropped_p} "
                    f"trailing_skipped={trailing_p} -> {pure_file}"
                )

    print(
        f"Total rows discarded due to trailing gap > {TRAILING_GAP_TOLERANCE_S}s: "
        f"{total_trailing_skipped}"
    )
    (output_root / "completed.txt").touch()
