"""Multi-turn conversations from SLUE-phase-2 SQA5, using its word alignment.

No LLM: every answer is computed from the alignment shipped with the corpus, so
it is exact by construction.

Source
------
``raw/misc/en/slu-phase-2-sqa5/{split}.jsonl`` -- 15,883 distinct spoken
Wikipedia excerpts of 40s, each with a full word-level alignment (word /
start_second / end_second, coverage 1.00 of the transcript), the reading speaker
and, per row, a SQuAD-style question in text AND as a separate audio file.

What is used and what is not
---------------------------
Everything derived from the alignment is ground truth and is used freely.

The QA pairs are NOT. SQA5 attaches a question to a document by matching the
answer STRING, not the topic: measured lexical overlap between question and
document has a median of 0.25 on train against 0.50 on the human-verified
split. Training on that teaches confident answers about content the audio does
not hold. QA turns are therefore emitted only for rows above
``--qa-overlap-min`` (0.5 by default, i.e. the verified median, ~12% of rows),
and that is a proxy for relevance, not a guarantee.

Two further tasks were dropped after measurement rather than shipped:

* "transcribe what follows" -- document ids look sequential (``Cheese_45``,
  ``Cheese_46``) and 8,904 consecutive pairs exist, always same speaker, but on
  14 pairs inspected all 14 have a gap: ``Cheese_45`` ends on "have a protein
  structure that" and ``Cheese_46`` opens on "about 82°C". Consecutive ids are
  not audio-adjacent, unlike MLS chapters.
* "same speaker on both clips?" -- one article is read by exactly one speaker,
  so same-speaker and same-topic always coincide and the label is confounded.
* "how long is this audio?" -- 15,638 of 15,883 documents last exactly 40.0s.

Tasks
-----
Transcription (whole, time window, first/last N seconds, halves, thirds),
sentences (first/last/nth, the one containing a word, the one at a time, when
one starts), localisation (when a word is said, its span, all its occurrences),
structure (word counts, first/last word, word order, neighbouring word, speech
rate, longest pause, spoken numbers), plus linked chains where consecutive turns
zoom in on one anchor, a switch to a second audio, an optional spoken-question
QA opening, and a final synthesis turn.

Shapes and proportions are inherited from template_commonvoice_multiturn.

Usage
-----
    python template_slue_multiturn.py \\
        --slue-root $DATA_FOLDER/raw/misc/en/slu-phase-2-sqa5 \\
        --output-dir out/ --splits train validation test --mode all
"""

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from template_commonvoice_multiturn import (  # noqa: E402
    ALL_MODES, BUNDLE_SIZES, CONTINUATION_EXPLICIT, CONT_TURN_DIST,
    MIXED_COMPOUND_OF_REST, MIXED_HYBRID_SHARE, TURN_DIST, assistant,
    count_audios, draw_compound_position, draw_turns, oxford_join, user_audio,
    user_text,
)
from template_mls_multiturn import (  # noqa: E402
    A, ORDINALS, ORDINALS_M, apply_language_note, build_compound, build_sequential,
    bundle_turn, draw_instruction_lang, fmt, qa_pairs, round1, single_turn,
    synthesis_turn,
)
from template_mls_multiturn import T as MLS_T  # noqa: E402

INSTRUCTION_LANGS = ("fr", "en")
# SQA5 is read English throughout; the instruction follows it most of the time
# and states the language when it does not. See template_mls_multiturn.
AUDIO_LANG = "en"

# Punctuation is its own token in SQA5 ("levels", ".", "Other"), so a sentence
# closes on a bare "." and the transcript has to be re-glued to read normally.
SENT_END = {".", "!", "?"}
GLUE_LEFT = {".", ",", ";", ":", "!", "?", ")", "]", "%", "'s", "n't"}
OPEN_BRACKET = ("(", "[")

MIN_SENTENCES = 3
MIN_COVERAGE = 0.9
MIN_HALF_SECONDS = 20.0
MIN_THIRD_SECONDS = 35.0
MAX_WINDOW_SHARE = 0.4
MIN_PAUSE = 0.5

WINDOW_SECONDS = (5, 8, 10, 12, 15)
SENTENCE_ROUNDED_RATE = 0.4


# ── Templates ────────────────────────────────────────────────────────────────
# Most phrasings are shared with the MLS generator; only what SQA5 adds or has
# to say differently lives here.

SLUE_T = {
    "word_order": {
        "fr": ["« {a} » est-il prononcé avant ou après « {b} » ?",
               "Lequel vient en premier, « {a} » ou « {b} » ?"],
        "en": ["Is \"{a}\" spoken before or after \"{b}\"?",
               "Which comes first, \"{a}\" or \"{b}\"?"],
    },
    "word_next": {
        "fr": ["Quel mot suit immédiatement « {w} » ?",
               "Qu'est-ce qui est dit juste après « {w} » ?"],
        "en": ["Which word comes right after \"{w}\"?",
               "What is said immediately after \"{w}\"?"],
    },
    "word_previous": {
        "fr": ["Quel mot précède « {w} » ?", "Qu'entend-on juste avant « {w} » ?"],
        "en": ["Which word comes just before \"{w}\"?",
               "What is heard right before \"{w}\"?"],
    },
    "count_words_between": {
        "fr": ["Combien de mots sont prononcés entre {a} et {b} secondes ?"],
        "en": ["How many words are spoken between {a} and {b} seconds?"],
    },
    "speech_rate": {
        "fr": ["À quel rythme parle le locuteur, en mots par minute ?",
               "Quel est le débit de parole, en mots par minute ?"],
        "en": ["How fast does the speaker talk, in words per minute?",
               "What is the speech rate, in words per minute?"],
    },
    "longest_pause": {
        "fr": ["Où se situe la plus longue pause ?",
               "À quel moment le locuteur marque-t-il la plus longue pause ?"],
        "en": ["Where is the longest pause?",
               "At what point does the speaker pause the longest?"],
    },
    "numbers_spoken": {
        "fr": ["Quels nombres sont prononcés dans cet extrait ?",
               "Relève les nombres que l'on entend."],
        "en": ["Which numbers are spoken in this excerpt?",
               "List the numbers that can be heard."],
    },
    # The excerpts are cut mid-article, so "the first sentence" is often a
    # fragment. Saying so in the question avoids teaching the model to invent a
    # clean opening.
    "sentence_first": {
        "fr": ["Quelle est la première phrase, même si elle est tronquée ?",
               "Donne la phrase d'ouverture telle qu'elle est entendue.",
               "Par quoi commence l'extrait ?"],
        "en": ["What is the first sentence, even if it is cut off?",
               "Give the opening sentence as heard.",
               "How does the excerpt begin?"],
    },
    "qa_answer": {
        "fr": ["Réponds à cette question à partir de l'audio : {q}",
               "En t'appuyant sur l'enregistrement : {q}",
               "Écoute l'extrait et réponds : {q}"],
        "en": ["Answer this question from the audio: {q}",
               "Based on the recording: {q}",
               "Listen to the excerpt and answer: {q}"],
    },
    "qa_spoken": {
        "fr": ["Voici une question, puis l'enregistrement. Réponds à partir de l'audio.",
               "Écoute cette question, puis l'extrait, et réponds.",
               "La question est posée à l'oral. Réponds en te basant sur l'enregistrement qui suit."],
        "en": ["Here is a question, then the recording. Answer from the audio.",
               "Listen to this question, then the excerpt, and answer.",
               "The question is spoken. Answer using the recording that follows."],
    },
    "qa_time": {
        "fr": ["À quel moment la réponse est-elle prononcée ?",
               "Et quand cela est-il dit ?"],
        "en": ["When is the answer spoken?", "And at what point is that said?"],
    },
    "qa_sentence": {
        "fr": ["Donne-moi la phrase complète d'où vient cette réponse.",
               "Et la phrase entière qui contient la réponse ?"],
        "en": ["Give me the full sentence the answer comes from.",
               "And the whole sentence containing the answer?"],
    },
}
T = {**MLS_T, **SLUE_T}

SLUE_A = {
    "order": {"fr": ["« {a} » d'abord, à {ta}s, puis « {b} » à {tb}s.",
                     "« {a} » vient avant, {ta}s contre {tb}s."],
              "en": ["\"{a}\" first, at {ta}s, then \"{b}\" at {tb}s.",
                     "\"{a}\" comes first, {ta}s against {tb}s."]},
    "rate": {"fr": ["Environ {n} mots par minute.", "À peu près {n} mots/minute."],
             "en": ["About {n} words per minute.", "Roughly {n} words/minute."]},
    "pause": {"fr": ["Une pause de {d}s à {t}s.", "La plus longue est à {t}s, {d}s."],
              "en": ["A {d}s pause at {t}s.", "The longest is at {t}s, {d}s long."]},
    "list": {"fr": ["{items}", "On entend {items}."],
             "en": ["{items}", "The excerpt has {items}."]},
}
ANS = {**A, **SLUE_A}

# Follow-up phrasings for the linked chains: each step points back at what the
# previous turn just returned, which is the whole point of the shape.
CH = {
    "zoom_sentence": {
        "fr": ["Et la phrase complète autour de ce mot ?",
               "Donne-moi la phrase entière dans laquelle il apparaît."],
        "en": ["And the full sentence around that word?",
               "Give me the whole sentence it appears in."],
    },
    "zoom_window": {
        "fr": ["Élargis : transcris les {n} secondes autour de ce moment.",
               "Et le passage autour, sur {n} secondes ?"],
        "en": ["Widen it: transcribe the {n} seconds around that point.",
               "And the passage around it, over {n} seconds?"],
    },
    "locate_sentence": {
        "fr": ["Dans quelle phrase se trouve-t-il ?",
               "Et la phrase qui le contient ?"],
        "en": ["Which sentence is it in?", "And the sentence containing it?"],
    },
    "locate_next": {
        "fr": ["Quel mot vient juste après ?", "Et le mot suivant ?"],
        "en": ["Which word comes right after it?", "And the next word?"],
    },
}

# Transition to the second audio, when the conversation switches.
SWITCH = {
    "fr": ["Passons à un autre extrait.", "Voici un autre enregistrement.",
           "Changeons d'audio."],
    "en": ["Let us move to another excerpt.", "Here is another recording.",
           "Let us switch to a different audio."],
}
# Same article, same reader, further along or further back -- but NOT contiguous,
# so the wording claims topical relation only, never adjacency.
SWITCH_SAME = {
    "fr": ["Voici un autre passage du même article.",
           "Restons sur le même sujet : voici un autre extrait de la même lecture.",
           "Toujours le même article, un autre passage."],
    "en": ["Here is another passage from the same article.",
           "Staying on the same subject: another excerpt from the same reading.",
           "Same article, a different passage."],
}
ORDER_IN_ARTICLE = {
    "fr": ["Lequel des deux extraits vient le plus tôt dans l'article ?",
           "Dans l'ordre de lecture, lequel des deux passages arrive en premier ?"],
    "en": ["Which of the two excerpts comes earlier in the article?",
           "In reading order, which of the two passages comes first?"],
}
ORDER_ANSWER = {
    "fr": ["Le {ord}.", "Le {ord} des deux."],
    "en": ["The {ord} one.", "The {ord} of the two."],
}


# ── Loading ──────────────────────────────────────────────────────────────────

def is_word(token):
    return any(c.isalnum() for c in token)


def detok(tokens):
    """Re-glue the token stream into readable text."""
    out = ""
    for token in tokens:
        if not out:
            out = token
        elif token in GLUE_LEFT or (token.startswith("'") and len(token) <= 3):
            out += token
        elif out.endswith(OPEN_BRACKET):
            out += token
        else:
            out += " " + token
    return out


def split_sentences(tokens, starts, ends):
    """Cut the aligned token stream into sentences, keeping each one's span."""
    out, cur = [], []
    for token, a, b in zip(tokens, starts, ends):
        cur.append((token, a, b))
        if token in SENT_END or (len(token) > 1 and token[-1] in SENT_END):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    sentences = []
    for group in out:
        spoken = [x for x in group if is_word(x[0])]
        if not spoken:
            continue
        sentences.append((detok([x[0] for x in group]), spoken[0][1], spoken[-1][2]))
    return sentences


STOP = set("the a an of in on at to for and or is are was were be been by with from "
           "as that this it its his her their what which who when where how many much "
           "did do does have has had if not no than then there some".split())


def overlap(question, document):
    """Share of the question's content words present in the document."""
    def bag(text):
        return {w for w in re.findall(r"[a-z0-9']+", text.lower())
                if w not in STOP and len(w) > 2}
    q = bag(question)
    return len(q & bag(document)) / len(q) if q else 0.0


def load_documents(root, split):
    """One record per document audio, with every question attached to it."""
    path = root / f"{split}.jsonl"
    if not path.exists():
        return []
    records = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            meta = row["custom_metadata"]
            doc = meta["document_id"]
            record = records.get(doc)
            if record is None:
                audio = next((t for t in row["conversations"]
                              if t.get("type") == "audio"), None)
                if audio is None:
                    continue
                w2t = meta.get("word2time") or {}
                tokens, starts, ends = (w2t.get("word"), w2t.get("start_second"),
                                        w2t.get("end_second"))
                if not tokens or not (len(tokens) == len(starts) == len(ends)):
                    continue
                duration = audio.get("duration") or 0.0
                sentences = split_sentences(tokens, starts, ends)
                if len(sentences) < MIN_SENTENCES:
                    continue
                # Coverage is read off the last SPOKEN token: punctuation is
                # timestamped -1.0 and would sink every document below the bar.
                if duration <= 0 or sentences[-1][2] / duration < MIN_COVERAGE:
                    continue
                record = records[doc] = {
                    "id": doc, "audio": audio["value"], "duration": duration,
                    "tokens": tokens, "starts": starts, "ends": ends,
                    "sentences": sentences,
                    # Built from the sentences, not the raw token stream: an
                    # excerpt cut mid-article often opens on a stray full stop.
                    "transcript": " ".join(s[0] for s in sentences),
                    "speaker": meta.get("document_speaker_id"),
                    "article": doc.rsplit("_", 1)[0],
                    # Chunk indices skip, so two excerpts of one article are not
                    # contiguous -- but they do keep the reading order.
                    "chunk": int(doc.rsplit("_", 1)[1]) if doc.rsplit("_", 1)[1].isdigit() else -1,
                    "questions": [],
                }

            spans = meta.get("answer_spans") or {}
            answers = spans.get("answer") or []
            question = meta.get("raw_question_text")
            if not answers or not question:
                continue
            # answer_spans carries the normalised form ("chicago"); the aligned
            # tokens over the same span give it back its case and punctuation.
            answer = answers[0]
            start, end = spans["start_second"][0], spans["end_second"][0]
            over = [t for t, s, e in zip(record["tokens"], record["starts"], record["ends"])
                    if s >= start - 1e-6 and e <= end + 1e-6]
            if over and " ".join(over).lower() == answer.lower():
                answer = detok(over)

            record["questions"].append({
                "text": question,
                "audio": meta.get("question_audio_filepath"),
                "audio_duration": meta.get("question_audio_duration"),
                "answer": answer,
                "start": start,
                "end": end,
                "overlap": overlap(question, record["transcript"]),
            })
    return list(records.values())


# ── Task generation on one document ──────────────────────────────────────────

def window_text(record, a, b):
    """Tokens spoken inside [a, b].

    Punctuation carries a -1.0 timestamp in SQA5, so it can only be placed by
    adjacency: a mark is kept when the word it follows is kept, otherwise every
    window comes back stripped of its commas and full stops while the full
    transcript keeps them.
    """
    tokens, starts, ends = record["tokens"], record["starts"], record["ends"]
    kept, previous = [], False
    for token, s, e in zip(tokens, starts, ends):
        if is_word(token):
            previous = s >= a - 1e-6 and e <= b + 1e-6
        if previous:
            kept.append(token)
    return detok(kept)


def sentence_window(record, a, b):
    return [t for t, s, e in record["sentences"] if a <= (s + e) / 2 <= b]


def word_index(record, min_len=5):
    """Positions of content words long enough to be a usable target."""
    return [i for i, t in enumerate(record["tokens"])
            if len(t.strip(".,;:!?()\"'")) >= min_len and t.strip(".,;:!?()\"'").isalpha()]


def clean(token):
    return token.strip(".,;:!?()\"'»«")


def occurrences(record, word):
    low = word.lower()
    return [(s, e) for t, s, e in zip(record["tokens"], record["starts"], record["ends"])
            if clean(t).lower() == low]


def unambiguous_starts(record):
    """Words whose start is far enough from its neighbours' to be asked about.

    "Which word starts at 27.3s?" has two answers when the next word begins
    0.04s later; 5% of unfiltered draws were ambiguous that way.
    """
    spoken = [i for i, t in enumerate(record["tokens"]) if is_word(t)]
    starts = record["starts"]
    return [i for k, i in enumerate(spoken)
            if (k == 0 or starts[i] - starts[spoken[k - 1]] >= 0.15)
            and (k == len(spoken) - 1 or starts[spoken[k + 1]] - starts[i] >= 0.15)]


def sentence_at(record, t):
    for text, s, e in record["sentences"]:
        if s <= t <= e:
            return text
    return min(record["sentences"], key=lambda x: abs((x[1] + x[2]) / 2 - t))[0]


def task_pool(record, lang):
    """Every (name, question, answer, standalone question) this audio supports."""
    out = []
    duration = record["duration"]
    sentences = record["sentences"]
    tokens, starts, ends = record["tokens"], record["starts"], record["ends"]
    spoken = [i for i, t in enumerate(tokens) if is_word(t)]

    out.append(("transcribe_all", fmt(T["transcribe_all"], lang), record["transcript"]))

    windows = [n for n in WINDOW_SECONDS if n <= duration * MAX_WINDOW_SHARE]
    if windows:
        n = random.choice(windows)
        rounded = random.random() < SENTENCE_ROUNDED_RATE
        body = (" ".join(sentence_window(record, 0, n)) if rounded
                else window_text(record, 0, n))
        if body:
            key = "window_first_sent" if rounded else "window_first"
            out.append(("window_first", fmt(T[key], lang, n=n), body))
        m = random.choice(windows)
        rounded = random.random() < SENTENCE_ROUNDED_RATE
        body = (" ".join(sentence_window(record, duration - m, duration)) if rounded
                else window_text(record, duration - m, duration))
        if body:
            key = "window_last_sent" if rounded else "window_last"
            out.append(("window_last", fmt(T[key], lang, n=m), body,
                        fmt(T[key], lang, standalone=True, n=m)))

    if duration > 25:
        a = round(random.uniform(2, duration - 15))
        b = a + random.choice([8, 10, 12])
        body = window_text(record, a, b)
        if body:
            out.append(("window_between", fmt(T["window_between"], lang, a=a, b=b), body))
            count = len([t for t in window_text(record, a, b).split() if is_word(t)])
            if count:
                out.append(("count_words_between",
                            fmt(T["count_words_between"], lang, a=a, b=b),
                            fmt(ANS["count"], lang, n=count)))

    mid = duration / 2
    first_half, second_half = sentence_window(record, 0, mid), sentence_window(record, mid, duration)
    if duration >= MIN_HALF_SECONDS and len(first_half) >= 2 and len(second_half) >= 2:
        out.append(("half_first", fmt(T["half_first"], lang), " ".join(first_half)))
        out.append(("half_second", fmt(T["half_second"], lang), " ".join(second_half),
                    fmt(T["half_second"], lang, standalone=True)))

    thirds = [sentence_window(record, duration * i / 3, duration * (i + 1) / 3)
              for i in range(3)]
    if duration >= MIN_THIRD_SECONDS and all(len(t) >= 2 for t in thirds):
        k = random.randrange(3)
        out.append(("third", fmt(T["third"], lang, ord=ORDINALS_M[lang][k]),
                    " ".join(thirds[k])))

    out.append(("sentence_first", fmt(T["sentence_first"], lang), sentences[0][0]))
    out.append(("sentence_last", fmt(T["sentence_last"], lang), sentences[-1][0],
                fmt(T["sentence_last"], lang, standalone=True)))
    if len(sentences) >= 3:
        k = random.randrange(1, min(len(sentences), len(ORDINALS[lang])) - 1)
        out.append(("sentence_nth", fmt(T["sentence_nth"], lang, ord=ORDINALS[lang][k]),
                    sentences[k][0]))
        out.append(("sentence_start_time",
                    fmt(T["sentence_start_time"], lang, ord=ORDINALS[lang][k]),
                    fmt(ANS["time"], lang, t=round1(sentences[k][1]))))

    text, s, e = random.choice(sentences)
    out.append(("sentence_at_time",
                fmt(T["sentence_at_time"], lang, t=round1((s + e) / 2)), text))

    unambiguous = unambiguous_starts(record)
    if unambiguous:
        i = random.choice(unambiguous)
        out.append(("word_at_time", fmt(T["word_at_time"], lang, t=round1(starts[i])),
                    clean(tokens[i])))

    candidates = word_index(record)
    if candidates:
        i = random.choice(candidates)
        target = clean(tokens[i])
        hits = occurrences(record, target)
        if hits:
            out.append(("word_time", fmt(T["word_time"], lang, w=target),
                        fmt(ANS["time"], lang, t=round1(hits[0][0]))))
            out.append(("word_span", fmt(T["word_span"], lang, w=target),
                        fmt(ANS["span"], lang, a=round1(hits[0][0]), b=round1(hits[0][1]))))
            if len(hits) > 1:
                times = oxford_join([f"{round1(a)}s" for a, _ in hits], lang)
                out.append(("word_occurrences", fmt(T["word_occurrences"], lang, w=target),
                            fmt(ANS["occurrences"], lang, n=len(hits), times=times)))
        holder = next((t for t, _, _ in sentences if target.lower() in t.lower()), None)
        if holder:
            out.append(("sentence_with_word", fmt(T["sentence_with_word"], lang, w=target),
                        holder))
        # Neighbours are only unambiguous when the anchor occurs once.
        if len(hits) == 1:
            after = next((j for j in spoken if j > i), None)
            before = next((j for j in reversed(spoken) if j < i), None)
            if after is not None:
                out.append(("word_next", fmt(T["word_next"], lang, w=target),
                            clean(tokens[after])))
            if before is not None:
                out.append(("word_previous", fmt(T["word_previous"], lang, w=target),
                            clean(tokens[before])))

    # Order needs two distinct anchors, each occurring once, far enough apart.
    unique = [i for i in candidates if len(occurrences(record, clean(tokens[i]))) == 1]
    if len(unique) >= 2:
        i, j = sorted(random.sample(unique, 2))
        if starts[j] - starts[i] > 2.0:
            first, second = clean(tokens[i]), clean(tokens[j])
            pair = [(first, starts[i]), (second, starts[j])]
            random.shuffle(pair)
            out.append(("word_order",
                        fmt(T["word_order"], lang, a=pair[0][0], b=pair[1][0]),
                        fmt(ANS["order"], lang, a=first, ta=round1(starts[i]),
                            b=second, tb=round1(starts[j]))))

    out.append(("count_sentences", fmt(T["count_sentences"], lang),
                fmt(ANS["count"], lang, n=len(sentences))))
    out.append(("first_word", fmt(T["first_word"], lang), clean(tokens[spoken[0]])))
    out.append(("last_word", fmt(T["last_word"], lang), clean(tokens[spoken[-1]]),
                fmt(T["last_word"], lang, standalone=True)))

    rate = round(len(spoken) / duration * 60 / 5) * 5
    out.append(("speech_rate", fmt(T["speech_rate"], lang), fmt(ANS["rate"], lang, n=rate)))

    gaps = [(starts[b] - ends[a], ends[a])
            for a, b in zip(spoken, spoken[1:]) if starts[b] - ends[a] >= MIN_PAUSE]
    if gaps:
        gap, at = max(gaps)
        out.append(("longest_pause", fmt(T["longest_pause"], lang),
                    fmt(ANS["pause"], lang, d=round1(gap), t=round1(at))))

    numbers = [(clean(t), s) for t, s in zip(tokens, starts)
               if clean(t) and any(c.isdigit() for c in clean(t))]
    if numbers:
        items = oxford_join([f"{w} ({round1(s)}s)" for w, s in numbers], lang)
        out.append(("numbers_spoken", fmt(T["numbers_spoken"], lang),
                    fmt(ANS["list"], lang, items=items)))

    if windows:
        n = min(windows)
        count = len([t for t in window_text(record, 0, n).split() if is_word(t)])
        if count:
            out.append(("count_words_window", fmt(T["count_words_window"], lang, n=n),
                        fmt(ANS["count"], lang, n=count)))

    return [t if len(t) == 4 else (t[0], t[1], t[2], t[1]) for t in out]


# ── Linked chains ────────────────────────────────────────────────────────────
# Consecutive turns about the SAME anchor, each one referring back to the
# previous answer. This is the shape the models fail at, so it is generated
# explicitly rather than hoped for as a by-product of drawing several tasks.

def chain_zoom(record, lang):
    """time -> word -> its sentence -> the passage around it."""
    inner = [i for i in unambiguous_starts(record)
             if 3 < record["starts"][i] < record["duration"] - 3]
    if not inner:
        return []
    i = random.choice(inner)
    t = record["starts"][i]
    items = [("word_at_time", fmt(T["word_at_time"], lang, t=round1(t)),
              clean(record["tokens"][i]), fmt(T["word_at_time"], lang, t=round1(t)))]
    holder = sentence_at(record, t)
    items.append(("sentence_at_time", fmt(CH["zoom_sentence"], lang), holder,
                  fmt(T["sentence_at_time"], lang, t=round1(t))))
    n = random.choice([6, 8, 10])
    body = window_text(record, max(0, t - n / 2), min(record["duration"], t + n / 2))
    if body:
        items.append(("window_between", fmt(CH["zoom_window"], lang, n=n), body,
                      fmt(T["window_between"], lang, a=round1(max(0, t - n / 2)),
                          b=round1(min(record["duration"], t + n / 2)))))
    return items


def chain_locate(record, lang):
    """word -> when it is said -> which sentence -> what follows it."""
    candidates = [i for i in word_index(record)
                  if len(occurrences(record, clean(record["tokens"][i]))) == 1]
    if not candidates:
        return []
    i = random.choice(candidates)
    target = clean(record["tokens"][i])
    items = [("word_time", fmt(T["word_time"], lang, w=target),
              fmt(ANS["time"], lang, t=round1(record["starts"][i])),
              fmt(T["word_time"], lang, w=target))]
    holder = next((t for t, _, _ in record["sentences"] if target.lower() in t.lower()), None)
    if holder:
        items.append(("sentence_with_word", fmt(CH["locate_sentence"], lang), holder,
                      fmt(T["sentence_with_word"], lang, w=target)))
    spoken = [j for j, t in enumerate(record["tokens"]) if is_word(t)]
    after = next((j for j in spoken if j > i), None)
    if after is not None:
        items.append(("word_next", fmt(CH["locate_next"], lang),
                      clean(record["tokens"][after]),
                      fmt(T["word_next"], lang, w=target)))
    return items


def chain_qa(record, lang, question, spoken):
    """question -> answer -> when -> the source sentence.

    The only shape that uses the SQuAD pairing, hence the overlap gate upstream.
    When `spoken`, the opening turn only announces the question; the question
    itself is inserted as audio by the caller.
    """
    opening = (fmt(T["qa_spoken"], lang) if spoken
               else fmt(T["qa_answer"], lang, q=question["text"]))
    items = [("qa_answer", opening, question["answer"], opening)]
    items.append(("qa_time", fmt(T["qa_time"], lang),
                  fmt(ANS["time"], lang, t=round1(question["start"])),
                  fmt(T["qa_time"], lang, standalone=True)))
    holder = sentence_at(record, question["start"])
    items.append(("qa_sentence", fmt(T["qa_sentence"], lang), holder,
                  fmt(T["qa_sentence"], lang, standalone=True)))
    return items


# ── Assembly ─────────────────────────────────────────────────────────────────

# Phrasings that read as follow-ups and must not open a conversation, plus the
# pairs where one only makes sense after the other.
NOT_OPENERS = {"window_last", "sentence_last", "half_second", "last_word",
               "qa_time", "qa_sentence"}
REQUIRES = {"half_second": "half_first", "qa_time": "qa_answer",
            "qa_sentence": "qa_answer"}


def select_tasks(pool, n):
    """Draw n distinct tasks, keeping openers first.

    Deduplicated by ANSWER as well as by task: on a short excerpt the first
    half, the first third and the opening sentence can be the same text, and
    three turns asking differently for one answer teach nothing.
    """
    by_name, seen = {}, set()
    for item in pool:
        if item[2] in seen:
            continue
        seen.add(item[2])
        by_name.setdefault(item[0], item)
    names = list(by_name)
    random.shuffle(names)

    openers = [x for x in names if x not in NOT_OPENERS]
    if not openers:
        return []
    first = random.choice(openers)
    chosen = [first] + [x for x in names if x != first][:max(0, n - 1)]

    for name, needed in REQUIRES.items():
        if name in chosen and needed not in chosen:
            chosen.remove(name)
    if "half_first" in chosen and "half_second" in chosen:
        chosen.remove("half_second")
        chosen.insert(chosen.index("half_first") + 1, "half_second")
    return [by_name[name] for name in chosen[:n]]


def build_hybrid(items, lang, audio, trace=None):
    bundle_size = 2 if len(items) < 4 else random.choice(BUNDLE_SIZES)
    n_turns = len(items) - bundle_size + 1
    if n_turns < 2:
        return None
    at = draw_compound_position(n_turns)
    if at == 0 and any(t[0] in NOT_OPENERS for t in items[:bundle_size]):
        at = 1
    turns, cursor = [], 0
    for i in range(n_turns):
        take = bundle_size if i == at else 1
        group = items[cursor:cursor + take]
        cursor += take
        clip = audio if i == 0 else None
        if take == 1:
            question = group[0][3] if i == 0 else group[0][1]
            turns += single_turn(question, group[0][2], lang, i == 0, clip, trace)
        else:
            turns += bundle_turn(group, lang, clip, trace)
    return turns


def build_mixed(items, lang, audio, trace=None, p_hybrid=MIXED_HYBRID_SHARE):
    if len(items) >= 3 and random.random() < p_hybrid:
        turns = build_hybrid(items, lang, audio, trace)
        if turns is not None:
            return turns
        if trace is not None:
            del trace[:]
    if random.random() < MIXED_COMPOUND_OF_REST:
        return build_compound(items, lang, audio, trace)
    return build_sequential(items, lang, audio, trace)


BUILDERS = {"sequential": build_sequential, "compound": build_compound,
            "mixed": build_mixed}


def build_chain_turns(items, lang, audio, trace=None):
    """Emit a linked chain: no "another question" lead, since each step already
    points back at the previous answer."""
    turns = []
    for i, item in enumerate(items):
        if trace is not None:
            trace.append(1)
        question = item[3] if i == 0 else item[1]
        turns.append(user_text(question[0].upper() + question[1:]))
        if i == 0 and audio is not None:
            turns.append(audio)
        turns.append(assistant(item[2]))
    return turns


def append_block(turns, items, lang, mode, trace, p_hybrid):
    """Answer more tasks about the audio already in the conversation."""
    if not items:
        return
    block = (build_mixed(items, lang, None, trace, p_hybrid) if mode == "mixed"
             else BUILDERS[mode](items, lang, None, trace))
    if block:
        turns += block


def build_conversation(record, mode, lang, args, extra=None, p_hybrid=MIXED_HYBRID_SHARE):
    trace = []
    turns = []

    question = None
    if random.random() < args.qa_rate:
        usable = [q for q in record["questions"] if q["overlap"] >= args.qa_overlap_min]
        question = random.choice(usable) if usable else None

    chain, spoken = [], False
    if question is not None:
        spoken = bool(question.get("audio")) and random.random() < args.spoken_question_rate
        chain = chain_qa(record, lang, question, spoken)
    elif random.random() < args.chain_rate:
        chain = random.choice((chain_zoom, chain_locate))(record, lang)

    audio = user_audio(record["audio"], record["duration"])
    pool = task_pool(record, lang)
    n = draw_turns(len(pool), TURN_DIST) or 2

    if chain:
        turns += build_chain_turns(chain, lang, audio, trace)
        if spoken:
            # The question audio goes between the announcement and the document.
            turns.insert(1, user_audio(question["audio"], question["audio_duration"]))
        rest = select_tasks(pool, max(0, n - len(chain)))
        append_block(turns, rest, lang, mode, trace, p_hybrid)
    else:
        items = select_tasks(pool, n)
        if not items:
            return None
        block = (build_mixed(items, lang, audio, trace, p_hybrid) if mode == "mixed"
                 else BUILDERS[mode](items, lang, audio, trace))
        if not block:
            return None
        turns += block

    if extra is not None:
        second = task_pool(extra, lang)
        picked = select_tasks(second, draw_turns(len(second), CONT_TURN_DIST) or 1)
        if picked:
            same_article = extra["article"] == record["article"]
            block = build_sequential(picked, lang,
                                     user_audio(extra["audio"], extra["duration"]))
            lead = fmt(SWITCH_SAME if same_article else SWITCH, lang)
            block[0]["value"] = f"{lead} {block[0]['value']}"
            turns += block
            # Only meaningful when both excerpts come from one reading.
            if same_article and record["chunk"] >= 0 and extra["chunk"] >= 0 \
                    and record["chunk"] != extra["chunk"]:
                question = fmt(ORDER_IN_ARTICLE, lang)
                first = record["chunk"] < extra["chunk"]
                turns += [user_text(question),
                          # Masculine: it qualifies "extrait", not "phrase".
                          assistant(fmt(ORDER_ANSWER, lang,
                                        ord=ORDINALS_M[lang][0 if first else 1]))]

    if random.random() < args.synthesis_rate:
        pairs = qa_pairs(turns)
        if len(pairs) >= 2:
            turns += synthesis_turn(pairs, lang)
    return turns


# ── Driver ───────────────────────────────────────────────────────────────────

def generate(args):
    root = Path(args.slue_root)
    out_root = Path(args.output_dir)
    modes = ALL_MODES if args.mode == "all" else [args.mode]

    for split in args.splits:
        records = load_documents(root, split)
        qa_ready = sum(1 for r in records
                       if any(q["overlap"] >= args.qa_overlap_min for q in r["questions"]))
        print(f"[{split}] {len(records)} documents exploitables, "
              f"{qa_ready} avec une question au-dessus du seuil de recouvrement",
              file=sys.stderr)
        if not records:
            continue
        random.shuffle(records)

        by_article = defaultdict(list)
        for i, r in enumerate(records):
            by_article[r["article"]].append(i)
        articles = list(by_article)

        capable = sum(1 for r in records if len(r["sentences"]) >= 3) / len(records)
        p_hybrid = min(1.0, MIXED_HYBRID_SHARE / capable) if capable else 0.0

        writers, counters = {}, {}
        for mode in modes:
            path = out_root / mode / f"{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            writers[mode] = path.open("w", encoding="utf-8")
            counters[mode] = 0

        stats = {m: {"audios": Counter(), "turns": Counter(), "family": Counter()}
                 for m in modes}

        for index, record in enumerate(records):
            if all(counters[m] >= args.max_samples for m in modes):
                break
            lang = draw_instruction_lang(AUDIO_LANG, args.instruction_match_rate)
            extra = None
            if random.random() < args.multi_audio_rate and len(records) > 1:
                # Two relations are on offer: an unrelated excerpt (new topic,
                # new voice) or another passage of the SAME reading, which is
                # related without being contiguous.
                siblings = [i for i in by_article[record["article"]]
                            if records[i]["id"] != record["id"]]
                if siblings and random.random() < args.same_article_rate:
                    extra = records[random.choice(siblings)]
                elif len(articles) > 1:
                    other = random.choice([a for a in articles if a != record["article"]])
                    extra = records[random.choice(by_article[other])]

            for mode in modes:
                if counters[mode] >= args.max_samples:
                    continue
                turns = build_conversation(record, mode, lang, args, extra, p_hybrid)
                if not turns:
                    continue
                apply_language_note(turns, lang, AUDIO_LANG, args.name_language_rate)
                writers[mode].write(json.dumps(
                    {"id": f"{record['id']}_slue_{counters[mode]}_{mode}",
                     "conversations": turns}, ensure_ascii=False) + "\n")
                counters[mode] += 1
                s = stats[mode]
                s["audios"][count_audios(turns)] += 1
                s["turns"][min(sum(1 for t in turns
                                   if t["from"] == "User" and t["type"] == "text"), 8)] += 1

        for w in writers.values():
            w.close()
        for mode in modes:
            total = counters[mode]
            print(f"[{split}]   {mode}: {total} conversations -> "
                  f"{out_root / mode / f'{split}.jsonl'}", file=sys.stderr)
            if not total:
                continue

            def pct(c):
                return "  ".join(f"{k}:{v / total * 100:.0f}%" for k, v in sorted(c.items()))
            print(f"[{split}]     audios par conversation: {pct(stats[mode]['audios'])}",
                  file=sys.stderr)
            print(f"[{split}]     tours utilisateur: {pct(stats[mode]['turns'])}",
                  file=sys.stderr)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slue-root", required=True,
                   help="Directory holding {train,validation,test,verified_test}.jsonl")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--splits", nargs="+", default=["train"])
    p.add_argument("--mode", default="all", choices=ALL_MODES + ["all"])
    p.add_argument("--instruction-match-rate", type=float, default=0.75,
                   help="How often the instruction is written in English, the "
                        "language of the audio, as in the production manifests.")
    p.add_argument("--name-language-rate", type=float, default=1.0,
                   help="When the instruction is in French, how often the audio "
                        "language is stated rather than left implicit.")
    p.add_argument("--chain-rate", type=float, default=0.35,
                   help="Share of conversations opening on a linked chain, where each "
                        "turn refers back to the previous answer.")
    p.add_argument("--qa-rate", type=float, default=0.25,
                   help="Share of conversations attempting a SQuAD QA opening; only "
                        "documents holding a question above --qa-overlap-min qualify.")
    p.add_argument("--qa-overlap-min", type=float, default=0.5,
                   help="Minimum share of the question's content words present in the "
                        "document. 0.5 is the median of the human-verified split; "
                        "train's median is 0.25. Set to 0 to keep every pair.")
    p.add_argument("--spoken-question-rate", type=float, default=0.4,
                   help="Within QA openings, how often the question is given as audio "
                        "instead of text.")
    p.add_argument("--multi-audio-rate", type=float, default=0.15,
                   help="Share of conversations switching to a second document.")
    p.add_argument("--same-article-rate", type=float, default=0.4,
                   help="Within those, how often the second excerpt comes from the "
                        "same article and reader -- related, but not contiguous, so "
                        "it also carries a 'which comes earlier?' turn.")
    p.add_argument("--synthesis-rate", type=float, default=0.15,
                   help="Share of conversations closing on a turn that only reformats "
                        "the exchange as JSON, a table or a list.")
    p.add_argument("--max-samples", type=int, default=1_000_000, help="Per mode, per split")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    generate(args)
