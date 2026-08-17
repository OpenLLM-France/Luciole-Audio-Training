"""Multi-turn conversations from Multilingual LibriSpeech, using its word alignment.

No LLM: every answer is computed from the forced alignment shipped with the
corpus, so it is exact by construction. MLS is CC BY 4.0.

Two sources, two conversation families
--------------------------------------
* ``concatenated/`` -- 80k chained audios of ~45s with a full transcript, a
  word-level alignment (word / start_second / end_second) and the speaker id.
  Everything that happens INSIDE one audio comes from here: windows, sentences,
  word localisation, counting.
* ``train/segments.txt`` -- the original 258k utterances with their position in
  the chapter. Sorting a chapter by start time gives 248k pairs whose gap is
  under 0.5s, i.e. audio that genuinely continues the previous one. That is what
  makes "now transcribe what follows" answerable, and it is the multi-turn shape
  where the audio changes every turn but the turns are chained.

Tasks
-----
windows (first/last N seconds, between A and B, halves, thirds), sentences
(first/last/nth, the one containing a word, the one at a time), localisation
(when is a word said, all its occurrences, when a sentence starts), structure
(how many sentences, first/last word, duration), continuation (transcribe what
follows / what preceded / do these two follow each other), comparison across
audios (same speaker, same book, which is longer), and a final synthesis turn
that reformats the whole conversation as JSON, a table or a list.

Shapes and proportions are inherited from template_commonvoice_multiturn.

Usage
-----
    python template_mls_multiturn.py \\
        --mls-root $DATA_FOLDER/raw/transcript/multilang/MultilingualLibriSpeech/mls_french \\
        --output-dir out/ --splits train dev test --mode all
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
    EXTRA_CLIP_DIST, MIXED_COMPOUND_OF_REST, MIXED_HYBRID_SHARE, TURN_DIST,
    assistant, count_audios, draw_compound_position, draw_turns, oxford_join,
    user_audio, user_text,
)

INSTRUCTION_LANGS = ("fr", "en")
SENTENCE_END = (".", "!", "?", "…")

# The instruction language follows the audio most of the time, and when it does
# not, the audio language is ALWAYS stated. Measured over nemo/asr/*/context --
# 408 manifests, 65.4M rows -- 92.9% of prompts are written in the language of
# the audio and every cross-language one names it or points at "its original
# language": the share left indeterminate is 0.00%. Drawing the note instead of
# forcing it would make these generators the only ones in the mix to break that.
LANG_NAME = {
    "fr": {"fr": "français", "en": "anglais", "de": "allemand", "es": "espagnol",
           "it": "italien", "nl": "néerlandais", "pt": "portugais", "pl": "polonais"},
    "en": {"fr": "French", "en": "English", "de": "German", "es": "Spanish",
           "it": "Italian", "nl": "Dutch", "pt": "Portuguese", "pl": "Polish"},
}
CROSS_LANG_NOTE = {
    "fr": ["L'audio est en {lang}.", "Enregistrement en {lang}.",
           "Cet extrait est en {lang}."],
    "en": ["The audio is in {lang}.", "Recording in {lang}.",
           "This excerpt is in {lang}."],
}
MLS_LANG_CODES = {"french": "fr", "english": "en", "german": "de", "spanish": "es",
                  "italian": "it", "dutch": "nl", "portuguese": "pt", "polish": "pl"}


def draw_instruction_lang(audio_lang, match_rate):
    """Instruction language, biased towards the language of the audio."""
    if audio_lang in INSTRUCTION_LANGS and random.random() < match_rate:
        return audio_lang
    other = [l for l in INSTRUCTION_LANGS if l != audio_lang]
    return random.choice(other or list(INSTRUCTION_LANGS))


def apply_language_note(turns, ilang, audio_lang, rate):
    """State the audio language when the instruction is not in it."""
    if ilang == audio_lang or random.random() >= rate:
        return
    name = LANG_NAME.get(ilang, {}).get(audio_lang)
    if not name or not turns or turns[0]["type"] != "text":
        return
    note = random.choice(CROSS_LANG_NOTE[ilang]).format(lang=name)
    turns[0]["value"] = f"{note} {turns[0]['value']}"

# Minimum quality bar. The coverage test matters: on ~10% of chains the
# alignment stops before the audio does, and a question about "the last
# seconds" would then get a truncated answer presented as complete.
MIN_SENTENCES = 3
MIN_COVERAGE = 0.9

# A half is only a meaningful ask on an audio long enough to have a middle, and a
# third needs three parts that each hold something. A window must stay a genuine
# excerpt rather than most of the audio.
MIN_HALF_SECONDS = 20.0
MIN_THIRD_SECONDS = 35.0
MAX_WINDOW_SHARE = 0.4

# A window is drawn from these, capped by the audio duration.
WINDOW_SECONDS = (5, 8, 10, 12, 15, 20)

# Share of window questions that ask for whole sentences rather than a strict
# time cut. A strict cut is exact but slices mid-clause; both phrasings exist so
# neither reads as the only way to ask.
SENTENCE_ROUNDED_RATE = 0.4

# Feminine in French, because they qualify "phrase". "tiers" is masculine and
# needs its own list, or the templates produce "le première tiers".
ORDINALS = {
    "fr": ["première", "deuxième", "troisième", "quatrième", "cinquième",
           "sixième", "septième", "huitième", "neuvième", "dixième"],
    "en": ["first", "second", "third", "fourth", "fifth",
           "sixth", "seventh", "eighth", "ninth", "tenth"],
}
ORDINALS_M = {
    "fr": ["premier", "deuxième", "troisième"],
    "en": ORDINALS["en"][:3],
}

# ── Templates ────────────────────────────────────────────────────────────────
# Three phrasings per task per language. `{}` placeholders are named.

T = {
    "transcribe_all": {
        "fr": ["Transcris cet audio.", "Écris ce qui est dit dans cet enregistrement.",
               "Donne-moi la transcription complète."],
        "en": ["Transcribe this audio.", "Write down what is said in this recording.",
               "Give me the full transcription."],
    },
    # The two MLS sources do not share a convention: concatenated/ ships restored
    # text (89% punctuated, 88% with capitals) while train/transcripts.txt is raw
    # (87% neither). Chapter walks read the raw one, so their prompts have to
    # announce it, exactly as the `nocasepunc` bucket of the ASR banks does --
    # otherwise "Transcris cet audio." means punctuation in one family and none
    # in the other, and punctuation looks random.
    "transcribe_raw": {
        "fr": ["Transcris cet audio, en minuscules et sans ponctuation.",
               "Écris ce qui est dit dans cet enregistrement, sans majuscules ni ponctuation.",
               "Donne-moi la transcription complète, tout en minuscules et sans ponctuation."],
        "en": ["Transcribe this audio in lowercase, without punctuation.",
               "Write down what is said in this recording, no capitals and no punctuation.",
               "Give me the full transcription, all lowercase and without punctuation."],
    },
    "window_first": {
        "fr": ["Transcris les {n} premières secondes.",
               "Que dit-on dans les {n} premières secondes ?",
               "Donne-moi le début, les {n} premières secondes."],
        "en": ["Transcribe the first {n} seconds.",
               "What is said in the first {n} seconds?",
               "Give me the opening, the first {n} seconds."],
    },
    "window_last": {
        "fr": ["Transcris les {n} dernières secondes.",
               "Et les {n} dernières secondes ?",
               "Que dit-on sur les {n} dernières secondes ?"],
        "en": ["Transcribe the last {n} seconds.",
               "And the last {n} seconds?",
               "What is said in the final {n} seconds?"],
    },
    "window_between": {
        "fr": ["Que dit-on entre {a} et {b} secondes ?",
               "Transcris le passage entre {a}s et {b}s.",
               "Donne-moi ce qui est prononcé de {a} à {b} secondes."],
        "en": ["What is said between {a} and {b} seconds?",
               "Transcribe the passage from {a}s to {b}s.",
               "Give me what is spoken between {a} and {b} seconds."],
    },
    "window_first_sent": {
        "fr": ["Transcris les {n} premières secondes, en phrases entières.",
               "Donne le début jusque vers {n} secondes, sans couper de phrase."],
        "en": ["Transcribe the first {n} seconds, whole sentences only.",
               "Give the opening up to about {n} seconds, without cutting a sentence."],
    },
    "window_last_sent": {
        "fr": ["Transcris la fin, les {n} dernières secondes environ, en phrases entières.",
               "Donne les dernières phrases, celles des {n} dernières secondes."],
        "en": ["Transcribe the ending, roughly the last {n} seconds, whole sentences.",
               "Give the closing sentences, those in the last {n} seconds."],
    },
    "half_first": {
        "fr": ["Transcris la première moitié de l'audio.",
               "Donne-moi la première moitié de ce qui est dit."],
        "en": ["Transcribe the first half of the audio.",
               "Give me the first half of what is said."],
    },
    "half_second": {
        "fr": ["Et la deuxième moitié ?", "Maintenant la seconde moitié.",
               "Transcris la deuxième moitié."],
        "en": ["And the second half?", "Now the second half.",
               "Transcribe the second half."],
    },
    "third": {
        "fr": ["Transcris le {ord} tiers de l'audio.",
               "Donne-moi le {ord} tiers de ce qui est dit."],
        "en": ["Transcribe the {ord} third of the audio.",
               "Give me the {ord} third of what is said."],
    },
    "sentence_first": {
        "fr": ["Quelle est la première phrase ?", "Donne-moi la phrase d'ouverture.",
               "Par quelle phrase commence l'audio ?"],
        "en": ["What is the first sentence?", "Give me the opening sentence.",
               "Which sentence does the audio start with?"],
    },
    "sentence_last": {
        "fr": ["Et la dernière phrase ?", "Quelle est la phrase finale ?",
               "Par quoi se termine l'audio ?"],
        "en": ["And the last sentence?", "What is the final sentence?",
               "How does the audio end?"],
    },
    "sentence_nth": {
        "fr": ["Donne-moi la {ord} phrase.", "Quelle est la {ord} phrase ?",
               "Transcris la {ord} phrase."],
        "en": ["Give me the {ord} sentence.", "What is the {ord} sentence?",
               "Transcribe the {ord} sentence."],
    },
    "sentence_at_time": {
        "fr": ["Quelle phrase est prononcée vers {t}s ?",
               "Écoute vers {t} secondes : quelle est la phrase complète ?",
               "Donne-moi la phrase que l'on entend à {t}s."],
        "en": ["Which sentence is spoken around {t}s?",
               "Listen around {t} seconds: what is the full sentence?",
               "Give me the sentence heard at {t}s."],
    },
    "sentence_start_time": {
        "fr": ["À quel moment commence la {ord} phrase ?",
               "Quand débute la {ord} phrase ?"],
        "en": ["When does the {ord} sentence start?",
               "At what time does the {ord} sentence begin?"],
    },
    "sentence_with_word": {
        "fr": ["Dans quelle phrase entend-on « {w} » ?",
               "Transcris la phrase qui contient « {w} ».",
               "Quelle phrase comporte le mot « {w} » ?"],
        "en": ["In which sentence is \"{w}\" heard?",
               "Transcribe the sentence containing \"{w}\".",
               "Which sentence has the word \"{w}\"?"],
    },
    "word_at_time": {
        "fr": ["Quel mot est prononcé à {t} secondes ?",
               "Que dit le locuteur à {t}s ?", "Quel mot commence à {t}s ?"],
        "en": ["Which word is spoken at {t} seconds?",
               "What does the speaker say at {t}s?", "Which word starts at {t}s?"],
    },
    "word_time": {
        "fr": ["Quand « {w} » est-il prononcé pour la première fois ?",
               "À quel moment entend-on « {w} » ?",
               "Donne-moi l'instant où « {w} » est dit."],
        "en": ["When is \"{w}\" first spoken?",
               "At what time is \"{w}\" heard?",
               "Give me the moment \"{w}\" is said."],
    },
    "word_span": {
        "fr": ["« {w} » commence et finit à quel moment ?",
               "Donne le début et la fin du mot « {w} »."],
        "en": ["When does \"{w}\" start and end?",
               "Give the start and end of the word \"{w}\"."],
    },
    "word_occurrences": {
        "fr": ["Combien de fois « {w} » est-il prononcé ?",
               "« {w} » revient combien de fois ?"],
        "en": ["How many times is \"{w}\" spoken?",
               "How often does \"{w}\" come up?"],
    },
    "count_sentences": {
        "fr": ["Combien de phrases compte cet audio ?",
               "Il y a combien de phrases en tout ?"],
        "en": ["How many sentences does this audio have?",
               "How many sentences are there in total?"],
    },
    "first_word": {
        "fr": ["Quel est le premier mot prononcé ?", "Par quel mot cela commence-t-il ?"],
        "en": ["What is the first word spoken?", "Which word does it start with?"],
    },
    "last_word": {
        "fr": ["Et le dernier mot ?", "Quel est le tout dernier mot prononcé ?"],
        "en": ["And the last word?", "What is the very last word spoken?"],
    },
    "duration": {
        "fr": ["Quelle est la durée de cet audio ?", "Ça dure combien de temps ?"],
        "en": ["How long is this audio?", "What is the duration of this recording?"],
    },
    "count_words_window": {
        "fr": ["Combien de mots sont prononcés dans les {n} premières secondes ?"],
        "en": ["How many words are spoken in the first {n} seconds?"],
    },
}

# Answer phrasings for the computed values.
A = {
    "time": {"fr": ["{t}s", "À {t}s.", "Vers {t} secondes."],
             "en": ["{t}s", "At {t}s.", "Around {t} seconds."]},
    "span": {"fr": ["De {a}s à {b}s.", "Entre {a}s et {b}s."],
             "en": ["From {a}s to {b}s.", "Between {a}s and {b}s."]},
    "count": {"fr": ["{n}", "Il y en a {n}.", "{n} en tout."],
              "en": ["{n}", "There are {n}.", "{n} in total."]},
    "duration": {"fr": ["{d} secondes.", "Environ {d}s."],
                 "en": ["{d} seconds.", "About {d}s."]},
    "occurrences": {"fr": ["{n} fois, à {times}.", "{n} occurrences : {times}."],
                    "en": ["{n} times, at {times}.", "{n} occurrences: {times}."]},
}

# Continuation between two consecutive segments of a chapter.
CONT_NEXT = {
    "fr": ["Voici la suite : transcris-la.", "Et la suite de l'enregistrement ?",
           "Transcris ce qui vient juste après.", "Voici ce qui suit, transcris.",
           "Avançons d'un extrait. Transcris celui-ci.",
           "Et après ça, qu'entend-on ? Voici l'extrait."],
    "en": ["Here is what follows: transcribe it.", "And the rest of the recording?",
           "Transcribe what comes right after.", "Here is the continuation, transcribe it.",
           "One clip forward. Transcribe this one.",
           "And after that, what is said? Here is the clip."],
}
CONT_PREV = {
    "fr": ["Et ce qui précédait, le voici : transcris-le.",
           "Voici le passage juste avant. Transcris-le.",
           "Remontons : voici ce qui vient avant, transcris-le.",
           "Et juste avant, qu'entend-on ? Voici l'extrait.",
           "Voici ce qui précède ce passage. Transcris.",
           "Reculons d'un extrait. Transcris celui-ci."],
    "en": ["And here is what came before: transcribe it.",
           "Here is the passage just before. Transcribe it.",
           "Let us go back: here is what precedes, transcribe it.",
           "And just before that, what is said? Here is the clip.",
           "Here is what comes before this passage. Transcribe it.",
           "One clip back. Transcribe this one."],
}
# Names the LAST two clips, not "the first" and "the second": the conversation
# usually holds three or more by then.
CONT_CHECK = {
    "fr": ["Ce dernier extrait fait-il bien suite au précédent ?",
           "Est-ce que ce dernier passage enchaîne sur celui d'avant ?"],
    "en": ["Does this last clip follow on from the previous one?",
           "Does this last passage continue the one before it?"],
}
CONT_CHECK_ANSWER = {
    "fr": {True: ["Oui, le second enchaîne directement sur le premier.",
                  "Oui, ils se suivent."],
           False: ["Non, ils ne se suivent pas.",
                   "Non, le second ne fait pas suite au premier."]},
    "en": {True: ["Yes, the second follows directly from the first.",
                  "Yes, they are consecutive."],
           False: ["No, they do not follow each other.",
                   "No, the second is not the continuation of the first."]},
}

# Closing turn on a chapter walk: one continuous transcript of everything heard.
# In a backward walk the answer is the REVERSE of the order the clips were given
# in, which is the whole point of asking.
MERGE = {
    "fr": ["Maintenant regroupe les {n} extraits en une seule transcription continue.",
           "Donne-moi les {n} passages d'affilée, comme un seul texte.",
           "Reprends l'ensemble des {n} extraits en une transcription unique, dans l'ordre."],
    "en": ["Now merge the {n} excerpts into one continuous transcript.",
           "Give me the {n} passages in a row, as a single text.",
           "Put all {n} excerpts together as one transcript, in order."],
}

# Tasks a chapter-walk clip can support. These segments come from
# transcripts.txt, which has NO alignment, so nothing time-indexed is available:
# only what the transcript and the duration give. Without them a walk would be
# four transcriptions in a row, while the chain family mixes tasks freely.
WALK_T = {
    "count_words": {
        "fr": ["Combien de mots sont prononcés dans cet extrait ?",
               "Il y a combien de mots en tout ?"],
        "en": ["How many words are spoken in this clip?",
               "How many words are there in total?"],
    },
    "word_occurrences_text": {
        "fr": ["Combien de fois le mot « {w} » est-il dit dans cet extrait ?",
               "« {w} » revient combien de fois ici ?"],
        "en": ["How many times is the word \"{w}\" said in this clip?",
               "How often does \"{w}\" come up here?"],
    },
    "longest_word": {
        "fr": ["Quel est le mot le plus long de cet extrait ?",
               "Donne-moi le mot le plus long qu'on entend."],
        "en": ["What is the longest word in this clip?",
               "Give me the longest word you can hear."],
    },
    "contains_word": {
        "fr": ["Est-ce que le mot « {w} » est prononcé dans cet extrait ?",
               "Entend-on « {w} » ici ?"],
        "en": ["Is the word \"{w}\" spoken in this clip?",
               "Can \"{w}\" be heard here?"],
    },
}
YESNO = {
    "fr": {True: ["Oui.", "Oui, il est bien prononcé."],
           False: ["Non.", "Non, il n'apparaît pas."]},
    "en": {True: ["Yes.", "Yes, it is spoken."],
           False: ["No.", "No, it does not appear."]},
}

CONT_CHECK_BACK = {
    "fr": ["Ce dernier extrait précède-t-il bien celui d'avant ?",
           "Est-ce que ce dernier passage vient juste avant le précédent ?"],
    "en": ["Does this last clip come right before the previous one?",
           "Is this last passage the one just before it?"],
}
CONT_CHECK_BACK_ANSWER = {
    "fr": {True: ["Oui, il précède directement le précédent.",
                  "Oui, il vient juste avant."],
           False: ["Non, il ne vient pas juste avant.",
                   "Non, ce n'est pas le passage qui précède."]},
    "en": {True: ["Yes, it comes directly before the previous one.",
                  "Yes, it is the passage just before."],
           False: ["No, it does not come just before.",
                   "No, that is not the preceding passage."]},
}

# Comparison across audios.
SAME_SPEAKER = {
    "fr": ["Est-ce la même voix sur les deux extraits ?",
           "Les deux enregistrements ont-ils le même locuteur ?"],
    "en": ["Is it the same voice on both clips?",
           "Do both recordings have the same speaker?"],
}
SAME_SPEAKER_ANSWER = {
    "fr": {True: ["Oui, c'est le même locuteur.", "Oui, la même voix dans les deux cas."],
           False: ["Non, ce sont deux locuteurs différents.", "Non, les voix diffèrent."]},
    "en": {True: ["Yes, it is the same speaker.", "Yes, the same voice in both."],
           False: ["No, these are two different speakers.", "No, the voices differ."]},
}
LONGER = {
    "fr": ["Lequel des deux est le plus long ?", "Quel extrait dure le plus longtemps ?"],
    "en": ["Which of the two is longer?", "Which clip lasts longer?"],
}
LONGER_ANSWER = {
    "fr": ["Le {ord}, {d1}s contre {d2}s."],
    "en": ["The {ord} one, {d1}s against {d2}s."],
}

# Final synthesis turn: reformats what was already answered, nothing new from the audio.
SYNTH = {
    "json": {"fr": ["Récapitule tout ça en JSON.",
                    "Reprends l'ensemble de mes questions et de tes réponses en JSON."],
             "en": ["Sum all of that up as JSON.",
                    "Take every question and answer above and give me JSON."]},
    "table": {"fr": ["Mets tout ça dans un tableau.",
                     "Reprends l'ensemble sous forme de tableau."],
              "en": ["Put all of that in a table.", "Lay the whole thing out as a table."]},
    "list": {"fr": ["Résume l'échange en une liste à puces.",
                    "Reprends chaque point en liste."],
             "en": ["Summarise the exchange as a bullet list.",
                    "List each point back to me."]},
}

FOLLOWUP_LEADS = {
    "fr": [""] * 8 + ["Autre question : ", "Et aussi : ", "Ensuite : ",
                      "Toujours sur cet extrait : ", "Sur le même enregistrement : "],
    "en": [""] * 8 + ["Another question: ", "And also: ", "Next: ",
                      "Still on this clip: ", "About the same recording: "],
}

COMPOUND_FRAMES_ANY = {
    "fr": ["{tasks}", "{tasks}", "Sur cet extrait : {tasks}",
           "Plusieurs choses sur cet audio : {tasks}", "Écoute et réponds : {tasks}",
           "J'aimerais savoir, pour cet enregistrement : {tasks}"],
    "en": ["{tasks}", "{tasks}", "On this clip: {tasks}",
           "Several things about this audio: {tasks}", "Listen and answer: {tasks}",
           "I would like to know, for this recording: {tasks}"],
}
COMPOUND_FRAMES_TWO = {
    "fr": ["Deux choses sur cet extrait : {tasks}", "J'ai deux questions : {tasks}"],
    "en": ["Two things about this clip: {tasks}", "I have two questions: {tasks}"],
}


# ── Loading ──────────────────────────────────────────────────────────────────

def split_sentences(words, starts, ends):
    """Cut the aligned word stream into sentences, keeping each one's span."""
    out, cur, start = [], [], None
    for word, a, b in zip(words, starts, ends):
        if start is None:
            start = a
        cur.append(word)
        if word.endswith(SENTENCE_END):
            out.append((" ".join(cur), start, b))
            cur, start = [], None
    if cur:
        out.append((" ".join(cur), start, ends[-1]))
    return out


def load_chains(root, split):
    """Join plan/ and aligned/ into one record per concatenated audio."""
    plan = {}
    with (root / "concatenated" / "plan" / f"{split}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            plan[r["id"]] = r

    records = []
    with (root / "concatenated" / "aligned" / f"{split}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            meta = plan.get(r["id"])
            if meta is None:
                continue
            w = r["custom_metadata"]["word2time"]
            words, starts, ends = w["word"], w["start_second"], w["end_second"]
            if not words or len(words) != len(starts) or len(words) != len(ends):
                continue
            duration = meta.get("duration") or 0.0
            if duration <= 0 or ends[-1] / duration < MIN_COVERAGE:
                continue
            sentences = split_sentences(words, starts, ends)
            if len(sentences) < MIN_SENTENCES:
                continue
            audio = next((t["value"] for t in r["conversations"]
                          if t.get("type") == "audio"), None)
            transcript = next((t["value"] for t in r["conversations"]
                               if t.get("from") == "Assistant"), None)
            if not audio or not transcript:
                continue
            records.append({
                "id": r["id"], "audio": audio, "duration": duration,
                "words": words, "starts": starts, "ends": ends,
                "sentences": sentences, "transcript": transcript,
                "speaker": meta.get("speaker"), "book": meta.get("book"),
            })
    return records


def load_neighbours(root, split):
    """Consecutive utterance pairs, found by sorting each chapter by start time.

    Utterance INDICES are useless for this: segments of one (speaker, book) are
    scattered across chapters, and only 1.5% of consecutive indices are adjacent
    audio. Sorting by chapter and start time instead yields 248k pairs under a
    0.5s gap on the French train split.
    """
    seg_dir = root / split
    if not (seg_dir / "segments.txt").exists():
        return []
    text = {}
    with (seg_dir / "transcripts.txt").open(encoding="utf-8") as f:
        for line in f:
            sid, _, body = line.partition("\t")
            text[sid.strip()] = body.strip()

    chapters = defaultdict(list)
    with (seg_dir / "segments.txt").open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                parts = line.split()
            if len(parts) < 4:
                continue
            sid, url, start, end = parts[0], parts[1], float(parts[2]), float(parts[3])
            chapters[url].append((start, end, sid))

    def path_of(sid):
        speaker, book, _ = sid.split("_")
        return str(seg_dir / "audio" / speaker / book / f"{sid}.flac")

    runs = []
    for url, segs in chapters.items():
        segs.sort()
        run = [segs[0]]
        for previous, current in zip(segs, segs[1:]):
            if current[0] - previous[1] < 0.5:
                run.append(current)
            else:
                if len(run) > 1:
                    runs.append(run)
                run = [current]
        if len(run) > 1:
            runs.append(run)

    out = []
    for run in runs:
        items = []
        for start, end, sid in run:
            body = text.get(sid)
            if not body:
                items = []
                break
            items.append({"id": sid, "audio": path_of(sid), "duration": round(end - start, 2),
                          "transcript": body,
                          "speaker": sid.split("_")[0], "book": sid.split("_")[1]})
        if len(items) > 1:
            out.append(items)
    return out


# ── Task generation on one chain ─────────────────────────────────────────────

# A phrasing that opens on "And ..." only works as a follow-up: in a bundle, or
# as the first turn, it refers to something that was never said.
ELLIPTIC_PREFIX = ("et ", "and ", "maintenant", "now ")


def is_elliptic(text):
    return text.lower().startswith(ELLIPTIC_PREFIX)


def fmt(pool, lang, standalone=False, **kw):
    """Draw a phrasing, avoiding the elliptic ones when the turn must stand alone."""
    options = pool[lang]
    if standalone:
        plain = [o for o in options if not is_elliptic(o)]
        options = plain or options
    return random.choice(options).format(**kw)


def window_text(record, a, b):
    return " ".join(w for w, s, e in zip(record["words"], record["starts"], record["ends"])
                    if s >= a - 1e-6 and e <= b + 1e-6)


def sentence_window(record, a, b):
    """Whole sentences whose midpoint falls in the window."""
    return [t for t, s, e in record["sentences"] if a <= (s + e) / 2 <= b]


def sentence_window_text(record, a, b):
    return " ".join(sentence_window(record, a, b))


def pick_word(record, min_len=5):
    """A content word long enough to be a usable target."""
    candidates = [w.strip(".,;:!?»«'\"()") for w in record["words"]]
    candidates = [w for w in candidates if len(w) >= min_len and w.isalpha()]
    return random.choice(candidates) if candidates else None


def round1(x):
    return f"{x:.1f}".rstrip("0").rstrip(".") if x else "0"


def task_pool(record, lang):
    """Every (name, question, answer) this audio can support."""
    out = []
    duration = record["duration"]
    sentences = record["sentences"]
    words, starts, ends = record["words"], record["starts"], record["ends"]

    out.append(("transcribe_all", fmt(T["transcribe_all"], lang), record["transcript"]))

    windows = [n for n in WINDOW_SECONDS if n <= duration * MAX_WINDOW_SHARE]
    if windows:
        n = random.choice(windows)
        if random.random() < SENTENCE_ROUNDED_RATE:
            body = sentence_window_text(record, 0, n)
            if body:
                out.append(("window_first", fmt(T["window_first_sent"], lang, n=n), body))
        else:
            body = window_text(record, 0, n)
            if body:
                out.append(("window_first", fmt(T["window_first"], lang, n=n), body))
        m = random.choice(windows)
        if random.random() < SENTENCE_ROUNDED_RATE:
            body = sentence_window_text(record, duration - m, duration)
            if body:
                out.append(("window_last", fmt(T["window_last_sent"], lang, n=m), body,
                            fmt(T["window_last_sent"], lang, standalone=True, n=m)))
        else:
            body = window_text(record, duration - m, duration)
            if body:
                out.append(("window_last", fmt(T["window_last"], lang, n=m), body,
                            fmt(T["window_last"], lang, standalone=True, n=m)))

    if duration > 25:
        a = round(random.uniform(5, duration - 15))
        b = a + random.choice([8, 10, 12])
        body = window_text(record, a, b)
        if body:
            out.append(("window_between", fmt(T["window_between"], lang, a=a, b=b), body))

    # Halves and thirds need an audio long enough for the cut to mean something,
    # and at least two sentences on each side: a half made of one sentence is the
    # same question as "what is the opening sentence?".
    mid = duration / 2
    first_half = sentence_window(record, 0, mid)
    second_half = sentence_window(record, mid, duration)
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
        out.append(("sentence_nth",
                    fmt(T["sentence_nth"], lang, ord=ORDINALS[lang][k]), sentences[k][0]))
        out.append(("sentence_start_time",
                    fmt(T["sentence_start_time"], lang, ord=ORDINALS[lang][k]),
                    fmt(A["time"], lang, t=round1(sentences[k][1]))))

    k = random.randrange(len(sentences))
    text, s, e = sentences[k]
    t = round1((s + e) / 2)
    out.append(("sentence_at_time", fmt(T["sentence_at_time"], lang, t=t), text))

    i = random.randrange(len(words))
    out.append(("word_at_time", fmt(T["word_at_time"], lang, t=round1(starts[i])),
                words[i].strip(".,;:!?»«")))

    target = pick_word(record)
    if target:
        hits = [(s, e) for w, s, e in zip(words, starts, ends)
                if w.strip(".,;:!?»«'\"()").lower() == target.lower()]
        if hits:
            out.append(("word_time", fmt(T["word_time"], lang, w=target),
                        fmt(A["time"], lang, t=round1(hits[0][0]))))
            out.append(("word_span", fmt(T["word_span"], lang, w=target),
                        fmt(A["span"], lang, a=round1(hits[0][0]), b=round1(hits[0][1]))))
            if len(hits) > 1:
                times = oxford_join([f"{round1(s)}s" for s, _ in hits], lang)
                out.append(("word_occurrences", fmt(T["word_occurrences"], lang, w=target),
                            fmt(A["occurrences"], lang, n=len(hits), times=times)))
        holder = next((t for t, _, _ in sentences if target.lower() in t.lower()), None)
        if holder:
            out.append(("sentence_with_word", fmt(T["sentence_with_word"], lang, w=target),
                        holder))

    out.append(("count_sentences", fmt(T["count_sentences"], lang),
                fmt(A["count"], lang, n=len(sentences))))
    out.append(("first_word", fmt(T["first_word"], lang),
                words[0].strip(".,;:!?»«")))
    out.append(("last_word", fmt(T["last_word"], lang), words[-1].strip(".,;:!?»«"),
                fmt(T["last_word"], lang, standalone=True)))
    out.append(("duration", fmt(T["duration"], lang),
                fmt(A["duration"], lang, d=round1(duration))))

    if windows:
        n = min(windows)
        count = len(window_text(record, 0, n).split())
        if count:
            out.append(("count_words_window", fmt(T["count_words_window"], lang, n=n),
                        fmt(A["count"], lang, n=count)))
    # Every item is (name, question, answer, standalone question); the standalone
    # one is only different where the task has an elliptic phrasing.
    return [t if len(t) == 4 else (t[0], t[1], t[2], t[1]) for t in out]


# ── Rendering ────────────────────────────────────────────────────────────────

def audio_turn(record):
    turn = user_audio(record["audio"], record["duration"])
    return turn


def single_turn(question, answer, lang, first, audio=None, trace=None):
    if trace is not None:
        trace.append(1)
    lead = "" if first else random.choice(FOLLOWUP_LEADS[lang])
    turns = [user_text(lead + (question[0].upper() + question[1:]))]
    if audio is not None:
        turns.append(audio)
    turns.append(assistant(answer))
    return turns


def bundle_turn(items, lang, audio=None, trace=None):
    if len(items) < 2:
        return single_turn(items[0][3], items[0][2], lang, True, audio, trace)
    if trace is not None:
        trace.append(len(items))
    # Inside a bundle every question is read on its own, so the standalone
    # phrasing is used: "And the last word?" has nothing to refer back to there.
    tasks = " ".join(t[3][0].upper() + t[3][1:] for t in items)
    pool = list(COMPOUND_FRAMES_ANY[lang])
    if len(items) == 2:
        pool += COMPOUND_FRAMES_TWO[lang]
    turns = [user_text(random.choice(pool).format(tasks=tasks))]
    if audio is not None:
        turns.append(audio)
    style = random.choice(("numbered", "restated"))
    if style == "numbered":
        body = "\n".join(f"{i}. {t[2]}" for i, t in enumerate(items, 1))
    else:
        label = "Réponse : " if lang == "fr" else "Answer: "
        body = "\n".join(f"{t[3]}\n{label}{t[2]}" for t in items)
    turns.append(assistant(body))
    return turns


def synthesis_turn(pairs, lang):
    """A turn that only reformats what was already answered."""
    kind = random.choice(list(SYNTH))
    question = random.choice(SYNTH[kind][lang])
    if kind == "json":
        body = json.dumps([{"question": q, "answer": a} for q, a in pairs],
                          ensure_ascii=False, indent=2)
    elif kind == "table":
        head = ("| Question | Réponse |\n|---|---|" if lang == "fr"
                else "| Question | Answer |\n|---|---|")
        rows = "\n".join(f"| {q} | {a.splitlines()[0]} |" for q, a in pairs)
        body = f"{head}\n{rows}"
    else:
        body = "\n".join(f"- {q} {a.splitlines()[0]}" for q, a in pairs)
    return [user_text(question), assistant(body)]


# ── Assembly ─────────────────────────────────────────────────────────────────

# Tasks that read as follow-ups ("And the last sentence?") and must not open a
# conversation, plus the pairs where one only makes sense after the other.
NOT_OPENERS = {"window_last", "sentence_last", "half_second", "last_word"}
REQUIRES = {"half_second": "half_first"}


def select_tasks(pool, n):
    """Draw n distinct tasks, keeping openers first and linked pairs in order.

    Deduplicated by ANSWER as well as by task: on a short chain the first half,
    the first third and the opening sentence are the same text, and three turns
    asking differently for one answer teach nothing.
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


def build_sequential(items, lang, audio, trace=None):
    turns = []
    for i, item in enumerate(items):
        question = item[3] if i == 0 else item[1]
        turns += single_turn(question, item[2], lang, i == 0,
                             audio if i == 0 else None, trace)
    return turns


def build_compound(items, lang, audio, trace=None):
    return bundle_turn(items[:random.choice(BUNDLE_SIZES)], lang, audio, trace)


def build_hybrid(items, lang, audio, trace=None):
    bundle_size = 2 if len(items) < 4 else random.choice(BUNDLE_SIZES)
    n_turns = len(items) - bundle_size + 1
    if n_turns < 2:
        return None
    at = draw_compound_position(n_turns)
    if at == 0:
        # The bundle would open the conversation; its members must be openers.
        if any(t[0] in NOT_OPENERS for t in items[:bundle_size]):
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
        del trace[:]
    if random.random() < MIXED_COMPOUND_OF_REST:
        return build_compound(items, lang, audio, trace)
    return build_sequential(items, lang, audio, trace)


BUILDERS = {"sequential": build_sequential, "compound": build_compound,
            "mixed": build_mixed}


def walk_task(seg, lang, absent_pool):
    """One non-transcription turn about a clip, from its transcript alone."""
    words = [w for w in seg["transcript"].split() if w]
    if len(words) < 5:
        return None
    kind = random.choice(("count_words", "word_occurrences_text", "longest_word",
                          "contains_word", "duration"))
    if kind == "count_words":
        return (fmt(WALK_T["count_words"], lang), fmt(A["count"], lang, n=len(words)))
    if kind == "longest_word":
        return (fmt(WALK_T["longest_word"], lang), max(words, key=len))
    if kind == "duration":
        return (fmt(T["duration"], lang),
                fmt(A["duration"], lang, d=round1(seg["duration"])))
    target = random.choice([w for w in words if len(w) >= 5] or words)
    if kind == "word_occurrences_text":
        n = sum(1 for w in words if w.lower() == target.lower())
        return (fmt(WALK_T["word_occurrences_text"], lang, w=target),
                fmt(A["count"], lang, n=n))
    # contains_word: half the time ask about a word taken from another chapter,
    # so "no" is a real answer rather than a phrasing the model never sees.
    present = random.random() < 0.5 or not absent_pool
    if not present:
        target = random.choice(absent_pool)
        if any(w.lower() == target.lower() for w in words):
            present = True
    else:
        present = True
    return (fmt(WALK_T["contains_word"], lang, w=target),
            random.choice(YESNO[lang][present]))


def build_continuation(run, lang, args, decoy=None):
    """Walk consecutive segments of one chapter, one per turn.

    The direction is drawn at EVERY turn, not once for the walk: starting from a
    clip somewhere in the run, the block of heard audio grows to the left or to
    the right. So a conversation can ask for what follows, then for what came
    before, and each claim stays true because the new clip is always adjacent to
    the block's current edge. Drawing it once produced conversations where the
    same two phrasings alternated for four turns.
    """
    n = min(len(run), random.choices([2, 3, 4], weights=[0.4, 0.35, 0.25], k=1)[0])
    lo = random.randrange(0, len(run) - n + 1)
    hi = lo + n - 1
    left = right = random.randrange(lo, hi + 1)

    picked = [run[left]]
    steps = []
    while right - left + 1 < n:
        can_left, can_right = left > lo, right < hi
        go_left = can_left and (not can_right or random.random() < args.backward_rate)
        if go_left:
            left -= 1
            picked.append(run[left])
            steps.append("prev")
        else:
            right += 1
            picked.append(run[right])
            steps.append("next")

    absent = [w for w in (decoy or {}).get("transcript", "").split() if len(w) >= 6]

    def maybe_task(seg):
        """Slip a non-transcription turn in, so a walk is not four ASR turns."""
        if random.random() >= args.walk_task_rate:
            return
        item = walk_task(seg, lang, absent)
        if item:
            turns.extend([user_text(item[0]), assistant(item[1])])
            pairs.append(item)

    turns = [user_text(fmt(T["transcribe_raw"], lang)),
             user_audio(picked[0]["audio"], picked[0]["duration"]),
             assistant(picked[0]["transcript"])]
    pairs = [(turns[0]["value"], picked[0]["transcript"])]
    maybe_task(picked[0])
    for seg, step in zip(picked[1:], steps):
        question = random.choice((CONT_PREV if step == "prev" else CONT_NEXT)[lang])
        turns += [user_text(question), user_audio(seg["audio"], seg["duration"]),
                  assistant(seg["transcript"])]
        pairs.append((question, seg["transcript"]))
        maybe_task(seg)
    # The closing check is about the LAST clip, so it follows the last step.
    backward = bool(steps) and steps[-1] == "prev"
    lead_pool = CONT_PREV if backward else CONT_NEXT
    chronological = run[left:right + 1]

    # Sometimes close on the yes/no check, with a decoy for the negative case.
    # The check has to mirror the direction: in a backward walk the last clip
    # PRECEDES the one before it, so asking "does it follow?" would be false.
    spoiled = False
    if random.random() < args.continuation_check_rate:
        truthful = decoy is None or random.random() < 0.5
        if not truthful:
            turns += [user_text(random.choice(lead_pool[lang])),
                      user_audio(decoy["audio"], decoy["duration"]),
                      assistant(decoy["transcript"])]
            spoiled = True
        check = CONT_CHECK_BACK if backward else CONT_CHECK
        answers = CONT_CHECK_BACK_ANSWER if backward else CONT_CHECK_ANSWER
        question = random.choice(check[lang])
        answer = random.choice(answers[lang][truthful])
        turns += [user_text(question), assistant(answer)]
        pairs.append((question, answer))

    # Merge everything heard into one transcript. Skipped when a decoy was
    # slipped in: the conversation no longer holds a single continuous passage.
    if len(picked) >= 2 and not spoiled and random.random() < args.merge_rate:
        question = fmt(MERGE, lang, n=len(picked))
        answer = " ".join(seg["transcript"] for seg in chronological)
        turns += [user_text(question), assistant(answer)]
        pairs.append((question, answer))
    return turns, pairs


def comparison_turns(a, b, lang):
    """Questions that only exist because the conversation holds two audios."""
    out = []
    if a.get("speaker") and b.get("speaker"):
        same = a["speaker"] == b["speaker"]
        question = random.choice(SAME_SPEAKER[lang])
        out.append((question, random.choice(SAME_SPEAKER_ANSWER[lang][same])))
    if a.get("duration") and b.get("duration"):
        question = random.choice(LONGER[lang])
        first = a["duration"] >= b["duration"]
        answer = random.choice(LONGER_ANSWER[lang]).format(
            ord=ORDINALS[lang][0 if first else 1],
            d1=round1(max(a["duration"], b["duration"])),
            d2=round1(min(a["duration"], b["duration"])))
        out.append((question, answer))
    return out


def qa_pairs(turns):
    """(question, answer) pairs as they appear, for the synthesis turn."""
    out, pending = [], None
    for turn in turns:
        if turn["from"] == "User" and turn["type"] == "text":
            pending = turn["value"]
        elif turn["from"] == "Assistant" and pending is not None:
            out.append((pending, turn["value"]))
            pending = None
    return out


def build_conversation(mode, lang, args, chain=None, run=None, extra=None, decoy=None,
                       p_hybrid=MIXED_HYBRID_SHARE):
    """One conversation, either inside a chain or walking a chapter."""
    trace = []
    if run is not None:
        turns, _ = build_continuation(run, lang, args, decoy)
    else:
        pool = task_pool(chain, lang)
        n = draw_turns(len(pool), TURN_DIST) or 2
        items = select_tasks(pool, n)
        if not items:
            return None
        builder = BUILDERS[mode]
        audio = audio_turn(chain)
        turns = (build_mixed(items, lang, audio, trace, p_hybrid) if mode == "mixed"
                 else builder(items, lang, audio, trace))
        if not turns:
            return None

        if extra is not None:
            lead = random.choice(CONTINUATION_EXPLICIT[lang])
            second = task_pool(extra, lang)
            picked = select_tasks(second, draw_turns(len(second), CONT_TURN_DIST) or 1)
            if picked:
                block = build_sequential(picked, lang, audio_turn(extra))
                block[0]["value"] = f"{lead} {block[0]['value']}"
                turns += block
                for question, answer in comparison_turns(chain, extra, lang):
                    turns += [user_text(question), assistant(answer)]

    if random.random() < args.synthesis_rate:
        pairs = qa_pairs(turns)
        if len(pairs) >= 2:
            turns += synthesis_turn(pairs, lang)
    return turns


# ── Driver ───────────────────────────────────────────────────────────────────

def generate(args):
    root = Path(args.mls_root)
    out_root = Path(args.output_dir)
    modes = ALL_MODES if args.mode == "all" else [args.mode]
    audio_lang = args.audio_lang or MLS_LANG_CODES.get(
        root.name.replace("mls_", "").lower(), "")
    if not audio_lang:
        print(f"[!] langue de l'audio non déduite de {root.name}, "
              f"consigne tirée à 50/50", file=sys.stderr)

    for split in args.splits:
        chains = load_chains(root, split)
        runs = load_neighbours(root, split) if args.continuation_rate > 0 else []
        print(f"[{split}] {len(chains)} chaines exploitables, "
              f"{len(runs)} suites contigues", file=sys.stderr)
        if not chains:
            continue
        random.shuffle(chains)
        random.shuffle(runs)
        # One speaker holds 22k of the 80k chains, so drawing the second audio at
        # random makes "same speaker" the winning guess. Pick the pair explicitly.
        by_speaker = defaultdict(list)
        for i, c in enumerate(chains):
            by_speaker[c["speaker"]].append(i)
        speakers = [k for k in by_speaker if k]

        capable = sum(1 for c in chains if len(c["sentences"]) >= 3) / len(chains)
        p_hybrid = min(1.0, MIXED_HYBRID_SHARE / capable) if capable else 0.0

        writers, counters = {}, {}
        for mode in modes:
            path = out_root / mode / f"{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            writers[mode] = path.open("w", encoding="utf-8")
            counters[mode] = 0

        stats = {m: {"audios": Counter(), "turns": Counter(), "family": Counter(),
                     "synth": 0} for m in modes}
        run_cursor = 0
        for index, chain in enumerate(chains):
            if all(counters[m] >= args.max_samples for m in modes):
                break
            lang = draw_instruction_lang(audio_lang, args.instruction_match_rate)
            use_run = runs and random.random() < args.continuation_rate
            run = runs[run_cursor % len(runs)] if use_run else None
            if use_run:
                run_cursor += 1
            extra = None
            if not use_run and random.random() < args.multi_audio_rate:
                same = random.random() < 0.5
                pool = by_speaker.get(chain["speaker"], [])
                if same and len(pool) > 1:
                    j = random.choice([i for i in pool if i != index])
                elif not same and len(speakers) > 1:
                    other = random.choice([k for k in speakers if k != chain["speaker"]])
                    j = random.choice(by_speaker[other])
                else:
                    j = (index + 1) % len(chains)
                extra = chains[j]
            decoy = random.choice(runs)[0] if runs else None

            for mode in modes:
                if counters[mode] >= args.max_samples:
                    continue
                turns = build_conversation(mode, lang, args, chain=chain, run=run,
                                           extra=extra, decoy=decoy, p_hybrid=p_hybrid)
                if not turns:
                    continue
                apply_language_note(turns, lang, audio_lang, args.name_language_rate)
                stem = run[0]["id"] if use_run else chain["id"]
                writers[mode].write(json.dumps(
                    {"id": f"{stem}_mlschain_{counters[mode]}_{mode}",
                     "conversations": turns}, ensure_ascii=False) + "\n")
                counters[mode] += 1
                s = stats[mode]
                s["audios"][count_audios(turns)] += 1
                s["turns"][min(sum(1 for t in turns if t["from"] == "User"
                                   and t["type"] == "text"), 8)] += 1
                s["family"]["continuation" if use_run else "chain"] += 1

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
            s = stats[mode]
            print(f"[{split}]     familles: {pct(s['family'])}", file=sys.stderr)
            print(f"[{split}]     audios par conversation: {pct(s['audios'])}", file=sys.stderr)
            print(f"[{split}]     tours utilisateur: {pct(s['turns'])}", file=sys.stderr)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mls-root", required=True,
                   help="mls_<lang> directory, the one holding concatenated/ and train/")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--splits", nargs="+", default=["train"])
    p.add_argument("--mode", default="all", choices=ALL_MODES + ["all"])
    p.add_argument("--continuation-rate", type=float, default=0.25,
                   help="Share of conversations that walk consecutive segments of a "
                        "chapter instead of staying inside one concatenated audio.")
    p.add_argument("--audio-lang", default=None,
                   help="Language code of the audio. Deduced from the mls_<lang> "
                        "directory name when omitted.")
    p.add_argument("--instruction-match-rate", type=float, default=0.75,
                   help="How often the instruction is written in the language of the "
                        "audio, as in the production ASR manifests.")
    p.add_argument("--name-language-rate", type=float, default=1.0,
                   help="When the instruction is NOT in the audio's language, how "
                        "often the audio language is stated rather than left implicit.")
    p.add_argument("--walk-task-rate", type=float, default=0.35,
                   help="In a chapter walk, probability of following a clip's "
                        "transcription with a non-transcription turn about it "
                        "(word count, longest word, does it contain X, duration).")
    p.add_argument("--merge-rate", type=float, default=0.35,
                   help="Within chapter walks holding at least two clips, how often "
                        "the conversation closes on one continuous transcript of the "
                        "whole run -- in chronological order, so a backward walk has "
                        "to give it back reversed.")
    p.add_argument("--backward-rate", type=float, default=0.4,
                   help="At each step of a chapter walk, probability of extending the "
                        "block backwards (asking for what PRECEDED) rather than "
                        "forwards, when both directions are still available.")
    p.add_argument("--continuation-check-rate", type=float, default=0.3,
                   help="Within those, how often the conversation closes on 'do these "
                        "two follow each other?', half the time with a decoy segment.")
    p.add_argument("--multi-audio-rate", type=float, default=0.15,
                   help="Share of chain conversations that switch to a second audio.")
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
