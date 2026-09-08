"""Generate multi-turn and multi-instruction conversations from CommonVoice + AST.

No LLM: every answer is an existing annotation, every question a template.

What it generates
-----------------
Three files, one per --mode, over the same joined material:

* ``sequential`` -- the audio appears ONCE, in turn 1; later user turns are
  text-only and elliptical ("Et en allemand ?"), so they can only be answered by
  keeping the audio in context.
* ``compound`` -- one user turn carrying 2-3 instructions, answered in one go.
* ``mixed`` -- the one to train on: 40% sequential, 40% hybrid (one bundled turn
  among single ones), 20% compound alone.

Any of them may CHANGE AUDIO part-way through (--multi-audio-rate, 15%): the new
clip arrives either explicitly ("Passons a un autre extrait. <instruction>") or
elliptically ("Et celui-ci ?", --elliptic-rate), the latter carrying the previous
task over to the new clip. 50% of switches change source language too
(--cross-lang-rate), which is what puts non-French audio in these conversations.

A bare "Et celui-ci ?" is ambiguous whenever the carried task is a translation --
nothing in it says the answer should be German rather than the transcript -- so 70%
of those switches name the language instead ("Et celui-ci, en allemand ?",
--elliptic-lang-rate); same when a carried transcript lands on a clip in another
language. The remaining 30% keep the bare form, which is what trains the carry-over.

Proportions
-----------
* content turns: 2 turns 35%, 3 turns 30%, 4 turns 25%, 5 turns 10% (TURN_DIST)
* bundled turn at turn 1 40%, turn 2 30%, later 30% (COMPOUND_POS_*)
* audios per conversation: 1 85%, 2 11%, 3 4% (EXTRA_CLIP_DIST)
* speaker accent / gender / age asked in 5% of conversations each, at most two
  per conversation (--accent-rate / --gender-rate / --age-rate, 0 to disable)
* opening language fr 55%, en 35%, rest 10% (--lang-share)
* at most 5% of clip draws reuse an already-used clip (--reuse-rate)

Every rate is a share of the OUTPUT; the conditional probability needed to reach
it is derived at run time from the pool (weighted_coverage), because the pool is
15% French while the draw is 55% French and only French clips can answer more
than two things.

The join
--------
``ast/{src}-{tgt}/{variant}/CommonVoice{SRC}2{TGT}/`` holds the translations but
not the transcript, which lives in ``<cv_root>/<lang>/validated.tsv``. Both share
the clip stem (``common_voice_fr_19344297``); joining on it is 100% on fr->en,
against ~17% when joining AST manifests to each other. So a French clip carries
the transcript plus 7 translations, and an ar/de/en/es/it/nl/pt clip the
transcript plus the French one -- three distinct asks minimum either way.

Where the instructions come from
--------------------------------
Turn 1 is drawn from ``data/contexts`` (--contexts-root), never rewritten:

* transcription -- ``{src}_asr_contexts.json``. Its ``0.95`` bucket is written in the
  audio's OWN language, so it only applies on the diagonal (French instructions over
  French audio: 31 prompts, English over English: 38). Off the diagonal the ``0.05``
  bucket gives 4 per instruction language, all of which either name the audio language
  or point at "its original language".
* translation -- ``translation/{src}-{tgt}_ast_contexts.json``, 10 per instruction
  language on every one of the 56 pairs (20 counting fr and en together): 5 naming
  source and target, 5 naming the target alone. The 3 bare stubs ("Traduis") are
  skipped -- turn 1 has to say what language to answer in.

Every later turn is templated in this file, in fr/en only, which is why the
instruction language is drawn from {fr, en}.
``data/assets/instruction_transcription_and_translation_fr-en.txt`` is no longer used.

Usage
-----
    python template_commonvoice_multiturn.py \
        --cv-root  $DATA_FOLDER/raw/transcript/multilang/CommonVoice/cv-corpus-22.0-2025-06-20 \
        --ast-root $DATA_FOLDER/nemo/ast \
        --source-langs fr en de es it nl pt ar \
        --output-dir out/ --mode all
"""

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Language naming
# ──────────────────────────────────────────────────────────────────────────────

# Target-language NOUNS, as used after "en"/"into" ("en allemand", "into German").
LANG_NAMES = {
    "fr": {"ar": "arabe", "de": "allemand", "en": "anglais", "es": "espagnol",
           "fr": "français", "it": "italien", "nl": "néerlandais", "pt": "portugais"},
    "en": {"ar": "Arabic", "de": "German", "en": "English", "es": "Spanish",
           "fr": "French", "it": "Italian", "nl": "Dutch", "pt": "Portuguese"},
}

# Feminine adjectives, used after "version". French needs its own table: the noun
# would give "sa version portugais".
LANG_ADJ = {
    "fr": {"ar": "arabe", "de": "allemande", "en": "anglaise", "es": "espagnole",
           "fr": "française", "it": "italienne", "nl": "néerlandaise", "pt": "portugaise"},
    "en": dict(LANG_NAMES["en"]),
}

INSTRUCTION_LANGS = ("fr", "en")

ALL_MODES = ["sequential", "compound", "mixed"]


def fmt(template, tgt, ilang, **extra):
    """Fill a template with both the noun (`{lang}`) and the feminine adjective
    (`{adj}`) form of the target language."""
    return template.format(lang=LANG_NAMES[ilang][tgt], adj=LANG_ADJ[ilang][tgt], **extra)

# Templates. Every list is sampled uniformly and independently, for surface variety.

# Turn-1 instructions come from data/contexts: per audio language for ASR, per pair for
# AST. Both banks are already multilingual, so a prompt is TAKEN in the instruction
# language rather than written fr->en and string-substituted.
#
# They replace data/assets/instruction_transcription_and_translation_fr-en.txt, whose
# transcription section asked for things CommonVoice never contains ("include all pauses
# and hesitations", "including any background noises or interjections"): the reference is
# the written sentence being read aloud, so there are no disfluencies and no noise tags
# to produce, and the instruction trains the model to ignore what it is told. The
# combined transcribe-then-translate ask has no counterpart in data/contexts and is now
# assembled from the fragments in `compound_turns`, which covered every wider shape
# already.
CONTEXTS_RELPATH = ("contexts",)
ASR_BANK_NAME = "{lang}_asr_contexts.json"
AST_BANK_RELNAME = "translation/{src}-{tgt}_ast_contexts.json"
# ASR files split prompts into a bucket written in the audio's own language and a
# smaller cross-lingual one. The keys are the sampling weights they were generated
# with; we index them by role, not by weight.
ASR_SAME_LANG_BUCKET = "0.95"
ASR_CROSS_LANG_BUCKET = "0.05"
ASR_VARIANT = "default_contexts"   # `nocasepunc` asks for lowercase-no-punct output
AST_VARIANT = "default_contexts"

# The cross-lingual ASR bucket mixes every instruction language in one list, with no
# tag. These are the four templates it was generated from, per language: formatting
# them with the audio language's name reconstructs exactly the entries that belong to
# `ilang`. Two name the language, two point at it as "its original language" -- all
# four are answerable, which is why the thin pool is acceptable.
ASR_CROSS_LANG_TEMPLATES = {
    "fr": ["Transcrivez l'audio dans sa langue originelle.",
           "Écrivez exactement en {lang} ce qui est dit dans cet enregistrement audio.",
           "Transcris en {lang}",
           "Peux-tu rédiger le texte de ce fichier audio dans sa langue source?"],
    "en": ["Transcribe the audio in its original language.",
           "Write exactly in {lang} what is said in this audio recording.",
           "Transcribe in {lang}",
           "Can you write the text of this audio file in its source language?"],
}

# Elliptical follow-ups, the point of `sequential`: meaningless without the previous
# turns, so they force the model to keep attending to the audio.
FOLLOWUP_TRANSLATE = {
    "fr": [
        "Et en {lang} ?",
        "En {lang} maintenant.",
        "Maintenant en {lang}.",
        "Et sa version {adj} ?",
        "Pareil, mais en {lang}.",
        "Et si c'était en {lang} ?",
    ],
    "en": [
        "And in {lang}?",
        "Now in {lang}.",
        "What about {lang}?",
        "And the {lang} version?",
        "Same thing, in {lang}.",
        "Now give me that in {lang}.",
    ],
}

FOLLOWUP_TRANSCRIBE = {
    "fr": [
        "Et le texte d'origine, c'était quoi ?",
        "Redonne-moi ce qui est dit dans l'audio.",
        "Et la transcription ?",
        "Qu'est-ce qui était dit exactement ?",
    ],
    "en": [
        "And what was the original text?",
        "Remind me what the audio says.",
        "And the transcription?",
        "What exactly was said?",
    ],
}
FOLLOWUP_TRANSCRIBE_NAMED = {
    "fr": [
        "Et le texte {adj} d'origine, c'était quoi ?",
        "Redonne-moi ce qui est dit, en {lang}.",
        "Et la transcription {adj} ?",
    ],
    "en": [
        "And what was the original {lang} text?",
        "Remind me what the audio says, in {lang}.",
        "And the {lang} transcription?",
    ],
}
# Same job -- saying which language to answer in when the conversation has not
# established it -- but by POINTING at the audio instead of naming its language. The
# ask stays determinate while the language itself has to be recognised from the audio,
# which naming it hands over for free. Same phrasing the contexts ASR bank uses for its
# cross-lingual prompts ("Transcrivez l'audio dans sa langue originelle.").
FOLLOWUP_TRANSCRIBE_SOURCE = {
    "fr": [
        "Et la transcription, dans la langue de l'audio ?",
        "Redonne-moi ce qui est dit, dans la langue d'origine.",
        "Et le texte d'origine, dans la langue parlée dans l'extrait ?",
        "Qu'est-ce qui était dit, dans la langue de l'enregistrement ?",
        "Et la transcription dans sa langue source ?",
    ],
    "en": [
        "And the transcription, in the audio language?",
        "Remind me what the audio says, in the source audio language.",
        "And the original text, in the language spoken in the clip?",
        "What exactly was said, in the language of the recording?",
        "And the transcription in its source language?",
    ],
}
# Split between the two when the language is not established. Kept at half so neither
# shape becomes the rule: naming is the easier ask, pointing is the one that trains
# language identification.
FOLLOWUP_TRANSCRIBE_NAMED_SHARE = 0.5

# Accent, gender and age are NOT in the assistant's own previous answers, so they are
# the questions that force a return to the audio rather than to its last transcript.
FOLLOWUP_ACCENT = {
    "fr": [
        "Tu reconnais l'accent ?",
        "La personne a quel accent ?",
        "Et l'accent, tu dirais quoi ?",
        "D'où vient l'accent de la voix ?",
    ],
    "en": [
        "Can you place the accent?",
        "What accent does the speaker have?",
        "And the accent, what would you say?",
        "Where is the speaker's accent from?",
    ],
}

# CommonVoice labels are capitalised noun phrases ("Francais du Canada"), so they are
# quoted rather than inflected into the sentence.
ACCENT_ANSWERS = {
    "fr": ["L'accent est du type « {accent} ».", "Cela ressemble à un accent « {accent} ».",
           "Je dirais un accent « {accent} »."],
    "en": ["The accent is of the \"{accent}\" kind.", "It sounds like a \"{accent}\" accent.",
           "I would say a \"{accent}\" accent."],
}

# Gender and age are SELF-DECLARED and absent on most clips, hence the low rate and
# answers phrased as impressions rather than assertions.
FOLLOWUP_GENDER = {
    "fr": ["C'est une voix d'homme ou de femme ?", "Tu dirais un homme ou une femme ?",
           "La voix, plutôt masculine ou féminine ?"],
    "en": ["Is that a man's or a woman's voice?", "Would you say a man or a woman?",
           "Does the voice sound male or female?"],
}
GENDER_ANSWERS = {
    "fr": {"m": ["On dirait une voix d'homme.", "Plutôt une voix masculine."],
           "f": ["On dirait une voix de femme.", "Plutôt une voix féminine."]},
    "en": {"m": ["It sounds like a man's voice.", "Rather a male voice."],
           "f": ["It sounds like a woman's voice.", "Rather a female voice."]},
}

FOLLOWUP_AGE = {
    "fr": ["Tu dirais quel âge ?", "La personne a environ quel âge ?",
           "Quel âge donnerais-tu à cette voix ?"],
    "en": ["How old would you say they are?", "Roughly what age is the speaker?",
           "What age would you give that voice?"],
}
AGE_ANSWERS = {
    "fr": ["La voix semble être celle d'une personne {age}.", "Je dirais une personne {age}.",
           "Ça sonne comme une personne {age}."],
    "en": ["The speaker sounds like someone {age}.", "I would say someone {age}.",
           "It sounds like a person {age}."],
}
# CommonVoice age buckets are decades, so the answer stays a range.
AGE_LABELS = {
    "fr": {"teens": "de moins de vingt ans", "twenties": "entre vingt et trente ans",
           "thirties": "entre trente et quarante ans", "fourties": "entre quarante et cinquante ans",
           "fifties": "entre cinquante et soixante ans", "sixties": "entre soixante et soixante-dix ans",
           "seventies": "de plus de soixante-dix ans", "eighties": "de plus de quatre-vingts ans",
           "nineties": "de plus de quatre-vingt-dix ans"},
    "en": {"teens": "in their teens", "twenties": "in their twenties",
           "thirties": "in their thirties", "fourties": "in their forties",
           "fifties": "in their fifties", "sixties": "in their sixties",
           "seventies": "in their seventies", "eighties": "in their eighties",
           "nineties": "in their nineties"},
}

# Compound frames, split by whether they commit to a count: a counted frame is only
# drawn when the number of asks matches.
COMPOUND_FRAMES_ANY = {
    "fr": ["Sur cet extrait en {src} : {tasks}", "Pour cet audio en {src} : {tasks}",
           "À partir de cet enregistrement en {src} : {tasks}"],
    "en": ["On this {src} clip: {tasks}", "For this {src} audio: {tasks}",
           "From this recording in {src}: {tasks}"],
}
COMPOUND_FRAMES_TWO = {
    "fr": ["Fais deux choses avec cet extrait en {src} : {tasks}",
           "Deux demandes sur cet audio en {src} : {tasks}"],
    "en": ["Do two things with this {src} clip: {tasks}",
           "Two things on this {src} audio: {tasks}"],
}
# Same job, once a merge ("transcris les deux audios") is in `tasks`: the frames
# above say "cet extrait"/"this clip" (singular) and name ONE clip's language, both
# wrong once the ask is about two clips that need not even share a language.
# No count in these on purpose: with a rare third clip in the room, "these two
# clips" would be wrong as often as the singular frames it replaces.
COMPOUND_FRAMES_DUAL = {
    "fr": ["Sur ces extraits : {tasks}", "Pour ces audios : {tasks}",
           "À partir de ces enregistrements : {tasks}"],
    "en": ["On these clips: {tasks}", "For these audios: {tasks}",
           "From these recordings: {tasks}"],
}

# Bare task fragments used to build a compound instruction.
TASK_FRAGMENT_TRANSCRIBE = {
    "fr": ["transcris-le", "donne la transcription", "écris ce qui est dit"],
    "en": ["transcribe it", "give the transcription", "write down what is said"],
}
TASK_FRAGMENT_TRANSCRIBE_NAMED = {
    "fr": ["transcris-le en {lang}", "donne la transcription {adj}",
           "écris en {lang} ce qui est dit"],
    "en": ["transcribe it in {lang}", "give the {lang} transcription",
           "write down what is said, in {lang}"],
}
TASK_FRAGMENT_TRANSLATE = {
    "fr": ["traduis-le en {lang}", "donne sa version {adj}", "rends-le en {lang}"],
    "en": ["translate it into {lang}", "give its {lang} version", "render it in {lang}"],
}

COORDINATORS = {"fr": [" puis ", " et ", ", ensuite "], "en": [" then ", " and ", ", then "]}

# Same, for a bundled turn that is NOT the first, so "compound" never fuses with
# "turn 1".
COMPOUND_FOLLOWUP_FRAMES = {
    "fr": ["Et en {langs} ?", "Maintenant en {langs}.", "Et en {langs}, ça donne quoi ?",
           "Donne-moi aussi les versions {adjs}."],
    "en": ["And in {langs}?", "Now in {langs}.", "And in {langs}, what does that give?",
           "Give me the {langs} versions too."],
}
# Same, when the still-unspoken transcript is one of the things being asked for.
COMPOUND_FOLLOWUP_WITH_TRANSCRIPT = {
    "fr": ["Et la transcription, plus la version {adjs} ?",
           "Redonne-moi le texte d'origine et sa version {adjs}.",
           "Le texte original et le {langs}, s'il te plaît."],
    "en": ["And the transcription, plus the {langs} version?",
           "Give me the original text and its {langs} version.",
           "The original text and the {langs} one, please."],
}

# Changing audio part-way through. EXPLICIT: a prefix before a normal instruction, so
# the turn stays self-contained. ELLIPTIC: no instruction at all, the task carries
# over from the previous segment -- only used when the new clip can answer it.
CONTINUATION_EXPLICIT = {
    "fr": ["Passons à un autre extrait.", "Autre audio maintenant.",
           "Voici un autre enregistrement.", "Changeons d'extrait.",
           "J'ai un deuxième audio."],
    "en": ["Let's move on to another clip.", "Now a different audio.",
           "Here is another recording.", "Switching to another clip.",
           "I have a second audio."],
}
CONTINUATION_ELLIPTIC = {
    "fr": ["Et celui-ci ?", "Même chose sur cet audio.", "Et sur cet extrait ?",
           "Pareil pour cet enregistrement.", "Et avec cet audio ?",
           "Idem sur celui-là."],
    "en": ["And this one?", "Same thing on this audio.", "What about this clip?",
           "Same for this recording.", "And with this audio?",
           "Likewise on that one."],
}
# Same switch, but the carried task is NAMED by its output language. A bare "Et
# celui-ci ?" after "Et la version française ?" is answerable two ways -- transcribe
# or translate -- and the target is only recoverable by assuming the very last ask
# carries over rather than the task the conversation opened on. Used for a share of
# translation carry-overs (--elliptic-lang-rate) so the bare form still trains the
# carry-over itself.
CONTINUATION_ELLIPTIC_TARGET = {
    "fr": ["Et celui-ci, en {lang} ?", "Même chose en {lang} sur cet audio.",
           "Et cet extrait, en {lang} ?", "Pareil en {lang} pour cet enregistrement.",
           "Et sa version {adj}, sur ce nouvel audio ?"],
    "en": ["And this one, in {lang}?", "Same thing in {lang} on this audio.",
           "What about this clip, in {lang}?", "Same in {lang} for this recording.",
           "And its {lang} version, on this new audio?"],
}
# Transcript carried over onto a clip in ANOTHER language: the output language is
# fixed by the audio, but naming it removes the doubt about which language is
# actually being heard.
CONTINUATION_ELLIPTIC_SOURCE = {
    "fr": ["Et celui-ci, qui est en {lang} ?", "Même chose sur cet audio en {lang}.",
           "Et cet extrait en {lang} ?", "Pareil pour cet enregistrement en {lang}."],
    "en": ["And this one, which is in {lang}?", "Same thing on this {lang} audio.",
           "What about this {lang} clip?", "Same for this recording in {lang}."],
}

# With several clips in play "quel accent ?" has no referent, so the question is
# qualified to point at the latest audio.
VOICE_QUALIFIER = {
    "fr": ["Sur ce dernier audio : {q}", "Pour l'extrait qu'on vient d'entendre : {q}"],
    "en": ["On this last audio: {q}", "About the clip we just heard: {q}"],
}

# Closing turn asking for everything already answered, regrouped into one structured
# reply. The material is entirely in the conversation: nothing new is transcribed or
# translated, the task is purely to reorganise. Only offered from RECAP_MIN_TURNS
# content turns on -- recapping a single turn is just repeating it.
RECAP_MIN_TURNS = 2
RECAP_QUESTION = {
    "fr": ["Récapitule tout ce que tu m'as donné.",
           "Regroupe-moi tout ça proprement.",
           "Peux-tu réorganiser toutes ces informations en une seule réponse ?",
           "Rassemble tout ça dans un seul message.",
           "Fais-moi une synthèse structurée de ce qu'on a vu.",
           "Reprends l'ensemble et présente-le de façon ordonnée.",
           "Remets tout ça au propre, en un seul bloc."],
    "en": ["Sum up everything you have given me.",
           "Group all of that together properly.",
           "Can you reorganise all this information into a single answer?",
           "Put it all together in one message.",
           "Give me a structured recap of what we covered.",
           "Take everything above and lay it out in order.",
           "Tidy all of that up into a single block."],
}
# An empty lead is one of the options: a recap that always opens on the same sentence
# would make the phrase, not the regrouping, the thing being learned.
RECAP_LEAD = {
    "fr": ["", "", "Voici le récapitulatif :", "Récapitulatif :", "Voici tout, regroupé :"],
    "en": ["", "", "Here is the recap:", "Recap:", "Here is everything, grouped:"],
}
# Which clip a block belongs to, when the conversation carried more than one.
RECAP_CLIP_LABEL = {
    "fr": "Extrait {n} ({lang})",
    "en": "Clip {n} ({lang})",
}
# `prose` is deliberately absent: the ask is to STRUCTURE what was said, and prose
# output would make the recap indistinguishable from the answers it regroups.
RECAP_STYLES = ("labelled", "numbered")

_OXFORD_AND = {"fr": "et", "en": "and"}


def oxford_join(parts, ilang):
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f" {_OXFORD_AND[ilang]} " + parts[-1]

# Three answer shapes, so no single separator is learned as "the" format.
ANSWER_STYLES = ("labelled", "prose", "numbered")
ANSWER_STYLES_NO_NUMBER = ("labelled", "prose")
# Once several audios share one answer with one line per clip, "numbered" is the
# format that disambiguates for free (the position IS the clip), so it is picked
# more often than the other two -- labelled/prose still show up, just not as the
# plurality.
_DUAL_ANSWER_STYLE_WEIGHTS = (0.25, 0.25, 0.5)


def numbered_style_mode(has_transcript, translations, dual_text, extra_duals):
    """How safe a bare '1. / 2. / 3.' rendering is for this answer, and how it
    should be weighted if used.

    With exactly ONE audio, "2" can only mean "answer to the 2nd question" --
    nothing else in the conversation for it to refer to, so it is unambiguous and
    gets no special weighting ("uniform"). With SEVERAL audios it is only safe
    when the mapping really is one line per clip ("clip", weighted toward
    numbered, since position alone then tells the clips apart) -- a merged clause
    padded with a second fact about a clip that already has one (`extra_duals`
    idx 0, or a clip1 translation alongside its own transcript) puts more lines
    than clips, which is genuinely ambiguous between "2nd question" and "2nd
    audio" and is excluded ("excluded").
    """
    total_items = ((1 if has_transcript else 0) + len(translations)
                   + (1 if dual_text is not None else 0) + len(extra_duals))
    distinct_clips = (1 + (1 if dual_text is not None else 0)
                      + (1 if any(idx == 1 for idx, _, _ in extra_duals) else 0))
    if distinct_clips == 1:
        return "uniform"
    return "clip" if total_items == distinct_clips else "excluded"


def pick_answer_style(mode="uniform"):
    if mode == "excluded":
        return random.choice(ANSWER_STYLES_NO_NUMBER)
    if mode == "clip":
        return random.choices(ANSWER_STYLES, weights=_DUAL_ANSWER_STYLE_WEIGHTS)[0]
    return random.choice(ANSWER_STYLES)

LABELS = {
    "fr": {"transcription": "Transcription", "translation": "Traduction ({lang})"},
    "en": {"transcription": "Transcription", "translation": "Translation ({lang})"},
}

PROSE_TRANSCRIPTION = {
    "fr": ["L'extrait dit : « {text} ».", "On entend : « {text} »."],
    "en": ["The clip says: \"{text}\".", "What is said is: \"{text}\"."],
}
PROSE_TRANSLATION = {
    "fr": ["En {lang}, cela donne : « {text} ».", "Sa version {adj} : « {text} »."],
    "en": ["In {lang}, that is: \"{text}\".", "Its {lang} version: \"{text}\"."],
}
# Same job as PROSE_TRANSCRIPTION/PROSE_TRANSLATION, but naming the clip -- used for
# the FIRST clip's own transcript/translation once a dual item also has content in
# the same prose answer, so "What is said is..." doesn't read as ownerless the moment
# a second clip's sentence follows it.
PROSE_FIRST_TRANSCRIBE = {
    "fr": ["Le premier extrait dit : « {text} ».",
           "Sur le premier extrait, on entend : « {text} »."],
    "en": ["The first clip says: \"{text}\".",
           "On the first clip, what is said is: \"{text}\"."],
}
PROSE_FIRST_TRANSLATE = {
    "fr": ["Sur le premier extrait, en {lang}, cela donne : « {text} ».",
           "Sa version {adj} (premier extrait) : « {text} »."],
    "en": ["On the first clip, in {lang}, that is: \"{text}\".",
           "Its {lang} version (first clip): \"{text}\"."],
}


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_cv_index(cv_root: Path, lang: str, tsv_name: str = "validated.tsv",
                  keep: set = None) -> dict:
    """Index the raw CommonVoice TSV by clip stem.

    `validated.tsv` rather than `train.tsv`: it is the superset the AST manifests were
    built from. `keep` restricts the index to stems the manifests reference, the eight
    full TSVs being ~1.4 GB of mostly unjoinable text.
    """
    path = cv_root / lang / tsv_name
    if not path.exists():
        raise FileNotFoundError(f"CommonVoice TSV not found: {path}")
    index = {}
    with path.open(encoding="utf-8", newline="") as f:
        # QUOTE_NONE: these files are tab-separated and not quoted, and the pt TSV has a
        # lone double quote that the default dialect reads as opening a field.
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            stem = Path(row["path"]).stem
            if keep is not None and stem not in keep:
                continue
            index[stem] = {
                "sentence": (row.get("sentence") or "").strip(),
                "client_id": (row.get("client_id") or "").strip(),
                "lang": lang,
                "accent": normalise_accent(row.get("accents")),
                "gender": normalise_gender(row.get("gender")),
                "age": normalise_age(row.get("age")),
            }
    return index


def discover_ast_manifests(ast_root: Path, src: str, split: str, variant: str) -> dict:
    """Map target language -> AST manifest path for a given source language.

    Looks for `ast/{src}-{tgt}/{variant}/CommonVoice{SRC}2{TGT}/{split}.jsonl`.
    Only CommonVoice-derived manifests are eligible: they are the ones whose
    audio stems join back to the raw TSV.
    """
    found = {}
    for direction in sorted(ast_root.glob(f"{src}-*")):
        tgt = direction.name.split("-", 1)[1]
        expected = f"CommonVoice{src.upper()}2{tgt.upper()}"
        manifest = direction / variant / expected / f"{split}.jsonl"
        if manifest.exists():
            found[tgt] = manifest
    return found


def load_ast_manifest(path: Path) -> dict:
    """Index one AST manifest by clip stem -> {text, audio, duration}."""
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            audio_path = duration = target = None
            for turn in record.get("conversations") or []:
                if turn.get("type") == "audio":
                    audio_path = turn.get("value")
                    duration = turn.get("duration")
                elif turn.get("from") == "Assistant":
                    target = (turn.get("value") or "").strip()
            if audio_path and target:
                out[Path(str(audio_path)).stem] = {
                    "text": target, "audio": str(audio_path), "duration": duration,
                }
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────────────────────────────────────

# Content-turn count. This DRIVES how many translations a conversation consumes;
# doing it the other way round made 79% of conversations exactly 5 turns long.
TURN_DIST = ((2, 0.35), (3, 0.30), (4, 0.25), (5, 0.10))

# Same for a segment continuing on a NEW clip: a single ask is fine there, and only
# French has more than one translation available anyway.
CONT_TURN_DIST = ((1, 0.45), (2, 0.35), (3, 0.20))

# How many clips a conversation uses, once it has been drawn as multi-audio.
EXTRA_CLIP_DIST = ((1, 0.75), (2, 0.25))

# Within-length weights solved so the GLOBAL bundle position lands on 40/30/30: a
# two-turn conversation has no third slot, so short lengths over-weight the first two.
COMPOUND_POS_TWO_TURN = (0.55, 0.45)
COMPOUND_POS_FIRST = 0.319
COMPOUND_POS_SECOND = 0.219

BUNDLE_SIZES = (2, 2, 3)

# `None` is a valid item key (the transcript), so "no preference" needs a sentinel.
_MISSING = object()


# Set in main(); the builders read it rather than taking a ninth parameter.
CONTEXTS_ROOT = None
# (kind, key, ilang) -> list of prompts. Filled on first draw; a conversation only ever
# touches the pairs its clips can answer, so loading all 56 AST files up front is waste.
_CONTEXT_CACHE = {}


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def asr_prompts(src, ilang):
    """Turn-1 transcription prompts for `src` audio, written in `ilang`.

    Two regimes, and they differ by an order of magnitude. On the diagonal
    (`ilang == src`, so French instructions over French audio) the whole same-language
    bucket applies: 31 prompts for fr, 38 for en. Off it, only the four cross-lingual
    templates match -- the rest of the file is written in the audio's own language and
    would silently switch the conversation's instruction language mid-way.
    """
    key = ("asr", src, ilang)
    if key not in _CONTEXT_CACHE:
        bank = _load_json(CONTEXTS_ROOT / ASR_BANK_NAME.format(lang=src))[ASR_VARIANT]
        if ilang == src:
            pool = list(bank[ASR_SAME_LANG_BUCKET])
        else:
            want = {t.format(lang=LANG_NAMES[ilang][src])
                    for t in ASR_CROSS_LANG_TEMPLATES[ilang]}
            pool = [p for p in bank[ASR_CROSS_LANG_BUCKET] if p in want]
        if not pool:
            raise ValueError(f"no {ilang} transcription prompt for {src} audio "
                             f"in {CONTEXTS_ROOT / ASR_BANK_NAME.format(lang=src)}")
        _CONTEXT_CACHE[key] = pool
    return _CONTEXT_CACHE[key]


def ast_prompts(src, tgt, ilang):
    """Turn-1 translation prompts for src->tgt, written in `ilang`.

    The pair file pools every instruction language into weight buckets with no tag, but
    every prompt names its TARGET language -- in its own language, so "en néerlandais"
    marks a French prompt and "naar het Nederlands" a Dutch one. Matching on the target
    name as spelled in `ilang` selects exactly the ten that belong to it: five naming
    both languages, five naming only the target. The three bare stubs ("Traduis") are
    dropped by the same rule, which is what we want here -- turn 1 has to say what
    language to answer in.
    """
    key = ("ast", (src, tgt), ilang)
    if key not in _CONTEXT_CACHE:
        path = CONTEXTS_ROOT / AST_BANK_RELNAME.format(src=src, tgt=tgt)
        bank = _load_json(path)[AST_VARIANT]
        name = LANG_NAMES[ilang][tgt]
        pool = [p for bucket in bank.values() for p in bucket if name in p]
        if not pool:
            raise ValueError(f"no {ilang} translation prompt for {src}->{tgt} in {path}")
        _CONTEXT_CACHE[key] = pool
    return _CONTEXT_CACHE[key]


def bank_prompt(task, ilang, src, tgt=None):
    """Draw one turn-1 instruction for `task` from data/contexts.

    No retargeting: the banks are stored per audio language and per pair, so the prompt
    already names the right languages in the right instruction language.
    """
    if task == "transcription":
        return random.choice(asr_prompts(src, ilang))
    return random.choice(ast_prompts(src, tgt, ilang))


def draw_turns(max_turns, dist=TURN_DIST):
    """Draw a content-turn count from `dist`, capped by what is available."""
    pool = [(n, w) for n, w in dist if n <= max_turns]
    if not pool:
        return 0
    return random.choices([n for n, _ in pool], weights=[w for _, w in pool], k=1)[0]


def draw_compound_position(n_turns):
    """Index of the turn carrying several instructions."""
    if n_turns <= 1:
        return 0
    if n_turns == 2:
        return random.choices((0, 1), weights=COMPOUND_POS_TWO_TURN, k=1)[0]
    rest = (1.0 - COMPOUND_POS_FIRST - COMPOUND_POS_SECOND) / (n_turns - 2)
    weights = [COMPOUND_POS_FIRST, COMPOUND_POS_SECOND] + [rest] * (n_turns - 2)
    return random.choices(range(n_turns), weights=weights, k=1)[0]


def conversation_items(transcript, targets, first_key=_MISSING, first_start=0):
    """Ordered pool of things a conversation can ask for.

    The transcript sits first only 60% of the time, so "turn 1 is the transcription"
    never becomes a rule. `first_key` pins one item near the front (None for the
    transcript): that is how an elliptic continuation asks the task it inherited.
    A LIST of keys pins each in order starting at `first_start` (0, 1, ... from
    there) -- how a dual and a rare triple item get guaranteed a spot ahead of
    everything else. `first_start` defaults to 0 (literally the front, what the
    elliptic case above needs -- the carried task MUST be turn 1's own ask) but the
    dual feature draws it randomly: forcing the second clip to position 0 every
    time it is active would make "what does the FIRST clip say" never the opening
    question of a dual conversation. `first_start=1` leaves position 0 to whatever
    item would have landed there naturally (often the transcript) and only
    guarantees the dual item(s) a spot from turn 2 on.
    """
    items = list(targets)
    if transcript:
        idx = 0 if random.random() < 0.6 else random.randrange(1, len(items) + 1)
        items.insert(idx, (None, transcript))
    if first_key is not _MISSING:
        keys = first_key if isinstance(first_key, (list, tuple)) else [first_key]
        for i, key in enumerate(keys):
            place_key(items, first_start + i, key)
    return items


def place_key(items, index, key):
    """Move the item answering `key` to `index`, if it is there to move.

    Used to steer which ask a segment ENDS on, so that the elliptic "and this
    one?" opening the next segment inherits a task the next clip can actually
    answer. Returns whether the move happened.
    """
    if index >= len(items):
        return False
    for i, (tgt, _) in enumerate(items):
        if tgt == key:
            items.insert(index, items.pop(i))
            return True
    return False


def quotable(text):
    """Strip a single trailing period before the text is dropped inside quotes.

    The prose templates close with `« {text} ».`, so a transcript that already
    ends in a period renders as `« ... ». ` preceded by its own stop -- a double
    punctuation that appears in every prose sample and would be learned as the
    house style. Other terminators (?, !, …) are meaningful and kept.
    """
    text = text.strip()
    return text[:-1].rstrip() if text.endswith(".") and not text.endswith("..") else text


def normalise_gender(raw):
    """Map the TSV gender column to 'm'/'f', or '' when unusable.

    `do_not_wish_to_say` is an explicit refusal and is treated as missing.
    """
    raw = (raw or "").strip()
    if raw.startswith("male"):
        return "m"
    if raw.startswith("female"):
        return "f"
    return ""


def normalise_age(raw):
    """Keep the TSV age bucket only when we have a rendering for it."""
    raw = (raw or "").strip().lower()
    return raw if raw in AGE_LABELS["en"] else ""


def normalise_accent(raw):
    """Pick a single usable accent label from the TSV `accents` column.

    The column is a free-text comma-separated list ("Français de France,Français
    du Canada") and is sometimes truncated mid-word. Keep the first entry, and
    drop anything that no longer looks like a complete label.
    """
    if not raw:
        return ""
    first = raw.split(",")[0].strip()
    return first if len(first) >= 4 else ""


def user_text(value):
    return {"from": "User", "value": value, "type": "text"}


def user_audio(value, duration):
    turn = {"from": "User", "value": value, "type": "audio"}
    if duration is not None:
        turn["duration"] = duration
    return turn


def assistant(value):
    return {"from": "Assistant", "value": value, "type": "text"}


def render_compound_answer(ilang, transcript, translations, style, dual_task=None, dual_text=None,
                           extra_duals=()):
    """Compose one answer covering several requested tasks.

    `translations` is an ordered list of (target_code, text), never containing the
    dual item -- callers pull that out into `dual_text` first (`dual_task` says
    whether it is the second clip's transcript, None, or one of its translations, a
    language code). `transcript` may be None when the compound only chains
    translations (and/or the dual item). `extra_duals` -- `(idx, task, text)` triples
    -- adds a second fact about the second clip and/or the rare third clip.
    """
    sep = " : " if ilang == "fr" else ": "
    dual_active = dual_text is not None or bool(extra_duals)
    parts = []
    if style == "labelled":
        if transcript:
            # Once a dual item is also being labelled, "Transcription" next to
            # "Second extrait" reads as a mismatched pair (task label vs clip label)
            # -- "Premier extrait" keeps both blocks naming the same kind of thing.
            label = first_clip_label(ilang, None) if dual_active else LABELS[ilang]["transcription"]
            parts.append(f"{label}{sep}{transcript}")
        for tgt, text in translations:
            label = first_clip_label(ilang, tgt) if dual_active else fmt(LABELS[ilang]["translation"], tgt, ilang)
            parts.append(f"{label}{sep}{text}")
        if dual_text is not None:
            parts.append(f"{dual_label(ilang, 0, dual_task)}{sep}{dual_text}")
        for idx, task, text in extra_duals:
            parts.append(f"{dual_label(ilang, idx, task)}{sep}{text}")
        return "\n".join(parts)

    if style == "prose":
        if transcript:
            pool = PROSE_FIRST_TRANSCRIBE[ilang] if dual_active else PROSE_TRANSCRIPTION[ilang]
            parts.append(random.choice(pool).format(text=quotable(transcript)))
        for tgt, text in translations:
            pool = PROSE_FIRST_TRANSLATE[ilang] if dual_active else PROSE_TRANSLATION[ilang]
            parts.append(fmt(random.choice(pool), tgt, ilang, text=quotable(text)))
        if dual_text is not None:
            parts.append(prose_dual(ilang, 0, dual_task, dual_text))
        for idx, task, text in extra_duals:
            parts.append(prose_dual(ilang, idx, task, text))
        return " ".join(parts)

    # numbered -- the position already disambiguates which clip an item belongs to
    # (no "first/second clip" wording needed, unlike labelled/prose), but a
    # translation still needs its language named or "1. blabla" / "2. blabla" would
    # leave the reader guessing which of several languages each number is in.
    entries = ([(None, transcript)] if transcript else []) + list(translations)
    if dual_text is not None:
        entries.append((dual_task, dual_text))
    entries.extend((task, text) for _, task, text in extra_duals)
    return "\n".join(
        f"{i}{fmt(' ({lang})', tgt, ilang) if tgt is not None else ''}{sep}{text}"
        for i, (tgt, text) in enumerate(entries, 1)
    )


# ── Dual-clip transcription ─────────────────────────────────────────────────
# A second clip, glued into segment 0's turn 1 alongside the first: not a shape of
# its own, but one more ASKABLE ITEM (key `DUAL_KEY`) sitting in the same `targets`
# pool as the real translations. That is what lets it flow through
# sequential/compound/hybrid unmodified: drawn as an opener, folded into a bundle,
# recapped, chained -- whatever the pool item it stands in for would get.
#
# `dual_task` says what the item actually IS -- None for the second clip's own
# transcript, a language code for one of ITS OWN translations -- drawn from the
# second clip's real pool (`build_conversation`), never invented: asking to
# "translate the second clip into English" when it already IS English audio reads
# as a non-sequitur, and only a real target on that clip avoids it. Every phrasing
# function below branches on it the same way `fmt` branches a translation target.
#
# Two clips in the same turn also means every question that used to say "it" /
# "en anglais ?" about the FIRST clip is now ambiguous -- `qualify_first_clip`
# below is what `render_compound_answer`'s dual-aware callers use to keep the
# first clip's own follow-ups pointed unambiguously at it, the same way
# `qualify_voice_question` already does for voice follow-ups.
DUAL_KEY = "__dual__"
# A rare THIRD clip, glued in the same way. It never merges with anything (that
# would need a three-way "transcris les trois audios" grammar this file doesn't
# have) -- always its own explicitly-numbered clause or turn.
DUAL_KEY2 = "__dual2__"
# Every dual-phrasing function below takes an explicit ordinal INDEX (0 = the
# second clip, 1 = the third) rather than inferring it from which sentinel key
# is involved, so the SAME functions also serve `dual_extra_task`/`dual_extra_text`
# in `build_compound` -- a second fact about the second clip (idx 0), never routed
# through the pool at all, used only to keep the padding clause of a merged ask
# from always being about the first clip.
DUAL_ORDINAL_WORD = {"fr": ("second", "troisième"), "en": ("second", "third")}
DUAL_LABEL_TRANSCRIBE = {"fr": "{ord} extrait", "en": "{ord} clip"}
DUAL_LABEL_TRANSLATE = {"fr": "{ord} extrait ({lang})", "en": "{ord} clip ({lang})"}
# Same, for the FIRST clip's own items -- used only once a dual item is also being
# labelled, so "Transcription" next to "Second extrait" doesn't read as one being
# the task and the other the clip.
FIRST_LABEL_TRANSCRIBE = {"fr": "Premier extrait", "en": "First clip"}
FIRST_LABEL_TRANSLATE = {"fr": "Premier extrait ({lang})", "en": "First clip ({lang})"}
# Standalone questions, for when the dual item is drawn on its own (sequential, or
# a non-bundled hybrid turn) -- opener vs follow-up, same split as FOLLOWUP_TRANSCRIBE.
# Openers are always about idx 0: `first_key` puts DUAL_KEY at position 0 and
# DUAL_KEY2 at position 1, so DUAL_KEY2's task is never the FIRST thing asked.
DUAL_QUESTION_TRANSCRIBE = {
    "fr": ["Transcris le second extrait.", "Qu'est-ce que dit le second extrait ?",
           "Peux-tu transcrire le deuxième audio ?"],
    "en": ["Transcribe the second clip.", "What does the second clip say?",
           "Can you transcribe the second audio?"],
}
DUAL_QUESTION_TRANSLATE = {
    "fr": ["Traduis le second extrait en {lang}.",
           "Le second extrait, en {lang}, ça donne quoi ?",
           "Donne-moi la version {adj} du second extrait."],
    "en": ["Translate the second clip into {lang}.",
           "The second clip, in {lang} -- what does it say?",
           "Give me the {lang} version of the second clip."],
}
# Same openers, but committing to "two clips" in the wording -- only correct when
# there really are just two (a third, DUAL_KEY2, sitting in the same targets pool
# would make "Voici deux extraits" false the moment it lands in the same turn).
DUAL_QUESTION_TRANSCRIBE_PAIR = {
    "fr": ["Voici deux extraits : donne-moi la transcription du second."],
    "en": ["Here are two clips: give me the transcript of the second one."],
}
DUAL_QUESTION_TRANSLATE_PAIR = {
    "fr": ["Voici deux extraits : traduis le second en {lang}."],
    "en": ["Here are two clips: translate the second one into {lang}."],
}
DUAL_FOLLOWUP_QUESTION_TRANSCRIBE = {
    "fr": ["Et le {ord} extrait ?", "Et sur le {ord} audio ?",
           "Pareil, mais sur le {ord} extrait."],
    "en": ["And the {ord} clip?", "What about the {ord} audio?",
           "Same thing, for the {ord} clip."],
}
DUAL_FOLLOWUP_QUESTION_TRANSLATE = {
    "fr": ["Et le {ord} extrait, en {lang} ?", "Pareil, mais le {ord} extrait en {lang}."],
    "en": ["And the {ord} clip, in {lang}?", "Same thing, the {ord} clip in {lang}."],
}
# Bare clauses, for when the dual item is bundled with other asks in one turn --
# joined by COORDINATORS exactly like TASK_FRAGMENT_TRANSCRIBE/TRANSLATE. The
# "_FIRST" pair says the same thing about the FIRST clip's own transcript/translation,
# used only once a dual item is also in the sentence, to keep "-le" from being read
# as pointing at whichever clip was mentioned last.
TASK_FRAGMENT_TRANSCRIBE_DUAL = {
    "fr": ["transcris aussi le {ord} extrait", "donne la transcription du {ord} extrait",
           "écris ce qui est dit dans le {ord} extrait"],
    "en": ["also transcribe the {ord} clip", "give the transcription of the {ord} clip",
           "write down what is said in the {ord} clip"],
}
TASK_FRAGMENT_TRANSLATE_DUAL = {
    "fr": ["traduis aussi le {ord} extrait en {lang}",
           "donne la version {adj} du {ord} extrait"],
    "en": ["also translate the {ord} clip into {lang}",
           "give the {lang} version of the {ord} clip"],
}
TASK_FRAGMENT_TRANSCRIBE_FIRST = {
    "fr": ["transcris le premier extrait", "donne la transcription du premier extrait"],
    "en": ["transcribe the first clip", "give the transcription of the first clip"],
}
TASK_FRAGMENT_TRANSLATE_FIRST = {
    "fr": ["traduis le premier extrait en {lang}", "donne la version {adj} du premier extrait"],
    "en": ["translate the first clip into {lang}", "give the {lang} version of the first clip"],
}
# One clause covering BOTH clips, used instead of two separate ones whenever they
# happen to share the same task -- "transcris les deux audios" rather than
# "transcris-le et transcris aussi le second extrait". Only ever the FIRST clip and
# DUAL_KEY (idx 0): DUAL_KEY2 never merges.
TASK_FRAGMENT_TRANSCRIBE_BOTH = {
    "fr": ["transcris les deux extraits", "transcris les deux audios",
           "donne la transcription des deux audios"],
    "en": ["transcribe both clips", "transcribe both audios",
           "give the transcription of both audios"],
}
# Same clause, but worded for when a THIRD clip (an `extra_duals` fact) is also
# riding along in the same turn -- "les deux audios"/"both audios" would be
# ambiguous once a third one is in the room, so this names the first two
# explicitly instead of claiming to be the whole set.
TASK_FRAGMENT_TRANSCRIBE_BOTH_QUALIFIED = {
    "fr": ["transcris les deux premiers extraits", "transcris les deux premiers audios",
           "donne la transcription des deux premiers audios"],
    "en": ["transcribe the first two clips", "transcribe the first two audios",
           "give the transcription of the first two audios"],
}
# Tacked onto a "transcribe/translate both clips" clause (never a standalone ask
# of its own) to ask for ONE continuous text instead of two separate answers --
# for consecutive audios that are really one thing split in two (a recording cut
# into parts), where the user wants them read back as a single block, not two
# labelled ones. Only offered when the merge would otherwise stand ALONE (see
# DUAL_GLUE_RATE): gluing alongside another, unrelated ask in the same turn has
# no sensible single-text answer.
TASK_FRAGMENT_TRANSCRIBE_GLUE_SUFFIX = {
    "fr": [" (et fais-en une seule transcription)", ", en une seule transcription",
           " et regroupe les deux transcriptions en une seule",
           " et mets les deux transcriptions à la suite"],
    "en": [" (and make it a single transcription)", ", as a single transcription",
           " and merge the two transcriptions into one",
           " and put the two transcriptions one after another"],
}
TASK_FRAGMENT_TRANSLATE_GLUE_SUFFIX = {
    "fr": [" (et fais-en une seule traduction)", ", en une seule traduction",
           " et regroupe les deux traductions en une seule",
           " et mets les deux traductions à la suite"],
    "en": [" (and make it a single translation)", ", as a single translation",
           " and merge the two translations into one",
           " and put the two translations one after another"],
}
# Share of an otherwise-standalone "transcribe/translate both clips" merge that
# gets this suffix (and therefore a glued answer) instead of the plain merge
# clause.
DUAL_GLUE_RATE = 0.2
TASK_FRAGMENT_TRANSLATE_BOTH = {
    "fr": ["traduis les deux extraits en {lang}", "traduis les deux audios en {lang}",
           "donne la version {adj} des deux audios"],
    "en": ["translate both clips into {lang}", "translate both audios into {lang}",
           "give the {lang} version of both audios"],
}
# Same disambiguation as TASK_FRAGMENT_TRANSCRIBE_BOTH_QUALIFIED, for the
# translate-merge clause.
TASK_FRAGMENT_TRANSLATE_BOTH_QUALIFIED = {
    "fr": ["traduis les deux premiers extraits en {lang}", "traduis les deux premiers audios en {lang}",
           "donne la version {adj} des deux premiers audios"],
    "en": ["translate the first two clips into {lang}", "translate the first two audios into {lang}",
           "give the {lang} version of the first two audios"],
}
# When the transcribe-merge (clip1+dual) and EVERY extra_duals fact are all a
# plain transcription with nothing else asked for (no translation anywhere),
# "transcribe both audios, then the third clip" is a clunky way to say the
# same thing for every clip in the turn -- collapse to one "transcribe them
# all" clause instead of merge-both + a separate clause per extra clip.
TASK_FRAGMENT_TRANSCRIBE_ALL = {
    "fr": ["transcris tous les extraits", "transcris tous les audios",
           "donne la transcription de tous les audios"],
    "en": ["transcribe all the clips", "transcribe all the audios",
           "give the transcription of all the audios"],
}
# Same job as COMPOUND_FOLLOWUP_FRAMES/_WITH_TRANSCRIPT, but wrapping a joined clause
# list (`{tasks}`) rather than a language list: those two assume every non-transcript
# item is a translation, which the dual item is not.
COMPOUND_FOLLOWUP_WITH_DUAL = {
    "fr": ["Ensuite : {tasks}", "Continue avec ceci : {tasks}", "Fais aussi ceci : {tasks}"],
    "en": ["Next: {tasks}", "Also do this: {tasks}", "Continue with this: {tasks}"],
}
PROSE_DUAL_TRANSCRIBE = {
    "fr": ["Le {ord} extrait dit : « {text} ».",
           "Sur le {ord} extrait, on entend : « {text} »."],
    "en": ["The {ord} clip says: \"{text}\".",
           "On the {ord} clip, what is said is: \"{text}\"."],
}
PROSE_DUAL_TRANSLATE = {
    "fr": ["Sa version {adj} : « {text} ».",
           "En {lang}, le {ord} extrait donne : « {text} »."],
    "en": ["Its {lang} translation: \"{text}\".",
           "In {lang}, the {ord} clip gives: \"{text}\"."],
}
# Same idiom as `qualify_voice_question`/`VOICE_QUALIFIER`, but pointing at the
# FIRST clip instead of the most recent one -- needed once a dual item has put a
# second clip in the room, so a bare "et en anglais ?" no longer reads as being
# about whichever clip was mentioned last.
FIRST_CLIP_QUALIFIER = {
    "fr": ["Sur le premier extrait : {q}", "Pour le tout premier audio : {q}"],
    "en": ["About the first clip: {q}", "For the very first audio: {q}"],
}


def dual_label(ilang, idx, dual_task):
    ord_word = DUAL_ORDINAL_WORD[ilang][idx]
    ord_cap = ord_word[0].upper() + ord_word[1:]
    if dual_task is None:
        return DUAL_LABEL_TRANSCRIBE[ilang].format(ord=ord_cap)
    return DUAL_LABEL_TRANSLATE[ilang].format(ord=ord_cap, lang=LANG_NAMES[ilang][dual_task])


def first_clip_label(ilang, tgt):
    if tgt is None:
        return FIRST_LABEL_TRANSCRIBE[ilang]
    return fmt(FIRST_LABEL_TRANSLATE[ilang], tgt, ilang)


def dual_question(ilang, idx, dual_task, first, has_third=False):
    # Openers are hard-coded to "second": only idx 0 (DUAL_KEY) is ever first.
    # `has_third` says whether a DUAL_KEY2 is ALSO in play for this conversation --
    # the "Voici deux extraits"/"Here are two clips" phrasings commit to an exact
    # clip count, so they are only offered when that count is really two.
    if first:
        base = DUAL_QUESTION_TRANSCRIBE if dual_task is None else DUAL_QUESTION_TRANSLATE
        pair = DUAL_QUESTION_TRANSCRIBE_PAIR if dual_task is None else DUAL_QUESTION_TRANSLATE_PAIR
        pool = list(base[ilang]) + ([] if has_third else list(pair[ilang]))
        return (random.choice(pool) if dual_task is None
                else fmt(random.choice(pool), dual_task, ilang))
    ord_word = DUAL_ORDINAL_WORD[ilang][idx]
    if dual_task is None:
        return random.choice(DUAL_FOLLOWUP_QUESTION_TRANSCRIBE[ilang]).format(ord=ord_word)
    return fmt(random.choice(DUAL_FOLLOWUP_QUESTION_TRANSLATE[ilang]), dual_task, ilang,
              ord=ord_word)


def dual_fragment(ilang, idx, dual_task):
    ord_word = DUAL_ORDINAL_WORD[ilang][idx]
    if dual_task is None:
        return random.choice(TASK_FRAGMENT_TRANSCRIBE_DUAL[ilang]).format(ord=ord_word)
    return fmt(random.choice(TASK_FRAGMENT_TRANSLATE_DUAL[ilang]), dual_task, ilang, ord=ord_word)


def prose_dual(ilang, idx, dual_task, text):
    ord_word = DUAL_ORDINAL_WORD[ilang][idx]
    if dual_task is None:
        return random.choice(PROSE_DUAL_TRANSCRIBE[ilang]).format(text=quotable(text), ord=ord_word)
    return fmt(random.choice(PROSE_DUAL_TRANSLATE[ilang]), dual_task, ilang,
              text=quotable(text), ord=ord_word)


_LEADING_AND = {"fr": "et ", "en": "and "}


def qualify_first_clip(question, ilang):
    """Point a question at the FIRST clip, once a dual item has put a second one
    in the room.

    Half of FOLLOWUP_TRANSLATE/FOLLOWUP_TRANSCRIBE opens on "Et"/"And" -- a
    connector doing the same job the qualifier prefix is about to do ("Sur le
    premier extrait : et en anglais ?" doubles up), so it is dropped first.
    """
    body = question[0].lower() + question[1:]
    prefix = _LEADING_AND[ilang]
    if body.startswith(prefix):
        body = body[len(prefix):]
    return random.choice(FIRST_CLIP_QUALIFIER[ilang]).format(q=body)


def build_task_fragments(ilang, include_transcript, translations, dual_task, dual_text, src=None,
                         extra_duals=()):
    """Bare clauses for a bundled ask: optional transcript, translations, optional
    second-clip item -- shared by `compound_turns` and any later bundled turn that
    also carries the dual item, so both phrase it the same way.

    When the first clip's transcript (or one of its translations) and the dual item
    happen to be the SAME task, they are merged into one "both clips" clause instead
    of two separate ones; otherwise each names its own clip explicitly. `extra_duals`
    -- `(idx, task, text)` triples -- are always their OWN clause, never merged: a
    second fact about the second clip, and/or the rare third clip.

    Returns `(fragments, merged, glued)` -- `merged` is what lets a caller treat a
    LONE "transcribe both clips" clause as already a complete two-clip
    instruction, rather than requiring a second, unrelated clause just to pass a
    "compound needs 2+ asks" gate that was written for two asks about the SAME
    clip. `glued` says the transcribe-merge clause asks for ONE continuous text
    (see `TASK_FRAGMENT_TRANSCRIBE_GLUE_SUFFIX`) rather than two separate answers
    -- only possible when the merge stands ALONE (no translation, no third clip
    fact riding along in the same turn), since there is no sensible single-text
    answer once something else is also being asked for.
    """
    has_dual = dual_text is not None
    merge_transcribe = has_dual and dual_task is None and include_transcript
    merge_lang = (dual_task if has_dual and dual_task is not None
                 and any(tgt == dual_task for tgt, _ in translations) else None)
    # clip1, the dual item, AND every extra clip are all being asked for a plain
    # transcription (clip1 may ALSO have separate translations on top -- those
    # still get their own clause below) -- fold the transcribe asks into one
    # "transcribe them all" clause instead of a merge clause plus a clause per
    # extra clip, since splitting out the extra clip(s) would just repeat the
    # same instruction the merge clause already gives.
    collapse_all_transcribe = (merge_transcribe and extra_duals
                               and all(task is None for _, task, _ in extra_duals))

    fragments = []
    glued = False
    if collapse_all_transcribe:
        fragments.append(random.choice(TASK_FRAGMENT_TRANSCRIBE_ALL[ilang]))
    elif merge_transcribe:
        pool = (TASK_FRAGMENT_TRANSCRIBE_BOTH_QUALIFIED if extra_duals
                else TASK_FRAGMENT_TRANSCRIBE_BOTH)
        fragment = random.choice(pool[ilang])
        if not translations and not extra_duals and random.random() < DUAL_GLUE_RATE:
            fragment += random.choice(TASK_FRAGMENT_TRANSCRIBE_GLUE_SUFFIX[ilang])
            glued = True
        fragments.append(fragment)
    elif include_transcript:
        if has_dual:
            fragments.append(random.choice(TASK_FRAGMENT_TRANSCRIBE_FIRST[ilang]))
        elif src is not None and src != ilang:
            fragments.append(fmt(random.choice(TASK_FRAGMENT_TRANSCRIBE_NAMED[ilang]),
                                 src, ilang))
        else:
            fragments.append(random.choice(TASK_FRAGMENT_TRANSCRIBE[ilang]))

    if merge_lang is not None:
        pool = (TASK_FRAGMENT_TRANSLATE_BOTH_QUALIFIED if extra_duals
                else TASK_FRAGMENT_TRANSLATE_BOTH)
        fragment = fmt(random.choice(pool[ilang]), merge_lang, ilang)
        # Alone the same way the transcribe merge is: no OTHER translation, no
        # separate transcript ask for clip 1, no third-clip fact riding along.
        if (not include_transcript and len(translations) == 1 and not extra_duals
                and random.random() < DUAL_GLUE_RATE):
            fragment += random.choice(TASK_FRAGMENT_TRANSLATE_GLUE_SUFFIX[ilang])
            glued = True
        fragments.append(fragment)
    for tgt, _ in translations:
        if tgt == merge_lang:
            continue
        pool = TASK_FRAGMENT_TRANSLATE_FIRST if has_dual else TASK_FRAGMENT_TRANSLATE
        fragments.append(fmt(random.choice(pool[ilang]), tgt, ilang))

    if has_dual and not merge_transcribe and merge_lang is None:
        fragments.append(dual_fragment(ilang, 0, dual_task))
    if not collapse_all_transcribe:
        for idx, task, _text in extra_duals:
            fragments.append(dual_fragment(ilang, idx, task))
    return fragments, (merge_transcribe or merge_lang is not None), glued


def join_fragments(ilang, fragments):
    tasks = fragments[0]
    for frag in fragments[1:]:
        tasks += random.choice(COORDINATORS[ilang]) + frag
    if not tasks.endswith("."):
        tasks += "."
    return tasks


# Share of mergeable compound turns ("transcris les deux audios") left to stand
# alone, rather than padded with one more clause about the first clip only. Kept
# high: a shared instruction on its own ("transcribe both clips") should be the
# COMMONEST shape a dual conversation takes, ahead of two separate per-clip asks,
# which in turn should be ahead of a shared instruction padded with a specific one.
DUAL_MERGE_ALONE_RATE = 0.85
# Of the padded ones, the share where the padding names the SECOND clip (a real
# second fact about it) instead of defaulting to the first.
DUAL_EXTRA_ON_SECOND_RATE = 0.5
# Share of dual draws where `dual_task` is biased toward a task the FIRST clip can
# also answer (always includes the transcript, since every clip has one -- see
# `build_conversation`), rather than drawn uniformly from the second clip's own
# pool. This is what makes a shared instruction ("transcribe/translate both
# clips") POSSIBLE often enough to be the commonest shape, rather than an
# incidental coincidence that only fires when two independently-drawn tasks
# happen to match.
DUAL_MERGE_BIAS_RATE = 1.0
# `build_compound` re-rolls a merge opportunity `pick_compound_selection` did not
# happen to take (see there): this is how often it does.
DUAL_FORCE_MERGE_RATE = 1.0
# Share of dual `sequential`/`hybrid` conversations where the dual item(s) are
# guaranteed a front slot starting at position 0 (so the opening question is about
# the SECOND clip) rather than position 1 (so a natural clip-1 item leads instead,
# and the dual item is guaranteed a slot from turn 2 on). Without this, the dual
# item's guaranteed-inclusion mechanism (`first_key`) would make the opening
# question about the second clip every single time one is active.
DUAL_LEAD_RATE = 0.5

MAX_VOICE_FOLLOWUPS = 2


def qualify_voice_question(question, ilang):
    """Point a voice question at the most recent clip."""
    body = question[0].lower() + question[1:]
    return random.choice(VOICE_QUALIFIER[ilang]).format(q=body)


def append_voice_followups(turns, ilang, meta, probs, qualify=False):
    """Append text-only questions about the voice itself: accent, gender, age.

    Each kind is drawn INDEPENDENTLY, at most MAX_VOICE_FOLLOWUPS of them; firing only
    one let accent, the most available, starve the other two down to 0.4%. Each
    probability is conditional on the label being present in the TSV.

    `qualify` names the clip asked about and MUST be set when the conversation carries
    several audios: `meta` describes the last clip only.

    Returns the number of follow-ups appended.
    """
    candidates = []
    if meta["accent"]:
        candidates.append(("accent", probs["accent"]))
    if meta["gender"]:
        candidates.append(("gender", probs["gender"]))
    if meta["age"]:
        candidates.append(("age", probs["age"]))
    random.shuffle(candidates)

    added = 0
    for kind, prob in candidates:
        if added >= MAX_VOICE_FOLLOWUPS or random.random() >= prob:
            continue
        if kind == "accent":
            question = random.choice(FOLLOWUP_ACCENT[ilang])
            answer = random.choice(ACCENT_ANSWERS[ilang]).format(accent=meta["accent"])
        elif kind == "gender":
            question = random.choice(FOLLOWUP_GENDER[ilang])
            answer = random.choice(GENDER_ANSWERS[ilang][meta["gender"]])
        else:
            question = random.choice(FOLLOWUP_AGE[ilang])
            answer = random.choice(AGE_ANSWERS[ilang]).format(
                age=AGE_LABELS[ilang][meta["age"]])
        if qualify:
            question = qualify_voice_question(question, ilang)
        turns.append(user_text(question))
        turns.append(assistant(answer))
        added += 1
    return added


def followup_transcribe(ilang, src, asked):
    """The elliptical "and the transcription?" turn, naming the audio language or not.

    A bare "What exactly was said?" only works once the conversation has established
    what language the clip is in. It often has not: the translation bank's `target_only`
    phrasings ("Translate this file into German.") name the target and nothing else, so
    after two German and Italian answers there is nothing saying the audio was French,
    and the ask has no determinate answer -- the model has to guess between transcribing
    and translating into the instruction language.

    `asked` is the user turns already emitted in this segment. The language counts as
    established when one of them spells its name in `ilang`, which is exactly what
    separates the bank's `full` phrasings from its `target_only` ones. Otherwise the ask
    either names the language or points at the audio's own ("in the source audio
    language"), half and half.
    """
    name = LANG_NAMES[ilang][src]
    if any(name in text for text in asked):
        return random.choice(FOLLOWUP_TRANSCRIBE[ilang])
    if random.random() < FOLLOWUP_TRANSCRIBE_NAMED_SHARE:
        return fmt(random.choice(FOLLOWUP_TRANSCRIBE_NAMED[ilang]), src, ilang)
    return random.choice(FOLLOWUP_TRANSCRIBE_SOURCE[ilang])


def collect_delivered(rec, trace):
    """The (key, text) pairs a segment actually answered, in the order it answered them.

    `trace` holds one entry per content turn, each listing the item keys that turn
    covered (None for the transcript). Read back from the trace rather than recomposed,
    so a recap can never list something the conversation did not say.
    """
    texts = dict(rec["targets"])
    return [(key, rec["transcript"] if key is None else texts[key])
            for group in trace for key in group]


def append_recap(turns, ilang, delivered, n_content_turns, rate):
    """Append a closing "group all of that together" turn, or nothing.

    `delivered` is one (record, items) pair per clip, in conversation order. The answer
    re-renders those items and nothing else: no new transcription, no new translation,
    which is what makes the turn answerable from context alone. With several clips the
    blocks are labelled by clip, because otherwise two transcripts in the same reply
    have nothing telling them apart.

    Gated on CONTENT TURNS, not on items: a lone compound turn already answered three
    things in one structured block, so recapping it produces a near-copy of the turn
    right above it.

    Returns whether a recap was appended.
    """
    if n_content_turns < RECAP_MIN_TURNS or random.random() >= rate:
        return False

    # A bare numbered list is fine when position maps one line to one clip. Across
    # `delivered` entries it does not (a recap block is one clip, "1." under "Clip 2"
    # says nothing about transcript vs. translation) -- forced "labelled". Within a
    # single entry, numbering is unambiguous UNLESS that entry itself carries a dual
    # item whose lines don't map one-to-one to its clips -- see `numbered_style_mode`.
    if len(delivered) > 1:
        style = "labelled"
    else:
        _rec0, _items0 = delivered[0]
        _transcript0 = next((text for key, text in _items0 if key is None), None)
        _dual_text0 = next((text for key, text in _items0 if key == DUAL_KEY), None)
        _dual2_text0 = next((text for key, text in _items0 if key == DUAL_KEY2), None)
        _extra_duals0 = ([(1, _rec0.get("dual2_task"), _dual2_text0)]
                         if _dual2_text0 is not None else [])
        _translations0 = [(key, text) for key, text in _items0
                          if key is not None and key not in (DUAL_KEY, DUAL_KEY2)]
        _mode0 = numbered_style_mode(_transcript0 is not None, _translations0,
                                     _dual_text0, _extra_duals0)
        if _mode0 != "excluded":
            style = random.choice(RECAP_STYLES)
        else:
            style = "labelled"
    blocks = []
    for index, (rec, items) in enumerate(delivered, 1):
        transcript = next((text for key, text in items if key is None), None)
        dual_text = next((text for key, text in items if key == DUAL_KEY), None)
        dual_task = rec.get("dual_task") if dual_text is not None else None
        dual2_text = next((text for key, text in items if key == DUAL_KEY2), None)
        extra_duals = ([(1, rec.get("dual2_task"), dual2_text)]
                       if dual2_text is not None else [])
        translations = [(key, text) for key, text in items
                        if key is not None and key not in (DUAL_KEY, DUAL_KEY2)]
        block = render_compound_answer(ilang, transcript, translations, style,
                                       dual_task=dual_task, dual_text=dual_text,
                                       extra_duals=extra_duals)
        if len(delivered) > 1:
            label = RECAP_CLIP_LABEL[ilang].format(
                n=index, lang=LANG_NAMES[ilang][rec["meta"]["lang"]])
            block = f"{label}\n{block}"
        blocks.append(block)

    lead = random.choice(RECAP_LEAD[ilang])
    body = "\n\n".join(blocks)
    turns.append(user_text(random.choice(RECAP_QUESTION[ilang])))
    turns.append(assistant(f"{lead}\n{body}" if lead else body))
    return True


def pick_compound_selection(targets, max_asks):
    """Choose what a single compound turn asks for.

    Kept deliberately short (2-3 asks): the point is compositionality in a
    single turn, and a five-part instruction is a different, rarer thing that
    would dominate the sample if allowed. Long chains belong in `sequential`.
    Returns (include_transcript, chosen_targets).
    """
    include_transcript = random.random() < 0.65
    n_translations = max(1, min(len(targets), max_asks - (1 if include_transcript else 0)))
    chosen = targets[:n_translations]
    if not include_transcript and len(chosen) < 2:
        # A single translation is not a compound instruction at all.
        include_transcript = True
    return include_transcript, chosen


def compound_turns(clip, transcript, chosen, ilang, include_transcript, src=None,
                   dual_clip=None, dual_task=None, dual_text=None, dual_clip2=None,
                   extra_duals=()):
    """Render one compound user turn plus its composed answer, or None.

    Every shape is assembled from fragments. data/contexts has no combined
    transcribe-then-translate section, and the fr->en bank that used to cover the
    narrow "transcript plus exactly one translation" case is gone; the fragments
    already handled everything wider and name the audio language when it differs
    from the instruction language, so there is nothing the bank was adding.

    `dual_clip`/`dual_task`/`dual_text` fold a second clip's task into the same
    turn: its audio right after the first, its clause joined in with the others
    (merged into one "both clips" clause when it matches the first clip's own task).
    `dual_clip2` is a rare third clip's audio. `extra_duals` -- `(idx, task, text)`
    triples -- are extra facts that never merge: a second fact about the second
    clip (its audio already covered by `dual_clip`) and/or the third clip's task.
    """
    fragments, merged, glued = build_task_fragments(ilang, include_transcript, chosen, dual_task,
                                                    dual_text, src, extra_duals=extra_duals)
    # A merged "both clips" clause is already a complete two-clip instruction on its
    # own; the 2-clause floor only exists so a single ordinary ask (one clip, one
    # task) never gets sold as "compound".
    if len(fragments) < (1 if merged else 2):
        return None
    tasks = join_fragments(ilang, fragments)
    # COMPOUND_FRAMES_ANY/TWO say "cet extrait"/"this clip" (singular, ONE clip's
    # language): wrong the moment any dual item -- merged or not -- is also being
    # named in `tasks`.
    has_dual = dual_text is not None or bool(extra_duals)

    if merged and len(fragments) == 1:
        # A lone merged clause ("transcris les deux audios") is already a full
        # instruction; a frame naming the clips again ("Pour ces audios : transcris
        # les deux audios.") would just repeat itself.
        instruction = tasks[0].upper() + tasks[1:]
    elif has_dual:
        instruction = random.choice(COMPOUND_FRAMES_DUAL[ilang]).format(tasks=tasks)
    else:
        # Counted frames only when the count matches.
        pool = list(COMPOUND_FRAMES_ANY[ilang])
        if len(fragments) == 2:
            pool += COMPOUND_FRAMES_TWO[ilang]
        instruction = random.choice(pool).format(tasks=tasks, src=LANG_NAMES[ilang][src])

    if glued:
        first_text = transcript if dual_task is None else chosen[0][1]
        answer = f"{first_text.strip()} {dual_text.strip()}"
    else:
        answer = render_compound_answer(
            ilang, transcript if include_transcript else None, chosen,
            pick_answer_style(numbered_style_mode(include_transcript and transcript is not None,
                                                  chosen, dual_text, extra_duals)),
            dual_task=dual_task, dual_text=dual_text, extra_duals=extra_duals,
        )
    turns = [user_text(instruction), user_audio(clip["audio"], clip["duration"])]
    if dual_clip is not None:
        turns.append(user_audio(dual_clip["audio"], dual_clip["duration"]))
    if dual_clip2 is not None:
        turns.append(user_audio(dual_clip2["audio"], dual_clip2["duration"]))
    turns.append(assistant(answer))
    return turns


def build_compound(stem, clip, transcript, targets, ilang, meta, probs, max_asks,
                   *, first_key=_MISSING, first_start=0, last_key=_MISSING, turn_dist=None,
                   min_turns=1, voice=True, trace=None, continuation=False, dual_clip=None,
                   dual_task=None, dual_clip2=None, dual2_task=None,
                   dual_extra_task=None, dual_extra_text=None):
    """A single compound turn and nothing else.

    `first_start` is unused here (accepted only so every builder shares one call
    signature) -- a lone turn has no "which turn asks it first" to randomize; the
    dual item's guaranteed spot comes from `targets` itself already carrying it up
    front (see `build_conversation`), which `pick_compound_selection` always keeps.

    The keyword arguments exist so every builder shares one call signature; a
    single turn has no length to draw and nothing to pin to the front. `dual_clip`
    is honoured only through `targets` carrying a `DUAL_KEY` entry -- `first_key`
    puts it at position 0, which `pick_compound_selection` always keeps. Likewise
    `dual_clip2`/`DUAL_KEY2` for the rare third clip, at position 1.

    `dual_extra_task`/`dual_extra_text` are a SECOND fact about the second clip,
    never routed through `targets` at all (no new audio to guarantee) -- just an
    alternative source for the padding clause below, so it is not always the first
    clip's.
    """
    include_transcript, chosen = pick_compound_selection(targets, max_asks)
    dual_text = next((text for tgt, text in chosen if tgt == DUAL_KEY), None)
    dual2_text = next((text for tgt, text in chosen if tgt == DUAL_KEY2), None)
    chosen = [(tgt, text) for tgt, text in chosen if tgt not in (DUAL_KEY, DUAL_KEY2)]
    extra_duals = [(1, dual2_task, dual2_text)] if dual2_text is not None else []

    would_merge = dual_text is not None and (
        (dual_task is None and include_transcript)
        or any(tgt == dual_task for tgt, _ in chosen))
    # `dual_task` was already biased (in `build_conversation`) toward a task the
    # first clip can ALSO answer, but `pick_compound_selection` draws
    # `include_transcript`/`chosen` independently of that, so half the time the
    # coincidence goes to waste (transcript matches, but this turn happens not to
    # be asking for the transcript). Forced back in here, so the earlier bias
    # actually turns into a merged instruction most of the time it is available,
    # rather than being diluted by an unrelated draw.
    if dual_text is not None and not would_merge and random.random() < DUAL_FORCE_MERGE_RATE:
        if dual_task is None:
            include_transcript = True
            would_merge = True
        else:
            match = next((t for t in targets
                         if t[0] == dual_task and t not in chosen), None)
            if match is not None:
                chosen = [match] + chosen[:max(0, max_asks - 2)]
                would_merge = True
    # `pick_compound_selection` always reserves a second slot alongside the dual item,
    # so a merge ("transcris les deux audios") would otherwise ALWAYS come padded with
    # one more clause -- and that clause was always about the first clip, since the
    # second clip only ever had the one task the merge just consumed. A real second
    # fact about it (`dual_extra_*`) lets the padding name the second clip instead;
    # failing that, dropping the padding entirely lets the merged ask stand alone.
    if would_merge and chosen:
        # Checked FIRST, over the whole mergeable pool: a shared instruction is
        # meant to be the commonest dual shape, so `DUAL_MERGE_ALONE_RATE` must
        # apply before any padding decision spends part of that mass.
        if random.random() < DUAL_MERGE_ALONE_RATE:
            chosen = []
        elif dual_extra_text is not None and random.random() < DUAL_EXTRA_ON_SECOND_RATE:
            extra_duals.append((0, dual_extra_task, dual_extra_text))
            chosen = []

    turns = compound_turns(clip, transcript, chosen, ilang, include_transcript,
                           meta["lang"], dual_clip=dual_clip if dual_text else None,
                           dual_task=dual_task, dual_text=dual_text,
                           dual_clip2=dual_clip2 if dual2_text else None,
                           extra_duals=extra_duals)
    if turns is not None and trace is not None:
        trace.append(([None] if include_transcript else [])
                     + ([DUAL_KEY] if dual_text is not None else [])
                     + ([DUAL_KEY2] if dual2_text is not None else [])
                     + [t for t, _ in chosen])
    return turns


def build_hybrid(stem, clip, transcript, targets, ilang, meta, probs, max_asks,
                 *, first_key=_MISSING, first_start=0, last_key=_MISSING, turn_dist=TURN_DIST,
                 min_turns=2, voice=True, trace=None, continuation=False, dual_clip=None,
                 dual_task=None, dual_clip2=None, dual2_task=None,
                 dual_extra_task=None, dual_extra_text=None):
    """Several user turns, exactly one of which bundles 2-3 instructions.

    The bundled turn is positioned over ALL turns, so "several instructions" is never
    learned as meaning "first turn". `targets` carrying a `DUAL_KEY`/`DUAL_KEY2` entry
    (forced to a front slot, in that order, starting at `first_start` -- see
    `conversation_items`) flows through the same item pool as everything else, so
    each can land in a single turn or inside the bundle, wherever the draw puts it.
    `dual_extra_task`/`dual_extra_text` (a second fact about the second clip, no
    `targets` entry of its own) is only honoured in the bundled turn, same as in
    `build_compound`.
    """
    src = meta["lang"]
    # A one-turn hybrid is just a compound.
    min_turns = max(2, min_turns)
    items = conversation_items(transcript, targets, first_key, first_start)
    bundle_size = random.choice(BUNDLE_SIZES)
    # A hybrid of T turns consumes (T-1) single asks plus the bundle.
    n_turns = draw_turns(len(items) - bundle_size + 1, turn_dist)
    if n_turns < min_turns:
        # Retry with the smallest bundle: three items leave no room for a second turn.
        bundle_size = 2
        n_turns = draw_turns(len(items) - bundle_size + 1, turn_dist)
        if n_turns < min_turns:
            return None

    compound_at = draw_compound_position(n_turns)
    # Only when the last turn is a single ask: a bundle cannot be inherited elliptically.
    if (last_key is not _MISSING and last_key != first_key
            and compound_at != n_turns - 1):
        place_key(items, n_turns + bundle_size - 2, last_key)

    # Leftover items are DROPPED, not appended to the last group, which would give the
    # conversation a second bundle.
    groups, cursor = [], 0
    for i in range(n_turns):
        take = bundle_size if i == compound_at else 1
        groups.append(items[cursor:cursor + take])
        cursor += take

    turns = []
    # User turns already emitted, read back by `followup_transcribe` to decide whether the
    # audio language is established. In a continuation the FIRST one is left out: it is
    # about to be overwritten by the elliptic lead, so it cannot be relied on.
    asked = []
    if trace is not None:
        trace.extend([tgt for tgt, _ in group] for group in groups)
    for i, group in enumerate(groups):
        tr_text = next((text for tgt, text in group if tgt is None), None)
        dual_text = next((text for tgt, text in group if tgt == DUAL_KEY), None)
        dual2_text = next((text for tgt, text in group if tgt == DUAL_KEY2), None)
        translations = [(tgt, text) for tgt, text in group
                        if tgt is not None and tgt not in (DUAL_KEY, DUAL_KEY2)]
        first = i == 0

        if len(group) == 1:
            tgt, text = group[0]
            if tgt is None:
                question = (bank_prompt("transcription", ilang, src) if first
                            else followup_transcribe(ilang, src, asked))
                if dual_clip is not None:
                    question = qualify_first_clip(question, ilang)
            elif tgt == DUAL_KEY:
                question = dual_question(ilang, 0, dual_task, first, has_third=dual_clip2 is not None)
            elif tgt == DUAL_KEY2:
                question = dual_question(ilang, 1, dual2_task, first)
            else:
                question = (bank_prompt("translation", ilang, src, tgt) if first
                            else fmt(random.choice(FOLLOWUP_TRANSLATE[ilang]), tgt, ilang))
                if dual_clip is not None:
                    question = qualify_first_clip(question, ilang)
            answer = text
        elif first:
            extra_duals = [(1, dual2_task, dual2_text)] if dual2_text is not None else []
            if dual_extra_text is not None and not extra_duals \
                    and random.random() < DUAL_EXTRA_ON_SECOND_RATE:
                extra_duals.append((0, dual_extra_task, dual_extra_text))
            block = compound_turns(clip, tr_text, translations, ilang,
                                   tr_text is not None, src,
                                   dual_clip=dual_clip if dual_text else None,
                                   dual_task=dual_task, dual_text=dual_text,
                                   dual_clip2=dual_clip2 if dual2_text else None,
                                   extra_duals=extra_duals)
            if block is None:
                return None
            turns.extend(block)
            if not continuation:
                asked.extend(t["value"] for t in block
                             if t["from"] == "User" and t["type"] == "text")
            continue
        elif dual_text is not None or dual2_text is not None:
            # The langs/adjs frames below assume every non-transcript item is a
            # translation, which the dual item is not -- phrase this bundle as a
            # joined clause list instead, same machinery as `compound_turns`. Only
            # DUAL_KEY (idx 0) is merge-eligible; DUAL_KEY2 is always its own clause.
            extra_duals = [(1, dual2_task, dual2_text)] if dual2_text is not None else []
            fragments, _merged, glued = build_task_fragments(ilang, tr_text is not None, translations,
                                                              dual_task, dual_text, src,
                                                              extra_duals=extra_duals)
            question = random.choice(COMPOUND_FOLLOWUP_WITH_DUAL[ilang]).format(
                tasks=join_fragments(ilang, fragments))
            if glued:
                first_text = tr_text if dual_task is None else translations[0][1]
                answer = f"{first_text.strip()} {dual_text.strip()}"
            else:
                answer = render_compound_answer(ilang, tr_text, translations,
                                                pick_answer_style(numbered_style_mode(
                                                    tr_text is not None, translations,
                                                    dual_text, extra_duals)),
                                                dual_task=dual_task, dual_text=dual_text,
                                                extra_duals=extra_duals)
        else:
            langs = oxford_join([LANG_NAMES[ilang][t] for t, _ in translations], ilang)
            adjs = oxford_join([LANG_ADJ[ilang][t] for t, _ in translations], ilang)
            frame_pool = (COMPOUND_FOLLOWUP_WITH_TRANSCRIPT if tr_text
                          else COMPOUND_FOLLOWUP_FRAMES)[ilang]
            question = random.choice(frame_pool).format(langs=langs, adjs=adjs)
            # This bundle is clip-1-only, but a dual item asked elsewhere in the
            # segment still put a second clip in the room.
            if dual_clip is not None:
                question = qualify_first_clip(question, ilang)
            # Clip-1-only content: always a single audio, so numbering is
            # unambiguous ("uniform") regardless of a dual item elsewhere in
            # the segment.
            answer = render_compound_answer(ilang, tr_text, translations,
                                            pick_answer_style("uniform"))

        turns.append(user_text(question))
        if not (first and continuation):
            asked.append(question)
        if first:
            turns.append(user_audio(clip["audio"], clip["duration"]))
            # Reached only through the single-item branch (the `elif first:` bundle
            # inserts its own audio inside `compound_turns`); `first_key` forces
            # DUAL_KEY to position 0, so a lone item 0 is always it, and dual_clip is
            # therefore always set here. dual_clip2 is inserted here TOO, even though
            # DUAL_KEY2 (forced to position 1) is not necessarily this turn's own
            # item -- `compound_at` can put it several turns later, in its own single
            # ask or a later bundle, and by then its audio must already have been
            # heard. Delivering both extra clips at turn 1 unconditionally (like
            # `build_sequential` does) is what guarantees that regardless of where
            # DUAL_KEY2 actually lands.
            if dual_clip is not None:
                turns.append(user_audio(dual_clip["audio"], dual_clip["duration"]))
            if dual_clip2 is not None:
                turns.append(user_audio(dual_clip2["audio"], dual_clip2["duration"]))
        turns.append(assistant(answer))

    if voice:
        append_voice_followups(turns, ilang, meta, probs)
    return turns


# How `mixed` splits between shapes.
MIXED_SHAPES = (("sequential", 0.40), ("compound", 0.20), ("hybrid", 0.40))
MIXED_HYBRID_SHARE = dict(MIXED_SHAPES)["hybrid"]
# Share of the non-hybrid mass going to compound. Redistributing the deficit with this
# fixed ratio keeps the global mix on 40/40/20 even where hybrids are impossible.
MIXED_COMPOUND_OF_REST = dict(MIXED_SHAPES)["compound"] / (
    dict(MIXED_SHAPES)["compound"] + dict(MIXED_SHAPES)["sequential"])
# `sequential` asks every item its own turn, so a dual/triple item glued into segment
# 0 can NEVER merge with the first clip's own task there -- only a bundled turn
# (`compound`, or `hybrid`'s bundled one) can produce a shared instruction
# ("transcribe both clips"). Once a dual item is active, `sequential`'s share of the
# draw is cut down and handed to `compound` instead, so a shared instruction is
# structurally reachable often enough to become the commonest dual shape (the
# DUAL_MERGE_BIAS_RATE/DUAL_MERGE_ALONE_RATE knobs then decide whether it actually
# fires and stands alone) rather than being capped by the ordinary 20% compound share.
DUAL_MIXED_SHAPES = (("sequential", 0.0), ("compound", 0.65), ("hybrid", 0.35))
DUAL_COMPOUND_OF_REST = dict(DUAL_MIXED_SHAPES)["compound"] / (
    dict(DUAL_MIXED_SHAPES)["compound"] + dict(DUAL_MIXED_SHAPES)["sequential"])


def build_mixed(stem, clip, transcript, targets, ilang, meta, probs, max_asks, **kw):
    """Draw one of the three shapes, so a single file carries all of them.

    The hybrid probability is CONDITIONAL and comes from the pool: a hybrid needs
    three answerable things and only French clips have that many, so drawing it at
    a flat 40% yields far less than 40% once seven single-target languages are in
    the pool. `probs["hybrid"]` is the compensated rate (see derive_shape_prob).
    """
    # Only on clips that can carry a bundle plus another turn; elsewhere the draw would
    # fail and fall back to sequential, halving the compound share.
    p_hybrid = probs.get("hybrid", MIXED_HYBRID_SHARE) if len(targets) >= 2 else 0.0
    compound_of_rest = MIXED_COMPOUND_OF_REST
    if kw.get("dual_clip") is not None:
        p_hybrid = max(p_hybrid, dict(DUAL_MIXED_SHAPES)["hybrid"])
        compound_of_rest = DUAL_COMPOUND_OF_REST
    if random.random() < p_hybrid:
        shape = "hybrid"
    else:
        shape = "compound" if random.random() < compound_of_rest else "sequential"
    builder = {"sequential": build_sequential, "compound": build_compound,
               "hybrid": build_hybrid}[shape]
    turns = builder(stem, clip, transcript, targets, ilang, meta, probs, max_asks, **kw)
    # `hybrid` legitimately declines on single-translation clips, i.e. everything but fr.
    if turns is None and shape == "hybrid":
        turns = build_sequential(stem, clip, transcript, targets, ilang, meta, probs,
                                 max_asks, **kw)
    return turns


def build_sequential(stem, clip, transcript, targets, ilang, meta, probs, max_asks=None,
                     *, first_key=_MISSING, first_start=0, last_key=_MISSING,
                     turn_dist=TURN_DIST, min_turns=2, voice=True, trace=None,
                     continuation=False, dual_clip=None, dual_task=None, dual_clip2=None,
                     dual2_task=None, dual_extra_task=None, dual_extra_text=None):
    """Audio in turn 1 only; every later user turn is text-only and elliptical.

    The turn count is drawn from TURN_DIST and then that many items are taken,
    so the length distribution is a parameter rather than a by-product of how
    many translations the clip happens to have. `targets` carrying a `DUAL_KEY`/
    `DUAL_KEY2` entry (forced to a front slot, in that order, starting at
    `first_start` -- see `conversation_items`) is just one more item drawn from
    the same pool, asked like any other -- only the extra clips' audio needs its
    own line, since they have to be heard before being asked about. The first
    clip's OWN turns are qualified ("Sur le premier extrait : ...") once a dual
    item is active, since a bare "et en anglais ?" no longer says which clip.
    """
    src = meta["lang"]
    items = conversation_items(transcript, targets, first_key, first_start)
    n_turns = draw_turns(len(items), turn_dist)
    if n_turns < min_turns:
        return None
    if last_key is not _MISSING and last_key != first_key and n_turns > 1:
        place_key(items, n_turns - 1, last_key)

    turns = []
    asked = []
    if trace is not None:
        trace.extend([tgt] for tgt, _ in items[:n_turns])
    for i, (tgt, text) in enumerate(items[:n_turns]):
        first = i == 0
        if tgt is None:
            question = (bank_prompt("transcription", ilang, src) if first
                        else followup_transcribe(ilang, src, asked))
            if dual_clip is not None:
                question = qualify_first_clip(question, ilang)
        elif tgt == DUAL_KEY:
            question = dual_question(ilang, 0, dual_task, first, has_third=dual_clip2 is not None)
        elif tgt == DUAL_KEY2:
            question = dual_question(ilang, 1, dual2_task, first)
        else:
            question = (bank_prompt("translation", ilang, src, tgt) if first
                        else fmt(random.choice(FOLLOWUP_TRANSLATE[ilang]), tgt, ilang))
            if dual_clip is not None:
                question = qualify_first_clip(question, ilang)
        turns.append(user_text(question))
        if not (first and continuation):
            asked.append(question)
        if first:
            turns.append(user_audio(clip["audio"], clip["duration"]))
            if dual_clip is not None:
                turns.append(user_audio(dual_clip["audio"], dual_clip["duration"]))
            if dual_clip2 is not None:
                turns.append(user_audio(dual_clip2["audio"], dual_clip2["duration"]))
        turns.append(assistant(text))

    if voice:
        append_voice_followups(turns, ilang, meta, probs)
    return turns


# ──────────────────────────────────────────────────────────────────────────────
# Composition: one conversation over one or several clips
# ──────────────────────────────────────────────────────────────────────────────

BUILDERS = {"sequential": build_sequential, "compound": build_compound,
            "mixed": build_mixed}


def apply_lead(turns, text=None, prefix=None):
    """Rewrite a segment's opening user turn so it reads as a continuation."""
    for turn in turns:
        if turn["from"] == "User" and turn["type"] == "text":
            turn["value"] = text if text is not None else f"{prefix} {turn['value']}"
            return


def count_audios(turns):
    return sum(1 for t in turns if t["type"] == "audio")


def elliptic_lead(carried, ilang, src, prev_src, rate):
    """The user turn opening an elliptic audio switch.

    `carried` is the task inherited from the previous segment: a target code for a
    translation, None for the transcript. It is spelled out in `rate` of the cases
    where a bare "and this one?" would leave the answer's language to be guessed --
    always for a translation, and for the transcript only when the clip changes
    source language.
    """
    if random.random() < rate:
        if carried is not None:
            return fmt(random.choice(CONTINUATION_ELLIPTIC_TARGET[ilang]), carried, ilang)
        if src != prev_src:
            return fmt(random.choice(CONTINUATION_ELLIPTIC_SOURCE[ilang]), src, ilang)
    return random.choice(CONTINUATION_ELLIPTIC[ilang])


def build_conversation(records, mode, probs, args, dual_rec=None, dual_rec2=None):
    """Chain one segment per clip into a single conversation.

    A segment is a normal conversation on one clip; only its opening changes. The
    instruction language is drawn once for the whole conversation.

    `dual_rec`, when given, glues a SECOND clip into segment 0's turn 1: both audios,
    one turn, its transcript folded into segment 0's `targets` as one more item
    (`DUAL_KEY`) alongside the real translations, forced to the front so whichever
    shape segment 0 draws (sequential/compound/hybrid) is guaranteed to ask it --
    same mechanism `last_key` already uses to steer where an item lands. `dual_rec2`
    (only ever set alongside `dual_rec`) does the same for a rare THIRD clip
    (`DUAL_KEY2`, forced to position 1). A second, distinct fact about `dual_rec`
    itself is also drawn here and passed down as `dual_extra_task`/`dual_extra_text`
    -- not routed through the pool (no new audio to guarantee), only ever picked up
    by `build_compound`'s padding decision.

    Returns (turns, n_clips_used) or None. A later segment whose builder declines just
    ends the conversation early instead of discarding what is already built.
    """
    ilang = random.choice(INSTRUCTION_LANGS)
    builder = BUILDERS[mode]
    turns, used, prev_trace, prev_src = [], 0, None, None
    # What each clip actually answered, for the optional closing recap.
    delivered, n_content_turns = [], 0

    # Decided up front: an elliptic switch constrains the segment BEFORE it, which has to
    # end on a task the next clip can answer.
    want_elliptic = [random.random() < args.elliptic_rate
                     for _ in range(max(0, len(records) - 1))]

    for index, rec in enumerate(records):
        continuation = index > 0
        trace = []
        kwargs = {"voice": False, "trace": trace, "continuation": continuation}
        carried = _MISSING
        if continuation:
            kwargs["turn_dist"] = CONT_TURN_DIST
            kwargs["min_turns"] = 1
            # Elliptic carries a SINGLE task the new clip can answer: after "translate into
            # German", that works on another French clip but not on a German one.
            last = prev_trace[-1] if prev_trace else None
            if (want_elliptic[index - 1] and last and len(last) == 1
                    and last[0] in rec["keys"]):
                carried = last[0]
                kwargs["first_key"] = carried
        if want_elliptic[index:index + 1] == [True]:
            shared = rec["keys"] & records[index + 1]["keys"]
            if shared:
                # Prefer a translation over the transcript so the carried task varies.
                pool = sorted(shared - {None}, key=str) or [None]
                kwargs["last_key"] = random.choice(pool)

        targets = rec["targets"]
        if index == 0 and dual_rec is not None:
            # Drawn from the second clip's OWN pool -- its transcript, or (as often)
            # one of its real translations -- so a translate-the-dual-item ask has a
            # genuine, distinct-language answer instead of always repeating the same
            # transcript under a translation-shaped question.
            dual_pool = conversation_items(dual_rec["transcript"], dual_rec["targets"])
            # Biased toward a task the first clip can ALSO answer (the transcript
            # always qualifies -- every clip has one -- a shared translation
            # target sometimes does too), so the two clips coincide on the same
            # ask often enough for `build_compound`'s merge ("transcris les deux
            # audios") to be the rule rather than a rare coincidence. The
            # unbiased draw is kept as the rest of the mass, so the dual item
            # still lands on a genuinely distinct-language answer sometimes.
            shared_pool = [item for item in dual_pool if item[0] in rec["keys"]]
            if shared_pool and random.random() < DUAL_MERGE_BIAS_RATE:
                # The transcript is the one shared task `pick_compound_selection`
                # includes most reliably (65% of compound turns, vs. a translation
                # target that has to also be among the few "chosen" ones) --
                # preferred within the shared pool so the bias actually lands on
                # a mergeable draw most of the time, not just a theoretically
                # shared one.
                transcript_shared = [item for item in shared_pool if item[0] is None]
                pool = (transcript_shared if transcript_shared and random.random() < 0.7
                        else shared_pool)
                dual_task, dual_text = random.choice(pool)
            else:
                dual_task, dual_text = random.choice(dual_pool)
            keys = [DUAL_KEY]
            # A new list: `targets` is the pool's own, reused (and reshuffled) across
            # later draws of this clip, and must not carry the dual item into those.
            targets = [(DUAL_KEY, dual_text)] + list(targets)
            kwargs["dual_clip"] = dual_rec["clip"]
            kwargs["dual_task"] = dual_task
            new_keys = rec["keys"] | {DUAL_KEY}

            # A second, DIFFERENT fact about the same second clip -- not a pool item,
            # only ever used by `build_compound` to keep a merged ask's padding clause
            # from always being about the first clip.
            other = [item for item in dual_pool if item[0] != dual_task]
            if other and random.random() < args.dual_extra_rate:
                kwargs["dual_extra_task"], kwargs["dual_extra_text"] = random.choice(other)

            dual2_task = None
            if dual_rec2 is not None:
                dual2_task, dual2_text = random.choice(
                    conversation_items(dual_rec2["transcript"], dual_rec2["targets"]))
                targets = [(DUAL_KEY2, dual2_text)] + targets
                keys.append(DUAL_KEY2)
                kwargs["dual_clip2"] = dual_rec2["clip"]
                kwargs["dual2_task"] = dual2_task
                new_keys = new_keys | {DUAL_KEY2}

            kwargs["first_key"] = keys
            # Forcing the dual item(s) to position 0 every time would make "what
            # does the FIRST clip say" never the opening question of a dual
            # conversation -- half the time they are guaranteed a spot from turn 2
            # on instead, leaving position 0 to whatever the pool would have put
            # there anyway (only `build_sequential`/`build_hybrid` read this;
            # `build_compound` bundles everything into one turn regardless).
            kwargs["first_start"] = 0 if random.random() < DUAL_LEAD_RATE else 1
            rec = {**rec, "targets": targets, "keys": new_keys, "dual_task": dual_task,
                  "dual2_task": dual2_task}

        segment = builder(rec["stem"], rec["clip"], rec["transcript"], targets,
                          ilang, rec["meta"], probs, args.max_compound_asks, **kwargs)
        if segment is None:
            if index == 0:
                return None
            break

        if continuation:
            # An elliptic lead in front of a bundled answer would be an instruction never given.
            if carried is not _MISSING and len(trace[0]) == 1:
                apply_lead(segment, text=elliptic_lead(
                    carried, ilang, rec["meta"]["lang"], prev_src,
                    args.elliptic_lang_rate))
            else:
                apply_lead(segment, prefix=random.choice(CONTINUATION_EXPLICIT[ilang]))

        turns.extend(segment)
        delivered.append((rec, collect_delivered(rec, trace)))
        n_content_turns += len(trace)
        prev_trace = trace
        prev_src = rec["meta"]["lang"]
        used = index + 1

    # Appended here, not by the builders: they must come last and describe the LAST clip.
    # A lone compound turn is left alone, that shape being 20% of `mixed` by design.
    content_texts = sum(1 for t in turns if t["from"] == "User" and t["type"] == "text")
    n_audios = count_audios(turns)
    # Before the voice questions, so what it regroups is exactly the transcription and
    # translation material -- accent, gender and age are not in `delivered` and a recap
    # that skipped them while sitting under them would read as an omission.
    append_recap(turns, ilang, delivered, n_content_turns, args.recap_rate)
    if not (content_texts == 1 and n_audios == 1):
        # The dual clip, when it is what segment 0 answered with, is the most
        # recently heard audio -- voice questions should point at it, not at the
        # first clip whose audio came before it in the same turn.
        if used == 1 and dual_rec2 is not None:
            voice_meta = dual_rec2["meta"]
        elif used == 1 and dual_rec is not None:
            voice_meta = dual_rec["meta"]
        else:
            voice_meta = records[used - 1]["meta"]
        append_voice_followups(turns, ilang, voice_meta, probs, qualify=n_audios > 1)
    return turns, used


# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────

def load_source(cv_root, ast_root, src, split, args):
    """Build the record pool for one source language.

    `keys` is the set of things a clip can answer (None for the transcript plus its
    target codes); the composer intersects it with the previous segment's last ask to
    decide whether an elliptic "and this one?" is answerable at all.
    """
    manifests = discover_ast_manifests(ast_root, src, split, args.variant)
    if args.targets:
        manifests = {k: v for k, v in manifests.items() if k in args.targets}
    if not manifests:
        return []

    by_stem = defaultdict(dict)
    for tgt, path in sorted(manifests.items()):
        loaded = load_ast_manifest(path)
        for stem, payload in loaded.items():
            by_stem[stem][tgt] = payload
        print(f"[{split}][{src}] {tgt}: {len(loaded)} samples from {path}", file=sys.stderr)

    # Manifests first, so the TSV can be filtered to joinable stems: eight full
    # CommonVoice TSVs are ~1.4 GB of text.
    cv = load_cv_index(cv_root, src, args.tsv_name, keep=set(by_stem))

    records = []
    for stem, per_target in by_stem.items():
        meta = cv.get(stem)
        if meta is None or not meta["sentence"]:
            continue
        shuffled = sorted(per_target.items())
        random.shuffle(shuffled)
        all_targets = [(tgt, payload["text"]) for tgt, payload in shuffled]
        targets = all_targets[: args.max_targets]
        records.append({
            "stem": stem,
            "clip": per_target[all_targets[0][0]],
            "transcript": meta["sentence"],
            "targets": targets,
            "meta": meta,
            "keys": {None} | {tgt for tgt, _ in targets},
        })
    print(f"[{split}][{src}] {len(records)} records "
          f"({len(records) / max(1, len(by_stem)) * 100:.1f}% of {len(by_stem)} manifest stems "
          f"joined to {args.tsv_name})", file=sys.stderr)
    random.shuffle(records)
    return records


def parse_lang_shares(specs):
    """Parse `lang=share` pairs into a dict."""
    shares = {}
    for spec in specs or []:
        lang, _, value = spec.partition("=")
        if not value:
            raise ValueError(f"--lang-share expects lang=share, got {spec!r}")
        shares[lang.strip()] = float(value)
    return shares


def lang_weights(sizes, shares):
    """Normalised draw probability per source language.

    Pool sizes are wildly uneven (en 1.14M clips against pt 23k) and French is
    the only language with more than one translation target, so drawing in
    proportion to size would both bury French and starve every long chain.
    Languages named in --lang-share get their share outright; the rest split the
    remainder in proportion to how much of them is left.
    """
    langs = [lang for lang, size in sizes.items() if size > 0]
    if not langs:
        return {}
    fixed = {lang: shares[lang] for lang in langs if lang in shares}
    free = [lang for lang in langs if lang not in fixed]
    rest = max(0.0, 1.0 - sum(fixed.values()))
    free_total = sum(sizes[lang] for lang in free) or 1
    raw = {lang: fixed[lang] if lang in fixed else rest * sizes[lang] / free_total
           for lang in langs}
    total = sum(raw.values())
    if total <= 0:
        return {lang: 1.0 / len(langs) for lang in langs}
    return {lang: weight / total for lang, weight in raw.items()}


def weighted_coverage(pools, weights, predicate):
    """Share of DRAWN clips satisfying `predicate`.

    Not the share of pooled clips: the pool is 15% French but the draw is 55%
    French, so any rate derived from raw pool composition is wrong by that
    factor. Every conditional probability in here goes through this.
    """
    return sum(weights.get(lang, 0.0)
               * sum(1 for r in records if predicate(r)) / max(1, len(records))
               for lang, records in pools.items())


def derive_voice_probs(pools, weights, args, split):
    """Turn requested output rates into per-label conditional probabilities.

    The flag is a share of generated conversations; the draw can only fire when the
    TSV declares the label, and that coverage varies by split and language.
    """
    wanted = {"accent": args.accent_rate, "gender": args.gender_rate, "age": args.age_rate}
    coverages = {kind: weighted_coverage(pools, weights, lambda r, k=kind: bool(r["meta"][k]))
                 for kind in wanted}
    probs = {}
    for kind, want in wanted.items():
        coverage = coverages[kind]
        if coverage <= 0:
            probs[kind] = 0.0
            continue
        probs[kind] = min(1.0, want / coverage)
        if want > coverage:
            print(f"[{split}] warning: {kind} requested at {want * 100:.1f}% but only "
                  f"{coverage * 100:.1f}% of joined clips declare one; capped there",
                  file=sys.stderr)
    print(f"[{split}] label coverage over drawn clips: "
          + "  ".join(f"{k} {coverages[k] * 100:.1f}%"
                      for k in ("accent", "gender", "age")), file=sys.stderr)
    return probs


def derive_shape_prob(pools, weights, split):
    """Conditional probability of drawing a hybrid, given the clip allows one.

    A hybrid needs three answerable things, so with fr<->* only French clips qualify
    and a flat 40% would deliver 40% OF THE FRENCH SHARE. Weighted by the language
    draw probability, not by pool size: fr is 15% of the records but 55% of the draws.
    """
    share = weighted_coverage(pools, weights, lambda r: len(r["targets"]) >= 2)
    if share <= 0:
        return 0.0
    prob = min(1.0, MIXED_HYBRID_SHARE / share)
    if MIXED_HYBRID_SHARE > share:
        print(f"[{split}] warning: {MIXED_HYBRID_SHARE * 100:.0f}% hybrids wanted but only "
              f"{share * 100:.1f}% of clips have two translations; capped there",
              file=sys.stderr)
    return prob


def build_reference_sets(source_langs):
    """Every user turn we can emit that asks for exactly ONE thing.

    Shape is read back off the output by exact membership rather than trusted from the
    draw; a keyword heuristic misreported it twice, in both directions.
    """
    single_ask, voice_questions, voice_kind = set(), set(), {}
    for kind, table in (("accent", FOLLOWUP_ACCENT), ("gender", FOLLOWUP_GENDER),
                        ("age", FOLLOWUP_AGE)):
        for ilang in INSTRUCTION_LANGS:
            for template in table[ilang]:
                variants = [template] + [f.format(q=template[0].lower() + template[1:])
                                         for f in VOICE_QUALIFIER[ilang]]
                for variant in variants:
                    voice_kind[variant] = kind
                    voice_questions.add(variant)

    # The language-qualified switches are templates: expanded over every language they
    # can name, so membership stays an exact match like everything else here.
    qualified = {ilang: {fmt(tpl, lang, ilang)
                         for table in (CONTINUATION_ELLIPTIC_TARGET,
                                       CONTINUATION_ELLIPTIC_SOURCE)
                         for tpl in table[ilang]
                         for lang in LANG_NAMES[ilang]}
                 for ilang in INSTRUCTION_LANGS}

    for ilang in INSTRUCTION_LANGS:
        single_ask |= set(FOLLOWUP_TRANSCRIBE[ilang])
        single_ask |= {fmt(t, lang, ilang) for t in FOLLOWUP_TRANSCRIBE_NAMED[ilang]
                       for lang in LANG_NAMES[ilang]}
        single_ask |= set(FOLLOWUP_TRANSCRIBE_SOURCE[ilang])
        single_ask |= set(CONTINUATION_ELLIPTIC[ilang])
        single_ask |= qualified[ilang]
        single_ask |= voice_questions
        for src in source_langs:
            single_ask |= set(asr_prompts(src, ilang))
        for tgt in LANG_NAMES[ilang]:
            for tpl in FOLLOWUP_TRANSLATE[ilang]:
                single_ask.add(fmt(tpl, tgt, ilang))
            for src in source_langs:
                if src != tgt:
                    single_ask |= set(ast_prompts(src, tgt, ilang))
    prefixes = tuple(p for ilang in INSTRUCTION_LANGS for p in CONTINUATION_EXPLICIT[ilang])
    elliptic = {p for ilang in INSTRUCTION_LANGS
                for p in list(CONTINUATION_ELLIPTIC[ilang]) + list(qualified[ilang])}
    # Recaps ask for no new task, so they are pulled out of `content` before the shape
    # is read: left in, every recap would look like a bundled turn and relabel the
    # conversation a hybrid.
    recap = {p for ilang in INSTRUCTION_LANGS for p in RECAP_QUESTION[ilang]}
    return single_ask, voice_questions, voice_kind, prefixes, elliptic, recap


def generate(args):
    cv_root = Path(args.cv_root)
    ast_root = Path(args.ast_root)
    out_root = Path(args.output_dir)
    shares = parse_lang_shares(args.lang_share)
    modes = ALL_MODES if args.mode == "all" else [args.mode]
    single_ask, voice_questions, voice_kind, cont_prefixes, elliptic_leads, recap_questions = \
        build_reference_sets(args.source_langs)

    for split in args.splits:
        pools = {}
        for src in args.source_langs:
            records = load_source(cv_root, ast_root, src, split, args)
            if records:
                pools[src] = records
            else:
                print(f"[{split}][{src}] no CommonVoice AST manifest, skipping",
                      file=sys.stderr)
        if not pools:
            print(f"[{split}] nothing to generate", file=sys.stderr)
            continue

        opening_weights = lang_weights({s: len(p) for s, p in pools.items()}, shares)
        probs = derive_voice_probs(pools, opening_weights, args, split)
        probs["hybrid"] = derive_shape_prob(pools, opening_weights, split)
        print(f"[{split}] opening language weights: " + "  ".join(
            f"{k} {v * 100:.0f}%" for k, v in sorted(opening_weights.items(),
                                                     key=lambda kv: -kv[1])),
            file=sys.stderr)

        cursors = {src: 0 for src in pools}
        passes = {src: 0 for src in pools}
        drawn = reused = 0

        def reuse_allowed():
            """Whether one more clip may be a second use of an already-used one.

            fr is the binding pool -- 593k clips against 1.19M draws at a 55% share --
            and empties before the file is full, after which the mix drifts to English.
            A French clip carries 7 translations and a conversation consumes 2 to 5, so
            a second pass over a reshuffled target list is a different conversation.
            """
            return reused + 1 <= args.reuse_rate * (drawn + 1)

        def effective_size(src):
            """Pool size for weighting; 0 when the language may not be drawn.

            Once a pool has wrapped every further draw from it is a second use, so the
            budget gates the whole second pass. Gating only the wrap let one allowed
            wrap open a free pass over the entire pool -- 14% reuse against a 5% cap.
            """
            left = len(pools[src]) - cursors[src]
            if passes[src] >= 1 or left <= 0:
                return len(pools[src]) if reuse_allowed() else 0
            return left

        def take(src):
            nonlocal drawn, reused
            if cursors[src] >= len(pools[src]):
                passes[src] += 1
                # Reshuffling the order matters as much as reshuffling the targets, else the
                # second pass pairs the same clips together again.
                random.shuffle(pools[src])
                cursors[src] = 0
            record = pools[src][cursors[src]]
            cursors[src] += 1
            drawn += 1
            record["uses"] = record.get("uses", 0) + 1
            if record["uses"] > 1:
                reused += 1
                # Permute the asks so the second conversation differs from the first.
                random.shuffle(record["targets"])
            return record

        def pick_lang(exclude=None):
            """Draw a source language from what is left, honouring --lang-share."""
            sizes = {s: effective_size(s) for s in pools if s != exclude}
            weights = lang_weights(sizes, shares)
            if not weights:
                return None
            langs = list(weights)
            return random.choices(langs, weights=[weights[s] for s in langs], k=1)[0]

        writers, counters = {}, {}
        for mode in modes:
            out_path = out_root / mode / f"{split}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            writers[mode] = out_path.open("w", encoding="utf-8")
            counters[mode] = 0

        voice_counts = defaultdict(Counter)
        shape_counts = defaultdict(Counter)
        length_counts = defaultdict(Counter)
        position_counts = defaultdict(Counter)
        audio_counts = defaultdict(Counter)
        cont_style = defaultdict(Counter)
        recap_counts = Counter()
        lang_counts = defaultdict(Counter)
        open_lang = defaultdict(Counter)
        groups = 0

        while any(effective_size(s) > 0 for s in pools):
            if all(counters[m] >= args.max_samples for m in modes):
                break
            primary = pick_lang()
            if primary is None:
                break
            group = [take(primary)]
            if random.random() < args.multi_audio_rate:
                n_extra = random.choices([n for n, _ in EXTRA_CLIP_DIST],
                                         weights=[w for _, w in EXTRA_CLIP_DIST], k=1)[0]
                for _ in range(n_extra):
                    lang = None
                    if random.random() < args.cross_lang_rate:
                        lang = pick_lang(exclude=group[-1]["meta"]["lang"])
                    if lang is None:
                        lang = pick_lang()
                    if lang is None:
                        break
                    group.append(take(lang))
            groups += 1

            # Drawn once per group, not per mode, like `group` itself: every mode builds
            # off the same clips. Independent of the multi-audio chain above -- this one
            # glues into segment 0's OWN turn rather than opening a new segment.
            dual_rec = dual_rec2 = None
            if random.random() < args.dual_transcription_rate:
                lang = None
                if random.random() < args.cross_lang_rate:
                    lang = pick_lang(exclude=group[0]["meta"]["lang"])
                if lang is None:
                    lang = pick_lang()
                if lang is not None:
                    dual_rec = take(lang)
                    # A rare THIRD clip, only ever alongside a second one.
                    if random.random() < args.dual_triple_rate:
                        lang2 = None
                        if random.random() < args.cross_lang_rate:
                            lang2 = pick_lang(exclude=dual_rec["meta"]["lang"])
                        if lang2 is None:
                            lang2 = pick_lang()
                        if lang2 is not None:
                            dual_rec2 = take(lang2)

            for mode in modes:
                if counters[mode] >= args.max_samples:
                    continue
                built = build_conversation(group, mode, probs, args, dual_rec=dual_rec,
                                           dual_rec2=dual_rec2)
                if built is None:
                    continue
                turns, used = built
                # A reused clip needs a distinct id, or two conversations collide on one key.
                pass_suffix = "" if group[0]["uses"] == 1 else f"_r{group[0]['uses']}"
                record = {"id": f"{group[0]['stem']}_cvchain{pass_suffix}_{mode}",
                          "conversations": turns}
                writers[mode].write(json.dumps(record, ensure_ascii=False) + "\n")
                counters[mode] += 1

                n_audios = count_audios(turns)
                audio_counts[mode][n_audios] += 1
                open_lang[mode][group[0]["meta"]["lang"]] += 1
                for rec in group[:used]:
                    lang_counts[mode][rec["meta"]["lang"]] += 1
                # The dual item(s) are forced into segment 0, so they are used whenever
                # build_conversation returned at all (a decline there returns None for
                # the whole conversation).
                if dual_rec is not None:
                    lang_counts[mode][dual_rec["meta"]["lang"]] += 1
                if dual_rec2 is not None:
                    lang_counts[mode][dual_rec2["meta"]["lang"]] += 1

                content, leads = [], []
                for turn in turns:
                    if turn["from"] != "User" or turn["type"] != "text":
                        continue
                    value = turn["value"]
                    kind = voice_kind.get(value)
                    if kind:
                        voice_counts[mode][kind] += 1
                        continue
                    if value in recap_questions:
                        recap_counts[mode] += 1
                        continue
                    if value in elliptic_leads:
                        cont_style[mode]["elliptic"] += 1
                        leads.append(len(content))
                    elif value.startswith(cont_prefixes):
                        cont_style[mode]["explicit"] += 1
                        leads.append(len(content))
                        # Strip the lead: what follows is an ordinary single instruction.
                        for prefix in cont_prefixes:
                            if value.startswith(prefix):
                                value = value[len(prefix):].strip()
                                break
                    content.append(value)

                # Read off the OPENING SEGMENT only: past a switch the turn index mixes
                # position with the previous segment length, and counting the whole
                # conversation labelled every multi-audio compound a hybrid.
                first = content[:leads[0]] if leads else content
                bundles = [i for i, v in enumerate(first) if v not in single_ask]
                if not bundles:
                    shape_counts[mode]["sequential"] += 1
                elif len(first) == 1:
                    shape_counts[mode]["compound"] += 1
                else:
                    shape_counts[mode]["hybrid"] += 1
                    position_counts[mode][min(bundles[0] + 1, 3)] += 1
                if len(content) > 1:
                    length_counts[mode][min(len(content), 6)] += 1

        for w in writers.values():
            w.close()

        pool_total = sum(len(p) for p in pools.values())
        print(f"[{split}] {groups} clip groups over {drawn}/{pool_total} records "
              f"({drawn / max(1, groups):.2f} clips per group), "
              f"{reused} reuses ({reused / max(1, drawn) * 100:.1f}% of draws)",
              file=sys.stderr)

        def pct(counter, denom):
            return "  ".join(f"{k}:{v / denom * 100:.0f}%" for k, v in sorted(counter.items()))

        for mode in modes:
            total = counters[mode]
            print(f"[{split}]   {mode}: {total} conversations -> "
                  f"{out_root / mode / f'{split}.jsonl'}", file=sys.stderr)
            if not total:
                continue
            if voice_counts[mode]:
                print(f"[{split}]     voice follow-ups: " + "  ".join(
                    f"{kind} {voice_counts[mode][kind] / total * 100:.1f}%"
                    for kind in ("accent", "gender", "age") if voice_counts[mode][kind]),
                    file=sys.stderr)
            if recap_counts[mode]:
                print(f"[{split}]     recap turns: "
                      f"{recap_counts[mode] / total * 100:.1f}% "
                      f"({recap_counts[mode]})", file=sys.stderr)
            if len(shape_counts[mode]) > 1:
                print(f"[{split}]     shapes: " + "  ".join(
                    f"{k} {v / total * 100:.0f}%"
                    for k, v in shape_counts[mode].most_common()), file=sys.stderr)
            print(f"[{split}]     audios per conversation: "
                  f"{pct(audio_counts[mode], total)}", file=sys.stderr)
            n_cont = sum(cont_style[mode].values())
            if n_cont:
                print(f"[{split}]     audio switches ({n_cont}): "
                      f"{pct(cont_style[mode], n_cont)}", file=sys.stderr)
            if open_lang[mode]:
                print(f"[{split}]     opening language: " + "  ".join(
                    f"{k} {v / total * 100:.0f}%"
                    for k, v in open_lang[mode].most_common()), file=sys.stderr)
            n_clips = sum(lang_counts[mode].values())
            if n_clips:
                print(f"[{split}]     clip languages: " + "  ".join(
                    f"{k} {v / n_clips * 100:.0f}%"
                    for k, v in lang_counts[mode].most_common()), file=sys.stderr)
            n_multi = sum(length_counts[mode].values())
            if n_multi:
                print(f"[{split}]     content turns ({n_multi} multi-turn): "
                      f"{pct(length_counts[mode], n_multi)}", file=sys.stderr)
            n_hybrid = sum(position_counts[mode].values())
            if n_hybrid:
                print(f"[{split}]     bundle at turn ({n_hybrid} hybrid openings): "
                      f"{pct(position_counts[mode], n_hybrid)}   (3 = 3rd or later)",
                      file=sys.stderr)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cv-root", required=True,
                   help="CommonVoice corpus root, the directory containing <lang>/validated.tsv")
    p.add_argument("--ast-root", required=True, help="$DATA_FOLDER/nemo/ast")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--source-langs", nargs="+",
                   default=["fr", "en", "de", "es", "it", "nl", "pt", "ar"],
                   help="Audio source languages to pool. Only fr has several translation "
                        "targets, so only fr carries long chains; the others contribute "
                        "transcription plus the French translation.")
    p.add_argument("--lang-share", nargs="*", default=["fr=0.55", "en=0.35"],
                   help="lang=share pairs fixing how often a language is drawn; languages left "
                        "out split the remainder in proportion to their remaining pool. The "
                        "default keeps fr above half and fr+en at 90% of openings.")
    p.add_argument("--multi-audio-rate", type=float, default=0.15,
                   help="Share of conversations that change audio part-way through. Composes "
                        "with the three shapes rather than replacing them: each segment is "
                        "still sequential, compound or hybrid.")
    p.add_argument("--dual-transcription-rate", type=float, default=0.0,
                   help="Share of conversations where a second clip is glued into segment "
                        "1's own turn 1 (both audios, no assistant reply between them), its "
                        "transcript folded into the normal ask pool as one more item. Composes "
                        "with sequential/compound/hybrid: whichever shape segment 1 draws, that "
                        "item is guaranteed to be asked -- alone, bundled with translations, "
                        "chained, recapped. 0 disables it.")
    p.add_argument("--dual-triple-rate", type=float, default=0.0,
                   help="Of the conversations already carrying a dual clip, the share that get "
                        "a THIRD clip glued into the same turn instead. Never merges into a "
                        "single 'both clips' clause (only the second clip does): always its own "
                        "explicitly-numbered ask. 0 disables it.")
    p.add_argument("--dual-extra-rate", type=float, default=0.0,
                   help="Share of dual conversations where the second clip's OWN pool also "
                        "supplies a second, distinct fact (its transcript plus one translation, "
                        "say) -- used only by the compound shape, to name the second clip in a "
                        "merged ask's padding clause instead of always the first. 0 disables it.")
    p.add_argument("--reuse-rate", type=float, default=0.05,
                   help="Hard cap on the share of clip draws reusing an already-used clip, "
                        "with its asks permuted. Only triggered by an exhausted pool: fr "
                        "empties past ~1M conversations. 0 disables reuse.")
    p.add_argument("--cross-lang-rate", type=float, default=0.50,
                   help="Within a multi-audio conversation, how often the next clip comes "
                        "from a DIFFERENT source language.")
    p.add_argument("--elliptic-rate", type=float, default=0.55,
                   help="Share of audio switches introduced with no instruction at all "
                        "(\"Et celui-ci ?\"), carrying the previous task over; the rest get an "
                        "explicit lead. Only applies when the new clip can answer that task.")
    p.add_argument("--elliptic-lang-rate", type=float, default=0.7,
                   help="Share of those elliptic switches that NAME the carried task's "
                        "language (\"Et celui-ci, en allemand ?\") instead of leaving it "
                        "implicit. Only applies where the bare form is ambiguous: a carried "
                        "translation, or a carried transcript on a clip in another language. "
                        "0 restores the fully bare form everywhere.")
    p.add_argument("--targets", nargs="*", default=None,
                   help="Restrict to these target languages (default: every one discovered)")
    p.add_argument("--splits", nargs="+", default=["train"])
    p.add_argument("--variant", default="context", choices=["context", "nocontext"],
                   help="AST manifest variant to read translations from")
    p.add_argument("--tsv-name", default="validated.tsv",
                   help="validated.tsv is a superset of train.tsv and joins more clips")
    p.add_argument("--mode", default="all", choices=ALL_MODES + ["all"])
    p.add_argument("--max-targets", type=int, default=6,
                   help="Cap on translations available to a conversation. A pool, not a "
                        "length: TURN_DIST decides how many are used. Too small and the "
                        "length distribution drifts.")
    p.add_argument("--max-compound-asks", type=int, default=3,
                   help="Cap on instructions packed into a single compound turn")
    p.add_argument("--max-samples", type=int, default=200_000, help="Per mode, per split")
    # Shares of generated conversations, converted at run time using the label coverage
    # measured on the clips actually drawn.
    p.add_argument("--accent-rate", type=float, default=0.05,
                   help="Share of conversations carrying an accent follow-up; these are the "
                        "non-transcribable questions that force a return to the audio.")
    p.add_argument("--gender-rate", type=float, default=0.05,
                   help="Same, for gender. The label is self-declared, so answers are phrased "
                        "as impressions.")
    p.add_argument("--age-rate", type=float, default=0.05,
                   help="Same, for age.")
    p.add_argument("--recap-rate", type=float, default=0.10,
                   help="Share of eligible conversations closing on a 'group all of that "
                        "together' turn, answered by restructuring what was already said "
                        f"and nothing else. Eligible = at least {RECAP_MIN_TURNS} content "
                        "turns. 0 disables.")
    p.add_argument("--contexts-root", default=None,
                   help="Override data/contexts, source of the turn-1 transcription "
                        "({lang}_asr_contexts.json) and translation "
                        "(translation/{src}-{tgt}_ast_contexts.json) instructions.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    CONTEXTS_ROOT = Path(args.contexts_root) if args.contexts_root else \
        Path(__file__).resolve().parent.parent.joinpath(*CONTEXTS_RELPATH)
    print(f"instruction banks: {CONTEXTS_ROOT}", file=sys.stderr)
    generate(args)
