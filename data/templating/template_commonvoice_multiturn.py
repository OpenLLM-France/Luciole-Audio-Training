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

# Turn-1 instructions come from the repo bank: 10 phrasings per task per language,
# all of which NAME the audio and output languages -- without that, "Transcribe this
# clip." over French audio never says which language to answer in. The bank is
# written for fr -> en; `substitute_languages` retargets it.
PROMPT_BANK_RELPATH = ("assets", "instruction_transcription_and_translation_fr-en.txt")
BANK_SOURCE_LANG = "fr"   # language the bank's prompts describe as the audio's
BANK_TARGET_LANG = "en"   # language the bank's prompts translate into
BANK_TASKS = ("transcription", "translation", "transcription_and_translation")

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

# With several clips in play "quel accent ?" has no referent, so the question is
# qualified to point at the latest audio.
VOICE_QUALIFIER = {
    "fr": ["Sur ce dernier audio : {q}", "Pour l'extrait qu'on vient d'entendre : {q}"],
    "en": ["On this last audio: {q}", "About the clip we just heard: {q}"],
}

_OXFORD_AND = {"fr": "et", "en": "and"}


def oxford_join(parts, ilang):
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f" {_OXFORD_AND[ilang]} " + parts[-1]

# Three answer shapes, so no single separator is learned as "the" format.
ANSWER_STYLES = ("labelled", "prose", "numbered")

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
BANK = None


def load_prompt_bank(path):
    """Load the repo's instruction bank and check it has what we draw from."""
    with open(path, encoding="utf-8") as f:
        bank = json.load(f)
    for task in BANK_TASKS:
        if task not in bank:
            raise ValueError(f"prompt bank {path} has no '{task}' section")
        for lang in INSTRUCTION_LANGS:
            if not bank[task].get(lang):
                raise ValueError(f"prompt bank {path}: '{task}' has no '{lang}' prompts")
    return bank


def fix_english_articles(text, names):
    """Repair a/an after a language substitution.

    The bank says "Provide an English translation"; swapping in German without
    touching the article yields "an German translation". Only English needs
    this -- the French prompts all use "en {langue}", which is article-free.
    """
    if not names:
        return text
    pattern = re.compile(r"\b([Aa]n?)(\s+)(" + "|".join(re.escape(n) for n in names) + r")\b")

    def repl(m):
        article, space, word = m.groups()
        correct = "an" if word[0].upper() in "AEIOU" else "a"
        return (correct.capitalize() if article[0].isupper() else correct) + space + word

    return pattern.sub(repl, text)


def substitute_languages(text, ilang, src, tgt):
    """Retarget a bank prompt from fr->en to src->tgt.

    Both names are swapped in ONE pass: doing them in sequence would rewrite
    French to English and then that same English back to French when the pair is
    reversed.
    """
    mapping = {LANG_NAMES[ilang][BANK_SOURCE_LANG]: LANG_NAMES[ilang][src]}
    if tgt is not None:
        mapping[LANG_NAMES[ilang][BANK_TARGET_LANG]] = LANG_NAMES[ilang][tgt]
    mapping = {k: v for k, v in mapping.items() if k != v}
    if not mapping:
        return text
    pattern = re.compile("|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)))
    out = pattern.sub(lambda m: mapping[m.group(0)], text)
    return fix_english_articles(out, set(mapping.values())) if ilang == "en" else out


def bank_prompt(task, ilang, src, tgt=None):
    """Draw one turn-1 instruction from the repo bank, retargeted to src/tgt."""
    return substitute_languages(random.choice(BANK[task][ilang]), ilang, src, tgt)


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


def conversation_items(transcript, targets, first_key=_MISSING):
    """Ordered pool of things a conversation can ask for.

    The transcript sits first only 60% of the time, so "turn 1 is the transcription"
    never becomes a rule. `first_key` pins one item to the front (None for the
    transcript): that is how an elliptic continuation asks the task it inherited.
    """
    items = list(targets)
    if transcript:
        idx = 0 if random.random() < 0.6 else random.randrange(1, len(items) + 1)
        items.insert(idx, (None, transcript))
    if first_key is not _MISSING:
        place_key(items, 0, first_key)
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


def render_compound_answer(ilang, transcript, translations, style):
    """Compose one answer covering several requested tasks.

    `translations` is an ordered list of (target_code, text). `transcript` may
    be None when the compound only chains translations.
    """
    sep = " : " if ilang == "fr" else ": "
    parts = []
    if style == "labelled":
        if transcript:
            parts.append(f"{LABELS[ilang]['transcription']}{sep}{transcript}")
        for tgt, text in translations:
            label = fmt(LABELS[ilang]["translation"], tgt, ilang)
            parts.append(f"{label}{sep}{text}")
        return "\n".join(parts)

    if style == "prose":
        if transcript:
            parts.append(random.choice(PROSE_TRANSCRIPTION[ilang]).format(
                text=quotable(transcript)))
        for tgt, text in translations:
            parts.append(fmt(random.choice(PROSE_TRANSLATION[ilang]), tgt, ilang,
                             text=quotable(text)))
        return " ".join(parts)

    # numbered
    items = ([transcript] if transcript else []) + [t for _, t in translations]
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))


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


def compound_turns(clip, transcript, chosen, ilang, include_transcript, src=None):
    """Render one compound user turn plus its composed answer, or None.

    Transcription + exactly one translation is the case the repo bank already
    covers, so it is drawn from there. Anything wider (several translations, or
    translations without the transcript) is assembled from fragments below --
    the bank has no phrasing for those.
    """
    if include_transcript and len(chosen) == 1 and src is not None:
        tgt, text = chosen[0]
        instruction = bank_prompt("transcription_and_translation", ilang, src, tgt)
        answer = render_compound_answer(ilang, transcript, chosen,
                                        random.choice(ANSWER_STYLES))
        return [user_text(instruction),
                user_audio(clip["audio"], clip["duration"]),
                assistant(answer)]

    fragments = []
    if include_transcript:
        # The frame already names the audio language; the fragment repeats it only when
        # the instruction language differs.
        if src is not None and src != ilang:
            fragments.append(fmt(random.choice(TASK_FRAGMENT_TRANSCRIBE_NAMED[ilang]),
                                 src, ilang))
        else:
            fragments.append(random.choice(TASK_FRAGMENT_TRANSCRIBE[ilang]))
    for tgt, _ in chosen:
        fragments.append(fmt(random.choice(TASK_FRAGMENT_TRANSLATE[ilang]), tgt, ilang))
    if len(fragments) < 2:
        return None

    tasks = fragments[0]
    for frag in fragments[1:]:
        tasks += random.choice(COORDINATORS[ilang]) + frag
    if not tasks.endswith("."):
        tasks += "."

    # Counted frames only when the count matches.
    pool = list(COMPOUND_FRAMES_ANY[ilang])
    if len(fragments) == 2:
        pool += COMPOUND_FRAMES_TWO[ilang]
    instruction = random.choice(pool).format(tasks=tasks,
                                             src=LANG_NAMES[ilang][src or BANK_SOURCE_LANG])

    answer = render_compound_answer(
        ilang, transcript if include_transcript else None, chosen,
        random.choice(ANSWER_STYLES),
    )
    return [
        user_text(instruction),
        user_audio(clip["audio"], clip["duration"]),
        assistant(answer),
    ]


def build_compound(stem, clip, transcript, targets, ilang, meta, probs, max_asks,
                   *, first_key=_MISSING, last_key=_MISSING, turn_dist=None, min_turns=1,
                   voice=True, trace=None):
    """A single compound turn and nothing else.

    The keyword arguments exist so every builder shares one call signature; a
    single turn has no length to draw and nothing to pin to the front.
    """
    include_transcript, chosen = pick_compound_selection(targets, max_asks)
    turns = compound_turns(clip, transcript, chosen, ilang, include_transcript,
                           meta["lang"])
    if turns is not None and trace is not None:
        trace.append(([None] if include_transcript else []) + [t for t, _ in chosen])
    return turns


def build_hybrid(stem, clip, transcript, targets, ilang, meta, probs, max_asks,
                 *, first_key=_MISSING, last_key=_MISSING, turn_dist=TURN_DIST,
                 min_turns=2, voice=True, trace=None):
    """Several user turns, exactly one of which bundles 2-3 instructions.

    The bundled turn is positioned over ALL turns, so "several instructions" is never
    learned as meaning "first turn".
    """
    src = meta["lang"]
    # A one-turn hybrid is just a compound.
    min_turns = max(2, min_turns)
    items = conversation_items(transcript, targets, first_key)
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
    if trace is not None:
        trace.extend([tgt for tgt, _ in group] for group in groups)
    for i, group in enumerate(groups):
        tr_text = next((text for tgt, text in group if tgt is None), None)
        translations = [(tgt, text) for tgt, text in group if tgt is not None]
        first = i == 0

        if len(group) == 1:
            tgt, text = group[0]
            if tgt is None:
                question = (bank_prompt("transcription", ilang, src) if first
                            else random.choice(FOLLOWUP_TRANSCRIBE[ilang]))
            else:
                question = (bank_prompt("translation", ilang, src, tgt) if first
                            else fmt(random.choice(FOLLOWUP_TRANSLATE[ilang]), tgt, ilang))
            answer = text
        elif first:
            block = compound_turns(clip, tr_text, translations, ilang,
                                   tr_text is not None, src)
            if block is None:
                return None
            turns.extend(block)
            continue
        else:
            langs = oxford_join([LANG_NAMES[ilang][t] for t, _ in translations], ilang)
            adjs = oxford_join([LANG_ADJ[ilang][t] for t, _ in translations], ilang)
            frame_pool = (COMPOUND_FOLLOWUP_WITH_TRANSCRIPT if tr_text
                          else COMPOUND_FOLLOWUP_FRAMES)[ilang]
            question = random.choice(frame_pool).format(langs=langs, adjs=adjs)
            answer = render_compound_answer(ilang, tr_text, translations,
                                            random.choice(ANSWER_STYLES))

        turns.append(user_text(question))
        if first:
            turns.append(user_audio(clip["audio"], clip["duration"]))
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
    if random.random() < p_hybrid:
        shape = "hybrid"
    else:
        shape = "compound" if random.random() < MIXED_COMPOUND_OF_REST else "sequential"
    builder = {"sequential": build_sequential, "compound": build_compound,
               "hybrid": build_hybrid}[shape]
    turns = builder(stem, clip, transcript, targets, ilang, meta, probs, max_asks, **kw)
    # `hybrid` legitimately declines on single-translation clips, i.e. everything but fr.
    if turns is None and shape == "hybrid":
        turns = build_sequential(stem, clip, transcript, targets, ilang, meta, probs,
                                 max_asks, **kw)
    return turns


def build_sequential(stem, clip, transcript, targets, ilang, meta, probs, max_asks=None,
                     *, first_key=_MISSING, last_key=_MISSING, turn_dist=TURN_DIST,
                     min_turns=2, voice=True, trace=None):
    """Audio in turn 1 only; every later user turn is text-only and elliptical.

    The turn count is drawn from TURN_DIST and then that many items are taken,
    so the length distribution is a parameter rather than a by-product of how
    many translations the clip happens to have.
    """
    src = meta["lang"]
    items = conversation_items(transcript, targets, first_key)
    n_turns = draw_turns(len(items), turn_dist)
    if n_turns < min_turns:
        return None
    if last_key is not _MISSING and last_key != first_key and n_turns > 1:
        place_key(items, n_turns - 1, last_key)

    turns = []
    if trace is not None:
        trace.extend([tgt] for tgt, _ in items[:n_turns])
    for i, (tgt, text) in enumerate(items[:n_turns]):
        first = i == 0
        if tgt is None:
            question = (bank_prompt("transcription", ilang, src) if first
                        else random.choice(FOLLOWUP_TRANSCRIBE[ilang]))
        else:
            question = (bank_prompt("translation", ilang, src, tgt) if first
                        else fmt(random.choice(FOLLOWUP_TRANSLATE[ilang]), tgt, ilang))
        turns.append(user_text(question))
        if first:
            turns.append(user_audio(clip["audio"], clip["duration"]))
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


def build_conversation(records, mode, probs, args):
    """Chain one segment per clip into a single conversation.

    A segment is a normal conversation on one clip; only its opening changes. The
    instruction language is drawn once for the whole conversation.

    Returns (turns, n_clips_used) or None. A later segment whose builder declines just
    ends the conversation early instead of discarding what is already built.
    """
    ilang = random.choice(INSTRUCTION_LANGS)
    builder = BUILDERS[mode]
    turns, used, prev_trace = [], 0, None

    # Decided up front: an elliptic switch constrains the segment BEFORE it, which has to
    # end on a task the next clip can answer.
    want_elliptic = [random.random() < args.elliptic_rate
                     for _ in range(max(0, len(records) - 1))]

    for index, rec in enumerate(records):
        continuation = index > 0
        trace = []
        kwargs = {"voice": False, "trace": trace}
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

        segment = builder(rec["stem"], rec["clip"], rec["transcript"], rec["targets"],
                          ilang, rec["meta"], probs, args.max_compound_asks, **kwargs)
        if segment is None:
            if index == 0:
                return None
            break

        if continuation:
            # An elliptic lead in front of a bundled answer would be an instruction never given.
            if carried is not _MISSING and len(trace[0]) == 1:
                apply_lead(segment, text=random.choice(CONTINUATION_ELLIPTIC[ilang]))
            else:
                apply_lead(segment, prefix=random.choice(CONTINUATION_EXPLICIT[ilang]))

        turns.extend(segment)
        prev_trace = trace
        used = index + 1

    # Appended here, not by the builders: they must come last and describe the LAST clip.
    # A lone compound turn is left alone, that shape being 20% of `mixed` by design.
    content_texts = sum(1 for t in turns if t["from"] == "User" and t["type"] == "text")
    n_audios = count_audios(turns)
    if not (content_texts == 1 and n_audios == 1):
        append_voice_followups(turns, ilang, records[used - 1]["meta"], probs,
                               qualify=n_audios > 1)
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

    for ilang in INSTRUCTION_LANGS:
        single_ask |= set(FOLLOWUP_TRANSCRIBE[ilang])
        single_ask |= set(CONTINUATION_ELLIPTIC[ilang])
        single_ask |= voice_questions
        for src in source_langs:
            single_ask |= {substitute_languages(t, ilang, src, None)
                           for t in BANK["transcription"][ilang]}
        for tgt in LANG_NAMES[ilang]:
            for tpl in FOLLOWUP_TRANSLATE[ilang]:
                single_ask.add(fmt(tpl, tgt, ilang))
            for src in source_langs:
                single_ask |= {substitute_languages(t, ilang, src, tgt)
                               for t in BANK["translation"][ilang]}
    prefixes = tuple(p for ilang in INSTRUCTION_LANGS for p in CONTINUATION_EXPLICIT[ilang])
    elliptic = {p for ilang in INSTRUCTION_LANGS for p in CONTINUATION_ELLIPTIC[ilang]}
    return single_ask, voice_questions, voice_kind, prefixes, elliptic


def generate(args):
    cv_root = Path(args.cv_root)
    ast_root = Path(args.ast_root)
    out_root = Path(args.output_dir)
    shares = parse_lang_shares(args.lang_share)
    modes = ALL_MODES if args.mode == "all" else [args.mode]
    single_ask, voice_questions, voice_kind, cont_prefixes, elliptic_leads = \
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

            for mode in modes:
                if counters[mode] >= args.max_samples:
                    continue
                built = build_conversation(group, mode, probs, args)
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

                content, leads = [], []
                for turn in turns:
                    if turn["from"] != "User" or turn["type"] != "text":
                        continue
                    value = turn["value"]
                    kind = voice_kind.get(value)
                    if kind:
                        voice_counts[mode][kind] += 1
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
    p.add_argument("--prompt-bank", default=None,
                   help="Override the repo instruction bank "
                        "(data/assets/instruction_transcription_and_translation_fr-en.txt)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    bank_path = Path(args.prompt_bank) if args.prompt_bank else \
        Path(__file__).resolve().parent.parent.joinpath(*PROMPT_BANK_RELPATH)
    BANK = load_prompt_bank(bank_path)
    print(f"prompt bank: {bank_path}", file=sys.stderr)
    generate(args)
