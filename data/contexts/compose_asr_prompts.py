"""Compose ASR prompts from compositional instruction files (data/contexts/compositional/<lang>.json).

A prompt is a transcription request (a base) that states each requested constraint:
  language      explicit (transcribe in French) or source (in the original language)
  nocasepunc    lowercase without punctuation (yes) or cased and punctuated (no)
  events        keep (yes) or drop (no) the [tags] of non-speech events
  disfluencies  keep (yes) or drop (no) the hesitations (euh, hmm)
An unset constraint is not mentioned: the base alone is a plain transcription request.

There is one file per instruction language, for audio of any language: the instruction language
is drawn per prompt (`instruction_languages`): the audio's but for `foreign_fraction` of the
prompts, shared equally between each of the `favored_languages` and the group of all the other
files (fr audio: en 1/2, others 1/2; it audio: en, fr, others 1/3 each). When it is the audio's, the language is mentioned
as `language_mention` says (mostly not); when it is another one, it is always mentioned, as the
audio language (explicit, from the file's `language_names`) or as the source language.

The constraints come in random order, in one of these formats (`formats`):
  fused      Transcris l'audio en minuscules et sans ponctuation, sans les hésitations comme « euh ».
  sentences  Transcris l'audio. Écris tout en minuscules, sans ponctuation. Sans les hésitations.
  lines      the same, one sentence per line
  bullets    Transcris l'audio :
             - en minuscules et sans ponctuation
             - sans les hésitations
  kv         Transcris l'audio.
             Casse : minuscules
             Ponctuation : aucune

An instruction file has:
  bases       the requests, by kind (imperative, question, noun: optional), with no constraint nor language
  axes        per constraint value, its three forms: clause (fits in a sentence: "en minuscules"),
              sentence (stands alone) and kv (lines "Clé : valeur")
  examples    per axis, what {ex} is replaced with (1-3 of them, joined with `or`); for the
              disfluencies, the `hesitations` of the audio language, written as `quotes` says ("« {} »")
  language_names  what {lang} is replaced with: the audio language, in the file's language
  registers   optional: an entry is then a string (any register) or {register: text} (tu / vous);
              the base is drawn among all the bases, its register (drawn when it has none) is the
              prompt's and only the entries of that register are used
  question_mark, colon, comma, and, or, bullets: the punctuation and words of the layouts
  attached    optional: the words of `and` / `or` written against the next word (Arabic و)
  serial_comma  optional, default true: false never puts a comma before the last `and` / `or`
              (German: no comma before "und" / "oder" in a list)

Usage:
    python data/contexts/compose_asr_prompts.py --nocasepunc yes --events yes -n 10
    python data/contexts/compose_asr_prompts.py --instruction-language en --all -n 3
"""
import argparse
import itertools
import json
import os
import random


AXES = {"language": ("explicit", "source"), "nocasepunc": ("yes", "no"), "events": ("yes", "no"),
        "disfluencies": ("yes", "no")}
FORMATS = {"fused": 0.3, "sentences": 0.25, "lines": 0.15, "bullets": 0.15, "kv": 0.15}
LANGUAGE_MENTION = {"none": 0.8, "explicit": 0.1, "source": 0.1}
FORMS = ("clause", "sentence", "kv")
MIN_PER_REGISTER = 2       # in a list with tu / vous entries, at least this many of each
# the base kinds each format can start with: a question cannot end in a colon
BASE_KINDS = {"fused": ("imperative", "question", "noun"), "sentences": ("imperative", "question", "noun"),
              "lines": ("imperative", "question", "noun"), "bullets": ("imperative", "noun"),
              "kv": ("imperative", "noun")}
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compositional")


def weighted(rng, weights):
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def capitalize(text):
    return text[:1].upper() + text[1:]


def in_register(entries, register):
    """The entries usable in `register`: the strings, and the {register: text} of that register."""
    return [e for e in entries if isinstance(e, str) or register in e]


def join_words(items, conj, comma=",", attached=False, serial=True):
    """'a', 'a et b', 'a, b et c'; 'a, b, et c' when c or b has its own conj or b a comma ('sans
    ponctuation et en minuscules, et en notant...', '..., et en minuscules et sans ponctuation',
    'en signalant les sons, par exemple [noise], et en laissant...'), unless `serial` is false.
    `attached`: 'a وb'."""
    if len(items) == 1:
        return items[0]
    word = f" {conj}" if attached else f" {conj} "
    nested = word in items[-2] or word in items[-1] or comma in items[-2]
    sep = f"{comma}{word}" if serial and nested else word
    return f"{f'{comma} '.join(items[:-1])}{sep}{items[-1]}"


def join_in(lang, items, word):
    """join_words with the `word` (and, or) and the punctuation of the instruction file `lang`."""
    return join_words(items, lang[word], lang.get("comma", ","), word in lang.get("attached", []),
                      lang.get("serial_comma", True))


class PromptComposer:
    """Prompts for audio in `audio_language`, written in the `instruction_languages`
    ({language: weight}, default: the audio's only); `compose` gives one prompt.
    hesitations: the hesitations of the audio language (["euh", "hmm"]), the examples of the
    disfluencies (clean_asr.py takes them from the `hesitations` of conf/rules.yaml)."""

    def __init__(self, directory, audio_language, hesitations, foreign_fraction=0.05, favored_languages=("fr", "en"),
                 instruction_languages=None, language_mention=None, formats=None):
        """instruction_languages: {language: weight}, instead of the weights of foreign_fraction and
        favored_languages."""
        self.audio_language = audio_language
        self.hesitations = list(hesitations)
        if not self.hesitations:
            raise ValueError(f"no hesitations given for {audio_language!r}: the disfluency prompts need examples")
        self.instruction_languages = instruction_languages or self.weights(directory, audio_language, foreign_fraction,
                                                                           favored_languages)
        self.language_mention = language_mention or LANGUAGE_MENTION
        self.formats = formats or FORMATS
        if set(self.formats) - set(FORMATS):
            raise ValueError(f"unknown formats {sorted(set(self.formats) - set(FORMATS))}, expected {list(FORMATS)}")
        if set(self.language_mention) - {"none", *AXES["language"]}:
            raise ValueError(f"language_mention: {sorted(self.language_mention)}, expected none, explicit, source")
        self.files = {}
        for name in self.instruction_languages:
            path = os.path.join(directory, f"{name}.json")
            with open(path, encoding="utf-8") as f:
                self.files[name] = json.load(f)
            self.validate(self.files[name], path)

    @staticmethod
    def weights(directory, audio_language, foreign_fraction, favored_languages):
        """{language: weight} of the instruction languages: the files of `directory`."""
        files = sorted(f[:-5] for f in os.listdir(directory) if f.endswith(".json"))
        if audio_language not in files:
            raise ValueError(f"no instruction file {audio_language}.json in {directory}")
        favored = [l for l in favored_languages if l in files and l != audio_language]
        others = [l for l in files if l != audio_language and l not in favored]
        share = foreign_fraction / (len(favored) + bool(others))
        return {audio_language: 1 - foreign_fraction, **{l: share for l in favored},
                **{l: share / len(others) for l in others}}

    def validate(self, lang, where):
        registers = lang.get("registers") or [None]

        def check(entries, at):
            """Every entry in a known register, and some entry for each register: in a list that
            addresses the listener, MIN_PER_REGISTER entries of each register."""
            for entry in entries:
                if isinstance(entry, dict) and set(entry) - set(registers):
                    raise ValueError(f"{where}: {at}: registers {sorted(entry)}, expected {registers}")
            addressed = any(isinstance(entry, dict) for entry in entries)
            for register in registers:
                n = sum(isinstance(entry, dict) and register in entry for entry in entries)
                if not in_register(entries, register) or (addressed and n < MIN_PER_REGISTER):
                    raise ValueError(f"{where}: {at}: {n} entries in the register {register!r}, "
                                     f"expected {MIN_PER_REGISTER}")

        if self.audio_language not in lang["language_names"]:
            raise ValueError(f"{where}: no name for the audio language {self.audio_language!r} in language_names")
        for fmt, kinds in BASE_KINDS.items():
            check([entry for kind in kinds for entry in lang["bases"].get(kind, [])], f"bases of {fmt}")
        for axis, values in AXES.items():
            for value in values:
                part = lang["axes"].get(axis, {}).get(value)
                if part is None or any(not part.get(form) for form in FORMS):
                    raise ValueError(f"{where}: axes.{axis}.{value} needs non-empty {FORMS}")
                for form in ("clause", "sentence"):
                    check(part[form], f"axes.{axis}.{value}.{form}")
                if "{ex}" in json.dumps(part, ensure_ascii=False) and axis != "disfluencies" \
                        and not lang["examples"].get(axis):
                    raise ValueError(f"{where}: axes.{axis} uses {{ex}} but has no examples")
        if "{}" not in lang.get("quotes", ""):
            raise ValueError(f"{where}: quotes must be a template with {{}}, like \"« {{}} »\"")

    def compose(self, rng, nocasepunc=None, events=None, disfluencies=None, fmt=None, instruction_language=None,
                mention=None):
        """A prompt asking for the given constraints: nocasepunc True (lowercase, no punctuation),
        False (case and punctuation) or None (not mentioned); events / disfluencies True (keep),
        False (drop) or None. `fmt`, `instruction_language` and `mention`
        (none, explicit, source) force a format, an instruction language or the language mention."""
        name = instruction_language or weighted(rng, self.instruction_languages)
        lang = self.files[name]
        if name != self.audio_language:
            if mention == "none":
                raise ValueError(f"instructions in {name} for {self.audio_language} audio must mention the language")
            mention = mention or rng.choice(AXES["language"])
        mention = mention or weighted(rng, self.language_mention)
        wanted = [(axis, "yes" if value else "no")
                  for axis, value in (("nocasepunc", nocasepunc), ("events", events),
                                      ("disfluencies", disfluencies)) if value is not None]
        if mention != "none":
            wanted.append(("language", mention))
        rng.shuffle(wanted)
        fmt = fmt or weighted(rng, self.formats)
        # the base, among all the bases of the kinds the format allows; the prompt takes its register
        kind, base = rng.choice([(kind, entry) for kind in BASE_KINDS[fmt] for entry in lang["bases"].get(kind, [])])
        register = next(iter(base)) if isinstance(base, dict) else rng.choice(lang.get("registers") or [None])

        def text(entry, axis=None):
            entry = entry if isinstance(entry, str) else entry[register]
            if "{ex}" in entry:
                pool = [lang["quotes"].format(h) for h in self.hesitations] if axis == "disfluencies" \
                    else lang["examples"][axis]
                entry = entry.replace("{ex}", join_in(lang, rng.sample(pool, rng.randint(1, min(3, len(pool)))),
                                                      "or"))
            return entry.replace("{lang}", lang["language_names"][self.audio_language])

        def parts(form):
            return [text(rng.choice(in_register(lang["axes"][axis][value][form], register)), axis)
                    for axis, value in wanted]

        base = text(base)
        end = lang["question_mark"] if kind == "question" else "."
        if not wanted:
            return base + end
        if fmt == "fused":
            # a comma after a long base, or one with its own clause ("Écris ce que tu entends, en
            # minuscules"): without it, the constraint reads as part of the base ("sans rien omettre en minuscules")
            long_base = len(base.split()) > 5 or lang.get("comma", ",") in base
            comma = lang.get("comma", ",") + " " if long_base or rng.random() < 0.3 else " "
            return f"{base}{comma}{join_in(lang, parts('clause'), 'and')}{end}"
        if fmt == "sentences":
            return " ".join([base + end] + parts("sentence"))
        if fmt == "lines":
            return "\n".join([base + end] + parts("sentence"))
        head = base + (lang["colon"] if rng.random() < 0.5 else end)
        if fmt == "bullets":
            bullet, cap = rng.choice(lang["bullets"]), rng.random() < 0.5
            return "\n".join([head] + [bullet.format(n=i + 1) + (capitalize(p) if cap else p)
                                       for i, p in enumerate(parts("clause"))])
        lines = [text(line, axis) for axis, value in wanted
                 for line in rng.choice(lang["axes"][axis][value]["kv"])]
        return "\n".join([head] + lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=DEFAULT_DIR, help="directory of the instruction files (default: %(default)s)")
    p.add_argument("--audio-language", default="fr", help="language of the audio (default: %(default)s)")
    p.add_argument("--hesitations", nargs="+", default=["euh", "hmm"],
                   help="hesitations of the audio language, the examples of the disfluencies (default: %(default)s)")
    p.add_argument("--instruction-language", default=None,
                   help="language of the instructions (default: the audio's); another one always mentions the language")
    p.add_argument("--mention", choices=["none", "explicit", "source"], help="force the language mention")
    p.add_argument("--nocasepunc", choices=["yes", "no"],
                   help="ask for lowercase without punctuation (yes) or with case and punctuation (no)")
    p.add_argument("--events", choices=["yes", "no"], help="ask to keep (yes) or drop (no) the [tags]")
    p.add_argument("--disfluencies", choices=["yes", "no"], help="ask to keep (yes) or drop (no) the hesitations")
    p.add_argument("--all", action="store_true", help="every combination of the constraints")
    p.add_argument("--format", choices=list(FORMATS), help="only this format")
    p.add_argument("-n", type=int, default=5, help="prompts per combination")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    instruction = args.instruction_language or args.audio_language
    composer = PromptComposer(args.dir, args.audio_language, args.hesitations, instruction_languages={instruction: 1.0})
    rng = random.Random(args.seed)
    yes_no = {"yes": True, "no": False, None: None}
    if args.all:
        combos = itertools.product((None, True, False), repeat=3)
    else:
        combos = [(yes_no[args.nocasepunc], yes_no[args.events], yes_no[args.disfluencies])]
    for nocasepunc, events, disfluencies in combos:
        print(f"### nocasepunc={nocasepunc} events={events} disfluencies={disfluencies}")
        for _ in range(args.n):
            print(composer.compose(rng, nocasepunc, events, disfluencies, args.format, mention=args.mention))
            print("--")


if __name__ == "__main__":
    main()
