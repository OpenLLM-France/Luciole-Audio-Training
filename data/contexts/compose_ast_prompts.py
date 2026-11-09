"""Compose translation (AST) prompts from the `ast` section of data/contexts/compositional/<lang>.json.

A prompt names the languages as its `mention` says, drawn per prompt (`MENTION`):
  target         Traduis cet audio en anglais
  source_target  Traduis cet audio de l'allemand en anglais
  implicit       Traduis cet audio (no language: the target is the language of the instruction,
                 so these prompts are always written in the target language)

The instruction language is drawn per prompt among the source and target languages, with a small
share for French and English, then for the other files (`instruction_weights`).

The `ast` section of an instruction file has:
  bases         {mention: {kind: [entries]}}, kinds: imperative, question, noun; an entry is a string
                (any register) or {register: text}, as in compose_asr_prompts.py
  source_names  optional, {language: text}: what {src} is replaced with when the name needs a
                preposition or an article ("de l'arabe", "du français"); default: language_names
Placeholders: {tgt} and {src_name} are the bare language_names, {src} is as above.

Usage:
    python data/contexts/compose_ast_prompts.py --source de --target en --instruction-language fr -n 10
    python data/contexts/compose_ast_prompts.py --source de --target en --instruction-language en --mention implicit
"""
import argparse
import json
import os
import random

from compose_asr_prompts import DEFAULT_DIR, MIN_PER_REGISTER, capitalize, in_register, weighted


MENTION = {"target": 0.55, "source_target": 0.30, "implicit": 0.15}
KINDS = ("imperative", "question", "noun")
SOURCE_OR_TARGET, FAVORED, OTHERS = 0.47, 0.04, 0.02   # the shares of the instruction languages
FAVORED_LANGUAGES = ("fr", "en")
# what the entries of each mention must and must not name
REQUIRED = {"target": ["{tgt}"], "source_target": ["{tgt}"], "implicit": []}
FORBIDDEN = {"target": ["{src}", "{src_name}"], "source_target": [], "implicit": ["{tgt}", "{src}", "{src_name}"]}


def instruction_weights(directory, source, target):
    """{language: weight}: source and target SOURCE_OR_TARGET each, FAVORED shared by French and English
    and OTHERS by the other files. A pair of French and English has no favored language left."""
    files = sorted(f[:-5] for f in os.listdir(directory) if f.endswith(".json"))
    favored = [l for l in FAVORED_LANGUAGES if l not in (source, target)]
    others = [l for l in files if l not in (source, target, *FAVORED_LANGUAGES)]
    return {source: SOURCE_OR_TARGET, target: SOURCE_OR_TARGET, **{l: FAVORED / len(favored) for l in favored},
            **{l: OTHERS / len(others) for l in others}}


class AstPromptComposer:
    """Prompts for the pair `source` -> `target`; `compose` gives one prompt."""

    def __init__(self, directory, source, target, instruction_languages=None, mention=None):
        self.source, self.target = source, target
        self.instruction_languages = instruction_languages or instruction_weights(directory, source, target)
        self.mention = mention or MENTION
        if set(self.mention) - set(MENTION):
            raise ValueError(f"mention: {sorted(self.mention)}, expected {list(MENTION)}")
        names = set(self.instruction_languages) | ({target} if self.mention.get("implicit") else set())
        self.files = {}
        for name in names:
            path = os.path.join(directory, f"{name}.json")
            with open(path, encoding="utf-8") as f:
                self.files[name] = json.load(f)
            self.validate(self.files[name], path)

    def validate(self, lang, where):
        section = lang.get("ast")
        if section is None:
            raise ValueError(f"{where}: no `ast` section")
        for language in (self.source, self.target):
            if language not in lang["language_names"]:
                raise ValueError(f"{where}: no name for {language!r} in language_names")
        registers = lang.get("registers") or [None]
        for mention in MENTION:
            entries = [e for kind in KINDS for e in section["bases"][mention].get(kind, [])]
            for entry in entries:
                if isinstance(entry, dict) and set(entry) - set(registers):
                    raise ValueError(f"{where}: ast.{mention}: registers {sorted(entry)}, expected {registers}")
                for text in ([entry] if isinstance(entry, str) else entry.values()):
                    if any(p not in text for p in REQUIRED[mention]) or any(p in text for p in FORBIDDEN[mention]):
                        raise ValueError(f"{where}: ast.{mention}: wrong placeholders in {text!r}")
                    if mention == "source_target" and "{src}" not in text and "{src_name}" not in text:
                        raise ValueError(f"{where}: ast.{mention}: no source language in {text!r}")
            for register in registers:
                n = sum(isinstance(e, dict) and register in e for e in entries)
                if not in_register(entries, register) or (any(isinstance(e, dict) for e in entries)
                                                          and n < MIN_PER_REGISTER):
                    raise ValueError(f"{where}: ast.{mention}: {n} entries in the register {register!r}, "
                                     f"expected {MIN_PER_REGISTER}")

    def compose(self, rng, instruction_language=None, mention=None):
        """One prompt; `instruction_language` and `mention` force them."""
        mention = mention or weighted(rng, self.mention)
        if mention == "implicit":          # the language of the instruction is what says the target
            if instruction_language not in (None, self.target):
                raise ValueError(f"an implicit prompt is written in the target language, not {instruction_language}")
            name = self.target
        else:
            name = instruction_language or weighted(rng, self.instruction_languages)
        lang = self.files[name]
        bases = lang["ast"]["bases"][mention]
        kind, base = rng.choice([(kind, entry) for kind in KINDS for entry in bases.get(kind, [])])
        register = next(iter(base)) if isinstance(base, dict) else None
        text = base if isinstance(base, str) else base[register]
        names = lang["language_names"]
        source = lang["ast"].get("source_names", names)[self.source]
        text = text.replace("{tgt}", names[self.target]).replace("{src_name}", names[self.source]) \
            .replace("{src}", source)
        return capitalize(text) + (lang["question_mark"] if kind == "question" else ".")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=DEFAULT_DIR, help="directory of the instruction files (default: %(default)s)")
    p.add_argument("--source", default="de", help="language of the audio (default: %(default)s)")
    p.add_argument("--target", default="en", help="language of the translation (default: %(default)s)")
    p.add_argument("--instruction-language", help="language of the instructions (default: drawn)")
    p.add_argument("--mention", choices=list(MENTION), help="force the language mention")
    p.add_argument("-n", type=int, default=10, help="prompts (default: %(default)s)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    instruction = {args.instruction_language: 1.0} if args.instruction_language else None
    mention = {args.mention: 1.0} if args.mention else None
    composer = AstPromptComposer(args.dir, args.source, args.target, instruction_languages=instruction, mention=mention)
    rng = random.Random(args.seed)
    for _ in range(args.n):
        print(composer.compose(rng, args.instruction_language, args.mention))


if __name__ == "__main__":
    main()
