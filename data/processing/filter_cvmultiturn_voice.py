"""Drop CVMultiTurn conversations that ask about the speaker's voice (accent, gender, age).

CVMultiTurn mixes transcription / translation chains with a few voice follow-ups
("Tu reconnais l'accent ?", "Is that a man's or a woman's voice?", ...), about 11.7 % of
the `mixed` train conversations. For an ASR/AST-only model those conversations are
dropped WHOLE rather than trimmed: a later recap turn ("Rassemble tout ça...") can
repeat the voice answer.

Voice questions are matched by exact membership against the templates of
data/templating/template_commonvoice_multiturn.py (with and without their
VOICE_QUALIFIER prefix), the same way that script reads its own output back.

Every split and its `_randomorder` twin is filtered line by line, so the shuffled file
keeps its order (no reshuffle needed). The source variant is left untouched; the
output goes to a sibling folder:

    CVMultiTurn/mixed/{train,dev,test}[_randomorder].jsonl
 -> CVMultiTurn/mixed_asr_ast/{train,dev,test}[_randomorder].jsonl

Usage:
    python data/processing/filter_cvmultiturn_voice.py
    python data/processing/filter_cvmultiturn_voice.py --variant compound
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "templating"))
from template_commonvoice_multiturn import (  # noqa: E402
    ACCENT_ANSWERS, AGE_ANSWERS, FOLLOWUP_ACCENT, FOLLOWUP_AGE, FOLLOWUP_GENDER,
    GENDER_ANSWERS, VOICE_QUALIFIER,
)

ROOT = "/data-server/datasets/audio/nemo/misc/multilang/context/CVMultiTurn"
SPLITS = ("train", "dev", "test")


def voice_questions():
    """Every voice question the generator can emit, keyed to its kind."""
    kinds = {}
    for kind, table in (("accent", FOLLOWUP_ACCENT), ("gender", FOLLOWUP_GENDER),
                        ("age", FOLLOWUP_AGE)):
        for ilang, templates in table.items():
            for t in templates:
                kinds[t] = kind
                for frame in VOICE_QUALIFIER[ilang]:
                    kinds[frame.format(q=t[0].lower() + t[1:])] = kind
    return kinds


def voice_answer_prefixes():
    """Fixed opening of every voice answer, used only as a post-filter sanity check."""
    out = set()
    for table in (ACCENT_ANSWERS, AGE_ANSWERS):
        for templates in table.values():
            out.update(t.split("{")[0] for t in templates)
    for by_gender in GENDER_ANSWERS.values():
        for templates in by_gender.values():
            out.update(templates)
    return tuple(out)


def filter_file(src, dst, questions, answers):
    kept = dropped = 0
    leaked = 0
    by_kind = {"accent": 0, "gender": 0, "age": 0}
    tmp = dst + ".tmp"
    with open(src) as fin, open(tmp, "w") as fout:
        for line in fin:
            turns = json.loads(line)["conversations"]
            hits = {questions[t["value"]] for t in turns
                    if t["from"] == "User" and t["type"] == "text" and t["value"] in questions}
            if hits:
                dropped += 1
                for k in hits:
                    by_kind[k] += 1
                continue
            leaked += any(t["from"] == "Assistant" and t["value"].startswith(answers) for t in turns)
            fout.write(line)
            kept += 1
    os.replace(tmp, dst)
    total = kept + dropped
    print(f"{os.path.basename(src)}: kept {kept}/{total}, dropped {dropped} "
          f"({100 * dropped / max(total, 1):.2f} %) {by_kind}, kept-with-voice-answer {leaked}")
    return kept, leaked


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=ROOT)
    p.add_argument("--variant", default="mixed", help="source subfolder (mixed, compound, sequential)")
    p.add_argument("--suffix", default="_asr_ast", help="output folder = <variant><suffix>")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    src_dir = os.path.join(args.root, args.variant)
    dst_dir = os.path.join(args.root, args.variant + args.suffix)
    assert os.path.abspath(src_dir) != os.path.abspath(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)

    questions, answers = voice_questions(), voice_answer_prefixes()
    counts = {}
    for split in SPLITS:
        for name in (f"{split}.jsonl", f"{split}_randomorder.jsonl"):
            src, dst = os.path.join(src_dir, name), os.path.join(dst_dir, name)
            if not os.path.exists(src):
                continue
            if os.path.exists(dst) and not args.overwrite:
                sys.exit(f"{dst} exists, pass --overwrite")
            counts[name], leaked = filter_file(src, dst, questions, answers)
            if leaked:
                print(f"  WARNING: {leaked} kept conversations still contain a voice answer")
        a, b = counts.get(f"{split}.jsonl"), counts.get(f"{split}_randomorder.jsonl")
        if a is not None and b is not None and a != b:
            sys.exit(f"{split}: {a} kept vs {b} in _randomorder, the two files differ")


if __name__ == "__main__":
    main()
