"""Lowercase spurious mid-sentence capitals left by the recase+punctuation model.

The `*_recasepunc` French manifests (ESLO, TCOF, CFPP2000, LesVocaux, ...)
contain capitalised function words in the middle of a sentence, with no
sentence boundary before them:

    "Bien sûr Et."          -> "Bien sûr et."
    "Ah Oui, Oui!"          -> "Ah oui, oui!"
    "y a Je sais pas"       -> "y a je sais pas"

Rule: a word from FUNCTION_WORDS that is Capitalised (not ALL-CAPS) is
lowercased when the previous non-space character is not a sentence boundary
(. ? ! … : « " etc.). It is kept as is when the next word is a capitalised
word outside the list, which protects proper nouns and titles ("de La Poste",
"Le Bon Coin", "Les Mille et Une Nuits"), and when it is hyphenated
("Notre-Dame", "Il-76").

Only Assistant text turns are touched; unchanged lines are copied verbatim.
Outputs are written next to the inputs, with `_fixcaps` inserted after
`_recasepunc` (or before `.jsonl` when absent):

    train_recasepunc_max30.jsonl -> train_recasepunc_fixcaps_max30.jsonl

Usage:
    python fix_spurious_caps.py --dry-run /data-server/.../fr/context/ESLO/*.jsonl
    python fix_spurious_caps.py /data-server/.../fr/{context,nocontext}/{ESLO,TCOF_*}/*.jsonl
"""
import argparse
import json
import os
import re
from collections import Counter

from tqdm import tqdm

from rules import words


# the words of `fix_spurious_caps` in conf/rules.yaml, capitalised
FUNCTION_WORDS = sorted(w.capitalize() for w in words("fr", "fix_spurious_caps", "function_words"))
ELIDED = sorted(w.capitalize() for w in words("fr", "fix_spurious_caps", "elided"))  # followed by ' or ’

FUNCTION_SET = set(FUNCTION_WORDS)
SENTENCE_BOUNDARY = set(".?!…:«\"“([—–-")

_word_alt = "|".join(sorted(FUNCTION_WORDS, key=len, reverse=True))
_elided_alt = "|".join(sorted(ELIDED, key=len, reverse=True))
# group 1: the candidate word (elided form keeps its apostrophe)
CANDIDATE_RE = re.compile(
    rf"(?<=\s)((?:{_elided_alt})['’](?=\w)|(?:{_word_alt})(?![\w'’-]))"
)
NEXT_WORD_RE = re.compile(r"\s+([^\s]+)")


def _prev_char(text, start):
    """Previous non-space character, skipping event tags: "[laughter] Ah" starts the text."""
    i = start - 1
    while i >= 0:
        if text[i].isspace():
            i -= 1
        elif text[i] == "]" and text.rfind("[", 0, i) >= 0:
            i = text.rfind("[", 0, i) - 1
        else:
            break
    return text[i] if i >= 0 else None


def _next_word_protects(text, end):
    """True when the following words look like a proper noun/title: the next
    word is capitalised and either outside FUNCTION_WORDS ("La Poste") or
    itself followed by a capitalised word ("Le Bon Coin")."""
    words = []
    pos = end
    while len(words) < 2:
        m = NEXT_WORD_RE.match(text, pos)
        if not m:
            break
        words.append(m.group(1))
        pos = m.end()
        if m.group(1)[-1] in ".,;:!?…»\"”)":
            break
    if not words:
        return False
    first = words[0].strip(".,;:!?…»\"”)")
    if not first or not first[0].isupper():
        return False
    if first not in FUNCTION_SET:
        return True
    return len(words) == 2 and words[1][:1].isupper()


def _spelled_letter(text, start, end):
    """"Y" is a spelled letter, not the pronoun y, when no lowercase word follows it ("Avec un Y.",
    "H E N R Y") or a capital letter alone comes before it ("A M A R Y a")."""
    nxt = NEXT_WORD_RE.match(text, end)
    prev = text[:start].split()
    return not (nxt and nxt.group(1)[:1].islower()) or bool(prev) and re.fullmatch(r"[A-ZÀ-Þ]\.?", prev[-1])


def fix_text(text, stats=None):
    def repl(m):
        word = m.group(1)
        prev = _prev_char(text, m.start())
        if prev is None or prev in SENTENCE_BOUNDARY:
            return word
        if word == "Y" and _spelled_letter(text, m.start(), m.end()):
            return word
        if not word.endswith(("'", "’")) and _next_word_protects(text, m.end()):
            if stats is not None:
                stats["kept_proper_noun"] += 1
            return word
        if stats is not None:
            stats[word] += 1
        return word[0].lower() + word[1:]

    return CANDIDATE_RE.sub(repl, text)


def fix_record(rec, stats):
    changed = False
    turns = rec.get("conversations")
    if turns is not None:
        for turn in turns:
            if turn.get("from") == "Assistant" and turn.get("type") == "text" and turn.get("value"):
                new = fix_text(turn["value"], stats)
                if new != turn["value"]:
                    turn["value"] = new
                    changed = True
    elif rec.get("text"):
        new = fix_text(rec["text"], stats)
        if new != rec["text"]:
            rec["text"] = new
            changed = True
    return changed


def output_path(path):
    d, name = os.path.split(path)
    if "_recasepunc" in name:
        name = name.replace("_recasepunc", "_recasepunc_fixcaps", 1)
    else:
        name = name[: -len(".jsonl")] + "_fixcaps.jsonl"
    return os.path.join(d, name)


def process(path, dry_run, n_examples, overwrite):
    out = output_path(path)
    if not dry_run and os.path.exists(out) and not overwrite:
        print(f"skip {path}: {out} exists (use --overwrite)")
        return
    stats = Counter()
    n = n_changed = 0
    examples = []
    fout = None if dry_run else open(out + ".tmp", "w", encoding="utf-8")
    with open(path, encoding="utf-8") as fin:
        for line in tqdm(fin, desc=os.path.basename(path), unit=" lines", leave=False):
            n += 1
            rec = json.loads(line)
            before = json.dumps(rec, ensure_ascii=False) if len(examples) < n_examples else None
            if fix_record(rec, stats):
                n_changed += 1
                if fout is not None:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if before is not None:
                    examples.append((json.loads(before), rec))
            elif fout is not None:
                fout.write(line if line.endswith("\n") else line + "\n")
    if fout is not None:
        fout.close()
        os.replace(out + ".tmp", out)

    kept = stats.pop("kept_proper_noun", 0)
    print(f"{path}\n  -> {'(dry-run)' if dry_run else out}")
    print(f"  lines changed: {n_changed}/{n} ({n_changed / max(n, 1):.2%}), "
          f"words lowercased: {sum(stats.values())}, kept before capitalised word: {kept}")
    print(f"  top: {', '.join(f'{w}={c}' for w, c in stats.most_common(12))}")
    for before, after in examples:
        b = before["conversations"][-1]["value"] if "conversations" in before else before["text"]
        a = after["conversations"][-1]["value"] if "conversations" in after else after["text"]
        print(f"    - {b[:160]!r}\n      {a[:160]!r}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="jsonl manifests to fix")
    p.add_argument("--dry-run", action="store_true", help="print stats and examples, write nothing")
    p.add_argument("--examples", type=int, default=5, help="number of changed lines to print per file")
    p.add_argument("--overwrite", action="store_true", help="overwrite existing *_fixcaps outputs")
    args = p.parse_args()
    for path in args.inputs:
        if "_fixcaps" in os.path.basename(path):
            continue
        process(path, args.dry_run, args.examples, args.overwrite)


if __name__ == "__main__":
    main()
