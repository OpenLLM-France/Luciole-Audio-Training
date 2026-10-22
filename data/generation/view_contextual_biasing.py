#!/usr/bin/env python3
"""View contextual-biasing manifests one example at a time.

For each line of the input jsonl, the user instruction is printed with its hotword list
coloured: hotwords actually spoken in the audio are shown in green, and a few of the
hotwords that are not spoken (preferably the ones most similar to the spoken ones) are
shown in red, followed by an ellipsis to signal the list is truncated. The transcription is
printed below, with the spoken hotwords highlighted in green. Press Enter for the next
example, or type q to quit.

Usage::

    python data/generation/view_contextual_biasing.py <manifest>_contextual_biasing.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

# Keep internal hyphens so compounds stay whole ("mother-in-law"); mirrors the generator.
WORD_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*", re.UNICODE)

GREEN = "\033[32m"
RED = "\033[31m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def lower_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def token_spans(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0).lower(), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def get_instruction(record: dict) -> str | None:
    for turn in record.get("conversations", []):
        if turn.get("from") == "User" and turn.get("type", "text") == "text":
            return turn.get("value")
    return None


def get_transcription(record: dict) -> str | None:
    for turn in reversed(record.get("conversations", [])):
        if turn.get("from") == "Assistant" and turn.get("type", "text") == "text":
            return turn.get("value")
    return None


def split_prompt(instruction: str, instruction_sep: str) -> tuple[str, list[str] | None]:
    """Split an augmented instruction into (prefix-up-to-and-including-label, [hotwords]).

    Returns (instruction, None) when no hotword block is found.
    """
    if instruction_sep not in instruction:
        return instruction, None
    base, last = instruction.rsplit(instruction_sep, 1)
    if ":" not in last:
        return instruction, None
    label, rest = last.split(":", 1)
    prefix = f"{base}{instruction_sep}{label}:"
    hotwords = [h.strip() for h in rest.split(",") if h.strip()]
    return prefix, hotwords or None


def find_present(hotwords: list[str], transcription: str) -> set[str]:
    """Hotwords whose lower-cased token sequence occurs in the transcription."""
    toks = lower_tokens(transcription)
    present: set[str] = set()
    for hw in hotwords:
        seq = lower_tokens(hw)
        n = len(seq)
        if n == 0:
            continue
        if any(toks[i : i + n] == seq for i in range(len(toks) - n + 1)):
            present.add(hw)
    return present


def highlight_spoken(text: str, spoken: list[str], color: str, reset: str) -> str:
    """Wrap each occurrence of a spoken hotword in ``text`` with colour codes."""
    if not color:
        return text
    spans: list[tuple[int, int]] = []
    toks = token_spans(text)
    seqs = [lower_tokens(hw) for hw in spoken]
    token_list = [t[0] for t in toks]
    for seq in seqs:
        n = len(seq)
        if n == 0:
            continue
        for i in range(len(token_list) - n + 1):
            if token_list[i : i + n] == seq:
                spans.append((toks[i][1], toks[i + n - 1][2]))
    if not spans:
        return text
    spans.sort()
    out: list[str] = []
    cursor = 0
    for start, end in spans:
        if start < cursor:
            continue  # overlapping match already coloured
        out.append(text[cursor:start])
        out.append(f"{color}{text[start:end]}{reset}")
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def distractor_score(hotword: str, present: list[str]) -> float:
    """Max string similarity of a distractor to any spoken hotword (to prefer confusable ones)."""
    return max(SequenceMatcher(None, hotword.lower(), p.lower()).ratio() for p in present)


def render(record: dict, index: int, args, palette) -> None:
    green, red, dim, bold, reset = palette
    instruction = get_instruction(record)
    transcription = get_transcription(record)
    header = f"Example {index}"
    if record.get("id"):
        header += f"   id={record['id']}"
    if record.get("dataset_name"):
        header += f"   dataset={record['dataset_name']}"
    print(f"\n{dim}{'=' * 70}{reset}")
    print(f"{bold}{header}{reset}\n")

    if instruction is None:
        print("(no user instruction)")
    else:
        prefix, hotwords = split_prompt(instruction, args.instruction_separator)
        if not hotwords:
            print(prefix)
        else:
            present_set = find_present(hotwords, transcription or "")
            present_list = [h for h in hotwords if h in present_set]
            # Choose which distractors to show (most similar to the spoken ones first), but
            # render everything in the real prompt order; hidden runs collapse to one ellipsis.
            absent_idx = [i for i, h in enumerate(hotwords) if h not in present_set]
            if present_list:
                absent_idx.sort(key=lambda i: distractor_score(hotwords[i], present_list), reverse=True)
            shown_distractor = set(absent_idx[: args.num_distractors])

            items: list[str | None] = []
            for i, hotword in enumerate(hotwords):
                if hotword in present_set:
                    items.append(f"{green}{hotword}{reset}")
                elif i in shown_distractor:
                    items.append(f"{red}{hotword}{reset}")
                elif not items or items[-1] is not None:
                    items.append(None)  # start a collapsed run of hidden hotwords
            rendered = ", ".join(f"{dim}…{reset}" if it is None else it for it in items)
            print(f"{prefix} {rendered} {dim}({len(hotwords)}){reset}")

    print()
    if transcription is None:
        print(f"{dim}(no transcription){reset}")
    else:
        spoken = sorted(find_present(hotwords, transcription), key=len, reverse=True) if hotwords else []
        print(f"{dim}Transcription:{reset}")
        print(highlight_spoken(transcription, spoken, green, reset))


def record_has_present(record: dict, instruction_sep: str) -> bool:
    """True when the example has at least one hotword spoken in the audio (a green hotword)."""
    instruction = get_instruction(record)
    transcription = get_transcription(record)
    if instruction is None or transcription is None:
        return False
    _, hotwords = split_prompt(instruction, instruction_sep)
    return bool(hotwords) and bool(find_present(hotwords, transcription))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretty-print a contextual-biasing manifest, one example at a time.")
    parser.add_argument("input", type=Path, help="Input jsonl manifest.")
    parser.add_argument("--num-distractors", type=int, default=6, help="How many absent hotwords to show in red.")
    parser.add_argument("--start", type=int, default=0, help="Skip the first N examples.")
    parser.add_argument(
        "--only-present-hotwords",
        action="store_true",
        help="Only show examples that have at least one hotword spoken in the audio (a green hotword).",
    )
    parser.add_argument(
        "--instruction-separator", default="\n\n", help="Separator between the instruction and the hotword prompt."
    )
    parser.add_argument("--color", choices=["auto", "always", "never"], default="auto", help="Coloured output.")
    args = parser.parse_args()

    use_color = args.color == "always" or (args.color == "auto" and sys.stdout.isatty())
    palette = (GREEN, RED, DIM, BOLD, RESET) if use_color else ("", "", "", "", "")

    if not args.input.exists():
        raise FileNotFoundError(f"Input manifest not found: {args.input}")

    shown = 0
    with args.input.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or i < args.start:
                continue
            record = json.loads(line)
            if args.only_present_hotwords and not record_has_present(record, args.instruction_separator):
                continue
            render(record, i + 1, args, palette)
            shown += 1
            try:
                reply = input(f"\n{palette[2]}[Enter = next, q = quit]{palette[4]} ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if reply.strip().lower() in ("q", "quit", "exit"):
                break
    if shown == 0:
        print("No examples to show.")


if __name__ == "__main__":
    main()
