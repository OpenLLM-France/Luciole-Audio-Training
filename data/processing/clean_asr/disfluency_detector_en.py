#!/usr/bin/env python3
"""Remove disfluencies from English text with a trained disfluency detector.

English counterpart of ``src/detect_disfluencies.py`` of linagora/flaubert_disfluency_detector:
the same API (``load_model``, ``annotate``, ``strip_fillers``, ``strip_disfluency_tags``,
``_FILLER_WORD_RE``, ``_PROTECT_WORDS``, ``remove_disfluencies`` and the command line), so that
the ``disfluency_model`` filter of clean_asr.py uses either (``lang: en`` loads this module),
but wrapping the English/Japanese model ``oldriku/Clausa``. The model repository has no code:
this file is it.

Clausa is a standard ``XLMRobertaForTokenClassification`` model: each word of a transcript is
predicted fluent (0) or disfluent (1), where "disfluent" covers filled pauses, repetitions,
self-corrections, false starts, partial words and English discourse markers ("well",
"you know", "I mean"). The model expects lower-cased, punctuation-free words, so the input is
normalized for the model while the *original* words (their casing and punctuation) are kept
for the output: only the words flagged disfluent are deleted.

The weights are downloaded from the Hugging Face Hub (``oldriku/Clausa`` by default); pass a
local folder to ``--model_id`` to use a local copy instead. Note that Clausa is released under
CC-BY-NC-SA-4.0 (non-commercial), unlike the Apache-2.0 French model.

Limitations. Clausa was trained on conversational turns and synthetic data; on very short,
isolated sentences it can over-flag a sentence-final word (reading it as a false start). It is
most reliable on running transcripts of spontaneous speech, which is its intended input.

Examples
--------
    python3 remove_disfluencies_en.py "i think i think the the boy is um running to the park"
    # I think the boy is running to the park

    python3 remove_disfluencies_en.py transcript.txt -o transcript.clean.txt
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Quiet down optional backends and advisory logs before transformers is imported.
os.environ.setdefault("USE_TF", "0")                # do not pull in TensorFlow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # ... and silence it if something else does
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import string
import warnings
from collections import namedtuple

import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

HF_MODEL_ID = "oldriku/Clausa"

# In Clausa's label scheme, class index 1 is the disfluent class.
_DISFLUENT_CLASS = 1

# Clausa labels long transcripts in windows; a word window well under its 512 sub-token limit.
MAX_WORDS = 128

# Default thresholds for the asymmetric mode (threshold < 0): lenient for repetitions and
# hesitations, strict for isolated single-word flags. Clausa returns plain probabilities, so
# these are probabilities (unlike the log-space thresholds of the French script).
ASYM_LOW, ASYM_HIGH = 0.4, 0.6

# Hesitation / filled-pause sounds (uh, um, er, erm, ah, hmm, mm, mh), used to justify a
# removal in the post-filter and the asymmetric mode.
# Not "ah" (an interjection, a word in conf/rules.yaml) nor "ER" (emergency room).
_HESITATION_RE = re.compile(r"^(?:[Uu]+[Hh]+|[Uu]+[Mm]+|[Ee]+r+m*|[Hh]+[Mm]+|[Mm]+[Hh]+|[Mm]+[Mm]+)$")
# What the "strip fillers" option removes directly from raw text: the filled-pause sounds
# above as standalone words, plus truncation fragments (letters followed by "-" before a space).
_FILLER_WORD_RE = re.compile(
    r"(?<![\w'’-])(?:[Uu]+[Hh]+|[Uu]+[Mm]+|[Ee]+r+m*|[Hh]+[Mm]+|[Mm]+[Hh]+|[Mm]+[Mm]+)(?![\w'’-])")
_TRUNC_RE = re.compile(r"(?i)(?<![\w'’-])([a-z]+)-(?=\s|$)")

# Number words: a trailing hyphen here is a compound-number artifact, not a truncation.
_NUMBER_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety", "hundred", "thousand", "million", "billion", "zero",
}
# Words never removed in isolation by the protect-list option (unless they are a repetition).
_PROTECT_WORDS = {"and", "but", "so", "or", "then", "because"} | _NUMBER_WORDS
# Never removed, even repeated: digits read one by one ("eight zero zero one", ICSI) repeat.
_ALWAYS_PROTECT = set(_NUMBER_WORDS)
# Word sequences never removed by the protect-list option unless repeated: discourse markers
# Switchboard labels disfluent, words of the speech all the same.
_PROTECT_PHRASES = {("you", "know"), ("i", "mean"), ("sort", "of"), ("kind", "of")}

# Punctuation stripped from the two ends of a word before it is shown to the model. Internal
# and edge apostrophes and hyphens are kept, so contractions ("don't") stay intact and
# truncations ("wor-") keep their trailing dash.
_EDGE_PUNCT = string.punctuation.replace("'", "").replace("-", "") + "“”«»…·—–"

# Everything needed to render one cleaned line. char_label / covered / ok: per character of
# the text, as the French repository gives them (clean_asr.py reads these): the label of the
# token the character is in, every character covered, always aligned (tokens are split on spaces).
Annotation = namedtuple("Annotation", "text tokens labels char_label covered ok")


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_model(model_id=HF_MODEL_ID, device="cpu"):
    """Load the English disfluency detector (Clausa).

    :param model_id: a Hugging Face repository id, or a path to a local model folder.
    :param device: ``"cpu"`` or e.g. ``"cuda:0"``.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    net = AutoModelForTokenClassification.from_pretrained(model_id)
    net.to(device)
    net.eval()
    return {"tokenizer": tokenizer, "net": net, "max_words": MAX_WORDS}


# --------------------------------------------------------------------------- #
# Pre-processing: raw token -> model word
# --------------------------------------------------------------------------- #

def _clean_word(token):
    """Lower-case a whitespace token and strip the punctuation on its two ends.

    Returns ``""`` for a token made only of punctuation (which is kept verbatim in the output
    and never sent to the model). Internal/edge apostrophes and hyphens are preserved so that
    contractions ("don't") and truncations ("wor-") stay recognizable.
    """
    return token.lower().strip(_EDGE_PUNCT)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #

def annotate(model, lines, threshold=0.5, batch_size=32, gpu=None, progress=True,
             word_level=True, asymmetric=None, postfilter=False, protect=True):
    """Run Clausa on ``lines`` and return one :class:`Annotation` per line.

    Each line is split on whitespace into raw tokens; the lower-cased, de-punctuated words are
    shown to the model, and the resulting per-word disfluency decision is mapped back onto the
    raw tokens so the original casing and punctuation can be preserved on output.

    :param word_level: when ``True`` (default), a word's disfluency score is the mean over its
        sub-word pieces (Clausa's majority-style decision); set to ``False`` to score a word by
        the maximum over its pieces (a word is disfluent as soon as any piece is).
    :param asymmetric: ``None`` (default) for a single ``threshold``; otherwise a
        ``(low, high)`` pair giving a lenient threshold to repetitions/hesitations and a strict
        one to isolated words.
    :param postfilter: when ``True``, keep a removal only if it is a hesitation, a truncation,
        or an exact repeat of the adjacent text (off by default).
    :param protect: when ``True`` (default), never remove a protect-list word (connector or
        number) unless it is a repetition.
    """
    tokenizer = model["tokenizer"]
    net = model["net"]
    max_words = model["max_words"]
    device = next(net.parameters()).device

    # Split every line and build the model windows (word-aligned, so no word is cut).
    per_line = []                       # [raw_tokens, word_token_index, words, scores]
    windows = []                        # [(line_idx, word_start, words_chunk)]
    for line in lines:
        raw_tokens = line.split()
        word_token_index, words = [], []
        for i, token in enumerate(raw_tokens):
            w = _clean_word(token)
            if w:
                word_token_index.append(i)
                words.append(w)
        per_line.append([raw_tokens, word_token_index, words, [0.0] * len(words)])
        line_idx = len(per_line) - 1
        for start in range(0, len(words), max_words):
            windows.append((line_idx, start, words[start:start + max_words]))

    # Run the model, batch by batch, with dynamic padding.
    batches = range(0, len(windows), batch_size)
    if progress and windows:
        batches = tqdm(batches, total=(len(windows) + batch_size - 1) // batch_size)
    for batch_start in batches:
        batch = windows[batch_start:batch_start + batch_size]
        batch_words = [w[2] for w in batch]
        enc = tokenizer(
            batch_words, is_split_into_words=True, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        )
        word_ids = [enc.word_ids(b) for b in range(len(batch_words))]
        enc = enc.to(device)
        with torch.no_grad():
            logits = net(**enc).logits                           # (B, L, num_labels)
            probs = torch.softmax(logits, dim=-1)[:, :, _DISFLUENT_CLASS].cpu()
        for b, (line_idx, word_start, _) in enumerate(batch):
            # Gather each word's sub-word disfluency probabilities (skip special tokens).
            buckets = {}
            for position, wid in enumerate(word_ids[b]):
                if wid is None:
                    continue
                buckets.setdefault(wid, []).append(float(probs[b, position]))
            scores = per_line[line_idx][3]
            for wid, plist in buckets.items():
                scores[word_start + wid] = (sum(plist) / len(plist)) if word_level else max(plist)

    # Decide labels, then map them from model words back onto the raw tokens.
    annotations = []
    for line, (raw_tokens, word_token_index, words, scores) in zip(lines, per_line):
        word_labels = _decide_word_labels(
            words, scores, threshold, asymmetric, postfilter, protect)
        token_labels = [0] * len(raw_tokens)
        for k, i in enumerate(word_token_index):
            token_labels[i] = word_labels[k]
        char_label, pos = [0] * len(line), 0
        for token, label in zip(raw_tokens, token_labels):
            pos = line.index(token, pos)
            char_label[pos:pos + len(token)] = [label] * len(token)
            pos += len(token)
        annotations.append(Annotation(line, raw_tokens, token_labels, char_label, [True] * len(line), True))
    return annotations


# --------------------------------------------------------------------------- #
# Decision heuristics (word level)
# --------------------------------------------------------------------------- #

def _decide_word_labels(words, scores, threshold, asymmetric, postfilter, protect, require_repair=True):
    """Turn per-word disfluency scores into binary labels, applying the enabled options."""
    n = len(words)
    if asymmetric:
        low, high = asymmetric
        labels = []
        for i, w in enumerate(words):
            lenient = bool(_HESITATION_RE.match(w)) or (
                (i > 0 and words[i - 1] == w) or (i + 1 < n and words[i + 1] == w))
            labels.append(1 if scores[i] > (low if lenient else high) else 0)
    else:
        labels = [1 if scores[i] > threshold else 0 for i in range(n)]
    if postfilter:
        labels = _postfilter_labels(words, labels)
    if protect:
        labels = _protect_labels(words, labels)
    if require_repair:
        labels = _repair_labels(words, labels)
    return labels


def _repair_labels(words, labels):
    """Un-remove the flagged words that end the line, unless hesitations or cut-off words: a false
    start is one when the speaker starts again after it. On a segment cut in the middle of a
    sentence ("But that's not really", "So in"), the model reads the words left unfinished as a
    false start, but they are spoken all the same; a line is never emptied either."""
    i = len(words)
    while i > 0 and labels[i - 1]:
        i -= 1
    for k in range(i, len(words)):
        if not (_HESITATION_RE.match(words[k]) or words[k].endswith("-")):
            labels[k] = 0
    return labels


def _postfilter_labels(words, labels):
    """Keep a removed word only if it is a hesitation, a truncation, or an exact repeat of the
    adjacent (kept) text; otherwise un-remove it. Operates on maximal removed runs."""
    n = len(words)
    i = 0
    while i < n:
        if not labels[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and labels[j + 1]:
            j += 1
        run = words[i:j + 1]
        length = len(run)
        justified = any(_HESITATION_RE.match(w) or w.endswith("-") for w in run)
        if not justified:
            following = words[j + 1:j + 1 + length]
            preceding = words[i - length:i] if i - length >= 0 else []
            justified = run in (following, preceding)
        if not justified:
            for k in range(i, j + 1):
                labels[k] = 0
        i = j + 1
    return labels


def _protect_labels(words, labels):
    """Un-remove a protect-list word (connector, number, discourse word) or phrase unless it is
    a repetition; a number word even then."""
    n = len(words)
    for i, w in enumerate(words):
        if not labels[i]:
            continue
        if w in _ALWAYS_PROTECT:
            labels[i] = 0
        elif w in _PROTECT_WORDS:
            repeat = ((i > 0 and words[i - 1] == w) or (i + 1 < n and words[i + 1] == w))
            if not repeat:
                labels[i] = 0
    for phrase in _PROTECT_PHRASES:
        k = len(phrase)
        for i in range(n - k + 1):
            if tuple(words[i:i + k]) == phrase and all(labels[i:i + k]) \
                    and tuple(words[i + k:i + 2 * k]) != phrase and tuple(words[max(0, i - k):i]) != phrase:
                labels[i:i + k] = [0] * k
    return labels


# --------------------------------------------------------------------------- #
# Filler stripping (raw text)
# --------------------------------------------------------------------------- #

def strip_fillers(text):
    """Remove filled-pause sounds (uh/um/er...) and truncation fragments from raw ``text``,
    tidying capitalisation and spacing the same way disfluency removal does."""
    def wrap_filler(match):
        return "<dsf>" + match.group(0) + "</dsf>"

    def wrap_truncation(match):
        if match.group(1).lower() in _NUMBER_WORDS:
            return match.group(0)
        return "<dsf>" + match.group(0) + "</dsf>"

    marked = _FILLER_WORD_RE.sub(wrap_filler, text)
    marked = _TRUNC_RE.sub(wrap_truncation, marked)
    return strip_disfluency_tags(marked)


# --------------------------------------------------------------------------- #
# Rendering: drop the flagged words and tidy the result
# --------------------------------------------------------------------------- #

# Characters that end a sentence (so the next word should start with a capital).
_SENT_END = ".!?…"
# Punctuation that becomes an orphan (to be dropped) when a sentence-initial disfluency is
# removed and leaves its trailing mark with nothing before it.
_ORPHAN_PUNCT = ".,;:!?…"
# Punctuation that attaches to the word on its left.
_ATTACH_LEFT = set(".,;:!?)]}…\"'")

_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?)\]}…])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[{])\s+")   # not '"': a closing quote too ('"key" file')


def _first_letter(text):
    """First alphabetic character of ``text`` (``""`` if there is none)."""
    for ch in text:
        if ch.isalpha():
            return ch
    return ""


def _build_tagged(raw_tokens, token_labels):
    """Join the raw tokens, wrapping each flagged token in ``<dsf>...</dsf>``."""
    out = []
    for token, label in zip(raw_tokens, token_labels):
        out.append("<dsf>" + token + "</dsf>" if label else token)
    return " ".join(out)


def strip_disfluency_tags(tagged, auto_capitalize=True):
    """Remove every ``<dsf>...</dsf>`` span from ``tagged`` and tidy the spacing.

    :param auto_capitalize: when ``True`` (default), each sentence whose opening disfluency
        (removed here) started with an upper-case letter has its new first word capitalized,
        so a sentence that began with a capitalized disfluency still starts with a capital.
    """
    parts = re.split(r"(<dsf>.*?</dsf>)", tagged, flags=re.DOTALL)
    out = []
    at_sentence_start = True   # no kept word has been emitted yet in the current sentence
    pending_capital = False    # the next kept letter should be capitalized
    drop_orphan = False        # a sentence-initial span was removed; drop its orphaned punctuation
    for index, part in enumerate(parts):
        if index % 2 == 1:
            # A removed disfluency span, "<dsf>...</dsf>"; capture its leading letter's case.
            if auto_capitalize and at_sentence_start and not pending_capital:
                if _first_letter(part[len("<dsf>"):-len("</dsf>")]).isupper():
                    pending_capital = True
            if at_sentence_start:
                drop_orphan = True
            # If a punctuation mark directly follows this removed span, drop the space that
            # preceded the span so the mark joins the previous kept word.
            next_plain = parts[index + 1] if index + 1 < len(parts) else ""
            if next_plain[:1] in _ATTACH_LEFT:
                while out and out[-1].isspace():
                    out.pop()
            continue
        for ch in part:
            if auto_capitalize and pending_capital and ch.isalpha():
                out.append(ch.upper())
                pending_capital = False
                at_sentence_start = False
                drop_orphan = False
            elif drop_orphan and at_sentence_start and ch in _ORPHAN_PUNCT:
                # Punctuation left stranded by a removed sentence-initial interjection: drop it.
                continue
            else:
                out.append(ch)
                if ch in _SENT_END:
                    at_sentence_start = True
                    pending_capital = False
                    drop_orphan = False
                elif not ch.isspace():
                    at_sentence_start = False
                    drop_orphan = False
    cleaned = "".join(out)
    cleaned = re.sub(r"</?dsf>", " ", cleaned)              # drop any unbalanced stray tag
    cleaned = re.sub(r"[,;:]\s*(?=[.!?…])", "", cleaned)  # drop a comma/etc. orphaned before a terminator
    cleaned = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", cleaned)  # "So, uh, we" -> "So, we", not "So,, we"
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = _SPACE_BEFORE_PUNCT.sub(r"\1", cleaned)
    cleaned = _SPACE_AFTER_OPEN.sub(r"\1", cleaned)
    return cleaned


def build_clean(annotation, auto_capitalize=True):
    """Render an :class:`Annotation` as text with disfluencies removed."""
    tagged = _build_tagged(annotation.tokens, annotation.labels)
    return strip_disfluency_tags(tagged, auto_capitalize=auto_capitalize)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def remove_disfluencies(model, lines, threshold=0.5, batch_size=32, gpu=None, progress=True,
                        word_level=True, auto_capitalize=True,
                        asymmetric=None, postfilter=False, protect=True,
                        strip_fillers_mode="pre", iterate=1):
    """Detect and remove disfluencies, returning one cleaned sentence per input line.

    :param word_level: when ``True`` (default), score each word by the mean over its sub-word
        pieces; set to ``False`` to score it by the maximum (more aggressive removal).
    :param auto_capitalize: when ``True`` (default) and a capitalized word was removed from the
        start of the sentence, the new first word is capitalized in turn.
    :param asymmetric: ``None`` for a single ``threshold`` (default), or a ``(low, high)`` pair
        for the asymmetric mode (lenient on repetitions/hesitations, strict elsewhere).
    :param postfilter: keep a removal only if it is a hesitation/truncation/repeat (off).
    :param protect: never remove a connector/number unless it is a repetition (on by default).
    :param strip_fillers_mode: ``"pre"`` (default, strip filled pauses/truncations before the
        model), ``"post"`` (strip them from the model output) or ``"none"``.
    :param iterate: re-apply the whole removal up to this many times, stopping once the text
        stops changing (default 1 = a single pass).
    """
    def clean_once(current):
        if strip_fillers_mode == "pre":
            current = [strip_fillers(line) for line in current]
        annotations = annotate(
            model, current, threshold=threshold, batch_size=batch_size, gpu=gpu,
            progress=progress, word_level=word_level,
            asymmetric=asymmetric, postfilter=postfilter, protect=protect,
        )
        cleaned = [build_clean(a, auto_capitalize=auto_capitalize) for a in annotations]
        if strip_fillers_mode == "post":
            cleaned = [strip_fillers(line) for line in cleaned]
        return cleaned

    current = lines
    for _ in range(max(1, iterate)):
        nxt = clean_once(current)
        if nxt == current:
            break
        current = nxt
    return current


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #

def read_lines(text_args):
    """Return the input lines, from text file(s) or from a single raw string."""
    if any(os.path.isfile(arg) for arg in text_args):
        lines = []
        for path in text_args:
            with open(path, "r", encoding="utf-8") as f:
                lines += [line.rstrip("\n") for line in f]
        return lines
    return [" ".join(text_args)]


def _resolve_gpu(gpu):
    gpu = gpu if gpu >= 0 else None
    if gpu is not None and not torch.cuda.is_available():
        warnings.warn("GPU is not available. Using CPU instead.")
        gpu = None
    return gpu


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Remove disfluencies from English text with a trained detector (Clausa).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "text",
        help="Input text file(s) (one sentence per line), or a raw string to clean",
        type=str, nargs="+",
    )
    parser.add_argument(
        "--model_id",
        help="Hugging Face repository id, or a path to a local model folder",
        default=HF_MODEL_ID, type=str,
    )
    parser.add_argument(
        "-o", "--output",
        help="Write the cleaned text to this file instead of printing it",
        default=None, type=str,
    )
    parser.add_argument(
        "--threshold",
        help="Decision threshold in [0, 1] (higher = more conservative); -1 for asymmetric",
        default=0.5, type=float,
    )
    parser.add_argument(
        "--strip_fillers", choices=("none", "pre", "post"), default="pre",
        help="Remove uh/um/er-type sounds and truncations before (pre, default) or after "
             "(post) the model; 'none' to disable",
    )
    parser.add_argument(
        "--iterate", type=int, default=1,
        help="Re-apply removal up to N times until the text stops changing",
    )
    parser.add_argument(
        "--postfilter", default=False, action="store_true",
        help="Keep a removal only if it is a hesitation, a truncation, or an exact repeat",
    )
    parser.add_argument(
        "--no_protect", dest="protect", default=True, action="store_false",
        help="Allow removing connector/number words (protect-list is on by default)",
    )
    parser.add_argument("--batch_size", help="Batch size", default=32, type=int)
    parser.add_argument(
        "--gpu", help="GPU index (0, 1, ...). Use -1 to run on CPU", default=0, type=int
    )
    parser.add_argument(
        "--subword_level",
        help="Score a word by the maximum over its sub-word pieces instead of the mean",
        default=False, action="store_true",
    )
    parser.add_argument(
        "--no_auto_capitalize",
        help="Do not capitalize the new first word when a capitalized word is removed at the start",
        default=False, action="store_true",
    )
    parser.add_argument(
        "-q", "--quiet", help="Do not show the progress bar", default=False, action="store_true"
    )
    args = parser.parse_args()

    gpu = _resolve_gpu(args.gpu)
    model = load_model(args.model_id, device="cpu" if gpu is None else "cuda:%d" % gpu)

    asymmetric = (ASYM_LOW, ASYM_HIGH) if args.threshold < 0 else None

    lines = read_lines(args.text)
    cleaned = remove_disfluencies(
        model, lines, threshold=args.threshold, batch_size=args.batch_size,
        gpu=gpu, progress=not args.quiet, word_level=not args.subword_level,
        auto_capitalize=not args.no_auto_capitalize,
        asymmetric=asymmetric, postfilter=args.postfilter, protect=args.protect,
        strip_fillers_mode=args.strip_fillers, iterate=args.iterate,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("\n".join(cleaned) + "\n")
        print(f"Cleaned text written to {args.output}")
    else:
        for line in cleaned:
            print(line)
