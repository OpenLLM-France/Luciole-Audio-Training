"""Rebuild YouTubeFr text from the raw subtitle cues.

Each manifest line is a merge of consecutive cues (concat_segments.merge_segments):
its id is the first cue id and its offset/duration span [first cue start, last cue end].
The cues come from kaldi/fr/YouTubeFr/casepunc/<split> (text identical to the raw
YouTube CSV) and are joined again without the old separators: never a comma, ". "
only when the previous cue has no final punctuation and the next starts with a
capital, a space otherwise. Dialogue dashes ("-Mais") are removed, sound notes
("*rires*") become tags or are removed, brackets are removed (the asides are spoken),
lines with censored words ("p%tain") are dropped. No text normalisation (digits, œ, symbols kept).
Lines whose cues cannot be found exactly keep their current text.

The only rebuild that reads kaldi/: the cues there were re-segmented and re-timed from the
raw CSV (raw/transcript/fr/YouTubeFr/fr/<video>.csv) by the first conversion, and the
manifest windows are made of those cues, so the raw CSV rows cannot be matched to them
directly (only by aligning the words). The kaldi cue text is the CSV text with ' apostrophes,
no quotes and "..." for "…".
"""
import bisect
import collections
import os
import re
import string
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import audio_turn, assistant_text, rebuild_manifest  # noqa: E402

KALDI = "/data-server/datasets/audio/kaldi/fr/YouTubeFr/casepunc"
PUNCT = set(string.punctuation) | {"…", "»", "«", "”", "“", "’"}
EPS = 0.002
ASTERISK = re.compile(r"\*([^*]{2,40})\*")   # "*rires*", "*Tousse*": sound notes of the subtitles
CENSORED = re.compile(r"\w[%*#@]+\w")       # "p%tain"
BRACKETS = re.compile(r"[()\[\]{}]")          # spoken asides "(je devrais dire : ...)": keep the words
EVENTS = [("laughter", r"\bri(re|res|t|gol)"), ("cough", r"\btouss"), ("music", r"\b(musique|chanson)"), ("applause", r"\bapplaudi")]
DIALOGUE_DASH = re.compile(r"(?:^|(?<=[.?!…] ))[-–—]+\s*(?=[^\W\d_])")   # "-Mais il faut" (speaker turn)


def load_split(split):
    text = {}
    with open(os.path.join(KALDI, split, "text"), encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split(" ", 1)
            text[p[0]] = p[1].strip() if len(p) > 1 else ""
    cues = collections.defaultdict(list)
    with open(os.path.join(KALDI, split, "segments"), encoding="utf-8") as f:
        for line in f:
            utt, audio, start, end = line.split()
            cues[audio].append((float(start), float(end), utt))
    for v in cues.values():
        v.sort()
    return text, cues


def event_tag(m):
    note = m[1].lower()
    return next((f" [{tag}] " for tag, p in EVENTS if re.search(p, note)), " ")


def join_cues(texts):
    out = ""
    for t in texts:
        t = DIALOGUE_DASH.sub("", ASTERISK.sub(event_tag, BRACKETS.sub(" ", t))).strip()
        if not t:
            continue
        if out and out[-1] not in PUNCT and t[0].isupper():
            out += ". " + t
        else:
            out += (" " if out else "") + t
    return re.sub(r"\s+", " ", out).strip()


def make_fn(splits, stats):
    def fn(rec):
        split = rec["dataset_name"].replace("YouTubeFr_casepunc_", "")
        if split not in splits:
            splits.clear()
            splits[split] = load_split(split)
        text, cues = splits[split]
        audio = audio_turn(rec)
        stem = os.path.splitext(os.path.basename(audio["value"]))[0]
        start = audio.get("offset", 0.0)
        end = start + audio["duration"]
        L = cues.get(stem, [])
        i = bisect.bisect_left(L, (start - EPS,))
        while i < len(L) and L[i][2] != rec["id"] and L[i][0] <= start + EPS:
            i += 1
        if i == len(L) or L[i][2] != rec["id"]:
            stats["unmatched"] += 1
            return assistant_text(rec)["value"]
        window, cur_end = [L[i]], L[i][1]
        for s, e, u in L[i + 1:]:
            if abs(cur_end - end) <= EPS or s - cur_end > 1.0 or e - start > 30:
                break
            window.append((s, e, u))
            cur_end = e
        if abs(cur_end - end) > EPS:
            stats["unmatched"] += 1
            return assistant_text(rec)["value"]
        cue_texts = [text.get(c[2], "") for c in window]
        new = join_cues(cue_texts)
        if CENSORED.search(new):
            return None, "censored"
        if not new:
            if any(cue_texts):
                return None, "sound_note_only"     # "*Cris*", "*charge l'arme*"
            stats["empty_kept_current"] += 1
            return assistant_text(rec)["value"]
        return new
    return fn


if __name__ == "__main__":
    names = sys.argv[1:] or [f"train_split{i}.jsonl" for i in range(7)]
    splits = {}
    for name in names:
        stats = collections.Counter()
        rebuild_manifest("YouTubeFr", name, make_fn(splits, stats))
        print(f"  {dict(stats)}")
