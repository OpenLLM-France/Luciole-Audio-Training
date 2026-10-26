"""Build [silence] examples from the gaps between the segments of long recordings.

A gap is a stretch of a recording that no segment of the manifest covers (all segments of
the clean/<Dataset>/intermediate/ file and of the context/ file of the same name, so a
segment dropped by a rebuild still counts as speech). Gaps of `--min-gap` to `--max-gap`
seconds are checked on the audio: no speech for Silero VAD (music and noise are not speech
for it either), an RMS level of at most `--max-db` dBFS (quiet music or room noise is around
-35 to -45) and at least `--min-db-below` dB under the neighbouring segments.
Each accepted gap gives one line, of one of four kinds drawn at random:

    alone    the gap alone (0.1 s trimmed on each side)     "[silence]"
    after    the previous segment extended over the gap     "<previous text> [silence]"
    before   the next segment started at the gap            "[silence] <next text>"
    between  previous segment + gap + next segment          "<previous text> [silence] <next text>"

(no line longer than `--max-duration` s). The lines keep the neighbouring segment's fields
and get the id "<segment id>_silence<gap start in ms>". They are written next to the input, with "_silence"
before ".jsonl" (clean/ESLO/intermediate/train_recasepunc_silence.jsonl),
and clean_asr.py reads them with the dataset's other inputs: in the base output the tag is
stripped ("alone" lines are dropped, the others are plain lines over a longer audio), in
events/ it is kept.

Usage:
    python data/processing/clean_asr/build_silence.py --datasets ESLO Multilingual_TEDx
"""
import argparse
import collections
import copy
import glob
import json
import os
import random

import numpy as np
import soundfile as sf
from multiprocessing import Pool

from rebuild.common import CLEAN, CONTEXT, assistant_text as text_turn, audio_turn

# datasets with an events/ variant and segments cut from long recordings
DATASETS = ["Multilingual_TEDx", "YouTubeFr", "ESLO", "TCOF_Adultes", "TCOF_Enfants", "CFPP2000", "CLAPI",
            "CFPB", "PFC", "ACSYNT", "LINAGORA_Meetings", "LVL-Atril-CTFNN1", "LVL-Atril-CTFAR"]
KINDS = ["alone", "after", "before", "between"]
SR = 16000
_vad = {}


def span(rec):
    a = audio_turn(rec)
    start = a.get("offset") or 0.0
    return a["value"], start, start + a["duration"]


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def find_gaps(recs, extra, min_gap, max_gap):
    """(wav, start, end, previous rec or None, next rec or None) for each uncovered stretch."""
    by_wav = collections.defaultdict(list)
    for rec in recs:
        wav, a, b = span(rec)
        by_wav[wav].append((a, b, rec))
    for rec in extra:                                   # context/ segments: speech, no usable text
        wav, a, b = span(rec)
        if wav in by_wav:
            by_wav[wav].append((a, b, None))
    gaps = []
    for wav, segs in by_wav.items():
        segs.sort(key=lambda x: (x[0], x[1]))
        end, last = segs[0][1], segs[0][2]
        for a, b, rec in segs[1:]:
            if min_gap <= a - end <= max_gap:
                gaps.append((wav, end, a, last, rec))
            if b > end:
                end, last = b, rec
            elif b == end and rec is not None:
                last = rec
    return gaps


def _init():
    import torch
    from silero_vad import load_silero_vad
    torch.set_num_threads(1)
    _vad["model"] = load_silero_vad()


def _read(wav, start, end):
    with sf.SoundFile(wav) as f:
        if f.samplerate != SR:
            return None
        f.seek(max(int(start * SR), 0))
        x = f.read(max(int((end - start) * SR), 0), dtype="float32", always_2d=True)
    return x.mean(axis=1)


def _db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)


def check(args):
    """'ok', 'speech', 'loud' or 'error' for one gap, with the gap and reference levels."""
    wav, start, end, ref_start, ref_end, threshold, max_db, min_db_below = args
    import torch
    from silero_vad import get_speech_timestamps
    try:
        x = _read(wav, start, end)
    except Exception:
        return "error", None, None
    if x is None or len(x) < SR // 2:
        return "error", None, None
    gap_db = _db(x)
    if gap_db > max_db:                                 # cheap test first: no VAD for loud gaps
        return "loud", round(gap_db, 1), None
    speech = get_speech_timestamps(torch.from_numpy(x), _vad["model"], threshold=threshold, sampling_rate=SR,
                                   min_speech_duration_ms=150)
    if speech:
        return "speech", None, None
    try:
        # reference level: up to 3 s of the neighbouring segments (speech)
        ref = [_read(wav, max(start - 3, ref_start), start - 0.1), _read(wav, end + 0.1, min(end + 3, ref_end))]
    except Exception:
        return "error", None, None
    ref = [r for r in ref if r is not None and len(r) > SR // 10]
    if not ref:
        return "error", None, None
    ref_db = max(_db(r) for r in ref)
    return ("ok" if gap_db <= ref_db - min_db_below else "loud"), round(gap_db, 1), round(ref_db, 1)


def make_line(kind, gap, max_duration):
    wav, start, end, prev, nxt = gap
    if kind == "alone":
        rec, a, b, text = prev or nxt, start + 0.1, end - 0.1, "[silence]"
    elif kind == "after":
        if prev is None or end - span(prev)[1] > max_duration:
            return None
        rec, a, b, text = prev, span(prev)[1], end, f"{text_turn(prev)['value']} [silence]"
    elif kind == "before":
        if nxt is None or span(nxt)[2] - start > max_duration:
            return None
        rec, a, b, text = nxt, start, span(nxt)[2], f"[silence] {text_turn(nxt)['value']}"
    else:
        if prev is None or nxt is None or span(nxt)[2] - span(prev)[1] > max_duration:
            return None
        rec, a, b = prev, span(prev)[1], span(nxt)[2]
        text = f"{text_turn(prev)['value']} [silence] {text_turn(nxt)['value']}"
    if rec is None:
        return None
    out = copy.deepcopy(rec)
    out["id"] = f"{rec['id']}_silence{round(start * 1000)}"   # a segment can border two gaps
    turn = audio_turn(out)
    turn["offset"], turn["duration"] = round(a, 3), round(b - a, 3)
    text_turn(out)["value"] = text
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--fraction", type=float, default=0.02, help="silence lines per line of the input")
    p.add_argument("--max-per-dataset", type=int, default=5000)
    p.add_argument("--min-gap", type=float, default=2.0, help="shorter gaps are pauses, untagged elsewhere")
    p.add_argument("--max-gap", type=float, default=10.0)
    p.add_argument("--max-duration", type=float, default=30.0)
    p.add_argument("--vad-threshold", type=float, default=0.3, help="lower than Silero's 0.5: stricter")
    p.add_argument("--max-db", type=float, default=-50.0, help="dBFS, 0 = full scale")
    p.add_argument("--min-db-below", type=float, default=15.0)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    with Pool(args.workers, initializer=_init) as pool:
        for ds in args.datasets:
            inputs = [f for f in sorted(glob.glob(f"{CLEAN}/{ds}/intermediate/train*.jsonl"))
                      if "max30" not in f and "_silence" not in f]
            per_file = -(-args.max_per_dataset // max(len(inputs), 1))
            for path in inputs:
                rng = random.Random(f"{args.seed}:{ds}:{os.path.basename(path)}")
                recs = load(path)
                ctx_path = f"{CONTEXT}/{ds}/{os.path.basename(path)}"
                extra = load(ctx_path) if os.path.exists(ctx_path) else []
                gaps = find_gaps(recs, extra, args.min_gap, args.max_gap)
                rng.shuffle(gaps)
                target = min(round(args.fraction * len(recs)), per_file)
                lines, stats, levels, i = [], collections.Counter(), [], 0
                while len(lines) < target and i < len(gaps):
                    batch = gaps[i:i + 4 * (target - len(lines)) + 16]
                    i += len(batch)
                    jobs = [(g[0], g[1] + 0.1, g[2] - 0.1, span(g[3])[1] if g[3] else g[1] - 3,
                             span(g[4])[2] if g[4] else g[2] + 3, args.vad_threshold, args.max_db, args.min_db_below)
                            for g in batch]
                    for g, (verdict, gap_db, ref_db) in zip(batch, pool.map(check, jobs, chunksize=4)):
                        stats[verdict] += 1
                        if verdict != "ok" or len(lines) >= target:
                            continue
                        levels.append(ref_db - gap_db)
                        kinds = KINDS[:]
                        rng.shuffle(kinds)
                        for k in kinds:
                            line = make_line(k, g, args.max_duration)
                            if line is not None:
                                lines.append(line)
                                stats["kind:" + k] += 1
                                break
                out = path[: -len(".jsonl")] + "_silence.jsonl"
                with open(out + ".tmp", "w", encoding="utf-8") as f:
                    for line in sorted(lines, key=lambda r: r["id"]):
                        f.write(json.dumps(line, ensure_ascii=False) + "\n")
                os.replace(out + ".tmp", out)
                hours = sum(audio_turn(r)["duration"] for r in lines) / 3600
                print(f"[{ds}] {os.path.basename(path)}: {len(gaps)} gaps, checked {i}, {dict(stats)}, "
                      f"median dB under speech {np.median(levels) if levels else 0:.1f} -> {out} "
                      f"({len(lines)} lines, {hours:.2f} h)", flush=True)


if __name__ == "__main__":
    main()
