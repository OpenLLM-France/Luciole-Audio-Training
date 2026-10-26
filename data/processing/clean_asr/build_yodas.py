"""Rebuild the Yodas ASR manifest from the raw YODAS data, keeping the subtitle punctuation.

The first Yodas manifest (nemo/asr/fr/context/Yodas) lost the subtitle
punctuation (no "?", most periods gone) and got commas inserted mid-phrase
("des, monuments"), leaving capitals with no boundary before them
("contenu Et aujourd'hui"). The raw cues still have it:

    raw cues: "plus tard… la permutation et ainsi de suite. Bien, donc ce sont en quelque sorte"
    Yodas   : "plus tard la permutation et ainsi de suite Bien donc ce sont en quelque sorte,"

Sources (raw/transcript/fr/YODAS):
    audios_merged/fr000/audio/<shard>/<video>-<first cue>-<start>-<end>.wav   segments merged from
                                                                             consecutive cues (times in 10 ms)
    fr000/text/*.txt                                                         one cue per line: "<cue id> <text>"

Each segment's text is the cues of its video lying within [start, end],
joined with a space, written as is (digits, symbols, [Musique], [ __ ]).
The merged wav drops the silences between cues, so the duration is read from
the file, not from end - start.
Segments with no cue text are skipped, and so are the ~0.2 % whose end is
before their start: the merge ran past the end of the video into the next one
(next by name), with a wav duration that matches neither cue range. Segments
whose first cue in range is not the one in their id are skipped too
(overlapping cue times, or a broken subtitle file with raw ASS lines).
The prompt is drawn from the `default_contexts` of
data/contexts/fr_asr_contexts.json, seeded by the id.
Lines are shuffled with --seed and written to
--output (clean/Yodas/intermediate/train.jsonl); then run
clean_asr.py (Yodas entry of conf/fr.yaml), which writes
clean/Yodas/train.jsonl.

Usage:
    python data/processing/clean_asr/build_yodas.py --limit 2000 --output /tmp/yodas_test/train.jsonl
    python data/processing/clean_asr/build_yodas.py
"""
import argparse
import glob
import hashlib
import json
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import soundfile as sf
from tqdm import tqdm


YODAS = "/data-server/datasets/audio/raw/transcript/fr/YODAS"
OUTPUT = "/data-server/datasets/audio/nemo/asr/fr/clean/Yodas/intermediate/train.jsonl"
CONTEXTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "contexts",
                        "fr_asr_contexts.json")


def parse_id(seg_id):
    video, idx, start, end = seg_id.rsplit("-", 3)
    return video, int(idx), int(start), int(end)


def list_segments(merged_dir, limit):
    segs = []
    for shard in sorted(os.listdir(merged_dir)):
        shard_dir = os.path.join(merged_dir, shard)
        if not os.path.isdir(shard_dir):
            continue
        for name in sorted(os.listdir(shard_dir)):
            if name.endswith(".wav"):
                segs.append(os.path.join(shard_dir, name))
                if limit and len(segs) >= limit:
                    return segs
    return segs


def load_cues(text_dir, videos):
    cues = defaultdict(list)
    for fn in tqdm(sorted(glob.glob(os.path.join(text_dir, "*.txt"))), desc="raw cues", leave=False):
        with open(fn, encoding="utf-8") as f:
            for line in f:
                cue_id, _, text = line.rstrip("\n").partition(" ")
                video, idx, start, end = parse_id(cue_id)
                if video in videos:
                    cues[video].append((start, end, idx, text.strip()))
    for v in cues.values():
        v.sort()
    return cues


def seeded_rng(seed, key):
    return random.Random(int.from_bytes(hashlib.md5(f"{seed}:{key}".encode()).digest()[:8], "little"))


def draw_prompt(contexts, rng):
    groups = contexts["default_contexts"]
    weights = [float(w) for w in groups]
    return rng.choice(rng.choices(list(groups.values()), weights=weights)[0])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yodas", default=YODAS, help="raw YODAS root")
    p.add_argument("--subset", default="fr000")
    p.add_argument("--output", default=OUTPUT)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--limit", type=int, default=0, help="only the first N segments (test)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--workers", type=int, default=32, help="threads reading wav durations")
    args = p.parse_args()

    out = args.output
    if os.path.exists(out) and not args.overwrite:
        raise SystemExit(f"{out} exists (use --overwrite)")
    with open(CONTEXTS, encoding="utf-8") as f:
        contexts = json.load(f)

    segs = list_segments(os.path.join(args.yodas, "audios_merged", args.subset, "audio"), args.limit)
    videos = {parse_id(os.path.basename(s)[:-4])[0] for s in segs}
    cues = load_cues(os.path.join(args.yodas, args.subset, "text"), videos)

    records, n_empty, n_cross, n_first_mismatch = [], 0, 0, 0
    for path in tqdm(segs, desc="segments", leave=False):
        seg_id = os.path.basename(path)[:-4]
        video, idx, start, end = parse_id(seg_id)
        if end < start:
            n_cross += 1
            continue
        inside = [c for c in cues.get(video, ()) if c[0] >= start and c[1] <= end]
        text = " ".join(t for _, _, _, t in inside if t)
        if not text:
            n_empty += 1
            continue
        if inside[0][2] != idx:  # overlapping cue times or broken subtitle file (raw ASS lines)
            n_first_mismatch += 1
            continue
        records.append({
            "id": seg_id,
            "dataset_name": f"YODAS_rawtext_{args.subset}",
            "conversations": [
                {"from": "User", "value": draw_prompt(contexts, seeded_rng(args.seed, seg_id)), "type": "text"},
                {"from": "User", "value": path, "type": "audio", "duration": None},
                {"from": "Assistant", "value": text, "type": "text"},
            ],
        })

    with ThreadPoolExecutor(args.workers) as pool:
        durations = list(tqdm(pool.map(lambda r: sf.info(r["conversations"][1]["value"]).duration, records),
                              total=len(records), desc="durations", leave=False))
    for rec, d in zip(records, durations):
        rec["conversations"][1]["duration"] = round(d, 2)

    random.Random(args.seed).shuffle(records)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(out + ".tmp", out)

    hours = sum(r["conversations"][1]["duration"] for r in records) / 3600
    print(f"{out}\n  {len(records)}/{len(segs)} segments ({hours:,.1f} h), no cue text: {n_empty}, "
          f"spanning two videos: {n_cross}, "
          f"first cue != id cue: {n_first_mismatch}")


if __name__ == "__main__":
    main()
