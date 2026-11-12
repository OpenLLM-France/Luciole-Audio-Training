"""Length in seconds of every audio file of some NeMo manifests, from the headers, as a TSV
"<audio path>\t<seconds>" (0 for an unreadable file) for the audio_bounds filter of clean_asr.py when the audio is not
reachable where the cleaning runs (Yodas2: audio on JZ only, run this there).

  python audio_durations.py $DATA_FOLDER/nemo/asr/it/clean/Yodas2/intermediate/train*.jsonl \
      -o $DATA_FOLDER/nemo/asr/it/clean/Yodas2/intermediate/audio_durations.tsv --workers 8
"""
import argparse
import json
import os
from multiprocessing import Pool

import soundfile as sf


def audio_paths(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            for t in rec.get("conversations") or []:
                if t.get("type") == "audio":
                    yield t["value"]
            if "audio_filepath" in rec:
                yield rec["audio_filepath"]


def length(path):
    try:
        info = sf.info(path)
        return path, info.frames / info.samplerate
    except Exception as e:
        return path, f"{type(e).__name__}: {e}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifests", nargs="*")
    p.add_argument("--paths", help="file with one audio path per line, instead of / with the manifests")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    paths = {a for m in args.manifests for a in audio_paths(m)}
    if args.paths:
        with open(args.paths, encoding="utf-8") as f:
            paths |= {line.strip() for line in f if line.strip()}
    paths = sorted(paths)
    with Pool(args.workers) as pool:
        res = pool.map(length, paths, chunksize=64)
    bad = [(a, r) for a, r in res if isinstance(r, str)]
    with open(args.output + ".tmp", "w", encoding="utf-8") as f:
        for a, r in res:
            f.write(f"{a}\t{0.0 if isinstance(r, str) else r:.3f}\n")   # unreadable: 0 s, its lines dropped
    os.replace(args.output + ".tmp", args.output)
    print(f"{len(paths)} audio files, {len(bad)} unreadable -> {args.output}")
    for a, r in bad[:20]:
        print("  unreadable:", a, r)


if __name__ == "__main__":
    main()
