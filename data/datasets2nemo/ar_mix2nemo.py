#!/usr/bin/env python3
"""
Convert the curated Arabic ASR mix (kaldi -> NeMo multiturn) into
/data-server/datasets/audio/nemo/asr/ar/{nocontext,context}/<Name>/<split>.jsonl

Uses the ssak NemoDataset/KaldiDataset machinery:
  KaldiDataset.load        -> reads kaldi dir (sox-pipe wav.scp, segments->offset)
  NemoDataset.kaldi_to_nemo-> kaldi row -> multiturn row (audio turn + text turn)
  NemoDataset.extend       -> merge several kaldi dirs into one split
  NemoDataset.filter       -> utt-id filtering (MGB2 WMER=0) / drop corrupt segments
  NemoDataset.normalize_audios -> resample non-16k sources to 16k (like normalize_audios.py)
  set_context              -> add natural-language prompts (context variant)

The kaldi wav.scp paths are stale (/media/nas, /media/storage0). We remap each
audio turn to the real file on /data-server by basename lookup (build_audio_index),
which is robust to the sub-directory layout differences between machines.

Usage:
  python ar_mix2nemo.py --only ESCWA MGB5      # subset
  python ar_mix2nemo.py                         # all 7 datasets
"""
import os
import sys
import logging
import argparse
from pathlib import Path

PROCESSING = Path(__file__).resolve().parent.parent / "processing"   # data/processing
sys.path.insert(0, str(PROCESSING))

from ssak.utils.kaldi_dataset import KaldiDataset
from ssak.utils.nemo_dataset import NemoDataset
from set_context import set_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ar_mix2nemo")

KALDI = "/data-server/datasets/audio/kaldi/ar/raw"
RAW = "/data-server/datasets/audio/raw/transcript/ar"
OUT = "/data-server/datasets/audio/nemo/asr/ar"
CONVERTED = "/data-server/datasets/audio/converted_audios/transcript/ar"
DEFAULT_FILTER_DIR = "/tmp/claude-1005/-home-abert-airflow/07599129-18b1-4367-86b9-623d318594b2/scratchpad/mix"

CTX_CATEGORY = "nocasepunc"    # category in contexts/ar_asr_contexts.json
MASC_MAX_DUR = 60.0            # drop MASC segments with corrupt (>60s) durations

# split -> list of kaldi dirs to merge. audio_root = /data-server tree indexed by
# basename. convert=True -> resample to 16k. filter -> {split: utt2dur/list file}.
DATASETS = {
    "MGB2": dict(
        audio_root=f"{RAW}/MSA/MGB2",
        convert=False,
        splits={"train": [f"{KALDI}/MSA/MGB2/train"],
                "dev":   [f"{KALDI}/MSA/MGB2/dev"],
                "test":  [f"{KALDI}/MSA/MGB2/test"]},
        filter={"train": "MGB2_wmer0.utt2dur"},
    ),
    "MASC": dict(
        audio_root=f"{RAW}/MISC/MASC/audios",
        convert=False,
        max_duration=MASC_MAX_DUR,                   # drop corrupt-duration segments
        splits={"train": [f"{KALDI}/MISC/MASC/clean_train"],
                "dev":   [f"{KALDI}/MISC/MASC/clean_dev"],
                "test":  [f"{KALDI}/MISC/MASC/clean_test"]},
    ),
    "ESCWA": dict(
        audio_root=f"{RAW}/MISC/ESCWA", convert=False,
        splits={"all": [f"{KALDI}/MISC/ESCWA"]},
    ),
    "MGB5": dict(
        audio_root=f"{RAW}/ALG/MGB5", convert=False,
        splits={"train": [f"{KALDI}/ALG/MGB5-train"],
                "test":  [f"{KALDI}/ALG/MGB5-test"]},
    ),
    "ARABIC-SPEECH": dict(
        audio_root=f"{RAW}/MSA/ARABIC-SPEECH-CORPUS", convert=True,
        splits={"train": [f"{KALDI}/MSA/ARABIC-SPEECH/train"],
                "test":  [f"{KALDI}/MSA/ARABIC-SPEECH/test"]},
    ),
    "TARIC": dict(
        audio_root=f"{RAW}/TN/TARIC_dataset_v1",
        audio_exclude=["audio_augmented"],           # keep originals, skip augmented
        convert=True,
        splits={"train": [f"{KALDI}/TN/TARIC/train"],
                "dev":   [f"{KALDI}/TN/TARIC/dev"],
                "test":  [f"{KALDI}/TN/TARIC/test"]},
    ),
    "TunSwitch": dict(
        audio_root=f"{RAW}/TN/TunSwitch", convert=True,
        splits={"train": [f"{KALDI}/TN/TunSwitch/TunSwitchCS_train", f"{KALDI}/TN/TunSwitch/TunSwitchTO_train"],
                "dev":   [f"{KALDI}/TN/TunSwitch/TunSwitchCS_dev",   f"{KALDI}/TN/TunSwitch/TunSwitchTO_dev"],
                "test":  [f"{KALDI}/TN/TunSwitch/TunSwitchCS_test",  f"{KALDI}/TN/TunSwitch/TunSwitchTO_test"]},
    ),
}


def build_audio_index(root, exclude=None):
    """basename (with and without extension) -> list of realpaths under root.

    A list (not a single path) because some datasets reuse the same basename in
    different sub-dirs (e.g. ARABIC-SPEECH train WAV_dir/ vs test_set/WAV_dir/).
    remap_audio disambiguates by the stale path's suffix.
    """
    from collections import defaultdict
    exclude = exclude or []
    idx = defaultdict(list)
    for dp, _, files in os.walk(root):
        if any(e in dp for e in exclude):
            continue
        for f in files:
            if f.lower().endswith((".wav", ".flac", ".mp3")):
                full = os.path.join(dp, f)
                idx[f].append(full)
                stem = os.path.splitext(f)[0]
                if stem != f:
                    idx[stem].append(full)
    return idx


SHIM_ROOT = "/home/abert/abert/ssak/tools/nemo/datasets2nemo/.shims"
# fallback text files, in order of preference, when a kaldi dir's `text` is
# a broken symlink (some MGB2 dev/test `text` point to a dead /media path).
TEXT_FALLBACKS = ["text.utf8", "text_utf8_latin", "text.utf8_norm"]


def usable_kaldi_dir(kaldi_dir):
    """Return a kaldi dir whose `text` is readable.

    If `<dir>/text` is fine, return it unchanged. If it is a broken symlink,
    build a shim dir that symlinks every kaldi file but swaps `text` for the
    first readable fallback. Never modifies the shared source dir.
    """
    text = os.path.join(kaldi_dir, "text")
    if os.path.isfile(text) and os.access(text, os.R_OK):
        return kaldi_dir
    alt = next((os.path.join(kaldi_dir, t) for t in TEXT_FALLBACKS
                if os.path.isfile(os.path.join(kaldi_dir, t))), None)
    if alt is None:
        raise FileNotFoundError(f"No readable text file in {kaldi_dir}")
    shim = os.path.join(SHIM_ROOT, kaldi_dir.strip(os.sep).replace(os.sep, "_"))
    os.makedirs(shim, exist_ok=True)
    for f in os.listdir(kaldi_dir):
        if f == "text":
            continue
        link = os.path.join(shim, f)
        if not os.path.lexists(link):
            os.symlink(os.path.join(kaldi_dir, f), link)
    tlink = os.path.join(shim, "text")
    if os.path.lexists(tlink):
        os.remove(tlink)
    os.symlink(alt, tlink)
    logger.info(f"[shim] {kaldi_dir}: broken text -> using {os.path.basename(alt)}")
    return shim


def expand_kaldi_dirs(dirs):
    """A configured dir may be a kaldi dir, or a parent of many (MASC countries)."""
    out = []
    for d in dirs:
        if os.path.isfile(os.path.join(d, "wav.scp")):
            out.append(d)
        else:
            for dp, _, files in os.walk(d):
                if "wav.scp" in files:
                    out.append(dp)
    return out


def build_split_manifest(kaldi_dirs, name):
    """Build ONE NeMo manifest for a split out of one or more kaldi dirs.

    A split can span several kaldi dirs that must end up in a single
    <split>.jsonl, e.g. MASC/train = 16 country dirs, TunSwitch/train =
    CS + TO. We load each kaldi dir, convert it, and accumulate its rows
    into `manifest` (that is all `.extend` does). For a single-dir split
    (ESCWA, MGB5, ...) the loop runs once = plain load -> convert.
    """
    manifest = NemoDataset(name=name)
    for kaldi_dir in expand_kaldi_dirs(kaldi_dirs):
        kaldi_dir = usable_kaldi_dir(kaldi_dir)     # repair broken `text` symlink if any
        kaldi = KaldiDataset(name=name, row_checking_kwargs=dict(show_warnings=False))
        kaldi.load(kaldi_dir)                       # read kaldi dir (segments -> offset)
        one_dir = NemoDataset(name=name)
        one_dir.kaldi_to_nemo(kaldi)                # kaldi rows -> multiturn rows
        manifest.extend(one_dir)                    # accumulate into the single split
    return manifest


def _best_suffix_match(stale, candidates):
    """Pick the candidate path sharing the most trailing path components with `stale`."""
    sp = stale.split(os.sep)
    best, best_n = candidates[0], -1
    for c in candidates:
        cp = c.split(os.sep)
        n = 0
        for a, b in zip(reversed(sp), reversed(cp)):
            if a == b:
                n += 1
            else:
                break
        if n > best_n:
            best_n, best = n, c
    return best


def remap_audio(nd, audio_index):
    """Repoint each audio turn to the real /data-server file.

    Match by basename; when several files share that basename (e.g. train
    WAV_dir/ vs test_set/WAV_dir/), pick the one whose path best matches the
    stale kaldi path's suffix. Returns the set of unmatched basenames.
    """
    missing = set()
    for row in nd:
        for turn in row.get_audio_turns():
            stale = turn.value
            base = os.path.basename(stale)
            cands = audio_index.get(base) or audio_index.get(os.path.splitext(base)[0]) or []
            if not cands:
                missing.add(base)
            elif len(cands) == 1:
                turn.value = cands[0]
            else:
                turn.value = _best_suffix_match(stale, cands)
    return missing


# ---- cleaning thresholds ----
MIN_DURATION = 0.1        # drop segments shorter than this (seconds)
LONG_SEC = 60.0           # a segment is "long" beyond this
LONG_MIN_CPS = 1.0        # long + fewer chars/sec than this = misaligned -> drop
AUDIO_TOL = 0.5           # allowed slack (s) for segment end vs audio file end


def _cps(row):
    """Characters (no spaces) per second of audio."""
    if not row.text or not row.duration:
        return 0.0
    return len(row.text.replace(" ", "")) / row.duration


def clean_content(nd, tag):
    """Filters that need no audio I/O. Logs how many rows each one drops."""
    def step(keep_pred, label):
        removed = nd.filter(keep_pred)          # filter keeps rows where pred is True
        if removed:
            logger.info(f"[{tag}] dropped {len(removed):,} rows: {label}")

    step(lambda r: bool(r.text and r.text.strip()), "empty text")
    step(lambda r: r.duration is not None and r.duration >= MIN_DURATION, f"duration < {MIN_DURATION}s")
    step(lambda r: not (r.duration and r.duration > LONG_SEC and _cps(r) < LONG_MIN_CPS),
         f"long audio (> {LONG_SEC:g}s) with short text (cps < {LONG_MIN_CPS:g})")


def clean_audio(nd, tag, workers=8):
    """Drop rows whose audio is absent/corrupted, or whose segment runs past the file end."""
    import soundfile as sf
    from concurrent.futures import ThreadPoolExecutor

    def probe(p):
        try:
            info = sf.info(p)
            return p, info.frames / info.samplerate
        except Exception:
            return p, None                      # missing or unreadable/corrupted

    paths = list(nd.get_audio_paths(unique=True))
    durs = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for p, d in ex.map(probe, paths):
            durs[p] = d
    n_bad_files = sum(1 for d in durs.values() if d is None)

    def keep(r):
        filedur = durs.get(r.audio_filepath)
        if filedur is None:
            return False                        # absent / corrupted audio
        return (r.offset or 0) + (r.duration or 0) <= filedur + AUDIO_TOL

    removed = nd.filter(keep)
    if removed:
        logger.info(f"[{tag}] dropped {len(removed):,} rows: absent/corrupted audio or "
                    f"out-of-bounds ({n_bad_files} unreadable files)")


def process(name, spec, filter_dir):
    audio_index = build_audio_index(spec["audio_root"], spec.get("audio_exclude"))
    logger.info(f"[{name}] audio index: {len(audio_index):,} entries")
    results = []
    for split, kaldi_dirs in spec["splits"].items():
        nd = build_split_manifest(kaldi_dirs, name)
        n0 = len(nd)

        missing = remap_audio(nd, audio_index)
        nd.filter(lambda r: r.audio_filepath.startswith("/data-server"))   # drop unmatched audio

        filt = spec.get("filter", {}).get(split)
        if filt:
            keep = set()
            with open(os.path.join(filter_dir, filt)) as f:
                for line in f:
                    if line.strip():
                        keep.add(line.split()[0])
            nd.filter(lambda r, k=keep: r.id in k)

        if spec.get("max_duration"):
            md = spec["max_duration"]
            nd.filter(lambda r, m=md: r.duration is not None and r.duration <= m)

        tag = f"{name}/{split}"
        clean_content(nd, tag)                       # empty text, dur<0.1, long+short-text

        if spec["convert"]:
            nd.normalize_audios(CONVERTED, target_sample_rate=16000,
                                target_extension="flac", num_workers=8, relative_to=RAW)

        clean_audio(nd, tag)                          # absent/corrupted/out-of-bounds audio

        if len(nd) == 0:
            logger.warning(f"[{tag}] empty after filtering, skipped")
            continue

        noctx = os.path.join(OUT, "nocontext", name, f"{split}.jsonl")
        ctx = os.path.join(OUT, "context", name, f"{split}.jsonl")
        nd.save(noctx, data_type="multiturn")

        set_context(noctx, ctx, CTX_CATEGORY, task="asr", language="ar",
                    dataset_name=name, force_context=False)

        logger.info(f"[{name}/{split}] {n0} -> {len(nd)} rows "
                    f"(unmatched audios: {len(missing)}) -> {noctx}")
        results.append((name, split, len(nd), len(missing)))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None, help="subset of dataset names")
    ap.add_argument("--filter_dir", default=DEFAULT_FILTER_DIR)
    args = ap.parse_args()

    summary = []
    for name in (args.only or list(DATASETS)):
        logger.info(f"===== {name} =====")
        summary += process(name, DATASETS[name], args.filter_dir)

    print("\n===== SUMMARY =====")
    for name, split, cnt, miss in summary:
        print(f"  {name:16s} {split:6s} {cnt:>8,} rows  (unmatched audios: {miss})")


if __name__ == "__main__":
    main()
