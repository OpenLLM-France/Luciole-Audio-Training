"""Build the _nonspeech inputs: the [silence] lines of build_silence.py plus [noise] and [music] lines,
speech segments of the same manifest mixed with sounds or music.

For each clean/<Dataset>/intermediate/<name>_silence.jsonl, the speech input it was built from
(clean/<Dataset>/intermediate/<name>.jsonl, else context/<Dataset>/<name>.jsonl) gives segments
(1.5 to 12 s, 2 words or more), drawn without replacement while there are enough, for lines of
five kinds in equal shares:

    alone    a sound/music clip alone (1.5 to 10 s)            "[music]"
    before   clip (1.5 to 5 s), then a speech segment         "[music] <text>"
    after    a speech segment, then a clip                    "<text> [music]"
    around   clip, speech segment, clip (same clip going on)  "[music] <text> [music]"
    between  two consecutive segments of a recording, a clip  "<text 1> [music] <text 2>"
             in between

--per-silence new silence lines per silence line (default 1.5, 1 in English), then as many noise
lines and as many music lines as there are silence lines in all (real + new): the three backgrounds
in equal numbers, each at most --max-passes times the number of files of its pool (noise: ~23k
files, so ~46k lines). --max-silence caps the silence total (real lines drawn at random; English
default 100000: Yodas2 en has 218k real ones -> 100k silence, 46k noise, 100k music).
--every-line (read speech with no _silence file, FLEURS): every line of context/<Dataset>/train.jsonl
gives one noise line and one music line, each before / after / around at random (no alone, no
between, no silence), in clean/<Dataset>/intermediate/train_nonspeech.jsonl.
Output: clean/<Dataset>/intermediate/<name>_nonspeech.jsonl = the silence lines + the new ones,
read by clean_asr.py in place of the _silence file (base: tags stripped, "alone" lines kept with an
empty text; events/: tags kept). The mixed audio: 16 kHz mono flac in
converted_audios/transcript/<lang>/<Dataset>/nonspeech/. `custom.nonspeech` keeps what was mixed.
Their lengths: <name>_nonspeech_durations.tsv next to the manifest, to give clean_asr.py
--durations (no header read per line).

Mixing: the clip is one continuous background. Where there is no speech it plays at --noise-db /
--music-db from the speech RMS (alone: -35 to -20 dBFS). Under the speech the same clip, from the
same position, is added by audiomentations' AddBackgroundNoise at --bed-snr (5 to 50 dB, as ssak's
SpeechAugment). The two cross-fade over 0.3 to 1 s inside the speech; the speech segments get
20 ms fades. Every clip is checked with Silero VAD (no speech, no singing) before use.

Backgrounds (raw/): noise = sounds/Clotho (development + validation, captions with voices dropped)
and the noise/distant_noises files of -50 dBFS and over, uniformly over the files; silence = the
distant_noises files of -100 to -50 dBFS (faint room noise; levels in noise/silence/distant_noises_levels.tsv,
under -100: digital silence, not used), kept at their own level, under -50 dBFS and 15 dB or more
under the speech (as build_silence.py's gaps); music = music/fma (instrumental genres or echonest
instrumentalness > 0.8) and music/JamendoMaxCaps ("instrumental" captions), half each.

    python data/processing/clean_asr/build_nonspeech.py --lang fr
    python data/processing/clean_asr/build_nonspeech.py --lang en --datasets Yodas2
    python data/processing/clean_asr/build_nonspeech.py --lang fr --datasets FLEURS --every-line
    python data/processing/clean_asr/build_nonspeech.py --lang fr --datasets LVL-Atril-CTFAR --limit 40 --output-root tmp/nonspeech_test
"""
import argparse
import ast
import collections
import copy
import csv
import glob
import json
import os
import random
import re
from multiprocessing import Pool

import numpy as np
import soundfile as sf

SR = 16000
DATA = os.environ["DATA_FOLDER"]
RAW, NEMO = f"{DATA}/raw", f"{DATA}/nemo"
KINDS = ["alone", "before", "after", "around", "between"]
EVERY_LINE_KINDS = ["before", "after", "around"]
VOICE_RE = re.compile(r"\b(speak\w*|talk\w*|voices?|conversation|chatter\w*|sing\w*|song|crowd|laugh\w*|shout\w*|"
                      r"yell\w*|scream\w*|cr(y|ies|ying)|murmur\w*|announc\w*|radio|television|tv|man|woman|men|women|"
                      r"people|person|child\w*|kids?|baby|boy|girl|whisper\w*|cheer\w*|chant\w*|words?)\b", re.I)
VOCAL_RE = re.compile(r"\b(vocals?|voices?|sing\w*|singer|lyric\w*|rap\w*|choir|spoken|speech|chant\w*|hum\w*)\b", re.I)
FMA_INSTRUMENTAL = {"Instrumental", "Ambient", "Classical", "Soundtrack", "Chiptune", "Drone", "Piano", "Minimalism",
                    "Post-Rock", "Ambient Electronic", "Chip Music", "Composed Music", "New Age", "Modern Jazz",
                    "Free-Jazz", "Jazz: Out", "Big Band/Swing", "Glitch", "IDM", "Techno", "House", "Downtempo"}
FMA_EXCLUDED = {"Spoken", "Poetry", "Talk Radio", "Audio Collage", "Radio", "Spoken Weird", "Comedy", "Spoken Word",
                "Hip-Hop", "Rap"}
TAG_RE = re.compile(r"\[[^\[\]]+\]")


# --- background pools --------------------------------------------------------------------------

def pool_clotho():
    out = []
    for split in ["development", "validation"]:
        captions = f"{RAW}/sounds/Clotho/clotho_captions_{split}.csv"
        if not os.path.exists(captions):
            continue
        with open(captions, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                path = f"{RAW}/sounds/Clotho/{split}/{row['file_name']}"
                if not any(VOICE_RE.search(row[f"caption_{i}"]) for i in range(1, 6)) and os.path.exists(path):
                    out.append(path)
    return out


def distant_noises(lo, hi):
    """distant_noises files whose RMS (whole file, noise/silence/distant_noises_levels.tsv) is in [lo, hi) dBFS."""
    with open(f"{RAW}/noise/silence/distant_noises_levels.tsv", encoding="utf-8") as f:
        return [f"{RAW}/noise/distant_noises/{r['file']}" for r in csv.DictReader(f, delimiter="\t")
                if lo <= float(r["rms_dbfs"]) < hi]


def pool_fma():
    import pandas as pd
    meta = f"{RAW}/music/fma/fma_metadata"
    tracks = pd.read_csv(f"{meta}/tracks.csv", index_col=0, header=[0, 1])
    names = pd.read_csv(f"{meta}/genres.csv", index_col=0)["title"].to_dict()
    echonest = pd.read_csv(f"{meta}/echonest.csv", index_col=0, header=[0, 1, 2])[
        ("echonest", "audio_features", "instrumentalness")].reindex(tracks.index)
    out = []
    for tid, genres, inst in zip(tracks.index, tracks[("track", "genres_all")], echonest):
        g = {names.get(x) for x in ast.literal_eval(genres)}
        if g & FMA_EXCLUDED or not (g & FMA_INSTRUMENTAL or inst > 0.8):
            continue
        path = f"{RAW}/music/fma/fma_large_flac/{tid:06d}.flac"
        if os.path.exists(path):
            out.append(path)
    return out


def pool_jamendo():
    out = []
    with open(f"{NEMO}/music/music-captioning/en/context/JamendoMaxCaps/train.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            caption = rec["conversations"][-1]["value"]
            if "instrumental" in caption.lower() and not VOCAL_RE.search(caption.replace("instrumental", "")):
                out.append(next(t["value"] for t in rec["conversations"] if t.get("type") == "audio"))
    return out


def background_pools():
    """{type: [(source, files, share)]}: noise uniformly over its files, music half FMA half Jamendo."""
    clotho, distant = pool_clotho(), distant_noises(-50, 1)
    n = len(clotho) + len(distant)
    return {"noise": [("clotho", clotho, len(clotho) / n), ("distant_noises", distant, len(distant) / n)],
            "music": [("fma", pool_fma(), 0.5), ("jamendo", pool_jamendo(), 0.5)],
            "silence": [("distant_noises", distant_noises(-100, -50), 1.0)]}


LEVELS = {"noise": (-50, 0), "music": (-50, 0), "silence": (-100, -50)}   # dBFS of a usable clip window


# --- audio -------------------------------------------------------------------------------------

def read(path, start=0.0, duration=None):
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        f.seek(max(int(start * sr), 0))
        x = f.read(-1 if duration is None else int(duration * sr), dtype="float32", always_2d=True).mean(axis=1)
    if sr != SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g).astype("float32")
    return x


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)


def ramp(n, k):
    """raised-cosine 0 -> 1 over the first k samples, ones after."""
    r = np.ones(n, dtype="float32")
    k = min(k, n)
    r[:k] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, k))
    return r


def tile(x, n):
    """x repeated (0.5 s crossfades) up to n samples."""
    if len(x) >= n:
        return x[:n]
    out = x.copy()
    while len(out) < n:
        k = min(int(0.5 * SR), len(out), len(x))
        r = ramp(k, k)
        out[-k:] = out[-k:] * r[::-1] + x[:k] * r
        out = np.concatenate([out, x[k:]])
    return out[:n]


def bed_under_speech(x, clip_file, start_sample, snr=None, rms=None):
    """The background AddBackgroundNoise (audiomentations, as ssak's SpeechAugment) adds under the
    speech x: the clip file from start_sample, at snr dB under the RMS of x, or at rms dBFS
    (silence: the clip's own level), parameters frozen."""
    from audiomentations import AddBackgroundNoise
    absolute = rms is not None
    t = AddBackgroundNoise(sounds_path=[clip_file], noise_rms="absolute" if absolute else "relative",
                           min_snr_db=snr or 0, max_snr_db=snr or 0,
                           min_absolute_rms_db=min(rms, -1) if absolute else -45, max_absolute_rms_db=min(rms, -1) if absolute else -45,
                           p=1.0, lru_cache_size=1)
    # the position in the file wraps around (a line longer than the file: the clip is tiled), and a
    # tail too short to start from goes back to 0: AddBackgroundNoise loops forever on an empty slice
    length = int(sf.info(clip_file).duration * SR)
    start_sample = start_sample % max(length, 1)
    if length - start_sample < int(0.5 * SR):
        start_sample = 0
    t.randomize_parameters(x, SR)
    t.parameters.update(snr_db=snr, rms_db=rms, noise_file_path=clip_file, noise_start_index=start_sample,
                        noise_end_index=start_sample + len(x))
    t.freeze_parameters()
    return t(samples=x, sample_rate=SR) - x


def mix(layout, clip, info, rng, fg_rel, snr, natural=False):
    """layout: list of ("speech", x) / ("bg", n_samples), in order. clip: the VAD-checked window
    of info["file"] from info["offset"]. natural (silence): the clip at its own level everywhere.
    Returns the mix and its levels (dBFS)."""
    total = sum(len(x) if k == "speech" else x for k, x in layout)
    speech, regions, pos = np.zeros(total, dtype="float32"), [], 0
    e = int(0.02 * SR)
    for k, x in layout:
        n = len(x) if k == "speech" else x
        if k == "speech":
            x = x * ramp(n, e) * ramp(n, e)[::-1]
            speech[pos:pos + n] = x
        regions.append((k, pos, pos + n, x if k == "speech" else None))
        pos += n
    voiced = [x for k, _, _, x in regions if k == "speech"]
    speech_db = rms_db(np.concatenate(voiced)) if voiced else None
    if natural:
        fg_db = rms_db(clip)
        if speech_db is not None and speech_db - fg_db < 15:
            raise ValueError("silence clip less than 15 dB under the speech")
    else:
        fg_db = rng.uniform(-35, -20) if speech_db is None else speech_db + fg_rel

    # foreground: the clip at fg_db where there is no speech, fading out (in) inside the
    # neighbouring speech over 0.3-1 s; the AddBackgroundNoise bed fades in (out) over the same span
    fg = tile(clip, total)
    fg = fg * 10 ** ((fg_db - rms_db(fg)) / 20)
    mask = np.zeros(total, dtype="float32")
    beds = np.zeros(total, dtype="float32")
    offset = int(info["offset"] * SR)
    edge = int(0.05 * SR)
    for i, (k, a, b, x) in enumerate(regions):
        if k == "bg":
            mask[a:b] = 1.0
            continue
        n = b - a
        prev_bg = i > 0 and regions[i - 1][0] == "bg"
        next_bg = i + 1 < len(regions) and regions[i + 1][0] == "bg"
        r_in = min(int(rng.uniform(0.3, 1.0) * SR), n // 2) if prev_bg else 0
        r_out = min(int(rng.uniform(0.3, 1.0) * SR), n // 2) if next_bg else 0
        if r_in:
            mask[a:a + r_in] = ramp(r_in, r_in)[::-1]
        if r_out:
            mask[b - r_out:b] = np.maximum(mask[b - r_out:b], ramp(r_out, r_out))
        bed = bed_under_speech(x, info["file"], offset + a, rms=fg_db) if natural else \
            bed_under_speech(x, info["file"], offset + a, snr)
        bed *= ramp(n, r_in or edge) * ramp(n, r_out or edge)[::-1]
        beds[a:b] = bed
    fg *= mask
    fg *= ramp(total, edge) * ramp(total, edge)[::-1]
    out = speech + fg + beds
    peak = np.abs(out).max()
    out = out / peak * 0.97 if peak > 0.97 else out
    return out, (None if speech_db is None else round(speech_db, 1)), round(fg_db, 1)


# --- speech ------------------------------------------------------------------------------------

def audio_turn(rec):
    return next(t for t in rec["conversations"] if t.get("type") == "audio")


def text_turn(rec):
    return next(t for t in reversed(rec["conversations"]) if t.get("from") == "Assistant")


def span(rec):
    a = audio_turn(rec)
    start = a.get("offset") or 0.0
    return a["value"], start, start + a["duration"]


def segment(rec):
    a = audio_turn(rec)
    return read(a["value"], a.get("offset") or 0.0, a["duration"])


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def speech_pool(recs, min_seg, max_seg):
    """usable segments and (prev, next) consecutive pairs of a recording."""
    segs = [r for r in recs if min_seg <= (audio_turn(r).get("duration") or 0) <= max_seg
            and len(TAG_RE.sub(" ", text_turn(r)["value"]).split()) >= 2]
    by_wav = collections.defaultdict(list)
    for r in segs:
        by_wav[span(r)[0]].append(r)
    pairs = []
    for v in by_wav.values():
        v.sort(key=lambda r: span(r)[1])
        pairs += [(p, q) for p, q in zip(v, v[1:]) if span(p)[2] <= span(q)[1] + 0.01]
    return segs, pairs


# --- workers -----------------------------------------------------------------------------------

_w = {}


def _init(pools, vad_threshold):
    import warnings
    import torch
    from silero_vad import load_silero_vad
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    _w.update(pools=pools, threshold=vad_threshold, vad=load_silero_vad())


def draw_clip(bg_type, duration, rng):
    """A clip of ~duration s (a shorter file is tiled), no speech for the VAD, not near-silent."""
    import torch
    from silero_vad import get_speech_timestamps
    sources = _w["pools"][bg_type]
    rejected = collections.Counter()
    for _ in range(50):
        source, files, _ = rng.choices(sources, weights=[s[2] for s in sources])[0]
        path = rng.choice(files)
        try:
            total = sf.info(path).duration
            start = rng.uniform(0, max(total - duration, 0))
            x = read(path, start, duration)
        except Exception:
            rejected["error"] += 1
            continue
        if len(x) < int(min(duration, 5.0) * SR):
            rejected["short"] += 1
            continue
        lo, hi = LEVELS[bg_type]
        if not lo <= rms_db(x) < hi:
            rejected["quiet" if rms_db(x) < lo else "loud"] += 1
            continue
        if get_speech_timestamps(torch.from_numpy(x), _w["vad"], threshold=_w["threshold"], sampling_rate=SR,
                                 min_speech_duration_ms=150):
            rejected["speech"] += 1
            continue
        return x, {"source": source, "file": path, "offset": round(start, 3)}, dict(rejected)
    raise RuntimeError(f"no usable {bg_type} clip in 50 draws")


def make_line(job):
    key, bg_type, kind, recs, audio_dir, p = job
    rng = random.Random(key)
    tag = f"[{bg_type}]"
    rel = rng.uniform(*p["music_db" if bg_type == "music" else "noise_db"])     # unused for silence
    snr = rng.uniform(*p["bed_snr"])
    g = int(rng.uniform(1.5, 10 if kind == "alone" else 5) * SR)
    try:
        texts = [text_turn(r)["value"] for r in recs]
        if kind == "alone":
            layout, text = [("bg", g)], tag
        elif kind == "before":
            layout, text = [("bg", g), ("speech", segment(recs[0]))], f"{tag} {texts[0]}"
        elif kind == "after":
            layout, text = [("speech", segment(recs[0])), ("bg", g)], f"{texts[0]} {tag}"
        elif kind == "around":
            g2 = int(rng.uniform(1.5, 5) * SR)
            layout, text = [("bg", g), ("speech", segment(recs[0])), ("bg", g2)], f"{tag} {texts[0]} {tag}"
        else:
            layout = [("speech", segment(recs[0])), ("bg", g), ("speech", segment(recs[1]))]
            text = f"{texts[0]} {tag} {texts[1]}"
        total = sum(len(x) if k == "speech" else x for k, x in layout)
        clip, info, rejected = draw_clip(bg_type, total / SR, rng)
        x, speech_db, fg_db = mix(layout, clip, info, rng, rel, snr, natural=bg_type == "silence")
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    out = copy.deepcopy(recs[0])
    out["id"] = f"{recs[0]['id']}_{bg_type}{kind}{key.rsplit(':', 1)[1]}"
    path = f"{audio_dir}/{out['id']}.flac"
    sf.write(path, x, SR, format="FLAC")
    a = audio_turn(out)
    a["value"], a["duration"] = path, round(len(x) / SR, 3)
    a.pop("offset", None)
    text_turn(out)["value"] = text
    out.setdefault("custom", {})["nonspeech"] = {
        "kind": kind, "type": bg_type, "speech_ids": [] if kind == "alone" else [r["id"] for r in recs],
        "fg_db": fg_db, "speech_db": speech_db, "rel_db": None if kind == "alone" else round(rel, 1),
        "bed_snr": None if kind == "alone" or bg_type == "silence" else round(snr, 1), "background": info, "rejected_clips": rejected}
    return out, None


# --- main --------------------------------------------------------------------------------------

def inputs(lang, datasets, every_line=False):
    """(dataset, silence file, speech input) for the _silence files of nemo/asr/<lang>/clean/;
    every_line: (dataset, None, context/<dataset>/train.jsonl)."""
    if every_line:
        out = [(ds, None, f"{NEMO}/asr/{lang}/context/{ds}/train.jsonl") for ds in datasets or []]
        for ds, _, src in out:
            if not os.path.exists(src):
                raise SystemExit(f"[{ds}] no {src}")
        return out
    out = []
    for sil in sorted(glob.glob(f"{NEMO}/asr/{lang}/clean/*/intermediate/*_silence.jsonl")):
        ds = sil.split("/")[-3]
        if datasets and ds not in datasets:
            continue
        name = os.path.basename(sil)[: -len("_silence.jsonl")] + ".jsonl"
        src = os.path.join(os.path.dirname(sil), name)
        if not os.path.exists(src):
            src = f"{NEMO}/asr/{lang}/context/{ds}/{name}"
        if not os.path.exists(src):
            raise SystemExit(f"[{ds}] no speech input for {sil}")
        out.append((ds, sil, src))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", required=True)
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--per-silence", type=float, help="default: 1.5, 1 in English;"
                   " new silence lines per silence line, then noise = music = all the silence lines")
    p.add_argument("--max-silence", type=int, default=0,
                   help="at most that many silence lines (real + new) per file, the real ones drawn at random "
                        "(English: 100000); noise and music follow the silence total, within their caps")
    p.add_argument("--every-line", action="store_true",
                   help="one noise + one music line per line of context/<Dataset>/train.jsonl (FLEURS), needs --datasets")
    p.add_argument("--max-passes", type=float, default=2.0, help="at most that many lines per file of the pool, per file")
    p.add_argument("--limit", type=int, default=0, help="at most N new lines per file (a test)")
    p.add_argument("--output-root", help="write under this directory instead of $DATA_FOLDER (a test)")
    p.add_argument("--min-seg", type=float, default=1.5)
    p.add_argument("--max-seg", type=float, default=12.0)
    p.add_argument("--music-db", type=float, nargs=2, default=[-12.0, 0.0], help="music RMS vs speech RMS, dB range")
    p.add_argument("--noise-db", type=float, nargs=2, default=[-15.0, -3.0])
    p.add_argument("--bed-snr", type=float, nargs=2, default=[5.0, 50.0],
                   help="speech RMS over the background under it, dB range (ssak SpeechAugment)")
    p.add_argument("--vad-threshold", type=float, default=0.3)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    per_silence = args.per_silence if args.per_silence is not None else (1.0 if args.lang == "en" else 1.5)
    if args.max_silence == 0 and args.lang == "en":
        args.max_silence = 100000
    root = os.path.abspath(args.output_root) if args.output_root else DATA
    params = {"music_db": args.music_db, "noise_db": args.noise_db, "bed_snr": args.bed_snr}

    files = inputs(args.lang, args.datasets, args.every_line)
    if not files:
        raise SystemExit("no _silence input")
    pools = background_pools()
    print("backgrounds:", {t: {s: len(f) for s, f, _ in v} for t, v in pools.items()}, flush=True)
    caps = {t: int(args.max_passes * sum(len(f) for _, f, _ in v)) for t, v in pools.items()}

    with Pool(args.workers, initializer=_init, initargs=(pools, args.vad_threshold)) as pool:
        for ds, sil, src in files:
            if args.every_line:
                sil = os.path.join(os.path.dirname(src).replace("/context/", "/clean/"), "intermediate",
                                   os.path.basename(src)[: -len(".jsonl")] + "_silence.jsonl")   # for the names only
                silence = []
                segs, pairs = speech_pool(load(src), 0.5, 1e9)
            else:
                silence = load(sil)
                segs, pairs = speech_pool(load(src), args.min_seg, args.max_seg)
            local = [r for r in segs[:50] if os.path.exists(span(r)[0])]
            if not local:
                print(f"[{ds}] {os.path.basename(src)}: audio not reachable here, skipped", flush=True)
                continue
            rng = random.Random(f"{args.seed}:{args.lang}:{ds}:{os.path.basename(sil)}")
            order, pair_order = rng.sample(segs, len(segs)), rng.sample(pairs, len(pairs))
            used = collections.Counter()

            def take(kind):
                if kind == "between" and pair_order:
                    i = used["pair"]
                    used["pair"] += 1
                    return list(pair_order[i % len(pair_order)])
                n = 2 if kind == "between" else 1
                out = [order[(used["seg"] + j) % len(order)] for j in range(n)]
                used["seg"] += n
                return out

            rel = os.path.relpath(os.path.dirname(os.path.dirname(sil)), f"{NEMO}/asr/{args.lang}/clean")
            audio_dir = f"{root}/converted_audios/transcript/{args.lang}/{rel}/nonspeech"
            os.makedirs(audio_dir, exist_ok=True)
            jobs = []
            if args.every_line:
                for bg_type in ["noise", "music"]:
                    for i, r in enumerate(segs[: args.limit or None]):
                        key = f"{args.seed}:{src}:{bg_type}:{i}"
                        jobs.append((key, bg_type, random.Random(key + ":kind").choice(EVERY_LINE_KINDS), [r], audio_dir, params))
            n_silence = min(round(per_silence * len(silence)), caps["silence"])
            if args.limit:
                n_silence = min(n_silence, args.limit)
            if args.max_silence and len(silence) + n_silence > args.max_silence:
                # real silence lines drawn at random so that real + new = max_silence (new ones kept: varied room noise)
                n_silence = min(n_silence, args.max_silence)
                kept = args.max_silence - n_silence
                print(f"[{ds}] {os.path.basename(sil)}: {kept} of the {len(silence)} silence lines kept (--max-silence)", flush=True)
                silence = random.Random(f"{args.seed}:{sil}:max_silence").sample(silence, kept)
            for bg_type in [] if args.every_line else ["silence", "noise", "music"]:
                n = n_silence if bg_type == "silence" else min(len(silence) + n_silence, caps[bg_type])
                if args.limit:
                    n = min(n, args.limit)
                for i in range(n):
                    kind = KINDS[i % len(KINDS)]
                    jobs.append((f"{args.seed}:{sil}:{bg_type}:{i}", bg_type, kind, take(kind), audio_dir, params))
            lines, errors = [], collections.Counter()
            for line, err in pool.imap_unordered(make_line, jobs, chunksize=4):
                if line is None:
                    errors[err] += 1
                else:
                    lines.append(line)
            out = sil[: -len("_silence.jsonl")] + "_nonspeech.jsonl"
            if args.output_root:
                out = os.path.join(root, os.path.relpath(out, DATA))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out + ".tmp", "w", encoding="utf-8") as f:
                for line in sorted(silence + lines, key=lambda r: r["id"]):
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
            os.replace(out + ".tmp", out)
            # the length of every mixed file, for clean_asr.py / drop_past_audio_end.py --durations
            # (else audio_bounds reads each header over NFS)
            with open(out[: -len(".jsonl")] + "_durations.tsv", "w", encoding="utf-8") as f:
                for line in lines:
                    a = audio_turn(line)
                    f.write(f"{a['value']}\t{a['duration']}\n")
            kinds =collections.Counter((l["custom"]["nonspeech"]["type"], l["custom"]["nonspeech"]["kind"]) for l in lines)
            srcs = collections.Counter(l["custom"]["nonspeech"]["background"]["source"] for l in lines)
            rej = collections.Counter()
            for l in lines:
                rej.update(l["custom"]["nonspeech"]["rejected_clips"])
            hours = sum(audio_turn(l)["duration"] for l in lines) / 3600
            print(f"[{ds}] {os.path.basename(sil)}: {len(silence)} silence + {len(lines)} new lines ({hours:.1f} h, "
                  f"{len(segs)} segments, {len(pairs)} pairs, segments used {used['seg']}, pairs used {used['pair']}) "
                  f"-> {out}\n    {dict(sorted(kinds.items()))} {dict(srcs)} clips rejected {dict(rej)} "
                  f"errors {sum(errors.values())} {errors.most_common(3)}", flush=True)


if __name__ == "__main__":
    main()
