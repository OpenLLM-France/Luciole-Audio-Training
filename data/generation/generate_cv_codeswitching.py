"""Build CVCodeSwitching: fake code-switched audio made of 2-5 CommonVoice clips in different languages.

Each example chains 2 to 5 clips of the clean CommonVoice manifests of the 8 Luciole languages
(fr, en, de, es, it, pt, nl, ar), with at least two languages; each clip is used at most once.

    plan   no audio I/O: draws the examples and writes a JSONL plan (one example per line, its clips)
    apply  reads the plan, writes one flac per example and the NeMo manifest

How an example is drawn (plan):
  - number of clips: N_SEGMENTS (2: 35 %, 3: 30 %, 4: 20 %, 5: 15 %); total raw duration <= --max-duration;
  - each transition repeats the language (REPEAT_LANGUAGE, rarer) or switches language, at least one switch;
  - a switch keeps the speaker a third of the time (SAME_SPEAKER_SWITCH), a repeat half of the time;
  - same-speaker switches come from multilingual speakers only (CommonVoice gives a contributor the
    same client_id in every language: ~1,800 such speakers in the clean train), alternating two of
    their languages (fr-en-fr); every other speaker is a one-language one, so the scarce clips of
    the multilingual speakers only serve the same-speaker switches;
  - a new speaker after a switch goes back to a language already in the example (fr-en-fr-en) or
    brings a new one (NEW_LANGUAGE); one-language clips are drawn by language, by their number left;
  - the plan stops when the multilingual speakers are used up (keeps the 1/3 share): the clips of
    one-language speakers left are not used.

How the audio is made (apply): each clip is cut to its speech found by Silero VAD (MARGIN_START /
MARGIN_END kept): the leading / trailing silence goes, and so do the clicks of the recorder when
the contributor starts / stops it (an energy threshold would keep them: short but loud). It is
scaled to TARGET_RMS_DB (speech frames, peak capped to -1 dBFS) and gets 20 ms fades; the clips are
separated by 0.15-0.8 s pauses of very low noise. --background of the examples (default 50 %) then
get one noise clip under the whole audio, pauses included (it ties the clips together, whose own
background noise differs), at --min-snr to --max-snr dB (5-50, as build_nonspeech.py) under the
speech: the noise pools and clip draw of data/processing/clean_asr/build_nonspeech.py (Clotho
without voices + distant_noises, checked with Silero VAD). The text does not change (no [noise]:
a background under the speech, as the base files of clean_asr).

Text: the clean transcripts joined by a space. Prompt: an ASR prompt of
data/contexts/compose_asr_prompts.py for one of the languages of the example, with no constraint
(the text is cased and punctuated), that never names one language (no mention, or "the source
language"). custom_metadata.segments gives the language, speaker, source clip, text and
start / end (s) of every clip in the new audio.

Usage:
    python data/generation/generate_cv_codeswitching.py plan
    python data/generation/generate_cv_codeswitching.py apply --jobs 48
    # a few examples to listen to (+ index.tsv):
    python data/generation/generate_cv_codeswitching.py apply --sample 30 --audio-dir tmp/cvcs_samples \
        --manifest tmp/cvcs_samples/samples.jsonl
"""
import argparse
import csv
import json
import logging
import os
import random
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "data" / "contexts"))
sys.path.insert(0, str(REPO_ROOT / "data" / "processing" / "clean_asr"))
from compose_asr_prompts import DEFAULT_DIR, PromptComposer  # noqa: E402
from rules import rules as language_rules  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LANGS = ["fr", "en", "de", "es", "it", "pt", "nl", "ar"]
CV_VERSION = "cv-corpus-22.0-2025-06-20"     # the clips_wav of the clean manifests
N_SEGMENTS = {2: 0.35, 3: 0.30, 4: 0.20, 5: 0.15}
REPEAT_LANGUAGE = 0.40       # next clip in the same language
SAME_SPEAKER_SWITCH = 1 / 3  # a language switch keeps the speaker (a multilingual one)
SAME_SPEAKER_REPEAT = 0.5    # a repeated language keeps the speaker
NEW_LANGUAGE = 0.25          # a new speaker brings a language not yet in the example (else one already there)
LANGUAGE_MENTION = {"none": 0.8, "source": 0.2}   # never explicit: the audio has several languages
SR = 16000
VAD_THRESHOLD = 0.5
MARGIN_START, MARGIN_END, FRAME = 0.05, 0.1, 0.02   # kept before / after the speech found by the VAD
TARGET_RMS_DB, PEAK_DB, NOISE_DB = -22.0, -1.0, -70.0
PAUSE = (0.15, 0.8)
BACKGROUND_VAD_THRESHOLD = 0.3     # a noise clip with speech in it is not used (build_nonspeech.py default)
FADE = 0.02


def data_folder():
    return os.environ["DATA_FOLDER"]


# ----------------------------------------------------------------------------- plan

def load_speakers(lang):
    """{clip file stem: client_id} of the CommonVoice validated clips of a language."""
    path = f"{data_folder()}/raw/transcript/multilang/CommonVoice/{CV_VERSION}/{lang}/validated.tsv"
    speakers = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            speakers[os.path.splitext(os.path.basename(row["path"]))[0]] = row["client_id"]
    return speakers


def load_clips():
    """The clips of the clean manifests: lists of (lang, speaker, audio, duration, text, id)."""
    clips = []
    for lang in LANGS:
        speakers = load_speakers(lang)
        path = f"{data_folder()}/nemo/asr/{lang}/clean/CommonVoice/train.jsonl"
        missing = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                audio = next(t for t in row["conversations"] if t["type"] == "audio")
                text = next(t for t in row["conversations"] if t["from"] == "Assistant")["value"].strip()
                speaker = speakers.get(os.path.splitext(os.path.basename(audio["value"]))[0])
                if speaker is None or not text:
                    missing += 1
                    continue
                clips.append((lang, speaker, audio["value"], float(audio["duration"]), text, row["id"]))
        logger.info(f"{lang}: {sum(c[0] == lang for c in clips)} clips ({missing} without speaker or text)")
    return clips


class Pools:
    """The unused clips: by (speaker, language), and per language those of the one-language speakers."""

    def __init__(self, clips, rng, max_per_speaker=None):
        """max_per_speaker: at most that many clips of a speaker in a language (the others are not
        used): a few CommonVoice contributors recorded tens of thousands of clips."""
        self.clips = clips
        self.used = np.ones(len(clips), dtype=bool)       # capped-out clips stay "used"
        by_speaker = defaultdict(list)
        for i, (lang, speaker, *_) in enumerate(clips):
            by_speaker[(speaker, lang)].append(i)
        self.langs_of = defaultdict(set)
        for speaker, lang in by_speaker:
            self.langs_of[speaker].add(lang)
        for key, ids in by_speaker.items():
            rng.shuffle(ids)
            del ids[max_per_speaker or len(ids):]
            self.used[ids] = False
        self.by_speaker = by_speaker
        self.count = Counter({key: len(ids) for key, ids in by_speaker.items()})
        self.multi = [s for s, langs in self.langs_of.items() if len(langs) > 1]   # still usable ones
        self.mono_order = {lang: [] for lang in LANGS}
        for i, (lang, speaker, *_) in enumerate(clips):
            if not self.used[i] and not self.is_multi(speaker):
                self.mono_order[lang].append(i)
        for ids in self.mono_order.values():
            rng.shuffle(ids)
        self.mono_left = Counter({lang: len(ids) for lang, ids in self.mono_order.items()})

    def is_multi(self, speaker):
        return len(self.langs_of[speaker]) > 1

    def take(self, i):
        lang, speaker = self.clips[i][:2]
        self.used[i] = True
        self.count[(speaker, lang)] -= 1
        if not self.is_multi(speaker):
            self.mono_left[lang] -= 1
        return i

    def give_back(self, i):
        lang, speaker = self.clips[i][:2]
        self.used[i] = False
        self.count[(speaker, lang)] += 1
        self.by_speaker[(speaker, lang)].append(i)
        if not self.is_multi(speaker):
            self.mono_left[lang] += 1
            self.mono_order[lang].append(i)

    def of_speaker(self, speaker, lang):
        """An unused clip of this speaker in this language, or None."""
        ids = self.by_speaker.get((speaker, lang))
        while ids:
            i = ids.pop()
            if not self.used[i]:
                return self.take(i)
        return None

    def mono_clip(self, lang, exclude):
        """An unused clip of a one-language speaker not in `exclude`, or None."""
        ids, skipped, clip = self.mono_order[lang], [], None
        while ids:
            i = ids.pop()
            if self.used[i]:
                continue
            if self.clips[i][1] in exclude:
                skipped.append(i)
                if len(skipped) > 50:
                    break
                continue
            clip = self.take(i)
            break
        ids.extend(reversed(skipped))
        return clip

    def mono_language(self, rng, exclude=(), among=None):
        """A language with one-language clips left, by their number, or None."""
        langs = [lang for lang in (among or LANGS) if self.mono_left[lang] > 0 and lang not in exclude]
        if not langs:
            return None
        return rng.choices(langs, weights=[self.mono_left[lang] for lang in langs])[0]

    def multi_speaker(self, rng, pattern, entry, prev_lang, exclude, tries=40):
        """(speaker, [language per clip]) of a multilingual speaker able to say a run: `pattern` is
        its inner transitions ("S" switch, "R" repeat), `entry` the transition into it ("S", "R" or
        None for the first run). The run alternates two languages L1 / L2 of the speaker."""
        for _ in range(tries):
            if not self.multi:
                return None
            k = rng.randrange(len(self.multi))
            speaker = self.multi[k]
            langs = [lang for lang in self.langs_of[speaker] if self.count[(speaker, lang)] > 0]
            if len(langs) < 2:            # used up: no more switch possible for this speaker
                self.multi[k] = self.multi[-1]
                self.multi.pop()
                continue
            if speaker in exclude:
                continue
            pairs = [(a, b) for a in langs for b in langs if a != b
                     and (entry is None or (entry == "R") == (a == prev_lang))]
            rng.shuffle(pairs)
            for a, b in pairs:
                seq, cur = [a], a
                for t in pattern:
                    if t == "S":
                        cur = b if cur == a else a
                    seq.append(cur)
                need = Counter(seq)
                if all(self.count[(speaker, lang)] >= n for lang, n in need.items()):
                    return speaker, seq
        return None


def draw_transitions(rng):
    """The transitions of an example: "Ss" / "Sn" language switch with the same / a new speaker,
    "Rs" / "Rn" same language with the same / a new speaker; at least one switch."""
    n = rng.choices(list(N_SEGMENTS), weights=list(N_SEGMENTS.values()))[0]
    while True:
        trans = []
        for _ in range(n - 1):
            if rng.random() < REPEAT_LANGUAGE:
                trans.append("R" + ("s" if rng.random() < SAME_SPEAKER_REPEAT else "n"))
            else:
                trans.append("S" + ("s" if rng.random() < SAME_SPEAKER_SWITCH else "n"))
        if any(t[0] == "S" for t in trans):
            return trans


def draw_example(pools, rng, max_duration):
    """(clip ids, None) of one example, or (None, reason): "multi" when no multilingual speaker
    can say a same-speaker run any more, "mono" when the one-language clips do not suffice, "long"."""
    trans = draw_transitions(rng)
    # runs of one speaker: cut at the transitions to a new speaker
    runs, current = [], []
    for t in trans:
        current.append(t[0])
        if t[1] == "n":
            runs.append((current[:-1], t[0]))
            current = []
    runs.append((current, None))
    # runs[i] = (inner transitions, transition into the next run)
    ids, speakers, langs, entry = [], set(), [], None
    reason = None
    for inner, out in runs:
        prev = langs[-1] if langs else None
        if "S" in inner:
            found = pools.multi_speaker(rng, inner, entry, prev, speakers)
            if found is None:
                reason = "multi"
                break
            speaker, seq = found
            for lang in seq:
                ids.append(pools.of_speaker(speaker, lang))
                langs.append(lang)
            speakers.add(speaker)
        else:
            if entry is None:
                lang = pools.mono_language(rng)
            elif entry == "R":
                lang = prev if pools.mono_left[prev] > 0 else None
            else:
                there = [l for l in set(langs) if l != prev]
                lang = None
                if there and rng.random() >= NEW_LANGUAGE:
                    lang = pools.mono_language(rng, among=there)
                lang = lang or pools.mono_language(rng, exclude=set(langs)) or pools.mono_language(rng, exclude={prev})
            clip = pools.mono_clip(lang, speakers) if lang else None
            if clip is None:
                reason = "mono"
                break
            ids.append(clip)
            langs.append(lang)
            speaker = pools.clips[clip][1]
            speakers.add(speaker)
            for _ in inner:                       # "R" only: same speaker, same language
                clip = pools.of_speaker(speaker, lang) or pools.mono_clip(lang, speakers)
                if clip is None:
                    reason = "mono"
                    break
                ids.append(clip)
                langs.append(lang)
                speakers.add(pools.clips[clip][1])
            if reason:
                break
        entry = out
    if reason is None and sum(pools.clips[i][3] for i in ids) > max_duration:
        reason = "long"
    if reason:
        for i in ids:
            pools.give_back(i)
        return None, reason
    return ids, None


def plan(args):
    rng = random.Random(args.seed)
    clips = load_clips()
    pools = Pools(clips, rng, args.max_per_speaker)
    logger.info(f"{len(pools.multi)} multilingual speakers, one-language clips {dict(pools.mono_left)}")
    stats, fails, examples = Counter(), Counter(), 0
    os.makedirs(os.path.dirname(os.path.abspath(args.plan)), exist_ok=True)
    with open(args.plan, "w", encoding="utf-8") as f:
        # stop when the multilingual speakers are used up: the same-speaker share would drop
        while pools.multi and fails["multi_in_a_row"] < 500 and fails["mono_in_a_row"] < 2000:
            ids, reason = draw_example(pools, rng, args.max_duration)
            if ids is None:
                fails[reason] += 1
                fails[f"{reason}_in_a_row"] += 1
                continue
            segs = [clips[i] for i in ids]
            fails["mono_in_a_row"] = 0
            if any(a[0] != b[0] and a[1] == b[1] for a, b in zip(segs, segs[1:])):
                fails["multi_in_a_row"] = 0       # only a same-speaker example shows they are not used up
            for a, b in zip(segs, segs[1:]):
                kind = ("switch" if a[0] != b[0] else "repeat") + ("_same_speaker" if a[1] == b[1] else "_new_speaker")
                stats[kind] += 1
            stats[f"n={len(ids)}"] += 1
            stats[f"langs={len({s[0] for s in segs})}"] += 1
            line = {"id": f"cvcs_{examples:07d}",
                    "segments": [{"lang": s[0], "speaker": s[1], "audio": s[2], "duration": s[3], "text": s[4],
                                  "source_id": s[5]} for s in segs]}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            examples += 1
            if examples % 50000 == 0:
                logger.info(f"{examples} examples, multilingual speakers left {len(pools.multi)}")
    switches = stats["switch_same_speaker"] + stats["switch_new_speaker"]
    logger.info(f"{examples} examples -> {args.plan}; failed draws {dict(fails)}")
    logger.info(f"one-language clips left unused: {dict(pools.mono_left)}")
    logger.info(f"language switches: {switches}, same speaker {stats['switch_same_speaker'] / max(switches, 1):.1%}; "
                f"repeats: same speaker {stats['repeat_same_speaker']}, new {stats['repeat_new_speaker']}")
    logger.info("segments: " + ", ".join(f"{k} {v}" for k, v in sorted(stats.items()) if k.startswith("n=")))
    logger.info("languages: " + ", ".join(f"{k} {v}" for k, v in sorted(stats.items()) if k.startswith("langs=")))


# ----------------------------------------------------------------------------- apply

def db(x):
    return 10 ** (x / 20)


_vad = {}


def init_worker(noise_pools=None):
    """The VAD, and the background noise pools of build_nonspeech.py (its worker state)."""
    if noise_pools:
        import build_nonspeech
        build_nonspeech._init(noise_pools, BACKGROUND_VAD_THRESHOLD)
        _vad["model"] = build_nonspeech._w["vad"]
        return
    import torch
    from silero_vad import load_silero_vad
    torch.set_num_threads(1)
    _vad["model"] = load_silero_vad()


def noise_pools():
    """The noise backgrounds of build_nonspeech.py: Clotho (captions without voices) and the
    distant_noises files of -50 dBFS and over, uniformly over the files."""
    import build_nonspeech as bn
    clotho, distant = bn.pool_clotho(), bn.distant_noises(-50, 1)
    n = len(clotho) + len(distant)
    return {"noise": [("clotho", clotho, len(clotho) / n), ("distant_noises", distant, len(distant) / n)]}


def add_background(out, speech_rms, snr, seed):
    """`out` with one noise clip under all of it (pauses too) at `snr` dB under the speech RMS."""
    import build_nonspeech as bn
    clip, info, _ = bn.draw_clip("noise", len(out) / SR, random.Random(seed))
    clip = bn.tile(clip, len(out))
    clip = clip * (speech_rms / (np.sqrt(np.mean(clip ** 2)) + 1e-9) / db(snr))
    fade = min(int(FADE * SR), len(clip) // 2)
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    clip[:fade] *= ramp
    clip[-fade:] *= ramp[::-1]
    out = out + clip.astype(np.float32)
    peak = np.abs(out).max()
    if peak > db(PEAK_DB):
        out = out * (db(PEAK_DB) / peak)
    return out, {**info, "snr_db": round(snr, 1)}


def speech_bounds(audio):
    """(start, end) samples of the speech (Silero VAD), None when it finds none. A click of the
    recorder at the start / end of a CommonVoice clip is no speech: it is left out."""
    import torch
    from silero_vad import get_speech_timestamps
    if "model" not in _vad:
        init_worker()
    speech = get_speech_timestamps(torch.from_numpy(audio), _vad["model"], threshold=VAD_THRESHOLD,
                                   sampling_rate=SR, min_speech_duration_ms=100, speech_pad_ms=0)
    if not speech:
        return None
    return speech[0]["start"], speech[-1]["end"]


def trim_and_level(audio):
    """The speech of the clip (VAD, MARGIN_START / MARGIN_END kept), at TARGET_RMS_DB, with fades."""
    bounds = speech_bounds(audio)
    if bounds is None:
        raise ValueError("no speech found")
    start = max(0, bounds[0] - int(MARGIN_START * SR))
    end = min(len(audio), bounds[1] + int(MARGIN_END * SR))
    audio = audio[start:end].astype(np.float32)
    frame = int(FRAME * SR)
    n = len(audio) // frame
    energy = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12) if n else np.array([1e-6])
    level = 20 * np.log10(energy)
    speech = energy[level > level.max() - 30]          # the RMS of the speech frames, not of the pauses
    audio = audio * (db(TARGET_RMS_DB) / max(np.sqrt(np.mean(speech ** 2)), 1e-6))
    peak = np.abs(audio).max()
    if peak > db(PEAK_DB):
        audio = audio * (db(PEAK_DB) / peak)
    fade = min(int(FADE * SR), len(audio) // 2)
    if fade:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        audio[:fade] *= ramp
        audio[-fade:] *= ramp[::-1]
    return audio


def build(task):
    """(manifest line, None) or (None, error) for one plan line."""
    line, out_path, fmt, seed, snr = task
    rng = np.random.default_rng(seed)
    try:
        parts, segments, t = [], [], 0.0
        for k, seg in enumerate(line["segments"]):
            audio, sr = sf.read(seg["audio"], dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SR:
                raise ValueError(f"{seg['audio']}: {sr} Hz")
            if k:
                pause = rng.normal(0, db(NOISE_DB), int(rng.uniform(*PAUSE) * SR)).astype(np.float32)
                parts.append(pause)
                t += len(pause) / SR
            audio = trim_and_level(audio)
            parts.append(audio)
            segments.append({"lang": seg["lang"], "speaker": seg["speaker"][:16], "source_id": seg["source_id"],
                             "source_audio": seg["audio"], "start": round(t, 3), "end": round(t + len(audio) / SR, 3),
                             "text": seg["text"]})
            t += len(audio) / SR
        out = np.concatenate(parts)
        background = None
        if snr is not None:
            speech = np.concatenate(parts[::2])
            out, background = add_background(out, np.sqrt(np.mean(speech ** 2)), snr, seed)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sf.write(out_path, out, SR, format=fmt.upper(), subtype="PCM_16")
        return {"id": line["id"], "audio": str(out_path), "duration": round(len(out) / SR, 3),
                "text": " ".join(s["text"] for s in segments), "segments": segments, "background": background}, None
    except Exception as e:  # noqa: BLE001 - one bad clip must not stop the run
        return None, f"{line['id']}: {e}"


class Prompts:
    """A composer per language (hesitations from conf/rules.yaml of clean_asr)."""

    def __init__(self):
        self.composers = {}

    def compose(self, rng, lang):
        if lang not in self.composers:
            hesitations = list(language_rules(lang).get("disfluencies", {}).get("hesitations") or {})
            self.composers[lang] = PromptComposer(DEFAULT_DIR, lang, hesitations, language_mention=LANGUAGE_MENTION)
        composer = self.composers[lang]
        weights = composer.instruction_languages
        instruction = rng.choices(list(weights), weights=list(weights.values()))[0]
        return composer.compose(rng, instruction_language=instruction, mention="source" if instruction != lang else None)


def apply(args):
    with open(args.plan, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    rng = random.Random(args.seed)
    if args.sample:
        lines = rng.sample(lines, min(args.sample, len(lines)))
    audio_dir = Path(args.audio_dir)
    tasks = [(line, audio_dir / line["id"][-7:-4] / f"{line['id']}.{args.format}", args.format, args.seed + i,
              rng.uniform(args.min_snr, args.max_snr) if rng.random() < args.background else None)
             for i, line in enumerate(lines)]
    pools = noise_pools() if args.background > 0 else None
    prompts = Prompts()
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
    errors, done, hours = 0, 0, 0.0
    with Pool(args.jobs, initializer=init_worker, initargs=(pools,)) as pool, open(args.manifest, "w", encoding="utf-8") as f:
        for result, error in pool.imap(build, tasks, chunksize=64):
            if error:
                errors += 1
                if errors <= 20:
                    logger.warning(error)
                continue
            langs = [s["lang"] for s in result["segments"]]
            row = {
                "id": result["id"],
                "dataset_name": "CVCodeSwitching",
                "conversations": [
                    {"from": "User", "value": prompts.compose(rng, rng.choice(langs)), "type": "text"},
                    {"from": "User", "value": result["audio"], "type": "audio", "duration": result["duration"]},
                    {"from": "Assistant", "value": result["text"], "type": "text"},
                ],
                "custom_metadata": {"languages": langs, "segments": result["segments"],
                                    **({"background": result["background"]} if result["background"] else {})},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            done += 1
            hours += result["duration"] / 3600
            if done % 50000 == 0:
                logger.info(f"{done}/{len(tasks)} examples, {hours:.0f} h")
    logger.info(f"{done} examples, {hours:.1f} h -> {args.manifest} ({errors} errors)")
    if args.sample:
        with open(audio_dir / "index.tsv", "w", encoding="utf-8") as idx, open(args.manifest, encoding="utf-8") as f:
            idx.write("audio\tduration\tlanguages\tspeakers\tbackground\tprompt\ttext\n")
            for line in f:
                row = json.loads(line)
                segs = row["custom_metadata"]["segments"]
                speakers = "-".join(chr(65 + sorted({s["speaker"] for s in segs}, key=[s["speaker"] for s in segs].index)
                                        .index(s["speaker"])) for s in segs)
                idx.write("\t".join([row["conversations"][1]["value"], str(row["conversations"][1]["duration"]),
                                     "-".join(s["lang"] for s in segs), speakers,
                                     (lambda b: f"{b['source']} {b['snr_db']} dB" if b else "-")(
                                         row["custom_metadata"].get("background")),
                                     row["conversations"][0]["value"].replace("\n", " / "),
                                     row["conversations"][2]["value"]]) + "\n")
        logger.info(f"index: {audio_dir / 'index.tsv'}")


def main():
    default_root = "$DATA_FOLDER"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["plan", "apply"])
    p.add_argument("--plan", default=None,
                   help=f"plan JSONL (default: {default_root}/nemo/misc/codeswitching/CVCodeSwitching/plan.jsonl)")
    p.add_argument("--manifest", default=None,
                   help=f"output manifest (default: {default_root}/nemo/misc/codeswitching/CVCodeSwitching/train.jsonl)")
    p.add_argument("--audio-dir", default=None,
                   help=f"output audio (default: {default_root}/converted_audios/transcript/multilang/CVCodeSwitching)")
    p.add_argument("--format", default="flac", choices=["flac", "wav"])
    p.add_argument("--max-duration", type=float, default=40.0, help="max raw duration of an example (s)")
    p.add_argument("--max-per-speaker", type=int, default=500,
                   help="plan: at most N clips of a speaker in a language (0: no limit)")
    p.add_argument("--background", type=float, default=0.5,
                   help="apply: share of the examples with one noise background under all the audio")
    p.add_argument("--min-snr", type=float, default=5.0, help="apply: background SNR range (dB under the speech)")
    p.add_argument("--max-snr", type=float, default=50.0)
    p.add_argument("--sample", type=int, default=0, help="apply: only N random examples of the plan, plus an index.tsv")
    p.add_argument("--jobs", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = f"{data_folder()}/nemo/misc/codeswitching/CVCodeSwitching"
    args.plan = args.plan or f"{out}/plan.jsonl"
    args.manifest = args.manifest or f"{out}/train.jsonl"
    args.audio_dir = args.audio_dir or f"{data_folder()}/converted_audios/transcript/multilang/CVCodeSwitching"
    plan(args) if args.mode == "plan" else apply(args)


if __name__ == "__main__":
    main()
