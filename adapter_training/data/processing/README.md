# MLS-French synthetic QA pipeline

End-to-end: plan + concat MLS French segments into 10–60s clips → forced-align
with WhisperX → emit a JSONL in the shape consumed by `template_slu_variants.py`
(and `template_timestamped_transcription.py`).

The pipeline is split so alignment is decoupled from MLS specifics:

- `concat_mls.py --mode plan` builds a concat plan (cheap, no audio I/O).
- `concat_mls.py --mode apply` concatenates the audio and writes a
  **generic ASR dataset** (conversations: audio + transcript, no task prompt).
- `align_transcripts.py` takes *any* such generic ASR dataset and only adds
  forced-alignment `word2time` — it is not MLS-aware.

## Why concat

MLS ships ~258k French segments, each 10–20s. To get clips closer to 60s, we
chain consecutive-in-chapter-time segments. Segment IDs are NOT chapter-ordered
(MLS shuffled them), so the canonical ordering lives in `segments.txt`:

```
seg_id    chapter_url    start_sec    end_sec
```

Step 01 groups by `(speaker, book, chapter_url)`, sorts by `start_sec`, and
chains adjacent segments (gap ≤ `--max_gap`) until total duration ∈
`[--min_dur, --max_dur]`.

## Transcript source (punctuation matters)

Raw MLS `transcripts.txt` is lowercased with **no punctuation**, which breaks
the sentence tasks (`time2sentence`, `word2sentence`) — without `.`/`?`/`!`
boundaries `extract_sentence` returns the whole clip as one "sentence".

So `concat_mls.py` defaults to the **recased+punctuated** transcripts at
`--recasepunc_dir` (`{split}_recasepunc.jsonl`, keyed by MLS seg_id). Ordering
and timings still come from `segments.txt`; only the text source changes. Pass
`--raw_text` to fall back to the lowercase, unpunctuated `transcripts.txt`.

## Run

All outputs live under `<mls_root>/concatenated/` (override with `--concat_root`):

```
concatenated/
  plan/{split}.jsonl     # step 1: the concat plan
  audios/{split}/*.flac  # step 2: concatenated audio
  {split}.jsonl          # step 2: generic ASR dataset (audio + transcript)
```

`concat_mls.py` defaults to `--split all` (train/dev/test) and the recasepunc
transcripts, so the common case is just:

```bash
SYN=/home/abert/abert/Audio-Adapter-Training/data/synthetic
cd $SYN/mls

# 1. Build the concat plans (cheap, no audio I/O) -> concatenated/plan/{split}.jsonl
python concat_mls.py --mode plan

# 2. Concat audio -> concatenated/audios/{split}/ + concatenated/{split}.jsonl
python concat_mls.py --mode apply --resume

# 3. Forced-align with WhisperX (adds custom_metadata.word2time)
python align_transcripts.py \
    --in_jsonl <mls_root>/concatenated/train.jsonl \
    --out_jsonl out/train.jsonl \
    --device cuda --resume
```

Use `--split dev` (etc.) to do one split, `--limit N` for smoke tests, and
`--resume` to skip IDs already written. Because step 3 only needs
`audio + transcript`, you can point `align_transcripts.py` at any
conversations-shaped ASR dataset, not just MLS.

## Use with the SLU templater

The output `out/train.jsonl` is ready for `template_slu_variants.py`. Note:
- `custom_metadata.word2time` is populated → all of `word2time`,
  `word2time_first`, `time2sentence`, `time2word`, `word2sentence` work.
- `answer_spans` is NOT populated → `answer_with_source`, `answer_with_time`,
  `format_json_answer` will be skipped (they need a real Q/A).

```bash
python ../template_slu_variants.py out/  \
    --output_dir out_variants --language fr \
    --tasks word2time word2time_first time2sentence time2word word2sentence
```

(The templater iterates over `*.jsonl` in `out/`, so put `train.jsonl`,
`dev.jsonl`, `test.jsonl` there.)

For full-clip timestamped transcription, point
`template_timestamped_transcription.py` at the same directory.

## Notes

- **Alignment quality**: WhisperX uses
  `jonatasgrosman/wav2vec2-large-xlsr-53-french` by default. Override with
  `--align_model` if you prefer NeMo NFA later (would need a separate runner).
- **Audio concat**: raw sample concatenation via `soundfile`. MLS FLACs are
  16 kHz mono. The VAD-removed silences between segments are NOT restored — if
  prosody matters you'd have to go back to the original LibriVox chapter MP3
  and cut `[min(start), max(end)]` from there.
- **Gap threshold**: `--max_gap 3.0` is conservative. Bump it to chain more
  aggressively (more long clips, more chance of topic drift inside one clip).



docker run -it --rm -v /media:/media -v /home:/home --gpus all --shm-size=4g --user $(id -u):$(id -g) --name align_audios nvcr.io/nvidia/nemo:25.09
export HOME=/home/abert
docker exec -it --user root:root align_audios bash
useradd -m -u 1002 abert -s /bin/sh

