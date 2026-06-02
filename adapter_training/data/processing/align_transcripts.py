"""
scitrera/dgx-spark-pytorch-dev:2.10.0-rc6-cu131
Forced-align a generic ASR dataset (audio + transcript) with WhisperX.

Input is any JSONL ASR dataset in the conversations shape (e.g. the output of
`concat_mls.py --mode apply`, but nothing here is MLS-specific). It needs an
audio turn and an Assistant transcript; any other turns (e.g. a task prompt)
are passed through untouched:

    {
      "id": "...",
      "conversations": [
        {"from": "User", "value": <audio_path>, "type": "audio", "duration": ...},
        {"from": "Assistant", "value": <transcript>, "type": "text"},
      ],
      "custom_metadata": {...}        # optional, passed through untouched
    }

For each record this runs whisperx.align() of the transcript against the audio
file and adds `custom_metadata.word2time`, leaving everything else as-is. The
output is the shape consumed by template_slu_variants.py (and
template_timestamped_transcription.py):

    "custom_metadata": {
      "word2time": {"word": [...], "start_second": [...], "end_second": [...]},
      ...
    }

answer_spans is intentionally not produced — a plain ASR dataset has no Q/A
annotation, so only the word2time/time2word/time2sentence/word2sentence tasks
(and the timestamped-transcription generator) are targeted, none of which need it.
"""
import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
import whisperx


def get_audio_turn(conversations):
    """Return (audio_path, duration_or_None) from a conversations list."""
    for turn in conversations:
        if turn.get("type") == "audio":
            dur = turn.get("duration")
            return turn["value"], (float(dur) if dur not in (None, "") else None)
    return None, None


def get_transcript(conversations):
    """Return the assistant transcript text from a conversations list."""
    for turn in conversations:
        if turn.get("from") == "Assistant" and turn.get("type") == "text":
            return turn["value"]
    return None


def align_one(audio_np, sr, text, align_model, metadata, device):
    duration = len(audio_np) / sr
    segments = [{"text": text, "start": 0.0, "end": duration}]
    result = whisperx.align(segments, align_model, metadata, audio_np, device,
                            return_char_alignments=False)
    words = result.get("word_segments", [])
    out = {"word": [], "start_second": [], "end_second": []}
    for w in words:
        out["word"].append(w["word"])
        out["start_second"].append(round(float(w.get("start", 0.0)), 3))
        out["end_second"].append(round(float(w.get("end", w.get("start", 0.0))), 3))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_jsonl", required=True,
                    help="Generic ASR dataset JSONL (conversations: prompt + audio + transcript)")
    ap.add_argument("--out_jsonl", required=True, help="Output SLU jsonl path")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--language", default="fr", help="Language code for the align model")
    ap.add_argument("--align_model",
                    default="jonatasgrosman/wav2vec2-large-xlsr-53-french",
                    help="HF id of the wav2vec2-CTC model used by whisperx.align")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="Skip record IDs already present in --out_jsonl")
    args = ap.parse_args()

    align_model, metadata = whisperx.load_align_model(
        language_code=args.language, device=args.device, model_name=args.align_model
    )

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.resume and out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
        print(f"Resume: skipping {len(done)} already-done records")

    n_done = 0
    n_err = 0
    mode = "a" if args.resume and out_path.exists() else "w"
    with open(args.in_jsonl) as fin, out_path.open(mode) as fout:
        for line in fin:
            rec = json.loads(line)
            if rec.get("id") in done:
                continue
            conversations = rec.get("conversations", [])
            audio_path, _ = get_audio_turn(conversations)
            text = get_transcript(conversations)
            if audio_path is None or not text:
                n_err += 1
                print(f"[err] {rec.get('id')}: missing audio turn or transcript")
                continue
            try:
                audio_np, sr = sf.read(audio_path, dtype="float32", always_2d=False)
                if audio_np.ndim > 1:
                    audio_np = audio_np.mean(axis=1)
                w2t = align_one(audio_np, sr, text, align_model, metadata, args.device)
            except Exception as e:
                n_err += 1
                print(f"[err] {rec.get('id')}: {e}")
                continue

            rec.setdefault("custom_metadata", {})["word2time"] = w2t
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 50 == 0:
                print(f"[ok] aligned {n_done} (errors: {n_err})")
            if args.limit and n_done >= args.limit:
                break

    print(f"Done. records={n_done} errors={n_err} -> {out_path}")


if __name__ == "__main__":
    main()
