"""Forced-align a generic ASR dataset (audio + transcript) with NeMo Forced
Aligner (NFA) — an alternative backend to align_transcripts.py (WhisperX).

Preferred on NVIDIA arm64 / Blackwell (e.g. DGX Spark), where WhisperX's
torchaudio / ctranslate2 dependencies are currently painful: NFA runs inside the
first-party NeMo NGC container and uses a NeMo CTC (or hybrid-CTC) ASR model.

Input is the conversations-shape ASR dataset produced by `concat_mls.py
--mode apply` (audio turn + Assistant transcript). Work is done in chunks; per
chunk this:

  1. writes a temp NeMo manifest ({audio_filepath, text}) for the chunk,
  2. shells out to NFA's align.py -> per-utterance word-level CTM files,
  3. parses the word CTMs into `custom_metadata.word2time` and appends them,

producing the same output shape as align_transcripts.py (consumed by
template_slu_variants.py / template_timestamped_transcription.py).

By default it resumes: records already present in the output are skipped, so a
crash only loses (at most) the in-flight chunk. Pass --force to rebuild from
scratch. Chunking is what makes resume useful — without it a crash mid-NFA
would redo the whole split.

NFA names each CTM by the audio file stem, and our clips are `<id>.flac`, so the
CTM for record `id` is `<output>/ctm/words/<id>.ctm`.

Use a punctuation-aware model (default below) to match the recased+punctuated
transcripts from concat_mls.py.
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from tqdm import tqdm


SPLITS = ["train", "dev", "test"]


def get_audio_turn(conversations):
    """Return the audio path from a conversations list (or None)."""
    for turn in conversations:
        if turn.get("type") == "audio":
            return turn.get("value")
    return None


def get_transcript(conversations):
    """Return the Assistant transcript text from a conversations list (or None)."""
    for turn in conversations:
        if turn.get("from") == "Assistant" and turn.get("type") == "text":
            return turn.get("value")
    return None


def get_duration(conversations):
    """Return the audio turn's duration (or 0.0 if absent) for length sorting."""
    for turn in conversations:
        if turn.get("type") == "audio":
            try:
                return float(turn.get("duration") or 0.0)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def truncate_partial_tail(path: Path):
    """Drop an incomplete trailing line (interrupted mid-write) so an appended
    resume stays valid JSONL. Complete records always end in '\\n' (we flush
    each line), so a missing final newline means the last line is torn."""
    with open(path, "rb+") as f:
        end = f.seek(0, os.SEEK_END)
        if end == 0:
            return
        f.seek(end - 1)
        if f.read(1) == b"\n":
            return
        nl, pos, buf = -1, end, b""
        while pos > 0:
            step = min(8192, pos)
            pos -= step
            f.seek(pos)
            buf = f.read(step) + buf
            idx = buf.rfind(b"\n")
            if idx != -1:
                nl = pos + idx
                break
        f.truncate(nl + 1 if nl != -1 else 0)
        print(f"Resume: trimmed an incomplete trailing line from {path}")


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def parse_word_ctm(ctm_path: Path):
    """Parse a NFA word-level CTM into a word2time dict.

    CTM line: `<utt_id> <channel> <start_s> <duration_s> <word> [<conf>]`.
    """
    out = {"word": [], "start_second": [], "end_second": []}
    with open(ctm_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            start = float(parts[2])
            dur = float(parts[3])
            word = parts[4]
            out["word"].append(word)
            out["start_second"].append(round(start, 3))
            out["end_second"].append(round(start + dur, 3))
    return out


def write_manifest(chunk, manifest_path: Path):
    """Write a NeMo manifest for the chunk; return the number of utterances."""
    n = 0
    with manifest_path.open("w") as mf:
        for rec in chunk:
            audio = get_audio_turn(rec.get("conversations", []))
            text = get_transcript(rec.get("conversations", []))
            if not audio or not text:
                continue
            mf.write(json.dumps({"audio_filepath": audio, "text": text},
                                ensure_ascii=False) + "\n")
            n += 1
    return n


def run_nfa(args, manifest_path: Path, nfa_out: Path, on_progress=None):
    """Invoke NFA's align.py on a manifest, writing word/segment/token CTMs.

    NeMo's (noisy) stdout/stderr is redirected to <nfa_out>/nfa.log so the
    console stays clean. While it runs, `on_progress(n)` is called with the
    number of word CTMs produced so far (drives the live progress bar).
    """
    nfa_out.mkdir(parents=True, exist_ok=True)
    words_dir = nfa_out / "ctm" / "words"
    log_path = nfa_out / "nfa.log"
    cmd = [
        "python", args.nfa_script,
        f"pretrained_name={args.model}",
        f"manifest_filepath={manifest_path}",
        f"output_dir={nfa_out}",
        f"batch_size={args.batch_size}",
        "save_output_file_formats=[ctm]",
    ]
    env = os.environ.copy()
    # Containers run with a host uid not in /etc/passwd make getpass.getuser()
    # (used by torch's inductor cache dir) raise KeyError; USER short-circuits it.
    env.setdefault("USER", "nemo")
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        while proc.poll() is None:
            if on_progress and words_dir.exists():
                try:
                    on_progress(sum(1 for _ in words_dir.glob("*.ctm")))
                except OSError:
                    pass
            time.sleep(1.0)
    if proc.returncode != 0:
        tail = ""
        try:
            with open(log_path) as f:
                tail = "".join(f.readlines()[-25:])
        except OSError:
            pass
        raise RuntimeError(
            f"NFA failed (rc={proc.returncode}) for {manifest_path}.\n"
            f"--- nfa.log tail ---\n{tail}")


def align_split(args, in_path: Path, out_path: Path):
    if not in_path.exists():
        print(f"[skip] no input for split: {in_path}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Default = resume; --force rebuilds from scratch.
    fresh = args.force or not out_path.exists()
    done = set()
    if not fresh:
        truncate_partial_tail(out_path)
        with out_path.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
        print(f"Resume: {len(done)} records already done in {out_path}")

    records = []
    with open(in_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("id") in done:
                continue
            records.append(rec)
            if args.limit and len(records) >= args.limit:
                break

    if not records:
        print(f"Nothing to do for {out_path}")
        return

    # Sort longest-first so each batch holds similar-duration clips (minimal
    # padding -> faster), and any OOM shows up on the very first chunk.
    records.sort(key=lambda r: get_duration(r.get("conversations", [])), reverse=True)

    mode = "w" if fresh else "a"
    tag = out_path.stem
    n_done = 0
    n_err = 0
    total = len(records)
    n_chunks = (total + args.chunk_size - 1) // args.chunk_size

    # (#1) Per-split header: confirms config + resume math at a glance.
    resume_note = "" if fresh else f" (resume: {len(done)} already done)"
    print(f"[{tag}] {total} clips to do{resume_note} -> {out_path} | "
          f"model={args.model} batch={args.batch_size} chunk={args.chunk_size} "
          f"({n_chunks} chunks)", flush=True)

    bar = tqdm(total=total, desc=f"align {tag}", unit="clip")
    with out_path.open(mode) as fout:
        for c, chunk in enumerate(chunked(records, args.chunk_size), start=1):
            bar.set_postfix(chunk=f"{c}/{n_chunks}", err=n_err)
            work = Path(tempfile.mkdtemp(prefix="nfa_"))
            try:
                manifest_path = work / "manifest.jsonl"
                if write_manifest(chunk, manifest_path) == 0:
                    n_err += len(chunk)
                    bar.n = n_done + n_err
                    bar.refresh()
                    continue
                nfa_out = work / "nfa_out"
                # Live-advance the bar from CTMs produced during this chunk's NFA run.
                base = n_done + n_err
                # (#3) A failing chunk is logged and skipped, not fatal: keep its
                # nfa.log next to the output and carry on so a single NFA hiccup
                # doesn't lose a multi-hour run (resume retries it later).
                try:
                    run_nfa(args, manifest_path, nfa_out,
                            on_progress=lambda k, base=base: (setattr(bar, "n", min(base + k, total)), bar.refresh()))
                except Exception as e:
                    saved = out_path.parent / f"{tag}_chunk{c}_fail.log"
                    src = nfa_out / "nfa.log"
                    if src.exists():
                        shutil.copy(src, saved)
                    n_err += len(chunk)
                    bar.n = n_done + n_err
                    bar.refresh()
                    tqdm.write(f"[err] {tag} chunk {c}/{n_chunks} failed ({len(chunk)} clips "
                               f"skipped): {e}  | log: {saved}")
                    continue
                words_dir = nfa_out / "ctm" / "words"
                for rec in chunk:
                    audio = get_audio_turn(rec.get("conversations", []))
                    ctm = words_dir / f"{Path(audio).stem}.ctm" if audio else None
                    if ctm is None or not ctm.exists():
                        n_err += 1
                        tqdm.write(f"[err] {rec.get('id')}: no CTM at {ctm}")
                        continue
                    rec.setdefault("custom_metadata", {})["word2time"] = parse_word_ctm(ctm)
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    n_done += 1
                # Authoritative position after merge.
                bar.n = n_done + n_err
                bar.set_postfix(chunk=f"{c}/{n_chunks}", err=n_err)
                bar.refresh()
            finally:
                shutil.rmtree(work, ignore_errors=True)
    bar.close()
    print(f"Done. records={n_done} errors={n_err} -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=True,
                    help="Folder containing {split}.jsonl ASR datasets to align.")
    ap.add_argument("--output_dir", default=None,
                    help="Where to write the aligned {split}.jsonl "
                         "(default: <input_dir>/aligned).")
    ap.add_argument("--split", default="all", choices=SPLITS + ["all"])
    ap.add_argument("--model", default="linagora/linto_stt_fr_fastconformer_pc",
                    help="NeMo ASR model (CTC or hybrid-CTC). Use a *_pc model to align "
                         "the recased+punctuated transcripts from concat_mls.py.")
    ap.add_argument("--nfa_script", default="/opt/NeMo/tools/nemo_forced_aligner/align.py",
                    help="Path to NeMo Forced Aligner align.py (in the NeMo NGC container).")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--chunk_size", type=int, default=3200,
                    help="Records per NFA invocation. Smaller = finer resume granularity "
                         "(less lost on a crash); larger = less per-chunk overhead.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="Rebuild from scratch, ignoring/overwriting any existing output "
                         "(default: resume, skipping IDs already present).")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "aligned"
    splits = SPLITS if args.split == "all" else [args.split]

    for split in splits:
        align_split(
            args,
            in_path=input_dir / f"{split}.jsonl",
            out_path=output_dir / f"{split}.jsonl",
        )


if __name__ == "__main__":
    main()
