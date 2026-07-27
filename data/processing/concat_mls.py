"""Plan and apply concatenation of MLS French segments into 10–60s clips.

Two modes (``--mode``):

  plan   Group segments by chapter, sort by start_sec, chain "adjacent" runs
         (gap < --max_gap) into clips whose total duration falls in
         [--min_dur, --max_dur], and write a JSONL plan. Cheap, no audio I/O.

  apply  Read a plan, concatenate the per-segment FLACs into one file per clip,
         and write a *generic ASR dataset* (audio + transcript) in the
         conversations shape consumed downstream:

             {
               "id": "...",
               "conversations": [
                 {"from": "User", "value": <audio_path>, "type": "audio", "duration": ...},
                 {"from": "Assistant", "value": <transcript>, "type": "text"},
               ],
               "custom_metadata": {"source": {...}}
             }

         No task prompt is emitted here — that is a templating concern. Forced
         alignment (word2time) is added separately by align_transcripts.py.

MLS segment IDs (`{speaker}_{book}_{seg}`) are NOT in chapter time order: the
upstream pipeline shuffled segments across chapters. The authoritative ordering
is in `segments.txt` (seg_id, chapter_url, start_sec, end_sec). The plan mode
relies on this to chain contextually adjacent segments.
"""
import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm


SPLITS = ["train", "dev", "test"]


def parse_segments(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            seg_id, url, start, end = parts
            rows.append((seg_id, url, float(start), float(end)))
    return rows


def parse_transcripts(path):
    out = {}
    with open(path) as f:
        for line in f:
            seg_id, _, text = line.rstrip("\n").partition("\t")
            out[seg_id] = text
    return out


def parse_recasepunc(path):
    """Map seg_id -> recased+punctuated transcript from a NeMo-style jsonl
    (the Assistant text turn). seg_ids match MLS `segments.txt`."""
    out = {}
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            seg_id = rec.get("id")
            if not seg_id:
                continue
            for turn in rec.get("conversations", []):
                if turn.get("from") == "Assistant" and turn.get("type") == "text":
                    if turn.get("value"):
                        out[seg_id] = turn["value"]
                    break
    return out


def load_seg_texts(args, split, split_root: Path):
    """Choose the per-segment transcript source: recased+punctuated manifest
    (default, needed for the sentence tasks) or raw MLS `transcripts.txt`."""
    if not args.raw_text and args.recasepunc_dir:
        rp_path = Path(args.recasepunc_dir) / f"{split}_recasepunc.jsonl"
        if rp_path.exists():
            print(f"[{split}] transcripts: recasepunc {rp_path}")
            return parse_recasepunc(rp_path)
        print(f"[{split}] recasepunc not found ({rp_path}); falling back to transcripts.txt")
    print(f"[{split}] transcripts: raw {split_root / 'transcripts.txt'}")
    return parse_transcripts(split_root / "transcripts.txt")


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
            return  # ends cleanly, nothing to trim
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


def seg_to_flac(audio_root: Path, seg_id: str) -> Path:
    speaker, book, _ = seg_id.split("_")
    return audio_root / speaker / book / f"{seg_id}.flac"


def concat_flacs(paths, out_path: Path) -> tuple[float, int]:
    """Concat the FLACs by raw sample concat. MLS FLACs are 16 kHz mono.

    MLS segments tile their chapter contiguously (segment N ends where N+1
    begins), so a chain reproduces a continuous slice of the source audio with
    no seam — no silence needs to be inserted between segments.
    """
    chunks = []
    sr = None
    for p in paths:
        data, this_sr = sf.read(p, dtype="float32", always_2d=False)
        if sr is None:
            sr = this_sr
        elif this_sr != sr:
            raise ValueError(f"sr mismatch: {p} has {this_sr}, expected {sr}")
        if data.ndim > 1:
            data = data.mean(axis=1)
        chunks.append(data)
    audio = np.concatenate(chunks)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, audio, sr, format="FLAC")
    return len(audio) / sr, sr


# ---------------------------------------------------------------------------
# Mode: plan
# ---------------------------------------------------------------------------
def run_plan(args, split, out_path: Path):
    random.seed(args.seed)
    split_root = Path(args.mls_root) / split
    segs = parse_segments(split_root / "segments.txt")
    texts = load_seg_texts(args, split, split_root)
    audio_root = split_root / "audio"

    by_chapter = defaultdict(list)
    for seg_id, url, start, end in segs:
        speaker, book, _ = seg_id.split("_")
        # Group by (speaker, book, chapter_url) — same reader+book+chapter file.
        by_chapter[(speaker, book, url)].append((start, end, seg_id))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_plans = 0
    with out_path.open("w") as fout:
        for (speaker, book, url), items in by_chapter.items():
            items.sort()
            # Walk adjacent runs.
            i = 0
            while i < len(items):
                chain = [items[i]]
                chain_dur = items[i][1] - items[i][0]
                j = i + 1
                while j < len(items):
                    prev_end = chain[-1][1]
                    nxt_start, nxt_end, _ = items[j]
                    gap = nxt_start - prev_end
                    next_dur = nxt_end - nxt_start
                    if gap > args.max_gap:
                        break
                    if chain_dur + next_dur > args.max_dur:
                        break
                    # Randomly stop early even though we could still extend, to
                    # spread the duration distribution toward shorter clips. Halve
                    # the chance while the chain is still a single segment (no
                    # concat yet) so we don't over-produce 1-segment clips.
                    stop_p = args.stop_prob / 2 if len(chain) == 1 else args.stop_prob
                    if chain_dur >= args.min_dur and random.random() < stop_p:
                        break
                    chain.append(items[j])
                    chain_dur += next_dur
                    j += 1

                if args.min_dur <= chain_dur <= args.max_dur and len(chain) >= 1:
                    seg_ids = [c[2] for c in chain]
                    paths = [str(seg_to_flac(audio_root, sid)) for sid in seg_ids]
                    if all(text in texts for text in (s for s in seg_ids)):
                        joined_text = " ".join(texts[s].strip() for s in seg_ids)
                        plan = {
                            "id": f"{speaker}_{book}_{seg_ids[0].split('_')[-1]}_chain{len(chain)}",
                            "speaker": speaker,
                            "book": book,
                            "chapter_url": url,
                            "seg_ids": seg_ids,
                            "flac_paths": paths,
                            "duration": round(chain_dur, 3),
                            "text": joined_text,
                        }
                        fout.write(json.dumps(plan, ensure_ascii=False) + "\n")
                        n_plans += 1
                        if args.limit and n_plans >= args.limit:
                            print(f"Wrote {n_plans} plans -> {out_path} (limit reached)")
                            return
                # Advance past whatever was consumed (at least 1).
                i = max(j, i + 1)

    print(f"Wrote {n_plans} plans -> {out_path}")


# ---------------------------------------------------------------------------
# Mode: apply
# ---------------------------------------------------------------------------
def run_apply(args, plan_path: Path, audio_out: Path, out_path: Path):
    audio_out.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.resume and out_path.exists():
        truncate_partial_tail(out_path)
        with out_path.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
        print(f"Resume: skipping {len(done)} already-done records")

    total = sum(1 for _ in open(plan_path))
    if args.limit:
        total = min(total, args.limit + len(done))

    n_done = 0
    n_err = 0
    mode = "a" if args.resume and out_path.exists() else "w"
    with open(plan_path) as fin, out_path.open(mode) as fout:
        bar = tqdm(fin, total=total, desc=f"concat {out_path.stem}", unit="clip")
        for line in bar:
            plan = json.loads(line)
            if plan["id"] in done:
                continue
            try:
                concat_path = audio_out / f"{plan['id']}.flac"
                if args.force or not concat_path.exists():
                    duration, _ = concat_flacs(plan["flac_paths"], concat_path)
                else:
                    duration = sf.info(concat_path).duration
            except Exception as e:
                n_err += 1
                bar.set_postfix(ok=n_done, err=n_err)
                tqdm.write(f"[err] {plan['id']}: {e}")
                continue

            rec = {
                "id": plan["id"],
                "conversations": [
                    {"from": "User", "value": str(concat_path), "type": "audio",
                     "duration": round(duration, 3)},
                    {"from": "Assistant", "value": plan["text"], "type": "text"},
                ],
                "custom_metadata": {
                    "source": {
                        "dataset": "MLS_fr",
                        "speaker": plan["speaker"],
                        "book": plan["book"],
                        "chapter_url": plan["chapter_url"],
                        "seg_ids": plan["seg_ids"],
                    },
                },
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            bar.set_postfix(ok=n_done, err=n_err)
            if args.limit and n_done >= args.limit:
                break

    print(f"Done. records={n_done} errors={n_err} -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="plan", choices=["plan", "apply"],
                    help="plan: build the concat plan; apply: concat audio + write ASR dataset")
    ap.add_argument("--concat_root", default=None,
                    help="Output root (default: <mls_root>/concatenated). Layout: "
                         "<root>/plan/{split}.jsonl (plans), <root>/audios/{split}/ "
                         "(concatenated FLACs), <root>/{split}.jsonl (final ASR dataset).")
    ap.add_argument("--limit", type=int, default=0, help="Cap on number of clips (0 = all)")

    # plan-mode args
    ap.add_argument("--split", default="all", choices=SPLITS + ["all"],
                    help="One split, or 'all' to process train/dev/test in one run "
                         "(then --out/--plan/--audio_out_dir are treated as directories)")
    ap.add_argument(
        "--mls_root",
        default=f"{os.environ['DATA_DIR']}/raw/transcript/multilang/MultilingualLibriSpeech/mls_french",
    )
    ap.add_argument(
        "--recasepunc_dir",
        default=f"{os.environ['DATA_DIR']}/nemo/asr/fr/nocontext/Multilingual_LibriSpeech_recasepunc",
        help="Dir holding {split}_recasepunc.jsonl (recased+punctuated transcripts, "
             "keyed by MLS seg_id). Used by default; needed for the sentence tasks.",
    )
    ap.add_argument("--raw_text", action="store_true",
                    help="Use raw MLS transcripts.txt (lowercased, no punctuation) "
                         "instead of the recasepunc transcripts.")
    ap.add_argument("--min_dur", type=float, default=10.0)
    ap.add_argument("--max_dur", type=float, default=60.0)
    ap.add_argument(
        "--max_gap",
        type=float,
        default=3.0,
        help="Max gap (s) between consecutive segments in the same chapter "
        "to still consider them contextually adjacent.",
    )
    ap.add_argument(
        "--stop_prob",
        type=float,
        default=0.1,
        help="Probability of ending a chain early at each boundary once it is "
        "already >= --min_dur (0 = greedy/longest clips; higher = more short "
        "clips). Halved while the chain is still a single segment.",
    )
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for --stop_prob randomness (reproducible plans).")

    # apply-mode args
    ap.add_argument("--force", action="store_true",
                    help="[apply] Re-concatenate audio even if the FLAC already exists "
                         "(e.g. after regenerating plans with a new --stop_prob).")
    ap.add_argument("--resume", action="store_true",
                    help="[apply] Skip plan IDs already present in the output file")
    args = ap.parse_args()

    concat_root = Path(args.concat_root) if args.concat_root else Path(args.mls_root) / "concatenated"
    splits = SPLITS if args.split == "all" else [args.split]

    for split in splits:
        if args.mode == "plan":
            run_plan(args, split, concat_root / "plan" / f"{split}.jsonl")
        else:
            run_apply(
                args,
                plan_path=concat_root / "plan" / f"{split}.jsonl",
                audio_out=concat_root / "audios" / split,
                out_path=concat_root / f"{split}.jsonl",
            )


if __name__ == "__main__":
    main()
