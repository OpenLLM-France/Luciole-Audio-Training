"""Word timestamps of the duplex manifests (custom.user_words, CTC peaks of an offline FastConformer) against the
frames where the streaming Nemotron RNNT emits the same words, on the frozen duplex encoder.

The agent channel of the transcribe model writes each word at its user_words end + data.lag. If those ends are
not where the streaming encoder has the word, the LLM has to guess the frame. Here, for each word that Nemotron
transcribes like the reference (normalized, difflib matching blocks), in 80 ms frames:

  d_end   = Nemotron frame of the word's last token - user_words end / 0.08
  d_start = Nemotron frame of the word's first token - user_words start / 0.08

Reading: mean of d_end = constant shift (a lag could absorb it); std of d_end, and its std within an utterance
(per-utterance mean removed) = jitter that no constant lag corrects.

Writes <out>/<lang>_ctx<N>.jsonl (per word) and <out>/summary.json. Run through timestamps_vs_nemotron.slurm.
"""

import argparse
import difflib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from omegaconf import open_dict

from nemo.collections.asr.models import ASRModel
from nemo.collections.speechlm2.parts.pretrained import load_pretrained_nemo

FRAME = 0.08


def norm(w: str) -> str:
    w = unicodedata.normalize("NFC", w).lower().replace("’", "'")
    return re.sub(r"[^\w']", "", w)


def read_lines(path: Path, n: int):
    """n lines taken with a stride over the file (the manifests are not shuffled)."""
    lines = [l for l in open(path) if l.strip()]
    stride = max(1, len(lines) // n)
    out = []
    for l in lines[::stride][:n]:
        x = json.loads(l)
        audio = [c for c in x["conversations"] if c.get("type") == "audio"]
        words = x.get("custom", {}).get("user_words")
        if len(audio) != 1 or not words or not words[0]:
            continue
        out.append({"id": x["id"], "audio": audio[0]["value"], "words": words[0]})
    return out


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    wav, s = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(1)
    if s != sr:
        import librosa

        wav = librosa.resample(wav, orig_sr=s, target_sr=sr)
    return wav


@torch.inference_mode()
def nemotron_words(asr, wavs, lang, context):
    """Per utterance: list of (word, first token frame, last token frame) from greedy RNNT."""
    asr.encoder.set_default_att_context_size(context)
    lens = torch.tensor([len(w) for w in wavs], device="cuda")
    audio = torch.zeros(len(wavs), int(lens.max()), device="cuda")
    for i, w in enumerate(wavs):
        audio[i, : len(w)] = torch.from_numpy(w)
    prompt = torch.full((len(wavs),), asr.cfg.model_defaults.prompt_dictionary[lang], device="cuda")
    enc, enc_len = asr.forward(input_signal=audio, input_signal_length=lens, prompt_indices=prompt)
    hyps = asr.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc, encoded_lengths=enc_len,
                                                        return_hypotheses=True)
    hyps = hyps[0] if isinstance(hyps, tuple) else hyps
    out = []
    for h in hyps:
        ids = h.y_sequence.tolist() if torch.is_tensor(h.y_sequence) else list(h.y_sequence)
        ts = h.timestamp["timestep"] if isinstance(h.timestamp, dict) else h.timestamp
        ts = ts.tolist() if torch.is_tensor(ts) else list(ts)
        assert len(ts) == len(ids), (len(ts), len(ids))
        pieces = asr.tokenizer.ids_to_tokens(ids)
        words = []  # [text_ids, first frame, last frame]
        for tid, piece, t in zip(ids, pieces, ts):
            if piece.startswith("▁") or not words:
                words.append([[tid], t, t])
            else:
                words[-1][0].append(tid)
                words[-1][2] = t
        out.append([(asr.tokenizer.ids_to_text(w), s, e) for w, s, e in words])
    return out


def stats(x):
    x = np.asarray(x, dtype=float)
    if not len(x):
        return None
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std()),
            "p5": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p95": q[4],
            "abs<=1": float((np.abs(x - np.median(x)) <= 1).mean())}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--asr", required=True, help="nemotron-3.5-asr-streaming-0.6b.nemo")
    p.add_argument("--manifest", nargs="+", required=True, help="lang=path.jsonl")
    p.add_argument("--contexts", type=json.loads, default=[[56, 3], [56, 6]])
    p.add_argument("--lines", type=int, default=1000, help="per manifest, with a stride")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    asr = load_pretrained_nemo(ASRModel, args.asr).cuda().eval()
    dec = asr.cfg.decoding
    with open_dict(dec):
        dec.compute_timestamps = True
        dec.rnnt_timestamp_type = "all"
    if hasattr(asr, "cur_decoder"):
        asr.change_decoding_strategy(dec, decoder_type="rnnt")
    else:
        asr.change_decoding_strategy(dec)

    summary = {}
    for spec in args.manifest:
        lang, path = spec.split("=", 1)
        rows = read_lines(Path(path), args.lines)
        wavs = [load_audio(r["audio"]) for r in rows]
        print(f"[{lang}] {len(rows)} utterances from {path}")
        for ctx in args.contexts:
            key = f"{lang}_ctx{ctx[1]}"
            d_end, d_start, d_end_centered, n_ref, n_match = [], [], [], 0, 0
            f = open(args.out / f"{key}.jsonl", "w")
            for b in range(0, len(rows), args.batch):
                hyps = nemotron_words(asr, wavs[b : b + args.batch], lang, ctx)
                for r, hw in zip(rows[b : b + args.batch], hyps):
                    ref = [(w, s, e) for w, s, e in r["words"] if norm(w)]
                    hw = [x for x in hw if norm(x[0])]
                    n_ref += len(ref)
                    sm = difflib.SequenceMatcher(a=[norm(w) for w, _, _ in ref], b=[norm(w) for w, _, _ in hw],
                                                 autojunk=False)
                    utt = []
                    for blk in sm.get_matching_blocks():
                        for k in range(blk.size):
                            (w, s, e), (_, hs, he) = ref[blk.a + k], hw[blk.b + k]
                            de, ds = he - e / FRAME, hs - s / FRAME
                            utt.append(de)
                            d_start.append(ds)
                            f.write(json.dumps({"id": r["id"], "word": w, "ref_start": s, "ref_end": e,
                                                "nem_first": hs, "nem_last": he, "d_start": ds, "d_end": de},
                                               ensure_ascii=False) + "\n")
                    n_match += len(utt)
                    d_end += utt
                    if len(utt) >= 3:
                        d_end_centered += list(np.asarray(utt) - np.mean(utt))
            f.close()
            summary[key] = {"matched_words": n_match / max(n_ref, 1), "d_end": stats(d_end),
                            "d_start": stats(d_start), "d_end_within_utt": stats(d_end_centered)}
            s = summary[key]
            print(f"\n== {key}: {n_match}/{n_ref} words matched ({s['matched_words']:.2f})")
            for k in ("d_end", "d_start", "d_end_within_utt"):
                v = s[k]
                print(f"  {k:17} mean {v['mean']:+.2f}  std {v['std']:.2f}  p5/p25/med/p75/p95 "
                      f"{v['p5']:+.1f} {v['p25']:+.1f} {v['median']:+.1f} {v['p75']:+.1f} {v['p95']:+.1f}  "
                      f"|x-med|<=1: {v['abs<=1']:.2f}")
            hist = np.bincount(np.clip(np.round(d_end).astype(int) + 8, 0, 16), minlength=17)
            print("  d_end histogram (frames -8..+8, clipped): " +
                  " ".join(f"{i - 8:+d}:{c}" for i, c in enumerate(hist) if c))
    with open(args.out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] {args.out}")


if __name__ == "__main__":
    main()
