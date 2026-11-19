"""Teacher-forcing diagnostic of the ASR head of a DuplexSTT checkpoint, on the validation ASR sets.

The ASR-only run (duplex_asr_only_4gpu) reaches a low teacher-forced asr_loss but its free-running
(autoregressive) transcripts are mostly empty. Three decodings of the same batches tell why:

  tf       teacher forcing, as in training: the ASR channel input is the reference token of the previous
           frame; per-frame argmax.
  tf_pad   same, but the ASR channel input is <pad> at every frame (no transcript history): what the
           head gets from the audio alone, i.e. the situation of free running after a missed word.
  tf_bos   same as tf_pad, but <bos>/<eos> are kept where the reference has them: the ASR input of a
           training run with asr_input_dropout=1.0.
  free     DuplexSTTStreamingInference.offline_inference with a silent agent, as in validation: <pad> as
           ASR input at frame 0 (_get_asr_bos_embedding). Without delay_source_text_by, training has the
           <bos> of the user supervision there when it starts at 0 (asr_input_bos_at_0 gives how often).
  free_bos same, with <bos> as ASR input at frame 0: how runs trained with delay_source_text_by=0 work.
  rnnt     the Nemotron RNNT decoder alone (frozen encoder of the duplex model + its own prompt/decoder/
           joint), language prompt from the dataset name, at the training context [56, 3]: the ceiling
           of the audio features. rnnt6, rnnt13: same with the [56, 6], [56, 13] lookaheads (--rnnt-contexts).

Per label class (pad / word start / continuation / bos-eos), the frame accuracy and the mean P(<pad>);
P(non-pad) around each word start (offsets -4..+3, single-token words for the positive offsets: their
next labels are <pad>, so a non-pad mass there means "too late"); WER of each decoding (same WER
class as the validation) and the share of empty outputs. Reading:

  tf word-start accuracy high, tf_pad low        -> the head starts words from its own transcript
                                                    history (exposure bias): ASR input dropout
  tf word-start accuracy low, non-pad mass after -> the word end comes too early for it: delay the
                                                    ASR channel (model.delay_source_text_by)
  both low, no mass after                        -> no usable signal (data, alignment, capacity)

Writes <out>/<dataset>.jsonl (one line per example: refs, hyps, per-frame timelines) and
<out>/summary.json. Run through asr_teacher_forcing.slurm.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.models import ASRModel
from nemo.collections.speechlm2 import DataModule, DuplexSTTDataset, DuplexSTTModel
from nemo.collections.speechlm2.parts.pretrained import load_pretrained_nemo
from nemo.collections.speechlm2.parts.metrics.wer import WER

OFFSETS = range(-4, 4)
TF_MODES = ("tf", "tf_pad", "tf_bos")
FREE_MODES = ("free", "free_bos")
CLASSES = ("pad", "start", "cont", "bos_eos")


def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x


def load_model(exp_dir: Path, ckpt: Path) -> DuplexSTTModel:
    cfg = OmegaConf.load(exp_dir / "exp_config.yaml")
    model = DuplexSTTModel(OmegaConf.to_container(cfg.model, resolve=True))
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    # nn.Module.load_state_dict: DuplexSTTModel's override retries a partial load on any mismatch,
    # which would hide a missing trained module.
    missing, unexpected = torch.nn.Module.load_state_dict(model, state, strict=False)
    print(f"[load] {ckpt}: {len(state)} tensors, missing {len(missing)}, unexpected {len(unexpected)}")
    for k in missing[:20] + unexpected[:20]:
        print(f"    {k}")
    trained = ("asr_head_adapter", "embed_asr_tokens", "lora_", "perception.proj", "asr_head.delta")
    lost = [k for k in missing if any(t in k for t in trained)]
    assert not lost, f"trained weights missing from the checkpoint: {lost[:5]}"
    return cfg, model.to(torch.bfloat16).cuda().eval()


def label_classes(labels: np.ndarray, pad: int, special: set[int], pieces: list[str]) -> np.ndarray:
    """Label class per frame: 0 pad, 1 word start, 2 continuation, 3 bos/eos. A word starts after a
    non-text label or on a token with a leading-space marker (words are pushed back to back when the
    previous one is still being emitted)."""
    cls = np.zeros(len(labels), dtype=np.int64)
    for t, tok in enumerate(labels):
        if tok == pad:
            continue
        if tok in special:
            cls[t] = 3
            continue
        prev = labels[t - 1] if t else pad
        new_word = prev == pad or prev in special or pieces[t][:1] in ("Ġ", "▁", " ")
        cls[t] = 1 if new_word else 2
    return cls


def timeline(tokens: np.ndarray, pad: int, special: set[int], cls: np.ndarray | None = None) -> str:
    """'.' pad, 'B' bos/eos, '|' word start (reference only, from cls), 'x' other token."""
    out = []
    for t, tok in enumerate(tokens):
        if tok == pad:
            out.append(".")
        elif tok in special:
            out.append("B")
        elif cls is not None and cls[t] == 1:
            out.append("|")
        else:
            out.append("x")
    return "".join(out)


def rnnt_transcribe(asr, audio, lens, lang, context):
    asr.encoder.set_default_att_context_size(context)
    prompt = torch.full((audio.shape[0],), asr.cfg.model_defaults.prompt_dictionary[lang], device=audio.device)
    enc, enc_len = asr.forward(input_signal=audio, input_signal_length=lens, prompt_indices=prompt)
    hyps = asr.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc, encoded_lengths=enc_len)
    hyps = hyps[0] if isinstance(hyps, tuple) else hyps
    return [(h.text if hasattr(h, "text") else h).strip() for h in hyps]


@torch.inference_mode()
def run_dataset(name, loader, model, args, asr=None):
    tok = model.tokenizer
    pad = model.text_pad_id
    special = {model.text_bos_id, model.text_eos_id}
    w_asr = model.cfg.get("duplex_asr_text_weight", 1.0)
    pad_emb = model.embed_asr_tokens(torch.tensor([pad], device="cuda")) * w_asr

    stats = defaultdict(lambda: np.zeros(2))  # key -> [sum, count]
    curve = {m: {o: np.zeros(2) for o in OFFSETS} for m in TF_MODES}
    rnnt_modes = {("rnnt" if c[1] == 3 else f"rnnt{c[1]}"): c for c in args.rnnt_contexts} if asr else {}
    lang = name.rsplit("_", 1)[-1]
    wers = {m: WER(verbose=False) for m in TF_MODES + FREE_MODES + tuple(rnnt_modes)}
    bos_emb = model.embed_asr_tokens(torch.tensor([model.text_bos_id], device="cuda"))
    pad_raw = model.embed_asr_tokens(torch.tensor([pad], device="cuda"))
    asr_bos = model._get_asr_bos_embedding
    empty = defaultdict(int)
    rows, n = [], 0
    out_f = open(args.out / f"{name}.jsonl", "w")

    for batch in loader:
        batch = to_device(batch["audio_data"], "cuda")
        B = batch["source_audio"].shape[0]

        # Free running first: prepare_inputs replaces some batch entries (prompt prepending).
        free = {}
        for m, first_emb in (("free", lambda: pad_raw), ("free_bos", lambda: bos_emb)):
            model._get_asr_bos_embedding = first_emb
            free[m] = model.streaming_inference.offline_inference(
                batch["source_audio"],
                batch["source_audio_lens"],
                prompt_tokens=batch.get("prompt_tokens"),
                prompt_token_lens=batch.get("prompt_token_lens"),
                silent_agent=torch.ones(B, dtype=torch.bool, device="cuda"),
            )
        model._get_asr_bos_embedding = asr_bos
        rnnt = {m: rnnt_transcribe(asr, batch["source_audio"].float(), batch["source_audio_lens"], lang, c)
                for m, c in rnnt_modes.items()}
        tf_batch = dict(batch)  # prepare_inputs puts the prompt-prepended source_tokens back in it
        inputs = model.prepare_inputs(tf_batch)
        embeds = inputs["input_embeds"]
        src = tf_batch["source_tokens"]
        if d := model.cfg.get("delay_source_text_by", 0):  # as label_prep.prepare_text_and_asr_labels
            src = torch.cat([torch.full_like(src[:, :d], pad), src[:, :-d]], dim=-1)
        asr_in = src[:, : embeds.shape[1]]  # = prepare_inputs' asr_inputs
        assert asr_in.shape == inputs["asr_labels"].shape
        assert (asr_in[:, 1:] == inputs["asr_labels"][:, :-1]).all()
        embeds_pad = embeds - model.embed_asr_tokens(asr_in) * w_asr + pad_emb
        keep = (asr_in == model.text_bos_id) | (asr_in == model.text_eos_id)
        embeds_bos = embeds_pad + (model.embed_asr_tokens(asr_in) * w_asr - pad_emb) * keep.unsqueeze(-1)
        logits = {"tf": model(embeds)["asr_logits"].float(), "tf_pad": model(embeds_pad)["asr_logits"].float(),
                  "tf_bos": model(embeds_bos)["asr_logits"].float()}
        probs = {m: l.softmax(-1) for m, l in logits.items()}

        prompt_lens = batch.get("prompt_token_lens")
        for i in range(B):
            p = int(prompt_lens[i]) if prompt_lens is not None else 0
            T = int(inputs["input_lens"][i])
            labels = inputs["asr_labels"][i, p:T].cpu().numpy()
            if not (labels != pad).any():
                continue
            pieces = tok.ids_to_tokens(labels.tolist())
            cls = label_classes(labels, pad, special, pieces)
            ref = batch["source_texts"][i]
            row = {"sample_id": batch["sample_id"][i], "ref": ref, "labels": timeline(labels, pad, special, cls)}
            stats["asr_input_bos_at_0"] += [int(asr_in[i, p] == model.text_bos_id), 1]

            for m in TF_MODES:
                pr = probs[m][i, p:T]
                pred = pr.argmax(-1).cpu().numpy()
                p_pad = pr[:, pad].cpu().numpy()
                for c, cname in enumerate(CLASSES):
                    sel = cls == c
                    stats[f"{m}/acc/{cname}"] += [(pred[sel] == labels[sel]).sum(), sel.sum()]
                    stats[f"{m}/p_pad/{cname}"] += [p_pad[sel].sum(), sel.sum()]
                # Word start predicted as <pad> (missed) vs a wrong token.
                sel = cls == 1
                stats[f"{m}/start_as_pad"] += [(pred[sel] == pad).sum(), sel.sum()]
                for t0 in np.flatnonzero(cls == 1):
                    single = t0 + 4 <= len(cls) - 1 and (labels[t0 + 1 : t0 + 4] == pad).all()
                    for o in OFFSETS:
                        if 0 <= t0 + o < len(cls) and (o <= 0 or single):
                            curve[m][o] += [1 - p_pad[t0 + o], 1]
                hyp = tok.ids_to_text([int(x) for x in pred if x != pad and x not in special]).strip()
                wers[m].update(name, [ref], [hyp])
                empty[m] += not hyp
                row[f"{m}_hyp"] = hyp
                row[m] = timeline(pred, pad, special)

            for m in FREE_MODES:
                gen = free[m]["tokens_text_src"][i, : T - p].cpu().numpy()
                hyp = tok.ids_to_text([int(x) for x in gen if x != pad and x not in special]).strip()
                wers[m].update(name, [ref], [hyp])
                empty[m] += not hyp
                row[f"{m}_hyp"], row[m] = hyp, timeline(gen, pad, special)
                # First frame where free running leaves the reference, and the reference class there.
                diff = np.flatnonzero(gen[: len(labels)] != labels)
                first = CLASSES[cls[diff[0]]] if len(diff) else "none"
                stats[f"{m}/first_divergence/{first}"] += [1, 0]
                row[f"{m}_first_divergence"] = first

            for m in rnnt_modes:
                wers[m].update(name, [ref], [rnnt[m][i]])
                empty[m] += not rnnt[m][i]
                row[f"{m}_hyp"] = rnnt[m][i]

            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            n += 1
        if n >= args.max_examples:
            break
    out_f.close()

    summary = {"examples": n}
    for k, (s, c) in sorted(stats.items()):
        summary[k] = int(s) if "/first_divergence/" in k else (float(s / c) if c else None)
    for m in curve:
        summary[f"{m}/p_nonpad_around_start"] = {o: float(s / c) if c else None for o, (s, c) in curve[m].items()}
    for m, w in wers.items():
        summary[f"{m}/wer"] = float(w.compute()[f"wer_{name}"])
        summary[f"{m}/empty"] = empty[m] / max(n, 1)

    print(f"\n===== {name}: {n} examples =====")
    print(f"{'':10}" + "".join(f"{c:>18}" for c in CLASSES))
    for m in TF_MODES:
        print(f"{m + ' acc':10}" + "".join(f"{summary[f'{m}/acc/{c}'] or 0:18.3f}" for c in CLASSES))
        print(f"{m + ' Ppad':10}" + "".join(f"{summary[f'{m}/p_pad/{c}'] or 0:18.3f}" for c in CLASSES))
    print("frames per class: " + ", ".join(f"{c} {int(stats[f'tf/acc/{c}'][1])}" for c in CLASSES))
    print(f"ASR input <bos> at frame 0: {summary['asr_input_bos_at_0']:.2f}")
    for m in TF_MODES:
        print(f"{m}: word start predicted as <pad>: {summary[f'{m}/start_as_pad']:.3f}")
        print(f"{m}: P(non-pad) at word start offset " + "  ".join(
            f"{o:+d}:{v:.2f}" for o, v in summary[f"{m}/p_nonpad_around_start"].items() if v is not None))
    for m in TF_MODES + FREE_MODES + tuple(rnnt_modes):
        print(f"{m:8} WER {summary[f'{m}/wer']:.3f}  empty {summary[f'{m}/empty']:.2f}")
    for m in FREE_MODES:
        print(f"{m}, reference class at the first divergence: " + ", ".join(
            f"{k.split('/')[-1]} {v}" for k, v in summary.items() if k.startswith(f"{m}/first_divergence/")))
    for row in rows[: args.show]:
        print(f"\n[{row['sample_id']}] {row['ref']}")
        for m in ("labels",) + TF_MODES + FREE_MODES:
            print(f"  {m:8} {row[m]}")
        for m in TF_MODES + FREE_MODES + tuple(rnnt_modes):
            print(f"  {m + '_hyp':12} {row[m + '_hyp']}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp-dir", type=Path, required=True, help="experiment folder (exp_config.yaml)")
    parser.add_argument("--ckpt", type=Path, required=True, help="Lightning .ckpt of that experiment")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["asr_fr", "asr_en"], help="validation_ds.datasets keys")
    parser.add_argument("--max-examples", type=int, default=200, help="per dataset")
    parser.add_argument("--show", type=int, default=6, help="examples printed per dataset")
    parser.add_argument("--rnnt-contexts", type=json.loads, default=[[56, 3], [56, 6], [56, 13]],
                        help="att_context_size of the Nemotron RNNT baselines (JSON); [] skips them")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    torch.set_float32_matmul_precision("medium")
    cfg, model = load_model(args.exp_dir, args.ckpt)
    # conversation_as_duplex tokenizes the transcripts with the LLM_MODEL tokenizer (set by run_duplex.slurm).
    os.environ.setdefault("LLM_MODEL", cfg.model.pretrained_llm)
    dataset = DuplexSTTDataset(
        tokenizer=model.tokenizer,
        frame_length=cfg.data.frame_length,
        source_sample_rate=cfg.data.source_sample_rate,
        input_roles=cfg.data.input_roles,
        output_roles=cfg.data.output_roles,
        aug_by_swap_role=False,
        cfg=cfg.data,
        model_cfg=cfg.model,
    )
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)
    asr = None
    if args.rnnt_contexts:
        asr = load_pretrained_nemo(ASRModel, cfg.model.pretrained_asr).cuda().eval()
        # The duplex encoder is frozen: the checkpoint's copy must be the pretrained one.
        enc = model.perception.encoder.state_dict()
        drift = max((v.float() - enc[k].float()).abs().max().item()
                    for k, v in asr.encoder.state_dict().items() if k in enc)
        print(f"[rnnt] {cfg.model.pretrained_asr}: max |encoder - duplex encoder| = {drift:.2e}")
    val = cfg.data.validation_ds
    base = val.copy()
    with open_dict(base):
        del base.datasets

    summaries = {"ckpt": str(args.ckpt)}
    for name in args.datasets:
        item = OmegaConf.merge(base, val.datasets[name])
        summaries[name] = run_dataset(name, datamodule._build_test_dataloader(item), model, args, asr)
    with open(args.out / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"\n[done] {args.out}")


if __name__ == "__main__":
    main()
