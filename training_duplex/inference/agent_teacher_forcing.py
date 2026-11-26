"""Teacher-forcing diagnostic of the agent channel of a single-head DuplexSTT checkpoint (predict_user_text:
false, e.g. duplex_transcribe: the transcript IS the agent channel), on the validation ASR sets.

Same reading as asr_teacher_forcing.py (which handles the separate ASR head of the two-head models), on the
agent channel: input of frame t = embed(agent token t) + audio frame t, label = agent token t + 1.

  tf       teacher forcing, as in training: the agent input is the reference token of the previous frame;
           per-frame argmax.
  tf_pad   same, but every text token of the agent input is replaced by <pad> (<bos>/<eos> kept): what the
           channel gets from the audio alone, i.e. the situation of free running after a missed word.
  free     DuplexSTTStreamingInference.offline_inference, as in validation.
  rnnt     the Nemotron RNNT decoder alone on the frozen encoder, at --rnnt-contexts: the ceiling of the
           audio features (rnnt = [56, 3], rnnt6 = [56, 6], ...).

Per label class (pad / word start / continuation / bos-eos), the frame accuracy and the mean P(<pad>);
P(non-pad) around each word start (offsets -4..+3, single-token words for the positive offsets: a non-pad
mass after the start means "too late"); WER of each decoding and the share of empty outputs. Reading:

  tf word-start accuracy high, tf_pad low        -> words started from the transcript history (exposure bias)
  tf word-start accuracy low, non-pad mass after -> the word end comes too early: more lag / lookahead
  tf word-start ok, wrong token (not <pad>)      -> timing fine, the word itself is not read from the audio
  both low, no mass after                        -> no usable signal (data, alignment, capacity)

--att-context sets the lookahead of the duplex encoder (default: the first one of the checkpoint's config).
Writes <out>/<dataset>.jsonl and <out>/summary.json. Run through agent_teacher_forcing.slurm.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from asr_teacher_forcing import CLASSES, OFFSETS, label_classes, load_model, rnnt_transcribe, timeline, to_device
from nemo.collections.asr.models import ASRModel
from nemo.collections.speechlm2 import DataModule, DuplexSTTDataset
from nemo.collections.speechlm2.parts.metrics.wer import WER
from nemo.collections.speechlm2.parts.pretrained import load_pretrained_nemo

TF_MODES = ("tf", "tf_pad")


@torch.inference_mode()
def run_dataset(name, loader, model, args, asr=None):
    tok = model.tokenizer
    pad = model.text_pad_id
    special = {model.text_bos_id, model.text_eos_id}
    w_txt = model.cfg.get("duplex_text_channel_weight", 1.0)
    pad_emb = model.embed_tokens(torch.tensor([pad], device="cuda")) * w_txt

    stats = defaultdict(lambda: np.zeros(2))  # key -> [sum, count]
    curve = {m: {o: np.zeros(2) for o in OFFSETS} for m in TF_MODES}
    rnnt_modes = {("rnnt" if c[1] == 3 else f"rnnt{c[1]}"): c for c in args.rnnt_contexts} if asr else {}
    lang = name.rsplit("_", 1)[-1]
    wers = {m: WER(verbose=False) for m in TF_MODES + ("free",) + tuple(rnnt_modes)}
    empty = defaultdict(int)
    rows, n = [], 0
    out_f = open(args.out / f"{name}.jsonl", "w")

    for batch in loader:
        batch = to_device(batch["audio_data"], "cuda")
        # Free running first: prepare_inputs replaces some batch entries (prompt prepending).
        free = model.streaming_inference.offline_inference(
            batch["source_audio"],
            batch["source_audio_lens"],
            prompt_tokens=batch.get("prompt_tokens"),
            prompt_token_lens=batch.get("prompt_token_lens"),
        )
        rnnt = {m: rnnt_transcribe(asr, batch["source_audio"].float(), batch["source_audio_lens"], lang, c)
                for m, c in rnnt_modes.items()}
        inputs = model.prepare_inputs(dict(batch))
        embeds = inputs["input_embeds"]
        labels_all = inputs["text_labels"]
        # text_inputs = labels shifted right by one frame (label_prep: inputs[:, :-1], labels[:, 1:]).
        txt_in = torch.cat([torch.full_like(labels_all[:, :1], pad), labels_all[:, :-1]], dim=-1)
        is_text = (txt_in != pad) & (txt_in != model.text_bos_id) & (txt_in != model.text_eos_id)
        embeds_pad = embeds - (model.embed_tokens(txt_in) * w_txt - pad_emb) * is_text.unsqueeze(-1)
        probs = {"tf": model(embeds)["text_logits"].float().softmax(-1),
                 "tf_pad": model(embeds_pad)["text_logits"].float().softmax(-1)}

        prompt_lens = batch.get("prompt_token_lens")
        for i in range(embeds.shape[0]):
            p = int(prompt_lens[i]) if prompt_lens is not None else 0
            T = int(inputs["input_lens"][i]) if "input_lens" in inputs else labels_all.shape[1]
            labels = labels_all[i, p:T].cpu().numpy()
            if not (labels != pad).any():
                continue
            pieces = tok.ids_to_tokens(labels.tolist())
            cls = label_classes(labels, pad, special, pieces)
            ref = batch["target_texts"][i]
            row = {"sample_id": batch["sample_id"][i], "ref": ref, "labels": timeline(labels, pad, special, cls)}

            for m in TF_MODES:
                pr = probs[m][i, p:T]
                pred = pr.argmax(-1).cpu().numpy()
                p_pad = pr[:, pad].cpu().numpy()
                for c, cname in enumerate(CLASSES):
                    sel = cls == c
                    stats[f"{m}/acc/{cname}"] += [(pred[sel] == labels[sel]).sum(), sel.sum()]
                    stats[f"{m}/p_pad/{cname}"] += [p_pad[sel].sum(), sel.sum()]
                sel = cls == 1
                stats[f"{m}/start_as_pad"] += [(pred[sel] == pad).sum(), sel.sum()]
                stats[f"{m}/start_wrong_token"] += [((pred[sel] != pad) & (pred[sel] != labels[sel])).sum(), sel.sum()]
                for t0 in np.flatnonzero(cls == 1):
                    single = t0 + 4 <= len(cls) - 1 and (labels[t0 + 1 : t0 + 4] == pad).all()
                    for o in OFFSETS:
                        if 0 <= t0 + o < len(cls) and (o <= 0 or single):
                            curve[m][o] += [1 - p_pad[t0 + o], 1]
                hyp = tok.ids_to_text([int(x) for x in pred if x != pad and x not in special]).strip()
                wers[m].update(name, [ref], [hyp])
                empty[m] += not hyp
                row[f"{m}_hyp"], row[m] = hyp, timeline(pred, pad, special)

            gen = free["tokens_text"][i, : T - p].cpu().numpy()
            hyp = tok.ids_to_text([int(x) for x in gen if x != pad and x not in special]).strip()
            wers["free"].update(name, [ref], [hyp])
            empty["free"] += not hyp
            row["free_hyp"], row["free"] = hyp, timeline(gen, pad, special)
            diff = np.flatnonzero(gen[: len(labels)] != labels)
            first = CLASSES[cls[diff[0]]] if len(diff) else "none"
            stats[f"free/first_divergence/{first}"] += [1, 0]
            row["free_first_divergence"] = first

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
    for m in TF_MODES:
        print(f"{m}: word start predicted as <pad>: {summary[f'{m}/start_as_pad']:.3f}, "
              f"as a wrong token: {summary[f'{m}/start_wrong_token']:.3f}")
        print(f"{m}: P(non-pad) at word start offset " + "  ".join(
            f"{o:+d}:{v:.2f}" for o, v in summary[f"{m}/p_nonpad_around_start"].items() if v is not None))
    for m in wers:
        print(f"{m:8} WER {summary[f'{m}/wer']:.3f}  empty {summary[f'{m}/empty']:.2f}")
    print("free, reference class at the first divergence: " + ", ".join(
        f"{k.split('/')[-1]} {v}" for k, v in summary.items() if k.startswith("free/first_divergence/")))
    for row in rows[: args.show]:
        print(f"\n[{row['sample_id']}] {row['ref']}")
        for m in ("labels",) + TF_MODES + ("free",):
            print(f"  {m:8} {row[m]}")
        for m in wers:
            print(f"  {m + '_hyp':12} {row[m + '_hyp']}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp-dir", type=Path, required=True, help="experiment folder (exp_config.yaml)")
    parser.add_argument("--ckpt", type=Path, required=True, help="Lightning .ckpt of that experiment")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["asr_fr", "asr_en"], help="validation_ds.datasets keys")
    parser.add_argument("--max-examples", type=int, default=256, help="per dataset")
    parser.add_argument("--show", type=int, default=6, help="examples printed per dataset")
    parser.add_argument("--att-context", type=json.loads, default=None,
                        help="lookahead of the duplex encoder (JSON, e.g. [56,6]); default: the config's first")
    parser.add_argument("--rnnt-contexts", type=json.loads, default=[[56, 3], [56, 6]],
                        help="att_context_size of the Nemotron RNNT baselines (JSON); [] skips them")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    torch.set_float32_matmul_precision("medium")
    cfg, model = load_model(args.exp_dir, args.ckpt)
    assert not model.predict_user_text, "two-head model: use asr_teacher_forcing.py"
    if args.att_context is not None:
        model.perception.encoder.set_default_att_context_size(args.att_context)
    print(f"[encoder] att_context_size = {model.perception.encoder.att_context_size}")
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
    val = cfg.data.validation_ds
    base = val.copy()
    with open_dict(base):
        del base.datasets

    summaries = {"ckpt": str(args.ckpt), "att_context": list(model.perception.encoder.att_context_size)}
    for name in args.datasets:
        item = OmegaConf.merge(base, val.datasets[name])
        summaries[name] = run_dataset(name, datamodule._build_test_dataloader(item), model, args, asr)
    with open(args.out / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"\n[done] {args.out}")


if __name__ == "__main__":
    main()
