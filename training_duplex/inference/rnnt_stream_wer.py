"""WER of the Nemotron RNNT in true streaming, on the examples of an asr_teacher_forcing.py run.

asr_teacher_forcing.py decodes the RNNT baselines offline: whole file through the encoder at the look-ahead of the
checkpoint. Here each example goes chunk by chunk through the cache-aware encoder and the RNNT state carries over
(stream_duplex.py: EncoderStream, RnntStreamer). Both are run on the same audio, with the language of the dataset
and with the "auto" prompt, and rnnt_offline_hyp is compared with the rnnt<R>_hyp of the earlier run: same text
means the same examples and the same offline baseline.

The RNNT needs no LLM: only the tokenizer and the audio come from the experiment, and the checkpoint is used to check
that its encoder is still the pretrained one. Writes <out>/<dataset>.jsonl and <out>/summary.json.
Run through rnnt_stream_wer.slurm.
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from asr_teacher_forcing import load_model, rnnt_transcribe
from stream_duplex import EncoderStream, FeatureBuffer, RnntStreamer, load_rnnt

from nemo.collections.speechlm2 import DataModule, DuplexSTTDataset


def stream_transcribe(asr, audio: np.ndarray) -> tuple[str, list[float]]:
    buf = FeatureBuffer(asr, online_normalization=False, pad_and_drop_preencoded=False)
    buf.append_audio(audio)
    enc_stream, rnnt = EncoderStream(asr.encoder), RnntStreamer(asr)
    times = []
    for chunk, chunk_len in iter(buf):
        t = time.perf_counter()
        enc_out, enc_len = enc_stream.step(chunk, chunk_len, last=buf.is_buffer_empty())
        rnnt.step(enc_out, enc_len)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return rnnt.text.strip(), times


@torch.inference_mode()
def run_dataset(name, loader, asr, context, previous, out_dir, max_examples):
    lang = name.rsplit("_", 1)[-1]
    examples = []
    for batch in loader:
        a = batch["audio_data"]
        for i, sid in enumerate(a["sample_id"]):
            examples.append((sid, a["source_audio"][i, : a["source_audio_lens"][i]].float().numpy()))
        if len(examples) >= max_examples:
            break
    examples = examples[:max_examples]
    assert [e[0] for e in examples] == [r["sample_id"] for r in previous], "not the examples of the earlier run"

    key = f"rnnt{context[1]}_hyp"
    langs = {"": lang, "_auto": "auto"}
    rows = [{"sample_id": sid, "ref": r["ref"], "rnnt_previous_hyp": r[key]} for (sid, _), r in zip(examples, previous)]
    chunk_times = []
    for suffix, code in langs.items():
        asr.set_inference_prompt(code)
        for row, (_, audio) in zip(rows, examples):
            row[f"rnnt_stream{suffix}_hyp"], times = stream_transcribe(asr, audio)
            chunk_times += times
            sig = torch.from_numpy(audio)[None].cuda()
            row[f"rnnt_offline{suffix}_hyp"] = rnnt_transcribe(asr, sig, torch.tensor([sig.shape[1]]).cuda(), code, context)[0]
    with open(out_dir / f"{name}.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ms = np.array(chunk_times) * 1000
    summary = {
        "examples": len(rows),
        "chunk_ms_p50": float(np.median(ms)),
        "chunk_ms_p99": float(np.percentile(ms, 99)),
        "offline_same_as_previous": sum(r["rnnt_offline_hyp"] == r["rnnt_previous_hyp"] for r in rows),
        "stream_same_as_offline": sum(r["rnnt_stream_hyp"] == r["rnnt_offline_hyp"] for r in rows),
        "stream_auto_same_as_offline_auto": sum(r["rnnt_stream_auto_hyp"] == r["rnnt_offline_auto_hyp"] for r in rows),
    }
    print(f"===== {name}: {summary}", flush=True)
    for r in rows[:5]:
        print(f"  REF    {r['ref']}\n  STREAM {r['rnnt_stream_hyp']}\n  OFFL   {r['rnnt_offline_hyp']}\n  AUTO   {r['rnnt_stream_auto_hyp']}\n")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp-dir", type=Path, required=True, help="experiment folder (exp_config.yaml)")
    p.add_argument("--ckpt", type=Path, required=True, help="Lightning .ckpt of that experiment")
    p.add_argument("--previous", type=Path, required=True, help="asr_teacher_forcing.py output folder of that checkpoint")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", default=["asr_fr", "asr_en"], help="validation_ds.datasets keys")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    torch.set_float32_matmul_precision("medium")
    cfg, model = load_model(args.exp_dir, args.ckpt)
    os.environ.setdefault("LLM_MODEL", cfg.model.pretrained_llm)
    context = list(cfg.model.perception.encoder.att_context_size)
    asr = load_rnnt(cfg.model.pretrained_asr, "auto", "cuda", keep_encoder=True).float()
    enc = model.perception.encoder.state_dict()
    drift = max((v.float() - enc[k].float()).abs().max().item() for k, v in asr.encoder.state_dict().items() if k in enc)
    print(f"[rnnt] {cfg.model.pretrained_asr}: context {context}, max |encoder - duplex encoder| = {drift:.2e} "
          f"(duplex copy cast to bf16 by load_model)")
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
    del model, enc
    gc.collect()
    torch.cuda.empty_cache()

    asr.encoder.set_default_att_context_size(context)
    val = cfg.data.validation_ds
    base = val.copy()
    with open_dict(base):
        del base.datasets
    summaries = {"ckpt": str(args.ckpt), "context": context, "drift": drift}
    for name in args.datasets:
        previous = [json.loads(line) for line in open(args.previous / f"{name}.jsonl")]
        item = OmegaConf.merge(base, val.datasets[name])
        loader = datamodule._build_test_dataloader(item)
        summaries[name] = run_dataset(name, loader, asr, context, previous, args.out, len(previous))
    with open(args.out / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"\n[done] {args.out}")


if __name__ == "__main__":
    main()
