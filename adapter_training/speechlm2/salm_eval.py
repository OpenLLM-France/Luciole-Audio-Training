# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from time import perf_counter
from typing import Optional, Any
import random
import json
from pathlib import Path

import lhotse.dataset
import torch
from lhotse import CutSet
from lhotse.serialization import SequentialJsonlWriter
from lhotse.dataset import IterableDatasetWrapper
from omegaconf import OmegaConf
import omegaconf
from transformers import GenerationConfig
from whisper_normalizer.english import EnglishTextNormalizer

from nemo.collections.asr.metrics.wer import word_error_rate_detail
from nemo.collections.common.data.lhotse.cutset import guess_parse_cutset
from nemo.collections.speechlm2 import SALM, SALMDataset
from nemo.core.config import hydra_runner
from nemo.utils import logging
from to_hf import load_model

class ToAudio(torch.utils.data.Dataset):
    def __getitem__(self, cuts: CutSet):
        audios, audio_lens = cuts.load_audio(collate=True)
        return {"cuts": cuts, "audios": audios, "audio_lens": audio_lens}

def get_output_manifest_path(cfg, name):
    if cfg.output_manifest is not None:
        output_folder = Path(cfg.output_folder)
        if cfg.ckpt_xp_path is not None:
            output_folder = output_folder / Path(cfg.ckpt_xp_path).name
        output_manifest = cfg.output_manifest
        if len(cfg.inputs)>1:
            output_manifest = output_manifest.replace(".jsonl", f"_{name}.jsonl")
        output_path = output_folder / Path(output_manifest)
        return output_path
    else: 
        return None

def evaluate(hyps, refs, dataset_config, results, name):
    results[name] = dict()
    if "wer" in dataset_config.metrics:
        wer, _, nins, ndel, nsub = word_error_rate_detail(hypotheses=hyps, references=refs, use_cer=False)
        logging.info(f"WER: {wer:.2%} [ins={nins:.2%} del={ndel:.2%} sub={nsub:.2%}]")
        results[name].update(dict(wer=wer*100, nins=nins*100, ndel=ndel*100, nsub=nsub*100))
    if "bleu" in dataset_config.metrics:
        from torchmetrics.text import BLEUScore
        bleu = BLEUScore()
        score = bleu(hyps, [[ref] for ref in refs]).item()*100
        results[name].update(dict(bleu=score))
        logging.info(f"BLEU: {score:.3f}")
    if "bert" in dataset_config.metrics:
        from bert_score import score
        if dataset_config.lang=="en":
            model = "~/.cache/huggingface/hub/models--google-bert--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"
        elif dataset_config.lang=="fr":
            model = "~/.cache/huggingface/hub/models--bert-base-multilingual-cased/snapshots/3f076fdb1ab68d5b2880cb87a0886f315b8146f8"
        P, R, F1 = score(hyps, refs, 
                         lang=dataset_config.lang, 
                         model_type=model,
                         num_layers=12)
        results[name].update(dict(bert_p=P.mean().item(), bert_r=R.mean().item(), bert_f1=F1.mean().item()))
    return results

def infer(model, cfg, dataset_config):
    cuts = guess_parse_cutset(dataset_config.manifest_path)#.sort_by_duration()
    dloader = torch.utils.data.DataLoader(
        dataset=ToAudio(),
        sampler=lhotse.dataset.DynamicCutSampler(cuts, max_cuts=dataset_config.batch_size),
        num_workers=1,
        batch_size=None,
    )
    if cfg.use_normalizer:
        normalizer = EnglishTextNormalizer()
    else:
        normalizer = lambda x: x

    eos_tokens = [model.text_eos_id]
    if cfg.extra_eos_tokens is not None:
        for t in cfg.extra_eos_tokens:
            tid = model.tokenizer.token_to_id(t)
            assert tid is not None, f"Token '{t}' is not in the model's vocabulary."
            eos_tokens.append(tid)

    prompt = []
    if cfg.system_prompt is not None:
        prompt.append({"role": "system", "content": cfg.system_prompt})
    # If no user prompt is provided, just use the audio placeholder.
    content = model.audio_locator_tag
    # Otherwise:
    # * if user prompt already has audio placeholder, add it as-is,
    # * if not, append audio placeholder at the end of user prompt
    if dataset_config.dataset_prompt is not None:
        content = dataset_config.dataset_prompt
        if model.audio_locator_tag not in content:
            content = f"{content} {model.audio_locator_tag}"
    prompt.append({"role": "user", "content": content})
    refs = []
    hyps = []
    input_durations = []
    infer_durations = []
    prompts = []
    for batch_idx, batch in enumerate(dloader):
        ts = perf_counter()
        batch_prompt = []
        if dataset_config.batch_size==1:
            for cut in batch["cuts"]:
                if cut.supervisions[0].custom:
                    batch_prompt.append([{"role": "user", "content": f"{cut.supervisions[0].custom.get('context', '')} {model.audio_locator_tag}"}])
                else:
                    batch_prompt.append([{"role": "user", "content": f"{model.audio_locator_tag}"}])
        else:
            batch_prompt = [prompt] * len(batch["cuts"])
        prompts.extend(batch_prompt)
        answer_ids = model.generate(
            prompts=batch_prompt,
            audios=batch["audios"].to(model.device, non_blocking=True),
            audio_lens=batch["audio_lens"].to(model.device, non_blocking=True),
            generation_config=GenerationConfig(
                max_new_tokens=cfg.max_new_tokens,
                bos_token_id=model.text_bos_id,
                eos_token_id=eos_tokens,
                pad_token_id=model.text_pad_id,
            ),
        )
        answer_ids = answer_ids.cpu()
        batch_infer_duration = perf_counter() - ts

        batch_duration = sum(c.duration for c in batch["cuts"])
        batch_refs = [normalizer(cut.supervisions[0].text) for cut in batch["cuts"]]
        batch_hyps = [
            normalizer(model.tokenizer.ids_to_text(parse_hyp(ans, eos_tokens)).strip()) for ans in answer_ids
        ]
        if dataset_config.verbose:
            batch_rtfx = batch_duration / batch_infer_duration
            if "wer" in dataset_config.metrics:
                batch_wer, _, nins, ndel, nsub = word_error_rate_detail(batch_hyps, batch_refs)
                logging.info(
                    f"Batch {batch_idx}: WER={batch_wer:.2%} [ins={nins:.2%} del={ndel:.2%} sub={nsub:.2%}] RTFx={batch_rtfx:.1f}"
                )
            else:
                logging.info(f"Batch {batch_idx}: RTFx={batch_rtfx:.1f}")

        refs.extend(batch_refs)
        hyps.extend(batch_hyps)
        input_durations.append(batch_duration)
        infer_durations.append(batch_infer_duration)
        if dataset_config.limit_val_batches>0 and batch_idx+1 >= dataset_config.limit_val_batches:
            break
    if dataset_config.output_path is not None:
        dataset_config.output_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Writing manifest to {dataset_config.output_path}")
        with SequentialJsonlWriter(dataset_config.output_path) as writer:
            for cut, ref, hyp, prompt in zip(cuts, refs, hyps, prompts):
                writer.write({"id": cut.id, "duration": cut.duration, "prompt": prompt, "text": ref, "pred_text": hyp})
    dataset_results = dict(hyps=hyps, refs=refs, durations=input_durations, infer_durations=infer_durations, prompts=prompts, ids=[cut.id for cut in cuts])
    return dataset_results

def load_result(dataset_config):
    with open(dataset_config.output_path, "r") as f:
        rows = [json.loads(line) for line in f]
    hyps = []
    refs = []
    prompts = []
    ids = []
    durations = []
    for row in rows:
        hyps.append(row["pred_text"])
        refs.append(row["text"])
        prompts.append(row["prompt"])
        ids.append(row["id"])
        durations.append(row["duration"])
    dataset_results = dict(hyps=hyps, refs=refs, prompts=prompts, ids=ids, durations=durations)
    return dataset_results

@dataclass
class SalmDatasetInfer:
    manifest_path: str
    output_path: str
    batch_size: int
    name: str
    dataset_prompt: str
    limit_val_batches: int
    metrics: Any
    verbose: bool
    show_examples: bool
    lang: str

    @classmethod
    def from_config(cls, dataset: Any, cfg: Any, i: int):
        """
        Factory to construct SalmDatasetInfer from either:
        - dataset: a string (manifest path)
        - dataset: a dict with overrides
        """

        # Case 1: dataset is a string
        if isinstance(dataset, str):
            name = str(i)
            manifest_path = dataset
            batch_size = cfg.batch_size
            dataset_prompt = cfg.user_prompt
            limit_val_batches = -1
            metrics = cfg.metric
            verbose = cfg.verbose
            show_examples = cfg.show_examples
            lang = cfg.lang

        # Case 2: dataset is a dict
        else:
            name = dataset.get("name", str(i))
            manifest_path = dataset["manifest"]
            batch_size = dataset.get("batch_size", cfg.batch_size)
            dataset_prompt = dataset.get("user_prompt", cfg.user_prompt)
            limit_val_batches = dataset.get("limit_val_batches", -1)
            metrics = dataset.get("metric", cfg.metric)
            verbose = dataset.get("verbose", cfg.verbose)
            show_examples = dataset.get("show_examples", cfg.show_examples)
            lang = dataset.get("lang", cfg.lang)

        # Normalize metrics to a list
        if isinstance(metrics, (list, tuple, omegaconf.listconfig.ListConfig)):
            metrics = list(metrics)
        else:
            metrics = [metrics]
        
        return cls(
            manifest_path=manifest_path,
            output_path=get_output_manifest_path(cfg, name),
            batch_size=batch_size,
            name=name,
            dataset_prompt=dataset_prompt,
            limit_val_batches=limit_val_batches,
            metrics=metrics,
            verbose=verbose,
            show_examples=show_examples,
            lang=lang
        )

@dataclass
class SalmEvalConfig:
    pretrained_name: Optional[str] = None
    ckpt: Optional[str] = None
    ckpt_xp_path: Optional[str] = None
    ckpt_class: Optional[str] = None
    inputs: Any = None
    batch_size: int = 64
    max_new_tokens: int = 128
    output_folder: Optional[str] = "evaluations"
    output_manifest: Optional[str] = "generations.jsonl"
    verbose: bool = True
    use_normalizer: bool = True
    device: str = "cuda"
    num_workers: int = 4
    extra_eos_tokens: Optional[list[str]] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    metric: str = "wer"
    show_examples: int = 0
    lang: str = "en"


@hydra_runner(config_path="conf", config_name="eval", schema=SalmEvalConfig)
def main(cfg: SalmEvalConfig):
    logging.info(f'Hydra config:\n{OmegaConf.to_yaml(cfg)}')

    with torch.device(cfg.device):
        torch.set_default_dtype(torch.bfloat16)
        if cfg.ckpt_xp_path is not None:
            xp_config = Path(cfg.ckpt_xp_path) / "exp_config.yaml"
            if cfg.ckpt is not None:
                ckpt_path = Path(cfg.ckpt_xp_path) / "checkpoints" / cfg.ckpt
            else:
                checkpoints_dir = Path(cfg.ckpt_xp_path) / "checkpoints"
                ckpt_candidates = list(checkpoints_dir.glob("*-last*"))
                if not ckpt_candidates:
                    raise FileNotFoundError(f"No checkpoint with '-last' found in {checkpoints_dir}")
                ckpt_path = ckpt_candidates[0]
            model = load_model(ckpt_path=ckpt_path, ckpt_class=cfg.ckpt_class, ckpt_config=xp_config)
        else:
            model = SALM.from_pretrained(cfg.pretrained_name)
        model = model.eval().to(torch.bfloat16).to(cfg.device)
        torch.set_default_dtype(torch.float32)
    results = dict()
    for i, dataset in enumerate(cfg.inputs):
        dataset_config = SalmDatasetInfer.from_config(dataset, cfg, i=i)
        print()
        print(f"Evaluating dataset {dataset_config.name}")
        if dataset_config.output_path is not None and dataset_config.output_path.exists():
            print(f"Loading dataset {dataset_config.name} results as output manifest already exists.")
            dataset_results = load_result(dataset_config)
        else:
            dataset_results = infer(model, cfg, dataset_config)
            
        results = evaluate(dataset_results["refs"], dataset_results["hyps"], dataset_config, results, dataset_config.name)
        # logging.info(f"RTFx: {rtfx:.1f}")
        for i, (id, ref, hyp, prompt) in enumerate(zip(dataset_results["ids"], dataset_results["refs"], dataset_results["hyps"], dataset_results["prompts"])):
            if i+1>dataset_config.show_examples:
                break
            print(f"\tDATA {id}")
            print(f"\t\tPrompt: {prompt}")
            print(f"\t\tRef: {ref}")
            print(f"\t\tPrediction: {hyp}")
            print()
    for result in results:
        logging.info(f"{result}: {results[result]}")
    
    output_result_path = dataset_config.output_path.parent / "results.json"
    if output_result_path.exists():
        with open(output_result_path, "r") as f:
            try:
                old_data = json.load(f)
            except json.JSONDecodeError:
                old_data = {}
    else:
        old_data = {}
    merged = {**old_data, **results}
    with open(output_result_path, "w") as f:
        json.dump(merged, f, indent=4)
    return results
    
def parse_hyp(answer: torch.Tensor, eos_tokens: list[int]):
    end = (answer == torch.isin(answer, torch.tensor(eos_tokens))).nonzero(as_tuple=True)[0]
    if end.numel() == 0:
        return answer
    end = end[0]
    return answer[:end]


if __name__ == '__main__':
    main()
