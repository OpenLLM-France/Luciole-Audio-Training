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
        if isinstance(dataset, str):
            manifest_path = dataset
            batch_size = cfg.batch_size
            name = str(i)
            dataset_prompt = cfg.user_prompt
            limit_val_batches = -1
            metric = cfg.metric
            verbose= cgf.verbose
            show_examples = cfg.show_examples
        else:
            manifest_path = dataset['manifest']
            batch_size = dataset.get("batch_size", cfg.batch_size)
            name = dataset.get("name", str(i))
            dataset_prompt = dataset.get("user_prompt", cfg.user_prompt)
            limit_val_batches = dataset.get("limit_val_batches", -1)
            metric = dataset.get("metric", cfg.metric)
            verbose = dataset.get("verbose", cfg.verbose)
            show_examples = dataset.get("show_examples", cfg.show_examples)
        print()
        print(f"Evaluating dataset {name}")
        cuts = guess_parse_cutset(manifest_path).sort_by_duration()
        dloader = torch.utils.data.DataLoader(
            dataset=ToAudio(),
            sampler=lhotse.dataset.DynamicCutSampler(cuts, max_cuts=batch_size),
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
        if dataset_prompt is not None:
            content = dataset_prompt
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
            if verbose:
                batch_wer, _, nins, ndel, nsub = word_error_rate_detail(batch_hyps, batch_refs)
                batch_rtfx = batch_duration / batch_infer_duration
                logging.info(
                    f"Batch {batch_idx}: WER={batch_wer:.2%} [ins={nins:.2%} del={ndel:.2%} sub={nsub:.2%}] RTFx={batch_rtfx:.1f}"
                )

            refs.extend(batch_refs)
            hyps.extend(batch_hyps)
            input_durations.append(batch_duration)
            infer_durations.append(batch_infer_duration)
            if limit_val_batches>0 and batch_idx+1 >= limit_val_batches:
                break
            
        results[name] = dict()
        if metric == "wer":
            wer, _, nins, ndel, nsub = word_error_rate_detail(hypotheses=hyps, references=refs, use_cer=False)
            logging.info(f"WER: {wer:.2%} [ins={nins:.2%} del={ndel:.2%} sub={nsub:.2%}]")
            results[name] = dict(wer=wer*100, nins=nins*100, ndel=ndel*100, nsub=nsub*100)
        rtfx = sum(input_durations) / sum(infer_durations)
        results[name]["rtfx"] = rtfx
        logging.info(f"RTFx: {rtfx:.1f}")
        if cfg.output_manifest is not None:
            output_folder = Path(cfg.output_folder)
            if cfg.ckpt_xp_path is not None:
                output_folder = output_folder / Path(cfg.ckpt_xp_path).name
            output_manifest = cfg.output_manifest
            if len(cfg.inputs)>1:
                output_manifest = output_manifest.replace(".jsonl", f"_{name}.jsonl")
            output_path = output_folder / Path(output_manifest)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            logging.info(f"Writing manifest to {output_path}")
            with SequentialJsonlWriter(output_path) as writer:
                for cut, ref, hyp, prompt in zip(cuts, refs, hyps, prompts):
                    writer.write({"id": cut.id, "duration": cut.duration, "text": ref, "pred_text": hyp})
        for i, (cut, ref, hyp, prompt) in enumerate(zip(cuts, refs, hyps, prompts)):
            if i+1>show_examples:
                break
            print(f"\tDATA {cut.id} ({cut.duration})")
            print(f"\t\tPrompt: {prompt}")
            print(f"\t\tRef: {ref}")
            print(f"\t\tPrediction: {hyp}")
            print()
    for result in results:
        logging.info(f"{result}: {results[result]}")
def parse_hyp(answer: torch.Tensor, eos_tokens: list[int]):
    end = (answer == torch.isin(answer, torch.tensor(eos_tokens))).nonzero(as_tuple=True)[0]
    if end.numel() == 0:
        return answer
    end = end[0]
    return answer[:end]


if __name__ == '__main__':
    main()
