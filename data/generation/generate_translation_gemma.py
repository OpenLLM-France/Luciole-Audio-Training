import os
import asyncio
import argparse
import json
import copy
import re
import unicodedata
from functools import partial
from pathlib import Path

from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from datatrove.data import DocumentsPipeline
from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters import LambdaFilter


SEP = "\n[[SEP]]\n"


# Typographic normalization of the translated turns. TranslateGemma emits typographer's
# punctuation (curly quotes, ellipsis, guillemets, no-break spaces) that the source English
# data does not have, so the two languages would otherwise disagree on characters that sound
# identical. Accents and the oe/ae ligatures are French SPELLING, not typography: never touched.
def _map_to(chars, replacement):
    return {c: replacement for c in chars}


_CHAR_MAP = {
    # apostrophes / single quotes / primes / acute accent / backtick
    **_map_to("‘’‚‛′´`", "'"),
    # double quotes and guillemets
    **_map_to("“”„‟″«»", '"'),
    # hyphen variants, en/em dash, horizontal bar
    **_map_to("‐‑‒–—―", "-"),
    "…": "...",
    # presentation-form ligatures (unlike œ/æ, these are pure typography)
    "ﬁ": "fi",
    "ﬂ": "fl",
    # The invisible ones stay as \u escapes on purpose: literal forms are unreviewable, and
    # U+2028 is a LINE SEPARATOR that git and editors flag as a stray terminator in the source.
    # no-break, narrow no-break, figure, thin, hair space; line + paragraph separator
    **_map_to("\u00a0\u202f\u2007\u2009\u200a\u2028\u2029", " "),
}
# zero-width space / non-joiner / joiner, word joiner, BOM
_ZERO_WIDTH = "\u200b\u200c\u200d\u2060\ufeff"
_NORM_TABLE = str.maketrans({**_CHAR_MAP, **{c: "" for c in _ZERO_WIDTH}})

# Only horizontal runs: newlines separate paragraphs inside a turn and must survive.
_SPACE_RUN = re.compile(r"[ \t]{2,}")

# French sets the padding INSIDE guillemets (« mot »), so the pair must collapse to
# "mot", not " mot ". Runs before _NORM_TABLE, which cannot tell an opener from a closer once
# both are '"'. Unpaired guillemets fall through to the table. \s is Unicode-aware for str
# patterns, so it already covers the no-break spaces French puts there.
_GUILLEMETS = re.compile(r"«\s*(.*?)\s*»", re.DOTALL)


def normalize_text(text):
    """ASCII-ify punctuation and spacing without touching letters.

    NFC first so decomposed accents (e + U+0301) become a single é, otherwise the combining
    mark would survive as its own character downstream.
    """
    if not isinstance(text, str) or not text:
        return text
    text = unicodedata.normalize("NFC", text)
    text = _GUILLEMETS.sub(r'"\1"', text)
    text = text.translate(_NORM_TABLE)
    text = _SPACE_RUN.sub(" ", text)
    return text.strip()


def normalize_docs(data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
    """Pipeline stage: normalize every text turn of the translated conversation."""
    for doc in data:
        for turn in doc.metadata.get("conversations", []):
            if turn.get("type") == "text":
                turn["value"] = normalize_text(turn.get("value"))
        yield doc


def _find_turn(conversations, from_role, turn_type="text"):
    for i, turn in enumerate(conversations):
        if turn.get("from", "").lower() == from_role.lower() and turn.get("type") == turn_type:
            return i, turn
    raise ValueError(f"No turn with from={from_role!r} and type={turn_type!r}")


def _translate_payload(text, source_lang, target_lang, max_tokens=512):
    return {
        "messages": [
            {
                "role": "user",
                "content": f"<<<source>>>{source_lang}<<<target>>>{target_lang}<<<text>>>{text}",
            }
        ],
        "max_tokens": max_tokens,
    }


async def simple_rollout(
    document,
    generate,
    source_lang,
    target_lang,
):
    _, user_turn = _find_turn(document.metadata["conversations"], "User")
    instruction = user_turn["value"]
    text = document.text

    combined = f"{instruction}{SEP}{text}"
    result = await generate(
        _translate_payload(combined, source_lang, target_lang, max_tokens=1024)
    )

    parts = result.text.split(SEP, 1)
    if len(parts) == 2 and result.finish_reason == "stop":
        translated_instruction, translated_text = parts[0].strip(), parts[1].strip()
        finish_reason_instruction = finish_reason_text = result.finish_reason
    else:
        # Fallback: separator drifted or generation truncated — re-translate independently.
        result_instruction, result_text = await asyncio.gather(
            generate(_translate_payload(instruction, source_lang, target_lang)),
            generate(_translate_payload(text, source_lang, target_lang)),
        )
        translated_instruction = result_instruction.text
        translated_text = result_text.text
        finish_reason_instruction = result_instruction.finish_reason
        finish_reason_text = result_text.finish_reason

    json_str = json.dumps(
        {
            "translated_instruction": translated_instruction,
            "translated_text": translated_text,
        }
    )
    return {
        "text": json_str,
        "finish_reason_instruction": finish_reason_instruction,
        "finish_reason_text": finish_reason_text,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path",
        type=str,
    )
    # parser.add_argument(
    #     "--model_name", type=str, default="Infomaniak-AI/vllm-translategemma-4b-it"
    # )
    parser.add_argument("--model_name", type=str, default="Infomaniak-AI/vllm-translategemma-27b-it")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_translation_gemma",
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--source_lang", type=str, default="en")
    parser.add_argument("--target_lang", type=str, default="fr")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: only process the first 10 rows.",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Skip the typographic normalization stage (curly quotes, ellipsis, "
        "no-break spaces, ... -> their ASCII equivalents). See normalize_text.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Folder name under data_cleaned/ for the output dataset. "
        "Defaults to the basename of data_path.",
    )
    args = parser.parse_args()

    source_lang = args.source_lang
    target_lang = args.target_lang
    data_root = Path(args.data_path).resolve()
    base_name = args.dataset_name or data_root.name

    output_path = (
        Path(args.output_dir)
        / f"{args.model_name.split('/')[1]}_{source_lang}-{target_lang}"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    config: InferenceConfig = InferenceConfig(
        server_type="vllm",
        model_name_or_path=args.model_name,
        # temperature=0.6,
        tp=args.tp,
        model_max_context=2048,
        max_concurrent_generations=500,
        metric_interval=120,
        model_kwargs=dict(gpu_memory_utilization=0.6),
    )

    def input_jsonl_adapter(self, data: dict, path: str, id_in_file: int | str):
        metadata = data.pop("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                pass
        if not isinstance(metadata, dict):
            metadata = {"metadata": metadata}
        _, assistant_turn = _find_turn(data["conversations"], "Assistant")
        return {
            "text": assistant_turn["value"],
            "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
            "media": data.pop("media", []),
            "metadata": metadata | data,  # remaining data goes into metadata
        }

    jsonl_files = sorted(data_root.rglob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found under {data_root}")

    sub_datasets = []
    for jsonl_file in jsonl_files:
        rel = jsonl_file.parent.relative_to(data_root)
        sub_datasets.append((str(jsonl_file.parent), str(Path(base_name) / rel)))

    # Deduplicate (multiple jsonl files in same folder)
    sub_datasets = list(dict(sub_datasets).items())

    def build_language_doc(doc, base_metadata, json_data):
        metadata = copy.deepcopy(base_metadata)
        metadata["language"] = target_lang

        conversations = copy.deepcopy(metadata.get("conversations", []))

        user_idx, _ = _find_turn(conversations, "User")
        conversations[user_idx] = {
            "from": "User",
            "type": "text",
            "value": json_data["translated_instruction"],
        }

        assistant_idx, _ = _find_turn(conversations, "Assistant")
        conversations[assistant_idx] = {
            "from": "Assistant",
            "type": "text",
            "value": json_data["translated_text"],
        }

        metadata["conversations"] = conversations

        return Document(
            id=f"{doc.id}_{target_lang}",
            text=doc.text,
            metadata=metadata,
        )

    def output_jsonl_adapter(self, data: dict):
        conversations = data.metadata.pop("conversations")
        out = {"id": data.id, "conversations": conversations}
        for key in ("dataset_name",):
            if key in data.metadata:
                out[key] = data.metadata[key]
        return out

    def filter_data(doc):
        result = doc.metadata["rollout_results"][0]
        return (
            result["finish_reason_instruction"] == "stop"
            and result["finish_reason_text"] == "stop"
        )

    def count_jsonl_lines(path: Path) -> int:
        if path.is_file():
            files = [path]
        else:
            files = list(path.rglob("*.jsonl"))
        total = 0
        for f in files:
            with open(f, "r") as fh:
                total += sum(1 for _ in fh)
        return total

    for sub_path, dataset_name in sub_datasets:
        print(f"\n{'='*60}\nProcessing {dataset_name} ({sub_path})\n{'='*60}")

        if (output_path / f"completed_{dataset_name.replace('/', '_')}.txt").exists():
            print(f"Skipping {dataset_name} (already completed)")
            continue

        pipeline = [
            JsonlReader(
                sub_path,
                adapter=input_jsonl_adapter,
                limit=10 if args.debug else -1,
                recursive=False,
            ),
            LambdaFilter(
                lambda doc: not any(
                    tag in Path(doc.metadata.get("file_path", "")).name
                    for tag in ("randomorder", "orig")
                )
            ),
            InferenceRunner(
                rollout_fn=partial(
                    simple_rollout, source_lang=source_lang, target_lang=target_lang
                ),
                config=config,
                records_per_chunk=5000,
                checkpoints_local_dir=f"{output_path}/checkpoints/{dataset_name}",
                output_writer=JsonlWriter(
                    f"{output_path}/data/{dataset_name}",
                    output_filename="${rank}_chunk_${chunk_index}.jsonl",
                ),
            ),
        ]

        inference_executor = LocalPipelineExecutor(
            pipeline=pipeline,
            logging_dir=f"{output_path}/logs/{dataset_name}",
            tasks=1,
            skip_completed=False,
        )
        inference_executor.run()

        def fix_data(data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
            for doc in data:
                base_metadata = copy.deepcopy(doc.metadata)
                base_metadata.pop("file_path", None)
                base_metadata["split"] = Path(doc.metadata.get("file_path", "")).stem

                raw_text = base_metadata["rollout_results"][0]["text"]
                cleaned = raw_text.replace("```json\n{", "{").replace("}\n```", "}")
                try:
                    json_data = json.loads(cleaned)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSON: {cleaned}") from e
                yield build_language_doc(doc, base_metadata, json_data)

        pipeline = [
            JsonlReader(
                f"{output_path}/data/{dataset_name}",
            ),
            LambdaFilter(filter_data),
            fix_data,
            *([] if args.no_normalize else [normalize_docs]),
            JsonlWriter(
                f"{output_path}/data_cleaned/{dataset_name}",
                output_filename="${split}.jsonl",
                adapter=output_jsonl_adapter,
                compression=None,
            ),
        ]

        post_process = LocalPipelineExecutor(
            pipeline=pipeline,
            logging_dir=f"{output_path}/logs_cleaned/{dataset_name}",
            tasks=1,
            skip_completed=False,
            depends=inference_executor,
        )
        post_process.run()

        input_lines = count_jsonl_lines(Path(sub_path))
        output_lines = count_jsonl_lines(Path(f"{output_path}/data_cleaned/{dataset_name}"))
        if args.debug:
            input_lines = min(input_lines, 10)

        if output_lines > input_lines:
            raise ValueError(
                f"[{dataset_name}] Output line count ({output_lines}) exceeds "
                f"input line count ({input_lines})."
            )
        elif output_lines < input_lines:
            print(
                f"WARNING: [{dataset_name}] output line count ({output_lines}) is smaller than "
                f"input line count ({input_lines}). Some rows may have been filtered out."
            )
        else:
            print(f"[{dataset_name}] Output line count matches input ({input_lines}).")

        with open(output_path / f"completed_{dataset_name.replace('/', '_')}.txt", "w") as f:
            pass
