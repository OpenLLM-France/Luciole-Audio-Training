import os
import asyncio
import argparse
import json
import copy
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
    instruction = document.metadata["conversations"][0]["value"]
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
        "--dataset_name",
        type=str,
        default=None,
        help="Folder name under data_cleaned/ for the output dataset. "
        "Defaults to the basename of data_path.",
    )
    args = parser.parse_args()

    source_lang = args.source_lang
    target_lang = args.target_lang
    dataset_name = args.dataset_name or Path(args.data_path).name

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
        return {
            "text": data["conversations"][-1]["value"],
            "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
            "media": data.pop("media", []),
            "metadata": metadata | data,  # remaining data goes into metadata
        }

    pipeline = [
        JsonlReader(
            f"{args.data_path}",
            adapter=input_jsonl_adapter,
            limit=10 if args.debug else -1,
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

    def build_language_doc(doc, base_metadata, json_data):
        metadata = copy.deepcopy(base_metadata)
        metadata["language"] = target_lang

        conversations = copy.deepcopy(metadata.get("conversations", []))

        conversations[0] = {
            "from": "User",
            "type": "text",
            "value": json_data["translated_instruction"],
        }

        conversations[-1] = {
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

    def fix_data(data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        for doc in data:
            base_metadata = copy.deepcopy(doc.metadata)
            base_metadata["split"] = Path(base_metadata.pop("file_path", "")).stem

            raw_text = base_metadata["rollout_results"][0]["text"]
            cleaned = raw_text.replace("```json\n{", "{").replace("}\n```", "}")
            try:
                json_data = json.loads(cleaned)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON: {cleaned}") from e
            yield build_language_doc(doc, base_metadata, json_data)

    def output_jsonl_adapter(self, data: dict):
        conversations = data.metadata.pop("conversations")
        return {"id": data.id, "conversations": conversations}

    def filter_data(doc):
        result = doc.metadata["rollout_results"][0]
        return (
            result["finish_reason_instruction"] == "stop"
            and result["finish_reason_text"] == "stop"
        )

    pipeline = [
        JsonlReader(
            f"{output_path}/data/{dataset_name}",
        ),
        LambdaFilter(filter_data),
        fix_data,
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

    input_lines = count_jsonl_lines(Path(args.data_path))
    output_lines = count_jsonl_lines(Path(f"{output_path}/data_cleaned/{dataset_name}"))
    if args.debug:
        input_lines = min(input_lines, 10)

    if output_lines > input_lines:
        raise ValueError(
            f"Output line count ({output_lines}) exceeds input line count ({input_lines})."
        )
    elif output_lines < input_lines:
        print(
            f"WARNING: output line count ({output_lines}) is smaller than input line count "
            f"({input_lines}). Some rows may have been filtered out during generation."
        )
    else:
        print(f"Output line count matches input line count ({input_lines}).")

    with open(output_path / f"completed_{dataset_name}.txt", "w") as f:
        pass
