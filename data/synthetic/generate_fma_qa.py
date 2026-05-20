import os
import re
import argparse
import json
import copy
import random
from functools import partial
from pathlib import Path

from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from datatrove.pipeline.filters import LambdaFilter
from datatrove.data import DocumentsPipeline
from datatrove.executor.local import LocalPipelineExecutor


PROMPT_TEMPLATE = """
You are given a music recording. 
Generate a question and the answer.
Both texts should be a sentence that is natural, clear and consice.
If you get partial in the metadata it means you have only one genre instead of all of them, adapt the sentences if this is the case.
When genre is "Spoken" it means it is an audiobook. Rewrite the genre in a nuatural way if needed, for example.

Genres: <metadata>

Answer using the following format:
Question: <question>
Answer: <answer>
"""


async def fma_qa_rollout(document, generate):
    """
    Build the prompt from the FMA prompt template with genre metadata,
    and send it to the text-only LLM.
    """
    custom_metadata = document.metadata.get("custom_metadata", {})
    genres = custom_metadata.get("genres", "")
    top_genres = custom_metadata.get("top_genres", "")

    # Randomly choose which metadata fields to include
    meta = {}
    choices = ["genres_only", "top_genres_only", "partial_genres", "partial_top_genres"]
    choice = random.choice(choices)

    if choice == "genres_only":
        if genres:
            meta["genres"] = genres
    elif choice == "top_genres_only":
        if top_genres:
            meta["top_genres"] = top_genres
    elif choice == "partial_genres":
        if isinstance(genres, list) and len(genres) > 1:
            meta["genres (partial)"] = [random.choice(genres)]
        elif genres:
            meta["genres"] = genres
    elif choice == "partial_top_genres":
        if isinstance(top_genres, list) and len(top_genres) > 1:
            meta["top_genres (partial)"] = [random.choice(top_genres)]
        elif top_genres:
            meta["top_genres"] = top_genres

    # Fallback: if meta ended up empty, include whatever is available
    if not meta:
        if genres:
            meta["genres"] = genres
        if top_genres:
            meta["top_genres"] = top_genres

    metadata_str = json.dumps(meta, indent=2)

    prompt_text = PROMPT_TEMPLATE.replace("<metadata>", metadata_str)

    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt_text,
            }
        ],
        "max_tokens": 2048,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    return await generate(payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--output_dir", type=str, default="output_fma_qa")
    parser.add_argument("--tp", type=int, default=1)
    args = parser.parse_args()

    output_path = Path(args.output_dir) / f"{args.model_name.split('/')[1]}"
    output_path.mkdir(parents=True, exist_ok=True)

    config: InferenceConfig = InferenceConfig(
        server_type="vllm",
        model_name_or_path=args.model_name,
        tp=args.tp,
        model_max_context=32768,
        max_concurrent_generations=500,
        metric_interval=120,
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
            "text": data["conversations"][-1]["value"] or "inference_pending",
            "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
            "media": data.pop("media", []),
            "metadata": metadata | data,  # remaining data goes into metadata
        }

    def has_genre_metadata(doc):
        cm = doc.metadata.get("custom_metadata", {})
        genres = cm.get("genres", "")
        top_genres = cm.get("top_genres", "")
        return (genres and genres != "[]") or (top_genres and top_genres != "[]")

    pipeline = [
        JsonlReader(
            f"{args.data_path}",
            adapter=input_jsonl_adapter,
            glob_pattern="**/*.jsonl",
        ),
        LambdaFilter(has_genre_metadata),
        InferenceRunner(
            rollout_fn=partial(fma_qa_rollout),
            config=config,
            records_per_chunk=5000,
            checkpoints_local_dir=f"{output_path}/checkpoints",
            output_writer=JsonlWriter(
                f"{output_path}/data",
                output_filename="${rank}_chunk_${chunk_index}.jsonl",
            ),
        ),
    ]

    inference_executor = LocalPipelineExecutor(
        pipeline=pipeline,
        logging_dir=f"{output_path}/logs",
        tasks=1,
        skip_completed=False,
    )
    inference_executor.run()

    def fix_data(data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        for doc in data:
            metadata = copy.deepcopy(doc.metadata)
            metadata["split"] = Path(metadata.pop("file_path", "")).stem

            raw_text = metadata["rollout_results"][0]["text"]

            # Parse "Question: ... Answer: ..." format
            q_match = re.search(r"Question:\s*(.+?)(?:\nAnswer:)", raw_text, re.DOTALL)
            a_match = re.search(r"Answer:\s*(.+)", raw_text, re.DOTALL)

            if not q_match or not a_match:
                continue

            question = q_match.group(1).strip()
            answer = a_match.group(1).strip()

            audio_turn = metadata["conversations"][0]
            metadata["conversations"] = [
                {"from": "User", "value": question, "type": "text"},
                {"from": "User", "value": audio_turn["value"], "type": "audio", "duration": audio_turn.get("duration", "")},
                {"from": "Assistant", "value": answer, "type": "text"},
            ]

            doc.metadata = metadata
            doc.text = answer
            yield doc

    def filter_data(doc):
        return doc.metadata["rollout_results"][0]["finish_reason"] == "stop"

    def output_jsonl_adapter(self, data: dict):
        conversations = data.metadata.pop("conversations")
        return {
            "id": data.id,
            "conversations": conversations,
        }

    pipeline = [
        JsonlReader(
            f"{output_path}/data",
        ),
        fix_data,
        LambdaFilter(
            filter_data,
            JsonlWriter(
                f"{output_path}/data_removed_finish_reason/FMA_QA",
                output_filename="${split}.jsonl",
                adapter=output_jsonl_adapter,
                compression=None,
            ),
        ),
        JsonlWriter(
            f"{output_path}/data_cleaned/FMA_QA",
            output_filename="${split}.jsonl",
            adapter=output_jsonl_adapter,
            compression=None,
        ),
    ]

    post_process = LocalPipelineExecutor(
        pipeline=pipeline,
        logging_dir=f"{output_path}/logs_cleaned",
        tasks=1,
        skip_completed=False,
        depends=inference_executor,
    )
    post_process.run()
    with open(os.path.join(args.output_dir, "completed.txt"), "w") as f:
        pass
