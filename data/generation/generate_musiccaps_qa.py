import os
import argparse
import json
import copy
from typing import Any
from functools import partial
from pathlib import Path

from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.data import DocumentsPipeline
from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor


# Instruction:
# <instruction>
#   "french_instruction": "<instruction>",
#   "french_text": "<text>"
# .replace("<instruction>", document.metadata["conversations"][0]["value"])
  
async def simple_rollout(
    document,
    generate,
):
    template = """I give you a text which is about (caption) a music recording.
    Generate a question and an answer. The question (mood, string, where to can it be played, quality, genre, etc...) must be answered by only the text. The answer must be short, ideally one or 2 sentences.
    Then translate the text, instruction, everything you said in French. Make it fill natural, talk only about music. Avoid using "described in the text" or something like that, the word "text" is FORBIDDEN.
    
    JSON Output Format:

```json
{
  "english_question": "<question>",
  "english_answer": "<answer>",
  "french_question": "<question>",
  "french_answer": "<answer>",
}
```

Text:
<text>
"""
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": template.replace("<text>", document.text)},
                ],
            }
        ],
        "max_tokens": 2048,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    return await generate(payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path",
        type=str,
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_musiccaps",
    )
    parser.add_argument("--tp", type=int, default=1)
    args = parser.parse_args()
    
    output_path = Path(args.output_dir) / f"{args.model_name.split('/')[1]}"
    output_path.mkdir(parents=True, exist_ok=True)

    config: InferenceConfig = InferenceConfig(
        server_type="vllm",
        model_name_or_path=args.model_name,
        # temperature=0.6,
        tp=1,
        model_max_context=32768,
        max_concurrent_generations=500,
        metric_interval=120,
        model_kwargs=dict(gpu_memory_utilization=0.4),
    )

    def input_jsonl_adapter(self, data: dict, path: str, id_in_file: int | str):
        metadata = data.pop("metadata", {})
        if isinstance(metadata, str):
            import json

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
        ),
        
        InferenceRunner(
            rollout_fn=partial(simple_rollout),
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
        skip_completed=False
    )
    # inference_executor.run()

    def build_language_doc(doc, base_metadata, json_data, lang):
        metadata = copy.deepcopy(base_metadata)
        metadata["language"] = lang

        conversations = copy.deepcopy(metadata.get("conversations", []))

        if lang=="fr_caption":
            
            conversations[0] = {
                "role": "User",
                "type": "text",
                "value": json_data[f"french_instruction"]
            }

            conversations[-1] = {
                "role": "User",
                "type": "text",
                "value": json_data[f"french_text"]
            }
        
        else:
        
            conversations.insert(
                0,
                {
                    "role": "User",
                    "type": "text",
                    "value": json_data[f"english_question"] if lang == "en" else json_data[f"french_question"]
                },
            )

            conversations[-1] = {
                "role": "User",
                "type": "text",
                "value": json_data[f"english_answer"] if lang == "en" else json_data[f"french_answer"]
            }
        metadata["conversations"] = conversations

        return Document(
            id=f"{doc.id}_{lang}",
            text=doc.text,
            metadata=metadata,
        )

    def fix_data(data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        for doc in data:
            base_metadata = copy.deepcopy(doc.metadata)
            base_metadata["split"] = Path(
                base_metadata.pop("file_path", "")
            ).stem

            raw_text = base_metadata["rollout_results"][0]["text"]
            cleaned = (
                raw_text
                .replace("```json\n{", "{")
                .replace("}\n```", "}")
            )
            try:
                json_data = json.loads(cleaned)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON: {cleaned}") from e
            yield build_language_doc(doc, base_metadata, json_data, "en")
            yield build_language_doc(doc, base_metadata, json_data, "fr")
            # yield build_language_doc(doc, base_metadata, json_data, "fr_caption")
    
    def output_jsonl_adapter(self, data: dict):
        conversations = data.metadata.pop("conversations")
        return {
            "id": data.id,
            "conversations": conversations
        }
        
    
    pipeline = [
        JsonlReader(
            f"{output_path}/data",
        ),
        fix_data,
        JsonlWriter(
            f"{output_path}/data_cleaned/MusicCaps", 
            output_filename="${language}/${split}.jsonl",
            adapter=output_jsonl_adapter,
            compression=None
        )
        
    ]

    post_process = LocalPipelineExecutor(
        pipeline=pipeline,
        logging_dir=f"{output_path}/logs_cleaned",
        tasks=1,
        skip_completed=False,
        depends=inference_executor
    )
    post_process.run()
    with open(os.path.join(args.output_dir, "completed.txt"), "w") as f:
        pass