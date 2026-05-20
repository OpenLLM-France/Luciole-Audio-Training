import os
import re
import json
import argparse
from functools import partial
from pathlib import Path

from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from datatrove.data import DocumentsPipeline
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters import LambdaFilter


async def simple_rollout(
    document,
    generate,
    source_lang,
    target_lang
):
    """
    Basic rollout that sends a single request per document.

    Returns the InferenceResult directly, which will be stored under document.metadata["rollout_results"].
    """
    template = """You are a professional translator. You are given a text in {source_lang}. Translate this text in {target_lang}. 
Just give the translated text if you are sure and nothing more, make it feel natural. 

Text:
{text}
"""
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": template.format(text=document.text, source_lang=source_lang, target_lang=target_lang)},
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
        default="output_translation",
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument(
        "--source_lang",
        type=str,
        default="en",
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        default="fr",
    )
    args = parser.parse_args()
    
    source_lang = args.source_lang
    target_lang = args.target_lang
    output_path = Path(args.output_dir) / f"{args.model_name.split('/')[1]}_{source_lang}-{target_lang}"
    output_path.mkdir(parents=True, exist_ok=True)

    langage_codes = {"fr": "French", "en": "English", "it": "Italian", "de": "German", "es": "Spanish", "pt": "Portuguese", "ar": "Arabic", "nl": "Dutch"}
    
    print(f"Translating from {langage_codes[source_lang]} to {langage_codes[target_lang]}")

    config: InferenceConfig = InferenceConfig(
        server_type="vllm",
        model_name_or_path=args.model_name,
        # temperature=0.6,
        tp=1,
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
            "text": data["conversations"][-1]["value"],
            "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
            "media": data.pop("media", []),
            "metadata": metadata | data,  # remaining data goes into metadata
        }

    pipeline = [
        JsonlReader(
            f"{args.data_path}/{args.source_lang}",
            adapter=input_jsonl_adapter,
        ),
        
        InferenceRunner(
            rollout_fn=partial(simple_rollout, source_lang=langage_codes[source_lang], target_lang=langage_codes[target_lang]),
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
        skip_completed=True
    )
    def fix_data(data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        for doc in data:
            doc.metadata["split"] = Path(doc.metadata.pop("file_path", "")).stem
            doc.text = doc.metadata["rollout_results"][0]["text"]
            yield doc 
    
    def filter_data(doc):
        return doc.metadata["rollout_results"][0]["finish_reason"] == "stop"
    
    def filter_chinese(doc):
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        return len(chinese_pattern.findall(doc.text))==0
    
    def output_jsonl_adapter(self, data: dict):
        metadata = data.metadata
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                pass
        if not isinstance(metadata, dict):
            metadata = {"metadata": metadata}
        conversations = metadata.pop("conversations")
        conversations[-1]["value"] = data.text
        conversations[-1]["from"] = "Assistant"
        return {
            "id": data.id,
            "conversations": conversations
        }
        
    
    pipeline = [
        JsonlReader(
            f"{output_path}/data",
        ),
        fix_data,
        LambdaFilter(
            filter_data, 
            JsonlWriter(
                f"{output_path}/data_removed_finish_reason/CommonVoice{source_lang.upper()}2{target_lang.upper()}", 
                output_filename="${split}.jsonl",
                adapter=output_jsonl_adapter,
                compression=None
            )
        ),
        LambdaFilter(
            filter_chinese, 
            JsonlWriter(
                f"{output_path}/data_removed_chinese/CommonVoice{source_lang.upper()}2{target_lang.upper()}", 
                output_filename="${split}.jsonl",
                adapter=output_jsonl_adapter,
                compression=None
            )
        ),
        JsonlWriter(
            f"{output_path}/data_cleaned/CommonVoice{source_lang.upper()}2{target_lang.upper()}", 
            output_filename="${split}.jsonl",
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