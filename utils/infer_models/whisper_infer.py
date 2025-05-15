
import os
import gc
import csv
import argparse
import jiwer
import numpy as np
import torch
# import dask.dataframe as dd
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
from datasets import Dataset, Audio
# from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import pandas as pd
import pyarrow.parquet as pq
import csv
from glob import glob
# from dask.diagnostics import ProgressBar

# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Dynamic padding collator for Whisper inputs"""
    processor: Any
    max_length: int = 448  # Optimal for Whisper memory usage

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Efficient batch processing with chunking
        input_features = [{"input_features": f["input_features"][:self.max_length]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Label processing with smart truncation
        label_features = [{"input_ids": f["labels"][:self.max_length]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # Mask labels and remove BOS token if present
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if labels.size(1) > 0 and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def convert_parquet_messages(messages_df, split):
    """Convert Parquet messages to structured format with memory optimization"""
    output_data = {"user_text": [], "user_audio": [], "system_text": [], "assistant_text": []}
    chunk_size = 100
    for start in range(0, len(messages_df), chunk_size):
        chunk = messages_df.iloc[start:start+chunk_size]
        for _, row in chunk.iterrows():
            message_dict = {k: None for k in output_data}
            messages = row.get('messages', [])
            if isinstance(messages, (list, np.ndarray)):
                for msg in messages:
                    if not isinstance(msg, dict):
                        continue
                    role = msg.get('role')
                    contents = msg.get('content', [])
                    if role == 'system':
                        text = next((c['text'] for c in contents if c.get('type') == 'text'), None)
                        message_dict['system_text'] = text
                    elif role == 'user':
                        text = next((c['text'] for c in contents if c.get('type') == 'text'), None)
                        message_dict['user_text'] = text
                        audio_item = next((c for c in contents if c.get('type') == 'audio'), None)
                        if audio_item:
                            # Convert audio contents to numpy array
                            arr_data, sr, path = None, None, None
                            if audio_item.get('array') is not None:
                                arr_data = np.array(audio_item['array'], dtype=np.float16)
                                sr = audio_item.get('sampling_rate')
                                path = audio_item.get('path')
                            elif audio_item.get('audio') is not None:
                                arr_data = np.array(audio_item['audio']['array'], dtype=np.float16)
                                sr = audio_item['audio'].get('sampling_rate')
                                path = audio_item['audio'].get('path')
                            message_dict['user_audio'] = {
                                "array": arr_data,
                                "sampling_rate": sr,
                                "path": os.path.basename(path) if path else None
                            }
                    elif role == 'assistant':
                        text = next((c['text'] for c in contents if c.get('type') == 'text'), None)
                        message_dict['assistant_text'] = text
            for k in output_data:
                output_data[k].append(message_dict[k])
    return output_data


def _read_parquet_without_dask(parquet_dirs, split='test') -> Dataset:
    """Load parquet files with memory optimization"""
    paths = []
    if isinstance(parquet_dirs, str):
        parquet_dirs = [parquet_dirs]
    for d in parquet_dirs:
        split_dir = os.path.join(d, split)
        if os.path.exists(split_dir):
            paths.extend(glob(os.path.join(split_dir, '*.parquet')))
    if not paths:
        raise FileNotFoundError(f"No parquet files for split {split}")

    result = {"user_text": [], "user_audio": [], "system_text": [], "assistant_text": []}
    for path in tqdm(paths, desc=f"Processing {split} parquet files", unit="file"):
        table = pq.read_table(path, columns=['messages'])
        df = table.to_pandas()
        del table
        part = convert_parquet_messages(df, split)
        for k in result:
            result[k].extend(part[k])
        del df, part
        gc.collect()

    return Dataset.from_dict(result)

def load_parquet_dataset(parquet_dirs, split='test') -> Dataset:
    """Load and filter Parquet dataset with audio/text entries."""
    dataset = _read_parquet_without_dask(parquet_dirs, split)
    
    # Filter out samples without audio or text
    dataset = dataset.filter(lambda x: x["user_audio"] is not None and x["assistant_text"] is not None)
    return dataset

def load_model(model_path: str, device: torch.device) -> AutoModelForSpeechSeq2Seq:
    """Enhanced model loading with PEFT and quantization support"""
    from peft import PeftModel, PeftConfig
    
    try:
        if os.path.isdir(model_path):
            for root, _, files in os.walk(model_path):
                if 'adapter_config.json' in files:
                    peft_config = PeftConfig.from_pretrained(root)
                    model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        peft_config.base_model_name_or_path,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        attn_implementation="flash_attention_2"
                    )
                    return PeftModel.from_pretrained(model, root).eval()
        
        return AutoModelForSpeechSeq2Seq.from_pretrained(
                model_path,
                device_map="auto" if device.type == "cuda" else None,
                local_files_only=True
            ).eval().to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")

@torch.inference_mode()
def whisper_infer(
    model: AutoModelForSpeechSeq2Seq,
    dataloader: DataLoader,
    processor: AutoProcessor,
    forced_decoder_ids: List[List[int]],
    device: torch.device,
    output_tsv_path: str
) -> tuple:
    """Optimized inference with memory management + writes results to .tsv"""
    pred, ref = [], []

    # Prepare TSV file
    with open(output_tsv_path, mode="w", newline='', encoding='utf-8') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter="\t")
        tsv_writer.writerow(["reference", "prediction"])  # header

        # Inference loop
        for batch in tqdm(dataloader, desc="Processing batches"):
            try:
                input_features = batch["input_features"].to(device, non_blocking=True)
                decoder_input = batch["labels"][:, :4].to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type), torch.backends.cuda.sdp_kernel(enable_flash=True):
                    generated_tokens = model.generate(
                        input_features=input_features,
                        decoder_input_ids=decoder_input,
                        forced_decoder_ids=forced_decoder_ids,
                        max_new_tokens=150,
                    )

                labels = batch["labels"].cpu().numpy()
                labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)

                batch_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                batch_refs = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

                # Write batch to TSV
                for r, p in zip(batch_refs, batch_preds):
                    tsv_writer.writerow([r.strip(), p.strip()])

                pred.extend(batch_preds)
                ref.extend(batch_refs)

            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    torch.cuda.empty_cache()
                    gc.collect()
                    print("Skipping batch due to OOM error")
                    continue
                else:
                    raise

            del input_features, decoder_input, batch
            torch.cuda.empty_cache()

    wer = jiwer.wer(
        [jiwer.RemovePunctuation()(s.lower()) for s in ref],
        [jiwer.RemovePunctuation()(s.lower()) for s in pred]
    ) * 100

    return {"eval/wer": wer}, pred, ref

def process_dataset(
    processor: AutoProcessor,
    dataset: Dataset,
    input_column: str = "user_audio",
    text_column: str = "assistant_text"
) -> Dataset:
    """Batched dataset processing with memory mapping"""
    def preprocess(batch):
        arrays = [x["array"] for x in batch[input_column]]
        sampling_rates = [x["sampling_rate"] for x in batch[input_column]]
        
        # Batch audio processing
        input_features = [
            processor.feature_extractor(
                arr, 
                sampling_rate=sr,
                return_tensors="pt"
            ).input_features[0].numpy()
            for arr, sr in zip(arrays, sampling_rates)
        ]
        
        # Batch text processing
        labels = processor.tokenizer(
            batch[text_column],
            padding=False,
            truncation=True
        ).input_ids
        
        return {"input_features": input_features, "labels": labels}

    return dataset.map(
        preprocess,
        batched=True,
        batch_size=100,
        remove_columns=dataset.column_names,
        num_proc=1,
        desc="Preprocessing dataset",
        load_from_cache_file=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper Parquet Inference")
    parser.add_argument("parquet_dirs", nargs="+", help="Paths to Parquet directories")
    parser.add_argument("model_path", help="Model path or size (e.g. openai/whisper-large-v3)")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--language", "-l", default="fr", help="Target language")
    parser.add_argument("--task", "-t", default="transcribe", choices=["transcribe", "translate"])
    parser.add_argument("--batch_size", "-bs", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=min(os.cpu_count(), 16))
    parser.add_argument("--output", "-o", default="results")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Load processor and forced_decoder_ids
        processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)

        # Load model
        model = load_model(args.model_path, device)

        # Load and process dataset
        raw_dataset = load_parquet_dataset(args.parquet_dirs, split=args.split)
        dataset = process_dataset(processor, raw_dataset)

        # Create DataLoader
        collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size,
            collate_fn=collator,
            num_workers=args.num_workers
        )

        # Run inference
        output_tsv = os.path.join(args.output, "batch_results.tsv")
        metrics, preds, refs = whisper_infer(
            model=model,
            dataloader=dataloader,
            processor=processor,
            forced_decoder_ids=forced_decoder_ids,
            device=device,
            output_tsv_path=output_tsv
        )
        print(f"WER: {metrics['eval/wer']:.2f}%")

    except Exception as e:
        print(f"Error occurred: {e}")