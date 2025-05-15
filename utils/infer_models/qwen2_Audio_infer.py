from tqdm import tqdm
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from datasets import Dataset
import gc
import os
import csv
from glob import glob
import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

# Data loading functions
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


# Batch processing and evaluation
def process_batch(batch_samples, device):
    """Process a batch of samples through the model"""
    import re
    batch_texts, batch_audios, references = [], [], []
    for sample in batch_samples:
        audio_info = sample.get('user_audio')
        if not audio_info or audio_info.get('array') is None:
            continue
        raw_arr = audio_info['array']
        if isinstance(raw_arr, list):
            audio_array = np.array(raw_arr, dtype=np.float32)
        else:
            audio_array = raw_arr.astype(np.float32)
        audio_data = np.ascontiguousarray(audio_array)

        conversation = [{
            "role": "user",
            "content": [
                {"type": "text", "text": 'Repeat after me in French\n'},
                {"type": "audio", "audio": audio_data},
            ]
        }]
        text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        batch_texts.append(text)
        batch_audios.append(audio_data)
        references.append(sample.get('assistant_text', ''))

    if not batch_texts:
        return [], []
    with torch.no_grad():
        inputs = processor(
            text=batch_texts,
            audio=batch_audios,
            return_tensors="pt",
            sampling_rate=16000,
            padding=True
        ).to(device)
        generated = model.generate(**inputs, max_new_tokens=150)
        generated_ids = generated[:, inputs.input_ids.size(1):]
        
    preds = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return preds, references


def evaluate_in_batches(dataset, batch_size, output_path, dataname, device):
    """Main evaluation loop"""
    os.makedirs(output_path, exist_ok=True)
    output_tsv = os.path.join(output_path, f"predictions_Qwen-audio_{dataname}.tsv")
    with open(output_tsv, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["reference", "prediction"])

    data_list = list(dataset)
    all_refs, all_preds = [], []
    for i in tqdm(range(0, len(data_list), batch_size), desc="Evaluating batches", unit="batch"):
        batch = data_list[i:i+batch_size]
        preds, refs = process_batch(batch, device)
        if not preds:
            continue
        with open(output_tsv, "a", encoding="utf-8", newline='') as f:
            writer = csv.writer(f, delimiter="\t")
            for ref, pred in zip(refs, preds):
                ref_c = ref.strip().lower()
                pred_c = pred.strip().lower()
                # pred_c = pred_c.split(':')[1]
                writer.writerow([ref_c, pred_c])
                f.flush()
                all_refs.append(ref_c)
                all_preds.append(pred_c)
    return all_refs, all_preds


# Main execution
if __name__ == "__main__":
    import argparse
    # Ensure GPU selection
    

    parser = argparse.ArgumentParser(description="Configure data source paths.")
    parser.add_argument(
        "-d", "--dataset_dirs",
        nargs='+',
        required=True,
        help="Directories containing train/test parquet splits."
    )
    parser.add_argument(
        "-bs", "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "-o", "--output_path",
        type=str,
        default="./results",
        help="Directory to save predictions TSV."
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=0,
        help="GPU to use"
    )
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    
    PARQUET_DIRS = args.dataset_dirs
    OUTPUT_PATH = args.output_path
    BATCH_SIZE = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    # Initialize processor & model
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct",
        quantization_config=quant_config,
        device_map={"": device}
    )
    model.eval()
    print("\nStarting evaluation...")
    for data_dir in PARQUET_DIRS:
        try:    
            print("Loading test dataset...")
            test_dataset = _read_parquet_without_dask(data_dir)
            print(f"Loaded {len(test_dataset)} samples")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            break
        print("\nStarting evaluation...")
        dataname = os.path.basename(data_dir.rstrip('/'))  # strip trailing slash
        print("\nStarting evaluation...")
        evaluate_in_batches(test_dataset, BATCH_SIZE, OUTPUT_PATH, dataname, device)

    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("Evaluation completed successfully!")
