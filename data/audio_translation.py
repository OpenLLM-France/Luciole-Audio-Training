import os
import random
import uuid
import re
import librosa
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import numpy as np
import soundfile as sf
import pandas as pd
from typing import List, Dict, Any
from multiprocessing import cpu_count
import json
from concurrent.futures import ThreadPoolExecutor
import glob
import argparse

# Load prompt dictionary (ensure JSON format is correct)
script_dir = os.path.dirname(os.path.abspath(__file__))
dict_prompts = os.path.join(script_dir, "assets", "instruction_transcription_and_translation_fr-en.txt")
assert os.path.exists(dict_prompts), f"File not found: {dict_prompts}"
with open(dict_prompts, 'r', encoding='utf-8') as f:
    prompts_dict = json.load(f)

# Optimized prompt access using precomputed lists
prompt_cache = {
    task: {lang: np.array(prompts) for lang, prompts in langs.items()}
    for task, langs in prompts_dict.items()
}

def create_arrow_schema():
    """Define the Arrow schema with proper audio data structure."""
    return pa.schema([
        ('messages', pa.list_(
            pa.struct([
                ('role', pa.string()),
                ('content', pa.list_(
                    pa.struct([
                        ('type', pa.string()),
                        ('text', pa.string()),
                        ('audio', pa.struct([
                            ('array', pa.list_(pa.float32())),
                            ('sampling_rate', pa.int32()),
                            ('path', pa.string())
                        ]))
                    ])
                ))
            ])
        ))
    ])

# def load_audio(audio_path: str, start_dur:float = None, end_dur: float = None sampling_rate=16000) -> tuple:
#     """Load audio file, convert to mono if needed, and resample to 16kHz.
    
#     If the file does not exist or fails to load, return (None, None).
#     """
#     if not os.path.exists(audio_path):
#         print(f"Warning: Audio file not found, skipping: {audio_path}")
#         return None, None
    
#     try:
#         array, sr = sf.read(audio_path)
#         if array.ndim > 1:  # Convert stereo to mono
#             array = array.mean(axis=1)
#         if sr != sampling_rate:  # Resample if needed
#             array = librosa.resample(array, orig_sr=sr, target_sr=sampling_rate)
#             sr = sampling_rate
#         return array, sr
#     except Exception as e:
#         print(f"Error loading audio {audio_path}: {e}, skipping this file.")
#         return None, None

def load_audio(audio_path: str, start_dur: float = None, end_dur: float = None, sampling_rate: int = 16000) -> tuple:
    """Load audio, convert to mono, resample to 16kHz, and trim by duration if specified.

    Returns (audio_array, sampling_rate) or (None, None) on failure.
    """
    if not os.path.exists(audio_path):
        print(f"Warning: Audio file not found, skipping: {audio_path}")
        return None, None

    try:
        array, sr = sf.read(audio_path)
        if array.ndim > 1:
            array = array.mean(axis=1)
        if sr != sampling_rate:
            array = librosa.resample(array, orig_sr=sr, target_sr=sampling_rate)
            sr = sampling_rate

        start_sample = int(start_dur * sr) if start_dur is not None else 0
        end_sample = int(end_dur * sr) if end_dur is not None else len(array)
        array = array[start_sample:end_sample]

        return array, sr
    except Exception as e:
        print(f"Error loading audio {audio_path}: {e}, skipping this file.")
        return None, None

def create_message_record(
    row: Dict[str, Any],
    audio_base_paths: List[str],             # ← a list of strings now with ',' to split the audios paths
) -> Dict[str, Any]:
    """Try each base path in turn; return the first valid message-record or None."""
    # parse start/end
    start_dur = float(row["start_time"]) if row.get("start_time") else None
    end_dur   = float(row["end_time"])   if row.get("end_time")   else None

    array, sr, audio_path = None, None, None

    # ──────── loop through each candidate root ────────
    for base in audio_base_paths:
        candidate = os.path.join(base, row.get("path", ""))
        arr, rate = load_audio(candidate, start_dur=start_dur, end_dur=end_dur)
        if arr is not None and rate is not None:
            array, sr, audio_path = arr, rate, candidate
            break

    # if none found, bail out
    if array is None:
        return None

    # build your messages exactly as before
    task        = "transcription_and_translation"
    lang        = "fr"
    instruction = np.random.choice(prompt_cache[task][lang])
    output      = f"{row.get('sentence','')}\n{row.get('translation','')}"

    return {
        "messages": [
            {"role": "system",
             "content": [{
                 "type": "text",
                 "text": "Vous êtes un assistant vocal utile.",
                 "audio": {"array": [], "sampling_rate": 0, "path": ""}
             }]}
          , {"role": "user",
             "content": [
               {"type": "audio", "text": "", "audio":{
                   "array": array.tolist(),
                   "sampling_rate": sr,
                   "path": audio_path
               }},
               {"type": "text", "text": instruction, "audio": {"array": [], "sampling_rate": 0, "path": ""}}
             ]}
          , {"role": "assistant",
             "content": [{
               "type": "text", "text": output,
               "audio": {"array": [], "sampling_rate": 0, "path": ""}
             }]}
        ]
    }

def process_batch(
    batch: List[Dict[str, Any]],
    audio_base_paths: List[str],
    audio_load_workers=4
) -> List[Dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=audio_load_workers) as executor:
        results = list(tqdm(
            executor.map(
                lambda row: create_message_record(row, audio_base_paths),
                batch
            ),
            total=len(batch), desc="Processing batch", leave=False
        ))
    return [r for r in results if r]

def save_shard(shard_data: List[Dict[str, Any]], output_path: str, split: str, shard_counter: int, data_id: str, parquet_prefix: str) -> None:
    """Save a single shard to a Parquet file."""
    schema = create_arrow_schema()
    filename = f"{parquet_prefix}--{data_id}--{split}--shard-{shard_counter}.parquet"
    output_file = os.path.join(output_path, filename)
    table = pa.Table.from_pylist(shard_data, schema)
    pq.write_table(table, output_file, compression='zstd')
    print(f"✅ Saved shard {shard_counter} with {len(shard_data)} records to {output_file}")

def count_processed_records(output_path: str, split: str, parquet_prefix: str) -> int:
    """
    Count the total number of records already processed in all Parquet files for a given split.
    """
    search_pattern = os.path.join(output_path, split, f"{parquet_prefix}*.parquet")
    parquet_files = glob.glob(search_pattern)
    total = 0
    for file in parquet_files:
        try:
            table = pq.read_table(file)
            total += table.num_rows
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return total

def get_last_shard_counter(output_path: str, split: str, parquet_prefix: str) -> int:
    """
    Determine the maximum shard counter from existing Parquet files for a given split.
    """
    search_pattern = os.path.join(output_path, split, f"{parquet_prefix}*.parquet")
    parquet_files = glob.glob(search_pattern)
    max_shard = 0
    pattern = re.compile(r'shard-(\d+)\.parquet$')
    for file in parquet_files:
        match = pattern.search(file)
        if match:
            shard_num = int(match.group(1))
            max_shard = max(max_shard, shard_num)
    return max_shard

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration for audio processing.")
    parser.add_argument('--max_shard_size', type=int, default=500, help='Max number of messages per Parquet file')
    parser.add_argument('--audio_load_workers', type=int, default=4, help='Number of workers to load audio')
    parser.add_argument('--sampling_rate', type=int, default=16000, help='Sampling rate for audio processing')
    parser.add_argument('--audio_base_paths', type=str, default='path/to/audios', help='Path to audio base directory')
    parser.add_argument('--output_base_path', type=str, default='path/to/save/data', help='Path to output base directory')
    parser.add_argument('--tsv_path', type=str, default='path/to/tsv_fils', help='Path to TSV files (should be like test.tsv, train.tsv, dev.tsv, valid.tsv)')
    parser.add_argument('--parquet_prefix', type=str, default='Audio--Transcription--Translation', help='Prefix for Parquet file names')
    parser.add_argument('--load_last', action='store_true', help='Resume processing by skipping records already processed in previous Parquet shards')

    args = parser.parse_args()

    # ========== Configuration ==========
    MAX_SHARD_SIZE = args.max_shard_size
    AUDIO_LOAD_WORKERS = args.audio_load_workers
    SAMPLING_RATE = args.sampling_rate
    # ===================================

    audio_base_paths = [p.strip() for p in args.audio_base_paths.split(',')]
    output_base_path = args.output_base_path
    tsv_path = args.tsv_path
    parquet_prefix = args.parquet_prefix

    # Generate a new data_id if processing new records.
    new_data_id = uuid.uuid4().hex[:8]

    for split in ['test', 'valid', 'dev', 'train']:
        
        print(f"\n{'='*40}\nProcessing {split} split\n{'='*40}")

        processed_count = 0
        shard_counter = 0

        # If resuming, count processed records and get the last shard counter.
        if args.load_last:
            processed_count = count_processed_records(output_base_path, split, parquet_prefix)
            shard_counter = get_last_shard_counter(output_base_path, split, parquet_prefix)
            print(f"Resuming split '{split}': {processed_count} records already processed in {shard_counter} shards.")

        # 1. Load data from TSV
        data_file = os.path.join(tsv_path, f'{split}.tsv')
        if not os.path.exists(data_file):
            print(f"❌ TSV file not found for split '{split}': {data_file}. Skipping...")
            continue
        dataset = pd.read_csv(data_file, sep="\t").to_dict('records')
        print(f"📂 Loaded {len(dataset)} records from {data_file}")

        # Skip already processed records
        if processed_count >= len(dataset):
            print(f"👌 All records in {data_file} have already been processed. Skipping {split} split.")
            continue
        else:
            dataset = dataset[processed_count:]
            print(f"🤏 Processing {len(dataset)} new records (skipped first {processed_count}).")
        
        # 2. Process data in batches
        batch_size = MAX_SHARD_SIZE * 2  # Process larger batches but save in shard-sized chunks
        batches = [dataset[i:i+batch_size] for i in range(0, len(dataset), batch_size)]
        
        # Create output directory for the split
        output_path = os.path.join(output_base_path, split)
        os.makedirs(output_path, exist_ok=True)
        
        # 3. Process each batch and save new shards
        for batch in tqdm(batches, desc=f"📦 Streaming {split} batches"):
            processed_data = process_batch(batch, audio_base_paths, AUDIO_LOAD_WORKERS)
            if not processed_data:
                continue
            
            # Split processed data into shards of exactly MAX_SHARD_SIZE messages
            for i in range(0, len(processed_data), MAX_SHARD_SIZE):
                shard_data = processed_data[i:i+MAX_SHARD_SIZE]
                shard_counter += 1
                save_shard(
                    shard_data, 
                    output_path, 
                    split, 
                    shard_counter, 
                    new_data_id,  # new data_id for new shards
                    parquet_prefix
                )
