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
audio_question_instructions = [
    "Listen to the audio question and provide a clear and concise answer.",
    "Transcribe the question from the audio, then answer it accurately.",
    "Based on the audio input, extract the question and respond to it.",
    "Given an audio recording of a user question, provide the best answer.",
    "Understand the spoken question in the audio and reply accordingly.",
    "Process the user's voice question and generate a relevant answer.",
    "Answer the question provided in the audio clip using natural language understanding.",
    "Transcribe the audio to text and answer the spoken question.",
    "Analyze the audio question and generate an appropriate response.",
    "You're given an audio file with a question. Convert it to text and respond with the answer."
]


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

def load_audio(audio_path: str, sampling_rate=16000) -> tuple:
    """Load audio file, convert to mono if needed, and resample to 16kHz.
    
    If the file does not exist or fails to load, return (None, None).
    """
    if not os.path.exists(audio_path):
        print(f"Warning: Audio file not found, skipping: {audio_path}")
        return None, None
    
    try:
        array, sr = sf.read(audio_path)
        if array.ndim > 1:  # Convert stereo to mono
            array = array.mean(axis=1)
        if sr != sampling_rate:  # Resample if needed
            array = librosa.resample(array, orig_sr=sr, target_sr=sampling_rate)
            sr = sampling_rate
        return array, sr
    except Exception as e:
        print(f"Error loading audio {audio_path}: {e}, skipping this file.")
        return None, None

def create_message_record(row: Dict[str, Any], audio_base_path: str) -> Dict[str, Any]:
    """Create a properly structured message record with audio data."""
    audio_path = os.path.join(audio_base_path, row.get("path", ""))
    
    array, sr = load_audio(audio_path, 16000)
    # Skip record if audio could not be loaded.
    if array is None or sr is None:
        return None

    instruction = np.random.choice(audio_question_instructions)
    output = f"👂 You asked: {row.get('speech_instruct', '')}\n💬 Here's what I think: {row.get('output', '')}"

    return {
        "messages": [
            {"role": "system",
             "content": [{
                 "type": "text",
                 "text": "Answer the user's question, which was originally asked in audio and transcribed into text.",
                 "audio": {"array": [], "sampling_rate": 0, "path": ""}
             }]
            },
            {"role": "user",
             "content": [
                 {
                     "type": "audio",
                     "text": "",
                     "audio": {
                         "array": array.tolist(),
                         "sampling_rate": sr,
                         "path": audio_path
                     }
                 },
                 {
                     "type": "text",
                     "text": instruction,
                     "audio": {"array": [], "sampling_rate": 0, "path": ""}
                 }
             ]
            },
            {"role": "assistant",
             "content": [{
                 "type": "text",
                 "text": output,
                 "audio": {"array": [], "sampling_rate": 0, "path": ""}
             }]
            }
        ]
    }

def process_batch(batch: List[Dict[str, Any]], audio_base_path: str, audio_load_workers=4) -> List[Dict[str, Any]]:
    """Process a batch of records in parallel."""
    processed = []
    with ThreadPoolExecutor(max_workers=audio_load_workers) as executor:
        results = list(tqdm(executor.map(lambda row: create_message_record(row, audio_base_path), batch),
                            total=len(batch), desc="Processing batch", leave=False))
    
    for record in results:
        if record:
            processed.append(record)
    return processed

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
    parser.add_argument('--audio_base_path', type=str, default='path/to/audio', help='Path to audio base directory')
    parser.add_argument('--output_base_path', type=str, default='path/to/output', help='Path to output base directory')
    parser.add_argument('--file_path', type=str, default='path/to/data/file.Json', help='Path to JSON files')
    parser.add_argument('--parquet_prefix', type=str, default='Audio--Transcription--Translation', help='Prefix for Parquet file names')
    parser.add_argument('--load_last', action='store_true', help='Resume processing by skipping records already processed in previous Parquet shards')

    args = parser.parse_args()

    # ========== Configuration ==========
    MAX_SHARD_SIZE = args.max_shard_size
    AUDIO_LOAD_WORKERS = args.audio_load_workers
    SAMPLING_RATE = args.sampling_rate
    # ===================================

    audio_base_path = args.audio_base_path
    output_base_path = args.output_base_path
    file_path = args.file_path
    parquet_prefix = args.parquet_prefix

    # Generate a new data_id if processing new records.
    new_data_id = uuid.uuid4().hex[:8]

    for split in ['test', 'dev', 'train']:
        print(f"\n{'='*40}\nProcessing {split} split\n{'='*40}")

        processed_count = 0
        shard_counter = 0

        # If resuming, count processed records and get the last shard counter.
        if args.load_last:
            processed_count = count_processed_records(output_base_path, split, parquet_prefix)
            shard_counter = get_last_shard_counter(output_base_path, split, parquet_prefix)
            print(f"Resuming split '{split}': {processed_count} records already processed in {shard_counter} shards.")

        # 1. Load data from TSV
        data_file = os.path.join(file_path, f'{split}.json')
        with open("your_file.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
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
            processed_data = process_batch(batch, audio_base_path, AUDIO_LOAD_WORKERS)
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
