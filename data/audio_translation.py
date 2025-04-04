import os
import random
import uuid
import librosa  # Added this import
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

# Load prompt dictionary (ensure JSON format is correct)
script_dir = os.path.dirname(os.path.abspath(__file__))
dict_prompts = os.path.join(
    script_dir, 
    "assets", 
    "instruction_transcription_and_translation_fr-en.txt"
    )
assert os.path.exists(dict_prompts), f"File not found: {dict_prompts}"
with open(dict_prompts, 'r', encoding='utf-8') as f:
    prompts_dict = json.load(f)

# Optimized prompt access using precomputed lists
prompt_cache = {
    task: {lang: np.array(prompts) for lang, prompts in langs.items()}
    for task, langs in prompts_dict.items()
}

def create_arrow_schema():
    """Define the Arrow schema with proper audio data structure"""
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
    """Load audio file, convert to mono if needed, and resample to 16kHz."""
    try:
        array, sr = sf.read(audio_path)
        if array.ndim > 1:  # Convert stereo to mono
            array = array.mean(axis=1)
        if sr != sampling_rate:  # Resample if the sampling rate is not 16kHz
            array = librosa.resample(array, orig_sr=sr, target_sr=sampling_rate)
            sr = sampling_rate
        return array, sr
    except Exception as e:
        print(f"Error loading audio {audio_path}: {str(e)}")
        raise

def create_message_record(row: Dict[str, Any], audio_base_path: str) -> Dict[str, Any]:
    """Create a properly structured message record with audio data"""
    audio_path = os.path.join(audio_base_path, row.get("path", ""))
    try:
        array, sr = load_audio(audio_path, 16000)
        task = "transcription_and_translation"
        lang = "fr"
        instruction = np.random.choice(prompt_cache[task][lang])
        output = f"{row.get('sentence', '')}\n{row.get('translation', '')}"

        return {
            "messages": [
                {"role": "system",
                 "content": [{
                     "type": "text",
                     "text": "You are a helpful speech assistant." if lang == "en" 
                             else "Vous êtes un assistant vocal utile.",
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
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None

def process_batch(batch: List[Dict[str, Any]], audio_base_path: str, audio_load_workers = 4) -> List[Dict[str, Any]]:
    """Process a batch of records in parallel"""
    processed = []
    with ThreadPoolExecutor(max_workers=audio_load_workers) as executor:
        results = list(tqdm(executor.map(lambda row: create_message_record(row, audio_base_path), batch),
                            total=len(batch), desc="Processing batch", leave=False))
    
    for record in results:
        if record:
            processed.append(record)
    return processed

def save_shard(shard_data: List[Dict[str, Any]], output_path: str, split: str, shard_counter: int, data_id: str, parquet_prefix: str) -> None:
    """Save a single shard to a Parquet file"""
    schema = create_arrow_schema()
    filename = f"{parquet_prefix}--{data_id}--{split}--shard-{shard_counter}.parquet"
    output_file = os.path.join(output_path, filename)
    table = pa.Table.from_pylist(shard_data, schema)
    pq.write_table(table, output_file, compression='zstd')
    print(f"✅ Saved shard {shard_counter} with {len(shard_data)} records to {output_file}")

if __name__ == '__main__':
    import argparse     
    parser = argparse.ArgumentParser(description="Configuration for audio processing.")
    parser.add_argument('--max_shard_size', type=int, default=500, help='Max number of messages per Parquet file')
    parser.add_argument('--audio_load_workers', type=int, default=4, help='Number of workers to load audio')
    parser.add_argument('--sampling_rate', type=int, default=16000, help='Sampling rate for audio processing')    
    parser.add_argument('--audio_base_path', type=str, default='/home/hnaoura/hnaouara/CommonVoice/cv-corpus-18.0-2024-06-14/fr/clips', help='Path to audio base directory')
    parser.add_argument('--output_base_path', type=str, default='/home/hnaoura/hnaouara/Audio_corpus/En-Fr/audio-translation/both', help='Path to output base directory')
    parser.add_argument('--tsv_path', type=str, default='/home/hnaoura/hnaouara/CV/', help='Path to TSV files')
    parser.add_argument('--parquet_prefix', type=str, default='Audio--Transcription--Translation', help='Prefix for Parquet file names')

    args = parser.parse_args()

    # ========== Configuration ==========
    MAX_SHARD_SIZE = args.max_shard_size
    AUDIO_LOAD_WORKERS = args.audio_load_workers
    SAMPLING_RATE = args.sampling_rate
    # ===================================
    
    audio_base_path = args.audio_base_path
    output_base_path = args.output_base_path
    tsv_path = args.tsv_path
    parquet_prefix = args.parquet_prefix
    
    data_id = uuid.uuid4().hex[:8]  # Unique ID for this dataset
    
    shard_counter = 0  # Track global shard count

    for split in ['dev', 'train']: # 'test', 
        print(f"\n{'='*40}\nProcessing {split} split\n{'='*40}")
        
        # 1. Load data
        data_file = os.path.join(tsv_path, f'{split}.tsv')
        dataset = pd.read_csv(data_file, sep="\t").to_dict('records')
        print(f"📂 Loaded {len(dataset)} records from {data_file}")
        
        # 2. Process data
        batch_size = MAX_SHARD_SIZE * 2  # Process larger batches but save in 200-message chunks
        batches = [dataset[i:i+batch_size] for i in range(0, len(dataset), batch_size)]
        
        # Create output directory
        output_path = os.path.join(output_base_path, split)
        os.makedirs(output_path, exist_ok=True)
        
        # 3. Process each batch and save
        for batch in tqdm(batches, desc=f"📦 Streaming {split} batches"):
            processed_data = process_batch(batch, audio_base_path, AUDIO_LOAD_WORKERS)
            if not processed_data:
                continue
            
            # Split processed data into shards of exactly 200 messages
            for i in range(0, len(processed_data), MAX_SHARD_SIZE):
                shard_data = processed_data[i:i+MAX_SHARD_SIZE]
                shard_counter += 1
                save_shard(
                    shard_data, 
                    output_path, 
                    split, 
                    shard_counter, 
                    data_id, 
                    parquet_prefix
                    )
