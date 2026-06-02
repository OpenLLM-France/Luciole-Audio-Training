import os
import json
import uuid
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import multiprocessing
import hashlib
import pickle

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from tqdm import tqdm


def set_resource_limits():
    """Set environment variables to limit thread usage and prevent resource exhaustion."""
    # Limit OpenBLAS threads
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # Set numpy to use single thread
    os.environ['NUMPY_NUM_THREADS'] = '1'
    
    # Disable OpenBLAS threading completely
    os.environ['GOTO_NUM_THREADS'] = '1'


def convert_parquet_messages_optimized(parquet_file):
    """Optimized version with reduced memory allocation and faster processing."""
    # Set thread limits in each process
    set_resource_limits()
    
    table = pq.read_table(parquet_file)
    messages_df = table.to_pandas()

    output_data = []
    
    # Pre-allocate and reuse objects to reduce memory allocation
    for _, row in messages_df.iterrows():
        messages = row.iloc[0]
        if not isinstance(messages, np.ndarray):
            continue
            
        structured_messages = []
        
        for message in messages:
            if not isinstance(message, dict):
                continue
                
            role = message.get("role")
            content_list = message.get("content", [])
            
            if role == "system":
                text = next((item.get("text") for item in content_list if item.get("type") == "text"), None)
                if text:
                    structured_messages.append({
                        "role": "system",
                        "content": [{"type": "text", "text": text}]
                    })
            
            elif role == "user":
                audio_item = next((item for item in content_list if item.get("type") == "audio"), None)
                text_item = next((item for item in content_list if item.get("type") == "text"), None)
                
                # Optimized audio processing
                audio_array = None
                sampling_rate = 16000
                audio_path = None
                
                if audio_item:
                    if audio_item.get("array") is not None:
                        audio_array = audio_item.get("array")
                        sampling_rate = int(audio_item.get("sampling_rate", 16000))
                        audio_path = audio_item.get("path")
                    elif isinstance(audio_item.get("audio"), dict):
                        audio = audio_item["audio"]
                        audio_array = np.array(audio["array"], dtype=np.float32)
                        sampling_rate = int(audio.get("sampling_rate", 16000))
                        audio_path = audio.get("path")
                
                structured_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "audio", "array": audio_array, "path": audio_path, "sampling_rate": sampling_rate},
                        {"type": "text", "text": text_item.get("text") if text_item else None}
                    ]
                })
            
            elif role == "assistant":
                output_text = next((item.get("text") for item in content_list if item.get("type") == "text"), None)
                if output_text:
                    structured_messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": output_text}]
                    })
        
        if structured_messages:
            output_data.append(structured_messages)
    
    return output_data


def process_single_parquet_file(args):
    """Process a single parquet file - designed for multiprocessing."""
    parquet_file, audio_dir, temp_dir, prefix = args
    
    # Set resource limits in worker process
    set_resource_limits()
    
    try:
        dialogs = convert_parquet_messages_optimized(parquet_file)
    except Exception as e:
        return [], f"Failed to parse {parquet_file.name}: {e}"
    
    entries = []
    audio_count = 0
    errors = []
    
    for i, dialog in enumerate(dialogs):
        try:
            user = next((m for m in dialog if m["role"] == "user"), None)
            assistant = next((m for m in dialog if m["role"] == "assistant"), None)
            
            if not user or not assistant:
                continue
            
            audio_obj = next((c for c in user["content"] if c["type"] == "audio"), None)
            text_prompt = next((c for c in user["content"] if c["type"] == "text"), "")
            assistant_text = next((c for c in assistant["content"] if c["type"] == "text"), None)
            
            if (audio_obj is None or audio_obj.get("array") is None or 
                assistant_text is None or assistant_text.get("text") is None):
                continue
            
            audio_array = audio_obj["array"]
            sampling_rate = int(audio_obj.get("sampling_rate", 16000))
            audio_filename = f"{prefix}--{hashmd5(audio_array)}.wav"
            audio_filepath = audio_dir / audio_filename
            
            # Write audio file
            sf.write(audio_filepath, audio_array, sampling_rate)
            audio_count += 1
            
            context = text_prompt.get("text") if text_prompt else ""
            answer = assistant_text.get("text") if assistant_text else ""
            
            entry = {
                "audio_filepath": str(audio_filepath),
                "offset": 0.0,
                "duration": round(len(audio_array) / sampling_rate, 3),
                "context": context or "",
                "answer": answer or ""
            }
            
            entries.append(entry)
            
        except Exception as e:
            errors.append(f"Dialog {i} in {parquet_file.name}: {e}")
            continue
    
    return entries, audio_count, errors


def process_parquet_folder_parallel(parquet_dir, output_jsonl, audio_dir, max_workers=None, prefix:str=None, pretty_json=False):
    """Parallel processing version with improved resource management."""
    # Set thread limits in main process
    set_resource_limits()
    
    parquet_dir = Path(parquet_dir)
    audio_dir = Path(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temp directory for intermediate files
    temp_dir = audio_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    print(f"🔍 Found {len(parquet_files)} parquet files in {parquet_dir}")
    
    # Conservative worker count to avoid resource exhaustion
    if max_workers is None:
        # Use fewer workers to prevent resource exhaustion
        max_workers = min(4, cpu_count() // 2, len(parquet_files))
    else:
        # Ensure we don't exceed reasonable limits
        max_workers = max(max_workers, 8)
    
    print(f"🚀 Using {max_workers} parallel workers (limited to prevent resource exhaustion)")
    
    total_entries = 0
    total_audio = 0
    all_errors = []
    
    # Prepare arguments for parallel processing
    process_args = [(pf, audio_dir, temp_dir, prefix) for pf in parquet_files]
    
    # Use context manager to ensure proper cleanup
    with open(output_jsonl, "w", encoding="utf-8") as out_file:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(process_single_parquet_file, args): args[0] 
                             for args in process_args}
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_file), total=len(parquet_files), 
                             desc="Processing parquet files"):
                parquet_file = future_to_file[future]
                
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per file
                    
                    if isinstance(result, tuple) and len(result) == 3:
                        entries, audio_count, errors = result
                        
                        # Write entries to JSONL
                        for entry in entries:
                            if pretty_json:
                                out_file.write(json.dumps(entry, ensure_ascii=False, indent=2) + "\n")
                            else:
                                out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        
                        total_entries += len(entries)
                        total_audio += audio_count
                        all_errors.extend(errors)
                    
                    else:
                        # Handle error case
                        entries, error_msg = result
                        print(f"❌ {error_msg}")
                        
                except Exception as e:
                    print(f"❌ Unexpected error processing {parquet_file.name}: {e}")
    
    # Clean up temp directory
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print(f"\n✅ Finished: {total_entries} JSONL entries written to {output_jsonl}")
    print(f"🎵 Saved {total_audio} audio files to {audio_dir}")
    
    if all_errors:
        print(f"⚠️  {len(all_errors)} errors encountered during processing")
        if len(all_errors) <= 10:
            for error in all_errors:
                print(f"   - {error}")
        else:
            print(f"   - First 10 errors:")
            for error in all_errors[:10]:
                print(f"     - {error}")


def process_parquet_folder_batch(parquet_dir, output_jsonl, audio_dir, batch_size=1000, prefix:str=None, pretty_json=False):
    """Memory-efficient batch processing version - recommended for resource-constrained environments."""
    # Set thread limits
    set_resource_limits()
    
    parquet_dir = Path(parquet_dir)
    audio_dir = Path(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    print(f"🔍 Found {len(parquet_files)} parquet files in {parquet_dir}")
    
    total_entries = 0
    total_audio = 0
    
    with open(output_jsonl, "w", encoding="utf-8") as out_file:
        for parquet_file in tqdm(parquet_files, desc="Processing parquet files"):
            try:
                dialogs = convert_parquet_messages_optimized(parquet_file)
            except Exception as e:
                print(f"❌ Failed to parse {parquet_file.name}: {e}")
                continue
            
            # Process in batches to manage memory
            batch_entries = []
            
            for i, dialog in enumerate(dialogs):
                try:
                    user = next((m for m in dialog if m["role"] == "user"), None)
                    assistant = next((m for m in dialog if m["role"] == "assistant"), None)
                    
                    if not user or not assistant:
                        continue
                    
                    audio_obj = next((c for c in user["content"] if c["type"] == "audio"), None)
                    text_prompt = next((c for c in user["content"] if c["type"] == "text"), "")
                    assistant_text = next((c for c in assistant["content"] if c["type"] == "text"), None)
                    
                    if (audio_obj is None or audio_obj.get("array") is None or 
                        assistant_text is None or assistant_text.get("text") is None):
                        continue
                    
                    audio_array = audio_obj["array"]
                    sampling_rate = int(audio_obj.get("sampling_rate", 16000))
                    audio_filename = f"{prefix}--{hashmd5(audio_array)}.wav"
                    audio_filepath = audio_dir / audio_filename
                    
                    sf.write(audio_filepath, audio_array, sampling_rate)
                    total_audio += 1
                    
                    context = text_prompt.get("text") if text_prompt else ""
                    answer = assistant_text.get("text") if assistant_text else ""
                    
                    entry = {
                        "audio_filepath": str(audio_filepath),
                        "offset": 0.0,
                        "duration": round(len(audio_array) / sampling_rate, 3),
                        "context": context or "",
                        "answer": answer or ""
                    }
                    
                    batch_entries.append(entry)
                    
                    # Write batch when it reaches batch_size
                    if len(batch_entries) >= batch_size:
                        for batch_entry in batch_entries:
                            if pretty_json:
                                out_file.write(json.dumps(batch_entry, ensure_ascii=False, indent=2) + "\n")
                            else:
                                out_file.write(json.dumps(batch_entry, ensure_ascii=False) + "\n")
                        total_entries += len(batch_entries)
                        batch_entries = []
                
                except Exception as e:
                    print(f"⚠️ Skipping dialog {i} in {parquet_file.name} due to error: {e}")
                    continue
            
            # Write remaining entries in the batch
            if batch_entries:
                for batch_entry in batch_entries:
                    out_file.write(json.dumps(batch_entry, ensure_ascii=False) + "\n")
                total_entries += len(batch_entries)
    
    print(f"\n✅ Finished: {total_entries} JSONL entries written to {output_jsonl}")
    print(f"🎵 Saved {total_audio} audio files to {audio_dir}")


def hashmd5(obj):
    """
    Hash an object into a deterministic string
    """
    return hashlib.md5(pickle.dumps(obj)).hexdigest()



if __name__ == "__main__":
    # Set resource limits before any imports or processing
    set_resource_limits()
    
    # Set multiprocessing start method
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set, ignore
        pass
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert a folder of Parquet files to NeMo-compatible JSONL format.")
    parser.add_argument("--input-dir", required=True, help="Directory containing .parquet files")
    parser.add_argument("--output-jsonl", required=True, help="Output path for the resulting .jsonl file")
    parser.add_argument("--audio-dir", required=True, help="Directory where extracted audio files will be saved")
    parser.add_argument("--prefix", type=str, default=None, help="Optional prefix for audio filenames")
    parser.add_argument("--mode", choices=["parallel", "batch", "original"], default="batch", help="Processing mode: parallel (faster but resource-intensive), batch (memory-efficient, RECOMMENDED), or original")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: conservative limit)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for batch processing mode")
    parser.add_argument("--pretty-json", action="store_true", help="Write pretty-printed JSON instead of JSONL")

    args = parser.parse_args()
    
    print("🔧 Setting resource limits to prevent thread exhaustion...")
    prefix = args.prefix
    path = args.input_dir
    if prefix is not None:
        print(f"   - Using prefix: {prefix}")
    else:
        print("   - No prefix specified for audio filenames so we will derive one from the input directory name")
        last_parts = path.strip("/").split("/")[-2:]
        if last_parts[-1] == 'train' or last_parts[-1] == 'validation' or last_parts[-1] == 'test' or last_parts[-1] == 'dev':
            last_parts = "--".join(last_parts)
        else:
            last_parts = last_parts[-1]
        prefix = last_parts
    print(f"   - Derived prefix: {prefix}")
        
    if args.mode == "parallel":
        print("⚠️  WARNING: Parallel mode may cause resource exhaustion on some systems.")
        print("   Consider using --mode batch if you encounter issues.")
        process_parquet_folder_parallel(path, args.output_jsonl, args.audio_dir, args.workers, prefix, args.pretty_json)
    elif args.mode == "batch":
        print("✅ Using batch mode (recommended for stability)")
        process_parquet_folder_batch(path, args.output_jsonl, args.audio_dir, args.batch_size, prefix, args.pretty_json)
    else:
        # Original function for comparison
        print("⚠️  Using original mode - may have resource issues")
        from original_code import process_parquet_folder
        process_parquet_folder(path, args.output_jsonl, args.audio_dir)