import argparse
import os
import uuid
import io  
from datasets import Dataset, load_dataset, load_from_disk
from tqdm import tqdm
import soundfile as sf

from adapter_training.utils.audio import conform_audio 

def _convert_bytes_to_audio(bytes_data):
    """Convert bytes data to audio array."""
    audio_stream = io.BytesIO(bytes_data)
    audio_array, sr = sf.read(audio_stream)
    return audio_array, sr

def get_instruction(row):
    """Extract instruction from available keys."""
    instruction = ""
    for key in ["instruction", "instruct", "prompt", "question"]:
        if key in row:
            instruction = row.get(key, "")
            break
    if "input" in row:
        input_text = row.get("input", "")
        if input_text:
            instruction += f" {input_text}".strip()
    return instruction.strip() if instruction else ""

def get_speech_instruction(row):
    """Extract speech instruction if available."""
    speech_instruction = ""
    for key in ["speech_input", "speech_instruction"]:
        if key in row and row[key] is not None:
            speech_instruction = row.get(key, "")
    return speech_instruction

def get_output(row, split):
    """Extract output with handling for 'response_interleaf'."""
    if "output" in row:
        return row["output"] if row["output"] is not None else ""
    elif "response_interleaf" in row:
        output = row["response_interleaf"]
        parts = output.split("\n\n")
        if split == "test" and len(parts) >= 2:
            return parts[1].strip() or ""
        elif split == "validation" and len(parts) >= 1:
            return parts[0].strip() or ""
        return output or ""
    for key in ["response", "answer", "assistant_response", "gpt4-answer"]:
        if key in row:
            return row[key] or ""
    return ""

def get_audio(row):
    """Extract audio data from row."""
    for key in ["audio", "input_audio", "context", "question_audio", "question_audio_path", "filename"]:
        if key in row and row[key] is not None:
            audio_data = row[key]
            if isinstance(audio_data, dict):
                return audio_data
            elif isinstance(audio_data, bytes):
                audio_array, sr = _convert_bytes_to_audio(audio_data)
                return {
                    "array": audio_array,
                    "sampling_rate": sr,
                    "path": row.get("audio_path", None)
                }
            elif isinstance(audio_data, str):
                if os.path.isfile(audio_data):
                    audio_array, sr = sf.read(audio_data)
                    return {
                        "array": audio_array,
                        "sampling_rate": sr,
                        "path": audio_data
                    }
                else:
                    print(f"🚫 Audio file {audio_data} does not exist.")
                    return None
            else:
                print("⚠️ Audio data is not a dictionary.")
                return None
    return None

def process_row(row, data_id, split, system_prompt, sampling_rate=16000):
    """Process a row into a conversation format with audio."""

    if row is None:
        print("🚫 Skipping row because row is None.")
        return None

    query_id = f"{data_id}_{split}_{uuid.uuid4()}"

    instruction = get_instruction(row)
    if not instruction:
        print(f"🚫 Skipping row {query_id} due to missing text instruction.")
        return None

    speech_instruct = get_speech_instruction(row) or instruction
    output = get_output(row, split)
    if not output:
        print(f"🚫 Skipping row {query_id} due to missing output.")
        return None

    output = output.replace("Omni", "Lucie")

    audio_data = get_audio(row)
    if not audio_data or "array" not in audio_data:
        print(f"🚫 Skipping row {query_id} due to missing audio data.")
        return None

    array = audio_data["array"]
    if not array.any():
        print(f"🔇 Skipping row {query_id} due to empty or silent audio array.")
        return None

    audio_path = audio_data.get("path", f'{query_id}.wav')
    sr = audio_data.get("sampling_rate", sampling_rate)
    try:
        if sr != sampling_rate:
            array = conform_audio(array, sr, sampling_rate)
    except Exception as e:
        print(f"⚠️ Skipping row {query_id} due to audio conversion error: {e}")
        return None

    conversation = []
    if system_prompt:
        conversation.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        })

    conversation.extend([
        {
            "role": "user",
            "content": [
                {"type": "audio", "array": array, "path": audio_path, "sampling_rate": sampling_rate},
                {"type": "text", "text": speech_instruct}
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": output}]
        }
    ])

    return {"messages": conversation}

def save_batch(messages, data_id, split, data_path, shard_idx, total_shards=None):
    """Save a list of messages as a Parquet shard."""
    dataset = Dataset.from_list(messages)
    os.makedirs(data_path, exist_ok=True)
    shard_info = f"{shard_idx:04d}--{total_shards:04d}" if total_shards else f"{shard_idx:04d}"
    data_path_split = os.path.join(data_path, split)
    os.makedirs(data_path_split, exist_ok=True)
    file_path = os.path.join(data_path_split, f"AudioInstruction--{data_id}--{split}--{shard_info}.parquet")
    dataset.to_parquet(file_path)
    print(f"✅ Saved {len(messages)} records to {file_path}")

def process_split(dataset_split, data_id, split, data_path, system_prompt, max_docs, streaming=False):
    """Process dataset split in batches."""
    messages = []
    shard_counter = 0
    total_shards = None
    if not streaming and hasattr(dataset_split, '__len__'):
        total_shards = (len(dataset_split) + max_docs - 1) // max_docs

    for row in tqdm(dataset_split, desc=f"Processing {split} rows"):
        message = process_row(row, data_id, split, system_prompt)
        if message:
            messages.append(message)
        if len(messages) >= max_docs:
            shard_counter += 1
            save_batch(messages, data_id, split, data_path, shard_counter, total_shards)
            messages = []

    if messages:
        shard_counter += 1
        save_batch(messages, data_id, split, data_path, shard_counter, total_shards)

def _has_parquet_files(folder_path):
    from pathlib import Path
    return any(Path(folder_path).glob("*.parquet"))

def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description="🎛️ Convert HF datasets to audio instruction format.")
    parser.add_argument("hf_dataset", nargs="+", help="📚 HuggingFace dataset names")
    parser.add_argument("--output", default="output", help="📂 Output directory")
    parser.add_argument("--system_prompt", default="You are a helpful speech assistant who understands user input and aids with various tasks", help="🧠 System prompt")
    parser.add_argument("--max_docs", type=int, default=200, help="📦 Max records per Parquet file")
    parser.add_argument("--streaming", action="store_true", help="🔄 Enable streaming mode")
    args = parser.parse_args()

    hf_datasets = args.hf_dataset
    for hf_dataset in tqdm(hf_datasets, desc="Processing datasets"):
        if os.path.isdir(hf_dataset):
            for sub in tqdm(os.listdir(hf_dataset), desc="Local Datasets"):
                sub_path = os.path.join(hf_dataset, sub)
                if not os.path.isdir(sub_path):
                    continue  # Skip files, we only want folders
                
                name_parts = sub.split("-")
                split_mtnsc = name_parts[-1] if len(name_parts) > 1 else "train"
                data_id = "-".join(name_parts[:-1]) if len(name_parts) > 1 else sub
                parquet_paths = [str(p) for p in Path(sub_path).rglob("*.parquet")]
                
                if not _has_parquet_files(sub_path):
                    print(f"🚫 Skipping {sub_path} because it does not contain Parquet files.")
                    continue
                
                print(f"✅ Loading dataset from: {sub_path}")
                parquet_paths = [str(p) for p in Path(sub_path).rglob("*.parquet")]
                
                try:
                    dataset = load_dataset(
                        "parquet",
                        data_files={"train": parquet_paths},
                        features=None,
                        streaming=True  # set to False if you want to inspect the data immediately
                        )
                except Exception as e:
                    print(f"❌ Failed to load dataset from {sub_path}: {e}")
                    continue
                for split in ["train", "validation", "test"]:
                    if split not in dataset:
                        continue
                    if os.path.basename(hf_dataset) == "Multitask-National-Speech-Corpus-v1-extend":
                        ssplit = split_mtnsc
                    else:
                        ssplit = split
                    process_split(
                        dataset[split], data_id, ssplit, args.output,
                        args.system_prompt, args.max_docs, streaming=True
                    )
            
        else:
            try:
                print(f"📡 Loading from 🤗 Hub: {hf_dataset}")
                if args.streaming:
                    dataset = load_dataset(hf_dataset, streaming=True)
                else:
                    dataset = load_dataset(hf_dataset)
            except Exception as e:
                print(f"❌ Error loading {hf_dataset}: {e}")
                continue

            data_id = hf_dataset.replace("/", "--")
            for split in ["train", "validation", "test"]:
                if split not in dataset:
                    continue
                process_split(
                    dataset[split], data_id, split, args.output,
                    args.system_prompt, args.max_docs, streaming=args.streaming
                )


if __name__ == "__main__":
    main()