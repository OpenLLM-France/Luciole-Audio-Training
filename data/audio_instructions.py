import argparse
import os
import uuid

from datasets import Dataset, load_dataset
from tqdm import tqdm

from utils.audio import conform_audio  # Ensure these utilities are properly defined


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
    """Extract output with special handling for 'response_interleaf'."""
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
            # Ensure the audio data is a dictionary
            if isinstance(audio_data, dict):
                return audio_data
            else:
                print("Audio data is not a dictionary.")
                return None
    return None

def process_row(row, data_id, split, system_prompt, sampling_rate=16000):
    """Process a row into a conversation format with audio."""

    if row is None:
        print("Skipping row because row is None.")
        return None

    query_id = f"{data_id}_{split}_{uuid.uuid4()}"

    # Extract text instructions
    instruction = get_instruction(row)
    if not instruction:
        print(f"Skipping row {query_id} due to missing text instruction.")
        return None

    speech_instruct = get_speech_instruction(row)
    if not speech_instruct:
        speech_instruct = instruction

    output = get_output(row, split)
    if not output:
        print(f"Skipping row {query_id} due to missing output.")
        return None
    output = output.replace("Omni", "Lucie")
    # Process audio
    audio_data = get_audio(row)
    if not audio_data or "array" not in audio_data:
        print(f"Skipping row {query_id} due to missing audio data.")
        return None

    array = audio_data["array"]
    if not array.any():  # True if all elements are zero
        print(f"Skipping row {query_id} due to empty or silent audio array.")
        return None

    audio_path = audio_data.get("path", f'{query_id}.wav')
    sr = audio_data.get("sampling_rate", sampling_rate)
    try:
        if sr != sampling_rate:
            array = conform_audio(array, sr, sampling_rate)
    except Exception as e:
        print(f"Skipping row {query_id} due to audio conversion error: {e}")
        return None

    # Build conversation
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
    """
    Convert a list of messages to a dataset and save as a Parquet shard.
    If total_shards is known, it is included in the file name.
    """
    dataset = Dataset.from_list(messages)
    os.makedirs(data_path, exist_ok=True)
    if total_shards is not None:
        shard_info = f"{shard_idx:04d}--{total_shards:04d}"
    else:
        shard_info = f"{shard_idx:04d}"
    file_path = os.path.join(data_path, f"AudioInstruction--{data_id}--{split}--{shard_info}.parquet")
    dataset.to_parquet(file_path)
    print(f"Saved {len(messages)} records to {file_path}")

def process_split(dataset_split, data_id, split, data_path, system_prompt, max_docs, streaming=False):
    """
    Process dataset split in batches and save each batch immediately.
    For non-streaming datasets, the total shard count is computed in advance.
    """
    messages = []
    shard_counter = 0

    # For non-streaming mode, we can compute the total number of shards.
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
            messages = []  # clear the batch

    if messages:  # save any remaining messages
        shard_counter += 1
        save_batch(messages, data_id, split, data_path, shard_counter, total_shards)

def main():
    parser = argparse.ArgumentParser(description="Convert HF datasets to audio instruction format.")
    parser.add_argument("hf_dataset", nargs="+", help="HuggingFace dataset names")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--system_prompt", default="You are a helpful speech assistant who understands user input and aids with various tasks", help="System prompt")
    parser.add_argument("--max_docs", type=int, default=200, help="Max records per Parquet file")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming mode for lower memory usage")
    args = parser.parse_args()

    for ds_name in tqdm(args.hf_dataset, desc="Datasets"):
        try:
            if args.streaming:
                dataset = load_dataset(ds_name, streaming=True)
            else:
                dataset = load_dataset(ds_name)
        except Exception as e:
            print(f"Error loading {ds_name}: {e}")
            continue

        data_id = ds_name.replace("/", "--")
        for split in ["train", "validation", "test"]:
            if split not in dataset:
                continue
            process_split(
                dataset[split], data_id, split, args.output,
                args.system_prompt, args.max_docs, streaming=args.streaming
            )

if __name__ == "__main__":
    main()
