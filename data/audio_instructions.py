import os
from tqdm import tqdm
from datasets import load_dataset, Dataset
import uuid
import argparse

from utils.audio import conform_audio  # Ensure these utilities are properly defined

def get_instruction(row):
    """Extract instruction from available keys."""
    for key in ["instruction", "instruct", "prompt", "question"]:
        if key in row:
            instruction = row.get(key, "")
            break
    else:
        instruction = ""
    if "input" in row:
        instruction += f" {row.get('input', '')}".strip()
    return instruction.strip()

def get_speech_instruction(row):
    """Extract speech instruction if available."""
    return row.get("speech_input") or row.get("speech_instruction")

def get_output(row, split):
    """Extract output with special handling for 'response_interleaf'."""
    if "output" in row:
        return row["output"]
    elif "response_interleaf" in row:
        output = row["response_interleaf"]
        parts = output.split("\n\n")
        if split == "test" and len(parts) >= 2:
            return parts[1].strip()
        elif split == "validation" and len(parts) >= 1:
            return parts[0].strip()
        return output
    for key in ["response", "answer", "assistant_response", "gpt4-answer"]:
        if key in row:
            return row[key]
    return ""

def get_audio(row):
    """Extract audio data from row."""
    for key in ["audio", "input_audio", "context", "question_audio", "question_audio_path"]:
        if key in row:
            return row[key]
    return None

def process_row(row, data_id, split, system_prompt, sampling_rate=16000):
    """Process a row into a conversation format with audio."""
    query_id = f"{data_id}_{split}_{uuid.uuid4()}"
    
    # Extract text instructions
    instruction = get_instruction(row)
    speech_instruct = get_speech_instruction(row) or instruction
    output = get_output(row, split)
    
    # Validate critical fields
    if not instruction or not output:
        return None
    
    # Process audio
    audio_data = get_audio(row)
    # if isinstance(audio_data, dict) and "array" in audio_data:
    array = audio_data["array"]
    audio_path = audio_data["path"] if audio_data["path"] is not None else f'{query_id}.wav'
    sr = audio_data.get("sampling_rate", sampling_rate)
    if sr != sampling_rate:
        array = conform_audio(array, sr, sampling_rate)
    # audio_entry = {"type": "audio", "path": audio_path, "sampling_rate": sampling_rate}
    
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
                {"type": "audio", "array" : array, "path": audio_path, "sampling_rate": sampling_rate},
                {"type": "text", "text": speech_instruct}
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": output}]
        }
    ])
    
    return {"messages": conversation}

def process_split(dataset_split, data_id, split, data_path, system_prompt, max_docs):
    """Process and save dataset split as Parquet shards."""
    messeges = []
    
    for row in tqdm(dataset_split, desc=f"Processing {split}"):
        try:
            messege = process_row(row, data_id, split, system_prompt)
            if messege:
                messeges.append(messege)
        except Exception as e:
            print(f"Skipping row due to error: {e}")
    
    if not messeges:
        return
    
    # Create and save dataset
    dataset = Dataset.from_list(messeges)
    # output_dir = os.path.join(data_path, split)
    os.makedirs(data_path, exist_ok=True)
    
    # Save in shards
    num_shards = (len(dataset) + max_docs - 1) // max_docs
    for shard_idx in range(num_shards):
        shard = dataset.shard(num_shards, shard_idx)
        shard.to_parquet(os.path.join(data_path, f"AudioInstruction--{data_id}--{split}--{shard_idx + 1:04d}--{num_shards:04d}.parquet"))

def main():
    parser = argparse.ArgumentParser(description="Convert HF datasets to audio instruction format.")
    parser.add_argument("hf_dataset", nargs="+", help="HuggingFace dataset names")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--system_prompt", default="You are a helpful speech assistant who understands user input and aids with various tasks", help="System prompt")
    parser.add_argument("--max_docs", type=int, default=200, help="Max records per Parquet file")
    args = parser.parse_args()
    
    for ds_name in args.hf_dataset:
        try:
            dataset = load_dataset(ds_name)
        except Exception as e:
            print(f"Error loading {ds_name}: {e}")
            continue
        
        data_id = ds_name.replace("/", "--")
        # data_path = os.path.join(args.output, data_id)
        
        for split in ["train", "validation", "test"]:
            if split not in dataset:
                continue
            process_split(
                dataset[split], data_id, split, args.output,
                args.system_prompt, args.max_docs
            )

if __name__ == "__main__":
    main()