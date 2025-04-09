import os
import glob
import random
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor
import sys

# ─── CONFIG ───────────────────────────────────────────────────────────────
INPUT_DIR   = sys.argv[1] 
OUTPUT_DIR  = sys.argv[2] 
if len(sys.argv) < 2:
    raise ValueError("You have to provide the input and output directories as arguments.")
if len(sys.argv) > 3:
    raise ValueError("You have to provide only the input and output directories as arguments.")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
if not os.path.exists(INPUT_DIR):
    raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")

FAIL_LOG = os.path.join(OUTPUT_DIR, "failed_files.log")

MAX_WORKERS = max(1, os.cpu_count() - 1)

# ─── PROMPTS & TEMPLATES ────────────────────────────────────────────────────
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

response_templates = [
    lambda q, a: f'After analyzing the audio, you\'re asking for:\n"{q}"\n\nHere’s what I came up with:\n{a}',
    lambda q, a: f'I’ve processed your audio, and here’s the question you’re asking:\n"{q}"\n\nMy response:\n{a}',
    lambda q, a: f'From the audio, this is the question you posed:\n"{q}"\n\nHere’s what I think:\n{a}',
    lambda q, a: f'Based on my analysis of the audio, here’s the question you’re asking:\n"{q}"\n\nHere’s my take:\n{a}',
    lambda q, a: f'After processing the audio, this is the question you raised:\n"{q}"\n\nLet me respond:\n{a}',
    lambda q, a: f'Upon reviewing the audio, you’re asking:\n"{q}"\n\nHere’s the result:\n{a}',
    lambda q, a: f'Upon listening to the audio, I understand you’re asking:\n"{q}"\n\nHere’s my reply:\n{a}',
    lambda q, a: f'After analyzing the audio, you asked:\n"{q}"\n\nHere’s what I came up with:\n{a}',
    lambda q, a: f'Following the analysis of the audio, here’s the question you’ve asked:\n"{q}"\n\nMy response:\n{a}',
    lambda q, a: f'After reviewing the audio, it seems like you’re asking:\n"{q}"\n\nHere’s my reply to your query:\n{a}'
]

# ─── FUNCTION TO CONVERT AND OVERWRITE PARQUET ───────────────────────────────
def convert_and_overwrite_parquet(parquet_file_path, out_path):
    # Read the original parquet file
    table = pq.read_table(parquet_file_path)
    df = table.to_pandas()

    modified_messages = []

    for _, row in df.iterrows():
        messages = row.iloc[0]

        if isinstance(messages, np.ndarray):
            structured_messages = []
            original_user_question = None

            for message in messages:
                if isinstance(message, dict):
                    role = message.get("role")
                    content = list(message.get("content", []))

                    if role == "system":
                        text = next((c.get("text") for c in content if c.get("type") == "text"), None)
                        structured_messages.append({"role": "system", "content": [{"type": "text", "text": text}]})

                    elif role == "user":
                        audio = next((c for c in content if c.get("type") == "audio"), None)
                        original_user_question = next((c.get("text") for c in content if c.get("type") == "text"), None)
                        new_user_text = random.choice(audio_question_instructions)

                        structured_messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "audio",
                                    "array": audio.get("array"),
                                    "path": audio.get("path"),
                                    "sampling_rate": audio.get("sampling_rate")
                                },
                                {"type": "text", "text": new_user_text}
                            ]
                        })

                    elif role == "assistant":
                        answer = next((c.get("text") for c in content if c.get("type") == "text"), None)
                        if original_user_question:
                            template = random.choice(response_templates)
                            full_response = template(original_user_question, answer)
                        else:
                            full_response = answer

                        structured_messages.append({
                            "role": "assistant",
                            "content": [{"type": "text", "text": full_response}]
                        })

            modified_messages.append([structured_messages])

    # Save updated data back to parquet file
    new_df = pd.DataFrame(modified_messages, columns=[df.columns[0]])
    table = pa.Table.from_pandas(new_df)
    parquet_base_name = parquet_file_path.split("/")[-1]
    parquet_file_new_path = os.path.join(out_path, parquet_base_name)
    pq.write_table(table, parquet_file_new_path)

    print(f"✅ Successfully updated and saved: {parquet_file_new_path}")

# ─── MAIN ──────────────────────────────────────────────────────────────────
def process_file(parquet_file):
    try:
        convert_and_overwrite_parquet(parquet_file, OUTPUT_DIR)
    except Exception as e:
        with open(FAIL_LOG, "a") as fail_log:
            fail_log.write(f"Failed to process {parquet_file}: {e}\n")
        print(f"⚠️ Failed to process: {parquet_file}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(FAIL_LOG):
        os.remove(FAIL_LOG)

    files = glob.glob(os.path.join(INPUT_DIR, "*.parquet"))
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        exe.map(process_file, files)

    if os.path.exists(FAIL_LOG):
        n = sum(1 for _ in open(FAIL_LOG))
        print(f"\n⚠️ {n} files failed (see {FAIL_LOG})")
    else:
        print("\n✅ All files processed!")
