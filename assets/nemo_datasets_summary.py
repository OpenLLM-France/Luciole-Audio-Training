import os
import json
import csv
import statistics
from pathlib import Path
from tqdm import tqdm

CSV_FILE = "datasets_metadata.csv"

FIELDNAMES = [
    "dataset_name",
    "split",
    "task_type",
    "sub_task",
    "language",
    "num_audio_segments",
    "num_samples",
    "total_duration_sec",
    "total_duration_dhms",
    "min_segment_duration_sec",
    "max_segment_duration_sec",
    "avg_instruction_words",
    "min_instruction_words",
    "max_instruction_words",
    "avg_response_words",
    "min_response_words",
    "max_response_words",
    "path",
    "note"
]

def seconds_to_dhms(seconds):
    """Convert seconds to d:h:m:s format."""
    seconds = int(seconds)
    d = seconds // 86400
    h = (seconds % 86400) // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if d > 0:
        return f"{d}:{h:02}:{m:02}:{s:02}"
    else:
        return f"00:{h:02}:{m:02}:{s:02}"

def parse_jsonl(jsonl_path):
    durations, instruction_counts, response_counts = [], [], []
    nb_samples = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            nb_samples += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "conversations" in data:  # multi-turn QA
                for turn in data["conversations"]:
                    turn_type = turn.get("type")
                    if turn_type == "audio":
                        durations.append(turn.get("duration", 0))
                    elif turn_type == "text":
                        wc = len(turn.get("value", "").split())
                        from_role = turn.get("from", "").lower()
                        if from_role == "user":
                            instruction_counts.append(wc)
                        elif from_role == "assistant":
                            response_counts.append(wc)

            elif "audio_filepath" in data:  # single-turn ASR/QA
                durations.append(data.get("duration", 0))
                if "context" in data and data["context"]:
                    instruction_counts.append(len(data["context"].split()))
                if "answer" in data and data["answer"]:
                    response_counts.append(len(data["answer"].split()))

    total_sec = round(sum(durations), 2)
    return {
        "num_audio_segments": len(durations),
        "num_samples": nb_samples,
        "total_duration_sec": total_sec,
        "total_duration_dhms": seconds_to_dhms(total_sec),
        "min_segment_duration_sec": round(min(durations), 2) if durations else 0,
        "max_segment_duration_sec": round(max(durations), 2) if durations else 0,
        "avg_instruction_words": round(statistics.mean(instruction_counts), 2) if instruction_counts else 0,
        "min_instruction_words": min(instruction_counts) if instruction_counts else 0,
        "max_instruction_words": max(instruction_counts) if instruction_counts else 0,
        "avg_response_words": round(statistics.mean(response_counts), 2) if response_counts else 0,
        "min_response_words": min(response_counts) if response_counts else 0,
        "max_response_words": max(response_counts) if response_counts else 0,
    }

def add_dataset_entry(jsonl_path, task_type, sub_task, language, note="NeMo dataset"):
    jsonl_path = Path(jsonl_path)
    parts = jsonl_path.parts
    
    dataset_name = jsonl_path.parent.name
    if "multitask" in parts:
        dataset_name = "AudioLLMs--Multitask-National-Speech-Corpus-v1-extend"
    split = jsonl_path.stem

    stats = parse_jsonl(jsonl_path)

    entry = {
        "dataset_name": dataset_name,
        "split": split,
        "task_type": task_type,
        "sub_task": sub_task,
        "language": language,
        "path": str(jsonl_path),
        "note": note,
        **stats
    }

    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(entry)

def scan_and_update(root_dir):
    root_dir = Path(root_dir)
    for jsonl_path in root_dir.rglob("*.jsonl"):
        parts = jsonl_path.parts

        if "asr" in parts:
            task_type = "ASR"
            idx = parts.index("asr")
            sub_task = parts[idx + 1] if len(parts) > idx + 1 and parts[idx + 1] not in ("fr", "en") else "asr"
        elif "ast" in parts:
            task_type = "AST"
            idx = parts.index("ast")
            sub_task = parts[idx + 1] if len(parts) > idx + 1 and parts[idx + 1] not in ("fr-en", "en-fr") else "ast"
        elif "question-answering" in parts:
            task_type = "QA"
            idx = parts.index("question-answering")
            sub_task = parts[idx + 1] if len(parts) > idx + 1 else "question-answering"
        elif "multitask" in parts:
            idx = parts.index("multitask")
            task_type = "Multitask"
            sub_task = parts[idx + 3] if len(parts) > idx + 3 else ""
        else:
            task_type = "UNKNOWN"
            sub_task = ""

        print(f" → task_type={task_type}, sub_task={sub_task.lower()}, file={jsonl_path}")
        
        if "multitask" not in parts:
            language = jsonl_path.parent.parent.name if len(jsonl_path.parent.parts) >= 2 else ""
        else:
            language = "en"

        add_dataset_entry(jsonl_path, task_type, sub_task.lower(), language)

if __name__ == "__main__":
    scan_and_update("/data-server/datasets/audio/nemo/single-turn")
    scan_and_update("/data-server/datasets/audio/nemo/multi-turn")
    print(f"✅ CSV generated: {CSV_FILE}")
