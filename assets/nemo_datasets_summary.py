import os
import json
import csv
import statistics
from pathlib import Path
from tqdm import tqdm

CSV_FILE = "datasets_metadata.csv"

FIELDNAMES = [
    "nom_dataset",
    "split",
    "type_tache",
    "sous_tache",
    "langue",
    "nb_segments_audio",
    "nb_samples",
    "duree_totale_sec",
    "duree_totale_dhms",
    "duree_segment_min_sec",
    "duree_segment_max_sec",
    "nb_mots_instruction_moy",
    "nb_mots_instruction_min",
    "nb_mots_instruction_max",
    "nb_mots_reponse_moy",
    "nb_mots_reponse_min",
    "nb_mots_reponse_max",
    "note",
    "path"
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
        "nb_segments_audio": len(durations),
        "nb_samples": nb_samples,
        "duree_totale_sec": total_sec,
        "duree_totale_dhms": seconds_to_dhms(total_sec),
        "duree_segment_min_sec": round(min(durations), 2) if durations else 0,
        "duree_segment_max_sec": round(max(durations), 2) if durations else 0,
        "nb_mots_instruction_moy": round(statistics.mean(instruction_counts), 2) if instruction_counts else 0,
        "nb_mots_instruction_min": min(instruction_counts) if instruction_counts else 0,
        "nb_mots_instruction_max": max(instruction_counts) if instruction_counts else 0,
        "nb_mots_reponse_moy": round(statistics.mean(response_counts), 2) if response_counts else 0,
        "nb_mots_reponse_min": min(response_counts) if response_counts else 0,
        "nb_mots_reponse_max": max(response_counts) if response_counts else 0,
    }

def add_dataset_entry(jsonl_path, type_tache, sous_tache, langue, note="NeMo dataset"):
    jsonl_path = Path(jsonl_path)
    parts = jsonl_path.parts
    
    dataset_name = jsonl_path.parent.name
    if "multitask" in parts:
        dataset_name = "AudioLLMs--Multitask-National-Speech-Corpus-v1-extend"
    split = jsonl_path.stem

    stats = parse_jsonl(jsonl_path)

    entry = {
        "nom_dataset": dataset_name,
        "split": split,
        "type_tache": type_tache,
        "sous_tache": sous_tache,
        "langue": langue,
        "path": str(jsonl_path.parent.as_posix()),
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
    # jsonl_files = list(root_dir.rglob("*.jsonl"))
    for jsonl_path in root_dir.rglob("*.jsonl"): #tqdm(jsonl_files, desc=f"Scanning {root_dir}", unit="file"):
        parts = jsonl_path.parts

        if "asr" in parts:
            type_tache = "ASR"
            idx = parts.index("asr")
            sous_tache = parts[idx + 1] if len(parts) > idx + 1 and parts[idx + 1] not in ("fr", "en") else "asr"
        elif "ast" in parts:
            type_tache = "AST"
            idx = parts.index("ast")
            sous_tache = parts[idx + 1] if len(parts) > idx + 1 and parts[idx + 1] not in ("fr-en", "en-fr") else "ast"
        elif "question-answering" in parts:
            type_tache = "QA"
            idx = parts.index("question-answering")
            sous_tache = parts[idx + 1] if len(parts) > idx + 1 else "question-answering"
        elif "multitask" in parts:
            idx = parts.index("multitask")
            type_tache = "Multitask"
            sous_tache = parts[idx + 3] if len(parts) > idx + 1 else ""
        else:
            type_tache = "UNKNOWN"
            sous_tache = ""
        print(f" → type_tache={type_tache}, sous_tache={sous_tache.lower()}, file={jsonl_path}")
        
        if "multitask" not in parts:
            langue = jsonl_path.parent.parent.name if len(jsonl_path.parent.parts) >= 2 else ""
        else:
            langue = "en"

        add_dataset_entry(jsonl_path, type_tache, sous_tache.lower(), langue)

if __name__ == "__main__":
    scan_and_update("/data-server/datasets/audio/nemo/single-turn")
    scan_and_update("/data-server/datasets/audio/nemo/multi-turn")
    print(f"✅ CSV généré : {CSV_FILE}")
