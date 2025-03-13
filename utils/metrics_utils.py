import os
import json
import torch

def save_metrics_to_json(metrics, filepath="train_log.json"):
    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            json.dump([], f)

    def convert_tensors(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, (int, float)):
            return obj
        elif isinstance(obj, dict):
            return {k: convert_tensors(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tensors(i) for i in obj]
        return obj

    metrics = convert_tensors(metrics)

    with open(filepath, "r") as f:
        logs = json.load(f)
    
    logs.append(metrics)

    with open(filepath, "w") as f:
        json.dump(logs, f, indent=4)
