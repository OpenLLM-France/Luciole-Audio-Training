import json
import os
from dataclasses import asdict

def load_config(config_path):
    with open(config_path, "r") as file:
        config = json.load(file)
    return config

def setup_directories(train_config):
    if not os.path.exists(train_config.output_dir):
        os.makedirs(train_config.output_dir, exist_ok=True)
    
    model_config_file = os.path.join(train_config.output_dir, "model_config.json")
    train_config_file = os.path.join(train_config.output_dir, "train_config.json")
    
    return model_config_file, train_config_file

def save_config_files(model_config, train_config, model_config_file, train_config_file):
    with open(model_config_file, "w") as json_file:
        json.dump(asdict(model_config), json_file, indent=4)
    
    with open(train_config_file, "w") as json_file:
        data = asdict(train_config)
        json.dump(data, json_file, indent=4)
