from configs import TrainConfig, ModelConfig

def load_all_configs(model_path: str, train_path: str):
    model_config = ModelConfig.load(model_path)
    train_config = TrainConfig.load(train_path)
    return model_config, train_config

def save_all_configs(model_config: ModelConfig, train_config: TrainConfig, output_dir: str):
    model_config.save(f"{output_dir}/model_config.json")
    train_config.save(f"{output_dir}/train_config.json")