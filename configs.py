import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional

@dataclass
class ModelConfig:
    llm_name_hf: str = "OpenLLM-France/Lucie-7B-Instruct-v1.1"
    llm_type: str = "decoder_only"
    llm_dim: int = 4096
    using_llm_type: str = None
    
    encoder_path_hf: Optional[str] = None
    encoder_path: str = "large-v3-turbo"
    encoder_projector_ds_rate: int = 5
    encoder_projector_hidden_dim: int = 4096
    encoder_projector_activation: str = "gelu"
    encoder_projector_dropout: float = 0.1
    encoder_dim: int = 1280
    encoder_ds_rate: int = 1
    mel_size: int = 128

    num_workers: int = 2
    fix_length_audio: int = -1
    prompt: Optional[str] = None
    prompt_template: str = None
    answer_template: str = None
    language: str = "en"
    normalization: Optional[str] = None
    text_length: int = 512
    process_online: bool = False
    
    # Decodeing params
    max_new_tokens: int = 200
    num_beams: int = 3
    do_sample: bool = True
    min_length: int = 2
    top_p: float = 0.9
    repetition_penalty: float = 2.0
    temperature: float = 0.7
    length_penalty: float = 1.5
    inference_mode: bool = True
    
    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)

    @staticmethod
    def load(path: str) -> "ModelConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return ModelConfig(**data)
    
@dataclass
class PeftConfig:
    peft_method: str = "lora"
    r: int = 8
    lora_alpha: int = 16
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.0
    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "PeftConfig":
        return PeftConfig(**data)
    
@dataclass
class TrainConfig:
    model_name: Optional[str] = "LucAS_ckp"
    batch_size_training: int = 1
    batch_size_validation: int = 1
    context_len: int = 4096
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    num_epochs: int = 3
    restart_train: bool = False
    warmup_step: int = 500
    validation_step: int = 100
    run_validation: bool = True
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    gamma: float = 0.86
    seed: int = 42
    logging_steps=10
    use_fp16: bool = True
    use_bf16: bool = False
    mixed_precision: bool = True
    use_peft: bool = False 
    peft_config: PeftConfig = field(default_factory=PeftConfig)
    freeze_llm: bool = True
    freeze_encoder: bool = True
    train_projector_only: bool = False
    num_freeze_layers: int = 1
    quantization: bool = True
    log_interval: int = 5
    log_file: Optional[str] = None
    run_test_during_validation: bool = False
    run_test_during_validation_file: Optional[str] = None
    run_test_during_validation_prompt: str = "<|ASR|>"
    metric: str = "acc"
    patience: int = 4
    use_gradient_checkpointing: bool = False
    output_dir: str = "/path/to/save/model"
    save_model: bool = True
    save_top_k_checkpoints: int = 3
    gradient_clip_val: float = 1.0
    
    # Audio specifications:
    chunk_duration_per_s: int = 30
    min_segment_duration: float = 0.1
    max_segment_duration: float = 30.0
    silence_thresh: int = -40
    min_silence_len: int = 400
    keep_silence: int = 300
    
    def save(self, path: str):
        data = asdict(self)
        if isinstance(self.peft_config, PeftConfig):
            data["peft_config"] = self.peft_config.to_dict()
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def load(path: str) -> "TrainConfig":
        with open(path, "r") as f:
            data = json.load(f)
        data["peft_config"] = PeftConfig.from_dict(data["peft_config"])
        return TrainConfig(**data)
