from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ModelConfig:
    # LLM Configuration
    llm_name_hf: str = field(
        default="OpenLLM-France/Lucie-7B-Instruct-v1.1", # "unsloth/Llama-3.2-1B-Instruct-bnb-4bit", # 
        metadata={"help": "Hugging Face LLM model, e.g., Llama"}
    )
    llm_type: str = field(
        default="decoder_only",
        metadata={"help": "Type of LLM (e.g., decoder_only, encoder_decoder)"}
    )
    using_llm_type: str = field(
        default=None, #"unsloth",
        metadata={"help": "Specific type of LLM in use"}
    )
    llm_dim: int = field(
        default=4096, # , #
        metadata={"help": "Dimension size of the LLM"}
    )

    # Encoder Configuration
    encoder_path_hf: Optional[str] = field(
        default=None , #"openai/whisper-small",
        metadata={"help": "Path to encoder model on Hugging Face"}
    )
    encoder_path: str = field(
        default="large-v3-turbo", # small
        metadata={"help": "Path to the Whisper model (e.g., 'small')"}
    )
    encoder_projector_ds_rate: int = field(
        default=5,
        metadata={"help": "Downsampling rate for encoder projector"}
    )
    encoder_projector_hidden_dim: int = field(
        default= 4096 , # 2048
        metadata={"help": "Encoder projector hidden dim"}
    )
    encoder_projector_activation:str = field(
        default="relu",
        metadata={"help": "encoder projector activation (relu | gelu)"}
    )
    encoder_projector_dropout:float = field(
        default=0.1,
        metadata={"help":"Encoder projector dropout"}
    )
    mel_size: int = field(
        default=128, #80
        metadata={"help": "Mel spectrogram size"}
    )
    encoder_dim: int = field(
        default=1280, #768,
        metadata={"help": "Dimensions of the encoder"}
    )
    encoder_ds_rate: int = field(
        default=1,
        metadata={"help": "Downsampling rate of the encoder"}
    )
    num_workers: int = field(
        default=2,
        metadata={"help": "Number of workers for data loading"}
    )
    max_duration: Optional[float] = field(
        default=None,
        metadata={"help": "Maximum duration of audio (in seconds)"}
    )
    min_duration: Optional[float] = field(
        default=None,
        metadata={"help": "Minimum duration of audio (in seconds)"}
    )
    fix_length_audio: int = field(
        default=-1,
        metadata={"help": "Fixed length for audio segments"}
    )
    prompt: Optional[str] = field(
        default=None,
        metadata={"help": "Initial prompt text"}
    )
    prompt_template: str = field(
        default=None,
        metadata={"help": "Template for prompt generation"}
    )
    answer_template: str = field(
        default=None,
        metadata={"help": "Template for answer generation"}
    )
    language: str = field(
        default="en",
        metadata={"help": "Processing language (e.g., 'ar' for Arabic)"}
    )
    normalization: Optional[str] = field(
        default=None,
        metadata={"help": "Normalization method"}
    )    
    text_length: int = field(
        default=512,
        metadata={"help": "Maximum text length for processing"}
    )    
    process_online: bool = field(
        default=False,
        metadata={"help": "Process data online if True"}
    )
    
    # Decoding Parameters
    max_new_tokens: int = field(
        default=200,  # Focus on concise output
        metadata={"help": "Maximum number of new tokens to generate"}
    )
    num_beams: int = field(
        default=3,  # Increase beams for better exploration
        metadata={"help": "Number of beams for beam search"}
    )
    do_sample: bool = field(
        default=True,  # Disable sampling for deterministic outputs
        metadata={"help": "Enable sampling during decoding"}
    )
    min_length: int = field(
        default=2,  # Keep minimum length at 1
        metadata={"help": "Minimum length of generated sequence"}
    )
    top_p: float = field(
        default=0.9,  # Irrelevant since sampling is disabled
        metadata={"help": "Top-p (nucleus) sampling value"}
    )
    repetition_penalty: float = field(
        default=2.0,  # Strong penalty to discourage repetition
        metadata={"help": "Penalty to discourage repetition"}
    )
    temperature: float = field(
        default=0.7,  # Irrelevant since sampling is disabled
        metadata={"help": "Temperature for sampling"}
    )
    length_penalty: float = field(
        default=1.5,  # Slight penalty for lengthy outputs
        metadata={"help": "Length penalty factor"}
    )
    inference_mode: bool = field(
        default=True,
        metadata={"help": "Enable inference mode if True"}
    )


@dataclass
class PeftConfig:
    peft_method: str = field(
        default="lora",
        metadata={"help": "PEFT method, e.g., lora, prefix, llama_adapter"}
    )
    r: int = field(
        default=16,
        metadata={"help": "Rank of the LoRA update"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "Scaling alpha for LoRA"}
    )
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "Modules targeted by LoRA"}
    )
    bias: str = field(
        default="none",
        metadata={"help": "Type of bias (none, all, layernorm_only)"}
    )
    task_type: str = field(
        default="CAUSAL_LM",
        metadata={"help": "Task type, e.g., CAUSAL_LM"}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for LoRA"}
    )
    

@dataclass
class TrainConfig:
    model_name: Optional[str] = field(
        default="LucAS_ckp",
        metadata={"help": "Model name or path"}
    )
    batch_size_training: int = field(
        default=1,
        metadata={"help": "Batch size for training"}
    )
    batch_size_validation: int = field(
        default=1,
        metadata={"help": "Batch size for validation"}
    )
    batching_strategy: str = field(
        default="packing",
        metadata={"help": "Batching strategy (e.g., packing, padding)"}
    )
    context_len: int = field(
        default=4096, # 4096, #
        metadata={"help": "Context length for LLM"}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Gradient accumulation steps"}
    )
    num_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"}
    )
    over_train: int = field(
        default=1,  
        metadata={"help": "Number of times to overtrain"}
    )
    warmup_step: int = field(
        default=500,
        metadata={"help": "Warmup steps"}
    )
    validation_step: int = field(
        default=20,
        metadata={"help": "Steps between validations"}
    )
    run_validation: bool = field(
        default=True,
        metadata={"help": "Run validation during training"}
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "Learning rate"}
    )
    weight_decay: float = field(
        default=1e-2,
        metadata={"help": "Weight decay factor"}
    )
    gamma: float = field(
        default=0.86,
        metadata={"help": "Learning rate decay factor"}
    )
    seed: int = field(
        default=3407,
        metadata={"help": "Random seed for reproducibility"}
    )
    use_fp16: bool = field(
        default=True,
        metadata={"help": "Use FP16 precision for training"}
    )
    mixed_precision: bool = field(
        default=True,
        metadata={"help": "Enable mixed precision"}
    )
    use_peft: bool = field(
        default=True,
        metadata={"help": "Enable PEFT for training"}
    )
    peft_config: PeftConfig = field(
        default_factory=PeftConfig,
        metadata={"help": "Configuration for PEFT"}
    )
    freeze_llm: bool = field(
        default=False,# False,
        metadata={"help": "Freeze LLM during fine-tuning"}
    )
    freeze_encoder: bool = field(
        default=True,
        metadata={"help": "Freeze encoder during training"}
    )
    freeze_layers: bool = field(
        default=False,
        metadata={"help": "Freeze certain layers during training"}
    )
    num_freeze_layers: int = field(
        default=1,
        metadata={"help": "Number of layers to freeze"}
    )
    quantization: bool = field(
        default=True,
        metadata={"help": "Enable quantization"}
    )
    log_interval: int = field(
        default=5,
        metadata={"help": "Logging interval (steps)"}
    )
    log_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to log file"}
    )
    run_test_during_validation: bool = field(
        default=False,
        metadata={"help": "Run tests during validation"}
    )
    run_test_during_validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "Test data file for validation"}
    )
    run_test_during_validation_prompt: str = field(
        default="<|ASR|>",
        metadata={"help": "Prompt for test during validation"}
    )
    metric: str = field(
        default="acc",
        metadata={"help": "Metric used to evaluate the model"}
    )
    patience: int = field(
        default=4,
        metadata={"help": "Patience for early stopping"}
    )
    use_gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Use gradient checkpointing or not"}
    )
    output_dir: str = field(
        default="path/to/output",
        metadata={"help": "Directory to save model outputs"}
    )
    save_model: bool = field(
        default=True,
        metadata={"help": "Save model after training"}
    )
