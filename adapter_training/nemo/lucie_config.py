from nemo.collections.llm.gpt.model import Llama3Config8B
from dataclasses import dataclass, field

@dataclass
class LucieConfig7B(Llama3Config8B):
    """Configuration for an 8B parameter Llama 3 model.

    Specific configuration for the 8B Llama 3 model with 32 layers,
    4096 hidden size, and 32 attention heads.
    """

    rotary_base: int = 20000000
    seq_length: int = 32_000
    num_layers: int = 32
    hidden_size: int = 4096
    ffn_hidden_size: int = 12288
    num_attention_heads: int = 32
    