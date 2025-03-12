import torch
from torch.nn import functional as F
import torch.nn as nn
import types


class WhisperWrappedEncoder:
    """
    Wrapper class to load and extend the Whisper encoder model.
    """
    @classmethod
    def load(cls, model_config):
        """
        Load the Whisper encoder based on the configuration.
        Args:
            model_config: Configuration object with the following attributes:
                - encoder_path_hf (str or None): Path to the Hugging Face pretrained model.
                - encoder_path (str or None): Path to the Whisper model from OpenAI's library.
        Returns:
            The loaded encoder model with extended functionality.
        """
        def extract_variable_length_features(self, x: torch.Tensor) -> torch.Tensor:
            """
            Process audio features with variable length.
            Args:
                x: torch.Tensor, shape = (batch_size, n_mels, n_ctx)
                   The mel spectrogram of the audio.
            Returns:
                torch.Tensor: Processed audio features of shape 
                              (batch_size, seq_len, feature_dim).
            """
            # Apply convolutional layers with GELU activations
            x = F.gelu(self.conv1(x))
            x = F.gelu(self.conv2(x))
        
            x = x.permute(0, 2, 1) # Rearrange dimensions to match the expected shape
            
            # Ensure shapes match with positional embedding
            assert x.shape[1:] == self.positional_embedding.shape, \
                f"Incorrect audio shape: {x.shape[1:]} != {self.positional_embedding.shape}"
                        
            x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype) # Add positional embeddings (trimmed to input length)
            for block in self.blocks: # Pass through transformer blocks
                x = block(x)

            x = self.ln_post(x) # Apply layer normalization
            return x

        if model_config.encoder_path_hf:
            # Load from Hugging Face pretrained models
            try:
                from transformers import AutoModel
                encoder = AutoModel.from_pretrained(
                    model_config.encoder_path_hf,
                    torch_dtype=torch.bfloat16
                ).encoder
            except ImportError as e:
                raise ImportError(
                    "Hugging Face transformers library is required but not installed."
                ) from e
        elif model_config.encoder_path:
            # Load from OpenAI's Whisper implementation
            try:
                import whisper
                encoder = whisper.load_model(
                    name=model_config.encoder_path, 
                    device="cpu",
                ).encoder
                # Dynamically add the method for variable-length feature extraction
                encoder.extract_variable_length_features = types.MethodType(
                    extract_variable_length_features, encoder
                )
            except ImportError as e:
                raise ImportError(
                    "OpenAI Whisper library is required but not installed."
                ) from e
        else:
            raise ValueError("Either `encoder_path_hf` or `encoder_path` must be provided in the configuration.")

        return encoder

def set_encoder(model_conf):
    encoder = WhisperWrappedEncoder.load(model_conf)
    return encoder