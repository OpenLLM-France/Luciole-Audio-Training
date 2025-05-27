from typing import Optional, Tuple, List, Dict, Any, Union
import torch
import torch.nn as nn
import functools
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EmbeddingLocator:
    """Utility class to locate embedding layers in various model architectures."""
    
    @staticmethod
    @functools.lru_cache(maxsize=32)  # Cache results for efficiency
    def find_embedding_layer(model):
        """Find the embedding layer in the model, with caching for better performance."""
        attribute_path = ["embed_tokens"]  # Start with direct attribute
        
        # Try progressively deeper nested paths
        for i in range(4):  # Up to 4 levels deep
            model_path = "model." * i
            full_path = f"{model_path}embed_tokens"
            attr_path = full_path.split(".")
            
            # Try to follow the attribute path
            current = model
            try:
                for attr in attr_path:
                    if attr:  # Skip empty string
                        current = getattr(current, attr)
                return current  # Found it!
            except AttributeError:
                continue  # Try next path
        
        # If we get here, embedding layer wasn't found
        raise ValueError("embed_tokens method not found in the LLM model structure.")

def get_embeddings(model, input_ids):
    """
    Get token embeddings from the model using LRU-cached embedding layer lookup.
    
    Args:
        model: The language model
        input_ids: Token IDs tensor [batch_size, seq_len]
        
    Returns:
        Embeddings tensor [batch_size, seq_len, hidden_size]
    """
    # Get embedding layer (cached for efficiency)
    embedding_layer = EmbeddingLocator.find_embedding_layer(model)
    
    # Check validity of input_ids
    vocab_size = embedding_layer.num_embeddings
    if torch.any(input_ids >= vocab_size) or torch.any(input_ids < 0):
        # Get min and max for better error message
        min_id = input_ids.min().item()
        max_id = input_ids.max().item()
        raise ValueError(
            f"Invalid input_ids detected. Found values in range [{min_id}, {max_id}], "
            f"but vocab size is {vocab_size}."
        )
    
    return embedding_layer(input_ids)

def compute_accuracy(predictions, targets, ignore_label=-100):
    """
    Compute token-level accuracy, ignoring padded positions.
    
    Args:
        predictions: Predicted token ids [batch_size, seq_len]
        targets: Target token ids [batch_size, seq_len] 
        ignore_label: Value to ignore in targets
        
    Returns:
        Scalar accuracy tensor
    """
    # Create mask for valid positions
    mask = targets != ignore_label
    
    # Select only valid positions for both predictions and targets
    valid_preds = predictions.masked_select(mask)
    valid_targets = targets.masked_select(mask)
    
    # Compute accuracy
    correct = torch.sum(valid_preds == valid_targets)
    total = torch.sum(mask)
    
    # Handle edge case of empty targets
    if total == 0:
        return torch.tensor(0.0, device=predictions.device)
        
    return (correct.float() / total.float())

def split_audio_tensor(audio_tensor, chunk_size=3000):
    """
    Splits an audio tensor along the time dimension into fixed-size chunks.
    
    Args:
        audio_tensor: Audio tensor with shape [batch, time, mel]
        chunk_size: Maximum chunk size for the time dimension
        
    Returns:
        List of tensors with shape [batch, time_chunk, mel]
    """
    batch_size, time_length, mel_dim = audio_tensor.shape
    splits = []

    # Process in chunk_size increments
    for start in range(0, time_length, chunk_size):
        end = min(start + chunk_size, time_length)
        chunk = audio_tensor[:, start:end, :]
        splits.append(chunk)

    return splits

def process_audio_batch(encoder, audio, chunk_processing=True, max_chunk_size=3000):
    """
    Process an audio batch through the encoder, handling chunking if needed.
    
    Args:
        encoder: Audio encoder model
        audio: Audio tensor [batch_size, num_chunks, frames_per_chunk, n_mels]
        chunk_processing: Whether to process in chunks even with single chunk
        max_chunk_size: Maximum chunk size when processing large inputs
        
    Returns:
        Encoder features [batch_size, seq_len, hidden_size]
    """
    bsz, num_chunks, frames_per_chunk, n_mels = audio.shape
    
    # Case 1: Multiple chunks in input
    if num_chunks > 1:
        logger.debug(f"Processing {num_chunks} audio chunks separately")
        # Process each chunk individually and concatenate
        encoder_feats = torch.cat([
            encoder.extract_variable_length_features(audio[:, i].permute(0, 2, 1))
            for i in range(num_chunks)
        ], dim=1)
        
    # Case 2: Single chunk but we should process in smaller pieces
    elif chunk_processing and frames_per_chunk > max_chunk_size:
        logger.debug(f"Single large chunk detected, processing in {frames_per_chunk//max_chunk_size+1} parts")
        # Flatten the audio tensor
        audio_flat = audio.reshape(bsz, -1, n_mels).permute(0, 2, 1)
        # Split into smaller chunks
        chunks = split_audio_tensor(audio_flat.permute(0, 2, 1), chunk_size=max_chunk_size)
        # Process each chunk and concatenate
        encoder_feats = torch.cat([
            encoder.extract_variable_length_features(chunk.permute(0, 2, 1))
            for chunk in chunks
        ], dim=1)
    
    # Case 3: Single chunk, process directly
    else:
        logger.debug("Processing single audio chunk")
        audio_flat = audio.reshape(bsz, -1, n_mels).permute(0, 2, 1)
        encoder_feats = encoder.extract_variable_length_features(audio_flat)
    
    return encoder_feats

class LucAS(nn.Module):
    """
    LucAS (Language Understanding with Audio and Speech) model.
    Combines audio encoder, language model, and projection layer.
    """
    def __init__(
        self,
        encoder: nn.Module,
        llm: nn.Module,
        encoder_projector: nn.Module,
        tokenizer,
        train_config,
        model_config,
        **kwargs
    ):  
        super().__init__()
        self.encoder = encoder
        self.llm = llm
        self.encoder_projector = encoder_projector
        self.tokenizer = tokenizer
        self.train_config = train_config
        self.model_config = model_config
        self.metric = getattr(train_config, 'metric', None)
        
        # Store fixed token IDs for convenience
        self._pad_token_id = self.tokenizer.pad_token_id
        self._bos_token_id = getattr(self.tokenizer, 'bos_token_id', None)
        self._eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)
        
        # Set default generation parameters
        self.generation_config = {
            'max_new_tokens': getattr(model_config, 'max_new_tokens', 128),
            'num_beams': getattr(model_config, 'num_beams', 4),
            'do_sample': getattr(model_config, 'do_sample', True),
            'min_length': getattr(model_config, 'min_length', 1),
            'top_p': getattr(model_config, 'top_p', 0.9),
            'repetition_penalty': getattr(model_config, 'repetition_penalty', 1.0),
            'length_penalty': getattr(model_config, 'length_penalty', 1.0),
            'temperature': getattr(model_config, 'temperature', 0.7),
        }
        
        # Record layer parameters for freezing/unfreezing
        self._component_params = {
            'encoder': list(self.encoder.parameters()),
            'llm': list(self.llm.parameters()),
            'projector': list(self.encoder_projector.parameters()),
        }
    def get_trainable_parameters(self):
        """Get the number of trainable parameters in the model."""
        return {
            'encoder': sum(p.numel() for p in self._component_params['encoder'] if p.requires_grad),
            'llm': sum(p.numel() for p in self._component_params['llm'] if p.requires_grad),
            'projector': sum(p.numel() for p in self._component_params['projector'] if p.requires_grad),
            'total': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
        
    def train_projector_only(self):
        """Freeze encoder and LLM, leaving only projector trainable."""
        freeze_component(self.encoder, 'Audio Encoder')
        freeze_component(self.llm, 'LLM')
        self.encoder.eval()  # Set encoder to eval mode
        self.llm.eval()  # Set LLM to eval mode
        self.encoder_projector.train()  # Set projector to train mode
        logger.info("Set up for projector-only training")
        
    def _prepare_inputs(self, input_ids=None, inputs_embeds=None, audio=None, modality_mask=None):
        """
        Prepare inputs for the model - handles audio encoding and embedding lookup.
        
        Returns:
            inputs_embeds: Prepared embeddings
            encoder_outs: Audio encoder outputs (if applicable)
        """
        encoder_outs = None
        
        # Process audio if provided
        if audio is not None:
            # Process audio through encoder
            if self.train_config.freeze_encoder:
                with torch.no_grad():
                    encoder_feats = process_audio_batch(self.encoder, audio)
            else:
                encoder_feats = process_audio_batch(self.encoder, audio)
                
            # Project encoded audio to hidden space
            encoder_outs = self.encoder_projector(encoder_feats)
        
        # Get text embeddings if input_ids provided
        with torch.no_grad():
            if input_ids is not None:
                # Replace -100 (ignore_index) with pad token for embedding lookup
                input_ids_clean = input_ids.clone()
                input_ids_clean[input_ids_clean == -100] = self._pad_token_id
                inputs_embeds = get_embeddings(self.llm, input_ids_clean.long())
        
        # Ensure we have embeddings from at least one source
        if inputs_embeds is None and encoder_outs is None:
            raise ValueError("Either input_ids, inputs_embeds, or audio must be provided.")
            
        return inputs_embeds, encoder_outs
    
    def _merge_modalities(self, inputs_embeds, encoder_outs, modality_mask):
        """
        Merge text embeddings with audio embeddings based on modality mask.
        
        Args:
            inputs_embeds: Text embeddings [batch_size, seq_len, hidden_dim]
            encoder_outs: Audio embeddings [batch_size, audio_len, hidden_dim]
            modality_mask: Mask indicating audio token positions [batch_size, seq_len]
            
        Returns:
            Merged embeddings [batch_size, seq_len, hidden_dim]
        """
        if modality_mask is None or encoder_outs is None:
            return inputs_embeds
            
        batch_size, seq_len, hidden_dim = inputs_embeds.size()
        
        # Find audio token start positions and lengths
        start_idxs = torch.argmax(modality_mask.float(), dim=1)
        lengths = torch.clamp(modality_mask.sum(dim=1), max=encoder_outs.shape[1]).long()
        
        # Create tensor to hold merged embeddings
        merged = torch.zeros_like(inputs_embeds)
        
        # Merge audio and text embeddings for each example in batch
        for i in range(batch_size):
            start = start_idxs[i].item()
            length = lengths[i].item()
            
            # Place audio embeddings at specified positions
            merged[i, start:start+length, :] = encoder_outs[i, :length, :]
            
            # Keep text embeddings for non-audio positions
            text_mask = ~modality_mask[i].unsqueeze(-1)
            merged[i] = merged[i] + inputs_embeds[i] * text_mask
            
        return merged

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token ids [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            inputs_embeds: Pre-computed input embeddings [batch_size, seq_len, hidden_size]
            labels: Target labels for language modeling [batch_size, seq_len]
            **kwargs: Additional keyword arguments
                - audio: Audio input tensor [batch_size, num_chunks, frames_per_chunk, n_mels]
                - modality_mask: Mask indicating audio token positions [batch_size, seq_len]
                
        Returns:
            outputs: Model outputs
            acc: Token prediction accuracy (if labels provided)
        """
        # Extract additional inputs
        audio = kwargs.get("audio", None)
        modality_mask = kwargs.get("modality_mask", None)
        
        # === Prepare text and audio embeddings ===
        
        inputs_embeds, encoder_outs = self._prepare_inputs(
            input_ids=input_ids, 
            inputs_embeds=inputs_embeds,
            audio=audio
        )
        
        # === Merge text and audio embeddings if needed ===
        if modality_mask is not None and encoder_outs is not None:
            inputs_embeds = self._merge_modalities(inputs_embeds, encoder_outs, modality_mask)

        # === Forward through LLM ===
        # with torch.no_grad():
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False
        )
        
        # === Compute accuracy if labels provided ===
        acc = None
        decoded_preds, decoded_labels = None, None
        if labels is not None:
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            # Shift for next-token prediction
            acc = compute_accuracy(preds[:, :-1], labels[:, 1:], ignore_label=-100)

            # Decode predictions and labels (cleaned from -100)
            decoded_preds = self.tokenizer.batch_decode(
                [p[l != -100] for p, l in zip(preds[:, :-1], labels[:, 1:])],
                skip_special_tokens=True
            )
            decoded_labels = self.tokenizer.batch_decode(
                [l[l != -100] for l in labels[:, 1:]],
                skip_special_tokens=True
            )

            # Save predictions
            out_path = os.path.join(self.train_config.output_dir, 'train_predictions.txt')
            with open(out_path, 'a', encoding='utf-8') as f:
                for pred, label in zip(decoded_preds, decoded_labels):
                    f.write(f"Pred : {pred.strip()}\nLabel: {label.strip()}\n---\n")
                    f.flush()
        return outputs, acc

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        audio: Optional[torch.FloatTensor] = None,
        modality_mask: Optional[torch.BoolTensor] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        """
        Generate text from inputs.
        
        Args:
            input_ids: Input token ids [batch_size, seq_len]
            inputs_embeds: Pre-computed input embeddings [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            audio: Audio input tensor [batch_size, num_chunks, frames_per_chunk, n_mels]
            modality_mask: Mask indicating audio token positions [batch_size, seq_len]
            use_cache: Whether to use past key values for faster generation
            **kwargs: Additional generation parameters
            
        Returns:
            Generated token sequences
        """
        # Prepare inputs (audio encoding + text embeddings)
        if audio is not None or modality_mask is not None:
            inputs_embeds, encoder_outs = self._prepare_inputs(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                audio=audio
            )
            
            # Merge modalities if needed
            if modality_mask is not None and encoder_outs is not None:
                inputs_embeds = self._merge_modalities(inputs_embeds, encoder_outs, modality_mask)
                # Use merged embeddings, so set input_ids to None
                input_ids = None
        
        # Build generation parameters
        gen_kwargs = self.generation_config.copy()
        # Add token IDs
        gen_kwargs.update({
            'bos_token_id': self._bos_token_id,
            'eos_token_id': self._eos_token_id, 
            'pad_token_id': self._pad_token_id,
        })
        # Update with any user-provided kwargs
        gen_kwargs.update(kwargs)
        
        # Generate with the LLM
        return self.llm.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **gen_kwargs,
        )
