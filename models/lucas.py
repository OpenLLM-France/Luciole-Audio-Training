from typing import Optional

import torch
import torch.nn as nn

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_embeddings(model, input_ids):
    # Determine where the embeddings are stored
    if hasattr(model, "embed_tokens"):
        embedding_layer = model.embed_tokens
    elif hasattr(model.model, "embed_tokens"):
        embedding_layer = model.model.embed_tokens
    elif hasattr(model.model.model, "embed_tokens"):
        embedding_layer = model.model.model.embed_tokens
    elif hasattr(model.model.model.model, "embed_tokens"):
        embedding_layer = model.model.model.model.embed_tokens
    else:
        raise ValueError("embed_tokens method not found in the LLM model structure.")
    
    vocab_size = embedding_layer.num_embeddings
    # Sanity check
    if input_ids.max() >= vocab_size or input_ids.min() < 0:
        raise ValueError(
            f"Invalid input_ids detected. Found values in range [{input_ids.min().item()}, {input_ids.max().item()}], "
            f"but vocab size is {vocab_size}."
        )
    
    return embedding_layer(input_ids)

def compute_accuracy(pad_outputs, pad_targets, ignore_label=-100):
    mask = pad_targets != ignore_label
    numerator = torch.sum(pad_outputs.masked_select(mask) == pad_targets.masked_select(mask))
    denominator = torch.sum(mask)
    return (numerator.float() / denominator.float())

def split_audio_tensor(audio_tensor, chunk_size=3000):
    """
    Splits an audio tensor along the time dimension into fixed-size chunks.
    Input shape: [batch, time, mel]
    Output: list of tensors [batch, time_chunk, mel]
    """
    batch_size, time_length, mel_dim = audio_tensor.shape
    splits = []

    for start in range(0, time_length, chunk_size):
        end = min(start + chunk_size, time_length)
        chunk = audio_tensor[:, start:end, :]
        splits.append(chunk)

    return splits

class LucAS(nn.Module):
    def __init__(self, encoder: nn.Module, llm: nn.Module, encoder_projector: nn.Module, tokenizer, train_config, model_config, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.llm = llm
        self.encoder_projector = encoder_projector
        self.tokenizer = tokenizer
        self.train_config = train_config
        self.model_config = model_config
        self.metric = self.train_config.metric

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        audio_mel = kwargs.get("audio", None)
        modality_mask = kwargs.get("modality_mask", None)
        encoder_outs = None

        # Process audio input
        if audio_mel is not None:
            if audio_mel.dim() == 2:  # Single sample
                audio_mel = audio_mel.unsqueeze(0)
            # print(f"Audio shape: {audio_mel.shape}")
            try:
                encoder_outs = self.encoder.extract_variable_length_features(audio_mel.permute(0, 2, 1))
            except AssertionError as e:
                print(f"Assertion error: {e}")
                chunks = split_audio_tensor(audio_mel, chunk_size=3000)
                encoder_outs_list = []
                for chunk in chunks:
                    out = self.encoder.extract_variable_length_features(chunk.permute(0, 2, 1))
                    encoder_outs_list.append(out)
                encoder_outs = torch.cat(encoder_outs_list, dim=1)
            # encoder_outs = self.encoder.extract_variable_length_features(audio_mel.permute(0, 2, 1))
            encoder_outs = self.encoder_projector(encoder_outs)

        # Process tokenized input (if provided)
        if input_ids is not None:
            input_ids = input_ids.clone()  # avoid in-place operations
            input_ids[input_ids == -100] = 0  # Replace padding with a valid token ID
            if input_ids.dtype != torch.long:
                input_ids = input_ids.long()

            try:
                inputs_embeds = get_embeddings(self.llm, input_ids)
            except ValueError as e:
                logger.error(f"Embedding error: {e}")
                raise e

        if inputs_embeds is None:
            raise ValueError("`inputs_embeds` cannot be None. Provide input_ids or precomputed embeddings.")

        # Handle modality masking
        if modality_mask is not None:
            modality_mask_start_indices = (modality_mask == True).float().argmax(dim=1)
            modality_lengths = torch.clamp(modality_mask.sum(dim=1), max=encoder_outs.shape[1]).tolist()

            encoder_outs_pad = torch.zeros_like(inputs_embeds)
            for i in range(encoder_outs.shape[0]):
                start_idx = modality_mask_start_indices[i]
                length = modality_lengths[i]
                encoder_outs_pad[i, start_idx:start_idx + length] = encoder_outs[i][:length]
            inputs_embeds = encoder_outs_pad + inputs_embeds * (~modality_mask[:, :, None])

        # Pass through LLM
        model_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            use_reentrant=False,
        )
        preds = torch.argmax(model_outputs.logits, -1)
        acc = -1 if labels is None else compute_accuracy(preds[:, :-1], labels[:, 1:], ignore_label=-100)
        return model_outputs, acc

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        if inputs_embeds is None and input_ids is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")
        model_outputs = self.llm.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.model_config.max_new_tokens,
            num_beams=self.model_config.num_beams,
            do_sample=self.model_config.do_sample,
            min_length=self.model_config.min_length,
            top_p=self.model_config.top_p,
            repetition_penalty=self.model_config.repetition_penalty,
            length_penalty=self.model_config.length_penalty,
            temperature=self.model_config.temperature,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs,
        )
        return model_outputs

