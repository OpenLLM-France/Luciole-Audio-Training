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

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        audio = kwargs.get("audio", None)
        modality_mask = kwargs.get("modality_mask", None)
        encoder_outs = None

        # ==== Audio encoding ====                
        if audio is not None:
            bsz, num_chunks, frames_per_chunk, n_mels = audio.shape

            if num_chunks > 1:
                logger.info(f"Multiple chunks detected ({num_chunks}), processing each separately.")
                # Process each chunk individually and concatenate
                encoder_feats = torch.cat([
                    self.encoder.extract_variable_length_features(audio[:, i].permute(0, 2, 1))
                    for i in range(num_chunks)
                ], dim=1)
            else:
                # Flatten and process the single chunk
                audio_flat = audio.reshape(bsz, -1, n_mels).permute(0, 2, 1)
                encoder_feats = self.encoder.extract_variable_length_features(audio_flat)

            # Project into hidden space
            encoder_outs = self.encoder_projector(encoder_feats)

        
        # ==== Text embeddings ====        
        if input_ids is not None:
            # Replace ignore_index with pad token for embedding lookup
            input_ids = input_ids.clone()
            input_ids[input_ids == -100] = self.tokenizer.pad_token_id
            inputs_embeds = get_embeddings(self.llm, input_ids.long())

        if inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")

        # ==== Modality mixing ====        
        if modality_mask is not None and encoder_outs is not None:
            # modality_mask: (batch_size, seq_len), True for audio-token positions
            # compute start and length per example
            start_idxs = modality_mask.float().argmax(dim=1)
            lengths = torch.clamp(modality_mask.sum(dim=1), max=encoder_outs.shape[1]).long()
            # build a tensor matching inputs_embeds shape to receive encoder_outs
            batch_size, seq_len, hid_dim = inputs_embeds.size()
            mixed = torch.zeros_like(inputs_embeds)
            for i in range(bsz):
                start = start_idxs[i].item()
                length = lengths[i].item()
                # copy encoder outputs into mixed
                mixed[i, start:start+length, :] = encoder_outs[i, :length, :]
                # retain text embeddings for non-audio positions
                text_mask = ~modality_mask[i].unsqueeze(-1)
                mixed[i] = mixed[i] + inputs_embeds[i] * text_mask
            inputs_embeds = mixed

        # ==== Forward through LLM ====        
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False
        )
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        acc = None
        if labels is not None:
            acc = compute_accuracy(preds[:, :-1], labels[:, 1:], ignore_label=-100)

        return outputs, acc

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

