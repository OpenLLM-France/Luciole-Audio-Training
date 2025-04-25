import os
import numpy as np
import torch
import whisper
from textwrap import dedent
from typing import Any, List, Tuple
from pydub import AudioSegment
from pydub.silence import split_on_silence
from datasets import Dataset
from dask.diagnostics import ProgressBar
import dask.dataframe as dd


def convert_parquet_messages(messages_df):
    """
    Convert Parquet rows of 'messages' into columns for user/system/assistant text and user audio dict.
    """
    output_data = {"user_text": [], "user_audio": [], "system_text": [], "assistant_text": []}

    for _, row in messages_df.iterrows():
        messages = row['messages']
        message_dict = {k: None for k in output_data}

        if isinstance(messages, (list, np.ndarray)):
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = msg.get('role')
                contents = msg.get('content', [])
                if role == 'system':
                    text = next((c['text'] for c in contents if c.get('type')=='text'), None)
                    message_dict['system_text'] = text
                elif role == 'user':
                    # text
                    text = next((c['text'] for c in contents if c.get('type')=='text'), None)
                    message_dict['user_text'] = text
                    # audio
                    audio_item = next((c for c in contents if c.get('type')=='audio'), None)
                    if audio_item:
                        arr = None
                        sr = None
                        path = None
                        if audio_item.get('array') is not None:
                            arr = np.array(audio_item['array'], dtype=np.float32)
                            sr = audio_item.get('sampling_rate')
                            path = audio_item.get('path')
                        elif audio_item.get('audio') is not None:
                            arr = np.array(audio_item['audio']['array'], dtype=np.float32)
                            sr = audio_item['audio'].get('sampling_rate')
                            path = audio_item['audio'].get('path')
                        if arr is not None:
                            message_dict['user_audio'] = {"array": arr, "sampling_rate": sr, "path": os.path.basename(path) if path else None}
                elif role == 'assistant':
                    text = next((c['text'] for c in contents if c.get('type')=='text'), None)
                    message_dict['assistant_text'] = text

        for k in output_data:
            output_data[k].append(message_dict[k])

    return output_data


def _read_parquet_using_dask(parquet_dirs, split='test') -> Dataset:
    paths = []
    if isinstance(parquet_dirs, str):
        parquet_dirs = [parquet_dirs]
    for d in parquet_dirs:
        p = os.path.join(d, split)
        if os.path.exists(p):
            paths.append(os.path.join(p, '*.parquet'))
    if not paths:
        raise FileNotFoundError(f"No parquet files for split {split}")

    with ProgressBar():
        df = dd.read_parquet(paths, engine='pyarrow', columns=['messages']).compute()
    converted = convert_parquet_messages(df)
    return Dataset.from_dict(converted)


def _encode_chunk(chunk: np.ndarray, mel_size: int) -> torch.Tensor:
    """Pad/trim to 30s then compute log-mel and permute to (time, n_mels)."""
    raw = whisper.pad_or_trim(chunk)
    mel = whisper.log_mel_spectrogram(raw, n_mels=mel_size).permute(1,0)
    return mel


def _make_pseudo_tokens(total_frames: int, tokenizer: Any, ignore_index: int) -> torch.Tensor:
    boa = tokenizer.convert_tokens_to_ids("<|start_of_audio|>")
    eoa = tokenizer.convert_tokens_to_ids("<|end_of_audio|>")
    total_tokens = ((total_frames + 1)//2)//5
    placeholder = torch.full((total_tokens,), ignore_index, dtype=torch.long)
    return torch.cat([torch.tensor([boa]), placeholder, torch.tensor([eoa])])


def split_audio_on_silence(
    audio_array: np.ndarray,
    sample_rate: int = 16000,
    silence_thresh: int = -40,
    min_silence_len: int = 400,
    keep_silence: int = 300,
    min_segment_duration: float = 0.1,
    max_segment_duration: float = 30.0
) -> List[AudioSegment]:
    """
    Split an audio array into segments on silence and merge to respect max duration.
    """
    if audio_array.dtype != np.int16:
        audio_array = (audio_array * 32767).astype(np.int16)
    seg = AudioSegment(
        audio_array.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )
    pieces = split_on_silence(
        seg,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence
    )
    chunks, current = [], AudioSegment.empty()
    for p in pieces:
        if len(current) + len(p) <= max_segment_duration * 1000:
            current += p
        else:
            if len(current) >= min_segment_duration * 1000:
                chunks.append(current)
                current = p
            else:
                current += p
    if len(current) >= min_segment_duration * 1000 or not chunks:
        chunks.append(current)
    return chunks


def process_audio_long(
    audio_array: np.ndarray,
    tokenizer: Any,
    mel_size: int,
    ignore_index: int = -100,
    sr: int = whisper.audio.SAMPLE_RATE,
    chunk_duration_s: int = 30,
    silence_thresh: int = -40,
    min_silence_len: int = 400,
    keep_silence: int = 300,
    min_segment_duration: float = 0.1,
    max_segment_duration: float = 30.0
) -> Tuple[torch.Tensor, int, List[torch.Tensor], int]:
    """
    Split long audio into mel-chunks and return pseudo-tokens, chunk list, and total frames.
    """
    # ensure float32
    if isinstance(audio_array, list):
        audio_array = np.array(audio_array, dtype=np.float32)
    elif audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)
    duration = len(audio_array) / sr

    if round(duration,2) <= chunk_duration_s:
        mel = _encode_chunk(audio_array, mel_size)
        total_frames = mel.shape[0]
        tokens = _make_pseudo_tokens(total_frames, tokenizer, ignore_index)
        return tokens, tokens.size(0), [mel], total_frames

    segments = split_audio_on_silence(
        audio_array, sr, silence_thresh, min_silence_len,
        keep_silence, min_segment_duration, max_segment_duration
    )
    mel_chunks = []
    for seg in segments:
        samples = np.array(seg.get_array_of_samples(), dtype=np.int16).astype(np.float32) / 32767.0
        mel_chunks.append(_encode_chunk(samples, mel_size))

    total_frames = sum(m.shape[0] for m in mel_chunks)
    tokens = _make_pseudo_tokens(total_frames, tokenizer, ignore_index)
    return tokens, tokens.size(0), mel_chunks, total_frames


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(
        self, dirs, tokenizer, model_config, train_config,
        split='test', inference_mode=False
    ):
        self.ds = _read_parquet_using_dask(dirs, split)
        self.tokenizer = tokenizer
        self.mel_size = getattr(model_config, 'mel_size', 80)
        self.IGNORE_INDEX = -100
        self.inference_mode = inference_mode
        # chunk params
        self.chunk_duration_s = getattr(train_config, 'chunk_duration_per_s', 30)
        self.silence_thresh = getattr(train_config, 'silence_thresh', -40)
        self.min_silence_len = getattr(train_config, 'min_silence_len', 400)
        self.keep_silence = getattr(train_config, 'keep_silence', 300)
        self.min_segment_duration = getattr(train_config, 'min_segment_duration', 0.1)
        self.max_segment_duration = getattr(train_config, 'max_segment_duration', 30.0)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        row = self.ds[idx]
        system = row['system_text'] or dedent(
            "You are a helpful speech assistant who understands user input and aids with various tasks"
        )
        user_text = row['user_text'] or ""
        audio = row['user_audio']
        arr = audio.get('array') if audio else None
        if arr is None:
            raise ValueError(f"No audio at index {idx}")
        sr = audio.get('sampling_rate', whisper.audio.SAMPLE_RATE)
        if sr != whisper.audio.SAMPLE_RATE:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=whisper.audio.SAMPLE_RATE)

        tokens, at_len, mel_chunks, a_len = process_audio_long(
            arr, self.tokenizer, self.mel_size, self.IGNORE_INDEX,
            whisper.audio.SAMPLE_RATE,
            self.chunk_duration_s, self.silence_thresh,
            self.min_silence_len, self.keep_silence,
            self.min_segment_duration, self.max_segment_duration
        )

        assistant = row['assistant_text'] or ""
        # build prompts and examples
        prompt = [
            {"role":"system","content":[{"type":"text","text":system}]},
            {"role":"user","content":[{"type":"text","text":user_text}]},
        ]
        full = prompt + [{"role":"assistant","content":[{"type":"text","text":assistant}]}]

        # tokenize prompt
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, tokenize=True, return_tensors='pt'
        ).squeeze(0)
        p_len = prompt_ids.size(0)

        if self.inference_mode:
            inp = torch.cat([tokens, prompt_ids])
            return {
                'input_ids': inp,
                'attention_mask': inp != self.IGNORE_INDEX,
                'audio': mel_chunks,
                'audio_chunk_lengths': [m.shape[0] for m in mel_chunks],
                'audio_token_length': at_len,
                'prompt_length': p_len,
                'target': assistant,
            }

        # training mode: labels include generated part
        example_ids = self.tokenizer.apply_chat_template(
            full, add_generation_prompt=False, tokenize=True, return_tensors='pt'
        ).squeeze(0)
        inp = torch.cat([tokens, example_ids])
        labels = inp.clone()
        labels[:at_len + p_len] = self.IGNORE_INDEX

        return {
            'input_ids': inp,
            'labels': labels,
            'attention_mask': inp != self.IGNORE_INDEX,
            'audio': mel_chunks,
            'audio_chunk_lengths': [m.shape[0] for m in mel_chunks],
            'audio_token_length': at_len,
            'prompt_length': p_len,
        }

    def pad(self, sequence: torch.Tensor, max_length: int, padding_idx: int = 0) -> torch.Tensor:
        """Pad or trim a tensor along first (time) dimension."""
        cur = sequence.size(0)
        if cur < max_length:
            pad_shape = (max_length - cur,) + sequence.shape[1:]
            pad = torch.full(pad_shape, padding_idx, dtype=sequence.dtype, device=sequence.device)
            return torch.cat([sequence, pad], dim=0)
        return sequence[:max_length]

    def data_collator(self, samples: List[dict]) -> dict:
        """
        Pad a batch of samples with variable number of chunks and variable chunk lengths.
        Returns a tensor of shape (B, max_chunks, max_frames, n_mels).
        """
        # collect chunk counts and lengths
        chunk_counts = [len(s['audio']) for s in samples]
        max_chunks = max(chunk_counts)
        all_lens = [l for s in samples for l in s['audio_chunk_lengths']]
        max_frames = max(all_lens)
        n_mels = self.mel_size

        # pad audio chunks
        padded_audios, chunk_masks = [], []
        for s in samples:
            chunks, lengths = s['audio'], s['audio_chunk_lengths']
            tensor_chunks, tensor_masks = [], []
            for mel, L in zip(chunks, lengths):
                pad_mel = self.pad(mel, max_frames, padding_idx=0)
                mask = torch.cat([torch.ones(L, dtype=torch.bool), torch.zeros(max_frames-L, dtype=torch.bool)])
                tensor_chunks.append(pad_mel)
                tensor_masks.append(mask)
            # pad missing chunks
            for _ in range(max_chunks - len(chunks)):
                tensor_chunks.append(torch.zeros((max_frames, n_mels)))
                tensor_masks.append(torch.zeros((max_frames,), dtype=torch.bool))
            padded_audios.append(torch.stack(tensor_chunks))
            chunk_masks.append(torch.stack(tensor_masks))

        audio_batch = torch.stack(padded_audios)         # (B, max_chunks, max_frames, n_mels)
        audio_chunk_mask = torch.stack(chunk_masks)      # (B, max_chunks, max_frames)

        # now pad input_ids, labels, attention_mask as before…
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [s['input_ids'] for s in samples], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = input_ids != self.tokenizer.pad_token_id
        labels = torch.nn.utils.rnn.pad_sequence(
            [s.get('labels', torch.tensor([])) for s in samples], batch_first=True,
            padding_value=self.IGNORE_INDEX
        ) if 'labels' in samples[0] else None

        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'audio': audio_batch,
            'audio_chunk_mask': audio_chunk_mask,
        }
        if labels is not None:
            batch['labels'] = labels
        return batch

def get_dataset(dataset_dir: str, tokenizer: Any, model_config, train_config, split: str = "test", **kwargs) -> SpeechDataset:
    return SpeechDataset(dataset_dir, tokenizer, model_config, train_config, split, **kwargs)

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from configs import *
    from models.lucas_setup import set_tokenizer
    
    model_config = ModelConfig()
    train_config = TrainConfig()
    
    torch.manual_seed(train_config.seed)

    # Initialize model and tokenizer
    tokenizer = set_tokenizer(model_config)

    parquet_dir = "/home/hnaouara/data-server/datasets/audio/instruct/audio-context"
    ds = get_dataset(parquet_dir, tokenizer, model_config, train_config, split="test")
    print("Dataset size:", len(ds))
    # Create DataLoader
    dataloader = DataLoader(
        ds,
        batch_size=1,
        collate_fn=ds.data_collator,
        shuffle=False,
        num_workers=1,
    )
    # Iterate through the DataLoader   
    sample = next(iter(dataloader))
    # print("Sample:", sample)
    
    print("Input IDs:", sample["input_ids"].shape)  
    print("Labels:", sample["labels"].shape)
    print("Audio:", sample["audio"].shape)
    print("Audio mel post_mask:", sample["audio_mel_post_mask"])
    print("Attention Mask:", sample["attention_mask"].shape)
    print("Modality Mask:", sample["modality_mask"].shape)
    
    # Decode the input_ids
    tokens = sample["input_ids"]
    labels = sample["labels"]   
    full = tokenizer.decode(tokens[tokens != -100], skip_special_tokens=False)
    print("FULL:\n", full)
    
    print("LABELS:\n", tokenizer.decode(labels[labels != -100], skip_special_tokens=False))
    
