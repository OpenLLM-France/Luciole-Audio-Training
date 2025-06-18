import os
import sys
sys.path.append('/home/hnaoura/hnaouara_Storage1/Audio_Adapter_Training')
import numpy as np
import torch
import whisper
from textwrap import dedent
from typing import Any, List, Tuple, Dict, Optional, Iterator
from pydub import AudioSegment
from pydub.silence import split_on_silence
from datasets import load_dataset, IterableDataset
import gc
import logging
from tqdm import tqdm
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

def convert_row(row: dict) -> dict:
    """Convert a single row from the parquet file into our desired format."""
    message_dict = {
        "user_text": None,
        "user_audio": None,
        "system_text": None,
        "assistant_text": None
    }
    
    messages = row.get('messages', [])
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
                        message_dict['user_audio'] = {
                            "array": arr, 
                            "sampling_rate": sr, 
                            "path": os.path.basename(path) if path else None
                        }
            elif role == 'assistant':
                text = next((c['text'] for c in contents if c.get('type')=='text'), None)
                message_dict['assistant_text'] = text
                
    return message_dict

def _read_parquet_streaming(parquet_dirs: List[str], split: str = 'test') -> IterableDataset:
    """Load parquet files in streaming mode."""
    import glob
    paths = []
    if isinstance(parquet_dirs, str):
        parquet_dirs = [parquet_dirs]
    for d in parquet_dirs:
        p = os.path.join(d, split)
        if os.path.exists(p):
            paths.extend(glob.glob(os.path.join(p, '*.parquet')))
    if not paths:
        raise FileNotFoundError(f"No parquet files for split {split}")
    
    # Create streaming dataset
    ds = load_dataset(
        'parquet',
        data_files=paths,
        split='train',
        streaming=True
    )
    
    # Convert each row on-the-fly
    ds = ds.map(convert_row, remove_columns=['messages'])
    return ds

@lru_cache(maxsize=8)
def _encode_chunk(chunk_bytes: bytes, mel_size: int, sr: int) -> torch.Tensor:
    """Convert audio bytes to log-mel spectrogram."""
    chunk = np.frombuffer(chunk_bytes, dtype=np.float32)
    raw = whisper.pad_or_trim(chunk)
    mel = whisper.log_mel_spectrogram(raw, n_mels=mel_size).permute(1,0)
    return mel

def _make_pseudo_tokens(total_frames: int, tokenizer: Any, ignore_index: int) -> torch.Tensor:
    """Generate pseudo-tokens for audio."""
    boa = tokenizer.convert_tokens_to_ids("<|start_of_audio|>")
    eoa = tokenizer.convert_tokens_to_ids("<|end_of_audio|>")
    total_tokens = ((total_frames + 1) // 2) // 5
    tokens = torch.full((total_tokens + 2,), ignore_index, dtype=torch.long)
    tokens[0] = boa
    tokens[-1] = eoa
    return tokens

def split_audio_on_silence(
    audio_array: np.ndarray,
    sample_rate: int = 16000,
    silence_thresh: int = -40,
    min_silence_len: int = 400,
    keep_silence: int = 300,
    min_segment_duration: float = 0.1,
    max_segment_duration: float = 30.0
) -> List[AudioSegment]:
    """Split audio on silence with memory optimization."""
    if audio_array.dtype != np.int16:
        audio_array = (audio_array * 32767).astype(np.int16)
        
    seg = AudioSegment(
        audio_array.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )
    del audio_array
    gc.collect()
    
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
    """Process audio with memory optimizations."""
    if isinstance(audio_array, list):
        audio_array = np.array(audio_array, dtype=np.float32)
    elif audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)
        
    duration = len(audio_array) / sr

    if round(duration, 2) <= chunk_duration_s:
        audio_bytes = audio_array.tobytes()
        mel = _encode_chunk(audio_bytes, mel_size, sr)
        total_frames = mel.shape[0]
        tokens = _make_pseudo_tokens(total_frames, tokenizer, ignore_index)
        return tokens, tokens.size(0), [mel], total_frames

    segments = split_audio_on_silence(
        audio_array, sr, silence_thresh, min_silence_len,
        keep_silence, min_segment_duration, max_segment_duration
    )
    
    del audio_array
    gc.collect()
    
    mel_chunks = []
    total_frames = 0
    
    for seg in segments:
        samples = np.array(seg.get_array_of_samples(), dtype=np.int16).astype(np.float32) / 32767.0
        samples_bytes = samples.tobytes()
        mel = _encode_chunk(samples_bytes, mel_size, sr)
        mel_chunks.append(mel)
        total_frames += mel.shape[0]
        del samples
        
    tokens = _make_pseudo_tokens(total_frames, tokenizer, ignore_index)
    return tokens, tokens.size(0), mel_chunks, total_frames

class SpeechDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        dirs, 
        tokenizer, 
        model_config, 
        train_config,
        split='test', 
        inference_mode=False
    ):
        self.ds = _read_parquet_streaming(dirs, split)
        self.tokenizer = tokenizer
        self.mel_size = getattr(model_config, 'mel_size', 80)
        self.IGNORE_INDEX = -100
        self.inference_mode = inference_mode
        self.chunk_duration_s = getattr(train_config, 'chunk_duration_per_s', 30)
        self.silence_thresh = getattr(train_config, 'silence_thresh', -40)
        self.min_silence_len = getattr(train_config, 'min_silence_len', 400)
        self.keep_silence = getattr(train_config, 'keep_silence', 300)
        self.min_segment_duration = getattr(train_config, 'min_segment_duration', 0.1)
        self.max_segment_duration = getattr(train_config, 'max_segment_duration', 30.0)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Streaming iterator processing one example at a time."""
        for row in self.ds:
            try:
                system = row['system_text'] or dedent(
                    "You are a helpful speech assistant who understands user input and aids with various tasks"
                )
                user_text = row['user_text'] or ""
                audio = row['user_audio']
                
                if audio is None:
                    continue
                    
                arr = audio.get('array')
                if isinstance(arr, list):
                    arr = np.array(arr, dtype=np.float32)
                elif arr is not None and arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                    
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

                del arr
                gc.collect()

                assistant = row['assistant_text'] or ""
                prompt_template = (
                    "<|start_header_id|>user<|end_header_id|>\n"
                    "{user_message}\n<|eot_id|>\n\n"
                    "<|start_header_id|>assistant<|end_header_id|>\n"
                )
                formatted_prompt = prompt_template.format(user_message=user_text.strip())
                prompt_ids = self.tokenizer.encode(formatted_prompt)
                p_len = len(prompt_ids)

                if self.inference_mode:
                    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.int64)
                    inp = torch.cat([tokens, prompt_tensor])
                    assistant = self.tokenizer.encode(assistant)
                    yield {
                        'input_ids': inp,
                        'attention_mask': inp != self.IGNORE_INDEX,
                        'audio': mel_chunks,
                        'audio_chunk_lengths': [m.shape[0] for m in mel_chunks],
                        'audio_token_length': at_len,
                        'prompt_length': p_len,
                        'labels': torch.tensor(assistant, dtype=torch.int64),
                    }
                else:
                    example = f"{formatted_prompt}\n{assistant.strip()}"
                    example_ids = self.tokenizer.encode(example) + [self.tokenizer.eos_token_id]
                    example_tensor = torch.tensor(example_ids, dtype=torch.int64)
                    inp = torch.cat([tokens, example_tensor])
                    labels = inp.clone()
                    labels[:at_len + p_len] = self.IGNORE_INDEX

                    yield {
                        'input_ids': inp,
                        'labels': labels,
                        'attention_mask': inp != self.IGNORE_INDEX,
                        'audio': mel_chunks,
                        'audio_chunk_lengths': [m.shape[0] for m in mel_chunks],
                        'audio_token_length': at_len,
                        'prompt_length': p_len,
                    }
                    
            except Exception as e:
                logging.error(f"Error processing example: {e}")
                continue

    def pad(self, sequence: torch.Tensor, max_length: int, padding_idx: int = 0) -> torch.Tensor:
        cur = sequence.size(0)
        if cur < max_length:
            pad_shape = (max_length - cur,) + sequence.shape[1:]
            pad = torch.full(pad_shape, padding_idx, dtype=sequence.dtype)
            return torch.cat([sequence, pad], dim=0)
        return sequence[:max_length]

    def data_collator(self, samples: List[dict]) -> dict:
        if not samples:
            return {}
            
        chunk_counts = [len(s['audio']) for s in samples]
        max_chunks = max(chunk_counts)
        all_lens = [l for s in samples for l in s['audio_chunk_lengths']]
        max_frames = max(all_lens)
        n_mels = self.mel_size

        batch_size = len(samples)
        audio_batch = torch.zeros((batch_size, max_chunks, max_frames, n_mels))
        audio_chunk_mask = torch.zeros((batch_size, max_chunks, max_frames), dtype=torch.bool)
        
        for b, s in enumerate(samples):
            chunks, lengths = s['audio'], s['audio_chunk_lengths']
            for c, (mel, L) in enumerate(zip(chunks, lengths)):
                pad_mel = self.pad(mel, max_frames, padding_idx=0)
                audio_batch[b, c, :pad_mel.size(0)] = pad_mel
                audio_chunk_mask[b, c, :L] = True
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [s['input_ids'] for s in samples], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = input_ids != self.tokenizer.pad_token_id
        
        prompt_lengths = [
            sample["audio_token_length"] + sample["prompt_length"] for sample in samples
        ]
        max_prompt_length = max(prompt_lengths)
        modality_masks = torch.zeros_like(attention_mask)
        for idx, sample in enumerate(samples):
            left_pad = max_prompt_length - (sample["audio_token_length"] + sample["prompt_length"])
            modality_masks[idx, left_pad:left_pad + sample["audio_token_length"]] = self.IGNORE_INDEX
            
        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'audio': audio_batch,
            'audio_chunk_mask': audio_chunk_mask,
            "modality_mask": modality_masks
        }
        
        if 'labels' in samples[0]:
            labels = torch.nn.utils.rnn.pad_sequence(
                [s['labels'] for s in samples], batch_first=True,
                padding_value=self.IGNORE_INDEX
            )
            batch['labels'] = labels
            
        return batch

def get_dataset(dataset_dir: str, tokenizer: Any, model_config, train_config, split: str = "test", **kwargs) -> SpeechDataset:
    return SpeechDataset(dataset_dir, tokenizer, model_config, train_config, split, **kwargs)

if __name__ == "__main__":
    from configs import *
    from models.lucas_setup import set_tokenizer
    from torch.utils.data import DataLoader
    model_config = ModelConfig()
    train_config = TrainConfig()
    
    torch.manual_seed(train_config.seed)
    tokenizer = set_tokenizer(model_config.llm_name_hf)
    parquet_dir = sys.argv[1]
    split = sys.argv[2]
    ds = get_dataset(
        parquet_dir, 
        tokenizer, 
        model_config,
        train_config,
        split=split,
        inference_mode=False
    )
    
    dataloader = DataLoader(
        ds,
        batch_size=3,
        collate_fn=ds.data_collator,
        shuffle=False,
        num_workers=1,
    )
    
    # Process and inspect first batch in detail
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing")):
        print("#" * 40)
        print(f"Batch #{batch_idx}")
        print("Input IDs shape:", batch["input_ids"].shape)
        print("Audio shape:", batch["audio"].shape)
        print("Audio Chunk Mask shape:", batch["audio_chunk_mask"].shape)
        
        if 'labels' in batch:
            print("Labels shape:", batch["labels"].shape)
        
        # Print detailed info for each sample in the batch
        for sample_idx in range(batch["input_ids"].shape[0]):
            print("\n" + "=" * 30)
            print(f"Sample #{sample_idx + 1} in batch")
            
            # Get the non-padded parts of each tensor
            input_ids = batch["input_ids"][sample_idx]
            attention_mask = batch["attention_mask"][sample_idx]
            actual_input = input_ids[attention_mask.bool()].cpu().numpy()  # Convert to numpy array
            
            # Filter out any invalid token IDs
            valid_tokens = actual_input[(actual_input >= 0) & (actual_input < len(tokenizer))]
            
            print("\nInput Text:")
            try:
                print(tokenizer.decode(valid_tokens, skip_special_tokens=False))
            except Exception as e:
                print(f"Error decoding tokens: {e}")
                print("Token IDs:", valid_tokens)
            
            if 'labels' in batch:
                labels = batch["labels"][sample_idx]
                actual_labels = labels[labels != ds.IGNORE_INDEX].cpu().numpy()
                valid_labels = actual_labels[(actual_labels >= 0) & (actual_labels < len(tokenizer))]
                
                print("\nLabels Text:")
                try:
                    print(tokenizer.decode(valid_labels, skip_special_tokens=False))
                except Exception as e:
                    print(f"Error decoding labels: {e}")
                    print("Label IDs:", valid_labels)
            
            print(f"\nModality Mask: {len(batch['modality_mask'][sample_idx])}")

        
        # Only show first batch for demo purposes
        if batch_idx >= 0:  # Change to higher number to see more batches
            break