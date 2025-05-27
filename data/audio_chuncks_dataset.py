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
from datasets import Dataset
import gc
import pyarrow.parquet as pq
from glob import glob
from functools import lru_cache
import logging
from tqdm import tqdm

# 1) Configure logging once, at top of your script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
def convert_parquet_messages(messages_df, split):
    """
    Convert Parquet rows of 'messages' into columns for user/system/assistant text and user audio dict.
    Memory-optimized version that processes data in batches.
    """
    output_data = {"user_text": [], "user_audio": [], "system_text": [], "assistant_text": []}
    
    # Process in chunks to reduce peak memory usage
    chunk_size = 100
    # for start in tqdm(range(0, len(messages_df), chunk_size),desc=f"convert_chunks[{split}]", unit="chunk"):
    for start in range(0, len(messages_df), chunk_size):
        chunk = messages_df.iloc[start:start+chunk_size]
        
        for _, row in chunk.iterrows():
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
                                # Store as np.float16 to save memory, will convert back when needed
                                arr_data = np.array(audio_item['array'], dtype=np.float16)
                                sr = audio_item.get('sampling_rate')
                                path = audio_item.get('path')
                            elif audio_item.get('audio') is not None:
                                arr_data = np.array(audio_item['audio']['array'], dtype=np.float16)
                                sr = audio_item['audio'].get('sampling_rate')
                                path = audio_item['audio'].get('path')
                            if arr_data is not None:
                                message_dict['user_audio'] = {
                                    "array": arr_data, 
                                    "sampling_rate": sr, 
                                    "path": os.path.basename(path) if path else None
                                }
                    elif role == 'assistant':
                        text = next((c['text'] for c in contents if c.get('type')=='text'), None)
                        message_dict['assistant_text'] = text
            
            for k in output_data:
                output_data[k].append(message_dict[k])
        
    return output_data


def _read_parquet_without_dask(parquet_dirs, split='test') -> Dataset:
    """Load parquet files without using Dask, with improved memory handling and tqdm progress."""
    # Gather all parquet paths
    paths = []
    if isinstance(parquet_dirs, str):
        parquet_dirs = [parquet_dirs]
    for d in parquet_dirs:
        p = os.path.join(d, split)
        if os.path.exists(p):
            paths.extend(glob(os.path.join(p, '*.parquet')))
    if not paths:
        raise FileNotFoundError(f"No parquet files for split {split}")

    result = {
        "user_text": [],
        "user_audio": [],
        "system_text": [],
        "assistant_text": []
    }

    # tqdm over files
    for path in tqdm(paths, desc=f"Processing {split} parquet files", unit="file"):
        # Read only the 'messages' column to save memory
        table = pq.read_table(path, columns=['messages'])
        df = table.to_pandas()
        del table  # free memory

        # Convert messages to your desired dict structure
        part_result = convert_parquet_messages(df, split)

        # Append to global result
        for k in result:
            result[k].extend(part_result[k])

        # Cleanup
        del df, part_result
        gc.collect()

    return Dataset.from_dict(result)


@lru_cache(maxsize=8)  # Cache a few mel spectrograms to avoid recomputation
def _encode_chunk(chunk_bytes: bytes, mel_size: int, sr: int) -> torch.Tensor:
    """
    Convert audio bytes to log-mel spectrogram. Using bytes for caching.
    """
    # Convert bytes back to array
    chunk = np.frombuffer(chunk_bytes, dtype=np.float32)
    raw = whisper.pad_or_trim(chunk)
    mel = whisper.log_mel_spectrogram(raw, n_mels=mel_size).permute(1,0)
    return mel


def _make_pseudo_tokens(total_frames: int, tokenizer: Any, ignore_index: int) -> torch.Tensor:
    """Generate pseudo-tokens for audio, optimized to use less memory."""
    boa = tokenizer.convert_tokens_to_ids("<|start_of_audio|>")
    eoa = tokenizer.convert_tokens_to_ids("<|end_of_audio|>")
    total_tokens = ((total_frames + 1)//2)//5
    # Using torch.full is more memory efficient than creating and concatenating
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
    """
    Split an audio array into segments on silence and merge to respect max duration.
    Memory-optimized version.
    """
    # Convert to int16 only if needed
    if audio_array.dtype != np.int16:
        audio_array = (audio_array * 32767).astype(np.int16)
        
    # Create AudioSegment
    seg = AudioSegment(
        audio_array.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )
    # Free memory
    del audio_array
    gc.collect()
    
    # Split on silence
    pieces = split_on_silence(
        seg,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence
    )
    
    # Process pieces into chunks
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
    Split long audio into mel-chunks with memory optimizations.
    """
    # Convert to float32 only when needed
    if isinstance(audio_array, list):
        audio_array = np.array(audio_array, dtype=np.float32)
    elif audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)
        
    duration = len(audio_array) / sr

    # Handle short audio efficiently
    if round(duration, 2) <= chunk_duration_s:
        # Convert to bytes for caching
        audio_bytes = audio_array.tobytes()
        mel = _encode_chunk(audio_bytes, mel_size, sr)
        total_frames = mel.shape[0]
        tokens = _make_pseudo_tokens(total_frames, tokenizer, ignore_index)
        return tokens, tokens.size(0), [mel], total_frames

    # Split longer audio
    segments = split_audio_on_silence(
        audio_array, sr, silence_thresh, min_silence_len,
        keep_silence, min_segment_duration, max_segment_duration
    )
    
    # Free original array
    del audio_array
    gc.collect()
    
    # Process mel chunks one at a time
    mel_chunks = []
    total_frames = 0
    
    for seg in segments:
        samples = np.array(seg.get_array_of_samples(), dtype=np.int16).astype(np.float32) / 32767.0
        # Convert to bytes for caching
        samples_bytes = samples.tobytes()
        mel = _encode_chunk(samples_bytes, mel_size, sr)
        mel_chunks.append(mel)
        total_frames += mel.shape[0]
        # Free memory
        del samples
        
    tokens = _make_pseudo_tokens(total_frames, tokenizer, ignore_index)
    return tokens, tokens.size(0), mel_chunks, total_frames


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(
        self, dirs, tokenizer, model_config, train_config,
        split='test', inference_mode=False,
        cache_size=100  # Added cache size parameter
    ):
        self.ds = _read_parquet_without_dask(dirs, split)
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
        
        # Item cache to reduce repeated processing
        self._item_cache = {}
        self._item_cache_size = cache_size
        self._item_cache_keys = []

    def __len__(self):
        return len(self.ds)
    
    def _add_to_cache(self, idx, item):
        """Add item to cache with LRU implementation."""
        if len(self._item_cache_keys) >= self._item_cache_size:
            # Remove least recently used item
            old_key = self._item_cache_keys.pop(0)
            del self._item_cache[old_key]
        
        self._item_cache[idx] = item
        self._item_cache_keys.append(idx)
    
    def _get_from_cache(self, idx):
        """Get item from cache and update LRU order."""
        if idx in self._item_cache:
            # Move to end of list (most recently used)
            self._item_cache_keys.remove(idx)
            self._item_cache_keys.append(idx)
            return self._item_cache[idx]
        return None

    def __getitem__(self, idx: int) -> dict:
        # Check cache first
        # logging.debug(f"__getitem__ start idx={idx}")
        cached_item = self._get_from_cache(idx)
        if cached_item is not None:
            return cached_item
            
        row = self.ds[idx]
        system = row['system_text'] or dedent(
            "You are a helpful speech assistant who understands user input and aids with various tasks"
        )
        user_text = row['user_text'] or ""
        audio = row['user_audio']
        
        if audio is None:
            raise ValueError(f"No audio at index {idx}")
            
        # Convert from float16 to float32 for processing
        arr = audio.get('array')
        if isinstance(arr, list):
            arr = np.array(arr, dtype=np.float32)
        elif arr.dtype != np.float32:
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

        # Free the original array
        del arr
        
        assistant = row['assistant_text'] or ""
        # build prompts and examples
        prompt_template = (
            "<|start_header_id|>user<|end_header_id|>\n"
            "{user_message}\n<|eot_id|>\n\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
        formatted_prompt = prompt_template.format(
                                user_message=user_text.strip()
                            )
        prompt_ids = self.tokenizer.encode(formatted_prompt)
        p_len = len(prompt_ids)

        if self.inference_mode:
            prompt_tensor = torch.tensor(prompt_ids, dtype=torch.int64)
            inp = torch.cat([tokens, prompt_tensor])
            assistant = self.tokenizer.encode(assistant)
            result = {
                'input_ids': inp,
                'attention_mask': inp != self.IGNORE_INDEX,
                'audio': mel_chunks,
                'audio_chunk_lengths': [m.shape[0] for m in mel_chunks],
                'audio_token_length': at_len,
                'prompt_length': p_len,
                'labels': torch.tensor(assistant, dtype=torch.int64),
            }
            # Add to cache
            self._add_to_cache(idx, result)
            return result

        example = f"{formatted_prompt}\n{assistant.strip()}"
        example_ids = self.tokenizer.encode(example) + [self.tokenizer.eos_token_id]
        example_tensor = torch.tensor(example_ids, dtype=torch.int64)
        inp = torch.cat([tokens, example_tensor])
        labels = inp.clone()
        labels[:at_len + p_len] = self.IGNORE_INDEX

        result = {
            'input_ids': inp,
            'labels': labels,
            'attention_mask': inp != self.IGNORE_INDEX,
            'audio': mel_chunks,
            'audio_chunk_lengths': [m.shape[0] for m in mel_chunks],
            'audio_token_length': at_len,
            'prompt_length': p_len,
        }
        
        # Add to cache
        self._add_to_cache(idx, result)
        # logging.debug(f"__getitem__ done idx={idx}")
        return result

    def pad(self, sequence: torch.Tensor, max_length: int, padding_idx: int = 0) -> torch.Tensor:
        """Pad or trim a tensor along first (time) dimension. Memory-optimized version."""
        cur = sequence.size(0)
        if cur < max_length:
            # Create padding tensor with correct shape
            pad_shape = (max_length - cur,) + sequence.shape[1:]
            pad = torch.full(pad_shape, padding_idx, dtype=sequence.dtype, device=sequence.device)
            return torch.cat([sequence, pad], dim=0)
        # Slice original tensor to avoid copy
        return sequence[:max_length]

    def data_collator(self, samples: List[dict]) -> dict:
        """
        Pad a batch of samples with variable number of chunks and variable chunk lengths.
        Memory-optimized version.
        """
        if not samples:
            return {}
            
        # collect chunk counts and lengths
        chunk_counts = [len(s['audio']) for s in samples]
        max_chunks = max(chunk_counts)
        all_lens = [l for s in samples for l in s['audio_chunk_lengths']]
        max_frames = max(all_lens)
        n_mels = self.mel_size

        # preallocate tensors to avoid repeated allocations
        batch_size = len(samples)
        audio_batch = torch.zeros((batch_size, max_chunks, max_frames, n_mels))
        audio_chunk_mask = torch.zeros((batch_size, max_chunks, max_frames), dtype=torch.bool)
        
        # fill in data one sample at a time
        for b, s in enumerate(samples):
            chunks, lengths = s['audio'], s['audio_chunk_lengths']
            for c, (mel, L) in enumerate(zip(chunks, lengths)):
                # Pad mel and create mask
                pad_mel = self.pad(mel, max_frames, padding_idx=0)
                audio_batch[b, c, :pad_mel.size(0)] = pad_mel
                audio_chunk_mask[b, c, :L] = True
        
        # Handle other tensors
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [s['input_ids'] for s in samples], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = input_ids != self.tokenizer.pad_token_id
        
        # Compute the total length for the prompt part (audio tokens + prompt tokens)
        prompt_lengths = [
            sample["audio_token_length"] + sample["prompt_length"] for sample in samples
        ]
        
        max_prompt_length = max(prompt_lengths)
        # Create a modality mask indicating positions of audio tokens.
        modality_masks = torch.zeros_like(attention_mask)
        for idx, sample in enumerate(samples):
            # The left padding added for each sample is:
            left_pad = max_prompt_length - (sample["audio_token_length"] + sample["prompt_length"])
            modality_masks[idx, left_pad:left_pad + sample["audio_token_length"]] = -100
            
        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'audio': audio_batch,
            'audio_chunk_mask': audio_chunk_mask,
            "modality_mask": modality_masks
        }
        
        # Add labels if present
        if 'labels' in samples[0]:
            labels = torch.nn.utils.rnn.pad_sequence(
                [s['labels'] for s in samples], batch_first=True,
                padding_value=self.IGNORE_INDEX
            )
            batch['labels'] = labels
            
        return batch

    def __iter__(self) -> Iterator[dict]:
        """Implement iterator for memory-efficient lazy loading."""
        for i in range(len(self)):
            yield self[i]

def get_dataset(dataset_dir: str, tokenizer: Any, model_config, train_config, split: str = "test", **kwargs) -> SpeechDataset:
    return SpeechDataset(dataset_dir, tokenizer, model_config, train_config, split, **kwargs)


if __name__ == "__main__":
    from configs import *
    from models.lucas_setup import set_tokenizer
    from torch.utils.data import DataLoader
    model_config = ModelConfig()
    train_config = TrainConfig()
    
    torch.manual_seed(train_config.seed)

    # Initialize model and tokenizer
    tokenizer = set_tokenizer(model_config.llm_name_hf)

    parquet_dir = sys.argv[1]
    
    ds = get_dataset(
        parquet_dir, 
        tokenizer, 
        model_config,
        train_config,
        split="test",
        inference_mode=False
        )
    print("Dataset size:", len(ds))
    
    # Use our memory-efficient data loader instead
    dataloader = DataLoader(
        ds,
        batch_size=2,
        collate_fn=ds.data_collator,
        shuffle=False,
        num_workers=1,
    )
    # Iterate through the DataLoader   
    samples = next(iter(dataloader))
    counter = 1
    batch_size = samples["input_ids"].shape[0]
    for i in range(batch_size):
        print("#" * 40)
        print(f"Sample #{i + 1}")
        print("Input IDs:", samples["input_ids"][i].shape)

        if "labels" in samples:
            print("Labels:", samples["labels"][i].shape)
        print("Audio:", samples["audio"][i].shape)
        print("Audio Chunk Mask:", samples["audio_chunk_mask"][i].shape)
        print("Attention Mask:", samples["attention_mask"][i].shape)

        tokens = samples["input_ids"][i]
        full = tokenizer.decode(tokens[tokens != -100], skip_special_tokens=False)
        print("FULL:\n", full)

        if "labels" in samples:
            labels = samples["labels"][i]
            print("LABELS:\n", tokenizer.decode(labels[labels != -100], skip_special_tokens=False))
