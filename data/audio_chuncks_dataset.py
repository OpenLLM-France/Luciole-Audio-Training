import os
import sys
import numpy as np
import torch
import whisper
import textwrap
import typing
import pydub
import datasets
import gc
import logging
from tqdm import tqdm
from functools import lru_cache
from pathlib import Path
import warnings
import time
from concurrent.futures import ThreadPoolExecutor
import threading

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
                        arr = np.array(audio_item['array'], dtype=np.float16)
                        sr = audio_item.get('sampling_rate')
                        path = audio_item.get('path')
                    elif audio_item.get('audio') is not None:
                        arr = np.array(audio_item['audio']['array'], dtype=np.float16)
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

features = datasets.Features({
    "messages": [
        {
            "role": datasets.Value("string"),
            "content": [
                {
                    "array": datasets.Sequence(datasets.Value("float32")),
                    "audio": {
                        "array": datasets.Sequence(datasets.Value("float32")),
                        "path": datasets.Value("string"),
                        "sampling_rate": datasets.Value("int64")
                    },
                    "path": datasets.Value("string"),
                    "sampling_rate": datasets.Value("float64"),
                    "text": datasets.Value("string"),
                    "type": datasets.Value("string")
                }
            ]
        }
    ]
})


@lru_cache(maxsize=32)  # Increased cache size
def _encode_chunk(chunk_bytes: bytes, mel_size: int, sr: int) -> torch.Tensor:
    """Convert audio bytes to log-mel spectrogram with error handling."""
    try:
        chunk = np.frombuffer(chunk_bytes, dtype=np.float16)
        if len(chunk) == 0:
            return torch.zeros((1, mel_size))
        
        raw = whisper.pad_or_trim(chunk)
        mel = whisper.log_mel_spectrogram(raw, n_mels=mel_size).permute(1, 0)
        return mel
    except Exception as e:
        logging.error(f"Error encoding chunk: {e}")
        return torch.zeros((1, mel_size))

def _make_pseudo_tokens(total_frames: int, tokenizer: typing.Any, ignore_index: int) -> torch.Tensor:
    """Generate pseudo-tokens for audio."""
    try:
        boa = tokenizer.convert_tokens_to_ids("<|start_of_audio|>")
        eoa = tokenizer.convert_tokens_to_ids("<|end_of_audio|>")
        total_tokens = max(1, ((total_frames + 1) // 2) // 5)
        tokens = torch.full((total_tokens + 2,), ignore_index, dtype=torch.long)
        tokens[0] = boa
        tokens[-1] = eoa
        return tokens
    except Exception as e:
        logging.error(f"Error creating pseudo tokens: {e}")
        # Return minimal tokens
        return torch.tensor([0, ignore_index, 1], dtype=torch.long)

def split_audio_on_silence(
    audio_array: np.ndarray,
    sample_rate: int = 16000,
    silence_thresh: int = -40,
    min_silence_len: int = 400,
    keep_silence: int = 300,
    min_segment_duration: float = 0.1,
    max_segment_duration: float = 30.0
) -> typing.List[pydub.AudioSegment]:
    """Split audio on silence with enhanced error handling."""
    if len(audio_array) == 0:
        return []
    
    try:
        # Convert to int16 if needed
        if audio_array.dtype != np.int16:
            audio_array = np.clip(audio_array, -1.0, 1.0)
            audio_array = (audio_array * 32767).astype(np.int16)
            
        seg = pydub.AudioSegment(
            audio_array.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
        
        # Free memory early
        del audio_array
        
        # Split on silence with timeout protection
        pieces = pydub.silence.split_on_silence(
            seg,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence
        )
        
        # If no pieces found, return the original segment
        if not pieces:
            return [seg]
        
        # Combine pieces intelligently
        chunks = []
        current = pydub.AudioSegment.empty()
        
        for piece in pieces:
            if len(current) + len(piece) <= max_segment_duration * 1000:
                current += piece
            else:
                if len(current) >= min_segment_duration * 1000:
                    chunks.append(current)
                current = piece
                    
        if len(current) >= min_segment_duration * 1000 or not chunks:
            chunks.append(current)
            
        return chunks if chunks else [seg]
        
    except Exception as e:
        logging.error(f"Error splitting audio: {e}")
        # Return original audio as single segment
        try:
            if audio_array.dtype != np.int16:
                audio_array = np.clip(audio_array, -1.0, 1.0)
                audio_array = (audio_array * 32767).astype(np.int16)
            seg = pydub.AudioSegment(
                audio_array.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,
                channels=1
            )
            return [seg]
        except:
            return []

def process_audio_long(
    audio_array: np.ndarray,
    tokenizer: typing.Any,
    mel_size: int,
    ignore_index: int = -100,
    sr: int = whisper.audio.SAMPLE_RATE,
    chunk_duration_s: int = 30,
    silence_thresh: int = -40,
    min_silence_len: int = 400,
    keep_silence: int = 300,
    min_segment_duration: float = 0.3,  # Now 0.3s minimum
    max_segment_duration: float = 30.0,
    max_duration: int = 60,  # 60s maximum
    min_duration: float = 0.3  # New minimum duration parameter
) -> typing.Tuple[torch.Tensor, int, typing.List[torch.Tensor], int]:
    """
    Process audio with strict duration constraints:
    - Skips audio shorter than min_duration (0.3s)
    - Skips audio longer than max_duration (60s)
    - Processes 0.3-30s audio directly
    - Splits 30-60s audio on silence
    """
    
    def create_dummy_output():
        """Helper to create consistent dummy output for skipped/invalid audio"""
        empty_mel = torch.zeros((1, mel_size))
        tokens = _make_pseudo_tokens(1, tokenizer, ignore_index)
        return tokens, tokens.size(0), [empty_mel], 1

    try:
        # Convert and validate input audio
        if isinstance(audio_array, list):
            audio_array = np.array(audio_array, dtype=np.float32)
        elif audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        duration = len(audio_array) / sr
        logging.debug(f"Audio duration: {duration:.3f}s")

        # Duration validation
        if duration < min_duration:
            logging.warning(f"Skipping audio: Too short ({duration:.3f}s < {min_duration}s)")
            return create_dummy_output()
            
        if duration > max_duration:
            logging.warning(f"Skipping audio: Too long ({duration:.3f}s > {max_duration}s)")
            return create_dummy_output()

        # Direct processing for short segments (0.3s - 30s)
        if duration <= chunk_duration_s:
            try:
                # Normalize and convert to bytes
                audio_norm = audio_array / np.max(np.abs(audio_array))
                audio_bytes = audio_norm.astype(np.float32).tobytes()
                
                mel = _encode_chunk(audio_bytes, mel_size, sr)
                total_frames = mel.shape[0]
                tokens = _make_pseudo_tokens(total_frames, tokenizer, ignore_index)
                return tokens, tokens.size(0), [mel], total_frames
            except Exception as e:
                logging.error(f"Short audio processing failed: {str(e)}")
                return create_dummy_output()

        # Silence-based splitting for medium segments (30s - 60s)
        logging.info(f"Processing medium audio ({duration:.2f}s) with silence splitting")
        try:
            segments = split_audio_on_silence(
                audio_array, 
                sr, 
                silence_thresh, 
                min_silence_len,
                keep_silence, 
                min_segment_duration,  # Now respects 0.3s minimum
                max_segment_duration
            )

            if not segments:
                logging.warning("Silence splitting produced no valid segments")
                return create_dummy_output()

            mel_chunks = []
            total_frames = 0
            valid_segments = 0

            for seg in segments:
                seg_duration = len(seg) / sr
                if seg_duration < min_duration:
                    logging.debug(f"Skipping segment: Too short ({seg_duration:.3f}s)")
                    continue
                    
                try:
                    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
                    samples = samples / 32767.0  # Proper normalization
                    mel = _encode_chunk(samples.tobytes(), mel_size, sr)
                    mel_chunks.append(mel)
                    total_frames += mel.shape[0]
                    valid_segments += 1
                except Exception as e:
                    logging.error(f"Segment processing failed: {str(e)}")

            if valid_segments == 0:
                logging.warning("No valid segments after processing")
                return create_dummy_output()

            tokens = _make_pseudo_tokens(total_frames, tokenizer, ignore_index)
            return tokens, tokens.size(0), mel_chunks, total_frames

        except Exception as e:
            logging.error(f"Silence splitting failed: {str(e)}")
            return create_dummy_output()

    except Exception as e:
        logging.error(f"Audio processing error: {str(e)}")
        return create_dummy_output()

def _read_parquet(parquet_dirs: typing.List[str], split: str = 'test') -> datasets.IterableDataset: #datasets.Dataset:
    import glob
    import torch.distributed as dist

    paths = []
    if isinstance(parquet_dirs, str):
        parquet_dirs = [parquet_dirs]
    for d in parquet_dirs:
        p = os.path.join(d, split)
        if os.path.exists(p):
            found_files = glob.glob(os.path.join(p, '*.parquet'))
            paths.extend(found_files)

    if not paths:
        raise FileNotFoundError(f"No parquet files for split {split}")

    logging.info(f"Found {len(paths)} parquet files for split {split}")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            ds = datasets.load_dataset(
                'parquet',
                data_files=paths,
                split='train',
                streaming=True,  # now loading all into memory
                features=features,
                cache_dir=None,
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"Failed to load dataset (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)
            else:
                raise

    ds = ds.map(convert_row, remove_columns=['messages'])

    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        if world_size > 1:
            ds = ds.shard(num_shards=world_size, index=rank)
            logging.info(f"[DDP] Sharded dataset: rank={rank} / {world_size}")
    else:
        logging.info("[DDP] Not initialized, using full dataset")

    return ds


class SpeechDataset(torch.utils.data.IterableDataset): #(torch.utils.data.Dataset):
    def __init__(
        self, 
        dirs: typing.Union[str, typing.List[str]], 
        tokenizer: typing.Any, 
        model_config: typing.Any, 
        train_config: typing.Any,
        split: str = 'test', 
        inference_mode: bool = False,
    ):
        self.ds = _read_parquet(dirs, split)
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
        self.max_duration = getattr(train_config, 'max_duration', 60)  # Default to 60 seconds
        self.min_duration = getattr(train_config, 'min_duration', 0.3)  # Default to 0.3 seconds
        
        if not hasattr(self.tokenizer, 'pad_token_id') or self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        logging.info(f"Initialized SpeechDataset with split='{split}', inference_mode={inference_mode}")

    def __iter__(self):
        for row in self.ds:
            try:
                system = row.get('system_text') or textwrap.dedent(
                    "You are a helpful speech assistant who understands user input and aids with various tasks"
                )
                user_text = row.get('user_text') or ""
                assistant = row.get('assistant_text') or ""
                audio = row.get('user_audio')

                if audio is None or not assistant.strip():
                    continue

                arr = audio.get('array')
                if arr is None or len(arr) == 0:
                    continue

                if isinstance(arr, list):
                    arr = np.array(arr, dtype=np.float16)
                elif arr.dtype != np.float16:
                    arr = arr.astype(np.float16)

                sr = audio.get('sampling_rate', whisper.audio.SAMPLE_RATE)
                if sr != whisper.audio.SAMPLE_RATE:
                    import librosa
                    arr = librosa.resample(arr, orig_sr=sr, target_sr=whisper.audio.SAMPLE_RATE)

                tokens, at_len, mel_chunks, a_len = process_audio_long(
                    arr, self.tokenizer, self.mel_size, self.IGNORE_INDEX,
                    whisper.audio.SAMPLE_RATE,
                    self.chunk_duration_s, self.silence_thresh,
                    self.min_silence_len, self.keep_silence,
                    self.min_segment_duration, self.max_segment_duration, 
                    self.max_duration, self.min_duration
                )

                if tokens is None or mel_chunks is None:
                    continue

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
                    assistant_ids = self.tokenizer.encode(assistant)
                    yield {
                        'input_ids': inp,
                        'attention_mask': inp != self.IGNORE_INDEX,
                        'audio': mel_chunks,
                        'audio_chunk_lengths': [m.shape[0] for m in mel_chunks],
                        'audio_token_length': at_len,
                        'prompt_length': p_len,
                        'labels': torch.tensor(assistant_ids, dtype=torch.int64),
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
                logging.warning(f"[SKIP] Failed to process a sample: {e}")        
                yield None

    def pad(self, sequence: torch.Tensor, max_length: int, padding_idx: int = 0) -> torch.Tensor:
        cur = sequence.size(0)
        if cur < max_length:
            pad_shape = (max_length - cur,) + sequence.shape[1:]
            pad = torch.full(pad_shape, padding_idx, dtype=sequence.dtype)
            return torch.cat([sequence, pad], dim=0)
        return sequence[:max_length]

    def data_collator(self, samples: typing.List[typing.Optional[dict]]) -> dict:
        # Filter out None samples
        samples = [s for s in samples if s is not None]
        if not samples:
            return {}

        try:
            chunk_counts = [len(s['audio']) for s in samples]
            max_chunks = max(chunk_counts)
            all_lens = [l for s in samples for l in s['audio_chunk_lengths']]
            max_frames = max(all_lens) if all_lens else 1
            n_mels = self.mel_size
            batch_size = len(samples)

            audio_batch = torch.zeros((batch_size, max_chunks, max_frames, n_mels))
            audio_chunk_mask = torch.zeros((batch_size, max_chunks, max_frames), dtype=torch.bool)

            for b, s in enumerate(samples):
                chunks, lengths = s['audio'], s['audio_chunk_lengths']
                for c, (mel, L) in enumerate(zip(chunks, lengths)):
                    if c < max_chunks and L > 0:
                        pad_mel = self.pad(mel, max_frames, padding_idx=0)
                        audio_batch[b, c, :pad_mel.size(0)] = pad_mel
                        audio_chunk_mask[b, c, :L] = True

            input_ids = torch.nn.utils.rnn.pad_sequence(
                [s['input_ids'] for s in samples], 
                batch_first=True, 
                padding_value=self.tokenizer.pad_token_id
            )
            attention_mask = input_ids != self.tokenizer.pad_token_id

            prompt_lengths = [
                s["audio_token_length"] + s["prompt_length"] for s in samples
            ]
            max_prompt_length = max(prompt_lengths)
            modality_masks = torch.zeros_like(attention_mask)

            for i, s in enumerate(samples):
                left_pad = max_prompt_length - (s["audio_token_length"] + s["prompt_length"])
                end_idx = min(left_pad + s["audio_token_length"], modality_masks.size(1))
                if end_idx > left_pad:
                    modality_masks[i, left_pad:end_idx] = self.IGNORE_INDEX

            batch = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'audio': audio_batch,
                'audio_chunk_mask': audio_chunk_mask,
                'modality_mask': modality_masks
            }

            if 'labels' in samples[0]:
                labels = torch.nn.utils.rnn.pad_sequence(
                    [s['labels'] for s in samples],
                    batch_first=True,
                    padding_value=self.IGNORE_INDEX
                )
                batch['labels'] = labels

            return batch

        except Exception as e:
            logging.error(f"Error in data collator: {e}")
            return {
                'input_ids': torch.tensor([[self.tokenizer.pad_token_id]]),
                'attention_mask': torch.tensor([[True]]),
                'audio': torch.zeros((1, 1, 1, self.mel_size)),
                'audio_chunk_mask': torch.tensor([[[False]]]),
                'modality_mask': torch.tensor([[self.IGNORE_INDEX]]),
                'labels': torch.tensor([[self.IGNORE_INDEX]]) if 'labels' in samples[0] else None
            }


def get_dataset(
    dataset_dir: typing.Union[str, typing.List[str]], 
    tokenizer: typing.Any, 
    model_config: typing.Any, 
    train_config: typing.Any, 
    split: str = "test", 
    **kwargs
) -> SpeechDataset:
    """Create a SpeechDataset instance."""
    return SpeechDataset(dataset_dir, tokenizer, model_config, train_config, split, **kwargs)

if __name__ == "__main__":
    # Import required modules
    try:
        from configs import *
        from models.lucas_setup import set_tokenizer
    except ImportError as e:
        logging.error(f"Error importing required modules: {e}")
        sys.exit(1)
        
    from torch.utils.data import DataLoader
    
    # Check command line arguments
    if len(sys.argv) < 3:
        print("Usage: python script.py <parquet_directory> <split>")
        sys.exit(1)
    
    model_config = ModelConfig()
    train_config = TrainConfig()
    
    torch.manual_seed(train_config.seed)
    tokenizer = set_tokenizer(model_config.llm_name_hf)
    parquet_dir = sys.argv[1]
    split = sys.argv[2]
    
    # Verify directory exists
    if not os.path.exists(parquet_dir):
        logging.error(f"Directory does not exist: {parquet_dir}")
        sys.exit(1)
    
    logging.info(f"Creating dataset from {parquet_dir} with split '{split}'")
    
    try:
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
            batch_size=1,  # Reduced batch size for stability
            collate_fn=ds.data_collator,
            shuffle=False,
            num_workers=0,  # Set to 0 for debugging
        )
        
        # Process and inspect batches
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            if not batch:  # Skip empty batches
                continue
                
            print("#" * 50)
            print(f"Batch #{batch_idx + 1}")
            print("Input IDs shape:", batch["input_ids"].shape)
            print("Audio shape:", batch["audio"].shape)
            print("Audio Chunk Mask shape:", batch["audio_chunk_mask"].shape)
            
            if 'labels' in batch:
                print("Labels shape:", batch["labels"].shape)
            
            # Print detailed info for each sample in the batch
            for sample_idx in range(batch["input_ids"].shape[0]):
                print("\n" + "=" * 40)
                print(f"Sample #{sample_idx + 1} in batch")
                
                # Get the non-padded parts of each tensor
                input_ids = batch["input_ids"][sample_idx]
                attention_mask = batch["attention_mask"][sample_idx]
                actual_input = input_ids[attention_mask.bool()]
                
                # Convert to numpy and filter valid tokens
                # if len(actual_input) > 0:
                #     actual_input = actual_input.cpu().numpy()
                #     valid_tokens = actual_input[
                #         (actual_input >= 0) & (actual_input < len(tokenizer))
                #     ]
                    
                #     # print(f"Valid tokens count: {len(valid_tokens)}")
                #     # print("\nInput Text:")
                #     # try:
                #     #     decoded_text = tokenizer.decode(valid_tokens, skip_special_tokens=False)
                #     #     print(decoded_text[:500] + "..." if len(decoded_text) > 500 else decoded_text)
                #     # except Exception as e:
                #     #     print(f"Error decoding tokens: {e}")
                #     #     print("Token IDs:", valid_tokens[:10], "...")
                # else:
                #     print("No valid input tokens found")
                
                # Handle labels
                if 'labels' in batch:
                    labels = batch["labels"][sample_idx]
                    actual_labels = labels[labels != ds.IGNORE_INDEX]
                    
                    if len(actual_labels) > 0:
                        actual_labels = actual_labels.cpu().numpy()
                        valid_labels = actual_labels[
                            (actual_labels >= 0) & (actual_labels < len(tokenizer))
                        ]
                        
                        print(f"\nValid label tokens count: {len(valid_labels)}")
                        print("Labels Text:")
                        try:
                            decoded_labels = tokenizer.decode(valid_labels, skip_special_tokens=False)
                            print(decoded_labels[:300] + "..." if len(decoded_labels) > 300 else decoded_labels)
                        except Exception as e:
                            print(f"Error decoding labels: {e}")
                            print("Label IDs:", valid_labels[:10], "...")
                    else:
                        print("No valid label tokens found")
                
                print(f"\nAudio chunks: {len([1 for chunk in batch['audio'][sample_idx] if chunk.sum() > 0])}")
                print(f"Modality mask shape: {batch['modality_mask'][sample_idx].shape}")

            # Only show first few batches for demo
            if batch_idx >= 3:  # Show first 3 batches
                break
                
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise