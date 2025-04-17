import os
import numpy as np
import itertools
from typing import Dict, Any
import whisper
import torch
from textwrap import dedent
import dask.dataframe as dd
from datasets import Dataset, DatasetDict
from dask.diagnostics import ProgressBar
import numpy as np

def convert_parquet_messages(messages_df):
    """
    Convert a list of messages with NumPy arrays in 'content' into the desired format.
    Ensures audio arrays are float32 and returns a dict with user text, user audio, system text (if present), 
    and assistant text.
    """
    output_data = {
        "user_text": [],
        "user_audio": [],
        "system_text": [],
        "assistant_text": []
    }

    for _, row in messages_df.iterrows():
        messages = row['messages']  # Assuming the column name is 'messages'

        if isinstance(messages, (np.ndarray, list)):
            message_dict = {
                "user_text": None,
                "user_audio": None,
                "system_text": None,
                "assistant_text": None
            }

            for message in messages:
                if isinstance(message, dict):
                    role = message.get("role")
                    content_list = list(message.get("content", []))

                    if role == "system":
                        system_prompt = next(
                            (item.get("text") for item in content_list if item.get("type") == "text"), None)
                        message_dict["system_text"] = system_prompt

                    elif role == "user":
                        audio_item = next((item for item in content_list if item.get("type") == "audio"), None)

                        audio_array = None
                        audio_path = None
                        sampling_rate = None

                        if audio_item:
                            if audio_item.get("array") is not None:
                                audio_array = np.array(audio_item.get("array"), dtype=np.float32)
                                sampling_rate = audio_item.get("sampling_rate")
                                audio_path = audio_item.get("path")

                            elif audio_item.get("audio") is not None:
                                audio_data = audio_item["audio"]
                                audio_array = np.array(audio_data.get("array"), dtype=np.float32)
                                sampling_rate = audio_data.get("sampling_rate")
                                audio_path = audio_data.get("path")

                            if audio_path:
                                audio_path = os.path.basename(audio_path)

                        text_item = next((item for item in content_list if item.get("type") == "text"), None)
                        message_dict["user_text"] = text_item.get("text") if text_item else None
                        message_dict["user_audio"] = {
                            "array": audio_array,
                            "path": audio_path,
                            "sampling_rate": sampling_rate
                        }

                    elif role == "assistant":
                        output_text = next((item.get("text") for item in content_list if item.get("type") == "text"), None)
                        message_dict["assistant_text"] = output_text

            # Append data to respective columns
            for key in output_data:
                output_data[key].append(message_dict[key])

    return output_data

def _read_parquet_using_dask(parquet_dirs, split='test'):
    """
    Read 'split' Parquet files from multiple directories using Dask, convert to Hugging Face Datasets format.

    Args:
        parquet_dirs (List[str]): List of dataset directories (each with train/test subfolders).
        split (str): Split to read, e.g., 'train' or 'test'.

    Returns:
        Dataset: Hugging Face Dataset containing all examples.
    """
    if isinstance(parquet_dirs, str):
        parquet_dirs = [parquet_dirs]  # Ensure parquet_dirs is always a list

    all_parquet_files = []

    for dir_path in parquet_dirs:
        split_path = os.path.join(dir_path, split)  # Join dir path with split (e.g., 'test')
        if os.path.exists(split_path):
            all_parquet_files.append(os.path.join(split_path, "*.parquet"))  # Add the path to parquet files
        else:
            print(f"❌ No [{split}] folder found in {dir_path}")

    if not all_parquet_files:
        raise FileNotFoundError(f"No parquet files found for split '{split}' in provided directories.")

    # Read all parquet files using Dask and convert them
    with ProgressBar():
        df = dd.read_parquet(all_parquet_files, engine="pyarrow", columns=["messages"])
        df_computed = df.compute()  # Convert Dask DataFrame to Pandas DataFrame
        converted = convert_parquet_messages(df_computed)

    # Convert to Hugging Face datasets
    ds = Dataset.from_dict(converted)
    return ds

def _encode_chunk(chunk: np.ndarray, mel_size: int):
    # pad/trim to exactly 30s
    raw = whisper.pad_or_trim(chunk)
    # mel: (n_mels, time) → permute to (time, n_mels)
    mel = whisper.log_mel_spectrogram(raw, n_mels=mel_size).permute(1, 0)
    return mel

def _make_pseudo_tokens(total_frames: int, tokenizer, ignore_index: int):
    # total_frames = sum of all chunk‑frames after downsampling
    boa = tokenizer.convert_tokens_to_ids("<|start_of_audio|>")
    eoa = tokenizer.convert_tokens_to_ids("<|end_of_audio|>")
    # we downsample mel time by factor ~10 → tokens_per_sec = 30s_chunk → 300 tokens
    placeholder = torch.full((total_frames,), ignore_index)
    return torch.cat([torch.tensor([boa]), placeholder, torch.tensor([eoa])])

def process_audio_long(
    audio_array: np.ndarray,
    tokenizer: Any,
    mel_size: int,
    ignore_index: int = -100,
    sr: int = whisper.audio.SAMPLE_RATE,
    chunk_duration_s: int = 30
) -> tuple:
    """
    Break `audio_array` into non-overlapping 30s chunks, encode each with Whisper,
    then concatenate both mel-features and pseudo-tokens.
    Returns:
        audio_pseudo_tokens:   (T_tokens,) 1D tensor
        token_length:          int
        full_mel:              (T_frames, mel_size) tensor
        total_frames:          int
    """
    # ensure numpy array
    if isinstance(audio_array, list):
        audio_array = np.array(audio_array, dtype=np.float32)

    chunk_size = chunk_duration_s * sr
    mel_chunks = []
    # 30s at 16 kHz → 480 k samples → mel frames ~300
    # each chunk’s mel_frames = (n_mels, time) permuted → time dimension
    for start in range(0, len(audio_array), chunk_size):
        chunk = audio_array[start : start + chunk_size]
        mel = _encode_chunk(chunk, mel_size)
        mel_chunks.append(mel)

    # concatenate along time axis
    full_mel = torch.cat(mel_chunks, dim=0)  # shape: (total_frames, mel_size)
    total_frames = full_mel.shape[0]

    # Heuristic to convert mel-frames → token count
    # Your original: computed_audio_length = ((audio_mel.shape[0] + 1)//2)//5
    total_tokens = ((total_frames + 1) // 2) // 5

    # build pseudo tokens of that length
    audio_pseudo_tokens = _make_pseudo_tokens(total_tokens, tokenizer, ignore_index)

    return audio_pseudo_tokens, audio_pseudo_tokens.size(0), full_mel, total_frames

def process_audio(audio_array: np.ndarray, tokenizer: Any, mel_size: int, 
                  fix_length_audio: int, ignore_index: int = -100) -> tuple:
    """Load and process audio into pseudo tokens and mel spectrogram features."""
    if isinstance(audio_array, list):
        audio_array = np.array(audio_array, dtype=np.float32)
    audio_raw = whisper.pad_or_trim(audio_array)
    audio_mel = whisper.log_mel_spectrogram(audio_raw, n_mels=mel_size).permute(1, 0)

    # Compute heuristic audio length
    computed_audio_length = ((audio_mel.shape[0] + 1) // 2) // 5
    audio_length = fix_length_audio if fix_length_audio > 0 else computed_audio_length

    # Create pseudo audio tokens
    boa_token_id = tokenizer.convert_tokens_to_ids("<|start_of_audio|>")
    eoa_token_id = tokenizer.convert_tokens_to_ids("<|end_of_audio|>")
    audio_placeholder = torch.full((audio_length,), ignore_index)

    audio_pseudo_tokens = torch.cat([
        torch.tensor([boa_token_id]),
        audio_placeholder,
        torch.tensor([eoa_token_id]),
    ])
    
    return audio_pseudo_tokens, audio_pseudo_tokens.size(0), audio_mel, audio_length

class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, dirs, tokenizer, model_config, train_config, split="test", **kwargs):
        self.ds = _read_parquet_using_dask(dirs, split)
        self.tokenizer = tokenizer
        self.mel_size = model_config.mel_size
        self.fix_length_audio = model_config.fix_length_audio
        self.IGNORE_INDEX = -100
        self.inference_mode = kwargs.get("inference_mode", False)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        system = row["system_text"] or dedent(
            "You are a helpful speech assistant who understands user input and aids with various tasks"
        )
        user_text = row["user_text"] or ""
        audio_dict = row["user_audio"] or {}
        arr = audio_dict.get("array")
        if arr is None:
            raise ValueError(f"No audio at index {idx}")
        sr = audio_dict.get("sampling_rate", whisper.audio.SAMPLE_RATE)
        if sr != whisper.audio.SAMPLE_RATE:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=whisper.audio.SAMPLE_RATE)
            
        # tokens, at_len, mel, a_len = process_audio(
        #         arr, self.tokenizer, self.mel_size, self.fix_length_audio, self.IGNORE_INDEX
        #     )
        
        tokens, at_len, mel, a_len = process_audio_long(
                arr, self.tokenizer, self.mel_size, self.IGNORE_INDEX
            )

        assistant = row["assistant_text"] or ""
        # format chat
        prompt = [
            {"role":"system","content":[{"type":"text","text":system}]},
            {"role":"user","content":[{"type":"text","text":user_text}]},
        ]
        full = prompt + [{"role":"assistant","content":[{"type":"text","text":assistant}]}]

        # encode
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        ).squeeze(0)
        p_len = prompt_ids.size(0)

        if self.inference_mode:
            inp = torch.cat([tokens, prompt_ids])
            return {
                "input_ids": inp,
                "attention_mask": inp != self.IGNORE_INDEX,
                "audio": mel,
                "audio_length": a_len,
                "audio_token_length": at_len,
                "prompt_length": p_len,
                "target": assistant,
            }

        example_ids = self.tokenizer.apply_chat_template(
            full, add_generation_prompt=False, tokenize=True, return_tensors="pt"
        ).squeeze(0)
        inp = torch.cat([tokens, example_ids])
        labels = inp.clone()
        labels[: at_len + p_len] = self.IGNORE_INDEX

        return {
            "input_ids": inp,
            "labels": labels,
            "attention_mask": inp != self.IGNORE_INDEX,
            "audio": mel,
            "audio_length": a_len,
            "audio_token_length": at_len,
            "prompt_length": p_len,
        }
    
    def pad(self, sequence: torch.Tensor, max_length: int, padding_idx: int = 0) -> torch.Tensor:
        """
        Pads a tensor sequence to the specified max_length.

        Args:
            sequence (torch.Tensor): Input sequence tensor.
            max_length (int): Desired sequence length.
            padding_idx (int): Padding value.

        Returns:
            torch.Tensor: Padded (or truncated) tensor.
        """
        current_length = sequence.size(0)
        if current_length < max_length:
            pad_shape = (max_length - current_length,) + sequence.shape[1:]
            padding_tensor = torch.full(pad_shape, padding_idx, dtype=sequence.dtype, device=sequence.device)
            sequence = torch.cat((sequence, padding_tensor), dim=0)
        else:
            sequence = sequence[:max_length]
        return sequence

    @classmethod
    def padding(cls, sequence, padding_length: int, padding_idx: int = 0, padding_side: str = "right"):
        """
        Pads (or truncates) a sequence to a given length from either the left or right.
        Supports list, tuple, torch.Tensor, or np.ndarray and returns a torch.Tensor.

        Args:
            sequence (list/tuple/torch.Tensor/np.ndarray): Input sequence.
            padding_length (int): Number of elements to pad (or remove if negative).
            padding_idx (int): Padding value.
            padding_side (str): 'left' or 'right'.

        Returns:
            torch.Tensor: Padded (or truncated) sequence.
        """
        if isinstance(sequence, (list, tuple)):
            sequence = list(sequence)
            if padding_length >= 0:
                if padding_side == "left":
                    sequence = [padding_idx] * padding_length + sequence
                else:
                    sequence = sequence + [padding_idx] * padding_length
            else:
                sequence = sequence[:padding_length]
            return torch.tensor(sequence, dtype=torch.int64)
        elif isinstance(sequence, torch.Tensor):
            if padding_length >= 0:
                padding_shape = (padding_length,) + tuple(sequence.shape[1:])
                padding_tensor = torch.full(padding_shape, padding_idx, dtype=sequence.dtype, device=sequence.device)
                if padding_side == "left":
                    sequence = torch.cat((padding_tensor, sequence), dim=0)
                else:
                    sequence = torch.cat((sequence, padding_tensor), dim=0)
            else:
                sequence = sequence[:padding_length]
            return sequence
        elif isinstance(sequence, np.ndarray):
            if padding_length >= 0:
                padding_shape = (padding_length,) + sequence.shape[1:]
                padding_array = np.full(padding_shape, padding_idx, dtype=sequence.dtype)
                if padding_side == "left":
                    sequence = np.concatenate((padding_array, sequence), axis=0)
                else:
                    sequence = np.concatenate((sequence, padding_array), axis=0)
            else:
                sequence = sequence[:padding_length]
            return torch.tensor(sequence, dtype=torch.int64)
        else:
            raise ValueError(f"Unsupported sequence type: {type(sequence)}")

    def data_collator(self, samples: list) -> dict:
        """
        Collates a list of samples into batch tensors for model input.

        Args:
            samples (list): List of samples (each a dict from __getitem__).

        Returns:
            dict: Batch dictionary containing padded input_ids, attention_mask, labels, audio features, etc.
        """
        assert samples, "Samples cannot be empty."

        # Compute the total length for the prompt part (audio tokens + prompt tokens)
        prompt_lengths = [
            sample["audio_token_length"] + sample["prompt_length"] for sample in samples
        ]
        answer_lengths = [
            sample["input_ids"].size(0) - (sample["audio_token_length"] + sample["prompt_length"])
            for sample in samples
        ]

        max_prompt_length = max(prompt_lengths)
        max_answer_length = max(answer_lengths)

        padded_input_ids = []
        padded_attention_masks = []
        for idx, sample in enumerate(samples):
            # Pad the sequence: first pad on the left (for the prompt part) then on the right (for the answer part)
            padded_ids = self.padding(
                self.padding(
                    sample["input_ids"], 
                    max_prompt_length - prompt_lengths[idx], 
                    self.tokenizer.pad_token_id, 
                    "left"
                ),
                max_answer_length - answer_lengths[idx], 
                self.tokenizer.pad_token_id, 
                "right"
            )
            padded_input_ids.append(padded_ids)

            attn_mask = self.padding(
                self.padding(
                    sample["attention_mask"].tolist(), 
                    max_prompt_length - prompt_lengths[idx], 
                    0, 
                    "left"
                ),
                max_answer_length - answer_lengths[idx], 
                0, 
                "right"
            )
            padded_attention_masks.append(attn_mask)

        input_ids = torch.stack(padded_input_ids).to(torch.int64)
        attention_mask = torch.stack(padded_attention_masks)

        # Pad audio mel spectrograms.
        audio_max_length = max(sample["audio"].shape[0] for sample in samples)
        padded_audios = [self.pad(sample["audio"], audio_max_length, 0) for sample in samples]
        audio_batch = torch.stack(padded_audios)

        # Create a post-mask for audio (downsampled mask based on a heuristic).
        audio_mel_post_masks = torch.zeros(len(samples), (audio_max_length + 1) // 2, dtype=torch.bool)
        for idx, sample in enumerate(samples):
            length = (sample["audio"].shape[0] + 1) // 2
            audio_mel_post_masks[idx, :length] = 1

        # Create a modality mask indicating positions of audio tokens.
        modality_masks = torch.zeros_like(attention_mask)
        for idx, sample in enumerate(samples):
            # The left padding added for each sample is:
            left_pad = max_prompt_length - (sample["audio_token_length"] + sample["prompt_length"])
            modality_masks[idx, left_pad:left_pad + sample["audio_token_length"]] = 1

        if self.inference_mode:
            audio_ids = [sample["audio_id"] for sample in samples]
            targets = [sample["target"] for sample in samples]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "audio_id": audio_ids,
                "audio": audio_batch,
                "audio_mel_post_mask": audio_mel_post_masks,
                "modality_mask": modality_masks,
                "target": targets,
            }

        # For training mode, pad the labels.
        padded_labels = []
        for idx, sample in enumerate(samples):
            padded_lbl = self.padding(
                self.padding(
                    sample["labels"], 
                    max_prompt_length - prompt_lengths[idx], 
                    self.IGNORE_INDEX, 
                    "left"
                ),
                max_answer_length - answer_lengths[idx], 
                self.IGNORE_INDEX, 
                "right"
            )
            padded_labels.append(padded_lbl)
        labels = torch.stack(padded_labels).to(torch.int64)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "audio": audio_batch,
            "audio_mel_post_mask": audio_mel_post_masks,
            "modality_mask": modality_masks,
        }
def get_dataset(dataset_dir: str, tokenizer: Any, model_config, train_config, split: str = "test", **kwargs) -> SpeechDataset:
    return SpeechDataset(dataset_dir, tokenizer, model_config, train_config, split, **kwargs)

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from AudioAdapterTraining.configs import *
    from AudioAdapterTraining.models.lucas_setup import set_tokenizer
    
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
    
