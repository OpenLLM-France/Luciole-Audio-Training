import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.audio import load_audio, get_audio_duration
import whisper
from pathlib import Path
from textwrap import dedent
import csv
import json
import ast

def resolve_audio_path(relative_path: str, base_paths) -> str:
    """
    Given a relative file path and a list (or single string) of base paths,
    return the first full path that exists. Otherwise, return None.
    """
    if isinstance(base_paths, str):
        base_paths = [base_paths]
    
    for base in base_paths:
        full_path = Path(base) / relative_path
        if full_path.exists():
            return str(full_path)
    return None

def load_data_as_dict(data_file):
    required_cols = ['path', 'sentence', 'translation', 'start_time', 'end_time', 'language', 'language_dist', 'tokens', 'nlu_tokens']
    
    def process_csv(file):
        is_csv = file.endswith(".csv")
        delimiter = "," if is_csv else "\t"
        
        with open(file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            data = []
            for row in reader:
                for col in required_cols:
                    row.setdefault(col, "" if col in ["translation", "language", "language_dist"] else "[]" if col in ["tokens", "nlu_tokens"] else 0.0)
                
                for col in ["tokens", "nlu_tokens"]:
                    try:
                        row[col] = ast.literal_eval(row[col]) if isinstance(row[col], str) else []
                    except (SyntaxError, ValueError):
                        row[col] = []
                
                row["start_time"] = float(row.get("start_time", 0.0))
                row["end_time"] = float(row.get("end_time", 0.0))
                
                data.append(row)
        return data
    
    def process_json(file):
        with open(file, 'r', encoding="utf-8") as f:
            return json.load(f)
    
    if isinstance(data_file, str):
        return process_json(data_file) if data_file.endswith(".json") else process_csv(data_file)
    elif isinstance(data_file, list):
        combined_data = []
        for f in data_file:
            combined_data.extend(process_json(f) if f.endswith(".json") else process_csv(f))
        return combined_data
    
    raise TypeError("data_file must be a string (file path) or a list of file paths")

def prepare_data(file, audio_paths):
    """
    Prepare a list of dictionaries by modifying the 'path' field to include full audio paths,
    and handling start/end times for audio segments.
    """
    data_list = load_data_as_dict(file)
    
    valid_data = []
    for item in data_list:
        resolved_path = resolve_audio_path(item["path"], audio_paths)
        if resolved_path:
            item["path"] = resolved_path
            item.setdefault("start_time", 0.0)
            item.setdefault("end_time", get_audio_duration(resolved_path) if Path(resolved_path).exists() else 0.0)
            valid_data.append(item)
        else:
            print(f"Warning: Audio file {item['path']} not found in given directories.")
    
    return valid_data


class SpeechDatasetTSV(Dataset):
    def __init__(self, file_data: str, audio_path: str, model_config, train_config, tokenizer, **kwargs):
        """
        Initializes the SpeechDatasetTSV.

        Args:
            file_data (str): Path to the dataset file containing data entries.
            audio_path (str): Base path to the audio files.
            model_config: Model configuration object.
            train_config: Training configuration object.
            tokenizer: Tokenizer instance.
            **kwargs: Additional keyword arguments (e.g., inference_mode flag).
        """
        super().__init__()
        self.data = file_data
        self.audio_path = audio_path
        self.tokenizer = tokenizer
        self.train_config = train_config
        self.model_config = model_config
        self.inference_mode = kwargs.get("inference_mode", False)  # Inference mode flag
        
        # Load model configurations
        self.IGNORE_INDEX = -100
        # self.prompt = model_config.prompt
        self.mel_size = model_config.mel_size
        self.max_duration = model_config.max_duration
        self.min_duration = model_config.min_duration
        self.prompt_template = model_config.prompt_template
        self.answer_template = model_config.answer_template
        self.fix_length_audio = model_config.fix_length_audio
        self.language = model_config.language
        self.normalization = model_config.normalization
        self.num_workers = model_config.num_workers
        self.text_length = model_config.text_length
        self.process_online = model_config.process_online

        # Load and store data entries from dataset
        self.data_list = prepare_data(self.data, self.audio_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Retrieves and processes a single data sample.

        Args:
            idx (int): Index of the sample.
            
        Returns:
            dict: Processed sample containing input_ids, attention_mask, audio features, etc.
        """
        data_dict = self.data_list[idx]

        # Extract relevant fields
        audio_id = data_dict.get("client_id") or data_dict.get("id", "")
        sentence = data_dict.get("sentence")
        instruction = data_dict.get("instruction")
        output = data_dict.get("output")
        speech_instruct = data_dict.get("speech_instruct")
        audio_path = data_dict.get("path")

        lang_src = data_dict.get("language", "")
        start_seg = float(data_dict.get("start_time", 0.0))
        end_seg = float(data_dict.get("end_time", 0.0))

        translation = data_dict.get("translation", "")
        lang_dist = data_dict.get("language_dist", "")
        nlu_tokens = data_dict.get("nlu_tokens", [])
        list_tokens = data_dict.get("tokens", [])

        if nlu_tokens and list_tokens and len(list_tokens) != len(nlu_tokens):
            raise ValueError(
                f"list_tokens and nlu_tokens must have the same length: "
                f"{len(list_tokens)} vs {len(nlu_tokens)}"
            )

        # Load and process audio
        audio_raw = load_audio(audio_path, start=start_seg, end=end_seg)
        audio_raw = whisper.pad_or_trim(audio_raw)
        audio_mel = whisper.log_mel_spectrogram(audio_raw, n_mels=self.mel_size).permute(1, 0)

        # Compute heuristic audio length
        computed_audio_length = ((audio_mel.shape[0] + 1) // 2) // 5
        audio_length = self.fix_length_audio if self.fix_length_audio > 0 else computed_audio_length

        # Create pseudo audio tokens
        boa_token_id = self.tokenizer.convert_tokens_to_ids("<|start_of_audio|>")
        eoa_token_id = self.tokenizer.convert_tokens_to_ids("<|end_of_audio|>")
        audio_placeholder = torch.full((audio_length,), -1, dtype=torch.int64)

        audio_pseudo_tokens = torch.cat([
            torch.tensor([boa_token_id], dtype=torch.int64),
            audio_placeholder,
            torch.tensor([eoa_token_id], dtype=torch.int64),
        ])
        audio_token_length = audio_pseudo_tokens.size(0)

        # Format system, user, and assistant messages
        sys_message = dedent(
            """You are a helpful speech assistant who understands user input and aids with various tasks"""
        ).strip()

        if sentence is not None:
            user_message = f"\nTranscribe the speech from the {lang_src} language into text"
            if translation:
                user_message += f", then translate it into the {lang_dist} target language"
            elif nlu_tokens:
                user_message += ", then analyze the transcribed text"

            assistant_message = f"Transcription of the Audio:\n{sentence}"
            if translation:
                assistant_message += f"\nTranslation into ({lang_dist}):\n{translation}"
            elif nlu_tokens and list_tokens:
                nlu_analysis = "\n".join(f"{word}: {token}" for word, token in zip(list_tokens, nlu_tokens))
                assistant_message += f"\nThe analysis of this transcription:\n{nlu_analysis}"
        elif speech_instruct is not None:
            user_message = f"\n{speech_instruct}"
            assistant_message = f"\n{output}"
            
        # Format the prompt
        prompt_template = self.prompt_template or  (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "{system_message}\n<|eot_id|>\n\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            "{user_message}\n<|eot_id|>\n\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
        formatted_prompt = prompt_template.format(
                                system_message=sys_message.strip(),
                                user_message=user_message.strip()
                            )

        # Tokenize the prompt
        prompt_ids = self.tokenizer.encode(formatted_prompt)
        prompt_length = len(prompt_ids)

        if self.inference_mode:
            prompt_tensor = torch.tensor(prompt_ids, dtype=torch.int64)
            input_ids = torch.cat((audio_pseudo_tokens, prompt_tensor))
            attention_mask = input_ids != -1  # Ensure valid masking

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "audio_id": audio_id,
                "audio": audio_mel,
                "audio_length": audio_length,
                "audio_token_length": audio_token_length,
                "prompt_length": prompt_length,
                "target": assistant_message,
            }

        # Training mode: tokenize full example with EOS token
        example = f"{formatted_prompt}\n{assistant_message}"
        example_ids = self.tokenizer.encode(example) + [self.tokenizer.eos_token_id]
        example_tensor = torch.tensor(example_ids, dtype=torch.int64)
        input_ids = torch.cat((audio_pseudo_tokens, example_tensor))

        # Create labels and mask out the audio and prompt tokens
        labels = input_ids.clone()
        labels[: audio_token_length + prompt_length] = self.IGNORE_INDEX

        attention_mask = input_ids != -1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "audio": audio_mel,
            "audio_length": audio_length,
            "audio_token_length": audio_token_length,
            "prompt_length": prompt_length,
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
        
def get_speech_dataset(file_data, audio_path, model_config, train_config, tokenizer):
    dataset = SpeechDatasetTSV(file_data, audio_path,  model_config, train_config, tokenizer)
    return dataset

if __name__ == '__main__':
    
    from torch.utils.data import DataLoader
    from configs import *
    from linastt.utils.env import *

    from models.model import model_factory
   
    model_config = ModelConfig()
    train_config = TrainConfig()
    
    torch.manual_seed(train_config.seed)
    # tokenizer = AutoTokenizer.from_pretrained(model_config.llm_name_hf)
    kwargs = {"metric": "acc"}
    # Initialize model and tokenizer
    model, tokenizer = model_factory(train_config, model_config, **kwargs)
    # tokenizer = AutoTokenizer.from_pretrained(model_config.llm_name_hf)
    device = auto_device()
    model.to(device)
    # Load and preprocess the dataset
    data_source = {
        "audio_path": "/home/hnaoura/hnaouara/Audio_corpus/Audio_instruct_GPT/test_audios",
        "dev": "/home/hnaoura/hnaouara/Audio_corpus/Audio_instruct_GPT/test_5.json",
    }



    dataset_val = get_speech_dataset(
        data_source["dev"], 
        data_source["audio_path"], 
        model_config, 
        train_config, 
        tokenizer
    )
    val_loader = DataLoader(
                dataset_val, 
                batch_size=1, 
                shuffle=False, 
                num_workers=1, 
                collate_fn=dataset_val.data_collator
            )

    
    sample = dataset_val[-1]

    input_ids = sample['input_ids'][sample['input_ids'] != -1]
    labels = sample['labels'][sample['labels'] != -100]
    print()
    print(f'input_ids : {tokenizer.decode(input_ids, skip_special_tokens=True)}')
    print(f'labels : {tokenizer.decode(labels, skip_special_tokens=True)}')
