import librosa
import numpy as np
import torch
from typing import Union

def load_audio(
    path: str,
    start: Union[None, float] = None,
    end: Union[None, float] = None,
    sample_rate: int = 16_000,
    mono: bool = True,
    return_format: str = 'array',
    verbose: bool = False
) -> Union[np.ndarray, torch.Tensor]:
    """
    Loads an audio file and returns it as either a numpy array or torch tensor.

    Args:
        path (str): Path to the audio file.
        start (float, optional): Start time (in seconds) from which to begin loading the audio.
        end (float, optional): End time (in seconds) at which to stop loading the audio.
        sample_rate (int, optional): Desired sample rate for the loaded audio.
        mono (bool, optional): Whether to convert the audio to mono.
        return_format (str, optional): Format for return, either 'array' (numpy array) or 'tensor' (PyTorch tensor).
        verbose (bool, optional): Whether to print additional information during loading.

    Returns:
        Union[np.ndarray, torch.Tensor]: Loaded audio in the specified format (array or tensor).
    """
    if verbose:
        print(f"Loading audio from {path}...")

    # Load the audio with librosa, specifying the start, end, and sample rate
    audio, sr = librosa.load(path, sr=sample_rate, mono=mono, offset=start, duration=(end - start if end else None))

    if verbose:
        print(f"Audio loaded with sample rate {sr} and shape {audio.shape}")

    # If the return format is 'tensor', convert to a torch tensor
    if return_format == 'tensor':
        audio = torch.tensor(audio)

    return audio

def get_audio_duration(path: str, verbose: bool = False) -> float:
    """
    Returns the duration of an audio file in seconds.

    Args:
        path (str): Path to the audio file.
        verbose (bool, optional): Whether to print additional information during loading.

    Returns:
        float: Duration of the audio file in seconds.
    """
    if verbose:
        print(f"Loading audio from {path} to get duration...")

    # Load the audio with librosa to get the duration
    y, sr = librosa.load(path, sr=None)  # sr=None to keep the original sample rate

    # Calculate the duration in seconds
    duration = librosa.get_duration(y=y, sr=sr)

    if verbose:
        print(f"Audio loaded with sample rate {sr}. Duration: {duration:.2f} seconds")

    return duration