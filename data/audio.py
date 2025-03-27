# Misc
import argparse
import sys
import os
import random
import shlex
import functools
from typing import Optional, List, Callable, Union
from numpy.typing import NDArray
from pathlib import Path
from packaging import version

# Load audios
import librosa
import numpy as np
import soxbindings as sox
import torch
import torchaudio

# Audio augmentation
import audiomentations
import torch
import torch.nn.functional as F
import torchaudio
from audiomentations.core.transforms_interface import (
    BaseWaveformTransform,
)
from audiomentations.core.utils import (
    calculate_desired_noise_rms,
    calculate_rms,
    convert_decibels_to_amplitude_ratio,
)
from audiomentations.core.audio_loading_utils import load_sound_file


AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".opus"]

def load_audio(path, start = None, end = None, sampling_rate = 16_000, mono = True, return_format = 'array'):
    """
    Load an audio file and return the data.

    Parameters
    ----------
    path: str
        path to the audio file
    start: float
        start time in seconds. If None, the file will be loaded from the beginning.
    end: float
        end time in seconds. If None the file will be loaded until the end.
    sampling_rate: int
        destination sampling rate in Hz
    mono: bool
        if True, convert to mono
    return_format: str (default: 'array')
        'array': numpy.array
        'torch': torch.Tensor
        'bytes': bytes

    verbose: bool
        if True, print the steps
    """
    assert return_format in ['array', 'torch', 'bytes']
    if not os.path.isfile(path):
        # Because soxbindings does not indicate the filename if the file does not exist
        raise RuntimeError(f"File not found: {path}")
    # Test if we have read permission on the file
    elif not os.access(path, os.R_OK):
        # os.system("chmod a+r %s" % path)
        raise RuntimeError(f"Missing reading permission for: {path}")

    must_cut = start or end

    if return_format == 'torch' and not must_cut:
        if must_cut: # This path is super slow and has been disabled
            start = float(start if start else 0)
            sr = torchaudio.info(path).sampling_rate
            offset = int(start * sr)
            num_frames = -1
            if end:
                end = float(end)
                num_frames = int((end - start) * sr)
            audio, sr = torchaudio.load(path, frame_offset=offset, num_frames=num_frames)
        else:
            audio, sr = torchaudio.load(path)

    else:

        with suppress_stderr():
            # stderr could print these harmless warnings:
            # 1/ Could occur with sox.read
            #   mp3: MAD lost sync
            #   mp3: recoverable MAD error
            # 2/ Could occur with sox.get_info
            #   wav: wave header missing extended part of fmt chunk
            if must_cut: # is not None:
                start = float(start if start else 0)
                sr = sox.get_info(path)[0].rate
                offset = int(start * sr)
                nframes = 0
                if end: # is not None:
                    end = float(end)
                    nframes = int((end - start) * sr)
                audio, sr = sox.read(path, offset = offset, nframes = nframes)
            else:
                audio, sr = sox.read(path)

        audio = np.float32(audio)

    audio = conform_audio(audio, sr, sampling_rate=sampling_rate, mono=mono, return_format=return_format)

    if sampling_rate is None:
        return (audio, sr)
    return audio


class suppress_stderr:
    """
    A context manager for doing a "deep suppression" of stdout and stderr in Python,
    i.e. will suppress all print, even if the print originates in a compiled C/Fortran sub-function.
    """
    def __enter__(self):
        self.errnull_file = open(os.devnull, 'w')
        self.old_stderr_fileno_undup = sys.stderr.fileno()
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())
        self.old_stderr = sys.stderr
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stderr = self.old_stderr
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)
        os.close(self.old_stderr_fileno)
        self.errnull_file.close()


def conform_audio(audio, sr, sampling_rate=16_000, mono=True, return_format='array'):
    """
    Conform the audio to the desired format (mono channel, fixed frequency -- 16kHz)
    """
    if mono:
        if len(audio.shape) == 1:
            pass
        elif len(audio.shape) > 2:
            raise RuntimeError("Audio with more than 2 dimensions not supported")
        elif min(audio.shape) == 1:
            audio = audio.reshape(audio.shape[0] * audio.shape[1])
        else:
            if isinstance(audio, torch.Tensor):
                audio = audio.numpy()
            else:
                audio = audio.transpose()
            audio = librosa.to_mono(audio)
    if sampling_rate is not None and sr != sampling_rate:
        if not isinstance(audio, torch.Tensor):
            audio = torch.Tensor(audio)

        # # We don't use librosa here because there is a problem with multi-threading
        # audio = librosa.resample(audio, orig_sr = sr, target_sr = sampling_rate)

        audio = torchaudio.transforms.Resample(sr, sampling_rate)(torch.Tensor(audio))

    if return_format == "torch" and not isinstance(audio, torch.Tensor):
        audio = torch.Tensor(audio)
    elif return_format != "torch":
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        elif isinstance(audio, list):
            audio = np.array(audio, dtype=np.float32)
        if return_format == "bytes":
            audio = array_to_bytes(audio)

    return audio


def array_to_bytes(audio):
    return (audio * 32768).astype(np.int16).tobytes()


def save_audio(path, audio, sampling_rate=16_000):
    """
    Save an audio signal into a wav file.
    """
    if isinstance(audio, torch.Tensor):
        audio = audio.numpy()
        audio = audio.transpose()
    elif isinstance(audio, list):
        audio = np.array(audio, dtype=np.float32)
    sox.write(path, audio, sampling_rate)


class Reverberation(BaseWaveformTransform):
    def __init__(
        self,
        path_dir: str,
        rir_list_files: list[str],
        p: Optional[float] = 1,
        rir_scale_factor = (0.5, 1.0),
        gain_scaling_factor = None,
    ):
        """
        :param path_dir: directory including the rir directory
        :param rir_lists: list of rir files
        :param sampling_rate: target sample rate
        :param p: The probability of applying this transform
        :param rir_scale_factor: It compresses or dilates the given impulse response.
                                 If 0 < scale_factor < 1, the impulse response is compressed
                                 (less reverb), while if scale_factor > 1 it is dilated (more reverb).
        """
        super().__init__(p)
        self.path = path_dir
        self.rir_scale_factor = rir_scale_factor
        if isinstance(gain_scaling_factor, float):
            gain_scaling_factor = (gain_scaling_factor, gain_scaling_factor)
        if not gain_scaling_factor:
            self.gain_scaler = None
        elif isinstance(gain_scaling_factor, (tuple, list)):
            assert len(gain_scaling_factor) == 2, f"{gain_scaling_factor=} should be a tuple of two floats"
            min_db, max_db = gain_scaling_factor
            assert min_db < max_db, f"{gain_scaling_factor=} should be a tuple of two floats with min < max"
            self.gain_scaler = audiomentations.Gain(min_gain_in_db=min_db, max_gain_in_db=max_db, p=1.0)
        else:
            raise ValueError(f"{gain_scaling_factor=} should be a float or a tuple of two floats")
        self.wavs = []
        for rir_file in rir_list_files:
            if not os.path.isfile(os.path.join(self.path, rir_file)):
                print(f"WARNING: {rir_file} not found")
            else:
                self.wavs += self._parse_rir_list(rir_file)
        assert self.wavs, f"Could not find any RIR files in {self.path=}"

    def apply(self, waveform, sampling_rate):
        # Cast waveform object type to torch tensor
        waveform = torch.as_tensor(waveform)

        if self.gain_scaler:
            waveform = self.gain_scaler(waveform, sampling_rate)

        samp_index = self.parameters["samp_index"]
        rir_samples = self._load_wav(self.wavs[samp_index], sampling_rate)
        self.parameters.pop("samp_index")
        self.parameters["reverberation_file_path"] = self.wavs[samp_index]

        # Compress or dilate RIR
        rir_scale_factor = self.rir_scale_factor
        if rir_scale_factor == "random":
            rir_scale_factor = random.uniform(0, 1)
        elif isinstance(rir_scale_factor, (tuple, list)):
            assert len(rir_scale_factor) == 2
            mini, maxi = rir_scale_factor
            rir_scale_factor = random.uniform(mini, maxi)
        if rir_scale_factor != 1:
            rir_samples = F.interpolate(
                rir_samples.unsqueeze(0),
                scale_factor=rir_scale_factor,
                mode="linear",
                align_corners=False,
            )
            rir_samples = rir_samples.transpose(1, -1)

        return self._reverberate(waveform, rir_samples, rescale_amp="avg")

    def randomize_parameters(self, waveform, sampling_rate):
        super().randomize_parameters(waveform, sampling_rate)
        if self.parameters["should_apply"]:
            self.parameters["samp_index"] = random.randint(0, len(self.wavs) - 1)

    def _parse_rir_list(self, rir_file):
        rir_parser = argparse.ArgumentParser()
        rir_parser.add_argument(
            "--rir-id",
            type=str,
            required=True,
            help="This id is unique for each RIR and the music may associate with a particular RIR by refering to this id",
        )
        rir_parser.add_argument("--room-id", type=str, required=True, help="This is the room that where the RIR is generated")
        rir_parser.add_argument("--receiver-position-id", type=str, default=None, help="receiver position id")
        rir_parser.add_argument("--source-position-id", type=str, default=None, help="source position id")
        rir_parser.add_argument(
            "--rt60",
            type=float,
            default=None,
            help="RT60 is the time required for reflections of a direct sound to decay 60 dB.",
        )
        rir_parser.add_argument("--drr", type=float, default=None, help="Direct-to-reverberant-ratio of the impulse response.")
        rir_parser.add_argument("--cte", type=float, default=None, help="Early-to-late index of the impulse response.")
        rir_parser.add_argument("--probability", type=float, default=None, help="probability of the impulse response.")
        rir_parser.add_argument(
            "rir_rspecifier",
            type=str,
            help="""rir rspecifier, it can be either a filename or a piped command.
                                E.g. data/impulses/Room001-00001.wav or "sox data/impulses/Room001-00001.wav -t wav - |" """,
        )

        rir_list = []
        current_rir_list = [rir_parser.parse_args(shlex.split(x.strip())) for x in open(os.path.join(self.path, rir_file))]
        for rir in current_rir_list:
            # check if the rspecifier is a pipe or not
            filepath = self.path + "/" + rir.rir_rspecifier
            if len(rir.rir_rspecifier.split()) == 1 and os.path.exists(filepath):
                rir_list.append(filepath)

        return rir_list

    def _load_wav(self, filepath, sampling_rate):
        speech_array, sr = torchaudio.load(filepath)
        if sr != sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, sampling_rate)
            speech_array = resampler(speech_array).squeeze()
        return speech_array

    def _reverberate(self, waveforms, rir_waveform, rescale_amp="avg"):
        """
        General function to contaminate a given signal with reverberation given a
        Room Impulse Response (RIR).
        It performs convolution between RIR and signal, but without changing
        the original amplitude of the signal.

        Arguments
        ---------
        waveforms : tensor
            The waveforms to normalize.
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        rir_waveform : tensor
            RIR tensor, shape should be [time, channels].
        rescale_amp : str
            Whether reverberated signal is rescaled (None) and with respect either
            to original signal "peak" amplitude or "avg" average amplitude.
            Choose between [None, "avg", "peak"].

        Returns
        -------
        waveforms: numpy.array
            Reverberated signal.

        """

        orig_shape = waveforms.shape

        if len(waveforms.shape) > 3 or len(rir_waveform.shape) > 3:
            raise NotImplementedError

        # if inputs are mono tensors we reshape to 1, waveform
        if len(waveforms.shape) == 1:
            waveforms = waveforms.unsqueeze(0).unsqueeze(-1)
        elif len(waveforms.shape) == 2:
            waveforms = waveforms.unsqueeze(-1)

        if len(rir_waveform.shape) == 1:  # convolve1d expects a 3d tensor !
            rir_waveform = rir_waveform.unsqueeze(0).unsqueeze(-1)
        elif len(rir_waveform.shape) == 2:
            rir_waveform = rir_waveform.unsqueeze(-1)

        # Compute the average amplitude of the clean
        orig_amplitude = self._compute_amplitude(waveforms, waveforms.size(1), rescale_amp)

        # Compute index of the direct signal, so we can preserve alignment
        value_max, direct_index = rir_waveform.abs().max(axis=1, keepdim=True)

        # Making sure the max is always positive (if not, flip)
        # mask = torch.logical_and(rir_waveform == value_max,  rir_waveform < 0)
        # rir_waveform[mask] = -rir_waveform[mask]

        # Use FFT to compute convolution, because of long reverberation filter
        waveforms = self._convolve1d(
            waveform=waveforms,
            kernel=rir_waveform,
            use_fft=True,
            rotation_index=direct_index,
        )

        # Rescale to the peak amplitude of the clean waveform
        waveforms = self._rescale(waveforms, waveforms.size(1), orig_amplitude, rescale_amp)

        if len(orig_shape) == 1:
            waveforms = waveforms.squeeze(0).squeeze(-1)
        if len(orig_shape) == 2:
            waveforms = waveforms.squeeze(-1)

        waveforms = conform_audio(waveforms, 16_000)

        return waveforms

    def _compute_amplitude(self, waveforms, lengths=None, amp_type="avg", scale="linear"):
        """Compute amplitude of a batch of waveforms.

        Arguments
        ---------
        waveform : tensor
            The waveforms used for computing amplitude.
            Shape should be `[time]` or `[batch, time]` or
            `[batch, time, channels]`.
        lengths : tensor
            The lengths of the waveforms excluding the padding.
            Shape should be a single dimension, `[batch]`.
        amp_type : str
            Whether to compute "avg" average or "peak" amplitude.
            Choose between ["avg", "peak"].
        scale : str
            Whether to compute amplitude in "dB" or "linear" scale.
            Choose between ["linear", "dB"].

        Returns
        -------
        The average amplitude of the waveforms.

        Example
        -------
        >>> signal = torch.sin(torch.arange(16000.0)).unsqueeze(0)
        >>> compute_amplitude(signal, signal.size(1))
        tensor([[0.6366]])
        """
        if len(waveforms.shape) == 1:
            waveforms = waveforms.unsqueeze(0)

        assert amp_type in ["avg", "peak"]
        assert scale in ["linear", "dB"]

        if amp_type == "avg":
            if lengths is None:
                out = torch.mean(torch.abs(waveforms), dim=1, keepdim=True)
            else:
                wav_sum = torch.sum(input=torch.abs(waveforms), dim=1, keepdim=True)
                out = wav_sum / lengths
        elif amp_type == "peak":
            out = torch.max(torch.abs(waveforms), dim=1, keepdim=True)[0]
        else:
            raise NotImplementedError

        if scale == "linear":
            return out
        elif scale == "dB":
            return torch.clamp(20 * torch.log10(out), min=-80)  # clamp zeros
        else:
            raise NotImplementedError

    def _convolve1d(
        self,
        waveform,
        kernel,
        padding=0,
        pad_type="constant",
        stride=1,
        groups=1,
        use_fft=False,
        rotation_index=0,
    ):
        """Use torch.nn.functional to perform 1d padding and conv.

        Arguments
        ---------
        waveform : tensor
            The tensor to perform operations on.
        kernel : tensor
            The filter to apply during convolution.
        padding : int or tuple
            The padding (pad_left, pad_right) to apply.
            If an integer is passed instead, this is passed
            to the conv1d function and pad_type is ignored.
        pad_type : str
            The type of padding to use. Passed directly to
            `torch.nn.functional.pad`, see PyTorch documentation
            for available options.
        stride : int
            The number of units to move each time convolution is applied.
            Passed to conv1d. Has no effect if `use_fft` is True.
        groups : int
            This option is passed to `conv1d` to split the input into groups for
            convolution. Input channels should be divisible by the number of groups.
        use_fft : bool
            When `use_fft` is passed `True`, then compute the convolution in the
            spectral domain using complex multiply. This is more efficient on CPU
            when the size of the kernel is large (e.g. reverberation). WARNING:
            Without padding, circular convolution occurs. This makes little
            difference in the case of reverberation, but may make more difference
            with different kernels.
        rotation_index : int
            This option only applies if `use_fft` is true. If so, the kernel is
            rolled by this amount before convolution to shift the output location.

        Returns
        -------
        The convolved waveform.

        Example
        -------
        >>> from speechbrain.dataio.dataio import read_audio
        >>> signal = read_audio('waveform/audio_samples/example1.wav')
        >>> signal = signal.unsqueeze(0).unsqueeze(2)
        >>> kernel = torch.rand(1, 10, 1)
        >>> signal = convolve1d(signal, kernel, padding=(9, 0))
        """
        if len(waveform.shape) != 3:
            raise ValueError("Convolve1D expects a 3-dimensional tensor")

        # Move time dimension last, which pad and fft and conv expect.
        waveform = waveform.transpose(2, 1)
        kernel = kernel.transpose(2, 1)

        # Padding can be a tuple (left_pad, right_pad) or an int
        if isinstance(padding, tuple):
            waveform = torch.nn.functional.pad(
                input=waveform,
                pad=padding,
                mode=pad_type,
            )

        # This approach uses FFT, which is more efficient if the kernel is large
        if use_fft:
            # Pad kernel to same length as signal, ensuring correct alignment
            zero_length = waveform.size(-1) - kernel.size(-1)

            # Handle case where signal is shorter
            if zero_length < 0:
                kernel = kernel[..., :zero_length]
                zero_length = 0

            # Perform rotation to ensure alignment
            zeros = torch.zeros(kernel.size(0), kernel.size(1), zero_length, device=kernel.device)
            after_index = kernel[..., rotation_index:]
            before_index = kernel[..., :rotation_index]
            kernel = torch.cat((after_index, zeros, before_index), dim=-1)

            # Multiply in frequency domain to convolve in time domain
            if version.parse(torch.__version__) > version.parse("1.6.0"):
                import torch.fft as fft

                result = fft.rfft(waveform) * fft.rfft(kernel)
                convolved = fft.irfft(result, n=waveform.size(-1))
            else:
                f_signal = torch.rfft(waveform, 1)
                f_kernel = torch.rfft(kernel, 1)
                sig_real, sig_imag = f_signal.unbind(-1)
                ker_real, ker_imag = f_kernel.unbind(-1)
                f_result = torch.stack(
                    [
                        sig_real * ker_real - sig_imag * ker_imag,
                        sig_real * ker_imag + sig_imag * ker_real,
                    ],
                    dim=-1,
                )
                convolved = torch.irfft(f_result, 1, signal_sizes=[waveform.size(-1)])

        # Use the implementation given by torch, which should be efficient on GPU
        else:
            convolved = torch.nn.functional.conv1d(
                input=waveform,
                weight=kernel,
                stride=stride,
                groups=groups,
                padding=padding if not isinstance(padding, tuple) else 0,
            )

        # Return time dimension to the second dimension.
        return convolved.transpose(2, 1)

    def _rescale(self, waveforms, lengths, target_lvl, amp_type="avg", scale="linear"):
        """This functions performs signal rescaling to a target level.

        Arguments
        ---------
        waveforms : tensor
            The waveforms to normalize.
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : tensor
            The lengths of the waveforms excluding the padding.
            Shape should be a single dimension, `[batch]`.
        target_lvl : float
            Target lvl in dB or linear scale.
        amp_type : str
            Whether one wants to rescale with respect to "avg" or "peak" amplitude.
            Choose between ["avg", "peak"].
        scale : str
            whether target_lvl belongs to linear or dB scale.
            Choose between ["linear", "dB"].

        Returns
        -------
        waveforms : tensor
            Rescaled waveforms.
        """

        assert amp_type in ["peak", "avg"]
        assert scale in ["linear", "dB"]

        batch_added = False
        if len(waveforms.shape) == 1:
            batch_added = True
            waveforms = waveforms.unsqueeze(0)

        waveforms = self._normalize(waveforms, lengths, amp_type)

        if scale == "linear":
            out = target_lvl * waveforms
        elif scale == "dB":
            out = self._dB_to_amplitude(target_lvl) * waveforms

        else:
            raise NotImplementedError("Invalid scale, choose between dB and linear")

        if batch_added:
            out = out.squeeze(0)

        return out

    def _normalize(self, waveforms, lengths=None, amp_type="avg", eps=1e-14):
        """This function normalizes a signal to unitary average or peak amplitude.

        Arguments
        ---------
        waveforms : tensor
            The waveforms to normalize.
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : tensor
            The lengths of the waveforms excluding the padding.
            Shape should be a single dimension, `[batch]`.
        amp_type : str
            Whether one wants to normalize with respect to "avg" or "peak"
            amplitude. Choose between ["avg", "peak"]. Note: for "avg" clipping
            is not prevented and can occur.
        eps : float
            A small number to add to the denominator to prevent NaN.

        Returns
        -------
        waveforms : tensor
            Normalized level waveform.
        """

        assert amp_type in ["avg", "peak"]

        batch_added = False
        if len(waveforms.shape) == 1:
            batch_added = True
            waveforms = waveforms.unsqueeze(0)

        den = self._compute_amplitude(waveforms, lengths, amp_type) + eps
        if batch_added:
            waveforms = waveforms.squeeze(0)
        return waveforms / den

    def _dB_to_amplitude(self, SNR):
        """Returns the amplitude ratio, converted from decibels.

        Arguments
        ---------
        SNR : float
            The ratio in decibels to convert.

        Example
        -------
        >>> round(dB_to_amplitude(SNR=10), 3)
        3.162
        >>> dB_to_amplitude(SNR=0)
        1.0
        """
        return 10 ** (SNR / 20)


global _augmenter8k, _augmenter16k
_augmenter8k = _augmenter16k = None

def reverberation_factory(
    path_parent: str= "/data-server/datasets/audio/noise",
    path_dir_16k: str= "simulated_rirs_16k",
    path_dir_8k: str= "simulated_rirs_8k",
    rir_lists: list= ["smallroom/rir_list", "mediumroom/rir_list", "largeroom/rir_list"],
    sampling_rate=16_000,
    verbose=True,
    **kwargs
    ):
    global _augmenter8k, _augmenter16k
    if sampling_rate <= 10_000:
        if _augmenter8k is None:
            if verbose: print("Loading augmenter for 8kHz...")
            rir_lists = [os.path.join(path_dir_8k, rir_list) for rir_list in rir_lists]
            _augmenter8k = Reverberation(path_parent, rir_lists, **kwargs)
        return _augmenter8k
    else:
        if _augmenter16k is None:
            if verbose: print("Loading augmenter for 16kHz...")
            rir_lists = [os.path.join(path_dir_16k, rir_list) for rir_list in rir_lists]
            _augmenter16k = Reverberation(path_parent, rir_lists, **kwargs)
        return _augmenter16k


class CombineAudios(BaseWaveformTransform):
    """Mix in another sound, e.g. a background music. Useful if your original sound is clean and
    you want to simulate an environment where background music is present.
    Can also be used for mixup, as in https://arxiv.org/pdf/1710.09412.pdf
    A folder of (background music) sounds to be mixed in must be specified. These sounds should
    ideally be at least as long as the input sounds to be transformed. Otherwise, the background
    sound will be repeated, which may sound unnatural.
    Note that the gain of the added music is relative to the amount of signal in the input if the parameter music_rms
    is set to "relative" (default option). This implies that if the input is completely silent, no music will be added.
    Here are some examples of datasets that can be downloaded and used as background music:
    * https://github.com/karolpiczak/ESC-50#download
    * https://github.com/microsoft/DNS-Challenge/
    """

    def __init__(
        self,
        min_snr_db: float = None,
        max_snr_db: float = None,
        music_rms: str = "relative",
        music_transform: Optional[
            Callable[[NDArray[np.float32], int], NDArray[np.float32]]
        ] = None,
        p: float = 1.0,
        lru_cache_size: int = 2,
    ):
        """
        :param min_snr_db: Minimum signal-to-music ratio in dB.
        :param max_snr_db: Maximum signal-to-music ratio in dB.
        :param music_transform: A callable waveform transform (or composition of transforms) that
            gets applied to the music before it gets mixed in. The callable is expected
            to input audio waveform (numpy array) and sample rate (int).
        :param p: The probability of applying this transform
        :param lru_cache_size: Maximum size of the LRU cache for storing music files in memory
        """
        super().__init__(p)

        if min_snr_db is not None:
            self.min_snr_db = min_snr_db
        else:
            self.min_snr_db = 0.7  # the default

        if max_snr_db is not None:
            self.max_snr_db = max_snr_db
        else:
            self.max_snr_db = 2.5  # the default

        assert self.min_snr_db <= self.max_snr_db

        self.music_rms = music_rms
        self._load_sound = functools.lru_cache(maxsize=lru_cache_size)(
            CombineAudios._load_sound
        )
        self.music_transform = music_transform

        self.music_clip = None

    @staticmethod
    def _load_sound(file_path, sample_rate):
        return load_sound_file(file_path, sample_rate)

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["snr_db"] = random.uniform(self.min_snr_db, self.max_snr_db)

    def apply(self, samples: NDArray[np.float32], sample_rate: int):
        assert self.music_clip is not None, f"Set self.music_clip before calling apply"
        music_clip = self.music_clip

        num_samples = len(samples)

        if self.music_transform:
            music_clip = self.music_transform(music_clip, sample_rate)

        music_rms = calculate_rms(music_clip)
        clean_rms = calculate_rms(samples)

        # Repeat the sound if it shorter than the input sound
        while len(music_clip) < num_samples:
            print("WARNING: music clip is repeated!")
            music_clip = np.concatenate((music_clip, music_clip))

        if len(music_clip) > num_samples:
            # Pad with silence
            at_start = random.random() < 0.5
            big_silence = len(music_clip) - num_samples
            small_silence = 0
            if big_silence > 32_000:
                small_silence = round(random.uniform(8_000, min(big_silence, 32_000)) )
            big_silence -= small_silence
            if at_start:
                samples = np.pad(samples, (small_silence, big_silence))
            else:
                samples = np.pad(samples, (big_silence, small_silence))

        desired_music_rms = calculate_desired_noise_rms(
            clean_rms, self.parameters["snr_db"]
        )

        # Adjust the music to match the desired music RMS
        music_clip = music_clip * (desired_music_rms / music_rms)

        # Return a mix of the input sound and the background music sound
        return samples + music_clip

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_load_sound"]
        return state


global _audio_combiner
_audio_combiner = None

def combine_audios_factory():
    global _audio_combiner
    if _audio_combiner is None:
        _audio_combiner = CombineAudios()
    return _audio_combiner


def cut_audio(waveform, sampling_rate, duration):

    waveform = conform_audio(waveform, sampling_rate, return_format='array')

    if isinstance(duration, (tuple, list)):
        # Random duration
        
        assert len(duration) == 2
        duration_range = duration
        duration = random.uniform(*duration_range)

    assert len(waveform.shape) == 1, f"{waveform.shape=}"
    max_len = waveform.shape[0]
    duration = min(duration, max_len / sampling_rate)
    max_start = max_len / sampling_rate - duration
    if max_start > 0:
        start = random.uniform(0, max_start)
    else:
        start = 0
    sample_start = round(start * sampling_rate)
    sample_end = min(round((start + duration) * sampling_rate), max_len)
    waveform = waveform[sample_start:sample_end]
    return waveform
    
