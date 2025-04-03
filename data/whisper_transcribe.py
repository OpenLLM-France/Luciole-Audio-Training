import os

# Relative import
import sys

from faster_whisper import WhisperModel

sys.path.append(os.path.dirname(__file__))
from audio import (
    load_audio,
    save_audio,
)

from utils import array_signature


def write_srt(out, segments):
    for i, segment in enumerate(segments):
        id = i + 1
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        start_min = int(start / 60)
        start_sec = start % 60
        end_min = int(end / 60)
        end_sec = end % 60
        out.write(f"{id}\n")
        out.write(f"{start_min:02}:{start_sec:05.2f} --> {end_min:02}:{end_sec:05.2f}\n")
        out.write(f"{text}\n\n")

global _asr
_asr = None

def transcribe(audio, language=None):
    global _asr
    if _asr is None:
        _asr = WhisperModel("large-v3-turbo")
    segments, _ = _asr.transcribe(audio, language=language)

    return "\n".join([seg.text.strip() for seg in segments])

    # for seg in segments:
    #     yield {
    #         "start": seg.start,
    #         "end": seg.end,
    #         "text": seg.text
    #     }

def transcribe_with_cache(cache_folder, audio, *kargs, also_dump_audio=False, **kwargs):
    if not cache_folder:
        return transcribe(audio, *kargs, **kwargs)
    cache_filename = array_signature(audio)
    cache_file = os.path.join(cache_folder, cache_filename + ".srt")
    if os.path.isfile(cache_file):
        with open(cache_file) as f:
            result = f.read().strip()
    else:
        result = transcribe(audio, *kargs, **kwargs)
        with open(cache_file, 'w') as f:
            f.write(result.strip() + "\n")
    if also_dump_audio:
        audio_file = os.path.join(cache_folder, cache_filename + ".wav")
        if not os.path.isfile(audio_file):
            save_audio(audio_file, audio)
    return result


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="+")
    parser.add_argument("--language", default=None, help="Target language")
    args = parser.parse_args()

    for audio_file in args.input:
        output_file, _ = os.path.splitext(audio_file)
        output_file += ".srt"

        with open(output_file, 'w') as out:
            write_srt(out, transcribe(load_audio(audio_file), language=args.language))

