#!/usr/bin/env python3
"""
Manifest-Based Audio Format Converter
Reads audio paths from manifest.jsonl and converts with preserved directory structure
Optimized for large-scale audio processing with parallel execution
"""

import os
import sys
import json
from pathlib import Path
import argparse
import librosa
import soundfile as sf
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def extract_audio_paths_from_manifest(manifest_path):
    """
    Extract audio file paths from manifest.jsonl file.
    
    Args:
        manifest_path: Path to manifest.jsonl file
    
    Returns:
        List of audio file paths
    """
    audio_paths = []
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Extract audio paths from conversations
                    if 'conversations' in data:
                        for conv in data['conversations']:
                            if conv.get('type') == 'audio' and 'value' in conv:
                                audio_path = conv['value']
                                if audio_path and Path(audio_path).exists():
                                    audio_paths.append(audio_path)
                                elif audio_path:
                                    print(f"Warning: Audio file not found: {audio_path}")
                    else:
                        # Fallback to 'audio_filepath' key
                        if 'audio_filepath' in data:
                            audio_path = data['audio_filepath']
                            if audio_path and Path(audio_path).exists():
                                audio_paths.append(audio_path)
                            elif audio_path:
                                print(f"Warning: Audio file not found: {audio_path}")
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    continue
        
        # Remove duplicates while preserving order
        audio_paths = list(dict.fromkeys(audio_paths))
        
    except FileNotFoundError:
        print(f"Error: Manifest file not found: {manifest_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading manifest: {e}")
        sys.exit(1)
    
    return audio_paths

def replace_base_path(original_path, old_base, new_base):
    """
    Replace the base path while preserving the rest of the directory structure.
    
    Args:
        original_path: Original file path
        old_base: Old base path to replace
        new_base: New base path
    
    Returns:
        New path with replaced base
    """
    original_path = Path(original_path)
    old_base = Path(old_base)
    new_base = Path(new_base)
    
    try:
        # Get the relative path from old base
        rel_path = original_path.relative_to(old_base)
        # Construct new path
        new_path = new_base / rel_path
        return new_path
    except ValueError:
        # If original_path is not relative to old_base, return as is under new_base
        return new_base / original_path.name

def convert_audio_to_wav(args_tuple):
    """
    Convert audio file to WAV format with 16kHz sample rate and float32 data type.
    Optimized for speed and memory efficiency.
    
    Args:
        args_tuple: Tuple of (input_path, output_path, target_sr)
    
    Returns:
        Tuple of (success: bool, input_path: str, output_path: str, error: str)
    """
    input_path, output_path, target_sr = args_tuple
    try:
        input_path = Path(input_path)
        output_path = Path(output_path)

        # ✅ Ensure output filename ends with .wav
        output_path = output_path.with_suffix(".wav")

        if output_path.exists():
            return (True, str(input_path), str(output_path), "already_exists")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        audio, sr = librosa.load(
            input_path,
            sr=target_sr,
            mono=True,
            dtype=np.float32,
            res_type="kaiser_fast"
        )

        sf.write(output_path, audio, target_sr, subtype='PCM_16', format='WAV')

        return (True, str(input_path), str(output_path), None)

    except Exception as e:
        return (False, str(input_path), str(output_path), str(e))

def process_manifest_parallel(manifest_path, old_base, new_base, target_sr=16000, num_workers=None):
    """
    Process audio files from manifest in parallel using multiprocessing.
    
    Args:
        manifest_path: Path to manifest.jsonl file
        old_base: Old base path to replace (e.g., "/lustre/.../recordings")
        new_base: New base path (e.g., "/lustre/.../converted_audios")
        target_sr: Target sample rate
        num_workers: Number of parallel workers (default: CPU count)
    """
    if num_workers is None:
        num_workers = cpu_count()
    
    # Extract audio paths from manifest
    print("Reading manifest and extracting audio paths...")
    audio_paths = extract_audio_paths_from_manifest(manifest_path)
    
    if not audio_paths:
        print("No valid audio paths found in manifest")
        return
    
    print(f"Found {len(audio_paths)} unique audio file(s)")
    print(f"Processing using {num_workers} workers...\n")
    
    # Prepare arguments for parallel processing
    conversion_args = []
    for audio_path in audio_paths:
        output_path = replace_base_path(audio_path, old_base, new_base)
        conversion_args.append((audio_path, output_path, target_sr))
    
    # Process files in parallel with progress bar
    success_count = 0
    skip_count = 0
    error_count = 0
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(convert_audio_to_wav, conversion_args),
            total=len(conversion_args),
            desc="Converting",
            unit="file"
        ))
    
    # Count results and show errors
    print("\n" + "="*80)
    for success, input_path, output_path, error in results:
        if success:
            if error == "already_exists":
                skip_count += 1
            else:
                success_count += 1
        else:
            error_count += 1
            print(f"✗ Error: {Path(input_path).name} - {error}")
    
    print("="*80)
    print(f"✓ Successfully converted: {success_count} files")
    if skip_count > 0:
        print(f"⊙ Skipped (already exists): {skip_count} files")
    if error_count > 0:
        print(f"✗ Failed: {error_count} files")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(
        description='Convert audio files from manifest.jsonl with preserved directory structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Basic usage with manifest
            python audio_converter.py -m manifest.jsonl \\
                --old-base /home/audio/recordings \\
                --new-base /home/fsn1/projects/rech/qgz/commun/audio/converted_audios

            # With custom workers and sample rate
            python audio_converter.py -m manifest.jsonl \\
                --old-base /path/to/recordings \\
                --new-base /path/to/converted \\
                -w 16 --sample-rate 22050

            # Use all CPU cores for maximum speed
            python audio_converter.py -m manifest.jsonl \\
                --old-base /old/path \\
                --new-base /new/path \\
                -w -1

            How it works:
            The script reads audio paths from manifest.jsonl and replaces the base path
            while preserving the entire directory structure after it.
            
            Example transformation:
                Input:  /home/.../recordings/fr/African.../speech/train/ca16/002/file.wav
                Output: /home/.../converted_audios/fr/African.../speech/train/ca16/002/file.wav
        """
    )
    
    parser.add_argument('-m', '--manifest', required=True,
                       help='Path to manifest.jsonl file')
    parser.add_argument('--old-base', required=True,
                       help='Old base path to replace (e.g., /path/to/recordings)')
    parser.add_argument('--new-base', required=True,
                       help='New base path (e.g., /path/to/converted_audios)')
    parser.add_argument('-w', '--workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU count, -1 for all cores)')
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='Target sample rate in Hz (default: 16000)')
    
    args = parser.parse_args()
    
    # Validate manifest file
    if not os.path.isfile(args.manifest):
        print(f"Error: Manifest file not found: {args.manifest}")
        sys.exit(1)
    
    # Set number of workers
    num_workers = args.workers
    if num_workers == -1:
        num_workers = cpu_count()
    elif num_workers is None:
        num_workers = cpu_count()
    
    # Show configuration
    print("="*80)
    print("Configuration:")
    print(f"  Manifest: {args.manifest}")
    print(f"  Old base: {args.old_base}")
    print(f"  New base: {args.new_base}")
    print(f"  Sample rate: {args.sample_rate} Hz")
    print(f"  Workers: {num_workers}")
    print("="*80 + "\n")
    
    # Process manifest
    process_manifest_parallel(
        args.manifest,
        args.old_base,
        args.new_base,
        target_sr=args.sample_rate,
        num_workers=num_workers
    )

if __name__ == "__main__":
    main()