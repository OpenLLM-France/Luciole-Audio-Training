import pandas as pd
import random
import os
import numpy as np
from tqdm import tqdm

def convert_audio_array_dtype(messages):
    """
    Ensures all 'audio' arrays in 'messages' are np.float32.
    Works on a list of message groups (e.g., a single row in the 'messages' column).
    """
    if isinstance(messages, np.ndarray):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                for item in msg.get("content", []):
                    if item.get("type") == "audio":
                        arr = item.get("array")
                        if isinstance(arr, list):
                            item["array"] = np.array(arr, dtype=np.float32)
                        elif isinstance(arr, np.ndarray) and arr.dtype != np.float32:
                            item["array"] = arr.astype(np.float32)
    return messages

def interleave_parquets(parquet_files, output_folder, group_by = 10, seed=42):

    random.seed(seed)  # For reproducibility

    # Shuffle the parquet files
    random.shuffle(parquet_files)

    # Create a new output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    i_output = 0
    while len(parquet_files) > 0:
        # Create a list to hold the dataframes for this group
        dfs = []
        current_files = []
        # Read up to 'group_by' parquet files
        for _ in tqdm(range(group_by), desc=f"Reading group {i_output}", leave=False):
            if len(parquet_files) == 0:
                break
            file = parquet_files.pop(0)
            df = pd.read_parquet(file)
            # Ensure 'messages' arrays are float32
            if "messages" in df.columns:
                df["messages"] = df["messages"].apply(convert_audio_array_dtype)
            
            dfs.append(df)
            current_files.append(file)

        # Concatenate the dataframes
        try:
            combined_df = pd.concat(dfs, ignore_index=True)
        except Exception as err:
            # Can fail because columns are not the same in all files
            raise RuntimeError(f"Error combining files {current_files}") from err

        # Shuffle the combined dataframe
        combined_df = combined_df.sample(frac=1).reset_index(drop=True)

        # Split the combined dataframe into n(=group_by) parts
        num_by_parquet = len(combined_df) // group_by + (1 if len(combined_df) % group_by > 0 else 0)
        for i in tqdm(range(0, group_by), desc=f"Writing group {i_output}", leave=False):

            # Get the current chunk of data
            chunk_df = combined_df.iloc[i * num_by_parquet:(i + 1) * num_by_parquet]
            if chunk_df.empty:
                continue

            output_file = os.path.join(output_folder, f'interleaved_{i_output:04d}_{i:03d}.parquet')
            chunk_df.to_parquet(output_file, index=False)

        i_output += 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Interleave multiple parquet files into a single output folder.')
    parser.add_argument('parquet_files', nargs='*', help='List of parquet files to interleave.')
    parser.add_argument('--file_list', type=str, help='Path to a text file containing one parquet path per line.')
    parser.add_argument('--output_folder', default='interleaved_output', help='Output folder path.')
    parser.add_argument('--group_by', type=int, default=10, help='Number of files per group.')

    args = parser.parse_args()

    if args.file_list:
        with open(args.file_list, 'r') as f:
            parquet_files = [line.strip() for line in f if line.strip()]
    else:
        parquet_files = args.parquet_files

    if not parquet_files:
        print("❌ No parquet files provided.")
        exit(1)

    interleave_parquets(parquet_files, args.output_folder, args.group_by)