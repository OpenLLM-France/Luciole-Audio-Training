import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
import argparse
from pathlib import Path

def extract_audio_duration_vectorized(messages_series):
    """Vectorized version using pandas operations"""
    def extract_single(messages):
        try:
            for msg in messages:
                if msg.get("role") == "user":
                    for item in msg.get("content", []):
                        if item.get("type") == "audio":
                            array = item.get("array")
                            sr = item.get("sampling_rate", 16000)
                            if array is not None:
                                return len(array) / sr
        except:
            pass
        return None
    
    return messages_series.map(extract_single)

def process_single_file(parquet_file):
    """Process a single parquet file - optimized for multiprocessing"""
    try:
        # Read only the messages column to save memory
        df = pd.read_parquet(parquet_file, columns=['messages'])
        durations = extract_audio_duration_vectorized(df["messages"])
        # Filter out None values and convert to numpy array in one step
        valid_durations = durations.dropna().values
        return valid_durations
    except Exception as e:
        print(f"⚠️ Error reading {parquet_file}: {e}")
        return np.array([])

def process_dataset_parallel(tag, dataset, split, pattern, max_workers=None):
    """Process dataset files in parallel"""
    parquet_files = glob.glob(pattern)
    
    if not parquet_files:
        return {
            "Tag": tag, "Dataset": dataset, "Split": split,
            "Num Files": 0, "Count": 0, "Total (hr)": 0,
            "Mean (sec)": 0, "Median (sec)": 0, "Min (sec)": 0, "Max (sec)": 0,
        }
    
    if max_workers is None:
        max_workers = min(len(parquet_files), mp.cpu_count())
    
    all_durations = []
    
    # Use ProcessPoolExecutor for CPU-bound tasks
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all files for processing
        future_to_file = {executor.submit(process_single_file, f): f for f in parquet_files}
        
        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_file), 
                          total=len(parquet_files), 
                          desc=f"{dataset} - {split}"):
            durations = future.result()
            if len(durations) > 0:
                all_durations.append(durations)
    
    # Concatenate all duration arrays efficiently
    if all_durations:
        durations_np = np.concatenate(all_durations)
    else:
        durations_np = np.array([])
    
    # Calculate statistics
    if len(durations_np) > 0:
        return {
            "Tag": tag,
            "Dataset": dataset,
            "Split": split,
            "Num Files": len(parquet_files),
            "Count": len(durations_np),
            "Total (hr)": round(durations_np.sum() / 3600, 2),
            "Mean (sec)": round(durations_np.mean(), 2),
            "Median (sec)": round(np.median(durations_np), 2),
            "Min (sec)": round(durations_np.min(), 2),
            "Max (sec)": round(durations_np.max(), 2),
        }
    else:
        return {
            "Tag": tag, "Dataset": dataset, "Split": split,
            "Num Files": len(parquet_files), "Count": 0, "Total (hr)": 0,
            "Mean (sec)": 0, "Median (sec)": 0, "Min (sec)": 0, "Max (sec)": 0,
        }

def discover_datasets(base_paths, tag="question-answering"):
    """
    Automatically discover datasets with train/test subfolders
    
    Args:
        base_paths: List of base paths to search for datasets
        tag: Tag to assign to all datasets
    
    Returns:
        List of (tag, dataset_name, split, pattern) tuples
    """
    datasets = []
    
    for base_path in base_paths:
        base_path = Path(base_path)
        
        if not base_path.exists():
            print(f"⚠️ Warning: Path does not exist: {base_path}")
            continue
        
        print(f"🔍 Examining: {base_path}")
        
        # Check if this path itself has train/test subdirectories (direct dataset path)
        train_dir = base_path / "train"
        test_dir = base_path / "test"
        
        if train_dir.exists() or test_dir.exists():
            # This is a direct dataset path
            dataset_name = base_path.name
            print(f"📂 Direct dataset found: {dataset_name}")
            
            for split in ['train', 'test']:
                split_dir = base_path / split
                if split_dir.exists() and split_dir.is_dir():
                    pattern = str(split_dir / "*.parquet")
                    parquet_files = list(split_dir.glob("*.parquet"))
                    if parquet_files:
                        datasets.append((tag, dataset_name, split, pattern))
                        print(f"  ✅ Found: {dataset_name}/{split} -> {len(parquet_files)} files")
                    else:
                        print(f"  ⚠️ No parquet files in: {dataset_name}/{split}")
        else:
            # Look for dataset directories within this path
            print(f"📁 Searching subdirectories in: {base_path}")
            found_any = False
            
            for dataset_dir in base_path.iterdir():
                if dataset_dir.is_dir():
                    dataset_name = dataset_dir.name
                    
                    # Check for train and test subdirectories
                    for split in ['train', 'test']:
                        split_dir = dataset_dir / split
                        if split_dir.exists() and split_dir.is_dir():
                            pattern = str(split_dir / "*.parquet")
                            parquet_files = list(split_dir.glob("*.parquet"))
                            if parquet_files:
                                datasets.append((tag, dataset_name, split, pattern))
                                print(f"  ✅ Found: {dataset_name}/{split} -> {len(parquet_files)} files")
                                found_any = True
                            else:
                                print(f"  ⚠️ No parquet files in: {dataset_name}/{split}")
            
            if not found_any:
                print(f"  ❌ No valid datasets found in: {base_path}")
    
    return datasets

def main():
    parser = argparse.ArgumentParser(description='Extract audio durations from parquet files')
    parser.add_argument('paths', nargs='+', help='Base paths to search for datasets')
    parser.add_argument('--tag', default='question-answering', help='Tag to assign to datasets')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    print(f"🔍 Searching for datasets in: {args.paths}")
    datasets = discover_datasets(args.paths, args.tag)
    
    if not datasets:
        print("❌ No datasets found!")
        return
    
    print(f"\n📊 Processing {len(datasets)} dataset splits...")
    results = []
    
    # Process each dataset
    for tag, dataset, split, pattern in datasets:
        result = process_dataset_parallel(tag, dataset, split, pattern, args.workers)
        results.append(result)
    
    # Display results
    df_summary = pd.DataFrame(results)
    print("\n" + "="*80)
    print("SUMMARY RESULTS:")
    print("="*80)
    print(df_summary.to_string(index=False))
    
    # Optional: Save to CSV
    output_file = "audio_duration_summary.csv"
    df_summary.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")

if __name__ == "__main__":
    main()