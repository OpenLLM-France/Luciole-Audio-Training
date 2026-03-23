import pandas as pd

# Load the CSV
df = pd.read_csv('/home/usertn2/Audio-Adapter-Training/assets/datasets_metadata.csv')

# Clean up column names (strip whitespace)
df.columns = df.columns.str.strip()

# Clean up string columns
for col in ['dataset_name', 'split', 'task_type', 'sub_task', 'language']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# Mappings for display
language_map = {
    'en': 'English (en)',
    'fr': 'French (fr)',
    'fr-en': 'French-English (fr-en)'
}

task_descriptions = {
    'ASR': 'Automatic Speech Recognition',
    'QA': 'Question Answering',
    'AST': 'Audio Speech Translation',
    'AQA': 'Audio Question Answering'
}

# 1. Overview Statistics
total_datasets = len(df)
task_types = sorted(df['task_type'].unique())
languages = sorted(df['language'].unique())
total_segments = df['num_audio_segments'].sum()
total_duration_sec = df['total_duration_sec'].sum()
total_duration_hours = total_duration_sec / 3600
total_duration_days = total_duration_hours / 24

print("## Overview Statistics")
print(f"")
print(f"| Metric | Value |")
print(f"|--------|--------|")
print(f"| **Total Datasets** | {total_datasets} unique dataset entries |")
print(f"| **Task Types** | {len(task_types)} ({', '.join(task_types)}) |")
print(f"| **Languages** | {len(languages)} ({', '.join([language_map.get(l, l).split(' ')[0] for l in languages])}) |")
print(f"| **Total Audio Segments** | {total_segments:,} segments |")
print(f"| **Total Duration** | ~{total_duration_hours:,.0f} hours ({total_duration_days:.1f} days) |")
print()

# 2. Dataset Distribution by Task Type
print("## Dataset Distribution by Task Type")
print(f"")
print(f"| Task Type | Description | Datasets | Audio Segments | Duration (Hours) |")
print(f"|-----------|-------------|----------|----------------|------------------|")

task_stats = df.groupby('task_type').agg({
    'dataset_name': 'count',
    'num_audio_segments': 'sum',
    'total_duration_sec': 'sum'
}).reset_index()

# Sort by count desc
task_stats = task_stats.sort_values('dataset_name', ascending=False)

for _, row in task_stats.iterrows():
    task = row['task_type']
    desc = task_descriptions.get(task, task)
    count = row['dataset_name']
    segments = row['num_audio_segments']
    duration = row['total_duration_sec'] / 3600
    print(f"| **{task}** | {desc} | {count} | {segments:,} | ~{duration:,.0f} hrs |")
print()

# 3. Dataset Distribution by Language
print("## Dataset Distribution by Language")
print(f"")
print(f"| Language | Datasets | Audio Segments | Duration (Hours) |")
print(f"|----------|----------|----------------|------------------|")

lang_stats = df.groupby('language').agg({
    'dataset_name': 'count',
    'num_audio_segments': 'sum',
    'total_duration_sec': 'sum'
}).reset_index()

# Sort by duration desc
lang_stats = lang_stats.sort_values('total_duration_sec', ascending=False)

for _, row in lang_stats.iterrows():
    lang = row['language']
    display_lang = language_map.get(lang, lang)
    count = row['dataset_name']
    segments = row['num_audio_segments']
    duration = row['total_duration_sec'] / 3600
    print(f"| **{display_lang}** | {count} | {segments:,} | ~{duration:,.0f} hrs |")
print()

# 4. Key Dataset Characteristics
print("## Key Dataset Characteristics")
print(f"")
print(f"| Characteristic | Value |")
print(f"|----------------|--------|")

# Weighted averages
avg_segment_duration = total_duration_sec / total_segments if total_segments else 0
min_segment_duration = df['min_segment_duration_sec'].min()
max_segment_duration = df['max_segment_duration_sec'].max()

total_instruction_words = (df['avg_instruction_words'] * df['num_audio_segments']).sum()
avg_instruction_length = total_instruction_words / total_segments if total_segments else 0

total_response_words = (df['avg_response_words'] * df['num_audio_segments']).sum()
avg_response_length = total_response_words / total_segments if total_segments else 0

print(f"| **Avg. Segment Duration** | {avg_segment_duration:.1f} seconds |")
print(f"| **Min Segment Duration** | {min_segment_duration} seconds |")
print(f"| **Max Segment Duration** | {max_segment_duration} seconds |")
print(f"| **Avg. Instruction Length** | {avg_instruction_length:.2f} words |")
print(f"| **Avg. Response Length** | {avg_response_length:.1f} words |")
print()

# 5. Split Distribution
print("## Split Distribution")
print(f"")
print(f"| Split | Datasets | Percentage |")
print(f"|-------|----------|------------|")

split_stats = df['split'].value_counts().reset_index()
split_stats.columns = ['split', 'count']
total_splits = split_stats['count'].sum()

# Sort by count desc
split_stats = split_stats.sort_values('count', ascending=False)

for _, row in split_stats.iterrows():
    split = row['split'].capitalize()
    count = row['count']
    percentage = (count / total_splits) * 100
    print(f"| **{split}** | {count} | {percentage:.1f}% |")
