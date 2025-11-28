# Meta data
- To build the datasets_metadata.csv we used : assets/nemo_datasets_summary.py

# Visualisation:
- To plot the distribution of the dataset: assets/plot_metadata.py

# Dataset Summary & Key Metrics

## Overview Statistics

| Metric | Value |
|--------|--------|
| **Total Datasets** | 82 unique dataset entries |
| **Task Types** | 4 (ASR, AST, QA, AQA) |
| **Languages** | 3 (English, French, French-English) |
| **Total Audio Segments** | 15,247,539 segments |
| **Total Duration** | ~1,167 hours (48.6 days) |

## Dataset Distribution by Task Type

| Task Type | Description | Datasets | Audio Segments | Duration (Hours) |
|-----------|-------------|----------|----------------|------------------|
| **ASR** | Automatic Speech Recognition | 54 | 13,439,646 | ~1,087 hrs |
| **QA** | Question Answering | 25 | 1,782,395 | ~76 hrs |
| **AST** | Audio Speech Translation | 4 | 265,140 | ~3.6 hrs |
| **AQA** | Audio Question Answering | 3 | 25,474 | ~0.4 hrs |

## Dataset Distribution by Language

| Language | Datasets | Audio Segments | Duration (Hours) |
|----------|----------|----------------|------------------|
| **French (fr)** | 56 | 11,945,052 | ~927 hrs |
| **English (en)** | 22 | 3,037,347 | ~237 hrs |
| **French-English (fr-en)** | 4 | 265,140 | ~3.6 hrs |

## Key Dataset Characteristics

| Characteristic | Value |
|----------------|--------|
| **Avg. Segment Duration** | 4.6 seconds |
| **Min Segment Duration** | 0.01 seconds |
| **Max Segment Duration** | 788.64 seconds |
| **Avg. Instruction Length** | 8.85 words |
| **Avg. Response Length** | 42.8 words |

## Split Distribution

| Split | Datasets | Percentage |
|-------|----------|------------|
| **Train** | 62 | 75.6% |
| **Test** | 16 | 19.5% |
| **Dev** | 4 | 4.9% |

---
*Data compiled from 82 dataset configurations across multiple speech and language understanding tasks*