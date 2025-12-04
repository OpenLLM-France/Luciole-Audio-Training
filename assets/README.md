# Meta data
- To build the datasets_metadata.csv we used : assets/nemo_datasets_summary.py

# Visualisation:
- To plot the distribution of the dataset: assets/plot_metadata.py

# Dataset Summary & Key Metrics

## Overview Statistics

| Metric | Value |
|--------|--------|
| **Total Datasets** | 82 unique dataset entries |
| **Task Types** | 4 (AQA, ASR, AST, QA) |
| **Languages** | 3 (English, French, French-English) |
| **Total Audio Segments** | 16,772,705 segments |
| **Total Duration** | ~48,503 hours (2020.9 days) |

## Dataset Distribution by Task Type

| Task Type | Description | Datasets | Audio Segments | Duration (Hours) |
|-----------|-------------|----------|----------------|------------------|
| **ASR** | Automatic Speech Recognition | 48 | 14,522,560 | ~37,359 hrs |
| **QA** | Question Answering | 27 | 1,959,531 | ~10,696 hrs |
| **AST** | Audio Speech Translation | 4 | 265,140 | ~287 hrs |
| **AQA** | Audio Question Answering | 3 | 25,474 | ~159 hrs |

## Dataset Distribution by Language

| Language | Datasets | Audio Segments | Duration (Hours) |
|----------|----------|----------------|------------------|
| **English (en)** | 26 | 10,477,900 | ~27,463 hrs |
| **French (fr)** | 52 | 6,029,665 | ~20,752 hrs |
| **French-English (fr-en)** | 4 | 265,140 | ~287 hrs |

## Key Dataset Characteristics

| Characteristic | Value |
|----------------|--------|
| **Avg. Segment Duration** | 10.4 seconds |
| **Min Segment Duration** | 0.01 seconds |
| **Max Segment Duration** | 788.64 seconds |
| **Avg. Instruction Length** | 8.61 words |
| **Avg. Response Length** | 31.2 words |

## Split Distribution

| Split | Datasets | Percentage |
|-------|----------|------------|
| **Train** | 42 | 51.2% |
| **Test** | 31 | 37.8% |
| **Dev** | 9 | 11.0% |

---
*Data compiled from 82 dataset configurations across multiple speech and language understanding tasks*