# 📊 Dataset Composition Report

> Generated: 2026-03-17 08:48

---

## 📌 Table of Contents

1. [Global Statistics](#1-global-statistics)
2. [Distribution by Task](#2-distribution-by-task)
3. [Distribution by Language](#3-distribution-by-language)
4. [Task × Language Cross-Table](#4-task--language-cross-table)
5. [Audio Duration Statistics](#5-audio-duration-statistics)
6. [Detailed Dataset Tables per Task](#6-detailed-dataset-tables-per-task)
   - [AQA](#aqa)
   - [ASR](#asr)
   - [AST](#ast)
   - [OTHER](#other)
   - [QA](#qa)
6.5. [Actual Sampling Weights](#65-actual-sampling-weights)
8. [Suggested Sampling Weights](#8-suggested-sampling-weights)
9. [Suggested max_steps for Training](#9-suggested-max_steps-for-training)
10. [Balance Notes](#10-balance-notes)

---

## 1. Global Statistics

| Metric                   | Value |
|--------------------------|-------|
| Total manifests          | **118** |
| Unique datasets          | **66** |
| Unique languages / pairs | **29** |
| Unique tasks             | **5** |
| Total samples            | **37.76M** (37,763,479) |
| Total audio duration     | **117,039.4 h** (4876d 15h 22m 38s) |

---

## 2. Distribution by Task

| Task  | Manifests | Datasets | Samples | Samples % | Hours     | Hours % |
| ----- | --------- | -------- | ------- | --------- | --------- | ------- |
| asr   | 51        | 23       | 26.77M  | 70.9%     | 84781.7 h | 72.4%   |
| ast   | 31        | 16       | 7.26M   | 19.2%     | 10606.7 h | 9.1%    |
| qa    | 13        | 12       | 1.82M   | 4.8%      | 10838.2 h | 9.3%    |
| other | 13        | 8        | 1.04M   | 2.7%      | 1559.1 h  | 1.3%    |
| aqa   | 10        | 8        | 870.5K  | 2.3%      | 9253.6 h  | 7.9%    |

---

## 3. Distribution by Language

| Language / Pair | Manifests | Samples | Samples % | Hours     | Hours % |
| --------------- | --------- | ------- | --------- | --------- | ------- |
| en              | 28        | 22.86M  | 60.5%     | 82919.7 h | 70.8%   |
| fr              | 36        | 4.81M   | 12.7%     | 15832.5 h | 13.5%   |
| de              | 5         | 1.20M   | 3.2%      | 3213.1 h  | 2.7%    |
| en→fr           | 1         | 1.14M   | 3.0%      | 1798.4 h  | 1.5%    |
| fr→en           | 3         | 870.4K  | 2.3%      | 1211.3 h  | 1.0%    |
| es              | 5         | 730.3K  | 1.9%      | 1764.9 h  | 1.5%    |
| fr→es           | 2         | 613.8K  | 1.6%      | 883.2 h   | 0.8%    |
| de→fr           | 1         | 607.5K  | 1.6%      | 962.1 h   | 0.8%    |
| fr→pt           | 2         | 606.2K  | 1.6%      | 870.7 h   | 0.7%    |
| fr→it           | 1         | 593.0K  | 1.6%      | 850.7 h   | 0.7%    |
| fr→nl           | 1         | 592.7K  | 1.6%      | 850.3 h   | 0.7%    |
| fr→de           | 1         | 592.5K  | 1.6%      | 850.0 h   | 0.7%    |
| nl              | 4         | 441.6K  | 1.2%      | 1662.4 h  | 1.4%    |
| en→de           | 1         | 429.3K  | 1.1%      | 587.6 h   | 0.5%    |
| es→fr           | 2         | 357.2K  | 0.9%      | 514.8 h   | 0.4%    |
| it              | 5         | 308.0K  | 0.8%      | 689.8 h   | 0.6%    |
| de→en           | 1         | 264.1K  | 0.7%      | 345.6 h   | 0.3%    |
| pt              | 4         | 153.5K  | 0.4%      | 350.2 h   | 0.3%    |
| es→en           | 2         | 128.2K  | 0.3%      | 195.8 h   | 0.2%    |
| fr→ar           | 1         | 116.0K  | 0.3%      | 166.2 h   | 0.1%    |
| it→fr           | 1         | 109.7K  | 0.3%      | 161.8 h   | 0.1%    |
| it→en           | 2         | 67.4K   | 0.2%      | 108.2 h   | 0.1%    |
| pt→fr           | 2         | 45.8K   | 0.1%      | 52.6 h    | 0.0%    |
| pt→en           | 2         | 45.7K   | 0.1%      | 70.9 h    | 0.1%    |
| nl→fr           | 1         | 43.4K   | 0.1%      | 54.1 h    | 0.0%    |
| es→pt           | 1         | 21.1K   | 0.1%      | 37.3 h    | 0.0%    |
| pt→es           | 1         | 11.5K   | 0.0%      | 20.6 h    | 0.0%    |
| es→it           | 1         | 5.6K    | 0.0%      | 9.9 h     | 0.0%    |
| it→es           | 1         | 2.3K    | 0.0%      | 4.7 h     | 0.0%    |

---

## 4. Task × Language Cross-Table

> Cell values = samples.  Rows = tasks, columns = languages/pairs.

| Task  | de    | de→en  | de→fr  | en     | en→de  | en→fr | es     | es→en  | es→fr  | es→it | es→pt | fr     | fr→ar  | fr→de  | fr→en  | fr→es  | fr→it  | fr→nl  | fr→pt  | it     | it→en | it→es | it→fr  | nl     | nl→fr | pt     | pt→en | pt→es | pt→fr |
| ----- | ----- | ------ | ------ | ------ | ------ | ----- | ------ | ------ | ------ | ----- | ----- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ----- | ----- | ------ | ------ | ----- | ------ | ----- | ----- | ----- |
| aqa   | 0     | 0      | 0      | 848.1K | 0      | 0     | 0      | 0      | 0      | 0     | 0     | 22.4K  | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0     | 0     | 0      | 0      | 0     | 0      | 0     | 0     | 0     |
| asr   | 1.20M | 0      | 0      | 20.12M | 0      | 0     | 730.3K | 0      | 0      | 0     | 0     | 3.83M  | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 308.0K | 0     | 0     | 0      | 441.6K | 0     | 153.5K | 0     | 0     | 0     |
| ast   | 0     | 264.1K | 607.5K | 0      | 429.3K | 1.14M | 0      | 128.2K | 357.2K | 5.6K  | 21.1K | 0      | 116.0K | 592.5K | 870.4K | 613.8K | 593.0K | 592.7K | 606.2K | 0      | 67.4K | 2.3K  | 109.7K | 0      | 43.4K | 0      | 45.7K | 11.5K | 45.8K |
| other | 0     | 0      | 0      | 709.1K | 0      | 0     | 0      | 0      | 0      | 0     | 0     | 326.2K | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0     | 0     | 0      | 0      | 0     | 0      | 0     | 0     | 0     |
| qa    | 0     | 0      | 0      | 1.19M  | 0      | 0     | 0      | 0      | 0      | 0     | 0     | 632.2K | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0     | 0     | 0      | 0      | 0     | 0      | 0     | 0     | 0     |

---

## 5. Audio Duration Statistics

| Task  | Avg Segment | Min Segment | Max Segment |
| ----- | ----------- | ----------- | ----------- |
| aqa   | 14.69 s     | 0.06 s      | 57.56 s     |
| asr   | 9.20 s      | 0.03 s      | 1347.41 s   |
| ast   | 5.44 s      | 0.03 s      | 1037.85 s   |
| other | 4.42 s      | 0.06 s      | 41.05 s     |
| qa    | 13.08 s     | 0.01 s      | 361.18 s    |

---

## 6. Detailed Dataset Tables per Task

> One table per task. Columns: Dataset · Split · Note · Language · Samples · % of total · Duration · Avg segment duration.
> Sorted by language then dataset name. Last row = task totals.

### AQA

> 10 manifests &nbsp;·&nbsp; 8 unique datasets &nbsp;·&nbsp; 2 language(s)

| Dataset                   | Split | Note | Language | Samples    | % Total   | Duration    | Avg Seg |
| ------------------------- | ----- | ---- | -------- | ---------- | --------- | ----------- | ------- |
| AcousticDS                | train |      | en       | 4.6K       | 0.01%     | 4.0 h       | 3.1 s   |
| CLEAR_v1.0.0_improved     | train |      | en       | 607.5K     | 1.61%     | 8,489 h     | 50.3 s  |
| CLEAR_v2.0.0_improved     | train |      | en       | 105.0K     | 0.28%     | 312.5 h     | 10.7 s  |
| MusicCaps                 | train |      | en       | 2.6K       | 0.01%     | 7.2 h       | 10.0 s  |
| MusicCaps                 | train |      | en       | 2.6K       | 0.01%     | 7.2 h       | 10.0 s  |
| clotho_aqa_improved       | train |      | en       | 5.8K       | 0.02%     | 36.4 h      | 22.6 s  |
| mispeech_MECAT-Caption    | train |      | en       | 19.8K      | 0.05%     | 55.4 h      | 10.1 s  |
| mispeech_MECAT-QA         | train |      | en       | 100.2K     | 0.27%     | 279.7 h     | 10.1 s  |
| MusicCaps                 | train |      | fr       | 2.6K       | 0.01%     | 7.2 h       | 10.0 s  |
| mispeech_MECAT-Caption2FR | train |      | fr       | 19.8K      | 0.05%     | 55.4 h      | 10.1 s  |
| **TOTAL (AQA)**           |       |      |          | **870.5K** | **2.31%** | **9,254 h** |         |

### ASR

> 51 manifests &nbsp;·&nbsp; 23 unique datasets &nbsp;·&nbsp; 7 language(s)

| Dataset                  | Split | Note                                   | Language | Samples    | % Total    | Duration     | Avg Seg |
| ------------------------ | ----- | -------------------------------------- | -------- | ---------- | ---------- | ------------ | ------- |
| CommonVoice              | train |                                        | de       | 607.9K     | 1.61%      | 962.7 h      | 5.7 s   |
| FLEURS                   | train |                                        | de       | 3.0K       | 0.01%      | 9.0 h        | 10.9 s  |
| Multilingual_LibriSpeech | train |                                        | de       | 469.9K     | 1.24%      | 1,967 h      | 15.1 s  |
| Multilingual_TEDx        | train |                                        | de       | 6.8K       | 0.02%      | 10.3 h       | 5.5 s   |
| VoxPopuli                | train |                                        | de       | 108.5K     | 0.29%      | 264.7 h      | 8.8 s   |
| CommonVoice              | train |                                        | en       | 1.14M      | 3.02%      | 1,798 h      | 5.7 s   |
| FLEURS                   | train |                                        | en       | 2.5K       | 0.01%      | 7.3 h        | 10.4 s  |
| Multilingual_LibriSpeech | train |                                        | en       | 10.81M     | 28.62%     | 44,660 h     | 14.9 s  |
| VoxPopuli                | train |                                        | en       | 182.5K     | 0.48%      | 522.6 h      | 10.3 s  |
| Yodas                    | train |                                        | en       | 7.98M      | 21.14%     | 19,447 h     | 8.8 s   |
| CommonVoice              | train |                                        | es       | 353.7K     | 0.94%      | 508.7 h      | 5.2 s   |
| FLEURS                   | train |                                        | es       | 2.8K       | 0.01%      | 8.8 h        | 11.3 s  |
| Multilingual_LibriSpeech | train |                                        | es       | 220.7K     | 0.58%      | 917.7 h      | 15.0 s  |
| Multilingual_TEDx        | train |                                        | es       | 102.2K     | 0.27%      | 177.8 h      | 6.3 s   |
| VoxPopuli                | train |                                        | es       | 50.9K      | 0.13%      | 151.9 h      | 10.7 s  |
| ACSYNT                   | train |                                        | fr       | 3.5K       | 0.01%      | 7.9 h        | 8.0 s   |
| AfricanAccentedFrench    | train | with punctuations, max duration is 30s | fr       | 11.5K      | 0.03%      | 13.7 h       | 4.3 s   |
| CFPB                     | train | with punctuations                      | fr       | 18.4K      | 0.05%      | 9.1 h        | 1.8 s   |
| CFPP2000                 | train | with punctuations                      | fr       | 30.3K      | 0.08%      | 38.3 h       | 4.6 s   |
| CLAPI                    | train | with punctuations                      | fr       | 17.8K      | 0.05%      | 21.9 h       | 4.4 s   |
| CommonVoice              | train |                                        | fr       | 593.1K     | 1.57%      | 850.8 h      | 5.2 s   |
| ESLO                     | train | with punctuations                      | fr       | 454.6K     | 1.20%      | 272.4 h      | 2.2 s   |
| FLEURS                   | train |                                        | fr       | 3.2K       | 0.01%      | 10.3 h       | 11.6 s  |
| LINAGORA_Meetings        | train |                                        | fr       | 6.2K       | 0.02%      | 10.3 h       | 6.0 s   |
| LVL-Atril-CTFAR          | train |                                        | fr       | 58.2K      | 0.15%      | 83.0 h       | 5.1 s   |
| LVL-Atril-CTFNN1         | train |                                        | fr       | 69.8K      | 0.18%      | 42.0 h       | 2.2 s   |
| LesVocaux                | train | with punctuations                      | fr       | 744        | 0.00%      | 10.3 h       | 49.7 s  |
| Multilingual_LibriSpeech | train |                                        | fr       | 258.2K     | 0.68%      | 1,077 h      | 15.0 s  |
| Multilingual_TEDx        | train |                                        | fr       | 116.0K     | 0.31%      | 175.8 h      | 5.5 s   |
| PFC                      | train |                                        | fr       | 67.4K      | 0.18%      | 62.3 h       | 3.3 s   |
| PxSLU                    | train | with punctuations                      | fr       | 2.0K       | 0.01%      | 4.2 h        | 7.8 s   |
| SimSamu                  | train |                                        | fr       | 3.1K       | 0.01%      | 2.6 h        | 3.0 s   |
| TCOF_Adultes             | train | with punctuations                      | fr       | 79.7K      | 0.21%      | 56.2 h       | 2.5 s   |
| TCOF_Enfants             | train | with punctuations                      | fr       | 73.1K      | 0.19%      | 51.5 h       | 2.5 s   |
| VoxForge                 | train | with punctuations                      | fr       | 22.4K      | 0.06%      | 37.2 h       | 6.0 s   |
| VoxPopuli                | train |                                        | fr       | 73.6K      | 0.19%      | 205.7 h      | 10.1 s  |
| Yodas                    | train |                                        | fr       | 332.7K     | 0.88%      | 2,373 h      | 25.7 s  |
| YouTubeFr                | train |                                        | fr       | 1.53M      | 4.06%      | 5,250 h      | 12.3 s  |
| CommonVoice              | train |                                        | it       | 172.8K     | 0.46%      | 254.7 h      | 5.3 s   |
| FLEURS                   | train |                                        | it       | 3.0K       | 0.01%      | 9.0 h        | 10.7 s  |
| Multilingual_LibriSpeech | train |                                        | it       | 59.6K      | 0.16%      | 247.4 h      | 14.9 s  |
| Multilingual_TEDx        | train |                                        | it       | 50.0K      | 0.13%      | 100.7 h      | 7.3 s   |
| VoxPopuli                | train |                                        | it       | 22.6K      | 0.06%      | 78.1 h       | 12.4 s  |
| CommonVoice              | train |                                        | nl       | 43.5K      | 0.12%      | 54.1 h       | 4.5 s   |
| FLEURS                   | train |                                        | nl       | 2.9K       | 0.01%      | 7.7 h        | 9.5 s   |
| Multilingual_LibriSpeech | train |                                        | nl       | 374.3K     | 0.99%      | 1,554 h      | 14.9 s  |
| VoxPopuli                | train |                                        | nl       | 21.0K      | 0.06%      | 46.4 h       | 8.0 s   |
| CommonVoice              | train |                                        | pt       | 22.9K      | 0.06%      | 26.3 h       | 4.1 s   |
| FLEURS                   | train |                                        | pt       | 2.8K       | 0.01%      | 10.2 h       | 13.1 s  |
| Multilingual_LibriSpeech | train |                                        | pt       | 37.5K      | 0.10%      | 161.0 h      | 15.4 s  |
| Multilingual_TEDx        | train |                                        | pt       | 90.2K      | 0.24%      | 152.8 h      | 6.1 s   |
| **TOTAL (ASR)**          |       |                                        |          | **26.77M** | **70.90%** | **84,782 h** |         |

### AST

> 31 manifests &nbsp;·&nbsp; 16 unique datasets &nbsp;·&nbsp; 22 language(s)

| Dataset             | Split | Note | Language | Samples   | % Total    | Duration     | Avg Seg |
| ------------------- | ----- | ---- | -------- | --------- | ---------- | ------------ | ------- |
| CoVoST              | train |      | de→en    | 264.1K    | 0.70%      | 345.6 h      | 4.7 s   |
| CommonVoiceDE2FR    | train |      | de→fr    | 607.5K    | 1.61%      | 962.1 h      | 5.7 s   |
| CoVoST              | train |      | en→de    | 429.3K    | 1.14%      | 587.6 h      | 4.9 s   |
| CommonVoiceEN2FR    | train |      | en→fr    | 1.14M     | 3.02%      | 1,798 h      | 5.7 s   |
| CoVoST              | train |      | es→en    | 91.9K     | 0.24%      | 131.4 h      | 5.1 s   |
| Multilingual_TEDx   | train |      | es→en    | 36.3K     | 0.10%      | 64.4 h       | 6.4 s   |
| CommonVoiceES2FR    | train |      | es→fr    | 353.5K    | 0.94%      | 508.5 h      | 5.2 s   |
| Multilingual_TEDx   | train |      | es→fr    | 3.7K      | 0.01%      | 6.3 h        | 6.2 s   |
| Multilingual_TEDx   | train |      | es→it    | 5.6K      | 0.01%      | 9.9 h        | 6.4 s   |
| Multilingual_TEDx   | train |      | es→pt    | 21.1K     | 0.06%      | 37.3 h       | 6.4 s   |
| CommonVoiceFR2AR    | train |      | fr→ar    | 116.0K    | 0.31%      | 166.2 h      | 5.2 s   |
| CommonVoiceFR2DE    | train |      | fr→de    | 592.5K    | 1.57%      | 850.0 h      | 5.2 s   |
| CoVoST              | train |      | fr→en    | 247.2K    | 0.65%      | 315.4 h      | 4.6 s   |
| CommonVoiceFR2EN    | train |      | fr→en    | 593.0K    | 1.57%      | 850.8 h      | 5.2 s   |
| Multilingual_TEDx   | train |      | fr→en    | 30.2K     | 0.08%      | 45.1 h       | 5.4 s   |
| CommonVoiceFR2ES    | train |      | fr→es    | 593.0K    | 1.57%      | 850.7 h      | 5.2 s   |
| Multilingual_TEDx   | train |      | fr→es    | 20.8K     | 0.06%      | 32.5 h       | 5.6 s   |
| CommonVoiceFR2IT    | train |      | fr→it    | 593.0K    | 1.57%      | 850.7 h      | 5.2 s   |
| CommonVoiceFR2NL    | train |      | fr→nl    | 592.7K    | 1.57%      | 850.3 h      | 5.2 s   |
| CommonVoiceFR2PT    | train |      | fr→pt    | 593.0K    | 1.57%      | 850.7 h      | 5.2 s   |
| Multilingual_TEDx   | train |      | fr→pt    | 13.3K     | 0.04%      | 20.0 h       | 5.4 s   |
| CoVoST              | train |      | it→en    | 42.9K     | 0.11%      | 60.5 h       | 5.1 s   |
| Multilingual_TEDx   | train |      | it→en    | 24.6K     | 0.07%      | 47.7 h       | 7.0 s   |
| Multilingual_TEDx   | train |      | it→es    | 2.3K      | 0.01%      | 4.7 h        | 7.6 s   |
| CommonVoiceIT2FR    | train |      | it→fr    | 109.7K    | 0.29%      | 161.8 h      | 5.3 s   |
| CommonVoiceNL2FR    | train |      | nl→fr    | 43.4K     | 0.12%      | 54.1 h       | 4.5 s   |
| CoVoST              | train |      | pt→en    | 14.9K     | 0.04%      | 17.9 h       | 4.3 s   |
| Multilingual_TEDx   | train |      | pt→en    | 30.9K     | 0.08%      | 53.0 h       | 6.2 s   |
| Multilingual_TEDx   | train |      | pt→es    | 11.5K     | 0.03%      | 20.6 h       | 6.5 s   |
| CommonVoicePT2FR    | train |      | pt→fr    | 22.9K     | 0.06%      | 26.3 h       | 4.1 s   |
| CommonVoicePT2FR_7B | train |      | pt→fr    | 22.9K     | 0.06%      | 26.3 h       | 4.1 s   |
| **TOTAL (AST)**     |       |      |          | **7.26M** | **19.23%** | **10,607 h** |         |

### OTHER

> 13 manifests &nbsp;·&nbsp; 8 unique datasets &nbsp;·&nbsp; 2 language(s)

| Dataset                                      | Split | Note | Language | Samples   | % Total   | Duration    | Avg Seg |
| -------------------------------------------- | ----- | ---- | -------- | --------- | --------- | ----------- | ------- |
| CommonVoice_Age-Gender_reco                  | train |      | en       | 667.2K    | 1.77%     | 1,036 h     | 5.6 s   |
| EmotionDS                                    | train |      | en       | 10.0K     | 0.03%     | 8.7 h       | 3.1 s   |
| VoxLingua_lang_identification                | train |      | en       | 13.5K     | 0.04%     | 36.3 h      | 9.7 s   |
| ssd                                          | train |      | en       | 4.4K      | 0.01%     | 3.6 h       | 3.0 s   |
| ssi-speech-emotion-recognition               | train |      | en       | 9.6K      | 0.03%     | 6.9 h       | 2.6 s   |
| ssr                                          | train |      | en       | 4.4K      | 0.01%     | 3.6 h       | 3.0 s   |
| CommonVoice_Age-Gender_reco                  | train |      | fr       | 288.1K    | 0.76%     | 407.7 h     | 5.1 s   |
| EmotionDS                                    | train |      | fr       | 10.0K     | 0.03%     | 8.7 h       | 3.1 s   |
| OreauFR_02                                   | train |      | fr       | 434       | 0.00%     | 0.3 h       | 2.8 s   |
| VoxLingua_lang_identification                | train |      | fr       | 13.5K     | 0.04%     | 36.5 h      | 9.7 s   |
| emotional-speech-audio-dataset-4languages-fr | train |      | fr       | 144       | 0.00%     | 0.2 h       | 4.2 s   |
| ssd                                          | train |      | fr       | 4.4K      | 0.01%     | 3.6 h       | 3.0 s   |
| ssi-speech-emotion-recognition               | train |      | fr       | 9.6K      | 0.03%     | 6.9 h       | 2.6 s   |
| **TOTAL (OTHER)**                            |       |      |          | **1.04M** | **2.74%** | **1,559 h** |         |

### QA

> 13 manifests &nbsp;·&nbsp; 12 unique datasets &nbsp;·&nbsp; 2 language(s)

| Dataset                                       | Split | Note | Language | Samples   | % Total   | Duration     | Avg Seg |
| --------------------------------------------- | ----- | ---- | -------- | --------- | --------- | ------------ | ------- |
| Menlo--instruction-speech-encodec-v1.5        | train |      | en       | 332.4K    | 0.88%     | 678.2 h      | 7.3 s   |
| VoxPopuli-QA                                  | train |      | en       | 568.0K    | 1.50%     | 4,540 h      | 28.8 s  |
| amuvarma--10k-filtered-tune-audio-text_answer | train |      | en       | 9.0K      | 0.02%     | 13.0 h       | 5.2 s   |
| gruhit-patel--alpaca_speech_instruct          | train |      | en       | 51.8K     | 0.14%     | 72.1 h       | 5.0 s   |
| liangtianle--science-question                 | train |      | en       | 3.5K      | 0.01%     | 7.7 h        | 7.9 s   |
| liangtianle--unsafety-question                | train |      | en       | 3.4K      | 0.01%     | 15.9 h       | 16.9 s  |
| slu-phase-2-sqa5                              | train |      | en       | 46.2K     | 0.12%     | 511.2 h      | 39.8 s  |
| worstchan--UltraChat-300K-SLAM-Omni           | train |      | en       | 164.5K    | 0.44%     | 342.1 h      | 7.5 s   |
| yijingwu--HeySQuAD_human                      | train |      | en       | 10.8K     | 0.03%     | 18.0 h       | 6.0 s   |
| CohereLabs--aya_collection                    | train |      | fr       | 723       | 0.00%     | 1.0 h        | 5.0 s   |
| ComparIA                                      | train |      | fr       | 12.9K     | 0.03%     | 21.5 h       | 6.0 s   |
| Vigogne--Alpaca                               | train |      | fr       | 49.7K     | 0.13%     | 79.7 h       | 5.8 s   |
| VoxPopuli-QA                                  | train |      | fr       | 568.9K    | 1.51%     | 4,538 h      | 28.7 s  |
| **TOTAL (QA)**                                |       |      |          | **1.82M** | **4.82%** | **10,838 h** |         |

---

## 6.5 Actual Sampling Weights

> These weights reflect **how often each dataset is actually drawn** during training, accounting for the full Task → Language → Dataset hierarchy from the YAML config and within-language balancing.

### Methodology

1. **Group weight** (`weight_group` from YAML) sets the probability budget for each Task and Language level.
2. **Within-language balancing** uses `sqrt(total_duration_sec)` (or `sqrt(num_samples)` if duration unavailable) as the raw score, then applies **temperature smoothing T=2.0**: `score_i ^ (1/T)`, normalised within the language group.
   - T=1 → proportional to √duration (larger datasets dominate).
   - T=2 (current) → moderate levelling of size differences.
   - T→∞ → uniform sampling within the language group.
3. **Effective weight** = group_weight × intra_lang_norm_weight.
4. **Sampling probability** = effective_weight / Σ(all effective_weights) → sums to 100% across the full training pipeline.
5. **Est. passes/epoch** ≈ (sampling_prob × total_samples) / dataset_samples — how many full sweeps the model makes through this dataset per training epoch.

---

### Summary: sampling probability by Task × Language

> Each row = one language group within a task. `Sampling %` sums to 100% across the whole table.

| Task      | Language | Datasets | Samples    | Duration      | Group weight | Sampling % |
| --------- | -------- | -------- | ---------- | ------------- | ------------ | ---------- |
| aqa       | en       | 7        | 848.1K     | 9191.1 h      | 0.0500       | 0.980%     |
| aqa       | fr       | 2        | 22.4K      | 62.5 h        | 0.0500       | 0.980%     |
| asr       | en       | 5        | 20.12M     | 66435.4 h     | 0.3000       | 5.882%     |
| asr       | fr       | 23       | 3.83M      | 10665.8 h     | 0.3000       | 5.882%     |
| asr       | de       | 5        | 1.20M      | 3213.1 h      | 0.0800       | 1.569%     |
| asr       | es       | 5        | 730.3K     | 1764.9 h      | 0.0800       | 1.569%     |
| asr       | it       | 5        | 308.0K     | 689.8 h       | 0.0800       | 1.569%     |
| asr       | nl       | 4        | 441.6K     | 1662.4 h      | 0.0800       | 1.569%     |
| asr       | pt       | 4        | 153.5K     | 350.2 h       | 0.0800       | 1.569%     |
| ast       | es→en    | 2        | 128.2K     | 195.8 h       | 0.1500       | 2.941%     |
| ast       | es→fr    | 2        | 357.2K     | 514.8 h       | 0.1500       | 2.941%     |
| ast       | fr→es    | 2        | 613.8K     | 883.2 h       | 0.1500       | 2.941%     |
| ast       | fr→pt    | 2        | 606.2K     | 870.7 h       | 0.1500       | 2.941%     |
| ast       | it→en    | 2        | 67.4K      | 108.2 h       | 0.1500       | 2.941%     |
| ast       | pt→en    | 2        | 45.7K      | 70.9 h        | 0.1500       | 2.941%     |
| ast       | pt→fr    | 2        | 45.8K      | 52.6 h        | 0.1500       | 2.941%     |
| ast       | de→en    | 1        | 264.1K     | 345.6 h       | 0.1500       | 2.941%     |
| ast       | de→fr    | 1        | 607.5K     | 962.1 h       | 0.1500       | 2.941%     |
| ast       | en→de    | 1        | 429.3K     | 587.6 h       | 0.1500       | 2.941%     |
| ast       | en→fr    | 1        | 1.14M      | 1798.4 h      | 0.1500       | 2.941%     |
| ast       | es→it    | 1        | 5.6K       | 9.9 h         | 0.1500       | 2.941%     |
| ast       | es→pt    | 1        | 21.1K      | 37.3 h        | 0.1500       | 2.941%     |
| ast       | fr→ar    | 1        | 116.0K     | 166.2 h       | 0.1500       | 2.941%     |
| ast       | fr→de    | 1        | 592.5K     | 850.0 h       | 0.1500       | 2.941%     |
| ast       | fr→en    | 3        | 870.4K     | 1211.3 h      | 0.1500       | 2.941%     |
| ast       | fr→it    | 1        | 593.0K     | 850.7 h       | 0.1500       | 2.941%     |
| ast       | fr→nl    | 1        | 592.7K     | 850.3 h       | 0.1500       | 2.941%     |
| ast       | it→es    | 1        | 2.3K       | 4.7 h         | 0.1500       | 2.941%     |
| ast       | it→fr    | 1        | 109.7K     | 161.8 h       | 0.1500       | 2.941%     |
| ast       | nl→fr    | 1        | 43.4K      | 54.1 h        | 0.1500       | 2.941%     |
| ast       | pt→es    | 1        | 11.5K      | 20.6 h        | 0.1500       | 2.941%     |
| other     | en       | 6        | 709.1K     | 1095.2 h      | 0.0500       | 0.980%     |
| other     | fr       | 7        | 326.2K     | 464.0 h       | 0.0500       | 0.980%     |
| qa        | en       | 9        | 1.19M      | 6198.1 h      | 0.3000       | 5.882%     |
| qa        | fr       | 4        | 632.2K     | 4640.1 h      | 0.3000       | 5.882%     |
| **TOTAL** |          |          | **37.76M** | **117,039 h** |              | **100.0%** |

---

### Per-task sampling weight detail

#### AQA

> 10 datasets &nbsp;·&nbsp; total sampling share: **1.961%**

| Dataset                                  | Language | Samples | Duration | Group wt | Intra-lang wt | Eff. weight | Sampling % | Est. passes |
| ---------------------------------------- | -------- | ------- | -------- | -------- | ------------- | ----------- | ---------- | ----------- |
| CLEAR_v1.0.0_improved                    | en       | 607.5K  | 8,489 h  | 0.0500   | 0.3457        | 0.01729     | 0.3390%    | 0.21×       |
| CLEAR_v2.0.0_improved                    | en       | 105.0K  | 312.5 h  | 0.0500   | 0.1514        | 0.00757     | 0.1485%    | 0.53×       |
| mispeech_MECAT-QA                        | en       | 100.2K  | 279.7 h  | 0.0500   | 0.1473        | 0.00737     | 0.1444%    | 0.54×       |
| mispeech_MECAT-Caption                   | en       | 19.8K   | 55.4 h   | 0.0500   | 0.0983        | 0.00491     | 0.0963%    | 1.83×       |
| clotho_aqa_improved ⚠ over-sampled       | en       | 5.8K    | 36.4 h   | 0.0500   | 0.0885        | 0.00442     | 0.0867%    | 5.64×       |
| MusicCaps ⚠ over-sampled                 | en       | 2.6K    | 7.2 h    | 0.0500   | 0.0589        | 0.00295     | 0.0578%    | 8.46×       |
| MusicCaps ⚠ over-sampled                 | en       | 2.6K    | 7.2 h    | 0.0500   | 0.0589        | 0.00295     | 0.0578%    | 8.46×       |
| AcousticDS                               | en       | 4.6K    | 4.0 h    | 0.0500   | 0.0509        | 0.00255     | 0.0499%    | 4.12×       |
| mispeech_MECAT-Caption2FR ⚠ over-sampled | fr       | 19.8K   | 55.4 h   | 0.0500   | 0.6251        | 0.03125     | 0.6128%    | 11.67×      |
| MusicCaps ⚠ over-sampled                 | fr       | 2.6K    | 7.2 h    | 0.0500   | 0.3749        | 0.01875     | 0.3676%    | 53.8×       |
| **TOTAL AQA**                            |          |         |          |          |               |             | **1.961%** |             |

#### ASR

> 51 datasets &nbsp;·&nbsp; total sampling share: **19.608%**

| Dataset                                 | Language | Samples | Duration | Group wt | Intra-lang wt | Eff. weight | Sampling %  | Est. passes |
| --------------------------------------- | -------- | ------- | -------- | -------- | ------------- | ----------- | ----------- | ----------- |
| Multilingual_LibriSpeech                | de       | 469.9K  | 1,967 h  | 0.0800   | 0.3366        | 0.02693     | 0.5280%     | 0.42×       |
| CommonVoice                             | de       | 607.9K  | 962.7 h  | 0.0800   | 0.2816        | 0.02252     | 0.4417%     | 0.27×       |
| VoxPopuli                               | de       | 108.5K  | 264.7 h  | 0.0800   | 0.2039        | 0.01631     | 0.3198%     | 1.11×       |
| Multilingual_TEDx ⚠ over-sampled        | de       | 6.8K    | 10.3 h   | 0.0800   | 0.0905        | 0.00724     | 0.1420%     | 7.93×       |
| FLEURS ⚠ over-sampled                   | de       | 3.0K    | 9.0 h    | 0.0800   | 0.0875        | 0.00700     | 0.1372%     | 17.43×      |
| Multilingual_LibriSpeech                | en       | 10.81M  | 44,660 h | 0.3000   | 0.3701        | 0.11102     | 2.1769%     | 0.08×       |
| Yodas                                   | en       | 7.98M   | 19,447 h | 0.3000   | 0.3006        | 0.09019     | 1.7684%     | 0.08×       |
| CommonVoice                             | en       | 1.14M   | 1,798 h  | 0.3000   | 0.1658        | 0.04973     | 0.9752%     | 0.32×       |
| VoxPopuli                               | en       | 182.5K  | 522.6 h  | 0.3000   | 0.1217        | 0.03652     | 0.7160%     | 1.48×       |
| FLEURS ⚠ over-sampled                   | en       | 2.5K    | 7.3 h    | 0.3000   | 0.0418        | 0.01254     | 0.2459%     | 36.9×       |
| Multilingual_LibriSpeech                | es       | 220.7K  | 917.7 h  | 0.0800   | 0.2876        | 0.02301     | 0.4511%     | 0.77×       |
| CommonVoice                             | es       | 353.7K  | 508.7 h  | 0.0800   | 0.2482        | 0.01985     | 0.3893%     | 0.42×       |
| Multilingual_TEDx                       | es       | 102.2K  | 177.8 h  | 0.0800   | 0.1908        | 0.01526     | 0.2993%     | 1.11×       |
| VoxPopuli                               | es       | 50.9K   | 151.9 h  | 0.0800   | 0.1834        | 0.01468     | 0.2878%     | 2.13×       |
| FLEURS ⚠ over-sampled                   | es       | 2.8K    | 8.8 h    | 0.0800   | 0.0900        | 0.00720     | 0.1412%     | 19.07×      |
| YouTubeFr                               | fr       | 1.53M   | 5,250 h  | 0.3000   | 0.1175        | 0.03525     | 0.6912%     | 0.17×       |
| Yodas                                   | fr       | 332.7K  | 2,373 h  | 0.3000   | 0.0964        | 0.02891     | 0.5668%     | 0.64×       |
| Multilingual_LibriSpeech                | fr       | 258.2K  | 1,077 h  | 0.3000   | 0.0791        | 0.02372     | 0.4651%     | 0.68×       |
| CommonVoice                             | fr       | 593.1K  | 850.8 h  | 0.3000   | 0.0746        | 0.02237     | 0.4386%     | 0.28×       |
| ESLO                                    | fr       | 454.6K  | 272.4 h  | 0.3000   | 0.0561        | 0.01683     | 0.3299%     | 0.27×       |
| VoxPopuli                               | fr       | 73.6K   | 205.7 h  | 0.3000   | 0.0523        | 0.01568     | 0.3075%     | 1.58×       |
| Multilingual_TEDx                       | fr       | 116.0K  | 175.8 h  | 0.3000   | 0.0503        | 0.01508     | 0.2957%     | 0.96×       |
| LVL-Atril-CTFAR                         | fr       | 58.2K   | 83.0 h   | 0.3000   | 0.0417        | 0.01250     | 0.2451%     | 1.59×       |
| PFC                                     | fr       | 67.4K   | 62.3 h   | 0.3000   | 0.0388        | 0.01163     | 0.2281%     | 1.28×       |
| TCOF_Adultes                            | fr       | 79.7K   | 56.2 h   | 0.3000   | 0.0378        | 0.01134     | 0.2223%     | 1.05×       |
| TCOF_Enfants                            | fr       | 73.1K   | 51.5 h   | 0.3000   | 0.0370        | 0.01110     | 0.2176%     | 1.12×       |
| LVL-Atril-CTFNN1                        | fr       | 69.8K   | 42.0 h   | 0.3000   | 0.0351        | 0.01054     | 0.2067%     | 1.12×       |
| CFPP2000                                | fr       | 30.3K   | 38.3 h   | 0.3000   | 0.0343        | 0.01030     | 0.2020%     | 2.52×       |
| VoxForge                                | fr       | 22.4K   | 37.2 h   | 0.3000   | 0.0341        | 0.01023     | 0.2005%     | 3.38×       |
| CLAPI                                   | fr       | 17.8K   | 21.9 h   | 0.3000   | 0.0299        | 0.00896     | 0.1757%     | 3.73×       |
| AfricanAccentedFrench ⚠ over-sampled    | fr       | 11.5K   | 13.7 h   | 0.3000   | 0.0266        | 0.00797     | 0.1562%     | 5.14×       |
| LINAGORA_Meetings ⚠ over-sampled        | fr       | 6.2K    | 10.3 h   | 0.3000   | 0.0248        | 0.00743     | 0.1456%     | 8.87×       |
| FLEURS ⚠ over-sampled                   | fr       | 3.2K    | 10.3 h   | 0.3000   | 0.0247        | 0.00742     | 0.1456%     | 17.21×      |
| LesVocaux ⚠ over-sampled                | fr       | 744     | 10.3 h   | 0.3000   | 0.0247        | 0.00741     | 0.1453%     | 73.77×      |
| CFPB                                    | fr       | 18.4K   | 9.1 h    | 0.3000   | 0.0240        | 0.00719     | 0.1410%     | 2.89×       |
| ACSYNT ⚠ over-sampled                   | fr       | 3.5K    | 7.9 h    | 0.3000   | 0.0232        | 0.00695     | 0.1362%     | 14.53×      |
| PxSLU ⚠ over-sampled                    | fr       | 2.0K    | 4.2 h    | 0.3000   | 0.0198        | 0.00593     | 0.1163%     | 22.47×      |
| SimSamu ⚠ over-sampled                  | fr       | 3.1K    | 2.6 h    | 0.3000   | 0.0176        | 0.00527     | 0.1033%     | 12.62×      |
| CommonVoice                             | it       | 172.8K  | 254.7 h  | 0.0800   | 0.2523        | 0.02018     | 0.3958%     | 0.86×       |
| Multilingual_LibriSpeech                | it       | 59.6K   | 247.4 h  | 0.0800   | 0.2505        | 0.02004     | 0.3929%     | 2.49×       |
| Multilingual_TEDx                       | it       | 50.0K   | 100.7 h  | 0.0800   | 0.2001        | 0.01601     | 0.3139%     | 2.37×       |
| VoxPopuli                               | it       | 22.6K   | 78.1 h   | 0.0800   | 0.1877        | 0.01502     | 0.2945%     | 4.93×       |
| FLEURS ⚠ over-sampled                   | it       | 3.0K    | 9.0 h    | 0.0800   | 0.1094        | 0.00875     | 0.1716%     | 21.4×       |
| Multilingual_LibriSpeech                | nl       | 374.3K  | 1,554 h  | 0.0800   | 0.4734        | 0.03787     | 0.7425%     | 0.75×       |
| CommonVoice                             | nl       | 43.5K   | 54.1 h   | 0.0800   | 0.2045        | 0.01636     | 0.3208%     | 2.79×       |
| VoxPopuli ⚠ over-sampled                | nl       | 21.0K   | 46.4 h   | 0.0800   | 0.1967        | 0.01574     | 0.3086%     | 5.56×       |
| FLEURS ⚠ over-sampled                   | nl       | 2.9K    | 7.7 h    | 0.0800   | 0.1254        | 0.01003     | 0.1967%     | 25.46×      |
| Multilingual_LibriSpeech ⚠ over-sampled | pt       | 37.5K   | 161.0 h  | 0.0800   | 0.3201        | 0.02561     | 0.5021%     | 5.05×       |
| Multilingual_TEDx                       | pt       | 90.2K   | 152.8 h  | 0.0800   | 0.3159        | 0.02527     | 0.4956%     | 2.07×       |
| CommonVoice ⚠ over-sampled              | pt       | 22.9K   | 26.3 h   | 0.0800   | 0.2035        | 0.01628     | 0.3193%     | 5.26×       |
| FLEURS ⚠ over-sampled                   | pt       | 2.8K    | 10.2 h   | 0.0800   | 0.1605        | 0.01284     | 0.2517%     | 34.07×      |
| **TOTAL ASR**                           |          |         |          |          |               |             | **19.608%** |             |

#### AST

> 31 datasets &nbsp;·&nbsp; total sampling share: **64.706%**

| Dataset                            | Language | Samples | Duration | Group wt | Intra-lang wt | Eff. weight | Sampling %  | Est. passes |
| ---------------------------------- | -------- | ------- | -------- | -------- | ------------- | ----------- | ----------- | ----------- |
| CoVoST                             | de→en    | 264.1K  | 345.6 h  | 0.1500   | 1.0000        | 0.15000     | 2.9412%     | 4.2×        |
| CommonVoiceDE2FR                   | de→fr    | 607.5K  | 962.1 h  | 0.1500   | 1.0000        | 0.15000     | 2.9412%     | 1.83×       |
| CoVoST                             | en→de    | 429.3K  | 587.6 h  | 0.1500   | 1.0000        | 0.15000     | 2.9412%     | 2.59×       |
| CommonVoiceEN2FR                   | en→fr    | 1.14M   | 1,798 h  | 0.1500   | 1.0000        | 0.15000     | 2.9412%     | 0.98×       |
| CoVoST ⚠ over-sampled              | es→en    | 91.9K   | 131.4 h  | 0.1500   | 0.5445        | 0.08167     | 1.6014%     | 6.58×       |
| Multilingual_TEDx ⚠ over-sampled   | es→en    | 36.3K   | 64.4 h   | 0.1500   | 0.4555        | 0.06833     | 1.3398%     | 13.95×      |
| CommonVoiceES2FR                   | es→fr    | 353.5K  | 508.5 h  | 0.1500   | 0.7495        | 0.11243     | 2.2044%     | 2.35×       |
| Multilingual_TEDx ⚠ over-sampled   | es→fr    | 3.7K    | 6.3 h    | 0.1500   | 0.2505        | 0.03757     | 0.7367%     | 75.95×      |
| Multilingual_TEDx ⚠ over-sampled   | es→it    | 5.6K    | 9.9 h    | 0.1500   | 1.0000        | 0.15000     | 2.9412%     | 198.34×     |
| Multilingual_TEDx ⚠ over-sampled   | es→pt    | 21.1K   | 37.3 h   | 0.1500   | 1.0000        | 0.15000     | 2.9412%     | 52.62×      |
| CommonVoiceFR2AR ⚠ over-sampled    | fr→ar    | 116.0K  | 166.2 h  | 0.1500   | 1.0000        | 0.15000     | 2.9412%     | 9.57×       |
| CommonVoiceFR2DE                   | fr→de    | 592.5K  | 850.0 h  | 0.1500   | 1.0000        | 0.15000     | 2.9412%     | 1.87×       |
| CommonVoiceFR2EN                   | fr→en    | 593.0K  | 850.8 h  | 0.1500   | 0.4425        | 0.06637     | 1.3014%     | 0.83×       |
| CoVoST                             | fr→en    | 247.2K  | 315.4 h  | 0.1500   | 0.3453        | 0.05179     | 1.0154%     | 1.55×       |
| Multilingual_TEDx ⚠ over-sampled   | fr→en    | 30.2K   | 45.1 h   | 0.1500   | 0.2123        | 0.03184     | 0.6244%     | 7.81×       |
| CommonVoiceFR2ES                   | fr→es    | 593.0K  | 850.7 h  | 0.1500   | 0.6934        | 0.10401     | 2.0395%     | 1.3×        |
| Multilingual_TEDx ⚠ over-sampled   | fr→es    | 20.8K   | 32.5 h   | 0.1500   | 0.3066        | 0.04599     | 0.9017%     | 16.35×      |
| CommonVoiceFR2IT                   | fr→it    | 593.0K  | 850.7 h  | 0.1500   | 1.0000        | 0.15000     | 2.9412%     | 1.87×       |
| CommonVoiceFR2NL                   | fr→nl    | 592.7K  | 850.3 h  | 0.1500   | 1.0000        | 0.15000     | 2.9412%     | 1.87×       |
| CommonVoiceFR2PT                   | fr→pt    | 593.0K  | 850.7 h  | 0.1500   | 0.7185        | 0.10778     | 2.1133%     | 1.35×       |
| Multilingual_TEDx ⚠ over-sampled   | fr→pt    | 13.3K   | 20.0 h   | 0.1500   | 0.2815        | 0.04222     | 0.8279%     | 23.53×      |
| CoVoST ⚠ over-sampled              | it→en    | 42.9K   | 60.5 h   | 0.1500   | 0.5149        | 0.07723     | 1.5144%     | 13.34×      |
| Multilingual_TEDx ⚠ over-sampled   | it→en    | 24.6K   | 47.7 h   | 0.1500   | 0.4851        | 0.07277     | 1.4268%     | 21.92×      |
| Multilingual_TEDx ⚠ over-sampled   | it→es    | 2.3K    | 4.7 h    | 0.1500   | 1.0000        | 0.15000     | 2.9412%     | 491.24×     |
| CommonVoiceIT2FR ⚠ over-sampled    | it→fr    | 109.7K  | 161.8 h  | 0.1500   | 1.0000        | 0.15000     | 2.9412%     | 10.12×      |
| CommonVoiceNL2FR ⚠ over-sampled    | nl→fr    | 43.4K   | 54.1 h   | 0.1500   | 1.0000        | 0.15000     | 2.9412%     | 25.56×      |
| Multilingual_TEDx ⚠ over-sampled   | pt→en    | 30.9K   | 53.0 h   | 0.1500   | 0.5675        | 0.08512     | 1.6691%     | 20.43×      |
| CoVoST ⚠ over-sampled              | pt→en    | 14.9K   | 17.9 h   | 0.1500   | 0.4325        | 0.06488     | 1.2721%     | 32.25×      |
| Multilingual_TEDx ⚠ over-sampled   | pt→es    | 11.5K   | 20.6 h   | 0.1500   | 1.0000        | 0.15000     | 2.9412%     | 96.59×      |
| CommonVoicePT2FR_7B ⚠ over-sampled | pt→fr    | 22.9K   | 26.3 h   | 0.1500   | 0.5000        | 0.07500     | 1.4706%     | 24.23×      |
| CommonVoicePT2FR ⚠ over-sampled    | pt→fr    | 22.9K   | 26.3 h   | 0.1500   | 0.5000        | 0.07500     | 1.4705%     | 24.23×      |
| **TOTAL AST**                      |          |         |          |          |               |             | **64.706%** |             |

#### OTHER

> 13 datasets &nbsp;·&nbsp; total sampling share: **1.961%**

| Dataset                                                     | Language | Samples | Duration | Group wt | Intra-lang wt | Eff. weight | Sampling % | Est. passes |
| ----------------------------------------------------------- | -------- | ------- | -------- | -------- | ------------- | ----------- | ---------- | ----------- |
| CommonVoice_Age-Gender_reco                                 | en       | 667.2K  | 1,036 h  | 0.0500   | 0.3988        | 0.01994     | 0.3909%    | 0.22×       |
| VoxLingua_lang_identification                               | en       | 13.5K   | 36.3 h   | 0.0500   | 0.1725        | 0.00863     | 0.1691%    | 4.73×       |
| EmotionDS                                                   | en       | 10.0K   | 8.7 h    | 0.0500   | 0.1208        | 0.00604     | 0.1184%    | 4.48×       |
| ssi-speech-emotion-recognition                              | en       | 9.6K    | 6.9 h    | 0.0500   | 0.1137        | 0.00569     | 0.1115%    | 4.39×       |
| ssd ⚠ over-sampled                                          | en       | 4.4K    | 3.6 h    | 0.0500   | 0.0971        | 0.00486     | 0.0952%    | 8.17×       |
| ssr ⚠ over-sampled                                          | en       | 4.4K    | 3.6 h    | 0.0500   | 0.0971        | 0.00486     | 0.0952%    | 8.17×       |
| CommonVoice_Age-Gender_reco                                 | fr       | 288.1K  | 407.7 h  | 0.0500   | 0.3438        | 0.01719     | 0.3371%    | 0.44×       |
| VoxLingua_lang_identification ⚠ over-sampled                | fr       | 13.5K   | 36.5 h   | 0.0500   | 0.1881        | 0.00940     | 0.1844%    | 5.16×       |
| EmotionDS                                                   | fr       | 10.0K   | 8.7 h    | 0.0500   | 0.1315        | 0.00657     | 0.1289%    | 4.87×       |
| ssi-speech-emotion-recognition                              | fr       | 9.6K    | 6.9 h    | 0.0500   | 0.1238        | 0.00619     | 0.1214%    | 4.77×       |
| ssd ⚠ over-sampled                                          | fr       | 4.4K    | 3.6 h    | 0.0500   | 0.1057        | 0.00529     | 0.1036%    | 8.89×       |
| OreauFR_02 ⚠ over-sampled                                   | fr       | 434     | 0.3 h    | 0.0500   | 0.0582        | 0.00291     | 0.0571%    | 49.66×      |
| emotional-speech-audio-dataset-4languages-fr ⚠ over-sampled | fr       | 144     | 0.2 h    | 0.0500   | 0.0489        | 0.00245     | 0.0479%    | 125.73×     |
| **TOTAL OTHER**                                             |          |         |          |          |               |             | **1.961%** |             |

#### QA

> 13 datasets &nbsp;·&nbsp; total sampling share: **11.765%**

| Dataset                                                      | Language | Samples | Duration | Group wt | Intra-lang wt | Eff. weight | Sampling %  | Est. passes |
| ------------------------------------------------------------ | -------- | ------- | -------- | -------- | ------------- | ----------- | ----------- | ----------- |
| VoxPopuli-QA                                                 | en       | 568.0K  | 4,540 h  | 0.3000   | 0.2495        | 0.07484     | 1.4674%     | 0.98×       |
| Menlo--instruction-speech-encodec-v1.5                       | en       | 332.4K  | 678.2 h  | 0.3000   | 0.1551        | 0.04653     | 0.9123%     | 1.04×       |
| slu-phase-2-sqa5 ⚠ over-sampled                              | en       | 46.2K   | 511.2 h  | 0.3000   | 0.1445        | 0.04335     | 0.8500%     | 6.95×       |
| worstchan--UltraChat-300K-SLAM-Omni                          | en       | 164.5K  | 342.1 h  | 0.3000   | 0.1307        | 0.03921     | 0.7688%     | 1.77×       |
| gruhit-patel--alpaca_speech_instruct                         | en       | 51.8K   | 72.1 h   | 0.3000   | 0.0885        | 0.02656     | 0.5208%     | 3.8×        |
| yijingwu--HeySQuAD_human ⚠ over-sampled                      | en       | 10.8K   | 18.0 h   | 0.3000   | 0.0626        | 0.01879     | 0.3684%     | 12.93×      |
| liangtianle--unsafety-question ⚠ over-sampled                | en       | 3.4K    | 15.9 h   | 0.3000   | 0.0607        | 0.01821     | 0.3571%     | 39.86×      |
| amuvarma--10k-filtered-tune-audio-text_answer ⚠ over-sampled | en       | 9.0K    | 13.0 h   | 0.3000   | 0.0577        | 0.01731     | 0.3395%     | 14.24×      |
| liangtianle--science-question ⚠ over-sampled                 | en       | 3.5K    | 7.7 h    | 0.3000   | 0.0507        | 0.01520     | 0.2980%     | 32.15×      |
| VoxPopuli-QA                                                 | fr       | 568.9K  | 4,538 h  | 0.3000   | 0.5720        | 0.17159     | 3.3645%     | 2.23×       |
| Vigogne--Alpaca ⚠ over-sampled                               | fr       | 49.7K   | 79.7 h   | 0.3000   | 0.2082        | 0.06247     | 1.2249%     | 9.32×       |
| ComparIA ⚠ over-sampled                                      | fr       | 12.9K   | 21.5 h   | 0.3000   | 0.1501        | 0.04502     | 0.8827%     | 25.84×      |
| CohereLabs--aya_collection ⚠ over-sampled                    | fr       | 723     | 1.0 h    | 0.3000   | 0.0698        | 0.02093     | 0.4103%     | 214.31×     |
| **TOTAL QA**                                                 |          |         |          |          |               |             | **11.765%** |             |

---

### Balance observations

- ⚠️  **Over-sampled datasets** (est. passes/epoch > 5 — overfitting risk): **Multilingual_TEDx** (it→es, 491.2×), **CohereLabs--aya_collection** (fr, 214.3×), **Multilingual_TEDx** (es→it, 198.3×), **emotional-speech-audio-dataset-4languages-fr** (fr, 125.7×), **Multilingual_TEDx** (pt→es, 96.6×).
- 📌  Most sampled dataset: **VoxPopuli-QA** (fr) at **3.364%** of total training steps.
- 🟠  Language **en**: configured at **13.7%** vs raw data share **60.5%** (↓ reduced by 46.8 pp).

---

## 8. Suggested Sampling Weights

> Weights are **automatically computed** at all three YAML hierarchy levels using `sqrt(total_duration_sec)` (or `sqrt(num_samples)` as fallback), with temperature smoothing T=2.0.

> **How to read this section:**
> - **Level 1 (Task weight)** → put on the task-level `group` node.
> - **Level 2 (Language weight)** → put on each language `group` node *inside* the task.
> - **Level 3 (Dataset weight)** → put on each `multimodal_conversation` node *inside* the language group.
> - Effective probability of a dataset = Task_w × Lang_w × Dataset_w.

---

### Level 1 — Task weights

> Put these on the **task-level group nodes**. Sum = 1.0 across all tasks.

| Task      | Datasets | Duration  | Samples | Task weight (L1) |
| --------- | -------- | --------- | ------- | ---------------- |
| asr       | 51       | 84781.7 h | 26.77M  | 0.468331         |
| ast       | 31       | 10606.7 h | 7.26M   | 0.269911         |
| qa        | 13       | 10838.2 h | 1.82M   | 0.115949         |
| aqa       | 9        | 9253.6 h  | 870.5K  | 0.078829         |
| other     | 13       | 1559.1 h  | 1.04M   | 0.066980         |
| **TOTAL** |          |           |         | **1.0000**       |

---

### Level 2 — Language weights (within each task)

> Put these on **language-level group nodes inside each task**. Sum = 1.0 *within* each task.

| Task  | Language | Datasets | Duration  | Samples | Task w (L1) | Lang w (L2) ← YAML | Effective share |
| ----- | -------- | -------- | --------- | ------- | ----------- | ------------------ | --------------- |
| aqa   | en       | 7        | 9191.1 h  | 848.1K  | 0.0788      | 0.864163           | 6.812%          |
| aqa   | fr       | 2        | 62.5 h    | 22.4K   | 0.0788      | 0.135837           | 1.071%          |
| asr   | fr       | 23       | 10665.8 h | 3.83M   | 0.4683      | 0.379530           | 17.775%         |
| asr   | en       | 5        | 66435.4 h | 20.12M  | 0.4683      | 0.205802           | 9.638%          |
| asr   | de       | 5        | 3213.1 h  | 1.20M   | 0.4683      | 0.103652           | 4.854%          |
| asr   | es       | 5        | 1764.9 h  | 730.3K  | 0.4683      | 0.100268           | 4.696%          |
| asr   | it       | 5        | 689.8 h   | 308.0K  | 0.4683      | 0.082950           | 3.885%          |
| asr   | nl       | 4        | 1662.4 h  | 441.6K  | 0.4683      | 0.069495           | 3.255%          |
| asr   | pt       | 4        | 350.2 h   | 153.5K  | 0.4683      | 0.058304           | 2.731%          |
| ast   | fr→en    | 3        | 1211.3 h  | 870.4K  | 0.2699      | 0.110962           | 2.995%          |
| ast   | fr→es    | 2        | 883.2 h   | 613.8K  | 0.2699      | 0.070801           | 1.911%          |
| ast   | fr→pt    | 2        | 870.7 h   | 606.2K  | 0.2699      | 0.068327           | 1.844%          |
| ast   | en→fr    | 1        | 1798.4 h  | 1.14M   | 0.2699      | 0.059199           | 1.598%          |
| ast   | es→fr    | 2        | 514.8 h   | 357.2K  | 0.2699      | 0.057594           | 1.555%          |
| ast   | es→en    | 2        | 195.8 h   | 128.2K  | 0.2699      | 0.056527           | 1.526%          |
| ast   | de→fr    | 1        | 962.1 h   | 607.5K  | 0.2699      | 0.050630           | 1.367%          |
| ast   | it→en    | 2        | 108.2 h   | 67.4K   | 0.2699      | 0.049240           | 1.329%          |
| ast   | fr→it    | 1        | 850.7 h   | 593.0K  | 0.2699      | 0.049095           | 1.325%          |
| ast   | fr→nl    | 1        | 850.3 h   | 592.7K  | 0.2699      | 0.049089           | 1.325%          |
| ast   | fr→de    | 1        | 850.0 h   | 592.5K  | 0.2699      | 0.049085           | 1.325%          |
| ast   | en→de    | 1        | 587.6 h   | 429.3K  | 0.2699      | 0.044758           | 1.208%          |
| ast   | pt→en    | 2        | 70.9 h    | 45.7K   | 0.2699      | 0.043220           | 1.167%          |
| ast   | pt→fr    | 2        | 52.6 h    | 45.8K   | 0.2699      | 0.041178           | 1.111%          |
| ast   | de→en    | 1        | 345.6 h   | 264.1K  | 0.2699      | 0.039197           | 1.058%          |
| ast   | fr→ar    | 1        | 166.2 h   | 116.0K  | 0.2699      | 0.032641           | 0.881%          |
| ast   | it→fr    | 1        | 161.8 h   | 109.7K  | 0.2699      | 0.032422           | 0.875%          |
| ast   | nl→fr    | 1        | 54.1 h    | 43.4K   | 0.2699      | 0.024659           | 0.666%          |
| ast   | es→pt    | 1        | 37.3 h    | 21.1K   | 0.2699      | 0.022459           | 0.606%          |
| ast   | pt→es    | 1        | 20.6 h    | 11.5K   | 0.2699      | 0.019368           | 0.523%          |
| ast   | es→it    | 1        | 9.9 h     | 5.6K    | 0.2699      | 0.016133           | 0.435%          |
| ast   | it→es    | 1        | 4.7 h     | 2.3K    | 0.2699      | 0.013417           | 0.362%          |
| other | en       | 6        | 1095.2 h  | 709.1K  | 0.0670      | 0.521199           | 3.491%          |
| other | fr       | 7        | 464.0 h   | 326.2K  | 0.0670      | 0.478801           | 3.207%          |
| qa    | en       | 9        | 6198.1 h  | 1.19M   | 0.1159      | 0.696333           | 8.074%          |
| qa    | fr       | 4        | 4640.1 h  | 632.2K  | 0.1159      | 0.303667           | 3.521%          |

> **Check**: Effective share = Task_w × Lang_w.  All rows sum to 100%.

---

### Level 3 — Dataset weights (within each Task × Language group)

> Put these on **individual dataset nodes** (`multimodal_conversation`). Sum = 1.0 *within* each language group.

#### AQA

| Dataset                   | Language | Samples | Duration | Task w (L1) | Lang w (L2) | Dataset w (L3) ← YAML | Effective prob |
| ------------------------- | -------- | ------- | -------- | ----------- | ----------- | --------------------- | -------------- |
| CLEAR_v1.0.0_improved     | en       | 607.5K  | 8,489 h  | 0.0788      | 0.8642      | 0.345736              | 2.3552%        |
| CLEAR_v2.0.0_improved     | en       | 105.0K  | 312.5 h  | 0.0788      | 0.8642      | 0.151446              | 1.0317%        |
| mispeech_MECAT-QA         | en       | 100.2K  | 279.7 h  | 0.0788      | 0.8642      | 0.147303              | 1.0034%        |
| mispeech_MECAT-Caption    | en       | 19.8K   | 55.4 h   | 0.0788      | 0.8642      | 0.098252              | 0.6693%        |
| clotho_aqa_improved       | en       | 5.8K    | 36.4 h   | 0.0788      | 0.8642      | 0.088462              | 0.6026%        |
| MusicCaps                 | en       | 2.6K    | 7.2 h    | 0.0788      | 0.8642      | 0.058937              | 0.4015%        |
| MusicCaps                 | en       | 2.6K    | 7.2 h    | 0.0788      | 0.8642      | 0.058937              | 0.4015%        |
| AcousticDS                | en       | 4.6K    | 4.0 h    | 0.0788      | 0.8642      | 0.050926              | 0.3469%        |
| mispeech_MECAT-Caption2FR | fr       | 19.8K   | 55.4 h   | 0.0788      | 0.1358      | 0.625055              | 0.6693%        |
| MusicCaps                 | fr       | 2.6K    | 7.2 h    | 0.0788      | 0.1358      | 0.374945              | 0.4015%        |

#### ASR

| Dataset                  | Language | Samples | Duration | Task w (L1) | Lang w (L2) | Dataset w (L3) ← YAML | Effective prob |
| ------------------------ | -------- | ------- | -------- | ----------- | ----------- | --------------------- | -------------- |
| Multilingual_LibriSpeech | de       | 469.9K  | 1,967 h  | 0.4683      | 0.1037      | 0.336596              | 1.6340%        |
| CommonVoice              | de       | 607.9K  | 962.7 h  | 0.4683      | 0.1037      | 0.281553              | 1.3668%        |
| VoxPopuli                | de       | 108.5K  | 264.7 h  | 0.4683      | 0.1037      | 0.203871              | 0.9897%        |
| Multilingual_TEDx        | de       | 6.8K    | 10.3 h   | 0.4683      | 0.1037      | 0.090511              | 0.4394%        |
| FLEURS                   | de       | 3.0K    | 9.0 h    | 0.4683      | 0.1037      | 0.087469              | 0.4246%        |
| Multilingual_LibriSpeech | en       | 10.81M  | 44,660 h | 0.4683      | 0.2058      | 0.370077              | 3.5669%        |
| Yodas                    | en       | 7.98M   | 19,447 h | 0.4683      | 0.2058      | 0.300627              | 2.8975%        |
| CommonVoice              | en       | 1.14M   | 1,798 h  | 0.4683      | 0.2058      | 0.165780              | 1.5978%        |
| VoxPopuli                | en       | 182.5K  | 522.6 h  | 0.4683      | 0.2058      | 0.121719              | 1.1732%        |
| FLEURS                   | en       | 2.5K    | 7.3 h    | 0.4683      | 0.2058      | 0.041797              | 0.4029%        |
| Multilingual_LibriSpeech | es       | 220.7K  | 917.7 h  | 0.4683      | 0.1003      | 0.287590              | 1.3505%        |
| CommonVoice              | es       | 353.7K  | 508.7 h  | 0.4683      | 0.1003      | 0.248152              | 1.1653%        |
| Multilingual_TEDx        | es       | 102.2K  | 177.8 h  | 0.4683      | 0.1003      | 0.190799              | 0.8960%        |
| VoxPopuli                | es       | 50.9K   | 151.9 h  | 0.4683      | 0.1003      | 0.183449              | 0.8614%        |
| FLEURS                   | es       | 2.8K    | 8.8 h    | 0.4683      | 0.1003      | 0.090010              | 0.4227%        |
| YouTubeFr                | fr       | 1.53M   | 5,250 h  | 0.4683      | 0.3795      | 0.117507              | 2.0886%        |
| Yodas                    | fr       | 332.7K  | 2,373 h  | 0.4683      | 0.3795      | 0.096351              | 1.7126%        |
| Multilingual_LibriSpeech | fr       | 258.2K  | 1,077 h  | 0.4683      | 0.3795      | 0.079073              | 1.4055%        |
| CommonVoice              | fr       | 593.1K  | 850.8 h  | 0.4683      | 0.3795      | 0.074555              | 1.3252%        |
| ESLO                     | fr       | 454.6K  | 272.4 h  | 0.4683      | 0.3795      | 0.056084              | 0.9969%        |
| VoxPopuli                | fr       | 73.6K   | 205.7 h  | 0.4683      | 0.3795      | 0.052279              | 0.9292%        |
| Multilingual_TEDx        | fr       | 116.0K  | 175.8 h  | 0.4683      | 0.3795      | 0.050268              | 0.8935%        |
| LVL-Atril-CTFAR          | fr       | 58.2K   | 83.0 h   | 0.4683      | 0.3795      | 0.041660              | 0.7405%        |
| PFC                      | fr       | 67.4K   | 62.3 h   | 0.4683      | 0.3795      | 0.038776              | 0.6892%        |
| TCOF_Adultes             | fr       | 79.7K   | 56.2 h   | 0.4683      | 0.3795      | 0.037796              | 0.6718%        |
| TCOF_Enfants             | fr       | 73.1K   | 51.5 h   | 0.4683      | 0.3795      | 0.036987              | 0.6574%        |
| LVL-Atril-CTFNN1         | fr       | 69.8K   | 42.0 h   | 0.4683      | 0.3795      | 0.035144              | 0.6247%        |
| CFPP2000                 | fr       | 30.3K   | 38.3 h   | 0.4683      | 0.3795      | 0.034344              | 0.6105%        |
| VoxForge                 | fr       | 22.4K   | 37.2 h   | 0.4683      | 0.3795      | 0.034088              | 0.6059%        |
| CLAPI                    | fr       | 17.8K   | 21.9 h   | 0.4683      | 0.3795      | 0.029866              | 0.5309%        |
| AfricanAccentedFrench    | fr       | 11.5K   | 13.7 h   | 0.4683      | 0.3795      | 0.026561              | 0.4721%        |
| LINAGORA_Meetings        | fr       | 6.2K    | 10.3 h   | 0.4683      | 0.3795      | 0.024757              | 0.4400%        |
| FLEURS                   | fr       | 3.2K    | 10.3 h   | 0.4683      | 0.3795      | 0.024744              | 0.4398%        |
| LesVocaux                | fr       | 744     | 10.3 h   | 0.4683      | 0.3795      | 0.024708              | 0.4392%        |
| CFPB                     | fr       | 18.4K   | 9.1 h    | 0.4683      | 0.3795      | 0.023975              | 0.4261%        |
| ACSYNT                   | fr       | 3.5K    | 7.9 h    | 0.4683      | 0.3795      | 0.023150              | 0.4115%        |
| PxSLU                    | fr       | 2.0K    | 4.2 h    | 0.4683      | 0.3795      | 0.019775              | 0.3515%        |
| SimSamu                  | fr       | 3.1K    | 2.6 h    | 0.4683      | 0.3795      | 0.017553              | 0.3120%        |
| CommonVoice              | it       | 172.8K  | 254.7 h  | 0.4683      | 0.0829      | 0.252310              | 0.9802%        |
| Multilingual_LibriSpeech | it       | 59.6K   | 247.4 h  | 0.4683      | 0.0829      | 0.250489              | 0.9731%        |
| Multilingual_TEDx        | it       | 50.0K   | 100.7 h  | 0.4683      | 0.0829      | 0.200083              | 0.7773%        |
| VoxPopuli                | it       | 22.6K   | 78.1 h   | 0.4683      | 0.0829      | 0.187745              | 0.7294%        |
| FLEURS                   | it       | 3.0K    | 9.0 h    | 0.4683      | 0.0829      | 0.109372              | 0.4249%        |
| Multilingual_LibriSpeech | nl       | 374.3K  | 1,554 h  | 0.4683      | 0.0695      | 0.473357              | 1.5406%        |
| CommonVoice              | nl       | 43.5K   | 54.1 h   | 0.4683      | 0.0695      | 0.204507              | 0.6656%        |
| VoxPopuli                | nl       | 21.0K   | 46.4 h   | 0.4683      | 0.0695      | 0.196712              | 0.6402%        |
| FLEURS                   | nl       | 2.9K    | 7.7 h    | 0.4683      | 0.0695      | 0.125424              | 0.4082%        |
| Multilingual_LibriSpeech | pt       | 37.5K   | 161.0 h  | 0.4683      | 0.0583      | 0.320072              | 0.8740%        |
| Multilingual_TEDx        | pt       | 90.2K   | 152.8 h  | 0.4683      | 0.0583      | 0.315932              | 0.8627%        |
| CommonVoice              | pt       | 22.9K   | 26.3 h   | 0.4683      | 0.0583      | 0.203528              | 0.5557%        |
| FLEURS                   | pt       | 2.8K    | 10.2 h   | 0.4683      | 0.0583      | 0.160468              | 0.4382%        |

#### AST

| Dataset             | Language | Samples | Duration | Task w (L1) | Lang w (L2) | Dataset w (L3) ← YAML | Effective prob |
| ------------------- | -------- | ------- | -------- | ----------- | ----------- | --------------------- | -------------- |
| CoVoST              | de→en    | 264.1K  | 345.6 h  | 0.2699      | 0.0392      | 1.000000              | 1.0580%        |
| CommonVoiceDE2FR    | de→fr    | 607.5K  | 962.1 h  | 0.2699      | 0.0506      | 1.000000              | 1.3665%        |
| CoVoST              | en→de    | 429.3K  | 587.6 h  | 0.2699      | 0.0448      | 1.000000              | 1.2081%        |
| CommonVoiceEN2FR    | en→fr    | 1.14M   | 1,798 h  | 0.2699      | 0.0592      | 1.000000              | 1.5978%        |
| CoVoST              | es→en    | 91.9K   | 131.4 h  | 0.2699      | 0.0565      | 0.544474              | 0.8307%        |
| Multilingual_TEDx   | es→en    | 36.3K   | 64.4 h   | 0.2699      | 0.0565      | 0.455526              | 0.6950%        |
| CommonVoiceES2FR    | es→fr    | 353.5K  | 508.5 h  | 0.2699      | 0.0576      | 0.749511              | 1.1651%        |
| Multilingual_TEDx   | es→fr    | 3.7K    | 6.3 h    | 0.2699      | 0.0576      | 0.250489              | 0.3894%        |
| Multilingual_TEDx   | es→it    | 5.6K    | 9.9 h    | 0.2699      | 0.0161      | 1.000000              | 0.4355%        |
| Multilingual_TEDx   | es→pt    | 21.1K   | 37.3 h   | 0.2699      | 0.0225      | 1.000000              | 0.6062%        |
| CommonVoiceFR2AR    | fr→ar    | 116.0K  | 166.2 h  | 0.2699      | 0.0326      | 1.000000              | 0.8810%        |
| CommonVoiceFR2DE    | fr→de    | 592.5K  | 850.0 h  | 0.2699      | 0.0491      | 1.000000              | 1.3248%        |
| CommonVoiceFR2EN    | fr→en    | 593.0K  | 850.8 h  | 0.2699      | 0.1110      | 0.442463              | 1.3252%        |
| CoVoST              | fr→en    | 247.2K  | 315.4 h  | 0.2699      | 0.1110      | 0.345253              | 1.0340%        |
| Multilingual_TEDx   | fr→en    | 30.2K   | 45.1 h   | 0.2699      | 0.1110      | 0.212284              | 0.6358%        |
| CommonVoiceFR2ES    | fr→es    | 593.0K  | 850.7 h  | 0.2699      | 0.0708      | 0.693422              | 1.3251%        |
| Multilingual_TEDx   | fr→es    | 20.8K   | 32.5 h   | 0.2699      | 0.0708      | 0.306578              | 0.5859%        |
| CommonVoiceFR2IT    | fr→it    | 593.0K  | 850.7 h  | 0.2699      | 0.0491      | 1.000000              | 1.3251%        |
| CommonVoiceFR2NL    | fr→nl    | 592.7K  | 850.3 h  | 0.2699      | 0.0491      | 1.000000              | 1.3250%        |
| CommonVoiceFR2PT    | fr→pt    | 593.0K  | 850.7 h  | 0.2699      | 0.0683      | 0.718526              | 1.3251%        |
| Multilingual_TEDx   | fr→pt    | 13.3K   | 20.0 h   | 0.2699      | 0.0683      | 0.281474              | 0.5191%        |
| CoVoST              | it→en    | 42.9K   | 60.5 h   | 0.2699      | 0.0492      | 0.514898              | 0.6843%        |
| Multilingual_TEDx   | it→en    | 24.6K   | 47.7 h   | 0.2699      | 0.0492      | 0.485102              | 0.6447%        |
| Multilingual_TEDx   | it→es    | 2.3K    | 4.7 h    | 0.2699      | 0.0134      | 1.000000              | 0.3621%        |
| CommonVoiceIT2FR    | it→fr    | 109.7K  | 161.8 h  | 0.2699      | 0.0324      | 1.000000              | 0.8751%        |
| CommonVoiceNL2FR    | nl→fr    | 43.4K   | 54.1 h   | 0.2699      | 0.0247      | 1.000000              | 0.6656%        |
| Multilingual_TEDx   | pt→en    | 30.9K   | 53.0 h   | 0.2699      | 0.0432      | 0.567483              | 0.6620%        |
| CoVoST              | pt→en    | 14.9K   | 17.9 h   | 0.2699      | 0.0432      | 0.432517              | 0.5046%        |
| Multilingual_TEDx   | pt→es    | 11.5K   | 20.6 h   | 0.2699      | 0.0194      | 1.000000              | 0.5228%        |
| CommonVoicePT2FR_7B | pt→fr    | 22.9K   | 26.3 h   | 0.2699      | 0.0412      | 0.500016              | 0.5557%        |
| CommonVoicePT2FR    | pt→fr    | 22.9K   | 26.3 h   | 0.2699      | 0.0412      | 0.499984              | 0.5557%        |

#### OTHER

| Dataset                                      | Language | Samples | Duration | Task w (L1) | Lang w (L2) | Dataset w (L3) ← YAML | Effective prob |
| -------------------------------------------- | -------- | ------- | -------- | ----------- | ----------- | --------------------- | -------------- |
| CommonVoice_Age-Gender_reco                  | en       | 667.2K  | 1,036 h  | 0.0670      | 0.5212      | 0.398759              | 1.3921%        |
| VoxLingua_lang_identification                | en       | 13.5K   | 36.3 h   | 0.0670      | 0.5212      | 0.172509              | 0.6022%        |
| EmotionDS                                    | en       | 10.0K   | 8.7 h    | 0.0670      | 0.5212      | 0.120773              | 0.4216%        |
| ssi-speech-emotion-recognition               | en       | 9.6K    | 6.9 h    | 0.0670      | 0.5212      | 0.113744              | 0.3971%        |
| ssd                                          | en       | 4.4K    | 3.6 h    | 0.0670      | 0.5212      | 0.097108              | 0.3390%        |
| ssr                                          | en       | 4.4K    | 3.6 h    | 0.0670      | 0.5212      | 0.097108              | 0.3390%        |
| CommonVoice_Age-Gender_reco                  | fr       | 288.1K  | 407.7 h  | 0.0670      | 0.4788      | 0.343799              | 1.1026%        |
| VoxLingua_lang_identification                | fr       | 13.5K   | 36.5 h   | 0.0670      | 0.4788      | 0.188098              | 0.6032%        |
| EmotionDS                                    | fr       | 10.0K   | 8.7 h    | 0.0670      | 0.4788      | 0.131467              | 0.4216%        |
| ssi-speech-emotion-recognition               | fr       | 9.6K    | 6.9 h    | 0.0670      | 0.4788      | 0.123816              | 0.3971%        |
| ssd                                          | fr       | 4.4K    | 3.6 h    | 0.0670      | 0.4788      | 0.105706              | 0.3390%        |
| OreauFR_02                                   | fr       | 434     | 0.3 h    | 0.0670      | 0.4788      | 0.058211              | 0.1867%        |
| emotional-speech-audio-dataset-4languages-fr | fr       | 144     | 0.2 h    | 0.0670      | 0.4788      | 0.048902              | 0.1568%        |

#### QA

| Dataset                                       | Language | Samples | Duration | Task w (L1) | Lang w (L2) | Dataset w (L3) ← YAML | Effective prob |
| --------------------------------------------- | -------- | ------- | -------- | ----------- | ----------- | --------------------- | -------------- |
| VoxPopuli-QA                                  | en       | 568.0K  | 4,540 h  | 0.1159      | 0.6963      | 0.249455              | 2.0141%        |
| Menlo--instruction-speech-encodec-v1.5        | en       | 332.4K  | 678.2 h  | 0.1159      | 0.6963      | 0.155085              | 1.2521%        |
| slu-phase-2-sqa5                              | en       | 46.2K   | 511.2 h  | 0.1159      | 0.6963      | 0.144507              | 1.1667%        |
| worstchan--UltraChat-300K-SLAM-Omni           | en       | 164.5K  | 342.1 h  | 0.1159      | 0.6963      | 0.130697              | 1.0552%        |
| gruhit-patel--alpaca_speech_instruct          | en       | 51.8K   | 72.1 h   | 0.1159      | 0.6963      | 0.088543              | 0.7149%        |
| yijingwu--HeySQuAD_human                      | en       | 10.8K   | 18.0 h   | 0.1159      | 0.6963      | 0.062636              | 0.5057%        |
| liangtianle--unsafety-question                | en       | 3.4K    | 15.9 h   | 0.1159      | 0.6963      | 0.060704              | 0.4901%        |
| amuvarma--10k-filtered-tune-audio-text_answer | en       | 9.0K    | 13.0 h   | 0.1159      | 0.6963      | 0.057711              | 0.4660%        |
| liangtianle--science-question                 | en       | 3.5K    | 7.7 h    | 0.1159      | 0.6963      | 0.050661              | 0.4090%        |
| VoxPopuli-QA                                  | fr       | 568.9K  | 4,538 h  | 0.1159      | 0.3037      | 0.571958              | 2.0139%        |
| Vigogne--Alpaca                               | fr       | 49.7K   | 79.7 h   | 0.1159      | 0.3037      | 0.208236              | 0.7332%        |
| ComparIA                                      | fr       | 12.9K   | 21.5 h   | 0.1159      | 0.3037      | 0.150053              | 0.5283%        |
| CohereLabs--aya_collection                    | fr       | 723     | 1.0 h    | 0.1159      | 0.3037      | 0.069754              | 0.2456%        |

---

### YAML skeleton — full three-level hierarchy

> Copy-paste skeleton. Replace `<path>` with actual manifest paths.

```yaml
input_cfg:
  - type: group
    weight: 0.078829   # L1 — AQA
    tags: {task: aqa}
    input_cfg:
      - type: group
        weight: 0.864163   # L2 — en within AQA
        tags: {task: aqa, lang: en}
        input_cfg:
          - type: multimodal_conversation
            weight: 0.345736   # L3 — CLEAR_v1.0.0_improved
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.151446   # L3 — CLEAR_v2.0.0_improved
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.147303   # L3 — mispeech_MECAT-QA
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.098252   # L3 — mispeech_MECAT-Caption
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.088462   # L3 — clotho_aqa_improved
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.058937   # L3 — MusicCaps
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.058937   # L3 — MusicCaps
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.050926   # L3 — AcousticDS
            manifest_filepath: <path>
      - type: group
        weight: 0.135837   # L2 — fr within AQA
        tags: {task: aqa, lang: fr}
        input_cfg:
          - type: multimodal_conversation
            weight: 0.625055   # L3 — mispeech_MECAT-Caption2FR
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.374945   # L3 — MusicCaps
            manifest_filepath: <path>

  - type: group
    weight: 0.468331   # L1 — ASR
    tags: {task: asr}
    input_cfg:
      - type: group
        weight: 0.379530   # L2 — fr within ASR
        tags: {task: asr, lang: fr}
        input_cfg:
          - type: multimodal_conversation
            weight: 0.117507   # L3 — YouTubeFr
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.096351   # L3 — Yodas
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.079073   # L3 — Multilingual_LibriSpeech
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.074555   # L3 — CommonVoice
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.056084   # L3 — ESLO
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.052279   # L3 — VoxPopuli
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.050268   # L3 — Multilingual_TEDx
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.041660   # L3 — LVL-Atril-CTFAR
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.038776   # L3 — PFC
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.037796   # L3 — TCOF_Adultes
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.036987   # L3 — TCOF_Enfants
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.035144   # L3 — LVL-Atril-CTFNN1
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.034344   # L3 — CFPP2000
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.034088   # L3 — VoxForge
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.029866   # L3 — CLAPI
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.026561   # L3 — AfricanAccentedFrench
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.024757   # L3 — LINAGORA_Meetings
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.024744   # L3 — FLEURS
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.024708   # L3 — LesVocaux
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.023975   # L3 — CFPB
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.023150   # L3 — ACSYNT
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.019775   # L3 — PxSLU
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.017553   # L3 — SimSamu
            manifest_filepath: <path>
      - type: group
        weight: 0.205802   # L2 — en within ASR
        tags: {task: asr, lang: en}
        input_cfg:
          - type: multimodal_conversation
            weight: 0.370077   # L3 — Multilingual_LibriSpeech
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.300627   # L3 — Yodas
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.165780   # L3 — CommonVoice
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.121719   # L3 — VoxPopuli
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.041797   # L3 — FLEURS
            manifest_filepath: <path>
      - type: group
        weight: 0.103652   # L2 — de within ASR
        tags: {task: asr, lang: de}
        input_cfg:
          - type: multimodal_conversation
            weight: 0.336596   # L3 — Multilingual_LibriSpeech
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.281553   # L3 — CommonVoice
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.203871   # L3 — VoxPopuli
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.090511   # L3 — Multilingual_TEDx
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.087469   # L3 — FLEURS
            manifest_filepath: <path>
      - type: group
        weight: 0.100268   # L2 — es within ASR
        tags: {task: asr, lang: es}
        input_cfg:
          - type: multimodal_conversation
            weight: 0.287590   # L3 — Multilingual_LibriSpeech
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.248152   # L3 — CommonVoice
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.190799   # L3 — Multilingual_TEDx
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.183449   # L3 — VoxPopuli
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.090010   # L3 — FLEURS
            manifest_filepath: <path>
      - type: group
        weight: 0.082950   # L2 — it within ASR
        tags: {task: asr, lang: it}
        input_cfg:
          - type: multimodal_conversation
            weight: 0.252310   # L3 — CommonVoice
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.250489   # L3 — Multilingual_LibriSpeech
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.200083   # L3 — Multilingual_TEDx
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.187745   # L3 — VoxPopuli
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.109372   # L3 — FLEURS
            manifest_filepath: <path>
      - type: group
        weight: 0.069495   # L2 — nl within ASR
        tags: {task: asr, lang: nl}
        input_cfg:
          - type: multimodal_conversation
            weight: 0.473357   # L3 — Multilingual_LibriSpeech
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.204507   # L3 — CommonVoice
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.196712   # L3 — VoxPopuli
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.125424   # L3 — FLEURS
            manifest_filepath: <path>
      - type: group
        weight: 0.058304   # L2 — pt within ASR
        tags: {task: asr, lang: pt}
        input_cfg:
          - type: multimodal_conversation
            weight: 0.320072   # L3 — Multilingual_LibriSpeech
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.315932   # L3 — Multilingual_TEDx
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.203528   # L3 — CommonVoice
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.160468   # L3 — FLEURS
            manifest_filepath: <path>

  - type: group
    weight: 0.269911   # L1 — AST
    tags: {task: ast}
    input_cfg:
      - type: group
        weight: 0.110962   # L2 — fr→en within AST
        tags: {task: ast, lang: fr→en}
        input_cfg:
          - type: multimodal_conversation
            weight: 0.442463   # L3 — CommonVoiceFR2EN
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.345253   # L3 — CoVoST
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.212284   # L3 — Multilingual_TEDx
            manifest_filepath: <path>
      - type: group
        weight: 0.070801   # L2 — fr→es within AST
        tags: {task: ast, lang: fr→es}
        input_cfg:
          - type: multimodal_conversation
            weight: 0.693422   # L3 — CommonVoiceFR2ES
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.306578   # L3 — Multilingual_TEDx
            manifest_filepath: <path>
      - type: group
        weight: 0.068327   # L2 — fr→pt within AST
        tags: {task: ast, lang: fr→pt}
        input_cfg:
          - type: multimodal_conversation
            weight: 0.718526   # L3 — CommonVoiceFR2PT
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.281474   # L3 — Multilingual_TEDx
            manifest_filepath: <path>
      - type: group
        weight: 0.059199   # L2 — en→fr within AST
        tags: {task: ast, lang: en→fr}
        input_cfg:
          - type: multimodal_conversation
            weight: 1.000000   # L3 — CommonVoiceEN2FR
            manifest_filepath: <path>
      - type: group
        weight: 0.057594   # L2 — es→fr within AST
        tags: {task: ast, lang: es→fr}
        input_cfg:
          - type: multimodal_conversation
            weight: 0.749511   # L3 — CommonVoiceES2FR
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.250489   # L3 — Multilingual_TEDx
            manifest_filepath: <path>
      - type: group
        weight: 0.056527   # L2 — es→en within AST
        tags: {task: ast, lang: es→en}
        input_cfg:
          - type: multimodal_conversation
            weight: 0.544474   # L3 — CoVoST
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.455526   # L3 — Multilingual_TEDx
            manifest_filepath: <path>
      - type: group
        weight: 0.050630   # L2 — de→fr within AST
        tags: {task: ast, lang: de→fr}
        input_cfg:
          - type: multimodal_conversation
            weight: 1.000000   # L3 — CommonVoiceDE2FR
            manifest_filepath: <path>
      - type: group
        weight: 0.049240   # L2 — it→en within AST
        tags: {task: ast, lang: it→en}
        input_cfg:
          - type: multimodal_conversation
            weight: 0.514898   # L3 — CoVoST
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.485102   # L3 — Multilingual_TEDx
            manifest_filepath: <path>
      - type: group
        weight: 0.049095   # L2 — fr→it within AST
        tags: {task: ast, lang: fr→it}
        input_cfg:
          - type: multimodal_conversation
            weight: 1.000000   # L3 — CommonVoiceFR2IT
            manifest_filepath: <path>
      - type: group
        weight: 0.049089   # L2 — fr→nl within AST
        tags: {task: ast, lang: fr→nl}
        input_cfg:
          - type: multimodal_conversation
            weight: 1.000000   # L3 — CommonVoiceFR2NL
            manifest_filepath: <path>
      - type: group
        weight: 0.049085   # L2 — fr→de within AST
        tags: {task: ast, lang: fr→de}
        input_cfg:
          - type: multimodal_conversation
            weight: 1.000000   # L3 — CommonVoiceFR2DE
            manifest_filepath: <path>
      - type: group
        weight: 0.044758   # L2 — en→de within AST
        tags: {task: ast, lang: en→de}
        input_cfg:
          - type: multimodal_conversation
            weight: 1.000000   # L3 — CoVoST
            manifest_filepath: <path>
      - type: group
        weight: 0.043220   # L2 — pt→en within AST
        tags: {task: ast, lang: pt→en}
        input_cfg:
          - type: multimodal_conversation
            weight: 0.567483   # L3 — Multilingual_TEDx
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.432517   # L3 — CoVoST
            manifest_filepath: <path>
      - type: group
        weight: 0.041178   # L2 — pt→fr within AST
        tags: {task: ast, lang: pt→fr}
        input_cfg:
          - type: multimodal_conversation
            weight: 0.500016   # L3 — CommonVoicePT2FR_7B
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.499984   # L3 — CommonVoicePT2FR
            manifest_filepath: <path>
      - type: group
        weight: 0.039197   # L2 — de→en within AST
        tags: {task: ast, lang: de→en}
        input_cfg:
          - type: multimodal_conversation
            weight: 1.000000   # L3 — CoVoST
            manifest_filepath: <path>
      - type: group
        weight: 0.032641   # L2 — fr→ar within AST
        tags: {task: ast, lang: fr→ar}
        input_cfg:
          - type: multimodal_conversation
            weight: 1.000000   # L3 — CommonVoiceFR2AR
            manifest_filepath: <path>
      - type: group
        weight: 0.032422   # L2 — it→fr within AST
        tags: {task: ast, lang: it→fr}
        input_cfg:
          - type: multimodal_conversation
            weight: 1.000000   # L3 — CommonVoiceIT2FR
            manifest_filepath: <path>
      - type: group
        weight: 0.024659   # L2 — nl→fr within AST
        tags: {task: ast, lang: nl→fr}
        input_cfg:
          - type: multimodal_conversation
            weight: 1.000000   # L3 — CommonVoiceNL2FR
            manifest_filepath: <path>
      - type: group
        weight: 0.022459   # L2 — es→pt within AST
        tags: {task: ast, lang: es→pt}
        input_cfg:
          - type: multimodal_conversation
            weight: 1.000000   # L3 — Multilingual_TEDx
            manifest_filepath: <path>
      - type: group
        weight: 0.019368   # L2 — pt→es within AST
        tags: {task: ast, lang: pt→es}
        input_cfg:
          - type: multimodal_conversation
            weight: 1.000000   # L3 — Multilingual_TEDx
            manifest_filepath: <path>
      - type: group
        weight: 0.016133   # L2 — es→it within AST
        tags: {task: ast, lang: es→it}
        input_cfg:
          - type: multimodal_conversation
            weight: 1.000000   # L3 — Multilingual_TEDx
            manifest_filepath: <path>
      - type: group
        weight: 0.013417   # L2 — it→es within AST
        tags: {task: ast, lang: it→es}
        input_cfg:
          - type: multimodal_conversation
            weight: 1.000000   # L3 — Multilingual_TEDx
            manifest_filepath: <path>

  - type: group
    weight: 0.066980   # L1 — OTHER
    tags: {task: other}
    input_cfg:
      - type: group
        weight: 0.521199   # L2 — en within OTHER
        tags: {task: other, lang: en}
        input_cfg:
          - type: multimodal_conversation
            weight: 0.398759   # L3 — CommonVoice_Age-Gender_reco
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.172509   # L3 — VoxLingua_lang_identification
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.120773   # L3 — EmotionDS
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.113744   # L3 — ssi-speech-emotion-recognition
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.097108   # L3 — ssd
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.097108   # L3 — ssr
            manifest_filepath: <path>
      - type: group
        weight: 0.478801   # L2 — fr within OTHER
        tags: {task: other, lang: fr}
        input_cfg:
          - type: multimodal_conversation
            weight: 0.343799   # L3 — CommonVoice_Age-Gender_reco
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.188098   # L3 — VoxLingua_lang_identification
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.131467   # L3 — EmotionDS
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.123816   # L3 — ssi-speech-emotion-recognition
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.105706   # L3 — ssd
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.058211   # L3 — OreauFR_02
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.048902   # L3 — emotional-speech-audio-dataset-4languages-fr
            manifest_filepath: <path>

  - type: group
    weight: 0.115949   # L1 — QA
    tags: {task: qa}
    input_cfg:
      - type: group
        weight: 0.696333   # L2 — en within QA
        tags: {task: qa, lang: en}
        input_cfg:
          - type: multimodal_conversation
            weight: 0.249455   # L3 — VoxPopuli-QA
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.155085   # L3 — Menlo--instruction-speech-encodec-v1.5
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.144507   # L3 — slu-phase-2-sqa5
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.130697   # L3 — worstchan--UltraChat-300K-SLAM-Omni
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.088543   # L3 — gruhit-patel--alpaca_speech_instruct
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.062636   # L3 — yijingwu--HeySQuAD_human
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.060704   # L3 — liangtianle--unsafety-question
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.057711   # L3 — amuvarma--10k-filtered-tune-audio-text_answer
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.050661   # L3 — liangtianle--science-question
            manifest_filepath: <path>
      - type: group
        weight: 0.303667   # L2 — fr within QA
        tags: {task: qa, lang: fr}
        input_cfg:
          - type: multimodal_conversation
            weight: 0.571958   # L3 — VoxPopuli-QA
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.208236   # L3 — Vigogne--Alpaca
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.150053   # L3 — ComparIA
            manifest_filepath: <path>
          - type: multimodal_conversation
            weight: 0.069754   # L3 — CohereLabs--aya_collection
            manifest_filepath: <path>

```

---

## 9. Suggested `max_steps` for Training

> Estimates assume a **weighted sampler** (no hard epochs). `steps_per_epoch` = ⌈total_samples / effective_batch_size⌉.

| Parameter             | Value |
|----------------------|------|
| Total training samples | **37,763,479** |
| Batch size (`batch_size`) | **4** |
| Gradient accumulation (`grad_accum`) | **1** |
| Effective batch size   | **4** |
| Steps per epoch        | **9,440,870** |
| Target epochs          | **3.0** |
| **Suggested max_steps** | **28,322,610** |

---

### Presets

| Preset       | Epochs | max_steps  | Note               |
| ------------ | ------ | ---------- | ------------------ |
| Conservative | 0.5    | 4,720,435  |                    |
| Recommended  | 1.0    | 9,440,870  |  ← **recommended** |
| Extended     | 3.0    | 28,322,610 |                    |
| Aggressive   | 5.0    | 47,204,350 |                    |

### Per-task breakdown

> How many steps are "effectively spent" on each task at the recommended epoch count (based on raw sample counts, not weights).

| Task      | Samples    | % of total | Steps @ 1 epoch |
| --------- | ---------- | ---------- | --------------- |
| asr       | 26.77M     | 70.9%      | 6,693,421       |
| ast       | 7.26M      | 19.2%      | 1,815,598       |
| qa        | 1.82M      | 4.8%       | 455,405         |
| other     | 1.04M      | 2.7%       | 258,820         |
| aqa       | 870.5K     | 2.3%       | 217,628         |
| **TOTAL** | **37.76M** | **100.0%** | **9,440,870**   |

> **Tip**: When using a weighted sampler, the actual "effective epochs" per dataset will differ from the raw counts above — see Section 6.5 (Est. passes/epoch) for the per-dataset breakdown.

---

## 10. Balance Notes

- ⚠️  Task **asr** dominates with **70.9%** of all samples.
- 🟡  Language **en**: **60.5%** of total samples.
- ✅  Language **fr**: **12.7%** of total samples.
- 📌  Top-3 languages (en, fr, de) account for **76.4%** of samples.
- ⚠️  Low-coverage (1 manifest only): en→fr, de→fr, fr→it, fr→nl, fr→de, en→de, de→en, fr→ar, it→fr, nl→fr, es→pt, pt→es, es→it, it→es.

---

_Report generated by `dataset_analysis.py`_