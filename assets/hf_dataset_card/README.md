---
pretty_name: Luciole Audio Training Dataset
license: cc-by-nc-sa-4.0
multilinguality:
- multilingual
task_categories:
- automatic-speech-recognition
- translation
- question-answering
- audio-classification
- audio-to-audio
tags:
- audio
- speech
- music
- sound
- audio-language-model
- speech-llm
- conversational
---

<!-- TEMPLATE-NOTE: this file is a template; assets/export_hf_dataset.py injects configs/language/size_categories into the header and fills the AUTOGEN blocks with the dataset/license tables. Edit the prose here, not the split list. This note is stripped from the published README. -->

# Luciole Audio Training Dataset

## Table of Contents

- [Dataset description](#dataset-description)
  - [Dataset structure](#dataset-structure)
  - [Data fields](#data-fields)
  - [Tasks](#tasks)
- [Getting the audio](#getting-the-audio)
- [Loading](#loading)
- [Datasets and licenses](#datasets-and-licenses)
- [Audio availability](#audio-availability)
  - [Text and audio](#text-and-audio)
  - [Text only](#text-only)
  - [Not included](#not-included)
- [Licensing](#licensing)
- [Attribution](#attribution)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Dataset description

**Luciole Audio Training Dataset** is a large, multilingual, multi-task collection of
**audio–text conversations** used to train the [OpenLLM-France](https://huggingface.co/OpenLLM-France)
*Luciole* audio-language models. It adapts a text LLM to understand audio by pairing speech,
music and environmental sounds with instruction-style dialogues (transcription, translation,
spoken question answering, audio/music/sound captioning and question answering, speaker and
language attributes, diarization, and more).

The repository holds **only the text and metadata** (the conversations and their references to
audio files). **The audio itself is not hosted on the Hub** — see
[Getting the audio](#getting-the-audio).

> ⚠️ **Licensing / redistribution.** This dataset aggregates **many** source datasets under
> **different licenses**. See [Datasets and licenses](#datasets-and-licenses) below: every split maps
> to exactly one source dataset, with its license and where its audio can be obtained. Datasets
> whose license forbids redistribution are **not included** here and must be obtained from their
> original source.

### Dataset structure

```
Luciole-Audio-Training-Dataset/
├── README.md
├── dataset_stats.csv        # per-split #examples / #audio / hours / license / audio_hosted
├── dataset_index.json       # split → source dataset, license, provenance
└── data/                    # the conversations (this is what lives on the Hub)
    ├── speech/  (asr, ast, qa, summarization, diarization, temporal, emotion_reco,
    │             gender_reco, age_reco, language_reco, voice_captioning, …)
    ├── music/   (qa, captioning)
    └── sound/   (qa, captioning)
```

The two top levels are **`<domain>/<task>`** (e.g. `speech/asr`, `music/qa`, `sound/qa`).
Each `<domain>/<task>` is exposed as a **configuration** **`<domain>.<task>`**, and each
**source dataset (per language)** is a **split** within it. For example, English ASR from
Common Voice is the `CommonVoice_en` split of the `speech.asr` config.

### Data fields

Every record, in every split, has exactly these four fields:

| field | type | description |
|-------|------|-------------|
| `id` | string | unique example id. |
| `conversations` | list | the dialogue turns (see below). |
| `language` | string | `"fr"`, `"en"`, … ; `"multilingual"` when a record mixes languages; for translation, source-target like `"en-fr"` (English → French). |
| `extra` | string | a JSON-serialized object with any extra source metadata; `dataset_name` is present for most records. Example: `{"dataset_name": "fleurs"}`. `"{}"` when there is none. |

Each turn in `conversations` has:

| field | type | description |
|-------|------|-------------|
| `from` | string | `"User"` or `"Assistant"` |
| `type` | string | `"text"` or `"audio"` |
| `value` | string | the text, **or** a **relative path** to an audio file (for `type: "audio"`) |
| `duration` | float | audio length in seconds (audio turns only; `null` otherwise) |
| `offset` | float | start offset in the audio file, in seconds (when the turn uses a segment; `null` otherwise) |

Audio is referenced by a **relative path** of the form
`audio/<domain>/<task>/<Split>/<file>`, e.g.
`audio/speech/asr/FLEURS_fr/16207707140941618664.wav`.

Example record:

```json
{
  "id": "735",
  "conversations": [
    {"from": "User", "value": "Faites une transcription complète du fichier audio.", "type": "text"},
    {"from": "User", "value": "audio/speech/asr/FLEURS_fr/16207707140941618664.wav", "type": "audio", "duration": 8.88},
    {"from": "Assistant", "value": "Les permis doivent être réservés à l'avance. …", "type": "text"}
  ],
  "language": "fr",
  "extra": "{\"dataset_name\": \"fleurs\"}"
}
```

### Tasks

The dataset covers the following tasks, organised by domain as `<domain>/<task>`:

**Speech** (`speech/…`)

- **`asr`** — *Automatic Speech Recognition*: transcribe spoken audio into text.
- **`ast`** — *Automatic Speech Translation*: translate spoken audio into text in another language (the split's `language` is the source-target pair, e.g. `en-fr` for the translation from English to French).
- **`qa`** — *Spoken Question Answering*: answer questions about spoken audio.
- **`summarization`** — *Summarization*: produce a summary of a spoken talk or meeting (e.g. AMI/ICSI meetings, TED talks).
- **`diarization`** — *Speaker Diarization*: segment the audio by speaker ("who spoke when"), with timestamps.
- **`temporal`** — *Temporal Localization*: locate words/sentences in time (word↔time, time↔sentence) and answer questions together with the timestamp of the answer.
- **`emotion_reco`** — *Emotion Recognition*: identify the emotion conveyed by the speaker.
- **`gender_reco`** — *Gender Recognition*: identify the speaker's (self-reported / perceived) gender.
- **`age_reco`** — *Age Recognition*: estimate the speaker's age range.
- **`language_reco`** — *Spoken Language Identification*: identify which language is being spoken.
- **`voice_captioning`** — *Voice Captioning*: describe the speaker's voice and acoustic characteristics (pitch, tone, accent, recording environment…).
- **`sentence_stress_detection`** — *Sentence-Stress Detection*: transcribe the audio and mark which words are stressed / emphasized.
- **`sentence_stress_reasoning`** — *Sentence-Stress Reasoning*: reason about how the stressed words change the meaning or intent of the utterance.
- **`task_switching`** — *Multi-turn, Multi-task Conversations*: dialogues that combine several of the above tasks (and several audio clips) within a single conversation.

**Music** (`music/…`)

- **`qa`** — *Music Question Answering*: answer questions about a music clip (genre, instruments, mood, tempo…).
- **`captioning`** — *Music Captioning*: produce a descriptive caption of a music clip.

**Sound** (`sound/…`)

- **`qa`** — *Environmental-sound Question Answering*: answer questions about non-speech / everyday sounds.
- **`captioning`** — *Environmental-sound Captioning*: describe non-speech / ambient sounds.

### Loading

Because each `<domain>/<task>` is a separate config, load a config and pick a split:

```python
from datasets import load_dataset

asr = load_dataset("OpenLLM-France/Luciole-Audio-Training-Dataset", "speech.asr")
fleurs_fr = load_dataset("OpenLLM-France/Luciole-Audio-Training-Dataset",
                         "speech.asr", split="FLEURS_fr")
```

See [`dataset_stats.csv`](dataset_stats.csv) for the size of every split and
[`dataset_index.json`](dataset_index.json) for the exact source dataset/license behind each one.

## Getting the audio

The audio is **not** part of this Hub repository. Most of it is served separately at:

> **[https://dl.labs.linagora.com/files/datasets/OpenLLM-France/Luciole-Audio-Training-Dataset/audio/](https://dl.labs.linagora.com/files/datasets/OpenLLM-France/Luciole-Audio-Training-Dataset/audio/)**

Download that `audio/` folder and place it next to `data/`; its layout mirrors `data/`
(`audio/<domain>/<task>/<Split>/…`), so each conversation's audio path resolves directly.
The folder can be downloaded in zip format (tar, targz and tarbz2 also supported) with the following command:
```bash
TOKEN=$(curl -s https://dl.labs.linagora.com/api/login \
  -H "Content-Type: application/json" -d '{}')

curl -H "X-Auth: $TOKEN" \
  "https://dl.labs.linagora.com/api/raw/datasets/OpenLLM-France/Luciole-Audio-Training-Dataset/audio/?algo=zip" \
  -o <output-name>.zip
```


**Some datasets' audio is not redistributable by us** (copyrighted source, YouTube-only, etc.).
For those, the conversations are still published but you must download the audio from the
original source listed below. See [Audio availability](#audio-availability).


## Datasets and licenses

This dataset is an **aggregation of third-party datasets**, each under its own license. The
table below lists every source dataset used, its license, what is published here, where to get
the audio, and its content (total samples and audio duration, broken down per task and language).

<!-- AUTOGEN:DATASET_TABLE -->

## Audio availability

Where to obtain the audio for each source dataset, grouped by what is published in this repository.

<!-- AUTOGEN:AUDIO_AVAILABILITY -->

## Licensing

**No single license covers the whole collection** — it is a *collection* in which each source
dataset retains its own license (see the table above). The curator's added layer (the
instruction wrappers, curation and formatting) and the collection as a whole are offered under
**CC-BY-NC-SA-4.0**, which is the most permissive single license consistent with the
NonCommercial + ShareAlike terms of the included sources:

- **NonCommercial (NC):** several sources are NC, so the combined dataset is non-commercial.
- **ShareAlike (SA):** several sources are NC-SA, so adaptations must stay under the same license.
- **Attribution (BY):** attribute the original datasets (this table) when you use it.

A few sources are **CC-BY-SA** (without NC); they are kept as **separate splits** (a collection,
not a remix) so their ShareAlike terms are respected without relicensing.

**NoDerivatives (ND) datasets** (e.g. Multilingual TEDx, SLUE-TED) may not be redistributed in
modified form. Our pipeline re-segments and reformats audio into conversations, which is a
derivative, so these are **excluded** — download them from their original source (see the table).

When in doubt, comply with the **most restrictive** applicable terms for the subset you use.
If you are a rights holder and believe something should not be included, please open an issue.

## Attribution

This dataset builds on the work of many others. Several sources are under **attribution
(CC-BY*)** licenses and **require credit**. Please attribute every source dataset you use
(the table above lists all of them with their license and origin); the attribution-required
ones are highlighted here:

<!-- AUTOGEN:ATTRIBUTION -->

## Citation

If you use this dataset, please cite the OpenLLM-France / Luciole project **and** the individual
source datasets you rely on (see the table above for sources).

## Acknowledgements

Built by [LINAGORA](https://linagora.com) / [OpenLLM-France](https://huggingface.co/OpenLLM-France).
This collection would not exist without the authors of the many source datasets it builds upon.
