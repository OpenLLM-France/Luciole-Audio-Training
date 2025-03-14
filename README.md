# Audio-Adapter-Training

## Overview

This project is focused on training a multimodal model to adapt LLM with audio. The goal is to develop a robust and efficient audio system for Lucie.

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Configs:
Before starting your training, you have to update the `configs.py` file to set up your training parameters (e.g., base model, output directory, and other important settings).

### Training:
To train the model, use the following command. But first, you should change the dataset needed for training and development:

```bash
python main.py --audio_path path/to/audio/train path/to/audio/train --dev path/to/dev.{json, tsv} --train path/to/train.{json, tsv} --gpus 1
```

## Inference:
To infer the model, you just need to enter the path to the model (the script will take the last checkpoint trained) and an audio file, like in the following command:

```bash
python infer.py path/to/model path/to/audio --gpus 1
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

