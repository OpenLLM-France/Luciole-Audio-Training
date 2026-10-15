# Luciole-Audio-Training

Training and export scripts for **Luciole Audio**: speech/audio-augmented LLMs (SALM)
built on the Luciole models (1B, 8B, 23B) with NVIDIA NeMo `speechlm2`.

## Repository layout

| Folder      | Content |
|-------------|---------|
| `data/`     | Dataset conversion to NeMo/Lhotse manifests, data generation, templating, TTS |
| `training/` | SALM training (`salm_train.py`), Hydra configs (`conf/`), Slurm launcher |
| `export/`   | Export of NeMo checkpoints to Hugging Face format (`to_hf.py`) |
| `utils/`    | Shared helpers (audio, inference, metrics) |
| `assets/`   | Model/dataset cards, plotting and upload scripts |
| `demo/`     | WebRTC demo app |
| `install/`  | Environment installation scripts (Jean-Zay) |

## Installation

**Data preparation** scripts (`data/`) import `luciole_audio.*`, so the repo must be installed:

```bash
pip install -e ".[data]"   # or [generation], [processing], [templating], [all-data], ...
```

The `tts` extra conflicts with the others (different `transformers` pin) and must be
installed in a separate environment.

## Usage

Training (submits `training/run_train.slurm`):

```bash
python training/slurm_launcher.py --config run/luciole-8b --gpus 4 --time 20:00:00
```

Hugging Face export:

```bash
sbatch export/run_hf_export.slurm
```

## Evaluation

Models are evaluated with [our fork of AudioBench](https://github.com/OpenLLM-France/AudioBench)
(results on the [leaderboard](https://openllm-france.github.io/AudioBench/)).

## License

AGPL v3, see [LICENSE](LICENSE).
