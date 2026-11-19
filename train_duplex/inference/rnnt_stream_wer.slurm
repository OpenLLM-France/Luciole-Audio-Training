#!/bin/bash
#SBATCH --job-name=rnnt_stream_wer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --account=wuh@h100
#SBATCH --constraint=h100
#SBATCH --hint=nomultithread
#SBATCH --output=/lustre/fsn1/projects/rech/qgz/%u/speechlm/duplex_diag/%x_%j.out
# WER of the Nemotron RNNT in true streaming (rnnt_stream_wer.py), on the examples of an asr_teacher_forcing.slurm run.
#   cd train_duplex/inference
#   sbatch rnnt_stream_wer.slurm <xp> <step> ["datasets"]
# Outputs in $SCRATCH/speechlm/duplex_diag/<xp>_<step>_rnnt_stream/.
set -euo pipefail
XP="$1"
STEP="$2"
DATASETS="${3:-asr_fr asr_en}"
EXP_DIR="$ALL_CCFRSCRATCH/audio/training/duplex/$XP"
PREVIOUS="$SCRATCH/speechlm/duplex_diag/${XP}_${STEP}"

module purge
module load arch/h100 nemo/2.7.3
conda activate "$SCRATCH/speechlm/envs/nemo27-salm_automodel_3"
NEMO_DIR="$HOME/NeMo_Duplex"
export PYTHONPATH="$NEMO_DIR" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NEMO_MODELS_CACHE="$SCRATCH/.cache" TMPDIR="${JOBSCRATCH:-/tmp}"
# $HOME/NeMo (installed editable) serves a running training: never use it here.
nemo_file="$(python -c "import nemo; print(nemo.__file__)")"
[[ "$nemo_file" == "$NEMO_DIR"/* ]] || { echo "❌ nemo imported from $nemo_file, not $NEMO_DIR" >&2; exit 1; }
HERE="${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f "$HERE/rnnt_stream_wer.py" ]] || { echo "❌ submit from train_duplex/inference" >&2; exit 1; }

python "$HERE/rnnt_stream_wer.py" --exp-dir "$EXP_DIR" --ckpt "$EXP_DIR/checkpoints/$STEP.ckpt" --previous "$PREVIOUS" \
    --out "${PREVIOUS}_rnnt_stream" --datasets $DATASETS
