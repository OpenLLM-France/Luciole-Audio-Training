#!/bin/bash
#SBATCH --job-name=timestamps_vs_nemotron
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --account=wuh@h100
#SBATCH --constraint=h100
#SBATCH --hint=nomultithread
#SBATCH --output=/lustre/fsn1/projects/rech/qgz/%u/speechlm/duplex_diag/%x_%j.out
# user_words timestamps against the streaming Nemotron RNNT emission frames (timestamps_vs_nemotron.py), on the
# CommonVoice dev sets of the duplex validation.
#   cd training_duplex/inference
#   mkdir -p $SCRATCH/speechlm/duplex_diag && sbatch timestamps_vs_nemotron.slurm [lines]
# Outputs in $SCRATCH/speechlm/duplex_diag/timestamps_vs_nemotron/.
set -euo pipefail
LINES="${1:-1000}"
DUPLEX="$ALL_CCFRSCRATCH/audio/duplex/nemo/asr"
OUT="$SCRATCH/speechlm/duplex_diag/timestamps_vs_nemotron"

module purge
module load arch/h100 nemo/2.7.3
conda activate "$SCRATCH/speechlm/envs/nemo27-salm_automodel_3"
NEMO_DIR="$HOME/NeMo_Duplex"
export PYTHONPATH="$NEMO_DIR" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NEMO_MODELS_CACHE="$SCRATCH/.cache" TMPDIR="${JOBSCRATCH:-/tmp}"
HERE="${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f "$HERE/timestamps_vs_nemotron.py" ]] || { echo "❌ submit from training_duplex/inference" >&2; exit 1; }

python "$HERE/timestamps_vs_nemotron.py" --asr "$SCRATCH/speechlm/models/nemotron-3.5-asr-streaming-0.6b.nemo" \
    --manifest "fr=$DUPLEX/fr/context/CommonVoice/dev.jsonl" "en=$DUPLEX/en/context/CommonVoice/dev.jsonl" \
    --lines "$LINES" --out "$OUT"
