#!/bin/bash
#SBATCH --job-name=agent_teacher_forcing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --account=wuh@h100
#SBATCH --constraint=h100
#SBATCH --hint=nomultithread
#SBATCH --output=/lustre/fsn1/projects/rech/qgz/%u/speechlm/duplex_diag/%x_%j.out
# Teacher-forcing diagnostic of the agent channel of a single-head model (agent_teacher_forcing.py) on the validation ASR sets.
#   cd training_duplex/inference
#   mkdir -p $SCRATCH/speechlm/duplex_diag && sbatch agent_teacher_forcing.slurm [xp] [step] [att_context] ["datasets"] [max_examples]
# defaults: duplex_transcribe_100k_v1, step-010000, the config's first lookahead ("" or e.g. [56,6]), "asr_fr asr_en" (validation_ds.datasets keys). Outputs in $SCRATCH/speechlm/duplex_diag/<xp>_<step>[_ctxN]/.
set -euo pipefail
XP="${1:-duplex_transcribe_100k_v1}"
STEP="${2:-step-010000}"
ATT="${3:-}"
DATASETS="${4:-asr_fr asr_en}"
MAX_EXAMPLES="${5:-256}"
EXP_DIR="$ALL_CCFRSCRATCH/audio/training/duplex/$XP"
OUT="$SCRATCH/speechlm/duplex_diag/${XP}_${STEP}${ATT:+_ctx$(echo "$ATT" | tr -dc "0-9," | cut -d, -f2)}"

module purge
module load arch/h100 nemo/2.7.3
conda activate "$SCRATCH/speechlm/envs/nemo27-salm_automodel_3"
NEMO_DIR="$HOME/NeMo_Duplex"
export PYTHONPATH="$NEMO_DIR" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NEMO_MODELS_CACHE="$SCRATCH/.cache" TMPDIR="${JOBSCRATCH:-/tmp}"
# $HOME/NeMo (installed editable) serves a running training: never use it here.
nemo_file="$(python -c "import nemo; print(nemo.__file__)")"
[[ "$nemo_file" == "$NEMO_DIR"/* ]] || { echo "❌ nemo imported from $nemo_file, not $NEMO_DIR" >&2; exit 1; }
# Slurm runs a copy of this script: find agent_teacher_forcing.py from the submission directory.
HERE="${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f "$HERE/agent_teacher_forcing.py" ]] || { echo "❌ submit from training_duplex/inference" >&2; exit 1; }

python "$HERE/agent_teacher_forcing.py" --exp-dir "$EXP_DIR" --ckpt "$EXP_DIR/checkpoints/$STEP.ckpt" --out "$OUT" --datasets $DATASETS --max-examples "$MAX_EXAMPLES" ${ATT:+--att-context "$ATT"}
