#!/bin/bash
#SBATCH --job-name=duplex_stt_luciole
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --account=wuh@h100
#SBATCH --constraint=h100

# Duplex STT training, single phase (conf/duplex_<MODEL>.yaml, MODEL=qa or transcribe). See README.md.
# Needs the NeMo fork branch luciole_duplexstt_3.1 (trainable_head_rows + conversation_as_duplex) in
# $NEMO_DIR: a `git archive` copy without .git, so the branch is in $NEMO_DIR/GIT_REVISION. The env has
# $HOME/NeMo installed editable; PYTHONPATH makes this copy win.
#
# Submit through duplex_launcher.py: it creates $ALL_CCFRSCRATCH/audio/training/duplex/<job_name>/ and
# puts the Slurm logs, config snapshots and the exp_manager output (checkpoints, TensorBoard) in it:
#
#   python duplex_launcher.py --model qa --job-name <xp>
#   python duplex_launcher.py --model transcribe --job-name <xp>
#   python duplex_launcher.py --job-name smoke --qos dev --gpus 1 --time 00:30:00 \
#       --set ++trainer.max_steps=50 --set ++trainer.val_check_interval=25
#
# A plain `sbatch run_duplex.slurm` from train_duplex/ still works, with its logs in ./logs/.

set -euo pipefail

# ================= USER CONFIG =================
LLM_MODEL="${LLM_MODEL:-$ALL_CCFRSCRATCH/OpenLLM-BPI-output/sft/final/Luciole-1B-Instruct-1.1}"
SPEECH_ENCODER_MODEL="${SPEECH_ENCODER_MODEL:-$SCRATCH/speechlm/models/nemotron-3.5-asr-streaming-0.6b.nemo}"
DATA_FOLDER="$ALL_CCFRSCRATCH/audio"
NEMO_DIR="${NEMO_DIR:-$HOME/NeMo_Duplex}"
# Two models, each = conf/duplex_<MODEL>.yaml + conf/data/input_cfg_<MODEL>.yaml + conf/data/val_cfg_<MODEL>.yaml:
#   qa         ASR/QA (v4 logic): the agent answers after the audio, 2 heads (agent + prompt-less ASR head)
#   transcribe Luciole-Audio-Transcribe (v5 logic): the agent writes the transcript/translation while the
#              user speaks, 1 head
# CONFIG_NAME / INPUT_CFG / VAL_CFG override each file separately.
MODEL="${MODEL:-qa}"
CONFIG_NAME="${CONFIG_NAME:-duplex_$MODEL}"
INPUT_CFG="${INPUT_CFG:-conf/data/input_cfg_$MODEL.yaml}"
VAL_CFG="${VAL_CFG:-conf/data/val_cfg_$MODEL.yaml}"
EXPERIMENT_FOLDER="${EXPERIMENT_FOLDER:-$ALL_CCFRSCRATCH/audio/training/duplex}"
SAVE_DIR="${SAVE_DIR:-$(pwd)/logs}"  # config snapshots; duplex_launcher.py sets it to the experiment folder
MAX_TIME_PER_RUN="${MAX_TIME_PER_RUN:-00:19:40:00}"  # SLURM --time minus 20 min: leaves room for the -last.ckpt
NEMO_MODULE="${NEMO_MODULE:-nemo/2.7.3}"
CONDA_ENV="${CONDA_ENV:-$SCRATCH/speechlm/envs/nemo27-salm_automodel_3}"
# ===============================================

LOCAL_FOLDER="${LOCAL_FOLDER:-$(pwd)}"
[[ "$INPUT_CFG" == /* ]] || INPUT_CFG="$LOCAL_FOLDER/$INPUT_CFG"
[[ "$VAL_CFG" == /* ]] || VAL_CFG="$LOCAL_FOLDER/$VAL_CFG"
umask 007  # shared project space: group qgz only
mkdir -p "$SAVE_DIR" "$EXPERIMENT_FOLDER"

module purge
module load arch/h100 "$NEMO_MODULE"
conda activate "$CONDA_ENV"
export PYTHONPATH="$NEMO_DIR${PYTHONPATH:+:$PYTHONPATH}"
nemo_file="$(python -c "import nemo; print(nemo.__file__)")"
echo "NeMo from $nemo_file"
[[ "$nemo_file" == "$NEMO_DIR"/* ]] || { echo "❌ nemo imported from $nemo_file, not $NEMO_DIR" >&2; exit 1; }

revision="$(cat "$NEMO_DIR/GIT_REVISION")"
[[ "$revision" == "luciole_duplexstt_3.1 "* ]] || { echo "❌ $NEMO_DIR is '$revision', expected luciole_duplexstt_3.1" >&2; exit 1; }
[[ -f "$SPEECH_ENCODER_MODEL" ]] || { echo "❌ missing $SPEECH_ENCODER_MODEL" >&2; exit 1; }
[[ "$LLM_MODEL" != *Nemotron* ]] || { echo "❌ LLM path must not contain 'Nemotron' (see conf/duplex_qa.yaml)" >&2; exit 1; }

export LLM_MODEL SPEECH_ENCODER_MODEL DATA_FOLDER
export HYDRA_FULL_ERROR=1
export TMPDIR="${JOBSCRATCH:-/tmp}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$(( SLURM_GPUS_ON_NODE * SLURM_NNODES ))
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NEMO_MODELS_CACHE="$SCRATCH/.cache"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((15000 + RANDOM % 20000))

echo "config=$CONFIG_NAME input_cfg=$INPUT_CFG val_cfg=$VAL_CFG llm=$LLM_MODEL asr=$SPEECH_ENCODER_MODEL nemo=$NEMO_DIR@${revision#* }"
cp "$LOCAL_FOLDER/conf/${CONFIG_NAME}.yaml" "$SAVE_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.yaml"
cp "$INPUT_CFG" "$SAVE_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.input_cfg.yaml"
cp "$VAL_CFG" "$SAVE_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.val_cfg.yaml"

torchrun \
  --nproc_per_node="$SLURM_GPUS_ON_NODE" --nnodes="$SLURM_NNODES" --node_rank="$SLURM_NODEID" \
  --rdzv_backend=c10d --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" --max_restarts=0 \
  "${LOCAL_FOLDER}/duplex_train.py" \
  --config-path="${LOCAL_FOLDER}/conf" \
  --config-name="$CONFIG_NAME" \
  ++data.train_ds.input_cfg="$INPUT_CFG" \
  ++data.validation_ds.datasets="$VAL_CFG" \
  ++trainer.num_nodes="$SLURM_NNODES" \
  ++exp_manager.exp_dir="$EXPERIMENT_FOLDER" \
  ++exp_manager.name="$SLURM_JOB_NAME" \
  ++exp_manager.max_time_per_run="$MAX_TIME_PER_RUN" \
  ${EXTRA_OVERRIDES:-}
