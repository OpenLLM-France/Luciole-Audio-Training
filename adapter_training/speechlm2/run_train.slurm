#!/bin/bash
#SBATCH --job-name=test_ddp-luciol-8B-LoRA
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --time=00:30:00
#SBATCH --output=Luciole_exp/logs/%x_%j.out
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --account=wuh@h100
#SBATCH --constraint=h100

# =====================================================
# Safety bash mode
# =====================================================
# set -euo pipefail

# =====================================================
# USER CONFIG (edit only this section)
# =====================================================

# --- Logging / saving scripts
SAVE_DIR="Luciole_exp"
LOG_DIR="${SAVE_DIR}/logs"

# --- Cache / experiment base folder
CACHE_ROOT="$SCRATCH/speechlm"
TMP_CACHE="${CACHE_ROOT}/.cache"
EXPERIMENT_FOLDER="${CACHE_ROOT}/nemo_experiments"

# --- NeMo fork
NEMO_FORK="$HOME/Work-folder/NeMo.luciole.2.7.0-rc0" # This is the last version of NeMo we found for now (1/2/2026)

# --- Models / data
LLM_MODEL="${TMP_CACHE}/Luciole-8B-hf"
SPEECH_ENCODER_MODEL="$SCRATCH/speechlm/models/canary-1b-v2.nemo"
DATA_FOLDER="$SCRATCH/../commun/audio"
PROMPT_FORMAT="qwen"

# --- Training hyperparams
MAX_DURATION=45
MAX_TOKENS=4096
ACCUMULATE_GRAD_BATCHES=1

# Choose ONE:
BATCH_TOKENS=1024
#BATCH_SIZE=2

# --- LoRA control
USE_LORA=true   # set false to disable LoRA

# --- Config file
CONFIG_NAME="config_multitask-v2.yaml"  # set the config you want to use

# --- Training script
TRAIN_SCRIPT="salm_train.py"

# --- Optional pip installs (not recommended to enable in job)
INSTALL_DEPS=false

# =====================================================
# Derived variables (DO NOT EDIT)
# =====================================================
LOCAL_FOLDER="$(pwd)"
CONFIG_PATH="${LOCAL_FOLDER}/conf"
EXPERIMENT_NAME="${SLURM_JOB_NAME}"

mkdir -p "$LOG_DIR"
mkdir -p "$TMP_CACHE" "$EXPERIMENT_FOLDER"

# =====================================================
# Load environment
# =====================================================
module purge
module load arch/h100 nemo/2.4.0

export PYTHONPATH="$PYTHONPATH:$NEMO_FORK"

# =====================================================
# Optional dependencies (NOT executed by default)
# =====================================================
# Some dependencies are needed to run newer NeMo versions
# inside the NeMo 2.4.0 environment.
#
# Run once manually if needed:
#
#   pip install nv-one-logger-core==2.3.1 \
#              nv-one-logger-pytorch-lightning-integration==2.3.1 \
#              nv-one-logger-training-telemetry==2.3.1
#
#   pip install kaldialign==0.9.1
#
# TIP: better to install in user space:
#   pip install --user ...
#
# OR enable INSTALL_DEPS=true (not recommended)

if [[ "$INSTALL_DEPS" == "true" ]]; then
    echo "⚠️ Installing pip dependencies..."

    pip install --user nv-one-logger-core==2.3.1 \
                     nv-one-logger-pytorch-lightning-integration==2.3.1 \
                     nv-one-logger-training-telemetry==2.3.1

    pip install --user kaldialign==0.9.1
fi

# =====================================================
# Check NeMo version
# =====================================================
python - <<EOF
import nemo
print(f"NeMo version: {nemo.__version__}")
EOF

# =====================================================
# Save SLURM + config for reproducibility
# =====================================================
cp "${LOCAL_FOLDER}/run_train.slurm" \
   "${SAVE_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.slurm"

cp "${CONFIG_PATH}/${CONFIG_NAME}" \
   "${SAVE_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.yaml"

# =====================================================
# Export training environment variables
# =====================================================
export LLM_MODEL
export SPEECH_ENCODER_MODEL
export DATA_FOLDER
export PROMPT_FORMAT

# Runtime
export HYDRA_FULL_ERROR=1
export TMPDIR="${JOBSCRATCH:-/tmp}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$(( SLURM_GPUS_ON_NODE * SLURM_NNODES ))

# Offline HF mode
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NEMO_MODELS_CACHE="$TMP_CACHE"

# =====================================================
# Batch method selection
# =====================================================
if [[ -n "${BATCH_TOKENS:-}" ]]; then
    batch_method="batch_tokens"
    batch_value="$BATCH_TOKENS"
elif [[ -n "${BATCH_SIZE:-}" ]]; then
    batch_method="batch_size"
    batch_value="$BATCH_SIZE"
else
    echo "❌ Error: set either BATCH_TOKENS or BATCH_SIZE"
    exit 1
fi

# =====================================================
# LoRA toggle
# =====================================================
if [[ "$USE_LORA" == "true" ]]; then
    LORA_ARG=""
else
    LORA_ARG="~model.lora"
fi

# =====================================================
# Experiment folder cleanup
# =====================================================
rm -rf "${EXPERIMENT_FOLDER:?}/${EXPERIMENT_NAME}"

# =====================================================
# Distributed setup (torchrun)
# =====================================================
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((15000 + RANDOM % 20000))

NNODES="$SLURM_NNODES"
NODE_RANK="$SLURM_NODEID"
NPROC_PER_NODE="$SLURM_GPUS_ON_NODE"

DISTRIBUTED_ARGS="
--nproc_per_node=${NPROC_PER_NODE}
--nnodes=${NNODES}
--node_rank=${NODE_RANK}
--rdzv_backend=c10d
--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}
--max_restarts=0
"

# =====================================================
# Print useful debug info
# =====================================================
echo "======================================"
echo " Job Name       : $SLURM_JOB_NAME"
echo " Job ID         : $SLURM_JOB_ID"
echo " Nodes          : $NNODES"
echo " GPUs per node  : $NPROC_PER_NODE"
echo " Master Addr    : $MASTER_ADDR"
echo " Master Port    : $MASTER_PORT"
echo " Batch method   : $batch_method"
echo " Batch value    : $batch_value"
echo " LoRA enabled   : $USE_LORA"
echo " Config         : $CONFIG_NAME"
echo " Train script   : $TRAIN_SCRIPT"
echo "======================================"

# =====================================================
# Launch training
# =====================================================
torchrun ${DISTRIBUTED_ARGS} \
  "${LOCAL_FOLDER}/${TRAIN_SCRIPT}" \
  --config-path="$CONFIG_PATH" \
  --config-name="$CONFIG_NAME" \
  ++data.train_ds.${batch_method}="$batch_value" \
  ++data.train_ds.max_tokens="$MAX_TOKENS" \
  ++data.train_ds.max_duration="$MAX_DURATION" \
  ++trainer.accumulate_grad_batches="$ACCUMULATE_GRAD_BATCHES" \
  ++trainer.num_nodes="$NNODES" \
  ++exp_manager.exp_dir="$EXPERIMENT_FOLDER" \
  ++exp_manager.name="$EXPERIMENT_NAME" \
  $LORA_ARG
