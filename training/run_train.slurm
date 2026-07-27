#!/bin/bash
#SBATCH --job-name=test_training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --account=wuh@h100
#SBATCH --constraint=h100

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


# =====================================================
# Safety bash mode
# =====================================================
set -euo pipefail

# =====================================================
# USER CONFIG (edit only this section)
# =====================================================

# --- Logging / saving scripts
SAVE_DIR="${SAVE_DIR:-$ALL_CCFRSCRATCH/audio/training/logs/${SLURM_JOB_NAME}}"

# --- Cache / experiment base folder
TMP_CACHE="$SCRATCH/.cache"
EXPERIMENT_FOLDER="${EXPERIMENT_FOLDER:-$ALL_CCFRSCRATCH/audio/training/speechlm2_experiments}"

# --- Models / data
LLM_MODEL="${LLM_MODEL:-$ALL_CCFRSCRATCH/OpenLLM-BPI-output/finetune/luciole_final/Luciole-1B-SFT-1.1}"
SPEECH_ENCODER_MODEL="${SPEECH_ENCODER_MODEL:-$SCRATCH/speechlm/models/parakeet-tdt-0.6b-v3.nemo}"
DATA_FOLDER="$ALL_CCFRSCRATCH/audio"
PROMPT_FORMAT="${PROMPT_FORMAT:-qwen}"

# --- Data version (override with: DATA_VERSION=v3 sbatch run_train.slurm)
DATA_VERSION=${DATA_VERSION:-v2}

# --- Config file
CONFIG_NAME="${CONFIG_NAME:-run/luciole}"  # Hydra config name (no .yaml); e.g. run/luciole, run/v2

# --- Soft time cap for Lightning (Days:HH:MM:SS). slurm_launcher.py computes
#     this as SLURM --time minus 15min (min 30min); fallback here is 15min.
MAX_TIME_PER_RUN="${MAX_TIME_PER_RUN:-00:00:15:00}"

# --- Training script
TRAIN_SCRIPT="salm_train.py"

# =====================================================
# Derived variables (DO NOT EDIT)
# =====================================================
LOCAL_FOLDER="${LOCAL_FOLDER:-$(pwd)}"
CONFIG_PATH="${LOCAL_FOLDER}/conf"
EXPERIMENT_NAME="${SLURM_JOB_NAME}"

mkdir -p "$SAVE_DIR"
mkdir -p "$TMP_CACHE" "$EXPERIMENT_FOLDER"

# =====================================================
# Load environment
# =====================================================
module purge
# Module is overridable so alternative stacks can be probed (e.g. nemo/2.7.3,
# which brings CUDA 13.2 + torch 2.12). It must stay CUDA-compatible with
# CONDA_ENV below: nemo/2.4.0 is CUDA 12.8, so pairing it with an env built
# from the 2.7.3 stack (or vice versa) will fail confusingly.
module load arch/h100 "${NEMO_MODULE:-nemo/2.4.0}"
# Conda env is overridable (Automodel runs use a dedicated env, e.g. salm_automodel).
# Default keeps the SALM env so existing DDP/HF runs are unchanged.
CONDA_ENV="${CONDA_ENV:-$SCRATCH/speechlm/envs/nemo24-salm}"
echo "Conda env: $CONDA_ENV"
conda activate "$CONDA_ENV"

# =====================================================
# Check NeMo version
# =====================================================
python - <<EOF
import nemo
print(f"NeMo version: {nemo.__version__} (from {nemo.__file__})")
EOF

# =====================================================
# Save SLURM + config for reproducibility
# =====================================================
# DATA_VERSION may be either a subfolder name (legacy: looks for the default
# filename inside it) or a full path to a specific input_cfg .yaml file.
if [[ "$DATA_VERSION" == *.yaml ]]; then
  INPUT_CFG="${CONFIG_PATH}/data/${DATA_VERSION}"
else
  INPUT_CFG="${CONFIG_PATH}/data/${DATA_VERSION}/input_cfg_train_weighted_randomorder_sharded.yaml"
fi

# The materialized slurm file is snapshotted by slurm_launcher.py into SAVE_DIR/<job_name>_<jobid>.slurm.
cp "${CONFIG_PATH}/${CONFIG_NAME}.yaml" \
   "${SAVE_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.yaml"

cp "$INPUT_CFG" \
   "${SAVE_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.input_cfg.yaml"

# =====================================================
# Export training environment variables
# =====================================================
export LLM_MODEL
export SPEECH_ENCODER_MODEL
export DATA_FOLDER
export PROMPT_FORMAT
export MLFLOW_ALLOW_FILE_STORE=true
# Runtime
export HYDRA_FULL_ERROR=1
export TMPDIR="${JOBSCRATCH:-/tmp}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$(( SLURM_GPUS_ON_NODE * SLURM_NNODES ))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

# Offline HF mode
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NEMO_MODELS_CACHE="$TMP_CACHE"

# =====================================================
# SLURM array: tasks after the first auto-resume
# =====================================================
# The first task starts the run and honors OVERWRITE/RESUME as usual; every later one
# must continue from the checkpoint the previous task wrote. Forcing RESUME=true keeps
# the experiment folder, and exp_manager's resume_if_exists does the rest.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" -ne "${SLURM_ARRAY_TASK_MIN:-1}" ]]; then
    echo "↻ Array task ${SLURM_ARRAY_TASK_ID} (not first) — forcing RESUME=true, OVERWRITE=false"
    RESUME=true
    OVERWRITE=false
fi

# =====================================================
# Experiment folder cleanup
# =====================================================
if [[ -d "${EXPERIMENT_FOLDER}/${EXPERIMENT_NAME}" ]]; then
    if [[ "${OVERWRITE:-false}" == "true" ]]; then
        echo "⚠ OVERWRITE=true — wiping ${EXPERIMENT_FOLDER}/${EXPERIMENT_NAME}"
        rm -rf "${EXPERIMENT_FOLDER:?}/${EXPERIMENT_NAME}"
    elif [[ "${RESUME:-false}" == "true" ]]; then
        echo "↻ RESUME=true — keeping ${EXPERIMENT_FOLDER}/${EXPERIMENT_NAME}; exp_manager will resume from its last checkpoint"
    else
        echo "❌ Experiment folder already exists: ${EXPERIMENT_FOLDER}/${EXPERIMENT_NAME}"
        echo "   Re-submit with --resume (resume from last ckpt) or --overwrite (wipe and restart), or change --job-name."
        exit 1
    fi
fi

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
echo " Config         : $CONFIG_NAME"
echo " Train script   : $TRAIN_SCRIPT"
echo " Data version   : $DATA_VERSION"
echo " Experiment fd  : $EXPERIMENT_FOLDER"
echo " Save dir       : $SAVE_DIR"
echo " LLM model      : $LLM_MODEL"
echo " Speech encoder : $SPEECH_ENCODER_MODEL"
echo " Prompt format  : $PROMPT_FORMAT"
echo "======================================"

# =====================================================
# Launch training
# =====================================================
torchrun ${DISTRIBUTED_ARGS} \
  "${LOCAL_FOLDER}/${TRAIN_SCRIPT}" \
  --config-path="$CONFIG_PATH" \
  --config-name="$CONFIG_NAME" \
  ++data.train_ds.input_cfg="$INPUT_CFG"\
  ++trainer.num_nodes="$NNODES" \
  ++exp_manager.exp_dir="$EXPERIMENT_FOLDER" \
  ++exp_manager.name="$EXPERIMENT_NAME" \
  ++exp_manager.max_time_per_run="$MAX_TIME_PER_RUN" \
  ${EXTRA_OVERRIDES:-}