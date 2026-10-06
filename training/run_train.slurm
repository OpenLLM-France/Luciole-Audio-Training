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
#SBATCH --mail-type=ALL

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

# --- Models / data
# (defined before the output folders: those are grouped per LLM, see llm_family below)
LLM_MODEL="${LLM_MODEL:-$ALL_CCFRSCRATCH/OpenLLM-BPI-output/sft/final/Luciole-1B-Instruct-1.1}"
SPEECH_ENCODER_MODEL="${SPEECH_ENCODER_MODEL:-$SCRATCH/speechlm/models/parakeet-tdt-0.6b-v3.nemo}"
DATA_FOLDER="$ALL_CCFRSCRATCH/audio"
PROMPT_FORMAT="${PROMPT_FORMAT:-qwen}"

# --- Output folders, grouped per LLM: <root>/<model-family>/<job_name>
# Runs of the same model stay together (Luciole-1B, Luciole-8B, Luciole-23B, ...)
# instead of piling up at the root. slurm_launcher.py computes the same paths (and
# can insert an extra --subdir level); this block only matters for a bare
# `sbatch run_train.slurm`, and both must keep agreeing.
llm_family() {
  local name; name="$(basename "${1%/}")"
  if [[ "$name" =~ [Ll]uciole[-_]?([0-9]+)[Bb] ]]; then
    echo "Luciole-${BASH_REMATCH[1]}B"
  else
    echo "$name"   # non-Luciole LLM: group by model dir name rather than at the root
  fi
}
MODEL_FAMILY="$(llm_family "$LLM_MODEL")"

SAVE_DIR="${SAVE_DIR:-$ALL_CCFRSCRATCH/audio/training/logs/${MODEL_FAMILY}/${SLURM_JOB_NAME}}"
EXPERIMENT_FOLDER="${EXPERIMENT_FOLDER:-$ALL_CCFRSCRATCH/audio/training/speechlm2_experiments/${MODEL_FAMILY}}"

# --- Cache
TMP_CACHE="$SCRATCH/.cache"

# --- Data version (override with: DATA_VERSION=v3 sbatch run_train.slurm)
DATA_VERSION=${DATA_VERSION:-v4}

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
module load arch/h100 "${NEMO_MODULE:-nemo/2.7.3}"
# Conda env is overridable (Automodel runs use a dedicated env, e.g. salm_automodel).
# Default keeps the SALM env so existing DDP/HF runs are unchanged.
CONDA_ENV="${CONDA_ENV:-$SCRATCH/speechlm/envs/nemo27-salm_automodel_3}"
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
# Resolve CHECKPOINT from CHECKPOINT_DIR (curriculum warm-start)
# =====================================================
# Lets slurm_launcher.py submit a later stage with --dependency=afterok before the
# earlier stage's checkpoint exists: CHECKPOINT_DIR is a literal path known at
# submission time, CHECKPOINT (the actual *-last.ckpt file) is only resolved here,
# once the dependency has been satisfied and the file is guaranteed to exist.
# *-last.ckpt are distcp DIRECTORIES: `ls -d` lists the entry itself, plain `ls -t`
# would list its content and return a bare "__0_0.distcp".
if [[ -n "${CHECKPOINT:-}" && -n "${CHECKPOINT_DIR:-}" ]]; then
    echo "❌ Both CHECKPOINT (${CHECKPOINT}) and CHECKPOINT_DIR (${CHECKPOINT_DIR}) are set;" \
         "a stale CHECKPOINT in the submitting shell would silently win. Unset one." >&2
    exit 1
fi
if [[ -n "${CHECKPOINT_DIR:-}" ]]; then
    CHECKPOINT="$(ls -td "${CHECKPOINT_DIR}"/*-last.ckpt 2>/dev/null | head -1)"
    if [[ -z "$CHECKPOINT" || ! -e "$CHECKPOINT" ]]; then
        echo "❌ No *-last.ckpt found in CHECKPOINT_DIR=${CHECKPOINT_DIR}" >&2
        exit 1
    fi
    export CHECKPOINT
    echo "Resolved CHECKPOINT from CHECKPOINT_DIR: $CHECKPOINT"
fi

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
echo " Model family   : $MODEL_FAMILY"
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