#!/bin/bash
#SBATCH --job-name=SpeechLM2-multitasks_Luciole-23B_DDP_8gpu-1bs_load-BF16-mdl_LoRA
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=92
#SBATCH --time=02:00:00
#SBATCH --output=Luciole_exp/logs/%x_%j.out
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --account=wuh@h100
#SBATCH --constraint=h100

#set -euo pipefail

# =====================================================
# Environment setup
# =====================================================

module purge
module load arch/h100 nemo/2.4.0

# Custom NeMo fork (if needed)
export PYTHONPATH="$PYTHONPATH:$HOME/NeMo.luciole"

# Create log dir BEFORE SLURM writes logs
mkdir -p Luciole_exp/logs

python - <<EOF
import nemo
print(f"NeMo version: {nemo.__version__}")
EOF

LOCAL_FOLDER="$(pwd)"

# Save submitted SLURM script
cp "$LOCAL_FOLDER/run_train.slurm" \
   "Luciole_exp/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.slurm"

# =====================================================
# Paths & caches
# =====================================================

CACHE_ROOT="$SCRATCH/speechlm"
TMP_CACHE="$CACHE_ROOT/.cache"
EXPERIMENT_FOLDER="$CACHE_ROOT/nemo_experiments"

mkdir -p "$TMP_CACHE" "$EXPERIMENT_FOLDER"

# =====================================================
# Training hyperparameters
# =====================================================

export LLM_MODEL="$TMP_CACHE/Luciole-23B-hf"
export SPEECH_ENCODER_MODEL="$SCRATCH/speechlm/models/canary-1b-v2.nemo"
export DATA_FOLDER="$SCRATCH/../commun/audio"
export PROMPT_FORMAT="qwen"

export BATCH_SIZE=1
export MAX_TOKENS=1024
export MAX_DURATION=30
export ACCUMULATE_GRAD_BATCHES=1

# LoRA toggle (Hydra-safe)
# export NO_LORA="~model.lora"
export NO_LORA=""

# =====================================================
# Runtime environment
# =====================================================

export OMP_NUM_THREADS=$(( SLURM_GPUS_ON_NODE * SLURM_NNODES ))
export TMPDIR="${JOBSCRATCH:-/tmp}"
export HYDRA_FULL_ERROR=1

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NEMO_MODELS_CACHE="$TMP_CACHE"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =====================================================
# Experiment
# =====================================================

EXPERIMENT_NAME="$SLURM_JOB_NAME"
rm -rf "$EXPERIMENT_FOLDER/$EXPERIMENT_NAME"

# =====================================================
# Configs
# =====================================================

CONFIG_PATH="$LOCAL_FOLDER/conf"
CONFIG_NAME="config_multitask-v2.yaml"

cp "$CONFIG_PATH/$CONFIG_NAME" \
   "Luciole_exp/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.yaml"

# =====================================================
# Distributed setup (torchrun ONLY, no srun wrapper)
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
# Launch training
# =====================================================

torchrun ${DISTRIBUTED_ARGS} \
  "$LOCAL_FOLDER/salm_train.py" \
  --config-path="$CONFIG_PATH" \
  --config-name="$CONFIG_NAME" \
  ++data.train_ds.batch_size="$BATCH_SIZE" \
  ++data.train_ds.max_tokens="$MAX_TOKENS" \
  ++data.train_ds.max_duration="$MAX_DURATION" \
  ++trainer.accumulate_grad_batches="$ACCUMULATE_GRAD_BATCHES" \
  ++trainer.num_nodes="$NNODES" \
  ++exp_manager.exp_dir="$EXPERIMENT_FOLDER" \
  ++exp_manager.name="$EXPERIMENT_NAME" \
  $NO_LORA