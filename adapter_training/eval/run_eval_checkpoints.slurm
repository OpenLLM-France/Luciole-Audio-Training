#!/bin/bash
#SBATCH --job-name=eval_ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=log/%x_%j.out
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --account=wuh@h100
#SBATCH --constraint=h100

# =====================================================
# Multi-GPU Multi-Checkpoint Evaluation
# Each GPU evaluates a different checkpoint simultaneously
# =====================================================

# =====================================================
# Environment Setup
# =====================================================
module purge
module load arch/h100 nemo/2.4.0
mkdir -p log

export PYTHONPATH="$HOME/NeMo:$PYTHONPATH"

LOCAL_FOLDER=$(pwd)
TMP_CACHE="$SCRATCH/speechlm/.cache"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=$JOBSCRATCH
export HYDRA_FULL_ERROR=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DATA_FOLDER="$ALL_CCFRSCRATCH/audio"

# =====================================================
# Configuration
# =====================================================
export XP="${XP:-<model_name>}"
CONFIG_NAME="${CONFIG_NAME:-eval_combined}"
NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}

echo "========================================"
echo "Multi-GPU Multi-Checkpoint Evaluation"
echo "========================================"
echo "Model: $XP"
echo "Config: $CONFIG_NAME"
echo "GPUs available: $NUM_GPUS"
echo "========================================"

# =====================================================
# Get list of checkpoints
# =====================================================
CKPT_DIR="$ALL_CCFRSCRATCH/audio/training/speechlm2_experiments/$XP/checkpoints"
CKPT_STEP=${CKPT_STEP:-10000}
CKPT_START=${CKPT_START:-0}

echo "Filtering Checkpoints:"
echo "  Start Step: $CKPT_START"
echo "  Step Interval: $CKPT_STEP"
echo "  Directory: $CKPT_DIR"

# Use python to filter checkpoints
CHECKPOINTS_LIST=$(python3 << EOF
import os
import re
import sys

ckpt_dir = "$CKPT_DIR"
try:
    step_interval = int("$CKPT_STEP")
    start_offset = int("$CKPT_START")
except ValueError:
    step_interval = 5000
    start_offset = 0

checkpoints = []
last_ckpt = None

# regex to match step number: step=12345.ckpt
pattern = re.compile(r"step=(\d+)\.ckpt")

if os.path.isdir(ckpt_dir):
    all_files = sorted(os.listdir(ckpt_dir))

    for f in all_files:
        if not f.endswith(".ckpt"):
            continue
        
        path = os.path.join(ckpt_dir, f)
        
        # Identify last checkpoint
        if "last" in f:
            last_ckpt = path
            continue
            
        match = pattern.search(f)
        if match:
            step = int(match.group(1))
            # Filter logic: step >= start AND (step - start) % interval == 0
            if step >= start_offset and (step - start_offset) % step_interval == 0:
                checkpoints.append((step, path))

    # Sort by step
    checkpoints.sort(key=lambda x: x[0])
    final_paths = [p for _, p in checkpoints]

    # Append last checkpoint if it exists and user wants it (default yes)
    if last_ckpt:
        # Avoid duplicates if last_ckpt uses step name format (unlikely but safe)
        if last_ckpt not in final_paths:
            final_paths.append(last_ckpt)

    print("\n".join(final_paths))
else:
    print("")
EOF
)

# Convert newline-separated string to array
IFS=$'\n' read -rd '' -a CHECKPOINTS <<< "$CHECKPOINTS_LIST"
NUM_CKPTS=${#CHECKPOINTS[@]}

echo "Found $NUM_CKPTS checkpoints:"
for i in "${!CHECKPOINTS[@]}"; do
    echo "  [$i] $(basename ${CHECKPOINTS[$i]})"
done
echo "========================================"

# =====================================================
# Run evaluation in parallel across GPUs
# =====================================================
# Function to evaluate a checkpoint on a specific GPU
evaluate_checkpoint() {
    local GPU_ID=$1
    local CKPT_PATH=$2
    local CKPT_NAME=$(basename $CKPT_PATH)
    
    echo "[GPU $GPU_ID] Starting evaluation of $CKPT_NAME"
    
    # Construct command with conditional dataset_filter
    CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python $LOCAL_FOLDER/salm_eval.py --config-path=$LOCAL_FOLDER/conf --config-name=$CONFIG_NAME ckpt=\"'$CKPT_NAME'\" device=\"cuda:0\""
    
    if [ ! -z "$DATASET_FILTER" ]; then
        CMD="$CMD dataset_filter=\"'$DATASET_FILTER'\""
    fi
    
    # Execute command
    eval $CMD 2>&1 | tee log/eval_${CKPT_NAME}_gpu${GPU_ID}.log &
}

# Distribute checkpoints across GPUs
CKPT_IDX=0
PIDS=()

# Start initial batch (one per GPU)
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    if [ $CKPT_IDX -lt $NUM_CKPTS ]; then
        CKPT_PATH=${CHECKPOINTS[$CKPT_IDX]}
        evaluate_checkpoint $GPU_ID "$CKPT_PATH"
        PIDS[$GPU_ID]=$!
        CKPT_IDX=$((CKPT_IDX + 1))
    fi
done

# As each GPU finishes, assign it the next checkpoint
while [ $CKPT_IDX -lt $NUM_CKPTS ]; do
    for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
        # Check if this GPU's process has finished
        if [ ! -z "${PIDS[$GPU_ID]}" ]; then
            if ! ps -p ${PIDS[$GPU_ID]} > /dev/null 2>&1; then
                # GPU is free, assign next checkpoint
                if [ $CKPT_IDX -lt $NUM_CKPTS ]; then
                    CKPT_PATH=${CHECKPOINTS[$CKPT_IDX]}
                    echo "[GPU $GPU_ID] Previous task complete, starting $(basename $CKPT_PATH)"
                    sleep 2 # avoid OOM
                    evaluate_checkpoint $GPU_ID "$CKPT_PATH"
                    PIDS[$GPU_ID]=$!
                    CKPT_IDX=$((CKPT_IDX + 1))
                fi
            fi
        fi
    done
    sleep 10  # Check every 10 seconds
done

# Wait for all remaining processes to complete
echo "Waiting for all evaluations to complete..."
wait

echo "========================================"
echo "All checkpoint evaluations completed!"
echo "========================================"
