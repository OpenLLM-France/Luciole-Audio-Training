#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=log/%x.out
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --account=wuh@h100
#SBATCH --constraint=h100

# =====================================================
# Environment Setup
# =====================================================
module purge
module load arch/h100 nemo/2.4.0
mkdir -p log

LOCAL_FOLDER=$(pwd)
TMP_CACHE="$SCRATCH/speechlm/.cache"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=$JOBSCRATCH
export HYDRA_FULL_ERROR=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# export NEMO_MODELS_CACHE=$TMP_CACHE
export DATA_FOLDER="$SCRATCH/../commun/audio"

# =====================================================
# Configs
# =====================================================
CONFIG_PATH="$LOCAL_FOLDER/conf"


# =====================================================
# Torchrun Setup
# =====================================================
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=6000
NODE_RANK=$SLURM_NODEID

DISTRIBUTED_ARGS="
  --nproc_per_node $SLURM_GPUS_ON_NODE \
  --nnodes $SLURM_NNODES \
  --node_rank $NODE_RANK \
  --rdzv_backend c10d \
  --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
  --max_restarts 0
"

# =====================================================
# Evaluation Launch
# =====================================================
export XP="<model_name>"
CONFIG_NAME="eval_combined.yaml"
srun torchrun $DISTRIBUTED_ARGS $LOCAL_FOLDER/salm_eval.py \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME
