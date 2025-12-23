#!/bin/bash
#SBATCH --job-name=SpeechLM2-multitasks_Qwen3-8B
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
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
EXPERIMENT_FOLDER="$SCRATCH/speechlm/nemo_experiments"

export PYTHONPATH="$PYTHONPATH:$HOME/NeMo.luciole"

export LLM_MODEL="$SCRATCH/speechlm/.cache/Qwen/Qwen3-8B"
export PROMPT_FORMAT="qwen"
export BATCH_TOKENS=512
export NO_LORA="~model.lora"  # pass as string 
# if you want to use lora just comment the line above and uncomment the line below
# export NO_LORA=""
export SPEECH_ENCODER_MODEL="$SCRATCH/speechlm/models/canary-1b-v2.nemo"
export DATA_FOLDER="$SCRATCH/../commun/audio"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=$JOBSCRATCH
export HYDRA_FULL_ERROR=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NEMO_MODELS_CACHE=$TMP_CACHE

EXPERIMENT_NAME=$SLURM_JOB_NAME
# rm -rf "$EXPERIMENT_FOLDER/$EXPERIMENT_NAME"

# =====================================================
# Configs
# =====================================================
CONFIG_PATH="$LOCAL_FOLDER/conf"
CONFIG_NAME="config_multitask-v2.yaml"

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
# Training Launch
# =====================================================
srun torchrun $DISTRIBUTED_ARGS $LOCAL_FOLDER/salm_train.py \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    data.train_ds.batch_tokens=$BATCH_TOKENS \
    trainer.num_nodes=$SLURM_NNODES \
    exp_manager.exp_dir=$EXPERIMENT_FOLDER \
    exp_manager.name=$EXPERIMENT_NAME \
    $NO_LORA