#!/bin/bash
#SBATCH --job-name=Export-HF
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
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

export EXPERIMENT_FOLDER="$ALL_CCFRSCRATCH/audio/training/speechlm2_experiments"
export SPEECH_ENCODER_MODEL="$SCRATCH/speechlm/models/canary-1b-v2.nemo"
export DATA_FOLDER="$ALL_CCFRSCRATCH/audio"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=$JOBSCRATCH
export HYDRA_FULL_ERROR=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NEMO_MODELS_CACHE=$TMP_CACHE
export HYDRA_FULL_ERROR=1

# EXPERIMENT_NAME=$SLURM_JOB_NAME

# =====================================================
# Configs
# =====================================================
CONFIG_PATH="$LOCAL_FOLDER/conf"
CONFIG_NAME="hf_export_config.yaml"

export LLM_MODEL="Qwen/Qwen3-1.7B"
export PROMPT_FORMAT="qwen"

python $LOCAL_FOLDER/to_hf.py \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \