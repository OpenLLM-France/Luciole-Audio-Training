#!/bin/bash
#SBATCH --job-name=export_experiment
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x.out
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --account=wuh@h100
#SBATCH --constraint=h100

# sbatch --dependency=afterany: --job-name=export_ run_export_experiment.slurm 

# =====================================================
# Environment Setup
# =====================================================
module purge
module load arch/h100 nemo/2.4.0

LOCAL_FOLDER=$(pwd)
TMP_CACHE="$SCRATCH/speechlm/.cache"

export EXPERIMENT_FOLDER="$ALL_CCFRSCRATCH/audio/training/speechlm2_experiments"
# export SPEECH_ENCODER_MODEL="$SCRATCH/speechlm/models/canary-1b-v2.nemo"
# export DATA_FOLDER="$ALL_CCFRSCRATCH/audio"
# export PROMPT_FORMAT="qwen"
NEMO_FORK="${NEMO_FORK:-$HOME/NeMo}"
export PYTHONPATH="$NEMO_FORK:${PYTHONPATH:-}"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=$JOBSCRATCH
export HYDRA_FULL_ERROR=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NEMO_MODELS_CACHE=$TMP_CACHE
# export HYDRA_FULL_ERROR=1

# =====================================================
# Configs
# =====================================================
# CONFIG_PATH="$LOCAL_FOLDER/conf"
# CONFIG_NAME="hf_export_config.yaml"

# Experiment to export: pass the name as the first argument, e.g.
#   sbatch run_export_experiment.slurm my_experiment_name
XP="${1:?Usage: sbatch run_export_experiment.slurm <experiment_name> [every]}"
EVERY="${2:-10000}"

python $LOCAL_FOLDER/export_experiment.py \
    --exp_dir $EXPERIMENT_FOLDER/$XP \
    --every $EVERY