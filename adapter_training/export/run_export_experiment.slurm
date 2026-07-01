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
conda activate $SCRATCH/speechlm/envs/nemo24-salm

LOCAL_FOLDER=$(pwd)
TMP_CACHE="$SCRATCH/speechlm/.cache"

export EXPERIMENT_FOLDER="$ALL_CCFRSCRATCH/audio/training/speechlm2_experiments"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=$JOBSCRATCH
export HYDRA_FULL_ERROR=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NEMO_MODELS_CACHE=$TMP_CACHE


# Experiment to export: pass the name as the first argument, e.g.
#   sbatch run_export_experiment.slurm my_experiment_name
XP="${1:?Usage: sbatch run_export_experiment.slurm <experiment_name> [every] [steps...]}"
EVERY="${2:-50000}"

# Extra step(s) to always export on top of the --every selection. Defaults to
# 10000 and 15000; override positionally or via env, e.g.
#   sbatch run_export_experiment.slurm my_xp 50000 12345 67890
#   STEPS="12345 67890" sbatch run_export_experiment.slurm my_xp
shift 2 2>/dev/null || shift $#
STEPS="${STEPS:-${*:-10000 15000}}"
STEPS_ARG=""
[ -n "$STEPS" ] && STEPS_ARG="--steps $STEPS"

# First (smallest-step) and last ('-last' in name) checkpoints are included by
# default. Disable either with INCLUDE_FIRST=0 / INCLUDE_LAST=0.
FIRST_ARG=""; [ "${INCLUDE_FIRST:-1}" = "0" ] && FIRST_ARG="--no-include-first"
LAST_ARG="";  [ "${INCLUDE_LAST:-1}"  = "0" ] && LAST_ARG="--no-include-last"

# Model class is auto-detected from the experiment's exp_config.yaml
# (SALMAutomodel when model.use_nemo_automodel, else SALM). Override only if
# needed: CLASS_PATH=nemo.collections.speechlm2.models.SALMAutomodel sbatch ...
CLASS_PATH_ARG=""
[ -n "${CLASS_PATH:-}" ] && CLASS_PATH_ARG="--class_path $CLASS_PATH"

python $LOCAL_FOLDER/export_experiment.py \
    --exp_dir $EXPERIMENT_FOLDER/$XP \
    --every $EVERY \
    $STEPS_ARG \
    $FIRST_ARG \
    $LAST_ARG \
    $CLASS_PATH_ARG