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
module load arch/h100 nemo/2.7.3
# conda activate happens below (its default no longer depends on $XP, but CONDA_ENV
# can still override it per-call -- see the comment there).

LOCAL_FOLDER=$(pwd)
TMP_CACHE="$SCRATCH/speechlm/.cache"

# Root of the experiments. Duplex runs live elsewhere:
#   EXPERIMENT_FOLDER=$ALL_CCFRSCRATCH/audio/training/duplex NEMO_DIR=$HOME/NeMo_Duplex \
#   CLASS_PATH=nemo.collections.speechlm2.models.DuplexSTTModel sbatch run_export_experiment.slurm <run>
export EXPERIMENT_FOLDER="${EXPERIMENT_FOLDER:-$ALL_CCFRSCRATCH/audio/training/speechlm2_experiments}"
umask 007  # hf_checkpoints/ lands in the shared project space: group qgz only

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=$JOBSCRATCH
export HYDRA_FULL_ERROR=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NEMO_MODELS_CACHE=$TMP_CACHE


# Experiment to export: pass its path RELATIVE TO $EXPERIMENT_FOLDER as the first
# argument. Training groups runs per LLM (Luciole-1B/, Luciole-8B/, ...) and optionally
# per xp subfolder, so that path usually has 2-3 components:
#   sbatch run_export_experiment.slurm Luciole-1B/xp_from_lora/stage2_from_lora_v2
XP="${1:?Usage: sbatch run_export_experiment.slurm <model-family>/[<subdir>/]<run> [every] [steps...]}"
EVERY="${2:-50000}"

# Conda env to export in. Pass an env NAME (under $SCRATCH/speechlm/envs) to force it:
#   sbatch --export=ALL,CONDA_ENV=nemo27-salm run_export_experiment.slurm ...
# Unset -> nemo27-salm_automodel_3, the training default since the nemo/2.7.3 migration.
# It has everything nemo27-salm has plus nemo_automodel (SALM itself never imports
# nemo_automodel, so it's a strict superset for a plain-SALM export too) -- one env
# covers both classes now, so there is no more automodel-vs-plain detection to do.
# Force CONDA_ENV to nemo24-salm(_automodel) when re-exporting a pre-nemo27 experiment
# (exp_config.yaml records no NeMo-version marker, so this can't be auto-detected).
CONDA_ENV="${CONDA_ENV:-nemo27-salm_automodel_3}"
echo "Conda env: $CONDA_ENV"
conda activate "$SCRATCH/speechlm/envs/$CONDA_ENV"
# NeMo copy to import instead of the env's editable one (e.g. $HOME/NeMo_Duplex for the duplex runs).
if [ -n "${NEMO_DIR:-}" ]; then
    export PYTHONPATH="$NEMO_DIR${PYTHONPATH:+:$PYTHONPATH}"
    echo "NeMo: $(python -c 'import nemo; print(nemo.__file__)' | tail -1)"
fi

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
# Re-export checkpoints already in hf_checkpoints/ (skipped otherwise): FORCE=1
FORCE_ARG=""; [ "${FORCE:-0}" = "1" ] && FORCE_ARG="--force"

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
    $CLASS_PATH_ARG \
    $FORCE_ARG
