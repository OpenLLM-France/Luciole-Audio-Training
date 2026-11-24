#!/bin/bash
#SBATCH --job-name=estimate_bins_duplex
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fsn1/projects/rech/qgz/%u/speechlm/duplex_align/logs/%x_%j.out
#SBATCH --hint=nomultithread
#SBATCH --account=qgz@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-dev

# Bucket duration bins for data.train_ds.bucket_duration_bins, estimated on the duplex examples of an input_cfg
# (durations of the whole conversation_as_duplex timelines, answers included). Without explicit bins NeMo uses
# linspace(min_duration, max_duration, num_buckets): 40 s wide buckets for max_duration 480, so the first one
# mixed 2 s ASR segments with 40 s QA timelines (padded batches up to 3.6x batch_duration: OOM of
# duplex_2heads_100k_4gpu at 11.3k steps).
# The bins printed are in seconds and the config takes frames: divide them by the frame length (0.08 s).
#   sbatch estimate_bins_duplex.slurm [input_cfg] [num_buckets] [num_examples]

set -euo pipefail
NEMO_MODULE="${NEMO_MODULE:-nemo/2.7.3}"
CONDA_ENV="${CONDA_ENV:-$SCRATCH/speechlm/envs/nemo27-salm_automodel_3}"
NEMO_DIR="${NEMO_DIR:-$HOME/NeMo_Duplex}"
HERE="${SLURM_SUBMIT_DIR:-$PWD}"
INPUT_CFG="${1:-conf/data/input_cfg_qa.yaml}"
[[ "$INPUT_CFG" == /* ]] || INPUT_CFG="$HERE/$INPUT_CFG"

module purge
module load arch/h100 "$NEMO_MODULE"
conda activate "$CONDA_ENV"
export PYTHONPATH="$NEMO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export DATA_FOLDER="$ALL_CCFRSCRATCH/audio"  # interpolated in the input_cfgs' manifest_filepath
export LLM_MODEL="${LLM_MODEL:-$ALL_CCFRSCRATCH/OpenLLM-BPI-output/sft/final/Luciole-1B-Instruct-1.1}"  # tokenizer

python "$NEMO_DIR/scripts/speech_recognition/estimate_duration_bins.py" "$INPUT_CFG" \
    --buckets "${2:-24}" --num_examples "${3:-30000}" --min_duration 0.5 --max_duration 480
