#!/bin/bash
#SBATCH --job-name=NemoInfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=31
#SBATCH --time=01:00:00
#SBATCH --output=log/infer_log.out
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --account=wuh@h100
#SBATCH --constraint=h100

set -e  # Exit immediately on error

# =====================================================
# Environment Setup
# =====================================================
module purge
module load arch/h100 nemo/2.3.1

LOCAL_FOLDER=$(pwd)
TMP_CACHE="$SCRATCH/speechlm/.cache"
EXPERIMENT_FOLDER="$SCRATCH/speechlm/nemo_experiments"

export HYDRA_FULL_ERROR=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NEMO_MODELS_CACHE=$TMP_CACHE
# export PYTHONUNBUFFERED=1   # Uncomment for real-time logs
# export NEMO_DEBUG=1         # Uncomment for extra debugging

# =====================================================
# Parallelism Setup
# =====================================================
TOTAL_GPUS=$(( SLURM_NNODES * SLURM_GPUS_ON_NODE ))

TP=1   # Tensor Parallelism
PP=1   # Pipeline Parallelism
CP=1   # Context Parallelism
DP=$(( TOTAL_GPUS / (TP * PP * CP) ))

export GLOBAL_BATCH=1   # Small batch to avoid OOM
export MICRO_BATCH=$(( GLOBAL_BATCH / DP ))
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# =====================================================
# Configs & Paths
# =====================================================
EXPERIMENT_NAME="Whisper_Llama-3.2-1B-model"

CONFIG_PATH="$EXPERIMENT_FOLDER/$EXPERIMENT_NAME/run_0"
CONFIG_NAME="config.yaml"
INPUT_CFG_INFER="$LOCAL_FOLDER/conf/input_cfg_infer.yaml"
CKPT_PATH="$EXPERIMENT_FOLDER/$EXPERIMENT_NAME/checkpoints/last/"

OUTPUT_DIR="$EXPERIMENT_FOLDER/outputs"
mkdir -p "$OUTPUT_DIR"

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
# Inference Launch
# =====================================================
srun torchrun $DISTRIBUTED_ARGS $LOCAL_FOLDER/speech_to_text_llm_validate.py \
    --config-path $CONFIG_PATH \
    --config-name $CONFIG_NAME \
    ~data.train_ds \
    data.common.global_batch_size=$GLOBAL_BATCH \
    data.common.micro_batch_size=$MICRO_BATCH \
    data.validation_ds.input_cfg=$INPUT_CFG_INFER \
    data.validation_ds.global_batch_size=$GLOBAL_BATCH \
    data.validation_ds.micro_batch_size=$MICRO_BATCH \
    data.validation_ds.num_workers=0 \
    data.validation_ds.pin_memory=true \
    data.validation_ds.defer_setup=true \
    data.validation_ds.use_lhotse=true \
    data.validation_ds.metric.name='bleu' \
    data.validation_ds.write_predictions_to_file=true \
    data.validation_ds.tokens_to_generate=128 \
    ++data.validation_ds.output_dir=$OUTPUT_DIR \
    data.common.prompt_format='llama3' \
    trainer.devices=-1 \
    trainer.num_nodes=$SLURM_NNODES \
    resume.resume_if_exists=true
    # ++resume.resume_from_path=$CKPT_PATH  # Uncomment to force load a checkpoint
