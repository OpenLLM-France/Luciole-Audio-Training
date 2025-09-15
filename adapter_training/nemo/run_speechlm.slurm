#!/bin/bash
#SBATCH --job-name=Test_SpeechLM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=log/log.out
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --account=wuh@h100
#SBATCH --constraint=h100

# =====================================================
# Environment Setup
# =====================================================
module purge
module load arch/h100 nemo/2.3.1

LOCAL_FOLDER=$(pwd)
TMP_CACHE="$SCRATCH/speechlm/.cache"
EXPERIMENT_FOLDER="$SCRATCH/speechlm/nemo_experiments"

export NUM_WORKERS=16
export BATCH_SIZE=2
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NEMO_MODELS_CACHE=$TMP_CACHE

# =====================================================
# Parallelism Setup
# =====================================================
TOTAL_GPUS=$(( SLURM_NNODES * SLURM_GPUS_ON_NODE ))

TP=1   # Tensor Parallelism
PP=1   # Pipeline Parallelism
CP=1   # Context Parallelism
DP=$(( TOTAL_GPUS / (TP * PP * CP) ))

GLOBAL_BATCH=$(( TOTAL_GPUS * BATCH_SIZE ))
MICRO_BATCH=$(( GLOBAL_BATCH / DP ))

EXPERIMENT_NAME="test"
OUTPUT_DIR=$EXPERIMENT_FOLDER/$EXPERIMENT_NAME/preds
rm -rf "$EXPERIMENT_FOLDER/$EXPERIMENT_NAME"
mkdir -p "$OUTPUT_DIR"

# =====================================================
# Configs
# =====================================================
CONFIG_PATH="$LOCAL_FOLDER/conf"
CONFIG_NAME="config.yaml"
INPUT_CFG_TRAIN="$CONFIG_PATH/input_cfg_train.yaml"
INPUT_CFG_VAL="$CONFIG_PATH/input_cfg_val.yaml"

# =====================================================
# Validation Checks
# =====================================================
if (( GLOBAL_BATCH % DP != 0 )); then
  echo "❌ ERROR: GLOBAL_BATCH ($GLOBAL_BATCH) must be divisible by DP ($DP)"
  exit 1
fi

if (( PP > 1 && MICRO_BATCH % PP != 0 )); then
  echo "❌ ERROR: MICRO_BATCH ($MICRO_BATCH) must be divisible by PP ($PP)"
  exit 1
fi

echo "✅ Memory-Optimized Configuration"
echo "➡️  Host: $HOSTNAME"
echo "➡️  Total GPUs: $TOTAL_GPUS | TP=$TP PP=$PP CP=$CP DP=$DP"
echo "➡️  Global batch: $GLOBAL_BATCH | Micro batch: $MICRO_BATCH"

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
srun torchrun $DISTRIBUTED_ARGS $LOCAL_FOLDER/speech_to_text_llm_train.py \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    data.common.global_batch_size=$GLOBAL_BATCH \
    data.common.micro_batch_size=$MICRO_BATCH \
    data.train_ds.input_cfg=$INPUT_CFG_TRAIN \
    data.train_ds.num_workers=$NUM_WORKERS \
    data.validation_ds.input_cfg=$INPUT_CFG_VAL \
    strategy.tensor_model_parallel_size=$TP \
    strategy.context_parallel_size=$CP \
    trainer.max_steps=3 \
    trainer.num_nodes=$SLURM_NNODES \
    name=$EXPERIMENT_NAME \
    max_time_per_run="00:00:30:00" \
    logger.log_dir=$EXPERIMENT_FOLDER \
    resume.resume_if_exists=true \
    name=$EXPERIMENT_NAME

# Notes:
# - For validation during training, add:
#   data.validation_ds.global_batch_size=$GLOBAL_BATCH \
#   data.validation_ds.micro_batch_size=$MICRO_BATCH \
#   trainer.limit_val_batches=10
