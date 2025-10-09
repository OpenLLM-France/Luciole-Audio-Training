#!/bin/bash
#SBATCH --job-name=NemoSpeechLM
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --time=20:00:00
#SBATCH --output=log/log-canary-peft_1e-4_to_5e-5.out
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --account=wuh@h100
#SBATCH --constraint=h100
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hnaouara@linagora.com,jlouradour@linagora.com

# =====================================================
# Environment Setup
# =====================================================
module purge
module load arch/h100 nemo/2.3.1

mkdir -p log

LOCAL_FOLDER=$(pwd)
TMP_CACHE="$SCRATCH/speechlm/.cache"
EXPERIMENT_FOLDER="$SCRATCH/speechlm/nemo_experiments"

export TMPDIR=$JOBSCRATCH
export NUM_WORKERS=16
export BATCH_SIZE=2                   # per GPU
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NEMO_MODELS_CACHE=$TMP_CACHE
export SPEECH_ENCODER_MODEL="$NEMO_MODELS_CACHE/nvidia/canary-1b-v2/canary-1b-v2.nemo"

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

EXPERIMENT_NAME="Canary-1b-v2_Llama-3.2-1B-${TOTAL_GPUS}gpu-${GLOBAL_BATCH}gbs--peft"

mkdir -p $EXPERIMENT_FOLDER/$EXPERIMENT_NAME
cp $LOCAL_FOLDER/run_speechlm.slurm $EXPERIMENT_FOLDER/$EXPERIMENT_NAME

# =====================================================
# Dataset Info (adjust as needed)
# =====================================================
NUM_SAMPLES=        # total number of training samples
FRACTION_EPOCH=4.0       # fraction of an epoch for max_steps

# Compute max_steps and limit_train_batches
STEPS_PER_EPOCH=$(python3 -c "import math; print(math.ceil($NUM_SAMPLES / $GLOBAL_BATCH))")
MAX_STEPS=$(python3 -c "import math; print(math.ceil($STEPS_PER_EPOCH * $FRACTION_EPOCH))")
LIMIT_TRAIN_BATCHES=$STEPS_PER_EPOCH

echo "✅ Memory-Optimized Configuration"
echo "➡️  Host: $HOSTNAME"
echo "➡️  Total GPUs: $TOTAL_GPUS | TP=$TP PP=$PP CP=$CP DP=$DP"
echo "➡️  Global batch: $GLOBAL_BATCH | Micro batch: $MICRO_BATCH"
echo "➡️  Steps per epoch: $STEPS_PER_EPOCH"
echo "➡️  Max steps: $MAX_STEPS | Limit train batches: $LIMIT_TRAIN_BATCHES"

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
    --config-path=$LOCAL_FOLDER/conf \
    --config-name=config_canary-1b-v2_linear.yaml \
    data.common.global_batch_size=$GLOBAL_BATCH \
    data.common.micro_batch_size=$MICRO_BATCH \
    data.train_ds.input_cfg=$LOCAL_FOLDER/conf/input_cfg_train.yaml \
    data.train_ds.num_workers=$NUM_WORKERS \
    ++data.train_ds.global_batch_size=$GLOBAL_BATCH \
    ++data.train_ds.micro_batch_size=$MICRO_BATCH \
    data.train_ds.shuffle=true \
    data.train_ds.seed=10 \
    data.train_ds.pin_memory=false \
    data.train_ds.max_duration=120 \
    data.train_ds.min_duration=0.1 \
    data.validation_ds.input_cfg=$LOCAL_FOLDER/conf/input_cfg_val.yaml \
    strategy.tensor_model_parallel_size=$TP \
    strategy.context_parallel_size=$CP \
    ++trainer.max_steps=$MAX_STEPS \
    ++trainer.limit_train_batches=$LIMIT_TRAIN_BATCHES \
    ++callbacks.checkpoint.every_n_train_steps=3000 \
    trainer.num_nodes=$SLURM_NNODES \
    name=$EXPERIMENT_NAME \
    max_time_per_run="00:20:00:00" \
    logger.log_dir=$EXPERIMENT_FOLDER \
    resume.resume_if_exists=true \
    # ~model.peft 

# To run using peft lora just comment or remove ~model.peft
