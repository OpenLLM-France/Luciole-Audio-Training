#!/bin/bash
#SBATCH --job-name=NemoSpeechLLM
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=log/log.out
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --account=wuh@h100
#SBATCH --constraint=h100

# Load required modules
module purge
module load arch/h100 nemo/2.3.1

TMP_CACHE="$SCRATCH/speechlm/.cache"
EXPERIMENT_FOLDER="$SCRATCH/speechlm/nemo_experiments"
EXPERIMENT_NAME="test"

LOCAL_FOLDER=$(pwd)

export NUM_WORKERS=8
export GLOBAL_BATCH=16
TOTAL_GPUS=$(( SLURM_NNODES * SLURM_GPUS_ON_NODE ))
export MICRO_BATCH=$(( GLOBAL_BATCH / TOTAL_GPUS ))
export TP=1
export CP=1

# Configuration
export CONFIG_PATH="$LOCAL_FOLDER/conf"
export CONFIG_NAME="config.yaml"
export INPUT_CFG_TRAIN="$CONFIG_PATH/input_cfg_train.yaml"
export INPUT_CFG_VAL="$CONFIG_PATH/input_cfg_val.yaml"

# Environment settings
export HYDRA_FULL_ERROR=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export NEMO_MODELS_CACHE=$TMP_CACHE

if (( TOTAL_GPUS > GLOBAL_BATCH )); then
  echo "❌ TOTAL_GPUS ($TOTAL_GPUS) > GLOBAL_BATCH ($GLOBAL_BATCH)"
  exit 1
fi

echo "Running on: $HOSTNAME"

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

DISTRIBUTED_ARGS=" \
       --nproc_per_node $SLURM_GPUS_ON_NODE \
       --nnodes $SLURM_NNODES \
       --node_rank $SLURM_NODEID \
       --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
       --rdzv_backend c10d \
       --max_restarts 0 \
       "


# Launch training
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
    trainer.max_steps=15 \
    trainer.num_nodes=$SLURM_NNODES \
    name=$EXPERIMENT_NAME \
    max_time_per_run="00:00:30:00" \
    logger.log_dir=$EXPERIMENT_FOLDER \
    resume.resume_if_exists=true \
    name=$EXPERIMENT_NAME \
    # resume.resume_from_directory=$EXPERIMENT_FOLDER/$EXPERIMENT_NAME \