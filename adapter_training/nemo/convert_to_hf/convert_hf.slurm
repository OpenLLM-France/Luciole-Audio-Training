#!/bin/bash
#SBATCH --job-name=NemoSpeechLLM-ExportHF
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=log/export_hf_log.out
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --account=wuh@h100
#SBATCH --constraint=h100

# --------------------
# Load required modules
# --------------------
module purge
module load arch/h100 nemo/2.3.1

# --------------------
# Paths & Variables
# --------------------
TMP_CACHE="$SCRATCH/speechlm/.cache"
EXPERIMENT_FOLDER="$SCRATCH/speechlm/nemo_experiments"
EXPERIMENT_NAME="Whisper_Llama-3.2-1B-8gpu-64gbs"

LOCAL_FOLDER=$(pwd)

export CONFIG_PATH="$EXPERIMENT_FOLDER/$EXPERIMENT_NAME/run_0"
export CONFIG_NAME="config.yaml"

export NUM_WORKERS=30

# --------------------
# Environment settings
# --------------------
export HYDRA_FULL_ERROR=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NEMO_MODELS_CACHE=$TMP_CACHE

# Memory optimization
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "➡️ Running on: $HOSTNAME"
echo "➡️ GPUs requested: $SLURM_GPUS_ON_NODE"
export TP=1                 # Split model across 4 GPUs
export PP=1                 # No pipeline parallelism
export CP=1                 # No context parallelism
# --------------------
# Run export script (single GPU, no torchrun needed)
# --------------------
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

# Launch training with memory optimizations
srun torchrun $DISTRIBUTED_ARGS $LOCAL_FOLDER/convert_to_hf_complete.py \
    --config-path $CONFIG_PATH \
    --config-name $CONFIG_NAME \
    ~data.train_ds \
    ~data.validation_ds \
    trainer.devices=1 \
    trainer.num_nodes=1 \
    resume.resume_if_exists=true \
    logger.log_dir=$EXPERIMENT_FOLDER \
    strategy.tensor_model_parallel_size=$TP \
    strategy.pipeline_model_parallel_size=$PP \
    strategy.context_parallel_size=$CP \
    ++hf_output_dir=$EXPERIMENT_FOLDER/${EXPERIMENT_NAME}-export-HF2 \
    name=$EXPERIMENT_NAME
