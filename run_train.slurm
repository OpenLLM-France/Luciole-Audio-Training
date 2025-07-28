#!/bin/bash
#SBATCH --job-name=<YOUR_JOB_NAME>
#SBATCH --output=logs/log_ddp_lucas_%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:4             
#SBATCH --ntasks-per-node=4      
#SBATCH --cpus-per-task=16       
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --constraint=h100
#SBATCH --account=<YOUR_ACCOUNT>  # Replace with your account
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<YOUR_EMAIL>  # Replace with your email

# -------------------- Load Environment --------------------
module purge
module load arch/h100
module load pytorch-gpu/py3/2.7.0
nvidia-smi
echo "Running on: $HOSTNAME"

# -------------------- Automatic Path Detection --------------------
LOCAL_FOLDER=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# -------------------- Environment Variables ---------------
export PYTHONPATH="$PYTHONPATH:$LOCAL_FOLDER"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

# Hugging Face offline cache
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$HOME/.cache/huggingface"
export XDG_CACHE_HOME="$HOME/.cache"

# -------------------- Define Paths ------------------------
code_dir=$LOCAL_FOLDER
data_dir="/path/to/your/dataset" 
output_dir="/path/to/output"

# -------------------- Launch Training ---------------------
# MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
# MASTER_PORT=29500
GPUS_PER_NODE=4
export OMP_NUM_THREADS=$GPUS_PER_NODE

torchrun \
  --nproc_per_node=$GPUS_PER_NODE \
  $code_dir/lightning_training.py \
    --dataset_dirs $data_dir \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --save_every_n_steps 100 \
    --precision bf16-mixed \
    --epochs 20 \
    --num_workers 7 \
    --devices $GPUS_PER_NODE \
    --num_nodes 1 \
    --accelerator gpu \
    --strategy ddp \
    --output_dir $output_dir \
    --auto_resume