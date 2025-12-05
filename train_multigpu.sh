#!/bin/bash

# Multi-GPU Training Script for TWIST2
# Usage: bash train_multigpu.sh <experiment_id> [num_gpus] [total_envs]
# Example: 
#   bash train_multigpu.sh 1204_twist2_multigpu 2        # 2 GPUs, default 8196 envs
#   bash train_multigpu.sh 1204_twist2_multigpu 4 16384  # 4 GPUs, 16384 total envs

export LD_LIBRARY_PATH=/opt/conda/envs/twist2/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/home/xhz/anaconda3/envs/twist2/lib:$LD_LIBRARY_PATH

# Disable MPI warnings
export OMPI_MCA_btl_base_warn_component_load=0

exptid=$1
num_gpus=${2:-$(nvidia-smi -L | wc -l)}  # Auto-detect number of GPUs
total_envs=${3:-8196}  # Default total environments

robot_name="g1"
task_name="${robot_name}_stu_future"
proj_name="${robot_name}_stu_future"

# Calculate environments per GPU
envs_per_gpu=$((total_envs / num_gpus))

echo "============================================"
echo "Multi-GPU Training Configuration"
echo "============================================"
echo "Experiment ID: ${exptid}"
echo "Number of GPUs: ${num_gpus}"
echo "Total Environments: ${total_envs}"
echo "Environments per GPU: ${envs_per_gpu}"
echo "============================================"

# Check if we have multiple GPUs
if [ "$num_gpus" -eq 1 ]; then
    echo "Only 1 GPU detected. Using single-GPU training mode."
    cd legged_gym/legged_gym/scripts
    python train.py \
        --task "${task_name}" \
        --proj_name "${proj_name}" \
        --num_envs ${total_envs} \
        --exptid "${exptid}" \
        --headless \
        --teacher_exptid "None"
else
    echo "Using distributed training with ${num_gpus} GPUs."
    cd legged_gym/legged_gym/scripts
    
    # Use torchrun for distributed training
    torchrun --standalone --nproc_per_node=${num_gpus} \
        train_distributed.py \
        --task "${task_name}" \
        --proj_name "${proj_name}" \
        --num_envs ${envs_per_gpu} \
        --exptid "${exptid}" \
        --headless \
        --teacher_exptid "None"
fi
