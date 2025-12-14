#!/bin/bash

# TWIST2 多 GPU 训练脚本
# 用法: bash train_multigpu.sh <experiment_id> [num_gpus] [total_envs]
# 示例:
#   bash train_multigpu.sh 1204_twist2_multigpu 2        # 2 张 GPU，总环境数默认 8196
#   bash train_multigpu.sh 1204_twist2_multigpu 4 16384  # 4 张 GPU，总环境数 16384

export LD_LIBRARY_PATH=/opt/conda/envs/twist2/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/home/xhz/anaconda3/envs/twist2/lib:$LD_LIBRARY_PATH

# 关闭 MPI 警告
export OMPI_MCA_btl_base_warn_component_load=0

exptid=$1
num_gpus=${2:-$(nvidia-smi -L | wc -l)}  # 自动检测 GPU 数量
total_envs=${3:-8196}  # 默认总环境数

robot_name="taks_t1"
task_name="${robot_name}_stu_future"
proj_name="${robot_name}_stu_future"

# 计算每张 GPU 分配的环境数
envs_per_gpu=$((total_envs / num_gpus))

echo "============================================"
echo "Multi-GPU Training Configuration"
echo "============================================"
echo "Experiment ID: ${exptid}"
echo "Number of GPUs: ${num_gpus}"
echo "Total Environments: ${total_envs}"
echo "Environments per GPU: ${envs_per_gpu}"
echo "============================================"

# 判断是否为多 GPU
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
        # --debug  # 调试模式（可视化）
        # --resume \
else
    echo "Using distributed training with ${num_gpus} GPUs."
    cd legged_gym/legged_gym/scripts
    
    # 使用 torchrun 进行分布式训练
    torchrun --standalone --nproc_per_node=${num_gpus} \
        train_distributed.py \
        --task "${task_name}" \
        --proj_name "${proj_name}" \
        --num_envs ${envs_per_gpu} \
        --exptid "${exptid}" \
        --headless \
        --teacher_exptid "None"
        # --debug  # 调试模式（可视化）
        # --resume \
fi