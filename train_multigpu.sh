#!/bin/bash

# TWIST2 单机多 GPU 分布式训练脚本
# 针对多 GPU 训练做了常用加速/稳定性优化：
#   - 更高效的梯度同步（单次 all_reduce）
#   - 混合精度训练（AMP）
#   - NCCL 通信优化
#   - 更合理的进程同步
#
# 用法：bash train_multigpu.sh <experiment_id> [num_gpus] [total_envs]
# 示例：
#   bash train_multigpu.sh 1204_twist2_multigpu 2        # 2 张 GPU，总 env 默认 8196
#   bash train_multigpu.sh 1204_twist2_multigpu 4 16384  # 4 张 GPU，总 env 16384

set -e  # 出错即退出

# ============================================
# 环境变量设置
# ============================================
export LD_LIBRARY_PATH=/opt/conda/envs/twist2/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/home/xhz/anaconda3/envs/twist2/lib:$LD_LIBRARY_PATH

# 关闭 MPI 组件加载告警（不影响功能，仅减少日志噪声）
export OMPI_MCA_btl_base_warn_component_load=0

# CUDA 优化
export CUDA_DEVICE_MAX_CONNECTIONS=1  # 降低连接数，偏向单流场景（有助于稳定/吞吐）
export CUDA_LAUNCH_BLOCKING=0  # 保持异步 kernel launch（不要开启阻塞）

# NCCL 多 GPU 通信优化
export NCCL_IB_DISABLE=1  # 单机通常无需 IB（避免误探测带来的问题）
export NCCL_P2P_LEVEL=NVL  # 有 NVLink 时优先使用（否则自动 fallback）
export NCCL_ASYNC_ERROR_HANDLING=1  # 异步错误处理，便于定位 NCCL 问题
export NCCL_DEBUG=WARN  # 需要排查 NCCL 时可改成 INFO

# PyTorch 显存分配优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # 降低碎片化概率

exptid=$1
num_gpus=${2:-$(nvidia-smi -L | wc -l)}  # 自动检测 GPU 数量
total_envs=${3:-8196}  # 总环境数默认值

robot_name="g1"
task_name="${robot_name}_stu_future"
proj_name="${robot_name}_stu_future"

# 计算每张 GPU 的环境数（确保均匀分配）
envs_per_gpu=$((total_envs / num_gpus))
actual_total=$((envs_per_gpu * num_gpus))

if [ "$actual_total" -ne "$total_envs" ]; then
    echo "Warning: Adjusting total_envs from ${total_envs} to ${actual_total} for even distribution"
    total_envs=$actual_total
fi

echo "============================================"
echo "Multi-GPU Distributed Training Configuration"
echo "============================================"
echo "Experiment ID: ${exptid}"
echo "Number of GPUs: ${num_gpus}"
echo "Total Environments: ${total_envs}"
echo "Environments per GPU: ${envs_per_gpu}"
echo "NCCL P2P Level: ${NCCL_P2P_LEVEL}"
echo "AMP Enabled: Yes"
echo "============================================"

# 判断单卡/多卡
if [ "$num_gpus" -eq 1 ]; then
    echo "Only 1 GPU detected. Using single-GPU training mode."
    cd legged_gym/legged_gym/scripts
    python train.py \
        --task "${task_name}" \
        --proj_name "${proj_name}" \
        --num_envs ${total_envs} \
        --exptid "${exptid}" \
        --headless \
        --teacher_exptid "None" \
        --debug  # 调试模式（可视化）
        # --resume \
else
    echo "Using distributed training with ${num_gpus} GPUs."
    echo "Starting at $(date)"
    cd legged_gym/legged_gym/scripts
    
    # 使用 torchrun 启动分布式训练（包含更稳的 rendezvous 配置）
    # --rdzv_backend=c10d: 使用 PyTorch 原生 rendezvous 后端
    # --rdzv_endpoint: 进程协同端口（单机固定即可）
    # --max_restarts=0: worker 出错不自动重启（便于发现真实错误）
    torchrun \
        --standalone \
        --nproc_per_node=${num_gpus} \
        --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:29500 \
        --max_restarts=0 \
        train_distributed.py \
        --task "${task_name}" \
        --proj_name "${proj_name}" \
        --num_envs ${envs_per_gpu} \
        --exptid "${exptid}" \
        --headless \
        --teacher_exptid "None" \
        --debug  # 调试模式（可视化）
        # --resume \
    
    echo "Training finished at $(date)"
fi
