#!/bin/bash

# Usage: bash train.sh <experiment_id> <device> [robot_name] [num_envs]
# 示例: bash train.sh 1215_twist2 cuda:0 taks_t1 10000

export LD_LIBRARY_PATH=/opt/conda/envs/twist2/lib:$LD_LIBRARY_PATH

# Set LD_LIBRARY_PATH for isaacgym
export LD_LIBRARY_PATH=/home/xhz/anaconda3/envs/twist2/lib:$LD_LIBRARY_PATH


cd legged_gym/legged_gym/scripts

exptid=$1
device=$2
robot_name=${3:-"g1"}    # 默认: g1
num_envs=${4:-10000}          # 默认: 10000

task_name="${robot_name}_stu_future"
proj_name="${robot_name}_stu_future"

echo "=========================================="
echo "Robot: ${robot_name}"
echo "Num Envs: ${num_envs}"
echo "Experiment ID: ${exptid}"
echo "Device: ${device}"
echo "=========================================="

python train.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --num_envs "${num_envs}" \
                --exptid "${exptid}" \
                --device "${device}" \
                --teacher_exptid "None" \
                --resume
                # --debug  # 调试模式（可视化）