#!/bin/bash

# Usage: bash train.sh <experiment_id> <device>

# bash train.sh 1103_twist2 cuda:0

export LD_LIBRARY_PATH=/opt/conda/envs/twist2/lib:$LD_LIBRARY_PATH

# Set LD_LIBRARY_PATH for isaacgym
# export LD_LIBRARY_PATH=/home/xhz/anaconda3/envs/twist2/lib:$LD_LIBRARY_PATH


cd legged_gym/legged_gym/scripts

robot_name="g1"
exptid=$1
device=$2

task_name="${robot_name}_stu_future"
proj_name="${robot_name}_stu_future"

python train.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --num_envs 8196 \
                --exptid "${exptid}" \
                --device "${device}" \
                --teacher_exptid "None" \
                --resume
                # --debug  # 调试模式（可视化）
