#!/bin/bash

# Script to convert student policy with future motion support to ONNX

# Usage: bash to_onnx.sh <robot_type> <checkpoint_path>
# robot_type: g1 or taks_t1

robot_type=$1
ckpt_path=$2

if [ -z "$robot_type" ] || [ -z "$ckpt_path" ]; then
    echo "Usage: bash to_onnx.sh <robot_type> <checkpoint_path>"
    echo "  robot_type: g1 or taks_t1"
    exit 1
fi

# 记录调用时所在目录，用于解析相对路径
CALL_DIR="$(pwd)"

cd legged_gym/legged_gym/scripts

# 将用户传入的相对路径转换为绝对路径（相对于调用目录解析）
ckpt_path_abs="$(cd "$CALL_DIR" && realpath "$ckpt_path")"

# Run the correct ONNX conversion script based on robot type
if [ "$robot_type" = "g1" ]; then
    python save_onnx_g1.py --ckpt_path "${ckpt_path_abs}"
elif [ "$robot_type" = "taks_t1" ]; then
    python save_onnx_taks_t1.py --ckpt_path "${ckpt_path_abs}"
else
    echo "Error: Unknown robot type '$robot_type'. Use 'g1' or 'taks_t1'."
    exit 1
fi