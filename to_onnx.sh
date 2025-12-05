#!/bin/bash

# Script to convert student policy with future motion support to ONNX

# bash to_onnx.sh $YOUR_POLICY_PATH

ckpt_path=$1

# 记录调用时所在目录，用于解析相对路径
CALL_DIR="$(pwd)"

cd legged_gym/legged_gym/scripts

# 将用户传入的相对路径转换为绝对路径（相对于调用目录解析）
ckpt_path_abs="$(cd "$CALL_DIR" && realpath "$ckpt_path")"

# Run the correct ONNX conversion script
python save_onnx.py --ckpt_path "${ckpt_path_abs}"