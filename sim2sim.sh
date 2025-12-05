#!/bin/bash
# export LD_LIBRARY_PATH=/opt/conda/envs/twist2/lib:$LD_LIBRARY_PATH

# Set LD_LIBRARY_PATH for isaacgym
export LD_LIBRARY_PATH=/home/xhz/anaconda3/envs/twist2/lib:$LD_LIBRARY_PATH

SCRIPT_DIR=$(dirname $(realpath $0))

# 使用说明
usage() {
    echo "Usage: $0 [robot_type] [ckpt_path]"
    echo "  robot_type: g1 (default) or taks_t1"
    echo "  ckpt_path:  path to onnx checkpoint (optional)"
    echo ""
    echo "Examples:"
    echo "  $0                          # 使用g1和默认checkpoint"
    echo "  $0 g1                       # 使用g1和默认checkpoint"
    echo "  $0 taks_t1                  # 使用taks_t1和默认checkpoint"
    echo "  $0 g1 path/to/model.onnx    # 使用g1和指定checkpoint"
    echo "  $0 taks_t1 path/to/model.onnx # 使用taks_t1和指定checkpoint"
    exit 1
}

# 解析参数
robot_type=${1:-g1}
ckpt_path=${2:-${SCRIPT_DIR}/assets/ckpts/twist2_1017_25k.onnx}

# 根据机器人类型选择XML文件
case $robot_type in
    g1)
        xml_file="../assets/g1/g1_sim2sim_29dof.xml"
        echo "Using G1 robot model"
        ;;
    taks_t1)
        xml_file="../assets/Taks_T1/Taks_T1_sim2sim.xml"
        echo "Using Taks_T1 robot model"
        ;;
    -h|--help)
        usage
        ;;
    *)
        echo "Error: Unknown robot type '$robot_type'"
        echo "Supported types: g1, taks_t1"
        usage
        ;;
esac

# 将传入路径视为相对于调用目录的相对路径，并转成绝对路径
CALL_DIR="$(pwd)"
ckpt_path="$(cd "$CALL_DIR" && realpath "$ckpt_path")"

echo "Robot type: $robot_type"
echo "XML file: $xml_file"
echo "Checkpoint: $ckpt_path"

cd deploy_real

# 根据机器人类型选择对应的服务器脚本
case $robot_type in
    g1)
        python server_low_level_g1_sim.py \
            --xml ${xml_file} \
            --policy ${ckpt_path} \
            --device cuda \
            --measure_fps 1 \
            --policy_frequency 100 \
            --limit_fps 1
        ;;
    taks_t1)
        python server_low_level_taks_t1_sim.py \
            --xml ${xml_file} \
            --policy ${ckpt_path} \
            --device cuda \
            --measure_fps 1 \
            --policy_frequency 100 \
            --limit_fps 1
        ;;
esac
