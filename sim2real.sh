export LD_LIBRARY_PATH=/opt/conda/envs/twist2/lib:$LD_LIBRARY_PATH

# Set LD_LIBRARY_PATH for isaacgym
# export LD_LIBRARY_PATH=/home/xhz/anaconda3/envs/twist2/lib:$LD_LIBRARY_PATH

source ~/anaconda3/bin/activate twist2

SCRIPT_DIR=$(dirname $(realpath $0))
# 根据机器人类型选择不同的policy路径
# g1: 使用原有ckpt
# taks_t1: 使用指定的未来学生模型
# 如需自定义，请修改对应分支的路径
# ============ 机器人类型选择 ============
# 可选: g1 或 taks_t1
ROBOT_TYPE=${1:-g1}

if [ "$ROBOT_TYPE" = "g1" ]; then
    ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx
elif [ "$ROBOT_TYPE" = "taks_t1" ]; then
    ckpt_path=${SCRIPT_DIR}/legged_gym/logs/taks_t1_stu_future/1227_taks_t1/model_100.onnx
else
    echo "错误: 未知的机器人类型 '$ROBOT_TYPE'"
    echo "用法: $0 [g1|taks_t1]"
    echo "  默认: g1"
    exit 1
fi

# ============ 网络接口配置 ============
# change the network interface name to your own that connects to the robot
# net=enp0s31f6
net=eno1

# ============ Taks-T1 服务器配置 ============
server_ip=192.168.1.208
cmd_port=5555

cd deploy_real

if [ "$ROBOT_TYPE" = "g1" ]; then
    echo "启动 G1 机器人 sim2real..."
    python server_low_level_g1_real.py \
        --policy ${ckpt_path} \
        --net ${net} \
        --device cuda \
        --use_hand
        # --smooth_body 0.5
        # --record_proprio
elif [ "$ROBOT_TYPE" = "taks_t1" ]; then
    echo "启动 Taks-T1 机器人 sim2real..."
    python server_low_level_taks_t1_real.py \
        --policy ${ckpt_path} \
        --device cuda \
        --server_ip ${server_ip} \
        --cmd_port ${cmd_port}
else
    echo "错误: 未知的机器人类型 '$ROBOT_TYPE'"
    echo "用法: $0 [g1|taks_t1]"
    echo "  默认: g1"
    exit 1
fi
