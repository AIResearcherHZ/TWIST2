# export LD_LIBRARY_PATH=/opt/conda/envs/twist2/lib:$LD_LIBRARY_PATH

# Set LD_LIBRARY_PATH for isaacgym
export LD_LIBRARY_PATH=/home/xhz/anaconda3/envs/twist2/lib:$LD_LIBRARY_PATH


SCRIPT_DIR=$(dirname $(realpath $0))
# 默认模型
ckpt_path=${1:-${SCRIPT_DIR}/assets/ckpts/twist2_1017_25k.onnx}
# 将传入路径视为相对于调用目录的相对路径，并转成绝对路径
CALL_DIR="$(pwd)"
ckpt_path="$(cd "$CALL_DIR" && realpath "$ckpt_path")"

cd deploy_real

python server_low_level_g1_sim.py \
    --xml ../assets/g1/g1_sim2sim_29dof.xml \
    --policy ${ckpt_path} \
    --device cuda \
    --measure_fps 1 \
    --policy_frequency 100 \
    --limit_fps 1 \
    # --record_proprio \
