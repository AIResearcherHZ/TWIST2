export LD_LIBRARY_PATH=/opt/conda/envs/twist2/lib:$LD_LIBRARY_PATH

# Set LD_LIBRARY_PATH for isaacgym
# export LD_LIBRARY_PATH=/home/xhz/anaconda3/envs/twist2/lib:$LD_LIBRARY_PATH

source ~/miniconda3/bin/activate twist2

SCRIPT_DIR=$(dirname $(realpath $0))
ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx

# change the network interface name to your own that connects to the robot
# net=enp0s31f6
net=eno1

cd deploy_real

python server_low_level_g1_real.py \
    --policy ${ckpt_path} \
    --net ${net} \
    --device cuda \
    --use_hand \
    # --smooth_body 0.5
    # --record_proprio \
