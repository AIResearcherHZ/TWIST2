export LD_LIBRARY_PATH=/opt/conda/envs/twist2/lib:$LD_LIBRARY_PATH

# Set LD_LIBRARY_PATH for isaacgym
# export LD_LIBRARY_PATH=/home/xhz/anaconda3/envs/twist2/lib:$LD_LIBRARY_PATH


SCRIPT_DIR=$(dirname $(realpath $0))
ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx

cd deploy_real

python server_low_level_g1_sim.py \
    --xml ../assets/g1/g1_sim2sim_29dof.xml \
    --policy ${ckpt_path} \
    --device cuda \
    --measure_fps 1 \
    --policy_frequency 100 \
    --limit_fps 1 \
    # --record_proprio \
