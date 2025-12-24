#!/bin/bash

# export LD_LIBRARY_PATH=/opt/conda/envs/twist2/lib:$LD_LIBRARY_PATH

# Set LD_LIBRARY_PATH for isaacgym
export LD_LIBRARY_PATH=/home/xhz/anaconda3/envs/twist2/lib:$LD_LIBRARY_PATH

# Robot type: unitree_g1_with_hands or taks_t1
# Can be overridden by command line argument: ./run_motion_server.sh taks_t1
robot_type="${1:-unitree_g1_with_hands}"

# Select motion file based on robot type
if [ "$robot_type" == "taks_t1" ]; then
    # Taks_T1 uses 32 DOF motion files
    # motion_file="../../GMR/data/TWIST2_dataset/example_motions_taks_t1/0807_yanjie_walk_001.pkl"
    motion_file="../../GMR/data/TWIST2_dataset/v1_v2_v3_taks_t1/0811_static_001.pkl"
else
    # G1 uses 29 DOF motion files
    motion_file="../../GMR/data/TWIST2_dataset/example_motions_g1/0807_yanjie_walk_001.pkl"
fi

# Change to deploy_real directory
cd deploy_real

# by default we use our own laptop as the redis server
redis_ip="localhost"
# this is my unitree g1's ip in wifi
# redis_ip="192.168.110.24"

# Run the motion server
python server_motion_lib.py \
    --motion_file ${motion_file} \
    --robot ${robot_type} \
    --vis \
    --redis_ip ${redis_ip}
    # --send_start_frame_as_end_frame \
    # --use_remote_control \
