
# bash eval.sh 1002_twist2 cuda:1

# export LD_LIBRARY_PATH=/opt/conda/envs/twist2/lib:$LD_LIBRARY_PATH

# Set LD_LIBRARY_PATH for isaacgym
export LD_LIBRARY_PATH=/home/xhz/anaconda3/envs/twist2/lib:$LD_LIBRARY_PATH

# 相对路径（相对于TWIST2根目录）
motion_file_rel="../GMR/data/TWIST2_dataset/example_motions_g1/0807_yanjie_walk_004.pkl"
# 转换为绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
motion_file="$(cd "$SCRIPT_DIR" && realpath "$motion_file_rel")"

exptid=$1
device=$2
robot_name=${3:-"g1"}
num_envs=${4:-1}

task_name="${robot_name}_stu_future"
proj_name="${robot_name}_stu_future"

cd legged_gym/legged_gym/scripts

echo "Evaluating student policy with future motion support..."
echo "Robot: ${robot_name}"
echo "Num Envs: ${num_envs}"
echo "Task: ${task_name}"
echo "Project: ${proj_name}"
echo "Experiment ID: ${exptid}"
echo ""

# Run the evaluation script
python play.py --task "${task_name}" \
               --proj_name "${proj_name}" \
               --teacher_exptid "None" \
               --exptid "${exptid}" \
               --num_envs "${num_envs}" \
               --record_video \
               --device "${device}" \
               --env.motion.motion_file "${motion_file}" \
               # --checkpoint 13000 \
               # --record_log \
               # --use_jit \
               # --teleop_mode