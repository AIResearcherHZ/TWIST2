from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfg, HumanoidMimicCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR


class TaksT1MimicCfg(HumanoidMimicCfg):
    class env(HumanoidMimicCfg.env):
        tar_motion_steps_priv = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                         50, 55, 60, 65, 70, 75, 80, 85, 90, 95,]
        
        num_envs = 4096
        num_actions = 32  # Taks_T1 has 32 DOF (including neck)
        n_priv = 0
        n_mimic_obs = 3*4 + 32  # 32 for dof pos
        n_proprio = len(tar_motion_steps_priv) * n_mimic_obs + 3 + 2 + 3*num_actions
        n_priv_latent = 4 + 1 + 2*num_actions
        extra_critic_obs = 3
        history_len = 10
        
        num_observations = n_proprio + n_priv_latent + history_len*n_proprio + n_priv + extra_critic_obs 
        num_privileged_obs = None

        env_spacing = 3.
        send_timeouts = True
        episode_length_s = 10
        
        randomize_start_pos = True
        randomize_start_yaw = False
        
        history_encoding = True
        contact_buf_len = 10
        
        normalize_obs = True
        
        enable_early_termination = True
        pose_termination = True
        pose_termination_dist = 0.7
        root_tracking_termination_dist = 0.8
        rand_reset = True
        track_root = False
        dof_err_w = [1.0, 1.0, 1.0, 1.0, 0.1, 0.1,  # Left Leg
                     1.0, 1.0, 1.0, 1.0, 0.1, 0.1,  # Right Leg
                     1.0, 1.0, 1.0,  # waist yaw, roll, pitch
                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # Left Arm (7 DOF with wrist)
                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # Right Arm (7 DOF with wrist)
                     1.0, 1.0, 1.0,  # Neck (yaw, roll, pitch)
                     ]
        
        global_obs = False
    
    class terrain(HumanoidMimicCfg.terrain):
        mesh_type = 'trimesh'
        height = [0, 0.00]
        horizontal_scale = 0.1
    
    class init_state(HumanoidMimicCfg.init_state):
        pos = [0, 0, 0.75]
        default_joint_angles = {
            'left_hip_pitch_joint': 0.0,
            'left_hip_roll_joint': 0.0,
            'left_hip_yaw_joint': 0.0,
            'left_knee_joint': 0.0,
            'left_ankle_pitch_joint': 0.0,
            'left_ankle_roll_joint': 0.0,
            
            'right_hip_pitch_joint': 0.0,
            'right_hip_roll_joint': 0.0,
            'right_hip_yaw_joint': 0.0,
            'right_knee_joint': 0.0,
            'right_ankle_pitch_joint': 0.0,
            'right_ankle_roll_joint': 0.0,
            
            'waist_yaw_joint': 0.0,
            'waist_roll_joint': 0.0,
            'waist_pitch_joint': 0.0,
            
            'left_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.0,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 0.0,
            'left_wrist_roll_joint': 0.0,
            'left_wrist_yaw_joint': 0.0,
            'left_wrist_pitch_joint': 0.0,
            
            'right_shoulder_pitch_joint': 0.0,
            'right_shoulder_roll_joint': 0.0,
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_joint': 0.0,
            'right_wrist_roll_joint': 0.0,
            'right_wrist_yaw_joint': 0.0,
            'right_wrist_pitch_joint': 0.0,
            
            'neck_yaw_joint': 0.0,
            'neck_roll_joint': 0.0,
            'neck_pitch_joint': 0.0,
        }
    
    class control(HumanoidMimicCfg.control):
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     'waist': 150,
                     'shoulder': 40,
                     'elbow': 40,
                     'wrist': 20,
                     'neck': 20,
                     }
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     'waist': 4,
                     'shoulder': 5,
                     'elbow': 5,
                     'wrist': 2,
                     'neck': 2,
                     }
        
        action_scale = 0.5
        decimation = 10
    
    class sim(HumanoidMimicCfg.sim):
        dt = 0.002
        
    class normalization(HumanoidMimicCfg.normalization):
        clip_actions = 5.0
    
    class asset(HumanoidMimicCfg.asset):
        file = f'{LEGGED_GYM_ROOT_DIR}/../assets/Taks_T1/Taks_T1_sim2sim.xml'
        # self_collisions = 1  # 1 to disable self-collisions
        
        torso_name: str = 'pelvis'
        chest_name: str = 'imu_in_torso'

        thigh_name: str = 'hip'
        shank_name: str = 'knee'
        foot_name: str = 'ankle_roll_link'
        waist_name: list = ['torso_link', 'waist_roll_link', 'waist_yaw_link']
        upper_arm_name: str = 'shoulder_roll_link'
        lower_arm_name: str = 'elbow_link'
        hand_name: str = 'wrist_pitch_link'

        feet_bodies = ['left_ankle_roll_link', 'right_ankle_roll_link']
        n_lower_body_dofs: int = 12

        penalize_contacts_on = ["shoulder", "elbow", "hip", "knee"]
        terminate_after_contacts_on = ['torso_link']
        
        # Inertia values for Taks_T1
        dof_armature = [0.0103, 0.0251, 0.0103, 0.0251, 0.003597, 0.003597] * 2 + \
                       [0.0103] * 3 + \
                       [0.003597] * 7 * 2 + \
                       [0.005] * 3  # neck
        
        collapse_fixed_joints = False
    
    class rewards(HumanoidMimicCfg.rewards):
        regularization_names = [
                        "feet_stumble",
                        "feet_contact_forces",
                        "lin_vel_z",
                        "ang_vel_xy",
                        "orientation",
                        "dof_pos_limits",
                        "dof_torque_limits",
                        "collision",
                        "torque_penalty",
                        "thigh_torque_roll_yaw",
                        "thigh_roll_yaw_acc",
                        "dof_acc",
                        "dof_vel",
                        "action_rate",
                        ]
        regularization_scale = 1.0
        regularization_scale_range = [0.8, 2.0]
        regularization_scale_curriculum = False
        regularization_scale_gamma = 0.0001
        
        class scales:
            tracking_joint_dof = 0.6
            tracking_joint_vel = 0.2
            tracking_root_translation = 0.6
            tracking_root_rotation = 0.6
            tracking_root_vel = 1.0
            tracking_keybody_pos = 2.0

            feet_slip = -0.1
            feet_contact_forces = -5e-4      
            feet_stumble = -1.25
            
            dof_pos_limits = -5.0
            dof_torque_limits = -1.0
            
            dof_vel = -1e-4
            dof_acc = -5e-8
            action_rate = -0.01
            
            feet_air_time = 5.0
            
            ang_vel_xy = -0.01
            
            ankle_dof_acc = -5e-8 * 2
            ankle_dof_vel = -1e-4 * 2
            idle_penalty = -0.001

        min_dist = 0.1
        max_dist = 0.4
        max_knee_dist = 0.4
        target_feet_height = 0.07
        only_positive_rewards = False
        tracking_sigma = 0.2
        tracking_sigma_ang = 0.125
        max_contact_force = 350
        soft_torque_limit = 0.95
        torque_safety_limit = 0.9
        
        termination_roll = 1.5
        termination_pitch = 1.5
        root_height_diff_threshold = 0.3

    class domain_rand:
        domain_rand_general = True
        
        randomize_gravity = (True and domain_rand_general)
        gravity_rand_interval_s = 4
        gravity_range = (-0.1, 0.1)
        
        randomize_friction = (True and domain_rand_general)
        friction_range = [0.1, 2.]
        
        randomize_base_mass = (True and domain_rand_general)
        added_mass_range = [-3., 3]
        
        randomize_base_com = (True and domain_rand_general)
        added_com_range = [-0.05, 0.05]
        
        push_robots = (True and domain_rand_general)
        push_interval_s = 4
        max_push_vel_xy = 1.0
        
        push_end_effector = (True and domain_rand_general)
        push_end_effector_interval_s = 2
        max_push_force_end_effector = 20.0

        randomize_motor = (True and domain_rand_general)
        motor_strength_range = [0.8, 1.2]

        action_delay = (True and domain_rand_general)
        action_buf_len = 8
        
        # ==================== 新增鲁棒性随机化 ====================
        # 动作噪声 - 模拟控制信号不完美（量化误差、通讯抖动）
        action_noise = (True and domain_rand_general)
        action_noise_std = 0.01
        
        # 关节编码器噪声 - 模拟编码器测量误差和零点偏移
        encoder_noise = (True and domain_rand_general)
        encoder_pos_noise_std = 0.005  # 位置噪声标准差 (rad)
        encoder_vel_noise_std = 0.01   # 速度噪声标准差 (rad/s)
        encoder_pos_bias_range = [-0.01, 0.01]  # 位置偏置范围 (rad)
        encoder_vel_bias_range = [-0.02, 0.02]  # 速度偏置范围 (rad/s)
        
        # IMU噪声和漂移 - 模拟真实IMU的测量特性
        imu_noise = (True and domain_rand_general)
        imu_ang_vel_noise_std = 0.02  # 角速度噪声 (rad/s)
        imu_lin_acc_noise_std = 0.05  # 线加速度噪声 (m/s^2)
        imu_ang_vel_bias_range = [-0.1, 0.1]
        imu_lin_acc_bias_range = [-0.2, 0.2]
        imu_bias_drift_std = 0.01  # 偏置漂移
        
        # 观测丢包 - 模拟传感器偶发失效
        observation_dropout = (True and domain_rand_general)
        observation_dropout_prob = 0.001  # 每个维度丢包概率 0.1%
        observation_dropout_mode = 'hold'  # 丢包时保持上一帧值 ('hold' or 'zero')
        
        # 关节故障 - 模拟电机故障（极低概率）
        joint_failure = (False and domain_rand_general)  # 默认关闭，太激进
        joint_failure_prob = 0.0001  # 每个关节失效概率 0.01%
        joint_failure_mode = 'weak'  # 弱化模式（扭矩衰减）
        joint_failure_weak_factor = 0.5  # 衰减因子
        
        # 传感器延迟尖峰 - 模拟偶发的通讯阻塞
        sensor_latency_spike = (True and domain_rand_general)
        sensor_latency_spike_prob = 0.001  # 0.1%概率发生延迟尖峰
        sensor_latency_max_steps = 10  # 最大延迟10步
        
        # 重力方向偏置 - 模拟基座倾斜/坡度
        slope_randomization = (True and domain_rand_general)
        gravity_bias_x_range = [-0.1, 0.1]  # x方向重力偏置 (m/s^2)
        gravity_bias_y_range = [-0.1, 0.1]  # y方向重力偏置 (m/s^2)
        gravity_bias_z_range = [-0.05, 0.05]  # z方向重力偏置 (m/s^2)
    
    class noise(HumanoidMimicCfg.noise):
        add_noise = True
        noise_increasing_steps = 3000
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.1
            lin_vel = 0.1
            ang_vel = 0.1
            gravity = 0.05
            imu = 0.1
    
    class evaluations:
        tracking_joint_dof = True
        tracking_joint_vel = True
        tracking_root_translation = True
        tracking_root_rotation = True
        tracking_root_vel = True
        tracking_root_ang_vel = True
        tracking_keybody_pos = True
        tracking_root_pose_delta_local = True
        tracking_root_rotation_delta_local = True

    class motion(HumanoidMimicCfg.motion):
        motion_curriculum = True
        motion_curriculum_gamma = 0.01
        key_bodies = ["left_wrist_pitch_link", "right_wrist_pitch_link", 
                      "left_ankle_roll_link", "right_ankle_roll_link", 
                      "left_knee_link", "right_knee_link", 
                      "left_elbow_link", "right_elbow_link", 
                      "head_mocap"]
        
        motion_file = f"../../../../motion_data/LAFAN1_taks_t1_gmr/dance1_subject2.pkl"

        reset_consec_frames = 30


class TaksT1MimicCfgPPO(HumanoidMimicCfgPPO):
    seed = 1
    class runner(HumanoidMimicCfgPPO.runner):
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'PPO'
        runner_class_name = 'OnPolicyRunner'
        max_iterations = 10001

        save_interval = 500
        experiment_name = 'test'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
    
    class algorithm(HumanoidMimicCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005
    
    class policy(HumanoidMimicCfgPPO.policy):
        action_std = [0.7] * 12 + [0.4] * 3 + [0.5] * 14 + [0.3] * 3  # 32 DOF
        init_noise_std = 0.8
        obs_context_len = 11
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = 'silu'
