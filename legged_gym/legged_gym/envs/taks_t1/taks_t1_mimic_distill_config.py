from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfg, HumanoidMimicCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR


class TaksT1MimicPrivCfg(HumanoidMimicCfg):
    class env(HumanoidMimicCfg.env):
        tar_motion_steps_priv = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                         50, 55, 60, 65, 70, 75, 80, 85, 90, 95,]
        tar_motion_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                         50, 55, 60, 65, 70, 75, 80, 85, 90, 95,]
        
        num_envs = 4096
        num_actions = 32  # Taks_T1 has 32 DOF
        obs_type = 'priv'
        n_priv_latent = 4 + 1 + 2*num_actions
        extra_critic_obs = 3
        n_priv = 0
        
        n_proprio = 3 + 2 + 3*num_actions
        n_priv_mimic_obs = len(tar_motion_steps_priv) * (21 + num_actions + 3*9)
        n_mimic_obs_single = 6 + 32
        n_mimic_obs = len(tar_motion_steps) * n_mimic_obs_single
        n_priv_info = 3 + 3 + 4 + 3*9 + 2 + 4 + 1 + 2*num_actions
        history_len = 10
        
        n_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info
        n_priv_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info
        
        num_observations = n_priv_obs_single
        num_privileged_obs = n_priv_obs_single

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
        rand_reset = True
        
        track_root = False
        root_tracking_termination_dist = 2.0
     
        dof_err_w = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                     1.0, 1.0, 1.0,
                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                     1.0, 1.0, 1.0,
                     ]
        
        global_obs = False
    
    class terrain(HumanoidMimicCfg.terrain):
        mesh_type = 'plane'
        height = [0, 0.00]
        horizontal_scale = 0.1
    
    class init_state(HumanoidMimicCfg.init_state):
        pos = [0, 0, 0.75]
        default_joint_angles = {
            'left_hip_pitch_joint': -0.2,
            'left_hip_roll_joint': 0.0,
            'left_hip_yaw_joint': 0.0,
            'left_knee_joint': 0.4,
            'left_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': 0.0,
            
            'right_hip_pitch_joint': -0.2,
            'right_hip_roll_joint': 0.0,
            'right_hip_yaw_joint': 0.0,
            'right_knee_joint': 0.4,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint': 0.0,
            
            'waist_yaw_joint': 0.0,
            'waist_roll_joint': 0.0,
            'waist_pitch_joint': 0.0,
            
            'left_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.4,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 0.8,
            'left_wrist_roll_joint': 0.0,
            'left_wrist_yaw_joint': 0.0,
            'left_wrist_pitch_joint': 0.0,
            
            'right_shoulder_pitch_joint': 0.0,
            'right_shoulder_roll_joint': -0.4,
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_joint': 0.8,
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
        damping = {'hip_yaw': 2,
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
        file = f'{LEGGED_GYM_ROOT_DIR}/../assets/Taks_T1/urdf/Taks_T1.urdf'
        
        torso_name: str = 'pelvis'
        chest_name: str = 'imu_in_torso'

        thigh_name: str = 'hip'
        shank_name: str = 'knee'
        foot_name: str = 'ankle_roll_link'
        waist_name: list = ['torso_link', 'waist_roll_link', 'waist_yaw_link']
        upper_arm_name: str = 'shoulder_roll_link'
        lower_arm_name: str = 'elbow_link'
        hand_name: list = ['right_wrist_pitch_link', 'left_wrist_pitch_link']

        feet_bodies = ['left_ankle_roll_link', 'right_ankle_roll_link']
        n_lower_body_dofs: int = 12

        penalize_contacts_on = ["shoulder", "elbow", "hip", "knee"]
        terminate_after_contacts_on = []
        
        dof_armature = [0.0103, 0.0251, 0.0103, 0.0251, 0.003597, 0.003597] * 2 + \
                       [0.0103] * 3 + \
                       [0.003597, 0.003597, 0.003597, 0.003597, 0.003597, 0.00425, 0.00425] * 2 + \
                       [0.005] * 3
        
        collapse_fixed_joints = False
    
    class rewards(HumanoidMimicCfg.rewards):
        regularization_names = []
        regularization_scale = 1.0
        regularization_scale_range = [0.8, 2.0]
        regularization_scale_curriculum = False
        regularization_scale_gamma = 0.0001
        
        class scales:
            tracking_joint_dof = 2.0
            tracking_joint_vel = 0.2
            tracking_root_translation_z = 1.0
            tracking_root_rotation = 1.0
            tracking_root_linear_vel = 1.0
            tracking_root_angular_vel = 1.0
            tracking_keybody_pos = 2.0
            tracking_keybody_pos_global = 2.0
            alive = 0.5
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

        min_dist = 0.1
        max_dist = 0.4
        max_knee_dist = 0.4
        feet_height_target = 0.2
        feet_air_time_target = 0.5
        only_positive_rewards = False
        tracking_sigma = 0.2
        tracking_sigma_ang = 0.125
        max_contact_force = 500
        soft_torque_limit = 0.95
        torque_safety_limit = 0.9
        
        termination_roll = 4.0
        termination_pitch = 4.0
        root_height_diff_threshold = 0.3

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
        
        push_end_effector = (False and domain_rand_general)
        push_end_effector_interval_s = 2
        max_push_force_end_effector = 10.0

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
        noise_increasing_steps = 50_000
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.1
            lin_vel = 0.1
            ang_vel = 0.1
            gravity = 0.05
            imu = 0.1
        
    class motion(HumanoidMimicCfg.motion):
        motion_curriculum = True
        motion_curriculum_gamma = 0.01
        reset_consec_frames = 30
        key_bodies = ["left_wrist_pitch_link", "right_wrist_pitch_link",
                      "left_ankle_roll_link", "right_ankle_roll_link",
                      "left_knee_link", "right_knee_link",
                      "left_elbow_link", "right_elbow_link",
                      "neck_pitch_link"]
        upper_key_bodies = ["left_wrist_pitch_link", "right_wrist_pitch_link",
                           "left_elbow_link", "right_elbow_link", "neck_pitch_link"]
        sample_ratio = 1.0
        motion_smooth = True
        motion_decompose = False

        motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/taks_t1_demo.yaml"


class TaksT1MimicStuCfg(TaksT1MimicPrivCfg):
    class env(TaksT1MimicPrivCfg.env):
        obs_type = 'student'
        tar_motion_steps = [1]
        n_mimic_obs_single = TaksT1MimicPrivCfg.env.n_mimic_obs_single
        n_mimic_obs = len(tar_motion_steps) * n_mimic_obs_single
        n_proprio = TaksT1MimicPrivCfg.env.n_proprio
        n_obs_single = n_mimic_obs + n_proprio
        num_observations = n_obs_single * (TaksT1MimicPrivCfg.env.history_len + 1)


class TaksT1MimicStuRLCfg(TaksT1MimicPrivCfg):
    class env(TaksT1MimicPrivCfg.env):
        obs_type = 'student'
        tar_motion_steps = [1]
        n_mimic_obs_single = TaksT1MimicPrivCfg.env.n_mimic_obs_single
        n_mimic_obs = len(tar_motion_steps) * n_mimic_obs_single
        n_proprio = TaksT1MimicPrivCfg.env.n_proprio
        n_obs_single = n_mimic_obs + n_proprio
        num_observations = n_obs_single * (TaksT1MimicPrivCfg.env.history_len + 1)


class TaksT1MimicPrivCfgPPO(HumanoidMimicCfgPPO):
    seed = 1
    class runner(HumanoidMimicCfgPPO.runner):
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'PPO'
        runner_class_name = 'OnPolicyRunnerMimic'
        max_iterations = 1_000_002

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
        action_std = [0.7] * 12 + [0.4] * 3 + [0.5] * 14 + [0.3] * 3
        init_noise_std = 1.0
        obs_context_len = 11
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = 'silu'
        layer_norm = True
        motion_latent_dim = 128


class TaksT1MimicStuCfgDAgger(TaksT1MimicPrivCfgPPO):
    seed = 1
    
    class teachercfg(TaksT1MimicPrivCfgPPO):
        pass
    
    class runner(TaksT1MimicPrivCfgPPO.runner):
        policy_class_name = 'DAggerActor'
        algorithm_class_name = 'DAgger'
        runner_class_name = 'DAggerRunner'
        max_iterations = 1_000_002
        warm_iters = 100
        
        save_interval = 500
        experiment_name = 'test'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        
        teacher_experiment_name = 'test'
        teacher_proj_name = 'taks_t1_priv_mimic'
        teacher_checkpoint = -1
        eval_student = False

    class algorithm:
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1e-4
        max_grad_norm = 1.0
        normalizer_update_iterations = 1000

    class policy:
        actor_hidden_dims = [1024, 1024, 512, 256]
        history_latent_dim = 128
        activation = 'silu'


class TaksT1MimicStuRLCfgDAgger(TaksT1MimicStuRLCfg):
    seed = 1
    
    class teachercfg(TaksT1MimicPrivCfgPPO):
        pass
    
    class runner(TaksT1MimicPrivCfgPPO.runner):
        policy_class_name = 'ActorCriticTeleop'
        algorithm_class_name = 'DaggerPPO'
        runner_class_name = 'OnPolicyDaggerRunner'
        max_iterations = 1_000_002
        warm_iters = 100
        
        save_interval = 500
        experiment_name = 'test'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        
        teacher_experiment_name = 'test'
        teacher_proj_name = 'taks_t1_priv_mimic'
        teacher_checkpoint = -1
        eval_student = False

    class algorithm(HumanoidMimicCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005
        
        dagger_coef_anneal_steps = 60000
        dagger_coef = 0.2
        dagger_coef_min = 0.1

    class policy(HumanoidMimicCfgPPO.policy):
        action_std = [0.7] * 12 + [0.4] * 3 + [0.5] * 14 + [0.3] * 3
        init_noise_std = 1.0
        obs_context_len = 11
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = 'silu'
        layer_norm = True
        motion_latent_dim = 128
