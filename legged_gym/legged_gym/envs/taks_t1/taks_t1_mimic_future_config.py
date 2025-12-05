from legged_gym.envs.taks_t1.taks_t1_mimic_distill_config import (
    TaksT1MimicPrivCfg, TaksT1MimicPrivCfgPPO
)
from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR


TAR_MOTION_STEPS_FUTURE = [0]


class TaksT1MimicStuFutureCfg(TaksT1MimicPrivCfg):
    class env(TaksT1MimicPrivCfg.env):
        obs_type = 'student_future'
        tar_motion_steps = [0]
        tar_motion_steps_future = TAR_MOTION_STEPS_FUTURE

        n_mimic_obs_single = 6 + 32
        n_mimic_obs = len(tar_motion_steps) * n_mimic_obs_single
        n_proprio = TaksT1MimicPrivCfg.env.n_proprio

        n_future_obs_single = 6 + 32
        n_future_obs = len(tar_motion_steps_future) * n_future_obs_single

        n_obs_single = n_mimic_obs + n_proprio
        num_observations = (n_obs_single *
                           (TaksT1MimicPrivCfg.env.history_len + 1) +
                           n_future_obs)

        enable_force_curriculum = True

        class force_curriculum:
            force_apply_links = ['left_wrist_pitch_link',
                                'right_wrist_pitch_link']
            force_scale_curriculum = True
            force_scale_initial_scale = 1.0
            force_scale_up_threshold = 210
            force_scale_down_threshold = 200
            force_scale_up = 0.02
            force_scale_down = 0.02
            force_scale_max = 1.0
            force_scale_min = 0.0
            apply_force_x_range = [-40.0, 40.0]
            apply_force_y_range = [-40.0, 40.0]
            apply_force_z_range = [-50.0, 5.0]
            zero_force_prob = [0.25, 0.25, 0.25]
            randomize_force_duration = [10, 50]
            max_force_estimation = True
            use_lpf = False
            force_filter_alpha = 0.05
            only_apply_z_force_when_walking = False
            only_apply_resistance_when_walking = True

    class motion(TaksT1MimicPrivCfg.motion):
        motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/taks_t1_demo.yaml"
        motion_curriculum = True
        motion_curriculum_gamma = 0.01
        motion_decompose = False
        motion_dr_enabled = True
        root_position_noise = [0.01, 0.05]
        root_orientation_noise = [0.1, 0.2]
        root_velocity_noise = [0.05, 0.1]
        joint_position_noise = [0.05, 0.1]
        motion_dr_resampling = True
        use_error_aware_sampling = False
        error_sampling_power = 5.0
        error_sampling_threshold = 0.15

    class domain_rand(TaksT1MimicPrivCfg.domain_rand):
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

        # 动作噪声
        action_noise = (True and domain_rand_general)
        action_noise_std = 0.01

        # 关节编码器噪声
        encoder_noise = (True and domain_rand_general)
        encoder_pos_noise_std = 0.005
        encoder_vel_noise_std = 0.01
        encoder_pos_bias_range = [-0.01, 0.01]
        encoder_vel_bias_range = [-0.02, 0.02]

        # IMU噪声和漂移
        imu_noise = (True and domain_rand_general)
        imu_ang_vel_noise_std = 0.02
        imu_lin_acc_noise_std = 0.05
        imu_ang_vel_bias_range = [-0.1, 0.1]
        imu_lin_acc_bias_range = [-0.2, 0.2]
        imu_bias_drift_std = 0.01

        # 观测丢包
        observation_dropout = (True and domain_rand_general)
        observation_dropout_prob = 0.001
        observation_dropout_mode = 'hold'

        # 关节故障
        joint_failure = (False and domain_rand_general)
        joint_failure_prob = 0.0001
        joint_failure_mode = 'weak'
        joint_failure_weak_factor = 0.5

        # 传感器延迟尖峰
        sensor_latency_spike = (True and domain_rand_general)
        sensor_latency_spike_prob = 0.001
        sensor_latency_max_steps = 10

        # 重力方向偏置
        slope_randomization = (True and domain_rand_general)
        gravity_bias_x_range = [-0.1, 0.1]
        gravity_bias_y_range = [-0.1, 0.1]
        gravity_bias_z_range = [-0.05, 0.05]

    class rewards(TaksT1MimicPrivCfg.rewards):
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
            action_rate = -0.1
            feet_air_time = 5.0
            ang_vel_xy = -0.02
            ankle_dof_acc = -5e-8 * 2
            ankle_dof_vel = -1e-4 * 2
            idle_penalty = -0.001
            # 未来动作一致性奖励（只在训练时生效）
            future_action_consistency = 1.0
            future_yaw_consistency = 0.6
            turning_smoothness = -0.01


class TaksT1MimicStuFutureCfgDAgger(TaksT1MimicStuFutureCfg):
    seed = 1

    class teachercfg(TaksT1MimicPrivCfgPPO):
        pass

    class runner(TaksT1MimicPrivCfgPPO.runner):
        policy_class_name = 'ActorCriticFuture'
        algorithm_class_name = 'DaggerPPO'
        runner_class_name = 'OnPolicyDaggerRunner'
        max_iterations = 10001
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
        save_to_wandb = False

    class algorithm(HumanoidMimicCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005
        dagger_coef_anneal_steps = 60000
        dagger_coef = 0.2
        dagger_coef_min = 0.1
        future_weight_decay = 0.95
        future_consistency_loss = 0.1

    class policy(HumanoidMimicCfgPPO.policy):
        action_std = [0.7] * 12 + [0.4] * 3 + [0.5] * 14 + [0.3] * 3
        init_noise_std = 1.0
        obs_context_len = 11
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = 'silu'
        layer_norm = True
        motion_latent_dim = 128
        future_encoder_dims = [256, 256, 128]
        future_attention_heads = 4
        future_dropout = 0.1
        temporal_embedding_dim = 64
        future_latent_dim = 128
        num_future_steps = len(TAR_MOTION_STEPS_FUTURE)
        num_future_observations = TaksT1MimicStuFutureCfg.env.n_future_obs
        num_experts = 4
        expert_hidden_dims = [256, 128]
        gating_hidden_dim = 128
        moe_temperature = 1.0
        moe_topk = None
        load_balancing_loss_weight = 0.01
