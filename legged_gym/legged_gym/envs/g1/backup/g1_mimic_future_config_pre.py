from legged_gym.envs.g1.g1_mimic_distill_config import G1MimicPrivCfg, G1MimicPrivCfgPPO
from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR


# 未来帧配置：
# - TAR_MOTION_STEPS_FUTURE_OBS: 用于 obs 输入，保持 [0] 以兼容 sim2sim/sim2real
# - TAR_MOTION_STEPS_FUTURE_REWARD: 用于奖励计算，10帧未来动作
TAR_MOTION_STEPS_FUTURE_OBS = [0]  # 保持原始 obs 维度
TAR_MOTION_STEPS_FUTURE_REWARD = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # 10帧用于奖励
class G1MimicStuFutureCfg(G1MimicPrivCfg):
    """Student policy config with future motion support and curriculum masking.
    Extends existing G1MimicPrivCfg to add future motion capabilities."""
    
    class env(G1MimicPrivCfg.env):
        obs_type = 'student_future'
        
        # Keep original student motion steps (current frame only)
        tar_motion_steps = [0]
        
        # Future motion frames for obs (keep [0] for sim2sim/sim2real compatibility)
        tar_motion_steps_future = TAR_MOTION_STEPS_FUTURE_OBS
        # Future motion frames for reward calculation (10 frames)
        tar_motion_steps_future_reward = TAR_MOTION_STEPS_FUTURE_REWARD
        
        
        # Observation dimensions (keep original structure)
        n_mimic_obs_single = 6 + 29
        n_mimic_obs = len(tar_motion_steps) * n_mimic_obs_single  # Current frame only
        n_proprio = G1MimicPrivCfg.env.n_proprio

        # Future observation dimensions (keep original for sim2sim/sim2real)
        n_future_obs_single = 6 + 29  # Masking disabled -> no indicator channel
        n_future_obs = len(tar_motion_steps_future) * n_future_obs_single  # 保持原始维度
        
        # Total observation size: maintain original structure + future observations (no force obs needed)
        n_obs_single = n_mimic_obs + n_proprio  # Current frame observation (for history)
        num_observations = n_obs_single * (G1MimicPrivCfg.env.history_len + 1) + n_future_obs
        
        
        # FALCON-style curriculum force application (domain randomization)
        # Set to None to completely disable force curriculum
        enable_force_curriculum = True  # Enable force disturbances during training, set to None/False to disable
        
        class force_curriculum:
            # Force application settings
            force_apply_links = ['left_rubber_hand', 'right_rubber_hand']  # Links to apply forces to
            
            # Force curriculum learning
            force_scale_curriculum = True
            force_scale_initial_scale = 1.0
            force_scale_up_threshold = 210    # Episode length threshold for scaling up force
            force_scale_down_threshold = 200  # Episode length threshold for scaling down force
            force_scale_up = 0.02            # Amount to increase force scale
            force_scale_down = 0.02          # Amount to decrease force scale
            force_scale_max = 1.0
            force_scale_min = 0.0
            
            # Force application ranges (in Newtons)
            apply_force_x_range = [-40.0, 40.0]
            apply_force_y_range = [-40.0, 40.0]
            apply_force_z_range = [-50.0, 5.0]
            
            # Force randomization
            zero_force_prob = [0.25, 0.25, 0.25]  # Probability of zeroing each force axis
            randomize_force_duration = [10, 50]  # Force duration range in steps (policy runs at 50Hz)
            
            # Advanced force settings
            max_force_estimation = True       # Use jacobian-based force estimation
            use_lpf = False                   # Low-pass filter applied forces
            force_filter_alpha = 0.05         # LPF coefficient
            
            # Task-specific force behavior
            only_apply_z_force_when_walking = False  # Restrict to Z-axis forces during walking
            only_apply_resistance_when_walking = True # Apply resistance forces against movement
    
    class motion(G1MimicPrivCfg.motion):
        motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/g1_demo.yaml"
        
        # Ensure motion curriculum is enabled for difficulty adaptation
        # Set to None to completely disable motion curriculum
        motion_curriculum = True  # Set to None/False to disable
        motion_curriculum_gamma = 0.01
        motion_decompose = False

        # use_adaptive_pose_termination = Truee
        
        # Motion Domain Randomization - Enable for robustness
        # Set to None to completely disable motion domain randomization
        motion_dr_enabled = True  # Set to None/False to disable
        root_position_noise = [0.01, 0.05]  # ±1-5cm noise range for root position
        root_orientation_noise = [0.1, 0.2]  # ±5.7-11.4° noise range for root orientation (in rad)
        root_velocity_noise = [0.05, 0.1]   # ±0.05-0.1 noise range for root velocity
        joint_position_noise = [0.05, 0.1]  # ±0.05-0.1 rad noise range for joint positions
        motion_dr_resampling = True          # Resample noise each timestep
        
        # Error Aware Sampling parameters
        use_error_aware_sampling = False      # Enable error aware sampling based on max key body error
        error_sampling_power = 5.0          # Power exponent for error-based probability calculation
        error_sampling_threshold = 0.15     # Threshold for max key body error normalization
    
    class domain_rand(G1MimicPrivCfg.domain_rand):
        # Master switch for all domain randomization
        # Set to None to completely disable all domain randomization
        domain_rand_general = True  # Set to None/False to disable all

        randomize_gravity = (True and domain_rand_general)
        gravity_rand_interval_s = 4
        gravity_range = (-0.1, 0.1)

        randomize_friction = (True and domain_rand_general)
        friction_range = [0.1, 2.]

        randomize_base_mass = (True and domain_rand_general)
        added_mass_range = [-1., 3]

        randomize_base_com = (True and domain_rand_general)
        added_com_range = [-0.1, 0.1]

        push_robots = (True and domain_rand_general)
        push_interval_s = 4
        max_push_vel_xy = 2.0

        push_end_effector = (False and domain_rand_general)
        push_end_effector_interval_s = 2
        max_push_force_end_effector = 10.0

        randomize_motor = (True and domain_rand_general)
        motor_strength_range = [0.1, 2.0]

        action_delay = (True and domain_rand_general)
        action_buf_len = 8
        
        # 惯量随机化 - 模拟电机转子惯量不确定性
        randomize_armature = (True and domain_rand_general)
        armature_range = [0.1, 2.0]  # 惯量缩放范围
        
        # 刚体惯性随机化 - 模拟连杆惯性不确定性
        randomize_link_inertia = (True and domain_rand_general)
        link_inertia_range = [0.1, 2.0]  # 刚体惯性缩放范围

    class rewards(G1MimicPrivCfg.rewards):
        # All reward scales can be set to None to completely disable that reward
        # Set any scale to None to skip computing that reward entirely
        class scales:
            tracking_joint_dof = 2.0  # Set to None to disable
            tracking_joint_vel = 0.2  # Set to None to disable
            tracking_root_translation_z = 1.0  # Set to None to disable
            tracking_root_rotation = 1.0  # Set to None to disable
            tracking_root_linear_vel = 1.0  # Set to None to disable
            tracking_root_angular_vel = 1.0  # Set to None to disable
            tracking_keybody_pos = 2.0  # Set to None to disable
            tracking_keybody_pos_global = 2.0  # Set to None to disable
            alive = 0.5  # Set to None to disable
            feet_slip = -0.1  # Set to None to disable
            feet_contact_forces = -5e-4  # Set to None to disable
            feet_stumble = -1.25  # Set to None to disable
            dof_pos_limits = -5.0  # Set to None to disable
            dof_torque_limits = -1.0  # Set to None to disable
            dof_vel = -1e-4  # Set to None to disable
            dof_acc = -1e-7  # Set to None to disable
            action_rate = -0.1  # Set to None to disable
            feet_air_time = 5.0  # Set to None to disable
            ang_vel_xy = -0.02  # Set to None to disable
            ankle_dof_acc = -1e-7 * 2  # Set to None to disable
            ankle_dof_vel = -1e-4 * 2  # Set to None to disable
            
            # 未来动作一致性奖励（只在训练时生效）- Set to None to disable
            future_action_consistency = 0.2  # Set to None to disable
            future_yaw_consistency = 0.1  # Set to None to disable
            turning_smoothness = -0.01  # Set to None to disable


class G1MimicStuFutureCfgDAgger(G1MimicStuFutureCfg):
    """DAgger training config for future motion student policy.
    Inherits from G1MimicStuFutureCfg and extends G1MimicStuRLTrackingCfgDAgger."""
    
    seed = 1
    
    class teachercfg(G1MimicPrivCfgPPO):
        pass
    
    class runner(G1MimicPrivCfgPPO.runner):
        policy_class_name = 'ActorCriticFuture'
        algorithm_class_name = 'DaggerPPO'
        runner_class_name = 'OnPolicyDaggerRunner'
        max_iterations = 10001
        warm_iters = 100
        
        # logging
        save_interval = 110
        experiment_name = 'test'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        
        teacher_experiment_name = 'test'
        teacher_proj_name = 'g1_priv_mimic'
        teacher_checkpoint = -1
        eval_student = False
        
        # Wandb model saving option
        save_to_wandb = False  # Set to False to disable wandb model saving

    class algorithm(HumanoidMimicCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005
        
        dagger_coef_anneal_steps = 60000  # Total steps to anneal dagger_coef to dagger_coef_min
        dagger_coef = 0.2
        dagger_coef_min = 0.1
        
        # Future motion specific parameters
        future_weight_decay = 0.95      # Decay weight for older future frames
        future_consistency_loss = 0.1   # Weight for consistency loss between future predictions

    class policy(HumanoidMimicCfgPPO.policy):
        action_std = [0.7] * 12 + [0.4] * 3 + [0.5] * 14
        init_noise_std = 1.0
        obs_context_len = 11
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = 'silu'
        layer_norm = True
        motion_latent_dim = 128
        
        # Future motion encoder parameters
        future_encoder_dims = [256, 256, 128]  # Separate encoder for future motion
        future_attention_heads = 4              # Multi-head attention for future frames
        future_dropout = 0.1                   # Dropout for future encoder
        temporal_embedding_dim = 64            # Temporal position embedding
        future_latent_dim = 128                # Future motion latent dimension
        num_future_steps = len(TAR_MOTION_STEPS_FUTURE_OBS)  # obs用的未来帧数
        
        # Explicit future observation dimensions (avoid miscalculations when tweaking configs)
        num_future_observations = G1MimicStuFutureCfg.env.n_future_obs  # 360
        
        # MoE specific parameters
        num_experts = 4                        # Number of expert networks
        expert_hidden_dims = [256, 128]        # Hidden dimensions for each expert
        gating_hidden_dim = 128                # Hidden dimension for gating network
        moe_temperature = 1.0                  # Temperature for gating softmax
        moe_topk = None                        # Number of top experts to use (None = use all)
        load_balancing_loss_weight = 0.01      # Weight for load balancing loss
