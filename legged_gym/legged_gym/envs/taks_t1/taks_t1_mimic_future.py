from isaacgym.torch_utils import *
from isaacgym import gymapi, gymtorch
import torch

from legged_gym.envs.taks_t1.taks_t1_mimic_distill import TaksT1MimicDistill
from .taks_t1_mimic_future_config import TaksT1MimicStuFutureCfg
from legged_gym.gym_utils.math import *
from pose.utils import torch_utils
from legged_gym.envs.base.legged_robot import euler_from_quaternion
from legged_gym.envs.base.humanoid_char import (
    convert_to_local_root_body_pos, convert_to_global_root_body_pos
)


class TaksT1MimicFuture(TaksT1MimicDistill):
    """Student policy environment with future motion support for Taks_T1."""

    def __init__(self, cfg: TaksT1MimicStuFutureCfg, sim_params, physics_engine,
                 sim_device, headless):
        self.future_cfg = cfg.env
        self.evaluation_mode = getattr(cfg.env, 'evaluation_mode', False)
        self.force_full_masking = getattr(cfg.env, 'force_full_masking', False)
        # Support None value to completely disable force curriculum
        _enable_force = getattr(cfg.env, 'enable_force_curriculum', False)
        self.enable_force_curriculum = bool(_enable_force) if _enable_force is not None else False

        if self.enable_force_curriculum:
            self.force_scale_curriculum = True
            self.episode_length_counter = None
            self.force_scale = None

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        num_motions = self._motion_lib.num_motions()
        self.motion_difficulty = 10 * torch.ones(
            (num_motions), device=self.device, dtype=torch.float)
        self.mean_motion_difficulty = 10.

        if self.obs_type == 'student_future':
            # Future motion target steps for OBS (keep small for sim2sim/sim2real)
            self._tar_motion_steps_future = torch.tensor(
                getattr(cfg.env, 'tar_motion_steps_future', [0]),
                device=self.device, dtype=torch.long)

            # Future motion target steps for REWARD (10 frames)
            self._tar_motion_steps_future_reward = torch.tensor(
                getattr(cfg.env, 'tar_motion_steps_future_reward',
                        [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]),
                device=self.device, dtype=torch.long)

            self._tar_motion_steps_future_idx = torch.searchsorted(
                self._tar_motion_steps_priv, self._tar_motion_steps_future)

            self.future_mask = torch.ones(
                (self.num_envs, len(self._tar_motion_steps_future)),
                device=self.device, dtype=torch.bool)

            print(f"Future motion enabled with "
                  f"{len(self._tar_motion_steps_future)} obs frames")
            print(f"Future obs steps: {self._tar_motion_steps_future.tolist()}")
            print(f"Future reward steps ({len(self._tar_motion_steps_future_reward)} frames): "
                  f"{self._tar_motion_steps_future_reward.tolist()}")

        if (hasattr(cfg.motion, 'use_error_aware_sampling') and
                cfg.motion.use_error_aware_sampling):
            self._error_log_counter = 0
            self.body_error_history = []
            print("Error aware sampling logging initialized")

        # Only initialize if enable_force_curriculum is True (not None or False)
        if self.enable_force_curriculum:
            self._init_force_curriculum_components(cfg)
            force_links = getattr(
                cfg.env.force_curriculum, 'force_apply_links',
                ['left_wrist_pitch_link', 'right_wrist_pitch_link'])
            print(f"Force curriculum enabled with force application to "
                  f"{len(force_links)} links: {force_links}")
        else:
            print("Force curriculum disabled (enable_force_curriculum is None or False)")

        # 通信延迟随机化初始化
        _comm_delay = getattr(cfg.domain_rand, 'comm_delay', False)
        self.enable_comm_delay = bool(_comm_delay) if _comm_delay is not None else False
        if self.enable_comm_delay:
            self._init_comm_delay_buffers(cfg)

        # 头部link index，用于头部姿态约束
        self._head_body_idx = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], 'neck_pitch_link')
        if self._head_body_idx == -1:
            print("Warning: neck_pitch_link not found, head orientation penalty disabled")
        else:
            print(f"Head orientation penalty enabled, neck_pitch_link index: {self._head_body_idx}")

    def _get_unified_motion_data(self):
        if (self.obs_type == 'student_future' and
                hasattr(self, '_tar_motion_steps_future')):
            all_steps = torch.cat([
                self._tar_motion_steps_priv, self._tar_motion_steps_future])
            num_priv_steps = self._tar_motion_steps_priv.shape[0]
            num_future_steps = self._tar_motion_steps_future.shape[0]
        else:
            all_steps = self._tar_motion_steps_priv
            num_priv_steps = self._tar_motion_steps_priv.shape[0]
            num_future_steps = 0

        total_steps = all_steps.shape[0]
        assert total_steps > 0, "Invalid number of target observation steps"

        motion_times = self._get_motion_times().unsqueeze(-1)
        obs_motion_times = all_steps * self.dt + motion_times
        motion_ids_tiled = torch.broadcast_to(
            self._motion_ids.unsqueeze(-1), obs_motion_times.shape)
        motion_ids_tiled = motion_ids_tiled.flatten()
        obs_motion_times = obs_motion_times.flatten()

        (root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel,
         body_pos, root_pos_delta_local,
         root_rot_delta_local) = self._motion_lib.calc_motion_frame(
            motion_ids_tiled, obs_motion_times)

        (root_pos, root_rot, root_vel, root_ang_vel, dof_pos,
         dof_vel) = self._apply_motion_domain_randomization(
            root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel)

        roll, pitch, yaw = euler_from_quaternion(root_rot)
        roll = roll.reshape(self.num_envs, total_steps, 1)
        pitch = pitch.reshape(self.num_envs, total_steps, 1)
        yaw = yaw.reshape(self.num_envs, total_steps, 1)

        root_vel_local = quat_rotate_inverse(root_rot, root_vel)
        root_ang_vel_local = quat_rotate_inverse(root_rot, root_ang_vel)

        whole_key_body_pos = body_pos[:, self._key_body_ids_motion, :]
        whole_key_body_pos_global = convert_to_global_root_body_pos(
            root_pos=root_pos, root_rot=root_rot, body_pos=whole_key_body_pos)

        root_pos = root_pos.reshape(
            self.num_envs, total_steps, root_pos.shape[-1])
        root_vel = root_vel.reshape(
            self.num_envs, total_steps, root_vel.shape[-1])
        root_rot = root_rot.reshape(
            self.num_envs, total_steps, root_rot.shape[-1])
        root_ang_vel = root_ang_vel.reshape(
            self.num_envs, total_steps, root_ang_vel.shape[-1])
        dof_pos = dof_pos.reshape(
            self.num_envs, total_steps, dof_pos.shape[-1])
        dof_vel = dof_vel.reshape(
            self.num_envs, total_steps, dof_vel.shape[-1])
        root_vel_local = root_vel_local.reshape(
            self.num_envs, total_steps, root_vel_local.shape[-1])
        root_ang_vel_local = root_ang_vel_local.reshape(
            self.num_envs, total_steps, root_ang_vel_local.shape[-1])
        root_pos_delta_local = root_pos_delta_local.reshape(
            self.num_envs, total_steps, root_pos_delta_local.shape[-1])
        root_rot_delta_local = root_rot_delta_local.reshape(
            self.num_envs, total_steps, root_rot_delta_local.shape[-1])
        whole_key_body_pos = whole_key_body_pos.reshape(
            self.num_envs, total_steps, -1)
        whole_key_body_pos_global = whole_key_body_pos_global.reshape(
            self.num_envs, total_steps, -1)

        root_pos_distance_to_target = (
            root_pos - self.root_states[:, 0:3].reshape(self.num_envs, 1, -1))

        return {
            'root_pos': root_pos,
            'root_vel': root_vel,
            'root_rot': root_rot,
            'root_ang_vel': root_ang_vel,
            'dof_pos': dof_pos,
            'dof_vel': dof_vel,
            'body_pos': body_pos,
            'root_pos_delta_local': root_pos_delta_local,
            'root_rot_delta_local': root_rot_delta_local,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'root_vel_local': root_vel_local,
            'root_ang_vel_local': root_ang_vel_local,
            'whole_key_body_pos': whole_key_body_pos,
            'whole_key_body_pos_global': whole_key_body_pos_global,
            'root_pos_distance_to_target': root_pos_distance_to_target,
            'num_priv_steps': num_priv_steps,
            'num_future_steps': num_future_steps,
            'total_steps': total_steps
        }

    def _build_future_obs_from_data(self, motion_data):
        if self.obs_type != 'student_future':
            return torch.zeros(self.num_envs, 0, device=self.device)

        if motion_data['num_future_steps'] == 0:
            return torch.zeros(self.num_envs, 0, device=self.device)

        num_priv_steps = motion_data['num_priv_steps']

        root_pos = motion_data['root_pos'][:, num_priv_steps:]
        root_vel_local = motion_data['root_vel_local'][:, num_priv_steps:]
        root_ang_vel_local = motion_data['root_ang_vel_local'][:, num_priv_steps:]
        roll = motion_data['roll'][:, num_priv_steps:]
        pitch = motion_data['pitch'][:, num_priv_steps:]
        dof_pos = motion_data['dof_pos'][:, num_priv_steps:]

        future_obs = torch.cat((
            root_vel_local[..., :2],
            root_pos[..., 2:3],
            roll, pitch,
            root_ang_vel_local[..., 2:3],
            dof_pos,
        ), dim=-1)

        return future_obs

    def _get_mimic_obs(self, motion_data=None):
        """Get mimic observations.
        Args:
            motion_data: Optional pre-computed motion data to avoid redundant sampling.
        """
        if motion_data is None:
            motion_data = self._get_unified_motion_data()
        num_steps = motion_data['num_priv_steps']

        root_pos = motion_data['root_pos'][:, :num_steps]
        root_vel = motion_data['root_vel'][:, :num_steps]
        root_rot = motion_data['root_rot'][:, :num_steps]
        root_ang_vel = motion_data['root_ang_vel'][:, :num_steps]
        dof_pos = motion_data['dof_pos'][:, :num_steps]
        dof_vel = motion_data['dof_vel'][:, :num_steps]
        root_pos_delta_local = motion_data['root_pos_delta_local'][:, :num_steps]
        root_rot_delta_local = motion_data['root_rot_delta_local'][:, :num_steps]
        roll = motion_data['roll'][:, :num_steps]
        pitch = motion_data['pitch'][:, :num_steps]
        yaw = motion_data['yaw'][:, :num_steps]
        root_vel_local = motion_data['root_vel_local'][:, :num_steps]
        root_ang_vel_local = motion_data['root_ang_vel_local'][:, :num_steps]
        whole_key_body_pos = motion_data['whole_key_body_pos'][:, :num_steps]
        whole_key_body_pos_global = motion_data['whole_key_body_pos_global'][
            :, :num_steps]
        root_pos_distance_to_target = motion_data['root_pos_distance_to_target'][
            :, :num_steps]

        priv_mimic_obs_buf = torch.cat((
            root_pos,
            root_pos_distance_to_target,
            roll, pitch, yaw,
            root_vel_local,
            root_ang_vel_local,
            root_pos_delta_local,
            root_rot_delta_local,
            dof_pos,
            whole_key_body_pos if not self.global_obs
            else whole_key_body_pos_global,
        ), dim=-1)

        mimic_obs_buf = torch.cat((
            root_vel_local[..., :2],
            root_pos[..., 2:3],
            roll, pitch,
            root_ang_vel_local[..., 2:3],
            dof_pos,
        ), dim=-1)[:, self._tar_motion_steps_idx_in_teacher, :]

        priv_mimic_obs = priv_mimic_obs_buf.reshape(self.num_envs, -1)
        mimic_obs = mimic_obs_buf.reshape(self.num_envs, -1)

        if self.obs_type == 'student_future':
            future_obs = self._build_future_obs_from_data(motion_data)
            return priv_mimic_obs, mimic_obs, future_obs
        else:
            return priv_mimic_obs, mimic_obs

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if self.enable_force_curriculum:
            self._update_force_curriculum(env_ids)
        if getattr(self, 'enable_comm_delay', False):
            self._resample_comm_delay(env_ids)

    def _init_force_curriculum_components(self, cfg):
        force_cfg = cfg.env.force_curriculum
        self.force_apply_links = getattr(
            force_cfg, 'force_apply_links',
            ['left_wrist_pitch_link', 'right_wrist_pitch_link'])
        self.force_apply_body_indices = []

        for link_name in self.force_apply_links:
            body_idx = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], link_name)
            if body_idx != -1:
                self.force_apply_body_indices.append(body_idx)
            else:
                print(f"Warning: Force application link '{link_name}' "
                      f"not found in robot model")

        self.force_apply_body_indices = torch.tensor(
            self.force_apply_body_indices, device=self.device, dtype=torch.long)

        self.force_scale_curriculum = getattr(
            force_cfg, 'force_scale_curriculum', True)
        self.force_scale_initial_scale = getattr(
            force_cfg, 'force_scale_initial_scale', 0.1)
        self.force_scale_up_threshold = getattr(
            force_cfg, 'force_scale_up_threshold', 210)
        self.force_scale_down_threshold = getattr(
            force_cfg, 'force_scale_down_threshold', 200)
        self.force_scale_up = getattr(force_cfg, 'force_scale_up', 0.02)
        self.force_scale_down = getattr(force_cfg, 'force_scale_down', 0.02)
        self.force_scale_max = getattr(force_cfg, 'force_scale_max', 1.0)
        self.force_scale_min = getattr(force_cfg, 'force_scale_min', 0.0)

        self.apply_force_x_range = torch.tensor(
            getattr(force_cfg, 'apply_force_x_range', [-40.0, 40.0]),
            device=self.device)
        self.apply_force_y_range = torch.tensor(
            getattr(force_cfg, 'apply_force_y_range', [-40.0, 40.0]),
            device=self.device)
        self.apply_force_z_range = torch.tensor(
            getattr(force_cfg, 'apply_force_z_range', [-50.0, 5.0]),
            device=self.device)

        # Pre-compute force range spans for efficiency
        self._force_x_span = self.apply_force_x_range[1] - self.apply_force_x_range[0]
        self._force_y_span = self.apply_force_y_range[1] - self.apply_force_y_range[0]
        self._force_z_span = self.apply_force_z_range[1] - self.apply_force_z_range[0]

        self.zero_force_prob = torch.tensor(
            getattr(force_cfg, 'zero_force_prob', [0.25, 0.25, 0.25]),
            device=self.device)
        self.randomize_force_duration = getattr(
            force_cfg, 'randomize_force_duration', [150, 250])

        self.force_scale = torch.full(
            (self.num_envs,), self.force_scale_initial_scale, device=self.device)
        self.episode_length_counter = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)

        self.applied_forces = torch.zeros(
            (self.num_envs, len(self.force_apply_body_indices), 3),
            device=self.device)
        self.force_duration_counter = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.force_duration_target = torch.randint(
            self.randomize_force_duration[0],
            self.randomize_force_duration[1] + 1,
            (self.num_envs,), device=self.device)

        print(f"Force curriculum initialized with "
              f"{len(self.force_apply_body_indices)} force application points")

    def _update_force_curriculum(self, env_ids):
        """Update force scale based on episode performance (curriculum learning)."""
        if not self.force_scale_curriculum:
            return
        if self.episode_length_counter is None or self.force_scale is None:
            return
        episode_lengths = self.episode_length_counter[env_ids]
        good_mask = episode_lengths > self.force_scale_up_threshold
        self.force_scale[env_ids[good_mask]] = torch.clamp(
            self.force_scale[env_ids[good_mask]] + self.force_scale_up,
            self.force_scale_min, self.force_scale_max)
        poor_mask = episode_lengths < self.force_scale_down_threshold
        self.force_scale[env_ids[poor_mask]] = torch.clamp(
            self.force_scale[env_ids[poor_mask]] - self.force_scale_down,
            self.force_scale_min, self.force_scale_max)
        self.episode_length_counter[env_ids] = 0
        self.force_duration_counter[env_ids] = 0
        self.force_duration_target[env_ids] = torch.randint(
            self.randomize_force_duration[0], self.randomize_force_duration[1] + 1,
            (len(env_ids),), device=self.device)

    def _apply_motion_domain_randomization(self, root_pos, root_rot, root_vel,
                                           root_ang_vel, dof_pos, dof_vel):
        return root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel

    # ==================== 通信延迟随机化 ====================
    def _init_comm_delay_buffers(self, cfg):
        """初始化通信延迟随机化所需的buffer。"""
        dr = cfg.domain_rand
        self.comm_delay_buf_len = getattr(dr, 'comm_delay_buf_len', 10)
        self.motor_pos_delay_range = getattr(dr, 'motor_pos_delay_range', [0, 3])
        self.motor_vel_delay_range = getattr(dr, 'motor_vel_delay_range', [0, 4])
        self.motor_torque_delay_range = getattr(dr, 'motor_torque_delay_range', [0, 3])
        self.imu_motor_delay_range = getattr(dr, 'imu_motor_delay_range', [-2, 2])
        self.resample_delay_per_episode = getattr(dr, 'resample_delay_per_episode', True)

        # 历史buffer: [num_envs, buf_len, dim]
        self.motor_pos_history = torch.zeros(
            (self.num_envs, self.comm_delay_buf_len, self.num_dof),
            device=self.device, dtype=torch.float)
        self.motor_vel_history = torch.zeros(
            (self.num_envs, self.comm_delay_buf_len, self.num_dof),
            device=self.device, dtype=torch.float)
        self.motor_torque_history = torch.zeros(
            (self.num_envs, self.comm_delay_buf_len, self.num_dof),
            device=self.device, dtype=torch.float)
        # IMU: ang_vel(3) + roll(1) + pitch(1)
        self.imu_history = torch.zeros(
            (self.num_envs, self.comm_delay_buf_len, 5),
            device=self.device, dtype=torch.float)

        # 每个env的延迟步数 [num_envs]
        self.motor_pos_delay = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.motor_vel_delay = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.motor_torque_delay = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.imu_delay = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # 初始采样延迟
        self._resample_comm_delay(torch.arange(self.num_envs, device=self.device))
        print(f"Comm delay enabled: pos[{self.motor_pos_delay_range}], "
              f"vel[{self.motor_vel_delay_range}], torque[{self.motor_torque_delay_range}], "
              f"imu_offset[{self.imu_motor_delay_range}]")

    def _resample_comm_delay(self, env_ids):
        """为指定env重新采样通信延迟。"""
        if not self.resample_delay_per_episode:
            return
        n = len(env_ids)
        # 电机各通道独立延迟
        self.motor_pos_delay[env_ids] = torch.randint(
            self.motor_pos_delay_range[0], self.motor_pos_delay_range[1] + 1,
            (n,), device=self.device, dtype=torch.long)
        self.motor_vel_delay[env_ids] = torch.randint(
            self.motor_vel_delay_range[0], self.motor_vel_delay_range[1] + 1,
            (n,), device=self.device, dtype=torch.long)
        self.motor_torque_delay[env_ids] = torch.randint(
            self.motor_torque_delay_range[0], self.motor_torque_delay_range[1] + 1,
            (n,), device=self.device, dtype=torch.long)
        # IMU相对电机的延迟偏移
        self.imu_delay[env_ids] = torch.randint(
            self.imu_motor_delay_range[0], self.imu_motor_delay_range[1] + 1,
            (n,), device=self.device, dtype=torch.long)
        # 重置历史buffer
        self.motor_pos_history[env_ids] = 0
        self.motor_vel_history[env_ids] = 0
        self.motor_torque_history[env_ids] = 0
        self.imu_history[env_ids] = 0

    def _update_comm_delay_buffers(self):
        """更新通信延迟历史buffer，每步调用。"""
        # 滚动buffer: 丢弃最老的，添加最新的
        self.motor_pos_history = torch.roll(self.motor_pos_history, -1, dims=1)
        self.motor_vel_history = torch.roll(self.motor_vel_history, -1, dims=1)
        self.motor_torque_history = torch.roll(self.motor_torque_history, -1, dims=1)
        self.imu_history = torch.roll(self.imu_history, -1, dims=1)
        # 写入最新值
        self.motor_pos_history[:, -1] = self.dof_pos
        self.motor_vel_history[:, -1] = self.dof_vel
        self.motor_torque_history[:, -1] = self.torques
        self.imu_history[:, -1, :3] = self.base_ang_vel
        self.imu_history[:, -1, 3] = self.roll
        self.imu_history[:, -1, 4] = self.pitch

    def _get_delayed_obs(self):
        """获取带通信延迟的观测值。返回延迟后的dof_pos, dof_vel, torques, ang_vel, roll, pitch。
        使用高效的gather操作实现per-env不同延迟。"""
        buf_len = self.comm_delay_buf_len
        env_idx = torch.arange(self.num_envs, device=self.device)

        # 计算每个env的索引: buf_len - 1 - delay (最新是buf_len-1)
        pos_idx = (buf_len - 1 - self.motor_pos_delay).clamp(0, buf_len - 1)
        vel_idx = (buf_len - 1 - self.motor_vel_delay).clamp(0, buf_len - 1)
        torque_idx = (buf_len - 1 - self.motor_torque_delay).clamp(0, buf_len - 1)
        imu_idx = (buf_len - 1 - self.imu_delay).clamp(0, buf_len - 1)

        # gather: [num_envs, buf_len, dim] -> [num_envs, dim]
        delayed_pos = self.motor_pos_history[env_idx, pos_idx]
        delayed_vel = self.motor_vel_history[env_idx, vel_idx]
        delayed_torque = self.motor_torque_history[env_idx, torque_idx]
        delayed_imu = self.imu_history[env_idx, imu_idx]

        return (delayed_pos, delayed_vel, delayed_torque,
                delayed_imu[:, :3], delayed_imu[:, 3], delayed_imu[:, 4])

    def pre_physics_step(self, actions):
        """Override pre_physics_step to include force updates."""
        super().pre_physics_step(actions)
        if self.enable_force_curriculum:
            self._calculate_ee_forces()

    def _calculate_ee_forces(self):
        """Calculate end-effector forces based on FALCON's curriculum force approach.
        Optimized with vectorized operations and pre-computed constants."""
        # Increment counters (in-place)
        self.episode_length_counter.add_(1)
        self.force_duration_counter.add_(1)

        # Check which environments need new force application
        need_new_forces = self.force_duration_counter >= self.force_duration_target
        num_new = need_new_forces.sum().item()

        if num_new > 0:
            # Reset force duration counter and set new targets
            self.force_duration_counter[need_new_forces] = 0
            self.force_duration_target[need_new_forces] = torch.randint(
                self.randomize_force_duration[0],
                self.randomize_force_duration[1] + 1,
                (num_new,), device=self.device)

            num_links = len(self.force_apply_body_indices)

            # Generate random forces for all axes at once using pre-computed spans
            rand_vals = torch.rand(num_new, num_links, 3, device=self.device)
            new_forces = torch.empty(num_new, num_links, 3, device=self.device)
            new_forces[..., 0] = (rand_vals[..., 0] * self._force_x_span +
                                  self.apply_force_x_range[0])
            new_forces[..., 1] = (rand_vals[..., 1] * self._force_y_span +
                                  self.apply_force_y_range[0])
            new_forces[..., 2] = (rand_vals[..., 2] * self._force_z_span +
                                  self.apply_force_z_range[0])

            # Apply zero force probability (vectorized)
            zero_mask = torch.rand(
                num_new, num_links, 3, device=self.device) < self.zero_force_prob
            new_forces[zero_mask] = 0.0

            # Update forces for environments that need new forces
            self.applied_forces[need_new_forces] = new_forces

        # Apply phase-based modulation (triangular wave) - in-place operations
        if not hasattr(self, 'force_phase'):
            self.force_phase = torch.zeros(self.num_envs, device=self.device)
        self.force_phase.add_(0.02)
        self.force_phase.fmod_(2.0)

        # Triangular wave: 0->1->0 over phase [0, 2) - vectorized
        phase_modulation = torch.where(
            self.force_phase < 1.0,
            self.force_phase,
            2.0 - self.force_phase)

        # Apply curriculum scaling and phase modulation (fully vectorized)
        scale_factor = (self.force_scale * phase_modulation).unsqueeze(-1).unsqueeze(-1)
        final_forces = self.applied_forces * scale_factor

        # Apply forces to simulation using fully vectorized tensor operations
        num_links = len(self.force_apply_body_indices)
        if num_links > 0:
            # Pre-compute global indices for all envs and links (cache on first call)
            if not hasattr(self, '_force_global_indices'):
                env_offsets = (torch.arange(self.num_envs, device=self.device)
                               .unsqueeze(1) * self.num_bodies)
                link_offsets = self.force_apply_body_indices.unsqueeze(0)
                self._force_global_indices = (env_offsets + link_offsets).flatten()

            # Create forces tensor and assign using advanced indexing
            all_forces = torch.zeros(
                (self.num_envs * self.num_bodies, 3),
                device=self.device, dtype=torch.float)
            all_forces[self._force_global_indices] = final_forces.reshape(-1, 3)

            # Apply forces using the tensor API
            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(all_forces),
                None,
                gymapi.ENV_SPACE)

    # ==================== 未来动作一致性奖励 ====================
    # 这些奖励只在训练时生效，不影响sim2sim/sim2real部署

    def _reward_future_action_consistency(self):
        """奖励当前动作与未来目标动作的一致性。
        通过比较当前动作与未来帧目标动作的差异，鼓励平滑过渡。
        这有助于减少转身时的惯量过大问题。
        """
        # Use cached flag instead of hasattr for GPU efficiency
        if not getattr(self, '_has_cached_future_data', False):
            return torch.zeros(self.num_envs, device=self.device)

        current_target_dof = (self.actions * self.cfg.control.action_scale +
                              self.default_dof_pos_all)
        future_dof_pos = self._cached_future_dof_pos
        dof_diff = current_target_dof - future_dof_pos
        dof_err = torch.sum(dof_diff ** 2, dim=-1)

        return torch.exp(-0.5 * dof_err)

    def _reward_future_yaw_consistency(self):
        """奖励当前角速度与未来目标角速度的一致性。
        特别关注yaw方向的角速度，减少转身惯量。
        """
        # Use cached flag instead of hasattr for GPU efficiency
        if not getattr(self, '_has_cached_future_data', False):
            return torch.zeros(self.num_envs, device=self.device)

        current_yaw_ang_vel = self.base_ang_vel[:, 2]
        future_yaw_ang_vel = self._cached_future_yaw_ang_vel
        yaw_diff = current_yaw_ang_vel - future_yaw_ang_vel
        yaw_err = yaw_diff ** 2

        return torch.exp(-2.0 * yaw_err)

    def _reward_turning_smoothness(self):
        """惩罚转身时的角速度突变。
        通过比较当前角速度与上一帧角速度的差异来实现。
        """
        # Use pre-initialized buffer instead of hasattr + clone for GPU efficiency
        if not hasattr(self, '_last_base_ang_vel_initialized'):
            self._last_base_ang_vel = torch.zeros_like(self.base_ang_vel)
            self._last_base_ang_vel_initialized = True
            return torch.zeros(self.num_envs, device=self.device)

        ang_vel_change = self.base_ang_vel - self._last_base_ang_vel
        yaw_acc = ang_vel_change[:, 2] ** 2
        # In-place copy instead of clone
        self._last_base_ang_vel.copy_(self.base_ang_vel)

        return yaw_acc

    def _cache_future_motion_data(self, motion_data):
        """缓存未来动作数据用于奖励计算。
        复用 _get_unified_motion_data 中已采样的数据，避免重复采样。
        """
        # Use cached flag for efficiency
        if not hasattr(self, '_tar_motion_steps_future_reward'):
            self._has_cached_future_data = False
            return

        # 复用已有的motion_data，避免重复采样
        if motion_data is not None and motion_data['num_future_steps'] > 0:
            num_priv_steps = motion_data['num_priv_steps']
            self._cached_future_dof_pos = motion_data['dof_pos'][:, num_priv_steps]
            self._cached_future_yaw_ang_vel = motion_data['root_ang_vel_local'][:, num_priv_steps, 2]
            self._has_cached_future_data = True
        else:
            # Fallback: 单独采样奖励用的未来帧
            reward_steps = self._tar_motion_steps_future_reward
            motion_times = self._get_motion_times().unsqueeze(-1)
            obs_motion_times = reward_steps * self.dt + motion_times
            motion_ids_tiled = torch.broadcast_to(
                self._motion_ids.unsqueeze(-1), obs_motion_times.shape
            ).flatten()
            obs_motion_times = obs_motion_times.flatten()

            (root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel,
             _, _, _) = self._motion_lib.calc_motion_frame(
                motion_ids_tiled, obs_motion_times)

            num_reward_steps = len(reward_steps)
            dof_pos = dof_pos.reshape(self.num_envs, num_reward_steps, -1)
            root_ang_vel_local = quat_rotate_inverse(root_rot, root_ang_vel)
            root_ang_vel_local = root_ang_vel_local.reshape(
                self.num_envs, num_reward_steps, -1)

            self._cached_future_dof_pos = dof_pos[:, 0]
            self._cached_future_yaw_ang_vel = root_ang_vel_local[:, 0, 2]
            self._has_cached_future_data = True

    def compute_observations(self):
        """Override to cache future motion data for reward computation."""
        # 更新通信延迟buffer
        if getattr(self, 'enable_comm_delay', False):
            self._update_comm_delay_buffers()

        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        self.base_yaw_quat = quat_from_euler_xyz(
            0*self.yaw, 0*self.yaw, self.yaw)

        if self.obs_type == 'student_future':
            # Get unified motion data ONCE and reuse for both mimic obs and caching
            motion_data = self._get_unified_motion_data()
            priv_mimic_obs, mimic_obs, future_obs = self._get_mimic_obs(motion_data)
            # 缓存未来动作数据用于奖励计算 (reuse motion_data, no redundant call)
            self._cache_future_motion_data(motion_data)
        else:
            priv_mimic_obs, mimic_obs = self._get_mimic_obs()
            future_obs = None

        # 根据是否启用通信延迟选择观测源
        if getattr(self, 'enable_comm_delay', False):
            (delayed_pos, delayed_vel, _, delayed_ang_vel,
             delayed_roll, delayed_pitch) = self._get_delayed_obs()
            obs_dof_pos = delayed_pos
            obs_dof_vel = delayed_vel
            obs_ang_vel = delayed_ang_vel
            obs_imu = torch.stack((delayed_roll, delayed_pitch), dim=1)
        else:
            obs_dof_pos = self.dof_pos
            obs_dof_vel = self.dof_vel
            obs_ang_vel = self.base_ang_vel
            obs_imu = imu_obs

        proprio_obs_buf = torch.cat((
            obs_ang_vel * self.obs_scales.ang_vel,
            obs_imu,
            self.reindex(
                (obs_dof_pos - self.default_dof_pos_all) *
                self.obs_scales.dof_pos),
            self.reindex(obs_dof_vel * self.obs_scales.dof_vel),
            self.reindex(self.action_history_buf[:, -1]),
        ), dim=-1)

        if self.cfg.noise.add_noise and self.headless:
            noise_scale = min(
                self.total_env_steps_counter /
                (self.cfg.noise.noise_increasing_steps * 24), 1.)
            proprio_obs_buf += ((2 * torch.rand_like(proprio_obs_buf) - 1) *
                                self.noise_scale_vec * noise_scale)
        elif self.cfg.noise.add_noise and not self.headless:
            proprio_obs_buf += ((2 * torch.rand_like(proprio_obs_buf) - 1) *
                                self.noise_scale_vec)

        dof_vel_start_dim = 3 + 2 + self.dof_pos.shape[1]
        ankle_idx = [4, 5, 10, 11]
        proprio_obs_buf[:, [dof_vel_start_dim + i for i in ankle_idx]] = 0.

        key_body_pos = self.rigid_body_states[:, self._key_body_ids, :3]
        key_body_pos = key_body_pos - self.root_states[:, None, :3]
        if not self.global_obs:
            key_body_pos = convert_to_local_root_body_pos(
                self.root_states[:, 3:7], key_body_pos)
        key_body_pos = key_body_pos.reshape(self.num_envs, -1)

        priv_info = torch.cat((
            self.base_lin_vel,
            self.root_states[:, 0:3],
            self.root_states[:, 3:7],
            key_body_pos,
            self.contact_forces[:, self.feet_indices, 2] > 5.,
            self.mass_params_tensor,
            self.friction_coeffs_tensor,
            self.motor_strength[0] - 1,
            self.motor_strength[1] - 1,
        ), dim=-1)

        obs_buf = torch.cat((
            mimic_obs,
            proprio_obs_buf,
        ), dim=-1)

        priv_obs_buf = torch.cat((
            priv_mimic_obs,
            proprio_obs_buf,
            priv_info,
        ), dim=-1)

        self.privileged_obs_buf = priv_obs_buf

        if self.obs_type == 'priv':
            self.obs_buf = priv_obs_buf
        elif self.obs_type == 'student_future':
            obs_components = [
                obs_buf,
                self.obs_history_buf.view(self.num_envs, -1)
            ]
            if future_obs is not None:
                future_obs_flat = future_obs.view(self.num_envs, -1)
                obs_components.append(future_obs_flat)
            self.obs_buf = torch.cat(obs_components, dim=-1)
        else:
            self.obs_buf = torch.cat([
                obs_buf, self.obs_history_buf.view(self.num_envs, -1)
            ], dim=-1)

        if self.cfg.env.history_len > 0:
            reset_mask = (self.episode_length_buf <= 1)

            if reset_mask.any():
                reset_indices = reset_mask.nonzero(as_tuple=False).squeeze(-1)
                self.privileged_obs_history_buf[reset_indices] = \
                    priv_obs_buf[reset_indices].unsqueeze(1).expand(
                        -1, self.cfg.env.history_len, -1)

            continue_mask = ~reset_mask
            if continue_mask.any():
                continue_indices = continue_mask.nonzero(
                    as_tuple=False).squeeze(-1)
                self.privileged_obs_history_buf[continue_indices, :-1] = \
                    self.privileged_obs_history_buf[continue_indices, 1:]
                self.privileged_obs_history_buf[continue_indices, -1] = \
                    priv_obs_buf[continue_indices]

            if self.obs_type == 'priv':
                self.obs_history_buf[:] = self.privileged_obs_history_buf[:]
            else:
                if reset_mask.any():
                    reset_indices = reset_mask.nonzero(
                        as_tuple=False).squeeze(-1)
                    self.obs_history_buf[reset_indices] = \
                        obs_buf[reset_indices].unsqueeze(1).expand(
                            -1, self.cfg.env.history_len, -1)

                if continue_mask.any():
                    continue_indices = continue_mask.nonzero(
                        as_tuple=False).squeeze(-1)
                    self.obs_history_buf[continue_indices, :-1] = \
                        self.obs_history_buf[continue_indices, 1:]
                    self.obs_history_buf[continue_indices, -1] = \
                        obs_buf[continue_indices]

    def _reward_idle_penalty(self):
        """Penalize joint movement when close to target."""
        joint_vel_magnitude = torch.sum(torch.square(self.dof_vel), dim=1)
        return joint_vel_magnitude

    def _reward_feet_flat_contact(self):
        """奖励脚掌全部着地（两只脚都接触地面）。"""
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        both_feet_contact = torch.all(contact, dim=1).float()
        return both_feet_contact

    def _reward_ankle_roll_pitch_penalty(self):
        """惩罚脚掌roll和pitch关节偏离默认位置（L1范数）。
        Taks_T1的ankle关节索引: [4, 5, 10, 11] (左右脚的roll和pitch)
        """
        ankle_dof_idx = [4, 5, 10, 11]
        ankle_pos_error = self.dof_pos[:, ankle_dof_idx] - \
            self.default_dof_pos_all[:, ankle_dof_idx]
        return torch.sum(torch.abs(ankle_pos_error), dim=1)

    def _reward_neck_dof_penalty(self):
        """惩罚头部neck三关节偏离默认位置（L1范数）。
        Taks_T1的neck关节索引: [29, 30, 31] (neck yaw, roll, pitch)
        """
        # neck_dof_idx = [29, 30, 31]
        neck_dof_idx = [30]
        neck_pos_error = self.dof_pos[:, neck_dof_idx] - \
            self.default_dof_pos_all[:, neck_dof_idx]
        return torch.sum(torch.abs(neck_pos_error), dim=1)

    def _reward_pelvis_lin_acc(self):
        """Penalize pelvis linear acceleration to encourage smooth motion.
        Uses pre-computed base_lin_acc for GPU efficiency."""
        return torch.sum(torch.square(self.base_lin_acc), dim=1)

    def _reward_pelvis_ang_acc(self):
        """Penalize pelvis angular acceleration to encourage smooth rotation.
        Computes angular acceleration from last_root_vel."""
        ang_acc = (self.root_states[:, 10:13] - self.last_root_vel[:, 3:6]) / self.dt
        return torch.sum(torch.square(ang_acc), dim=1)

    def _reward_flat_orientation(self):
        """Penalize non-flat base orientation (roll and pitch).
        Uses projected_gravity which is already computed for GPU efficiency."""
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_head_orientation_penalty(self):
        """惩罚头部link姿态偏离直立（roll和pitch偏离0）。
        通过head link的四元数计算其相对于世界坐标系的roll和pitch。
        """
        if not hasattr(self, '_head_body_idx') or self._head_body_idx == -1:
            return torch.zeros(self.num_envs, device=self.device)
        
        # 获取头部link的四元数 (num_envs, 4)
        head_quat = self.rigid_body_states[:, self._head_body_idx, 3:7]
        
        # 计算头部相对于世界坐标系的roll和pitch
        # 使用projected gravity方法：将重力向量投影到头部坐标系
        gravity_vec = torch.tensor([0., 0., -1.], device=self.device)
        gravity_vec = gravity_vec.unsqueeze(0).expand(self.num_envs, -1)
        
        # 将世界坐标系的重力向量转换到头部坐标系
        head_projected_gravity = quat_rotate_inverse(head_quat, gravity_vec)
        
        # head_projected_gravity的x和y分量表示头部的roll和pitch偏离
        # 如果头部直立，projected_gravity应该是[0, 0, -1]
        return torch.sum(torch.square(head_projected_gravity[:, :2]), dim=1)