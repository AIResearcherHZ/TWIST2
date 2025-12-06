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
            self._tar_motion_steps_future = torch.tensor(
                getattr(cfg.env, 'tar_motion_steps_future',
                        [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]),
                device=self.device, dtype=torch.long)

            self._tar_motion_steps_future_idx = torch.searchsorted(
                self._tar_motion_steps_priv, self._tar_motion_steps_future)

            self.future_mask = torch.ones(
                (self.num_envs, len(self._tar_motion_steps_future)),
                device=self.device, dtype=torch.bool)

            print(f"Future motion enabled with "
                  f"{len(self._tar_motion_steps_future)} future frames")
            print(f"Future steps: {self._tar_motion_steps_future.tolist()}")

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

    def _get_mimic_obs(self):
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

        self.zero_force_prob = getattr(
            force_cfg, 'zero_force_prob', [0.25, 0.25, 0.25])
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
        pass

    def _apply_motion_domain_randomization(self, root_pos, root_rot, root_vel,
                                           root_ang_vel, dof_pos, dof_vel):
        return root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel

    # ==================== 未来动作一致性奖励 ====================
    # 这些奖励只在训练时生效，不影响sim2sim/sim2real部署

    def _reward_future_action_consistency(self):
        """奖励当前动作与未来目标动作的一致性。
        通过比较当前动作与未来帧目标动作的差异，鼓励平滑过渡。
        这有助于减少转身时的惯量过大问题。
        """
        if not hasattr(self, '_cached_future_dof_pos'):
            return torch.zeros(self.num_envs, device=self.device)

        # 当前动作转换为目标关节位置
        current_target_dof = (self.actions * self.cfg.control.action_scale +
                              self.default_dof_pos_all)

        # 与未来帧的目标关节位置比较
        future_dof_pos = self._cached_future_dof_pos  # (num_envs, num_dof)
        dof_diff = current_target_dof - future_dof_pos
        dof_err = torch.sum(dof_diff ** 2, dim=-1)

        return torch.exp(-0.5 * dof_err)

    def _reward_future_yaw_consistency(self):
        """奖励当前角速度与未来目标角速度的一致性。
        特别关注yaw方向的角速度，减少转身惯量。
        """
        if not hasattr(self, '_cached_future_yaw_ang_vel'):
            return torch.zeros(self.num_envs, device=self.device)

        # 当前yaw角速度
        current_yaw_ang_vel = self.base_ang_vel[:, 2]

        # 未来目标yaw角速度
        future_yaw_ang_vel = self._cached_future_yaw_ang_vel  # (num_envs,)

        # 计算角速度差异
        yaw_diff = current_yaw_ang_vel - future_yaw_ang_vel
        yaw_err = yaw_diff ** 2

        return torch.exp(-2.0 * yaw_err)

    def _reward_turning_smoothness(self):
        """惩罚转身时的角速度突变。
        通过比较当前角速度与上一帧角速度的差异来实现。
        """
        if not hasattr(self, '_last_base_ang_vel'):
            self._last_base_ang_vel = self.base_ang_vel.clone()
            return torch.zeros(self.num_envs, device=self.device)

        # 角速度变化率（角加速度）
        ang_vel_change = self.base_ang_vel - self._last_base_ang_vel

        # 特别关注yaw方向的突变
        yaw_acc = ang_vel_change[:, 2] ** 2

        # 更新上一帧角速度
        self._last_base_ang_vel = self.base_ang_vel.clone()

        return yaw_acc

    def _cache_future_motion_data(self, motion_data):
        """缓存未来动作数据用于奖励计算。
        在compute_observations中调用。
        """
        if motion_data['num_future_steps'] == 0:
            return

        num_priv_steps = motion_data['num_priv_steps']

        # 缓存未来帧的目标关节位置（取第一个未来帧）
        self._cached_future_dof_pos = motion_data['dof_pos'][:, num_priv_steps]

        # 缓存未来帧的yaw角速度
        future_ang_vel_local = motion_data['root_ang_vel_local'][:, num_priv_steps]
        self._cached_future_yaw_ang_vel = future_ang_vel_local[:, 2]

    def compute_observations(self):
        """Override to cache future motion data for reward computation."""
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        self.base_yaw_quat = quat_from_euler_xyz(
            0*self.yaw, 0*self.yaw, self.yaw)

        if self.obs_type == 'student_future':
            priv_mimic_obs, mimic_obs, future_obs = self._get_mimic_obs()
            # 缓存未来动作数据用于奖励计算
            motion_data = self._get_unified_motion_data()
            self._cache_future_motion_data(motion_data)
        else:
            priv_mimic_obs, mimic_obs = self._get_mimic_obs()
            future_obs = None

        proprio_obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            imu_obs,
            self.reindex(
                (self.dof_pos - self.default_dof_pos_all) *
                self.obs_scales.dof_pos),
            self.reindex(self.dof_vel * self.obs_scales.dof_vel),
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
