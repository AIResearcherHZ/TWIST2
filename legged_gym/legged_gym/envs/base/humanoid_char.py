import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor

from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.base.legged_robot import LeggedRobot, euler_from_quaternion
from .humanoid_char_config import HumanoidCharCfg

from pose.utils import torch_utils

from termcolor import cprint

class HumanoidChar(LeggedRobot):
    def __init__(self, cfg: HumanoidCharCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True if not hasattr(cfg.env, 'debug_viz') else cfg.env.debug_viz
        self.init_done = False
        self._parse_cfg(self.cfg)
        self.domain_rand_general = self.cfg.domain_rand.domain_rand_general
        
        # Pre init for motion loading
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'
        
        BaseTask.__init__(self, self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        self.global_counter = 0
        self.total_env_steps_counter = 0
        
    def _init_buffers(self):
        super()._init_buffers()
        
        self._ref_body_pos = torch.zeros_like(self.rigid_body_states[..., :3])
        self.feet_force_sum = torch.ones(self.num_envs, 2, device=self.device)
        key_bodies = self.cfg.motion.key_bodies
        upper_key_bodies = self.cfg.motion.upper_key_bodies
        self._key_body_ids = self._build_body_ids_tensor(key_bodies)
        self._upper_key_body_ids = self._build_body_ids_tensor(upper_key_bodies)
        cprint(f"[HumanoidChar] key_bodies ids: {self._key_body_ids}", "green")
        cprint(f"[HumanoidChar] num of key bodies: {len(self._key_body_ids)}", "green")
        cprint(f"[HumanoidChar] upper_key_bodies ids: {self._upper_key_body_ids}", "green")
        cprint(f"[HumanoidChar] num of upper key bodies: {len(self._upper_key_body_ids)}", "green")
        self.init_yaw = torch.zeros(self.num_envs, device=self.device)
        
        # ==================== 新增域随机化缓冲区 ====================
        self._init_domain_rand_buffers()

    def _create_envs(self):
        super()._create_envs()
        if self.cfg.env.record_video:
            camera_props = gymapi.CameraProperties()
            camera_props.width = 720*2
            camera_props.height = 480*2
            self._rendering_camera_handles = []
            for i in range(self.num_envs):
                cam_pos = np.array([2, 0, 0.3])
                camera_handle = self.gym.create_camera_sensor(self.envs[i], camera_props)
                self._rendering_camera_handles.append(camera_handle)
                self.gym.set_camera_location(camera_handle, self.envs[i], gymapi.Vec3(*cam_pos), gymapi.Vec3(*0*cam_pos))
                
    def render_record(self, mode="rgb_array"):
        self.gym.step_graphics(self.sim)
        # self.gym.clear_lines(self.viewer)
        self.gym.render_all_camera_sensors(self.sim)
        imgs = []
        for i in range(self.num_envs):
            cam = self._rendering_camera_handles[i]
            root_pos = self.root_states[i, :3].cpu().numpy()
            cam_pos = root_pos + np.array([0, -1.5, 0.3])
            self.gym.set_camera_location(cam, self.envs[i], gymapi.Vec3(*cam_pos), gymapi.Vec3(*root_pos))
            img = self.gym.get_camera_image(self.sim, self.envs[i], cam, gymapi.IMAGE_COLOR)
            w, h = img.shape
            imgs.append(img.reshape([w, h // 4, 4]))
                
        return imgs
    
    def step(self, actions):
        actions = self.reindex(actions)
        actions.to(self.device)
        action_tensor = actions.clone()
        
        # 应用动作噪声
        action_tensor = self._apply_action_noise(action_tensor)
        
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), action_tensor[:, None, :].clone()], dim=1)
        
        if self.cfg.domain_rand.action_delay:
            # Curriculum for action delay - linear increase from 0 to 0.5 probability
            start_step = 5000 * 24  # Starting step for curriculum
            target_step = 20000 * 24  # Target step where probability reaches 0.5
            
            if self.total_env_steps_counter <= start_step:
                delay_prob = 0.0
            elif self.total_env_steps_counter >= target_step:
                delay_prob = 0.5
            else:
                # Linear interpolation between start and target steps
                delay_prob = 0.5 * (self.total_env_steps_counter - start_step) / (target_step - start_step)
            
            # Directly sample delay from [0,1] with probability [1-delay_prob, delay_prob]
            if torch.rand(1, device=self.device) < delay_prob:
                self.delay = torch.tensor(1.0, device=self.device, dtype=torch.float)
            else:
                self.delay = torch.tensor(0.0, device=self.device, dtype=torch.float)
                
            indices = -self.delay - 1
            action_tensor = self.action_history_buf[:, indices.long()]

        self.global_counter += 1
        self.total_env_steps_counter += 1
        clip_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale
        self.actions = torch.clip(action_tensor, -clip_actions, clip_actions).to(self.device)
        self.render()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
        
    def reset_idx(self, env_ids, init=False):
        if len(env_ids) == 0:
            return
            
        dof_pos = self.default_dof_pos_all.clone()

        # reset robot states
        self._reset_dofs(env_ids, dof_pos, torch.zeros_like(dof_pos))
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)  # no resample commands
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_root_vel[:] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.  # reset obs history buffer TODO no 0s
        self.contact_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        self.feet_land_time[env_ids] = 0.
        self._reset_buffers_extra(env_ids)
        
        # 重置域随机化缓冲区
        self._reset_domain_rand_buffers(env_ids)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['metric_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids] * self.reward_scales[key]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        return
    
    def _reset_buffers_extra(self, env_ids):
        pass
                                                                                                                                                                                                                                                                                                                                                                   
    def _reset_dofs(self, env_ids, dof_pos, dof_vel):
        self.dof_pos[env_ids] = dof_pos[env_ids] * torch_rand_float(0.8, 1.2, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = dof_vel[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt

        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)

        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()

        self.episode_length[env_ids] = self.episode_length_buf[env_ids].float()

        self.reset_idx(env_ids)

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_root_pos[:] = self.root_states[:, 0:3]
        self.last_root_rot[:] = self.root_states[:, 3:7]

        if self.cfg.rewards.regularization_scale_curriculum:
            if torch.mean(self.episode_length.float()).item()> 420.:
                self.cfg.rewards.regularization_scale *= (1. + self.cfg.rewards.regularization_scale_gamma)
            elif torch.mean(self.episode_length.float()).item() < 50.:
                self.cfg.rewards.regularization_scale *= (1. - self.cfg.rewards.regularization_scale_gamma)
            self.cfg.rewards.regularization_scale = max(min(self.cfg.rewards.regularization_scale, self.cfg.rewards.regularization_scale_range[1]), self.cfg.rewards.regularization_scale_range[0])


        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.draw_key_bodies_actual()
            self.draw_key_bodies_motion()
        
        
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0)
        self._resample_commands(env_ids.nonzero(as_tuple=False).flatten())

        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
    
    def _randomize_gravity(self, external_force = None):
        if self.cfg.domain_rand.randomize_gravity and external_force is None:
            min_gravity, max_gravity = self.cfg.domain_rand.gravity_range
            external_force = torch.rand(3, dtype=torch.float, device=self.device,
                                        requires_grad=False) * (max_gravity - min_gravity) + min_gravity


        sim_params = self.gym.get_sim_params(self.sim)
        if external_force is None:
            gravity = torch.Tensor([0, 0, -9.81]).to(self.device)
        else:
            gravity = external_force + torch.Tensor([0, 0, -9.81]).to(self.device)
        self.gravity_vec[:, :] = gravity.unsqueeze(0) / torch.norm(gravity)
        sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])
        self.gym.set_sim_params(self.sim, sim_params)
    
    # ==================== 新增域随机化方法 ====================
    def _init_domain_rand_buffers(self):
        """Initialize buffers for new domain randomization features."""
        # 编码器偏置 - 每个环境的每个关节有固定偏置
        if hasattr(self.cfg.domain_rand, 'encoder_noise') and self.cfg.domain_rand.encoder_noise:
            pos_bias_range = self.cfg.domain_rand.encoder_pos_bias_range
            vel_bias_range = self.cfg.domain_rand.encoder_vel_bias_range
            self.encoder_pos_bias = torch.rand(self.num_envs, self.num_dof, device=self.device) * \
                (pos_bias_range[1] - pos_bias_range[0]) + pos_bias_range[0]
            self.encoder_vel_bias = torch.rand(self.num_envs, self.num_dof, device=self.device) * \
                (vel_bias_range[1] - vel_bias_range[0]) + vel_bias_range[0]
        else:
            self.encoder_pos_bias = torch.zeros(self.num_envs, self.num_dof, device=self.device)
            self.encoder_vel_bias = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        
        # IMU偏置 - 每个环境有固定偏置
        if hasattr(self.cfg.domain_rand, 'imu_noise') and self.cfg.domain_rand.imu_noise:
            ang_bias_range = self.cfg.domain_rand.imu_ang_vel_bias_range
            self.imu_ang_vel_bias = torch.rand(self.num_envs, 3, device=self.device) * \
                (ang_bias_range[1] - ang_bias_range[0]) + ang_bias_range[0]
        else:
            self.imu_ang_vel_bias = torch.zeros(self.num_envs, 3, device=self.device)
        
        # 观测历史缓冲区 - 用于丢包时保持上一帧值
        self.last_obs_buf = None
        
        # 关节故障状态 - 每个环境的每个关节是否故障
        if hasattr(self.cfg.domain_rand, 'joint_failure') and self.cfg.domain_rand.joint_failure:
            self.joint_failure_mask = torch.ones(self.num_envs, self.num_dof, device=self.device)
        else:
            self.joint_failure_mask = torch.ones(self.num_envs, self.num_dof, device=self.device)
        
        # 传感器延迟尖峰状态
        if hasattr(self.cfg.domain_rand, 'sensor_latency_spike') and self.cfg.domain_rand.sensor_latency_spike:
            self.sensor_latency_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
            self.sensor_latency_obs = None
        
        # 重力偏置 - 模拟坡度
        if hasattr(self.cfg.domain_rand, 'slope_randomization') and self.cfg.domain_rand.slope_randomization:
            x_range = self.cfg.domain_rand.gravity_bias_x_range
            y_range = self.cfg.domain_rand.gravity_bias_y_range
            z_range = self.cfg.domain_rand.gravity_bias_z_range
            self.gravity_bias = torch.zeros(self.num_envs, 3, device=self.device)
            self.gravity_bias[:, 0] = torch.rand(self.num_envs, device=self.device) * (x_range[1] - x_range[0]) + x_range[0]
            self.gravity_bias[:, 1] = torch.rand(self.num_envs, device=self.device) * (y_range[1] - y_range[0]) + y_range[0]
            self.gravity_bias[:, 2] = torch.rand(self.num_envs, device=self.device) * (z_range[1] - z_range[0]) + z_range[0]
        else:
            self.gravity_bias = torch.zeros(self.num_envs, 3, device=self.device)
    
    def _reset_domain_rand_buffers(self, env_ids):
        """Reset domain randomization buffers for specified environments."""
        if len(env_ids) == 0:
            return
        
        # 重新采样编码器偏置
        if hasattr(self.cfg.domain_rand, 'encoder_noise') and self.cfg.domain_rand.encoder_noise:
            pos_bias_range = self.cfg.domain_rand.encoder_pos_bias_range
            vel_bias_range = self.cfg.domain_rand.encoder_vel_bias_range
            self.encoder_pos_bias[env_ids] = torch.rand(len(env_ids), self.num_dof, device=self.device) * \
                (pos_bias_range[1] - pos_bias_range[0]) + pos_bias_range[0]
            self.encoder_vel_bias[env_ids] = torch.rand(len(env_ids), self.num_dof, device=self.device) * \
                (vel_bias_range[1] - vel_bias_range[0]) + vel_bias_range[0]
        
        # 重新采样IMU偏置
        if hasattr(self.cfg.domain_rand, 'imu_noise') and self.cfg.domain_rand.imu_noise:
            ang_bias_range = self.cfg.domain_rand.imu_ang_vel_bias_range
            self.imu_ang_vel_bias[env_ids] = torch.rand(len(env_ids), 3, device=self.device) * \
                (ang_bias_range[1] - ang_bias_range[0]) + ang_bias_range[0]
        
        # 重新采样关节故障状态
        if hasattr(self.cfg.domain_rand, 'joint_failure') and self.cfg.domain_rand.joint_failure:
            self.joint_failure_mask[env_ids] = 1.0
            failure_prob = self.cfg.domain_rand.joint_failure_prob
            failure_mask = torch.rand(len(env_ids), self.num_dof, device=self.device) < failure_prob
            weak_factor = self.cfg.domain_rand.joint_failure_weak_factor
            self.joint_failure_mask[env_ids] = torch.where(failure_mask, 
                torch.full_like(self.joint_failure_mask[env_ids], weak_factor),
                self.joint_failure_mask[env_ids])
        
        # 重新采样重力偏置
        if hasattr(self.cfg.domain_rand, 'slope_randomization') and self.cfg.domain_rand.slope_randomization:
            x_range = self.cfg.domain_rand.gravity_bias_x_range
            y_range = self.cfg.domain_rand.gravity_bias_y_range
            z_range = self.cfg.domain_rand.gravity_bias_z_range
            self.gravity_bias[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * (x_range[1] - x_range[0]) + x_range[0]
            self.gravity_bias[env_ids, 1] = torch.rand(len(env_ids), device=self.device) * (y_range[1] - y_range[0]) + y_range[0]
            self.gravity_bias[env_ids, 2] = torch.rand(len(env_ids), device=self.device) * (z_range[1] - z_range[0]) + z_range[0]
    
    def _apply_action_noise(self, actions):
        """Apply noise to actions to simulate control signal imperfections."""
        if hasattr(self.cfg.domain_rand, 'action_noise') and self.cfg.domain_rand.action_noise:
            noise_std = self.cfg.domain_rand.action_noise_std
            noise = torch.randn_like(actions) * noise_std
            return actions + noise
        return actions
    
    def _get_noisy_dof_pos(self, dof_pos):
        """Apply encoder noise and bias to joint positions."""
        if hasattr(self.cfg.domain_rand, 'encoder_noise') and self.cfg.domain_rand.encoder_noise:
            noise_std = self.cfg.domain_rand.encoder_pos_noise_std
            noise = torch.randn_like(dof_pos) * noise_std
            return dof_pos + noise + self.encoder_pos_bias
        return dof_pos
    
    def _get_noisy_dof_vel(self, dof_vel):
        """Apply encoder noise and bias to joint velocities."""
        if hasattr(self.cfg.domain_rand, 'encoder_noise') and self.cfg.domain_rand.encoder_noise:
            noise_std = self.cfg.domain_rand.encoder_vel_noise_std
            noise = torch.randn_like(dof_vel) * noise_std
            return dof_vel + noise + self.encoder_vel_bias
        return dof_vel
    
    def _get_noisy_ang_vel(self, ang_vel):
        """Apply IMU noise and bias to angular velocity."""
        if hasattr(self.cfg.domain_rand, 'imu_noise') and self.cfg.domain_rand.imu_noise:
            noise_std = self.cfg.domain_rand.imu_ang_vel_noise_std
            noise = torch.randn_like(ang_vel) * noise_std
            return ang_vel + noise + self.imu_ang_vel_bias
        return ang_vel
    
    def _apply_observation_dropout(self, obs):
        """Apply random dropout to observations."""
        if hasattr(self.cfg.domain_rand, 'observation_dropout') and self.cfg.domain_rand.observation_dropout:
            dropout_prob = self.cfg.domain_rand.observation_dropout_prob
            dropout_mode = self.cfg.domain_rand.observation_dropout_mode
            
            dropout_mask = torch.rand_like(obs) < dropout_prob
            
            if dropout_mode == 'hold' and self.last_obs_buf is not None:
                obs = torch.where(dropout_mask, self.last_obs_buf, obs)
            elif dropout_mode == 'zero':
                obs = torch.where(dropout_mask, torch.zeros_like(obs), obs)
            
            self.last_obs_buf = obs.clone()
        return obs
    
    def _apply_sensor_latency_spike(self, obs):
        """Apply occasional sensor latency spikes."""
        if hasattr(self.cfg.domain_rand, 'sensor_latency_spike') and self.cfg.domain_rand.sensor_latency_spike:
            spike_prob = self.cfg.domain_rand.sensor_latency_spike_prob
            max_steps = self.cfg.domain_rand.sensor_latency_max_steps
            
            # 检查是否触发新的延迟尖峰
            new_spike = torch.rand(self.num_envs, device=self.device) < spike_prob
            self.sensor_latency_counter[new_spike] = torch.randint(1, max_steps + 1, 
                (new_spike.sum().item(),), device=self.device)
            
            # 对于有延迟的环境，保持旧观测
            if self.sensor_latency_obs is not None:
                latency_mask = self.sensor_latency_counter > 0
                obs[latency_mask] = self.sensor_latency_obs[latency_mask]
            
            # 更新计数器和缓存
            self.sensor_latency_counter = torch.clamp(self.sensor_latency_counter - 1, min=0)
            self.sensor_latency_obs = obs.clone()
        return obs
    
    def _apply_joint_failure(self, torques):
        """Apply joint failure effects to torques."""
        if hasattr(self.cfg.domain_rand, 'joint_failure') and self.cfg.domain_rand.joint_failure:
            return torques * self.joint_failure_mask
        return torques
    
    def _get_gravity_with_slope(self):
        """Get gravity vector with slope randomization applied."""
        if hasattr(self.cfg.domain_rand, 'slope_randomization') and self.cfg.domain_rand.slope_randomization:
            base_gravity = torch.tensor([0., 0., -9.81], device=self.device)
            gravity_with_bias = base_gravity.unsqueeze(0) + self.gravity_bias
            return gravity_with_bias
        return self.gravity_vec
    
    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.cfg.domain_rand.gravity_rand_interval = np.ceil(self.cfg.domain_rand.gravity_rand_interval_s / self.dt)
    
    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self.root_states[env_ids, 0:3] = root_pos
        self.root_states[env_ids, 3:7] = root_rot
        self.root_states[env_ids, 7:10] = root_vel
        self.root_states[env_ids, 10:13] = root_ang_vel

        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = dof_vel
        return

    def check_termination(self):
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        height_cutoff = self.root_states[:, 2] < self.cfg.rewards.termination_height
        
        roll_cut = torch.abs(self.roll) > self.cfg.rewards.termination_roll
        pitch_cut = torch.abs(self.pitch) > self.cfg.rewards.termination_pitch

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs

        self.reset_buf |= self.time_out_buf
        self.reset_buf |= height_cutoff
        self.reset_buf |= roll_cut
        self.reset_buf |= pitch_cut

    def get_first_contact(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        first_contact = (self.feet_air_time > 0.) * contact_filt
        return first_contact
    
    def update_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        self.feet_first_contact = (self.feet_air_time > 0) * contact_filt
        self.feet_air_time += self.dt
        self.feet_air_time *= ~contact_filt
        self.feet_land_time += self.dt
        self.feet_land_time = self.feet_land_time * contact
        
    def update_feet_force_sum(self):
        self.feet_force_sum += self.contact_forces[:, self.feet_indices, 2] * self.dt
        
    def _get_noise_scale_vec(self, cfg):
        noise_scale_vec = torch.zeros(1, self.cfg.env.n_proprio, device=self.device)
        if not self.cfg.noise.add_noise:
            return noise_scale_vec
        noise_start_dim = 2 + self.cfg.commands.num_commands
        noise_scale_vec[:, noise_start_dim:noise_start_dim+3] = self.cfg.noise.noise_scales.ang_vel
        noise_scale_vec[:, noise_start_dim+3:noise_start_dim+5] = self.cfg.noise.noise_scales.imu
        noise_scale_vec[:, noise_start_dim+5:noise_start_dim+5+self.num_dof] = self.cfg.noise.noise_scales.dof_pos
        noise_scale_vec[:, noise_start_dim+5+self.num_dof:noise_start_dim+5+2*self.num_dof] = self.cfg.noise.noise_scales.dof_vel
        return noise_scale_vec
    
    def compute_observations(self):
        
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        self.base_yaw_quat = quat_from_euler_xyz(0*self.yaw, 0*self.yaw, self.yaw)
        
        # 应用编码器噪声和IMU噪声
        noisy_ang_vel = self._get_noisy_ang_vel(self.base_ang_vel)
        noisy_dof_pos = self._get_noisy_dof_pos(self.dof_pos)
        noisy_dof_vel = self._get_noisy_dof_vel(self.dof_vel)
        
        obs_buf = torch.cat((
                            noisy_ang_vel * self.obs_scales.ang_vel,   # 3 dims
                            imu_obs,    # 2 dims
                            self.reindex((noisy_dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
                            self.reindex(noisy_dof_vel * self.obs_scales.dof_vel),
                            self.reindex(self.action_history_buf[:, -1]),
                            ),dim=-1)
        if self.cfg.noise.add_noise and self.headless:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec * min(self.total_env_steps_counter / (self.cfg.noise.noise_increasing_steps * 24),  1.)
        elif self.cfg.noise.add_noise and not self.headless:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec
        else:
            obs_buf += 0.

        if self.cfg.domain_rand.domain_rand_general:
            priv_latent = torch.cat((
                self.mass_params_tensor,
                self.friction_coeffs_tensor,
                self.motor_strength[0] - 1, 
                self.motor_strength[1] - 1,
                self.base_lin_vel,
            ), dim=-1)
        else:
            priv_latent = torch.zeros((self.num_envs, self.cfg.env.n_priv_latent), device=self.device)

        self.obs_buf = torch.cat([obs_buf, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        
        # 应用观测丢包和传感器延迟尖峰
        self.obs_buf = self._apply_observation_dropout(self.obs_buf)
        self.obs_buf = self._apply_sensor_latency_spike(self.obs_buf)


        if self.cfg.env.history_len > 0:
            self.obs_history_buf = torch.where(
                (self.episode_length_buf <= 1)[:, None, None],
                torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
                torch.cat([
                    self.obs_history_buf[:, 1:],
                    obs_buf.unsqueeze(1)
                ], dim=1)
            )
    
    def _build_body_ids_tensor(self, body_names):
        body_ids = []
        for body_name in body_names:
            body_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], body_name)
            assert body_id != -1, f"Body {body_name} not found"
            body_ids.append(body_id)
        body_ids = torch.tensor(body_ids, device=self.device, dtype=torch.long)
        return body_ids
            
    def draw_key_bodies_actual(self):
        
        if "g1" in self.__class__.__name__ or "G1" in self.__class__.__name__:
            sphere_size = 0.04
        elif "t1" in self.__class__.__name__ or "T1" in self.__class__.__name__:
            sphere_size = 0.04
        elif "toddy" in self.__class__.__name__ or "Toddy" in self.__class__.__name__:
            sphere_size = 0.02
        else:
            sphere_size = 0.04
            
        geom = gymutil.WireframeSphereGeometry(sphere_size, 32, 32, None, color=(1, 0, 0))
        rigid_body_pos = self.rigid_body_states[:, self._key_body_ids, :3].clone()
        for id in range(self.num_envs):
            for i in range(rigid_body_pos.shape[1]):
                pose = gymapi.Transform(gymapi.Vec3(rigid_body_pos[id, i, 0], rigid_body_pos[id, i, 1], rigid_body_pos[id, i, 2]), r=None)
                gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[id], pose)
    
    def draw_key_bodies_motion(self):
        # color = (0, 1, 0)
        color = (0, 1, 1)
        
        if "g1" in self.__class__.__name__ or "G1" in self.__class__.__name__:
            sphere_size = 0.04
        elif "t1" in self.__class__.__name__ or "T1" in self.__class__.__name__:
            sphere_size = 0.04
        elif "toddy" in self.__class__.__name__ or "Toddy" in self.__class__.__name__:
            sphere_size = 0.02
        else:
            sphere_size = 0.04
            
        geom = gymutil.WireframeSphereGeometry(sphere_size, 32, 32, None, color=color)
        ref_key_body_pos = self._ref_body_pos[:, self._key_body_ids, :3] - self._ref_root_pos[:, None, :]
        ref_key_body_pos_local = convert_to_local_root_body_pos(self._ref_root_rot, ref_key_body_pos)
        draw_root_pos = self.root_states[:, :3].clone()
        draw_root_pos[:, 2] = self._ref_root_pos[:, 2]
        ref_roll, ref_pitch, _ = euler_from_quaternion(self._ref_root_rot)
        draw_root_rot = quat_from_euler_xyz(ref_roll, ref_pitch, self.yaw)
        ref_key_body_pos_global = convert_to_global_root_body_pos(root_pos=draw_root_pos, root_rot=draw_root_rot, body_pos=ref_key_body_pos_local)
        for id in range(self.num_envs):
            for i in range(ref_key_body_pos.shape[1]):
                pose = gymapi.Transform(gymapi.Vec3(ref_key_body_pos_global[id, i, 0], ref_key_body_pos_global[id, i, 1], ref_key_body_pos_global[id, i, 2]), r=None)
                gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[id], pose)
        
        # # draw local upper key bodies
        # geom = gymutil.WireframeSphereGeometry(0.04, 32, 32, None, color=(0, 0, 1))
        # upper_key_body_pos = self._ref_body_pos[:, self._upper_key_body_ids, :3] - self._ref_root_pos[:, None, :]
        # upper_key_body_pos_local = convert_to_local_root_body_pos(self._ref_root_rot, upper_key_body_pos)
        # draw_root_pos = self.root_states[:, :3].clone()
        # draw_root_pos[:, 2] = self._ref_root_pos[:, 2]
        # upper_key_body_pos_global = convert_to_global_root_body_pos(root_pos=draw_root_pos, root_rot=self.root_states[:, 3:7], body_pos=upper_key_body_pos_local)
        # for id in range(self.num_envs):
        #     for i in range(upper_key_body_pos.shape[1]):
        #         pose = gymapi.Transform(gymapi.Vec3(upper_key_body_pos_global[id, i, 0], upper_key_body_pos_global[id, i, 1], upper_key_body_pos_global[id, i, 2]), r=None)
        #         gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[id], pose)

        # draw global whole body
        draw_gloabl = True
        if draw_gloabl:
            # color = (0, 1, 1)
            color = (0, 1, 0)
            geom = gymutil.WireframeSphereGeometry(sphere_size, 32, 32, None, color=color)
            ref_key_body_pos = self._ref_body_pos[:, self._key_body_ids, :3]        
            for id in range(self.num_envs):
                for i in range(ref_key_body_pos.shape[1]):
                    pose = gymapi.Transform(gymapi.Vec3(ref_key_body_pos[id, i, 0], ref_key_body_pos[id, i, 1], ref_key_body_pos[id, i, 2]), r=None)
                    gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[id], pose)
    

    ######### utils #########
    def get_episode_log(self, env_ids=0):
        log = {
            "ang vel": self.base_ang_vel[env_ids].cpu().numpy().tolist(),
            "dof pos": self.dof_pos[env_ids].cpu().numpy().tolist(),
            "dof vel": self.dof_vel[env_ids].cpu().numpy().tolist(),
            "action": self.action_history_buf[env_ids, -1].cpu().numpy().tolist(),
            "torque": self.torques[env_ids].cpu().numpy().tolist(),
        }
        
        return log
    
    ######### Rewards #########
    def compute_reward(self):
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() 
            if name in self.cfg.rewards.regularization_names:
                self.rew_buf += rew * self.reward_scales[name] * self.cfg.rewards.regularization_scale
            else: 
                self.rew_buf += rew * self.reward_scales[name]
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        if self.cfg.rewards.clip_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=-0.5)
        
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

                
@torch.jit.script
def convert_to_global_root_body_pos(root_pos, root_rot, body_pos):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    root_rot_expand = root_rot.unsqueeze(-2)
    root_rot_expand = root_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_root_rot_expand = root_rot_expand.reshape(root_rot_expand.shape[0] * root_rot_expand.shape[1], 
                                                   root_rot_expand.shape[2])
    flat_body_pos = body_pos.reshape(body_pos.shape[0] * body_pos.shape[1], body_pos.shape[2])
    flat_global_body_pos = torch_utils.quat_rotate(flat_root_rot_expand, flat_body_pos)
    global_body_pos = flat_global_body_pos.reshape(body_pos.shape[0], body_pos.shape[1], body_pos.shape[2]) # (num_envs, num_bodies, 3)
    global_body_pos += root_pos.unsqueeze(1)
    return global_body_pos


@torch.jit.script
def convert_to_local_root_body_pos(root_rot, body_pos):
    # type: (Tensor, Tensor) -> Tensor
    root_inv_rot = torch_utils.quat_conjugate(root_rot)
    root_rot_expand = root_inv_rot.unsqueeze(-2)
    root_rot_expand = root_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_root_rot_expand = root_rot_expand.reshape(root_rot_expand.shape[0] * root_rot_expand.shape[1], 
                                                   root_rot_expand.shape[2])
    flat_body_pos = body_pos.reshape(body_pos.shape[0] * body_pos.shape[1], body_pos.shape[2])
    flat_local_body_pos = torch_utils.quat_rotate(flat_root_rot_expand, flat_body_pos)
    local_body_pos = flat_local_body_pos.reshape(body_pos.shape[0], body_pos.shape[1], body_pos.shape[2])

    return local_body_pos