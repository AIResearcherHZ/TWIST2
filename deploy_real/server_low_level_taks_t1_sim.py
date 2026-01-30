import argparse
import json
import time
import numpy as np
import redis
import mujoco
import torch
from rich import print
from collections import deque
import mujoco.viewer as mjv
from tqdm import tqdm
import os
from data_utils.rot_utils import quatToEuler
from robot_control.fall_detector import FallDetector, FallProtectionController

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class OnnxPolicyWrapper:
    """Minimal wrapper so ONNXRuntime policies mimic TorchScript call signature."""

    def __init__(self, session, input_name, output_index=0):
        self.session = session
        self.input_name = input_name
        self.output_index = output_index

    def __call__(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(obs_tensor, torch.Tensor):
            obs_np = obs_tensor.detach().cpu().numpy()
        else:
            obs_np = np.asarray(obs_tensor, dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: obs_np})
        result = outputs[self.output_index]
        if not isinstance(result, np.ndarray):
            result = np.asarray(result, dtype=np.float32)
        return torch.from_numpy(result.astype(np.float32))


def load_onnx_policy(policy_path: str, device: str) -> OnnxPolicyWrapper:
    if ort is None:
        raise ImportError("onnxruntime is required for ONNX policy inference but is not installed.")
    providers = []
    available = ort.get_available_providers()
    if device.startswith('cuda'):
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        else:
            print("CUDAExecutionProvider not available in onnxruntime; falling back to CPUExecutionProvider.")
    providers.append('CPUExecutionProvider')
    session = ort.InferenceSession(policy_path, providers=providers)
    input_name = session.get_inputs()[0].name
    print(f"ONNX policy loaded from {policy_path} using providers: {session.get_providers()}")
    return OnnxPolicyWrapper(session, input_name)


class RealTimePolicyController:
    def __init__(self, 
                 xml_file, 
                 policy_path, 
                 device='cuda', 
                 record_video=False,
                 record_proprio=False,
                 measure_fps=False,
                 limit_fps=True,
                 policy_frequency=50,
                 fall_protection=True,
                 fall_roll_threshold=1.0,
                 fall_pitch_threshold=1.0,
                 ):
        self.measure_fps = measure_fps
        self.limit_fps = limit_fps
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.redis_pipeline = self.redis_client.pipeline()
        except Exception as e:
            print(f"Error connecting to Redis: {e}")

        self.device = device
        self.policy = load_onnx_policy(policy_path, device)

        # Create MuJoCo sim
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.model.opt.timestep = 0.001
        self.data = mujoco.MjData(self.model)
        
        self.viewer = mjv.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0
        self.viewer.cam.distance = 2.0

        self.num_actions = 32  # Taks_T1 has 32 DOF
        self.sim_duration = 100000.0
        self.sim_dt = 0.001
        # real frequency = 1 / (decimation * sim_dt)
        # ==> decimation = 1 / (real frequency * sim_dt)
        self.sim_decimation = 1 / (policy_frequency * self.sim_dt)
        print(f"sim_decimation: {self.sim_decimation}")

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

        # Taks_T1 specific configuration
        # Joint order from XML actuator section:
        # left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll (6)
        # right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle_pitch, right_ankle_roll (6)
        # waist_yaw, waist_roll, waist_pitch (3)
        # left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow, left_wrist_roll, left_wrist_yaw, left_wrist_pitch (7)
        # right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow, right_wrist_roll, right_wrist_yaw, right_wrist_pitch (7)
        # neck_yaw, neck_roll, neck_pitch (3)
        self.default_dof_pos = np.array([
                # left leg (6): hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                # right leg (6): hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll  
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                # waist (3): yaw, roll, pitch
                0.0, 0.0, 0.0,
                # left arm (7): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_yaw, wrist_pitch
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                # right arm (7): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_yaw, wrist_pitch
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                # neck (3): yaw, roll, pitch
                0.0, 0.0, 0.0,
            ])

        # MuJoCo qpos: [x, y, z, qw, qx, qy, qz, ...joint_angles...]
        # pelvis pos from XML: pos="0 0 0.75"
        self.mujoco_default_dof_pos = np.concatenate([
            np.array([0, 0, 0.75]),  # root position from XML pelvis pos
            np.array([1, 0, 0, 0]),  # quaternion (w, x, y, z)
            np.array([
                # left leg (6)
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                # right leg (6)
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                # waist (3)
                0.0, 0.0, 0.0,
                # left arm (7)
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                # right arm (7)
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                # neck (3)
                0.0, 0.0, 0.0,
            ])
        ])

        # Stiffness values aligned with taks_t1_mimic_distill_config.py control settings
        # hip_yaw: 100, hip_roll: 100, hip_pitch: 150, knee: 150, ankle: 40
        # waist: 150, shoulder: 20, elbow: 20, wrist: 10, neck: 2.5
        self.stiffness = np.array([
                # left leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
                100, 100, 100, 150, 40, 40,
                # right leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
                100, 100, 100, 150, 40, 40,
                # waist: yaw, roll, pitch
                150, 150, 150,
                # left arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_yaw, wrist_pitch
                40, 40, 40, 40, 20, 20, 20,
                # right arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_yaw, wrist_pitch
                40, 40, 40, 40, 20, 20, 20,
                # neck: yaw, roll, pitch
                10, 10, 10,
            ])
        # Damping values aligned with taks_t1_mimic_distill_config.py control settings
        # hip_yaw: 5, hip_roll: 5, hip_pitch: 5, knee: 5, ankle: 2
        # waist: 5, shoulder: 2, elbow: 2, wrist: 1, neck: 0.5
        self.damping = np.array([
                # left leg
                3.3, 3.3, 3.3, 5.0, 2.0, 2.0,
                # right leg
                3.3, 3.3, 3.3, 5.0, 2.0, 2.0,
                # waist
                5.0, 5.0, 5.0,
                # left arm
                2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0,
                # right arm
                2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0,
                # neck
                0.25, 0.25, 0.25,
            ])

        # Torque limits from XML actuatorfrcrange
        # left/right leg: hip_pitch=120, hip_roll=97, hip_yaw=97, knee=120, ankle=27
        # waist: all 97
        # arm: shoulder/elbow=27, wrist=7
        # neck: 3
        self.torque_limits = np.array([
                # left leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
                120, 97, 97, 120, 27, 27,
                # right leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
                120, 97, 97, 120, 27, 27,
                # waist: yaw, roll, pitch
                97, 97, 97,
                # left arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_yaw, wrist_pitch
                27, 27, 27, 27, 7, 7, 7,
                # right arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_yaw, wrist_pitch
                27, 27, 27, 27, 7, 7, 7,
                # neck: yaw, roll, pitch
                3, 3, 3,
            ])

        # Action scale from taks_t1_mimic_distill_config.py: action_scale = 0.5
        self.action_scale = np.array([
                # left leg (6)
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                # right leg (6)
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                # waist (3)
                0.5, 0.5, 0.5,
                # left arm (7)
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                # right arm (7)
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                # neck (3) - smaller scale for finer control
                0.5, 0.5, 0.5,
            ])

        self.ankle_idx = [4, 5, 10, 11]

        # Default mimic obs for Taks_T1 (38 dims: 6 root + 32 DOF)
        self.default_mimic_obs = np.concatenate([
            np.array([0, 0]),      # xy velocity
            np.array([0.75]),      # z position
            np.array([0, 0]),      # roll/pitch
            np.array([0]),         # yaw angular velocity
            self.default_dof_pos   # 32 DOF
        ]).astype(np.float32)

        self.n_mimic_obs = 38  # 6 + 32
        self.n_proprio = 3 + 2 + 3*32    # from config analysis
        self.n_obs_single = 38 + 3 + 2 + 3*32  # n_mimic_obs + n_proprio = 38 + 101 = 139
        self.history_len = 10
        
        self.total_obs_size = self.n_obs_single * (self.history_len + 1) + self.n_mimic_obs   # 139*11 + 38 = 1567

        print(f"TWIST2 Taks_T1 Controller Configuration:")
        print(f"  n_mimic_obs: {self.n_mimic_obs}")
        print(f"  n_proprio: {self.n_proprio}")
        print(f"  n_obs_single: {self.n_obs_single}")
        print(f"  history_len: {self.history_len}")
        print(f"  total_obs_size: {self.total_obs_size}")

        # Initialize history buffer
        self.proprio_history_buf = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_obs_single, dtype=np.float32))

        # Recording
        self.record_video = record_video
        self.record_proprio = record_proprio
        self.proprio_recordings = [] if record_proprio else None
        
        # Fall protection
        self.fall_protection_enabled = fall_protection
        if fall_protection:
            self.fall_detector = FallDetector(
                roll_threshold=fall_roll_threshold,
                pitch_threshold=fall_pitch_threshold,
                detection_window=5,
                detection_ratio=0.6
            )
            self.fall_controller = FallProtectionController(self.fall_detector)
            print(f"Fall protection enabled: roll_threshold={np.degrees(fall_roll_threshold):.1f}°, pitch_threshold={np.degrees(fall_pitch_threshold):.1f}°")
        else:
            self.fall_detector = None
            self.fall_controller = None

    def reset_sim(self):
        """Reset simulation to initial state"""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def reset(self, init_pos):
        """Reset robot to initial position"""
        self.data.qpos[:] = init_pos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def extract_data(self):
        """Extract robot state data"""
        n_dof = self.num_actions
        dof_pos = self.data.qpos[7:7+n_dof]
        dof_vel = self.data.qvel[6:6+n_dof]
        quat = self.data.qpos[3:7]
        ang_vel = self.data.qvel[3:6]
        sim_torque = self.data.ctrl
        return dof_pos, dof_vel, quat, ang_vel, sim_torque

    def run(self):
        """Main simulation loop"""
        print("Starting TWIST2 simulation...")

        # Video recording setup
        if self.record_video:
            import imageio
            mp4_writer = imageio.get_writer('twist2_simulation.mp4', fps=30)
        else:
            mp4_writer = None

        self.reset_sim()
        self.reset(self.mujoco_default_dof_pos)

        steps = int(self.sim_duration / self.sim_dt)
        pbar = tqdm(range(steps), desc="Simulating TWIST2...")

        # Send initial proprio to redis
        initial_obs = np.zeros(self.n_obs_single, dtype=np.float32)
        self.redis_pipeline.set("state_body_taks_t1", json.dumps(initial_obs.tolist()))
        self.redis_pipeline.set("state_hand_left_taks_t1", json.dumps(np.zeros(7).tolist()))
        self.redis_pipeline.set("state_hand_right_taks_t1", json.dumps(np.zeros(7).tolist()))
        self.redis_pipeline.execute()

        measure_fps = self.measure_fps
        fps_measurements = []
        fps_iteration_count = 0
        fps_measurement_target = 1000
        last_policy_time = None

        # Add policy execution FPS tracking for frequent printing
        policy_execution_times = []
        policy_step_count = 0
        policy_fps_print_interval = 100
        
        # Initialize fall protection scales
        kp_scale = 1.0
        kd_scale = 1.0
        pd_target = self.default_dof_pos.copy()

        try:
            for i in pbar:
                t_start = time.time()
                dof_pos, dof_vel, quat, ang_vel, sim_torque = self.extract_data()
                
                if i % self.sim_decimation == 0:
                    # Build proprioceptive observation
                    rpy = quatToEuler(quat)
                    obs_body_dof_vel = dof_vel.copy()
                    obs_body_dof_vel[self.ankle_idx] = 0.
                    obs_proprio = np.concatenate([
                        ang_vel * 0.25,
                        rpy[:2], # only use roll and pitch
                        (dof_pos - self.default_dof_pos),
                        obs_body_dof_vel * 0.05,
                        self.last_action
                    ])

                    print("roll/pitch (rpy):", rpy[:2], "obs_proprio roll/pitch:", obs_proprio[3:5])

                    # 打印关节编号与位置，便于对应 policy idx -> SDK jid
                    # dof_with_idx = [f"{i}:{dof_pos[i]:.4f}" for i in range(self.num_actions)]
                    # print("dof_pos (policy_idx:value):", ", ".join(dof_with_idx))

                    state_body = np.concatenate([
                        ang_vel,
                        rpy[:2],
                        dof_pos]) # 3+2+29 = 34 dims

                    # Send proprio to redis
                    
                    self.redis_pipeline.set("state_body_taks_t1", json.dumps(state_body.tolist()))
                    self.redis_pipeline.set("state_hand_left_taks_t1", json.dumps(np.zeros(7).tolist()))
                    self.redis_pipeline.set("state_hand_right_taks_t1", json.dumps(np.zeros(7).tolist()))
                    self.redis_pipeline.set("state_neck_taks_t1", json.dumps(np.zeros(2).tolist()))
                    self.redis_pipeline.set("t_state", int(time.time() * 1000)) # current timestamp in ms
                    self.redis_pipeline.execute()

                    # Get mimic obs from Redis
                    keys = ["action_body_taks_t1", "action_hand_left_taks_t1",
                            "action_hand_right_taks_t1", "action_neck_taks_t1"]
                    for key in keys:
                        self.redis_pipeline.get(key)
                    redis_results = self.redis_pipeline.execute()
                    
                    # Use default values if Redis data is None
                    if redis_results[0] is None:
                        action_mimic = self.default_mimic_obs.copy()
                    else:
                        action_mimic = np.array(json.loads(redis_results[0]),
                                                dtype=np.float32)
                    
                    if redis_results[3] is None:
                        action_neck = np.zeros(2, dtype=np.float32)
                    else:
                        action_neck = np.array(json.loads(redis_results[3]),
                                               dtype=np.float32)

                    # Handle G1 format (35 dims) -> Taks_T1 format (38 dims)
                    if len(action_mimic) == 35:
                        neck_joints = np.array([action_neck[0], 0.0, action_neck[1]],
                                               dtype=np.float32)
                        action_mimic = np.concatenate([action_mimic, neck_joints])

                    # Construct observation for TWIST2 controller
                    obs_full = np.concatenate([action_mimic, obs_proprio])
                    # Update history
                    obs_hist = np.array(self.proprio_history_buf).flatten()
                    self.proprio_history_buf.append(obs_full)
                    future_obs = action_mimic.copy()
                    # Combine all observations: current + history + future (set to current frame for now)
                    obs_buf = np.concatenate([obs_full, obs_hist, future_obs])
                    

                    # Ensure correct total observation size
                    assert obs_buf.shape[0] == self.total_obs_size, f"Expected {self.total_obs_size} obs, got {obs_buf.shape[0]}"
                    
                    # Run policy
                    obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()
                    
                    # Measure and track policy execution FPS
                    current_time = time.time()
                    if last_policy_time is not None:
                        policy_interval = current_time - last_policy_time
                        current_policy_fps = 1.0 / policy_interval
                        
                        # For frequent printing (every 100 steps)  
                        policy_execution_times.append(policy_interval)
                        policy_step_count += 1
                        
                        # Print policy execution FPS every 100 steps
                        if policy_step_count % policy_fps_print_interval == 0:
                            recent_intervals = policy_execution_times[-policy_fps_print_interval:]
                            avg_interval = np.mean(recent_intervals)
                            avg_execution_fps = 1.0 / avg_interval
                            print(f"Policy Execution FPS (last {policy_fps_print_interval} steps): {avg_execution_fps:.2f} Hz (avg interval: {avg_interval*1000:.2f}ms)")
                        
                        # For detailed measurement (every 1000 steps)
                        if measure_fps:
                            fps_measurements.append(current_policy_fps)
                            fps_iteration_count += 1
                            
                            if fps_iteration_count == fps_measurement_target:
                                avg_fps = np.mean(fps_measurements)
                                max_fps = np.max(fps_measurements)
                                min_fps = np.min(fps_measurements)
                                std_fps = np.std(fps_measurements)
                                print(f"\n=== Policy Execution FPS Results (steps {fps_iteration_count-fps_measurement_target+1}-{fps_iteration_count}) ===")
                                print(f"Average Policy FPS: {avg_fps:.2f}")
                                print(f"Max Policy FPS: {max_fps:.2f}")
                                print(f"Min Policy FPS: {min_fps:.2f}")
                                print(f"Std Policy FPS: {std_fps:.2f}")
                                print(f"Expected FPS (from decimation): {1.0/(self.sim_decimation * self.sim_dt):.2f}")
                                print(f"=================================================================================\n")
                                # Reset for next 1000 measurements
                                fps_measurements = []
                                fps_iteration_count = 0
                    last_policy_time = current_time
                    
                    self.last_action = raw_action
                    raw_action = np.clip(raw_action, -10., 10.)
                    scaled_actions = raw_action * self.action_scale
                    pd_target = scaled_actions + self.default_dof_pos
                    
                    # Fall protection check
                    kp_scale = 1.0
                    kd_scale = 1.0
                    if self.fall_protection_enabled and self.fall_controller:
                        kp_scale, kd_scale, is_fallen = self.fall_controller.check_and_protect_from_rpy(rpy[0], rpy[1])
                        if is_fallen:
                            pd_target = dof_pos  # 保持当前位置

                    # self.redis_client.set("action_low_level_unitree_g1", json.dumps(raw_action.tolist()))
                    
                    # Update camera to follow pelvis
                    pelvis_pos = self.data.xpos[self.model.body("pelvis").id]
                    self.viewer.cam.lookat = pelvis_pos
                    self.viewer.sync()
                    
                    if mp4_writer is not None:
                        img = self.viewer.read_pixels()
                        mp4_writer.append_data(img)

                    # Record proprio if enabled
                    if self.record_proprio:
                        proprio_data = {
                            'timestamp': time.time(),
                            'dof_pos': dof_pos.tolist(),
                            'dof_vel': dof_vel.tolist(),
                            'rpy': rpy.tolist(),
                            'ang_vel': ang_vel.tolist(),
                            'target_dof_pos': action_mimic.tolist()[-29:],
                        }
                        self.proprio_recordings.append(proprio_data)

               
                # PD control with fall protection scaling
                torque = (pd_target - dof_pos) * self.stiffness * kp_scale - dof_vel * self.damping * kd_scale
                torque = np.clip(torque, -self.torque_limits, self.torque_limits)
                
                self.data.ctrl[:] = torque
                mujoco.mj_step(self.model, self.data)
                
                # Sleep to maintain real-time pace
                if self.limit_fps:
                    elapsed = time.time() - t_start
                    if elapsed < self.sim_dt:
                        time.sleep(self.sim_dt - elapsed)

                    
        except Exception as e:
            print(f"Error in run: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if mp4_writer is not None:
                mp4_writer.close()
                print("Video saved as twist2_simulation.mp4")
            
            # Save proprio recordings if enabled
            if self.record_proprio and self.proprio_recordings:
                import pickle
                with open('twist2_proprio_recordings.pkl', 'wb') as f:
                    pickle.dump(self.proprio_recordings, f)
                print("Proprioceptive recordings saved as twist2_proprio_recordings.pkl")

            if self.viewer:
                self.viewer.close()
            print("Simulation finished.")


def main():
    parser = argparse.ArgumentParser(description='Run TWIST2 policy in simulation')
    parser.add_argument('--xml', type=str, default='../assets/g1/g1_sim2sim.xml',
                        help='Path to MuJoCo XML file')
    parser.add_argument('--policy', type=str, required=True,
                        help='Path to TWIST2 ONNX policy file')
    parser.add_argument('--device', type=str, 
                        default='cuda',
                        help='Device to run policy on (cuda/cpu)')
    parser.add_argument('--record_video', action='store_true',
                        help='Record video of simulation')
    parser.add_argument('--record_proprio', action='store_true',
                        help='Record proprioceptive data')
    parser.add_argument("--measure_fps", help="Measure FPS", default=0, type=int)
    parser.add_argument("--limit_fps", help="Limit FPS with sleep", default=1, type=int)
    parser.add_argument("--policy_frequency", help="Policy frequency", default=100, type=int)
    parser.add_argument('--fall_protection', action='store_true', default=True,
                        help='Enable fall protection')
    parser.add_argument('--no_fall_protection', action='store_true',
                        help='Disable fall protection')
    parser.add_argument('--fall_roll_threshold', type=float, default=1.0,
                        help='Fall detection roll threshold in radians (default: 1.0 rad = 57 deg)')
    parser.add_argument('--fall_pitch_threshold', type=float, default=1.0,
                        help='Fall detection pitch threshold in radians (default: 1.0 rad = 57 deg)')
    args = parser.parse_args()
    
    # Handle fall protection flag
    fall_protection = not args.no_fall_protection
    
    # Verify policy file exists
    if not os.path.exists(args.policy):
        print(f"Error: Policy file {args.policy} does not exist")
        return
    
    # Verify XML file exists
    if not os.path.exists(args.xml):
        print(f"Error: XML file {args.xml} does not exist")
        return
    
    print(f"Starting TWIST2 simulation controller...")
    print(f"  XML file: {args.xml}")
    print(f"  Policy file: {args.policy}")
    print(f"  Device: {args.device}")
    print(f"  Record video: {args.record_video}")
    print(f"  Record proprio: {args.record_proprio}")
    print(f"  Measure FPS: {args.measure_fps}")
    print(f"  Limit FPS: {args.limit_fps}")
    print(f"  Fall protection: {fall_protection}")
    if fall_protection:
        print(f"  Fall roll threshold: {np.degrees(args.fall_roll_threshold):.1f} deg")
        print(f"  Fall pitch threshold: {np.degrees(args.fall_pitch_threshold):.1f} deg")
    controller = RealTimePolicyController(
        xml_file=args.xml,
        policy_path=args.policy,
        device=args.device,
        record_video=args.record_video,
        record_proprio=args.record_proprio,
        measure_fps=args.measure_fps,
        limit_fps=args.limit_fps,
        policy_frequency=args.policy_frequency,
        fall_protection=fall_protection,
        fall_roll_threshold=args.fall_roll_threshold,
        fall_pitch_threshold=args.fall_pitch_threshold,
    )
    controller.run()


if __name__ == "__main__":
    main()
