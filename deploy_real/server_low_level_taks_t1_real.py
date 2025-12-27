#!/usr/bin/env python3
"""
Taks-T1 Sim2Real 部署脚本
- 使用 taks SDK 的 MIT 控制模式
- 缓启动：线性5s升kp,kd站起来
- 缓关闭(Ctrl+C)：线性5s降kp,kd为0
"""
import sys
sys.path.insert(0, '/home/xhz/TWIST2/taks_sdk')

import argparse
import json
import time
import signal
import numpy as np
import torch
import redis
from collections import deque
from tqdm import tqdm
import os
from rich.console import Console
from rich.table import Table
from rich.live import Live

import taks
from data_utils.rot_utils import quatToEuler

try:
    import onnxruntime as ort
except ImportError:
    ort = None

# ============ 全局控制参数配置 (方便调试) ============
# 控制频率
CONTROL_FREQ = 50.0  # Hz
CONTROL_DT = 1.0 / CONTROL_FREQ  # 秒

# 缓启动/缓关闭时间 (秒)
RAMP_UP_TIME = 5.0
RAMP_DOWN_TIME = 5.0

# ============ 全局 KP/KD 配置 ============
# 左腿: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
# 右腿: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
# 腰部: yaw, roll, pitch
# 左臂: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_yaw, wrist_pitch
# 右臂: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_yaw, wrist_pitch
# 脖子: yaw, roll, pitch

GLOBAL_KP = np.array([
    # 左腿 (6)
    50, 50, 50, 50, 40, 40,
    # 右腿 (6)
    50, 50, 50, 50, 40, 40,
    # 腰部 (3)
    50, 50, 50,
    # 左臂 (7)
    10, 10, 10, 10, 10, 10, 10,
    # 右臂 (7)
    10, 10, 10, 10, 10, 10, 10,
    # 脖子 (3)
    5, 5, 5,
], dtype=np.float32)

GLOBAL_KD = np.array([
    # 左腿 (6)
    2, 2, 2, 4, 2, 2,
    # 右腿 (6)
    2, 2, 2, 4, 2, 2,
    # 腰部 (3)
    4, 4, 4,
    # 左臂 (7)
    5, 5, 5, 5, 1, 1, 1,
    # 右臂 (7)
    5, 5, 5, 5, 1, 1, 1,
    # 脖子 (3)
    0.5, 0.5, 0.5,
], dtype=np.float32)

# Taks-T1 关节ID映射 (SDK中的关节编号)
# 全身关节: 1-7(右臂), 9-15(左臂), 17-19(腰), 20-22(脖子), 23-28(右腿), 29-34(左腿)
# 但policy输出顺序是: 左腿(6), 右腿(6), 腰(3), 左臂(7), 右臂(7), 脖子(3) = 32 DOF
POLICY_TO_SDK_JOINT_MAP = {
    # 左腿 (policy idx 0-5) -> SDK j29-j34
    0: 29,  # left_hip_pitch
    1: 30,  # left_hip_roll
    2: 31,  # left_hip_yaw
    3: 32,  # left_knee
    4: 33,  # left_ankle_pitch
    5: 34,  # left_ankle_roll
    # 右腿 (policy idx 6-11) -> SDK j23-j28
    6: 23,   # right_hip_pitch
    7: 24,   # right_hip_roll
    8: 25,   # right_hip_yaw
    9: 26,   # right_knee
    10: 27,  # right_ankle_pitch
    11: 28,  # right_ankle_roll
    # 腰部 (policy idx 12-14) -> SDK j17-j19
    12: 17,  # waist_yaw
    13: 18,  # waist_roll
    14: 19,  # waist_pitch
    # 左臂 (policy idx 15-21) -> SDK j9-j15
    15: 9,   # left_shoulder_pitch
    16: 10,  # left_shoulder_roll
    17: 11,  # left_shoulder_yaw
    18: 12,  # left_elbow
    19: 13,  # left_wrist_roll
    20: 14,  # left_wrist_yaw
    21: 15,  # left_wrist_pitch
    # 右臂 (policy idx 22-28) -> SDK j1-j7
    22: 1,   # right_shoulder_pitch
    23: 2,   # right_shoulder_roll
    24: 3,   # right_shoulder_yaw
    25: 4,   # right_elbow
    26: 5,   # right_wrist_roll
    27: 6,   # right_wrist_yaw
    28: 7,   # right_wrist_pitch
    # 脖子 (policy idx 29-31) -> SDK j20-j22
    29: 20,  # neck_yaw
    30: 21,  # neck_roll
    31: 22,  # neck_pitch
}

# SDK关节ID到policy索引的反向映射
SDK_TO_POLICY_JOINT_MAP = {v: k for k, v in POLICY_TO_SDK_JOINT_MAP.items()}

# 关节名称映射
JOINT_NAMES = {
    0: "L_hip_pitch", 1: "L_hip_roll", 2: "L_hip_yaw", 3: "L_knee", 4: "L_ankle_pitch", 5: "L_ankle_roll",
    6: "R_hip_pitch", 7: "R_hip_roll", 8: "R_hip_yaw", 9: "R_knee", 10: "R_ankle_pitch", 11: "R_ankle_roll",
    12: "waist_yaw", 13: "waist_roll", 14: "waist_pitch",
    15: "L_shoulder_pitch", 16: "L_shoulder_roll", 17: "L_shoulder_yaw", 18: "L_elbow",
    19: "L_wrist_roll", 20: "L_wrist_yaw", 21: "L_wrist_pitch",
    22: "R_shoulder_pitch", 23: "R_shoulder_roll", 24: "R_shoulder_yaw", 25: "R_elbow",
    26: "R_wrist_roll", 27: "R_wrist_yaw", 28: "R_wrist_pitch",
    29: "neck_yaw", 30: "neck_roll", 31: "neck_pitch",
}


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


class TaksT1RealController:
    """Taks-T1 真机控制器"""
    
    def __init__(self, 
                 policy_path,
                 device='cuda',
                 server_ip='192.168.1.208',
                 cmd_port=5555,
                 kp_override=None,
                 kd_override=None,
                 control_freq_override=None):
        
        self.device = device
        self.policy = load_onnx_policy(policy_path, device)
        self.server_ip = server_ip
        self.cmd_port = cmd_port
        
        # 使用全局变量或覆盖值
        self.kp = kp_override if kp_override is not None else GLOBAL_KP.copy()
        self.kd = kd_override if kd_override is not None else GLOBAL_KD.copy()
        self.control_freq = control_freq_override if control_freq_override is not None else CONTROL_FREQ
        self.control_dt = 1.0 / self.control_freq
        
        # Redis for motion server communication
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.redis_pipeline = self.redis_client.pipeline()
        except Exception as e:
            print(f"Warning: Redis connection failed: {e}")
        
        # Robot state
        self.robot = None
        self.imu = None
        self.num_actions = 32
        
        # Default positions (all zeros for Taks-T1)
        self.default_dof_pos = np.zeros(self.num_actions, dtype=np.float32)
        
        # Scaling factors
        self.ang_vel_scale = 0.25
        self.dof_vel_scale = 0.05
        self.dof_pos_scale = 1.0
        self.action_scale = 0.5
        self.ankle_idx = [4, 5, 10, 11]
        
        # Observation structure
        self.n_mimic_obs = 38  # 6 + 32
        self.n_proprio = 3 + 2 + 3 * 32  # ang_vel(3) + rpy(2) + dof_pos(32) + dof_vel(32) + last_action(32)
        self.n_obs_single = self.n_mimic_obs + self.n_proprio  # 38 + 101 = 139
        self.history_len = 10
        self.total_obs_size = self.n_obs_single * (self.history_len + 1) + self.n_mimic_obs  # 139*11 + 38 = 1567
        
        print(f"Taks-T1 Real Controller Configuration:")
        print(f"  Control Frequency: {self.control_freq:.1f} Hz (dt={self.control_dt:.4f}s)")
        print(f"  KP range: [{self.kp.min():.1f}, {self.kp.max():.1f}]")
        print(f"  KD range: [{self.kd.min():.1f}, {self.kd.max():.1f}]")
        print(f"  n_mimic_obs: {self.n_mimic_obs}")
        print(f"  n_proprio: {self.n_proprio}")
        print(f"  n_obs_single: {self.n_obs_single}")
        print(f"  history_len: {self.history_len}")
        print(f"  total_obs_size: {self.total_obs_size}")
        
        # History buffer
        self.proprio_history_buf = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_obs_single, dtype=np.float32))
        
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        
        # Control state
        self.running = False
        self.shutdown_requested = False
        self.current_kp_scale = 0.0
        self.current_kd_scale = 0.0
        
        # 频率统计（记录调用时间点，用于计算间隔）
        self.get_state_timestamps = deque(maxlen=100)
        self.control_mit_timestamps = deque(maxlen=100)
        self.loop_timestamps = deque(maxlen=100)
        self.console = Console()
        self.print_counter = 0
        self.print_interval = int(self.control_freq)  # 每秒打印一次表格
        self.target_dt = self.control_dt  # 目标控制周期
        
        # Default mimic obs
        self.default_mimic_obs = np.concatenate([
            np.array([0, 0]),      # xy velocity
            np.array([0.75]),      # z position
            np.array([0, 0]),      # roll/pitch
            np.array([0]),         # yaw angular velocity
            self.default_dof_pos   # 32 DOF
        ]).astype(np.float32)
        
    def connect_robot(self):
        """连接机器人"""
        print(f"连接到 Taks-T1 服务器: {self.server_ip}:{self.cmd_port}")
        taks.connect(self.server_ip, cmd_port=self.cmd_port)
        
        print("注册 Taks-T1 设备...")
        self.robot = taks.register("Taks-T1")
        print("✓ Taks-T1 注册成功")
        time.sleep(0.5)
        
        print("注册 Taks-T1-imu 设备...")
        self.imu = taks.register("Taks-T1-imu")
        print("✓ Taks-T1-imu 注册成功")
        
        print("等待10秒后开始控制...")
        for i in range(10, 0, -1):
            print(f"  {i}...")
            time.sleep(1)
        print("开始控制!")
        
    def disconnect_robot(self):
        """断开机器人连接"""
        taks.disconnect()
        print("✓ 已断开连接")
        
    def get_robot_state(self):
        """获取机器人状态"""
        t_get_start = time.time()
        # 获取关节状态
        joint_states = self.robot.GetState()
        
        # 获取IMU数据
        quat_data = self.imu.get_quat()
        ang_vel_data = self.imu.get_ang_vel()
        
        # 解析关节位置和速度 (按policy顺序排列)
        dof_pos = np.zeros(self.num_actions, dtype=np.float32)
        dof_vel = np.zeros(self.num_actions, dtype=np.float32)
        
        if joint_states:
            for sdk_jid, state in joint_states.items():
                if sdk_jid in SDK_TO_POLICY_JOINT_MAP:
                    policy_idx = SDK_TO_POLICY_JOINT_MAP[sdk_jid]
                    dof_pos[policy_idx] = state.get('pos', 0.0)
                    dof_vel[policy_idx] = state.get('vel', 0.0)
        
        # 解析四元数 (w, x, y, z)
        if quat_data and isinstance(quat_data, dict) and 'w' in quat_data:
            quat = np.array([quat_data['w'], quat_data['x'], quat_data['y'], quat_data['z']], dtype=np.float32)
            # print(f"[DEBUG get_robot_state] quat: {quat}")
        else:
            quat = np.array([1, 0, 0, 0], dtype=np.float32)
            # print(f"[DEBUG get_robot_state] quat_data: {quat_data}")
        
        # 解析角速度
        if ang_vel_data:
            ang_vel = np.array([ang_vel_data['x'], ang_vel_data['y'], ang_vel_data['z']], dtype=np.float32)
        else:
            ang_vel = np.zeros(3, dtype=np.float32)
        
        # 记录get时间点
        self.get_state_timestamps.append(time.time())
        
        return dof_pos, dof_vel, quat, ang_vel
    
    def send_mit_command(self, target_pos, kp_scale, kd_scale):
        """发送MIT控制命令"""
        # 构建MIT控制数据
        mit_data = {}
        for policy_idx in range(self.num_actions):
            sdk_jid = POLICY_TO_SDK_JOINT_MAP[policy_idx]
            mit_data[sdk_jid] = {
                'kp': float(self.kp[policy_idx] * kp_scale),
                'kd': float(self.kd[policy_idx] * kd_scale),
                'q': float(target_pos[policy_idx]),
                'dq': 0.0,
                'tau': 0.0
            }
        
        # DEBUG: 打印所有关节的目标位置
        # print(f"[DEBUG send_mit_command] kp_scale: {kp_scale}")
        # print(f"[DEBUG send_mit_command] kd_scale: {kd_scale}")
        # print(f"[DEBUG send_mit_command] target_pos: {target_pos}")
        # print(f"[DEBUG send_mit_command] mit_data sample (jid=11): {mit_data.get(11)}")
        
        # 发送命令
        self.robot.controlMIT(joints=mit_data)
        self.control_mit_timestamps.append(time.time())
        
        # 每隔一定次数打印MIT数据表格
        self.print_counter += 1
        if self.print_counter >= self.print_interval:
            self.print_counter = 0
            self._print_mit_table(mit_data)
    
    def _print_ramp_table(self, dof_pos, target_pos, kp_scale, kd_scale, phase="Ramp"):
        """用rich表格打印缓启动/缓关闭位置信息"""
        table = Table(title=f"{phase} | KP Scale: {kp_scale:.2f} | KD Scale: {kd_scale:.2f}")
        table.add_column("Policy ID", style="cyan", justify="center")
        table.add_column("SDK ID", style="magenta", justify="center")
        table.add_column("Name", style="green")
        table.add_column("Current Pos", style="yellow", justify="right")
        table.add_column("Target Pos", style="blue", justify="right")
        table.add_column("Error", style="red", justify="right")
        
        for policy_idx in range(self.num_actions):
            sdk_jid = POLICY_TO_SDK_JOINT_MAP[policy_idx]
            error = dof_pos[policy_idx] - target_pos[policy_idx]
            table.add_row(
                str(policy_idx),
                str(sdk_jid),
                JOINT_NAMES.get(policy_idx, "unknown"),
                f"{dof_pos[policy_idx]:.4f}",
                f"{target_pos[policy_idx]:.4f}",
                f"{error:.4f}"
            )
        
        self.console.clear()
        self.console.print(table)
    
    def ramp_up(self):
        """缓启动：线性5s升kp,kd，目标位置固定为0.0"""
        print(f"缓启动中 ({RAMP_UP_TIME}s)...")
        start_time = time.time()
        step_count = 0
        print_interval = int(self.control_freq)  # 每秒打印一次
        
        # 目标位置固定为全零
        target_pos = np.zeros(self.num_actions, dtype=np.float32)
        
        while True:
            elapsed = time.time() - start_time
            if elapsed >= RAMP_UP_TIME:
                self.current_kp_scale = 1.0
                self.current_kd_scale = 1.0
                break
            
            # 线性插值
            self.current_kp_scale = elapsed / RAMP_UP_TIME
            self.current_kd_scale = elapsed / RAMP_UP_TIME
            
            # 获取当前状态
            dof_pos, _, _, _ = self.get_robot_state()
            # 发送固定的零位置目标
            self.send_mit_command(target_pos, self.current_kp_scale, self.current_kd_scale)
            
            # 每隔一定次数打印表格
            step_count += 1
            if step_count >= print_interval:
                step_count = 0
                self._print_ramp_table(dof_pos, target_pos, 
                                      self.current_kp_scale, self.current_kd_scale, 
                                      phase=f"Ramp Up ({elapsed:.1f}s/{RAMP_UP_TIME}s)")
            
            time.sleep(self.control_dt)
        
        print(f"\n✓ 缓启动完成")
    
    def ramp_down(self):
        """缓关闭：线性5s降kp,kd为0，目标位置固定为0.0"""
        print(f"\n缓关闭中 ({RAMP_DOWN_TIME}s)...")
        start_time = time.time()
        initial_kp_scale = self.current_kp_scale
        initial_kd_scale = self.current_kd_scale
        step_count = 0
        print_interval = int(self.control_freq)  # 每秒打印一次
        
        # 目标位置固定为全零
        target_pos = np.zeros(self.num_actions, dtype=np.float32)
        
        while True:
            elapsed = time.time() - start_time
            if elapsed >= RAMP_DOWN_TIME:
                self.current_kp_scale = 0.0
                self.current_kd_scale = 0.0
                break
            
            # 线性插值降低
            progress = elapsed / RAMP_DOWN_TIME
            self.current_kp_scale = initial_kp_scale * (1.0 - progress)
            self.current_kd_scale = initial_kd_scale * (1.0 - progress)
            
            # 获取当前位置
            dof_pos, _, _, _ = self.get_robot_state()
            # 发送固定的零位置目标
            self.send_mit_command(target_pos, self.current_kp_scale, self.current_kd_scale)
            
            # 每隔一定次数打印表格
            step_count += 1
            if step_count >= print_interval:
                step_count = 0
                self._print_ramp_table(dof_pos, target_pos, 
                                      self.current_kp_scale, self.current_kd_scale, 
                                      phase=f"Ramp Down ({elapsed:.1f}s/{RAMP_DOWN_TIME}s)")
            
            time.sleep(self.control_dt)
        
        # 最后发送零位置、零力矩
        self.send_mit_command(target_pos, 0.0, 0.0)
        print(f"\n✓ 缓关闭完成")
    
    def _print_mit_table(self, mit_data):
        """用rich表格打印MIT数据"""
        # 计算频率（基于调用间隔）
        def calc_freq(timestamps):
            if len(timestamps) < 2:
                return 0.0
            intervals = np.diff(list(timestamps))
            return 1.0 / np.mean(intervals) if len(intervals) > 0 and np.mean(intervals) > 0 else 0.0
        
        get_freq = calc_freq(self.get_state_timestamps)
        mit_freq = calc_freq(self.control_mit_timestamps)
        loop_freq = calc_freq(self.loop_timestamps)
        target_freq = 1.0 / self.target_dt
        
        table = Table(title=f"MIT Control Data | Loop: {loop_freq:.1f}/{target_freq:.0f}Hz | Get: {get_freq:.1f}Hz | Send: {mit_freq:.1f}Hz")
        table.add_column("Policy ID", style="cyan", justify="center")
        table.add_column("SDK ID", style="magenta", justify="center")
        table.add_column("Name", style="green")
        table.add_column("Position", style="yellow", justify="right")
        table.add_column("KP", style="blue", justify="right")
        table.add_column("KD", style="blue", justify="right")
        table.add_column("Tau", style="red", justify="right")
        
        for policy_idx in range(self.num_actions):
            sdk_jid = POLICY_TO_SDK_JOINT_MAP[policy_idx]
            data = mit_data[sdk_jid]
            table.add_row(
                str(policy_idx),
                str(sdk_jid),
                JOINT_NAMES.get(policy_idx, "unknown"),
                f"{data['q']:.4f}",
                f"{data['kp']:.2f}",
                f"{data['kd']:.2f}",
                f"{data['tau']:.2f}"
            )
        
        self.console.clear()
        self.console.print(table)
    
    def signal_handler(self, signum, frame):
        """处理Ctrl+C信号"""
        print("\n\n收到退出信号 (Ctrl+C)...")
        self.shutdown_requested = True
    
    def run(self):
        """主控制循环"""
        # 注册信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            # 连接机器人
            self.connect_robot()
            
            # 缓启动
            self.ramp_up()
            
            self.running = True
            print("开始主控制循环...")
            print("按 Ctrl+C 安全退出")
            
            step_count = 0
            next_loop_time = time.time()  # 下一次循环的目标时间
            
            # ========== 注释掉policy执行循环，只测试缓慢启动 ==========
            # while self.running and not self.shutdown_requested:
            #     loop_start = time.time()
            #     
            #     # 等待到目标时间点
            #     sleep_time = next_loop_time - loop_start
            #     if sleep_time > 0:
            #         time.sleep(sleep_time)
            #     
            #     actual_start = time.time()
            #     
            #     # 获取机器人状态
            #     dof_pos, dof_vel, quat, ang_vel = self.get_robot_state()
            #     
            #     # 计算RPY
            #     rpy = quatToEuler(quat)
            #     
            #     # 构建本体感知观测
            #     obs_body_dof_vel = dof_vel.copy()
            #     obs_body_dof_vel[self.ankle_idx] = 0.0
            #     
            #     obs_proprio = np.concatenate([
            #         ang_vel * self.ang_vel_scale,
            #         rpy[:2],  # roll, pitch
            #         (dof_pos - self.default_dof_pos) * self.dof_pos_scale,
            #         obs_body_dof_vel * self.dof_vel_scale,
            #         self.last_action
            #     ])
            #     
            #     # 发送状态到Redis (用于motion server)
            #     if self.redis_client:
            #         state_body = np.concatenate([ang_vel, rpy[:2], dof_pos])
            #         self.redis_pipeline.set("state_body_taks_t1", json.dumps(state_body.tolist()))
            #         self.redis_pipeline.set("state_hand_left_taks_t1", json.dumps(np.zeros(7).tolist()))
            #         self.redis_pipeline.set("state_hand_right_taks_t1", json.dumps(np.zeros(7).tolist()))
            #         self.redis_pipeline.set("state_neck_taks_t1", json.dumps(np.zeros(2).tolist()))
            #         self.redis_pipeline.set("t_state", int(time.time() * 1000))
            #         self.redis_pipeline.execute()
            #     
            #     # 从Redis获取mimic观测
            #     action_mimic = self.default_mimic_obs.copy()
            #     action_neck = np.zeros(2, dtype=np.float32)
            #     
            #     if self.redis_client:
            #         keys = ["action_body_taks_t1", "action_neck_taks_t1"]
            #         for key in keys:
            #             self.redis_pipeline.get(key)
            #         redis_results = self.redis_pipeline.execute()
            #         
            #         if redis_results[0] is not None:
            #             action_mimic = np.array(json.loads(redis_results[0]), dtype=np.float32)
            #         if redis_results[1] is not None:
            #             action_neck = np.array(json.loads(redis_results[1]), dtype=np.float32)
            #         
            #         # Handle G1 format (35 dims) -> Taks_T1 format (38 dims)
            #         if len(action_mimic) == 35:
            #             neck_joints = np.array([action_neck[0], 0.0, action_neck[1]], dtype=np.float32)
            #             action_mimic = np.concatenate([action_mimic, neck_joints])
            #     
            #     # 构建完整观测
            #     obs_full = np.concatenate([action_mimic, obs_proprio])
            #     
            #     # 更新历史
            #     obs_hist = np.array(self.proprio_history_buf).flatten()
            #     self.proprio_history_buf.append(obs_full)
            #     
            #     future_obs = action_mimic.copy()
            #     
            #     # 组合所有观测
            #     obs_buf = np.concatenate([obs_full, obs_hist, future_obs])
            #     
            #     assert obs_buf.shape[0] == self.total_obs_size, \
            #         f"Expected {self.total_obs_size} obs, got {obs_buf.shape[0]}"
            #     
            #     # 运行policy
            #     obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)
            #     with torch.no_grad():
            #         raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()
            #     
            #     self.last_action = raw_action.copy()
            #     
            #     # 计算目标位置
            #     raw_action = np.clip(raw_action, -10.0, 10.0)
            #     target_dof_pos = self.default_dof_pos + raw_action * self.action_scale
            #     
            #     # 发送控制命令
            #     self.send_mit_command(target_dof_pos, self.current_kp_scale, self.current_kd_scale)
            #     
            #     # 计算下一次循环的目标时间
            #     next_loop_time += self.target_dt
            #     
            #     # 记录实际循环时间
            #     actual_loop_time = time.time() - actual_start
            #     self.loop_times.append(actual_loop_time)
            #     
            #     step_count += 1
            #     
            #     # 检测是否超时
            #     if actual_loop_time > self.target_dt:
            #         # 如果超时，重新同步时间
            #         next_loop_time = time.time() + self.target_dt
            
            print("\n缓启动测试完成，等待Ctrl+C退出...")
            while not self.shutdown_requested:
                time.sleep(0.1)
                    
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # ========== 注释掉缓关闭，只测试缓慢启动 ==========
            # # 缓关闭
            # self.running = False
            # self.ramp_down()
            
            # 断开连接
            self.disconnect_robot()
            print("控制器已退出")


def main():
    parser = argparse.ArgumentParser(description='Taks-T1 Sim2Real 部署')
    parser.add_argument('--policy', type=str, required=True,
                        help='ONNX policy文件路径')
    parser.add_argument('--device', type=str, default='cuda',
                        help='运行设备 (cuda/cpu)')
    parser.add_argument('--server_ip', type=str, default='192.168.1.208',
                        help='Taks-T1服务器IP')
    parser.add_argument('--cmd_port', type=int, default=5555,
                        help='命令端口')
    
    args = parser.parse_args()
    
    # 验证文件存在
    if not os.path.exists(args.policy):
        print(f"错误: Policy文件不存在: {args.policy}")
        return
    
    print("=" * 50)
    print("Taks-T1 Sim2Real 部署")
    print("=" * 50)
    print(f"  Policy: {args.policy}")
    print(f"  Device: {args.device}")
    print(f"  Server: {args.server_ip}:{args.cmd_port}")
    print(f"  缓启动时间: {RAMP_UP_TIME}s")
    print(f"  缓关闭时间: {RAMP_DOWN_TIME}s")
    print("=" * 50)
    
    print("\n安全警告:")
    print("  - 确保机器人处于安全环境")
    print("  - 随时准备按 Ctrl+C 安全退出")
    print("=" * 50 + "\n")
    
    controller = TaksT1RealController(
        policy_path=args.policy,
        device=args.device,
        server_ip=args.server_ip,
        cmd_port=args.cmd_port
    )
    
    controller.run()


if __name__ == "__main__":
    main()
