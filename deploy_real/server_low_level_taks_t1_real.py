#!/usr/bin/env python3
"""
Taks-T1 Sim2Real 部署脚本 (重构版)
- 位置控制模式 (MIT: kp/kd有值, q有值, dq=0, tau=0)
- 训练配置: sim.dt=0.002, decimation=10, 控制dt=0.02s=50Hz
- 缓启动, EMA平滑, 跌倒保护
"""
import sys
sys.path.insert(0, '/home/rex/桌面/TWIST2/taks_sdk')

import argparse
import json
import time
import signal
import numpy as np
import torch
import redis
import os
from collections import deque
from rich.console import Console
from rich.table import Table

import taks
from taks import sync_get_all
from data_utils.rot_utils import quatToEuler
from robot_control.fall_detector import FallDetector, FallProtectionController

try:
    import onnxruntime as ort
except ImportError:
    ort = None

# ============ 训练配置参数 ============
SIM_DT = 0.002
DECIMATION = 10
CONTROL_DT = SIM_DT * DECIMATION  # 0.02s = 50Hz
CONTROL_FREQ = 1.0 / CONTROL_DT

# 缓启动时间
RAMP_UP_TIME = 5.0
TRANSITION_TIME = 2.0

# ============ 全局 KP/KD 配置 ============
# 左腿(6), 右腿(6), 腰(3), 左臂(7), 右臂(7), 脖子(3) = 32 DOF
GLOBAL_KP = np.array([
    50, 150, 150, 50, 40, 40,  # 左腿
    50, 150, 150, 50, 40, 40,  # 右腿
    150, 150, 150,              # 腰部
    20, 20, 20, 20, 10, 10, 10, # 左臂
    20, 20, 20, 20, 10, 10, 10, # 右臂
    1, 1, 1,                    # 脖子
], dtype=np.float32)

GLOBAL_KD = np.array([
    50, 50, 50, 50, 2, 2,      # 左腿
    50, 50, 50, 50, 2, 2,      # 右腿
    4, 4, 4,                    # 腰部
    5, 5, 5, 5, 1, 1, 1,       # 左臂
    5, 5, 5, 5, 1, 1, 1,       # 右臂
    0.1, 0.1, 0.1,             # 脖子
], dtype=np.float32)

# GLOBAL_KP = np.array([
#     100, 100, 100, 150, 40, 40,  # 左腿
#     100, 100, 100, 150, 40, 40,  # 右腿
#     150, 150, 150,               # 腰部
#     40, 40, 40, 40, 20, 20, 20,  # 左臂
#     40, 40, 40, 40, 20, 20, 20,  # 右臂
#     20, 20, 20,                  # 脖子
# ], dtype=np.float32)

# GLOBAL_KD = np.array([
#     2, 2, 2, 4, 2, 2,  # 左腿
#     2, 2, 2, 4, 2, 2,  # 右腿
#     4, 4, 4,           # 腰部
#     5, 5, 5, 5, 2, 2, 2,  # 左臂
#     5, 5, 5, 5, 2, 2, 2,  # 右臂
#     2, 2, 2,             # 脖子
# ], dtype=np.float32)

# 关节ID映射
POLICY_TO_SDK_JOINT_MAP = {
    0: 29, 1: 30, 2: 31, 3: 32, 4: 33, 5: 34,   # 左腿
    6: 23, 7: 24, 8: 25, 9: 26, 10: 27, 11: 28, # 右腿
    12: 17, 13: 18, 14: 19,                      # 腰部
    15: 9, 16: 10, 17: 11, 18: 12, 19: 13, 20: 14, 21: 15,  # 左臂
    22: 1, 23: 2, 24: 3, 25: 4, 26: 5, 27: 6, 28: 7,        # 右臂
    29: 20, 30: 21, 31: 22,                      # 脖子
}
SDK_TO_POLICY_JOINT_MAP = {v: k for k, v in POLICY_TO_SDK_JOINT_MAP.items()}

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

# 关节物理限位
JOINT_LIMITS_LOWER = np.array([
    -2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,
    -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618,
    -2.618, -0.52, -0.52,
    -3.0892, -1.5882, -2.618, -0.7, -2.67, -0.9, -0.9,
    -3.0892, -2.2515, -2.618, -0.7, -2.67, -0.9, -0.9,
    -1.57, -0.87, 0.0,
], dtype=np.float32)

JOINT_LIMITS_UPPER = np.array([
    2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,
    2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618,
    2.618, 0.52, 0.52,
    2.6704, 2.2515, 2.618, 1.57, 2.67, 0.9, 0.9,
    2.6704, 1.5882, 2.618, 1.57, 2.67, 0.9, 0.9,
    1.57, 0.87, 0.50,
], dtype=np.float32)


class EMASmoother:
    """EMA平滑器"""
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.value = None
        
    def smooth(self, new_value):
        if self.value is None:
            self.value = new_value.copy()
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value
    
    def reset(self):
        self.value = None


class ObservationFilter:
    """观测滤波器 - 对IMU和电机数据进行EMA滤波"""
    def __init__(self, alpha_imu=0.3, alpha_motor=0.5):
        # IMU滤波器
        self.quat_filter = EMASmoother(alpha=alpha_imu)
        self.ang_vel_filter = EMASmoother(alpha=alpha_imu)
        # 电机滤波器
        self.dof_pos_filter = EMASmoother(alpha=alpha_motor)
        self.dof_vel_filter = EMASmoother(alpha=alpha_motor)
    
    def filter_imu(self, quat, ang_vel):
        """滤波IMU数据"""
        quat_filtered = self.quat_filter.smooth(quat)
        # 四元数归一化
        quat_filtered = quat_filtered / max(np.linalg.norm(quat_filtered), 1e-6)
        ang_vel_filtered = self.ang_vel_filter.smooth(ang_vel)
        return quat_filtered, ang_vel_filtered
    
    def filter_motor(self, dof_pos, dof_vel):
        """滤波电机数据"""
        dof_pos_filtered = self.dof_pos_filter.smooth(dof_pos)
        dof_vel_filtered = self.dof_vel_filter.smooth(dof_vel)
        return dof_pos_filtered, dof_vel_filtered
    
    def reset(self):
        """重置所有滤波器"""
        self.quat_filter.reset()
        self.ang_vel_filter.reset()
        self.dof_pos_filter.reset()
        self.dof_vel_filter.reset()


class OnnxPolicyWrapper:
    """ONNX策略包装器"""
    def __init__(self, session, input_name):
        self.session = session
        self.input_name = input_name

    def __call__(self, obs_tensor):
        obs_np = obs_tensor.detach().cpu().numpy() if isinstance(obs_tensor, torch.Tensor) else np.asarray(obs_tensor, dtype=np.float32)
        return torch.from_numpy(self.session.run(None, {self.input_name: obs_np})[0].astype(np.float32))


def load_onnx_policy(policy_path, device):
    if ort is None:
        raise ImportError("onnxruntime is required")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device.startswith('cuda') else ['CPUExecutionProvider']
    session = ort.InferenceSession(policy_path, providers=providers)
    print(f"ONNX policy loaded: {policy_path}, providers: {session.get_providers()}")
    return OnnxPolicyWrapper(session, session.get_inputs()[0].name)


class TaksT1RealController:
    """Taks-T1 真机控制器"""
    
    def __init__(self, policy_path, device='cuda', server_ip='192.168.36.36', cmd_port=5555,
                 smooth_body=0.0, fall_protection=True, fall_roll_threshold=1.0, fall_pitch_threshold=1.0):
        self.device = device
        self.policy = load_onnx_policy(policy_path, device)
        self.server_ip = server_ip
        self.cmd_port = cmd_port
        
        self.kp = GLOBAL_KP.copy()
        self.kd = GLOBAL_KD.copy()
        self.control_dt = CONTROL_DT
        self.num_actions = 32
        self.default_dof_pos = np.zeros(self.num_actions, dtype=np.float32)
        
        # 缩放因子 (与训练配置一致)
        self.ang_vel_scale = 0.25
        self.dof_vel_scale = 0.05
        self.dof_pos_scale = 1.0
        self.action_scale = np.full(self.num_actions, 0.5, dtype=np.float32)  # 训练配置action_scale=0.5
        self.ankle_idx = [4, 5, 10, 11]
        
        # 观测结构
        self.n_mimic_obs = 38
        self.n_proprio = 101
        self.n_obs_single = self.n_mimic_obs + self.n_proprio
        self.history_len = 10
        self.total_obs_size = self.n_obs_single * (self.history_len + 1) + self.n_mimic_obs
        
        # 历史缓冲 - 使用numpy数组替代deque，提高效率
        self.proprio_history_buf = np.zeros((self.history_len, self.n_obs_single), dtype=np.float32)
        self._hist_idx = 0  # 环形缓冲区索引
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        
        # 默认mimic观测
        self.default_mimic_obs = np.concatenate([
            np.array([0, 0, 0.75, 0, 0, 0]),
            self.default_dof_pos
        ]).astype(np.float32)
        
        # IMU缓存
        self.last_valid_quat = np.array([1, 0, 0, 0], dtype=np.float32)
        self.last_valid_ang_vel = np.zeros(3, dtype=np.float32)
        
        # Redis
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.redis_pipeline = self.redis_client.pipeline()
        except Exception as e:
            print(f"Redis连接失败: {e}")
        
        # EMA平滑
        self.body_smoother = EMASmoother(alpha=smooth_body) if smooth_body > 0.0 else None
        if self.body_smoother:
            print(f"EMA平滑已启用: alpha={smooth_body}")
        
        # 跌倒保护
        self.fall_protection_enabled = fall_protection
        if fall_protection:
            self.fall_detector = FallDetector(roll_threshold=fall_roll_threshold, pitch_threshold=fall_pitch_threshold)
            self.fall_controller = FallProtectionController(self.fall_detector)
            print(f"跌倒保护已启用: roll={np.degrees(fall_roll_threshold):.1f}°, pitch={np.degrees(fall_pitch_threshold):.1f}°")
        else:
            self.fall_detector = self.fall_controller = None
        
        # 观测滤波器
        self.obs_filter = ObservationFilter(alpha_imu=0.3, alpha_motor=0.5)
        print(f"观测滤波已启用: alpha_imu=0.3, alpha_motor=0.5")
        
        # 控制状态
        self.robot = self.imu = None
        self.running = False
        self.shutdown_requested = False
        self.current_kp_scale = 0.0
        self.current_kd_scale = 0.0
        
        # 频率统计
        self.loop_timestamps = deque(maxlen=100)
        self.console = Console()
        self.print_counter = 0
        self.print_interval = int(CONTROL_FREQ)
        
        print(f"控制配置: dt={self.control_dt:.4f}s ({CONTROL_FREQ:.0f}Hz), sim_dt={SIM_DT}, decimation={DECIMATION}")
        print(f"观测维度: mimic={self.n_mimic_obs}, proprio={self.n_proprio}, total={self.total_obs_size}")
        
    def connect_robot(self):
        print(f"连接 Taks-T1: {self.server_ip}:{self.cmd_port}")
        taks.connect(self.server_ip, cmd_port=self.cmd_port)
        self.robot = taks.register("Taks-T1")
        print("已注册全身设备")
        time.sleep(4.0)
        print("等待4秒后注册IMU...")
        self.imu = taks.register("Taks-T1-imu")
        time.sleep(1)
        print("设备注册完成")
        
    def disconnect_robot(self):
        taks.disconnect()
        print("已断开连接")
    
    def get_robot_state(self):
        """同步读取电机和IMU状态，保证时间步统一"""
        motor_state, imu_data, sync_ts = sync_get_all(self.robot, self.imu)
        
        dof_pos = np.zeros(self.num_actions, dtype=np.float32)
        dof_vel = np.zeros(self.num_actions, dtype=np.float32)
        
        if motor_state:
            for sdk_jid, state in motor_state.items():
                if sdk_jid in SDK_TO_POLICY_JOINT_MAP:
                    idx = SDK_TO_POLICY_JOINT_MAP[sdk_jid]
                    dof_pos[idx] = state.get('pos', 0.0)
                    dof_vel[idx] = state.get('vel', 0.0)
        
        quat = self.last_valid_quat.copy()
        ang_vel = self.last_valid_ang_vel.copy()
        
        if imu_data:
            quat_data = imu_data.get('quat')
            if quat_data and 'w' in quat_data:
                q = np.array([quat_data['w'], quat_data['x'], quat_data['y'], quat_data['z']], dtype=np.float32)
                if np.abs(q).sum() > 0.1:
                    self.last_valid_quat = quat = q
            ang_vel_data = imu_data.get('ang_vel')
            if ang_vel_data and 'x' in ang_vel_data:
                self.last_valid_ang_vel = ang_vel = np.array([ang_vel_data['x'], ang_vel_data['y'], ang_vel_data['z']], dtype=np.float32)
        
        # 应用滤波
        quat, ang_vel = self.obs_filter.filter_imu(quat, ang_vel)
        dof_pos, dof_vel = self.obs_filter.filter_motor(dof_pos, dof_vel)
        
        return dof_pos, dof_vel, quat, ang_vel
    
    def send_mit_command(self, target_pos, kp_scale, kd_scale):
        """发送MIT控制命令 (位置控制: q有值, dq=0, tau=0)"""
        mit_data = {}
        for idx in range(self.num_actions):
            sdk_jid = POLICY_TO_SDK_JOINT_MAP[idx]
            mit_data[sdk_jid] = {
                'kp': float(self.kp[idx] * kp_scale),
                'kd': float(self.kd[idx] * kd_scale),
                'q': float(target_pos[idx]),
                'dq': 0.0,
                'tau': 0.0
            }
        self.robot.controlMIT(joints=mit_data)
        return mit_data
    
    def _print_table(self, mit_data, phase=""):
        def calc_freq(ts):
            if len(ts) < 2: return 0.0
            return 1.0 / np.mean(np.diff(list(ts)))
        
        loop_freq = calc_freq(self.loop_timestamps)
        table = Table(title=f"{phase} | Loop: {loop_freq:.1f}/{CONTROL_FREQ:.0f}Hz")
        table.add_column("ID", style="cyan", justify="center")
        table.add_column("SDK", style="magenta", justify="center")
        table.add_column("Name", style="green")
        table.add_column("Pos", style="yellow", justify="right")
        table.add_column("KP", style="blue", justify="right")
        table.add_column("KD", style="blue", justify="right")
        
        for idx in range(self.num_actions):
            sdk_jid = POLICY_TO_SDK_JOINT_MAP[idx]
            d = mit_data[sdk_jid]
            table.add_row(str(idx), str(sdk_jid), JOINT_NAMES.get(idx, "?"),
                         f"{d['q']:.4f}", f"{d['kp']:.1f}", f"{d['kd']:.1f}")
        
        self.console.clear()
        self.console.print(table)
    
    def _ease_out(self, t: float) -> float:
        """Ease-out曲线: 由快到慢, t in [0,1] -> [0,1]"""
        return 1.0 - (1.0 - t) ** 2
    
    def _ease_in(self, t: float) -> float:
        """Ease-in曲线: 由慢到快, t in [0,1] -> [0,1]"""
        return t ** 2
    
    def ramp_up(self):
        """非线性缓启动: kp/kd从0到目标值，使用ease-out曲线（由快到慢）"""
        print(f"缓启动 ({RAMP_UP_TIME}s)...")
        start = time.time()
        target_pos = np.zeros(self.num_actions, dtype=np.float32)
        
        while True:
            elapsed = time.time() - start
            if elapsed >= RAMP_UP_TIME:
                self.current_kp_scale = self.current_kd_scale = 1.0
                break
            
            # 非线性ease-out曲线：由快到慢
            t = elapsed / RAMP_UP_TIME
            scale = self._ease_out(t)
            self.current_kp_scale = self.current_kd_scale = scale
            self.get_robot_state()
            mit_data = self.send_mit_command(target_pos, self.current_kp_scale, self.current_kd_scale)
            
            self.print_counter += 1
            if self.print_counter >= self.print_interval:
                self.print_counter = 0
                self._print_table(mit_data, f"Ramp Up {elapsed:.1f}s/{RAMP_UP_TIME}s (scale={scale:.2f})")
            
            time.sleep(self.control_dt)
        print("✓ 缓启动完成")
    
    def signal_handler(self, signum, frame):
        """信号处理: 确保先断开连接再退出"""
        if self.shutdown_requested:
            print("\n强制退出...")
            self.disconnect_robot()
            sys.exit(1)
        print("\n收到退出信号，开始安全关闭...")
        self.shutdown_requested = True
    
    def run(self):
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            self.connect_robot()
            self.ramp_up()
            self.running = True
            print("开始主控制循环, Ctrl+C退出")
            
            transition_start = time.time()
            next_loop_time = time.time()
            
            while self.running and not self.shutdown_requested:
                sleep_time = next_loop_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                self.loop_timestamps.append(time.time())
                
                dof_pos, dof_vel, quat, ang_vel = self.get_robot_state()
                rpy = quatToEuler(quat)
                
                obs_dof_vel = dof_vel.copy()
                obs_dof_vel[self.ankle_idx] = 0.0
                
                obs_proprio = np.concatenate([
                    ang_vel * self.ang_vel_scale,
                    rpy[:2],
                    (dof_pos - self.default_dof_pos) * self.dof_pos_scale,
                    obs_dof_vel * self.dof_vel_scale,
                    self.last_action
                ])
                
                # Redis通信
                action_mimic = self.default_mimic_obs.copy()
                if self.redis_client:
                    state_body = np.concatenate([ang_vel, rpy[:2], dof_pos])
                    self.redis_pipeline.set("state_body_taks_t1", json.dumps(state_body.tolist()))
                    self.redis_pipeline.get("action_body_taks_t1")
                    self.redis_pipeline.get("action_neck_taks_t1")
                    results = self.redis_pipeline.execute()
                    
                    if results[1]:
                        action_mimic = np.array(json.loads(results[1]), dtype=np.float32)
                        if self.body_smoother:
                            action_mimic = self.body_smoother.smooth(action_mimic)
                        if len(action_mimic) == 35 and results[2]:
                            neck = np.array(json.loads(results[2]), dtype=np.float32)
                            action_mimic = np.concatenate([action_mimic, [neck[0], 0.0, neck[1]]])
                
                # 构建观测 - 使用环形缓冲区
                obs_full = np.concatenate([action_mimic, obs_proprio])
                # 获取历史观测（从新到旧）
                hist_indices = [(self._hist_idx - i) % self.history_len for i in range(self.history_len)]
                obs_hist = self.proprio_history_buf[hist_indices].flatten()
                # 更新环形缓冲区
                self._hist_idx = (self._hist_idx + 1) % self.history_len
                self.proprio_history_buf[self._hist_idx] = obs_full
                obs_buf = np.concatenate([obs_full, obs_hist, action_mimic])
                
                # 运行policy
                obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()
                
                self.last_action = raw_action.copy()
                
                # 计算目标位置
                raw_action = np.clip(raw_action, -5.0, 5.0)
                policy_target = self.default_dof_pos + raw_action * self.action_scale
                policy_target = np.clip(policy_target, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)
                
                # 平滑过渡
                trans_elapsed = time.time() - transition_start
                if trans_elapsed < TRANSITION_TIME:
                    t = trans_elapsed / TRANSITION_TIME
                    blend = t * t * (3.0 - 2.0 * t)
                    target_dof_pos = blend * policy_target
                else:
                    target_dof_pos = policy_target
                
                # 跌倒保护
                kp_scale, kd_scale = self.current_kp_scale, self.current_kd_scale
                if self.fall_protection_enabled and self.fall_controller:
                    kp_scale, kd_scale, is_fallen = self.fall_controller.check_and_protect_from_rpy(rpy[0], rpy[1])
                    if is_fallen:
                        print("\n[跌倒检测] 断开连接...")
                        break
                
                # 发送命令
                mit_data = self.send_mit_command(target_dof_pos, kp_scale, kd_scale)
                
                self.print_counter += 1
                if self.print_counter >= self.print_interval:
                    self.print_counter = 0
                    self._print_table(mit_data, "Running")
                
                next_loop_time += self.control_dt
                if time.time() > next_loop_time:
                    next_loop_time = time.time() + self.control_dt
                    
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
            self.disconnect_robot()
            print("控制器已退出")


def main():
    parser = argparse.ArgumentParser(description='Taks-T1 Sim2Real 部署')
    parser.add_argument('--policy', type=str, required=True, help='ONNX policy路径')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--server_ip', type=str, default='192.168.36.36')
    parser.add_argument('--cmd_port', type=int, default=5555)
    parser.add_argument('--smooth_body', type=float, default=0.1, help='EMA平滑系数 (0=关闭,1=完全开启)')
    parser.add_argument('--no_fall_protection', action='store_true')
    parser.add_argument('--fall_roll_threshold', type=float, default=1.0)
    parser.add_argument('--fall_pitch_threshold', type=float, default=1.0)
    args = parser.parse_args()
    
    if not os.path.exists(args.policy):
        print(f"错误: Policy不存在: {args.policy}")
        return
    
    print("=" * 50)
    print("Taks-T1 Sim2Real 部署 (重构版)")
    print("=" * 50)
    print(f"  训练配置: sim_dt={SIM_DT}, decimation={DECIMATION}, 控制dt={CONTROL_DT}s ({CONTROL_FREQ:.0f}Hz)")
    print(f"  控制模式: 位置控制 (MIT: kp/kd有值, q有值, dq=0, tau=0)")
    print(f"  Policy: {args.policy}")
    print(f"  Server: {args.server_ip}:{args.cmd_port}")
    print(f"  EMA平滑: {args.smooth_body if args.smooth_body > 0 else '关闭'}")
    print(f"  跌倒保护: {'关闭' if args.no_fall_protection else '开启'}")
    print("=" * 50)
    
    controller = TaksT1RealController(
        policy_path=args.policy,
        device=args.device,
        server_ip=args.server_ip,
        cmd_port=args.cmd_port,
        smooth_body=args.smooth_body,
        fall_protection=not args.no_fall_protection,
        fall_roll_threshold=args.fall_roll_threshold,
        fall_pitch_threshold=args.fall_pitch_threshold
    )
    controller.run()


if __name__ == "__main__":
    main()