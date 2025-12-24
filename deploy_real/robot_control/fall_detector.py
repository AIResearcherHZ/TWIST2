#!/usr/bin/env python3
"""
跌倒检测模块 - 基于IMU姿态判断机器人是否跌倒
"""
import numpy as np
import threading
import time
from typing import Callable, Optional
from data_utils.rot_utils import quatToEuler


class FallDetector:
    """
    基于IMU姿态的跌倒检测器
    
    检测原理：
    - 当roll或pitch角度超过阈值时，判定为跌倒
    - 使用滑动窗口平滑检测，避免误触发
    """
    
    def __init__(self, 
                 roll_threshold: float = 1.0,      # roll阈值 (rad), 约57度
                 pitch_threshold: float = 1.0,     # pitch阈值 (rad), 约57度
                 detection_window: int = 5,        # 检测窗口大小
                 detection_ratio: float = 0.6):    # 触发比例
        """
        Args:
            roll_threshold: roll角度阈值(弧度)，超过此值判定为跌倒
            pitch_threshold: pitch角度阈值(弧度)，超过此值判定为跌倒
            detection_window: 滑动窗口大小
            detection_ratio: 窗口内超过阈值的比例达到此值时触发跌倒
        """
        self.roll_threshold = roll_threshold
        self.pitch_threshold = pitch_threshold
        self.detection_window = detection_window
        self.detection_ratio = detection_ratio
        
        self._fall_history = []
        self._is_fallen = False
        self._lock = threading.Lock()
        
    def update(self, quat: np.ndarray) -> bool:
        """
        更新IMU数据并检测是否跌倒
        
        Args:
            quat: 四元数 [w, x, y, z]
            
        Returns:
            bool: 是否检测到跌倒
        """
        rpy = quatToEuler(quat)
        roll, pitch = rpy[0], rpy[1]
        
        # 判断当前帧是否超过阈值
        current_fall = (abs(roll) > self.roll_threshold or 
                       abs(pitch) > self.pitch_threshold)
        
        with self._lock:
            self._fall_history.append(current_fall)
            if len(self._fall_history) > self.detection_window:
                self._fall_history.pop(0)
            
            # 计算窗口内跌倒帧的比例
            if len(self._fall_history) >= self.detection_window:
                fall_ratio = sum(self._fall_history) / len(self._fall_history)
                if fall_ratio >= self.detection_ratio:
                    self._is_fallen = True
                    
        return self._is_fallen
    
    def update_from_rpy(self, roll: float, pitch: float) -> bool:
        """
        直接使用roll/pitch更新检测状态
        
        Args:
            roll: roll角度(弧度)
            pitch: pitch角度(弧度)
            
        Returns:
            bool: 是否检测到跌倒
        """
        current_fall = (abs(roll) > self.roll_threshold or 
                       abs(pitch) > self.pitch_threshold)
        
        with self._lock:
            self._fall_history.append(current_fall)
            if len(self._fall_history) > self.detection_window:
                self._fall_history.pop(0)
            
            if len(self._fall_history) >= self.detection_window:
                fall_ratio = sum(self._fall_history) / len(self._fall_history)
                if fall_ratio >= self.detection_ratio:
                    self._is_fallen = True
                    
        return self._is_fallen
    
    @property
    def is_fallen(self) -> bool:
        """返回当前跌倒状态"""
        with self._lock:
            return self._is_fallen
    
    def reset(self):
        """重置跌倒检测状态"""
        with self._lock:
            self._fall_history = []
            self._is_fallen = False


class FallProtectionController:
    """
    跌倒保护控制器
    
    功能：
    - 并行监控IMU姿态
    - 检测到跌倒时触发保护回调
    - 支持渐进式降低kp/kd
    """
    
    def __init__(self,
                 fall_detector: FallDetector,
                 on_fall_callback: Optional[Callable] = None,
                 kp_decay_rate: float = 0.1,      # kp衰减速率
                 kd_decay_rate: float = 0.1):     # kd衰减速率
        """
        Args:
            fall_detector: 跌倒检测器实例
            on_fall_callback: 跌倒时的回调函数
            kp_decay_rate: kp衰减速率 (每次调用decay时的衰减比例)
            kd_decay_rate: kd衰减速率
        """
        self.fall_detector = fall_detector
        self.on_fall_callback = on_fall_callback
        self.kp_decay_rate = kp_decay_rate
        self.kd_decay_rate = kd_decay_rate
        
        self._kp_scale = 1.0
        self._kd_scale = 1.0
        self._protection_active = False
        self._lock = threading.Lock()
        
    def check_and_protect(self, quat: np.ndarray) -> tuple:
        """
        检查跌倒状态并返回当前kp/kd缩放值
        
        Args:
            quat: 四元数 [w, x, y, z]
            
        Returns:
            tuple: (kp_scale, kd_scale, is_fallen)
        """
        is_fallen = self.fall_detector.update(quat)
        
        with self._lock:
            if is_fallen and not self._protection_active:
                self._protection_active = True
                print("\n" + "="*50)
                print("[FALL DETECTED] 检测到跌倒! 启动保护模式...")
                print("="*50 + "\n")
                if self.on_fall_callback:
                    self.on_fall_callback()
            
            if self._protection_active:
                # 快速将kp/kd降为0
                self._kp_scale = 0.0
                self._kd_scale = 0.0
                
            return self._kp_scale, self._kd_scale, is_fallen
    
    def check_and_protect_from_rpy(self, roll: float, pitch: float) -> tuple:
        """
        使用roll/pitch检查跌倒状态
        
        Args:
            roll: roll角度(弧度)
            pitch: pitch角度(弧度)
            
        Returns:
            tuple: (kp_scale, kd_scale, is_fallen)
        """
        is_fallen = self.fall_detector.update_from_rpy(roll, pitch)
        
        with self._lock:
            if is_fallen and not self._protection_active:
                self._protection_active = True
                print("\n" + "="*50)
                print("[FALL DETECTED] 检测到跌倒! 启动保护模式...")
                print(f"  Roll: {np.degrees(roll):.1f}°, Pitch: {np.degrees(pitch):.1f}°")
                print("  kp/kd 已降为 0")
                print("="*50 + "\n")
                if self.on_fall_callback:
                    self.on_fall_callback()
            
            if self._protection_active:
                self._kp_scale = 0.0
                self._kd_scale = 0.0
                
            return self._kp_scale, self._kd_scale, is_fallen
    
    @property
    def kp_scale(self) -> float:
        with self._lock:
            return self._kp_scale
    
    @property
    def kd_scale(self) -> float:
        with self._lock:
            return self._kd_scale
    
    @property
    def is_protection_active(self) -> bool:
        with self._lock:
            return self._protection_active
    
    def reset(self):
        """重置保护状态"""
        with self._lock:
            self._kp_scale = 1.0
            self._kd_scale = 1.0
            self._protection_active = False
        self.fall_detector.reset()


if __name__ == "__main__":
    # 测试代码
    detector = FallDetector(roll_threshold=0.5, pitch_threshold=0.5)
    controller = FallProtectionController(detector)
    
    # 模拟正常姿态
    normal_quat = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(10):
        kp, kd, fallen = controller.check_and_protect(normal_quat)
        print(f"Normal: kp={kp:.2f}, kd={kd:.2f}, fallen={fallen}")
    
    # 模拟跌倒姿态 (大约45度倾斜)
    fall_quat = np.array([0.924, 0.383, 0.0, 0.0])  # 约45度roll
    for _ in range(10):
        kp, kd, fallen = controller.check_and_protect(fall_quat)
        print(f"Falling: kp={kp:.2f}, kd={kd:.2f}, fallen={fallen}")
