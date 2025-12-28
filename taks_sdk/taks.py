#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taks SDK 客户端
"""

import zmq
import json
import threading
import time
from typing import Dict, Optional, Callable

# ============ 全局状态 ============
_ctx: Optional[zmq.Context] = None
_dealer: Optional[zmq.Socket] = None
_address: Optional[str] = None
_lock = threading.Lock()

# 缓存
_motor_cache: Dict[str, Dict] = {}
_imu_cache: Dict[str, Dict] = {}

# 配置
RECV_TIMEOUT_MS = 50  # 接收超时(ms)，放宽以减少误报
SEND_TIMEOUT_MS = 20  # 发送超时(ms)
HWM = 200  # 高水位标记


# ============ 连接管理 ============
def connect(address: str, cmd_port: int = 5555):
    """连接服务器"""
    global _ctx, _dealer, _address
    disconnect()
    
    _ctx = zmq.Context()
    _address = address
    _dealer = _ctx.socket(zmq.DEALER)
    _dealer.setsockopt(zmq.RCVTIMEO, RECV_TIMEOUT_MS)
    _dealer.setsockopt(zmq.SNDTIMEO, SEND_TIMEOUT_MS)
    _dealer.setsockopt(zmq.SNDHWM, HWM)
    _dealer.setsockopt(zmq.RCVHWM, HWM)
    _dealer.setsockopt(zmq.LINGER, 0)
    _dealer.connect(f"tcp://{address}:{cmd_port}")
    print(f"✓ 已连接到 {address}:{cmd_port}")


def disconnect():
    """断开连接"""
    global _ctx, _dealer, _address
    
    # 在断开前先失能所有电机
    if _dealer:
        try:
            # 发送 disable_all 命令给所有已注册的设备
            for device_type in ["Taks-T1", "Taks-T1-semibody", "Taks-T1-leftarm", "Taks-T1-rightarm"]:
                try:
                    result = _send({'device': device_type, 'cmd': 'disable_all'})
                    if result and result.get('ok'):
                        print(f"✓ {device_type} 电机已失能")
                except:
                    pass
            time.sleep(0.2)  # 等待失能完成
        except:
            pass
    
    if _dealer:
        _dealer.close()
        _dealer = None
    if _ctx:
        _ctx.term()
        _ctx = None
    _address = None


def _send(msg: dict, wait_response: bool = True) -> Optional[dict]:
    """发送消息，可选择等待响应"""
    if not _dealer:
        raise RuntimeError("未连接，请先调用 connect()")
    
    data = json.dumps(msg).encode('utf-8')
    with _lock:
        try:
            _dealer.send_multipart([b'', data], zmq.NOBLOCK if not wait_response else 0)
        except zmq.Again:
            return {'ok': False, 'error': 'send_timeout'}
        
        if not wait_response:
            return None
        
        try:
            frames = _dealer.recv_multipart()
            return json.loads(frames[-1].decode('utf-8')) if frames else {'ok': False, 'error': 'empty'}
        except zmq.Again:
            return {'ok': False, 'error': 'recv_timeout'}
        except Exception as e:
            return {'ok': False, 'error': str(e)}


def _send_batch(msgs: list) -> list:
    """批量发送并接收（按device匹配响应）"""
    if not _dealer:
        raise RuntimeError("未连接")
    
    # 构建期望的响应映射
    expected = {}
    for i, msg in enumerate(msgs):
        device = msg.get('device', '')
        if 'imu' in device.lower():
            expected['imu'] = i
        else:
            expected['motor'] = i
    
    results = [{'ok': False, 'error': 'no_response'} for _ in msgs]
    received = 0
    
    with _lock:
        # 批量发送
        for msg in msgs:
            _dealer.send_multipart([b'', json.dumps(msg).encode('utf-8')], zmq.NOBLOCK)
        
        # 批量接收（根据响应内容匹配）
        while received < len(msgs):
            try:
                frames = _dealer.recv_multipart()
                if not frames:
                    continue
                resp = json.loads(frames[-1].decode('utf-8'))
                
                # 根据响应内容判断类型
                if resp.get('device') == 'imu' or 'data' in resp:
                    # IMU响应
                    if 'imu' in expected:
                        results[expected['imu']] = resp
                        received += 1
                elif 'joints' in resp:
                    # 电机query响应
                    if 'motor' in expected:
                        results[expected['motor']] = resp
                        received += 1
                else:
                    # 其他响应，按顺序填充
                    for i, r in enumerate(results):
                        if r.get('error') == 'no_response':
                            results[i] = resp
                            received += 1
                            break
            except zmq.Again:
                # 超时，跳出
                break
            except:
                break
    
    return results


# ============ 设备类 ============
class TaksDevice:
    """Taks电机设备"""
    
    # 关节映射
    JOINT_MAP = {
        "Taks-T1": [1,2,3,4,5,6,7, 9,10,11,12,13,14,15, 17,18,19, 20,21,22, 23,24,25,26,27,28, 29,30,31,32,33,34],
        "Taks-T1-leftarm": list(range(9, 16)),
        "Taks-T1-rightarm": list(range(1, 8)),
        "Taks-T1-semibody": [1,2,3,4,5,6,7, 9,10,11,12,13,14,15, 17,18,19, 20,21,22],
    }
    
    def __init__(self, device_type: str):
        self.device_type = device_type
        self.joints = self.JOINT_MAP.get(device_type, [])
    
    def _register(self):
        return _send({'device': self.device_type, 'cmd': 'register'})
    
    def GetState(self) -> Optional[Dict[int, Dict]]:
        """获取所有关节状态 {jid: {'pos', 'vel', 'tau'}}"""
        result = _send({'device': self.device_type, 'cmd': 'query', 'jids': []})
        if result and result.get('ok') and 'joints' in result:
            state = {int(k): v for k, v in result['joints'].items()}
            _motor_cache[self.device_type] = state
            return state
        return _motor_cache.get(self.device_type)
    
    def GetPosition(self) -> Optional[Dict[int, float]]:
        """获取位置"""
        state = self.GetState()
        return {jid: s['pos'] for jid, s in state.items()} if state else None
    
    def SetPosition(self, **kwargs):
        """设置位置: SetPosition(j1=0.1, j2=0.2)"""
        joints = {int(k[1:]): v for k, v in kwargs.items() if k.startswith('j') and v is not None}
        if joints:
            _send({'device': self.device_type, 'cmd': 'pos', 'joints': joints}, wait_response=False)
    
    def controlMIT(self, joints: dict):
        """MIT控制: {jid: {'q', 'dq', 'tau', 'kp', 'kd'}}"""
        _send({'device': self.device_type, 'cmd': 'mit', 'joints': joints}, wait_response=False)


class IMUDevice:
    """IMU设备"""
    
    def __init__(self):
        self.device_type = "Taks-T1-imu"
    
    def _register(self):
        return _send({'device': self.device_type, 'cmd': 'register'})
    
    def get_all_data(self) -> Optional[Dict]:
        """获取全部数据 {'ang_vel', 'lin_acc', 'quat', 'rpy'}"""
        result = _send({'device': self.device_type, 'cmd': 'get_all'})
        if result and result.get('ok') and 'data' in result:
            _imu_cache.update(result['data'])
            return result['data']
        return _imu_cache if _imu_cache else None
    
    def get_rpy(self) -> Optional[Dict]:
        result = _send({'device': self.device_type, 'cmd': 'rpy'})
        if result and result.get('ok') and 'data' in result:
            return result['data']
        return _imu_cache.get('rpy')
    
    def calibrate_zero(self) -> bool:
        result = _send({'device': self.device_type, 'cmd': 'cal_zero'})
        return result.get('ok', False) if result else False


# ============ 全局函数 ============
def register(device_type: str):
    """注册设备"""
    if device_type == "Taks-T1-imu":
        dev = IMUDevice()
    else:
        dev = TaksDevice(device_type)
    dev._register()
    return dev


def batch_query(robot: TaksDevice, imu: IMUDevice) -> tuple:
    """并行查询电机+IMU"""
    msgs = [
        {'device': robot.device_type, 'cmd': 'query', 'jids': []},
        {'device': imu.device_type, 'cmd': 'get_all'}
    ]
    results = _send_batch(msgs)
    
    # 电机状态
    motor_state = None
    if results[0].get('ok') and 'joints' in results[0]:
        motor_state = {int(k): v for k, v in results[0]['joints'].items()}
        _motor_cache[robot.device_type] = motor_state
    else:
        motor_state = _motor_cache.get(robot.device_type)
    
    # IMU数据
    imu_data = None
    if results[1].get('ok') and 'data' in results[1]:
        imu_data = results[1]['data']
        _imu_cache.update(imu_data)
    else:
        imu_data = _imu_cache if _imu_cache else None
    
    return motor_state, imu_data