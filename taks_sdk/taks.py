#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taks SDK - ZeroMQ 客户端库
特点：
1. DEALER socket：双向通信
2. SUB socket：订阅状态广播
3. JSON 协议：简单可靠
"""

import zmq
import json
import threading
import time
from typing import List, Optional, Dict, Callable

# ============ 全局状态 ============
_ctx: Optional[zmq.Context] = None
_dealer: Optional[zmq.Socket] = None
_sub: Optional[zmq.Socket] = None
_address: Optional[str] = None
_state_callbacks: Dict[str, Callable] = {}
_recv_thread: Optional[threading.Thread] = None
_running: bool = False
_lock = threading.Lock()

# IMU 缓存
_imu_cache: Dict[str, Dict] = {}
_imu_cache_lock = threading.Lock()

# 电机状态缓存
_motor_state_cache: Dict[str, Dict] = {}  # device_type -> {jid: state}
_motor_cache_lock = threading.Lock()


# ============ 连接管理 ============
def connect(address: str, cmd_port: int = 5555, sub_port: int = 5556):
    """连接到服务器"""
    global _ctx, _dealer, _sub, _address, _recv_thread, _running
    
    disconnect()
    
    _ctx = zmq.Context()
    _address = address
    
    _dealer = _ctx.socket(zmq.DEALER)
    _dealer.setsockopt(zmq.RCVTIMEO, 200)
    _dealer.setsockopt(zmq.SNDTIMEO, 50)
    _dealer.setsockopt(zmq.LINGER, 0)
    _dealer.connect(f"tcp://{address}:{cmd_port}")
    
    _sub = _ctx.socket(zmq.SUB)
    _sub.setsockopt(zmq.RCVTIMEO, 10)
    _sub.setsockopt(zmq.LINGER, 0)
    _sub.connect(f"tcp://{address}:{sub_port}")
    _sub.setsockopt(zmq.SUBSCRIBE, b'')
    
    _running = True
    _recv_thread = threading.Thread(target=_receive_loop, daemon=True)
    _recv_thread.start()
    
    print(f"✓ 已连接到 {address}:{cmd_port}")

def disconnect():
    """断开连接"""
    global _ctx, _dealer, _sub, _address, _recv_thread, _running
    
    _running = False
    if _recv_thread and _recv_thread.is_alive():
        _recv_thread.join(timeout=1.0)
    _recv_thread = None
    
    if _dealer:
        _dealer.close()
        _dealer = None
    if _sub:
        _sub.close()
        _sub = None
    if _ctx:
        _ctx.term()
        _ctx = None
    _address = None

def _receive_loop():
    """接收状态广播"""
    while _running and _sub:
        try:
            data = _sub.recv()
            try:
                result = json.loads(data.decode('utf-8'))
                device = result.get('device', '')
                if device in _state_callbacks:
                    _state_callbacks[device](result)
            except:
                pass
        except zmq.Again:
            continue
        except:
            if _running:
                pass

def _send_and_wait(msg: dict, timeout: float = 1.0) -> dict:
    """发送命令并等待响应"""
    if not _dealer:
        raise RuntimeError("未连接，请先调用 connect()")
    
    with _lock:
        data = json.dumps(msg).encode('utf-8')
        _dealer.send_multipart([b'', data])
        
        try:
            frames = _dealer.recv_multipart()
            if frames:
                return json.loads(frames[-1].decode('utf-8'))
            return {'ok': False, 'error': 'empty response'}
        except zmq.Again:
            return {'ok': False, 'error': 'timeout'}
        except Exception as e:
            return {'ok': False, 'error': str(e)}

def _send_batch(msgs: list) -> list:
    """批量发送命令并等待所有响应（并行）"""
    if not _dealer:
        raise RuntimeError("未连接，请先调用 connect()")
    
    results = []
    with _lock:
        # 批量发送所有消息
        for msg in msgs:
            data = json.dumps(msg).encode('utf-8')
            _dealer.send_multipart([b'', data])
        
        # 批量接收所有响应
        for _ in msgs:
            try:
                frames = _dealer.recv_multipart()
                if frames:
                    results.append(json.loads(frames[-1].decode('utf-8')))
                else:
                    results.append({'ok': False, 'error': 'empty response'})
            except zmq.Again:
                results.append({'ok': False, 'error': 'timeout'})
            except Exception as e:
                results.append({'ok': False, 'error': str(e)})
    
    return results

def _send_nowait(msg: dict):
    """发送命令，不等待响应"""
    if not _dealer:
        raise RuntimeError("未连接，请先调用 connect()")
    
    data = json.dumps(msg).encode('utf-8')
    with _lock:
        _dealer.send_multipart([b'', data], zmq.NOBLOCK)

# ============ 关节控制类 ============
class JointControl:
    """单关节控制"""
    __slots__ = ('_dev', '_jid')
    
    def __init__(self, device: 'TaksDevice', jid: int):
        self._dev = device
        self._jid = jid
    
    def SetPosition(self, position: float):
        """设置位置"""
        _send_nowait({'device': self._dev.device_type, 'cmd': 'pos', 'joints': {self._jid: position}})
    
    def GetPosition(self) -> Optional[float]:
        """获取位置"""
        result = _send_and_wait({'device': self._dev.device_type, 'cmd': 'query', 'jids': [self._jid]})
        if result.get('ok') and 'joints' in result:
            jdata = result['joints'].get(str(self._jid))
            return jdata.get('pos') if jdata else None
        return None
    
    def GetVelocity(self) -> Optional[float]:
        """获取速度"""
        result = _send_and_wait({'device': self._dev.device_type, 'cmd': 'query', 'jids': [self._jid]})
        if result.get('ok') and 'joints' in result:
            jdata = result['joints'].get(str(self._jid))
            return jdata.get('vel') if jdata else None
        return None
    
    def GetTorque(self) -> Optional[float]:
        """获取力矩"""
        result = _send_and_wait({'device': self._dev.device_type, 'cmd': 'query', 'jids': [self._jid]})
        if result.get('ok') and 'joints' in result:
            jdata = result['joints'].get(str(self._jid))
            return jdata.get('tau') if jdata else None
        return None
    
    def GetState(self) -> Optional[Dict]:
        """获取完整状态"""
        result = _send_and_wait({'device': self._dev.device_type, 'cmd': 'query', 'jids': [self._jid]})
        if result.get('ok') and 'joints' in result:
            return result['joints'].get(str(self._jid))
        return None
    
    def controlMIT(self, kp: Optional[float] = None, kd: Optional[float] = None,
                   q: float = 0, dq: float = 0, tau: float = 0):
        """MIT控制"""
        _send_nowait({
            'device': self._dev.device_type,
            'cmd': 'mit',
            'joints': {self._jid: {'kp': kp, 'kd': kd, 'q': q, 'dq': dq, 'tau': tau}}
        })

# ============ 设备类 ============
class TaksDevice:
    """Taks设备主类"""
    def __init__(self, device_type: str):
        self.device_type = device_type
        
        if not _dealer:
            raise RuntimeError("请先调用 connect() 连接网络")
        
        # 关节映射
        joint_map = {
            "leftarm": list(range(9, 16)),
            "rightarm": list(range(1, 8)),
            "semibody": [1,2,3,4,5,6,7, 9,10,11,12,13,14,15, 17,18,19, 20,21,22],
        }
        
        if device_type == "Taks-T1":
            joints = [1,2,3,4,5,6,7, 9,10,11,12,13,14,15, 17,18,19, 20,21,22, 23,24,25,26,27,28, 29,30,31,32,33,34]
        else:
            joints = []
            for key, jlist in joint_map.items():
                if key in device_type:
                    joints = jlist
                    break
        
        for i in joints:
            setattr(self, f"j{i}", JointControl(self, i))
    
    def _register(self):
        """注册设备"""
        return _send_and_wait({'device': self.device_type, 'cmd': 'register'})
    
    def SetPosition(self, **kwargs):
        """批量位置控制: SetPosition(j1=0.1, j2=0.2, ...)"""
        joints = {}
        for key, val in kwargs.items():
            if val is not None and key.startswith('j'):
                jid = int(key[1:])
                joints[jid] = val
        
        if joints:
            _send_nowait({'device': self.device_type, 'cmd': 'pos', 'joints': joints})
    
    def GetPosition(self) -> Optional[Dict[int, float]]:
        """获取所有关节位置"""
        result = _send_and_wait({'device': self.device_type, 'cmd': 'query', 'jids': []})
        if result.get('ok') and 'joints' in result:
            pos_dict = {int(k): v['pos'] for k, v in result['joints'].items()}
            with _motor_cache_lock:
                if self.device_type not in _motor_state_cache:
                    _motor_state_cache[self.device_type] = {}
                for jid, pos in pos_dict.items():
                    if jid not in _motor_state_cache[self.device_type]:
                        _motor_state_cache[self.device_type][jid] = {}
                    _motor_state_cache[self.device_type][jid]['pos'] = pos
            return pos_dict
        # 返回缓存
        with _motor_cache_lock:
            if self.device_type in _motor_state_cache:
                return {jid: state.get('pos', 0.0) for jid, state in _motor_state_cache[self.device_type].items()}
        return None
    
    def GetVelocity(self) -> Optional[Dict[int, float]]:
        """获取所有关节速度"""
        result = _send_and_wait({'device': self.device_type, 'cmd': 'query', 'jids': []})
        if result.get('ok') and 'joints' in result:
            vel_dict = {int(k): v['vel'] for k, v in result['joints'].items()}
            with _motor_cache_lock:
                if self.device_type not in _motor_state_cache:
                    _motor_state_cache[self.device_type] = {}
                for jid, vel in vel_dict.items():
                    if jid not in _motor_state_cache[self.device_type]:
                        _motor_state_cache[self.device_type][jid] = {}
                    _motor_state_cache[self.device_type][jid]['vel'] = vel
            return vel_dict
        # 返回缓存
        with _motor_cache_lock:
            if self.device_type in _motor_state_cache:
                return {jid: state.get('vel', 0.0) for jid, state in _motor_state_cache[self.device_type].items()}
        return None
    
    def GetTorque(self) -> Optional[Dict[int, float]]:
        """获取所有关节力矩"""
        result = _send_and_wait({'device': self.device_type, 'cmd': 'query', 'jids': []})
        if result.get('ok') and 'joints' in result:
            tau_dict = {int(k): v['tau'] for k, v in result['joints'].items()}
            with _motor_cache_lock:
                if self.device_type not in _motor_state_cache:
                    _motor_state_cache[self.device_type] = {}
                for jid, tau in tau_dict.items():
                    if jid not in _motor_state_cache[self.device_type]:
                        _motor_state_cache[self.device_type][jid] = {}
                    _motor_state_cache[self.device_type][jid]['tau'] = tau
            return tau_dict
        # 返回缓存
        with _motor_cache_lock:
            if self.device_type in _motor_state_cache:
                return {jid: state.get('tau', 0.0) for jid, state in _motor_state_cache[self.device_type].items()}
        return None
    
    def GetState(self) -> Optional[Dict[int, Dict]]:
        """获取所有关节完整状态"""
        result = _send_and_wait({'device': self.device_type, 'cmd': 'query', 'jids': []})
        if result.get('ok') and 'joints' in result:
            state_dict = {int(k): v for k, v in result['joints'].items()}
            with _motor_cache_lock:
                _motor_state_cache[self.device_type] = state_dict
            return state_dict
        # 返回缓存
        with _motor_cache_lock:
            if self.device_type in _motor_state_cache:
                return _motor_state_cache[self.device_type].copy()
        return None
    
    def controlMIT(self, joints: dict, kp=None, kd=None, q=None, dq=None, tau=None):
        """批量MIT控制
        
        Args:
            joints: {jid: {'kp': ..., 'kd': ..., 'q': ..., 'dq': ..., 'tau': ...}}
            kp, kd, q, dq, tau: 默认值
        """
        mit_data = {}
        for jid, params in joints.items():
            mit_data[jid] = {
                'kp': params.get('kp', kp),
                'kd': params.get('kd', kd),
                'q': params.get('q', q if q is not None else 0),
                'dq': params.get('dq', dq if dq is not None else 0),
                'tau': params.get('tau', tau if tau is not None else 0)
            }
        
        if mit_data:
            _send_nowait({'device': self.device_type, 'cmd': 'mit', 'joints': mit_data})
    
    def subscribe_state(self, callback: Callable[[Dict], None]):
        """订阅状态回调"""
        _state_callbacks[self.device_type] = callback
    
    def unsubscribe_state(self):
        """取消订阅"""
        _state_callbacks.pop(self.device_type, None)
    
    def close(self):
        self.unsubscribe_state()
    
    def __repr__(self):
        return f"<{self.device_type} {'已连接' if _address else '未连接'}>"

# ============ IMU 设备类 ============
class IMUDevice:
    """​IMU设备类"""
    def __init__(self):
        self.device_type = "Taks-T1-imu"
        if not _dealer:
            raise RuntimeError("请先调用 connect() 连接网络")
    
    def _register(self):
        return _send_and_wait({'device': self.device_type, 'cmd': 'register'})
    
    def _get_with_cache(self, cmd: str, cache_key: str) -> Optional[Dict]:
        """获取数据，失败时返回缓存"""
        result = _send_and_wait({'device': self.device_type, 'cmd': cmd})
        if result.get('ok'):
            data = result.get('data')
            if data:
                with _imu_cache_lock:
                    _imu_cache[cache_key] = data
                return data
        # 返回缓存
        with _imu_cache_lock:
            return _imu_cache.get(cache_key)
    
    def get_ang_vel(self) -> Optional[Dict]:
        """获取角速度 -> {'x', 'y', 'z'}"""
        return self._get_with_cache('ang_vel', 'ang_vel')
    
    def get_lin_acc(self) -> Optional[Dict]:
        """获取线加速度 -> {'x', 'y', 'z'}"""
        return self._get_with_cache('lin_acc', 'lin_acc')
    
    def get_quat(self) -> Optional[Dict]:
        """获取四元数 -> {'w', 'x', 'y', 'z'}"""
        return self._get_with_cache('quat', 'quat')
    
    def get_rpy(self) -> Optional[Dict]:
        """获取欧拉角 -> {'roll', 'pitch', 'yaw'}"""
        return self._get_with_cache('rpy', 'rpy')
    
    def get_all_data(self) -> Optional[Dict]:
        """一次性获取所有IMU数据 -> {'ang_vel', 'lin_acc', 'quat', 'rpy'}"""
        result = _send_and_wait({'device': self.device_type, 'cmd': 'get_all'})
        if result.get('ok') and 'data' in result:
            data = result['data']
            # 更新缓存
            with _imu_cache_lock:
                if 'ang_vel' in data:
                    _imu_cache['ang_vel'] = data['ang_vel']
                if 'lin_acc' in data:
                    _imu_cache['lin_acc'] = data['lin_acc']
                if 'quat' in data:
                    _imu_cache['quat'] = data['quat']
                if 'rpy' in data:
                    _imu_cache['rpy'] = data['rpy']
            return data
        # 返回缓存
        with _imu_cache_lock:
            return {
                'ang_vel': _imu_cache.get('ang_vel'),
                'lin_acc': _imu_cache.get('lin_acc'),
                'quat': _imu_cache.get('quat'),
                'rpy': _imu_cache.get('rpy')
            }
    
    def calibrate_zero(self) -> bool:
        """角度值标定零位"""
        result = _send_and_wait({'device': self.device_type, 'cmd': 'cal_zero'})
        return result.get('ok', False)
    
    def calibrate_rpy(self) -> bool:
        """陀螺静态校准（别名）"""
        return self.calibrate_gyro()
    
    def calibrate_gyro(self) -> bool:
        """陀螺静态校准"""
        result = _send_and_wait({'device': self.device_type, 'cmd': 'cal_gyro'})
        return result.get('ok', False)
    
    def __repr__(self):
        return f"<IMU {'已连接' if _address else '未连接'}>"

# ============ 全局函数 ============
def register(device_type: str):
    """注册设备"""
    if device_type == "Taks-T1-imu":
        device = IMUDevice()
        device._register()
        return device
    device = TaksDevice(device_type)
    device._register()
    return device

def batch_query(robot: 'TaksDevice', imu: 'IMUDevice') -> tuple:
    """并行查询电机状态和IMU数据
    
    Returns:
        (motor_state, imu_data) - 电机状态字典和IMU数据字典
    """
    msgs = [
        {'device': robot.device_type, 'cmd': 'query', 'jids': []},
        {'device': imu.device_type, 'cmd': 'get_all'}
    ]
    results = _send_batch(msgs)
    
    # 解析电机状态
    motor_state = None
    if results[0].get('ok') and 'joints' in results[0]:
        motor_state = {int(k): v for k, v in results[0]['joints'].items()}
        with _motor_cache_lock:
            _motor_state_cache[robot.device_type] = motor_state
    else:
        with _motor_cache_lock:
            if robot.device_type in _motor_state_cache:
                motor_state = _motor_state_cache[robot.device_type].copy()
    
    # 解析IMU数据
    imu_data = None
    if results[1].get('ok') and 'data' in results[1]:
        imu_data = results[1]['data']
        with _imu_cache_lock:
            if 'ang_vel' in imu_data:
                _imu_cache['ang_vel'] = imu_data['ang_vel']
            if 'lin_acc' in imu_data:
                _imu_cache['lin_acc'] = imu_data['lin_acc']
            if 'quat' in imu_data:
                _imu_cache['quat'] = imu_data['quat']
            if 'rpy' in imu_data:
                _imu_cache['rpy'] = imu_data['rpy']
    else:
        with _imu_cache_lock:
            imu_data = {
                'ang_vel': _imu_cache.get('ang_vel'),
                'lin_acc': _imu_cache.get('lin_acc'),
                'quat': _imu_cache.get('quat'),
                'rpy': _imu_cache.get('rpy')
            }
    
    return motor_state, imu_data