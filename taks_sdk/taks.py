#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taks SDK - ZeroMQ 客户端库（同步版）
特点：
1. DEALER socket：双向通信
2. SUB socket：订阅机器人状态广播
3. 纯同步 API，无异步接口
"""

import zmq
import struct
import threading
from typing import List, Optional, Dict, Callable
from enum import IntEnum

# ============ 配置常量 ============
DEVICE_PREFIXES = {
    "Taks-T1": bytes([0xC7, 0xE9, 0x67, 0x34, 0x67, 0x7F]),
    "Taks-T1-leftarm": bytes([0xC7, 0xE9, 0x67, 0x34, 0x6A, 0x97]),
    "Taks-T1-rightarm": bytes([0xC7, 0xE9, 0x67, 0x34, 0x6B, 0x0A]),
    "Taks-T1-semibody": bytes([0xC7, 0xE9, 0x67, 0x34, 0x6B, 0x08]),
    "Taks-T1-imu": bytes([0xC7, 0xE9, 0x67, 0x34, 0x68, 0xF7]),
}

class CMD(IntEnum):
    POS_CTRL = 0x01
    POS_QUERY = 0x02
    VEL_QUERY = 0x03
    TRQ_QUERY = 0x04
    MIT_CTRL = 0x05
    # IMU 命令
    IMU_ANG_VEL = 0x10
    IMU_LIN_ACC = 0x11
    IMU_QUAT = 0x12
    IMU_RPY = 0x13
    IMU_CAL_ZERO = 0x14
    IMU_CAL_GYRO = 0x15
    REGISTER = 0xFF

class RESP(IntEnum):
    ACK = 0x00
    STATE = 0x01
    IMU_DATA = 0x02
    ERROR = 0xFF

# ============ 全局连接状态 ============
_ctx: Optional[zmq.Context] = None
_dealer: Optional[zmq.Socket] = None
_sub: Optional[zmq.Socket] = None
_address: Optional[str] = None
_state_callbacks: Dict[str, Callable] = {}
_recv_thread: Optional[threading.Thread] = None
_running: bool = False
_lock = threading.Lock()

# ============ 二进制协议编解码 ============
def _encode_position(joints: Dict[int, float]) -> bytes:
    buf = bytearray([CMD.POS_CTRL, len(joints)])
    for jid, val in joints.items():
        buf.extend(struct.pack('Bf', jid, val))
    return bytes(buf)

def _encode_mit(joints: Dict[int, Dict]) -> bytes:
    buf = bytearray([CMD.MIT_CTRL, len(joints)])
    nan = float('nan')
    for jid, p in joints.items():
        kp = p.get('kp') if p.get('kp') is not None else nan
        kd = p.get('kd') if p.get('kd') is not None else nan
        q, dq, tau = p.get('q', 0), p.get('dq', 0), p.get('tau', 0)
        buf.extend(struct.pack('Bfffff', jid, kp, kd, q, dq, tau))
    return bytes(buf)

def _encode_query(cmd_type: int, jids: List[int] = None) -> bytes:
    count = len(jids) if jids else 0
    buf = bytearray([cmd_type, count])
    if jids:
        buf.extend(jids)
    return bytes(buf)

def _decode_response(data: bytes) -> Dict:
    """解码服务器响应"""
    if len(data) < 1:
        return {'ok': False, 'error': 'empty response'}
    
    resp_type = data[0]
    if resp_type == RESP.ACK:
        return {'ok': True, 'code': data[1] if len(data) > 1 else 0}
    elif resp_type == RESP.STATE:
        count = data[1] if len(data) > 1 else 0
        joints = {}
        offset = 2
        for i in range(count):
            if offset + 13 > len(data):
                break
            jid, pos, vel, tau = struct.unpack_from('Bfff', data, offset)
            joints[jid] = {'pos': pos, 'vel': vel, 'tau': tau}
            offset += 13
        return {'ok': True, 'joints': joints}
    elif resp_type == RESP.IMU_DATA:
        imu_type = data[1] if len(data) > 1 else 0
        if imu_type == CMD.IMU_QUAT and len(data) >= 18:
            w, x, y, z = struct.unpack_from('ffff', data, 2)
            return {'ok': True, 'w': w, 'x': x, 'y': y, 'z': z}
        elif len(data) >= 14:
            x, y, z = struct.unpack_from('fff', data, 2)
            return {'ok': True, 'x': x, 'y': y, 'z': z}
        return {'ok': False, 'error': 'invalid IMU data'}
    elif resp_type == RESP.ERROR:
        msg_len = data[1] if len(data) > 1 else 0
        msg = data[2:2+msg_len].decode('utf-8', errors='ignore') if msg_len > 0 else 'unknown'
        return {'ok': False, 'error': msg}
    return {'ok': False, 'error': f'unknown response: {data[:20].hex()}'}

# ============ 连接管理 ============
def connect(address: str, cmd_port: int = 5555, sub_port: int = 5556):
    """连接到服务器"""
    global _ctx, _dealer, _sub, _address, _recv_thread, _running
    
    disconnect()
    
    _ctx = zmq.Context()
    _address = address
    
    # DEALER socket：发送命令，接收响应
    _dealer = _ctx.socket(zmq.DEALER)
    _dealer.setsockopt(zmq.RCVTIMEO, 5000)
    _dealer.setsockopt(zmq.SNDTIMEO, 1000)
    _dealer.connect(f"tcp://{address}:{cmd_port}")
    
    # SUB socket：订阅状态广播
    _sub = _ctx.socket(zmq.SUB)
    _sub.setsockopt(zmq.RCVTIMEO, 100)
    _sub.connect(f"tcp://{address}:{sub_port}")
    _sub.setsockopt(zmq.SUBSCRIBE, b'')  # 订阅所有
    
    # 启动接收线程（用于状态广播回调）
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
    """接收状态广播的后台线程"""
    while _running and _sub:
        try:
            data = _sub.recv()
            # 解析设备类型
            for dtype, prefix in DEVICE_PREFIXES.items():
                if data.startswith(prefix):
                    payload = data[len(prefix):]
                    result = _decode_response(payload)
                    if dtype in _state_callbacks:
                        try:
                            _state_callbacks[dtype](result)
                        except Exception as e:
                            print(f"✗ 回调异常: {e}")
                    break
        except zmq.Again:
            continue
        except Exception as e:
            if _running:
                print(f"✗ 接收异常: {e}")

def _send_and_wait(prefix: bytes, payload: bytes, timeout: float = 1.0) -> Dict:
    """发送命令并等待响应"""
    if not _dealer:
        raise RuntimeError("未连接，请先调用 connect()")
    
    with _lock:
        # 发送：[empty, prefix + payload]
        _dealer.send_multipart([b'', prefix + payload])
        
        # 等待响应
        try:
            frames = _dealer.recv_multipart()
            if frames:
                data = frames[-1]
                result = _decode_response(data)
                return result
            return {'ok': False, 'error': 'empty response'}
        except zmq.Again:
            return {'ok': False, 'error': 'timeout'}
        except Exception as e:
            return {'ok': False, 'error': str(e)}

def _send_fire_and_forget(prefix: bytes, payload: bytes):
    """发送命令，不等待响应（高频控制用）"""
    if not _dealer:
        raise RuntimeError("未连接，请先调用 connect()")
    _dealer.send_multipart([b'', prefix + payload], zmq.NOBLOCK)

# ============ 客户端API类 ============
class JointControl:
    """单关节控制"""
    __slots__ = ('_dev', '_jid')
    
    def __init__(self, device: 'TaksDevice', jid: int):
        self._dev = device
        self._jid = jid
    
    def _query(self, cmd: int, attr: str) -> Optional[float]:
        """通用查询方法"""
        result = _send_and_wait(self._dev._prefix, _encode_query(cmd, [self._jid]))
        if result.get('ok') and 'joints' in result:
            jdata = result['joints'].get(self._jid)
            return jdata.get(attr) if jdata else None
        return None
    
    def SetPosition(self, position: float):
        """设置位置"""
        _send_fire_and_forget(self._dev._prefix, _encode_position({self._jid: position}))
    
    def GetPosition(self) -> Optional[float]:
        """获取位置"""
        return self._query(CMD.POS_QUERY, 'pos')
    
    def GetVelocity(self) -> Optional[float]:
        """获取速度"""
        return self._query(CMD.VEL_QUERY, 'vel')
    
    def GetTorque(self) -> Optional[float]:
        """获取力矩"""
        return self._query(CMD.TRQ_QUERY, 'tau')
    
    def GetState(self) -> Optional[Dict]:
        """获取关节完整状态"""
        result = _send_and_wait(self._dev._prefix, _encode_query(CMD.POS_QUERY, [self._jid]))
        if result.get('ok') and 'joints' in result:
            return result['joints'].get(self._jid)
        return None
    
    def controlMIT(self, kp: Optional[float] = None, kd: Optional[float] = None,
                   q: float = 0, dq: float = 0, tau: float = 0):
        """MIT控制"""
        _send_fire_and_forget(self._dev._prefix, _encode_mit({self._jid: {'kp': kp, 'kd': kd, 'q': q, 'dq': dq, 'tau': tau}}))

class TaksDevice:
    """Taks设备主类"""
    def __init__(self, device_type: str):
        if device_type not in DEVICE_PREFIXES:
            raise ValueError(f"不支持的设备类型: {device_type}")
        
        self.device_type = device_type
        self._prefix = DEVICE_PREFIXES[device_type]
        
        if not _dealer:
            raise RuntimeError("请先调用 connect() 连接网络")
        
        # 关节配置表
        joint_map = {
            "leftarm": (list(range(9, 16)), 9),
            "rightarm": (list(range(1, 8)), 1),
            "semibody": ([1,2,3,4,5,6,7, 9,10,11,12,13,14,15, 17,18,19, 20,21,22], None),
        }
        # 全身模式
        if device_type == "Taks-T1":
            joints = [1,2,3,4,5,6,7, 9,10,11,12,13,14,15, 17,18,19, 20,21,22, 23,24,25,26,27,28, 29,30,31,32,33,34]
            for i in joints:
                setattr(self, f"j{i}", JointControl(self, i))
            self._jstart = None
        else:
            self._jstart = None
            for key, (joints, jstart) in joint_map.items():
                if key in device_type:
                    for i in joints:
                        setattr(self, f"j{i}", JointControl(self, i))
                    self._jstart = jstart
                    break
        
    
    def _register(self):
        """注册设备"""
        msg = f"注册设备：{self.device_type}".encode('utf-8')
        return _send_and_wait(self._prefix, msg)
    
    def SetPosition(self, j1=None, j2=None, j3=None, j4=None, j5=None, j6=None, j7=None):
        """批量位置控制"""
        positions = {'j1': j1, 'j2': j2, 'j3': j3, 'j4': j4, 'j5': j5, 'j6': j6, 'j7': j7}
        joint_data = {}
        for key, val in positions.items():
            if val is not None and key.startswith('j'):
                idx = int(key[1:])
                jid = self._jstart + idx - 1 if self._jstart else idx
                joint_data[jid] = val
        
        if joint_data:
            data = _encode_position(joint_data)
            _send_fire_and_forget(self._prefix, data)
    
    def _query_all(self, cmd: int, attr: str = None) -> Optional[Dict]:
        """通用批量查询"""
        result = _send_and_wait(self._prefix, _encode_query(cmd))
        if not (result.get('ok') and 'joints' in result):
            return None
        joints = result['joints']
        return {jid: state[attr] for jid, state in joints.items()} if attr else joints
    
    def GetPosition(self) -> Optional[Dict[int, float]]:
        """获取所有关节位置"""
        return self._query_all(CMD.POS_QUERY, 'pos')
    
    def GetVelocity(self) -> Optional[Dict[int, float]]:
        """获取所有关节速度"""
        return self._query_all(CMD.VEL_QUERY, 'vel')
    
    def GetTorque(self) -> Optional[Dict[int, float]]:
        """获取所有关节力矩"""
        return self._query_all(CMD.TRQ_QUERY, 'tau')
    
    def GetState(self) -> Optional[Dict[int, Dict]]:
        """获取所有关节完整状态"""
        return self._query_all(CMD.POS_QUERY)
    
    def controlMIT(self, joints: dict, kp=None, kd=None, q=None, dq=None, tau=None):
        """批量MIT控制"""
        mit_data = {}
        for jidx, params in joints.items():
            jid = (self._jstart + jidx - 1) if self._jstart else jidx
            mit_data[jid] = {
                'kp': params.get('kp', kp),
                'kd': params.get('kd', kd),
                'q': params.get('q', q if q is not None else 0),
                'dq': params.get('dq', dq if dq is not None else 0),
                'tau': params.get('tau', tau if tau is not None else 0)
            }
        
        if mit_data:
            data = _encode_mit(mit_data)
            _send_fire_and_forget(self._prefix, data)
    
    def subscribe_state(self, callback: Callable[[Dict], None]):
        """订阅状态回调"""
        _state_callbacks[self.device_type] = callback
    
    def unsubscribe_state(self):
        """取消订阅"""
        _state_callbacks.pop(self.device_type, None)
    
    def close(self):
        self.unsubscribe_state()
    
    def __repr__(self):
        status = f"已连接到 {_address}" if _address else "未连接"
        return f"<{self.device_type} {status}>"

class IMUDevice:
    """IMU设备类"""
    def __init__(self):
        self.device_type = "Taks-T1-imu"
        self._prefix = DEVICE_PREFIXES[self.device_type]
        if not _dealer:
            raise RuntimeError("请先调用 connect() 连接网络")
    
    def _query(self, cmd: int) -> Dict:
        """发送IMU查询命令"""
        payload = bytes([cmd, 0])
        return _send_and_wait(self._prefix, payload)
    
    def get_ang_vel(self) -> Optional[Dict]:
        """获取角速度 (rad/s) -> {'x': float, 'y': float, 'z': float}"""
        result = self._query(CMD.IMU_ANG_VEL)
        if result.get('ok'):
            return {'x': result.get('x', 0), 'y': result.get('y', 0), 'z': result.get('z', 0)}
        return None
    
    def get_lin_acc(self) -> Optional[Dict]:
        """获取线加速度 (m/s²) -> {'x': float, 'y': float, 'z': float}"""
        result = self._query(CMD.IMU_LIN_ACC)
        if result.get('ok'):
            return {'x': result.get('x', 0), 'y': result.get('y', 0), 'z': result.get('z', 0)}
        return None
    
    def get_quat(self) -> Optional[Dict]:
        """获取四元数 -> {'w': float, 'x': float, 'y': float, 'z': float}"""
        result = self._query(CMD.IMU_QUAT)
        if result.get('ok'):
            return {'w': result.get('w', 1), 'x': result.get('x', 0), 'y': result.get('y', 0), 'z': result.get('z', 0)}
        return None
    
    def get_rpy(self) -> Optional[Dict]:
        """获取欧拉角 (rad) -> {'roll': float, 'pitch': float, 'yaw': float}"""
        result = self._query(CMD.IMU_RPY)
        if result.get('ok'):
            return {'roll': result.get('x', 0), 'pitch': result.get('y', 0), 'yaw': result.get('z', 0)}
        return None
    
    def calibrate_zero(self) -> bool:
        """角度值标定零位"""
        result = self._query(CMD.IMU_CAL_ZERO)
        return result.get('ok', False) and result.get('code', 1) == 0
    
    def calibrate_rpy(self) -> bool:
        """陀螺静态校准（别名）"""
        return self.calibrate_gyro()
    
    def calibrate_gyro(self) -> bool:
        """陀螺静态校准"""
        result = self._query(CMD.IMU_CAL_GYRO)
        return result.get('ok', False) and result.get('code', 1) == 0
    
    def __repr__(self):
        status = f"已连接到 {_address}" if _address else "未连接"
        return f"<IMU {status}>"

# ============ 全局函数 ============
def register(device_type: str):
    """注册设备"""
    if device_type == "Taks-T1-imu":
        return IMUDevice()
    device = TaksDevice(device_type)
    device._register()
    return device