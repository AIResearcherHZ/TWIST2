#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taks SDK 简单使用示例
最小化的示例代码，展示基本用法
"""
import taks
import time

def simple_rightarm_example():
    """简单的右臂控制示例"""
    print("=== 简单右臂控制示例 ===\n")
    
    # 第一步：连接网络
    taks.connect("192.168.1.208", cmd_port=5555)
    print("✓ 已连接到服务器")
    
    # 第二步：注册设备
    device = taks.register("Taks-T1-rightarm")
    print("✓ 设备注册成功\n")
    
    # 第三步：控制关节
    print("--- 控制关节 ---")
    device.j1.SetPosition(0.5)
    print("设置 J1 位置: 0.5 rad")
    
    device.j2.SetPosition(1.0)
    print("设置 J2 位置: 1.0 rad")
    
    device.j3.SetPosition(0.3)
    print("设置 J3 位置: 0.3 rad\n")
    
    # 第四步：查询状态
    print("--- 查询状态 ---")
    pos = device.j2.GetPosition()
    print(f"J2 位置: {pos}")
    
    vel = device.j2.GetVelocity()
    print(f"J2 速度: {vel}")
    
    tau = device.j2.GetTorque()
    print(f"J2 力矩: {tau}")
    
    # 第五步：断开连接
    taks.disconnect()
    print("✓ 连接已关闭")

def simple_leftarm_example():
    """简单的左臂控制示例"""
    print("=== 简单左臂控制示例 ===\n")
    
    # 第一步：连接网络
    taks.connect("192.168.1.208", cmd_port=5555)
    print("✓ 已连接到服务器")
    
    # 第二步：注册设备
    device = taks.register("Taks-T1-leftarm")
    print("✓ 设备注册成功\n")
    
    # 第三步：控制关节
    print("--- 控制关节 ---")
    device.j9.SetPosition(0.5)
    print("设置 J9 位置: 0.5 rad")
    
    device.j10.SetPosition(1.0)
    print("设置 J10 位置: 1.0 rad")
    
    device.j11.SetPosition(0.3)
    print("设置 J11 位置: 0.3 rad\n")
    
    # 第四步：查询状态
    print("--- 查询状态 ---")
    pos = device.j10.GetPosition()
    print(f"J10 位置: {pos}")
    
    vel = device.j10.GetVelocity()
    print(f"J10 速度: {vel}")
    
    tau = device.j10.GetTorque()
    print(f"J10 力矩: {tau}\n")
    
    # 第五步：断开连接
    taks.disconnect()
    print("✓ 连接已关闭")

def simple_semibody_example():
    """简单的半身控制示例"""
    print("=== 简单半身控制示例 ===\n")
    
    # 第一步：连接网络
    taks.connect("192.168.1.208", cmd_port=5555)
    print("✓ 已连接到服务器")
    
    # 第二步：注册设备
    device = taks.register("Taks-T1-semibody")
    print("✓ 设备注册成功\n")
    
    # 第三步：控制关节
    print("--- 控制关节 ---")
    device.j2.controlMIT(kp=3.0, kd=1.0, q=-5, dq=0, tau=0)
    print("设置 J2 位置: 5 rad")

    device.j10.controlMIT(kp=3.0, kd=1.0, q=-5, dq=0, tau=0)
    print("设置 J10 位置: 5 rad\n")
    
    device.j17.controlMIT(kp=1, kd=0.1, q=-1.5, dq=0, tau=0)
    print("设置 J17 位置: 1.5 rad")
    
    # 第四步：查询状态
    print("--- 查询状态 ---")
    device.j1.GetPosition()
    print("查询 J1 位置")
    
    device.j1.GetVelocity()
    print("查询 J1 速度")
    
    device.j1.GetTorque()
    print("查询 J1 力矩\n")
    
    # 第五步：断开连接
    taks.disconnect()
    print("✓ 连接已关闭")

def simple_batch_control_example():
    """简单的批量控制示例"""
    print("\n=== 简单批量控制示例 ===\n")
    
    # 连接网络并注册设备
    taks.connect("192.168.1.208", cmd_port=5555)
    device = taks.register("Taks-T1-leftarm")
    print("✓ 已连接到服务器\n")
    
    # 批量设置位置
    print("--- 批量设置位置 ---")
    device.SetPosition(
        j1=0.2,
        j2=0.3,
        j3=0.4,
        j4=0.5,
        j5=0.6,
        j6=0.7,
        j7=0.8
    )
    print("设置所有关节位置")
    time.sleep(1)
    
    # 批量查询
    print("\n--- 批量查询 ---")
    device.GetPosition()
    print("查询所有关节位置")
    
    device.GetVelocity()
    print("查询所有关节速度")
    
    device.GetTorque()
    print("查询所有关节力矩\n")
    
    taks.disconnect()
    print("✓ 连接已关闭")

def simple_mit_control_example():
    """简单的 MIT 控制示例"""
    print("\n=== 简单 MIT 控制示例 ===\n")
    
    # 连接网络并注册设备
    taks.connect("192.168.1.208", cmd_port=5555)
    device = taks.register("Taks-T1-rightarm")
    print("✓ 已连接到服务器\n")
    
    # 单个关节 MIT 控制
    print("--- 单个关节 MIT 控制 ---")
    device.j1.controlMIT(kp=10.0, kd=1.0, q=1.0, dq=0, tau=0)
    print("J1 MIT 控制: kp=10.0, kd=1.0, q=1.0")
    time.sleep(1)
    
    # 多个关节 MIT 控制
    print("\n--- 多个关节 MIT 控制 ---")
    device.controlMIT(
        joints={
            1: {'q': 1.0},
            2: {'q': -1.0},
            3: {'q': 0.5}
        },
        kp=10.0,
        kd=1.0
    )
    print("多个关节 MIT 控制")
    
    taks.disconnect()
    print("✓ 连接已关闭")

def simple_imu_example():
    """简单的IMU传感器示例"""
    print("\n=== 简单IMU传感器示例 ===\n")
    
    # 第一步：连接网络
    taks.connect("192.168.1.208", cmd_port=5555)
    print("✓ 已连接到服务器")
    
    # 第二步：注册IMU设备
    imu = taks.register("Taks-T1-imu")
    print("✓ IMU设备注册成功\n")
    
    # 第三步：读取IMU数据
    print("--- 读取IMU数据 ---")
    
    # 获取角速度 (rad/s)
    ang_vel = imu.get_ang_vel()
    if ang_vel:
        print(f"角速度 (rad/s): x={ang_vel['x']}, y={ang_vel['y']}, z={ang_vel['z']}")
    
    # 获取线加速度 (m/s²)
    lin_acc = imu.get_lin_acc()
    if lin_acc:
        print(f"线加速度 (m/s²): x={lin_acc['x']}, y={lin_acc['y']}, z={lin_acc['z']}")
    
    # 获取欧拉角 (rad)
    rpy = imu.get_rpy()
    if rpy:
        print(f"欧拉角 (rad): roll={rpy['roll']}, pitch={rpy['pitch']}, yaw={rpy['yaw']}")
    
    # 获取四元数
    quat = imu.get_quat()
    if quat:
        print(f"四元数: w={quat['w']}, x={quat['x']}, y={quat['y']}, z={quat['z']}")
    
    # 第四步：IMU标定
    print("\n--- IMU标定 ---")
    
    # 角度值标定零位
    if imu.calibrate_zero():
        print("✓ 角度值标定零位成功")
    else:
        print("✗ 角度值标定零位失败")
    
    time.sleep(0.5)
    
    # 陀螺静态校准
    if imu.calibrate_gyro():
        print("✓ 陀螺静态校准成功")
    else:
        print("✗ 陀螺静态校准失败")
    
    # 第五步：断开连接
    print()
    taks.disconnect()
    print("✓ 连接已关闭")

def simple_ankle_example():
    """简单的踝关节控制示例
    
    踝关节解算说明：
    - J27(pitch) + J28(roll) = 右脚踝
    - J33(pitch) + J34(roll) = 左脚踝
    - 用户输入脚踝角度(pitch, roll)，SDK自动解算为电机角度
    - 查询时SDK自动将电机角度解算回脚踝角度
    """
    print("\n=== 简单踝关节控制示例 ===\n")
    
    # 第一步：连接网络
    taks.connect("192.168.1.208", cmd_port=5555)
    print("✓ 已连接到服务器")
    
    # 第二步：注册全身设备（踝关节在全身设备中）
    robot = taks.register("Taks-T1")
    # print("✓ 设备注册成功\n")
    # time.sleep(10)
    
    # # ========== 右脚踝位置控制 ==========
    # print("--- 右脚踝位置控制 ---")
    # # 设置右脚踝 pitch=0.15 rad, roll=0.1 rad
    # robot.j27.SetPosition(0.15)   # pitch
    # robot.j28.SetPosition(0.1)  # roll
    # print("右脚踝设置: pitch=0.15 rad, roll=0.1 rad")
    # time.sleep(1)
    
    # # ========== 左脚踝位置控制 ==========
    # print("\n--- 左脚踝位置控制 ---")
    # # 设置左脚踝 pitch=-0.15 rad, roll=-0.1 rad
    # robot.j33.SetPosition(-0.15)   # pitch
    # robot.j34.SetPosition(-0.1)  # roll
    # print("左脚踝设置: pitch=-0.15 rad, roll=-0.1 rad")
    # time.sleep(1)
    
    # ========== 右脚踝 MIT 控制 ==========
    print("\n--- 右脚踝 MIT 控制 ---")
    # MIT控制：pitch=-0.15, roll=0.0，增加 KP 和等待时间
    robot.j27.controlMIT(kp=10.0, kd=3.0, q=-0.3, dq=0, tau=0)   # pitch，提高 KP
    print("右脚踝MIT控制: pitch=-0.15, roll=0.0 (KP=50, KD=3)")
    time.sleep(3)  # 增加等待时间到3秒
    
    # # ========== 左脚踝 MIT 控制 ==========
    # print("\n--- 左脚踝 MIT 控制 ---")
    # # MIT控制：pitch=0.15, roll=0.1, pitch_vel=0, roll_vel=0
    # robot.j33.controlMIT(kp=20.0, kd=2.0, q=0.15, dq=0, tau=0)  # pitch
    # robot.j34.controlMIT(kp=20.0, kd=2.0, q=0.1, dq=0, tau=0)  # roll
    # print("左脚踝MIT控制: pitch=0.15, roll=0.1, pitch_vel=0, roll_vel=0")
    # time.sleep(1)
    
    # ========== 查询踝关节状态 ==========
    print("\n--- 查询踝关节状态 ---")
    
    # 查询右脚踝
    r_pitch_pos = robot.j27.GetPosition()
    r_roll_pos = robot.j28.GetPosition()
    print(f"右脚踝位置: pitch={r_pitch_pos} rad, roll={r_roll_pos} rad")
    
    r_pitch_vel = robot.j27.GetVelocity()
    r_roll_vel = robot.j28.GetVelocity()
    print(f"右脚踝速度: pitch_vel={r_pitch_vel} rad/s, roll_vel={r_roll_vel} rad/s")
    
    r_pitch_tau = robot.j27.GetTorque()
    r_roll_tau = robot.j28.GetTorque()
    print(f"右脚踝扭矩: tau_pitch={r_pitch_tau} Nm, tau_roll={r_roll_tau} Nm")
    
    # 查询左脚踝
    l_pitch_pos = robot.j33.GetPosition()
    l_roll_pos = robot.j34.GetPosition()
    print(f"\n左脚踝位置: pitch={l_pitch_pos} rad, roll={l_roll_pos} rad")
    
    l_pitch_vel = robot.j33.GetVelocity()
    l_roll_vel = robot.j34.GetVelocity()
    print(f"左脚踝速度: pitch_vel={l_pitch_vel} rad/s, roll_vel={l_roll_vel} rad/s")
    
    l_pitch_tau = robot.j33.GetTorque()
    l_roll_tau = robot.j34.GetTorque()
    print(f"左脚踝扭矩: tau_pitch={l_pitch_tau} Nm, tau_roll={l_roll_tau} Nm")
    
    # ========== 归零 ==========
    print("\n--- 踝关节归零 ---")
    # robot.j27.SetPosition(0.0)
    # robot.j28.SetPosition(0.0)
    # robot.j33.SetPosition(0.0)
    # robot.j34.SetPosition(0.0)
    print("所有踝关节归零")
    
    # 第三步：断开连接
    print()
    taks.disconnect()
    print("✓ 连接已关闭")

def simple_fullbody_example():
    """简单的 Taks-T1 全身控制示例
    
    全身关节分布：
    - J1-J7: 右臂 (right_hand CAN)
    - J9-J15: 左臂 (left_hand CAN)
    - J17-J19: 腰椎 (waist_neck CAN)
    - J20-J22: 脖子 (waist_neck CAN)
    - J23-J28: 右腿 (right_leg CAN)
    - J29-J34: 左腿 (left_leg CAN)
    """
    print("\n=== Taks-T1 全身控制示例 ===\n")
    
    # 第一步：连接网络
    taks.connect("192.168.1.208", cmd_port=5555)
    print("✓ 已连接到服务器")
    
    # 第二步：注册全身设备
    robot = taks.register("Taks-T1")
    print("✓ 全身设备注册成功\n")
    
    # ========== 上肢控制 ==========
    print("--- 上肢控制 ---")
    robot.j1.SetPosition(0.0)
    robot.j2.SetPosition(0.0)
    robot.j3.SetPosition(0.0)
    print("右臂 J1/J2/J3 归零")
    
    robot.j9.SetPosition(0.0)
    robot.j10.SetPosition(0.0)
    robot.j11.SetPosition(0.0)
    print("左臂 J9/J10/J11 归零")
    time.sleep(0.5)
    
    # ========== 腰椎和脖子控制 ==========
    print("\n--- 腰椎和脖子控制 ---")
    robot.j17.controlMIT(kp=250.0, kd=5.0, q=0.0, dq=0, tau=0)
    robot.j18.controlMIT(kp=250.0, kd=5.0, q=0.0, dq=0, tau=0)
    robot.j19.controlMIT(kp=250.0, kd=5.0, q=0.0, dq=0, tau=0)
    print("腰椎 J17/J18/J19 归零")
    
    robot.j20.controlMIT(kp=1.0, kd=0.5, q=0.0, dq=0, tau=0)
    robot.j21.controlMIT(kp=1.0, kd=0.5, q=0.0, dq=0, tau=0)
    robot.j22.controlMIT(kp=1.0, kd=0.5, q=0.0, dq=0, tau=0)
    print("脖子 J20/J21/J22 归零")
    time.sleep(0.5)
    
    # ========== 下肢控制 ==========
    print("\n--- 下肢控制 ---")
    robot.j23.controlMIT(kp=300.0, kd=8.0, q=0.0, dq=0, tau=0)
    robot.j24.controlMIT(kp=250.0, kd=5.0, q=0.0, dq=0, tau=0)
    robot.j25.controlMIT(kp=250.0, kd=5.0, q=0.0, dq=0, tau=0)
    robot.j26.controlMIT(kp=300.0, kd=8.0, q=0.0, dq=0, tau=0)
    robot.j27.controlMIT(kp=20.0, kd=2.0, q=0.0, dq=0, tau=0)
    robot.j28.controlMIT(kp=20.0, kd=2.0, q=0.0, dq=0, tau=0)
    print("右腿 J23-J28 归零")
    
    robot.j29.controlMIT(kp=300.0, kd=8.0, q=0.0, dq=0, tau=0)
    robot.j30.controlMIT(kp=250.0, kd=5.0, q=0.0, dq=0, tau=0)
    robot.j31.controlMIT(kp=250.0, kd=5.0, q=0.0, dq=0, tau=0)
    robot.j32.controlMIT(kp=300.0, kd=8.0, q=0.0, dq=0, tau=0)
    robot.j33.controlMIT(kp=20.0, kd=2.0, q=0.0, dq=0, tau=0)
    robot.j34.controlMIT(kp=20.0, kd=2.0, q=0.0, dq=0, tau=0)
    print("左腿 J29-J34 归零")
    time.sleep(0.5)
    
    # ========== 查询状态 ==========
    print("\n--- 查询全身状态 ---")
    robot.GetPosition()
    print("查询所有关节位置")
    
    robot.GetVelocity()
    print("查询所有关节速度")
    
    robot.GetTorque()
    print("查询所有关节力矩\n")
    
    taks.disconnect()
    print("✓ 连接已关闭")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("Taks SDK 简单使用示例")
    print("="*50 + "\n")
    
    # ========== 单臂示例 ==========
    # try:
    #     simple_rightarm_example()
    # except Exception as e:
    #     print(f"❌ 错误: {e}\n")

    # try:
    #     simple_leftarm_example()
    # except Exception as e:
    #     print(f"❌ 错误: {e}\n")
    
    # ========== 半身示例 ==========
    # try:
    #     simple_semibody_example()
    # except Exception as e:
    #     print(f"❌ 错误: {e}\n")
    
    # ========== IMU传感器示例 ==========
    # try:
    #     simple_imu_example()
    # except Exception as e:
    #     print(f"❌ 错误: {e}\n")
    
    # ========== 踝关节控制示例 ==========
    try:
        simple_ankle_example()
    except Exception as e:
        print(f"❌ 错误: {e}\n")
    
    # ========== 批量控制示例 ==========
    # try:
    #     simple_batch_control_example()
    # except Exception as e:
    #     print(f"❌ 错误: {e}\n")
    
    # try:
    #     simple_mit_control_example()
    # except Exception as e:
    #     print(f"❌ 错误: {e}\n")
    
    # ========== Taks-T1 全身控制示例 ==========
    # try:
    #     simple_fullbody_example()
    # except Exception as e:
    #     print(f"❌ 错误: {e}\n")
    
    print("="*50)
    print("所有示例执行完成")
    print("="*50 + "\n")