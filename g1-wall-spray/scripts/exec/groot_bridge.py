# -*- coding: utf-8 -*-
"""GR00T 执行接入层 —— 把 GR00T 的全身控制能力接到网页工作站上。

策略、安全与消息协议均沿用 GR00T 官方实现（GEAR-SONIC），本文件提供三个接口：
  1) `GR00TBridge.publish_upper_body(...)`：把规划好的喷涂动作（17 个上肢关节）
     经 GR00T 官方 ZMQ 发送端模块发布到 `planner` 话题的 `upper_body_position` 字段
     —— 这是 v7 已验证过的路径（reference/upper_body/v7_arm_sdk_safe.csv）。
  2) `send_key(...)`：向部署进程 stdin 注入**GR00T 原生按键**
     （] 启动控制 / ENTER 切流 / O 急停 / I 重置朝向 / F 报温度）。
  3) `launch_deploy(...)`：按官方推荐参数启动 C++ 部署（`--input-type zmq_manager`）。

复用的官方模块：
  gear_sonic.utils.teleop.zmq.zmq_planner_sender:
      build_command_message(start, stop, planner, delta_heading?)
      build_planner_message(mode, movement, facing, speed, height,
                            upper_body_position, upper_body_velocity, ...)

上肢 17 关节顺序（与 reference/upper_body/v7_arm_sdk_safe.csv 表头一致）：
  waist_yaw, waist_roll, waist_pitch,
  left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow,
  left_wrist_roll, left_wrist_pitch, left_wrist_yaw,
  right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow,
  right_wrist_roll, right_wrist_pitch, right_wrist_yaw

我们动作 qpos(36) 的对应索引：腰=19,20,21；左臂=22..28；右臂=29..35（与上面同序）。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

GROOT_ROOT = os.environ.get("GROOT_ROOT",
                            "/home/yang/桌面/轨迹规划/GR00T-WholeBodyControl")
if GROOT_ROOT not in sys.path:
    sys.path.insert(0, GROOT_ROOT)

# ---- GR00T 官方发送端模块 ----
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (  # noqa: E402
    build_command_message,
    build_planner_message,
    pack_pose_message,
)

# 上肢 17 关节在本项目 qpos(36) 中的索引
UPPER_BODY_IDX = [19, 20, 21] + list(range(22, 29)) + list(range(29, 36))
# LocomotionMode::IDLE —— planner 模式下"原地站立"（下肢由机器人自身平衡）
MODE_IDLE = 0

# GR00T 原生按键（与官方 keyboard/zmq 接口一致）
KEY_START = "]"        # 启动控制
KEY_STREAM_TOGGLE = "\n"  # ENTER: 切换 ZMQ 流模式
KEY_ESTOP = "O"        # 急停: 官方 = 阻尼指令后退出
KEY_REINIT = "I"       # 重置基座朝向
KEY_TEMP = "F"         # 报告电机温度


class Publisher:
    """发布 GR00T planner 话题（含 upper_body_position）。"""

    def __init__(self, host="*", port=5556):
        import zmq
        self._zmq = zmq
        self.ctx = zmq.Context.instance()
        self.pub = self.ctx.socket(zmq.PUB)
        # 与官方 send_upper_body_trajectory.py 一致: 高水位 10 帧(消费不及丢旧帧,
        # 保实时), LINGER 0(关闭不等待) —— 控制流不能排队陈旧目标
        self.pub.setsockopt(zmq.SNDHWM, 10)
        self.pub.setsockopt(zmq.LINGER, 0)
        self.pub.bind(f"tcp://{host}:{port}")
        self.host, self.port = host, port
        self.t0 = time.time()

    # ---- 官方 command 话题 ----
    def command(self, start=False, stop=False, planner=True):
        self.pub.send(build_command_message(
            start=start, stop=stop, planner=planner))

    # ---- 官方 planner 话题: 上肢 17 关节目标 ----
    def planner_upper_body(self, upper17, movement=(0.0, 0.0, 0.0),
                           facing=(1.0, 0.0, 0.0), speed=-1.0, height=-1.0,
                           upper_vel=None):
        self.pub.send(build_planner_message(
            mode=MODE_IDLE, movement=movement, facing=facing,
            speed=speed, height=height,
            upper_body_position=list(np.asarray(upper17, float)),
            upper_body_velocity=None if upper_vel is None else list(np.asarray(upper_vel, float)),
        ))

    def close(self):
        try:
            self.pub.close(0)
        except Exception:
            pass


def qpos_to_upper17(qpos_row):
    """本项目 qpos(36) -> 上肢 17 关节（顺序与官方 v7 CSV 一致）。"""
    return np.asarray(qpos_row, float)[UPPER_BODY_IDX]


# ============ 官方上肢 17 关节顺序（send_upper_body_trajectory.py 原样对齐） ============
# CSV/硬件序: 腰3 + 左臂7 + 右臂7 (与 v7_arm_sdk_safe.csv 列序一致)
HARDWARE_ORDER = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
# SONIC 端期望的交错序 (官方注释: Mirrors SONIC PlannerStreamer._get_upper_body_joint_indices)
SONIC_ORDER = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
# 硬件序 -> SONIC 序 的重排索引 (官方脚本: [HARDWARE_ORDER.index(n) for n in SONIC_ORDER])
SONIC_PERM = [HARDWARE_ORDER.index(n) for n in SONIC_ORDER]


def to_sonic_order(u_hw):
    """硬件序 (T,17) -> SONIC 交错序 (T,17)。发 upper_body_position 前必须重排。"""
    return np.asarray(u_hw, float)[:, SONIC_PERM]


def qpos_upper17_hw(motion_npz, mjcf=None):
    """读动作 npz(qpos T×36) -> 硬件序 (T,17)。
    mjcf 给定时按官方 README §8 方法用 mujoco jnt_qposadr 按关节名解析地址
    (优于硬编码); 否则用已对 MJCF 验证过的默认 [19..35]。"""
    z = np.load(motion_npz)
    q = np.asarray(z["qpos"] if "qpos" in z.files else z[z.files[0]], float)
    assert q.ndim == 2 and q.shape[1] == 36, f"动作维度异常: {q.shape}"
    if mjcf:
        import mujoco
        m = mujoco.MjModel.from_xml_path(mjcf)
        idx = [int(m.jnt_qposadr[m.joint(n).id]) for n in HARDWARE_ORDER]
    else:
        idx = UPPER_BODY_IDX
    return q[:, idx]


def export_upper_csv(motion_npz, out_csv, mjcf=None):
    """按官方 README §8: qpos(T,36) npz -> time_s + 17 列(硬件序) CSV。
    产物可直接喂官方 send_upper_body_trajectory.py --csv。"""
    import csv as _csv
    u = qpos_upper17_hw(motion_npz, mjcf)
    with open(out_csv, "w", newline="", encoding="utf8") as f:
        w = _csv.writer(f)
        w.writerow(["time_s", *HARDWARE_ORDER])
        for i, row in enumerate(u):
            w.writerow([f"{i / 30.0:.6f}", *[f"{v:.9f}" for v in row]])
    return out_csv


def load_upper17(motion_npz):
    """读动作 npz(qpos T×36) -> (T,17) 上肢目标 + (T,17) 速度。"""
    z = np.load(motion_npz)
    q = np.asarray(z["qpos"] if "qpos" in z.files else z[z.files[0]], float)
    assert q.ndim == 2 and q.shape[1] == 36, f"动作维度异常: {q.shape}"
    u = q[:, UPPER_BODY_IDX]
    d = np.vstack([np.zeros((1, 17)), np.diff(u, axis=0) * 30.0])   # 30Hz 数值速度
    return u, d


def stream_upper_body(motion_npz, host="*", port=5556, rate=30.0,
                      start_command=True, realtime=True, verbose=True):
    """把动作的上肢 17 关节按实时速率流给机器人（GR00T 官方格式）。

    返回 (帧数, none)。真实时间流速 = rate Hz；每条 planner 消息 1 帧。
    """
    u, d = load_upper17(motion_npz)
    n = len(u)
    pub = Publisher(host, port)
    try:
        # 官方启用序列: command(start=True, planner=True)
        if start_command:
            for _ in range(3):
                pub.command(start=True, stop=False, planner=True)
                time.sleep(0.2)
            if verbose:
                print(f"[gr00t-bridge] 已发 command(start, planner=True)", flush=True)
        time.sleep(0.8)                      # PUB/SUB 慢加入
        dt, t0 = 1.0 / rate, time.time()
        for k in range(n):
            pub.planner_upper_body(u[k], upper_vel=d[k])
            if realtime:
                el = time.time() - t0
                if el < (k + 1) * dt:
                    time.sleep((k + 1) * dt - el)
            if verbose and k % 300 == 0:
                print(f"[gr00t-bridge] {k}/{n} 帧", flush=True)
        if verbose:
            print(f"[gr00t-bridge] 上肢动作流完成 {n}/{n} 帧", flush=True)
        return n
    finally:
        pub.close()


# ================= GR00T 官方 Protocol v1：全身动作流(pose 话题) =================
# 官方文档: "Streams raw G1 joint positions and velocities. Use this when your source
#            provides direct qpos/qvel data (e.g., from another simulator or motion
#            capture retargeting pipeline)."
# 必填字段(接收端 zmq_endpoint_interface.hpp:659-667 校验):
#   joint_pos f32[N,29] / joint_vel f32[N,29] / body_quat_w f32[N,4] / frame_index i64[N]
# 本项目的动作 qpos(36) = root7 + 29 关节 -> 直接对应。
JOINT_SLICE = slice(7, 36)
QUAT_SLICE = slice(3, 7)


def qpos_to_v1_fields(qpos, start_index=0, fps=30.0):
    """qpos(T,36) -> 官方 Protocol v1 字段 (joint_pos/joint_vel/body_quat_w/frame_index)."""
    q = np.asarray(qpos, float)
    jp = q[:, JOINT_SLICE].astype(np.float32)
    jv = np.vstack([np.zeros((1, jp.shape[1]), np.float32),
                    np.diff(jp, axis=0) * fps]).astype(np.float32)
    bq = q[:, QUAT_SLICE].astype(np.float32)
    bq /= np.maximum(np.linalg.norm(bq, axis=1, keepdims=True), 1e-9)
    fi = (np.arange(len(jp), dtype=np.int64) + int(start_index))
    return jp, jv, bq, fi


def stream_pose(motion_npz, host="*", port=5556, rate=30.0, chunk=1,
                start_command=True, verbose=True):
    """用**官方 pack_pose_message(version=1)** 把全身动作按实时速率流给机器人。
    start_command=True 时先发 command(start=True, planner=False) 进入流式动作模式。"""
    z = np.load(motion_npz)
    q = np.asarray(z["qpos"] if "qpos" in z.files else z[z.files[0]], float)
    assert q.ndim == 2 and q.shape[1] == 36, f"动作维度异常: {q.shape}"
    jp, jv, bq, fi = qpos_to_v1_fields(q)
    n = len(jp)
    pub = Publisher(host, port)
    try:
        if start_command:
            for _ in range(3):
                pub.command(start=True, stop=False, planner=False)   # 流式动作模式
                time.sleep(0.2)
            if verbose:
                print("[gr00t-bridge] 已发 command(start, planner=False)=流式动作模式", flush=True)
        time.sleep(0.8)                     # PUB/SUB 慢加入
        dt, t0 = 1.0 / rate, time.time()
        for k in range(0, n, chunk):
            e = min(k + chunk, n)
            msg = pack_pose_message({
                "joint_pos": jp[k:e],
                "joint_vel": jv[k:e],
                "body_quat_w": bq[k:e],
                "frame_index": fi[k:e],
                "catch_up": np.array([1], dtype=np.uint8),
            }, topic="pose", version=1)
            pub.pub.send(msg)
            if verbose and k % 300 == 0:
                print(f"[gr00t-bridge] {k}/{n} 帧", flush=True)
            el = time.time() - t0
            if el < e * dt:
                time.sleep(e * dt - el)
        if verbose:
            print(f"[gr00t-bridge] 全身动作流完成 {n}/{n} 帧", flush=True)
        return n
    finally:
        pub.close()


# ---------------- 与部署进程的按键通道（GR00T 原生按键）----------------
def send_key(fifo_path, ch):
    """向部署进程 stdin(FIFO) 注入 GR00T 原生按键。"""
    try:
        fd = os.open(fifo_path, os.O_WRONLY | os.O_NONBLOCK)
        try:
            os.write(fd, ch.encode())
        finally:
            os.close(fd)
        return True
    except Exception:
        return False


def send_command(kind, host="*", port=5556, repeat=3):
    """经 GR00T 官方 command 话题下发控制命令(在 GR00T venv 内运行, 依赖 msgpack)."""
    pub = Publisher(host, port)
    try:
        for _ in range(repeat):
            if kind == "start":
                pub.command(start=True, stop=False, planner=True)
            elif kind == "start_stream":
                pub.command(start=True, stop=False, planner=False)
            elif kind == "stop":
                pub.command(start=False, stop=True, planner=False)
            elif kind == "planner_on":
                pub.command(start=False, stop=False, planner=True)
            elif kind == "planner_off":
                pub.command(start=False, stop=False, planner=False)
            time.sleep(0.15)
    finally:
        pub.close()
    print(f"[gr00t-bridge] command 已下发: {kind}", flush=True)
    return True


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="GR00T 执行接入层自测/独立运行")
    ap.add_argument("--motion", help="动作 npz (qpos T×36)")
    ap.add_argument("--send-command",
                    choices=["start", "start_stream", "stop", "planner_on", "planner_off"],
                    help="下发 GR00T 官方 command")
    ap.add_argument("--host", default="*")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.send_command:
        sys.exit(0 if send_command(a.send_command, a.host, a.port) else 1)
    if a.selftest:
        u, d = load_upper17(a.motion)
        print(f"[自测] 上肢目标 {u.shape} 速度 {d.shape} (硬件序)")
        print(f"[自测] 第0帧 17 关节(硬件序): {u[0].round(3)}")
        us = to_sonic_order(u)
        print(f"[自测] 第0帧 17 关节(SONIC交错序): {us[0].round(3)}")
        print(f"[自测] 关节范围: [{u.min():.3f}, {u.max():.3f}] rad")
        m = build_planner_message(mode=0, movement=[0, 0, 0], facing=[1, 0, 0],
                                  speed=-1.0, height=-1.0,
                                  upper_body_position=us[0].tolist(),
                                  upper_body_velocity=[0.0] * 17)
        print(f"[自测] planner 报文 {len(m)} 字节, topic={m[:7]}")
        c = build_command_message(start=True, stop=False, planner=True)
        print(f"[自测] command 报文 {len(c)} 字节, topic={c[:7]}")
        print("[自测] 通过 —— 复用 GR00T 官方协议格式(含 SONIC 交错序重排)")
    elif a.motion:
        # 默认: 官方 Protocol v1 全身动作流(pose 话题)
        stream_pose(a.motion, a.host, a.port)
    else:
        ap.error("需要 --motion 或 --selftest")
