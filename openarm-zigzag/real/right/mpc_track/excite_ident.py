#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MPC 辨识激励轨迹(逐关节多频正弦) + 全程数据记录
================================================================
目的: 辨识"指令位置→实际位置"的每关节二阶动态(跟踪滞后/等效惯量/
重力 droop), 为 PreviewMPC 参数校准与 MIT 增益整定提供实测数据。

流程: 臂回 home → 逐关节(k=1..7, 各 --secs 秒)激励:
    q_k(t) = home_k + Σ_i A_i·sin(2π f_i t + φ_i)
  其余关节保持 home。同时 100Hz 记录 (t, q, dq, tau) 与指令轨迹
  → npz 落盘(供 identify_params.py 离线辨识)。

安全: 峰值幅度 ~4.3°/速率 ~0.2rad/s(远低于 watchdog 阈值); 软限位
  内缩 3° 钳位; watchdog 未 OK 不启动; Ctrl+C = 保持当前位形。
前置: 终端 A(control) + 终端 B(watchdog) 已起, 臂已 --to-home。
用法:
  /usr/bin/python3 excite_ident.py [--secs 22] [--amp-scale 1.0]
      [--out ident_data.npz]
"""

import argparse
import json
import math
import os
import threading
import time

import numpy as np
import rclpy
import yaml
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from std_msgs.msg import String

_HERE = os.path.dirname(os.path.abspath(__file__))
SAFETY = os.path.abspath(os.path.join(_HERE, "..", "config", "real_safety.yaml"))
ACTION = "/right_joint_trajectory_controller/follow_joint_trajectory"
NAMES = ["openarm_right_joint%d" % i for i in range(1, 8)]
D2R = math.pi / 180.0
MARGIN = math.radians(3.0)

# 激励频率/幅度(度): 4 频叠加, 峰值 ~4.3°, 峰值速率 ~11°/s ≈ 0.19rad/s
FREQS = [0.15, 0.35, 0.70, 1.20]                    # Hz, 覆盖 MPC 关注带宽
AMPS = [1.8, 1.3, 0.8, 0.4]                          # 度
PHASE = [0.0, 1.3, 2.6, 4.1]                         # 固定相位(可复现)
CMD_HZ = 100


class ExciteNode(Node):

    def __init__(self, cfg_path):
        super().__init__("excite_ident")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.home = list(cfg["home_rad"])
        self.limits = [cfg["soft_limits_rad"][j] for j in NAMES]
        self.act = ActionClient(self, FollowJointTrajectory, ACTION)
        self._q = None
        self.status = None               # watchdog 状态 OK/TRIPPED/RETURNING
        self.rec_t = []
        self.rec_q = []
        self.rec_dq = []
        self.rec_tau = []
        self.rec_on = False
        self.t0 = None
        self.create_subscription(JointState, "/joint_states",
                                 self._on_js, qos_profile_sensor_data)
        self.create_subscription(String, "/safety/status",
                                 lambda m: setattr(self, "status", m.data),
                                 10)

    def _on_js(self, msg):
        # 按关节名取(项目教训: 广播器顺序不保证)
        try:
            q = [float(msg.position[msg.name.index(n)]) for n in NAMES]
            dq = [float(msg.velocity[msg.name.index(n)]) for n in NAMES]
            tau = [float(msg.effort[msg.name.index(n)]) for n in NAMES]
        except (ValueError, IndexError):
            return
        self._q = q
        if self.rec_on and self.t0 is not None:
            self.rec_t.append(round(time.monotonic() - self.t0, 4))
            self.rec_q.append(q)
            self.rec_dq.append(dq)
            self.rec_tau.append(tau)

    def wait_watchdog_ok(self, max_wait=600.0):
        t0 = time.monotonic()
        while self.status is None:
            if time.monotonic() - t0 > 2.0:
                return          # watchdog 未运行(仅 control 时)
            time.sleep(0.05)
        last = None
        t0 = time.monotonic()
        while self.status != "OK":
            if self.status != last:
                print("watchdog: %s, 等待 OK ..." % self.status)
                last = self.status
            if time.monotonic() - t0 > max_wait:
                raise SystemExit("等待 watchdog 超时, 放弃")
            time.sleep(2.0)
        print("watchdog OK, 继续执行")

    def wait_q(self, timeout=10.0):
        t0 = time.monotonic()
        while self._q is None:
            if time.monotonic() - t0 > timeout:
                raise SystemExit("读不到 /joint_states —— control 层在跑吗?")
            time.sleep(0.05)
        return self._q

    def send(self, jt):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = jt
        while not self.act.wait_for_server(timeout_sec=1.0):
            self.get_logger().info("等待 JTC action ...")
        fut = self.act.send_goal_async(goal)
        while not fut.done():
            time.sleep(0.02)
        gh = fut.result()
        if not gh.accepted:
            raise SystemExit("目标被 JTC 拒收")
        res = gh.get_result_async()
        while not res.done():
            time.sleep(0.1)
        return res.result().status


def build_traj(q_home, secs):
    """单关节通用激励轨迹由调用方按关节生成; 此处生成 100Hz 位置序列。"""
    n = int(secs * CMD_HZ)
    ts = [i / CMD_HZ for i in range(n + 1)]
    return ts


def excite_wave(amp_scale):
    """返回 f(t_sec) -> 偏移(rad), 多频叠加。"""
    def wave(t):
        return sum(a * amp_scale * D2R *
                   math.sin(2 * math.pi * f * t + ph)
                   for f, a, ph in zip(FREQS, AMPS, PHASE))
    return wave


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--safety-yaml", default=SAFETY)
    ap.add_argument("--secs", type=float, default=22.0,
                    help="每关节激励时长(秒), 默认 22")
    ap.add_argument("--amp-scale", type=float, default=1.0,
                    help="幅度缩放(首次建议 1.0; 保守可 0.6)")
    ap.add_argument("--out", default=os.path.join(here, "ident_data.npz"))
    args = ap.parse_args()

    rclpy.init()
    node = ExciteNode(args.safety_yaml)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    spin_t = threading.Thread(target=ex.spin, daemon=True)
    spin_t.start()

    node.wait_watchdog_ok()
    q0 = list(node.wait_q())
    off = max(abs(q0[k] - node.home[k]) for k in range(7)) / D2R
    if off > 2.0:
        raise SystemExit("臂不在 home(偏差 %.2f°)——先 move_small.py "
                         "--to-home" % off)
    print("=" * 62)
    print("MPC 辨识激励: 逐关节 %.0fs × 7, 峰值 ~%.1f°(x%.2f), "
          "速率 ≤0.2rad/s" % (args.secs, sum(AMPS), args.amp_scale))
    print("全程其余关节保持 home; Ctrl+C = 停止并保存已录数据")
    print("=" * 62)

    wave = excite_wave(args.amp_scale)
    node.t0 = time.monotonic()
    node.rec_on = True
    meta = {"freqs": FREQS, "amps_deg": AMPS, "phase": PHASE,
            "secs": args.secs, "amp_scale": args.amp_scale,
            "cmd_hz": CMD_HZ, "home": node.home, "segments": []}
    ref_t_all, ref_q_all, ref_joint = [], [], []
    try:
        peak = sum(AMPS) * args.amp_scale * D2R
        APPROACH, RETREAT = 2.0, 1.5
        for j in range(1, 8):
            i = j - 1
            lo, hi = node.limits[i]
            # 激励中心: 钳进 [lo+裕量+峰值, hi−裕量−峰值]——home 贴限位的
            # 关节(J2 负向/J4)自动平移到安全区中点(2026-09-09)
            center = min(max(q0[i], lo + MARGIN + peak),
                         hi - MARGIN - peak)
            t_total = APPROACH + args.secs + RETREAT
            t_seg = [k / CMD_HZ for k in range(int(t_total * CMD_HZ) + 1)]
            q_seg = []                       # 本段完整 7 关节指令
            for t in t_seg:
                row = list(q0)
                if t < APPROACH:
                    # 接近段: home → center(余弦缓动)
                    s = 0.5 * (1 - math.cos(math.pi * t / APPROACH))
                    row[i] = q0[i] + (center - q0[i]) * s
                elif t < APPROACH + args.secs:
                    # 激励段: center + 多频波(1s 幅度淡入, 起点连续)
                    env = min((t - APPROACH) / 1.0, 1.0)
                    row[i] = center + wave(t - APPROACH) * env
                else:
                    # 退回段: 当前值 → home(余弦缓动)
                    u = min((t - APPROACH - args.secs) / RETREAT, 1.0)
                    s = 0.5 * (1 - math.cos(math.pi * u))
                    prev = q_seg[-1][i] if q_seg else center
                    row[i] = prev + (q0[i] - prev) * s
                row[i] = min(max(row[i], lo + MARGIN), hi - MARGIN)
                q_seg.append(row)
            print(">>> J%d 激励中(中心 %+.1f°, 摆幅 ±%.1f°, "
                  "%.0f+%.0f+%.1fs) ..."
                  % (j, center / D2R, peak / D2R, APPROACH, args.secs,
                     RETREAT))
            t_start = time.monotonic() - node.t0
            jt = JointTrajectory()
            jt.joint_names = NAMES
            for k, t in enumerate(t_seg):
                p = JointTrajectoryPoint()
                p.positions = q_seg[k]
                p.time_from_start = Duration(
                    seconds=int(t), nanoseconds=int(
                        round((t - int(t)) * 1e9))).to_msg()
                jt.points.append(p)
            node.send(jt)
            t_end = time.monotonic() - node.t0
            meta["segments"].append({
                "joint": j, "t_start": round(t_start, 3),
                "t_end": round(t_end, 3),
                "wave_start": round(t_start + APPROACH + 1.0, 3),
                "wave_end": round(t_end - 0.8, 3),
                "center_deg": round(center / D2R, 2)})
            ref_t_all.extend(t_seg)
            ref_q_all.extend(q_seg)
            ref_joint.append(j)
            time.sleep(1.0)                  # 换关节间隙
    except KeyboardInterrupt:
        print("\n[Ctrl+C] 停止激励, 保持当前位形")
    finally:
        node.rec_on = False
        npz = dict(
            t_js=np.array(node.rec_t),
            q_js=np.array(node.rec_q),
            dq_js=np.array(node.rec_dq),
            tau_js=np.array(node.rec_tau),
            t_ref=np.array(ref_t_all),
            q_ref=np.array(ref_q_all),
            ref_joint=np.array(ref_joint),
            meta=json.dumps(meta, ensure_ascii=False),
            names=np.array(NAMES))
        np.savez_compressed(args.out, **npz)
        print("数据已写 %s (JS 采样 %d, 指令点 %d)"
              % (args.out, len(node.rec_t), len(ref_t_all)))
        # 末尾回 home(安全收位)
        try:
            cur = node.wait_q()
            jt = JointTrajectory()
            jt.joint_names = NAMES
            p = JointTrajectoryPoint()
            p.positions = [float(v) for v in node.home]
            p.time_from_start = Duration(seconds=3).to_msg()
            jt.points.append(p)
            node.send(jt)
            print("已回 home")
        except Exception as e:  # noqa: BLE001
            print("回 home 失败(电机保持): %s" % e)
    ex.shutdown()
    spin_t.join(timeout=2.0)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
