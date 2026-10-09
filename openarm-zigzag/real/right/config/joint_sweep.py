#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""J1→J7 顺序微动扫查(仿真对照实验的实机侧)
==========================================
从当前位形(应先 --to-home 回 home)出发, 逐关节朝 URDF 正方向动 deg 度
→ 停住保持 → 回原位 → 下一关节。全程按关节名记录 /joint_states,
落盘 JSON 供 compare_sweep.py 与仿真日志对比。

用法(阶段 D control 层已起、watchdog OK 后, 人工执行):
  /usr/bin/python3 move_small.py --to-home          # 先回 home
  /usr/bin/python3 joint_sweep.py                   # 默认每关节 +5°
  /usr/bin/python3 joint_sweep.py --deg 3 --speed 0.08

安全: 与 move_small 同口径 —— JTC 直连(不经 MoveIt)、限速、软限位
钳位 2.5°、运动前等 watchdog。Ctrl+C = 当前目标取消(电机保持)。
⚠ 方向终裁以肉眼为准: 读数链对符号错误是自洽的(命令与反馈同错),
  数值对比查不出来, 必须目测本关节转向与 RViz 仿真是否一致。
"""

import argparse
import json
import math
import os
import sys
import threading
import time

import rclpy
import yaml
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint

JOINTS = ["openarm_right_joint%d" % i for i in range(1, 8)]
D2R = math.pi / 180.0
DESC = {1: "肩俯仰(+后仰 / -前抬)",
        2: "肩侧摆(绕立柱, +一侧 / -另一侧)",
        3: "大臂滚转(自转)",
        4: "肘(只做正方向, 负向贴限位)",
        5: "腕滚转",
        6: "腕俯仰",
        7: "腕偏航"}
MARGIN = math.radians(2.5)   # 软限位钳位余量(> watchdog 2° 触发余量)
HOME_TOL = 15.0              # 离 home 超过此角度(度)要求显式确认


class SweepNode(Node):

    def __init__(self, cfg_path):
        super().__init__("joint_sweep")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.home = list(cfg["home_rad"])
        self.limits = [cfg["soft_limits_rad"][j] for j in JOINTS]
        self.act = ActionClient(
            self, FollowJointTrajectory,
            "/right_joint_trajectory_controller/follow_joint_trajectory")
        self._q = None
        self.status = None
        self.log = {"mode": "real", "samples": [], "segments": []}
        self.t0 = None
        self.create_subscription(JointState, "/joint_states",
                                 self._on_js, qos_profile_sensor_data)
        self.create_subscription(String, "/safety/status",
                                 lambda m: setattr(self, "status", m.data),
                                 10)

    def _on_js(self, msg):
        # 必须按关节名取(2026-09-09 教训: broadcaster 可能不按 J1..J7 排序)
        try:
            q = [float(msg.position[msg.name.index(n)]) for n in JOINTS]
        except (ValueError, IndexError):
            return
        self._q = q
        if self.t0 is not None:
            self.log["samples"].append(
                [round(time.monotonic() - self.t0, 4), q])

    def wait_watchdog_ok(self, max_wait=600.0):
        t0 = time.monotonic()
        while self.status is None:
            if time.monotonic() - t0 > 2.0:
                return          # watchdog 未运行
            time.sleep(0.05)
        last = None
        t0 = time.monotonic()
        while self.status != "OK":
            if self.status != last:
                if self.status == "RETURNING":
                    print("watchdog 回位进行中, 等待其完成(不抢占 JTC)...")
                elif self.status == "TRIPPED":
                    print("watchdog TRIPPED。确认安全后人工复位: "
                          'ros2 topic pub -1 /safety/reset std_msgs/Bool '
                          '"{data: true}"')
                last = self.status
            if time.monotonic() - t0 > max_wait:
                raise SystemExit("等待 watchdog 复位超时, 放弃")
            time.sleep(2.0)
        print("watchdog OK, 继续执行")

    def wait_q(self, timeout=10.0):
        t0 = time.monotonic()
        while self._q is None:
            if time.monotonic() - t0 > timeout:
                raise SystemExit("10s 未收到 /joint_states——control 层没起来?")
            time.sleep(0.05)
        return self._q

    def send(self, points):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINTS
        goal.trajectory.points = points
        while not self.act.wait_for_server(timeout_sec=1.0):
            self.get_logger().info("等待 JTC action ...")
        fut = self.act.send_goal_async(goal)
        while not fut.done():
            time.sleep(0.02)
        gh = fut.result()
        if not gh.accepted:
            raise SystemExit("目标被拒收(JTC)")
        res = gh.get_result_async()
        while not res.done():
            time.sleep(0.1)
        return res.result().status


def interp(q0, q1, speed):
    """关节空间线性插值, 每关节限速 speed(rad/s), 50ms 步长。"""
    worst = max(abs(q1[i] - q0[i]) for i in range(7))
    duration = max(worst / speed, 0.5)
    steps = max(int(duration / 0.05), 1)
    pts = []
    for k in range(1, steps + 1):
        t = k / steps
        pt = JointTrajectoryPoint()
        pt.positions = [q0[i] + t * (q1[i] - q0[i]) for i in range(7)]
        pt.velocities = [0.0] * 7
        pt.time_from_start = Duration(seconds=0.05 * k).to_msg()
        pts.append(pt)
    return pts


def settle_avg(node, last_frac=0.5):
    """取最近 last_frac 秒的 /joint_states 平均(沉降窗口)。"""
    node._q = None
    t0 = time.monotonic()
    buf = []
    while time.monotonic() - t0 < last_frac:
        if node._q is not None:
            buf.append(node._q)
            node._q = None
        time.sleep(0.02)
    if not buf:
        raise SystemExit("沉降窗口内无 /joint_states 数据")
    n = len(buf)
    return [sum(b[i] for b in buf) / n for i in range(7)]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--safety-yaml", default=os.path.join(here,
                                                          "real_safety.yaml"))
    ap.add_argument("--deg", type=float, default=5.0,
                    help="每关节微动幅度(度), 默认 5; 必须与仿真侧一致")
    ap.add_argument("--speed", type=float, default=0.1,
                    help="关节限速(rad/s), 默认 0.1")
    ap.add_argument("--from-current", action="store_true",
                    help="允许在偏离 home >15° 的位形上开始(不建议)")
    ap.add_argument("--out", default=os.path.join(here,
                                                  "sweep_real_latest.json"))
    args = ap.parse_args()

    rclpy.init()
    node = SweepNode(args.safety_yaml)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    spin_t = threading.Thread(target=ex.spin, daemon=True)
    spin_t.start()

    node.wait_watchdog_ok()
    q0 = list(node.wait_q())
    off = max(abs(q0[k] - node.home[k]) for k in range(7)) / D2R
    if off > HOME_TOL and not args.from_current:
        raise SystemExit("当前位形离 home 最大 %.1f°(>%d°)。先 "
                         "`move_small.py --to-home` 回位, 或确认安全后加 "
                         "--from-current。" % (off, HOME_TOL))

    print("=" * 60)
    print("J1→J7 顺序微动扫查: 每关节 +%.1f°(URDF 正方向)去→回, "
          "限速 %.2f rad/s" % (args.deg, args.speed))
    for j in range(1, 8):
        print("  J%d %s" % (j, DESC[j]))
    print("⚠ 每个关节动时请目测转向, 与 RViz 仿真对照(方向终裁依据)。")
    print("=" * 60)
    input("确认周围安全、急停可及后, 按回车开始 ...")

    node.log["deg"] = args.deg
    node.log["speed_rad_s"] = args.speed
    node.log["q_start"] = q0
    node.t0 = time.monotonic()
    results = {}
    try:
        for j in range(1, 8):
            i = j - 1
            q_up = list(q0)
            q_up[i] += args.deg * D2R
            lo, hi = node.limits[i]
            if not (lo + MARGIN <= q_up[i] <= hi - MARGIN):
                print("J%d 目标 %.1f° 贴软限位 [%.1f°, %.1f°], 跳过"
                      % (j, q_up[i] / D2R, lo / D2R, hi / D2R))
                continue
            seg = {"joint": j, "target_deg": args.deg,
                   "q_base": q0, "q_up": q_up}
            print(">>> J%d %s — 朝 URDF 正方向 %.1f° ..." % (j, DESC[j],
                                                            args.deg))
            node.send(interp(q0, q_up, args.speed))
            time.sleep(2.0)                    # 保持 + 沉降
            q_go = settle_avg(node)
            seg["q_go_avg"] = q_go
            node.send(interp(q_up, q0, args.speed))
            time.sleep(1.0)
            q_back = settle_avg(node)
            seg["q_back_avg"] = q_back
            node.log["segments"].append(seg)
            got = (q_go[i] - q0[i]) / D2R
            resid = max(abs(q_back[k] - q0[k]) for k in range(7)) / D2R
            results[j] = got
            print("    达成 %+.2f°(指令 %+.1f), 回位残余 %.2f°"
                  % (got, args.deg, resid))
    except KeyboardInterrupt:
        print("\n[中断] 取消目标, 电机保持当前位置")
        cur = node.wait_q()
        node.send(interp(cur, cur, args.speed))
    finally:
        node.log["results_deg"] = results
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(node.log, f)
        print("日志已写 %s (%d 采样, %d 关节段)"
              % (args.out, len(node.log["samples"]),
                 len(node.log["segments"])))
    ex.shutdown()
    spin_t.join(timeout=2.0)
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
