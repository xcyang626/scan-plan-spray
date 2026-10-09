#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""J1→J7 顺序微动扫查(仿真对照实验的仿真侧)
==========================================
与 real/right/config/joint_sweep.py 同一动作序列(J1→J7 每关节 +deg
去→回, 同限速), 但直连 MoveIt 仿真的 JTC——与实机同一控制器通道,
对照才公平(不经 MoveIt 规划)。全程按关节名记录 /joint_states 落盘,
供 real/right/config/compare_sweep.py 对比。

用法(先起仿真):
  # 终端 1: 右臂 MoveIt demo(与走线仿真同一环境)
  source /opt/ros/jazzy/setup.bash
  source ~/桌面/openarm/openarm_ros2_ws/install/setup.bash
  ros2 launch ~/桌面/openarm实验/仿真/moveit_sim/right_config/right_demo.launch.py
  # 终端 2(等 JTC 起来, 约 10s):
  /usr/bin/python3 joint_sweep_sim.py                # 默认每关节 +5°
  /usr/bin/python3 joint_sweep_sim.py --deg 3        # 与实机侧参数一致!

仿真无 watchdog/软限位文件; demo 出厂状态为全零位形, 即扫查基位。
"""

import argparse
import json
import math
import os
import threading
import time

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

JOINTS = ["openarm_right_joint%d" % i for i in range(1, 8)]
D2R = math.pi / 180.0
ACTION = "/right_joint_trajectory_controller/follow_joint_trajectory"


class SweepNode(Node):

    def __init__(self):
        super().__init__("joint_sweep_sim")
        self.act = ActionClient(self, FollowJointTrajectory, ACTION)
        self._q = None
        self.log = {"mode": "sim", "samples": [], "segments": []}
        self.t0 = None
        self.create_subscription(JointState, "/joint_states",
                                 self._on_js, qos_profile_sensor_data)

    def _on_js(self, msg):
        # 按关节名取, 不按位切片(仿真 URDF 带 finger_joint1, 可能排最前)
        try:
            q = [float(msg.position[msg.name.index(n)]) for n in JOINTS]
        except (ValueError, IndexError):
            return
        self._q = q
        if self.t0 is not None:
            self.log["samples"].append(
                [round(time.monotonic() - self.t0, 4), q])

    def wait_q(self, timeout=30.0):
        t0 = time.monotonic()
        while self._q is None:
            if time.monotonic() - t0 > timeout:
                raise SystemExit("%ds 未收到 /joint_states——demo 起了吗?"
                                 % int(timeout))
            time.sleep(0.05)
        return self._q

    def send(self, points):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINTS
        goal.trajectory.points = points
        while not self.act.wait_for_server(timeout_sec=1.0):
            self.get_logger().info("等待 JTC action(demo 起来约需 10s) ...")
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
    """关节空间线性插值, 每关节限速 speed(rad/s), 50ms 步长(与实机同)。"""
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
    ap.add_argument("--deg", type=float, default=5.0,
                    help="每关节微动幅度(度), 默认 5; 必须与实机侧一致")
    ap.add_argument("--speed", type=float, default=0.1,
                    help="关节限速(rad/s), 默认 0.1(与实机同)")
    ap.add_argument("--out", default=os.path.join(here,
                                                  "sweep_sim_latest.json"))
    args = ap.parse_args()

    rclpy.init()
    node = SweepNode()
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    spin_t = threading.Thread(target=ex.spin, daemon=True)
    spin_t.start()

    q0 = list(node.wait_q())
    print("J1→J7 顺序微动扫查(仿真): 每关节 +%.1f° 去→回, 限速 %.2f rad/s"
          % (args.deg, args.speed))
    print("基位(°):", ["%.2f" % (v / D2R) for v in q0])

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
            seg = {"joint": j, "target_deg": args.deg,
                   "q_base": q0, "q_up": q_up}
            print(">>> J%d +%.1f° ..." % (j, args.deg))
            node.send(interp(q0, q_up, args.speed))
            time.sleep(2.0)
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
        print("\n[中断] 取消目标")
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
