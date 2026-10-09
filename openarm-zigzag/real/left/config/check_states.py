#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
状态核对工具(只读, 不动机器人)
==============================
读 /joint_states, 打印每关节 URDF 系角度, 并与标定初始位形/软限位对照。

用法:
  /usr/bin/python3 check_states.py                 # 打印一次后退出
  /usr/bin/python3 check_states.py --watch         # 持续刷新(配合微动测试)

用途: control 层起来后, 核对"MoveIt 眼中的关节值"与"真实位形"是否一致——
这是真机部署最重要的对齐验证。核对手法:
  1. 目测当前实际位形, 对照打印度数是否合理(±2° 内)
  2. 用 move_small.py 微动某关节, --watch 下打印值应同向同量变化
"""

import argparse
import math
import os
import threading
import time

import rclpy
import yaml
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

JOINTS = ["openarm_left_joint%d" % i for i in range(1, 8)]
R2D = 180.0 / math.pi


class CheckNode(Node):

    def __init__(self, cfg_path):
        super().__init__("check_states")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.limits = [cfg["soft_limits_rad"][j] for j in JOINTS]
        self.home = list(cfg["home_rad"])
        self._cv = threading.Event()
        self.create_subscription(JointState, "/joint_states",
                                 self._on_js, qos_profile_sensor_data)

    def _on_js(self, msg):
        if msg.name and msg.name != JOINTS:
            try:
                idx = [msg.name.index(n) for n in JOINTS]
            except ValueError:
                return
            q = [msg.position[i] for i in idx]
        else:
            q = list(msg.position[:7])
        self._q = q
        self._cv.set()

    def wait_one(self, timeout=10.0):
        return self._cv.wait(timeout)

    def print_report(self):
        q = self._q
        print("=" * 78)
        print("%-26s %9s %13s %13s %9s" %
              ("关节", "当前(°)", "限位下限(°)", "限位上限(°)", "偏差(°)"))
        worst = 0.0
        for i, n in enumerate(JOINTS):
            dev = q[i] - self.home[i]
            worst = max(worst, abs(dev))
            print("%-26s %9.2f %13.2f %13.2f %9.2f"
                  % (n, q[i] * R2D, self.limits[i][0] * R2D,
                     self.limits[i][1] * R2D, dev * R2D))
        in_lim = all(self.limits[i][0] <= q[i] <= self.limits[i][1]
                     for i in range(7))
        print("软限位内: %s | 与标定初始位形最大偏差: %.2f°"
              % ("是" if in_lim else "否(越界!)", worst * R2D))
        print("标定初始位形参考: "
              + " ".join("J%d %.2f" % (i + 1, v * R2D)
                         for i, v in enumerate(self.home)) + " (°)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safety-yaml",
                    default=os.path.join(os.path.dirname(
                        os.path.abspath(__file__)), "real_safety.yaml"))
    ap.add_argument("--watch", action="store_true",
                    help="持续刷新(Ctrl+C 退出), 配合微动测试")
    args = ap.parse_args()
    rclpy.init()
    node = CheckNode(args.safety_yaml)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    spin_t = threading.Thread(target=ex.spin, daemon=True)
    spin_t.start()

    if not args.watch:
        if not node.wait_one(10.0):
            print("10s 未收到 /joint_states——control 层没起来?")
            raise SystemExit(1)
        node.print_report()
        # 先停 executor 再关闭, 避免退出时 "terminate called" 噪音
        ex.shutdown()
        spin_t.join(timeout=2.0)
        node.destroy_node()
        rclpy.try_shutdown()
        return

    print("持续监控中, Ctrl+C 退出")
    try:
        while True:
            if node._cv.is_set():
                node._cv.clear()
                node.print_report()
            time.sleep(1.0)
    except KeyboardInterrupt:
        ex.shutdown()
        spin_t.join(timeout=2.0)
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
