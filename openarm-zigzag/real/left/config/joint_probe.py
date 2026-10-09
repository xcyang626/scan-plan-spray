#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单关节单向验证共用模块(被 move_j1.py ~ move_j7.py 调用)。

行为: 从当前位置向一个方向动 deg 度 → 停住保持(脚本退出后 JTC
保持最后指令, 臂不动) → 人工对照 RViz 模型同向同量。
回 home: /usr/bin/python3 move_small.py --to-home

安全: 目标钳位在软限位内 2.5°(大于 watchdog 2° 触发余量),
超界拒绝执行; 限位实时读 real_safety.yaml。
executor 放后台线程转, 保证 /joint_states 回调随时被处理。
"""

import math
import os
import sys
import threading
import time

import rclpy
import yaml
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

_HERE = os.path.dirname(os.path.abspath(__file__))
SAFETY = os.path.join(_HERE, "real_safety.yaml")
ACTION = "/left_joint_trajectory_controller/follow_joint_trajectory"
NAMES = ["openarm_left_joint%d" % i for i in range(1, 8)]

# 各关节语义(2026-09-04 修正: joint1=俯仰, joint2=侧摆)
DESC = {1: "肩俯仰(+后仰 / -前抬)",
        2: "肩侧摆(绕立柱, +一侧 / -另一侧)",
        3: "大臂滚转(自转)",
        4: "肘(只做正方向, 负向贴限位)",
        5: "腕滚转",
        6: "腕俯仰",
        7: "腕偏航"}
DEFAULT_DEG = {1: 5.0, 2: 4.0, 3: 10.0, 4: 10.0, 5: 10.0, 6: 10.0, 7: 10.0}
MARGIN = math.radians(2.5)   # 钳位余量, 大于 watchdog 的 2° 触发余量
SPEED = 0.10                 # rad/s, 慢速便于观察


def probe(joint, argv):
    deg = float(argv[1]) if len(argv) > 1 else DEFAULT_DEG[joint]
    with open(SAFETY, encoding="utf-8") as f:
        lim = yaml.safe_load(f)["soft_limits_rad"]
    if isinstance(lim, dict):
        lo, hi = lim[NAMES[joint - 1]]
    else:
        lo, hi = lim[joint - 1]

    rclpy.init()
    node = Node("move_j%d" % joint)
    box = {"q": None}
    node.create_subscription(JointState, "/joint_states",
                             lambda m: box.update(q=list(m.position[:7])), 10)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    act = ActionClient(node, FollowJointTrajectory, ACTION)

    try:
        t0 = time.monotonic()
        while box["q"] is None and time.monotonic() - t0 < 5.0:
            time.sleep(0.05)
        if box["q"] is None:
            sys.exit("等不到 /joint_states —— 终端 A 的 control 层在跑吗?")
        q = box["q"]

        tgt = q[joint - 1] + math.radians(deg)
        if tgt < lo + MARGIN or tgt > hi - MARGIN:
            sys.exit("拒绝: J%d 目标 %.2f° 超出安全范围 [%.2f, %.2f]°\n"
                     "(软限位 %s, 保留 2.5° watchdog 余量)"
                     % (joint, math.degrees(tgt), math.degrees(lo + MARGIN),
                        math.degrees(hi - MARGIN),
                        "[%.2f, %.2f]" % (math.degrees(lo),
                                          math.degrees(hi))))

        dt = max(abs(math.radians(deg)) / SPEED, 1.5)
        jt = JointTrajectory()
        jt.joint_names = NAMES
        for t, pos in ((0.0, q), (dt, q[:joint - 1] + [tgt] + q[joint:])):
            p = JointTrajectoryPoint()
            p.positions = [float(v) for v in pos]
            p.time_from_start.sec = int(t)
            p.time_from_start.nanosec = int(round((t - int(t)) * 1e9))
            jt.points.append(p)

        if not act.wait_for_server(timeout_sec=5.0):
            sys.exit("JTC action 不可用 —— control 层的 JTC 起了吗?")
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = jt
        fut = act.send_goal_async(goal)

        def wait_future(f, timeout):
            t0 = time.monotonic()
            while not f.done() and time.monotonic() - t0 < timeout:
                time.sleep(0.05)
            return f.result() if f.done() else None

        gh = wait_future(fut, 10.0)
        if gh is None or not gh.accepted:
            sys.exit("目标被 JTC 拒收(或超时)")
        res = wait_future(gh.get_result_async(), 60.0)

        t0 = time.monotonic()
        while time.monotonic() - t0 < 1.0:
            time.sleep(0.05)
        now = box["q"][joint - 1] if box["q"] else float("nan")
        ok = res is not None and res.status == 4
        print("=" * 62)
        print("J%d (%s)" % (joint, DESC[joint]))
        print("  命令: %+.1f°  |  完成: %s" %
              (deg, "成功, 已停住保持" if ok else
               "失败(status=%s)" % (res.status if res else "无响应")))
        print("  当前读数: %.2f° (起点 %.2f°, 读数变化 %+.2f°)"
              % (math.degrees(now), math.degrees(q[joint - 1]),
                 math.degrees(now - q[joint - 1])))
        print("  请对照: RViz 模型 J%d 动的方向/幅度 = 实物?  记录后回 home:"
              % joint)
        print("    /usr/bin/python3 move_small.py --to-home")
        print("=" * 62)
    finally:
        executor.remove_node(node)
        executor.shutdown()
        spin.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()
