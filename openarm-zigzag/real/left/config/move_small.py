#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
微动测试工具(直接操作 JTC, 不经过 MoveIt)
==========================================
用法:
  # 单关节微动: J3 正向 3°, 限速 0.1 rad/s(去+回, 结束停在原位)
  /usr/bin/python3 move_small.py --joint 3 --deg 3

  # 慢速回标定初始位形(= 安全回位动作, 手动触发)
  /usr/bin/python3 move_small.py --to-home

  # 自定义限速(默认 0.1 rad/s, 真机首跑不要加快)
  /usr/bin/python3 move_small.py --joint 5 --deg -3 --speed 0.05

每关节独立验证: 目的是确认 (1) 关节编号与物理轴对应 (2) 正方向与 URDF 一致
(3) 读数连续无跳变。任何一次执行中按 Ctrl+C = 取消目标(电机保持当前位置)。
"""

import argparse
import math
import os
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

JOINTS = ["openarm_left_joint%d" % i for i in range(1, 8)]
D2R = math.pi / 180.0


class Mover(Node):

    def __init__(self, cfg_path):
        super().__init__("move_small")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.home = list(cfg["home_rad"])
        self.limits = [cfg["soft_limits_rad"][j] for j in JOINTS]
        self.act = ActionClient(
            self, FollowJointTrajectory,
            "/left_joint_trajectory_controller/follow_joint_trajectory")
        self._q = None
        self.status = None   # watchdog 状态 OK/TRIPPED/RETURNING(不在则 None)
        self.create_subscription(JointState, "/joint_states",
                                 self._on_js, qos_profile_sensor_data)
        self.create_subscription(String, "/safety/status",
                                 lambda m: setattr(self, "status", m.data),
                                 10)

    def wait_watchdog_ok(self, max_wait=600.0):
        """运动前等 watchdog 就绪(2026-09-04 修复"远距离回家中途停下"):
        - watchdog 不在 → 直接过(旧行为);
        - RETURNING  → watchdog 正在送臂回家, 等它完成, 不和它抢 JTC;
        - TRIPPED(触发锁定) → 提示人工复位, 等待 OK。
        状态 OK 时臂必在软限位+余量内, 此后的小段回程不会再触发。"""
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
                    print("watchdog 处于触发锁定(TRIPPED)。到终端 B 确认触发"
                          "原因, 确认安全后人工复位:")
                    print('  ros2 topic pub -1 /safety/reset std_msgs/Bool '
                          '"{data: true}"')
                    print("等待复位中 ...")
                else:
                    print("watchdog 状态: %r, 等待 ..." % self.status)
                last = self.status
            if time.monotonic() - t0 > max_wait:
                raise SystemExit("等待 watchdog 复位超时(%ds), 放弃" % max_wait)
            time.sleep(2.0)
        print("watchdog OK, 继续执行")

    def _on_js(self, msg):
        if msg.name and msg.name != JOINTS:
            try:
                idx = [msg.name.index(n) for n in JOINTS]
            except ValueError:
                return
            self._q = [msg.position[i] for i in idx]
        else:
            self._q = list(msg.position[:7])

    def wait_q(self, timeout=10.0):
        t0 = time.monotonic()
        while self._q is None:
            if time.monotonic() - t0 > timeout:
                raise SystemExit("10s 未收到 /joint_states——control 层没起来?")
            time.sleep(0.05)
        return self._q

    def send(self, traj):
        while not self.act.wait_for_server(timeout_sec=1.0):
            self.get_logger().info("等待 JTC action ...")
        fut = self.act.send_goal_async(traj)
        while not fut.done():
            time.sleep(0.02)
        gh = fut.result()
        if not gh.accepted:
            raise SystemExit("目标被拒绝")
        res = gh.get_result_async()
        while not res.done():
            time.sleep(0.1)
        return res.result().status

    def build(self, points):
        # JTC action 需要 FollowJointTrajectory.Goal 包装, 不是裸 JointTrajectory
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINTS
        goal.trajectory.points = points
        return goal


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safety-yaml",
                    default=os.path.join(os.path.dirname(
                        os.path.abspath(__file__)), "real_safety.yaml"))
    ap.add_argument("--joint", type=int, choices=range(1, 8),
                    help="微动的关节编号 1-7")
    ap.add_argument("--deg", type=float, default=3.0,
                    help="微动幅度(度), 正=URDF 正方向")
    ap.add_argument("--speed", type=float, default=0.1,
                    help="关节速度上限(rad/s), 默认 0.1 保守值")
    ap.add_argument("--to-home", action="store_true",
                    help="慢速回标定初始位形后停住")
    args = ap.parse_args()
    if not args.joint and not args.to_home:
        raise SystemExit("指定 --joint N(微动) 或 --to-home(回位)")

    rclpy.init()
    node = Mover(args.safety_yaml)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    spin_t = threading.Thread(target=ex.spin, daemon=True)
    spin_t.start()

    node.wait_watchdog_ok()          # watchdog TRIPPED/RETURNING 时不抢它的活
    q0 = list(node.wait_q())
    print("当前(°):", ["%.2f" % (v / D2R) for v in q0])

    if args.to_home:
        print("慢速回标定初始位形(限速 %.2f rad/s) ..." % args.speed)
        status = node.send(node.build(interp(q0, node.home, args.speed)))
        time.sleep(1.0)          # 等 MIT 收敛沉降后再读数
        q_end = node.wait_q()
        drift = max(abs(q_end[k] - node.home[k]) for k in range(7)) / D2R
        if status != 4 and node.status in ("TRIPPED", "RETURNING"):
            print("目标被取消(status=%d)且 watchdog 已接管回位(当前: %s)。"
                  "等它送回家后重新运行本命令做最后找平即可。"
                  % (status, node.status))
        else:
            print("完成, 状态=%d, 与 home 残余偏差 %.2f°" % (status, drift))
        ex.shutdown()
        spin_t.join(timeout=2.0)
        rclpy.try_shutdown()
        return

    i = args.joint - 1
    q_up = list(q0)
    q_up[i] += args.deg * D2R
    lo, hi = node.limits[i]
    if not (lo <= q_up[i] <= hi):
        raise SystemExit("目标超出软限位 [%.1f°, %.1f°], 拒绝执行"
                         % (lo / D2R, hi / D2R))
    print("J%d 微动 %+.1f° (限速 %.2f rad/s): 去 → 回 ..." %
          (args.joint, args.deg, args.speed))
    node.send(node.build(interp(q0, q_up, args.speed)))
    node.send(node.build(interp(q_up, q0, args.speed)))
    # JTC goal 容差 3.4° 偏宽, 结果返回时电机可能仍在收敛 → 等 1s 沉降再读
    time.sleep(1.0)
    q_end = node.wait_q()
    drift = max(abs(q_end[k] - q0[k]) for k in range(7)) / D2R
    print("回到原位, 残余偏差 %.2f°(应接近 0)" % drift)
    ex.shutdown()
    spin_t.join(timeout=2.0)
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
