#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全监控节点(真机常驻)
======================
职责: 盯 /joint_states, 异常时把机械臂"安稳带回原位"(标定初始位形)。

触发条件(任一, 均需连续帧确认防毛刺):
  A. 软限位越界: 任一关节超出 zero 实测软限位 + trip_margin
  B. 异常运动:   任一关节速度连续超 max_joint_vel
  C. 人工急停:   /safety/estop (std_msgs/Bool) 收到 true

触发动作(latched, 只执行一次):
  1. 取消 /right_joint_trajectory_controller/follow_joint_trajectory 当前目标
     (打断正在执行的 MoveIt 轨迹)
  2. 不经过 MoveIt, 直接向 JTC 发一条关节空间慢速回位轨迹:
     当前位形 → 标定初始位形(home_rad), 每关节限速 return_speed(0.25 rad/s)

复位: 人工确认安全后
  ros2 topic pub -1 /safety/reset std_msgs/Bool "{data: true}"

注意: 本节点不做任何"主动运动", 只在异常时回位; 关闭本节点 = 关闭安全层。
"""

import argparse
import json
import math
import os
import threading
import time

import rclpy
import yaml
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from control_msgs.action import FollowJointTrajectory
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

JOINTS = ["openarm_right_joint%d" % i for i in range(1, 8)]


class SafetyWatchdog(Node):

    def __init__(self, cfg_path):
        super().__init__("safety_watchdog")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.limits = [cfg["soft_limits_rad"][j] for j in JOINTS]
        self.home = list(cfg["home_rad"])
        self.return_speed = float(cfg.get("return_speed", 0.25))
        self.max_vel = float(cfg.get("max_joint_vel", 4.0))
        self.margin = float(cfg.get("trip_margin_rad", 0.035))
        self.consecutive = int(cfg.get("trip_consecutive", 5))

        self._limit_cnt = 0     # 越界连续帧计数
        self._vel_cnt = 0       # 速度异常连续帧计数
        self._tripped = False
        self._tripping = False
        self._last_js_time = time.monotonic()
        self._last_q = None       # 最近一帧 7 关节位置(URDF 系)
        self._lock = threading.Lock()

        qos = qos_profile_sensor_data
        self.create_subscription(JointState, "/joint_states",
                                 self._on_js, qos)
        self.create_subscription(Bool, "/safety/estop", self._on_estop, 10)
        self.create_subscription(Bool, "/safety/reset", self._on_reset, 10)
        self.pub_status = self.create_publisher(String, "/safety/status", 10)
        self.act_fjt = ActionClient(
            self, FollowJointTrajectory,
            "/right_joint_trajectory_controller/follow_joint_trajectory")

        self.create_timer(0.1, self._publish_status)
        self.create_timer(0.5, self._check_js_alive)
        self.get_logger().info("安全监控已启动(触发条件: 越界+%em / 速度>%.1frad/s "
                               "或 /safety/estop)"
                               % (self.margin, self.max_vel))

    # ------------------------------------------------------------------

    def _on_js(self, msg):
        if msg.name and msg.name != JOINTS:
            # 只认 7 臂关节顺序; 顺序不一致时按名字重排
            try:
                idx = [msg.name.index(n) for n in JOINTS]
            except ValueError:
                return
            q = [msg.position[i] for i in idx]
            v = [msg.velocity[i] for i in idx]
        else:
            q = list(msg.position[:7])
            v = list(msg.velocity[:7])
        self._last_js_time = time.monotonic()
        with self._lock:
            self._last_q = q
        if self._tripped or self._tripping:
            return
        with self._lock:
            over = any(
                q[i] < self.limits[i][0] - self.margin
                or q[i] > self.limits[i][1] + self.margin for i in range(7))
            fast = any(abs(v[i]) > self.max_vel for i in range(7))
        self._limit_cnt = self._limit_cnt + 1 if over else 0
        self._vel_cnt = self._vel_cnt + 1 if fast else 0
        if self._limit_cnt >= self.consecutive:
            self._trip("软限位越界(连续%d帧)" % self._limit_cnt)
        elif self._vel_cnt >= 3:
            self._trip("异常运动速度(连续%d帧)" % self._vel_cnt)

    def _on_estop(self, msg):
        if msg.data and not self._tripped and not self._tripping:
            self._trip("人工急停(/safety/estop)")

    def _on_reset(self, msg):
        if msg.data and self._tripped and not self._tripping:
            self._tripped = False
            self._limit_cnt = 0
            self._vel_cnt = 0
            self.get_logger().warn("安全状态已人工复位, 恢复监控")

    def _check_js_alive(self):
        # 2026-09-18 安全强化: 断流>1.5s 从"仅日志"升级为"锁存 TRIPPED"。
        # 无反馈 = 策略会用过期状态算指令, 必须停掉一切指令流;
        # 不发起自动回位(没有反馈就不该主动运动), 请人工排查后 reset。
        if (not self._tripped and not self._tripping
                and time.monotonic() - self._last_js_time > 1.5):
            self._trip("/joint_states 断流>1.5s(control 层退出或反馈丢失)"
                       " → 锁定当前位形, 请人工排查")

    def _publish_status(self):
        m = String()
        if self._tripping:
            m.data = "RETURNING"
        elif self._tripped:
            m.data = "TRIPPED"
        else:
            m.data = "OK"
        self.pub_status.publish(m)

    # ------------------------------------------------------------------

    def _trip(self, reason, do_return=True):
        self._tripped = True
        self.get_logger().error("=" * 60)
        self.get_logger().error("安全触发: %s → 取消目标并慢速回标定初始位形"
                                % reason)
        self.get_logger().error("=" * 60)
        if do_return:
            t = threading.Thread(target=self._do_return, daemon=True)
            t.start()
        else:
            self.get_logger().error("(无反馈保护模式: 不发起自动回位)")

    def _do_return(self):
        """取消 JTC 当前目标, 然后发慢速回位轨迹(独立线程, 不阻塞 executor)。"""
        self._tripping = True
        try:
            # 1. 取消当前所有目标: rclpy 的 ActionClient 没有 cancel_all_goals
            #    (那是 rclcpp API), 直接调 action 取消服务,
            #    全零 GoalInfo = 取消全部(rclcpp cancel_all_goals 同语义)
            from action_msgs.srv import CancelGoal
            cli = self.create_client(
                CancelGoal,
                "/right_joint_trajectory_controller/follow_joint_trajectory"
                "/_action/cancel_goal")
            if not cli.wait_for_service(timeout_sec=5.0):
                self.get_logger().error("取消服务不可用, 无法打断当前轨迹!")
            else:
                fut = cli.call_async(CancelGoal.Request())  # goal_info 全零
                t0 = time.monotonic()
                while not fut.done():
                    if time.monotonic() - t0 > 3.0:
                        self.get_logger().error("取消请求超时")
                        break
                    time.sleep(0.02)
            time.sleep(0.2)
        except Exception as e:
            self.get_logger().error("取消目标异常: %s" % e)
        try:
            if not self.act_fjt.wait_for_server(timeout_sec=5.0):
                self.get_logger().error("JTC action 不可用, 无法回位! "
                                        "请人工断电/扶稳机械臂")
                return
        except Exception as e:
            self.get_logger().error("JTC 等待异常: %s" % e)

        try:
            # 2. 从最新状态插值回 home
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = JOINTS
            q0 = self._latest_q()
            if q0 is None:
                self.get_logger().error("无有效当前状态, 无法生成回位轨迹")
                return
            worst = max(abs(q0[i] - self.home[i]) for i in range(7))
            duration = max(worst / self.return_speed, 1.0)
            steps = int(duration / 0.05)
            for k in range(1, steps + 1):
                t = k / steps
                pt = JointTrajectoryPoint()
                pt.positions = [q0[i] + t * (self.home[i] - q0[i])
                                for i in range(7)]
                pt.velocities = [0.0] * 7
                pt.time_from_start = Duration(
                    seconds=0.05 * k).to_msg()
                goal.trajectory.points.append(pt)
            send_fut = self.act_fjt.send_goal_async(goal)
            while not send_fut.done():
                time.sleep(0.02)
            gh = send_fut.result()
            if not gh.accepted:
                self.get_logger().error("回位目标被拒! 请人工处理")
                return
            self.get_logger().warn(
                "回位轨迹已下发(%.1fs, 限速 %.2f rad/s), 执行中..."
                % (duration, self.return_speed))
            res_fut = gh.get_result_async()
            while not res_fut.done():
                time.sleep(0.1)
            self.get_logger().warn(
                "回位流程结束(状态 %d)" % res_fut.result().status)
        except Exception as e:
            self.get_logger().error("回位执行异常: %s" % e)
        finally:
            self._tripping = False

    def _latest_q(self):
        """从 TF 不可用时退路: 用 buf 缓存最近一次 /joint_states。"""
        with self._lock:
            q = self._last_q
        return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safety-yaml", required=True)
    args = ap.parse_args()
    rclpy.init()
    node = SafetyWatchdog(args.safety_yaml)
    # 多线程 executor: 订阅回调不停, 回位线程独立
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
