#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenArm 轨迹桥接节点（rclpy 完整解耦版）
=========================================

用途
----
在不改动 motorbridge/libdm 电机栈（arm_dm_control.py）的前提下，对外暴露
ros2_control 客户端的【标准跟随轨迹接口】（control_msgs/action/FollowJointTrajectory）。
这样 MoveIt / ros2 控制器 / 自定义客户端都能用统一的「轨迹点序列」去控制 6 个电机。

数据流
------
  [controller_client]  --(FollowJointTrajectory action)-->  本节点
      本节点轨迹插值 -> 写 /dm_arm/joint_command (sensor_msgs/JointState)
      读 /dm_arm/joint_states (sensor_msgs/JointState) --> 发布 /joint_states 供 robot_state_publisher

设计说明（重要）
----------------
本进程内运行【两个独立节点】，各自用独立的 SingleThreadedExecutor（互不阻塞）：
  - StateForwarder：订阅 /dm_arm/joint_states，转存到线程安全共享区，并发布 /joint_states。
  - TrajectoryServer：ActionServer 跟随轨迹，执行回调在自己的 executor 线程里阻塞式跑
    轨迹插值并写 /dm_arm/joint_command。

这么拆是刻意避开 rclpy 已知缺陷：在同一节点里让 MultiThreadedExecutor + ActionServer
配合（轨迹执行回调阻塞 + 定时器读反馈）会触发 wait set 竞态导致段错误。拆开后每个
executor 单线程、无跨线程 wait set 复用，最稳。

接口
----
  action server  : follow_joint_trajectory   (control_msgs/action/FollowJointTrajectory)
  发布 /joint_states          (sensor_msgs/JointState)    当前 6 关节位置
  订阅 /dm_arm/joint_states   (sensor_msgs/JointState)    底层节点反馈
  发布 /dm_arm/joint_command  (sensor_msgs/JointState)    目标位置(弧度)

【重要】关节顺序固定为 openarm_joint1..6，对应 CAN ID 1..6。目标位置是电机原始
编码器空间的绝对弧度（和 /dm_arm/joint_states 输出的数值一致），不做 URDF 零位
对齐，请以当前状态为基准规划轨迹。

【运行】纯 DDS 客户端，但为与 root 身份的控制节点互通，需以 sudo 运行（见 run_bridge.sh）：
    sudo -E env HOME=/home/yang /usr/bin/python3 arm_trajectory_bridge.py
"""
from __future__ import annotations

import sys
import threading
import time

try:
    import rclpy  # noqa: F401
except ImportError:
    sys.path.append("/home/yang/miniconda3/lib/python3.14/site-packages")  # noqa: E402
    import rclpy  # noqa: E402

from rclpy.node import Node  # noqa: E402
from rclpy.action import ActionServer, GoalResponse, CancelResponse  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from control_msgs.action import FollowJointTrajectory  # noqa: E402

# 关节顺序固定 = CAN ID 1..6
JOINT_NAMES = [f"openarm_joint{i}" for i in range(1, 7)]
N_JOINTS = len(JOINT_NAMES)

# 线程安全共享区：存放最新的底层反馈，供两个节点传递
class SharedLatest:
    def __init__(self):
        self._lock = threading.Lock()
        self._pos = [0.0] * N_JOINTS
        self._got = False

    def update(self, positions):
        with self._lock:
            self._pos = [float(x) for x in positions]
            self._got = True

    def cur_pos(self):
        with self._lock:
            return list(self._pos) if self._got else [0.0] * N_JOINTS


class StateForwarder(Node):
    """订阅底层反馈 -> 共享区 + 发布标准 /joint_states。单独一节点，保持实时。"""
    def __init__(self, shared: SharedLatest):
        super().__init__("arm_state_forwarder")
        self._shared = shared
        self._state_pub = self.create_publisher(JointState, "joint_states", 10)
        self.create_subscription(JointState, "dm_arm/joint_states",
                                 self._on_state, 10)

    def _on_state(self, msg: JointState):
        self._shared.update(msg.position[:N_JOINTS])
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = list(JOINT_NAMES)
        out.position = list(msg.position[:N_JOINTS])
        out.velocity = list(msg.velocity[:N_JOINTS]) if len(msg.velocity) >= N_JOINTS else []
        out.effort = list(msg.effort[:N_JOINTS]) if len(msg.effort) >= N_JOINTS else []
        self._state_pub.publish(out)


class TrajectoryServer(Node):
    """ActionServer：轨迹插值 -> 写 /dm_arm/joint_command。独立 executor，可阻塞。"""
    def __init__(self, shared: SharedLatest):
        super().__init__("arm_trajectory_server")
        self._shared = shared
        self._cmd_pub = self.create_publisher(JointState, "dm_arm/joint_command", 10)

        self._as = ActionServer(
            self, FollowJointTrajectory, "follow_joint_trajectory",
            execute_callback=self._execute,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb)

        self._traj = None        # (joint_order 列表, points 列表)
        self._traj_start = None  # time.monotonic()
        self._cancel_evt = threading.Event()

        self.get_logger().info(
            f"桥接就绪: 轨迹 action=follow_joint_trajectory, 关节={JOINT_NAMES}")

    # ---------- Action 回调 ----------
    def _goal_cb(self, goal_request):
        # goal 的关节名在 goal.trajectory.joint_names 里（Goal 顶层没有该属性）
        names = goal_request.trajectory.joint_names
        if not names:
            self.get_logger().warn("拒绝空关节名的目标")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_cb(self, goal_handle):
        self.get_logger().info("收到取消请求，正在中断轨迹")
        self._cancel_evt.set()
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle):
        req = goal_handle.request
        joints = list(req.trajectory.joint_names)
        order = []
        for jn in joints:
            if jn not in JOINT_NAMES:
                goal_handle.abort()
                res = FollowJointTrajectory.Result()
                res.error_code = FollowJointTrajectory.Result.INVALID_GOALS
                res.error_string = f"未知关节名: {jn}"
                return res
            order.append(JOINT_NAMES.index(jn))

        if not req.trajectory.points:
            goal_handle.abort()
            res = FollowJointTrajectory.Result()
            res.error_code = FollowJointTrajectory.Result.INVALID_GOALS
            res.error_string = "轨迹为空"
            return res

        self._traj = (list(order), list(req.trajectory.points))
        self._traj_start = None
        self._cancel_evt.clear()
        self.get_logger().info(
            f"收到轨迹: {len(req.trajectory.points)} 点, 关节={joints}")

        # 阻塞式轨迹回放：按相对时钟线性插值并下发
        play_start = time.monotonic()
        while rclpy.ok():
            if self._cancel_evt.is_set():
                self._traj = None
                goal_handle.canceled()
                res = FollowJointTrajectory.Result()
                res.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                res.error_string = "已取消"
                return res
            elapsed = time.monotonic() - play_start
            if self._step(elapsed):
                break
            time.sleep(0.02)

        self._traj = None
        goal_handle.succeed()
        res = FollowJointTrajectory.Result()
        res.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        res.error_string = ""
        self.get_logger().info("轨迹完成")
        return res

    def _point_pos(self, p, order):
        out = [0.0] * N_JOINTS
        for k, idx in enumerate(order):
            if k < len(p.positions):
                out[idx] = float(p.positions[k])
        return out

    @staticmethod
    def _pt_t(p):
        tfs = p.time_from_start
        return tfs.sec + tfs.nanosec * 1e-9

    def _step(self, elapsed):
        """按 elapsed 线性插值当前段并下发；返回 True 表示轨迹已播完。"""
        order, points = self._traj
        total = self._pt_t(points[-1])
        if elapsed >= total:
            self._send(self._point_pos(points[-1], order))
            return True
        for i in range(len(points) - 1):
            t0, t1 = self._pt_t(points[i]), self._pt_t(points[i + 1])
            if t0 <= elapsed <= t1:
                a = (elapsed - t0) / (t1 - t0) if t1 > t0 else 0.0
                p0 = self._point_pos(points[i], order)
                p1 = self._point_pos(points[i + 1], order)
                self._send([p0[j] + (p1[j] - p0[j]) * a for j in range(N_JOINTS)])
                return False
        # elapsed 落在第一个点之前
        self._send(self._point_pos(points[0], order))
        return False

    def _send(self, positions):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINT_NAMES)
        msg.position = positions
        self._cmd_pub.publish(msg)


def main():
    import argparse
    p = argparse.ArgumentParser(description="OpenArm 轨迹桥接节点")
    p.parse_args()

    rclpy.init()
    shared = SharedLatest()
    fwd = StateForwarder(shared)
    srv = TrajectoryServer(shared)

    from rclpy.executors import SingleThreadedExecutor
    ex_fwd = SingleThreadedExecutor()
    ex_srv = SingleThreadedExecutor()
    ex_fwd.add_node(fwd)
    ex_srv.add_node(srv)

    t = threading.Thread(target=ex_srv.spin, daemon=True)
    t.start()

    try:
        ex_fwd.spin()
    except KeyboardInterrupt:
        pass
    finally:
        ex_fwd.shutdown()
        ex_srv.shutdown()
        fwd.destroy_node()
        srv.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()