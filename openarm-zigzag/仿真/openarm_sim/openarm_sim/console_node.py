#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
终端控制台节点（独立进程，真实终端 stdin）
==========================================
与 scene_node 分离运行: launch 只提供 RViz 场景（不发布关节状态），
本进程通过 ros2 run 直接跑在用户终端里，保证 input() 能读到键盘。

坐标系约定: 用户操作数值与限位 = zero 文件原值（电机反馈系，度），
内部经 zero_bridge 换算到 URDF 系发布 /urdf_joint_states。
"""

import os
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from .kinematics import UrdfChain
from .term_console import TerminalConsole
from .zero_bridge import ZeroBridge, load_zero_file

_HERE = os.path.dirname(os.path.realpath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_URDF_SRC = os.path.abspath(os.path.join(_HERE, "..", "config", "v1.urdf"))

FINGER_JOINTS = ["openarm_right_finger_joint1", "openarm_right_finger_joint2"]


class ConsoleNode(Node):
    def __init__(self):
        super().__init__("joint_console")
        # 三个可覆盖参数: 零位文件(缺省=左臂 标定/zero)、URDF、TCP 指尖偏移
        self.declare_parameter("zero_file_path", "")
        self.declare_parameter("urdf_path", _URDF_SRC)
        self.declare_parameter("tcp_offset_m", 0.12)

        zero_path = self.get_parameter("zero_file_path").value
        if not zero_path:
            zero_path = os.path.join(_REPO_ROOT, "标定", "zero")
        self.tcp_off = float(self.get_parameter("tcp_offset_m").value)

        # bridge: 电机系↔URDF系 换算;  chain: URDF 正运动学(FK)
        self.bridge = ZeroBridge(load_zero_file(zero_path))
        self.chain = UrdfChain(self.get_parameter("urdf_path").value)
        self.finger_open = self.bridge.grip_to_finger_m(
            self.bridge.zero_position_rad[7])

        # 当前控制状态（电机系，rad）
        self._lock = threading.Lock()
        self.current_q_motor = [v * 3.141592653589793 / 180.0
                                for v in self.bridge.hanging_motor_deg()]
        self.current_finger = self.finger_open

        self.pub_js = self.create_publisher(JointState,
                                            "/urdf_joint_states", 10)
        self.js_timer = self.create_timer(0.033, self.publish_js)
        self.tcp_text = "TCP: --"
        self.get_logger().info(
            "终端控制台就绪: 操作数值 = zero 文件原值（电机系），"
            "TCP 指尖偏移 %.2f m" % self.tcp_off)

    # ------------------------------------------------------------------

    def set_control_input(self, q_motor_rad, finger_m):
        with self._lock:
            self.current_q_motor = [float(v) for v in q_motor_rad]
            self.current_finger = float(finger_m)

    def publish_js(self):
        with self._lock:
            qm = list(self.current_q_motor)
            finger = self.current_finger
        q_urdf = [self.bridge.motor_to_urdf(q, i) for i, q in enumerate(qm)]
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.chain.joint_names + FINGER_JOINTS
        msg.position = q_urdf + [finger, finger]
        self.pub_js.publish(msg)
        # TCP 读数（终端实时打印用）
        _, p = self.chain.fk(q_urdf, self.tcp_off)
        self.tcp_text = ("TCP: x=%+.3f  y=%+.3f  z=%+.3f m"
                         % (p[0], p[1], p[2]))


def main(args=None):
    rclpy.init(args=args)
    node = ConsoleNode()
    # ROS 回调放后台守护线程, 主线程留给终端 input() 交互循环——
    # 若用 rclpy.spin 阻塞主线程, 终端将读不到键盘
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,),
                                   daemon=True)
    spin_thread.start()

    def on_change():
        node.set_control_input(node._console.q_rad(),
                               node._console.finger_m())

    cons = TerminalConsole(
        node.bridge.motor_limits_deg(7), on_change=on_change,
        get_tcp_text=lambda: node.tcp_text,
        initial_deg=node.bridge.hanging_motor_deg(),
        home_deg=[v * 180.0 / 3.141592653589793
                  for v in node.bridge.zero_position_rad[:7]])
    node._console = cons
    try:
        cons.run()
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
