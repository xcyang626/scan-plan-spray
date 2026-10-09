#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
场景节点（阶段2）: RViz 场景 + 关节控制（zero 软限位）
======================================================
职责:
  1. 加载实机 zero 文件（zero_bridge 换算到 URDF 系），提供关节软限位；
  2. control_mode:
     - "static"  发布初始位形（全零/标定手摆位形），供场景确认；
     - "sliders" 启动 tkinter 滑条面板（限位 = zero 文件软限位），
       实时发布 /urdf_joint_states 控制 RViz 中的研究臂（openarm_right_*）；
  3. 场景元素: 墙体/TCP 工作平面由 URDF link 渲染（launch 注入）；
     箭头/文字/TCP 球标走 /sim/marker（TCP 球随当前位形实时移动）。

无 CAN、无物理引擎，纯运动学显示。
"""

import math
import os
import threading

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray

from .kinematics import UrdfChain
from .zero_bridge import ZeroBridge, load_zero_file

# 仓库根（scripts→openarm_sim→仿真→根）; realpath 兼容 symlink-install
_HERE = os.path.dirname(os.path.realpath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_URDF_SRC = os.path.abspath(os.path.join(_HERE, "..", "config", "v1.urdf"))

FINGER_JOINTS = ["openarm_right_finger_joint1", "openarm_right_finger_joint2"]


class SceneNode(Node):
    def __init__(self):
        super().__init__("scene_node")
        self.declare_parameter("zero_file_path", "")
        self.declare_parameter("urdf_path", _URDF_SRC)
        self.declare_parameter("initial_pose", "urdf_zero")   # urdf_zero / zero_file
        self.declare_parameter("control_mode", "static")      # static / sliders
        self.declare_parameter("publish_js", True)            # 是否发布关节状态
        self.declare_parameter("wall_distance_m", 2.00)       # 墙: 基座前方 2m
        self.declare_parameter("tcp_plane_m", 0.30)           # TCP 工作平面: 30cm
        self.declare_parameter("tcp_offset_m", 0.12)          # link7 原点→指尖

        zero_path = self.get_parameter("zero_file_path").value
        if not zero_path:
            zero_path = os.path.join(_REPO_ROOT, "标定", "zero")
        urdf_path = self.get_parameter("urdf_path").value
        self.wall_d = float(self.get_parameter("wall_distance_m").value)
        self.plane_d = float(self.get_parameter("tcp_plane_m").value)
        self.tcp_off = float(self.get_parameter("tcp_offset_m").value)
        self.control_mode = str(self.get_parameter("control_mode").value).lower()
        init_pose = str(self.get_parameter("initial_pose").value).lower()

        # ---------- 数据加载 ----------
        self.bridge = ZeroBridge(load_zero_file(zero_path))
        self.chain = UrdfChain(urdf_path)
        # 初始位形: 零位(全 0, 手臂自然下垂) 或 zero 文件标定初始位形
        if init_pose == "zero_file":
            self.init_q = list(self.bridge.urdf_home_rad[:7])
        else:
            self.init_q = [0.0] * 7
            init_pose = "urdf_zero"
        # 夹爪开合: zero 文件夹爪零位(电机系) → finger 平移
        zero_raw = load_zero_file(zero_path)
        self.finger_open = self.bridge.grip_to_finger_m(
            zero_raw["zero_position_rad"][7])

        # ---------- 当前控制状态（滑条模式由 GUI 线程写入，电机系 rad）----------
        self._lock = threading.Lock()
        self.current_q_motor = [self.bridge.urdf_to_motor(0.0, i)
                                for i in range(7)]
        self.current_finger = self.finger_open
        # 滑条限位（度）: zero 文件原值（电机系），与标定文件逐位一致
        self.slider_limits_deg = self.bridge.motor_limits_deg(7)
        self.slider_initial_deg = self.bridge.hanging_motor_deg()
        self.publish_js_flag = bool(self.get_parameter("publish_js").value)

        # ---------- FK 关键信息 ----------
        R, p = self.chain.fk(self.init_q, self.tcp_off)
        axis = R @ np.array([0.0, 0.0, 1.0])
        self.get_logger().info(
            "研究臂: URDF %s, 初始位形=%s: TCP(=%.2fm) 位置 [%.3f, %.3f, %.3f] m"
            % (self.bridge.side, init_pose, self.tcp_off, *p))
        self.get_logger().info(
            "墙体: body +x 水平距离 %.2f m 竖直平面(半透明); "
            "TCP 工作平面: %.2f m(与墙平行)" % (self.wall_d, self.plane_d))
        self.get_logger().info("控制模式: %s" % self.control_mode)

        # ---------- ROS 接口 ----------
        self.pub_js = self.create_publisher(JointState, "/urdf_joint_states", 10)
        self.pub_mk = self.create_publisher(MarkerArray, "/sim/markers", 10)
        # 单个 Marker 话题: RViz 手动添加显示时更可靠（MarkerArray 配置易失）
        self.pub_single = self.create_publisher(Marker, "/sim/marker", 10)
        # js:=false 时由外部节点（如 joint_console）负责发布关节状态
        if self.publish_js_flag:
            # static: 10Hz 初始位形; sliders: 30Hz 跟随滑条
            js_period = 0.1 if self.control_mode == "static" else 0.033
            self.js_timer = self.create_timer(js_period, self.publish_js)
        self.mk_timer = self.create_timer(0.2, self.publish_markers)
        self.publish_markers()

    # ------------------------------------------------------------------

    def set_control_input(self, q_motor_rad, finger_m):
        """滑条面板回调（Tk 主线程调用）→ 更新当前控制状态（电机系）。"""
        with self._lock:
            self.current_q_motor = [float(v) for v in q_motor_rad]
            self.current_finger = float(finger_m)

    def get_current_urdf(self):
        """当前位形（URDF 系 7 关节 + finger m）。滑条模式跟随电机系状态换算，
        其余模式返回初始位形。"""
        if self.control_mode != "sliders":
            return list(self.init_q), self.finger_open
        with self._lock:
            qm = list(self.current_q_motor)
            finger = self.current_finger
        q_urdf = [self.bridge.motor_to_urdf(q, i) for i, q in enumerate(qm)]
        return q_urdf, finger

    def publish_js(self):
        """关节状态（7 关节 + 双 finger）→ RViz。"""
        if self.control_mode == "static":
            q_urdf, finger = list(self.init_q), self.finger_open
        else:
            q_urdf, finger = self.get_current_urdf()
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.chain.joint_names + FINGER_JOINTS
        msg.position = [float(v) for v in q_urdf] + [finger, finger]
        self.pub_js.publish(msg)

    def publish_markers(self):
        arr = MarkerArray()
        # 清除旧 marker（id 复用覆盖即可，这里用 DELETEALL 更稳）
        wipe = Marker()
        wipe.action = Marker.DELETEALL
        arr.markers.append(wipe)

        # 1. 正前方指示箭头: 原点 → 墙（墙/工作平面已由 URDF link 渲染）
        arrow = Marker()
        arrow.header.frame_id = "world"
        arrow.ns = "scene"
        arrow.id = 2
        arrow.type = Marker.ARROW
        p0, p1 = [0.0, 0.0, 0.02], [self.wall_d, 0.0, 0.02]
        arrow.points = [Point(x=p0[0], y=p0[1], z=p0[2]),
                        Point(x=p1[0], y=p1[1], z=p1[2])]
        arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.01, 0.03, 0.03
        arrow.color.r, arrow.color.a = 1.0, 0.9
        arr.markers.append(arrow)

        # 2. 墙面文字标注
        text = Marker()
        text.header.frame_id = "world"
        text.ns = "scene"
        text.id = 3
        text.type = Marker.TEXT_VIEW_FACING
        text.pose.position.x = self.wall_d
        text.pose.position.z = 1.55
        text.scale.z = 0.08
        text.color.r, text.color.g, text.color.b, text.color.a = 1.0, 1.0, 1.0, 0.9
        text.text = "WALL  x=%.2f m" % self.wall_d
        arr.markers.append(text)

        # 3. 工作平面文字标注
        text2 = Marker()
        text2.header.frame_id = "world"
        text2.ns = "scene"
        text2.id = 6
        text2.type = Marker.TEXT_VIEW_FACING
        text2.pose.position.x = self.plane_d
        text2.pose.position.z = 1.08
        text2.scale.z = 0.06
        text2.color.r, text2.color.g, text2.color.b, text2.color.a = \
            0.4, 0.7, 1.0, 0.9
        text2.text = "TCP plane  x=%.2f m" % self.plane_d
        arr.markers.append(text2)

        # 4. TCP 球标（随当前位形实时移动）
        q_urdf, _ = self.get_current_urdf()
        R, p = self.chain.fk(q_urdf, self.tcp_off)
        # TCP 读数（ROS 线程写字符串；Tk 面板 _poll_tcp 轮询 / 控制台实时打印）
        self.tcp_text = ("TCP: x=%+.3f  y=%+.3f  z=%+.3f m"
                         % (p[0], p[1], p[2]))
        if getattr(self, "_panel", None) is not None:
            self._panel.tcp_text = self.tcp_text
        ball = Marker()
        ball.header.frame_id = "world"
        ball.ns = "scene"
        ball.id = 4
        ball.type = Marker.SPHERE
        ball.pose.position.x, ball.pose.position.y, ball.pose.position.z = p
        ball.scale.x = ball.scale.y = ball.scale.z = 0.03
        ball.color.r, ball.color.g, ball.color.b, ball.color.a = 0.1, 0.9, 0.9, 0.95
        arr.markers.append(ball)

        # 双通道发布: MarkerArray + 逐个单 Marker（RViz 两种显示方式都可用）
        self.pub_mk.publish(arr)
        for m in arr.markers:
            if m.action == Marker.DELETEALL:
                continue
            self.pub_single.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = SceneNode()
    if node.control_mode == "sliders":
        # Tk 主线程 + rclpy 后台线程（Tk 不允许跨线程操作）
        from .joint_panel import JointSliderPanel
        spin_thread = threading.Thread(
            target=rclpy.spin, args=(node,), daemon=True)
        spin_thread.start()

        def on_change():
            node.set_control_input(node._panel.q_rad(),
                                   node._panel.finger_m())

        panel = JointSliderPanel(
            node.slider_limits_deg, on_change=on_change,
            initial_deg=node.slider_initial_deg)
        node._panel = panel
        print(">>> 滑条控制面板已打开（若被 RViz 遮挡请查看任务栏）")
        try:
            panel.run()
        finally:
            pass
    else:
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
    try:
        node.destroy_node()
        rclpy.shutdown()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
