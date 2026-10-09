#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RViz 桥接节点：把标定节点的 joint_states 转成 URDF 关节名供可视化
=================================================================
订阅 /calibration_node/joint_states (openarm_joint1..7 + openarm_gripper)
按 arm_side 参数映射到 openarm_left_* 或 openarm_right_* URDF 关节。

映射依据官方 v1.urdf 限位与电机限位对照:
  J1 电机 ±80°  ⊂ URDF 限位 → 直映射
  J2 电机 [-115°,0°] → URDF = -q（电机+10° = URDF -10° 下限，完全吻合）
  J3-J7 直映射（限位区间一致）
  夹爪电机角 → finger 平移 0~0.044m 线性映射
"""
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

GRIP_MIN = -60.0 * math.pi / 180.0   # 官方 MECH_LIM_V1 夹爪负限位
GRIP_MAX = 0.0                       # 官方夹爪正限位
FINGER_MAX = 0.044                   # URDF finger 平移行程 (m)


def build_mapping(side: str):
    """按臂侧生成关节映射表: (电机关节名, URDF 关节名, 符号, 偏移 rad)。
    偏移由官方 MECH_LIM_V1(出厂编码) 与 URDF 限位满程对满程唯一确定:
      left:  J1 [-80,200]→URDF[-200,80]: q-120°   J2 [-100,100]→URDF[-190,10]: q-90°
      right: J1 [-80,200]→URDF[-80,200]: q        J2 [-100,100]→URDF[-10,190]: q+90°
      J3-J7 官方限位与 URDF 完全一致，直映射。
    """
    p = "openarm_%s_" % side
    d2r = math.pi / 180.0
    if side == "left":
        table = [
            ("openarm_joint1", p + "joint1", 1.0, -120.0 * d2r),
            ("openarm_joint2", p + "joint2", 1.0, -90.0 * d2r),
            ("openarm_joint3", p + "joint3", 1.0, 0.0),
            ("openarm_joint4", p + "joint4", 1.0, 0.0),
            ("openarm_joint5", p + "joint5", 1.0, 0.0),
            ("openarm_joint6", p + "joint6", 1.0, 0.0),
            ("openarm_joint7", p + "joint7", 1.0, 0.0),
        ]
    else:   # right
        table = [
            ("openarm_joint1", p + "joint1", 1.0, 0.0),
            ("openarm_joint2", p + "joint2", 1.0, 90.0 * d2r),
            ("openarm_joint3", p + "joint3", 1.0, 0.0),
            ("openarm_joint4", p + "joint4", 1.0, 0.0),
            ("openarm_joint5", p + "joint5", 1.0, 0.0),
            ("openarm_joint6", p + "joint6", 1.0, 0.0),
            ("openarm_joint7", p + "joint7", 1.0, 0.0),
        ]
    fingers = (p + "finger_joint1", p + "finger_joint2")
    return table, fingers


class RvizBridge(Node):
    def __init__(self):
        super().__init__("rviz_bridge")
        self.declare_parameter("arm_side", "left")
        side = self.get_parameter("arm_side").value.lower()
        self.mapping, self.fingers = build_mapping(side)
        self.sub = self.create_subscription(
            JointState, "/calibration_node/joint_states",
            self.on_joint_states, 10)
        self.pub = self.create_publisher(JointState, "/urdf_joint_states", 10)
        self.get_logger().info(
            "RViz 桥接就绪: side=%s, %d 关节 + 夹爪" % (side, len(self.mapping)))

    def on_joint_states(self, msg: JointState):
        src = dict(zip(msg.name, msg.position))
        names = []
        positions = []
        for motor_name, urdf_name, sign, offset in self.mapping:
            if motor_name in src:
                names.append(urdf_name)
                positions.append(sign * src[motor_name] + offset)
        # 夹爪: 电机角(官方限位 [-60°,0°]) → 双 finger 平移 0~0.044m
        if "openarm_gripper" in src:
            q = src["openarm_gripper"]
            ratio = max(0.0, min(1.0,
                       (q - GRIP_MIN) / (GRIP_MAX - GRIP_MIN)))
            opening = ratio * FINGER_MAX
            names += list(self.fingers)
            positions += [opening, opening]
        if names:
            out = JointState()
            out.header.stamp = self.get_clock().now().to_msg()
            out.name = names
            out.position = [float(p) for p in positions]
            self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = RvizBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
