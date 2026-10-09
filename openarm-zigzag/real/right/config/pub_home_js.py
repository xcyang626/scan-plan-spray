#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 real_safety.yaml 的 home_rad 以 20Hz 发布 /joint_states。

仅供零位可视化(view_home.launch.py)使用: 只读 YAML 并发话题,
不加载硬件、不碰电机。
"""

import math
import os
import time

import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import JointState

_HERE = os.path.dirname(os.path.abspath(__file__))
JOINTS = ["openarm_right_joint%d" % i for i in range(1, 8)]


def main():
    rclpy.init()
    # 纯发布节点: 不需要执行器(无订阅/服务), 直接定时发即可
    node = Node("home_js_publisher")
    with open(os.path.join(_HERE, "real_safety.yaml"), encoding="utf-8") as f:
        home = list(yaml.safe_load(f)["home_rad"])[:7]
    # RViz 的 TF 插值需要连续话题流, 因此以 20Hz 循环发布同一 home 位形
    pub = node.create_publisher(JointState, "/joint_states", 10)
    print("发布 home 位形(度): "
          + " ".join("J%d=%.2f" % (i + 1, v * 180.0 / math.pi)
                     for i, v in enumerate(home)), flush=True)
    msg = JointState()
    msg.name = JOINTS  # 关节名与 URDF 一致(left/right 由本目录决定)
    try:
        while rclpy.ok():
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.position = home
            pub.publish(msg)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
