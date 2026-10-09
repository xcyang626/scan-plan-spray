#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实时镜像视图(仿真层): RViz 模型由控制层 /joint_states 实时驱动
================================================================
与 view_home.launch.py(静态 home)的区别: 不发静态关节态, 模型完全
跟随终端 A 控制层的 /joint_states 实时刷新 —— 显示的是"URDF 系认为
的臂"(= 命令位形)。实物与它的偏差(方向/幅度/零位)= 实机与仿真的
差别, 就是微动阶段要观察的对象。

只订阅 /joint_states + 发布 TF, 不加载硬件插件、不碰 CAN、不碰电机,
可与 control 层同机共存。

多终端布局(阶段 F):
  终端 A  控制层: ros2 launch right_real_control.launch.py
  终端 B  仿真层: ros2 launch view_live.launch.py      ← 本文件
  终端 B' 监控层: check_states.py --watch
  终端 C' 命令层: move_j1.py ~ move_j7.py(逐关节, 动→保持→Ctrl+C 自动回 home)

用法:
  source /opt/ros/jazzy/setup.bash
  source ~/桌面/openarm/openarm_ros2_ws/install/setup.bash
  source ~/桌面/openarm实验/real/common/ws/install/setup.bash
  ros2 launch ~/桌面/openarm实验/real/right/config/view_live.launch.py
"""

import os

from launch import LaunchDescription
from launch_ros.actions import Node

_HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_HERE, "openarm_right_real.urdf"),
          encoding="utf-8") as f:
    urdf = f.read()


def generate_launch_description():
    # robot_state_publisher: 由 /joint_states 实时驱动 URDF TF(镜像控制层位形)
    rsp = Node(
        package="robot_state_publisher", executable="robot_state_publisher",
        parameters=[{"robot_description": urdf}], output="screen",
        remappings=[("joint_states", "/joint_states")])
    # RViz 复用静态视图配置, 仅作为显示端
    rviz = Node(
        package="rviz2", executable="rviz2",
        arguments=["-d", os.path.join(_HERE, "view_home.rviz")],
        output="log")
    return LaunchDescription([rsp, rviz])
