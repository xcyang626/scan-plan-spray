#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""零位(home)可视化: RViz 查看 zero 文件零位对应的机械臂位形
==========================================================
用途: 真机上电前, 先目视确认新坐标系下的 home 位形(≈垂直下垂)
以及各关节正方向。只用 robot_state_publisher + 静态 joint_states,
不加载硬件插件、不碰 CAN、不碰电机。

用法:
  source /opt/ros/jazzy/setup.bash
  ros2 launch (在本目录): ros2 launch view_home.launch.py
可选交互(拖滑杆验证关节方向, 若装了 joint_state_publisher_gui):
  先 Ctrl+C 本 launch, 再:
  ros2 run joint_state_publisher_gui joint_state_publisher_gui \
      --ros-args -p robot_description:="$(cat ~/桌面/openarm实验/real/left/config/openarm_left_real.urdf)"
  并保持 robot_state_publisher 运行(可单独 ros2 run 启动)。
"""

import os

import yaml
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

_HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_HERE, "openarm_left_real.urdf"),
          encoding="utf-8") as f:
    urdf = f.read()
_ = yaml  # home 值由 pub_home_js.py 自行读取


def generate_launch_description():
    # robot_state_publisher: 只加载 URDF 发布 TF(关节态由下面 pub 提供)
    rsp = Node(
        package="robot_state_publisher", executable="robot_state_publisher",
        parameters=[{"robot_description": urdf}], output="screen")
    # pub_home_js.py: 20Hz 发布 real_safety.yaml 里的 home 位形(不碰电机)
    pub = ExecuteProcess(
        cmd=["/usr/bin/python3", os.path.join(_HERE, "pub_home_js.py")],
        output="screen")
    rviz = Node(
        package="rviz2", executable="rviz2",
        arguments=["-d", os.path.join(_HERE, "view_home.rviz")],
        output="log")
    return LaunchDescription([rsp, pub, rviz])  # 三个进程同起同停
