#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真机 control 层 launch(只起 robot_state_publisher + ros2_control + 控制器)
==========================================================================
执行本 launch = 电机使能并锁定当前位形(不做任何大动作)。
MoveIt/RViz 不在此处启动——首跑流程先单独验证 control 层读数与锁位,
确认无误后另开 right_real_moveit.launch.py(见 rl_track/README.md §使用)。

用法: ros2 launch right_real_control.launch.py
"""

import os

from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node

_HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(_HERE, "openarm_right_real.urdf")
CONTROLLERS = os.path.join(_HERE, "ros2_controllers_real.yaml")


def generate_launch_description():
    with open(URDF, encoding="utf-8") as f:
        urdf = f.read()

    # rsp: 发布 URDF TF;  control: 加载 ZeroOffsetHW 插件——on_activate 会
    # 使能全部电机并锁定当前位形(设计行为, 不会复零位/大动作)
    rsp = Node(
        package="robot_state_publisher", executable="robot_state_publisher",
        output="screen", parameters=[{"robot_description": urdf}])
    control = Node(
        package="controller_manager", executable="ros2_control_node",
        output="screen",
        parameters=[{"robot_description": urdf}, CONTROLLERS])
    # 先广播状态, 再起控制器; 控制器名 right_joint_trajectory_controller
    # 的 type 由 ros2_controllers_real.yaml 决定——2026-09-14 起是
    # OpenArmMpcController(MPC 插件接管 JTC 槽位), 起来后即占用命令
    # 接口并保持锁位; 日志会打印 "OpenArmMpcController"。
    jsb = Node(
        package="controller_manager", executable="spawner",
        arguments=["joint_state_broadcaster", "-c", "/controller_manager"])
    jtc = Node(
        package="controller_manager", executable="spawner",
        arguments=["right_joint_trajectory_controller", "-c", "/controller_manager"])

    return LaunchDescription([
        rsp,
        control,
        TimerAction(period=2.0, actions=[jsb]),
        TimerAction(period=4.0, actions=[jtc]),
    ])
