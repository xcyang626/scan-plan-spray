#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
单臂(左臂) MoveIt demo launch
=============================
目的: 官方 demo.launch.py 是 bimanual(双臂), RViz 里出现两条臂 + 目标 ghost
+ 规划动画 ghost, 视觉混乱。本 launch:
  - URDF 用单臂版 openarm_left_moveit.urdf(基座位姿与官方双臂一致,
    hand_tcp=link7+0.186 与官方 TCP 语义一致, 含 fake ros2_control 块)
  - SRDF 只定义 left_arm 组
  - OMPL 管线参数照抄官方 jazzy 实测值(AddTimeOptimalParameterization 等)
  - RViz 预设关掉 Query Goal State / Loop Animation, 只显示一条实体臂

用法: ros2 launch left_demo.launch.py
"""

import os

import yaml
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node

_HERE = os.path.dirname(os.path.abspath(__file__))

# jazzy 官方 move_group 实测 OMPL 管线参数(2025-09 dump)
OMPL_PARAMS = {
    "planning_plugins": ["ompl_interface/OMPLPlanner"],
    "request_adapters": [
        "default_planning_request_adapters/ResolveConstraintFrames",
        "default_planning_request_adapters/ValidateWorkspaceBounds",
        "default_planning_request_adapters/CheckStartStateBounds",
        "default_planning_request_adapters/CheckStartStateCollision",
    ],
    "response_adapters": [
        "default_planning_response_adapters/AddTimeOptimalParameterization",
        "default_planning_response_adapters/ValidateSolution",
        "default_planning_response_adapters/DisplayMotionPath",
    ],
}


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_launch_description():
    urdf = _read(os.path.join(_HERE, "openarm_left_moveit.urdf"))
    srdf = _read(os.path.join(_HERE, "openarm_left.srdf"))
    kin = _load_yaml(os.path.join(_HERE, "kinematics.yaml"))
    joint_limits = _load_yaml(os.path.join(_HERE, "joint_limits.yaml"))
    moveit_ctrl = _load_yaml(os.path.join(_HERE, "moveit_controllers.yaml"))

    moveit_params = {
        "robot_description": urdf,
        "robot_description_semantic": srdf,
        "robot_description_kinematics": kin,
        "robot_description_planning": joint_limits,
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": OMPL_PARAMS,
        # trajectory_execution 内容平铺到顶层(move_group 按顶层键读,
        # 嵌套在 trajectory_execution 键下会读到 "No controller_names")
        "moveit_manage_controllers": True,
        **moveit_ctrl,
    }

    rsp = Node(
        package="robot_state_publisher", executable="robot_state_publisher",
        output="screen", parameters=[{"robot_description": urdf}])
    control = Node(
        package="controller_manager", executable="ros2_control_node",
        output="both",
        parameters=[{"robot_description": urdf},
                    os.path.join(_HERE, "ros2_controllers.yaml")])
    # 顺序: jsb(2s) → JTC(3s) → move_group(1s) → RViz(4s)
    jsb = Node(
        package="controller_manager", executable="spawner",
        arguments=["joint_state_broadcaster", "-c", "/controller_manager"])
    jtc = Node(
        package="controller_manager", executable="spawner",
        arguments=["left_joint_trajectory_controller", "-c", "/controller_manager"])
    move_group = Node(
        package="moveit_ros_move_group", executable="move_group",
        output="screen", parameters=[moveit_params])
    rviz = Node(
        package="rviz2", executable="rviz2", name="rviz2", output="log",
        arguments=["-d", os.path.join(_HERE, "openarm_left.rviz")],
        parameters=[moveit_params])

    return LaunchDescription([
        rsp,
        control,
        TimerAction(period=2.0, actions=[jsb]),
        TimerAction(period=3.0, actions=[jtc]),
        TimerAction(period=1.0, actions=[move_group]),
        TimerAction(period=4.0, actions=[rviz]),
    ])
