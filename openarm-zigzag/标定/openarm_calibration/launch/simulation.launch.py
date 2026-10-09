#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""launch: 仿真标定（vcan0 + 电机仿真器 + 标定节点，可选 URDF/RViz 可视化）。

rviz:=true 时额外启动:
  rviz_bridge            标定 joint_states → URDF 关节名
  robot_state_publisher  加载官方 v1.urdf（双臂，右臂随动）
  rviz2                  3D 实时显示标定运动
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("openarm_calibration")
    sim_params = os.path.join(share, "config", "sim_params.yaml")
    urdf_path = os.path.join(share, "config", "v1.urdf")
    rviz_cfg = os.path.join(share, "config", "rviz_calib.rviz")

    # URDF 内容作为 robot_description
    with open(urdf_path, "r", encoding="utf-8") as f:
        robot_description = f.read()

    use_rviz = LaunchConfiguration("rviz")

    return LaunchDescription([
        DeclareLaunchArgument("allow_motion", default_value="false"),
        DeclareLaunchArgument("arm_side", default_value="right"),
        DeclareLaunchArgument("rviz", default_value="false"),

        # ---- 电机仿真器（vcan0 上模拟 8 台达妙电机）----
        Node(
            package="openarm_calibration",
            executable="dm_motor_sim.py",
            name="dm_motor_sim",
            output="screen",
        ),
        # ---- 标定节点 ----
        Node(
            package="openarm_calibration",
            executable="calibration_node",
            name="calibration_node",
            output="screen",
            parameters=[
                sim_params,
                {"allow_motion": LaunchConfiguration("allow_motion")},
                {"arm_side": LaunchConfiguration("arm_side")},
            ],
        ),
        # ---- RViz 桥接（条件启动）----
        Node(
            package="openarm_calibration",
            executable="rviz_bridge",
            name="rviz_bridge",
            output="screen",
            condition=IfCondition(use_rviz),
            parameters=[{"arm_side": LaunchConfiguration("arm_side")}],
        ),
        # ---- robot_state_publisher（条件启动）----
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            condition=IfCondition(use_rviz),
            parameters=[{"robot_description": robot_description}],
            remappings=[("joint_states", "/urdf_joint_states")],
        ),
        # ---- RViz2（条件启动）----
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            condition=IfCondition(use_rviz),
            arguments=["-d", rviz_cfg],
        ),
    ])
