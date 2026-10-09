#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""launch: openarm_calibration 标定节点（自动加载 config/calibration_params.yaml）。"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(get_package_share_directory(
        "openarm_calibration"), "config", "calibration_params.yaml")

    return LaunchDescription([
        # 参数声明: YAML 为主配置; allow_motion 是安全总闸(默认拒绝运动),
        # arm_side 决定电机表与 zero 产物文件(left/right 不互串)
        DeclareLaunchArgument(
            "params_file", default_value=default_params,
            description="标定参数 YAML 路径"),
        DeclareLaunchArgument(
            "allow_motion", default_value="false",
            description="安全总闸: false 时标定服务拒绝启动"),
        DeclareLaunchArgument(
            "arm_side", default_value="right",
            description="right / left"),
        # 标定节点本体: share 目录 YAML 打底, launch 参数逐项覆盖
        Node(
            package="openarm_calibration",
            executable="calibration_node",
            name="calibration_node",
            output="screen",
            parameters=[
                LaunchConfiguration("params_file"),
                {"allow_motion": LaunchConfiguration("allow_motion")},
                {"arm_side": LaunchConfiguration("arm_side")},
            ],
        ),
    ])
