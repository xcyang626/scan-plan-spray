#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真机 MoveIt 层 launch(move_group + RViz + 安全监控)
====================================================
前置: right_real_control.launch.py 已在另一终端运行(control 层就绪)。
本 launch 不操作电机, 只提供规划/可视化/安全层。

安全监控 safety_watchdog.py 常驻(独立进程):
  - 软限位越界 / 关节速度异常 / 手动急停话题 /safety/estop
  - 触发即取消 JTC 当前目标, 并慢速回标定初始位形(不经过 MoveIt)

用法: ros2 launch right_real_moveit.launch.py
"""

import os
import sys

import yaml
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node

_HERE = os.path.dirname(os.path.abspath(__file__))
# 单臂 MoveIt 公共配置在 仿真/moveit_sim/right_config/
# 注意层级: real/right/config 到实验根目录要上三级(2026-09-09 修正,
# 原两级拼出 real/仿真/... 导致 FileNotFoundError)
_RIGHT_CFG = os.path.abspath(
    os.path.join(_HERE, "..", "..", "..", "仿真", "moveit_sim", "right_config"))

# jazzy 官方 move_group 实测 OMPL 管线参数(OMPL 管线与 仿真/moveit_sim/left_config/left_demo.launch.py 同源实测值)
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


def _load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_launch_description():
    with open(os.path.join(_HERE, "openarm_right_real.urdf"),
              encoding="utf-8") as f:
        urdf = f.read()
    with open(os.path.join(_RIGHT_CFG, "openarm_right.srdf"),
              encoding="utf-8") as f:
        srdf = f.read()
    kin = _load_yaml(os.path.join(_RIGHT_CFG, "kinematics.yaml"))
    moveit_ctrl = _load_yaml(os.path.join(_HERE, "moveit_controllers_real.yaml"))
    joint_limits = _load_yaml(os.path.join(_HERE, "joint_limits_real.yaml"))

    moveit_params = {
        "robot_description": urdf,
        "robot_description_semantic": srdf,
        "robot_description_kinematics": kin,
        "robot_description_planning": joint_limits,
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": OMPL_PARAMS,
        "moveit_manage_controllers": True,
        # 真机 MIT 跟踪存在 ~0.3-0.7° 稳态误差/噪声, 默认起始容差
        # 0.01rad 会让执行立刻 MOTION_PLAN_INVALIDATED_BY_ENV_CHANGE。
        # 0.0 = 关闭起始偏差监视(真机标准做法); 安全由 watchdog +
        # JTC 容差兜底。必须平铺顶层键(嵌套 dict 不生效, 见坑表)。
        "trajectory_execution.allowed_start_tolerance": 0.0,
        # 执行时长监视放宽(2026-09-04): 实测默认等效 1.0/0.0, JTC 真机
        # 跟踪在标称时长临界点被误杀(TIMED_OUT, 上层报为 env change)。
        # 二次放宽: 1.5x 生效但 JTC 实际耗时≈标称+滞后, 再加一倍余量。
        "trajectory_execution.allowed_execution_duration_scaling": 2.0,
        "trajectory_execution.allowed_goal_duration_margin": 2.0,
        **moveit_ctrl,
    }

    move_group = Node(
        package="moveit_ros_move_group", executable="move_group",
        output="screen", parameters=[moveit_params])
    rviz = Node(
        package="rviz2", executable="rviz2", name="rviz2", output="log",
        arguments=["-d", os.path.join(_RIGHT_CFG, "openarm_right.rviz")],
        parameters=[moveit_params])
    # 安全监控: 用系统 python3 起(脚本有 shebang, 亦可直接执行)
    watchdog = ExecuteProcess(
        cmd=["/usr/bin/python3",
             os.path.join(_HERE, "safety_watchdog.py"),
             "--safety-yaml", os.path.join(_HERE, "real_safety.yaml")],
        output="screen")

    return LaunchDescription([
        move_group,
        rviz,
        # watchdog 延后起: 等 control 层状态广播稳定后再开始判定
        TimerAction(period=6.0, actions=[watchdog]),
    ])
