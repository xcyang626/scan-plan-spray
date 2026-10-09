#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""launch: 之字形仿真场景（阶段2）。

墙体/TCP 工作平面以固定 link 注入 URDF（随 robot_state_publisher 渲染，
只要 RobotModel 可见就可见，不依赖 Marker 显示）。

参数:
  rviz:=false            不启动 RViz
  zero_file:=/abs/path   指定 zero 文件（缺省 仓库根/标定/zero）
  wall_distance_m:=2.0   墙: 基座前方水平距离
  tcp_plane_m:=0.5       TCP 工作平面: 距基座水平距离（与墙平行）
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# 注入 URDF 的场景元素模板（半透明材质, RViz RobotModel 按rgba渲染）
_WALL_TPL = (
    '<link name="scene_wall">'
    '<visual><geometry><box size="0.03 2.0 1.8"/></geometry>'
    '<material name="scene_wall_mat"><color rgba="0.85 0.85 0.9 0.55"/>'
    '</material></visual></link>'
    '<joint name="scene_wall_joint" type="fixed">'
    '<parent link="world"/><child link="scene_wall"/>'
    '<origin xyz="{x:.4f} 0 0.9" rpy="0 0 0"/></joint>')
_PLANE_TPL = (
    '<link name="scene_tcp_plane">'
    '<visual><geometry><box size="0.004 1.0 1.0"/></geometry>'
    '<material name="scene_plane_mat"><color rgba="0.2 0.5 1.0 0.25"/>'
    '</material></visual></link>'
    '<joint name="scene_tcp_plane_joint" type="fixed">'
    '<parent link="world"/><child link="scene_tcp_plane"/>'
    '<origin xyz="{x:.4f} 0 0.75" rpy="0 0 0"/></joint>')


def _inject_scene(urdf_text: str, wall_d: float, plane_d: float) -> str:
    """把墙与 TCP 工作平面作为固定 link 注入 URDF 根元素。"""
    inject = _WALL_TPL.format(x=wall_d + 0.015) + _PLANE_TPL.format(x=plane_d)
    return urdf_text.replace("</robot>", inject + "</robot>")


def _setup(context, *args, **kwargs):
    share = get_package_share_directory("openarm_sim")
    urdf_path = os.path.join(share, "config", "v1.urdf")
    rviz_cfg = os.path.join(share, "config", "zigzag_sim.rviz")
    wall_d = float(LaunchConfiguration("wall_distance_m").perform(context))
    plane_d = float(LaunchConfiguration("tcp_plane_m").perform(context))
    gui_raw = LaunchConfiguration("gui").perform(context).lower()
    if gui_raw in ("1", "true"):
        control_mode = "sliders"       # Tk 滑条面板
    elif gui_raw == "console":
        control_mode = "console"       # 终端交互控制台（独立进程，见 run_scene.sh）
    else:
        control_mode = "static"        # 无交互，仅初始位形
    # js:=false 时 scene_node 不发布关节状态（console 模式由 joint_console 发布）
    js = LaunchConfiguration("js").perform(context).lower() in ("1", "true")

    with open(urdf_path, "r", encoding="utf-8") as f:
        robot_description = _inject_scene(f.read(), wall_d, plane_d)

    return [
        # ---- 场景节点（初始位形 + TCP 球/箭头 Marker）----
        Node(
            package="openarm_sim",
            executable="scene_node",
            name="scene_node",
            output="screen",
            parameters=[
                {"urdf_path": urdf_path},
                {"zero_file_path": LaunchConfiguration("zero_file")},
                {"wall_distance_m": wall_d},
                {"tcp_plane_m": plane_d},
                {"control_mode": control_mode},
                {"publish_js": js},
            ],
        ),
        # ---- robot_state_publisher（含场景墙/工作平面）----
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}],
            remappings=[("joint_states", "/urdf_joint_states")],
        ),
        # ---- RViz2 ----
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            condition=IfCondition(LaunchConfiguration("rviz")),
            arguments=["-d", rviz_cfg],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("js", default_value="true"),
        DeclareLaunchArgument("zero_file", default_value=""),
        DeclareLaunchArgument("wall_distance_m", default_value="2.0"),
        DeclareLaunchArgument("tcp_plane_m", default_value="0.3"),
        OpaqueFunction(function=_setup),
    ])
