#!/bin/bash
# MoveIt 之字形走线执行（官方 openarm_bimanual_moveit_config 方案）
# 前置(终端1): 官方 demo 已启动（注意 arm_type 必须指到 v1.0, 默认是 v2.0!）
#   ros2 launch openarm_bimanual_moveit_config demo.launch.py \
#       arm_type:=openarm_v1.0 use_fake_hardware:=true
# 本脚本(终端2): 执行之字形
set -e
source /opt/ros/jazzy/setup.bash
source "$HOME/桌面/openarm/openarm_ros2_ws/install/setup.bash"
exec /usr/bin/python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/zigzag_moveit.py" "$@"
