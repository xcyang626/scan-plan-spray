#!/bin/bash
# ros2 CLI 快捷包装（自动处理 sudo 环境变量剥离问题）
# 用法: ./ros2.sh topic echo /calibration_node/joint_states --once
#       ./ros2.sh service call /calibration_node/start std_srvs/srv/Trigger
source /opt/ros/jazzy/setup.bash
WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$WS_DIR/install/setup.bash" ] && source "$WS_DIR/install/setup.bash"

exec sudo -E env HOME="$HOME" PATH="$PATH" \
    PYTHONPATH="$PYTHONPATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    /opt/ros/jazzy/bin/ros2 "$@"
