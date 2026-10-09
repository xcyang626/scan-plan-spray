#!/bin/bash
# 阶段2 场景一键启动: 初始位形/滑条面板/终端控制台 + 墙体场景 + RViz
# 用法:
#   ./run_scene.sh                    # RViz + Tk 滑条面板
#   ./run_scene.sh true console       # RViz + 终端交互控制台（终端内直接操作）
#   ./run_scene.sh true false         # 仅 RViz + 初始位形（无交互）
set -e

source /opt/ros/jazzy/setup.bash
# 官方 openarm_description 包（解析 URDF 里的 package:// mesh 路径）
if [ -f "$HOME/桌面/openarm/openarm_ros2_ws/install/setup.bash" ]; then
    source "$HOME/桌面/openarm/openarm_ros2_ws/install/setup.bash"
fi
SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SIM_DIR/install/setup.bash"
set -uo pipefail

# 规避 ~/.ros/log 写权限问题（沙箱/受限环境），ROS2 launch 日志目录指到 /tmp
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/openarm_sim_ros_log}"
mkdir -p "$ROS_LOG_DIR"

RVIZ="${1:-true}"
GUI="${2:-true}"

# console 模式: launch 后台提供场景（不发布关节状态），
# 终端控制台用 ros2 run 直接跑在前台——保证 stdin（键盘输入）可用
if [ "$GUI" = "console" ]; then
    ros2 launch openarm_sim zigzag_sim.launch.py \
        rviz:="$RVIZ" gui:=false js:=false "$@" &
    LAUNCH_PID=$!
    sleep 3
    ros2 run openarm_sim joint_console
    RC=$?
    kill $LAUNCH_PID 2>/dev/null
    exit $RC
fi

exec ros2 launch openarm_sim zigzag_sim.launch.py rviz:="$RVIZ" gui:="$GUI"
