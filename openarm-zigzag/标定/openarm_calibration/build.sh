#!/bin/bash
# OpenArm 零点标定包编译脚本（兼容 conda 环境共存）
# 原理: ROS2 Jazzy 依赖系统 Python3.12, conda python3.14 会破坏 ABI 匹配,
#       因此显式指定 CMake 使用 /usr/bin/python3, 无需退出 conda。
set -e

# 先 source ROS2 再开严格模式（官方 setup.bash 与 set -u 冲突）
source /opt/ros/jazzy/setup.bash
set -uo pipefail

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS_DIR"

colcon build --packages-select openarm_calibration --symlink-install \
    --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 "$@"

# 保证节点脚本有执行权限（symlink-install 直接引用源文件权限）
chmod +x openarm_calibration/scripts/calibration_node \
         openarm_calibration/run_calibration.sh

echo "编译完成。启动: ./openarm_calibration/run_calibration.sh [true|false] [right|left]"
