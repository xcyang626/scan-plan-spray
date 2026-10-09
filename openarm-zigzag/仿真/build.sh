#!/bin/bash
# openarm_sim 编译脚本（兼容 conda 环境共存，约定同 openarm_calibration）
set -e

# 先 source ROS2 再开严格模式（官方 setup.bash 与 set -u 冲突）
source /opt/ros/jazzy/setup.bash
set -uo pipefail

SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SIM_DIR"

colcon build --packages-select openarm_sim --symlink-install \
    --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 "$@"

chmod +x openarm_sim/scripts/scene_node

echo "编译完成。启动: ./run_scene.sh"
