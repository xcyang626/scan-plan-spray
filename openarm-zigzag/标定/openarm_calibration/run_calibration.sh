#!/bin/bash
# OpenArm 零点标定启动脚本（root 运行，CAN 权限要求）
# 用法:
#   ./run_calibration.sh                 # 默认(allow_motion=false, 安全锁定)
#   ./run_calibration.sh true right      # 解锁运动 + 右臂(7电机无夹爪, 参数右表)
#   ./run_calibration.sh true left       # 解锁运动 + 左臂
set -e

# 先 source 全部环境再开严格模式（colcon/ROS 的 setup.bash 与 set -u 冲突）
source /opt/ros/jazzy/setup.bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# 加载工作区（必须存在，否则找不到包）
if [ -f "$WS_DIR/install/setup.bash" ]; then
    source "$WS_DIR/install/setup.bash"
fi
set -uo pipefail

ALLOW="${1:-false}"
SIDE="${2:-left}"

# 右臂(7 电机无夹爪)用独立参数表: 电机表去夹爪 + zero 落盘 zero_right,
# 防止覆盖左臂的 标定/zero
PARAMS_ARG=""
if [ "$SIDE" = "right" ]; then
    PARAMS_ARG="params_file:=$SCRIPT_DIR/config/calibration_params_right.yaml"
fi

# 禁用 Fast DDS SHM 传输(节点以 root 运行时 SHM 段为 root 属主,
# 普通用户 CLI 与节点数据不互通——2026-09-09 实测)。UDP 回环两侧通用。
export FASTRTPS_DEFAULT_PROFILES_FILE="$SCRIPT_DIR/config/fastdds_noshm.xml"

exec sudo -E env HOME=/home/yang PATH="$PATH" \
    PYTHONPATH="$PYTHONPATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    FASTRTPS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE" \
    /opt/ros/jazzy/bin/ros2 launch openarm_calibration calibration.launch.py \
    allow_motion:="$ALLOW" arm_side:="$SIDE" $PARAMS_ARG
