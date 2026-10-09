#!/bin/bash
# 仿真标定一键脚本: 创建 vcan0 → 启动电机仿真器 + 标定节点（可选 RViz 3D 可视化）
# 用法:
#   ./run_simulation.sh                              # 安全锁定, 无 RViz
#   ./run_simulation.sh true right                   # 解锁 + 右臂
#   ./run_simulation.sh true right rviz              # 解锁 + 右臂 + URDF/RViz 3D 显示
set -e

source /opt/ros/jazzy/setup.bash
WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$WS_DIR/install/setup.bash" ] && source "$WS_DIR/install/setup.bash"
# 官方 openarm_description 包（解析 URDF 里的 package:// mesh 路径）
if [ -f "$HOME/桌面/openarm/openarm_ros2_ws/install/setup.bash" ]; then
    source "$HOME/桌面/openarm/openarm_ros2_ws/install/setup.bash"
fi
set -uo pipefail

ALLOW="${1:-false}"
SIDE="${2:-left}"
# 第 3 参数任意非空值(如 rviz) = 开启 RViz 3D 显示
RVIZ="false"
if [ -n "${3:-}" ]; then RVIZ="true"; fi

# 创建虚拟 CAN 总线（vcan 无需波特率）
sudo modprobe vcan
sudo ip link del dev vcan0 2>/dev/null || true
sudo ip link add dev vcan0 type vcan
sudo ip link set vcan0 txqueuelen 100
sudo ip link set vcan0 up

exec sudo -E env HOME=/home/yang PATH="$PATH" \
    PYTHONPATH="$PYTHONPATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    /opt/ros/jazzy/bin/ros2 launch openarm_calibration simulation.launch.py \
    allow_motion:="$ALLOW" arm_side:="$SIDE" rviz:="$RVIZ"
