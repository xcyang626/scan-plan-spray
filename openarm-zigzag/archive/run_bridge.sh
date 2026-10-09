#!/usr/bin/env bash
# 启动 OpenArm 轨迹桥接节点。
# 该节点是纯 DDS 客户端，但控制节点(arm_dm_control)以 root 运行，
# 普通用户进程无法互通，因此这里同样用 sudo -E 保留完整环境。
cd "$(dirname "$0")"
source /opt/ros/jazzy/setup.bash
set -euo pipefail

sudo -E env HOME=/home/yang \
  PYTHONPATH="${PYTHONPATH:-}" \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
  /usr/bin/python3 "$(pwd)/arm_trajectory_bridge.py" "$@"