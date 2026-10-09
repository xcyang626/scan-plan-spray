#!/usr/bin/env bash
# 启动 OpenArm DM-device 控制节点。
# 打开达妙设备需要 root；用 sudo -E 保留 ROS2 环境变量，并显式带上 HOME，
# 这样 root 也能从 /home/yang/.cache 找到 libdm_device.so（还要 libusb 权限）。
cd "$(dirname "$0")"

# 先 source ROS2 再开启严格模式，避免 setup.bash 在 set -u 下报未绑定变量
source /opt/ros/jazzy/setup.bash
set -euo pipefail

# sudo 默认会清掉一些环境变量；把 ROS2 需要的路径显式取出来，交给 env 重新注入
# 需要 root 才能打开达妙设备。用 2>&1 | grep 过滤掉达妙驱动的底层
# [DM-DEVICE TX]/[DM-DEVICE RX] 刷屏帧（走一次 sudo 密码提示后正常），
# 让终端只显示节点自己的状态/日志，便于看清“节点就绪”。
sudo -E env HOME=/home/yang \
  PYTHONPATH="${PYTHONPATH:-}" \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
  /usr/bin/python3 "$(pwd)/arm_dm_control.py" "$@" 2>&1 | \
  grep -v -- "\[DM-DEVICE"