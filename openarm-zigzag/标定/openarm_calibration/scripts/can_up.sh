#!/bin/bash
# can0 一键配置/恢复脚本
# 用法: ./can_up.sh
set -e
sudo ip link set can0 down 2>/dev/null || true
sudo ip link set can0 type can bitrate 1000000 sample-point 0.75 \
     dbitrate 5000000 fd on dsample-point 0.75 dsjw 2
sudo ip link set can0 txqueuelen 100
sudo ip link set can0 up
ip -br link show can0

# 禁用 USB 自动挂起（防止 gs_usb 适配器空闲时掉线导致接口归零）
for d in /sys/bus/usb/devices/*/power/autosuspend; do
    echo -1 2>/dev/null | sudo tee "$d" >/dev/null 2>&1 || true
done
sudo sh -c 'for d in /sys/bus/usb/devices/*/power/control; do echo on 2>/dev/null > $d 2>/dev/null || true; done'
echo "USB autosuspend 已对全部设备禁用（重启后失效，需重新执行）"
