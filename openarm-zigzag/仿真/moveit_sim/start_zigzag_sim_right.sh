#!/bin/bash
# ============================================================================
# OpenArm 右臂 MoveIt 之字形仿真 一键启动(左臂版 start_zigzag_sim.sh 的镜像)
# ============================================================================
# 效果: 弹出 RViz 窗口(右臂), 机械臂自动执行:
#   回零位(0.15倍速) → 之字形起点(0.2倍速) → 缓慢之字形走线(TCP 5cm/s)
#   → 终点缓慢回零(0.1倍速)
# 路线: workspace_right.json / dense_zigzag_right.json(左臂工作区 y→−y 镜像)
#
# 用法:
#   ./start_zigzag_sim_right.sh            # 右臂稠密 IK 模式
#
# 结束后 RViz 保持打开; 按回车退出并清理。
# ⚠ 与左臂脚本共用 /controller_manager 与 MoveIt 话题, 不可同时跑两份仿真。
# ============================================================================
set -e

SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIDE=right

# ROS 环境必须用系统解释器; 若当前 shell 在 conda 环境里先退出来
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "提示: 检测到 conda 环境 [$CONDA_DEFAULT_ENV], 临时移出 PATH 以免干扰 ros2"
    export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "$CONDA_PREFIX" | paste -sd:)
fi
source /opt/ros/jazzy/setup.bash
source "$HOME/桌面/openarm/openarm_ros2_ws/install/setup.bash"
export ROS_LOG_DIR=/tmp/openarm_sim_ros_log

# ---- 1. 清理残留进程(防 URDF/重复节点冲突) ----
pkill -9 -f '[m]ove_group' 2>/dev/null || true
pkill -9 -f '[r]os2_control_node' 2>/dev/null || true
pkill -9 -f '[c]omponent_container' 2>/dev/null || true
pkill -9 -f '[r]viz2' 2>/dev/null || true
sleep 1

# ---- 2. 启动右臂单臂 MoveIt demo ----
echo "== 启动右臂 MoveIt demo (RViz 窗口即将弹出) =="
ros2 launch "$SIM_DIR/right_config/right_demo.launch.py" &
DEMO_PID=$!

# ---- 3. 等待 MoveIt 服务就绪 ----
echo "== 等待 MoveIt 服务就绪 (约 20~40s) =="
READY=0
for i in $(seq 1 60); do
    if ros2 service list 2>/dev/null | grep -q '/compute_cartesian_path'; then
        READY=1
        break
    fi
    sleep 1
done
if [ "$READY" != "1" ]; then
    echo "错误: MoveIt 服务 60s 内未就绪, 查看上方 demo 日志排查"
    kill $DEMO_PID 2>/dev/null || true
    exit 1
fi

# ---- 4. 执行右臂之字形 ----
echo "=================================================================="
echo "== 开始右臂之字形执行: 回零位 → 起点 → 走线(约120s) → 回零 =="
echo "=================================================================="
cd "$SIM_DIR"
/usr/bin/python3 zigzag_moveit.py --side right

echo ""
echo "=================================================================="
echo "== 仿真结束 ✔  RViz 保持打开, 可拖动 [MotionPlanning] 交互查看 =="
echo "=================================================================="
read -rp "按回车关闭所有仿真进程并退出 ..."
kill $DEMO_PID 2>/dev/null || true
pkill -9 -f '[m]ove_group' 2>/dev/null || true
pkill -9 -f '[r]viz2' 2>/dev/null || true
pkill -9 -f '[r]os2_control_node' 2>/dev/null || true
echo "已清理。"
