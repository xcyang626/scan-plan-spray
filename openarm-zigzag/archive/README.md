# archive/ —— 已废弃历史路线存档

> ⚠️ **不要在真机上使用本目录的任何脚本**：该路线直接输出电机原始编码器
> 空间目标（无 URDF 零位对齐），误用会撞限位。现行真机主线见 `../real/`。

## 背景（2025-08 路线，已被 ZeroOffsetHW 主线替代）

本目录保存的是项目最早的真机驱动尝试：达妙 **DM-USB2FDCAN** 私有 USB 协议
适配器 + `motorbridge(libdm_device.so)` 用户态库，直接驱动 6 个 DM8009 电机
（当时未接 7 轴与夹爪）。因不生成标准 `can0` 接口、无法对接官方
openarm_can/ros2_control 栈、且无零位换算，后整体切换为
gs_usb + SocketCAN + `openarm_zero_hw/ZeroOffsetHW` 主线（见
`../docs/架构文档.md` §4.1），本路线于 2026-09-11 从根目录移入此处存档。

## 文件清单

| 文件 | 说明 |
|---|---|
| `arm_dm_control.py` | DM-Device ROS2 控制节点：6 电机 pos_vel 模式（速度受限位置控制），发布 `/dm_arm/joint_states`、订阅 `/dm_arm/joint_command`、提供 enable/disable 服务。头注释含"两个独立 SingleThreadedExecutor 规避 rclpy wait set 竞态"的设计笔记 |
| `arm_trajectory_bridge.py` | FollowJointTrajectory action → `/dm_arm/joint_command` 桥接节点，供 MoveIt/JTC 对接 |
| `run_arm_control.sh` | 控制节点启动脚本（sudo -E 保留 ROS 环境开达妙设备，过滤底层收发刷屏） |
| `run_bridge.sh` | 桥接节点启动脚本（同上环境处理） |

## 如确需运行（仅离线/台架验证，勿接臂）

```bash
cd archive
./run_arm_control.sh          # 终端 A：控制节点（需 root + libdm_device.so）
./run_bridge.sh               # 终端 B：轨迹桥接
```
