# real/common —— 双臂共用层

> 2026-09-08 双臂重构: 原 `real/` 单左臂结构拆分为 `common/`(共用) +
> `left/`(左臂) + `right/`(右臂)。权威变更记录见
> `../交接_20260921_RL全程验收进行时.md`。

## 目录内容

| 文件/目录 | 说明 |
|---|---|
| `ws/` | 驱动工作区(openarm_zero_hw/ZeroOffsetHW)。**完全臂无关**: 关节名前缀/方向符号/零位偏移/夹爪开关/MPC 话题全部来自 URDF ros2_control 参数(由 gen_real_config.py 按臂生成)。 |
| `gen_real_config.py` | 双臂配置生成器。`--side left/right` 选择 zero 文件、基底 URDF、通道方向符号、输出目录(`real/<side>/config/`)。各臂差异集中在脚本内 `SIDES` 表。 |
| `js_blackbox.py` | /joint_states 黑匣子(100Hz CSV → 启动目录 `js_blackbox_时间戳.csv`; 2026-09-22 起不再写 /tmp, 不覆盖旧记录), 臂无关。 |

## 使用规则

1. **一个 CAN 口(can0)同一时刻只接一只臂**: 左右臂电机都用出厂默认
   CAN ID(发送 0x01-07/0x08, 反馈 +0x10), 同总线会冲突。换臂 = 物理换插 +
   重新 `can_up.sh` + 启动对应臂的配置。
2. 每臂的运行配置/工具在 `real/<side>/config/`, MPC 轨在
   `real/<side>/mpc_track/`——**在哪个目录下跑的工具就是哪只臂的**,
   关节名(`openarm_left/right_jointN`)、控制器名、MPC 话题均已写死对应侧,
   防串臂。
3. 修改公共驱动/生成器后, 两侧都要回归(左臂至少 gen --side left 幂等校验,
   右臂至少 py_compile)。
4. 标定工具(`标定/openarm_calibration`)同为双臂通用:
   `run_calibration.sh true left` 用左臂 8 电机表(zero → `标定/zero`),
   `run_calibration.sh true right` 用右臂 7 电机表(zero → `标定/zero_right`,
   不碰左臂文件)。

## 重建驱动(移动/修改 ws 后必须)

```bash
conda deactivate   # 或确保 /usr/bin/python3 优先
source /opt/ros/jazzy/setup.bash
source ~/桌面/openarm/openarm_ros2_ws/install/setup.bash
cd ~/桌面/openarm实验/real/common/ws
rm -rf build install log     # 移动过目录必须清 CMake 绝对路径缓存
colcon build --packages-select openarm_zero_hw \
    --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
```
