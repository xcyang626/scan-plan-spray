# openarm_mpc_controller —— 预览 MPC 控制器插件(线路一)

把右臂 MPC 从"独立节点 + 话题注入"升级为 **ros2_control 控制器插件**:
MoveIt2 零修改照常 plan/execute, 轨迹经 FollowJointTrajectory 进来,
插件在 `update()`(100Hz)里"读状态 → 插值参考窗 → 闭式预览 MPC → 写
position 指令"。设计/数学规格/行为差异见
**`docs/MPC控制器插件_线路一改造设计.md`**(先读它)。

## 关键事实

- **槽位名接管**: 插件以 `right_joint_trajectory_controller` 这个名字
  注册类型(`ros2_controllers_real.yaml` 的 type 行), 所以
  `zigzag_moveit.py` / `move_small.py` / MoveIt / launch **零修改**;
  **回滚 = type 行换回 `joint_trajectory_controller/JointTrajectoryController`
  再重启 control 层**(注释就在 yaml 里)。
- **数学 = `real/right/mpc_track/mpc_core.py` 逐点移植**(N=25, dt=10ms,
  w=1000/200/1, a_max=1.2; a 饱和 + 净空盒投影), 增益一致性由
  `mpc_gain_dump` + `check_gain_parity.py` 验收(实测 worst 6.7e-16)。
- **软限位单一权威仍是 `real_safety.yaml`**: 控制器按
  `safety_yaml` 参数直读, 配置里不复制数值。
- 旧话题注入通道已关(`mpc_enable=false`, gen_real_config.py 同步改默认),
  `mpc_exec.py` 不可能再与插件抢指令源。
- 与旧行为的差异只有两处, 其余逐点一致: ① 中止后 **hold 当前指令**
  (旧版回落 JTC 的陈旧 home 指令); ② 新增"测量-指令偏差 > 0.40 rad →
  abort"(替代 JTC 容差监视)。

## 构建(离线, 不碰电机)

```bash
cd ~/桌面/openarm实验/real/common/ws
source /opt/ros/jazzy/setup.bash
colcon build --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 \
    -Dyaml-cpp_DIR=/usr/lib/x86_64-linux-gnu/cmake/yaml-cpp \
    --packages-select openarm_mpc_controller
# ⚠ -Dyaml-cpp_DIR 必加: 防 miniconda 的 yaml-cpp 0.9 劫持(系统是 0.8,
#   劫持后插件 dlopen 失败 undefined symbol, 本机已踩)
```

## 验收(操作员, 按序)

```bash
# 1) 增益一致性(应 worst < 1e-9, 实测 6.7e-16)
./build/openarm_mpc_controller/mpc_gain_dump > /tmp/k_cpp.txt
/usr/bin/python3 ~/桌面/openarm实验/real/right/mpc_track/check_gain_parity.py /tmp/k_cpp.txt

# 2) 离线复核仍代表插件行为(同一套数学):
/usr/bin/python3 ~/桌面/openarm实验/real/right/mpc_track/mpc_offline_check.py \
    --ref ~/桌面/openarm实验/仿真/moveit_sim/dense_zigzag_right.json

# 3) 实机: 见 real/交接_20260921_RL全程验收进行时.md §五 runbook(四终端)。
#    终端 A 应看到: "OpenArmMpcController (MPC 接管 JTC 槽位, 线路一插件化)"
#    及 7 行净空盒; **不会再有** "指令源切换: MPC 流"(注入通道已关)。
#    move_small.py --to-home 本身就在走 MPC(顺带验证回零);
#    然后 0.4 倍速 smooth 轨对照首跑, 现象应与 9/11 mpc_exec 首跑一致。
```

异常处置: goal 被拒先看终端 A 的"拒绝 goal:"日志(起点偏差 2° 闸门/
estop 挂起/并发 goal); estop 挂起需重启 control 层清除;
紧急程度递增 = Ctrl+C → `/safety/estop` → 物理断电(纪律不变)。

## mock 冒烟测试(无硬件, 已跑通)

`/tmp/mpc_plugin_test/`(ROS_DOMAIN_ID=42 隔离): mock_components 7 关节 +
本插件, 三场景——正常 goal SUCCEED / estop 中途 ABORT / estop 挂起后
REJECT, 全过。测试不进仓库, 需要时按该目录 `run.sh` + `test_client.py`。
