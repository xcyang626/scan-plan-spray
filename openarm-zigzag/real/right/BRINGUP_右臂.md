# 右臂 Bring-up 手册(7 电机, 无夹爪)

> 前置阅读: `../交接_20260921_RL全程验收进行时.md` → `../common/README.md`。
>
> **当前进度(2026-09-08 晚)**: 阶段 B(标定, zero_right 手工拼装)与
> 阶段 C(配置生成)已完成; 正式之字路线已生成待仿真验证。
> **下一步 = 仿真(`../moveit_sim/start_zigzag_sim_right.sh`) →
> 阶段 A/D/E/F/G(实机)**。阶段 F 微动测试重点裁决 J3/J5 方向
> (channel_sign [−1,+1,+1,+1,+1,+1,+1] 中限位对称不可分辨的关节)。
> 纪律不变: 运动命令全部由人执行; 异常即停、贴日志分析后再动;
> 物理急停是最后防线。执行期间人不离臂。
> ⚠ 右臂是镜像几何: **方向符号、零位、限位全部未真机验证**, 本手册
> 阶段 B(标定)与阶段 F(逐关节微动)就是要把这两件事验证掉, 不可跳过。

## 阶段 A: CAN 上电(不动电机)

```bash
cd ~/桌面/openarm实验/标定/openarm_calibration/scripts
./can_up.sh
ip -s link show can0 | grep -A1 'TX'
cansend can0 7FF#0000000000000000
ip -s link show can0 | grep -A1 'TX'
```

【预期】第二条 TX packets 比第一条 +1。

## 阶段 B: 右臂零位标定(⚠ 全程真实运动并碰撞机械限位)

标定会把当前位形写入电机 flash 零点(set_zero), 之后 d≈0、限位实测。
**清空工作空间, 扶住底座, 远离末端; 左臂 zero 文件不会被覆盖**(右臂写
`标定/zero_right`), 但先做个双保险备份:

```bash
cp ~/桌面/openarm实验/标定/zero ~/桌面/openarm实验/标定/zero_left_20260908.bak
```

B1. 安全锁定启动(电机绝不运动), 人工把右臂摆到接近折叠的初始位形:

```bash
cd ~/桌面/openarm实验/标定
./openarm_calibration/run_calibration.sh
```

【预期】日志报 `电机在线 7/7`(7 台都应答——**若报 ESC_ID=0x0X 无反馈,
先查供电/ID, 不要继续**; 右臂没有 ID 0x08 夹爪, 有 8 台应答反而说明
接错臂/接错线)。

另开终端确认读数:

```bash
cd ~/桌面/openarm实验/标定
./openarm_calibration/ros2.sh topic echo /calibration_node/status --once
```

B2. 确认姿态安全后, Ctrl-C 掉 B1, 解锁重启(右臂自动带 7 电机参数表):

```bash
./openarm_calibration/run_calibration.sh true right
```

另开终端触发标定并观察:

```bash
cd ~/桌面/openarm实验/标定
./openarm_calibration/ros2.sh service call /calibration_node/start std_srvs/srv/Trigger
./openarm_calibration/ros2.sh topic echo /calibration_node/status
```

【预期】顺序: 缓慢回零位基准 → J1 防护摆动 45° → J7→J2 逐关节双向碰限位
(无夹爪段) → J1 标定 → 全部回零位 → `set_zero 写入电机零点` → `保存标定数据`
→ 保持零位。约 5-8 分钟。**任何异常它都会自动失能; 紧急用 abort 服务或断电。**

B3. 完成后目视验证臂在零位, Ctrl-C 失能。确认产物:

```bash
head -c 300 ~/桌面/openarm实验/标定/zero_right; echo
```

【预期】`"status": "ok"`, `"arm_side": "right"`, 数组均为 7 个元素。

★ **停止点 B**: 把标定 status 话题的收尾输出 + zero_right 内容贴给 AI,
确认后再进阶段 C。

## 阶段 C: 生成右臂真机配置(不动电机)

```bash
cd ~/桌面/openarm实验/real/common
/usr/bin/python3 gen_real_config.py --side right --hand off
```

【预期】7 行核对表: `openarm_right_jointN: d=... 软限位=[...]` + 尾行
`通道映射=[1, 2, 3, 4, 5, 6, 7] 通道方向=[1, 1, 1, 1, 1, 1, 1]
MPC话题=/right_mpc_position_commands` + `夹爪 ESC8: 不接触`。
方向符号此刻是**待验证初值(恒等)**, 阶段 F 逐关节确认。

## 阶段 D: 终端 A —— 右臂 control 层(⚠ 电机将使能并锁定当前位形)

```bash
source /opt/ros/jazzy/setup.bash
source ~/桌面/openarm/openarm_ros2_ws/install/setup.bash
source ~/桌面/openarm实验/real/common/ws/install/setup.bash
export ROS_LOG_DIR=/tmp/openarm_real_ros_log
ros2 launch ~/桌面/openarm实验/real/right/config/right_real_control.launch.py
```

【预期验证点】① `CAN=can0 can_fd=on prefix=right_`; ② 7 行
`openarm_right_jointN: d=... kp=... kd=...`; ③ **`MPC 指令通道已订阅:
/right_mpc_position_commands`**(注意话题是 right); ④ `积分器 ki=[12, ...]`;
⑤ `已锁定当前位置(URDF 系)` + spawner `Configured and activated`。
【异常】任何快速/大幅运动 → 立即 Ctrl+C(自动失能) → 贴日志。

## 阶段 E: 读数核对(不动电机, 只读)

另开终端:

```bash
source /opt/ros/jazzy/setup.bash
source ~/桌面/openarm/openarm_ros2_ws/install/setup.bash
source ~/桌面/openarm实验/real/common/ws/install/setup.bash
export ROS_LOG_DIR=/tmp/openarm_real_ros_log
ros2 topic hz /joint_states          # 应约 100Hz
/usr/bin/python3 ~/桌面/openarm实验/real/right/config/check_states.py
```

【预期】7 个关节角与**目测位形一致**, "软限位内: 是"。
【异常】某关节差 90°/180° 量级 → 零位/映射有错, **停止一切运动步骤**,
贴输出发 AI。

## 阶段 F: 逐关节微动验证方向(低速, 一次一个关节)

四终端布局(2026-09-09 起, 实时看到实机 vs 仿真模型的差别):

```bash
cd ~/桌面/openarm实验/real/right/config
# 终端 A  控制层(阶段 D 已起): right_real_control.launch.py
# 终端 B  仿真层: RViz 模型由 /joint_states 实时驱动(=URDF 命令位形),
#                实物与它的偏差就是"实机-仿真差别", 直接目视:
source /opt/ros/jazzy/setup.bash
source ~/桌面/openarm/openarm_ros2_ws/install/setup.bash
source ~/桌面/openarm实验/real/common/ws/install/setup.bash
ros2 launch view_live.launch.py
# 终端 B' 监控层:
/usr/bin/python3 check_states.py --watch
# 终端 C' 命令层(一次只动一个关节, J1 先做; 每关节独立终端或顺序执行):
/usr/bin/python3 move_j1.py              # 默认 5°, 可: move_j3.py 5 --speed 0.08
/usr/bin/python3 move_j2.py
/usr/bin/python3 move_j3.py
/usr/bin/python3 move_j4.py
/usr/bin/python3 move_j5.py
/usr/bin/python3 move_j6.py
/usr/bin/python3 move_j7.py
```

命令层行为: 朝 URDF 正方向慢动 → **到位后停住保持**(不限时, 慢慢看)→
**按 Ctrl+C 自动慢速回 home 并退出**(回程中再按 Ctrl+C = 就地停住)。
动/回全程限速默认 0.1 rad/s。若某关节贴软限位被拒, 人工把它摆离限位
或先 `move_small.py --to-home` 归位。

【预期】每关节朝 URDF 正方向缓慢动到位, 实物转向/幅度与终端 B 的
RViz 模型**同向同量**, Ctrl+C 后自动回 home、残余偏差接近 0。
⚠ 数值读数查不出 channel_sign 方向错误——命令与反馈经同一个(错)符号
换算后读数自洽; **方向终裁只认"实物 vs RViz 模型是否同向"**。
**若某关节朝反方向动** → 该关节 channel_sign 应为 -1:
  1. Ctrl+C 停掉终端 A(control);
  2. 改 `real/common/gen_real_config.py` 里 `SIDES["right"]["channel_sign"]`
     对应位为 -1(记录进交接文档);
  3. 重跑阶段 C 的 gen; 重启阶段 D; 只重验改过的关节。

(批量扫查工具仍在: joint_sweep.py / joint_sweep_sim.py /
compare_sweep.py, 供后续 A/B 定量对比用, 阶段 F 不依赖。)

★ **停止点 F**: 7 关节方向全部确认后, 把结论(哪个关节翻转了)告诉 AI,
AI 回填 gen 的 SIDES 表并写入交接文档。

## 阶段 G: 终端 B —— MoveIt + watchdog, 复位, 回 home

```bash
source /opt/ros/jazzy/setup.bash
source ~/桌面/openarm/openarm_ros2_ws/install/setup.bash
source ~/桌面/openarm实验/real/common/ws/install/setup.bash
export ROS_LOG_DIR=/tmp/openarm_real_ros_log
ros2 launch ~/桌面/openarm实验/real/right/config/right_real_moveit.launch.py
```

【预期】move_group 就绪、RViz(右臂)、`安全监控已启动`。瘫姿会先触发一次
慢速回位(RETURNING), 正常。然后:

```bash
ros2 topic echo /safety/status        # 稳定 OK 后 Ctrl+C
/usr/bin/python3 ~/桌面/openarm实验/real/right/config/move_small.py --to-home
```

【预期】右臂竖直下垂, 基本无动作。到此右臂 bring-up 完成,
与左臂 9/7 走线前的状态等价。

## 应急速查(任何已 source 的终端)

```bash
ros2 topic pub -1 /safety/estop std_msgs/Bool "{data: true}"    # 急停→自动回家
ros2 topic echo /safety/status                                  # OK/TRIPPED/RETURNING
ros2 topic pub -1 /safety/reset std_msgs/Bool "{data: true}"    # 人工确认后复位
# 最终手段: 拍急停 / 断电(电机立即卸力)
```

## 本手册尚未覆盖(下阶段)

- 右臂之字路线: **临时版已绘出**(2026-09-08, 左臂工作区镜像,
  见 `assets/figures/moveit_zigzag_right_route.png` 与
  `仿真/moveit_sim/make_dense_zigzag_right.py`)。其限位是"左臂实测镜像"的临时值,
  本手册阶段 B 标定完成后需用右臂 real_safety.yaml **重新生成正式版**,
  经离线复核后才可上真机。
- MPC 轨(right/mpc_track)文件已就位, 需在右臂正式路线 + MPC 离线复核
  之后再上真机。
