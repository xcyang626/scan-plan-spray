# MPC 控制器插件化改造设计(线路一)

> 2026-09-14。目标: 把右臂 MPC 从"独立节点 + 话题注入"(线路二)升级为
> **ros2_control 控制器插件**(线路一), MoveIt2 侧零修改、照常
> plan/execute, 轨迹经 FollowJointTrajectory 自动由 MPC 控制器执行。
> 参考文章: CSDN ZPC8210/163364973《MoveIt2 + MPC 集成路线》线路一;
> weixin_29057695/158109667(PlannerManager 规划器插件路线, 本文未采用,
> 见 §八"未采用路线")。

## 一、架构: 原链路 → 新链路

原(线路二, 9/08 起, 已实机验证):
```
dense_zigzag_right.json → mpc_exec.py(自建参考时间轴, 100Hz)
  → /right_mpc_position_commands 话题
  → ZeroOffsetHW 内部注入通道(流新鲜<0.2s 优先, 否则回落 JTC)
  → MIT 电机
```

新(线路一, 本文档):
```
MoveIt2 plan (或 zigzag_moveit.py --dense 直接下发)
  → FollowJointTrajectory action
  → controller_manager → OpenArmMpcController 插件 (update() 100Hz)
       ① 读硬件状态(state interfaces)
       ② 从收到的轨迹插值出参考窗 r_0..r_N
       ③ 闭式预览 MPC 解 a0(§三)
       ④ 写位置指令(command interfaces)
  → ZeroOffsetHW write()(不感知 MPC 存在) → MIT 电机
```

ZeroOffsetHW 完全不改代码。旧注入通道保留但默认关闭(§五.3)。

## 二、接入方式: 槽位名接管(一行回滚)

`move_small.py`、`moveit_controllers_real.yaml`、`zigzag_moveit.py`
全部硬编码 `right_joint_trajectory_controller` 这个控制器名。因此插件
**直接以该名字注册类型**:

```yaml
# ros2_controllers_real.yaml
right_joint_trajectory_controller:
  type: openarm_mpc_controller/OpenArmMpcController   # ← 换回
  # type: joint_trajectory_controller/JointTrajectoryController  # ← 旧 JTC(回滚用)
```

- 所有调用方(脚本/MoveIt/launch)一行不改, 自动走 MPC;
- **回滚 = 把 type 行换回去, 重新 spawn**(或整段注释);
- 控制器启动日志会明确打印 `OpenArmMpcController (MPC 接管 JTC 槽位)`,
  防止排查时被名字误导。
- JTC 与 MPC 控制器**不能同时激活**(都占用 position 指令接口), 本来就互斥。

## 三、MPC 数学规格(正式确定, 与 mpc_core.py 逐点一致)

### 3.1 预测模型

逐关节**独立双积分模型**(7 关节解耦、无耦合动力学项):

```
状态 x = [q; q̇] ∈ R²,  输入 u = q̈ (加速度型)
连续: q' = q̇,  q̇' = u
精确 ZOH 离散:  p_j = q0 + j·h·v0 + h² Σ_{m<j}(j-m-0.5)·a_m
               v_j = v0 + h·Σ_{m<j} a_m        (h = dt = 0.01s)
```

不建耦合动力学/M 惯量阵(博客"加速度型 NMPC"那档), 理由:
① 右臂辨识 ident_result.json 仅 J1 可信(R²=0.98), J2-J7 R²≈0, 耦合
模型没有数据支撑(9/14 交接 §四); ② 电机 MIT 位置内环(kp/kd)吸收模型
误差, 外环 MPC 只需产出平滑可跟踪的位置指令——0.4 首跑已验证该口径
(指令态跟踪 1.00°)。

### 3.2 优化目标

N 步跟踪代价(N=25, 预览 250ms):

```
J = Σ_{j=1..N} [ w_p·(q_j − r_j)² + w_v·(q̇_j − ṙ_j)² + w_a·a_{j-1}² ]
    w_p=1000, w_v=200, w_a=1
ṙ_j = 参考窗有限差分 (r_j − r_{j−1})/h   (移动参考自动前馈)
```

**闭式解**(无 QP 求解器依赖, 建时预计算): 堆叠预测方程为
`[p误差; v误差] = e + M·a`, `a* = (MᵀWM+R)⁻¹MᵀW·(−e)`, 只取首条指令
a0, 即线性增益 `a0 = −K·e`, K 逐关节预计算为 (2N,) 行向量。
每周期 7 关节各一次 50 维点积, 100Hz 下计算量可忽略(不超周期, 实测
0.4/0.8 两档 Python 版同款数学全程无超时)。

### 3.3 约束条件

| 约束 | 处理 | 来源 |
|---|---|---|
| 加速度 \|a\| ≤ a_max=1.2 rad/s² | 闭式解后饱和(软约束) | mpc_config.json |
| 关节位置 ∈ 净空盒 [软限位+3°, 软限位−3°] | 预测位置逐步投影(硬) | real_safety.yaml + clearance_deg |
| 关节速度 | **无硬约束**——由参考时间轴有界保证(pretimed smooth ≤ SCALE·VEL_CAPS; MoveIt 重定时含速度约束) | 时间参数化层 |

速度硬化(带 \|q̇\|≤q̇max 的 QP, 博客线路一 OSQP 档)列为后续可选升级
(§七), 不在本次: 换求解器=换行为, 需要重新实机验证, 与"行为保持
已验证口径"的改造原则冲突。

### 3.4 与博客线路一的逐项对照

| 博客线路一要素 | 本文实现 |
|---|---|
| x=[q; q̇] 状态 | ✔ 双积分模型 |
| u=q̈ 加速度型 / OSQP 或 Acados | u=q̈ 但线性闭式解(无求解器依赖); QP 列为后续 |
| 跟踪代价 xᵀQx + uᵀRu | ✔ w_p/w_v/w_a 三项 N 步和 |
| 位置/速度/加速度限幅 | 位置硬化投影 + 加速度饱和; 速度交给时间参数化层 |
| update() 四步(读状态→插值参考→解 MPC→下发) | ✔ 逐条对应 |
| 250Hz~1kHz + PREEMPT_RT | 100Hz(controller_manager update_rate=100, 与被验证的 mpc_exec 同频); RT 内核非必需 |

## 四、控制器行为规格

- **action**: `~/follow_joint_trajectory`(接管槽位名后即
  `/right_joint_trajectory_controller/follow_joint_trajectory`), 语义对齐 JTC:
  accept → execute → succeed/abort/cancel。
- **起点校验**: goal 首点 vs 当前测量, 任一关节 > `start_tolerance_rad`
  (默认 0.035 ≈ 2°) → REJECT(防跳变)。
- **参考插值**: 轨迹点间线性; `t_ref = t_elapsed / speed_scale`
  (speed_scale<1 拉长时间, 语义同 mpc_exec --speed-scale; pretimed
  smooth 轨已烘焙 SCALE, 用 1.0)。预览窗按 `r(t_ref + j·dt)` 采样。
- **模型状态初始化**: goal 起始时从测量位形初始化 q/v, 之后按模型推进
  (与 mpc_exec 完全一致——只吃初始测量, 不吃过程反馈; MIT 内环负责
  实际跟踪)。
- **结束判定**: t_ref 越过轨迹终点后持续向终点收敛, 7 关节测量均在
  `goal_tolerance_rad`(默认 0.20, 对应旧 JTC 最松档)内 → SUCCEED,
  然后继续 hold 终点指令(goal_time=0 语义, 不限时)。
- **保护**(新增, 替代 JTC 的容差监视): 测量与指令偏差任一关节 >
  `abort_deviation_rad`(默认 0.40, 对应旧 JTC 最松 trajectory 容差)
  → ABORT + hold。watchdog `/safety/status` ∈ {TRIPPED, RETURNING} 或
  `/safety/estop`=true → ABORT + hold。
- **hold 语义**: 无 goal / 结束后 / 中止后, 持续写最后一条位置指令。
  ⚠ 与旧线路二的差异: 旧版停流后回落 JTC 的**陈旧指令**(约等于 home),
  新版 hold 在**当前 MPC 指令**(约等于事发位置)。hold 更安全(无回抽
  跳变); 且回 home 显式走 move_small.py --to-home(经 MPC 控制器平滑执行)。
- **接口**: 指令=position×7; 状态=position+velocity×7(ZeroOffsetHW 导出)。

## 五、配置与产物

1. 新包 `real/common/ws/src/openarm_mpc_controller/`(与 zero_hw 同 ws,
   一次 colcon 构建)。数学核 `preview_mpc.{hpp,cpp}` 纯函数、Eigen 实现,
   与 `mpc_core.py` 逐行对应; 附 `mpc_gain_dump` 小工具导出 C++ 增益,
   与 Python `PreviewMPC.K` 数值比对(一致性验收, 见 README)。
2. `ros2_controllers_real.yaml`: type 换为插件 + 追加参数段
   (joints/N/权重/a_max/speed_scale/容差/safety_yaml 路径)。
   软限位单一权威仍是 **real_safety.yaml**(控制器按 `safety_yaml`
   路径直读, 不在 yaml 里复制数值)。
3. **旧注入通道关闭**: `gen_real_config.py` 的 mpc_enable 默认 true→false,
   现有 `openarm_right_real.urdf` 同步改 false。⚠ 关闭后旧 `mpc_exec.py`
   即使误启动也无法驱动电机(话题无消费者), 两条 MPC 通路不可能打架。
   左臂未涉及。
4. launch / moveit_controllers_real.yaml / zigzag_moveit.py /
   move_small.py: **零修改**(槽位名接管)。

## 六、部署与验证步骤(操作员执行, AI 不动电机)

```bash
# 0) 构建(离线, 不碰电机)
cd ~/桌面/openarm实验/real/common/ws
colcon build --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
# 1) 增益一致性(应输出 7 行 max|ΔK| ≈ 1e-12 量级)
./build/openarm_mpc_controller/mpc_gain_dump > /tmp/k_cpp.txt
/usr/bin/python3 ~/桌面/openarm实验/real/right/mpc_track/check_gain_parity.py /tmp/k_cpp.txt
# 2) 离线复核仍是代表(同一套数学): mpc_offline_check.py --ref <新dense>
# 3) 实机(§五 9/14 交接 runbook 照旧, 控制器已被插件接管):
#    终端 A 日志应出现 OpenArmMpcController; 先 move_small.py --to-home
#    (这一步本身就在验证 MPC 路径), 再 zigzag_moveit.py --side right --real ...
```

验证序: 先 0.4 倍速 smooth 轨对照(应与 9/11 mpc_exec 首跑现象一致:
平滑无抖动、无 watchdog 事件), 过了再谈提速/MPC 参考直接吃 dense。

## 七、后续可选(不在本次)

- OSQP-QP 硬化速度约束(博客线路一完整档): 换求解器需重新实机验证;
- 逐关节辨识模型(J2-J7 激励辨识重做后, 双积分 → 含阻尼二阶);
- ILC 前馈叠加在 MPC 参考上(docs/ILC 文档路径不变)。

## 八、未采用路线(为什么)

- **PlannerManager 规划器插件**(weixin_29057695/158109667): 把 MPC 放在
  规划层, 规划时就做优化/动态避障。我们的主线是"MoveIt 出路径, MPC 出
  时间与平滑", 规划层替换动 SRDF/管线配置且与 dense 产物链不匹配。
- **继续线路二**(现 mpc_exec.py): 已验证但属博客明确标注的"快速原型"
  档; 话题注入绕过 ros2_control 生态, watchdog 之外无生命周期/抢占语义。
  代码保留于 `real/right/mpc_track/`(注入通道关闭后自然失活, 可随时回退)。
