# rl_track —— 右臂 RL 执行节点(真机)

> RL 策略(IsaacLab 训练 → MuJoCo 验证)在真机上的执行节点。
> 设计文档: [`docs/RL轨迹提速_奖励与时间设计.md`](../../../docs/RL轨迹提速_奖励与时间设计.md)
> 训练链路: [`仿真/isaaclab_sim/rl/README.md`](../../../仿真/isaaclab_sim/rl/README.md)

## ⚠ 安全铁律(与全项目一致)

1. **AI 不执行任何让机器人运动的命令**; 本文档只给指令, 由操作员本人执行。
2. 标定节点与 control/MoveIt **绝不能同时运行**。
3. 三级急停降级: 停流(Ctrl+C) → `/safety/estop` → 物理断电。
4. 首次上机必须 **分段验证**(见下), 不要一上来跑全程。

## 原理

```
actor_right.npz (纯 numpy MLP, 确定性动作)
   ↑ 67 维观测(与训练逐项一致)
/joint_states (100Hz) ──→ rl_exec.py ──→ q_cmd = q_ref(t) + 0.25·tanh(a)
                                            │ 钳位: 软限位 −3°
                                            │ 速率: 0.05rad/条 @50Hz = 2.5rad/s
                                            │ 残差低通 5Hz + 启动渐入 1.5s
                                            ▼
                    /right_rl_position_commands (RL 独立话题)
                                            ▼
             zero_offset_hw 注入通道(mpc_enable=true 时订阅)
               · ±6.5rad 绝对钳位 + 0.2rad/条速率钳位(驱动侧, 仍在)
               · 0.2s 断流自动回落控制器保持
               · watchdog / estop 立即停流
```

**纯 RL, 无 MPC 计算**: 策略 = 4 层 numpy MLP, 链路里没有任何 MPC 求解。
与 MPC 方案(OpenArmMpcController 插件)完全分离, 切换协议见
`docs/RL与MPC方案对比与切换.md`。

## 文件

| 文件 | 说明 |
|---|---|
| `rl_exec.py` | 执行节点(50Hz, 只发布不驱动电机) |
| `actor_right.npz` | 策略权重(从训练产物复制过来) |
| `ref_right.npz` | 参考轨迹(与训练/MuJoCo 同一份, 默认指向 rl/ 目录) |
| `grav_ff_right.npz` | 重力前馈表(make_grav_ff.py 产物, 与训练同一份; 见下节) |
| `grav_ff_right_iter1.npz` | ILC 合成前馈表(基线+Δτ, 2026-09-22; 见 ILC 节) |
| `ilc_delta_right_iter1.npz` | ILC 修正量 Δτ(仅审计用, 部署读合成表) |
| `ilc_learn.py` | ILC 学习器: 从运行 CSV 重构欠账 → 生成修正表(离线) |
| `ilc_check_real.py` | 故障注入孪生: 端到端验证学习律(离线) |
| `rl_policy.py` | 观测装配 + MLP 前向 + 指令钳位(纯 numpy, 无 ROS) |
| `verify_deploy.py` | 离线部署验证(配置核对 / 安全预检 / MuJoCo 回放; `--ff-file` 可审计覆盖表) |

## 重力前馈 + 参考速度通道(2026-09-16, 官方 MIT 五元组方案)

**来源**: 官方 `openarm_simple_hardware.cpp` 本就导出 position/velocity/
effort 三个命令接口、MIT 帧 `{kp,kd,p_des,v_des,t_ff}` 五元组全量下发;
我们的 `zero_offset_hw` 此前只用 p_des(v_des=0, t_ff=积分器)。本方案把
五元组用满 —— **t_ff=重力前馈(消准静态下垂) + v_des=参考速度(消动态
滞后)**, pd_screen A/B: max|e| −30%, mean|e| −60%, 力矩成本不变。

**数据流**(单一权威源): `仿真/mujoco_sim/make_grav_ff.py`(v2,
**真机描述 URDF 权威动力学**——与训练 USD 的 URDF 逐项一致, 已审计;
v1 用 MJCF 近似质量欠补偿 J2 达 42%, 已废弃) 生成 `grav_ff_right.npz`
→ 训练 env(自动启用, DR 0.8~1.2)/`rl_validate --grav-ff`/
本目录 `rl_exec` 三侧共用。**参考轨迹重生成后必须重跑并同步两处副本**,
`verify_deploy.py` 审计配对。

**执行**(前馈默认自动: 表在即开; vref 默认开):

```bash
/usr/bin/python3 rl_exec.py --max-t 15 --log-csv        # 21 项消息: 位置+速度+前馈
/usr/bin/python3 rl_exec.py --log-csv                   # 全程
# 退回旧行为: --grav-ff 0 --vref 0
# CSV 末尾多 ff1..ff7 列 [Nm]
```

**安全机制**(全部已在链路里):
- 驱动侧(需重编译 zero_offset_hw + 重启 control): 前馈 ±[15,15,20,20,
  5³] Nm(0.75×额定, URDF 权威 J2 峰值 9.9Nm)+速率 0.5Nm/条; 速度
  ±1.5rad/s+速率 0.1/条; **断流(>0.2s)回落 JTC 时 v_des 回 JTC 接口、
  t_ff 归零**; 7/14/21 项消息全兼容;
- 节点侧: 前馈启动淡入/结束·回家·保持淡出各 1s; 速度参考首尾自然为 0
  无需渐变;
- 训练侧: 前馈幅度 DR 0.8~1.2(URDF 口径=USD 物理残余误差小);
- `verify_deploy.py`: 训练↔部署配对 + 行数 + 峰值审计。

**首次上机前**: 确认 zero_offset_hw 已用新源码重编译并重启 control。
⚠ **旧驱动 + 新消息 = RL 指令流被整条忽略**(旧回调只认 7 项 → 判流
过期回落 JTC, 现象"节点在发、臂不动", 无猛拉风险); `--grav-ff 0
--vref 0` 可临时退回纯位置模式。

## ILC 前馈修正表(2026-09-22, 治理低频猎振)

**⚠ 当前权威表 = `grav_ff_right_iter2.npz`**(合成峰值 8.96Nm, 审计全绿
`verify_deploy_report_iter2.json`)。iter-1 因学习工具的滤波口径 bug
(50Hz 帧网格误按 10Hz → Q 实际截止 0.4Hz, 猎振带漏入)已废弃, 其已部署
版本存档为 `ilc_delta_right_iter1_deployed_20260922.npz`; 0922 首次
60s 复验即因该 bug 部分未达预期, 修正后真慢欠账跨日相关 0.87~1.00。

**背景**: 0921 全程首跑 73.4s 急停的根因 = URDF 外未建模负载(线缆拖拽)
在悬伸位形增大 → 重力前馈欠补偿 → 驱动软件积分器补上又过冲(慢猎振,
主峰 0.1~0.3Hz)。真慢(<0.08Hz)欠账量级 J1 2.2 / J2 0.9 / J3 1.6 Nm。

**方案**( docs/ILC可行性_实施路径文档.md 的力矩前馈实现, 无需重训——
策略训练含 FF DR 0.8~1.2): 从运行 CSV 在 50Hz 帧网格重构驱动反馈力矩
`τ_fb = kp·e_cmd + kd·ė_cmd + integ`(积分器按 zero_offset_hw 同律),
取 0.08Hz 零相位低通(只学位形相关慢变欠账, 剔除猎振),
`Δτ = clip(0.6·τ̂, ±[3,3,3,2,1,1,1]Nm)` 逐帧叠加进前馈表。

**部署与回滚**(运行时零代码改动, 基线表不动):

```bash
# 审计(必须全绿后再上机; 默认行为不变, 基线报告不受影响)
conda activate mujoco && python verify_deploy.py \
    --ff-file grav_ff_right_iter2.npz --out verify_deploy_report_iter2.json
# 执行(操作员): 唯一区别是 --grav-ff-file
/usr/bin/python3 rl_exec.py --max-t 60 --log-csv \
    --grav-ff-file grav_ff_right_iter2.npz          # 分段复验
/usr/bin/python3 rl_exec.py --log-csv \
    --grav-ff-file grav_ff_right_iter2.npz          # 全程
# 回滚: 去掉 --grav-ff-file(或删表文件), 立即回到已认证基线
```

**复验判据**(0922 修订): 60s+ 段慢误差(J2/J5, <0.3Hz)峰峰应明显小于
0921 同段; `cmd−q` 均值与积分器蓄能下降; 无新增高频抖动。0921 的
"0.5~2Hz 峰峰 2.9°"口径出自已丢失的 100Hz 黑匣子, 10Hz rl_exec CSV
无法等价复算 —— 复验时黑匣子(新版落当前目录, 不再覆盖)必录, 两套
日志同存以便对齐口径。

**迭代**: 每次跑完生成下一迭代(累积律, 自动继承已部署修正):
```bash
/usr/bin/python3 ilc_learn.py --csv rl_exec_<新>.csv \
    --prev-delta ilc_delta_right_iter2.npz --iter-tag iter3
```
预期残差每迭代 ×0.4。学习细节与复验结论见
`real/交接_20260922_ILC就绪与复验指引.md`。

## 实时调速(2026-09-23, /rl/speed_scale)

**运行中改变参考时间轴倍率**(策略仍为 143.6s 时序训练的 actor_right,
无重训; 依据《RL轨迹提速》速度剖面跟随奖励 + obs 的 qd_ref 缩放设计)。

```bash
# 跑动中任意时刻(另一终端):
ros2 topic pub -1 /rl/speed_scale std_msgs/Float64 "{data: 1.25}"   # 提速到 1.25×
ros2 topic pub -1 /rl/speed_scale std_msgs/Float64 "{data: 0.7}"    # 减速
ros2 topic pub -1 /rl/speed_scale std_msgs/Float64 "{data: 0.0}"    # 暂停(指令冻结,通道保鲜)
ros2 topic pub -1 /rl/speed_scale std_msgs/Float64 "{data: 1.0}"    # 回原速
```

**保护**: 硬限 [0, 1.5](`--speed-min/max`), 变化率限幅 0.5/s(参考速度无
阶跃), vref/观测的 qd_ref 同步缩放, ILC 表按 idx 索引不受影响, 全部
安全链(速率钳位/看门狗/急停)不动。CSV 末列 spd 记录实际倍率。
**孪生矩阵结论**(speed_matrix_report.json): 0.5~1.5× mean 误差 1.18~1.23°
无拐点, |τ| 5.9→10.1Nm 次线性(钳位 15/15/20/20/5³ 内)。真机阶梯:
15s @1.25 → 60s @1.25 → 全程; 持续低于 0.5× 可能入摩擦带(勿长用)。

## 部署前检查清单

- [ ] **跑 `verify_deploy.py` 并全绿**(最重要的一步, 见下)
- [ ] IsaacLab 训练收敛(play.py 显示关节误差 RMS < 1°, 全程无失败)
- [ ] MuJoCo 验证通过: `仿真/mujoco_sim/rl_validate.py --actor actor_right.npz`
      (对比 JTC 基线 HF σ: J1 0.043 / J2 0.053 rad/s)
- [ ] `ref_right.npz` 与 `actor_right.npz` 来自同一次训练(元数据 `T_total` 一致)
- [ ] 机械臂已回 home(`move_small.py --to-home`), 偏差 ≤2°
- [ ] 黑匣子已启动: `real/common/js_blackbox.py`(100Hz CSV, 抖动归因)
- [ ] 现场有物理急停, 人手在急停上

## 零号前置: 标定一致性(必做)

参考轨迹是在某个标定版本下生成的。**重标定后必须重跑参考与重训**,
否则关节限位/home 的坐标系对不上, 轻则跟踪差, 重则撞限位。

```bash
# 离线验证(不接触机器人, 秒级~分钟级): 配置核对 + 安全性预检 + 物理回放
conda activate mujoco
cd ~/桌面/openarm实验/real/right/rl_track
python verify_deploy.py            # 全套(含 MuJoCo 回放, 约 6 分钟)
python verify_deploy.py --limit-check-only   # 只做配置核对+安全预检(秒级)
```

全绿(`总判定: 可以上机`)才继续。任一项 `[!!]` 都不要上机。

> **2026-09-11 实例**: 右臂当天 15:19 用 official-zero-v2 重标定, J6 可用范围
> 比旧标定小 15.9°; 旧参考轨迹的 J6 超出新限位 8°, verify_deploy 直接拦下。
> 处置: 重做工作区扫描→之字形→参考→重训(产物已更新)。教训是
> **软限位/home/参考必须同源**, 现已全部从 `real_safety.yaml` 读取
> (`zigzag_ref.py` / `zigzag_env.py` / `rl_policy.py`), 不再有硬编码副本。

## 使用(操作员执行)

```bash
# 0) 标准上电: can_up.sh → 终端A(control) → 终端B(watchdog/MoveIt)
# 1) 回 home
cd ~/桌面/openarm实验/real/right/config && /usr/bin/python3 move_small.py --to-home

# 2) 分段 1: 前 15s(转移段, 最慢)。结束时节点自动缓速回 home, 无需手动回位
cd ~/桌面/openarm实验/real/right/rl_track
/usr/bin/python3 rl_exec.py --max-t 15 --log-csv

# 3) 分段 2: 前 60s(进入走线)。结束同样自动缓速回 home
#    (0922 修复长行程单位 bug 后, 长行程回家按规划需 ~23s(0.1rad/s
#    巡航), 属正常 —— 此前的同场景会触发"背离 home 的协调跳变"猛拉)
/usr/bin/python3 rl_exec.py --max-t 60 --log-csv

# 4) 全程(≈143.6s), 手放急停
/usr/bin/python3 rl_exec.py --log-csv

# 5) 转移段抖动治理实验: 转移段时间轴 ×1.5(总时长 ≈140s)
/usr/bin/python3 rl_exec.py --transfer-boost 1.5 --log-csv
#    分段验证时同样可用: --transfer-boost 1.5 --max-t 15 --log-csv
```

### ⚠ 为什么不能在分段后直接 `move_small.py --to-home`（重要教训）

**直接 Ctrl+C 或让节点停流 ≠ 安全**。停流后 0.2s 驱动回落 JTC, 而 JTC
保持的目标是 **home** —— 臂离 home 远时, 回落瞬间等于一步跳到 home,
全 PD 力矩猛拉(2026-09-14 实测造成碰撞)。因此:

- **分段执行结束** → 节点自动执行"缓速回家段"(0.1 rad/s S 曲线),
  到 home 后才停流, 回落零阶跃;
- **Ctrl+C 第一次** → 位置保持(冻结当前指令持续发布), 不会回落;
- **Ctrl+C 第二次** → 强制停流(会猛拉回 home!) —— 除非紧急, 不要这样停;
- **保持/中止后的回收**: 另开终端
  `/usr/bin/python3 rl_exec.py --return-home`(从当前位置缓速回 home);
- 只有臂**本来就在 home 附近**(±2°) 时才用 `move_small.py --to-home`。

### 中止 / 应急

```bash
ros2 topic pub -1 /safety/estop std_msgs/Bool "{data: true}"   # 软急停
# 急停 = 停止轨迹 + 0.25rad/s 缓速滑回 home(2026-09-18 修复断流回落
#   限速, 旧行为 12rad/s 扑跳已消除); 急停后 reset 恢复
ros2 topic pub -1 /safety/reset std_msgs/Bool "{data: true}"   # 复位
# 最终手段: 物理断电
```

终端 A 应出现 `指令源: RL`(2026-09-18 起注入通道完全分离: RL 与 MPC 各自独立状态, 日志直接显示源名 RL/MPC/JTC)。

## 每段验证要看什么

1. **终端输出**: 结束后打印"最大跟踪偏差 X°" —— 应 < 3°(仿真基准: MuJoCo 验证 mean ~2°);
2. **CSV**(`rl_exec_<时间戳>.csv`, 10Hz 记录): 用 `real/common/js_blackbox.py` 的
   CSV 或本 CSV 算走线段速度 HF σ, 与 JTC 基线 0.043/0.053 对比;
3. **走线时长**: 应 ≈150s(±5s, 当前参考 150.5s); 明显偏差 → 查 ref/actor 是否同批;
4. **听/看**: 有无异常啸叫、抖动、单关节跳跃。

## 异常处置

| 现象 | 处置 |
|---|---|
| 单关节明显抖动/啸叫 | Ctrl+C 停流 → 检查 CSV → 若确认策略抖动, 训练侧 w_action_rate/w_action_smooth ×2 重训 |
| 跟踪偏差 > 5° | 停流 → 查是否参考与 actor 不匹配(ref/actor 不同次) |
| watchdog TRIPPED / RETURNING | 自动停流, 按 `move_small.py` 提示人工复位 |
| 任何异常运动 | 物理急停 |

> **⚠ 2026-09-17 事故复盘(已修复)**: 中止全程时第一次 Ctrl+C 触发
> rclpy 内置 SIGINT 处理器直接关闭 ROS 上下文("位置保持"分支的 publish
> 全部失效) → 指令流断 → 驱动 0.2s 回落 JTC(其保持目标=control 激活时
> 位形) → 猛拉+剧烈抖动+watchdog TRIPPED。修复: `rclpy.init(signal_
> handler_options=SignalHandlerOptions.NO)`, 第一次 Ctrl+C 现在真正
> 位置保持。教训: 评测 rl_exec 行为必须在真机/完整 rclpy 环境下,
> 中止永远优先 `--return-home` 而非强停。

## 执行器增益(2026-09-14 论文基线)

PD/力矩/动作幅值已按论文经验公式重设(arXiv 2508.08241 §S3:
kp=I·ω², kd=2Iζω, ζ=2, ω=10Hz, I=kg²·I_rotor; α=0.25·τ_rated/kp;
力矩=达妙官方额定值):

| J | 电机 | kp | kd | τ_额定 | α |
|---|---|---|---|---|---|
| J1/J2 | DM-J8009P | 48.0 | 3.05 | 20 | 0.104 rad |
| J3/J4 | DM-J4340 | 126.3 | 8.04 | 9 | 0.0178 rad |
| J5-J7 | DM-J4310 | 7.1 | 0.45 | 3 | 0.106 rad |

配套项: 折算转子惯量(armature) = 0.0122/0.032/0.0018(逐关节);
硬件 tmax 寄存器保持峰值 54/28/10 不变(物理余量), 额定预算由
策略侧执行。真机应用: `gen_real_config.py --side right --scheme rl
--hand off --kp "48,48,126.3,126.3,7.1,7.1,7.1" --kd
"3.05,3.05,8.04,8.04,0.45,0.45,0.45"` + 重启 control。
整定细化流程: `docs/PD参数整定流程.md`。

## 与 MPC 方案的关系

RL 方案与 MPC 方案(OpenArmMpcController 插件)**完全分离、互斥运行**,
切换协议见 `docs/RL与MPC方案对比与切换.md`。若 RL 效果不达标, 切回 MPC:

```bash
cd real/right/mpc_track
/usr/bin/python3 mpc_exec.py --ref ../../仿真/isaaclab_sim/rl/ref_right.npz
```

```bash
# 切回 MPC 方案(两步):
#   1) 重新生成配置:  cd real/common
#      /usr/bin/python3 gen_real_config.py --side right --hand off   # 默认 --scheme mpc
#   2) ros2_controllers_real.yaml 槽位 = openarm_mpc_controller/...(已是)
# 切到 RL 方案:
#      /usr/bin/python3 gen_real_config.py --side right --scheme rl --hand off
#   且槽位建议换回 joint_trajectory_controller/JointTrajectoryController
# 每次切换后: 重启 control 层 + verify_deploy.py --scheme <rl|mpc> 全绿再上机
```
