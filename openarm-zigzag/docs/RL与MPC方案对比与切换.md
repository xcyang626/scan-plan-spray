# RL 与 MPC 方案对比与切换协议

> 2026-09-14。两套执行方案**完全分离、互斥运行**, 共享的只有:
> 之字形参考轨迹来源(`dense_zigzag_right.json` → `zigzag_ref.py`)、
> 驱动 `ZeroOffsetHW`、安全设施(watchdog/estop/黑匣子)与评估脚本
> (`analyze_real_run.py` 对两方案的 CSV 通用)。
> 目的: 同一任务、同一参考、同一判据下对比选型。

## 1. 两方案是什么

```
┌─ MPC 方案(线路一, 2026-09-14 插件化) ─────────────────────┐
│ dense_zigzag_right.json (+MoveIt/脚本)                     │
│   → FollowJointTrajectory action                          │
│   → OpenArmMpcController 插件(接管 JTC 槽位, 100Hz)        │
│      内部: 轨迹插值参考窗 → 闭式预览 MPC(N=25,dt=10ms)     │
│   → 指令接口 → MIT 电机                                    │
│ 注入通道: 关(mpc_enable=false)                             │
└──────────────────────────────────────────────────────────┘

┌─ RL 方案(纯强化学习, 无任何 MPC 计算) ─────────────────────┐
│ ref_right.npz(50Hz 时间参数化参考, 速度层级内含)           │
│   → rl_exec.py: 观测(67维) → actor numpy 前向(4层MLP)     │
│      q_cmd = q_ref + 0.25·tanh(a)  [低通5Hz + 渐入1.5s]    │
│   → /right_rl_position_commands(RL 独立话题)               │
│   → ZeroOffsetHW 注入通道(mpc_enable=true) → MIT 电机      │
│ 策略训练: IsaacLab PPO(MIT 执行器复刻, 见 rl/README)       │
└──────────────────────────────────────────────────────────┘
```

**RL 链路里没有 MPC**: 策略输出就是最终位置指令, 预览信息来自观测里的
参考前视点(+0.04~0.32s), 不经过任何 MPC 求解。两方案唯一历史上的交集是
运输层话题名, 现已分开(RL = `/right_rl_position_commands`)。

## 2. 对比维度(选型判据)

| 维度 | MPC 方案 | RL 方案 |
|---|---|---|
| 跟踪精度 | 高(闭环最优, 历史实测 ~1°) | 真机 mean 1.51° / max 12.2°(J2 重力滞后) |
| 时间预算 | 由参考+MPC 带宽决定 | 参考直接编码(150.5s 精确达成) |
| 平滑性 | 高(HF ~0.005 宣称) | 附加抖动 ≈ JTC 基线(0.043/0.061 @3×速度) |
| 对模型误差 | 敏感(双积分模型, 无摩擦/耦合) | 域随机化覆盖, 真机验证位置预测准 |
| 计算负载 | 每步解闭式 MPC(100Hz) | 一次 MLP 前向(50Hz, ~0.1ms) |
| 改任务 | 重规划+重定时 | 重生成参考+重训(~40min) |
| 可解释性 | 权重显式可调 | 黑盒, 靠奖励设计 |
| 安全回退 | 控制器生命周期管理 | 断流回落 + 位置保持 + 缓速回家 |

## 3. 切换协议(两步 + 一验)

```bash
cd ~/桌面/openarm实验/real/common
# → MPC 方案(默认):
/usr/bin/python3 gen_real_config.py --side right --hand off
# → RL 方案:
/usr/bin/python3 gen_real_config.py --side right --scheme rl --hand off
```

同时检查控制器槽位(`real/right/config/ros2_controllers_real.yaml`, 一行):

| 方案 | 槽位 type | 注入通道 |
|---|---|---|
| MPC | `openarm_mpc_controller/OpenArmMpcController` | 关（插件直接输出） |
| RL(推荐) | `joint_trajectory_controller/JointTrajectoryController` | 开（单话题：`/right_rl_position_commands`） |
| RL(可跑) | MPC 插件未动 | 开（同上；注入流新鲜时优先于插件输出） |

切换后: **重启 control 层**(配置在驱动 launch 时读入), 然后:

```bash
cd ~/桌面/openarm实验/real/right/rl_track
python verify_deploy.py --scheme rl    # 或 --scheme mpc, 全绿才上机
```

## 4. 对比实验协议(同一判据)

1. 两方案各跑一遍全程(同一 `dense_zigzag_right.json` 几何);
2. 数据: MPC 侧 MoveIt/脚本执行时开黑匣子; RL 侧 `rl_exec.py --log-csv`
   + 黑匣子;
3. 统一用 `real/right/rl_track/analyze_real_run.py` 出指标:
   跟踪误差(mean/p95/max)、分相位用时、HF σ 与附加抖动、|qd| 峰值;
4. 注意公平性: MPC 走线速度与 RL 不同时, HF σ 要做参考 HF 正交分解
   (脚本已内置); 时长以各方案参考/计划值为基准分别核对。

## 5. 已知事项（2026-09-18 注入通道完全分离后更新）

- **方案隔离在配置层强制**：驱动的注入通道只订阅 `mpc_topic` 参数指定的
  **唯一话题**（`gen_real_config --scheme` 写入：RL → `/right_rl_position_commands`，
  MPC → 不开注入 `mpc_enable=false` 走控制器插件）。运行时**物理上不可能**
  同时接收两个方案的指令 —— 无需人工防抢源。
- **无残留启动**：注入源上升沿以当前实际位置重置基线，上一会话无论
  正常/中止/急停/崩溃结束，新会话都从"臂在哪里"开始（历史启动踢跳
  事故的结构性修复）；
- **流断回退**：注入流断 → 驱动以 0.25 rad/s 限速滑回 JTC 目标
  （与看门狗慢速回位无缝衔接、闭锁至追上）；terminal A 日志显示
  `指令源: RL | MPC | JTC`。
- 旧 `mpc_exec.py`(线路二) 已被插件方案取代(自然失活), 不要再跑。
- 仍然建议：两方案的执行节点/控制器不要同时激活（虽已结构性隔离，
  同时运行会让日志与数据归属混乱）。
