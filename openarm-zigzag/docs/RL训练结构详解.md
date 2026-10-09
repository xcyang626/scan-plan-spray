# RL 训练结构详解（右臂之字形走线 · 2026-09-17 定稿版）

> 本文是**当前验证过的 RL 结构**的完整快照，与产出
> `v1_2026-09-17_11-47-10`（600 迭代）checkpoint 的代码逐项一致——
> 该 checkpoint 通过 verify_deploy 全绿（mean 2.88°/max 6.08°/τ 10.9Nm），
> 真机 60s 跟踪 mean 3.02°/max 6.03° 与孪生预测吻合。
> **改训练策略前先读本文；改完同步更新本文。**

---

## 0. 一图流总览

```
ref_right.npz (50Hz 参考: q_ref/qd_ref, 7179 帧, 143.6s)
grav_ff_right.npz (50Hz 前馈表: 重力+科氏+摩擦换向预补偿, make_grav_ff.py v3)
        │
IsaacLab 2048 envs (GPU 管线, 200Hz 物理 ×4 decimation = 50Hz 策略)
   策略 π(a|o) = MLP[256,256,128] ELU, 67 维观测 → 7 维动作
   q_cmd = clamp(q_ref[t] + α·tanh(a), 软限位∓3°)     α=逐关节残差幅值
   v_des = qd_ref[t]  (use_vel_ref=True)               ← MIT 通道参数4
   执行器 DmMITActuator(200Hz):
     τ = kp·(q_cmd−q) + kd·(v_des−v) + I + τ_fric + gff[t]·s_env
     I += ki·EMA(e_p)·dt, 钳位±ic, |EMA|>0.5 清零;  τ_fric=−b·v−c·tanh(v/v_lin)
        │
导出 actor_right.npz (纯 numpy MLP) → rl_exec.py(50Hz, 21 项消息)
        → zero_offset_hw → 达妙 MIT 指令 {kp,kd,p_des,v_des,t_ff}
```

| 层 | 值 | 出处 |
|---|---|---|
| 物理 | 200Hz, implicitfast, replicate_physics=True | zigzag_env.py:92,98 |
| 策略 | 50Hz（decimation=4） | zigzag_env.py:92 |
| 回合 | 随机窗口 12s = 600 步 | zigzag_env.py:93,131 |
| 并行 | 2048 envs, spacing 3.0 | zigzag_env.py:103 |

---

## 1. 任务定义

右臂跟踪 143.6s 之字形走线参考轨迹（速度层级：转移 0.12 rad/s < 水平
4.7cm/s < 竖直 14cm/s）。回合 = 从参考轨迹**随机位置**起（避开首 0.5s
静止段）的 12s 窗口；初态 = 参考态 + 噪声（q: σ=0.0131rad≈0.75°,
qd: σ=0.05rad/s）。轨迹/软限位/home **同源** real_safety.yaml。

## 2. 动作空间（7 维连续）

```
q_cmd = clamp( q_ref[t] + α ⊙ tanh(a), cmd_lo, cmd_hi )
α = 0.25·τ_budget/kp (逐关节) = [0.0417, 0.05, 0.0964, 0.0534, 0.0583×3] rad
cmd 盒 = 软限位 ∓ 3° (clearance_rad=0.0524)
```
- 训练侧: 原始动作先 clamp ±3 再 tanh；1 步动作延迟概率 delay_prob=0.3
  （2026-09-17 实训曾试 0.45 未验证，现保持 0.3）；
- 部署侧(rl_exec): 残差经 5Hz 低通 + 1.5s 启动渐入 + 0.05rad/条速率限幅；
  21 项消息 = q_cmd(7) + v_des(7) + τ_ff(7)。

## 3. 观测空间（67 维，全部真机可得）

| 段 | 维数 | 内容 | 归一化 |
|---|---|---|---|
| 1 | 7 | (q − q_mid)/q_half | 软限位中点/半宽 |
| 2 | 7 | qd / 10 | |
| 3 | 7 | tanh(上一动作) | |
| 4 | 7 | (q_ref[t] − q_mid)/q_half | 当前参考 |
| 5 | 28 | 参考预览 t+{2,4,8,16} 步(+0.04/0.08/0.16/0.32s) | 同段4 |
| 6 | 7 | qd_ref[t] / 10 | 参考速度 |
| 7 | 1 | t/ref_len 进度 | |
| 8 | 3 | 相位 one-hot(转移出/走线/转移回) | |
加性观测噪声(DR): q σ=0.002rad, qd σ=0.02rad/s。

## 4. 执行器与控制律（DmMITActuator, 200Hz, 与真机逐项一致）

```
τ = kp⊙e_p + kd⊙(v_des−v) + I + τ_fric + gff[t]⊙s_env     (URDF 系)
e_f ← 0.3·e_p + 0.7·e_f (EMA, 时间常数 τ=0.0281s)
I  ← clamp(I + ki·e_f·dt, ±ic);  |e_f|>0.5rad 时 I、e_f 清零
τ_fric = −b⊙v − c⊙tanh(v/0.05)
s_env = 逐 env 前馈缩放 ~ U(0.8, 1.2) (回合重置时采样)
```

| J | kp | kd | ki | ±ic | effort_limit(1.3×额定预算) | armature | gff 峰值 |
|---|---|---|---|---|---|---|---|
| J1 | 120 | 7.4 | 12 | 10 | 26 | 0.0122 | 5.57 |
| J2 | 100 | 12.2 | 12 | 10 | 26 | 0.0122 | 9.53 |
| J3 | 70 | 7.5 | 12 | 6 | 28(物理封顶) | 0.032 | 4.07 |
| J4 | 126.3 | 8.04 | 10 | 8 | 28(物理封顶) | 0.032 | 4.03 |
| J5-7 | 30 | 1.5 | 8 | 3 | 9.1 | 0.005 | ≤0.75 |

- **effort_limit=1.3×额定预算 v3**（2026-09-17 操作员裁定：短时 130% 额定
  可接受；J3/J4 按硬件 28 封顶。w_torque 归一仍用额定——100~130% 段受
  二次软惩罚。历史教训保留：峰值 54/28/10 曾致行为惩罚淹没探索、训练发散）
- 增益随机化 DR：kp/kd ±20%、ki ±30%、effort ±10%（回合重置采样）；
- **反粘滑包（2026-09-17 晚）**：摩擦 DR 加宽 粘性 (0.05,0.6)/库仑 (0,1.0)、
  `fric_v_lin` 0.05→0.02（更尖锐 breakaway）、`w_action_rate/smooth` 0.05→0.10、
  `delay_prob` 0.3→0.45 —— 针对真机 35~55s 确定性粘滑振（仿真摩擦太温和
  是根因，三级部署旋钮无效后转训练侧）；
- 前馈表 gff = G(q_ref)+C(q_ref,q̇_ref) + 0.2·q̇_ref + 0.3·tanh(q̇_ref/0.05)
  （make_grav_ff.py v3, 真机 URDF 权威动力学, 与部署同一份 npz）。

## 5. 奖励函数（每步, 50Hz）

```
r = +2.0 · exp(−Σe²/(2·0.06²))                       # 跟踪(σ=3.4°)
    +0.6 · exp(−Σ(qd−q̇_ref)²/(2·0.40²))              # 速度剖面
    −0.05 · ‖Δa‖²                                     # 动作速率
    −0.05 · ‖a_t−2a_{t−1}+a_{t−2}‖²                   # 动作平滑
    （↑2026-09-17 反粘滑包: 两项已 0.05→0.10, 见 §4；延迟概率 0.3→0.45）
    −0.01 · Σ(q̈/100)²                                 # 加速度
    −0.0015 · Σ(τ/τ_额定)²                            # 力矩(归一用额定)
    −0.5  · Σ relu(|qd|−2.0)²                         # 速度上限
    −0.5  · Σ relu(0.1 − margin)²                     # 限位裕度
    −30.0 · Σ relu(|e|−0.035)²                        # 尾部加权(2°起征)
    −2.0  · max(tail)·relu(Δmax|e|/dt)                # 苗头抑制
    −1.0    (fail 终止时)
    +2.0 · exp(−Σe²/(2·0.06²))                        # 窗口完成 bonus
```
- e 均相对 q_ref[t−1]；fail: max|e|>0.30rad(17°) 终止；
- **w_tail/w_growth 曾经对照实验置 0（v7/v9, 引发动作 std 发散），
  2026-09-17 恢复 30/2 = 验证过的 11-47-10 同款**；
- 已废弃项: w_jerk=0（50Hz 二阶差分噪声爆炸）。

## 6. 训练算法（train.py 实际生效值）

| 项 | 值 |
|---|---|
| 算法 | PPO (rsl_rl 5.4.2, OnPolicyRunner) |
| rollout | 48 步/env × 2048 envs（~4s） |
| γ / λ | 0.999 / 0.95 |
| LR | 3e-4 **fixed**（adaptive 在收敛后 KL 走低 → LR 反复放大 →
  系统性衰减，两次独立复现；desired_kl 参数保留不生效） |
| entropy | 0.005 |
| clip | 0.2（value clipped, coef 1.0） |
| epochs / minibatches | 5 / 4, max_grad_norm 1.0 |
| 网络 | actor/critic 均 MLP[256,256,128] ELU, 无 obs 归一化 |
| 初始 std | 0.3（scalar, 可学习；1.0 会开局就撞 17° fail） |
| seed | 42 |
| 诊断 | train.py 每 50 迭代打印逐项奖励([dbg] 行) |

**已知训练现象**：adaptive LR 下奖励见顶后系统性衰减（两次复现，根因
=收敛后 KL 走低→LR 反复放大）→ **已改固定 LR + 课程化(方差项 300 迭代
ramp)+熵退火**，2026-09-18 课程化首训: 峰值 1059、全程最平滑策略
(qdd_rms 1.8)、真机 60s mean 1.16°/max 3.46°/τ峰值 0.3×额定/振动窗口
消除(std 3.8°→0.7°)、sim2real 系数重标定为 1.0/1.13 —— 稳步训练法
(课程化+固定LR+熵退火+checkpoint验证选取)自此为标准流程。

## 7. 场景/模型

- USD: `openarm_right_rl.usd`（夹爪 fixed 并入 link7——夹爪棱柱关节
  kp200 在 200Hz 数值发散；对应真机 ESC8 锁定）；
- 无地面（Grid USD 走 NVIDIA 云端致启动失败；臂固定于台座无坠落自由度）；
- 软限位/home: real_safety.yaml（2026-09-11 official-zero-v2 标定）；
- MJCF 孪生（MuJoCo 验证链用）: 质量+限位已由 sync_mjcf_inertia.py
  对齐权威 URDF/real_safety.yaml。

## 8. sim→real 映射表

| 仿真 | 真机 | 通道 |
|---|---|---|
| DmMITActuator 200Hz | zero_offset_hw write() 100Hz + 固件环 | MIT 五元组 |
| gff 表 ×s_env | rl_exec 发 t_ff（淡入1s/淡出1s） | 21 项消息第 15~21 项 |
| use_vel_ref→v_des | rl_exec 发 v_des | 21 项消息第 8~14 项 |
| fail_err 17° | watchdog 软限位外 2°×5帧 + estop | 安全网 |
| 增益 DR ±20% | 真机域内整定免重训 | verify_deploy 审计 |

## 9. 验证链（改训练后必跑）

1. `verify_deploy.py`（部署精确口径: rl_policy 同代码 + FF + 当前律）
   全绿才上机；
2. `pd_screen.py`（纯 PD 孪生筛选, 与 actor 无关, 换 PD 时跑）；
3. 真机分段: 15s → 60s → 全程, 黑匣子必录（J2 力矩 9~13Nm 判据）。

## 10. 教训清单（改代码前必读）

1. effort_limit 用额定不用峰值（峰值=行为惩罚淹没探索, 训练发散）；
2. 评估权威序 = rl_policy(部署代码) > rl_validate(简化复刻)；
3. 参考重生成 → 重跑 make_grav_ff.py + 同步两处副本 + verify_deploy 拦截；
4. make_mjcf.py 重跑 → 必须重跑 sync_mjcf_inertia.py（质量+限位）；
5. MJCF motor 只有 forcerange, ctrlrange=[0,0]（力矩钳位读 forcerange）；
6. rclpy 默认 SIGINT 会杀位置保持（rl_exec 已用 SignalHandlerOptions.NO 修复）；
7. 训练奖励 ~600 迭代后缓慢退化 → checkpoint 按验证选取；
8. 部署旋钮与训练成对: 训练 use_vel_ref=True ↔ rl_exec --vref 默认开；
   改一侧必须改另一侧并重训。
9. **注入通道完全分离(2026-09-18, 取代共用状态+切换架构)**: RL 与 MPC
   各自独立话题/指令状态/新鲜度(InjectSrc ×2), 源上升沿以实际位置重置
   基线(残留自动清除), 两源同时活跃=冻结+告警, 总断流限速滑回 JTC。
   历史: 共用 mpc_cmd_ 状态 + 残留指令 → 16:09 启动踢跳(τ 28Nm)等
   一系列事故, 结构性根因已消除。
10. **盲区仪器化**: 回家段指令 rl_exec_*_return.csv 单独记录; 黑匣子
    增加注入通道 c1~c7+指令龄列 —— 任何"臂动了不知谁发的指令"事故,
    对比 c 列与 q 列即可区分 指令源问题 vs 硬件/供电问题。
