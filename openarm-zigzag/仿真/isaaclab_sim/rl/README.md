# RL 轨迹提速 —— IsaacLab 训练 (右臂之字形走线)

> 目标: 用 RL 策略替换 MPC 执行轨, 在保证**平稳/平滑/无晃动**前提下把全程
> 走线从 ~300s 压到 **~160s**。设计权威文档:
> [`docs/RL轨迹提速_奖励与时间设计.md`](../../../docs/RL轨迹提速_奖励与时间设计.md)

## 目录

```
rl/
├── zigzag_ref.py         ① 参考轨迹生成 → ref_right.npz (速度层级+平滑时间轴)
├── make_rl_usd.py        ② 生成 RL 专用 USD (夹爪改 fixed, 只跑一次)
├── openarm_right_rl.usd  ② 产物: 7 自由度(夹爪并入 link7; 二进制, 不入库)
├── openarm_zigzag/       ③ RL 环境包
│   ├── zigzag_env.py        DirectRLEnv + 配置(奖励/随机化/观测)
│   └── mit_actuator.py      达妙 MIT 执行器复刻(与真机驱动同控制律)
├── train.py              ④ 训练 (rsl_rl PPO)
├── play.py               ⑤ 评估 (指标 + 曲线图, 写进该次 run 目录)
├── export_actor.py       ⑥ checkpoint → actor npz (纯 numpy, 无框架依赖)
├── ref_right.npz         ① 产物: 50Hz 参考 (q_ref/qd_ref/相位/元数据)
└── (剖面图)              ① 产物: 参考剖面图 → 归档在
                            assets/figures/ref_right_profile.png
```

## 用法(全部在 isaaclab conda 环境)

```bash
conda activate isaaclab
cd ~/桌面/openarm实验/仿真/isaaclab_sim/rl

# ① 参考轨迹(改动 dense_zigzag_right.json 或速度参数后重跑)
python zigzag_ref.py                  # → ref_right.npz + 剖面图, 打印核算
#   目标时长: --target-total 158 (默认) ; 平稳性门限超限会自动降速重生成

# ② RL 专用 USD(仅在 URDF/夹爪结构变化时需要; 已生成可跳过)
python make_rl_usd.py

# ④ 训练(2048 envs, ~40min/2000 iter, RTX 5080)
python train.py --headless --num_envs 2048 --max_iterations 2000

# ⑤ 评估(滚参考窗口, 出关节/TCP 误差 + 速度 HF σ)
python play.py --checkpoint logs/rsl_rl/openarm_zigzag/<run>/model_final.pt

# ⑥ 导出给 MuJoCo/真机
python export_actor.py --checkpoint logs/.../model_final.pt --out actor_right.npz
#   (train.py 结束时也会自动导出 actor_final.npz)
```

## 关键设计(详见 docs 文档)

- **速度层级在参考里, 不在奖励里**: 转移 0.12 rad/s(最慢) < 水平 4.7cm/s
  < 竖直 14cm/s(最快, 3×)。总时长 156.7s 由构造保证, 奖励只负责"跟得稳"。
- **平滑由数学保证**: Hann 滑窗圆滑几何(拐角 TCP 偏差 2.7mm) + 高斯卷积
  时间轴(速度处处连续、相位间斜坡过渡、首尾软起软停)。
- **观测 67 维**(全部真机可得, 无 privileged): q + qd + 上一动作 + 当前参考
  + 4 组参考预览(+0.04/0.08/0.16/0.32s) + 参考速度 + 进度 + 相位 one-hot。
- **动作 = 位置残差**: `q_cmd = q_ref(t) + 0.25·tanh(a)`, 钳位软限位−3°。
- **尾部加权奖励(2026-09-16, 安全优先)**: `w_tail·relu(|e|−2°)²` +
  苗头抑制 `w_growth·relu(Δ|e|max)·tail` —— 大偏离平方级问责、压苗头
  而非事后拉回; 与 fail_err 终止同向, 均匀平方惩罚对尾部不敏感的补强。
  现有 obs 已含 (q,q_ref,qd,qd_ref,+0.32s 预览), 策略具备提前动所需的
  全部信息, 故不改 obs 维度(避免 4 处部署代码同步风险)。
- **MIT 执行器复刻**: 与 `zero_offset_hw.cpp` 逐项一致(kp/kd、ki 积分、
  EMA α=0.3、±0.5rad 清零、积分钳位、力矩上限)。
- **重力前馈 + v_des(2026-09-16, 官方 MIT 五元组方案)**: `grav_ff_`
  `right.npz`(make_grav_ff.py **v2, 真机描述 URDF 权威动力学**——与
  训练 USD 同源; v1 的 MJCF 近似质量欠补偿 J2 达 42%, 已废弃)存在时
  训练 env 自动启用(逐 env 缩放 DR 0.8~1.2); `use_vel_ref=True` 下
  `set_joint_velocity_target(qd_ref)` 与部署 `rl_exec --vref` 同源
  —— MIT 律变为 kp·e_p + kd·(v_ref−v) + I + g(q_ref), t_ff/v_des
  走官方 MIT 五元组通道(驱动 21 项消息, 7/14/21 全兼容)。任一开关
  变更(前馈有无/vref)都必须重训并三侧同步; 参考轨迹重生成后必须
  重跑 make_grav_ff.py。

## 踩过的坑(改代码前必读)

1. **`replicate_physics` 必须 True** —— 本版本 GPU 管线下 False 会触发
   PhysX fabric device assert。
2. **GPU 管线不支持逐环境写 DOF 属性**(armature/摩擦) → 模型随机化只能放在
   执行器侧(kp/kd/ki/力矩限, Python 缓冲)。
3. **关节必须加折算转子惯量 `armature=0.02`** —— 否则腕部 J5–J7 零动作即
   发散到 20 rad/s(0.005 边缘, 0.01 起稳定)。
4. **夹爪必须 fixed**(`make_rl_usd.py`) —— 夹爪棱柱关节 0.017kg + kp200 在
   200Hz 下数值发散并污染整条求解; 与真机 ESC8 锁定语义一致。
5. **奖励核宽度按实际误差量级定** —— σ_track 太小(0.03)时误差 0.15rad 处
   奖励≈1e-6, 策略拿不到梯度; 现用 0.10。
6. **不要用关节 jerk 项** —— 50Hz 二阶差分噪声被平方放大(jerk_rms 4000+),
   会彻底主导奖励; 改用动作空间二阶平滑 `w_action_smooth`。
7. **崩溃后 Isaac Sim 进程不退出**会占满显存, 重跑前先按 nvidia-smi 清理。

## 与其余两条链路的一致性

| | IsaacLab 训练 | MuJoCo 验证 | 真机 |
|---|---|---|---|
| 参考 | `ref_right.npz` | 同 | 同 |
| 策略 | torch actor | `actor_right.npz`(numpy) | 同 |
| 执行器 | `DmMITActuator` | `rl_validate.py` MIT 100Hz | `zero_offset_hw.cpp` |
| 关节数 | 7(夹爪 fixed) | 7(夹爪位置执行器) | 8(ESC8 锁定) |

MuJoCo 验证: `仿真/mujoco_sim/rl_validate.py`; 真机: `real/right/rl_track/rl_exec.py`。
