# mpc_track —— MoveIt2 + MPC 执行轨(实验)
> 主角: MPC 精确跟踪 MoveIt2/dense 参考路径。与主线(`real/config`)隔离,
> 驱动/CAN/MIT 链路完全复用, 仅新增一条指令注入通道。

## 架构

```
MoveIt2/dense 参考路径(dense_zigzag.json)
   → 参考生成器([home]+走线+[home] 时间参数化, 双约束: TCP+关节速率)
   → 闭式预览 MPC(逐关节双积分模型, N=25, dt=10ms,
      w_p=1000/w_v=200/w_a=1.0, a_max=1.2, 预览窗自动减速)
   → /left_mpc_position_commands (100Hz 指令流)
   → 驱动注入通道(新鲜<0.2s 时优先, 否则回落 JTC) → MIT 电机
安全: /safety/status != OK 或 estop → 立即停流 → JTC 保持 → watchdog 语义完整
```

## 模型说明
- 运动学模型来自 URDF(限位/轴系/d 换算已验证)。
- 动力学(重力)用**实测扰动补偿**替代: 驱动内软件积分器收敛值即
  所需重力前馈(免动力学库)。升级路线: pinocchio 动力学库 →
  真 NMPC(Acados) + PREEMPT_RT(参考 CSDN 163364973 路线 1)。

## 离线仿真验证(2026-09-07, 实机参数)
| 指标 | 主线(JTC+梯形) | MPC 轨 |
|---|---|---|
| 走线跟踪偏差 | 接近段 J4 短缺 19° | max 1.00° |
| 指令 HF σ (J1/J2) | 0.043/0.053 rad/s | 0.005/0.005 (≈8×) |
| 指令峰值速率 | 480°/s(翻转点) | 全程 <30°/s |
| 总时长 | 191s | 207s(TCP 不变, 翻转摊慢) |

## 真机使用
前置: 终端 A(新驱动) + 终端 B(watchdog) 在跑, 臂在 home。
```bash
cd ~/桌面/openarm实验/real/left/mpc_track
/usr/bin/python3 mpc_exec.py \
    --ref ../../../仿真/moveit_sim/dense_zigzag.json --speed-scale 0.4
```
- 终端 A 会出现 `指令源切换: MPC 流` —— 确认注入生效。
- 中止: Ctrl+C(停流, 臂原地保持) → estop → 物理急停。
- 完成后自动停流(臂保持 home)。

## 参数(mpc_config.json)
N=25, dt=0.01, w_p=1000, w_v=200, w_a=1.0, a_max=1.2,
tcp_speed=0.05, weave_joint_speed=0.3(翻转段关节速率),
approach/return_joint_speed=0.15/0.1, clearance_deg=3.0。
调参顺序: w_a(平滑) → w_p(贴合) → N(远见)。全部见文件内说明。
