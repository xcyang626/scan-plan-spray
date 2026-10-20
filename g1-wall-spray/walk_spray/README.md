# 边走边喷实验（walk-while-spraying）

2026-10-13 首轮仿真实验。**独立目录，不污染原站立喷涂方案**：runner 以 import 方式复用
`scripts/exec/groot_spray_runner.py` 的全部机制（子类只叠加行走速度），状态文件重定向到本目录。

## 方案

- 动作：V3 站姿段（`stand_motion.npz`，2907 帧 / 96.9s，从 v6 双姿态动作按骨盆高切出，蹲姿段不参与）
- 行走：WBC `loco_cmd=[0, ±v, 0]`（+y=左；范数>0.05 自动切 walk 策略），v=0.1 m/s
- **无世界系修正**：任务伺服关闭，手臂=规划关节角原始注入；偏航保持保留（±0.045 rad/s 权限，只锁朝向不修手臂）
- `--walk-mode`：
  - `const`：恒定向左（第一轮）
  - `sync`：**腕横移方向同步**——手臂往左喷就往左走、往右喷往右走，行间竖直过渡停住
    （方向序列 6Hz 预计算：FK 腕心 y 数值微分 + 中值平滑 + 滞回三态化；实测 +1314/−1304/停289 帧，行交替完美平衡）

## 运行

```bash
cd walk_spray
VENV=~/桌面/轨迹规划/GR00T-WholeBodyControl/.venv_sim_cpu/bin/python
$VENV walk_spray_runner.py --motion stand_motion.npz --walk-v 0.1 --walk-mode sync \
    --headless --stand 4 --duration 170 --out log/walk_sync.npz
~/miniconda3/envs/isaaclab/bin/python analyze_walk_spray.py log/walk_sync.npz   # 出图+统计
```

## 结果（三轮，全部未摔倒：骨盆 z min ≥ 0.715）

| 实验 | 净漂移 | yaw 漂移 | 墙面落点 y 散布 | 图 |
|---|---|---|---|---|
| A 恒左 0.1 + 喷涂臂 | 螺旋（−2.6, −2.2）m | **+1609°** | ±4.5m | fig_walk_spray_analysis.png |
| B 恒左 0.1 + 零臂（对照） | (+4.0, +5.6)m 直线 | **2.0°** | ——（无臂） | fig_walk_only_analysis.png |
| C 同向往返 0.1 + 喷涂臂 | **(−0.1, +0.9)m 基本原地** | **+1442°** | ±3m | fig_walk_sync_analysis.png |

对照 B 证实：walk 策略侧移本身没问题（yaw 稳 2°，实际侧速 0.058 m/s=指令 58%，带 0.04 m/s 前向偏）。
A/C 的唯一共性瓶颈 = **喷涂手臂大幅横甩的角动量反作用把基座带转**（与 2026-10-10 站立复盘
"单右臂横甩 yaw 漂 +130~165°"同根；行走态偏航保持 ±0.045 rad/s 权限被碾压，~15°/s 自旋）。
用户"同向往返"方案（C）成功解决走远问题：净漂移 9.7m 预期 → 0.89m。

## 下一步选项（按代价排序）

1. **行走期放开 yaw_hold 权限**：行走态本来就在 walk 策略，把 loco_cmd[2] 限幅从 ±0.045 提到 ±0.3 —— 最便宜的试验；
2. **左臂镜像配重**：`groot_spray_runner.MIRROR_LEFT` 代码已存在但符号约定未修正（站立实测曾摔），需按 URDF 逐关节验证符号后启用 —— 角动量一阶抵消，治本；
3. 世界系手臂修正（本实验刻意关闭；即原任务伺服通道）；
4. 实机路线：SONIC planner 的 facing 朝向指令直接锁向（比 WBC yaw-rate 闭环强得多）。
