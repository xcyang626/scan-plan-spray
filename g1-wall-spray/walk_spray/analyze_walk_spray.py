# -*- coding: utf-8 -*-
"""边走边喷实验效果分析: 骨盆路径/朝向漂移/枪尖世界轨迹/墙面覆盖.

用法: ~/miniconda3/envs/isaaclab/bin/python analyze_walk_spray.py [log.npz]
输出: walk_spray/fig_walk_spray_analysis.png + 控制台统计
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts", "exec"))
from arm_task_servo import fk_jac1, quat2rotm   # noqa: E402

LOG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "log", "walk_spray.npz")
GUN_LEN = 0.12
WALL_X = 0.74
BAND = (0.107, 0.617)          # 规划喷涂带(静止站立时的墙面 u 范围)
ROWS = (0.75, 1.45)            # 站姿行高度范围

z = np.load(LOG)
t, qp = z["t"], z["qpos"]
print(f"日志 {len(t)} 帧, t [{t[0]:.1f}, {t[-1]:.1f}]s")

def yaw_of(q):                  # (w,x,y,z)
    return np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]),
                      1 - 2 * (q[2] ** 2 + q[3] ** 2))

yaw = np.degrees(np.array([yaw_of(qp[k, 3:7]) for k in range(len(qp))]))
tip = np.zeros((len(qp), 3))
ok_fk = np.zeros(len(qp), bool)
for k in range(len(qp)):
    q10 = np.concatenate([qp[k, 19:22], qp[k, 29:36]])
    p_rel, a_rel, _, _ = fk_jac1(q10)
    Rb = quat2rotm(qp[k, 3:7])
    p_w = Rb @ p_rel + qp[k, 0:3]
    a_w = Rb @ a_rel
    tip[k] = p_w + GUN_LEN * a_w
    ok_fk[k] = True

# 朝向展开(-180..180 -> 连续)
yaw_c = np.unwrap(np.radians(yaw))
yaw_deg = np.degrees(yaw_c)

# 墙面覆盖: 枪轴朝 +x 且离墙不太远才算有效喷射
ax = tip - qp[:, 0:3]           # 方向近似(含枪长), 用轴精确算:
axs = np.zeros((len(qp), 3))
for k in range(len(qp)):
    q10 = np.concatenate([qp[k, 19:22], qp[k, 29:36]])
    _, a_rel, _, _ = fk_jac1(q10)
    axs[k] = quat2rotm(qp[k, 3:7]) @ a_rel
toward = axs[:, 0] > 0.3
dist = qp[:, 0]                 # 基座离墙原点 x
wall_y, wall_z, wall_t = [], [], []
for k in range(len(qp)):
    if not toward[k] or not np.isfinite(tip[k]).all():
        continue
    dx = WALL_X - tip[k, 0]
    if dx <= 0 or dx > 1.5:      # 枪在墙后或离墙太远
        continue
    s = dx / axs[k, 0]
    wall_y.append(tip[k, 1] + s * axs[k, 1])
    wall_z.append(tip[k, 2] + s * axs[k, 2])
    wall_t.append(t[k])
wall_y = np.array(wall_y); wall_z = np.array(wall_z); wall_t = np.array(wall_t)

# ---- 统计 ----
T = t[-1] - t[0]
dy = qp[-1, 1] - qp[0, 1]
print(f"骨盆: 起点 ({qp[0,0]:.2f},{qp[0,1]:.2f}) 终点 ({qp[-1,0]:.2f},{qp[-1,1]:.2f})")
print(f"位移: dx={qp[-1,0]-qp[0,0]:+.2f}m dy={dy:+.2f}m |v|={(np.hypot(qp[-1,0]-qp[0,0], dy)/T):.3f}m/s "
      f"(指令 0.1 前向x/左向y)")
print(f"yaw: 起 {yaw_deg[0]:+.1f}° 终 {yaw_deg[-1]:+.1f}° 最大|漂移| {np.abs(yaw_deg-yaw_deg[0]).max():.1f}°")
print(f"骨盆 z: min {qp[:,2].min():.3f} mean {qp[:,2].mean():.3f} (站立≈0.77, <0.35 视为摔倒)")
print(f"墙面有效喷射点 {len(wall_y)}/{len(qp)}; 墙面 y 覆盖 [{wall_y.min() if len(wall_y) else 0:.2f},"
      f"{wall_y.max() if len(wall_y) else 0:.2f}] (静止时应对齐带 {BAND})")

# ---- 图 ----
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
a1 = axes[0, 0]
a1.plot(qp[:, 1], qp[:, 0], "-", lw=1.2, label="骨盆")
a1.plot(tip[:, 1], tip[:, 0], "-", lw=0.8, alpha=0.8, label="枪尖")
a1.plot(qp[0, 1], qp[0, 0], "g^", ms=9, label="起点")
a1.plot(qp[-1, 1], qp[-1, 0], "rv", ms=9, label="终点")
a1.axvline(0, color="k", lw=0.5)
a1.set_xlabel("y 左向 (m)"); a1.set_ylabel("x 前向 (m)")
a1.set_title(f"俯视轨迹 (指令: 向左 0.1 m/s)"); a1.legend(); a1.axis("equal"); a1.grid(alpha=0.3)

a2 = axes[0, 1]
a2.plot(t, yaw_deg, lw=1.2)
a2.set_xlabel("t (s)"); a2.set_ylabel("yaw (°, 连续)")
a2.set_title(f"基座朝向漂移 (终 {yaw_deg[-1]:+.0f}°)"); a2.grid(alpha=0.3)

a3 = axes[1, 0]
sc = a3.scatter(wall_y, wall_z, c=wall_t, s=2, cmap="viridis")
a3.add_patch(plt.Rectangle((BAND[0], ROWS[0]), BAND[1]-BAND[0], ROWS[1]-ROWS[0],
                           fill=False, ec="r", ls="--", lw=1.5,
                           label="静止喷涂规划带"))
a3.set_xlabel("墙面 y (m)"); a2 and a3.set_ylabel("墙面高度 z (m)")
a3.set_title(f"枪尖墙面落点 (n={len(wall_y)}, 颜色=时间)")
a3.legend(); a3.grid(alpha=0.3)
plt.colorbar(sc, ax=a3, label="t (s)")

a4 = axes[1, 1]
a4.plot(t, qp[:, 2], lw=1.0)
a4.set_xlabel("t (s)"); a4.set_ylabel("骨盆 z (m)")
a4.set_title(f"骨盆高度 (min {qp[:,2].min():.2f})"); a4.grid(alpha=0.3)

fig.suptitle("边走边喷实验: 站立喷涂动作 + 左移 0.1 m/s (无世界系修正)", fontsize=13)
fig.tight_layout()
tag = os.path.splitext(os.path.basename(LOG))[0]
out = os.path.join(HERE, f"fig_{tag}_analysis.png")
fig.savefig(out, dpi=110)
print("图已存", out)
