#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""右臂之字轨迹腕部跳变分析: 定位大幅翻转段, 归因到 J5/J6/J7 与机制。"""
import json
import math
import sys
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "openarm_sim"))
sys.path.insert(0, os.path.join(_HERE, "..", "isaaclab_sim"))

from openarm_sim.kinematics import UrdfChain  # noqa: E402

DENSE = os.path.join(_HERE, "dense_zigzag_right.json")
URDF = os.path.join(_HERE, "..", "isaaclab_sim", "openarm_right.urdf")
WS = os.path.join(_HERE, "..", "isaaclab_sim", "workspace_right.json")

with open(DENSE, encoding="utf-8") as f:
    d = json.load(f)
P = np.array(d["points"])
N = len(P)
print("轨迹点数: %d" % N)

# --- 1) 逐关节单步跳变统计(度) ---
step_deg = np.degrees(np.abs(np.diff(P, axis=0)))
print("\n== 各关节单步最大跳变(°) ==")
for j in range(7):
    print("  J%d: max=%.1f  >30°步数=%d" % (j + 1, step_deg[:, j].max(),
                                            (step_deg[:, j] > 30).sum()))

# J7 角度范围检查(是否贴 ±180° 回绕风险)
print("\nJ5 范围: [%.1f, %.1f]°  J6: [%.1f, %.1f]°  J7: [%.1f, %.1f]°" % (
    np.degrees(P[:, 4]).min(), np.degrees(P[:, 4]).max(),
    np.degrees(P[:, 5]).min(), np.degrees(P[:, 5]).max(),
    np.degrees(P[:, 6]).min(), np.degrees(P[:, 6]).max()))

# --- 2) 定位最大跳变段 (综合指标: 7 关节最大单步) ---
worst = step_deg.max(axis=1)
order = np.argsort(worst)[::-1][:10]
print("\n== 跳变最大的 10 步 ==")
chain = UrdfChain(URDF, tip_link="openarm_right_hand_tcp")
for idx in order:
    q_a, q_b = P[idx], P[idx + 1]
    _, pa = chain.fk(q_a, 0.12)
    _, pb = chain.fk(q_b, 0.12)
    d_cart = float(np.linalg.norm(pb - pa))
    jdetail = " ".join("J%d=%.0f°" % (j + 1, step_deg[idx, j])
                       for j in range(7))
    print("  步@%d: TCP位移=%.1fmm | %s" % (idx, d_cart * 1000, jdetail))

# --- 3) 与上一段的窗口上下文: 定位问题点在矩形中的位置 ---
with open(WS, encoding="utf-8") as f:
    ws = json.load(f)
rect = ws["rectangle_m"]
print("\n矩形: y[%.3f,%.3f] z[%.3f,%.3f]" % (
    rect["y_min"], rect["y_max"], rect["z_min"], rect["z_max"]))

# 最大跳变处的 TCP 位置
i0 = int(order[0])
_, p_worst = chain.fk(P[i0 + 1], 0.12)
print("最大跳变步 @%d, TCP=(x=%.3f, y=%.3f, z=%.3f)  (y占矩形 %.0f%%, z占 %.0f%%)"
      % (i0, p_worst[0], p_worst[1], p_worst[2],
         100 * (p_worst[1] - rect["y_min"]) /
         (rect["y_max"] - rect["y_min"]),
         100 * (p_worst[2] - rect["z_min"]) /
         (rect["z_max"] - rect["z_min"])))

# --- 4) 腕部等价解检查: 问题点邻域解是否存在腕部连续替代解 ---
# wrist flip 等价变换: (J5+π, -J6, J7+π) 产生相同接近轴(对球腕)
# 逐点检查: 对最大跳变段前后各点, 构造等价解做 FK, 验证等价性
def ax_of(q):
    R, _ = chain.fk(q, 0.12)
    return R @ np.array([0.0, 0.0, 1.0])

qa, qb = P[i0], P[i0 + 1]
for tag, q in (("跳变前", qa), ("跳变后", qb)):
    q_alt = q.copy()
    q_alt[4] += math.pi      # J5+180
    q_alt[5] = -q_alt[5]     # J6 取反
    q_alt[6] += math.pi      # J7+180
    a1, a2 = ax_of(q), ax_of(q_alt)
    ang = math.degrees(math.acos(max(-1.0, min(1.0, float(a1 @ a2)))))
    print("  %s点 等价解(J5+180,J6取反,J7+180) 接近轴差: %.2f°  等价解J=[%s]"
          % (tag, ang, ", ".join("%.0f" % math.degrees(v) for v in q_alt)))

# --- 5) 时间预算核算: dt 公式与执行限速 ---
VEL = np.array([0.5, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8])
seg = np.array(d["seg_lens"])[1:]
dt_jt = np.abs(np.diff(P, axis=0)) / VEL
dt = np.maximum(dt_jt.max(axis=1), np.maximum(seg / 0.05, 0.02))
print("\n== 时间预算 ==")
print("  总时长(1x): %.1fs  其中 TCP 限速主导段占 %.0f%%" % (
    dt.sum(), 100 * (dt > seg / 0.05 + 1e-9).sum() / len(dt)))
top_t = np.argsort(dt)[::-1][:5]
print("  耗时最长的 5 段:")
for i in top_t:
    print("    步@%d dt=%.2fs (关节需求 %.2fs / TCP需求 %.2fs) 跳变J=%s"
          % (i, dt[i], dt_jt[i].max(), seg[i] / 0.05,
             ["J%d:%.0f°" % (j + 1, step_deg[i, j])
              for j in range(7) if step_deg[i, j] > 5]))

# 若不重定时(执行端固定 0.02s/步), 最大关节角速度
fixed_dt = 0.02
omega = step_deg / fixed_dt  # deg/s
print("\n若固定 %.0fms/步执行: 最大角速度 J5=%.0f°/s J6=%.0f°/s J7=%.0f°/s "
      "(限速对应 %.0f°/s)" % (
          fixed_dt * 1000, omega[:, 4].max(), omega[:, 5].max(),
          omega[:, 6].max(), math.degrees(0.8)))
