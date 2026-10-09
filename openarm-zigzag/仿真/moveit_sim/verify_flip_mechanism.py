#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补充验证: 跳变步与 (a)J6过零/奇异 (b)zigzag段边界(图集种子触发点) 的对应关系。"""
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "openarm_sim"))
sys.path.insert(0, os.path.join(_HERE, "..", "isaaclab_sim"))
from openarm_sim.kinematics import UrdfChain  # noqa: E402

DENSE = os.path.join(_HERE, "dense_zigzag_right.json")
WS = os.path.join(_HERE, "..", "isaaclab_sim", "workspace_right.json")
URDF = os.path.join(_HERE, "..", "isaaclab_sim", "openarm_right.urdf")

with open(DENSE, encoding="utf-8") as f:
    d = json.load(f)
P = np.array(d["points"])
seg = np.array(d["seg_lens"])[1:]
with open(WS, encoding="utf-8") as f:
    ws = json.load(f)
rect = ws["rectangle_m"]
chain = UrdfChain(URDF, tip_link="openarm_right_hand_tcp")

# 段边界: zigzag 路点间隔 seg_lens 恒定(3cm/6mm->5mm细分), 段长度=每行长度
# 每段 nsub 步, 由 seg_lens 单值判断: seg_lens 全为行长度/nsub
step_m = d.get("step_m", 0.005)
row_len = None
uniq = np.unique(np.round(seg, 6))
print("seg_lens 唯一值: %s (步长 %.1fmm)" % (
    [round(float(u) * 1000, 1) for u in uniq], step_m * 1000))

# 重建 zigzag 路点行边界: 每行 y0->y1 或 y1->y0, 行长 = |y1-y0|
row_len = (rect["y_max"] - rect["y_min"])
nsub = int(round(row_len / step_m))
n_steps_per_row = nsub
print("行长约 %.1fcm, 每行 %d 步" % (row_len * 100, n_steps_per_row))

# 跳变步 -> 行号/行内位置
step_deg = np.degrees(np.abs(np.diff(P, axis=0)))
jump_idx = np.where(step_deg.max(axis=1) > 30)[0]
print("\n== 跳变步(>30°) 定位 ==")
tcp = np.array([chain.fk(q, 0.12)[1] for q in P])
for i in jump_idx:
    row = int(i // n_steps_per_row)
    in_row = i % n_steps_per_row
    frac = in_row / n_steps_per_row
    going = "y0->y1" if row % 2 == 0 else "y1->y0"
    print("  步@%d: 第%d行(%s) 行内 %.0f%%  TCP=(y=%.3f,z=%.3f)  "
          "J5=%.0f° J6=%.1f° J7=%.0f°"
          % (i, row + 1, going, frac * 100,
             tcp[i + 1][1], tcp[i + 1][2],
             step_deg[i, 4], step_deg[i, 5], step_deg[i, 6]))

# J6 邻域行为: 跳变步前后 J6 轨迹与 0° 关系 + 限位距离
print("\n== J6 在跳变步附近的轨迹(°) ==")
for i in jump_idx[:3]:
    lo, hi = max(0, i - 6), min(len(P), i + 7)
    print("  步@%d J6: %s" % (i, " ".join(
        "%.1f" % math.degrees(P[k, 5]) for k in range(lo, hi))))
    print("        J5: %s" % " ".join(
        "%.0f" % math.degrees(P[k, 4]) for k in range(lo, hi)))
    print("        J7: %s" % " ".join(
        "%.0f" % math.degrees(P[k, 6]) for k in range(lo, hi)))

# 图集解在该处的腕部相角(种子污染检查)
atlas = {(r[0], r[1]): np.array(r[2:9])
         for r in ws.get("admitted_solutions", [])}
gy, gz = ws["grid"]["y_lo"], ws["grid"]["z_lo"]
print("\n== 跳变步处图集解(种子) vs 轨迹解 的腕部差 ==")
for i in jump_idx[:5]:
    cy = int(round((tcp[i + 1][1] - gy) / 0.01))
    cz = int(round((tcp[i + 1][2] - gz) / 0.01))
    s = None
    for r in range(0, 4):
        for a in range(-r, r + 1):
            for b in range(-r, r + 1):
                s = atlas.get((cy + a, cz + b))
                if s is not None:
                    break
            if s is not None:
                break
        if s is not None:
            break
    if s is None:
        print("  步@%d: 无图集解" % i)
        continue
    d5 = math.degrees(abs(s[4] - P[i + 1, 4]))
    d6 = math.degrees(abs(s[5] - P[i + 1, 5]))
    d7 = math.degrees(abs(s[6] - P[i + 1, 6]))
    d_arm = math.degrees(np.abs(s[:4] - P[i + 1, :4]).max())
    print("  步@%d: 图集解与轨迹解 差 J1-J4=%.0f° J5=%.0f° J6=%.0f° J7=%.0f°"
          % (i, d_arm, d5, d6, d7))
