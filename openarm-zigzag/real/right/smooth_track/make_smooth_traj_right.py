#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""smooth_track 右臂版: 时间轴平滑 → smooth_zigzag_right.json
================================================================
背景(左臂 2026-09-07 黑匣子实测, 右臂 2026-09-09 真机复现): 走线
"一卡一卡"的主频 = dense 每 5mm 路点的重定时台阶(Δq/dt 逐点跳变,
_retiming 只限速不限加速度, 行末方向反转处速度直接跳变)。

方案(与左臂 smooth_track 同源, 真机已验证): 路径保持原样(逐点不变
→ TCP 偏差恒为 0), 只平滑"时间轴":
  1. 轻度 SG(窗 5) 滤掉 IK 级微小抖动(偏差 <2mm 校验);
  2. dt 序列滑动平均(窗 9) + 总时长守恒 → 速度曲线连续;
  3. 输出 pretimed 格式, zigzag_moveit 检测到后直接使用, 不再重定时。

⚠ 时间轴已按 SCALE=0.8 烘焙(二档提速)。zigzag_moveit 的
  --speed-scale 对 pretimed 轨无效——换速度须改 SCALE 重跑本脚本。
  0.4 首跑版备份: smooth_zigzag_right.json.bak_20260914_scale0.4首跑
安全: 软限位/跳变/TCP 偏差全程校验。用法:
  /usr/bin/python3 make_smooth_traj_right.py [--sg-window 5] [--dt-window 9]
产出: 本目录 smooth_zigzag_right.json
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import yaml
from scipy.signal import savgol_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
_DENSE = os.path.abspath(os.path.join(
    _HERE, "..", "..", "..", "仿真", "moveit_sim", "dense_zigzag_right.json"))
_SAFETY = os.path.abspath(os.path.join(
    _HERE, "..", "config", "real_safety.yaml"))
_ISA = os.path.abspath(os.path.join(
    _HERE, "..", "..", "..", "仿真", "isaaclab_sim"))
_PKG = os.path.abspath(os.path.join(
    _HERE, "..", "..", "..", "仿真", "openarm_sim"))
NAMES = ["openarm_right_joint%d" % i for i in range(1, 8)]
CLEARANCE_DEG = 3.0
VEL_CAPS = [0.5, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8]   # 与主线 _retiming 一致
TCP_SPEED = 0.05
SCALE = 0.8


def smooth_velocity_profile(dq_max, dt_raw, win):
    """标量速度剖面平滑(位置不动, 只动时间):
    v_raw_i = dq_max_i/dt_raw_i → 滑动平均 → dt_new_i = dq_max_i/v_s_i。
    总时长均匀缩放回原值; v 下限 0.05 rad/s 防 dt 爆炸。"""
    v_raw = dq_max / dt_raw
    k = np.ones(win) / win
    wsum = np.convolve(np.ones_like(v_raw), k, mode="same")
    v_s = np.convolve(v_raw, k, mode="same") / np.maximum(wsum, 1e-9)
    v_s = np.clip(v_s, 0.05, None)
    dt_new = dq_max / v_s
    dt_new = np.maximum(dt_new, 0.02)
    dt_new *= dt_raw.sum() / dt_new.sum()
    return dt_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sg-window", type=int, default=5,
                    help="路径微抖 SG 窗长(奇数, 默认 5≈2.5cm)")
    ap.add_argument("--dt-window", type=int, default=9,
                    help="时间轴平滑窗长(点数, 默认 9)")
    args = ap.parse_args()
    if args.sg_window not in (0,) and (args.sg_window % 2 == 0
                                       or args.sg_window < 5):
        sys.exit("sg-window 必须为奇数且 >=5, 或 0(关闭)")

    sys.path.insert(0, _ISA)
    sys.path.insert(0, _PKG)
    from openarm_sim.kinematics import UrdfChain
    chain = UrdfChain(os.path.join(_ISA, "openarm_right.urdf"),
                      tip_link="openarm_right_hand_tcp")

    dense = json.load(open(_DENSE, encoding="utf-8"))
    P = np.array(dense["points"])
    seg = np.array(dense["seg_lens"])
    cfg = yaml.safe_load(open(_SAFETY, encoding="utf-8"))
    lim = cfg["soft_limits_rad"]
    lo = np.array([lim[n][0] for n in NAMES])
    hi = np.array([lim[n][1] for n in NAMES])
    c = math.radians(CLEARANCE_DEG)

    # 1. 路径保持原样(实测: 走线尖角几何, 任何空间平滑都会切角);
    #    --sg-window >=5 时才启用 SG(默认 5: 左臂真机验证配置)
    if args.sg_window >= 5:
        S = np.array([savgol_filter(P[:, k], args.sg_window, 3)
                      for k in range(7)]).T
        S[0], S[-1] = P[0], P[-1]
        S = np.clip(S, lo + c, hi - c)
    else:
        S = P.copy()

    # 2. 原始 dt(与主线 _retiming 同式) → 速度剖面平滑 → 反推 dt
    dt = np.zeros(len(P))
    for i in range(1, len(P)):
        dtj = max(abs(P[i][k] - P[i-1][k]) / VEL_CAPS[k] for k in range(7))
        dtt = seg[i] / TCP_SPEED
        dt[i] = max(dtj, dtt, 0.02) / SCALE
    dq_max = np.abs(np.diff(P, axis=0)).max(axis=1)     # 每段最坏关节(rad)
    dt_s = smooth_velocity_profile(dq_max, dt[1:], args.dt_window)
    dt_s = np.concatenate(([max(dt[1] / SCALE, 0.05)], dt_s))
    times = np.cumsum(dt_s)

    # 3. 速度台阶量化对比(平滑前/后): 相邻段指令速率变化
    def rate_steps(pos, dtv):
        v = []
        for i in range(1, len(pos)):
            r = max(abs(pos[i][k] - pos[i-1][k])
                    for k in range(7)) / dtv[i-1]
            v.append(r)
        return np.abs(np.diff(v))
    steps0 = rate_steps(P, dt[1:])
    steps1 = rate_steps(S, dt_s)
    print("==== smooth_track 校验 ====")
    print("SG 窗 %d | dt 窗 %d | 点数 %d | 总时长 %.0fs → %.0fs"
          % (args.sg_window, args.dt_window, len(P),
             dt[1:].sum(), dt_s.sum()))
    print("速度台阶(相邻段指令速率变化): 平滑前 mean=%.2f°/s max=%.1f°/s"
          " → 平滑后 mean=%.2f°/s max=%.1f°/s"
          % (math.degrees(steps0.mean()), math.degrees(steps0.max()),
             math.degrees(steps1.mean()), math.degrees(steps1.max())))
    # 4. 校验: 限位 / TCP 偏差(SG 后 vs 原路径) / 跳变
    dev = []
    for i in range(0, len(S), 10):
        _, pa = chain.fk(S[i], 0.12)
        dmin = 1e9
        for j in range(max(0, i-4), min(len(P), i+5)):
            _, pb = chain.fk(P[j], 0.12)
            dmin = min(dmin, float(np.linalg.norm(pa - pb)))
        dev.append(dmin)
    dmax = max(dev)
    margin = min((S.min(axis=0) - lo).min(), (hi - S.max(axis=0)).min())
    jump = np.degrees(np.abs(np.diff(S, axis=0)).max(axis=0))
    print("TCP 平滑偏差: max=%.2f mm" % (dmax * 1000))
    print("软限位最小裕量: %.2f°" % math.degrees(margin))
    print("单点最大跳变: %s°" % ["%.1f" % v for v in jump])
    if dmax * 1000 > 2.0:
        sys.exit("TCP 偏差超 2mm, 减小 --sg-window 重试")

    out = os.path.join(_HERE, "smooth_zigzag_right.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"points": S.tolist(), "seg_lens": seg.tolist(),
                   "pretimed": True,
                   "times": [round(float(v), 4) for v in times],
                   "step_m": float(np.mean(seg[1:])),
                   "smoothed": {"sg_window": args.sg_window,
                                "dt_window": args.dt_window,
                                "scale": SCALE}}, f)
    print("已写出", out)


if __name__ == "__main__":
    main()
