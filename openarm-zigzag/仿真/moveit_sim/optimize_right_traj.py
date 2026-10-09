#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
右臂之字路点冗余度全局优化(离线)
================================
把整条之字路线的路点关节解作为优化变量(N×7), 用 scipy least_squares
(TRF, 带硬边界)全局求解:
  残差 = w_pos×位置误差(3/点) + w_ax×接近轴正交误差(3/点)
         + w_sm×相邻路点关节移动量(7/步)          ← 连贯性的来源
  边界 = 执行盒(real_safety 软限位内缩 3°)硬约束, 永不越限。
初值 = workspace_right.json 图集解中距各路点最近者(同支热启动)。

这是对"贪心逐步解算"的根本性替代: 贪心解每步局部最优, 会在 y≈0 分岔、
腕部限位等处锁死; 全局优化把平滑性作为目标一次分配到整条轨迹。

用法: /usr/bin/python3 optimize_right_traj.py [--zmin ..] [--ymax ..]
                                             [--ymin ..] [--zmax ..]
产出: optimized_waypoints_right.json (路点 + 优化解 + 误差报告)
      供 make_dense_zigzag_right.py 原生模式自动取用。
"""

import json
import math
import os
import sys

import numpy as np
from scipy.optimize import least_squares

_HERE = os.path.dirname(os.path.abspath(__file__))
_ISA_DIR = os.path.abspath(os.path.join(_HERE, "..", "isaaclab_sim"))
_PKG_DIR = os.path.abspath(os.path.join(_HERE, "..", "openarm_sim"))
sys.path.insert(0, _ISA_DIR)
sys.path.insert(0, _PKG_DIR)

from openarm_sim.kinematics import UrdfChain     # noqa: E402

RIGHT_WS = os.path.join(_ISA_DIR, "workspace_right.json")
RIGHT_URDF = os.path.join(_ISA_DIR, "openarm_right.urdf")
_SAFETY_R = os.path.abspath(os.path.join(
    _HERE, "..", "..", "real", "right", "config", "real_safety.yaml"))
OUT = os.path.join(_HERE, "optimized_waypoints_right.json")

TCP_OFF = 0.12
W_POS, W_AX, W_SM = 30.0, 6.0, 1.0   # 残差权重: 任务主导, 平滑参与分配


def arg_val(name, cast=float):
    if name in sys.argv:
        return cast(sys.argv[sys.argv.index(name) + 1])
    return None


def load_right_limits():
    import yaml
    with open(_SAFETY_R, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    lim = cfg["soft_limits_rad"]
    names = ["openarm_right_joint%d" % i for i in range(1, 8)]
    if isinstance(lim, dict):
        arr = np.array([[float(lim[n][0]), float(lim[n][1])] for n in names])
    else:
        arr = np.array([[float(l[0]), float(l[1])] for l in lim])
    c = math.radians(3.0)
    arr[:, 0] += c
    arr[:, 1] -= c
    return arr


def main():
    zmin = arg_val("--zmin")
    zmax = arg_val("--zmax")
    ymin = arg_val("--ymin")
    ymax = arg_val("--ymax")

    with open(RIGHT_WS, encoding="utf-8") as f:
        ws = json.load(f)
    atlas = {(r[0], r[1]): np.array(r[2:9])
             for r in ws.get("admitted_solutions", [])}
    sols_scan = [np.array(q) for q in ws["zigzag_joint_solutions_rad"]]
    lim = load_right_limits()
    rect = dict(ws["rectangle_m"])
    if zmin is not None:
        rect["z_min"] = max(rect["z_min"], zmin)
    if zmax is not None:
        rect["z_max"] = min(rect["z_max"], zmax)
    if ymin is not None:
        rect["y_min"] = max(rect["y_min"], ymin)
    if ymax is not None:
        rect["y_max"] = min(rect["y_max"], ymax)
    print("矩形: y[%.3f, %.3f] z[%.3f, %.3f]  %.1f×%.1fcm" % (
        rect["y_min"], rect["y_max"], rect["z_min"], rect["z_max"],
        (rect["y_max"] - rect["y_min"]) * 100,
        (rect["z_max"] - rect["z_min"]) * 100))

    chain = UrdfChain(RIGHT_URDF, tip_link="openarm_right_hand_tcp")
    axis_t = np.array(ws["target_axis"], dtype=float)
    axis_t = axis_t / np.linalg.norm(axis_t)

    # 之字路点(行距 3cm 往返, 与扫描同款)
    y0, z0 = rect["y_min"], rect["z_min"]
    y1, z1 = rect["y_max"], rect["z_max"]
    spacing = ws.get("zigzag_spacing_m", 0.03)
    wps = []
    z = z0
    direction = 1.0
    while z <= z1 + 1e-9:
        wps.append(np.array([ws["plane_x"], y0 if direction > 0 else y1, z]))
        wps.append(np.array([ws["plane_x"], y1 if direction > 0 else y0, z]))
        direction *= -1.0
        z += spacing
    N = len(wps)
    print("路点 %d 个 (N×7 = %d 优化变量)" % (N, N * 7))

    # 热启动优先级:
    # A) --seed-mirror <旧优化产物>: 旧链是已验证的同支平滑链, 符号翻转
    #    (J1/J7)下同一物理位形满足 new_J = −old_J(其余不变), 镜像+裁剪后
    #    即新坐标系的同族种子(2026-09-09: 这是唯一靠得住的跨符号种子——
    #    图集逐点最近会跨族, 顺序贪心会撞贴中线陡梯度带)。
    # B) 图集顺序贪心(保连通), 仍失败的路点软约束兜底。
    seed_file = None
    if "--seed-mirror" in sys.argv:
        seed_file = sys.argv[sys.argv.index("--seed-mirror") + 1]
    x0 = np.zeros((N, 7))
    prev = None
    if seed_file and os.path.isfile(seed_file):
        with open(seed_file, encoding="utf-8") as f:
            old = json.load(f)
        old_q = np.array(old["solutions"], dtype=float)
        assert len(old_q) == N, "旧链路点数 %d != 当前 %d" % (len(old_q), N)
        for k in range(N):
            m = old_q[k].copy()
            m[0], m[6] = -m[0], -m[6]     # 翻转关节 = 同一物理位形
            x0[k] = np.clip(m, lim[:, 0], lim[:, 1])
        print("种子: 旧优化链镜像(--seed-mirror %s)" % os.path.basename(seed_file))
    else:
        SEED_STEP_MAX = math.radians(25.0)    # 略大于图集 20°/格连通门限
        SEED_RING = 12                        # 邻域搜索圈数(格, 1cm/格)
        cell_of = lambda w: (int(round((w[1] - ws["grid"]["y_lo"]) / 0.01)),
                             int(round((w[2] - ws["grid"]["z_lo"]) / 0.01)))
        for k, w in enumerate(wps):
            cy, cz = cell_of(w)
            cand = []
            for r in range(0, SEED_RING + 1):
                for a in range(-r, r + 1):
                    for b in range(-r, r + 1):
                        if max(abs(a), abs(b)) != r:
                            continue          # 只加外圈, 避免重复
                        q = atlas.get((cy + a, cz + b))
                        if q is not None:
                            cand.append(q)
            if prev is None:
                pool, gate_hit = cand, True
            else:
                pool = [q for q in cand
                        if float(np.abs(q - prev).max()) <= SEED_STEP_MAX]
                gate_hit = bool(pool)
                if not pool:
                    # 门限内无解: 软约束兜底——步长超限部分按距离惩罚
                    pool = cand
            best, bd = None, 1e9
            for q in pool:
                _, p = chain.fk(q, TCP_OFF)
                d = float(np.linalg.norm(p - w))
                if prev is not None and not gate_hit:
                    over = float(np.abs(q - prev).max()) - SEED_STEP_MAX
                    if over > 0:
                        d += 0.05 * over      # 0.05m/rad: 强但非硬的连通偏置
                if d < bd:
                    best, bd = q, d
            if best is None or bd > 0.05:     # 图集盲区: 最近扫描路点解兜底
                d2 = [float(np.linalg.norm(sw - w)) for sw in wps]
                best = sols_scan[int(np.argmin(d2))]
                print("  ⚠ 路点 %d 图集无近解(TCP 距 %.0fmm), 用扫描路点解兜底"
                      % (k, bd * 1000 if best is not None else -1))
            x0[k] = np.clip(best, lim[:, 0], lim[:, 1])
            prev = x0[k]
    seed_steps = np.abs(np.diff(x0, axis=0))
    print("种子链相邻路点最大移动(°/3cm):",
          {("J%d" % (j + 1)): round(float(np.degrees(seed_steps[:, j]).max()), 1)
           for j in range(7)})

    def fk_axis(q):
        R, p = chain.fk(q, TCP_OFF)
        return p, R @ np.array([0.0, 0.0, 1.0])

    def make_residuals(w_pos, w_ax, w_sm):
        def residuals(x):
            Q = x.reshape(N, 7)
            res = []
            for k in range(N):
                p, a = fk_axis(Q[k])
                res.extend(w_pos * (p - wps[k]))
                e_a = a - axis_t
                e_a = e_a - axis_t * float(e_a @ axis_t)  # 轴正交分量
                res.extend(w_ax * e_a)
            dQ = (Q[1:] - Q[:-1]).ravel()
            res.extend(w_sm * dQ)
            return np.asarray(res)
        return residuals

    x0 = x0.ravel()          # 每行已按执行盒裁剪
    print("优化中(trust-region reflective, 硬边界=执行盒)...")
    lo_b = np.tile(lim[:, 0], N)
    hi_b = np.tile(lim[:, 1], N)
    # 同伦续接: 先硬拟合任务 → 平滑权重分档渐增, 轨迹在任务流形内
    # 渐进弯曲(局部优化器跨不了姿态族鸿沟, 续接保证始终不出族)
    x = x0
    sol = None
    for w_pos, w_ax, w_sm in ((300.0, 60.0, 0.0),
                              (100.0, 20.0, 0.15),
                              (60.0, 12.0, 0.6),
                              (30.0, 6.0, 1.0),
                              (30.0, 6.0, 0.0)):   # 末级纯任务投影
        res_f = make_residuals(w_pos, w_ax, w_sm)
        sol = least_squares(res_f, x, bounds=(lo_b, hi_b),
                            method="trf", x_scale=0.1, verbose=0,
                            max_nfev=2000)
        x = sol.x
        print("  阶段 w=(%g,%g,%g): cost=%.5f" % (w_pos, w_ax, w_sm, sol.cost))
    Q = np.clip(sol.x.reshape(N, 7), lim[:, 0], lim[:, 1])

    # 误差报告
    max_pos = max_ax = 0.0
    for k in range(N):
        p, a = fk_axis(Q[k])
        max_pos = max(max_pos, float(np.linalg.norm(p - wps[k])))
        ax_err = math.acos(max(-1.0, min(1.0, float(a @ axis_t))))
        max_ax = max(max_ax, ax_err)
    steps = np.abs(np.diff(Q, axis=0))
    print("优化完成: cost=%.6f  位置最大误差 %.2fmm  轴最大误差 %.2f°" % (
        sol.cost, max_pos * 1000, math.degrees(max_ax)))
    print("相邻路点最大关节移动(°/3cm):",
          {("J%d" % (j + 1)): round(float(np.degrees(steps[:, j]).max()), 1)
           for j in range(7)})

    out = {
        "note": "全局冗余度优化产物(optimize_right_traj.py); 路点与 "
                "workspace_right.json 同源, 解为全局平滑解",
        "rect": rect,
        "plane_x": ws["plane_x"],
        "target_axis": list(axis_t),
        "zigzag_spacing_m": spacing,
        "waypoints": [w.tolist() for w in wps],
        "solutions": [q.tolist() for q in Q],
        "report": {"max_pos_err_m": max_pos,
                   "max_axis_err_deg": math.degrees(max_ax),
                   "max_wp_step_deg": [float(np.degrees(steps[:, j]).max())
                                       for j in range(7)]},
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("已写出", OUT)


if __name__ == "__main__":
    main()
