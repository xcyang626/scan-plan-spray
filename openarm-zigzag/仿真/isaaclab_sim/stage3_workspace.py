#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段3: 工作区间扫描 + 长方形选取 + 之字形路径生成（纯 numpy，无 Isaac 依赖）
==========================================================================
约束（用户确认）:
  - TCP 位于基座正前方 x = 0.30 m 的竖直平面上（与 2m 墙平行）；
  - 夹爪垂直于墙: TCP 接近轴对准 +x（指向墙）；
  - 全部关节在 zero 文件实测限位内（URDF 系换算值）。

方法:
  1. 自适应定界: 关节空间随机采样 + FK（仅位置约束）估计平面内可达范围；
  2. 网格扫描: 平面网格化（1cm），逐点 warm-start DLS IK（含 5D 轴约束），
     失败格随机重启兜底；
  3. 假洞修复: 只与可达格相邻的失败格反复"邻域解传播"（相邻解做种子 +
     随机重启），直到收敛——2026-09-07 复查确认原扫描留下的中间空洞
     148/148 格全是 DLS 局部极小假象，单个最大内接矩形被它挤成窄条；
  4. 连续支泛洪: 从扫描链起始格 BFS, 只承认"邻格解 warm start 且臂关节
     跳变 ≤20°"的格 → 同支连通执行域(矩形域内保证无构型支跳变点);
  5. 最大内接长方形: 同支域腐蚀 3cm 安全边距后取最大面积矩形；
  6. 之字形路径: 长方形内行距 3cm 往返扫描，输出路径点 JSON。

限位: 默认用 real_safety.yaml 软限位内缩 3°（= make_dense_zigzag /
make_smooth_traj 的真机执行限位），扫描可达 = 执行可达，矩形边界不再
出现"扫描可行、真机 watchdog 取消"；yaml 缺失时回退 URDF 全行程(仅仿真)。

产出: workspace.json / assets/figures/workspace_<side>.svg（可达图+长方形+之字形）/ 控制台报告
"""

import json
import math
import os
import sys
from collections import deque

import numpy as np

_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.abspath(os.path.join(_SIM_DIR, "..", "openarm_sim"))
sys.path.insert(0, _PKG_DIR)
sys.path.insert(0, _SIM_DIR)

from common import (ARM_URDF, SIDE, ZERO_FILE, load_bridge,  # noqa: E402
                    load_chain)
from openarm_sim.ik_solver import DlsIK  # noqa: E402


def _fig_dir():
    """图件输出目录: 仓库根 assets/figures/（图件集中归档, 不散落在代码旁）。

    从本文件向上找含 assets/figures 的目录; 不在仓库内时退回本文件所在目录。
    """
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        cand = os.path.join(d, "assets", "figures")
        if os.path.isdir(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            return os.path.dirname(os.path.abspath(__file__))
        d = parent


PLANE_X = 0.30          # TCP 工作平面
TARGET_AXIS = (1.0, 0.0, 0.0)   # 接近轴: 指向墙 (+x)
CELL = 0.01             # 网格步长 1cm
MARGIN_CELLS = 3        # 长方形安全边距 3cm
ZIGZAG_SPACING = 0.03   # 之字形行距 3cm

# 真机执行限位来源(与 make_dense_zigzag.py 的 WATCHDOG_CLEARANCE 同值):
# 软限位再内缩 3°, 覆盖 watchdog 触发余量 2°(trip_margin) + ~1° 跟踪噪声
# (双臂重构 2026-09-08: 按 --side 取 real/<side>/config/real_safety.yaml)
WATCHDOG_CLEARANCE = 3.0   # deg
SCAN_RESTARTS = 6          # 网格扫描失败格随机重启次数
REPAIR_ROUNDS = 80         # 假洞修复轮数上限
REPAIR_RESTARTS = 10       # 修复阶段单格随机重启次数


def _side_and_paths():
    """解析 --side(默认 left), 返回 (side, suffix, safety_yaml 路径)。"""
    side = "left"
    if "--side" in sys.argv:
        side = sys.argv[sys.argv.index("--side") + 1]
    suffix = "" if side == "left" else "_right"
    safety = os.path.abspath(os.path.join(
        _SIM_DIR, "..", "..", "real", side, "config", "real_safety.yaml"))
    return side, suffix, safety


def urdf_limits(bridge):
    return np.array([[bridge.motor_to_urdf(bridge.motor_limits_rad[i][0], i),
                      bridge.motor_to_urdf(bridge.motor_limits_rad[i][1], i)]
                     for i in range(7)])


def exec_limits(bridge, safety_yaml, side):
    """扫描限位 = 真机执行限位: real_safety.yaml 软限位内缩 3°(URDF 系)。
    与稠密轨迹/平滑阶段的关节盒完全一致, 矩形内每个格子真机都可直接执行。"""
    import yaml
    if not os.path.isfile(safety_yaml):
        if side != "left":
            raise SystemExit("右臂必须有 real_safety.yaml(先 gen_real_config "
                             "--side right), 无 zero 桥接可回退")
        print("[警告] 找不到 %s, 回退 URDF 全行程限位(仅仿真演示用)"
              % safety_yaml)
        return urdf_limits(bridge)
    with open(safety_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    lim = cfg["soft_limits_rad"]
    names = ["openarm_%s_joint%d" % (side, i) for i in range(1, 8)]
    if isinstance(lim, dict):
        arr = np.array([[float(lim[n][0]), float(lim[n][1])] for n in names])
    else:
        arr = np.array([[float(l[0]), float(l[1])] for l in lim])
    c = math.radians(WATCHDOG_CLEARANCE)
    arr[:, 0] += c
    arr[:, 1] -= c
    return arr


def solve_robust(ik, rng, limits, seed, tgt, axis, restarts):
    """warm start 优先, 失败随机重启兜底(与 make_dense_zigzag 同策略)。"""
    q, ok, _ = ik.solve(seed, tgt, axis)
    if ok:
        return q, True
    for _ in range(restarts):
        q2, ok, _ = ik.solve(rng.uniform(limits[:, 0], limits[:, 1]),
                             tgt, axis)
        if ok:
            return q2, True
    return q, False


def main():
    side, suffix, safety_yaml = _side_and_paths()
    bridge = load_bridge() if side == "left" else None
    if side == "left":
        chain = load_chain(ARM_URDF)
    else:
        # 右臂链: 显式 tip(common.load_chain 写死左臂 tip link)
        from openarm_sim.kinematics import UrdfChain
        chain = UrdfChain(os.path.join(_SIM_DIR, "openarm_right.urdf"),
                          tip_link="openarm_right_hand_tcp")
    limits = exec_limits(bridge, safety_yaml, side)
    print("扫描侧: %s | 限位: %s" % (side, safety_yaml))
    ik = DlsIK(chain, limits, tcp_offset_m=0.12)
    rng = np.random.default_rng(42)

    axis = np.array(TARGET_AXIS)
    # 种子位形: 各关节限位中点附近的"迎墙"姿态
    q_mid = limits.mean(axis=1)

    # --reuse-occupancy: 复用现有 workspace.json 的占据图, 跳过扫描/修复
    # (泛洪阶段会逐格重新求解), 便于只更新分支域/矩形时快速迭代
    occ = None
    sols = {}
    if "--reuse-occupancy" in sys.argv:
        with open(os.path.join(_SIM_DIR, "workspace%s.json" % suffix),
                  encoding="utf-8") as f:
            gp = json.load(f)["grid"]
        y_lo, z_lo = gp["y_lo"], gp["z_lo"]
        ny, nz = gp["ny"], gp["nz"]
        occ = np.array([[c == "1" for c in row] for row in gp["occupancy"]])
        ys_grid = y_lo + np.arange(ny) * CELL
        zs_grid = z_lo + np.arange(nz) * CELL
        y_hi = y_lo + ny * CELL
        z_hi = z_lo + nz * CELL
        print("复用占据图: 可达 %d 格" % int(occ.sum()))

    if occ is None:
        # ---------- 1. 自适应定界（位置约束随机采样）----------
        n = 20000
        qs = rng.uniform(limits[:, 0], limits[:, 1], size=(n, 7))
        ys, zs = [], []
        for q in qs:
            _, p = chain.fk(q, 0.12)
            if abs(p[0] - PLANE_X) < 0.02:
                ys.append(p[1])
                zs.append(p[2])
        if len(ys) < 50:
            raise SystemExit("随机采样在 x=%.2f 平面命中过少，请检查限位/偏移"
                             % PLANE_X)
        y_lo, y_hi = min(ys) - 0.02, max(ys) + 0.02
        z_lo, z_hi = min(zs) - 0.02, max(zs) + 0.02
        print("平面命中 %d/%d, 粗界 y[%.2f, %.2f] z[%.2f, %.2f]"
              % (len(ys), n, y_lo, y_hi, z_lo, z_hi))

        ny = int((y_hi - y_lo) / CELL) + 1
        nz = int((z_hi - z_lo) / CELL) + 1
        print("网格: %d x %d (%d 点)" % (ny, nz, ny * nz))

        # ---------- 2. 网格扫描（warm-start IK）----------
        occ = np.zeros((ny, nz), dtype=bool)
        sols = {}
        ys_grid = y_lo + np.arange(ny) * CELL
        zs_grid = z_lo + np.arange(nz) * CELL

        prev_row_seed = q_mid
        total_ok = 0
        nfail1 = 0
        for iy in range(ny):
            seed = prev_row_seed
            for iz in range(nz):
                tgt = np.array([PLANE_X, ys_grid[iy], zs_grid[iz]])
                q, ok, iters = ik.solve(seed, tgt, axis)
                if not ok:
                    # 换种子重试一次（中点位形）
                    q, ok, iters = ik.solve(q_mid, tgt, axis)
                if not ok:
                    # 随机重启兜底: DLS 陷局部极小≠不可达(假洞主因)
                    q, ok = solve_robust(ik, rng, limits, q_mid, tgt, axis,
                                         SCAN_RESTARTS)
                if ok:
                    occ[iy, iz] = True
                    sols[(iy, iz)] = q
                    seed = q
                    total_ok += 1
                else:
                    nfail1 += 1
            if occ[iy].any():
                idx = np.where(occ[iy])[0]
                prev_row_seed = sols[(iy, int(idx[len(idx) // 2]))]
            if iy % 10 == 0:
                print("  行 %d/%d, 累计可达 %d" % (iy, ny, total_ok))
        print("首轮扫描: 可达 %d, 失败 %d" % (total_ok, nfail1))

        # ---------- 2b. 假洞修复(邻域解传播) ----------
        # 只复查"与可达格相邻"的失败格: 相邻解是最好的种子, 1cm 邻域内
        # warm start 基本一发即中; 修复后的格子继续向洞内传播, 直到收敛。
        # 四周都是失败格的"深洞"格不重复烧算力——若整片真是不可达,
        # 边界轮会先失败退出; 若洞是假的, 传播几轮就能填满。
        nfix_total = 0
        for rnd in range(REPAIR_ROUNDS):
            frontier = []
            for iy in range(ny):
                for iz in range(nz):
                    if occ[iy, iz]:
                        continue
                    if ((iy > 0 and occ[iy-1, iz])
                            or (iy < ny-1 and occ[iy+1, iz])
                            or (iz > 0 and occ[iy, iz-1])
                            or (iz < nz-1 and occ[iy, iz+1])):
                        frontier.append((iy, iz))
            if not frontier:
                break
            nfix = 0
            for iy, iz in frontier:
                tgt = np.array([PLANE_X, ys_grid[iy], zs_grid[iz]])
                ok = False
                # 邻域解做种子(8 邻域)
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if (dy or dz) and 0 <= iy+dy < ny \
                                and 0 <= iz+dz < nz and occ[iy+dy, iz+dz]:
                            q, ok, _ = ik.solve(sols[(iy+dy, iz+dz)], tgt, axis)
                            if ok:
                                break
                    if ok:
                        break
                if not ok:
                    q, ok = solve_robust(ik, rng, limits, q_mid, tgt, axis,
                                         REPAIR_RESTARTS)
                if ok:
                    occ[iy, iz] = True
                    sols[(iy, iz)] = q
                    nfix += 1
            nfix_total += nfix
            print("  修复轮 %d: 边界失败格 %d, 修好 %d" % (rnd + 1,
                                                          len(frontier), nfix))
            if nfix == 0:
                break
        print("修复完成: 共修好 %d 格, 最终可达 %d"
              % (nfix_total, int(occ.sum())))

    # ---------- 2c. 多支泛洪: 枚举所有构型支, 选"最大执行矩形"的支 ----------
    # 网格"可达"≠一条连续关节轨迹能走遍: 不同格可能分属不同构型支
    # (肘上/肘下等), 相邻 1cm 格的臂关节(J1-J4)解可差 90~180°, 直线
    # 插值会让 TCP 中途甩出大弧线(2026-09-08 dense 实测 J2 单步 181°,
    # 真机执行危险)。2026-09-08 右臂实测教训: 固定从扫描首格泛洪会
    # 选中低区小支(z≤0.63), 高区整片被关在外面 → 改为枚举全部支,
    # 每支各自求"腐蚀 3cm 后最大内接矩形", 选矩形最大的支作为执行域。
    ARM_JUMP = math.radians(20.0)
    ARM = slice(0, 4)          # J1-J4 臂关节(腕 J5-J7 自旋不影响 TCP)
    adm = np.zeros_like(occ)
    adm_sols = {}
    un = occ.copy()
    branches = []              # (mask, sols, (area, rect_cells))
    while un.any():
        iy0, iz0 = [int(v) for v in np.argwhere(un)[0]]
        tgt = np.array([PLANE_X, ys_grid[iy0], zs_grid[iz0]])
        q0, ok, _ = ik.solve(q_mid, tgt, axis)
        if not ok:
            q0, ok = solve_robust(ik, rng, limits, q_mid, tgt, axis, 20)
        if not ok:
            un[iy0, iz0] = False      # 孤立且求解失败的格(罕见), 丢弃
            continue
        mask = np.zeros_like(occ)
        bsols = {(iy0, iz0): np.asarray(q0)}
        mask[iy0, iz0] = True
        dq = deque([(iy0, iz0)])
        while dq:
            cy, cz = dq.popleft()
            qn = bsols[(cy, cz)]
            for dy, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                jy, jz = cy + dy, cz + dz
                if not (0 <= jy < ny and 0 <= jz < nz):
                    continue
                if mask[jy, jz] or not occ[jy, jz]:
                    continue
                t2 = np.array([PLANE_X, ys_grid[jy], zs_grid[jz]])
                q2, ok2, _ = ik.solve(qn, t2, axis)
                if not ok2:
                    continue
                if np.abs(q2[ARM] - qn[ARM]).max() > ARM_JUMP:
                    continue
                mask[jy, jz] = True
                bsols[(jy, jz)] = q2
                dq.append((jy, jz))
        un &= ~mask
        # 该支腐蚀 3cm 后的最大内接矩形
        er = mask.copy()
        for _ in range(MARGIN_CELLS):
            er = er & np.roll(er, 1, 0) & np.roll(er, -1, 0) \
                    & np.roll(er, 1, 1) & np.roll(er, -1, 1)
        best_b = (0, None)
        heights = np.zeros(nz, dtype=int)
        for yy in range(ny):
            heights = np.where(er[yy], heights + 1, 0)
            stack = []
            for zz in range(nz + 1):
                h = heights[zz] if zz < nz else 0
                start = zz
                while stack and stack[-1][1] >= h:
                    s_iz, s_h = stack.pop()
                    area = s_h * (zz - s_iz)
                    if area > best_b[0]:
                        best_b = (area, (yy - s_h + 1, s_iz, yy, zz - 1))
                    start = s_iz
                stack.append((start, h))
        branches.append((mask, bsols, best_b))
        if best_b[1] is not None:
            b_r = (y_lo + best_b[1][0] * CELL, z_lo + best_b[1][1] * CELL,
                   y_lo + best_b[1][2] * CELL, z_lo + best_b[1][3] * CELL)
            print("  支 %d: %d 格, 最大矩形 %.1f×%.1fcm  y[%.3f,%.3f] z[%.3f,%.3f]"
                  % (len(branches), int(mask.sum()),
                     (b_r[2] - b_r[0]) * 100, (b_r[3] - b_r[1]) * 100,
                     b_r[0], b_r[2], b_r[1], b_r[3]))
        else:
            print("  支 %d: %d 格, 腐蚀后无矩形" % (len(branches),
                                                    int(mask.sum())))

    if not branches or all(b[2][1] is None for b in branches):
        raise SystemExit("所有构型支都给不出执行矩形(腐蚀 3cm 后为空)")
    best = max(branches, key=lambda b: b[2][0])
    adm, adm_sols, (rect_area, rect_cells) = best[0], best[1], best[2]
    print("连续支泛洪: 共 %d 支, 选中的执行域 %d 格 / 可达 %d 格"
          % (len(branches), int(adm.sum()), int(occ.sum())))

    # ---------- 3. 执行矩形(选中支的最大内接矩形, 已含 3cm 边距) ----------
    (iy0, iz0, iy1, iz1) = rect_cells
    rect = (y_lo + iy0 * CELL, z_lo + iz0 * CELL,
            y_lo + iy1 * CELL, z_lo + iz1 * CELL)
    print("长方形(带 %dcm 边距): y[%.3f, %.3f] z[%.3f, %.3f]  %.1fcm x %.1fcm"
          % (MARGIN_CELLS, rect[0], rect[2], rect[1], rect[3],
             (rect[2] - rect[0]) * 100, (rect[3] - rect[1]) * 100))

    # ---------- 4. 之字形路径（平面内往返扫描）----------
    y0, z0, y1, z1 = rect
    waypoints = []
    z = z0
    direction = 1.0
    while z <= z1 + 1e-9:
        waypoints.append([PLANE_X, y0 if direction > 0 else y1, z])
        waypoints.append([PLANE_X, y1 if direction > 0 else y0, z])
        direction *= -1.0
        z += ZIGZAG_SPACING
    # 限位/可达/同支检查: 用 admit 域逐格解做 warm start 沿路径传递;
    # 不再随机重启兜底——矩形 ⊂ admit 域, 逐格解必然存在, 野重启可能
    # 落到别的构型支(禁止)。失败只可能是域边界取整误差, 警告并跳过。
    q_prev = None
    wp_sols = []
    wp_keep = []
    for wp in waypoints:
        iy = int(round((wp[1] - y_lo) / CELL))
        iz = int(round((wp[2] - z_lo) / CELL))
        q, ok = None, False
        for seed in (adm_sols.get((iy, iz)), q_prev, q_mid):
            if seed is None:
                continue
            q, ok, _ = ik.solve(np.asarray(seed), np.array(wp), axis)
            if ok:
                break
        if not ok:
            print("[警告] 之字形点无同支解(已跳过): %s 单元(%d,%d) admit=%s"
                  % ([round(v, 3) for v in wp], iy, iz,
                     bool(adm[iy, iz])))
            continue
        q_prev = q
        wp_keep.append(wp)
        wp_sols.append([round(float(v), 5) for v in q])
    waypoints = wp_keep

    # ---------- 5. 产出 ----------
    out = {
        "plane_x": PLANE_X, "target_axis": list(TARGET_AXIS),
        "limits_source": ("real_safety.yaml soft_limits 内缩 %g°"
                          % WATCHDOG_CLEARANCE),
        "exec_clearance_deg": WATCHDOG_CLEARANCE,
        "joint_limits_rad": [[float(a), float(b)] for a, b in limits],
        "grid": {"cell": CELL, "y_lo": y_lo, "z_lo": z_lo,
                 "ny": ny, "nz": nz,
                 "occupancy": ["".join("1" if occ[iy, iz] else "0"
                                       for iz in range(nz))
                               for iy in range(ny)],
                 "admitted": ["".join("1" if adm[iy, iz] else "0"
                                      for iz in range(nz))
                              for iy in range(ny)]},
        "reachable_cells": int(occ.sum()),
        "admitted_cells": int(adm.sum()),
        "n_branches": len(branches),
        "branch_arm_jump_deg": 20.0,
        # 选中支的逐格解图集([iy, iz, q1..q7]): 供稠密走线在硬邻域
        # 用扫描已验证的同支解做种子(2026-09-08 J1 限位邻域教训)
        "admitted_solutions": [[iy, iz] + [round(float(v), 5) for v in q]
                               for (iy, iz), q in sorted(adm_sols.items())],
        "rectangle_m": {"y_min": rect[0], "z_min": rect[1],
                        "y_max": rect[2], "z_max": rect[3]},
        "zigzag_spacing_m": ZIGZAG_SPACING,
        "zigzag_waypoints": waypoints,
        "zigzag_joint_solutions_rad": wp_sols,
    }
    out_path = os.path.join(_SIM_DIR, "workspace%s.json" % suffix)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # SVG 可视化: 可达图 + 同支执行域 + 长方形 + 之字形
    W, H = ny, nz
    sc = 8
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">'
           % (W * sc + 80, H * sc + 60)]
    svg.append('<rect width="100%" height="100%" fill="#202020"/>')
    for iy in range(ny):
        for iz in range(nz):
            if occ[iy, iz] and not adm[iy, iz]:
                svg.append('<rect x="%d" y="%d" width="%d" height="%d" '
                           'fill="#606060"/>' % (40 + iy * sc,
                                                 20 + (H - 1 - iz) * sc,
                                                 sc, sc))
            elif adm[iy, iz]:
                svg.append('<rect x="%d" y="%d" width="%d" height="%d" '
                           'fill="#3a6ea5"/>' % (40 + iy * sc,
                                                 20 + (H - 1 - iz) * sc,
                                                 sc, sc))
    # 长方形（红框）
    rx = 40 + (rect[0] - y_lo) / CELL * sc
    rz = 20 + (H - 1 - (rect[3] - z_lo) / CELL) * sc
    rw = (rect[2] - rect[0]) / CELL * sc
    rh = (rect[3] - rect[1]) / CELL * sc
    svg.append('<rect x="%.0f" y="%.0f" width="%.0f" height="%.0f" '
               'fill="none" stroke="red" stroke-width="2"/>' % (rx, rz, rw, rh))
    # 之字形（黄线）
    pts = " ".join("%.0f,%.0f" % (40 + (wp[1] - y_lo) / CELL * sc,
                                  20 + (H - 1 - (wp[2] - z_lo) / CELL) * sc)
                   for wp in waypoints)
    svg.append('<polyline points="%s" fill="none" stroke="yellow" '
               'stroke-width="1.5"/>' % pts)
    svg.append('<text x="40" y="14" fill="white" font-size="12">'
               'TCP 平面 x=%.2fm | 蓝格=同支执行域 | 灰格=可达但在另支 | '
               '红框=工作长方形 | 黄线=之字形</text>' % PLANE_X)
    svg.append('<text x="40" y="%d" fill="white" font-size="11">'
               '横轴 y(m) %.2f→%.2f | 纵轴 z(m) %.2f→%.2f（上为高）</text>'
               % (H * sc + 45, y_lo, y_hi, z_lo, z_hi))
    svg.append('</svg>')
    svg_path = os.path.join(_fig_dir(), "workspace%s.svg" % (suffix or "_left"))
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))

    print("=" * 70)
    print("工作区间扫描完成: 可达 %d/%d 格（%.1f%%）, 同支执行域 %d 格"
          % (occ.sum(), occ.size, 100.0 * occ.sum() / occ.size,
             int(adm.sum())))
    print("产出: %s" % out_path)
    print("      %s" % svg_path)
    print("之字形路径点: %d 个, 行距 %.0fcm" % (len(waypoints),
                                               ZIGZAG_SPACING * 100))


if __name__ == "__main__":
    main()
