#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
右臂之字形稠密关节轨迹生成 + 路线绘图(双模式)
==============================================
模式 A(原生, 正式): 仿真/isaaclab_sim/workspace_right.json 存在且含
  zigzag_waypoints(stage3_workspace.py --side right 的工作区扫描产物,
  限位=右臂 real_safety.yaml 实测内缩 3°)→ 直接用其路点/关节解,
  稠密求解 + 安全闸。
模式 B(镜像, 临时): 无原生扫描时, 从左臂 workspace.json 镜像(y→−y):
  符号映射(FK 全路点验证) + 限位镜像 + 右臂 IK 精化——限位是临时值。

共用安全闸(与左臂 make_dense_zigzag 同款):
  连续 warm start(臂关节 J1-J4 单步 >20° 拒绝/终止) + 输出前全轨迹复核
  + 终点 FK 复核。

用法: /usr/bin/python3 make_dense_zigzag_right.py
产出: dense_zigzag_right.json + assets/figures/moveit_zigzag_right_route.png
      (workspace_right.json 属扫描产物, 本脚本不覆盖)
"""

import itertools
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ISA_DIR = os.path.abspath(os.path.join(_HERE, "..", "isaaclab_sim"))
_PKG_DIR = os.path.abspath(os.path.join(_HERE, "..", "openarm_sim"))
sys.path.insert(0, _ISA_DIR)
sys.path.insert(0, _PKG_DIR)

from openarm_sim.ik_solver import DlsIK          # noqa: E402
from openarm_sim.kinematics import UrdfChain     # noqa: E402


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


STEP = 0.005            # 采样步长 5mm(与左臂一致)
TCP_OFF = 0.12          # 路点=指尖位置, IK 链带 0.12 指尖偏移(阶段3语义)
ARM_JUMP_DEG = 20.0     # 臂关节 J1-J4 单步跳变阈值(换支判定, 与左臂一致)
ARM_JOINTS = slice(0, 4)
# 执行重定时参数(与 zigzag_moveit.py 一致, 仅用于时长预算)
VEL_LIMITS = [0.5, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8]
TCP_SPEED = 0.05

LEFT_WS = os.path.join(_ISA_DIR, "workspace.json")
RIGHT_WS = os.path.join(_ISA_DIR, "workspace_right.json")
RIGHT_URDF = os.path.join(_ISA_DIR, "openarm_right.urdf")
_SAFETY_R = os.path.abspath(os.path.join(
    _HERE, "..", "..", "real", "right", "config", "real_safety.yaml"))


def load_right_limits():
    """右臂扫描限位: real_safety.yaml 软限位内缩 3°(执行限位口径)。"""
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


def mirror_mode():
    """模式 B: 从左臂 workspace 镜像(临时限位)。
    返回 (wps_r, sols_r, lim_r, provisional, rect_r, axis_r, right)。"""
    with open(LEFT_WS, encoding="utf-8") as f:
        ws = json.load(f)
    wps_l = [np.array(w) for w in ws["zigzag_waypoints"]]
    sols_l = [np.array(q) for q in ws["zigzag_joint_solutions_rad"]]
    lim_l = np.array(ws["joint_limits_rad"])          # 左臂扫描限位(内缩3°)
    axis_l = np.array(ws["target_axis"])
    axis_r = np.array([axis_l[0], -axis_l[1], axis_l[2]])
    wps_r = [np.array([w[0], -w[1], w[2]]) for w in wps_l]

    right = UrdfChain(RIGHT_URDF, tip_link="openarm_right_hand_tcp")

    # ---- 符号映射搜索: FK_right(s⊙q_left) 命中镜像路点 ----
    print("== 符号映射搜索(镜像机构定理) ==")
    found = None
    for s in itertools.product([1.0, -1.0], repeat=7):
        ok = all(float(np.linalg.norm(
            right.fk(np.array(sols_l[i]) * s, TCP_OFF)[1] - wps_r[i])) <= 5e-3
            for i in range(0, len(wps_l), 5))
        if ok:
            found = list(s)
            break
    if found is None:
        sys.exit("[致命] 未找到一致的符号映射——请反馈排查, 不要强行生成")
    print("符号映射(URDF关节序 J1..J7): %s" % found)

    # ---- 限位镜像(临时值: 左臂实测限位的镜像) ----
    lim_r = lim_l.copy()
    for i in range(7):
        if found[i] < 0:
            lim_r[i] = [-lim_l[i][1], -lim_l[i][0]]

    ik = DlsIK(right, lim_r, tcp_offset_m=TCP_OFF)
    rng = np.random.default_rng(7)
    sols_r = []
    for i, (q_seed, tgt) in enumerate(zip(sols_l, wps_r)):
        q0 = np.clip(np.array(q_seed) * found, lim_r[:, 0], lim_r[:, 1])
        q, ok, _ = ik.solve(q0, tgt, target_axis=axis_r)
        if not ok:
            for _ in range(20):
                q2, ok2, _ = ik.solve(
                    rng.uniform(lim_r[:, 0], lim_r[:, 1]), tgt,
                    target_axis=axis_r)
                if ok2:
                    q, ok = q2, True
                    break
        if not ok:
            sys.exit("[致命] 右臂路点 %d IK 未收敛: %s" % (i, tgt))
        sols_r.append(q)
    print("36 路点右臂解全部收敛(镜像模式)")
    rect_l = ws["rectangle_m"]
    rect_r = {"y_min": -rect_l["y_max"], "y_max": -rect_l["y_min"],
              "z_min": rect_l["z_min"], "z_max": rect_l["z_max"]}
    return wps_r, sols_r, lim_r, True, rect_r, axis_r, right, {}, 0.0, 0.0


def native_mode(zmin_override=None, ymax_taper=None, ymax_cap=None,
                zmax_cap=None, ymin_cap=None):
    """模式 A: 右臂原生工作区扫描(stage3_workspace.py --side right 产物)。
    zmin_override: 抬高矩形底边(米)——低区角落 J5/J6 腕部行程饱和。
    ymax_taper: z=z_min 行的 y_max 收窄值(米, 线性过渡到 z_max 的原
    y_max)——跨中线方向 J6 正向行程仅 ~11°, 低行到达中线即饱和。
    返回 (wps_r, sols_r, lim_r, provisional, rect_r, axis_r, right)。"""
    with open(RIGHT_WS, encoding="utf-8") as f:
        ws = json.load(f)
    wps_scan = [np.array(w) for w in ws["zigzag_waypoints"]]
    sols_scan = [np.array(q) for q in ws["zigzag_joint_solutions_rad"]]
    lim_r = load_right_limits()          # 权威限位(real_safety 内缩 3°)
    rect_r = dict(ws["rectangle_m"])
    if ymin_cap is not None:
        rect_r["y_min"] = max(rect_r["y_min"], float(ymin_cap))
        print("== y_min 内收: %.3f m ==" % rect_r["y_min"])
    if ymax_cap is not None:
        # y_max 硬帽: 路线不过机体中线(y≈0 处 J1 姿态族分岔 + J6 正向
        # 行程仅 ~11°), 完全生活在自然姿态族内(用户确认无需跨中线)
        rect_r["y_max"] = min(rect_r["y_max"], float(ymax_cap))
        print("== y_max 硬帽: %.3f m ==" % rect_r["y_max"])
    if zmin_override is not None:
        if zmin_override <= rect_r["z_min"] or zmin_override >= rect_r["z_max"]:
            sys.exit("[致命] --zmin %.3f 不在扫描矩形 z 范围内" % zmin_override)
        rect_r["z_min"] = float(zmin_override)
        print("== 矩形底边抬高: z_min = %.3f m ==" % zmin_override)
    if zmax_cap is not None:
        rect_r["z_max"] = min(rect_r["z_max"], float(zmax_cap))
        print("== 矩形顶边压低: z_max = %.3f m ==" % rect_r["z_max"])
    axis_r = np.array(ws["target_axis"])
    right = UrdfChain(RIGHT_URDF, tip_link="openarm_right_hand_tcp")
    print("== 路线来源: 右臂原生扫描 workspace_right.json "
          "(可达 %d 格, 同支 %d 格) ==" %
          (ws.get("reachable_cells", -1), ws.get("admitted_cells", -1)))
    # 同支解图集: 目标点邻域的扫描已验证解, 供稠密走线做种子
    atlas = {(r[0], r[1]): np.array(r[2:9]) for r in
             ws.get("admitted_solutions", [])}
    gp_y_lo = ws["grid"]["y_lo"]
    gp_z_lo = ws["grid"]["z_lo"]

    # 之字路点(行距 3cm, 与扫描同款往返); y_max 按 z 线性锥形(低行收窄)
    y0, z0, y1, z1 = rect_r["y_min"], rect_r["z_min"], rect_r["y_max"], \
        rect_r["z_max"]
    spacing = ws.get("zigzag_spacing_m", 0.03)
    wps_r = []
    z = z0
    direction = 1.0
    while z <= z1 + 1e-9:
        t = (z - z0) / (z1 - z0) if ymax_taper is not None else 0.0
        y_hi = y1 if (ymax_taper is None or ymax_cap is not None) else \
            ymax_taper + t * (y1 - ymax_taper)
        wps_r.append(np.array([ws["plane_x"], y0 if direction > 0 else y_hi,
                               z]))
        wps_r.append(np.array([ws["plane_x"], y_hi if direction > 0 else y0,
                               z]))
        direction *= -1.0
        z += spacing
    # 全局优化产物优先: optimize_right_traj.py 的平滑解(矩形一致时)
    opt_p = os.path.join(_HERE, "optimized_waypoints_right.json")
    if os.path.isfile(opt_p):
        with open(opt_p, encoding="utf-8") as f:
            opt = json.load(f)
        o_rect = opt.get("rect", {})
        o_wps = opt.get("waypoints", [])
        same = bool(o_wps) and all([
            abs(o_rect.get(k, 9) - rect_r[k]) < 1e-9
            for k in ("y_min", "y_max", "z_min", "z_max")]) \
            and len(o_wps) == len(wps_r) and all(
            float(np.linalg.norm(np.array(a) - np.array(b))) < 1e-6
            for a, b in zip(o_wps, wps_r))
        if same and opt.get("solutions"):
            sols_r = [np.array(q) for q in opt["solutions"]]
            native_mode.opt_used = True
            r = opt.get("report", {})
            print("== 路点解: 全局优化产物 (位置最大误差 %.2fmm, "
                  "轴最大误差 %.2f°, 路点步进最大 %s) ==" % (
                      r.get("max_pos_err_m", 0) * 1000,
                      r.get("max_axis_err_deg", 0),
                      ["%.1f" % v for v in r.get("max_wp_step_deg", [])]))

    # 首点种子: 图集解中 TCP 距离最近者(拓扑在正确姿态族内;
    # 全局优化产物已就位时跳过——sols_r 已是优化链)
    if not getattr(native_mode, "opt_used", False):
        c0 = (int(round((wps_r[0][1] - gp_y_lo) / 0.01)),
              int(round((wps_r[0][2] - gp_z_lo) / 0.01)))
        best = None
        for r in range(0, 8):
            for a in range(-r, r + 1):
                for b in range(-r, r + 1):
                    q = atlas.get((c0[0] + a, c0[1] + b))
                    if q is not None:
                        best = q
                        break
                if best is not None:
                    break
            if best is not None:
                break
        if best is None:
            d2 = [float(np.linalg.norm(w - wps_r[0])) for w in wps_scan]
            best = sols_scan[int(np.argmin(d2))]
        sols_r = [best]
        print("首点种子 = 图集最近解 (格 %s, J1=%.1f°)" % (
            str(c0), math.degrees(best[0])))
    else:
        print("首点种子 = 全局优化解 (跳过图集搜索)")
    return (wps_r, sols_r, lim_r, False, rect_r, axis_r, right,
            atlas, gp_y_lo, gp_z_lo)


def main():
    zmin = ymax_taper = ymax_cap = zmax_cap = ymin_cap = None
    if "--zmin" in sys.argv:
        zmin = float(sys.argv[sys.argv.index("--zmin") + 1])
    if "--ymax-taper" in sys.argv:
        ymax_taper = float(sys.argv[sys.argv.index("--ymax-taper") + 1])
    if "--ymax" in sys.argv:
        ymax_cap = float(sys.argv[sys.argv.index("--ymax") + 1])
    if "--zmax" in sys.argv:
        zmax_cap = float(sys.argv[sys.argv.index("--zmax") + 1])
    if "--ymin" in sys.argv:
        ymin_cap = float(sys.argv[sys.argv.index("--ymin") + 1])
    use_native = os.path.isfile(RIGHT_WS)
    if use_native:
        with open(RIGHT_WS, encoding="utf-8") as f:
            cand = json.load(f)
        if not cand.get("zigzag_waypoints") or "mirror_source" in cand:
            use_native = False               # 旧的镜像产物, 走镜像模式
    if use_native:
        (wps_r, sols_r, lim_r, provisional, rect_r, axis_r, right,
         atlas, gy_lo, gz_lo) = native_mode(zmin, ymax_taper, ymax_cap, zmax_cap,
                            ymin_cap)
    else:
        if not os.path.isfile(_SAFETY_R):
            sys.exit("[致命] 缺 %s——先 gen_real_config --side right 生成"
                     % _SAFETY_R)
        (wps_r, sols_r, lim_r, provisional, rect_r, axis_r, right,
         atlas, gy_lo, gz_lo) = mirror_mode()

    # ---- 稠密求解(与左臂 make_dense_zigzag 同闸 + 连续性增强) ----
    # 教训(2026-09-08): 失败步用"全盒随机重启"会解到别的构型支(被 20° 闸
    # 拒掉)→ 致命退出; 改为笛卡尔中点细分(利用扫描已证明的 1cm 同支链,
    # 局部求解必然连续), 随机重启只留 5 次兜底。腕部(J5-J7)单步 >30° 时
    # 做小扰动重试取最小步长解——用户要求全程无大幅重构型动作。
    ik = DlsIK(right, lim_r, tcp_offset_m=TCP_OFF)
    rng = np.random.default_rng(7)
    WRIST, WRIST_MAX_DEG, n_wrist_retry = slice(4, 7), 30.0, [0]

    def solve_centered(q_prev, target, max_iters=80):
        """最小冗余运动 IK: DLS 任务步 + 零空间阻尼(冗余自由度向 q_prev
        收敛)。整条路线 = 冗余空间的测地线, 无构型重构、无贴限位滑行
        ——直接回应"完全连贯流畅、不要大变换"的要求。"""
        q = np.clip(q_prev.copy(), lim_r[:, 0], lim_r[:, 1])
        t = np.asarray(target, dtype=float)
        ax_t = axis_r / np.linalg.norm(axis_r)
        for _ in range(max_iters):
            e, p, ax = ik._error(q, t, ax_t)
            pos_err = float(np.linalg.norm(e[:3]))
            ax_err = math.acos(max(-1.0, min(1.0,
                float(np.dot(ax, ax_t)))))
            if pos_err < 1e-3 and ax_err < math.radians(2.0):
                return q, True
            J = ik._jacobian(q, ax_t)
            JJt = J @ J.T + (0.08 ** 2) * np.eye(6)
            lam = np.linalg.solve(JJt, e)
            dq = J.T @ lam
            s = float(np.abs(dq).max())
            if s > 0.25:
                dq *= 0.25 / s
            # 零空间: N = I − Jᵀ(JJt)⁻¹J (2维: 自旋+肘形)。
            # ①连续阻尼(向 q_prev 收敛=最小重构) ②限位回避(接近盒边
            # 15% 的关节被主动拉回安全区)——只靠①会贴限位滑行撞墙。
            N = np.eye(7) - J.T @ np.linalg.solve(JJt, J)
            pull = np.zeros(7)
            for j in range(7):
                lo_s, hi_s = lim_r[j]
                mg = (hi_s - lo_s) * 0.15
                if q[j] > hi_s - mg:
                    pull[j] = (hi_s - mg) - q[j]
                elif q[j] < lo_s + mg:
                    pull[j] = (lo_s + mg) - q[j]
            # 退火: 任务误差接近收敛阈值时拉力/阻尼归零, 防 q 振荡不停
            g = 0.8 * min(1.0, max(0.0, (pos_err - 0.0012) / 0.0028,
                                   (ax_err - math.radians(2.0))
                                   / math.radians(6.0)))
            dq = dq + N @ (g * (0.15 * (q_prev - q) + 0.8 * pull))
            # 限位: 单步内钳到盒边
            q_next = q + dq
            over_hi = q_next > lim_r[:, 1]
            over_lo = q_next < lim_r[:, 0]
            q_next[over_hi] = lim_r[over_hi, 1]
            q_next[over_lo] = lim_r[over_lo, 0]
            if float(np.abs(q_next - q).max()) < 1e-6:
                break                      # 顶死限位且误差未消
            q = q_next
        e, p, ax = ik._error(q, t, ax_t)
        ok = (float(np.linalg.norm(e[:3])) < 1e-3
              and math.acos(max(-1.0, min(1.0, float(np.dot(ax, ax_t)))))
              < math.radians(2.0))
        return q, ok

    def solve_to(q_prev, target, where, depth=0, tgt_from=None, seed=None):
        """走线单步求解(四级策略, 全程连贯优先):
        ① 优化链插值解(seed=相邻优化路点解的关节插值)——全局平滑解;
        ② 最小冗余运动(零空间阻尼+限位回避);
        ③ 图集种子(扫描同支已验证解);
        ④ 笛卡尔中点细分递归。所有候选都过同一闸门:
           臂关节 J1-J4 相对 q_prev ≤20°(防换支甩弧)。"""
        cands = []
        # ① 优化链插值解: 直接采纳(它就是全局平滑解本身)
        if seed is not None:
            q_c, ok_c, _ = ik.solve(np.asarray(seed), target,
                                    target_axis=axis_r)
            if ok_c and float(np.abs(
                    q_c[ARM_JOINTS] - q_prev[ARM_JOINTS]).max()) \
                    <= math.radians(ARM_JUMP_DEG):
                return q_c
        # ② centered
        q_c, ok_c = solve_centered(q_prev, target)
        if ok_c and float(np.abs(
                q_c[ARM_JOINTS] - q_prev[ARM_JOINTS]).max()) \
                <= math.radians(ARM_JUMP_DEG):
            cands.append(q_c)
        # ③ 图集种子(3×3 邻域, 按 ARM 距 q_prev 排序)
        if atlas:
            ciy = int(round((target[1] - gy_lo) / 0.01))
            ciz = int(round((target[2] - gz_lo) / 0.01))
            near = []
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    s = atlas.get((ciy + a, ciz + b))
                    if s is not None:
                        near.append((float(np.abs(
                            s[ARM_JOINTS] - q_prev[ARM_JOINTS]).max()), s))
            near.sort(key=lambda t: t[0])
            for _, s in near:
                q_c, ok_c, _ = ik.solve(np.asarray(s), target,
                                        target_axis=axis_r)
                if ok_c and float(np.abs(
                        q_c[ARM_JOINTS] - q_prev[ARM_JOINTS]).max()) \
                        <= math.radians(ARM_JUMP_DEG):
                    cands.append(q_c)
        # ③ 细分递归
        if not cands:
            if depth >= 3:
                print("[致命] %s: 三级策略仍无同支解" % where)
                print("  目标 TCP: %s" % [round(float(v), 4) for v in target])
                print("  q_prev(°): %s" % [round(math.degrees(v), 1)
                                           for v in q_prev])
                print("  q_prev 距限位余量(°): %s" % [
                    round(min(math.degrees(q_prev[j] - lim_r[j][0]),
                              math.degrees(lim_r[j][1] - q_prev[j])), 1)
                    for j in range(7)])
                q_pos, ok_pos, _ = ik.solve(q_prev, target, target_axis=None)
                print("  仅位置约束: %s" % ok_pos)
                sys.exit(1)
            mid = 0.5 * (tgt_from + target)
            qm = solve_to(q_prev, mid, where, depth + 1, tgt_from=tgt_from)
            return solve_to(qm, target, where, depth + 1, tgt_from=mid)
        # 多候选取腕部动作最小者(用户要求无大幅变换)
        if len(cands) > 1:
            return min(cands, key=lambda c: float(
                np.abs(c[4:7] - q_prev[4:7]).max()))
        return cands[0]

    points = [sols_r[0].tolist()]
    seg_lens = [0.0]
    q = np.clip(sols_r[0].copy(), lim_r[:, 0], lim_r[:, 1])
    Qwp = np.asarray(sols_r)                 # 优化链(原生模式=全局优化解)
    for i in range(len(wps_r) - 1):
        a, b = wps_r[i], wps_r[i + 1]
        nsub = max(int(np.linalg.norm(b - a) / STEP), 1)
        tgt_from = a
        for k in range(1, nsub + 1):
            t = k / nsub
            target = a + (b - a) * t
            seed = (1 - t) * Qwp[i] + t * Qwp[i + 1]   # 优化链关节插值
            q = solve_to(q, target, "段 %d 步 %d/%d" % (i, k, nsub),
                         tgt_from=tgt_from, seed=seed)
            tgt_from = target
            points.append(q.tolist())
            seg_lens.append(float(np.linalg.norm(b - a) / nsub))
    if n_wrist_retry[0]:
        print("腕部大步重试: %d 处改善" % n_wrist_retry[0])

    P = np.array(points)
    # 终点 FK 复核
    _, p_end = right.fk(q, TCP_OFF)
    print("共 %d 点, 终点指尖误差 %.4f m" % (
        len(points), float(np.linalg.norm(p_end - wps_r[-1]))))
    # 输出前全轨迹复核: 臂关节单步 >20° 拒绝写文件
    arm_step = np.abs(np.diff(P[:, ARM_JOINTS], axis=0)).max(axis=1)
    if (arm_step > math.radians(ARM_JUMP_DEG)).any():
        sys.exit("[致命] 输出拦截: 臂关节单步跳变超 %g°(最大 %.1f°)"
                 % (ARM_JUMP_DEG, math.degrees(arm_step.max())))
    wrist_step = np.abs(np.diff(P[:, 4:], axis=0)).max(axis=1)
    print("换支复核通过: 臂关节最大单步 %.1f° | 腕部最大单步 %.1f°(绕TCP自旋)"
          % (math.degrees(arm_step.max()), math.degrees(wrist_step.max())))
    # 限位裕量
    margin = np.min(np.concatenate(
        [P - lim_r[:, 0], lim_r[:, 1] - P], axis=1).min(axis=1))
    print("全程距扫描限位最小裕量 %.2f°(=watchdog 触发线, 裕量>0 即安全)"
          % math.degrees(float(margin)))

    # ---- 时长预算(与执行侧重定时公式一致) ----
    seg = np.array(seg_lens)[1:]
    dt_jt = np.abs(np.diff(P, axis=0)) / np.array(VEL_LIMITS)
    dt = np.maximum(dt_jt.max(axis=1), np.maximum(seg / TCP_SPEED, 0.02))
    for s_scale in (0.4, 0.8, 1.0):
        print("speed-scale %.1f: 走线段预计 %.0fs" % (s_scale, dt.sum() / s_scale))

    # ---- 落盘(稠密轨迹; workspace_right.json 属扫描产物, 不覆盖) ----
    with open(os.path.join(_HERE, "dense_zigzag_right.json"), "w",
              encoding="utf-8") as f:
        json.dump({"points": points, "seg_lens": seg_lens,
                   "step_m": STEP}, f)
    print("已写出 dense_zigzag_right.json")

    # ---- 绘图 ----
    plot_route(P, wps_r, rect_r, dt, provisional, lim_r, right)


def plot_route(P, wps_r, rect_r, dt, provisional, lim, right):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # TCP 轨迹(右臂 FK)与时间轴(scale 0.4 真机执行节奏)
    tcp = np.array([right.fk(qk, TCP_OFF)[1] for qk in P])
    t = np.concatenate([[0.0], np.cumsum(dt)]) / 0.4   # 真机 0.4 倍秒

    lim_tag = ("PROVISIONAL limits = mirrored left calibration"
               if provisional else
               "OFFICIAL limits = right arm real_safety.yaml")
    fig = plt.figure(figsize=(17, 10))
    fig.suptitle("Right-arm zigzag route  |  " + lim_tag, fontsize=12)

    # (1) 走线平面 y-z
    ax1 = fig.add_subplot(2, 2, 1)
    wp_r = np.array(wps_r)
    ax1.plot(wp_r[:, 1], wp_r[:, 2], "o", ms=5, mfc="none", mec="r",
             label="waypoints (%d)" % len(wp_r))
    ax1.plot(tcp[:, 1], tcp[:, 2], "-", lw=0.8, color="tab:blue",
             label="dense TCP path")
    ax1.add_patch(plt.Rectangle(
        (rect_r["y_min"], rect_r["z_min"]),
        rect_r["y_max"] - rect_r["y_min"],
        rect_r["z_max"] - rect_r["z_min"],
        fill=False, ec="g", ls="--", label="work rect"))
    ax1.set_xlabel("y [m]  (right arm side: y<0)")
    ax1.set_ylabel("z [m]")
    ax1.set_title("weave plane x=0.30m (front view)")
    ax1.axis("equal")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3)

    # (2) 关节轨迹 vs 限位
    ax2 = fig.add_subplot(2, 2, 2)
    colors = plt.cm.tab10.colors
    for j in range(7):
        ax2.plot(t, np.degrees(P[:, j]), lw=0.8, color=colors[j],
                 label="J%d" % (j + 1))
        ax2.axhline(math.degrees(lim[j][0]), color=colors[j], ls=":", lw=0.7)
        ax2.axhline(math.degrees(lim[j][1]), color=colors[j], ls=":", lw=0.7)
    ax2.set_xlabel("time [s] at speed-scale 0.4")
    ax2.set_ylabel("joint angle [deg]  (dotted = limit)")
    ax2.set_title("joint trajectories vs limits")
    ax2.legend(fontsize=7, ncol=2, loc="upper right")
    ax2.grid(alpha=0.3)

    # (3) 3D 等轴测(含基座方位)
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax3.plot(tcp[:, 0], tcp[:, 1], tcp[:, 2], lw=0.8, color="tab:blue")
    ax3.scatter(wp_r[:, 0], wp_r[:, 1], wp_r[:, 2], s=8, c="r")
    ax3.scatter([0], [0], [0], s=40, c="k", marker="s")
    ax3.text(0, 0, 0, "  base", fontsize=8)
    ax3.set_xlabel("x [m]")
    ax3.set_ylabel("y [m]")
    ax3.set_zlabel("z [m]")
    ax3.set_title("3D view")
    try:
        ax3.set_box_aspect((1, 2, 2))
    except Exception:
        pass

    # (4) TCP 速度剖面
    ax4 = fig.add_subplot(2, 2, 4)
    seg_tcp = np.linalg.norm(np.diff(tcp, axis=0), axis=1)
    ax4.plot(t[1:], seg_tcp / dt * 0.4 * 100, lw=0.5, color="tab:green")
    ax4.set_xlabel("time [s] at speed-scale 0.4")
    ax4.set_ylabel("TCP speed [cm/s]  (target 2.0)")
    ax4.set_title("TCP speed profile (retime: max(joint-rate, TCP-rate))")
    ax4.grid(alpha=0.3)

    out = os.path.join(_fig_dir(), "moveit_zigzag_right_route.png")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=140)
    print("已写出", out)


if __name__ == "__main__":
    main()
