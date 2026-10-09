#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zigzag_ref.py —— RL 参考轨迹生成: 速度层级 + 时间分配 + 平滑时间轴
====================================================================
输入: 仿真/moveit_sim/dense_zigzag_right.json (稠密关节轨迹, 5mm TCP 步距)
输出: rl/ref_right.npz (50Hz q_ref/qd_ref + 相位标签 + 元数据) + 剖面图

任务链 = [home→起点] + [之字形走线] + [终点→home], 速度层级(用户要求):
  相位 0 转移段(去/回零位)  中等   TCP 与水平行一致 ~6 cm/s
  相位 1 走线水平行         中等   TCP ~6 cm/s
  相位 2 走线竖直换行        最快   TCP ~18 cm/s (3×水平)

平滑原理(保证"运动连续、无速度突变、不同速度衔接顺滑"):
  1. 全链(含转移段)按关节弧长 0.01 rad 均匀重采样;
  2. Hann 滑窗(半宽 W_SMOOTH ≈ 2cm TCP)圆滑: 拐角换向摊到小弧长,
     TCP 偏差 ≤4mm —— 转弯不再是速度阶跃源;
  3. 每步双约束定时长 T_k = max(TCP 速率约束, 关节速率可行性),
     转移段按关节速率 0.12 rad/s 摊时(固定安全速度);
  4. 标量进度 u(t) 分段线性 → 高斯核(σ_T)时间卷积: 速度处处连续、
     转移↔走线速度大小斜坡过渡、首尾自然归零(软起软停);
  5. 几何(圆滑后链)与时间(卷积后 u)解耦, 无样条过冲。
平稳性闭环: 生成后核算峰值关节速度/加速度, 超门限自动降速 8% 重生成。
用法: <isaaclab env python> zigzag_ref.py [--target-total 158]
"""

import argparse
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_MOVEIT = os.path.abspath(os.path.join(_HERE, "..", "..", "moveit_sim"))
_SAFETY = os.path.abspath(os.path.join(
    _HERE, "..", "..", "..", "real", "right", "config", "real_safety.yaml"))
_ISA = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ISA)
sys.path.insert(0, os.path.join(_ISA, "..", "openarm_sim"))

from openarm_sim.kinematics import UrdfChain          # noqa: E402


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


POLICY_HZ = 50
DT = 1.0 / POLICY_HZ
TCP_OFF = 0.12          # 指尖偏移(与 make_dense_zigzag_right 一致)
V_JOINT_CAP = 0.40      # 走线段关节速率可行性上限 [rad/s](双约束之一)
MIN_SEG_T = 0.02        # 单步最短时长 [s]
SETTLE = 0.5            # 首末静止保持 [s]
RES_Q = 0.01            # 全链均匀重采样关节弧长 [rad]
W_SMOOTH = 4            # Hann 滑窗半宽 [重采样点] ≈ 2cm TCP(拐角半径)
SIGMA_T = 0.30          # 进度时间轴高斯核 σ [s](段间速度过渡宽度)

V_HORIZON = 0.060       # 水平行 TCP 速率 [m/s] —— 中等
V_VERTICAL = 0.180      # 竖直换行 TCP 速率 [m/s] —— 最快 (3×水平)

# 平稳性验收门限(超限自动降速)
# 门限口径: 真机 watchdog 慢速保护 max_joint_vel=4.0 rad/s, 此处留 ~7× 裕度;
# qdd 为瞬时值, RL 策略有 0.32s 预览可前馈, 且驱动 MIT 力矩受限自然平滑
QD_PEAK_MAX = 0.55      # 峰值关节速度 [rad/s]
QDD_PEAK_MAX = 6.5      # 峰值关节加速度 [rad/s²]


def load_home(safety_yaml=_SAFETY):
    """从 real_safety.yaml 读 home_rad(权威来源)。
    硬编码的 home 副本曾与重标定后的真机配置冲突, 使参考轨迹不可用。"""
    import re
    txt = open(safety_yaml, encoding="utf-8").read()
    m = re.search(r"home_rad:\s*\[([^\]]+)\]", txt)
    if not m:
        raise SystemExit("[致命] %s 缺 home_rad" % safety_yaml)
    h = [float(v) for v in m.group(1).split(",")]
    if len(h) != 7:
        raise SystemExit("[致命] home_rad 长度 %d != 7" % len(h))
    print("home(来自 %s): %s" % (os.path.basename(safety_yaml),
                                [round(v, 6) for v in h]))
    return np.array(h)


def resample_chain(P, home, dq_res=RES_Q):
    """全链 [home→P0→dense→Pn→home] 按关节弧长均匀重采样。
    返回 (Q[n,7], pos[n]): pos[i] = 第 i 点对应的链位置(浮点索引)。"""
    chain = np.vstack([home[None, :], P, home[None, :]])
    seg = np.linalg.norm(np.diff(chain, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(cum[-1] / dq_res), 8)
    s_tgt = np.linspace(0.0, cum[-1], n)
    Q = np.column_stack([np.interp(s_tgt, cum, chain[:, j])
                         for j in range(7)])
    pos = np.interp(s_tgt, cum, np.arange(len(chain), dtype=float))
    return Q, pos


def hann_smooth(Q, half=W_SMOOTH):
    """Hann 滑窗圆滑(端点保持): 拐角换向摊到 ~2·half 点弧长。"""
    n = len(Q)
    w = 0.5 * (1.0 + np.cos(np.pi * (np.arange(2 * half + 1) - half)
                            / (half + 1.0)))
    w = w / w.sum()
    pad = np.vstack([np.repeat(Q[:1], half, axis=0), Q,
                     np.repeat(Q[-1:], half, axis=0)])
    out = np.empty_like(Q)
    for j in range(7):
        out[:, j] = np.convolve(pad[:, j], w, mode="same")[half:half + n]
    return out


def classify_steps(tcp):
    """按 TCP 位移把步分类: 0=斜步/其他 1=水平 2=竖直。"""
    d = np.diff(tcp, axis=0)
    dy, dz = np.abs(d[:, 1]), np.abs(d[:, 2])
    ph = np.zeros(len(d), dtype=int)
    ph[dy > 5 * dz] = 1
    ph[dz > 5 * dy] = 2
    return ph


def build_once(P, home, scale, right):
    """scale = 时间膨胀因子(速度 ∝ 1/scale)。生成参考并核算。"""
    # 1) 全链重采样 + 圆滑
    Q0, pos = resample_chain(P, home)
    Q = hann_smooth(Q0, W_SMOOTH)
    n = len(Q)
    tcp = np.array([right.fk(q, TCP_OFF)[1] for q in Q])
    ph_cls = classify_steps(tcp)                     # 0=斜步 1=水平 2=竖直
    # 走线区间 = 链位置 ∈ (1, len(P)) 即 dense 部分
    i_weave0 = int(np.interp(1.0, pos, np.arange(n)))
    i_weave1 = int(np.interp(float(len(P)), pos, np.arange(n)))

    # 2) 双约束定时长
    dq_step = np.abs(np.diff(Q, axis=0)).max(axis=1)
    d_tcp = np.linalg.norm(np.diff(tcp, axis=0), axis=1)
    v_seg = np.where(ph_cls == 2, V_VERTICAL, V_HORIZON) / scale
    T = np.maximum(d_tcp / v_seg,
                   np.maximum(dq_step / V_JOINT_CAP, MIN_SEG_T))

    # 3) 进度 u(t) 分段线性 → 高斯卷积(端点外推 = 软起软停)
    t_edge = SETTLE + np.concatenate([[0.0], np.cumsum(T)])
    T_total = float(t_edge[-1]) + SETTLE
    tt = np.arange(0.0, T_total + DT * 0.5, DT)
    tt = tt[tt <= T_total + 1e-9]
    u_nodes = np.linspace(0.0, 1.0, n)
    u_raw = np.interp(tt, t_edge, u_nodes)
    half = int(math.ceil(3 * SIGMA_T / DT))
    x = np.arange(-half, half + 1) * DT
    ker = np.exp(-x ** 2 / (2 * SIGMA_T ** 2))
    ker /= ker.sum()
    u_pad = np.concatenate([np.full(half, u_raw[0]), u_raw,
                            np.full(half, u_raw[-1])])
    u_s = np.convolve(u_pad, ker, mode="same")[half:half + len(u_raw)]
    u_s = u_s / u_s[-1]                              # 终点精确回 home

    # 4) 采样参考(链内线性插值)
    fi = u_s * (n - 1)
    i0 = np.clip(np.floor(fi).astype(int), 0, n - 2)
    fr = (fi - i0)[:, None]
    q_ref = Q[i0] * (1 - fr) + Q[i0 + 1] * fr
    qd_ref = np.gradient(q_ref, DT, axis=0)
    qdd_ref = np.gradient(qd_ref, DT, axis=0)

    # 相位标签(秒界): 卷积使实际切换早/晚 ~2σ, 标签按名义进度
    tt_out_end = float(np.interp(u_nodes[i_weave0], u_s, tt))
    tt_back_start = float(np.interp(u_nodes[i_weave1], u_s, tt))
    phase_t = np.where(tt < tt_out_end, 0,
                       np.where(tt < tt_back_start, 1, 2)).astype(np.int8)

    # 5) 核算
    tcp_ref = np.array([right.fk(q, TCP_OFF)[1] for q in q_ref])
    d_tcp_t = np.linalg.norm(np.diff(tcp_ref, axis=0), axis=1) / DT
    tcp_poly = np.array([right.fk(q, TCP_OFF)[1] for q in P])
    m_weave = phase_t == 1
    dev = np.array([float(np.min(np.linalg.norm(tcp_poly - p, axis=1)))
                    for p in tcp_ref[m_weave]])
    return dict(
        samples=(tt, q_ref, qd_ref, qdd_ref, phase_t, T_total,
                 tt_out_end, tt_back_start),
        qd_peak=float(np.abs(qd_ref).max()),
        qdd_peak=float(np.abs(qdd_ref).max()),
        qd_per_joint=np.abs(qd_ref).max(axis=0),
        qdd_per_joint=np.abs(qdd_ref).max(axis=0),
        dqd_step=float(np.abs(np.diff(qd_ref, axis=0)).max()),
        dev=dev, d_tcp_t=d_tcp_t,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=os.path.join(
        _MOVEIT, "dense_zigzag_right.json"))
    ap.add_argument("--target-total", type=float, default=158.0,
                    help="参考总时长目标 [s]; 只缩放走线速度, 转移段固定")
    ap.add_argument("--out", default=os.path.join(_HERE, "ref_right.npz"))
    ap.add_argument("--plot", action="store_true", default=True)
    ap.add_argument("--no-plot", dest="plot", action="store_false")
    args = ap.parse_args()

    with open(args.ref, encoding="utf-8") as f:
        data = json.load(f)
    P = np.asarray(data["points"], dtype=float)
    # home 必须与真机配置一致(单一权威来源) —— 硬编码副本曾导致
    # 参考轨迹与重标定后的真机配置冲突
    home = load_home(_SAFETY)

    right = UrdfChain(os.path.join(_ISA, "openarm_right.urdf"),
                      tip_link="openarm_right_hand_tcp")

    # 定标(时间膨胀 scale, 全链统一缩放) + 平稳性迭代(超限降速 8%)
    scale = 1.0
    if args.target_total:
        res0 = build_once(P, home, 1.0, right)
        T_move0 = res0["samples"][5] - 2 * SETTLE   # 全部运动时长 @scale=1
        # 转移段与走线统一 TCP 口径 → 全链线性缩放
        scale = max(T_move0 / max(args.target_total - 2 * SETTLE, 1.0), 0.5)
    for attempt in range(10):
        res = build_once(P, home, scale, right)
        if (res["qd_peak"] <= QD_PEAK_MAX
                and res["qdd_peak"] <= QDD_PEAK_MAX):
            break
        scale *= 1.08
        print("[迭代 %d] qd峰值 %.2f / qdd峰值 %.2f 超限 → 降速 ×%.3f"
              % (attempt + 1, res["qd_peak"], res["qdd_peak"], 1.0 / scale))
    else:
        print("[警告] 10 次迭代仍超门限, 采用最后结果(建议人工复核)")

    (tt, q_ref, qd_ref, qdd_ref, phase_t, T_total, tt_out_end,
     tt_back_start) = res["samples"]
    qd_max, qdd_max = res["qd_per_joint"], res["qdd_per_joint"]
    v_weave_t = res["d_tcp_t"][phase_t[1:] == 1]
    dev = res["dev"]
    T_weave = tt_back_start - tt_out_end
    v_h, v_v = V_HORIZON / scale, V_VERTICAL / scale

    print("== 参考轨迹核算 ==")
    print("总时长 %.1fs (静持 %.1f×2 + 转移出 %.1fs + 走线 %.1fs + "
          "转移回 %.1fs)" % (
              T_total, SETTLE, tt_out_end - SETTLE, T_weave,
              T_total - SETTLE - tt_back_start))
    print("速度层级(TCP 统一口径): 水平=转移 %.2f cm/s | "
          "竖直 %.2f cm/s (3×)" % (v_h * 100, v_v * 100))
    print("走线实测 TCP 速率: 中位 %.2f / P95 %.2f cm/s" % (
        np.median(v_weave_t) * 100, np.percentile(v_weave_t, 95) * 100))
    print("峰值关节速度 [rad/s]:", np.round(qd_max, 3),
          "(门限 %.2f)" % QD_PEAK_MAX)
    print("峰值关节加速度 [rad/s²]:", np.round(qdd_max, 2),
          "(门限 %.1f)" % QDD_PEAK_MAX)
    print("参考速度最大单步变化 %.5f rad/s per %dms (无阶跃)" % (
        res["dqd_step"], DT * 1000))
    print("几何保真(走线相位): 采样 TCP 对名义折线最大偏差 %.2f mm"
          % (dev.max() * 1000))

    # ---- 落盘 ----
    meta = dict(
        policy_hz=POLICY_HZ, dt=DT, settle=SETTLE, sigma_t=SIGMA_T,
        w_smooth=W_SMOOTH, res_q=RES_Q,
        v_horizon=float(v_h), v_vertical=float(v_v),
        scale=float(scale), T_total=float(T_total),
        T_out=float(tt_out_end - SETTLE), T_back=float(
            T_total - SETTLE - tt_back_start), T_weave=float(T_weave),
        t_phase_out=float(tt_out_end),
        t_phase_weave_end=float(tt_back_start),
        joint_cap=V_JOINT_CAP, source=os.path.basename(args.ref),
        home_rad=home.tolist(), tcp_off=TCP_OFF,
        qd_peak=float(res["qd_peak"]), qdd_peak=float(res["qdd_peak"]),
        geom_dev_max_m=float(dev.max()),
    )
    np.savez_compressed(
        args.out, t=tt.astype(np.float32), q_ref=q_ref.astype(np.float32),
        qd_ref=qd_ref.astype(np.float32), phase=phase_t,
        meta_json=json.dumps(meta, ensure_ascii=False, indent=1))
    print("已写出 %s (%d 帧 @%dHz)" % (args.out, len(tt), POLICY_HZ))

    # ---- 绘图 ----
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
        ph_c = ["tab:purple", "tab:blue", "tab:green"]
        for ax in axes:
            for p in range(3):
                m = phase_t == p
                if m.any():
                    ax.axvspan(tt[m][0], tt[m][-1], alpha=0.10,
                               color=ph_c[p])
        a = axes[0]
        for j in range(7):
            a.plot(tt, np.degrees(q_ref[:, j]), lw=0.7, label="J%d" % (j + 1))
        a.set_ylabel("q_ref [deg]")
        a.legend(fontsize=7, ncol=4, loc="upper right")
        a.set_title("RL 参考轨迹(右臂): 紫=转移出 蓝=之字走线 绿=转移回")
        a = axes[1]
        for j in range(7):
            a.plot(tt, qd_ref[:, j], lw=0.7)
        a.set_ylabel("qd_ref [rad/s]")
        a = axes[2]
        a.plot(tt[1:], res["d_tcp_t"] * 100, lw=0.6, color="tab:red")
        a.axhline(v_h * 100, ls="--", lw=0.8, color="gray")
        a.axhline(v_v * 100, ls=":", lw=0.8, color="gray")
        a.set_ylabel("TCP 速率 [cm/s]")
        a = axes[3]
        for j in range(7):
            a.plot(tt[1:], qdd_ref[1:, j], lw=0.5)
        a.set_ylabel("qdd_ref [rad/s²]")
        a.set_xlabel("t [s]")
        fig.tight_layout()
        out_png = os.path.join(_fig_dir(), "ref_right_profile.png")
        fig.savefig(out_png, dpi=130)
        print("已写出", out_png)


if __name__ == "__main__":
    main()
