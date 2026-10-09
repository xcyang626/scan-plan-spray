#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ILC 仿真验证实验（用于技术文档第 7 章）
========================================
对象模型: 关节空间双积分 + 粘滞/库仑摩擦 + 重力偏置（代表性参数，
          需实测辨识替换; 解耦假设适用条件见文档 7.1 节）。
控制器:   与真机一致的 PD 反馈(kp=70, kd=3.0, 10ms 周期) + ILC 前馈 u_k。
ILC:      P 型 + 时间超前采样 + 零相位 Q 滤波器（实验 A/B: 有/无滤波器，
          实验 C: 大增益发散对照）。
轨迹:     dense_zigzag_right.json 真实 J5 关节轨迹片段(811 点中取 4.0s)。
输出:     ilc_sim_results.json (逐迭代误差) + assets/figures/ilc_sim_curves.png。
"""
import json
import os

import numpy as np
from scipy.signal import butter, filtfilt

_HERE = os.path.dirname(os.path.abspath(__file__))
# 输入轨迹: 项目根目录 仿真/moveit_sim/ 下的右臂稠密之字形轨迹(相对本文件定位,
# 项目目录整体移动后依然有效)
DENSE = os.path.join(_HERE, "..", "仿真", "moveit_sim", "dense_zigzag_right.json")


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


# ---------------- 仿真配置 ----------------
DT = 0.010            # 控制周期 10ms(与 mpc_core 一致)
KP, KD = 70.0, 3.0    # 真机 J2 实测 PD 增益(openarm_zero_hw)
GAMMA = 0.6           # 学习增益(实验主组)
GAMMA_BAD = 2.0       # 大增益对照组(实验 C)
N_ITER = 12           # 主组迭代次数
U_LIM = 3.0           # ILC 前馈限幅 Nm(安全钳位)
NOISE_STD = 0.0002    # 编码器测量噪声 rad(≈0.011°, 14bit 量级)

# 真实对象参数(含不确定性; ILC 不知道这些)
J_TRUE, B_TRUE = 1.15, 0.92         # 惯量/粘滞(文档示例值, 失配 15%)
FC_TRUE = 0.35                       # 库仑摩擦 Nm
GRAV_BIAS = 1.8                      # 重力偏置 Nm(近似单关节)
LOAD_TORQUE = 0.25                   # 周期内慢变负载扰动 Nm

# 轨迹: 真实 J5 关节轨迹平滑段(避开 0-450 点的腕部跳变带),
# 按真机重定时节奏压缩: 每步 0.05s(真实 dt≈0.1s×0.4 speed-scale) → 10s/1000 周期
with open(DENSE, encoding="utf-8") as f:
    pts = np.array(json.load(f)["points"])[:, 4]
seg = pts[500:700]                                  # 200 点平滑段
qd = np.interp(np.linspace(0, 1, 1000), np.arange(len(seg)), seg)
qd_dot = np.gradient(qd, DT)
N = len(qd)
print("轨迹段: J5[500:700], N=%d, 峰值角速度=%.2f rad/s" % (N, np.abs(qd_dot).max()))


def plant(q, dq, tau, t):
    """真实对象: 双积分 + 摩擦 + 重力 + 负载扰动。"""
    fric = B_TRUE * dq + FC_TRUE * np.tanh(dq / 0.05)
    load = LOAD_TORQUE * np.sin(2 * np.pi * t / 4.0)      # 慢变负载
    ddq = (tau - fric - GRAV_BIAS - load) / J_TRUE
    return ddq


def q_filter(u, enabled, fc=8.0):
    """零相位低通 Q 滤波器(2 阶 Butterworth, filtfilt 无相位滞后)。"""
    if not enabled:
        return u
    b, a = butter(2, fc / (0.5 / DT), btype="low")
    return filtfilt(b, a, u)


def run_ilc(gamma, use_q, n_iter, seed=0):
    """返回 (err_rms[k], err_max[k], u_final, e_last)。"""
    rng = np.random.default_rng(seed)
    u = np.zeros(N)                                   # ILC 前馈(初始 0)
    hist_rms, hist_max = [], []
    e_last = None
    for k in range(n_iter):
        q = qd[0] - GRAV_BIAS / (KP + 1e-9)           # 起点带稳态下垂
        dq = 0.0
        e_rec = np.zeros(N)
        for i in range(N):
            # 反馈 + ILC 前馈
            e_meas = qd[i] - (q + rng.normal(0, NOISE_STD))
            tau = KP * (qd[i] - q) + KD * (qd_dot[i] - dq) + u[i]
            ddq = plant(q, dq, tau, i * DT)
            dq += ddq * DT
            q += dq * DT
            e_rec[i] = qd[i] - q
        e_last = e_rec
        hist_rms.append(float(np.sqrt(np.mean(e_rec ** 2))))
        hist_max.append(float(np.max(np.abs(e_rec))))
        # 学习更新: 时间超前采样 + Q 滤波 + 限幅
        e_lead = np.concatenate([e_rec[1:], [e_rec[-1]]])
        u_new = u + gamma * e_lead
        u_new = q_filter(u_new, use_q)
        u = np.clip(u_new, -U_LIM, U_LIM)
    return hist_rms, hist_max, u, e_last


# ---- 实验 A: 主组 gamma=0.6 + Q 滤波 ----
rmsA, maxA, uA, eA = run_ilc(GAMMA, True, N_ITER)
# ---- 实验 B: 无 Q 滤波器(对照) ----
rmsB, maxB, uB, eB = run_ilc(GAMMA, False, N_ITER)
# ---- 实验 C: 大增益 gamma=2.0(发散对照, 有 Q) ----
rmsC, maxC, uC, eC = run_ilc(GAMMA_BAD, True, N_ITER)

print("== 实验 A: gamma=%.1f + Q滤波(8Hz零相位) ==" % GAMMA)
for k, (r, m) in enumerate(zip(rmsA, maxA)):
    print("  iter %2d: RMS=%.4f rad (%.3f°)  max=%.4f rad (%.2f°)"
          % (k, r, np.degrees(r), m, np.degrees(m)))
print("== 实验 B: 无 Q 滤波器 ==")
print("  iter 0: RMS=%.4f | iter %d: RMS=%.4f (终值)" % (rmsB[0], len(rmsB) - 1, rmsB[-1]))
print("== 实验 C: gamma=%.1f(过学习对照) ==" % GAMMA_BAD)
for k in (0, 1, 2, 3, len(rmsC) - 1):
    print("  iter %2d: RMS=%.4f rad" % (k, rmsC[k]))

# ---- 落盘 ----
out = {
    "config": {"dt": DT, "kp": KP, "kd": KD, "gamma": GAMMA, "gamma_bad": GAMMA_BAD,
               "n_iter": N_ITER, "u_lim": U_LIM, "q_filter": "2阶Butterworth 8Hz 零相位",
               "noise_std_rad": NOISE_STD,
               "plant": {"J": J_TRUE, "B": B_TRUE, "fc": FC_TRUE,
                         "grav_bias": GRAV_BIAS, "load_amp": LOAD_TORQUE},
               "traj": "dense_zigzag_right.json J5 前400点重采样4s"},
    "expA_main": {"rms": rmsA, "max": maxA},
    "expB_nofilter": {"rms": rmsB, "max": maxB},
    "expC_highgamma": {"rms": rmsC, "max": maxC},
}
with open(os.path.join(_HERE, "ilc_sim_results.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

# ---- 绘图 ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t = np.arange(N) * DT
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
it = np.arange(1, N_ITER + 1)
ax = axes[0]
ax.semilogy(it, np.degrees(rmsA), "o-", label="RMS")
ax.semilogy(it, np.degrees(maxA), "s--", label="max")
ax.set_xlabel("iteration k")
ax.set_ylabel("tracking error [deg]")
ax.set_title("A: main (gamma=0.6, Q-filter on)")
ax.grid(alpha=0.3, which="both")
ax.legend()

ax = axes[1]
ax.semilogy(it, np.degrees(rmsB), "o-", label="no Q-filter")
ax.semilogy(it, np.degrees(rmsA), "s--", label="with Q-filter")
ax.set_xlabel("iteration k")
ax.set_title("B: Q-filter effect (gamma=0.6)")
ax.grid(alpha=0.3, which="both")
ax.legend()

ax = axes[2]
ax.semilogy(np.arange(1, N_ITER + 1), np.degrees(rmsC), "o-", color="tab:red",
            label="gamma=2.0 (bad)")
ax.semilogy(np.arange(1, N_ITER + 1), np.degrees(rmsA), "s--", color="tab:blue",
            label="gamma=0.6 (good)")
ax.set_xlabel("iteration k")
ax.set_title("C: over-learning divergence check")
ax.grid(alpha=0.3, which="both")
ax.legend()
fig.suptitle("ILC convergence on real zigzag trajectory (J5), PD=J2 real gains, dt=10ms")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(_fig_dir(), "ilc_sim_curves.png"), dpi=140)
print("已写出 ilc_sim_results.json / assets/figures/ilc_sim_curves.png")
