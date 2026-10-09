#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每关节二阶动态辨识: 指令位置 → 实际位置
================================================================
模型(MIT PD 闭环等效二阶):
    q_cmd − q = A1·q̈ + A2·q̇ + B0
    A1 = J/k  ⇒ 固有频率 ω_n = 1/√A1
    A2 = c/k  ⇒ 速度跟踪滞后时间常数(秒)
    B0 = −τ_g/k ⇒ 重力 droop 残差(积分器补偿后的稳态偏置)

方法(每关节两段式):
  1. SG 滤波微分 → 线性回归 [q̈, q̇, 1] → 初值;
  2. 输出误差法精修: 用 (A1,A2,B0) 仿真二阶plants驱动同一 q_cmd,
     Nelder-Mead 最小化 ∫(q_sim − q_meas)² —— 避开数值微分噪声。

用法: /usr/bin/python3 identify_params.py [--data ident_data.npz]
       [--selftest]   # 合成数据自检(无需硬件)
产出: ident_result.json + 终端报告(MPC 参数建议)
"""

import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.signal import savgol_filter

_HERE = os.path.dirname(os.path.abspath(__file__))


def simulate_plant(q_cmd, t, A1, A2, B0, q0):
    """二阶 plants: A1·q̈ = (q_cmd − q) − A2·q̇ − B0。
    每个输出采样(测量网格)之间 Heun 子步积分, 输入线性插值。"""
    n = len(t)
    q = np.empty(n)
    v = np.empty(n)
    q[0], v[0] = q0, 0.0
    sub = 5                                   # 每输出步 5 个子步
    h = (t[1] - t[0]) / sub if n > 1 else 0.001
    qh, vh = q0, 0.0
    for k in range(1, n):
        t0_ = t[k - 1]
        for s in range(sub):
            qc0 = float(np.interp(t0_ + s * h, t, q_cmd))
            qc1 = float(np.interp(t0_ + (s + 1) * h, t, q_cmd))
            a0 = ((qc0 - qh) - A2 * vh - B0) / A1
            qm = qh + 0.5 * h * vh
            vm = vh + 0.5 * h * a0
            am = ((qc1 - qm) - A2 * vm - B0) / A1
            qh += h * vm
            vh += h * am
        q[k], v[k] = qh, vh
    return q, v


def fit_output_error(q_cmd, t, q_meas, q0, x0):
    """输出误差法: 优化 (A1, A2, B0) 最小化仿真输出与实测的偏差。"""

    def cost(x):
        A1, A2, B0 = x
        if A1 <= 1e-3 or A2 < 0 or not all(
                math.isfinite(v) for v in x):
            return 1e12
        q_sim, _ = simulate_plant(q_cmd, t, A1, A2, B0, q0)
        m = float(np.mean((q_sim - q_meas) ** 2))
        return m if math.isfinite(m) else 1e12

    res = minimize(cost, x0, method="Nelder-Mead",
                   options={"maxiter": 400, "xatol": 1e-5, "fatol": 1e-10})
    return res.x, math.sqrt(res.fun)


def identify_joint(t, q_meas, q_cmd, sg_win=31):
    """单关节: 修剪边缘 → SG 微分回归初值 → 输出误差精修。"""
    # 修剪前后各 0.6s(JTC 起停瞬态)
    keep = (t > t[0] + 0.6) & (t < t[-1] - 0.6)
    t, q_meas, q_cmd = t[keep], q_meas[keep], q_cmd[keep]
    dt = float(np.median(np.diff(t)))
    win = min(sg_win if sg_win % 2 else sg_win + 1, (len(t) // 4) * 2 + 1)
    q_f = savgol_filter(q_meas, win, 3)
    dq = savgol_filter(q_meas, win, 3, deriv=1, delta=dt)
    ddq = savgol_filter(q_meas, win, 3, deriv=2, delta=dt)
    X = np.column_stack([ddq, dq, np.ones_like(ddq)])
    y = q_cmd - q_f
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    A1_0, A2_0, B0_0 = max(coef[0], 1e-3), max(coef[1], 0.0), coef[2]
    (A1, A2, B0), rmse = fit_output_error(
        q_cmd, t, q_meas, q_meas[0], [A1_0, A2_0, B0_0])
    q_sim, _ = simulate_plant(q_cmd, t, A1, A2, B0, q_meas[0])
    ss_res = float(np.sum((q_meas - q_sim) ** 2))
    ss_tot = float(np.sum((q_meas - q_meas.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    wn = 1.0 / math.sqrt(A1) if A1 > 0 else 0.0
    zeta = A2 / (2.0 * math.sqrt(A1)) if A1 > 0 else 0.0
    return dict(A1=A1, A2=A2, B0=B0, wn_rad_s=wn, zeta=zeta,
                rmse_deg=math.degrees(rmse), r2=r2,
                init=dict(A1=A1_0, A2=A2_0, B0=B0_0))


def selftest():
    """合成数据自检: 已知参数二阶 plants + 噪声 → 验证辨识回收。"""
    sys.path.insert(0, _HERE)
    from excite_ident import FREQS, AMPS, PHASE, D2R
    truth = [dict(A1=0.040, A2=0.060, B0=math.radians(-0.5)),
             dict(A1=0.025, A2=0.045, B0=math.radians(-0.3))]
    rng = np.random.default_rng(7)
    secs = 22.0
    t = np.arange(0, secs, 0.01)
    print("==== 自检(合成数据, 已知真值) ====")
    ok_all = True
    for k, tv in enumerate(truth):
        def wave(tt, k_=k):
            return sum(a * D2R * math.sin(2 * math.pi * f * tt + ph)
                       for f, a, ph in zip(FREQS, AMPS, PHASE))
        q_cmd = np.array([wave(tt) for tt in t])
        qs, _ = simulate_plant(q_cmd, t, tv["A1"], tv["A2"], tv["B0"],
                               q_cmd[0])
        q_meas = qs + rng.normal(0, math.radians(0.05), len(t))  # 0.05° 噪声
        r = identify_joint(t, q_meas, q_cmd)
        e1 = abs(r["A1"] - tv["A1"]) / tv["A1"] * 100
        e2 = abs(r["A2"] - tv["A2"]) / tv["A2"] * 100
        e0 = math.degrees(abs(r["B0"] - tv["B0"]))
        ok = e1 < 10 and e2 < 10 and e0 < 0.1 and r["r2"] > 0.95
        ok_all = ok_all and ok
        print("关节%d: A1 %.4f(真%.3f, 误差%.1f%%) A2 %.4f(真%.3f, %.1f%%) "
              "B0 %+.3f°(真%+.2f°) | ωn=%.2frad/s ζ=%.2f R²=%.4f %s"
              % (k + 1, r["A1"], tv["A1"], e1, r["A2"], tv["A2"], e2,
                 math.degrees(r["B0"]), math.degrees(tv["B0"]), r["wn_rad_s"], r["zeta"],
                 r["r2"], "✓" if ok else "✗"))
    print("自检%s" % ("通过 ✓" if ok_all else "未通过 ✗"))
    return ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(_HERE, "ident_data.npz"))
    ap.add_argument("--sg-win", type=int, default=31,
                    help="微分滤波窗(采样点, 默认 31≈0.31s)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if selftest() else 1)

    d = np.load(args.data, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    t_js = d["t_js"]
    q_js = d["q_js"]
    t_ref = d["t_ref"]
    q_ref = d["q_ref"]
    names = [str(x) for x in d["names"]]
    dt_js = float(np.median(np.diff(t_js)))
    print("JS 采样: %d 点, 中位 dt=%.1fms | 指令点 %d"
          % (len(t_js), dt_js * 1000, len(t_ref)))
    results = {}
    for seg in meta["segments"]:
        j = seg["joint"]
        # 优先用激励段纯波形窗口(排除接近/退回缓动段)
        ws = seg.get("wave_start", seg["t_start"] + 0.5)
        we = seg.get("wave_end", seg["t_end"] - 0.5)
        m = (t_js >= ws) & (t_js <= we)
        if m.sum() < 200:
            print("J%d: 窗口内样本不足(%d), 跳过" % (j, m.sum()))
            continue
        t = t_js[m]
        q_meas = q_js[m, j - 1]
        q_cmd = np.interp(t, t_ref, q_ref[:, j - 1])
        r = identify_joint(t, q_meas, q_cmd, sg_win=args.sg_win)
        results[j] = r
        print("J%d: A1=%.4fs² A2=%.4fs B0=%+.3f° | ωn=%.2frad/s "
              "(%.2fHz) ζ=%.2f | RMSE=%.3f° R²=%.4f"
              % (j, r["A1"], r["A2"], math.degrees(r["B0"]), r["wn_rad_s"],
                 r["wn_rad_s"] / 2 / math.pi, r["zeta"],
                 r["rmse_deg"], r["r2"]))

    print("\n==== MPC 参数建议 ====")
    for j, r in sorted(results.items()):
        if r["zeta"] < 0.35:
            print("J%d: ζ=%.2f 偏欠阻尼 → MIT kd 可上调试验; "
                  "MPC w_v 建议加倍" % (j, r["zeta"]))
        lag_ms = r["A2"] * 1000
        print("J%d: 指令→实际滞后 ≈ %.0fms(MPC dt=10ms 的 %.1f 倍), "
              "droop 残差 %+.3f°(积分器残余)"
              % (j, lag_ms, lag_ms / 10, r["B0"]))
    with open(os.path.join(_HERE, "ident_result.json"), "w",
              encoding="utf-8") as f:
        json.dump({"meta": meta, "dt_js_s": dt_js, "results": results},
                  f, ensure_ascii=False, indent=1)
    print("已写出 ident_result.json")


if __name__ == "__main__":
    main()
