#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_real_run.py —— 真机运行数据分析(rl_exec CSV + 100Hz 黑匣子)
====================================================================
用法: <有 numpy/matplotlib 的环境> analyze_real_run.py <rl_exec.csv> <blackbox.csv>
产出: real_run_report.json + assets/figures/real_run_report.png + 终端摘要

指标口径(重要):
  · 跟踪误差 = 实测 q − q_ref(10Hz 记录), 按相位/按关节分解;
  · HF σ = 减 0.5s 滑动均值后的速度标准差。与 JTC 基线对比时注意:
    基线是 2cm/s 慢速走线(参考 HF≈0), 本任务 6.2cm/s + 快换行,
    故另给"附加抖动" = sqrt(实测HF² − 参考HF²) 做正交分解。
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, '.')
from rl_policy import Policy  # noqa: E402


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


def hf(x, dt, w=0.5):
    n = max(int(w / dt), 5)
    ma = np.convolve(x, np.ones(n) / n, mode='same')
    return float((x - ma).std())


def main(csv_path, bb_path):
    pol = Policy()
    rows = np.genfromtxt(csv_path, delimiter=',', names=True)
    t = rows['t']
    idx = rows['idx'].astype(int)
    Q = np.stack([rows['q%d' % j] for j in range(1, 8)], axis=1)
    QR = np.stack([rows['qref%d' % j] for j in range(1, 8)], axis=1)
    E = np.degrees(Q - QR)
    ph = np.array([pol.phase_of(min(i, pol.n - 1)) for i in idx])
    dur = float(t[-1] - t[0])

    # ---- 黑匣子(用 rl_exec 文件名里的起点做 wall-time 对齐) ----
    # 文件名形如 rl_exec_20260914_103648.csv
    base = csv_path.split('/')[-1]
    date_s, hms = base.replace('rl_exec_', '').split('_')
    t0 = time.mktime(time.strptime(
        "%s-%s-%s %s:%s:%s" % (date_s[0:4], date_s[4:6], date_s[6:8],
                               hms[0:2], hms[2:4], hms[4:6]),
        "%Y-%m-%d %H:%M:%S"))
    B = np.genfromtxt(bb_path, delimiter=',', usecols=(1, 2, 3, 4, 5, 6, 7,
                                                       8, 9))
    epoch = B[:, 0] + B[:, 1] / 1e9
    m = (epoch >= t0 - 1) & (epoch <= t0 + dur + 2.0)
    tb = epoch[m] - t0
    Qb = B[m, 2:9]
    dtb = float(np.median(np.diff(tb)))
    qd = np.gradient(Qb, tb, axis=0)

    hf_real = [hf(qd[:, j], dtb) for j in range(7)]
    hf_ref = [hf(pol.q_ref[:, j], pol.dt) for j in range(7)]
    excess = [float(np.sqrt(max(hf_real[j] ** 2 - hf_ref[j] ** 2, 0)))
              for j in range(7)]

    rep = dict(
        run_csv=csv_path,
        duration_s=dur, planned_s=float(pol.meta["T_total"]),
        frames=len(t), last_idx=int(idx[-1]), ref_frames=pol.n,
        err_deg=dict(
            mean=float(np.abs(E).mean()),
            p95=float(np.percentile(np.abs(E), 95)),
            max=float(np.abs(E).max()),
            per_joint_mean=[float(np.abs(E[:, j]).mean()) for j in range(7)],
            per_joint_max=[float(np.abs(E[:, j]).max()) for j in range(7)],
            per_phase={n: dict(
                dur_s=float(t[ph == p][-1] - t[ph == p][0]),
                mean=float(np.abs(E[ph == p]).mean()),
                max=float(np.abs(E[ph == p]).max()))
                for p, n in enumerate(pol.phase_labels) if (ph == p).any()},
        ),
        end_dev_from_home_deg=float(np.degrees(
            np.abs(Q[-1] - pol.home).max())),
        hf_sigma=dict(real=[float(x) for x in hf_real],
                      ref=[float(x) for x in hf_ref],
                      excess=excess,
                      jtc_baseline={"J1": 0.043, "J2": 0.053}),
        qd_peak=[float(x) for x in np.abs(qd).max(axis=0)],
        watchdog_limit=4.0,
    )
    with open('real_run_report.json', 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)

    print("== 真机运行分析 ==")
    print("时长 %.1fs (计划 %.1fs) | 进度 %d/%d 帧" % (
        dur, rep["planned_s"], idx[-1], pol.n))
    print("误差 mean %.2f° / p95 %.2f° / max %.2f°" % (
        rep["err_deg"]["mean"], rep["err_deg"]["p95"],
        rep["err_deg"]["max"]))
    for n, v in rep["err_deg"]["per_phase"].items():
        print("  %-12s %.1fs  mean %.2f° max %.2f°" % (
            n, v["dur_s"], v["mean"], v["max"]))
    print("每关节 mean/max:", end=" ")
    print(" ".join("J%d %.1f/%.1f" % (j + 1,
          rep["err_deg"]["per_joint_mean"][j],
          rep["err_deg"]["per_joint_max"][j]) for j in range(7)))
    print("HF σ 实测 J1 %.4f J2 %.4f (JTC 基线 0.043/0.053; 本任务参考自身 "
          "HF %.4f/%.4f, 附加抖动 %.4f/%.4f)" % (
              hf_real[0], hf_real[1], hf_ref[0], hf_ref[1],
              excess[0], excess[1]))
    print("|qd| 峰值:", np.round(rep["qd_peak"], 2), "(watchdog 4.0)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
        pc = ["tab:purple", "tab:blue", "tab:green"]
        for a in ax:
            for p in range(3):
                mm = ph == p
                if mm.any():
                    a.axvspan(t[mm][0], t[mm][-1], alpha=0.08, color=pc[p])
        ax[0].plot(t, np.degrees(Q[:, 1]), lw=0.6, label="J2 实测")
        ax[0].plot(t, np.degrees(QR[:, 1]), lw=0.8, ls="--", alpha=0.6,
                   label="J2 参考")
        ax[0].set_ylabel("J2 [deg]")
        ax[0].legend(fontsize=8)
        ax[0].set_title("真机全程运行: 紫=转移出 蓝=走线 绿=转移回")
        for j in range(7):
            ax[1].plot(t, E[:, j], lw=0.5)
        ax[1].set_ylabel("跟踪误差 [deg]")
        ax[2].plot(tb, qd[:, 1], lw=0.3, alpha=0.6)
        w = max(int(0.5 / dtb), 5)
        ax[2].plot(tb, np.convolve(qd[:, 1], np.ones(w) / w, mode='same'),
                   lw=1.2, color='k')
        ax[2].set_ylabel("J2 qd [rad/s] 灰=原始 黑=0.5s均值")
        ax[2].set_xlabel("t [s]")
        fig.tight_layout()
        fig.savefig(os.path.join(_fig_dir(), 'real_run_report.png'), dpi=130)
        print("已写出 real_run_report.json / assets/figures/real_run_report.png")
    except Exception as e:  # noqa: BLE001
        print("绘图失败:", e)


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else 'rl_exec_20260914_103648.csv'
    bb = sys.argv[2] if len(sys.argv) > 2 else None
    if bb is None:  # 2026-09-22 起黑匣子落当前目录(时间戳名); 兼容旧 /tmp 路径
        import glob as _g
        cands = sorted(_g.glob('js_blackbox_*.csv'))
        bb = cands[-1] if cands else '/tmp/js_blackbox.csv'
    main(csv, bb)
