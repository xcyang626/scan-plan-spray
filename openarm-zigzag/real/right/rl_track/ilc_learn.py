#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ilc_learn.py —— 从 rl_exec 运行 CSV 生成 ILC 前馈修正表(离线, 不接触机器人)
================================================================================
背景(2026-09-21 全程首跑 73.4s 急停复盘): URDF 外未建模负载(线缆拖拽)在悬伸
位形增大 → 重力前馈欠补偿 → 驱动软件积分器补上又过冲 → 低频猎振。

原理 —— 欠账力矩重构(不是裸误差学习):
  驱动力矩(关节系) = ff + kp·e_cmd + kd·ė_cmd + integ
    e_cmd = cmd − q(CSV 直接有); integ = ki·∫EMA(e_cmd)dt —— 误差的确定性
    函数, 从 CSV 按 50Hz 逐步重构(EMA α=0.3 / 钳位 / 清零规则与
    zero_offset_hw.cpp 一致; e_cmd 峰值 0.03rad << 0.5rad 清零线, 不触发)。
  准静态下 τ_fb = kp·e + kd·ė + integ ≈ 未建模负载 + 重力模型残差,
  其低频分量(随位形慢变)正是可前馈学习的内容; 猎振本身(<0.3Hz 闭
  环极限环, 相位锁定于本次运行的积分器状态)不可学习, 用 Q 滤波器剔除。

学习律(迭代 k+1, 本表为 iter-1, Δτ_0 = 0):
  Δτ_{k+1}[idx] = clip( γ · LPF_0.08Hz{ τ_fb[idx] } , ±clamp )
    γ = 0.6(压缩因子 0.4/迭代, 文档 §4.2, 对重构误差 ±50% 鲁棒);
    clamp = ±[3,3,3,2,1,1,1] Nm(文档 §6, 在驱动 ff 钳位 15/15/20/20/5³ 内)。

边界处理:
  · t < 1.5s(前馈淡入窗)与急停前 0.5s 不学习;
  · 覆盖段(有数据的 idx)之后: 默认 --extend hold(保持末值 —— 第二半程
    悬伸更深, 负载有增无减的一阶外推; 受钳位约束, 由 iter-2 实测替换),
    可选 zero(回退为不补偿);
  · 训练含 FF DR 0.8~1.2, 策略容忍此量级前馈修正, 无需重训。

部署/回滚(运行时零代码改动):
    /usr/bin/python3 rl_exec.py --log-csv --grav-ff-file grav_ff_right_ilc1.npz
    回滚 = 去掉 --grav-ff-file(回到 grav_ff_right.npz 基线)。

用法(系统 python3, 需 numpy/scipy/matplotlib):
    /usr/bin/python3 ilc_learn.py --csv rl_exec_20260921_160512.csv
输出: grav_ff_right_ilc1.npz / ilc_delta_right_iter1.npz
      ilc_learn_report.json / ilc_learn_curves.png
⚠ 本脚本纯离线; 上机执行一律由操作员按 rl_track/README.md 规程进行。
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np

try:
    from scipy.signal import butter, filtfilt
except ImportError:            # 延迟到调用时才需要 scipy(见 lpf0)
    butter = filtfilt = None

_HERE = os.path.dirname(os.path.abspath(__file__))


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

# ---- 驱动常量(单一权威来源审计: verify_deploy.py A 节已核对三方一致) ----
KP = np.array([120.0, 100.0, 70.0, 126.3, 30.0, 30.0, 30.0])   # URDF/训练一致
KD = np.array([7.4, 12.2, 7.5, 8.04, 1.5, 1.5, 1.5])
KI = np.array([12.0, 12.0, 12.0, 10.0, 8.0, 8.0, 8.0])          # 驱动默认
ICLAMP = np.array([10.0, 10.0, 6.0, 8.0, 3.0, 3.0, 3.0])        # 积分器钳位
EMA_A = 0.3                                                     # 驱动 EMA α

# ---- ILC 参数(docs/ILC可行性_实施路径文档.md §4/§6 + 0922 修订) ----
GAMMA = 0.6             # 学习增益(压缩因子 0.4/迭代)
FC_HZ = 0.08            # Q 低通截止: 保留负载慢变, 剔除 <0.3Hz 猎振带
TAU_CLAMP = np.array([3.0, 3.0, 3.0, 2.0, 1.0, 1.0, 1.0])       # Δτ 钳位 [Nm]
MASK_HEAD_S = 1.5       # 前馈淡入窗不学习
RAMP_S = 2.0            # 学习段起点 Δτ 线性渐入时长(防力矩阶跃)
MASK_TAIL_S = 0.5       # 急停瞬变不学习
TABLE_HZ = 50.0         # 输出表节拍(与 ref/驱动注入同 idx)


def lpf0(x, fc, fs):
    """2 阶零相位 Butterworth 低通(逐列)。"""
    if butter is None:
        raise RuntimeError("需要 scipy(filtfilt)——用系统 python3 运行")
    b, a = butter(2, fc / (0.5 * fs), btype="low")
    return filtfilt(b, a, x, axis=0)


def load_run_csv(path):
    """rl_exec CSV → (idx, q, cmd, qref, ff)。校验: 无断档/单调/含 ff 列。"""
    d = np.genfromtxt(path, delimiter=",", names=True)
    need = (["t", "idx"] + ["q%d" % (j + 1) for j in range(7)]
            + ["cmd%d" % (j + 1) for j in range(7)]
            + ["qref%d" % (j + 1) for j in range(7)])
    missing = [k for k in need if k not in d.dtype.names]
    if missing:
        raise SystemExit("CSV 缺列: %s" % missing)
    idx = d["idx"].astype(np.int64)
    if np.any(np.diff(idx) <= 0):
        raise SystemExit("idx 非单调 —— 不是单次连续运行?")
    gap = np.diff(idx) - np.diff(idx).min()
    if gap.max() > 0:
        print("⚠ idx 断档 %d 处(最大缺 %d 帧) —— 断档区间线性插值"
              % (int((gap > 0).sum()), int(gap.max())))
    col = lambda p: np.column_stack([d["%s%d" % (p, j + 1)] for j in range(7)])
    has_ff = all(("ff%d" % (j + 1)) in d.dtype.names for j in range(7))
    return dict(idx=idx, t=d["t"], q=col("q"), cmd=col("cmd"),
                qref=col("qref"), ff=col("ff") if has_ff else None)


def reconstruct_feedback_tau(run, verbose=True):
    """重构驱动反馈力矩 τ_fb = kp·e + kd·ė + integ(关节系, 50Hz idx 网格)。
    ⚠ 序列在 50Hz 轨迹帧网格上(每样本 0.02s), 内部一律按 TABLE_HZ 计算。
    2026-09-22 修复: 初版误传 CSV 的 10Hz —— 积分器重构快 5×/kd 项小 5×/
    Q 实际截止 5×(猎振带 0.1~0.3Hz 漏入表内, 即 iter-1 复验 J2/J3 反相
    打架的根源); 孪生工具 ilc_check_real 自身 fs 自洽, 不受影响。"""
    idx, q, cmd = run["idx"], run["q"], run["cmd"]
    i0, i1 = int(idx[0]), int(idx[-1])
    grid = np.arange(i0, i1 + 1)                      # 50Hz 帧号
    e = np.column_stack([np.interp(grid, idx, cmd[:, j] - q[:, j])
                         for j in range(7)])          # e_cmd = cmd − q
    # ė_cmd: 差分 + 轻度平滑(后续 0.08Hz 低通会再压噪声)
    edot = np.gradient(e, 1.0 / TABLE_HZ, axis=0)
    edot = lpf0(edot, 1.0, TABLE_HZ)
    # 积分器重构(zero_offset_hw.cpp 同律: EMA→钳位→累加→integ 钳位)
    integ = np.zeros(7)
    err_f = np.zeros(7)
    integ_rec = np.empty_like(e)
    dt = 1.0 / TABLE_HZ
    for i in range(len(e)):
        err_f = EMA_A * e[i] + (1.0 - EMA_A) * err_f
        big = np.abs(err_f) > 0.5
        err_f[big] = 0.0
        integ[big] = 0.0
        integ = np.clip(integ + KI * err_f * dt, -ICLAMP, ICLAMP)
        integ_rec[i] = integ
    tau_fb = KP * e + KD * edot + integ_rec
    if verbose:
        print("  重构积分器峰值: %s Nm (钳位 %s, 未触限)"
              % (np.round(np.abs(integ_rec).max(axis=0), 2).tolist(),
                 ICLAMP.astype(int).tolist()))
    return grid, e, tau_fb


def main():
    ap = argparse.ArgumentParser(description="ILC 前馈修正表生成(离线)")
    ap.add_argument("--csv", default=os.path.join(
        _HERE, "rl_exec_20260921_160512.csv"),
        help="rl_exec 运行记录(默认 0921 全程首跑)")
    ap.add_argument("--ref", default=os.path.join(_HERE, "ref_right.npz"))
    ap.add_argument("--base-ff", default=os.path.join(
        _HERE, "grav_ff_right.npz"), help="基线前馈表(部署权威副本)")
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--fc", type=float, default=FC_HZ,
                    help="Q 低通截止 [Hz](默认 0.08; iter-2 起 0.03 只跟真慢漂移)")
    ap.add_argument("--prev-delta", default=None,
                    help="上一迭代已部署的 Δτ 表(iter-2 起; 累积律 "
                         "Δτ_new = clip(γ·LPF{τ̂+Δτ_prev}), 自动继承已生效部分")
    ap.add_argument("--clamps", default=None,
                    help="逐关节 Δτ 钳位逗号列表(默认 3,3,3,2,1,1,1; "
                         "积分器仍贴钳位的关节可提, 须 < 驱动 ff 钳位)")
    ap.add_argument("--iter-tag", default="iter1")
    ap.add_argument("--extend", choices=["hold", "zero"], default="hold",
                    help="覆盖段之后: hold=保持末值(默认), zero=不补偿")
    ap.add_argument("--out-table", default=None)
    ap.add_argument("--out-delta", default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    clamps = TAU_CLAMP
    if args.clamps:
        clamps = np.array([float(x) for x in args.clamps.split(",")])
        assert len(clamps) == 7, "钳位须 7 个"
    prev = None
    if args.prev_delta:
        prev = np.load(args.prev_delta)["delta"]

    print("== ILC 学习: %s ==" % os.path.basename(args.csv))
    run = load_run_csv(args.csv)
    fs = 1.0 / np.median(np.diff(run["t"]))
    print("  记录: %d 行, %.1f~%.1fs, 采样 %.1fHz, idx %d~%d"
          % (len(run["t"]), run["t"][0], run["t"][-1], fs,
             run["idx"][0], run["idx"][-1]))

    grid, e, tau_fb = reconstruct_feedback_tau(run)
    # 学习信号: 当前总欠账 = τ̂(反馈实际供给) + 上一迭代已部署的 Δτ;
    # 只取低频(可学习)分量, γ 折扣后为新表(累积律, 对未漂移负载自动恒定)
    if prev is not None:
        tau_fb = tau_fb + prev[grid]
        print("  累积: τ̂ += Δτ_prev(逐帧), 滤波 fc=%.2fHz(真 50Hz 口径)"
              % args.fc)
    tau_learn = lpf0(tau_fb, args.fc, TABLE_HZ)
    delta = args.gamma * tau_learn
    delta = np.clip(delta, -clamps, clamps)

    # 边界掩蔽: 淡入窗 + 急停尾; 起点线性渐入(防 Δτ 力矩阶跃)
    t_grid = grid / TABLE_HZ
    t0, t1 = run["t"][0], run["t"][-1]
    mask = ((t_grid >= t0 + MASK_HEAD_S)
            & (t_grid <= t1 - MASK_TAIL_S))
    delta[~mask] = 0.0
    ramp = np.clip((t_grid - (t0 + MASK_HEAD_S)) / RAMP_S, 0.0, 1.0)
    delta *= ramp[:, None]
    n_cov = int(mask.sum())
    print("  学习窗口: %.1f~%.1fs(%d 帧, 掩蔽首 %.1fs/尾 %.1fs)"
          % (t_grid[mask][0], t_grid[mask][-1], n_cov,
             MASK_HEAD_S, MASK_TAIL_S))

    ref = np.load(args.ref, allow_pickle=True)
    n_ref = ref["q_ref"].shape[0]
    T_ref = n_ref / TABLE_HZ
    full = np.zeros((n_ref, 7))
    g_lo, g_hi = int(grid[0]), int(grid[-1])
    n_end = min(g_hi, n_ref - 1)
    full[g_lo:n_end + 1] = delta[:n_end - g_lo + 1]
    # 学习窗末帧的值一路保持到轨迹末: 覆盖段内被掩蔽的尾部与覆盖段之后
    # 都无阶跃(急停污染段只影响学习, 不影响部署表)
    i_le = int(np.nonzero(mask)[0][-1])             # 学习窗最后帧
    hold_val = delta[i_le].copy()
    if args.extend == "zero":
        print("  学习窗后(%.1fs~)不补偿 (--extend zero)"
              % ((g_lo + i_le + 1) / TABLE_HZ))
    else:
        full[g_lo + i_le:] = hold_val
        if n_ref > g_lo + i_le + 1:
            print("  %.1fs~轨迹末(%.1fs): 保持学习窗末值 %s Nm (--extend hold)"
                  % ((g_lo + i_le + 1) / TABLE_HZ, T_ref,
                     np.round(hold_val, 2).tolist()))
    step = np.abs(np.diff(full, axis=0)).max(axis=0)
    if step.max() > 0.1:
        print("  ⚠ Δτ 帧间跳变超 0.1Nm/帧: %s —— 检查边界/断档"
              % np.round(step, 3).tolist())
    else:
        print("  Δτ 帧间跳变 ≤ %.3f Nm/帧(平滑)" % step.max())

    # ---- 量化报告 ----
    err_deg = np.degrees(e)
    print("\n  本跑(输入)误差: mean|qref-q| 未知此表; e_cmd mean %.2f° / "
          "max %.2f°" % (np.abs(err_deg).mean(), np.abs(err_deg).max()))
    print("  欠账重构 τ_fb 低频分量峰值 [Nm]: %s"
          % np.round(np.abs(tau_learn).max(axis=0), 2).tolist())
    print("  Δτ(iter-%s, γ=%.2f) 峰值 [Nm]: %s  钳位 ±%s"
          % (args.iter_tag.replace("iter", ""), args.gamma,
             np.round(np.abs(full[g_lo:n_end + 1]).max(axis=0), 2).tolist(),
             clamps.astype(int).tolist()))
    # 预测: 学习带内残差 = τ̂ − Δτ = (1−γ)·τ̂
    resid = (1.0 - args.gamma) * tau_learn
    print("  预测学习带内残差(×%.0f%%): 峰值 %s Nm → 下次迭代再收 60%%"
          % ((1 - args.gamma) * 100,
             np.round(np.abs(resid).max(axis=0), 2).tolist()))

    base = np.load(args.base_ff)["gff"]
    if len(base) < n_ref:
        raise SystemExit("基线前馈表行数 %d < 参考 %d" % (len(base), n_ref))
    comp = base[:n_ref] + full
    print("  合成表峰值 %.2f Nm @J%d (基线 %.2f; 驱动 ff 钳位 15/15/20/20/5³)"
          % (np.abs(comp).max(), int(np.abs(comp).max(axis=0).argmax()) + 1,
             np.abs(base).max()))

    def sha(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]

    meta = dict(
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        tool="ilc_learn.py(欠账重构律, 2026-09-22)",
        source_csv=os.path.abspath(args.csv), source_csv_sha16=sha(args.csv),
        base_ff=os.path.abspath(args.base_ff), base_ff_sha16=sha(args.base_ff),
        ref=os.path.abspath(args.ref), n_ref=int(n_ref),
        covered_idx=[int(grid[0]), int(n_end)],
        extend=args.extend, gamma=args.gamma, fc_hz=args.fc,
        mask=[MASK_HEAD_S, MASK_TAIL_S], ramp_s=RAMP_S,
        tau_clamp=clamps.tolist(),
        prev_delta=(os.path.abspath(args.prev_delta)
                    if args.prev_delta else None),
        drive_const=dict(kp=KP.tolist(), kd=KD.tolist(), ki=KI.tolist(),
                         iclamp=ICLAMP.tolist(), ema_a=EMA_A),
        note="Δτ = clip(γ·LPF{kp·e_cmd + kd·ė_cmd + integ}, ±clamp); "
             "integ 由 CSV 按 50Hz 重构; 猎振(<0.3Hz)由 Q 滤波剔除不学习",
    )
    out_tab = args.out_table or os.path.join(
        _HERE, "grav_ff_right_%s.npz" % args.iter_tag)
    out_del = args.out_delta or os.path.join(
        _HERE, "ilc_delta_right_%s.npz" % args.iter_tag)
    np.savez(out_tab, gff=comp.astype(np.float64),
             meta_json=json.dumps(meta, ensure_ascii=False))
    np.savez(out_del, delta=full.astype(np.float64),
             meta_json=json.dumps(meta, ensure_ascii=False))
    rep = dict(meta,
               delta_peak=np.abs(full).max(axis=0).tolist(),
               tau_learn_peak=np.abs(tau_learn).max(axis=0).tolist(),
               composed_peak=float(np.abs(comp).max()),
               e_cmd_mean_deg=float(np.abs(err_deg).mean()),
               e_cmd_max_deg=float(np.abs(err_deg).max()))
    out_rep = os.path.join(_HERE, "ilc_learn_report.json")
    with open(out_rep, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print("\n  已写出: %s\n          %s\n          %s"
          % (os.path.basename(out_tab), os.path.basename(out_del),
             os.path.basename(out_rep)))

    # ---- 曲线 ----
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("(无 matplotlib, 跳过曲线)")
            return
        tt = grid / TABLE_HZ
        fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
        ax = axes[0]
        for j in (0, 1, 3):
            ax.plot(tt, err_deg[:, j], alpha=0.35, label="e_cmd J%d" % (j + 1))
        ax.set_ylabel("cmd−q [°]")
        ax.set_title("0921 full run: drive-loop error e_cmd (input)")
        ax.legend(ncol=3)
        ax.grid(alpha=0.3)
        ax = axes[1]
        for j in (0, 1, 3):
            ax.plot(tt, tau_learn[:, j], label="τ̂_fb J%d" % (j + 1))
            ax.plot(tt, tau_fb[:, j], alpha=0.15, color=ax.lines[-1].get_color())
        ax.set_ylabel("deficit recon [Nm]")
        ax.legend(ncol=3)
        ax.grid(alpha=0.3)
        ax = axes[2]
        for j in (0, 1, 3):
            ax.plot(tt, full[:len(tt), j], label="Δτ J%d" % (j + 1))
        for j in range(7):
            if j in (0, 1, 3):
                continue
            ax.plot(tt, full[:len(tt), j], alpha=0.4, lw=0.8)
        ax.axvline((n_end) / TABLE_HZ, ls="--", color="k", alpha=0.5)
        ax.text((n_end) / TABLE_HZ, ax.get_ylim()[1], " covered end",
                va="top", fontsize=8)
        ax.set_ylabel("Δτ [Nm]")
        ax.set_xlabel("t [s]")
        ax.legend(ncol=3)
        ax.grid(alpha=0.3)
        fig.suptitle("ILC iter-1 (gamma=%.2f, fc=%.2fHz) - see "
                     "ilc_learn_report.json" % (args.gamma, args.fc))
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out_png = os.path.join(_fig_dir(), "ilc_learn_curves.png")
        fig.savefig(out_png, dpi=130)
        print("          " + os.path.basename(out_png))


if __name__ == "__main__":
    main()
