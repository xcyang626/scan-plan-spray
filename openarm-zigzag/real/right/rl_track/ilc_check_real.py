#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ilc_check_real.py —— 故障注入孪生: 端到端验证 ILC 学习律(离线, 不接触机器人)
================================================================================
问题: ilc_learn.py 的"欠账重构律"在闭环里到底压不压得住猎振?
方法(ILC 文档 Phase-1 的孪生等价, 而非用真实表对孪生故障的循环论证):
  1) 给 MuJoCo 孪生注入未建模负载 τ_load = KX·(q−home) + FJ·tanh(q̇/v0)
     (位形相关线缆拖拽 + 换向摩擦; KX 按实测 τ̂ 量级标定, J1≈8 Nm/rad 等);
  2) 基线跑(基线前馈表, 无 Δτ)→ 应复现"误差随悬伸发散 + 积分器蓄能";
  3) 用与 ilc_learn.py 同一套重构律从基线跑学出 Δτ_twin;
  4) 带表跑(同故障)→ 量化误差/积分器蓄能下降。
回放循环与 verify_deploy.replay_mujoco 同一套常量与驱动律(含积分器 EMA/
钳位/清零), 保证结论对真机栈的代表性与离线审计一致。

用法(mujoco conda 环境):
    python ilc_check_real.py [--seconds 143.6] [--kx-scale 1.0]
输出: ilc_check_real_report.json
⚠ 纯离线; 上机执行一律由操作员按 rl_track/README.md 规程进行。
"""

import argparse
import json
import os
import sys
import time

import numpy as np

try:
    from scipy.signal import butter, filtfilt

    def lpf0(x, fc, fs):
        b, a = butter(2, fc / (0.5 * fs), btype="low")
        return filtfilt(b, a, x, axis=0)

    def bandpass(x, lo, hi, fs):
        b, a = butter(2, [lo / (fs / 2), hi / (fs / 2)], btype="band")
        return filtfilt(b, a, x, axis=0)
except ImportError:
    # mujoco 环境无 scipy: FFT 零相位滤波(数值效果对慢变信号等价)
    def _fft_mask(n, f_lo, f_hi, fs):
        f = np.fft.rfftfreq(n, 1.0 / fs)
        m = np.ones_like(f)
        m[f < f_lo] = 1.0 / (1.0 + (f_lo / np.maximum(f[f < f_lo], 1e-9))**4)
        if f_hi is not None:
            m[f > f_hi] = 1.0 / (1.0 + (f[f > f_hi] / f_hi)**4)
        return m[:, None]

    def lpf0(x, fc, fs):
        X = np.fft.rfft(x, axis=0)
        return np.fft.irfft(X * _fft_mask(len(x), 0.0, fc, fs), len(x), axis=0)

    def bandpass(x, lo, hi, fs):
        X = np.fft.rfft(x, axis=0)
        return np.fft.irfft(X * _fft_mask(len(x), lo, hi, fs), len(x), axis=0)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from rl_policy import Policy                                    # noqa: E402
from verify_deploy import TRAIN_KD, TRAIN_KP, TRAIN_TAU, _MJCF  # noqa: E402
# 驱动常量(与 verify_deploy.replay_mujoco / zero_offset_hw 同一口径)
ARMATURE = np.array([0.0122, 0.0122, 0.032, 0.032, 0.005, 0.005, 0.005])
KP = np.array(TRAIN_KP)
KD = np.array(TRAIN_KD)
KI = np.array([12.0, 12.0, 12.0, 10.0, 8.0, 8.0, 8.0])
IC = np.array([10.0, 10.0, 6.0, 8.0, 3.0, 3.0, 3.0])
TU = np.array(TRAIN_TAU)
# 学习律常量与 ilc_learn.py 保持同源(ilc_learn 延迟导入 scipy, 本环境无 scipy
# 时滤波用上面的 FFT 回退, 数值等价)
from ilc_learn import EMA_A, GAMMA, TAU_CLAMP                   # noqa: E402

import mujoco                                                   # noqa: E402

# 注入负载模型: τ_load = KX·(q−home) + FJ·tanh(q̇/V0)
# KX 标定依据: 0921 实测 τ̂ 峰值 [10.2, 3.6, 6.8, 3.0, 1.3, 1.9, 1.4] Nm
# 与各关节行程 (J1~±1, J2~3.3, J3~2, J4~2 rad) 之比。
KX = np.array([8.0, 1.2, 3.5, 1.5, 0.3, 0.3, 0.3])
FJ = np.array([0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1])
V0 = 0.05


def run_replay(pol, gff, delta=None, seconds=None, fault=True,
               speed=1.0):
    """与 verify_deploy.replay_mujoco 同律的回放; 返回逐帧记录。
    speed: 参考时间轴倍率(0.5~1.5), qd_scale/vref 同步缩放(0923)。"""
    model = mujoco.MjModel.from_xml_path(_MJCF)
    data = mujoco.MjData(model)
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                              "openarm_right_joint%d" % i)
            for i in range(1, 8)]
    qadr = np.array([model.jnt_qposadr[j] for j in jids])
    vadr = np.array([model.jnt_dofadr[j] for j in jids])
    model.dof_armature[vadr] = ARMATURE
    data.qpos[qadr] = pol.home
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    PHYS_DT, MIT_HZ = 0.002, 100
    n_sub = int(round(pol.dt / PHYS_DT))
    pol.reset_stream()
    integ, err_f = np.zeros(7), np.zeros(7)
    prev_a = np.zeros(7)
    n_max = pol.n if seconds is None else min(pol.n, int(seconds / pol.dt))
    rec = dict(idx=[], q=[], cmd=[], qref=[], tau=[], integ=[])
    pos = 0.0
    k = 0
    while True:
        done = pos >= pol.n - 1 or k >= n_max
        if done:
            break
        idx = int(min(pos, pol.n - 1))
        q = data.qpos[qadr].copy()
        qd = data.qvel[vadr].copy()
        a = pol.action(q, qd, prev_a, idx, qd_scale=speed)
        cmd = pol.command(q, qd, prev_a, idx, qd_scale=speed)
        prev_a = a
        tau_load = (KX * (q - pol.home) + FJ * np.tanh(qd / V0)
                    if fault else np.zeros(7))
        for s in range(n_sub):
            if s % max(n_sub // int(round(1.0 / (MIT_HZ * PHYS_DT))),
                       1) == 0:
                e = cmd - data.qpos[qadr]
                err_f = 0.3 * e + 0.7 * err_f
                big = np.abs(err_f) > 0.5
                err_f[big] = 0.0
                integ[big] = 0.0
                integ = np.clip(integ + KI * err_f * 0.01, -IC, IC)
                drive = (KP * e - KD * (data.qvel[vadr]
                                        - pol.qd_ref[idx] * speed)
                         + integ + (gff[idx] if gff is not None else 0.0)
                         + (delta[idx] if delta is not None else 0.0))
                data.ctrl[:7] = np.clip(drive - tau_load, -TU, TU)
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        rec["idx"].append(idx)
        rec["q"].append(data.qpos[qadr].copy())
        rec["cmd"].append(cmd.copy())
        rec["qref"].append(pol.q_ref[idx])
        rec["tau"].append(data.ctrl[:7].copy())
        rec["integ"].append(integ.copy())
        pos += speed
        k += 1
    return {k: np.array(v) for k, v in rec.items()}


def learn_from_rec(rec, gamma=GAMMA, fc=0.08):
    """与 ilc_learn.py 同一套重构律: τ̂ = LPF{kp·e+kd·ė+integ} → Δτ。"""
    q, cmd = rec["q"], rec["cmd"]
    e = cmd - q
    fs = 1.0 / pol_dt()
    edot = lpf0(np.gradient(e, 1.0 / fs, axis=0), 1.0, fs)
    tau_fb = KP * e + KD * edot + rec["integ"]
    delta = np.clip(gamma * lpf0(tau_fb, fc, fs), -TAU_CLAMP, TAU_CLAMP)
    # 学习窗起点渐入 2s(与 ilc_learn 同): 掩蔽前 1.5s + 渐入
    n = len(e)
    ramp = np.clip((np.arange(n) / fs - 1.5) / 2.0, 0.0, 1.0)
    return delta * ramp[:, None], tau_fb


def pol_dt():
    return 0.02


def band_p2p(x, lo, hi, fs=50.0):
    xb = bandpass(x, lo, hi, fs)
    return xb.max(axis=0) - xb.min(axis=0)


def summarize(name, rec, delta_used):
    err = np.degrees(rec["qref"] - rec["q"])
    integ_pk = np.abs(rec["integ"]).max(axis=0)
    print("== %s ==" % name)
    print("  |qref-q| mean %.2f°  p95 %.2f°  max %.2f°"
          % (np.abs(err).mean(), np.percentile(np.abs(err), 95),
             np.abs(err).max()))
    print("  末 1/3 误差 mean %.2f°  (悬伸段发散检查)"
          % np.abs(err[-len(err) // 3:]).mean())
    print("  0.1~0.5Hz 峰峰: %s °" % np.round(
        band_p2p(np.radians(err), 0.1, 0.5), 2).tolist())
    print("  积分器蓄能峰值: %s Nm (钳位 %s)"
          % (np.round(integ_pk, 1).tolist(),
             IC.astype(int).tolist()))
    if delta_used is not None:
        print("  Δτ 峰值: %s Nm" % np.round(
            np.abs(delta_used).max(axis=0), 2).tolist())
    return dict(err_mean=float(np.abs(err).mean()),
                err_p95=float(np.percentile(np.abs(err), 95)),
                err_max=float(np.abs(err).max()),
                err_tail_mean=float(np.abs(err[-len(err) // 3:]).mean()),
                band_p2p_deg=band_p2p(np.radians(err), 0.1, 0.5).tolist(),
                integ_peak=integ_pk.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=None,
                    help="只跑前 N 秒(默认全程)")
    ap.add_argument("--kx-scale", type=float, default=1.0)
    ap.add_argument("--out", default=os.path.join(
        _HERE, "ilc_check_real_report.json"))
    args = ap.parse_args()
    global KX
    KX = KX * args.kx_scale

    pol = Policy()
    gff = np.load(os.path.join(_HERE, "grav_ff_right.npz"))["gff"][:pol.n]
    print("孪生故障注入: KX=%s Nm/rad" % KX.tolist())

    print("\n[1/3] 基线跑(无 Δτ, 带故障)...")
    t0 = time.time()
    rec_b = run_replay(pol, gff, None, args.seconds, fault=True)
    print("  (%.0fs)" % (time.time() - t0))
    m_b = summarize("基线", rec_b, None)

    print("\n[2/3] 从基线跑学习 Δτ_twin(同一套重构律)...")
    delta_t, tau_fb = learn_from_rec(rec_b)
    print("  τ̂ 峰值: %s Nm"
          % np.round(np.abs(tau_fb).max(axis=0), 2).tolist())
    print("  Δτ_twin 峰值: %s Nm" % np.round(
        np.abs(delta_t).max(axis=0), 2).tolist())

    print("\n[3/3] 带表跑(同故障)...")
    t0 = time.time()
    rec_i = run_replay(pol, gff, delta_t, args.seconds, fault=True)
    print("  (%.0fs)" % (time.time() - t0))
    m_i = summarize("ILC", rec_i, delta_t)

    print("\n== 结论 ==")
    print("  误差 mean: %.2f° → %.2f° (%.0f%% 下降)"
          % (m_b["err_mean"], m_i["err_mean"],
             100 * (1 - m_i["err_mean"] / m_b["err_mean"])))
    print("  末段误差:  %.2f° → %.2f°"
          % (m_b["err_tail_mean"], m_i["err_tail_mean"]))
    print("  积分器蓄能(J1/J2/J3): %s → %s Nm"
          % (np.round(m_b["integ_peak"][:3], 1).tolist(),
             np.round(m_i["integ_peak"][:3], 1).tolist()))
    ok = m_i["err_mean"] < m_b["err_mean"] and \
        np.max(m_i["integ_peak"]) <= np.max(m_b["integ_peak"]) + 1e-6
    print("  判定: %s" % ("学习有效(误差与积分器蓄能双降)" if ok
                          else "未达预期 —— 检查重构律/注入标定"))
    out = dict(kx=KX.tolist(), fj=FJ.tolist(), gamma=GAMMA,
               baseline=m_b, ilc=m_i,
               delta_twin_peak=np.abs(delta_t).max(axis=0).tolist(),
               effective=bool(ok))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("已写出 %s" % args.out)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    t = np.arange(len(rec_b["q"])) * 0.02
    eb = np.degrees(rec_b["qref"] - rec_b["q"])
    ei = np.degrees(rec_i["qref"] - rec_i["q"])
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax = axes[0]
    for j in (0, 1, 2):
        ax.plot(t, eb[:, j], alpha=0.45, label="baseline J%d" % (j + 1))
        ax.plot(t, ei[:, j], label="ILC J%d" % (j + 1))
    ax.set_ylabel("qref-q [deg]")
    ax.set_title("twin + injected cable-drag fault: baseline vs ILC "
                 "(kx-scale %.2f)" % args.kx_scale)
    ax.legend(ncol=3)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(t, np.abs(rec_b["integ"]).max(axis=1), alpha=0.6,
            label="baseline max|integ|")
    ax.plot(t, np.abs(rec_i["integ"]).max(axis=1), label="ILC max|integ|")
    ax.axhline(IC.min(), ls=":", color="k", label="min clamp 3Nm")
    ax.set_ylabel("integrator energy [Nm]")
    ax.set_xlabel("t [s]")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png = args.out.replace(".json", ".png")
    fig.savefig(png, dpi=130)
    print("已写出 %s" % png)


if __name__ == "__main__":
    main()
