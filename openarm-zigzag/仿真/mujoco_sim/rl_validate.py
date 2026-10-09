#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rl_validate.py —— MuJoCo 验证: numpy actor + MIT 控制律跑完整任务链
======================================================================
与 IsaacLab/真机的一致性设计:
  - 执行器: openarm_right_rl.xml 电机执行器(力矩上限=ESC 实测), 关节阻尼 0,
    MIT 律 tau = kp*e_p + kd*(0-vd) + I 在 Python 侧 100Hz 复刻
    (EMA α=0.3, |err|>0.5rad 清零, I 钳位 ±[10,10,6,8,3,3,3]);
  - 策略 50Hz(actor.npz 纯 numpy 前向, 确定性均值动作);
  - 参考 ref_right.npz(与训练同一份)。
产出: rl_validate_report.json + rl_validate.png

⚠ 评估口径纪律(2026-09-17 教训): 本脚本是**简化复刻**(不含部署侧的
残差低通/启动渐入/动作延迟建模), 同一策略在此可能比部署口径差数倍 ——
**下结论/选 checkpoint 一律以 real/right/rl_track/verify_deploy.py
(部署代码路径 + 完整链路) 为准**; 本脚本用于快速迭代与回归对比。

用法(mujoco conda 环境):
    python rl_validate.py --actor ../isaaclab_sim/rl/logs/.../actor_final.npz
未训练时可用 --random 冒烟。
"""

import argparse
import json
import math
import os

import mujoco
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(_HERE, "openarm_right_rl.xml")
REF = os.path.abspath(os.path.join(_HERE, "..", "isaaclab_sim", "rl",
                                   "ref_right.npz"))

KP = np.array([120.0, 70.0, 70.0, 90.0, 30.0, 30.0, 30.0])
KD = np.array([3.6, 3.0, 2.0, 2.2, 1.5, 1.5, 1.2])
KI = np.array([12.0, 12.0, 12.0, 10.0, 8.0, 8.0, 8.0])
ICLAMP = np.array([10.0, 10.0, 6.0, 8.0, 3.0, 3.0, 3.0])
TAU_MAX = np.array([54.0, 54.0, 28.0, 28.0, 10.0, 10.0, 10.0])
SOFT_LO = np.array([-0.3490, -0.1336, -1.4069, 0.0505, -1.5189, -0.2543,
                    -1.5049])
SOFT_HI = np.array([4.4420, 3.2685, 1.6237, 2.4200, 1.5265, 0.9463,
                    1.5661])
# 折算转子惯量 [kg·m²]: 与 IsaacLab 训练一致(否则腕部发散, 见 zigzag_env.ARMATURE)
ARMATURE = 0.02
Q_MID = (SOFT_LO + SOFT_HI) / 2
Q_HALF = (SOFT_HI - SOFT_LO) / 2
CLEARANCE = math.radians(3.0)
ACTION_SCALE = 0.25
POLICY_HZ = 50
PHYS_DT = 0.002            # 与 xml timestep 一致
MIT_HZ = 100


def mlp_forward(obs, params):
    """ELU MLP 前向(与 rsl_rl MLPModel/导出 npz 一致)。
    层键为 W0/W2/W4/W6(下标跳过激活层), 按数值序遍历而非连续 range。"""
    h = obs
    w_keys = sorted([k for k in params if k.startswith("W")],
                    key=lambda s: int(s[1:]))
    for i, wk in enumerate(w_keys):
        h = h @ params[wk].T + params["b" + wk[1:]]
        if i < len(w_keys) - 1:
            h = np.where(h > 0, h, np.expm1(h))    # ELU
    return h


def build_obs(q, qd, prev_a, q_ref_t, q_ref_previews, qd_ref_t, t_frac,
              phase):
    return np.concatenate([
        (q - Q_MID) / Q_HALF,
        qd / 10.0,
        np.tanh(prev_a),
        (q_ref_t - Q_MID) / Q_HALF,
    ] + [(p - Q_MID) / Q_HALF for p in q_ref_previews] + [
        qd_ref_t / 10.0,
        [t_frac],
        phase,
    ])[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", default=None, help="actor npz 路径")
    ap.add_argument("--random", action="store_true", help="随机动作冒烟")
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--grav-ff", nargs="?", const=1.0, default=0.0, type=float,
                    help="重力前馈比例 0~1(默认关; 表=make_grav_ff.py 产物)")
    ap.add_argument("--grav-ff-file", default=os.path.abspath(
        os.path.join(_HERE, "..", "isaaclab_sim", "rl",
                     "grav_ff_right.npz")))
    ap.add_argument("--no-vref", action="store_true",
                    help="退回 v_des=0 旧律(默认 v_des=参考速度, 与部署一致)")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                              "openarm_right_joint%d" % i)
            for i in range(1, 8)]
    fqids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                               "openarm_right_finger_joint%d" % i)
             for i in (1, 2)]
    qadr = np.array([model.jnt_qposadr[j] for j in jids])
    vadr = np.array([model.jnt_dofadr[j] for j in jids])
    fadr = [model.jnt_qposadr[j] for j in fqids]
    # 折算转子惯量(与 IsaacLab 训练一致)
    model.dof_armature[vadr] = ARMATURE
    tcp_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")

    d = np.load(args.ref, allow_pickle=True)
    q_ref = d["q_ref"]
    qd_ref = d["qd_ref"]
    meta = d["meta_json"].item()
    if isinstance(meta, (bytes, str)):
        meta = json.loads(meta)
    T = float(meta["T_total"])
    t_out = float(meta["t_phase_out"])
    t_back = float(meta["t_phase_weave_end"])
    dt_pol = 1.0 / POLICY_HZ
    n_sub_pol = int(round(dt_pol / PHYS_DT))         # 10
    n_sub_mit = max(n_sub_pol // int(round(1.0 / (MIT_HZ * PHYS_DT))), 1)

    params = None
    gff = None
    if args.grav_ff > 0:
        gff = np.load(args.grav_ff_file)["gff"]
        assert len(gff) >= len(q_ref), \
            "grav_ff 行数(%d) < 参考帧数(%d)" % (len(gff), len(q_ref))
        print("重力前馈: %s ×%.2f" % (os.path.basename(args.grav_ff_file),
                                      args.grav_ff))
    rng = np.random.default_rng(args.seed)
    if args.actor:
        params = dict(np.load(args.actor))
        print("actor: %d 层" % len([k for k in params
                                    if str(k).startswith("W")]))
    preview = (2, 4, 8, 16)

    # 初始状态 = home(t=0 参考)
    data.qpos[qadr] = q_ref[0]
    data.qpos[fadr] = 0.02
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    integ = np.zeros(7)
    err_f = np.zeros(7)
    prev_a = np.zeros(7)
    steps = int(T / dt_pol)
    rec_t, rec_q, rec_qd, rec_qref, rec_tau, rec_tcp = [], [], [], [], [], []
    obs_t = 0.0
    for k in range(steps):
        t = k * dt_pol
        idx = min(int(round(t / dt_pol)), len(q_ref) - 1)
        q = data.qpos[qadr].copy()
        qd = data.qvel[vadr].copy()
        q_ref_t = q_ref[idx]
        previews = [q_ref[min(idx + p, len(q_ref) - 1)] for p in preview]
        phase = [1.0, 0.0, 0.0]
        if t >= t_out:
            phase = [0.0, 1.0, 0.0]
        if t >= t_back:
            phase = [0.0, 0.0, 1.0]
        obs = build_obs(q, qd, prev_a, q_ref_t, previews, qd_ref[idx],
                        idx / len(q_ref), phase)
        if params is not None:
            a = np.tanh(mlp_forward(obs, params)[0])
        elif args.random:
            a = rng.uniform(-1, 1, 7)
        else:
            a = np.zeros(7)
        prev_a = np.clip(a, -3, 3)
        q_cmd = np.clip(q_ref_t + ACTION_SCALE * prev_a,
                        SOFT_LO + CLEARANCE, SOFT_HI - CLEARANCE)

        for s in range(n_sub_pol):
            # MIT 律 100Hz(每 n_sub_mit 物理步)
            if s % n_sub_mit == 0:
                qq = data.qpos[qadr]
                vq = data.qvel[vadr]
                err = q_cmd - qq
                err_f = 0.3 * err + 0.7 * err_f
                big = np.abs(err_f) > 0.5
                err_f[big] = 0.0
                integ[big] = 0.0
                integ = np.clip(integ + KI * err_f * 0.01, -ICLAMP, ICLAMP)
                vd = 0.0 if args.no_vref else qd_ref[idx]
                tau = np.clip(KP * err - KD * (vq - vd) + integ
                              + (gff[idx] * args.grav_ff
                                 if gff is not None else 0.0),
                              -TAU_MAX, TAU_MAX)
                data.ctrl[:7] = tau
            data.ctrl[7:9] = 0.02          # 夹爪保持
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

        rec_t.append(t)
        rec_q.append(data.qpos[qadr].copy())
        rec_qd.append(data.qvel[vadr].copy())
        rec_qref.append(q_ref_t)
        rec_tau.append(data.ctrl[:7].copy())
        rec_tcp.append(data.site_xpos[tcp_sid].copy())
        if k % (POLICY_HZ * 20) == 0:
            e = np.abs(rec_q[-1] - rec_qref[-1]).max() * 57.3
            print("  t=%5.1fs |Δq|∞=%.2f°" % (t, e), flush=True)

    Q = np.array(rec_q)
    Qref = np.array(rec_qref)
    QD = np.array(rec_qd)
    TAU = np.array(rec_tau)
    TCP = np.array(rec_tcp)
    tt = np.array(rec_t)
    E = Q - Qref

    # 指标
    ph = np.zeros(len(tt), dtype=int)
    ph[tt >= t_out] = 1
    ph[tt >= t_back] = 2
    # 用 MuJoCo FK 重算参考 TCP(参考关节→TCP)
    tcp_ref_all = np.empty((len(q_ref), 3))
    tmp = mujoco.MjData(model)
    for i, qr in enumerate(q_ref):
        tmp.qpos[qadr] = qr
        tmp.qpos[fadr] = 0.02
        mujoco.mj_forward(model, tmp)
        tcp_ref_all[i] = tmp.site_xpos[tcp_sid]
    # 采样时刻(idx 与仿真步一一对应)
    idxs = np.minimum(np.arange(len(TCP)), len(q_ref) - 1)
    tcp_ref_s = tcp_ref_all[idxs]

    def hf_sigma(x):
        w = int(0.5 / dt_pol)
        ma = np.convolve(x, np.ones(w) / w, mode="same")
        return float((x - ma).std())

    report = dict(
        total_s=float(T),
        grav_ff=float(args.grav_ff),
        joint_err_deg={"mean": float(np.degrees(np.abs(E)).mean()),
                       "p95": float(np.percentile(np.degrees(np.abs(E)),
                                                  95)),
                       "max": float(np.degrees(np.abs(E)).max())},
        per_phase={},
        hf_sigma={"J1": hf_sigma(QD[:, 0]), "J2": hf_sigma(QD[:, 1])},
        tau_max_per_joint=[float(x) for x in np.abs(TAU).max(axis=0)],
        baseline_hf={"J1": 0.043, "J2": 0.053},
    )
    for p, name in enumerate(("transfer_out", "weave", "transfer_back")):
        m = ph == p
        report["per_phase"][name] = dict(
            dur_s=float(tt[m][-1] - tt[m][0]) if m.any() else 0.0,
            joint_err_deg_max=float(np.degrees(np.abs(E[m]).max())),
            tcp_err_mm_max=float(
                np.linalg.norm(TCP[m] - tcp_ref_s[m], axis=1).max() * 1000),
            tcp_err_mm_mean=float(
                np.linalg.norm(TCP[m] - tcp_ref_s[m], axis=1).mean() * 1000),
        )
    print("== MuJoCo 验证 ==")
    print("关节误差: mean %.2f° p95 %.2f° max %.2f°" % (
        report["joint_err_deg"]["mean"], report["joint_err_deg"]["p95"],
        report["joint_err_deg"]["max"]))
    for k, v in report["per_phase"].items():
        print("%-14s %.1fs | 关节max %.2f° | TCP mean %.2fmm max %.2fmm" % (
            k, v["dur_s"], v["joint_err_deg_max"], v["tcp_err_mm_mean"],
            v["tcp_err_mm_max"]))
    print("速度 HF σ: J1 %.4f J2 %.4f (JTC 基线 0.043/0.053)"
          % (report["hf_sigma"]["J1"], report["hf_sigma"]["J2"]))
    out_json = os.path.join(_HERE, "rl_validate_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("已写出", out_json)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        pc = ["tab:purple", "tab:blue", "tab:green"]
        for a_ in ax:
            for p in range(3):
                m = ph == p
                if m.any():
                    a_.axvspan(tt[m][0], tt[m][-1], alpha=0.08, color=pc[p])
        for j in range(7):
            ax[0].plot(tt, np.degrees(Q[:, j]), lw=0.7)
            ax[0].plot(tt, np.degrees(Qref[:, j]), lw=0.7, ls="--", alpha=0.4)
        ax[0].set_ylabel("q [deg] 实=MuJoCo 虚=参考")
        for j in range(7):
            ax[1].plot(tt, np.degrees(E[:, j]), lw=0.6)
        ax[1].set_ylabel("跟踪误差 [deg]")
        speed = np.linalg.norm(np.diff(TCP, axis=0), axis=1) / dt_pol
        ax[2].plot(tt[1:], speed * 100, lw=0.5, color="tab:red")
        ax[2].set_ylabel("TCP 速率 [cm/s]")
        for j in range(7):
            ax[3].plot(tt, TAU[:, j], lw=0.5)
        ax[3].set_ylabel("tau [Nm]")
        ax[3].set_xlabel("t [s]")
        fig.tight_layout()
        fig.savefig(os.path.join(_HERE, "rl_validate.png"), dpi=130)
        print("已写出 rl_validate.png")
    except Exception as e:  # noqa: BLE001
        print("绘图失败:", e)


if __name__ == "__main__":
    main()
