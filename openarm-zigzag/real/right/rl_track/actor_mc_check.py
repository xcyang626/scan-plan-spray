#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""actor_mc_check.py —— 部署 actor 的全程蒙特卡洛稳健性检查(上机前)
======================================================================
用 verify_deploy 同款回放路径(rl_policy 部署代码 + MIT 250Hz + FF v3 +
vref), 叠加 pd_screen 同款不确定性场景(摩擦/负载/延迟/增益容差),
对**完整 143.6s 轨迹**做 N 次并行回放 —— 回答"真机全程最坏会到多少"。

用法(mujoco conda 环境, rl_track 目录):
    python actor_mc_check.py --scenarios 12
产出: actor_mc_report.json + 控制台判定
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import mujoco
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
XML = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "仿真",
                                   "mujoco_sim", "openarm_right_rl.xml"))
FF = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "仿真",
                                  "isaaclab_sim", "rl",
                                  "grav_ff_right.npz"))
POLICY_HZ = 50
ARMATURE = np.array([0.0122, 0.0122, 0.032, 0.032, 0.005, 0.005, 0.005])
RATED = np.array([20., 20., 27., 27., 7., 7., 7.])
PAYLOAD_BODIES = ("openarm_right_link6", "openarm_right_link7",
                  "finger_openarm_right_finger_joint1",
                  "finger_openarm_right_finger_joint2",
                  "openarm_right_hand_tcp")
_G = {}


def gen_scenarios(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        out.append(dict(
            idx=i,
            fric_visc=rng.uniform(0.05, 0.6, 7),
            fric_coul=rng.uniform(0.0, 1.0, 7),
            payload=float(rng.uniform(0.8, 1.2)),
            delay_ms=float(rng.choice((0.0, 20.0))),
            kp_tol=1.0 + rng.uniform(-0.05, 0.05, 7),
            kd_tol=1.0 + rng.uniform(-0.05, 0.05, 7)))
    return out


def run_one(task):
    """一次 (场景) 的全轨迹回放。worker 内独立加载 Policy(线程态隔离)。"""
    scen, actor_path = task
    sys.path.insert(0, _HERE)
    from rl_policy import Policy
    import mujoco as mj
    pol = Policy(actor_path=os.path.join(_HERE, "actor_right.npz"),
                 ref_path=os.path.join(_HERE, "ref_right.npz"))
    if actor_path != "deployed":
        pol.params = dict(np.load(actor_path))
    q_ref, qd_ref = pol.q_ref, pol.qd_ref

    model = mj.MjModel.from_xml_path(XML)
    data = mj.MjData(model)
    jids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT,
                          "openarm_right_joint%d" % i) for i in range(1, 8)]
    qadr = np.array([model.jnt_qposadr[j] for j in jids])
    vadr = np.array([model.jnt_dofadr[j] for j in jids])
    model.dof_armature[vadr] = ARMATURE
    model.dof_damping[vadr] = scen["fric_visc"]
    model.dof_frictionloss[vadr] = scen["fric_coul"]
    for b in PAYLOAD_BODIES:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, b)
        if bid >= 0:
            model.body_mass[bid] *= scen["payload"]
            model.body_inertia[bid] *= scen["payload"]

    gff = np.load(FF)["gff"]
    # MIT 增益 = 部署 URDF 值(= verify_deploy TRAIN_KP/KD, 训练常量同源)
    kp = np.array([120., 100., 70., 126.3, 30., 30., 30.]) * scen["kp_tol"]
    kd = np.array([7.4, 12.2, 7.5, 8.04, 1.5, 1.5, 1.5]) * scen["kd_tol"]
    ki = np.array([12., 12., 12., 10., 8., 8., 8.])
    ic = np.array([10., 10., 6., 8., 3., 3., 3.])
    tau_lim = model.actuator_forcerange[:7, 1].copy()
    delay = int(round(scen["delay_ms"] / 1000.0 * POLICY_HZ))

    data.qpos[qadr] = pol.home
    data.qpos[[model.jnt_qposadr[mj.mj_name2id(
        model, mj.mjtObj.mjOBJ_JOINT, "openarm_right_finger_joint%d" % f)]
        for f in (1, 2)]] = 0.02
    data.qvel[:] = 0
    mj.mj_forward(model, data)

    PHYS_DT = 0.002
    n_sub = int(round(pol.dt / PHYS_DT))
    n_mit = max(int(round(1.0 / (250 * PHYS_DT))), 1)
    pol.reset_stream()
    integ, err_f, prev_a = np.zeros(7), np.zeros(7), np.zeros(7)
    buf_q, buf_v, buf_f = [q_ref[0].copy()], [qd_ref[0].copy()], \
        [gff[0].copy()]
    E = np.empty((pol.n, 7)); TAU = np.empty((pol.n, 7))
    for k in range(pol.n):
        a = pol.action(data.qpos[qadr].copy(), data.qvel[vadr].copy(),
                       prev_a, k)
        cmd = pol.command(data.qpos[qadr].copy(), data.qvel[vadr].copy(),
                          prev_a, k)
        prev_a = a
        for s in range(n_sub):
            if s % n_mit == 0:
                if s == 0:
                    buf_q.append(cmd)
                    buf_v.append(qd_ref[min(k, len(qd_ref) - 1)])
                    buf_f.append(gff[min(k, len(gff) - 1)])
                    for b in (buf_q, buf_v, buf_f):
                        if len(b) > delay + 1:
                            b.pop(0)
                dl = delay if len(buf_q) > delay else 0
                q_c, v_c, f_c = buf_q[-1 - dl], buf_v[-1 - dl], \
                    buf_f[-1 - dl]
                err = q_c - data.qpos[qadr]
                err_f = 0.3 * err + 0.7 * err_f
                big = np.abs(err_f) > 0.5
                err_f[big] = 0; integ[big] = 0
                integ = np.clip(integ + ki * err_f * 0.01, -ic, ic)
                tau = kp * err - kd * (data.qvel[vadr] - v_c) + integ + f_c
                data.ctrl[:7] = np.clip(tau, -tau_lim, tau_lim)
            data.ctrl[7:9] = 0.02
            mj.mj_step(model, data)
        E[k] = np.degrees(data.qpos[qadr] - q_ref[min(k, len(q_ref) - 1)])
        TAU[k] = tau
    j = np.unravel_index(np.abs(E).argmax(), E.shape)
    meta = pol.meta
    tmax = j[0] * pol.dt
    ph = ("转移出" if tmax < meta["t_phase_out"] else
          ("走线" if tmax < meta["t_phase_weave_end"] else "转移回"))
    return dict(scen=scen["idx"], delay_ms=scen["delay_ms"],
                payload=round(scen["payload"], 2),
                coul=round(float(scen["fric_coul"].mean()), 2),
                mean=round(float(np.abs(E).mean()), 3),
                p95=round(float(np.percentile(np.abs(E), 95)), 3),
                max=round(float(np.abs(E).max()), 3),
                worst_joint=int(j[1] + 1), worst_t=round(float(tmax), 1),
                phase=ph,
                tau_over=round(float((np.abs(TAU) / RATED).max()), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gate-max", type=float, default=4.5,
                    help="真机口径 max|e| 门限(°)")
    args = ap.parse_args()
    scens = gen_scenarios(args.scenarios, args.seed)
    actor = os.path.join(_HERE, "actor_right.npz")
    tasks = [(sc, "deployed") for sc in scens]
    t0 = time.time()
    ctx = mp.get_context("fork")
    print("场景 %d 个 × 全程 143.6s | workers=%d" % (len(tasks), args.
                                                    scenarios and mp.
                                                    cpu_count() // 2))
    with ctx.Pool(min(len(tasks), 10)) as pool:
        res = list(pool.imap_unordered(run_one, tasks))
    res.sort(key=lambda r: -r["max"])
    print("\n== 全程蒙特卡洛(部署 actor, 场景间同种子) ==")
    print("场景  延迟ms 负载  库仑 | mean  p95   max  最差关节/时刻/相位 | τ/额定")
    for r in res:
        print("%3d   %4.0f  %.2f  %.2f | %5.2f %5.2f %5.2f  J%d @%3ds %s"
              " | %.2f" % (r["scen"], r["delay_ms"], r["payload"], r["coul"],
                           r["mean"], r["p95"], r["max"], r["worst_joint"],
                           r["worst_t"], r["phase"], r["tau_over"]))
    worst = res[0]
    np_mean = float(np.mean([r["mean"] for r in res]))
    np_max = float(np.max([r["max"] for r in res]))
    gate = worst["max"] <= args.gate_max and worst["tau_over"] <= 1.3
    print("\n最坏场景: mean %.2f° max %.2f° τ%.2f×额定 | 12场景均值 mean %.2f°"
          % (worst["mean"], worst["max"], worst["tau_over"], np_mean))
    print("判定(门限 max ≤ %.1f° 且 τ ≤ 1.3×额定): %s" % (
        args.gate_max, "PASS — 可上真机全程" if gate else "FAIL"))
    with open(os.path.join(_HERE, "actor_mc_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"),
                       scenarios=res, worst=worst,
                       gate_pass=bool(gate)), f, ensure_ascii=False, indent=1)
    print("已写出 actor_mc_report.json | 用时 %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
