#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pd_screen.py —— 训练前离线 PD 安全筛选(数字孪生全轨迹回放)
======================================================================
目的(方案 2026-09-16 认可: docs/PD参数整定流程.md 的前置环节):
  上真机/开训练**之前**, 对候选 (kp, kd) 组合做安全优先评估 ——
    1. 全轨迹纯 PD 回放(与 rl_validate.py 同一套 MIT 律复刻:
       100Hz tau = kp·e + kd·(0-v) + I, EMA 0.3, |err_f|>0.5rad 清零,
       I 钳位; 指令 50Hz = q_ref 采样, 等价真机 α=0 纯参考回放);
    2. 蒙特卡洛不确定性场景: 关节粘性/库仑摩擦(区间取训练 env 的
       fric_visc_range / fric_coul_range)、腕部负载 ±20%、指令延迟
       0/10/20ms、固件增益容差 ±5%(候选间用同一批场景种子, 公平对比);
    3. 硬约束一票否决(任一场景违反即 FAIL):
         max|e| ≤ --e-max-deg (默认 4°)
         max|τ_指令| ≤ 额定(20/20/27/27/7/7/7, v2 力矩预算)
         抬臂位形 ring-down ζ(J1-J4) ≥ --zeta-min (默认 0.45)
    4. 次级指标(mean|e| / HF σ / ∫τ² 发热代理 / J2 下垂)只在安全集内排序。
  理论依据: 场景法(Calafiore & Campi 2006) —— N 个随机场景全过,
  违反概率 ε ≲ 3/N(95% 置信, Chernoff 界)。

用法(mujoco conda 环境):
    python pd_screen.py                    # v2 基线 × 9 候选 × 12 场景
    python pd_screen.py --smoke            # 冒烟: 中心候选 × 2 场景 × 前 40s
    python pd_screen.py --grav-ff 1.0      # 叠加重力前馈对比(消下垂)
    python pd_screen.py --kp-scales 1.0 --kd-scales 0.75,1.0,1.3   # 只扫 kd
    python pd_screen.py --kp-csv 120,100,70,126.3,30,30,30 \
                        --kd-csv 7.4,12.2,7.5,8.04,1.5,1.5,1.5      # 单组显式候选
产出: pd_screen_report.json + 控制台排序表(含安全裕度)。
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import time

import mujoco
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(_HERE, "openarm_right_rl.xml")
REF = os.path.abspath(os.path.join(_HERE, "..", "isaaclab_sim", "rl",
                                   "ref_right.npz"))

# ---- 常量(与 zigzag_env.py / rl_validate.py 同源, 2026-09-15 v2 基线) ----
KP_NOM = np.array([120.0, 100.0, 70.0, 126.3, 30.0, 30.0, 30.0])
KD_NOM = np.array([7.4, 12.2, 7.5, 8.04, 1.5, 1.5, 1.5])
KI = np.array([12.0, 12.0, 12.0, 10.0, 8.0, 8.0, 8.0])
ICLAMP = np.array([10.0, 10.0, 6.0, 8.0, 3.0, 3.0, 3.0])
TAU_RATED = np.array([20.0, 20.0, 27.0, 27.0, 7.0, 7.0, 7.0])   # v2 预算
ARMATURE = np.array([0.0122, 0.0122, 0.032, 0.032, 0.005, 0.005, 0.005])
FRIC_VISC_RANGE = (0.05, 0.40)    # [N·m·s/rad] zigzag_env.fric_visc_range
FRIC_COUL_RANGE = (0.0, 0.60)     # [N·m]       zigzag_env.fric_coul_range
PAYLOAD_RANGE = (0.8, 1.2)        # 腕部远端质量缩放
DELAY_CHOICES_MS = (0.0, 20.0)     # 50Hz 指令通道, 延迟按 20ms 粒度量化的语义
GAIN_TOL = 0.05                   # 固件 kp/kd 寄存器容差
# 远端负载体(场景缩放质量与惯量, 代表工具/负载变化)
PAYLOAD_BODIES = ("openarm_right_link6", "openarm_right_link7",
                  "finger_openarm_right_finger_joint1",
                  "finger_openarm_right_finger_joint2",
                  "openarm_right_hand_tcp")
POLICY_HZ = 50
PHYS_DT = 0.002
# MIT 律更新率: 与 rl_validate.py 一致取 250Hz。100Hz 离散 PD 会让腕部
# 数值失稳(kd·dt/I_armature > 2, 实验见 2026-09-16 对照: 100Hz 发散/
# 250Hz max|e|2.26°); 真机固件环 kHz 级, 250Hz 是已对真机验证过的口径。
MIT_HZ = 250
RING_STEP_DEG = 2.0               # 与 PD整定流程 §5.1 真机微动一致
RING_SETTLE_S = 1.0
RING_RECORD_S = 2.5

# fork 前由 main 填充, worker 只读(fork COW 共享)
_G = {}


# ---------------------------------------------------------------- 场景 ----
def gen_scenarios(n, seed):
    """公共随机数: 所有候选用同一批场景 → 差异只来自 PD 本身。"""
    rng = np.random.default_rng(seed)
    scens = []
    for i in range(n):
        scens.append(dict(
            idx=i,
            fric_visc=rng.uniform(*FRIC_VISC_RANGE, 7),
            fric_coul=rng.uniform(*FRIC_COUL_RANGE, 7),
            payload=float(rng.uniform(*PAYLOAD_RANGE)),
            delay_ms=float(rng.choice(DELAY_CHOICES_MS)),
            kp_tol=1.0 + rng.uniform(-GAIN_TOL, GAIN_TOL, 7),
            kd_tol=1.0 + rng.uniform(-GAIN_TOL, GAIN_TOL, 7),
        ))
    return scens


def ring_scenarios():
    """ring 固定场景: 标称 + 低摩擦(欠阻尼最坏方向)。"""
    return [dict(idx=-1, fric_visc=np.full(7, FRIC_VISC_RANGE[0]),
                 fric_coul=np.full(7, FRIC_COUL_RANGE[0]),
                 payload=1.0, delay_ms=0.0,
                 kp_tol=np.ones(7), kd_tol=np.ones(7)),
            dict(idx=-2, fric_visc=np.full(7, np.mean(FRIC_VISC_RANGE)),
                 fric_coul=np.full(7, FRIC_COUL_RANGE[0]),
                 payload=PAYLOAD_RANGE[0], delay_ms=0.0,
                 kp_tol=(1 - GAIN_TOL) * np.ones(7),
                 kd_tol=(1 - GAIN_TOL) * np.ones(7))]


# ------------------------------------------------------------ 仿真底座 ----
def _load_model():
    return mujoco.MjModel.from_xml_path(XML)


def _setup_sim(model, data, scen):
    """把不确定性场景写进模型(臂 armature 常量 + 场景摩擦/负载)。"""
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                              "openarm_right_joint%d" % i)
            for i in range(1, 8)]
    vadr = np.array([model.jnt_dofadr[j] for j in jids])
    qadr = np.array([model.jnt_qposadr[j] for j in jids])
    fqids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                               "openarm_right_finger_joint%d" % i)
             for i in (1, 2)]
    fadr = np.array([model.jnt_qposadr[j] for j in fqids])

    model.dof_armature[vadr] = ARMATURE
    model.dof_damping[vadr] = scen["fric_visc"]
    model.dof_frictionloss[vadr] = scen["fric_coul"]
    if abs(scen["payload"] - 1.0) > 1e-9:
        for b in PAYLOAD_BODIES:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
            if bid >= 0:
                model.body_mass[bid] *= scen["payload"]
                model.body_inertia[bid] *= scen["payload"]
        try:
            mujoco.mj_setConst(model, data)
        except Exception:
            pass                    # 旧版 mujoco 无此接口时直接用
    return qadr, vadr, fadr


# ------------------------------------------------------------ 回放核心 ----
def run_replay(task):
    """一次 (候选, 场景) 的全轨迹纯 PD 回放 → 指标 dict。"""
    ci, kp, kd, scen, t_end, grav_ff_scale = task
    q_ref, qd_ref, meta = _G["ref"]
    grav_ff = _G.get("grav_ff")                 # (n_ref, 7) 或 None

    model = _load_model()
    data = mujoco.MjData(model)
    qadr, vadr, fadr = _setup_sim(model, data, scen)

    dt_pol = 1.0 / POLICY_HZ
    n_sub = int(round(dt_pol / PHYS_DT))        # 10
    n_mit = max(int(round(1.0 / (MIT_HZ * PHYS_DT))), 1)   # 250Hz→每2物理步
    T = float(meta["T_total"])
    t_out = float(meta["t_phase_out"])
    t_back = float(meta["t_phase_weave_end"])
    steps = min(int(T / dt_pol), int(t_end / dt_pol))
    # 指令通道 50Hz → 延迟按策略步量化(20ms 粒度), 语义=命令老化
    delay_steps = int(round(scen["delay_ms"] / 1000.0 * POLICY_HZ))
    kp_e = kp * scen["kp_tol"]
    kd_e = kd * scen["kd_tol"]
    # motor 执行器只有 forcerange(物理峰值 54/28/10), ctrlrange 是 [0,0]
    tau_lim = model.actuator_forcerange[:7, 1].copy()

    data.qpos[qadr] = q_ref[0]
    data.qpos[fadr] = 0.02
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    integ = np.zeros(7)
    err_f = np.zeros(7)
    cmd_buf = [q_ref[0].copy()]                 # 指令延迟队列(MIT 分辨率 10ms)
    vbuf = [qd_ref[0].copy()]                   # 速度参考同步延迟
    use_vref = _G.get("vref", True)

    rec_q = np.empty((steps, 7))
    rec_qref = np.empty((steps, 7))
    rec_qd = np.empty((steps, 7))
    rec_tau_pre = np.empty((steps, 7))
    rec_tau_app = np.empty((steps, 7))
    ff = np.zeros(7)

    for k in range(steps):
        idx = min(k, len(q_ref) - 1)
        q_cmd = q_ref[idx]
        if grav_ff is not None and grav_ff_scale > 0:
            ff = grav_ff[idx] * grav_ff_scale
        rec_qref[k] = q_cmd
        for s in range(n_sub):
            if s % n_mit == 0:
                if s == 0:
                    cmd_buf.append(q_cmd)
                    if len(cmd_buf) > delay_steps + 1:
                        cmd_buf.pop(0)
                    vbuf.append(qd_ref[idx])
                    if len(vbuf) > delay_steps + 1:
                        vbuf.pop(0)
                if delay_steps and len(cmd_buf) > delay_steps:
                    eff = cmd_buf[-1 - delay_steps]
                    vef = vbuf[-1 - delay_steps]
                else:
                    eff = cmd_buf[-1]
                    vef = vbuf[-1]
                q = data.qpos[qadr]
                vq = data.qvel[vadr]
                err = eff - q
                err_f = 0.3 * err + 0.7 * err_f
                big = np.abs(err_f) > 0.5
                err_f[big] = 0.0
                integ[big] = 0.0
                integ = np.clip(integ + KI * err_f * 0.01, -ICLAMP, ICLAMP)
                tau_pre = kp_e * err - kd_e * (
                    vq - (vef if use_vref else 0.0)) + integ + ff
                tau_app = np.clip(tau_pre, -tau_lim, tau_lim)
                data.ctrl[:7] = tau_app
            data.ctrl[7:9] = 0.02
            mujoco.mj_step(model, data)
        rec_q[k] = data.qpos[qadr]
        rec_qd[k] = data.qvel[vadr]
        rec_tau_pre[k] = tau_pre
        rec_tau_app[k] = tau_app

    t = np.arange(steps) * dt_pol
    Edeg = np.degrees(rec_q - rec_qref)
    ph = np.zeros(steps, dtype=int)
    ph[t >= t_out] = 1
    ph[t >= t_back] = 2

    def hf_sigma(x):
        w = max(int(0.5 / dt_pol), 3)
        ma = np.convolve(x, np.ones(w) / w, mode="same")
        return float((x - ma).std())

    w2 = max(int(2.0 / dt_pol), 3)
    droop_lf = [float(np.convolve(Edeg[:, j], np.ones(w2) / w2,
                                   mode="same").mean()) for j in range(7)]
    return dict(
        cand=ci, scen=scen["idx"], steps=int(steps),
        err_mean_deg=float(np.abs(Edeg).mean()),
        err_p95_deg=float(np.percentile(np.abs(Edeg), 95)),
        err_max_deg=float(np.abs(Edeg).max()),
        err_max_lift_deg=float(np.abs(Edeg[ph == 0]).max())
        if (ph == 0).any() else 0.0,
        tau_over_rated_max=float((np.abs(rec_tau_pre) / TAU_RATED).max()),
        hf_j1=hf_sigma(rec_qd[:, 0]),
        hf_j2=hf_sigma(rec_qd[:, 1]),
        effort=float((rec_tau_app ** 2).sum() * dt_pol),
        droop_lf_deg=droop_lf,
    )


# ------------------------------------------------------------ ring 测试 ----
def estimate_zeta(x, dt):
    """对数衰减法估 ζ(同 tune_metrics.py ring 思路, 正负峰都取)。
    返回 (zeta, f_osc); 峰不足(过阻尼)→ (None, None)。"""
    x = x - x.mean()
    std = x.std()
    if std < 1e-6:
        return None, None
    peaks = []          # (idx, val): 正峰与负峰各找一遍
    for sgn in (1.0, -1.0):
        y = sgn * x
        for i in range(1, len(y) - 1):
            if y[i] > y[i - 1] and y[i] >= y[i + 1] and abs(y[i]) > std * 0.5:
                peaks.append((i, x[i]))
    peaks.sort()
    if len(peaks) < 3:
        return None, None
    # 同号相邻极值相隔一个振荡周期
    deltas, periods = [], []
    for i in range(len(peaks) - 1):
        (i0, v0), (i1, v1) = peaks[i], peaks[i + 1]
        if np.sign(v0) != np.sign(v1) or abs(v0) <= abs(v1):
            continue
        deltas.append(math.log(abs(v0) / max(abs(v1), 1e-12)))
        periods.append((i1 - i0) * dt)
    if not deltas:
        return None, None
    delta = float(np.median(deltas))
    f_osc = 1.0 / float(np.median(periods))
    zeta = delta / math.sqrt(4 * math.pi ** 2 + delta ** 2)
    return float(zeta), float(f_osc)


def run_ring(task):
    """抬臂位形下的逐关节 2° 阶跃 ring-down → ζ(J1-J4 为硬约束输入)。
    同时上报到位率(稳态/阶跃幅): <0.5 说明该位形下 PD 权限推不动该关节,
    结果只作参考(信息项, 不一票否决)。"""
    ci, kp, kd, scen, frac = task
    q_ref, _, meta = _G["ref"]

    model = _load_model()
    data = mujoco.MjData(model)
    qadr, vadr, fadr = _setup_sim(model, data, scen)

    # 位形取抬臂段(transfer_out)的 frac 比例处 —— 振动主诉工况
    t_out = float(meta["t_phase_out"])
    idx_lift = min(int(frac * t_out * POLICY_HZ), len(q_ref) - 1)
    q_hold = q_ref[idx_lift].copy()
    step = np.radians(RING_STEP_DEG)

    data.qpos[qadr] = q_hold
    data.qpos[fadr] = 0.02
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    kp_e = kp * scen["kp_tol"]
    kd_e = kd * scen["kd_tol"]
    n_mit = max(int(round(1.0 / (MIT_HZ * PHYS_DT))), 1)
    tau_lim = model.actuator_forcerange[:7, 1].copy()

    def hold(cmd, seconds):
        n = int(seconds / PHYS_DT)
        for r in range(n):
            if r % n_mit == 0:
                err = cmd - data.qpos[qadr]
                data.ctrl[:7] = np.clip(
                    kp_e * err - kd_e * data.qvel[vadr], -tau_lim, tau_lim)
            data.ctrl[7:9] = 0.02
            mujoco.mj_step(model, data)

    out = {}
    for j in range(7):
        hold(q_hold, RING_SETTLE_S)
        cmd = q_hold.copy()
        cmd[j] += step
        n_rec = int(RING_RECORD_S / PHYS_DT)
        x = np.empty(n_rec)
        for r in range(n_rec):
            if r % n_mit == 0:
                err = cmd - data.qpos[qadr]
                data.ctrl[:7] = np.clip(
                    kp_e * err - kd_e * data.qvel[vadr], -tau_lim, tau_lim)
            data.ctrl[7:9] = 0.02
            mujoco.mj_step(model, data)
            x[r] = data.qpos[qadr[j]]
        x = np.degrees(x - x[-int(0.3 / PHYS_DT):].mean())
        zeta, f_osc = estimate_zeta(x, PHYS_DT)
        out["J%d" % (j + 1)] = None if zeta is None else round(zeta, 3)
        out["J%d_f" % (j + 1)] = None if f_osc is None else round(f_osc, 2)
        # 到位率 = 阶跃后实际位移 / 阶跃幅
        reach_deg = float(np.degrees(data.qpos[qadr[j]] - q_hold[j]))
        out["J%d_reach" % (j + 1)] = round(reach_deg / RING_STEP_DEG, 2)
    return dict(cand=ci, scen=scen["idx"], ring_cfg_idx=idx_lift, zeta=out)


# ---------------------------------------------------------------- 主流程 ----
def main():
    ap = argparse.ArgumentParser(
        description="训练前离线 PD 安全筛选(数字孪生全轨迹回放)")
    ap.add_argument("--kp-scales", default="0.8,1.0,1.25",
                    help="候选 = v2 基线 kp × 这些系数(逗号分隔)")
    ap.add_argument("--kd-scales", default="0.75,1.0,1.3")
    ap.add_argument("--kp-csv", default=None, help="显式单组 kp(覆盖网格)")
    ap.add_argument("--kd-csv", default=None)
    ap.add_argument("--scenarios", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--grav-ff", type=float, default=0.0,
                    help="重力前馈比例 0~1(qfrc_bias@参考)")
    ap.add_argument("--no-vref", action="store_true",
                    help="退回 v_des=0 旧律(默认 v_des=参考速度, 与部署一致)")
    ap.add_argument("--e-max-deg", type=float, default=4.0,
                    help="真机口径的 max|e| 门限(硬约束作用于 仿真值×"
                         "--sim2real-factor)")
    ap.add_argument("--sim2real-factor", type=float, default=1.2,
                    help="孪生→真机误差放大系数(孪生无线缆/装配公差, "
                         "2026-09-18 重标定 1.13×(权威孪生后), 旧 2.0 退役)")
    ap.add_argument("--zeta-min", type=float, default=0.45)
    ap.add_argument("--tau-gate", type=float, default=1.3,
                    help="力矩硬约束 = 该倍数×额定(2026-09-17 预算 v3: 短时"
                         "1.3×额定可接受, J3/J4 物理封顶 28)")
    ap.add_argument("--t-end", type=float, default=1e9, help="截断回放秒数")
    ap.add_argument("--ring-frac", type=float, default=0.5,
                    help="ring 位形 = 抬臂段(transfer_out)该比例处, "
                         "默认 0.5=抬臂中点(用户报告的振动工况)")
    ap.add_argument("--no-ring", action="store_true")
    ap.add_argument("--workers", type=int, default=max(
        min((os.cpu_count() or 4) - 2, 10), 1))
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟: 中心候选 × 2 场景 × 前 40s × 无 ring")
    ap.add_argument("--out", default=os.path.join(_HERE,
                                                  "pd_screen_report.json"))
    args = ap.parse_args()
    if args.smoke:
        args.kp_scales = "1.0"
        args.kd_scales = "1.0"
        args.scenarios = min(args.scenarios, 2)
        args.t_end = min(args.t_end, 40.0)
        args.no_ring = True

    # 参考与重力前馈(fork 前算好, worker COW 共享)
    d = np.load(REF, allow_pickle=True)
    q_ref, qd_ref = d["q_ref"], d["qd_ref"]
    meta = d["meta_json"].item()
    if isinstance(meta, (bytes, str)):
        meta = json.loads(meta)
    _G["ref"] = (q_ref, qd_ref, meta)
    if args.grav_ff > 0:
        # 与训练/部署同一份表(URDF 权威动力学, make_grav_ff v2 产物);
        # 表缺失时回退 URDF 现算
        from make_grav_ff import URDF, compute_grav_ff
        ff_file = os.path.abspath(os.path.join(
            _HERE, "..", "isaaclab_sim", "rl", "grav_ff_right.npz"))
        if os.path.isfile(ff_file):
            gff = np.load(ff_file)["gff"]
            assert len(gff) >= len(q_ref), "grav_ff 行数不足, 重跑 make_grav_ff.py"
        else:
            gff = compute_grav_ff(URDF, q_ref, qd_ref)
        _G["grav_ff"] = gff

    # 候选与场景
    if args.kp_csv and args.kd_csv:
        cands = [(np.array([float(x) for x in args.kp_csv.split(",")]),
                  np.array([float(x) for x in args.kd_csv.split(",")]))]
    else:
        cands = []
        for ks in [float(x) for x in args.kp_scales.split(",")]:
            for ds in [float(x) for x in args.kd_scales.split(",")]:
                cands.append((KP_NOM * ks, KD_NOM * ds))
    scens = gen_scenarios(args.scenarios, args.seed)

    tasks = [(ci, kp, kd, sc, args.t_end, args.grav_ff)
             for ci, (kp, kd) in enumerate(cands) for sc in scens]
    _G["vref"] = not args.no_vref
    t0 = time.time()
    print("候选 %d 组 × 场景 %d 个 = 回放 %d 次 | workers=%d | grav_ff=%.2f"
          " | vref=%s"
          % (len(cands), len(scens), len(tasks), args.workers, args.grav_ff,
             "on" if _G["vref"] else "off"))
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(run_replay, tasks,
                                                  chunksize=1)):
            results.append(r)
            if (i + 1) % 10 == 0 or i + 1 == len(tasks):
                print("  回放 %d/%d (%.0fs)" % (i + 1, len(tasks),
                                                time.time() - t0), flush=True)
    per_cand = [[None] * len(scens) for _ in cands]
    for r in results:
        per_cand[r["cand"]][r["scen"]] = r

    # ring(每候选 2 个固定场景: 标称 + 低摩擦)
    ring_out = []
    if not args.no_ring:
        rtasks = [(ci, kp, kd, sc, args.ring_frac)
                  for ci, (kp, kd) in enumerate(cands)
                  for sc in ring_scenarios()]
        with ctx.Pool(args.workers) as pool:
            ring_out = list(pool.imap_unordered(run_ring, rtasks,
                                                chunksize=1))
    ring_by_cand = {}
    for r in ring_out:
        ring_by_cand.setdefault(r["cand"], []).append(r)

    # ---- 汇总: 硬约束取各场景最坏, 次级取均值 ----
    rows = []
    for ci, (kp, kd) in enumerate(cands):
        rs = [r for r in per_cand[ci] if r is not None]
        zetas = []
        for r in ring_by_cand.get(ci, []):
            for j in range(4):            # J1-J4 为硬约束对象
                z = r["zeta"]["J%d" % (j + 1)]
                if z is not None:
                    zetas.append(z)
        zeta_min = min(zetas) if zetas else None   # None=全过阻尼(视作≥0.7)
        err_max = max(r["err_max_deg"] for r in rs)
        err_lift = max(r["err_max_lift_deg"] for r in rs)
        err_est = err_max * args.sim2real_factor    # 折算真机估计
        tau_ratio = max(r["tau_over_rated_max"] for r in rs)
        ok_e = err_est <= args.e_max_deg
        ok_t = tau_ratio <= args.tau_gate
        ok_z = zeta_min is None or zeta_min >= args.zeta_min
        rows.append(dict(
            cand=ci,
            kp=[round(float(x), 2) for x in kp],
            kd=[round(float(x), 2) for x in kd],
            hard=dict(err_max_deg=round(err_max, 3),
                      err_max_lift_deg=round(err_lift, 3),
                      err_est_deg=round(err_est, 3),
                      tau_over_rated_max=round(tau_ratio, 3),
                      zeta_min_j14=None if zeta_min is None
                      else round(zeta_min, 3)),
            ok=dict(err=ok_e, tau=ok_t, zeta=ok_z),
            passed=bool(ok_e and ok_t and ok_z),
            margin_e=round(args.e_max_deg / max(err_est, 1e-6), 2),
            margin_tau=round(args.tau_gate / max(tau_ratio, 1e-6), 2),
            secondary=dict(
                err_mean_deg=round(float(np.mean(
                    [r["err_mean_deg"] for r in rs])), 3),
                err_p95_deg=round(float(np.mean(
                    [r["err_p95_deg"] for r in rs])), 3),
                hf_j1=round(max(r["hf_j1"] for r in rs), 4),
                hf_j2=round(max(r["hf_j2"] for r in rs), 4),
                effort=round(max(r["effort"] for r in rs), 0),
                droop_j2_deg=round(max(
                    abs(r["droop_lf_deg"][1]) for r in rs), 3)),
        ))
    # 排序: 安全集在前, 集内按 mean|e|
    rows.sort(key=lambda r: (not r["passed"],
                             r["secondary"]["err_mean_deg"]))

    # ---- 报告 ----
    n = len(scens)
    report = dict(
        meta=dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"),
                  tool="pd_screen.py(v1, 2026-09-16)",
                  ref=REF, xml=XML,
                  scenarios=n, seed=args.seed,
                  thresholds=dict(e_max_deg=args.e_max_deg,
                                  sim2real_factor=args.sim2real_factor,
                                  tau_budget_nm=[float(x)
                                                 for x in TAU_RATED],
                                  zeta_min=args.zeta_min),
                  grav_ff=args.grav_ff, t_end=args.t_end,
                  scenario_bounds=dict(
                      fric_visc=FRIC_VISC_RANGE, fric_coul=FRIC_COUL_RANGE,
                      payload=PAYLOAD_RANGE,
                      delay_ms=list(DELAY_CHOICES_MS), gain_tol=GAIN_TOL),
                  scenario_approach=(
                      "N=%d 随机场景全部通过时, 违反概率 ε≲3/N≈%.0f%% "
                      "(95%% 置信, Calafiore & Campi 2006)"
                      % (n, 300.0 / max(n, 1)))),
        candidates=rows)
    if ring_out:
        report["ring_detail"] = ring_out
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    print("\n== PD 筛选结果(硬约束一票否决 → 安全集内按 mean|e| 排序) ==")
    print("%-4s %-6s %-6s %-5s %-8s %-8s %-8s %-6s %-8s %-6s" % (
        "序", "kp×", "kd×", "判定", "max|e|°", "抬臂°", "τ/额定", "ζmin",
        "mean°", "裕度e"))
    for i, r in enumerate(rows):
        ks = r["kp"][0] / KP_NOM[0]
        ds = r["kd"][0] / KD_NOM[0]
        print("%-4d %-6.2f %-6.2f %-5s %-8.2f %-8.2f %-8.2f %-6s %-8.2f %-6.2f"
              % (i, ks, ds, "PASS" if r["passed"] else "FAIL",
                 r["hard"]["err_max_deg"], r["hard"]["err_max_lift_deg"],
                 r["hard"]["tau_over_rated_max"],
                 "-" if r["hard"]["zeta_min_j14"] is None
                 else "%.2f" % r["hard"]["zeta_min_j14"],
                 r["secondary"]["err_mean_deg"], r["margin_e"]))
    fails = [r for r in rows if not r["passed"]]
    if fails:
        print("\nFAIL 明细(违反项):")
        for r in fails:
            why = [k for k, v in r["ok"].items() if not v]
            print("  kp[0]=%.0f kd[0]=%.1f → %s" % (
                r["kp"][0], r["kd"][0], ",".join(why)))
    print("\n场景法: " + report["meta"]["scenario_approach"])
    print("用时 %.1fs | 已写出 %s" % (time.time() - t0, args.out))


if __name__ == "__main__":
    main()
