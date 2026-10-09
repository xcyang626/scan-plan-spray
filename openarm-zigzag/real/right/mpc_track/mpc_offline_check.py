#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""MPC 离线复核: 用本 MPC(PreviewMPC) 对新路线做闭环仿真验证(不碰硬件)
================================================================
背景: mpc_track README 挂着的实机前置"MPC 离线复核(用本 MPC 核对新
路线闭环)"。本脚本复刻 mpc_exec.run 的参考生成(分段速率时间参数化)
与指令闭环(PreviewMPC.step → 双积分推进 → 软限位投影), 把真机端换成
**实机辨识模型**(ident_result.json 的指令→位置二阶环, 含驱动 PD+ki)
+ 代表性未建模扰动(重力偏置/库仑摩擦, 量级见 DISTURB, 可调)。

输出(控制台 + JSON):
  - 走线段跟踪偏差(指令态/辨识对象态 vs 参考, °, max/RMS)
  - 指令峰值速率 / HF σ(相邻周期速率差的标准差, 抖动指标)
  - a 饱和占比 / 软限位最小裕量(投影后, °) / 参考时长
判读参考(左臂 2026-09-07 离线复核口径): 跟踪 max≈1°, 峰值速率<30°/s。

用法:
  /usr/bin/python3 mpc_offline_check.py [--ref dense...] [--speed-scale 0.4]
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from mpc_core import PreviewMPC          # noqa: E402

NAMES = ["openarm_right_joint%d" % i for i in range(1, 8)]
ISA = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "仿真", "isaaclab_sim"))
PKG = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "仿真", "openarm_sim"))

# 未建模扰动(代表性, 仅用于看偏差量级; 基座 J1/J3 轴近竖直无重力项)。
# 折算: 等效指令偏移 d_rad = (grav + fc·sign(q̇)) / kp_j, kp=真机 MIT 增益
# (真机 ki 积分器会再把它压掉大半, 故此口径偏保守)。
DISTURB = {
    "grav_nm": [0.0, 2.0, 0.0, 1.0, 0.4, 0.3, 0.15],
    "fc_nm":   [0.30, 0.30, 0.20, 0.20, 0.10, 0.10, 0.10],
    "kp":      [120.0, 70.0, 70.0, 90.0, 30.0, 30.0, 30.0],
}
D2G = 180.0 / math.pi


def build_reference(cfg, pts, speed_scale, seg_tcp):
    """与 mpc_exec.MpcExec.build_reference 逐行同逻辑(去 logger)。"""
    home = np.array(cfg["home_rad"])
    chain_pts = [home] + [np.array(p) for p in pts] + [home]
    tcp = cfg["tcp_speed"] * speed_scale
    v_app = cfg["approach_joint_speed"]
    v_ret = cfg["return_joint_speed"]
    lens, speeds, caps = [], [], []
    n_dense = len(chain_pts) - 2
    for k in range(len(chain_pts) - 1):
        a, b = chain_pts[k], chain_pts[k + 1]
        if 0 < k <= n_dense:
            lens.append(float(seg_tcp[k - 1]))
            speeds.append(tcp)
            caps.append(cfg["weave_joint_speed"])
        else:
            lens.append(float(np.abs(b - a).max()))
            speeds.append(v_app if k == 0 else v_ret)
            caps.append(0.0)
    T = []
    for k in range(len(lens)):
        t_tcp = lens[k] / speeds[k] if speeds[k] > 0 else 0.0
        t_joint = 0.0
        if caps[k] > 0.0:
            jn = float(np.abs(chain_pts[k + 1] - chain_pts[k]).max())
            t_joint = jn / caps[k]
        T.append(max(t_tcp, t_joint))
    total = sum(T)
    dt = cfg["dt"]
    m = int(total / dt) + 1
    R = np.zeros((m + cfg["N"] + 2, 7))
    for j in range(len(R)):
        tt = min(j * dt, total)
        acc = 0.0
        for k in range(len(T)):
            if tt <= acc + T[k] or k == len(T) - 1:
                f = 0.0 if T[k] <= 0 else min(max((tt - acc) / T[k], 0.0), 1.0)
                R[j] = chain_pts[k] + f * (chain_pts[k + 1] - chain_pts[k])
                break
            acc += T[k]
    return R, dt, total, (T[0], sum(T[1:1 + n_dense]), T[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=os.path.join(
        _HERE, "..", "..", "..", "仿真", "moveit_sim",
        "dense_zigzag_right.json"))
    ap.add_argument("--speed-scale", type=float, default=0.4)
    ap.add_argument("--config", default=os.path.join(_HERE, "mpc_config.json"))
    ap.add_argument("--ident", default=os.path.join(_HERE, "ident_result.json"))
    ap.add_argument("--plant-wn", type=float, default=12.0,
                    help="代表性对象闭环带宽 rad/s(ζ=1)。右臂 ident_result.json "
                         "的 J2-J7 R²≈0 不可用, 故默认用代表性对象(左臂 9/07 "
                         "离线复核同口径); 仅 J1 的辨识(R²=0.98)可用")
    ap.add_argument("--out", default=os.path.join(_HERE,
                                                  "mpc_offline_check.json"))
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))
    with open(os.path.abspath(os.path.join(_HERE, "..", "config",
                                           "real_safety.yaml")),
              encoding="utf-8") as f:
        s = yaml.safe_load(f)
    cfg["home_rad"] = list(s["home_rad"])
    cfg["soft_lo"] = {n: s["soft_limits_rad"][n][0] for n in NAMES}
    cfg["soft_hi"] = {n: s["soft_limits_rad"][n][1] for n in NAMES}

    data = json.load(open(args.ref, encoding="utf-8"))
    pts = data["points"]

    # 走线段 TCP 长度(与 mpc_exec 同法: isaaclab 右臂链 + 0.12 指尖段)
    sys.path.insert(0, ISA)
    sys.path.insert(0, PKG)
    from common import UrdfChain          # noqa: E402
    chain = UrdfChain(os.path.join(ISA, "openarm_right.urdf"),
                      tip_link="openarm_right_hand_tcp")
    seg_tcp = [0.0]
    for i in range(1, len(pts)):
        _, pa = chain.fk(np.array(pts[i - 1]), 0.12)
        _, pb = chain.fk(np.array(pts[i]), 0.12)
        seg_tcp.append(float(np.linalg.norm(pb - pa)))

    R, dt, total, T_split = build_reference(cfg, pts, args.speed_scale,
                                            seg_tcp)
    print("参考: 接近 %.1fs + 走线 %.1fs + 回家 %.1fs = %.0fs"
          % (T_split[0], T_split[1], T_split[2], total))

    lo_c = np.array([cfg["soft_lo"][n] for n in NAMES]) \
        + math.radians(cfg["clearance_deg"])
    hi_c = np.array([cfg["soft_hi"][n] for n in NAMES]) \
        - math.radians(cfg["clearance_deg"])

    mpc = PreviewMPC(7, N=cfg["N"], dt=cfg["dt"], w_p=cfg["w_p"],
                     w_v=cfg["w_v"], w_a=cfg["w_a"], a_max=cfg["a_max"])

    # 辨识对象: 指令→位置 二阶连续环(实机 100Hz 扫频, RMSE≈0.2°)
    # 对象模型: 代表性闭环二阶环(带宽 plant_wn, ζ=1)。右臂辨识 J2-J7
    # 的 R²≈0(模型不可用), 仅 J1 R²=0.98 —— 结果 JSON 里保留辨识 R² 供查。
    ident = json.load(open(args.ident, encoding="utf-8"))["results"]
    wn = np.full(7, float(args.plant_wn))
    ze = np.ones(7)
    ident_r2_report = [round(float(ident.get(str(i + 1), {})
                                  .get("r2", 0.0)), 3) for i in range(7)]
    d_rad = (np.array(DISTURB["grav_nm"]) + np.array(DISTURB["fc_nm"])) \
        / np.array(DISTURB["kp"])

    q = np.array(cfg["home_rad"])         # 与真机同: 臂在 home 起步
    v = np.zeros(7)
    pq, pdq = q.copy(), v.copy()          # 辨识对象态
    k = 0
    n_sat = 0
    n_cmd = 0
    n_clip = 0
    min_soft_margin = math.inf
    err_cmd_max = err_cmd_weave_mean = 0.0
    err_pl_max = err_pl_weave_mean = 0.0
    rate_peak = 0.0
    rate_spike = None                     # (k, joint, rate°/s, pre-clip qn)
    rate_prev = np.zeros(7)
    hf_acc = []
    rate_hist = []                        # 逐步"最快关节速率", 供分位数
    pl_err_max_j = np.zeros(7)
    weave0, weave1 = int(T_split[0] / dt), int((T_split[0] + T_split[1]) / dt)
    n_weave = 0

    while True:                            # 与 mpc_exec.run 同款判界
        k += 1
        if k >= len(R) - cfg["N"]:
            break
        ref_win = R[k:k + cfg["N"] + 1]
        qn, vn = [], []
        sat = False
        for j in range(7):
            a = mpc.step(q[j], v[j], ref_win[:, j], j)
            if abs(a) >= cfg["a_max"] - 1e-9:
                sat = True
            a = max(-cfg["a_max"], min(cfg["a_max"], a))
            qn.append(q[j] + dt * v[j] + 0.5 * dt * dt * a)
            vn.append(v[j] + dt * a)
        qn_raw = np.array(qn)
        qn = np.clip(qn_raw, lo_c, hi_c)
        clipped = float(np.abs(qn - qn_raw).max()) > 1e-9
        n_clip += clipped
        vn = np.array(vn)
        n_sat += sat
        n_cmd += 1
        min_soft_margin = min(
            min_soft_margin,
            float(np.min(np.minimum(qn - lo_c + math.radians(
                cfg["clearance_deg"]),
                hi_c - qn + math.radians(cfg["clearance_deg"])))))
        rate = (qn - q) / dt
        rmax = float(np.abs(rate).max()) * D2G
        rate_peak = max(rate_peak, rmax)
        rate_hist.append(rmax)
        hf_acc.append(float(np.abs(rate - rate_prev).max()) * D2G)
        if rmax > 60.0 and (rate_spike is None or rmax > rate_spike[2]):
            rate_spike = (k, int(np.argmax(np.abs(rate))) + 1, rmax,
                          [round(x, 4) for x in qn_raw],
                          "CLIPPED" if clipped else "no-clip")
        rate_prev = rate
        err_c = float(np.abs(qn - R[min(k, len(R) - 1)]).max()) * D2G
        err_cmd_max = max(err_cmd_max, err_c)
        # 辨识对象: 1ms 子步 ZOH 跟随指令(扰动=等效指令偏移, 方向随速度)
        # 注意: 仅 R² 合格的关节(≥0.8)的对象态判读有效, 其余仅供参考
        for _ in range(10):
            h = 0.001
            u = qn - d_rad * np.sign(pdq)
            acc_p = wn * wn * (u - pq) - 2.0 * ze * wn * pdq
            pdq = pdq + h * acc_p
            pq = pq + h * pdq
        err_p = np.abs(pq - R[min(k, len(R) - 1)]) * D2G   # 度
        pl_err_max_j = np.maximum(pl_err_max_j, err_p)
        err_pl_max = max(err_pl_max, float(err_p.max()))
        if weave0 < k <= weave1:
            n_weave += 1
            err_cmd_weave_mean += err_c
            err_pl_weave_mean += float(err_p.max())
        q, v = qn, vn

    res = {
        "ref": os.path.abspath(args.ref),
        "speed_scale": args.speed_scale,
        "ref_total_s": round(total, 1),
        "n_commands": n_cmd,
        "a_saturation_frac": round(n_sat / max(n_cmd, 1), 4),
        "n_clip_events": int(n_clip),
        "min_soft_margin_deg": round(min_soft_margin * D2G, 3),
        "track_cmd_max_deg": round(err_cmd_max, 3),
        "track_plant_max_deg": round(err_pl_max, 3),
        "plant_track_max_deg_per_joint": [round(float(x), 2)
                                          for x in pl_err_max_j],
        "ident_r2_per_joint": ident_r2_report,
        "track_plant_weave_mean_deg": round(
            err_pl_weave_mean / max(n_weave, 1), 3),
        "cmd_rate_peak_deg_s": round(rate_peak, 2),
        "cmd_rate_p99_deg_s": round(float(np.percentile(rate_hist, 99)), 2),
        "rate_spike_event": rate_spike,
        "cmd_hf_step_mean_deg_s": round(float(np.mean(hf_acc)), 3),
        "cmd_hf_step_max_deg_s": round(float(np.max(hf_acc)), 3),
        "plant": "代表性闭环二阶环 wn=%.1f ζ=1 + 重力/摩擦等效指令偏移; "
                 "右臂辨识 J2-J7 R²≈0 不可用(留档 ident_r2_per_joint)"
                 % args.plant_wn,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print("\n==== MPC 离线复核结果 ====")
    print("指令 a 饱和占比      : %.2f%%" % (100 * res["a_saturation_frac"]))
    print("投影裁剪事件数       : %d" % res["n_clip_events"])
    print("软限位最小裕量       : %.2f° (设计内缩 3.0°)" % res["min_soft_margin_deg"])
    print("跟踪偏差 指令态 max  : %.2f°" % res["track_cmd_max_deg"])
    print("跟踪偏差 对象态 max  : %.2f° | 走线段均值 %.3f°"
          % (res["track_plant_max_deg"], res["track_plant_weave_mean_deg"]))
    print("指令峰值速率         : %.1f°/s | p99 %.1f°/s" % (
        res["cmd_rate_peak_deg_s"], res["cmd_rate_p99_deg_s"]))
    if rate_spike:
        print("速率尖峰定位         : 步 %d, J%d, %.1f°/s, %s"
              % (rate_spike[0], rate_spike[1], rate_spike[2], rate_spike[4]))
    print("指令速率步进 mean/max: %.2f / %.2f °/s" % (
        res["cmd_hf_step_mean_deg_s"], res["cmd_hf_step_max_deg_s"]))
    print("已写出", args.out)
    gates = [("指令态跟踪 max < 2°", res["track_cmd_max_deg"] < 2.0),
             ("对象态跟踪 max < 6°(含保守扰动)", res["track_plant_max_deg"] < 6.0),
             ("指令峰值速率 p99 < 30°/s", res["cmd_rate_p99_deg_s"] < 30.0),
             ("软限位最小裕量 ≥ 2.9°", res["min_soft_margin_deg"] >= 2.9),
             ("a 饱和占比 < 5%", res["a_saturation_frac"] < 0.05)]
    print("\n判读: " + " | ".join(
        "%s %s" % (n, "✓" if okx else "✗") for n, okx in gates))
    sys.exit(0 if all(o for _, o in gates) else 1)


if __name__ == "__main__":
    main()
