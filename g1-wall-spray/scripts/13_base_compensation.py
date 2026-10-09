# -*- coding: utf-8 -*-
"""步骤13: 基座扰动下的世界系腕心锁定验证 (用户: 机器人轻微移动时手不跟动).

原理: 规划目标在世界系锁定(墙面点), 执行时用实测基座位姿 T_WB(t) 把世界目标
反投影回当前基座系 p_B* = T_WB(t)^-1 · p_W*, 上身IK(腰3+臂7, ±5°限位)实时增量重解。
腕部/枪在世界系的位置与指向保持, 不随基座漂移。

本脚本: 在当前 307/157 点轨迹上注入正弦基座扰动(平移±3cm / yaw±1.5° / roll,pitch±0.6°),
对比 (a)无补偿(直接回放原关节角) 与 (b)世界系锁定补偿 的:
  世界系腕心位置误差 / 枪轴指向偏差 / 墙面夹角(须>=60°) / 腰限位 / 关节速度增量
输出: output/base_comp_report.{json, png}

用法: python 13_base_compensation.py [--traj output/trajectory.npz]
"""
import os, sys, json, argparse, importlib.util
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_plan03():
    spec = importlib.util.spec_from_file_location(
        "plan03", os.path.join(BASE, "scripts", "03_plan_trajectories.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["plan03"] = m
    spec.loader.exec_module(m)
    return m


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default=os.path.join(BASE, "output", "trajectory.npz"))
    ap.add_argument("--out", default=os.path.join(BASE, "output", "base_comp_report"))
    ap.add_argument("--dx", type=float, default=0.03)     # 平移扰动幅值 (m)
    ap.add_argument("--dyaw", type=float, default=np.deg2rad(1.5))
    ap.add_argument("--drp", type=float, default=np.deg2rad(0.6))
    a = ap.parse_args()
    P03 = load_plan03()
    t = np.load(a.traj, allow_pickle=True)
    q0, pz0 = t["q"].astype(float), t["pelvis"].astype(float)
    d_cmd = t["axis"].astype(float)
    p_cmd = np.column_stack([np.full(len(q0), P03.D_ROBOT - P03.D_WRIST),
                             t["rob_y_r"].astype(float),
                             t["rob_z"].astype(float)])      # 基座系(标称)腕心目标
    N, T = len(q0), np.arange(len(q0)) / 30.0               # 30Hz 执行节拍

    # ---- 基座扰动 T_WB(t): 机器人位姿漂移(世界系) ----
    dx = a.dx * np.sin(2 * np.pi * 0.11 * T)                # 前后 ±3cm
    dy = 0.02 * np.sin(2 * np.pi * 0.07 * T + 1.0)          # 左右 ±2cm
    dz = 0.008 * np.sin(2 * np.pi * 0.23 * T)               # 上下 ±8mm
    yaw = a.dyaw * np.sin(2 * np.pi * 0.05 * T)             # 偏航 ±1.5°
    roll = a.drp * np.sin(2 * np.pi * 0.09 * T + 0.5)       # 侧倾 ±0.6°
    pit = a.drp * np.sin(2 * np.pi * 0.13 * T + 2.0)        # 俯仰 ±0.6°

    def T_WB(i):
        R = rot_z(yaw[i]) @ rot_y(pit[i]) @ rot_x(roll[i])
        return R, np.array([dx[i], dy[i], dz[i]])

    # ---- 世界系锁定目标: 规划时基座在原点 => p_W* = p_cmd ----
    p_W, d_W = p_cmd.copy(), d_cmd.copy()

    # ---- (a) 无补偿: 原关节角随基座动 ----
    _, ax0 = P03.fk_wrist(q0, pz0)
    err_nc, ang_nc = [], []
    for i in range(N):
        R, tr = T_WB(i)
        p_w = R @ p_cmd[i] + tr
        ax_w = R @ ax0[i]
        err_nc.append(np.linalg.norm(p_w - p_W[i]))
        ang_nc.append(90.0 - np.degrees(np.arccos(np.clip(ax_w[0], -1, 1))))
    err_nc = np.array(err_nc); ang_nc = np.array(ang_nc)

    # ---- (b) 世界系锁定: 目标反投影到当前基座系 -> 上身IK(批量+残余热启动+60°守卫) ----
    q1 = q0.copy(); pz1 = pz0.copy()
    stroke = t["kind"] == "stroke"
    p_B = np.zeros((N, 3)); d_B = np.zeros((N, 3))
    for i in range(N):                                   # 反投影(世界目标 -> 当前基座系)
        R, tr = T_WB(i)
        p_B[i] = R.T @ (p_W[i] - tr)
        d_B[i] = R.T @ d_W[i]
    d_B /= np.linalg.norm(d_B, axis=1, keepdims=True)
    qf, pf, e1, e2, cv = P03.batch_ik(q0.copy(), pz0.copy(), p_B, d_B,
                                      "stand", iters=250)     # 整批: q0 即好种子(扰动小)
    q1[cv] = qf[cv]; pz1[cv] = pf[cv]
    n_ik = int(cv.sum())
    for i in np.where(~cv)[0]:                            # 残余: 邻点热启动
        seed = q1[i - 1] if i and cv[i - 1] else q0[i]
        qf2, pf2, _, _, cv2 = P03.batch_ik(seed[None].copy(),
                                           np.array([pz1[i]]), p_B[i:i+1], d_B[i:i+1],
                                           "stand", iters=400)
        if cv2[0]:
            q1[i], pz1[i] = qf2[0], float(pf2[0]); n_ik += 1
    # 60°守卫(批量): 世界系角<60 的 stroke 点 -> 世界目标朝垂直偏+0.8° 反投影重解
    _, ax_t = P03.fk_wrist(q1, pz1)
    ang_t = np.zeros(N)
    for i in range(N):
        R, _ = T_WB(i)
        ang_t[i] = 90.0 - np.degrees(np.arccos(np.clip((R @ ax_t[i])[0], -1, 1)))
    low = np.where(stroke & (ang_t < 60.0))[0]
    if len(low):
        tb = np.deg2rad(0.8)
        th = np.arccos(np.clip(d_W[low, 0], -1, 1)) - tb
        dW2 = np.column_stack([np.cos(th), np.zeros(len(low)),
                               np.sign(d_W[low, 2]) * np.sqrt(np.maximum(0, 1 - np.cos(th)**2))])
        dB2 = np.zeros_like(dW2)
        for k, i in enumerate(low):
            R, _ = T_WB(i)
            dB2[k] = R.T @ dW2[k]
        dB2 /= np.linalg.norm(dB2, axis=1, keepdims=True)
        qf3, pf3, _, _, cv3 = P03.batch_ik(q1[low].copy(), pz1[low], p_B[low], dB2,
                                           "stand", iters=90, ax_tol=0.02)
        _, ax3 = P03.fk_wrist(qf3, pf3)
        for k, i in enumerate(low):
            if not cv3[k]:
                continue
            R, _ = T_WB(i)
            a3 = 90.0 - np.degrees(np.arccos(np.clip((R @ ax3[k])[0], -1, 1)))
            j_lo = np.abs(qf3[k] - q1[max(i-1,0)]).max()
            j_hi = np.abs(q1[min(i+1,N-1)] - qf3[k]).max()
            if a3 >= 60.0 and max(j_lo, j_hi) < 0.6:
                q1[i], pz1[i] = qf3[k], float(pf3[k])
    p_c, ax_c = P03.fk_wrist(q1, pz1)
    err_c, ang_c = [], []
    for i in range(N):
        R, tr = T_WB(i)
        p_w = R @ p_c[i] + tr
        ax_w = R @ ax_c[i]
        err_c.append(np.linalg.norm(p_w - p_W[i]))
        ang_c.append(90.0 - np.degrees(np.arccos(np.clip(ax_w[0], -1, 1))))
    err_c = np.array(err_c); ang_c = np.array(ang_c)

    # ---- 指标 ----
    dq = np.abs(q1 - q0).max()                              # 补偿引起的关节改动量
    vel_add = np.abs(np.diff(q1 - q0, axis=0)).max() * 30   # 附加关节速度
    _wlim = float(os.environ.get("G1W_WAIST_RP", "0")) or 0.0872665
    waist_ok = bool((np.abs(q1[:, 1]) <= _wlim + 1e-6).all()
                    and (np.abs(q1[:, 2]) <= _wlim + 1e-6).all())
    rep = dict(
        扰动=dict(平移x=f"±{a.dx*100:.0f}cm", 平移y="±2cm", 平移z="±0.8cm",
                  yaw=f"±{np.degrees(a.dyaw):.1f}°", roll_pitch=f"±{np.degrees(a.drp):.1f}°"),
        无补偿=dict(世界腕心误差max_mm=round(float(err_nc.max() * 1000), 1),
                     世界腕心误差mean_mm=round(float(err_nc.mean() * 1000), 1),
                     墙面夹角min_deg=round(float(ang_nc.min()), 2)),
        世界系锁定=dict(IK收敛=f"{n_ik}/{N}",
                         世界腕心误差max_mm=round(float(err_c.max() * 1000), 1),
                         世界腕心误差mean_mm=round(float(err_c.mean() * 1000), 1),
                         墙面夹角min_deg=round(float(ang_c[stroke].min()), 2),
                         腰限位保持=waist_ok,
                         最大关节改动_rad=round(float(dq), 3),
                         附加关节速度max_rad_s=round(float(vel_add), 2)),
        结论=("基座扰动下, 世界系锁定把腕心误差从 "
              f"{err_nc.max()*1000:.1f}mm 压到 {err_c.max()*1000:.1f}mm; "
              f"角度规范与腰限位保持" if waist_ok else "存在超限!"))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rep, open(a.out + ".json", "w", encoding="utf8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))

    # ---- 对比图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.join(BASE, "scripts"))
    from common import setup_cn_font
    setup_cn_font()
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    ax = axs[0]
    ax.plot(T, err_nc * 1000, label=f"无补偿 (max {err_nc.max()*1000:.1f}mm)", color="#e5484d")
    ax.plot(T, err_c * 1000, label=f"世界系锁定 (max {err_c.max()*1000:.1f}mm)", color="#29d391")
    ax.axhline(8, color="#f5a623", ls="--", lw=1.2, label="喷涂精度线 8mm")
    ax.set_ylabel("世界系腕心误差 (mm)"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("基座扰动下的世界系腕心锁定 (平移±3cm / yaw±1.5° / roll,pitch±0.6°)")
    ax = axs[1]
    ax.plot(T, dx * 100, label="dx (cm)"); ax.plot(T, dy * 100, label="dy (cm)")
    ax.plot(T, dz * 100, label="dz (cm)")
    ax2 = ax.twinx()
    ax2.plot(T, np.degrees(yaw), ":", color="#888", label="yaw (°)")
    ax.set_ylabel("基座平移漂移 (cm)"); ax2.set_ylabel("偏航 (°)")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right"); ax.grid(alpha=.3)
    ax = axs[2]
    ax.plot(T, np.degrees(q1[:, 1]), label="补偿后 腰roll")
    ax.plot(T, np.degrees(q1[:, 2]), label="补偿后 腰pitch")
    ax.axhline(5, color="#e5484d", ls="--", lw=1); ax.axhline(-5, color="#e5484d", ls="--", lw=1)
    ax.set_ylabel("腰 roll/pitch (°)"); ax.set_xlabel("t (s)")
    ax.set_ylim(-6, 6); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(a.out + ".png", dpi=120)
    print(f"[报告] {a.out}.json / {a.out}.png")


if __name__ == "__main__":
    main()
