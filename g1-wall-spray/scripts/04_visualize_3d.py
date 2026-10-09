# -*- coding: utf-8 -*-
"""步骤4: 三维可视化 — 舱室墙面段点云 + 墙面之字形轨迹 + 机器人之字形轨迹 + G1 机器人.

输出:
  fig_3d_scene.png     3D 全景(墙面点云+双色轨迹+G1 位姿)
  fig_3d_front.png     正对墙视角
  fig_3d_side.png      侧视(垂直/倾斜指向关系)
  g1_poses.npz         展示位姿的关节角
"""
import numpy as np
import os, json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("G1W_OUT", os.path.join(BASE, "output"))
IN = os.environ.get("G1W_IN", OUT)

X_FACE = 212.292
FLOOR_Z = 57.563
D_WRIST = 0.40
GUN_LEN = 0.12

# --------- G1 运动学(与03一致) ---------
WAIST = [
    ((0.0, 0.0, 0.0), (0, 0, 0), 'z', -2.618, 2.618),
    ((-0.0039635, 0, 0.035), (0, 0, 0), 'x', -0.52, 0.52),
    ((0.0, 0.0, 0.019), (0, 0, 0), 'y', -0.52, 0.52),
]
ARM_R = [
    ((0.0039563, -0.10021, 0.23778), (-0.27931, 5.4949e-05, 0.00019159), 'y', -3.0892, 2.6704),
    ((0.0, -0.038, -0.013831), (0.27925, 0, 0), 'x', -2.2515, 1.5882),
    ((0.0, -0.00624, -0.1032), (0, 0, 0), 'z', -2.618, 2.618),
    ((0.015783, 0, -0.080518), (0, 0, 0), 'y', -1.0472, 2.0944),
    ((0.100, -0.00188791, -0.010), (0, 0, 0), 'x', -1.9722, 1.9722),
    ((0.038, 0, 0), (0, 0, 0), 'y', -1.6144, 1.6144),
    ((0.046, 0, 0), (0, 0, 0), 'z', -1.6144, 1.6144),
]
CHAIN = WAIST + ARM_R
LEG_L = [
    ((0.0, 0.064452, -0.1027), (0, 0, 0), 'y'),
    ((0.0, 0.052, -0.030465), (0, -0.1749, 0), 'x'),
    ((0.025001, 0, -0.12412), (0, 0, 0), 'z'),
    ((-0.078273, 0.0021489, -0.17734), (0, 0.1749, 0), 'y'),
    ((0.0, -9.4445e-05, -0.30001), (0, 0, 0), 'y'),
    ((0.0, 0.0, -0.017558), (0, 0, 0), 'x'),
]
L_THIGH = 0.194          # hip_pitch->knee
L_SHANK = 0.318          # knee->ankle(含踝-足底)


def fk_chain_poses(q, pz):
    """返回 (关节原点列表[11], 末端姿态R, 各关节轴向量)"""
    N = q.shape[0]
    R = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
    p = np.broadcast_to(np.array([0.0, 0.0, 0.0]), (N, 3)).copy()
    p[:, 2] = pz
    joints = [p.copy()]
    e_ax = {'x': [1., 0., 0.], 'y': [0., 1., 0.], 'z': [0., 0., 1.]}
    axes = []
    for i, (xyz, rpy, ax, lo, hi) in enumerate(CHAIN):
        p = p + np.einsum('nij,j->ni', R, np.array(xyz))
        joints.append(p.copy())
        Rr = R @ (rotxf(np.full(N, rpy[0])) @ rotyf(np.full(N, rpy[1])) @ rotzf(np.full(N, rpy[2])))
        axes.append(np.einsum('nij,j->ni', Rr, np.array(e_ax[ax])))
        R = np.einsum('nij,njk->nik', Rr, AXF[ax](q[:, i]))
    return joints, R, axes


def rotxf(a):
    c, s = np.cos(a), np.sin(a)
    R = np.zeros((len(a), 3, 3))
    R[:, 0, 0] = 1; R[:, 1, 1] = c; R[:, 1, 2] = -s
    R[:, 2, 1] = s; R[:, 2, 2] = c
    return R


def rotyf(a):
    c, s = np.cos(a), np.sin(a)
    R = np.zeros((len(a), 3, 3))
    R[:, 0, 0] = c; R[:, 0, 2] = s; R[:, 1, 1] = 1
    R[:, 2, 0] = -s; R[:, 2, 2] = c
    return R


def rotzf(a):
    c, s = np.cos(a), np.sin(a)
    R = np.zeros((len(a), 3, 3))
    R[:, 0, 0] = c; R[:, 0, 1] = -s; R[:, 1, 0] = s; R[:, 1, 1] = c; R[:, 2, 2] = 1
    return R


AXF = {'x': rotxf, 'y': rotyf, 'z': rotzf}


def leg_ik(pelvis_z, base_x, base_y, side):
    """蹲姿/站姿 腿部2连杆IK(矢状面): 足底贴地. 返回 [(hip, knee, ankle, toe)] (世界系)"""
    hip_y = base_y + (0.0664 if side == 'L' else -0.0664)   # hip_pitch y 偏移
    hip_x = base_x
    hip_z = pelvis_z - 0.1027
    foot_z = FLOOR_Z
    dz = hip_z - foot_z                    # hip->足底 垂直距离
    r = np.hypot(dz, 0.0)                  # 足在髋正下方(前后偏移0)
    r = min(r, L_THIGH + L_SHANK - 1e-3)
    a = np.arccos(np.clip((L_THIGH**2 + r**2 - L_SHANK**2) / (2 * L_THIGH * r), -1, 1))
    knee_x = hip_x + L_THIGH * np.sin(a)
    knee_z = hip_z - L_THIGH * np.cos(a)
    ankle_z = foot_z + 0.038
    toe_x, toe_y, toe_z = base_x + 0.12, hip_y, FLOOR_Z
    return [(hip_x, hip_y, hip_z), (knee_x, hip_y, knee_z),
            (hip_x, hip_y, ankle_z), (toe_x, toe_y, toe_z)]


def main():
    from common import setup_cn_font
    setup_cn_font()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    t = np.load(os.path.join(OUT, "trajectory.npz"), allow_pickle=True)
    _cfgp = os.environ.get("G1W_CFG", "")
    cfg = json.load(open(_cfgp, encoding="utf8")) if _cfgp and os.path.exists(_cfgp) else None
    CLOUD = bool(cfg and cfg.get("cloud") and os.path.exists(cfg["cloud"]))
    if CLOUD:   # 点云模式: 世界系=机器人系(基座在0), 常量由配置给出
        FX = float(cfg.get("wall_x") or cfg.get("d_robot") or 0.74)
        U0v, FZv = 0.0, 0.0
    else:
        FX, U0v, FZv = X_FACE, 7.789, FLOOR_Z
    sec = np.load(os.path.join(IN, "wall_section.npz"))
    meta = json.load(open(os.path.join(IN, "wall_section_meta.json"), encoding="utf8"))
    LBL = meta["labels"]

    u_w, h_w = t["wall_u"], t["wall_h"]
    wall_xyz, rob_xyz = t["wall_xyz"], t["rob_xyz"]
    cls, kind, zone, mode = t["cls"], t["kind"], t["zone"], t["mode"]
    axis, conv, q, pelvis = t["axis"], t["ik_ok"], t["q"], t["pelvis"]
    u_robot = float(t["u_robot"])
    base_x, base_y = FX - 0.74, U0v + u_robot     # 标称站位0.74m(保证带0.70~0.78)
    SINGLE = "#1f7ac2"          # 统一走线颜色
    colors = dict(green=SINGLE, red=SINGLE, yellow=SINGLE)

    # ---- 背景点云: 点云模式用用户点云(机器人系=世界系), 否则舱室场景 ----
    if CLOUD:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("cloud_scan", os.path.join(BASE, "scripts", "11_cloud_scan.py"))
        _cs = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_cs)
        cpts = _cs.load_cloud(cfg["cloud"])
        n_sub = min(len(cpts), 60000)
        _cid = np.random.default_rng(0).choice(len(cpts), n_sub, replace=False)
        P = cpts[_cid]                        # (x,y,z) 直接即世界坐标
        PW = P.copy()
        L = np.zeros(n_sub, int)
        lab_c = {0: "#9fb8c8"}
        lab_n = {0: "点云"}
    else:
        pts = sec["points"].astype(float)     # (u,h,d)
        lab = sec["label"]
        # 转世界: x = FX - d, y = U0v + u, z = FZv + h
        n_sub = min(len(pts), 60000)
        idx = np.random.default_rng(0).choice(len(pts), n_sub, replace=False)
        P = pts[idx]; L = lab[idx]
        PW = np.stack([FX - P[:, 2], U0v + P[:, 0], FZv + P[:, 1]], 1)
        lab_c = {0: "#b8b8b8", 1: "#9fd49f", 2: "#c77fdb", 3: "#ffd83d", 4: "#3f8fe8", 5: "#3a3a3a"}
        lab_n = {0: "地面", 1: "墙", 2: "门框", 3: "窗", 4: "加强筋", 5: "管道/其他"}

    fig = plt.figure(figsize=(24, 13))

    # ============ 视图1: 3D 全景 ============
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    for k in np.unique(L):
        m = L == k
        ax.scatter(PW[m, 0], PW[m, 1], PW[m, 2], s=2.5, c=lab_c[k], alpha=0.55,
                   depthshade=False)
    # 墙面轨迹(抬高1cm避免埋进点云)
    walloff = wall_xyz + np.array([0.01, 0, 0])
    segs, segc = [], []
    for i in range(1, len(u_w)):
        if kind[i] == "stroke" and kind[i - 1] == "stroke":
            segs.append([walloff[i - 1], walloff[i]])
            segc.append(colors[cls[i]])
    ax.add_collection3d(Line3DCollection(segs, colors=segc, linewidths=2.2))
    # 机器人轨迹
    segs, segc = [], []
    for i in range(1, len(u_w)):
        segs.append([rob_xyz[i - 1], rob_xyz[i]])
        segc.append(colors[cls[i]])
    ax.add_collection3d(Line3DCollection(segs, colors=segc, linewidths=2.2,
                                         linestyles="dashed"))
    ax.plot(rob_xyz[:, 0], rob_xyz[:, 1], rob_xyz[:, 2], ".", color="#444", ms=2)
    # 站位与地面
    ax.scatter([base_x], [base_y], [0], marker="s", s=120, c="k")
    ax.text(base_x, base_y, 0.05, "G1站位", fontsize=9)
    gx, gy = np.meshgrid(np.linspace(base_x - 0.7, FX, 4),
                         np.linspace(7.589, 8.789, 4))
    ax.plot_surface(gx, gy, np.full_like(gx, FZv - 0.001), alpha=0.10, color="gray")
    # G1 展示位姿: 直接取自最终导出动作 v2 (MJCF-IK 验证过的真实帧)
    import csv as _csv, glob as _glob
    _meta = json.load(open(os.path.join(OUT, "trajectory_meta.json")))
    _D = _meta["d_robot_nominal"]
    base_x, base_y = FX - _D, U0v + _meta["u_robot"]
    _csvf = sorted(_glob.glob(os.path.join(OUT, "ufo_motion", "v*",
                                           "wall_spray_motion_qpos.csv")))
    kf = []
    if _csvf:
        _rows = list(_csv.reader(open(_csvf[-1])))
        _hdr = _rows[0]; _C = {n: i for i, n in enumerate(_hdr)}
        def _f(r, n):
            try: return float(r[_C[n]])
            except Exception: return np.nan
        track_idx = [i for i, r in enumerate(_rows[1:]) if r[_C["phase"]].startswith("track")]
        for name, frac in [("站姿喷上区(起始)", 0.06), ("站姿喷上区(顶部)", 0.30),
                           ("站姿喷上区(回落)", 0.52), ("蹲姿喷下区(底部)", 0.75),
                           ("蹲姿喷下区(中段)", 0.92)]:
            i = track_idx[int(frac * (len(track_idx) - 1))]
            r = _rows[1 + i]
            q10 = np.array([_f(r, "waist_yaw"), 0, 0] +
                           [_f(r, f"right_{j}_joint") for j in
                            ("shoulder_pitch", "shoulder_roll", "shoulder_yaw",
                             "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")])
            pz = _f(r, "root_tz")
            wrist = np.array([base_x + _f(r, "wrist_x"), base_y + _f(r, "wrist_y"),
                              FZv + pz + _f(r, "wrist_z")])
            axis_v = np.array([_f(r, "axis_dx"), _f(r, "axis_dy"), _f(r, "axis_dz")])
            nrm = np.linalg.norm(axis_v)
            axis_v = axis_v / nrm if nrm > 1e-6 else np.array([1.0, 0, 0])
            wall_pt = wrist + axis_v * ((FX - wrist[0]) / max(axis_v[0], 1e-6))
            kf.append((name, q10, pz, wrist, axis_v, wall_pt, r[_C["phase"]][6:]))
    else:
        # 独立运行目录尚无动作导出: 用 03 的关节解直接取 5 个展示位姿
        _N = len(q)
        for name, frac in [("位姿(起始)", 0.06), ("位姿(上部)", 0.30), ("位姿(中部)", 0.52),
                           ("位姿(下部)", 0.75), ("位姿(末段)", 0.92)]:
            i = int(frac * (_N - 1))
            q10 = np.asarray(q[i], float)
            pz = float(pelvis[i])
            wrist = np.asarray(rob_xyz[i], float).copy()
            _a = np.asarray(axis[i], float)
            axis_v = _a / max(np.linalg.norm(_a), 1e-6)
            wall_pt = wrist + axis_v * ((FX - wrist[0]) / max(axis_v[0], 1e-6))
            kf.append((name, q10, pz, wrist, axis_v, wall_pt, "track"))
    OUT_POS = os.path.join(OUT, 'ufo_motion', 'g1_poses.npz')
    poses = []
    for name, qi, pzi, wrist_w, axis_w, wall_pt, cls_name in kf:
        draw_g1(ax, qi, pzi, base_x, base_y, wrist_w, axis_w, wall_pt,
                SINGLE, name)
        poses.append(dict(name=name, q=qi.tolist(),
                          pelvis=float(pzi), wrist=wrist_w.tolist(),
                          axis=axis_v.tolist()))
    os.makedirs(os.path.dirname(OUT_POS), exist_ok=True)
    np.savez(os.path.join(OUT_POS), **{p["name"]: np.array(p["q"] + [p["pelvis"]])
                                       for p in poses})
    ax.set_xlim(base_x - 0.35, FX + 0.1)
    ax.set_ylim(base_y - 0.65, base_y + 0.65)
    ax.set_zlim(FZv - 0.05, FZv + 1.7)
    ax.set_box_aspect((0.95, 1.3, 1.75))
    ax.view_init(elev=12, azim=-75)
    ax.set_title("3D 全景: 墙面点云 + 墙面/机器人之字形轨迹 + G1")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")

    # ============ 视图2: 正对墙 ============
    ax2 = fig.add_subplot(1, 3, 2)
    for k in np.unique(L):
        m = L == k
        ax2.scatter(P[m, 0], P[m, 1], s=0.8, c=lab_c[k], alpha=0.4)
    for i in range(1, len(u_w)):
        if kind[i] == "stroke" and kind[i - 1] == "stroke":
            ax2.plot([u_w[i - 1], u_w[i]], [h_w[i - 1], h_w[i]], "-", color=colors[cls[i]],
                     lw=2.0, solid_capstyle="round")
        else:
            ax2.plot([u_w[i - 1], u_w[i]], [h_w[i - 1], h_w[i]], "-", color=colors[cls[i]],
                     lw=1.6, alpha=0.9)
    if CLOUD:
        for o in cfg.get("obstacles", []):
            ax2.add_patch(plt.Rectangle((o["u"][0], o["h"][0]),
                                        o["u"][1] - o["u"][0], o["h"][1] - o["h"][0],
                                        fill=False, ec="#e07b00", lw=1.6))
            ax2.text(o["u"][0], o["h"][1] + 0.015, o["kind"], color="#e07b00", fontsize=8)
        ax2.set_xlim(float(u_w.min()) - 0.10, float(u_w.max()) + 0.10)
        ax2.set_ylim(float(h_w.min()) - 0.10, float(h_w.max()) + 0.12)
    else:
        ax2.add_patch(plt.Rectangle((0, 0), 0.73, 0.90, fill=False, ec="navy", lw=1.6))
        ax2.add_patch(plt.Rectangle((0, 0.90), 0.73, 0.6, fill=False, ec="purple", lw=1.6))
        ax2.text(0.02, 0.42, "蹲姿矩形 [0,0.90]", color="navy", fontsize=10, weight="bold")
        ax2.text(0.02, 1.26, "站姿矩形 [0.90,1.50]", color="purple", fontsize=10, weight="bold")
        ax2.set_xlim(-0.06, 0.86); ax2.set_ylim(-0.06, 1.58)
    ax2.set_aspect("equal")
    ax2.set_title("墙面之字形轨迹(正对墙)")
    ax2.set_xlabel("u (m)"); ax2.set_ylabel("h (m)"); ax2.grid(alpha=0.25)

    # ============ 视图3: 侧视 ============
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.axvline(0, color="k", lw=2)                      # 墙面
    _ribd = max([o.get("depth", 0) for o in cfg.get("obstacles", [])], default=0.125) \
        if CLOUD else 0.125
    ax3.axvline(-_ribd, color="#3f8fe8", lw=3, alpha=0.6)  # 最深凸出(点云模式取点云实测)
    ax3.axvline(-0.4, color="#888", lw=1, ls="--")
    ax3.text(-0.4, 1.52, "腕心平面(距墙0.40m)", ha="center", fontsize=8)
    idxr = np.linspace(0, len(u_w) - 1, 130).astype(int)
    for i in idxr:
        if not conv[i]:
            continue
        ax3.plot([0, -0.4], [h_w[i], rob_xyz[i, 2] - FZv], "-",
                 color=SINGLE, lw=0.6, alpha=0.5)
    ax3.plot(-0.4, 0, "ks", ms=8)
    ax3.text(-0.42, -0.06, "G1站位", fontsize=9, ha="right")
    ax3.axhline(0.15, color="r", ls=":", lw=1.4)
    ax3.text(-0.68, 0.16, "安全下限0.15m", color="r", fontsize=8)
    ax3.axhline(0.90, color="k", ls="-.", lw=1.2)
    ax3.text(-0.68, 0.92, "蹲/站分界0.90m", fontsize=8)
    ax3.set_xlim(-0.75, 0.1); ax3.set_ylim(-0.1, 1.6); ax3.set_aspect("equal")
    ax3.set_title("侧视: 末端指向关系(蓝=加强筋凸出125mm)")
    ax3.set_xlabel("距墙距离 (m)"); ax3.set_ylabel("h (m)"); ax3.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_3d_scene.png"), dpi=125)
    print("[预览] fig_3d_scene.png")
    plt.close(fig)


def draw_g1(ax, qi, pzi, base_x, base_y, wrist_w, axis_w, wall_pt, color, name):
    """在3D图上画 G1 杆状图: 腿(2连杆IK)+躯干+臂(FK)+喷枪+喷雾锥"""
    joints, R_end, axes = fk_chain_poses(qi[None], np.array([pzi]))
    J = [j[0] for j in joints]                     # 11个原点(基座系)
    # 基座->世界
    off = np.array([base_x, base_y, 0.0])
    Jw = [np.array(j) + off for j in J]
    pelvis, waist_yaw, waist_roll, waist_pitch = Jw[0], Jw[1], Jw[2], Jw[3]
    sh_p, sh_r, sh_y, elbow, wr_r, wr_p, wr_y = Jw[4], Jw[5], Jw[6], Jw[7], Jw[8], Jw[9], Jw[10]
    # 腿
    for side in ("L", "R"):
        hip, knee, ankle, toe = leg_ik(pzi, base_x, base_y, side)
        ax.plot([pelvis[0], hip[0]], [pelvis[1], hip[1]], [pelvis[2], hip[2]],
                color="#555", lw=3)
        ax.plot([hip[0], knee[0], ankle[0], toe[0]], [hip[1], knee[1], ankle[1], toe[1]],
                [hip[2], knee[2], ankle[2], toe[2]], color="#555", lw=3)
    # 躯干
    ax.plot([pelvis[0], waist_pitch[0]], [pelvis[1], waist_pitch[1]],
            [pelvis[2], waist_pitch[2]], color="#333", lw=6)
    ax.plot([waist_pitch[0], sh_p[0]], [waist_pitch[1], sh_p[1]],
            [waist_pitch[2], sh_p[2]], color="#333", lw=6)
    # 臂
    ax.plot([sh_p[0], sh_r[0], sh_y[0], elbow[0], wr_r[0], wr_p[0], wr_y[0]],
            [sh_p[1], sh_r[1], sh_y[1], elbow[1], wr_p[1], wr_r[1], wr_y[1]],
            [sh_p[2], sh_r[2], sh_y[2], elbow[2], wr_r[2], wr_p[2], wr_y[2]],
            color="#c46a1f", lw=3.5)
    # 头
    head = waist_pitch + np.array([0.02, 0, 0.24])
    ax.plot([waist_pitch[0], head[0]], [waist_pitch[1], head[1]],
            [waist_pitch[2], head[2]], color="#333", lw=4)
    ax.scatter([head[0]], [head[1]], [head[2]], s=60, c="#333")
    # 喷枪 + 喷雾
    gun_tip = wrist_w + axis_w * GUN_LEN
    ax.plot([wr_y[0], wrist_w[0], gun_tip[0]], [wr_y[1], wrist_w[1], gun_tip[1]],
            [wr_y[2], wrist_w[2], gun_tip[2]], color="#c46a1f", lw=3)
    ax.plot([gun_tip[0], wall_pt[0]], [gun_tip[1], wall_pt[1]], [gun_tip[2], wall_pt[2]],
            color=color, lw=0.8, alpha=0.35, ls=":")
    for j in Jw:
        ax.scatter([j[0]], [j[1]], [j[2]], s=8, c="#c46a1f")
    ax.scatter([wrist_w[0]], [wrist_w[1]], [wrist_w[2]], s=26, c="k")
    ax.text(pelvis[0], pelvis[1], pelvis[2] + 0.06, name, fontsize=8, color="#333")


if __name__ == "__main__":
    main()
