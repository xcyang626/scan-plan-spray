# -*- coding: utf-8 -*-
"""步骤2: 宇树 G1 (29DOF, 无灵巧手) 工作区域蒙特卡洛计算.

方法(与方案文件 ws_calc.py / ws_squat.py 一致, 换 G1 运动学):
  - 关节链: 腰(yaw/roll/pitch) + 单臂7关节, 末端 = 腕yaw关节中心(末端前一关节),
    喷枪轴 = 腕yaw系 x 轴(前臂延长线)
  - 约束1: 腕心位于距墙 0.40m 的平面 -> 机器人离墙 d∈[0.70,0.78] (腰锁定后定位保证带,
    标称0.74m), 故基座系下腕平面 x_w ∈ [0.30, 0.38];
    对 x_w={0.30,0.34,0.38} 分别建占据网格后取交集 = 定位误差下"保证垂直"工作区域
  - 约束2(喷涂质量, 用户定稿): 喷枪轴与墙面夹角须在 60°~90°(越接近90°越好)
    -> 垂直模式 ≤15°(与法向), 倾斜模式 ≤30°(与法向), 用于上下/左右边缘指向
  - 站姿: 骨盆 0.77~0.79m; 蹲姿: 骨盆 0.42~0.70m (腿部FK求可行下限)
  - 腰部扭转范围 ±30° 与 ±60° 两种都计算, 供轨迹规划选择

输出: output/g1_workspace.npz + fig_workspace.png + g1_workspace_stats.json
"""
import numpy as np
import os, json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("G1W_OUT", os.path.join(BASE, "output"))
_WRP = float(os.environ.get("G1W_WAIST_RP", "0"))   # 网页端腰限覆盖(rad); 0=用源码默认

# ---------------------------------------------------------------- G1 运动学
# 腰部三自由度: yaw放开, roll/pitch 收紧至 ±5°(用户更新: 更保守的稳定裕度)
WAIST = [
    ((0.0, 0.0, 0.0),            (0, 0, 0), 'z', -2.618, 2.618),      # waist_yaw
    ((-0.0039635, 0, 0.035),     (0, 0, 0), 'x',
     -(_WRP if _WRP > 0 else 0.0872665), (_WRP if _WRP > 0 else 0.0872665)),  # waist_roll
    ((0.0, 0.0, 0.019),          (0, 0, 0), 'y',
     -(_WRP if _WRP > 0 else 0.0872665), (_WRP if _WRP > 0 else 0.0872665)),  # waist_pitch
]
ARM_R = [
    ((0.0039563, -0.10021, 0.23778), (-0.27931, 5.4949e-05, 0.00019159), 'y', -3.0892, 2.6704),
    ((0.0, -0.038, -0.013831),       (0.27925, 0, 0),                   'x', -2.2515, 1.5882),
    ((0.0, -0.00624, -0.1032),       (0, 0, 0),                         'z', -2.618, 2.618),
    ((0.015783, 0, -0.080518),       (0, 0, 0),                         'y', -1.0472, 2.0944),
    ((0.100, -0.00188791, -0.010),   (0, 0, 0),                         'x', -1.9722, 1.9722),
    ((0.038, 0, 0),                  (0, 0, 0),                         'y', -1.6144, 1.6144),
    ((0.046, 0, 0),                  (0, 0, 0),                         'z', -1.6144, 1.6144),
]
ARM_L = [
    ((0.0039563, 0.10022, 0.23778), (0.27931, 5.4949e-05, -0.00019159), 'y', -3.0892, 2.6704),
    ((0.0, 0.038, -0.013831),       (-0.27925, 0, 0),                   'x', -1.5882, 2.2515),
    ((0.0, 0.00624, -0.1032),       (0, 0, 0),                          'z', -2.618, 2.618),
    ((0.015783, 0, -0.080518),      (0, 0, 0),                          'y', -1.0472, 2.0944),
    ((0.100, 0.00188791, -0.010),   (0, 0, 0),                          'x', -1.9722, 1.9722),
    ((0.038, 0, 0),                 (0, 0, 0),                          'y', -1.6144, 1.6144),
    ((0.046, 0, 0),                 (0, 0, 0),                          'z', -1.6144, 1.6144),
]
LEG_L = [
    ((0.0, 0.064452, -0.1027),  (0, 0, 0),        'y', -2.5307, 2.8798),
    ((0.0, 0.052, -0.030465),   (0, -0.1749, 0),  'x', -0.5236, 2.9671),
    ((0.025001, 0, -0.12412),   (0, 0, 0),        'z', -2.7576, 2.7576),
    ((-0.078273, 0.0021489, -0.17734), (0, 0.1749, 0), 'y', -0.0873, 2.8798),
    ((0.0, -9.4445e-05, -0.30001), (0, 0, 0),      'y', -0.8727, 0.5236),
    ((0.0, 0.0, -0.017558),     (0, 0, 0),        'x', -0.2618, 0.2618),
]

PELVIS_STAND = (0.77, 0.79)   # 站姿骨盆
_CROUCH_LO = float(os.environ.get("G1W_CROUCH_LO", "0")) or 0.60
PELVIS_CROUCH = (_CROUCH_LO, 0.70)  # 蹲姿骨盆地板(实测 height_sweep: 策略可稳到0.29, 缺省0.60保守)
SOLE_TO_ANKLE = 0.038
X_PLANES = (0.30, 0.34, 0.38)  # 腕平面: 离墙0.70/0.74/0.78m (腰锁定后0.90m物理不可达)
PLANE_TOL = 0.03
COS_PERP = np.cos(np.deg2rad(15))
COS_TILT = np.cos(np.deg2rad(55))   # 倾斜采样上限(超集), 占据网格再按 TILT_GRID 过滤
TILT_GRID = 30.0                    # 倾斜判据: 与墙面法向≤30° <=> 与墙面夹角≥60°(喷涂质量)
AXIS_DOWN_Z = -0.02                 # 下倾样本阈值: 枪轴任意真下倾即计入(近垂直归 perp)
Y_RNG = (-0.75, 0.75)
Z_RNG = (0.0, 1.75)
RES = 0.02
N_Y = int(round((Y_RNG[1] - Y_RNG[0]) / RES))
N_Z = int(round((Z_RNG[1] - Z_RNG[0]) / RES))
Y_EDGES = np.linspace(Y_RNG[0], Y_RNG[1], N_Y + 1)
Z_EDGES = np.linspace(Z_RNG[0], Z_RNG[1], N_Z + 1)
Y_CEN = 0.5 * (Y_EDGES[:-1] + Y_EDGES[1:])
Z_CEN = 0.5 * (Z_EDGES[:-1] + Z_EDGES[1:])


def rotx(a):
    c, s = np.cos(a), np.sin(a)
    R = np.zeros((len(a), 3, 3))
    R[:, 0, 0] = 1; R[:, 1, 1] = c; R[:, 1, 2] = -s
    R[:, 2, 1] = s; R[:, 2, 2] = c
    return R


def roty(a):
    c, s = np.cos(a), np.sin(a)
    R = np.zeros((len(a), 3, 3))
    R[:, 0, 0] = c; R[:, 0, 2] = s; R[:, 1, 1] = 1
    R[:, 2, 0] = -s; R[:, 2, 2] = c
    return R


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    R = np.zeros((len(a), 3, 3))
    R[:, 0, 0] = c; R[:, 0, 1] = -s; R[:, 1, 0] = s; R[:, 1, 1] = c; R[:, 2, 2] = 1
    return R


AXF = {'x': rotx, 'y': roty, 'z': rotz}


def fk_chain(joints, q, pelvis_z):
    """批量FK. q:(N,k), pelvis_z: 标量或(N,). 返回末端位置(N,3)与末端姿态(N,3,3)"""
    N = q.shape[0]
    R = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
    p = np.broadcast_to(np.array([0.0, 0.0, 0.0]), (N, 3)).copy()
    p[:, 2] = pelvis_z
    for i, (xyz, rpy, ax, lo, hi) in enumerate(joints):
        p = p + np.einsum('nij,j->ni', R, np.array(xyz))
        Rr = R @ (rotx(np.full(N, rpy[0])) @ roty(np.full(N, rpy[1])) @ rotz(np.full(N, rpy[2])))
        R = np.einsum('nij,njk->nik', Rr, AXF[ax](q[:, i]))
    return p, R


def sample_crouch_pelvis(n=400000, seed=11):
    rng = np.random.default_rng(seed)
    hip = rng.uniform(-2.5307, 2.8798, n)
    knee = rng.uniform(-0.0873, 2.8798, n)
    ankle = -(hip + knee)
    ok = (ankle >= -0.8727) & (ankle <= 0.5236)
    # hip/knee/ankle 三列同口径过滤(ok 掩膜+同截断), 否则踝角取的是未过滤前
    # len(hip) 个样本, 与过滤后的 hip/knee 错位配对
    hip, knee, ankle = hip[ok][:200000], knee[ok][:200000], ankle[ok][:200000]
    q = np.stack([hip, np.zeros_like(hip), np.zeros_like(hip),
                  knee, ankle, np.zeros_like(hip)], 1)
    p, _ = fk_chain(LEG_L, q, 0.0)
    pelvis = -p[:, 2] + SOLE_TO_ANKLE
    pelvis = pelvis[pelvis > 0.2]
    print(f"[蹲姿腿FK] 骨盆高度范围 [{pelvis.min():.3f}, {pelvis.max():.3f}] m, "
          f"P2={np.percentile(pelvis, 2):.3f}, P10={np.percentile(pelvis, 10):.3f}")
    return float(np.percentile(pelvis, 2))


def sample_workspace(arm, yaw_range, pelvis_range, n=4_000_000, seed=7, nq_sub=20000):
    """一次FK得到腕心与枪轴, 再按垂直/下倾/上倾+三个腕平面分类; 同时保存关节角子样本"""
    chain = WAIST + arm
    rng = np.random.default_rng(seed)
    lo = np.array([j[3] for j in chain]); hi = np.array([j[4] for j in chain])
    q = rng.uniform(lo, hi, size=(n, len(chain)))
    q[:, 0] = np.clip(q[:, 0], yaw_range[0], yaw_range[1])
    pz = rng.uniform(pelvis_range[0], pelvis_range[1], n)
    p_end, R_end = fk_chain(chain, q, pz)
    axis = R_end[:, :, 0]
    keep = axis[:, 0] > 0.2
    p_end, axis, pz, q = p_end[keep], axis[keep], pz[keep], q[keep]
    n = len(p_end)
    tilt = np.arccos(np.clip(axis[:, 0], -1, 1))          # 枪轴与墙面法向夹角
    # 关节角子样本(供轨迹IK种子): 只取腕平面带内样本(x∈[0.28,0.52]), 均匀抽取
    band = (p_end[:, 0] > 0.28) & (p_end[:, 0] < 0.52)
    nb = int(band.sum())
    if nb > nq_sub:
        sub_idx = rng.choice(np.where(band)[0], size=nq_sub, replace=False)
    else:
        sub_idx = np.where(band)[0]
    q_sub = np.column_stack([q[sub_idx], pz[sub_idx]])    # 最后一列 = 骨盆高度
    end_sub = np.column_stack([p_end[sub_idx][:, :3], tilt[sub_idx]])
    res = {}
    for x_w in X_PLANES:
        mp = np.abs(p_end[:, 0] - x_w) < PLANE_TOL
        perp = mp & (tilt <= np.deg2rad(15))
        tdn = mp & (tilt <= np.deg2rad(55)) & (axis[:, 2] < AXIS_DOWN_Z)
        tup = mp & (tilt <= np.deg2rad(55)) & (axis[:, 2] > -AXIS_DOWN_Z)
        res[x_w] = dict(
            perp=np.stack([p_end[perp][:, 1], p_end[perp][:, 2],
                           np.degrees(tilt[perp])], 1),
            tilt_d=np.stack([p_end[tdn][:, 1], p_end[tdn][:, 2],
                             np.degrees(tilt[tdn])], 1),
            tilt_u=np.stack([p_end[tup][:, 1], p_end[tup][:, 2],
                             np.degrees(tilt[tup])], 1),
        )
    res['q_sub'] = q_sub
    res['end_sub'] = end_sub     # (x,y,z,tilt) 对应 q_sub
    return res


def occ_grid(yz, tilt=None, max_tilt=None):
    """占据网格; 若给 tilt 则只统计 tilt<=max_tilt 的点"""
    if tilt is not None and max_tilt is not None:
        yz = yz[tilt <= max_tilt]
    H, _, _ = np.histogram2d(yz[:, 0], yz[:, 1], bins=[Y_EDGES, Z_EDGES])
    return H > 0


def and_grids(samples_per_plane, key, max_tilt=None):
    g = None
    for x_w in X_PLANES:
        s = samples_per_plane[x_w][key]
        gi = occ_grid(s[:, :2], s[:, 2] if key != 'perp' else None, max_tilt)
        g = gi if g is None else (g & gi)
    return g


def solid_z_range(g, frac=0.85):
    """网格中"实心"z区间: 该z层的y向宽度 ≥ 0.85×最大宽度"""
    widths = np.array([(Y_CEN[row].max() - Y_CEN[row].min()) if row.any() else 0.0
                       for row in g.T])
    wmax = widths.max()
    ok = np.where(widths >= frac * max(wmax, 1e-9))[0]
    return (float(Z_CEN[ok.min()]), float(Z_CEN[ok.max()])) if len(ok) else (np.nan, np.nan)


def main():
    from common import setup_cn_font
    setup_cn_font()
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p2 = sample_crouch_pelvis()
    crouch_lo = max(p2, _CROUCH_LO)   # 实测地板(G1W_CROUCH_LO)与腿FK下限取大
    PELVIS_CROUCH = (crouch_lo, 0.70)
    print(f"[蹲姿采样范围] 骨盆 {PELVIS_CROUCH}; 站姿 {PELVIS_STAND}")
    print("[站位保证带] 腰锁定后 x_w∈[0.30,0.38] -> 离墙 d∈[0.70,0.78]m (标称0.74m)")

    SLICE_BAND = (-0.40, 0.40)
    all_stats = {}
    saved = {}
    for yaw_deg in (30, 60):
        YAW = (-np.deg2rad(yaw_deg), np.deg2rad(yaw_deg))
        for arm_name, arm in [("right", ARM_R), ("left", ARM_L)]:
            for mode, pr in [("stand", PELVIS_STAND), ("crouch", PELVIS_CROUCH)]:
                # 确定性种子: 逐字符 ord 累积哈希(内置 hash 受 PYTHONHASHSEED 影响,
                # 同参数不同进程结果不同, 不可复现)
                _sd = 0
                for ch in arm_name:
                    _sd = (_sd * 31 + ord(ch)) % 1000003
                r = sample_workspace(arm, YAW, pr, n=4_000_000,
                                     seed=7 + (_sd + 31 * yaw_deg) % 97)
                key = f"{arm_name}_{mode}_y{yaw_deg}"
                from scipy import ndimage
                ST = np.ones((3, 3), bool)
                def close(g):
                    return ndimage.binary_closing(g, ST) if g.any() else g
                g_perp = close(and_grids(r, 'perp'))
                g_td = close(and_grids(r, 'tilt_d', max_tilt=TILT_GRID))   # 下倾≤30°(与墙面夹角≥60°)
                g_tu = close(and_grids(r, 'tilt_u', max_tilt=TILT_GRID))
                yext_per_plane = {}
                for x_w in X_PLANES:
                    gp = occ_grid(r[x_w]['perp'][:, :2])
                    rr = np.where(gp.any(1))[0]
                    yext_per_plane[x_w] = (float(Y_CEN[rr.min()]), float(Y_CEN[rr.max()])) if len(rr) else None
                rows = np.where(g_perp.any(1))[0]
                y_ext = (float(Y_CEN[rows.min()]), float(Y_CEN[rows.max()])) if len(rows) else None
                zr = solid_z_range(g_perp)
                # 段宽度带内的 z 范围
                band = (Y_CEN >= SLICE_BAND[0]) & (Y_CEN <= SLICE_BAND[1])
                band_cols = np.where(g_perp[band].any(0))[0]
                z_band = (float(Z_CEN[band_cols.min()]), float(Z_CEN[band_cols.max()])) if len(band_cols) else None
                all_stats[key] = dict(
                    y_ext=y_ext, z_solid=zr, z_band=z_band, y_per_plane=yext_per_plane,
                    n_perp=int(g_perp.sum()), n_td=int(g_td.sum()), n_tu=int(g_tu.sum()))
                print(f"[{key:16s}] 垂直保证区 y[{y_ext[0]:+.2f},{y_ext[1]:+.2f}] "
                      f"z实心[{zr[0]:.2f},{zr[1]:.2f}] 段带内z[{z_band[0]:.2f},{z_band[1]:.2f}] "
                      f"格数 perp={g_perp.sum()} td={g_td.sum()} tu={g_tu.sum()} "
                      f"单平面y: " + " ".join(f"xw{x_w}={yext_per_plane[x_w]}" for x_w in X_PLANES))
                saved[f"{key}_perp"] = g_perp
                saved[f"{key}_tiltd"] = g_td
                saved[f"{key}_tiltu"] = g_tu
                saved[f"{key}_qsub"] = r['q_sub'].astype(np.float32)
                saved[f"{key}_endsub"] = r['end_sub'].astype(np.float32)

    saved["ys"] = Y_CEN; saved["zs"] = Z_CEN
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "g1_workspace.npz"), **saved)
    with open(os.path.join(OUT, "g1_workspace_stats.json"), "w", encoding="utf8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    # ---- 图: 8宫格(站/蹲 × 左/右臂 × ±30/±60) ----
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    ys, zs = Y_CEN, Z_CEN
    for j, yaw_deg in enumerate((30, 60)):
        for i, mode in enumerate(("stand", "crouch")):
            for k, arm_name in enumerate(("right", "left")):
                ax = axes[i, j * 2 + k]
                g = saved[f"{arm_name}_{mode}_y{yaw_deg}_perp"]
                yy, zz = np.meshgrid(ys, zs, indexing="ij")
                ax.scatter(yy[g], zz[g], s=1.2, c="tab:blue")
                gd = saved[f"{arm_name}_{mode}_y{yaw_deg}_tiltd"]
                gu = saved[f"{arm_name}_{mode}_y{yaw_deg}_tiltu"]
                ax.scatter(yy[gd], zz[gd], s=0.8, c="tab:green", alpha=0.35)
                ax.scatter(yy[gu], zz[gu], s=0.8, c="tab:orange", alpha=0.35)
                ax.axvspan(-0.40, 0.40, color="gold", alpha=0.10)
                ax.axhspan(0, 1.5, color="gray", alpha=0.08)
                ax.set_xlim(-0.75, 0.75); ax.set_ylim(0, 1.75); ax.set_aspect("equal")
                ax.set_title(f"{mode} 腰±{yaw_deg}° {arm_name}臂\n蓝:垂直 绿:下倾30° 橙:上倾30°")
                ax.set_xlabel("y (m)"); ax.set_ylabel("z (m)"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_workspace.png"), dpi=110)
    print("[预览]", os.path.join(OUT, "fig_workspace.png"))


if __name__ == "__main__":
    main()
