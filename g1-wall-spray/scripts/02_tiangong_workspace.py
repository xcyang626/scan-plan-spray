# -*- coding: utf-8 -*-
"""步骤2b: 天工3 (Tiangong3) 工作区域蒙特卡洛计算 —— 逻辑直接复用 02_g1_workspace.

差异仅在机器人常数(来自 tiangong3_kinematics.py 的 URDF 程序化提取):
  - 站姿骨盆 ~1.02m(直腿FK), 蹲姿带由天工腿FK采样;
  - 腕平面: 天工臂长(肩前平举腕心 ~0.51m), 标称离墙 d_robot=0.90m,
    腕平面 x_w = d_robot-0.40 ± 0.04 -> (0.46,0.50,0.54);
  - 覆盖网格按天工 KPI 扩到 y±1.0m / z 0~2.3m (原地覆盖 0~2m).
腰部 yaw 档位仍 ±30°/±60°(03 规划同口径); 输出键名与 02 一致(right_stand_y30...).

输出: $G1W_OUT/tiangong_workspace.npz + fig_tiangong_workspace.png
      + tiangong_workspace_stats.json
"""
import os

import numpy as np

import tiangong3_kinematics as T3
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "ws_g1", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "02_g1_workspace.py"))
ws_g1 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(ws_g1)
fk_chain, rotx, roty, rotz = ws_g1.fk_chain, ws_g1.rotx, ws_g1.roty, ws_g1.rotz

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("G1W_OUT", os.path.join(BASE, "output"))
_CROUCH_LO = float(os.environ.get("G1W_CROUCH_LO", "0")) or 0.62

WAIST, ARM_R, ARM_L, LEG_R = T3.WAIST, T3.ARM_R, T3.ARM_L, T3.LEG_R

PELVIS_STAND = (0.96, 1.005)             # 腿几何上限1.005直腿; 下限0.96轻屈膝(与03同带)
SOLE_TO_ANKLE = T3.SOLE_TO_ANKLE
X_PLANES = (0.46, 0.50, 0.54)            # 腕平面: 离墙 0.86/0.90/0.94m(标称0.90)
PLANE_TOL = 0.03
COS_PERP = np.cos(np.deg2rad(15))
TILT_GRID = 30.0
AXIS_DOWN_Z = -0.02
Y_RNG = (-1.00, 1.00)
Z_RNG = (0.0, 2.30)
RES = 0.02
N_Y = int(round((Y_RNG[1] - Y_RNG[0]) / RES))
N_Z = int(round((Z_RNG[1] - Z_RNG[0]) / RES))
Y_EDGES = np.linspace(*Y_RNG, N_Y + 1)
Z_EDGES = np.linspace(*Z_RNG, N_Z + 1)
Y_CEN = 0.5 * (Y_EDGES[:-1] + Y_EDGES[1:])
Z_CEN = 0.5 * (Z_EDGES[:-1] + Z_EDGES[1:])


def sample_crouch_pelvis(n=400000, seed=11):
    """天工腿FK采样蹲姿骨盆范围(平面三关节, 其余置零; hip_yaw/knee/ankle 限位过滤)."""
    rng = np.random.default_rng(seed)
    hip = rng.uniform(-2.79, 2.53, n)      # hip_pitch(腿下垂为0, 弯曲为正方向实测调)
    knee = rng.uniform(-0.087, 2.53, n)
    ankle = -(hip + knee)
    ok = (ankle >= -1.187) & (ankle <= 0.524)
    hip, knee, ankle = (hip[ok][:200000], knee[ok][:200000], ankle[ok][:200000])
    q = np.stack([hip, np.zeros_like(hip), np.zeros_like(hip),
                  knee, ankle, np.zeros_like(hip)], 1)
    p, _ = fk_chain(LEG_R, q, 0.0)
    pelvis = -p[:, 2] + SOLE_TO_ANKLE
    pelvis = pelvis[(pelvis > 0.4) & (pelvis < 1.05)]
    print(f"[蹲姿腿FK] 骨盆高度范围 [{pelvis.min():.3f}, {pelvis.max():.3f}] m, "
          f"P2={np.percentile(pelvis, 2):.3f}, P10={np.percentile(pelvis, 10):.3f}")
    return float(np.percentile(pelvis, 2)), float(np.percentile(pelvis, 10))


def sample_workspace(arm, yaw_range, pelvis_range, n=4_000_000, seed=7, nq_sub=20000):
    chain = WAIST + arm
    rng = np.random.default_rng(seed)
    lo = np.array([j[3] for j in chain])
    hi = np.array([j[4] for j in chain])
    q = rng.uniform(lo, hi, size=(n, len(chain)))
    q[:, 0] = np.clip(q[:, 0], yaw_range[0], yaw_range[1])
    pz = rng.uniform(pelvis_range[0], pelvis_range[1], n)
    p_end, R_end = fk_chain(chain, q, pz)
    axis = R_end[:, :, 0]
    keep = axis[:, 0] > 0.2
    p_end, axis, pz, q = p_end[keep], axis[keep], pz[keep], q[keep]
    tilt = np.arccos(np.clip(axis[:, 0], -1, 1))
    band = (p_end[:, 0] > 0.40) & (p_end[:, 0] < 0.62)
    nb = int(band.sum())
    sub_idx = (rng.choice(np.where(band)[0], size=nq_sub, replace=False)
               if nb > nq_sub else np.where(band)[0])
    res = {}
    for x_w in X_PLANES:
        mp = np.abs(p_end[:, 0] - x_w) < PLANE_TOL
        res[x_w] = dict(
            perp=np.stack([p_end[mp][:, 1], p_end[mp][:, 2],
                           np.degrees(tilt[mp])], 1),
            tilt_d=np.stack([p_end[(mp) & (axis[:, 2] < AXIS_DOWN_Z)][:, 1],
                             p_end[(mp) & (axis[:, 2] < AXIS_DOWN_Z)][:, 2],
                             np.degrees(tilt[(mp) & (axis[:, 2] < AXIS_DOWN_Z)])], 1),
            tilt_u=np.stack([p_end[(mp) & (axis[:, 2] > -AXIS_DOWN_Z)][:, 1],
                             p_end[(mp) & (axis[:, 2] > -AXIS_DOWN_Z)][:, 2],
                             np.degrees(tilt[(mp) & (axis[:, 2] > -AXIS_DOWN_Z)])], 1),
        )
    res['q_sub'] = np.column_stack([q[sub_idx], pz[sub_idx]])
    res['end_sub'] = np.column_stack([p_end[sub_idx][:, :3], tilt[sub_idx]])
    return res


def occ_grid(yz, tilt=None, max_tilt=None):
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
    widths = np.array([(Y_CEN[row].max() - Y_CEN[row].min()) if row.any() else 0.0
                       for row in g.T])
    ok = np.where(widths >= frac * max(widths.max(), 1e-9))[0]
    return (float(Z_CEN[ok.min()]), float(Z_CEN[ok.max()])) if len(ok) else (np.nan, np.nan)


def main():
    from common import setup_cn_font
    setup_cn_font()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage

    p2, p10 = sample_crouch_pelvis()
    crouch_lo = max(p2, _CROUCH_LO)
    PELVIS_CROUCH = (crouch_lo, 1.00)
    print(f"[蹲姿采样范围] 骨盆 {PELVIS_CROUCH}; 站姿 {PELVIS_STAND}")
    print(f"[站位] 天工标称离墙 0.90m, 腕平面 x_w∈[{X_PLANES[0]},{X_PLANES[2]}] "
          f"(站位保证带 0.86~0.94m)")

    SLICE_BAND = (-0.40, 0.40)
    all_stats = {}
    saved = {}
    ST = np.ones((3, 3), bool)

    def close(g):
        return ndimage.binary_closing(g, ST) if g.any() else g

    for yaw_deg in (30, 60):
        YAW = (-np.deg2rad(yaw_deg), np.deg2rad(yaw_deg))
        for arm_name, arm in [("right", ARM_R), ("left", ARM_L)]:
            for mode, pr in [("stand", PELVIS_STAND), ("crouch", PELVIS_CROUCH)]:
                _sd = 0
                for ch in arm_name:
                    _sd = (_sd * 31 + ord(ch)) % 1000003
                r = sample_workspace(arm, YAW, pr, n=4_000_000,
                                     seed=7 + (_sd + 31 * yaw_deg) % 97)
                key = f"{arm_name}_{mode}_y{yaw_deg}"
                g_perp = close(and_grids(r, 'perp'))
                g_td = close(and_grids(r, 'tilt_d', max_tilt=TILT_GRID))
                g_tu = close(and_grids(r, 'tilt_u', max_tilt=TILT_GRID))
                rows = np.where(g_perp.any(1))[0]
                y_ext = ((float(Y_CEN[rows.min()]), float(Y_CEN[rows.max()]))
                         if len(rows) else None)
                zr = solid_z_range(g_perp)
                bandm = (Y_CEN >= SLICE_BAND[0]) & (Y_CEN <= SLICE_BAND[1])
                band_cols = np.where(g_perp[bandm].any(0))[0]
                z_band = ((float(Z_CEN[band_cols.min()]), float(Z_CEN[band_cols.max()]))
                          if len(band_cols) else None)
                all_stats[key] = dict(
                    y_ext=y_ext, z_solid=zr, z_band=z_band,
                    n_perp=int(g_perp.sum()), n_td=int(g_td.sum()), n_tu=int(g_tu.sum()))
                print(f"[{key:16s}] 垂直保证区 y[{y_ext[0]:+.2f},{y_ext[1]:+.2f}] "
                      f"z实心[{zr[0]:.2f},{zr[1]:.2f}] 段带内z[{z_band[0]:.2f},{z_band[1]:.2f}] "
                      f"格数 perp={g_perp.sum()} td={g_td.sum()} tu={g_tu.sum()}")
                saved[f"{key}_perp"] = g_perp
                saved[f"{key}_tiltd"] = g_td
                saved[f"{key}_tiltu"] = g_tu
                saved[f"{key}_qsub"] = r['q_sub'].astype(np.float32)
                saved[f"{key}_endsub"] = r['end_sub'].astype(np.float32)

    saved["ys"] = Y_CEN
    saved["zs"] = Z_CEN
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "tiangong_workspace.npz"), **saved)
    with open(os.path.join(OUT, "tiangong_workspace_stats.json"), "w", encoding="utf8") as f:
        import json
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

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
                ax.axvspan(*SLICE_BAND, color="gold", alpha=0.10)
                ax.axhspan(0, 2.0, color="gray", alpha=0.08)
                ax.set_xlim(*Y_RNG)
                ax.set_ylim(*Z_RNG)
                ax.set_aspect("equal")
                ax.set_title(f"天工3 {mode} 腰±{yaw_deg}° {arm_name}臂\n"
                             f"蓝:垂直 绿:下倾30° 橙:上倾30°")
                ax.set_xlabel("y (m)")
                ax.set_ylabel("z (m)")
                ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_tiangong_workspace.png"), dpi=110)
    print("[预览]", os.path.join(OUT, "fig_tiangong_workspace.png"))


if __name__ == "__main__":
    main()
