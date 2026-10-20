# -*- coding: utf-8 -*-
"""步骤5-Lite: 天工3 动作导出 (回放通道).

与 G1 的 05_export_ufo_motion(增量 MJCF-IK) 不同, 天工 v1 直接组装:
  trajectory.npz 的 q10(腰3+右臂7 路点) + 骨盆高 -> 立方样条插值到 30Hz
  -> 腿部平面 2 连杆 IK(骨盆高 -> 髋/膝/踝, 双脚平放)
  -> qpos38 (free7 + 腿12 + 腰3 + 头2 + 左臂7 + 右臂7, 场景模型关节序)

用途: 网页【检查无误→生成动作】的天工分支 + 06 回放(场景=models/tiangong3_mjcf)。
执行通道(策略上机)不在本脚本范围。
用法: MOTION_VER=v.. G1W_OUT=<run目录> python 05_export_tiangong.py
输出: <run>/ufo_motion/<ver>/wall_spray_motion.npz + motion_meta.json
"""
import json
import os

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("G1W_OUT", os.path.join(BASE, "output"))
VER = os.environ.get("MOTION_VER", "t3_v1")
FPS = 30.0
DT_WP = float(os.environ.get("G1W_T3_DT", "0.36"))     # 每路点时间(≈G1 156s/433点口径)

# ---- 天工腿平面参数(URDF: hip 串 0.2243 + 大腿 0.306 + 小腿 0.40 + 踝到脚底 0.075) ----
LEG_DHIP = 0.2243
LEG_L1 = 0.306
LEG_L2 = 0.400
SOLE = 0.075


KNEE_LIM_HI = 2.531                              # URDF 膝限位上限(平面模型硬约束)
LEG_D_MAX = float(np.sqrt(LEG_L1**2 + LEG_L2**2
                          + 2*LEG_L1*LEG_L2*np.cos(KNEE_LIM_HI)))   # ≈0.2295
PZ_FLOOR = LEG_DHIP + LEG_D_MAX + SOLE           # ≈0.5288: 平面模型膝限位下的骨盆地板
                                               # (02b 全链P2=0.426 需非平面自由度, 平面下发达不到)


def leg_ik_planar(pz):
    """骨盆高 -> (髋pitch, 膝, 踝pitch), 膝前弯分支; 踝平放=-(髋+膝).
    D 上限按膝限位反解(非伸直极限) —— 膝永远不超 URDF 限位."""
    D = float(np.clip(pz - LEG_DHIP - SOLE, LEG_D_MAX,   # 下限=膝限位(弯最多2.531)
                      LEG_L1 + LEG_L2 - 0.02))            # 上限=伸直留微弯
    # 关节角(非内角): D² = L1²+L2²+2·L1·L2·cos(膝关节角) -> cos_k = (D²-L1²-L2²)/(2L1L2)
    cos_k = (D ** 2 - LEG_L1 ** 2 - LEG_L2 ** 2) / (2 * LEG_L1 * LEG_L2)
    knee = np.arccos(np.clip(cos_k, -1, 1))
    cos_h = (LEG_L1 ** 2 + D ** 2 - LEG_L2 ** 2) / (2 * LEG_L1 * D)
    hip = -np.arccos(np.clip(cos_h, -1, 1))            # 髋前倾为负(膝盖向前)
    return hip, knee, -(hip + knee)


def leg_fk_z(hip, knee):
    """平面 FK 验算: 该(髋,膝)下的骨盆高."""
    return LEG_DHIP + LEG_L1 * np.cos(hip) + LEG_L2 * np.cos(hip + knee) + SOLE


def main():
    traj = np.load(os.path.join(OUT, "trajectory.npz"))
    q_wp, pz_wp = traj["q"], traj["pelvis"]
    n = len(q_wp)
    t_wp = np.arange(n) * DT_WP
    # 立方样条插值到 30Hz
    from scipy.interpolate import CubicSpline
    T_end = t_wp[-1]
    t30 = np.arange(0, T_end, 1.0 / FPS)
    q10 = CubicSpline(t_wp, q_wp, axis=0)(t30)
    pz = np.clip(CubicSpline(t_wp, pz_wp)(t30), PZ_FLOOR, 1.005)  # 下限=平面模型膝限位地板(~0.53), 上限直腿
    if pz_wp.min() < PZ_FLOOR - 0.02:
        print(f"[t3-05][警告] 轨迹骨盆低至 {pz_wp.min():.3f} < 平面模型地板 {PZ_FLOOR:.3f} "
              f"— 帧被抬高, 蹲姿段落点将上漂 (03 蹲姿地板应与 05 对齐, 检查 G1W_CROUCH_LO)")

    # 腿部 IK(先定分支: 髋负膝正 -> FK 复核)
    hip, knee, ank = leg_ik_planar(0.9)
    assert abs(leg_fk_z(hip, knee) - 0.9) < 0.005, "腿平面IK自检失败"

    T = len(t30)
    qpos = np.zeros((T, 38))
    qpos[:, 2] = pz                                    # 骨盆高度(freejoint z)
    qpos[:, 3] = 1.0                                   # 四元数 w
    for i in range(T):
        h, k, a = leg_ik_planar(pz[i])
        for off in (7, 13):                            # 左腿 7..12 / 右腿 13..18
            qpos[i, off] = h
            qpos[i, off + 3] = k
            qpos[i, off + 4] = a
        qpos[i, 19:22] = q10[i, 0:3]                   # 腰 yaw/roll/pitch
        qpos[i, 31:38] = q10[i, 3:10]                  # 右臂 7
    # 平滑腿部(样条端点抖动)
    from scipy.ndimage import uniform_filter1d
    for c in (7, 8, 9, 10, 11, 13, 14, 15, 16, 17):
        qpos[:, c] = uniform_filter1d(qpos[:, c], 9)

    d = os.path.join(OUT, "ufo_motion", VER)
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "wall_spray_motion.npz"), qpos=qpos)
    json.dump({"robot": "tiangong3", "fps": FPS, "frames": int(T),
               "duration_s": round(T / FPS, 1), "export": "05_export_tiangong(v1 直组装)"},
              open(os.path.join(d, "motion_meta.json"), "w", encoding="utf8"),
              ensure_ascii=False, indent=1)
    print(f"[t3-05] 天工动作: {T} 帧 ({T/FPS:.1f}s), 骨盆 [{pz.min():.3f},{pz.max():.3f}]m")
    print(f"[t3-05] 输出 {d}/wall_spray_motion.npz")
    try:                                    # 真机指令流(关节角 CSV+对接元信息), 失败不影响回放
        import importlib.util as _su
        _sp = _su.spec_from_file_location(
            "t3real", os.path.join(BASE, "scripts", "07_export_tiangong_real.py"))
        _m = _su.module_from_spec(_sp); _sp.loader.exec_module(_m)
        _m.export(os.path.join(d, "wall_spray_motion.npz"))
    except Exception as _e:
        print(f"[t3-05] 真机导出跳过: {_e}")


if __name__ == "__main__":
    main()
