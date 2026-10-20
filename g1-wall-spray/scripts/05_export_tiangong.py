# -*- coding: utf-8 -*-
"""步骤5-Lite: 天工3 动作导出 (回放通道).

与 G1 的 05_export_ufo_motion(增量 MJCF-IK) 不同, 天工 v1 直接组装:
  trajectory.npz 的 q10(腰3+右臂7 路点) + 骨盆高 -> 立方样条插值到 30Hz
  -> 腿部 IK(真实 URDF 腿链: 骨盆高 -> 髋/膝/踝, 双脚平放触地)
  -> qpos38 (free7 + 腿12 + 腰3 + 头2 + 左臂7 + 右臂7, 场景模型关节序)

腿部模型说明(2026-10 修正):
  旧版用自研"平面二连杆"(LEG_DHIP/LEG_L1/LEG_L2 硬编码), 与天工 URDF 实际
  腿几何不符(骨盆高 0.80 以下偏 2~8.4cm, 且踝 x 前移 0.10m), 回放里脚浮空/穿地.
  现改用 tiangong3_kinematics.leg_ik(真实 URDF 链解析降维):
    - 脚掌平贴地面(脚底触地), 侧向锁定(脚不外扩);
    - "给定骨盆高 -> 腿关节角 -> 真实链 FK 回骨盆高" 闭环误差 < 5mm;
    - 超出腿链可达范围(踝铰链到限位)的骨盆高**显式拒绝**(收窄带宽并报警),
      不再给一个偏 8cm 的假解;
    - freejoint 的 z 写**FK 实到的骨盆高**(不是指令值), 与腿关节自洽.

用途: 网页【检查无误→生成动作】的天工分支 + 06 回放(场景=models/tiangong3_mjcf)。
执行通道(策略上机)不在本脚本范围。
用法: MOTION_VER=v.. G1W_OUT=<run目录> python 05_export_tiangong.py
输出: <run>/ufo_motion/<ver>/wall_spray_motion.npz + motion_meta.json
"""
import json
import os

import numpy as np

import tiangong3_kinematics as T3

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("G1W_OUT", os.path.join(BASE, "output"))
VER = os.environ.get("MOTION_VER", "t3_v1")
FPS = 30.0
DT_WP = float(os.environ.get("G1W_T3_DT", "0.36"))     # 每路点时间(≈G1 156s/433点口径)

# ---- 腿可达范围(由真实 URDF 链 + 脚底触地模型算出, 见 tiangong3_kinematics) ----
PZ_MAX = T3.LEG_PZ_MAX      # 直腿极限骨盆高 ≈0.9992m(脚高实测 0.0568 后)
PZ_MIN = T3.LEG_PZ_MIN      # 踝铰链到限位时的骨盆高 ≈0.485m
PZ_HARD_MAX = T3.PELVIS_STAND_TOP   # 规划带上限(站姿带 0.96~0.987, 与 03 同源)


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
    pz_raw = CubicSpline(t_wp, pz_wp)(t30)
    # 骨盆高可达性判定(不静默截断): 越界帧显式拒绝并收窄到可达带
    pz = np.clip(pz_raw, PZ_MIN, PZ_MAX)
    n_low = int((pz_raw < PZ_MIN - 1e-9).sum())
    n_high = int((pz_raw > PZ_MAX + 1e-9).sum())
    if n_low or n_high:
        print(f"[t3-05][警告] 轨迹骨盆高越界: {n_low} 帧 < {PZ_MIN:.3f}m(腿链够不到), "
              f"{n_high} 帧 > {PZ_MAX:.3f}m(直腿极限) —— 已收窄到可达带 "
              f"[{PZ_MIN:.3f},{PZ_MAX:.3f}]m, 对应行喷涂落点会偏, 请检查 03 蹲姿地板")
    if pz_raw.min() < PZ_MIN - 1e-9:
        print(f"[t3-05][警告] 轨迹骨盆低至 {pz_raw.min():.3f} < 腿链可达下限 {PZ_MIN:.3f} "
              f"— 帧被抬高, 蹲姿段落点将上漂 (03 蹲姿地板应与 05 对齐, 检查 G1W_CROUCH_LO)")
    if pz_raw.min() < 0.60 - 1e-9:
        print(f"[t3-05][警告] 轨迹骨盆低至 {pz_raw.min():.3f} < 天工 SDK 骨盆高指令范围下限 "
              f"0.60m(/hric/robot/stand_cmd 的 linear.z, 范围 0.60~0.98m) —— "
              f"真机自带控制器可能不接受该下蹲量, 需实测确认")
    if pz_raw.max() > PZ_HARD_MAX + 0.015:
        print(f"[t3-05][警告] 轨迹骨盆高至 {pz_raw.max():.3f} > 规划带上限 {PZ_HARD_MAX:.3f} "
              f"— 超站姿带, 请检查 03 的站姿带")

    # 腿 IK(真实 URDF 链): 骨盆高 -> 4 个腿关节角, 脚掌平贴地面
    from scipy.ndimage import uniform_filter1d
    pz_s = uniform_filter1d(pz, 5)          # 骨盆高轻微平滑(样条过冲), 再做 IK
    leg = T3.leg_ik(pz_s)
    qleg, ok = leg["q"], leg["ok"]
    if not np.all(ok):
        bad = np.where(~ok)[0]
        print(f"[t3-05][错误] {len(bad)} 帧腿 IK 不可达/未收敛(骨盆高 "
              f"{pz_s[bad].min():.3f}~{pz_s[bad].max():.3f}m) —— 拒绝导出这些帧")
        raise SystemExit("[t3-05] 腿 IK 不可达, 拒绝导出(不做假解)")

    # 闭环校验: 真实链 FK 回算骨盆高, 与指令骨盆高误差必须 < 5mm
    q6 = np.zeros((len(pz_s), 6))
    q6[:, 0] = qleg[:, 0]; q6[:, 1] = qleg[:, 1]
    q6[:, 3] = qleg[:, 2]; q6[:, 4] = qleg[:, 3]
    h_fk = T3.leg_pelvis_h_fk(q6)            # 脚底触地时的骨盆高
    err_mm = np.abs(h_fk - pz_s) * 1000.0
    print(f"[t3-05] 腿闭环自检: FK 骨盆高 vs 指令  最大偏差 {err_mm.max():.3f} mm "
          f"(均值 {err_mm.mean():.3f} mm), 踝 x {leg['ankle_x'].min():+.3f}~"
          f"{leg['ankle_x'].max():+.3f}m(站姿 {T3.LEG_X0:+.3f}m)")
    assert err_mm.max() < 5.0, f"腿 IK 闭环误差 {err_mm.max():.2f}mm 超过 5mm"

    T = len(t30)
    qpos = np.zeros((T, 38))
    qpos[:, 2] = h_fk                                   # 骨盆高度(freejoint z): 写 FK 实到值
    qpos[:, 3] = 1.0                                   # 四元数 w
    for off in (7, 13):                                # 左腿 7..12 / 右腿 13..18
        qpos[:, off] = qleg[:, 0]                      # hip_pitch
        qpos[:, off + 1] = qleg[:, 1] if off == 13 else -qleg[:, 1]  # hip_roll(左右镜像)
        qpos[:, off + 3] = qleg[:, 2]                  # knee_pitch
        qpos[:, off + 4] = qleg[:, 3]                  # ankle_pitch
    qpos[:, 19:22] = q10[:, 0:3]                       # 腰 yaw/roll/pitch
    qpos[:, 31:38] = q10[:, 3:10]                      # 右臂 7

    d = os.path.join(OUT, "ufo_motion", VER)
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "wall_spray_motion.npz"), qpos=qpos)
    json.dump({"robot": "tiangong3", "fps": FPS, "frames": int(T),
               "duration_s": round(T / FPS, 1),
               "export": "05_export_tiangong(v2 真实URDF腿IK)",
               "leg_model": "tiangong3_kinematics.leg_ik(真实URDF链, 脚底触地)",
               "pelvis_z_m": [round(float(pz.min()), 4), round(float(pz.max()), 4)],
               "leg_fk_pelvis_z_m": [round(float(h_fk.min()), 4),
                                     round(float(h_fk.max()), 4)],
               "leg_roundtrip_err_mm": round(float(err_mm.max()), 4),
               "leg_pz_range_m": [PZ_MIN, PZ_MAX],
               "pelvis_clipped_frames": [n_low, n_high]},
              open(os.path.join(d, "motion_meta.json"), "w", encoding="utf8"),
              ensure_ascii=False, indent=1)
    print(f"[t3-05] 天工动作: {T} 帧 ({T/FPS:.1f}s), 骨盆指令 [{pz.min():.3f},{pz.max():.3f}]m "
          f"FK实到 [{h_fk.min():.3f},{h_fk.max():.3f}]m")
    print(f"[t3-05] 输出 {d}/wall_spray_motion.npz")
    try:                                    # 真机指令流(关节角 CSV+对接元信息), 失败不影响回放
        import importlib.util as _su
        _sp = _su.spec_from_file_location(
            "t3real", os.path.join(BASE, "scripts", "07_export_tiangong_real.py"))
        _m = _su.module_from_spec(_sp); _sp.loader.exec_module(_m)
        # 显式 NO_BLEND: 执行器 config 自带 prepare/restore 插值段, 再叠一层渐入渐出
        # 会让首尾各 60 帧变成全零位斜坡(与执行器的插值重复), 08 的 hdf5 也按
        # NO_BLEND 原始轨迹吃 —— 这里必须与 07 __main__ 的口径一致.
        _m.export(os.path.join(d, "wall_spray_motion.npz"), blend_s=0.0)
    except Exception as _e:
        print(f"[t3-05] 真机导出跳过: {_e}")


if __name__ == "__main__":
    main()
