# -*- coding: utf-8 -*-
"""步骤7: 天工3 真机动作导出 (从回放 qpos38 提取关节指令流).

回放通道的 wall_spray_motion.npz 是 MuJoCo qpos38(含 freejoint, 世界位姿) —— 真机
不需要 freejoint: 骨盆高度已由腿部关节实现. 本脚本抽出 31 个铰链关节(URDF 命名,
rad), 做三件真机安全处理:
  1) 限位钳制: 05 立方样条在路点间小幅过冲(实测 shoulder_yaw ~0.24rad), 全帧 clip;
  2) 限速重定时(帧级): 关节大跳(支切换~3rad)的帧对按 v_max 拉长时间, 消 >10rad/s 尖峰;
  3) 首尾 BLEND_S 秒从全零位线性渐入/渐出到动作首末帧(防直接跳第一帧猛跳).
输出(与 npz 同目录):
  tiangong3_motion_real.csv   首列 t_s, 后 31 列=关节名(rad), 按时间升序
  tiangong3_motion_real.json  对接元信息: 关节清单/单位/帧率/时长/限位/速度峰值
用法: G1W_OUT=<run目录> MOTION_VER=v.. python 07_export_tiangong_real.py
      G1W_T3_NO_BLEND=1 关渐入渐出; G1W_T3_BLEND_S=2 渐变时长; G1W_T3_VMAX=2.5 限速
"""
import json
import os

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("G1W_OUT", os.path.join(BASE, "output"))
VER = os.environ.get("MOTION_VER", "t3_v1")
FPS = 30.0
BLEND_S = float(os.environ.get("G1W_T3_BLEND_S", "2.0"))
NO_BLEND = os.environ.get("G1W_T3_NO_BLEND", "") not in ("", "0")
V_MAX = float(os.environ.get("G1W_T3_VMAX", "2.5"))   # 关节速度上限(rad/s)

# MJCF/URDF 铰链序(qpos[7:38]); 名称=URDF joint name
JOINTS = [
    "hip_pitch_l_joint", "hip_roll_l_joint", "hip_yaw_l_joint",
    "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
    "hip_pitch_r_joint", "hip_roll_r_joint", "hip_yaw_r_joint",
    "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "head_yaw_joint", "head_pitch_joint",
    "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint", "elbow_yaw_l_joint",
    "wrist_pitch_l_joint", "wrist_roll_l_joint",
    "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint", "elbow_yaw_r_joint",
    "wrist_pitch_r_joint", "wrist_roll_r_joint",
]


def _load_limits():
    """关节限位 {name: (lo,hi)}: MuJoCo 模型优先, 退回 URDF 解析(不依赖 mujoco)."""
    try:
        import mujoco
        mm = mujoco.MjModel.from_xml_path(
            os.path.join(BASE, "models", "tiangong3_mjcf", "tiangong3_scene.xml"))
        return {mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_JOINT, j):
                (float(mm.jnt_range[j][0]), float(mm.jnt_range[j][1]))
                for j in range(mm.njnt) if mm.jnt_type[j] == 3}
    except Exception:
        import xml.etree.ElementTree as ET
        cands = [os.path.join(os.path.expanduser("~"), "桌面", "tiangong3",
                              "tiangong3_web", "tiangong3.urdf"),
            os.path.join(BASE, "..", "tiangong3", "tiangong3_web", "tiangong3.urdf"),
            os.path.join(BASE, "models", "tiangong3_mjcf", "tiangong3.urdf")]
        urdf = next((c for c in cands if os.path.exists(c)), None)
        if urdf is None:
            return {}
        out = {}
        for j in ET.parse(urdf).getroot().iter("joint"):
            if j.get("type") != "revolute":
                continue
            lm = j.find("limit")
            out[j.get("name")] = (float(lm.get("lower")), float(lm.get("upper")))
        return out


UPPER_IDX = list(range(12, 31))         # 腰3+头2+左臂7+右臂7 (JOINTS 下标)


def export(npz_path, blend_s=BLEND_S, v_max=V_MAX):
    m = np.load(npz_path)
    qpos = m["qpos"].astype(float)
    q = qpos[:, 7:].copy()                        # (N,31) 铰链关节
    pz = qpos[:, 2:3].copy()                      # (N,1) 骨盆高度(米, 随关节同步重定时)
    lim = _load_limits()
    assert lim, "关节限位表为空(mujoco 与 URDF 均不可用), 拒绝导出"

    # 1) 限位钳制(样条过冲)
    n_clip = 0
    for name, (lo, hi) in lim.items():
        if name not in JOINTS:
            continue
        k = JOINTS.index(name)
        qc = np.clip(q[:, k], lo, hi)
        n_clip += int((qc != q[:, k]).sum())
        q[:, k] = qc

    # 2) 帧级限速重定时: 正常帧 1/30s, 大跳帧对按 v_max 拉时; 重采样回均匀 30Hz
    qp = np.hstack([q, pz])                       # 32列: 31关节 + 骨盆高(末列, 米)
    dt = np.full(len(qp), 1.0 / FPS)
    dt[1:] = np.maximum(dt[1:], np.abs(np.diff(qp[:, :31], axis=0)).max(1) / v_max)
    t_src = np.concatenate([[0.0], np.cumsum(dt)[:-1]])   # 帧时刻=前缀和(去尾, 与q等长)
    t_new = np.arange(0, t_src[-1], 1.0 / FPS)
    # 线性重采样(非样条): 样条在拉时区边界会过冲, 重新顶破限位/速度上限
    idx = np.searchsorted(t_src, t_new, side="right") - 1
    idx = np.clip(idx, 0, len(t_src) - 2)
    w = ((t_new - t_src[idx]) / (t_src[idx + 1] - t_src[idx]))[:, None]
    qp = qp[idx] * (1 - w) + qp[idx + 1] * w
    t = t_new
    for name, (lo, hi) in lim.items():        # 二次钳制(重采样残余)
        if name in JOINTS:
            k = JOINTS.index(name)
            qp[:, k] = np.clip(qp[:, k], lo, hi)

    # 3) 首尾渐入渐出(全零位 <-> 动作首末帧); 骨盆高渐入到首帧值(零位骨盆=首帧站立高)
    if blend_s > 0:
        nb = int(round(blend_s * FPS))
        ramp = np.linspace(0.0, 1.0, nb)[:, None]
        qp0 = qp[0].copy(); qp0[31] = qp[0, 31]
        qp1 = qp[-1].copy(); qp1[31] = qp[-1, 31]
        qp = np.vstack([qp0 * ramp + (1 - ramp) * np.hstack([np.zeros(31), [qp[0, 31]]]),
                        qp,
                        qp1 * ramp[::-1] + (1 - ramp[::-1]) * np.hstack([np.zeros(31), [qp[-1, 31]]])])
        t = np.arange(len(qp)) / FPS
    q, pz = qp[:, :31], qp[:, 31:]

    # 自检报告: 限位/速度
    over = [n for n in JOINTS if n in lim
            and (q[:, JOINTS.index(n)].min() < lim[n][0] - 1e-6
                 or q[:, JOINTS.index(n)].max() > lim[n][1] + 1e-6)]
    dq = np.abs(np.diff(q, axis=0)) * FPS
    vmax_arr = dq.max(0)

    out_csv = os.path.join(os.path.dirname(npz_path), "tiangong3_motion_real.csv")
    with open(out_csv, "w", encoding="utf8") as f:
        f.write("t_s," + ",".join(JOINTS) + "\n")
        for i in range(len(q)):
            f.write(",".join(f"{v:.6f}" for v in [t[i], *q[i]]) + "\n")
    # 上身通道(天工自带整机控制: 平衡/下蹲归控制器, 只下发腰/头/双臂 + 骨盆高参考)
    up_names = [JOINTS[k] for k in UPPER_IDX]
    out_up = os.path.join(os.path.dirname(npz_path), "tiangong3_motion_upper.csv")
    with open(out_up, "w", encoding="utf8") as f:
        f.write("t_s," + ",".join(up_names) + ",pelvis_z_m\n")
        for i in range(len(q)):
            row = [t[i], *q[i][UPPER_IDX], pz[i, 0]]
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")
    meta = dict(
        robot="tiangong3", fps=FPS, n_frames=int(len(q)),
        duration_s=round(float(t[-1]), 2), units="rad",
        joint_order=JOINTS, n_joints=len(JOINTS),
        retiming_v_max=v_max,
        blend_s=blend_s, blend_note=("首尾各 %.1fs 从全零位线性渐入/渐出到动作首末帧"
                                     "(G1W_T3_NO_BLEND=1 可关)" % blend_s),
        frames_limit_clipped=n_clip,
        joints_limit_over=over,
        joint_max_vel_rad_s={n: round(float(v), 2)
                             for n, v in zip(JOINTS, vmax_arr)},
        overall_max_vel_rad_s=round(float(vmax_arr.max()), 2),
        speed_note="关节速度为位置差分估计(重定时后); 真机首试建议 0.25~0.5x 缩放播放",
        channels=dict(
            full_body="tiangong3_motion_real.csv — 31关节全身位置流(含腿, 仅当天工控制器接受全身位置指令时用)",
            upper="tiangong3_motion_upper.csv — 腰3+头2+双臂14位置流 + pelvis_z_m 骨盆高度参考(米)。"
                  "天工自带整机控制(SONIC不可用): 平衡/下蹲交给自带控制器, 按同时刻 pelvis_z_m 跟踪下蹲,"
                  "臂按关节列走; 骨盆跟踪偏差会直接表现为腕落点漂移(±5cm 骨盆≈±5cm 落点)"),
        source=os.path.basename(npz_path))
    out_json = out_csv.replace(".csv", ".json")
    json.dump(meta, open(out_json, "w", encoding="utf8"),
              ensure_ascii=False, indent=1)
    print(f"[真机导出] {out_csv}\n[真机导出] 上身通道 {out_up}")
    print(f"[真机导出] 元信息 {out_json}")
    print(f"[真机导出] {len(q)} 帧 {t[-1]:.1f}s | 限位钳制 {n_clip} 帧次, 残余越界 "
          f"{over if over else '无'} | 关节速度峰值 {vmax_arr.max():.2f} rad/s (限 {v_max})")
    return out_csv, out_json


if __name__ == "__main__":
    p = os.path.join(OUT, "ufo_motion", VER, "wall_spray_motion.npz")
    assert os.path.exists(p), f"动作文件不存在: {p} (先跑 05, 或检查 MOTION_VER)"
    export(p, blend_s=(0.0 if NO_BLEND else BLEND_S))
