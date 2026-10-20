# -*- coding: utf-8 -*-
"""天工3 (Tiangong3) 运动学表提取 —— 从 URDF 程序化解析, 输出 02 工作区脚本的元组格式.

链: 腰3 (pelvis→waist_yaw→roll→pitch) + 右/左臂7 (waist_pitch→shoulder→…→wrist_roll),
末端 tcp 为 fixed 关节(枪装夹参考). 腿用右腿6 (骨盆高度FK).
URDF 轴全部 x/y/z 对齐; rpy 仅取主分量(hip_pitch/waist_yaw 的 0.0013 z 分量 ~0.08° 忽略).

用法: python tiangong3_kinematics.py   # 打印全部表 + 站姿骨盆/腕距
      (被 02_tiangong_workspace.py import: T3.WAIST / T3.ARM_R / T3.ARM_L / T3.LEG_R / ...)
"""
import os
import xml.etree.ElementTree as ET

import numpy as np

URDF = os.environ.get(
    "T3_URDF",
    os.path.expanduser("~/桌面/tiangong3/tiangong3_urdf/urdf/tiangong3.urdf"))


def _f(s):
    return tuple(float(x) for x in s.split())


def _dominant_rpy(rpy):
    """URDF rpy -> 02 元组用的固定旋转; 仅保留主分量(角度绝对值最大的那个),
    其余(<0.18°)忽略 —— 天工 URDF 实际只有单分量 rpy + 0.0013 级寄生分量."""
    rpy = np.array(rpy, float)
    k = int(np.argmax(np.abs(rpy)))
    out = [0.0, 0.0, 0.0]
    out[k] = float(rpy[k])
    return tuple(out)


def _joint_tuple(j):
    o = j.find("origin")
    ax = j.find("axis")
    lim = j.find("limit")
    xyz = _f(o.get("xyz") or "0 0 0")
    rpy = _f(o.get("rpy") or "0 0 0")
    axis = _f(ax.get("xyz") if ax is not None else "0 0 0")
    chars = {0: "x", 1: "y", 2: "z"}
    k = int(np.argmax(np.abs(axis)))
    assert abs(axis[k]) > 0.99, f"{j.get('name')} 轴非对齐: {axis}"
    lo, hi = float(lim.get("lower")), float(lim.get("upper"))
    return (xyz, _dominant_rpy(rpy), chars[k], lo, hi)


def _chain(root, names, fixed_last=None):
    out = [_joint_tuple(root.find(f"joint[@name='{n}']")) for n in names]
    if fixed_last:                       # tcp: 只取平移
        j = root.find(f"joint[@name='{fixed_last}']")
        o = j.find("origin")
        out.append((_f(o.get("xyz") or "0 0 0"), (0, 0, 0), None, 0.0, 0.0))
    return out


_root = ET.parse(URDF).getroot()

WAIST = _chain(_root, ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"])
ARM_R = _chain(_root, ["shoulder_pitch_r_joint", "shoulder_roll_r_joint",
                       "shoulder_yaw_r_joint", "elbow_pitch_r_joint",
                       "elbow_yaw_r_joint", "wrist_pitch_r_joint",
                       "wrist_roll_r_joint"])
ARM_L = _chain(_root, ["shoulder_pitch_l_joint", "shoulder_roll_l_joint",
                       "shoulder_yaw_l_joint", "elbow_pitch_l_joint",
                       "elbow_yaw_l_joint", "wrist_pitch_l_joint",
                       "wrist_roll_l_joint"])
LEG_R = _chain(_root, ["hip_pitch_r_joint", "hip_roll_r_joint", "hip_yaw_r_joint",
                       "knee_pitch_r_joint", "ankle_pitch_r_joint",
                       "ankle_roll_r_joint"])
TCP_R = _f(_root.find("joint[@name='right_tcp_joint']/origin").get("xyz") or "0 0 0")

# 腰 roll/pitch 限位收紧(与 G1 同思路: 稳定裕度), 由环境变量覆盖
_WRP = float(os.environ.get("G1W_WAIST_RP", "0"))
if _WRP > 0:
    WAIST = [((xyz, rpy, ax,
               -min(_WRP, hi - 0.001) if lo < 0 else lo,
               min(_WRP, hi - 0.001) if hi > 0 else hi)
              if ax in ("x", "y") else (xyz, rpy, ax, lo, hi))
             for (xyz, rpy, ax, lo, hi) in WAIST]

SOLE_TO_ANKLE = 0.075       # ankle_roll 系到脚底(由脚碰撞体估)


def fk_end(joints, q, pelvis_z=0.0):
    """单姿态FK(02.fk_chain 的标量便捷版), 供标定用."""
    R = np.eye(3)
    p = np.array([0.0, 0.0, pelvis_z])
    for i, (xyz, rpy, ax, lo, hi) in enumerate(joints):
        p = p + R @ np.array(xyz)
        Rx = np.eye(3) if rpy[0] == 0 else _rotx(rpy[0])
        Ry = np.eye(3) if rpy[1] == 0 else _roty(rpy[1])
        Rz = np.eye(3) if rpy[2] == 0 else _rotz(rpy[2])
        R = R @ Rx @ Ry @ Rz
        if ax is not None:
            R = R @ {"x": _rotx, "y": _roty, "z": _rotz}[ax](q[i])
    return p, R


def _rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _roty(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def pelvis_stand():
    """直腿零位 FK 的骨盆高度(腿垂至脚底)."""
    p, _ = fk_end(LEG_R, np.zeros(6))
    return float(-p[2] + SOLE_TO_ANKLE)


def d_wrist_nominal():
    """名义腕距: 肩前平举姿态下 腰0+臂 的腕心前向距离(x).
    肩前平举 = shoulder_pitch 90°(臂从下垂转前平) + 肘微曲."""
    q = np.zeros(10)
    q[0] = 0.0                       # waist_yaw=0
    q[3] = np.pi / 2                 # shoulder_pitch_r: 下垂->前平
    q[4] = 0.2                       # shoulder_roll 微开
    p, R = fk_end(WAIST + ARM_R, q)
    return float(p[0]), float(p[2])


if __name__ == "__main__":
    print("== 天工3 运动学表 (02 元组格式) ==")
    for nm, tbl in (("WAIST", WAIST), ("ARM_R", ARM_R), ("ARM_L", ARM_L), ("LEG_R", LEG_R)):
        print(f"{nm}:")
        for row in tbl:
            print("   ", row)
    print("TCP_R 平移:", TCP_R)
    ps = pelvis_stand()
    dx, dz = d_wrist_nominal()
    print(f"\n站姿骨盆高度(直腿FK): {ps:.3f} m")
    print(f"肩前平举腕心: 前向 {dx:.3f} m, 高度 {dz:.3f} m (骨盆系)")
    print(f"tcp 在腕roll系下: {TCP_R}")
