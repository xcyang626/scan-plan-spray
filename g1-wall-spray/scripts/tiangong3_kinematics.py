# -*- coding: utf-8 -*-
"""天工3 (Tiangong3) 运动学表提取 —— 从 URDF 程序化解析, 输出 02 工作区脚本的元组格式.

链: 腰3 (pelvis→waist_yaw→roll→pitch) + 右/左臂7 (waist_pitch→shoulder→…→wrist_roll),
末端 tcp 为 fixed 关节(枪装夹参考). 腿用右/左腿6 (骨盆高度FK).
URDF 轴全部 x/y/z 对齐; rpy 只保留**绝对值最大的那个分量**(hip_pitch/waist_yaw 上
0.0013 rad 的 z 寄生分量对 waist_yaw 反而是主分量, 会被保留, 见 _dominant_rpy).

另含天工腿的**真实 URDF 运动学模型**(leg_ik/leg_fk): 05 真机导出用它做
"给定骨盆高 -> 腿关节角 -> FK 回骨盆高" 的闭环(误差 < 5mm), 不用自研平面二连杆.

用法: python tiangong3_kinematics.py   # 打印全部表 + 站姿骨盆/腕距 + 腿可达范围
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
    """URDF rpy -> 02 元组用的固定旋转; 只保留**绝对值最大的那个分量**,
    其余忽略 —— 天工 URDF 实际只有单分量 rpy + 0.0013 级寄生分量.
    注意: 保留的是绝对值最大者, 不是"人为认定的主分量": 例如 waist_yaw 的
    rpy=(0,0,0.0013415) 里这个 0.0013 的 z 分量本身最大, 故被保留(0.077°, 无害)."""
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
LEG_L = _chain(_root, ["hip_pitch_l_joint", "hip_roll_l_joint", "hip_yaw_l_joint",
                       "knee_pitch_l_joint", "ankle_pitch_l_joint",
                       "ankle_roll_l_joint"])
TCP_R = _f(_root.find("joint[@name='right_tcp_joint']/origin").get("xyz") or "0 0 0")

# 腰 roll/pitch 限位收紧(与 G1 同思路: 稳定裕度), 由环境变量覆盖
_WRP = float(os.environ.get("G1W_WAIST_RP", "0"))
if _WRP > 0:
    WAIST = [((xyz, rpy, ax,
               -min(_WRP, hi - 0.001) if lo < 0 else lo,
               min(_WRP, hi - 0.001) if hi > 0 else hi)
              if ax in ("x", "y") else (xyz, rpy, ax, lo, hi))
             for (xyz, rpy, ax, lo, hi) in WAIST]

# 踝铰链到脚底的距离.
# 实测: 天工 URDF 的脚部**只有视觉网格、没有碰撞体**, 网格 ankle_roll_r_link.STL
# 的最低点在踝系 z=-0.05677 -> 脚底距踝 56.8mm.
# 旧值 0.075 是"由脚碰撞体估"的, 那个碰撞体并不存在 -> 导出动作的脚恒悬空
# 18.2mm(站姿 1.00m 时最明显), 真机上表现为机器人比计划高 18mm / 落点偏高.
SOLE_TO_ANKLE = 0.0568      # ankle_roll 系到脚底(实测脚部视觉网格最低点)


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
    """名义腕距: 肩前平举姿态下 腰0+臂 的腕心**前向距离**(x, 机器人朝 +x).
    肩前平举 = shoulder_pitch -90°(臂从下垂转前平) + 肘微曲.
    注意符号: 天工 URDF 里 shoulder_pitch 正向是**向后**摆(实测 +90° 时腕心
    x=-0.510, 手臂后摆), 要让腕心前伸必须取 -90°(x=+0.510)."""
    q = np.zeros(10)
    q[0] = 0.0                       # waist_yaw=0
    q[3] = -np.pi / 2                # shoulder_pitch_r: 下垂->前平(负向才是前)
    q[4] = 0.2                       # shoulder_roll 微开
    p, R = fk_end(WAIST + ARM_R, q)
    return float(p[0]), float(p[2])


# ================= 腿模型(真实 URDF 链) =================
# 天工腿的 hip_pitch 轴在 URDF 里带 10° 固定倾斜(rpy), hip_roll 把它扳回来, 所以
# "腿平面"不是纯矢状面(大腿前摆时带侧向分量). 这里把腿链**解析降维**成两个自由度
# (q0=hip_pitch, q3=knee_pitch):
#   踝位(骨盆系) = p2(q0) + R1(q0)·v(q3)
#   R1 = Rx(tilt)·Ry(q0)·Rx(u),  u = q1 - tilt
#   p2 = hip_pitch 原点 + Rx(tilt)·Ry(q0)·(hip_roll 原点)   (随 q0 变)
#   v  = (vx0 + l2·sin q3, 0, vz0 + l2·cos q3)              (腿平面内)
# 脚掌平贴地面(脚底触地)对 u 与 beta=q3+q4 有解析解, 于是只剩 2 个残差
# (侧向锁定 + 骨盆高) 对 2 个未知量 —— 牛顿法即可收敛到 1e-14.
# 用途: 05 真机导出的"给定骨盆高 -> 腿关节角 -> FK 回骨盆高"闭环(误差 < 5mm).
_LEG_TILT = LEG_R[0][1][0]                 # hip_pitch 的 rpy x 分量(URDF 倾斜轴)
_LEG_CA, _LEG_SA = np.cos(_LEG_TILT), np.sin(_LEG_TILT)
_LEG_P1 = np.array(LEG_R[0][0])            # hip_pitch 原点(骨盆系)
_LEG_DR = np.array(LEG_R[1][0])            # hip_roll 原点(在 hip_pitch 角后的系里)
_LEG_VX0 = LEG_R[3][0][0]                  # 腿平面内 x 偏置(knee 原点 x)
_LEG_VZ0 = LEG_R[2][0][2] + LEG_R[3][0][2]   # 腿平面内 z 偏置(hip_yaw + knee)
_LEG_L2 = LEG_R[4][0][2]                   # 小腿长(踝铰链 z 偏置, 负值)


def _leg_p2(q0):
    """hip_roll 轴心在骨盆系的位置(随 hip_pitch 变)."""
    q0 = np.atleast_1d(np.asarray(q0, float))
    c0, s0 = np.cos(q0), np.sin(q0)
    px = _LEG_P1[0] + s0 * _LEG_DR[2]
    py = _LEG_P1[1] + _LEG_CA * _LEG_DR[1] - _LEG_SA * c0 * _LEG_DR[2]
    pz = _LEG_P1[2] + _LEG_SA * _LEG_DR[1] + _LEG_CA * c0 * _LEG_DR[2]
    return px, py, pz


def _leg_w(q0, q3):
    """腿降维 FK: (q0,q3) -> 踝在 R1 系下的向量 w=(wx,wy,wz) 与 D=cos(beta)."""
    q0 = np.atleast_1d(np.asarray(q0, float))
    q3 = np.atleast_1d(np.asarray(q3, float))
    c0, s0 = np.cos(q0), np.sin(q0)
    D = np.sqrt(_LEG_CA ** 2 * c0 ** 2 + _LEG_SA ** 2)
    vx = _LEG_VX0 + _LEG_L2 * np.sin(q3)
    vz = _LEG_VZ0 + _LEG_L2 * np.cos(q3)
    wx = c0 * vx + s0 * (_LEG_CA * c0 / D) * vz
    wy = _LEG_SA * s0 * (vx + _LEG_CA * s0 * vz / D)
    wz = D * vz - _LEG_CA * s0 * vx
    return wx, wy, wz, D


def _leg_ankle(q0, q3):
    """踝在骨盆系的位置(真实 URDF 链等效)."""
    px, py, pz = _leg_p2(q0)
    wx, wy, wz, _ = _leg_w(q0, q3)
    return px + wx, py + wy, pz + wz


_p0, _p1_, _p2_ = _leg_ankle(np.array([0.0]), np.array([0.0]))
LEG_X0 = float(_p0[0])                     # 站姿踝 x(骨盆系) —— 脚不前移基准
LEG_Y0 = float(_p1_[0])                    # 站姿踝 y(骨盆系) —— 侧向锁定基准
LEG_PZ_MAX = float(SOLE_TO_ANKLE - _p2_[0])   # 直腿时骨盆最大高度
LEG_PZ_MIN = 0.485                         # 实测可达下限(踝铰链到限位, 见 leg_ik 扫描)

# 站姿骨盆带(02 工作区 / 03 规划共用, 单一真源):
#   上限 = 直腿几何上限 - 12mm(留一点膝弯余量; 旧值 1.005 是按脚高 0.075 估的
#   直腿上限 1.0174 留同样余量, 脚高改实测 0.0568 后上限随之降到 0.9992,
#   站姿带必须跟着降, 否则规划出够不到的骨盆高);
#   下限 = 轻屈膝(够到底部 60° 下倾行).
PELVIS_STAND_LO = 0.96
PELVIS_STAND_TOP = float(LEG_PZ_MAX - 0.012)


def leg_fk_chain(joints, q6):
    """腿链批量 FK: q6 (N,6) -> 踝位置(骨盆系, N,3) 与脚系旋转(N,3,3)."""
    q6 = np.atleast_2d(np.asarray(q6, float))
    N = len(q6)
    R = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
    p = np.zeros((N, 3))
    for i, (xyz, rpy, ax, lo, hi) in enumerate(joints):
        p = p + np.einsum('nij,j->ni', R, np.array(xyz))
        c, s = np.cos(rpy[0]), np.sin(rpy[0])
        Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        c, s = np.cos(rpy[1]), np.sin(rpy[1])
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        c, s = np.cos(rpy[2]), np.sin(rpy[2])
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        Rr = R @ Rx @ Ry @ Rz
        a = q6[:, i]
        c, s = np.cos(a), np.sin(a)
        A = np.zeros((N, 3, 3))
        if ax == 'x':
            A[:, 0, 0] = 1; A[:, 1, 1] = c; A[:, 1, 2] = -s
            A[:, 2, 1] = s; A[:, 2, 2] = c
        elif ax == 'y':
            A[:, 0, 0] = c; A[:, 0, 2] = s; A[:, 1, 1] = 1
            A[:, 2, 0] = -s; A[:, 2, 2] = c
        else:
            A[:, 0, 0] = c; A[:, 0, 1] = -s; A[:, 1, 0] = s
            A[:, 1, 1] = c; A[:, 2, 2] = 1
        R = np.einsum('nij,njk->nik', Rr, A)
    return p, R


def leg_fk(q6):
    """右腿链 FK(校验用): q6 (N,6) -> 踝位置(骨盆系) 与脚系旋转."""
    return leg_fk_chain(LEG_R, q6)


def leg_pelvis_h_fk(q6, sole=SOLE_TO_ANKLE):
    """用真实腿链 FK 反算"脚底触地时的骨盆高"(脚掌俯仰时按脚尖/脚跟支点)."""
    p, R = leg_fk_chain(LEG_R, q6)
    xh = R[:, :, 0]
    phi = np.arctan2(-xh[:, 2], xh[:, 0])      # 脚掌俯仰(脚尖下为正)
    xc = np.where(phi >= 0, 0.17, -0.09)        # 触地点: 脚尖 / 脚跟
    h_ankle = sole * np.cos(phi) + xc * np.sin(phi)
    return h_ankle - p[:, 2]


def _leg_newton(pz, q0, q3, iters=80, tol=1e-13, max_step=0.25):
    """2x2 牛顿法解 (侧向锁定, 骨盆高) 两个残差.
    限步长 + 保留最优迭代, 避免个别 pz 牛顿发散(正弦周期性会让发散解"看起来"
    残差很小, 实际关节角已跑到 ±10rad 之外)."""
    pz = np.atleast_1d(np.asarray(pz, float))
    q0 = np.atleast_1d(np.asarray(q0, float)).copy()
    q3 = np.atleast_1d(np.asarray(q3, float)).copy()

    def resid(a0, a3):
        ax, ay, az = _leg_ankle(a0, a3)
        return ay - LEG_Y0, az - (SOLE_TO_ANKLE - pz)

    best = (q0.copy(), q3.copy(), np.full(len(pz), np.inf))
    for _ in range(iters):
        r0, r1 = resid(q0, q3)
        err = np.maximum(np.abs(r0), np.abs(r1))
        upd = err < best[2]
        if np.any(upd):
            best = (np.where(upd, q0, best[0]), np.where(upd, q3, best[1]),
                    np.where(upd, err, best[2]))
        if np.max(err) < tol:
            break
        e = 1e-7
        f0, f1 = resid(q0 + e, q3)
        g0, g1 = resid(q0, q3 + e)
        J00 = (f0 - r0) / e; J01 = (g0 - r0) / e
        J10 = (f1 - r1) / e; J11 = (g1 - r1) / e
        det = J00 * J11 - J01 * J10
        det = np.where(np.abs(det) < 1e-16, 1e-16, det)
        d0 = np.clip((-r0 * J11 + r1 * J01) / det, -max_step, max_step)
        d3 = np.clip((-r1 * J00 + r0 * J10) / det, -max_step, max_step)
        q0 = q0 + d0
        q3 = q3 + d3
    return best[0], best[1]


def leg_ik(pz):
    """真实 URDF 腿 IK: 目标骨盆高 -> 腿关节角 [hip_pitch, hip_roll,
    knee_pitch, ankle_pitch](hip_yaw / ankle_roll 恒 0).

    模型(与 05 导出、回放同源):
      - 真实 URDF 腿链的连杆长度与偏置(不是自研平面二连杆);
      - 脚掌平贴地面(脚底触地, 脚掌俯仰=0);
      - 侧向锁定: 踝的 y 保持站姿值, 双脚不外扩;
      - 分支: 取"脚踝留在身体正下方"那支(大腿前倾的深蹲支, 脚不后滑).

    返回 dict: q (N,4), h (N,) FK 回算骨盆高, ok (N,) bool, ankle_x (N,)."""
    pz = np.atleast_1d(np.asarray(pz, float))
    N = len(pz)
    lo = np.array([LEG_R[0][3], LEG_R[1][3], LEG_R[3][3], LEG_R[4][3]])
    hi = np.array([LEG_R[0][4], LEG_R[1][4], LEG_R[3][4], LEG_R[4][4]])
    d = LEG_PZ_MAX - pz
    # 初值: 站姿线性外推(大腿前倾 ~2.4 rad/m, 膝 ~4.5 rad/m)
    seeds = [(-0.07 - 2.39 * d, 0.106 + 4.48 * d),
             (0.0, 0.3), (-0.5, 1.0), (-0.9, 1.6), (-1.3, 2.4)]
    best = None
    for g0, g3 in seeds:
        q0, q3 = _leg_newton(pz, np.full(N, g0), np.full(N, g3))
        ax, ay, az = _leg_ankle(q0, q3)
        r0 = ay - LEG_Y0
        r1 = az - (SOLE_TO_ANKLE - pz)
        err = np.maximum(np.abs(r0), np.abs(r1))
        # 评分: 先要解正确且关节在限位内, 再要"脚踝留在身体下"那支
        #   (大腿前倾 q0<0 的深蹲支; 另一支是大腿垂直/脚后滑, 踝 x 差 0.3m)
        _c0 = np.cos(q0)
        _D = np.sqrt(_LEG_CA ** 2 * _c0 ** 2 + _LEG_SA ** 2)
        q1 = _LEG_TILT + np.arctan2(-_LEG_SA, _LEG_CA * _c0)
        q4 = np.arccos(np.clip(_D, -1, 1)) - q3
        qc = np.column_stack([q0, q1, q3, q4])
        lim_ok = np.all((qc >= lo - 1e-9) & (qc <= hi + 1e-9), axis=1)
        score = np.where(err < 1e-6,
                         np.where(lim_ok,
                                  np.where(q0 < -0.005, np.abs(ax - LEG_X0),
                                           1.0 + np.abs(ax - LEG_X0)),
                                  1e2),
                         1e3 + err)
        if best is None:
            best = (score, err, q0, q3)
        else:
            take = score < best[0]
            best = (np.where(take, score, best[0]),
                    np.where(take, err, best[1]),
                    np.where(take, q0, best[2]), np.where(take, q3, best[3]))
    score, err, q0, q3 = best
    c0 = np.cos(q0)
    D = np.sqrt(_LEG_CA ** 2 * c0 ** 2 + _LEG_SA ** 2)
    q1 = _LEG_TILT + np.arctan2(-_LEG_SA, _LEG_CA * c0)      # 脚掌平贴 -> u 解析解
    q4 = np.arccos(np.clip(D, -1, 1)) - q3                    # beta = q3 + q4
    q = np.column_stack([q0, q1, q3, q4])
    ax, ay, az = _leg_ankle(q0, q3)
    h = SOLE_TO_ANKLE - az
    ok = (err < 1e-6) & np.all((q >= lo - 1e-9) & (q <= hi + 1e-9), axis=1)
    return dict(q=q, h=h, ok=ok, ankle_x=ax)


if __name__ == "__main__":
    print("== 天工3 运动学表 (02 元组格式) ==")
    for nm, tbl in (("WAIST", WAIST), ("ARM_R", ARM_R), ("ARM_L", ARM_L),
                    ("LEG_R", LEG_R), ("LEG_L", LEG_L)):
        print(f"{nm}:")
        for row in tbl:
            print("   ", row)
    print("TCP_R 平移:", TCP_R)
    ps = pelvis_stand()
    dx, dz = d_wrist_nominal()
    print(f"\n站姿骨盆高度(直腿FK): {ps:.3f} m")
    print(f"肩前平举腕心: 前向 {dx:.3f} m, 高度 {dz:.3f} m (骨盆系)")
    print(f"tcp 在腕roll系下: {TCP_R}")
    print(f"\n== 腿模型(真实URDF链) ==")
    print(f"站姿踝(骨盆系) x={LEG_X0:+.4f} y={LEG_Y0:+.4f} | 骨盆高上限 {LEG_PZ_MAX:.4f} m")
    for pz in (1.005, 1.0, 0.96, 0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.48):
        r = leg_ik(np.array([pz]))
        q = r["q"][0]
        q6 = np.array([q[0], q[1], 0.0, q[2], q[3], 0.0])
        hf = float(leg_pelvis_h_fk(q6[None, :])[0])
        print(f"  pz={pz:.3f} -> hip_pitch {q[0]:+.4f} hip_roll {q[1]:+.4f} "
              f"knee {q[2]:+.4f} ankle {q[3]:+.4f} | FK骨盆高 {hf:.4f} "
              f"(误差 {(hf - pz) * 1000:+.2f} mm) 踝x {r['ankle_x'][0]:+.4f} 可行 {r['ok'][0]}")
