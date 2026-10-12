# -*- coding: utf-8 -*-
"""执行层任务空间伺服: 按实际底座位姿闭环补偿 腰3+右臂7 注入目标.

背景 (2026-10-10 实测诊断): 规划/动作按"底座在原点、姿态竖直"解算关节角, 但执行期
底座由 RL 策略控制, 实际会漂 —— 站姿带 yaw 缓慢漂移(中位 +22°, 最大 +40°), 深蹲段
骨盆前倾 +18°. 开环注入下腕心世界误差被 tan(枪倾角) 放大: 站姿行整体低 0.12m,
蹲姿行(倾角60°)低 0.3~0.47m, 墙面 0.34~0.61m 完全漏喷.

原理: 每个策略步 (50Hz) 用解析 FK/雅可比 (与 03 的 CHAIN 完全一致) 在实际底座位姿下
求 [腕心位置(3); 枪轴(3)] 残差, DLS 求腰+臂 10 关节修正量, 累加到动作帧目标上
(积分伺服, 修正量限幅), 再交给原 PD 通道. 关节限位内裁剪, 播放起点清零.

依赖: 仅 numpy. 独立于 03 (runner 在 GR00T venv 内运行, 不 import 规划脚本);
CHAIN 常量与 03_plan_trajectories.py 必须保持一致 (修改臂型时两处同步).
"""
import numpy as np

# ---- G1 腰3 + 右臂7 运动链 (与 03 一致; 元素: 平移, 固定rpy, 旋转轴, 下限, 上限) ----
_WRP = float(__import__("os").environ.get("G1W_WAIST_RP", "0") or 0) or 0.0872665
WAIST = [
    ((0.0, 0.0, 0.0), (0, 0, 0), 'z', -2.618, 2.618),
    ((-0.0039635, 0, 0.035), (0, 0, 0), 'x', -_WRP, _WRP),
    ((0.0, 0.0, 0.019), (0, 0, 0), 'y', -_WRP, _WRP),
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
JLO = np.array([j[3] for j in CHAIN])
JHI = np.array([j[4] for j in CHAIN])

_AX = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}


def _rot(axis, a):
    """单一角度旋转矩阵 (axis: 'x'|'y'|'z')."""
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def quat2rotm(q):
    """wxyz 四元数 -> 3x3 (MuJoCo qpos[3:7] 顺序)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def fk_jac1(q10):
    """单构型 FK + 雅可比. 返回 (p_rel(3), axis(3), Jp(3,10), Ja(3,10)),
    均为骨盆系(根平移0): p_rel[2] 已含 pz=0 约定, 世界量由调用方拼底座."""
    p = np.zeros(3)
    R = np.eye(3)
    orgs = np.empty((10, 3))
    axv = np.empty((10, 3))
    for i, (xyz, rpy, ax, lo, hi) in enumerate(CHAIN):
        p = p + R @ np.asarray(xyz)
        Rf = R @ (_rot("x", rpy[0]) @ _rot("y", rpy[1]) @ _rot("z", rpy[2]))
        axv[i] = Rf @ _AX[ax]
        orgs[i] = p
        R = Rf @ _rot(ax, q10[i])
    axis = R[:, 0]
    Jp = np.cross(axv, p[None, :] - orgs).T          # (3,10)
    Ja = np.cross(axv, axis[None, :]).T
    return p, axis, Jp, Ja


class ArmTaskServo:
    """积分式任务空间伺服 (50Hz 调用). 用法:
        servo = ArmTaskServo()
        # 播放起点: servo.reset()
        q10_cmd = servo.step(q10_motion, p_star, a_star, base_pos, base_quat, q10_meas)
    p_star/a_star: 规划世界系目标(底座原点竖直假设下的动作帧 FK, 由调用方预算);
    base_pos/base_quat: 实测底座; q10_meas: 实测腰+臂关节.

    关节权限(实测教训 2026-10-10): 腰 yaw 完全禁用 —— 伺服用它追 y 向误差时,
    上身旋转的反作用会把骨盆带偏, 形成正反馈(实测机器人来回甩到 ±114°, dq 顶满);
    腰 roll/pitch 轻权限(它们的反作用弱), 主力 = 右臂 7 关节.
    """

    #            腰yaw 腰roll 腰pitch  肩p 肩r 肩y 肘 腕r 腕p 腕y
    COL_SCALE = (0.0,  0.3,  0.3,    1,  1,  1,  1,  1,  1,  1)

    def __init__(self, lam=0.10, dq_step=0.03, dq_total=0.35, gain=1.0,
                 trim_every=25, trim_gain=0.12, trim_step=0.02, trim_total=0.15):
        self.lam = lam            # DLS 阻尼
        self.dq_step = dq_step    # 单步修正限幅 (rad, 50Hz -> 最大 1.5 rad/s)
        self.dq_total = dq_total  # 累计修正限幅 (rad)
        self.gain = gain
        self.col = np.asarray(self.COL_SCALE, float)
        # 慢速实测微调环: 快环(投影命令)不补偿关节 PD 重力下垂(实测腕部 ~5cm 慢性
        # 偏低), 用实测关节每 trim_every 步(0.5s)小幅积分收回 —— 带宽 ~0.1Hz,
        # 远低于任何振荡模式, 不会再激起手部抖动(快环实测关节反馈曾致 6Hz 极限环)
        self.trim_every = trim_every
        self.trim_gain = trim_gain
        self.trim_step = trim_step
        self.trim_total = trim_total
        self.reset()

    def reset(self):
        self.dq = np.zeros(10)
        self.dq_slow = np.zeros(10)
        self._k = 0
        self.err_p = self.err_a = float("nan")

    def step(self, q10_motion, p_star, a_star, base_pos, base_quat, q10_meas=None):
        """双环解耦(2026-10-10 两轮实测教训的最终结构):
        快环: 残差在**投影命令**(动作帧+dq)上计算 —— 模型前馈性质, 与 PD 跟踪
        延迟解耦(实测关节的快反馈曾与 PD 耦合成 ~6Hz 极限环, 手部可见抖动);
        dq 吃掉底座位姿漂移. 慢环: 实测关节残差每 0.5s 低增益积分出 dq_slow
        (关节 PD 重力下垂等慢性偏差, 实测 ~5cm), **只叠加进最终注入命令, 不进
        快环残差** —— 否则两个积分器在同一命令上互相抵消(v5 曾因此反而更差).
        """
        q10_proj = np.clip(q10_motion + self.dq, JLO, JHI)
        Rb = quat2rotm(base_quat)
        # 慢环: 实测残差 = 快环残差 + PD下垂; 快环收敛后只剩下垂分量
        if q10_meas is not None:
            self._k += 1
            if self._k % self.trim_every == 0:
                pm, am, _, _ = fk_jac1(np.clip(np.asarray(q10_meas, float), JLO, JHI))
                em = np.concatenate(
                    [p_star - (Rb @ pm + np.asarray(base_pos)),
                     0.5 * (a_star - Rb @ am)]) * self.trim_gain
                _, _, Jp2, Ja2 = fk_jac1(q10_proj)
                J2 = np.vstack([Rb @ Jp2, Rb @ Ja2]) * self.col
                dq_t = J2.T @ np.linalg.solve(
                    J2 @ J2.T + (self.lam ** 2) * np.eye(6), em)
                np.clip(dq_t, -self.trim_step, self.trim_step, out=dq_t)
                self.dq_slow = np.clip(self.dq_slow + dq_t,
                                       -self.trim_total, self.trim_total)
        p_rel, a_rel, Jp, Ja = fk_jac1(q10_proj)
        p_w = Rb @ p_rel + np.asarray(base_pos)
        a_w = Rb @ a_rel
        e = np.concatenate([p_star - p_w, 0.5 * (a_star - a_w)]) * self.gain
        self.err_p, self.err_a = float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
        if self.err_p < 0.003 and self.err_a < 0.01:
            return np.clip(q10_motion + self.dq + self.dq_slow, JLO, JHI)
        # 世界系雅可比: 位置列旋到世界; 轴向同旋 (轴是方向量); 按关节权限缩列
        J = np.vstack([Rb @ Jp, Rb @ Ja]) * self.col    # (6,10)
        JT = J.T
        dq = JT @ np.linalg.solve(J @ JT + (self.lam ** 2) * np.eye(6), e)
        np.clip(dq, -self.dq_step, self.dq_step, out=dq)
        self.dq = np.clip(self.dq + dq, -self.dq_total, self.dq_total)
        return np.clip(q10_motion + self.dq + self.dq_slow, JLO, JHI)


def precompute_motion_targets(motion_qpos, wall_x=0.74, d_wrist=0.40):
    """预算动作每帧的 规划世界系 (腕心p*, 枪轴a*, 落点高度h*).
    规划假设底座 xy=0, 竖直, pz=motion[root_z] —— 与 03/05 同口径.
    返回 (P*, A*, H*) 三个 (N,3)/(N,3)/(N,) 数组; H* 供执行后落点误差统计."""
    N = motion_qpos.shape[0]
    P = np.empty((N, 3)); A = np.empty((N, 3)); H = np.empty(N)
    for i in range(N):
        q10 = np.concatenate([motion_qpos[i, 19:22], motion_qpos[i, 29:36]])
        p_rel, a_rel, _, _ = fk_jac1(q10)
        P[i] = p_rel + np.array([0.0, 0.0, motion_qpos[i, 2]])
        A[i] = a_rel
        if a_rel[0] > 0.3:
            H[i] = P[i, 2] + (wall_x - P[i, 0]) * a_rel[2] / a_rel[0]
        else:
            H[i] = np.nan
    return P, A, H
