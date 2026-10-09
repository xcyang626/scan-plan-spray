# -*- coding: utf-8 -*-
"""
数值逆运动学（阶段3/4 共用，纯 numpy，无 ROS/Isaac 依赖）
==========================================================
DLS（阻尼最小二乘）+ 关节限位裁剪 + 接近轴约束（默认 5D：位置 3D +
TCP 接近轴对准目标方向，绕轴自旋放开）+ warm start。

雅可比用有限差分（8 次 FK/迭代），FK 由 kinematics.UrdfChain 提供。
"""

import math

import numpy as np

_Z = np.array([0.0, 0.0, 1.0])


class DlsIK:

    def __init__(self, chain, limits_rad, tcp_offset_m=0.0,
                 damping=0.08, pos_tol=1e-3, axis_tol_deg=2.0,
                 max_iters=80, max_step=0.25, fd_h=1e-5):
        """limits_rad: (7,2) URDF 系关节限位；tcp_offset_m: 指尖偏移。"""
        self.chain = chain
        self.limits = np.asarray(limits_rad, dtype=float)
        self.tcp_off = tcp_offset_m
        self.damping = damping
        self.pos_tol = pos_tol
        self.axis_tol = axis_tol_deg * math.pi / 180.0
        self.max_iters = max_iters
        self.max_step = max_step
        self.h = fd_h

    # ------------------------------------------------------------------

    def _pose(self, q):
        """当前位形的 TCP 位姿: 返回(位置, 接近轴单位向量)。"""
        R, p = self.chain.fk(q, self.tcp_off)
        return p, R @ _Z  # 接近轴 = 旋转矩阵第 3 列(URDF Z 轴旋到 TCP 系)

    def _error(self, q, target_pos, target_axis):
        """6D 残差: [目标位置−当前位置, 目标轴−当前轴]（与雅可比同构、符号一致）。"""
        p, axis = self._pose(q)
        e_pos = target_pos - p
        if target_axis is None:
            return np.concatenate([e_pos, np.zeros(3)]), p, axis
        return np.concatenate([e_pos, target_axis - axis]), p, axis

    def _jacobian(self, q, target_axis):
        """有限差分雅可比(每次迭代 8 次 FK): 上 3 行位置、下 3 行接近轴,
        行序与 _error 残差一致, 保证最小二乘方向正确。"""
        J = np.zeros((6, len(q)))
        p0, a0 = self._pose(q)
        for k in range(len(q)):
            q2 = q.copy()
            q2[k] += self.h
            p2, a2 = self._pose(q2)
            J[:3, k] = (p2 - p0) / self.h
            if target_axis is not None:
                J[3:, k] = (a2 - a0) / self.h
        return J

    # ------------------------------------------------------------------

    def solve(self, q0, target_pos, target_axis=None):
        """从 q0 迭代到 target_pos（target_axis: 接近轴期望方向，可 None）。
        返回 (q, converged, iters)。"""
        q = np.clip(np.asarray(q0, dtype=float),
                    self.limits[:, 0], self.limits[:, 1])
        target_pos = np.asarray(target_pos, dtype=float)
        if target_axis is not None:
            target_axis = np.asarray(target_axis, dtype=float)
            target_axis = target_axis / np.linalg.norm(target_axis)

        for it in range(1, self.max_iters + 1):
            e, p, axis = self._error(q, target_pos, target_axis)
            pos_err = np.linalg.norm(e[:3])
            ax_err = (0.0 if target_axis is None
                      else math.acos(max(-1.0, min(1.0,
                          float(np.dot(axis, target_axis))))))
            if pos_err < self.pos_tol and ax_err < self.axis_tol:
                return q, True, it

            # DLS 核心步: 解 (J·Jᵀ + λ²I)·Δ = Jᵀ·e,
            # 阻尼项 λ² 在奇异位形附近压制过大的关节步长
            J = self._jacobian(q, target_axis)
            JJt = J @ J.T + (self.damping ** 2) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            # 步长限幅（防奇异附近关节速度尖峰）
            mx = np.max(np.abs(dq))
            if mx > self.max_step:
                dq *= self.max_step / mx
            q = np.clip(q + dq, self.limits[:, 0], self.limits[:, 1])

        # 迭代耗尽: 用最终位形复核一次收敛判据(供调用方决定是否接受)
        p, axis = self._pose(q)
        pos_err = np.linalg.norm(p - target_pos)
        ax_err = (0.0 if target_axis is None
                  else math.acos(max(-1.0, min(1.0,
                      float(np.dot(axis, target_axis))))))
        converged = pos_err < self.pos_tol and ax_err < self.axis_tol
        return q, converged, self.max_iters
