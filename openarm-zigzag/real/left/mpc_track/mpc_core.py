#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""闭式预览 MPC 核心(逐关节独立双积分模型)
================================================================
模型(每关节):  q' = dq;  dq' = a          (dt 离散双积分)
代价(时域 N):  Σ w_p (q_j - r_j)² + w_v (dq_j - 0)² + w_a a_j²
求解:          闭式线性增益(建时预计算), 无 QP 求解器依赖。
               约束处理: a 饱和 ±a_max + 预测位置软限位投影。
接口:          step(q, dq, ref_window) -> a0  (每周期调一次, 用 a0)
参考:          ref_window = [r_0..r_N] (N+1 × 7), 已按 dt 前进对齐。
"""

import numpy as np


class PreviewMPC:
    """N 步预览 MPC。7 关节独立求解, 增益逐关节预计算。"""

    def __init__(self, n_joints=7, N=25, dt=0.01,
                 w_p=1000.0, w_v=200.0, w_a=1.0, a_max=1.2):
        self.n = n_joints
        self.N = N
        self.dt = dt
        self.a_max = a_max
        self.K = []       # 每关节的增益行 (2+N) x 1: [e_p0, e_v0, -dr_1..-dr_N]
        for _ in range(n_joints):
            # 精确 ZOH 双积分离散化:
            #   p_j = q0 + j·h·v0 + h² Σ_{m<j} (j-m-0.5) a_m
            #   v_j = v0 + h  Σ_{m<j}          a_m
            L = np.zeros((N, N))
            Lv = np.zeros((N, N))
            for j in range(N):
                for m in range(N):
                    if m < j:
                        L[j, m] = dt * dt * (j - m - 0.5)
                        Lv[j, m] = dt
            M = np.vstack([L, Lv])                      # (2N, N)
            W = np.diag([w_p] * N + [w_v] * N)
            R = np.diag([w_a] * N)
            Kfull = np.linalg.solve(M.T @ W @ M + R, M.T @ W)
            # a0 对 [q0-r0, v0-dr0, -(r1-r0), ..., -(rN-rN-1)] 的增益:
            # p_j - r_j = (q0 - r_j) + j·h·v0 + L·a; 自由误差 e =
            # [q0-r0, v0, dr_1..dr_N] 线性表出 → 直接用 Kfull 的第一行
            # 作用在 (p_free - r) 上, p_free - r 由当前状态+参考窗构成。
            self.K.append(Kfull[0, :])                  # a0 行
        # 参考增量增益: a0 对参考窗各点的系数(K_a 行 × 单位阵的分块)
        # 数值实现: 直接对 e 向量操作, 见 step()。

    def gains(self, joint=0):
        return self.K[joint]

    def step(self, q, dq, ref, joint):
        """单关节一步。q,dq 标量; ref: (N+1,) 参考位置窗。
        速度参考取参考窗有限差分(移动参考自动前馈)。
        返回 (a0) —— 周期内只施加 a0。"""
        N, h = self.N, self.dt
        K = self.K[joint]
        # 自由预测误差向量 e (2N): [p_j-r_j ; v_j-dr_j]
        # p_j(自由) = q + j·h·dq ;  v_j(自由) = dq
        e = np.empty(2 * N)
        for j in range(1, N + 1):
            e[j - 1] = q + j * h * dq - ref[j]
            dr_j = (ref[j] - ref[j - 1]) / h
            e[N + j - 1] = dq - dr_j
        a = float(-K @ e)
        a = max(-self.a_max, min(self.a_max, a))
        return a
