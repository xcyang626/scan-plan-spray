#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线生成之字形稠密关节轨迹（纯 numpy，复用阶段3的 DlsIK + UrdfChain）
====================================================================
背景: MoveIt /compute_cartesian_path 锁 6D 姿态时 KDL 沿直线中途无解
（17 段里 13 段 fraction<1），而 workspace.json 的解来自 5D IK
（位置 + 接近轴, 绕轴自旋放开），6D 属过约束。

方案: 对 36 个之字形路点连线按 5mm 步长稠密采样, 用 DlsIK warm start
逐点求解（容差 pos 1mm / 轴 2°），输出整条关节轨迹 + 每步 TCP 位移，
供 zigzag_moveit.py --dense 直接交给 ExecuteTrajectory 执行。
TCP 直线度由 IK 容差保证，绕轴自旋连续漂移（对贴墙走线无影响）。

用法: /usr/bin/python3 make_dense_zigzag.py [输出json]
产出: dense_zigzag.json {points: [[t?,7关节]...], seg_lens: [...]}
"""

import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ISA_DIR = os.path.abspath(os.path.join(_HERE, "..", "isaaclab_sim"))
_PKG_DIR = os.path.abspath(os.path.join(_HERE, "..", "openarm_sim"))
sys.path.insert(0, _ISA_DIR)
sys.path.insert(0, _PKG_DIR)

from common import load_bridge, load_chain          # noqa: E402
from openarm_sim.ik_solver import DlsIK             # noqa: E402
from stage3_workspace import urdf_limits            # noqa: E402

STEP = 0.005            # 采样步长 5mm（与 GetCartesianPath max_step 一致）
WS = os.path.join(_ISA_DIR, "workspace.json")

# 真机软限位(rad, URDF 关节序)——权威来源 real/config/real_safety.yaml
# (gen_real_config.py 生成), 启动时直接读取, 不再手抄常数
# (2026-09-04 教训: 手抄两位小数与 yaml 精确值差 ~0.005°, IK 边界点越限)。
# workspace.json 的关节解经 FK 验证本就是该系(模型系)的值, 直接使用,
# 不做任何旧系平移(2026-09-04 曾错误地减 OLD_HOME 致整条轨迹平移,
# 真机首跑轨迹完全错误, 已回滚该逻辑)。
_SAFETY_YAML = os.path.abspath(os.path.join(_HERE, "..", "..",
                                            "real", "config",
                                            "real_safety.yaml"))

# watchdog 触发线 = 软限位内缩 2°, IK 解再预留 3°(2° 触发余量 +
# ~1° 跟踪噪声): 命令值绝不允许越过触发线, 否则执行数秒即被
# watchdog 取消(2026-09-04 真机实测, J5/J6 贴边界所致)。
WATCHDOG_CLEARANCE = 3.0  # deg

# 相邻 5mm 路点间臂关节(J1-J4)单步最大变化: 超过即判定 IK 换了构型支
# (elbow-up↔down / 肩翻)。换支两端虽都在路径上, 但关节空间直线插值会让
# TCP 中途甩出大弧线(实测 J2 单步 181°), 真机执行危险, 必须拒绝。
# 腕关节(J5 自旋/J7 偏航)绕 TCP 转动、J6 力臂仅 0.12m, 跳变不影响 TCP
# 走线(重定时会把翻转摊到数秒), 允许。
ARM_JUMP_DEG = 20.0
ARM_JOINTS = slice(0, 4)   # J1-J4


def _load_soft_limits():
    import yaml  # 系统自带, 仍无 ROS 依赖
    with open(_SAFETY_YAML, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    lim = cfg["soft_limits_rad"]
    names = ["openarm_left_joint%d" % i for i in range(1, 8)]
    if isinstance(lim, dict):
        arr = np.array([[float(lim[n][0]), float(lim[n][1])]
                        for n in names])
    else:
        arr = np.array([[float(l[0]), float(l[1])] for l in lim])
    c = float(np.radians(WATCHDOG_CLEARANCE))
    arr[:, 0] += c
    arr[:, 1] -= c
    return arr


def main():
    with open(WS, encoding="utf-8") as f:
        ws = json.load(f)
    wps = [np.array(w) for w in ws["zigzag_waypoints"]]
    # 解即模型系(=真机当前 URDF 系, 零位=竖直下垂, d≈0)的值, 直接使用
    sols = [np.array(q) for q in ws["zigzag_joint_solutions_rad"]]

    bridge = load_bridge()
    chain = load_chain()
    # 限位 = real_safety.yaml 真值(rad, URDF 序)
    limits = _load_soft_limits()
    # 注意: 阶段3 IK 的 tcp_offset=0.12 语义 = 路点即指尖位置, 求解的是 TCP link
    ik = DlsIK(chain, limits, tcp_offset_m=0.12)
    rng = np.random.default_rng(7)
    axis = np.array(ws.get("target_axis", [1.0, 0.0, 0.0]))

    def solve_robust(q_seed, target):
        """warm start 优先, 失败随机重启兜底(与阶段3网格扫描同策略)。
        返回 (q, ok, arm_jump): arm_jump=True 表示只搜到换支解。"""
        q, ok, _ = ik.solve(q_seed, target, target_axis=axis)
        if ok:
            return q, True, False
        best, best_jump = None, None
        for _ in range(20):
            q2, ok2, _ = ik.solve(
                rng.uniform(limits[:, 0], limits[:, 1]), target,
                target_axis=axis)
            if ok2:
                j2 = float(np.abs(q2[ARM_JOINTS] - q_seed[ARM_JOINTS]).max())
                if j2 <= math.radians(ARM_JUMP_DEG):
                    return q2, True, False
                if best is None or j2 < best_jump:
                    best, best_jump = q2, j2
        if best is not None:
            return best, True, True
        return q, False, False

    def solve_continuous(q_prev, target, where):
        """连续性优先的求解: warm start 解若臂关节跳变超阈值, 视为换支,
        随机重启找同支解; 实在只有换支解则报错终止(路径不可安全执行)。"""
        q, ok, _ = ik.solve(q_prev, target, target_axis=axis)
        if ok:
            j = float(np.abs(q[ARM_JOINTS] - q_prev[ARM_JOINTS]).max())
            if j <= math.radians(ARM_JUMP_DEG):
                return q
        q2, ok2, jumped = solve_robust(q_prev, target)
        if ok2 and not jumped:
            return q2
        if ok2:   # 只有换支解
            j = float(np.degrees(np.abs(
                q2[ARM_JOINTS] - q_prev[ARM_JOINTS]).max()))
            print("[致命] %s: 仅存在换支解(臂关节最近跳 %.1f° > %g°), "
                  "TCP 插值会甩出大弧线, 终止。请缩小矩形/行距或反馈排查"
                  % (where, j, ARM_JUMP_DEG))
            sys.exit(1)
        print("[致命] %s: IK 未收敛(含随机重启)" % where)
        sys.exit(1)

    points = [sols[0].tolist()]
    seg_lens = [0.0]
    q = np.clip(sols[0].copy(), limits[:, 0], limits[:, 1])
    for i in range(len(wps) - 1):
        a, b = wps[i], wps[i + 1]
        nsub = max(int(np.linalg.norm(b - a) / STEP), 1)
        for k in range(1, nsub + 1):
            target = a + (b - a) * (k / nsub)
            # 目标点是"指尖位置"(路点), IK 链已带 0.12 指尖偏移 → 直接传路点
            q = solve_continuous(q, target, "段 %d 步 %d/%d" % (i, k, nsub))
            points.append(q.tolist())
            seg_lens.append(float(np.linalg.norm(b - a) / nsub))
    # 终点复核: FK 到最后一点, 检查指尖是否落在终点路点
    R, p = chain.fk(q, 0.12)
    print("共 %d 点, 终点指尖误差 %.4f m"
          % (len(points), float(np.linalg.norm(p - wps[-1]))))

    # 换支硬闸(输出前全轨迹复核): 臂关节(J1-J4)任何单步跳变超阈值都
    # 拒绝写文件——TCP 会甩大弧线, 真机危险(2026-09-08 实测 J2 181°)。
    P = np.array(points)
    arm_step = np.abs(np.diff(P[:, ARM_JOINTS], axis=0)).max(axis=1)
    n_over = int((arm_step > math.radians(ARM_JUMP_DEG)).sum())
    if n_over:
        i = int(np.argmax(arm_step))
        sys.exit("[致命] 输出拦截: 轨迹含 %d 处臂关节单步跳变超 %g°"
                 "(最大 %.1f° @ 点 %d), 已拒绝写 %s"
                 % (n_over, ARM_JUMP_DEG, math.degrees(arm_step.max()),
                    i, out))
    wrist_step = np.abs(np.diff(P[:, 4:], axis=0)).max(axis=1)
    print("换支复核通过: 臂关节最大单步 %.1f° (阈值 %g°) | "
          "腕部最大单步 %.1f°(绕 TCP 自旋, 重定时摊缓, 不影响走线)"
          % (math.degrees(arm_step.max()), ARM_JUMP_DEG,
             math.degrees(wrist_step.max())))

    out = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(_HERE, "dense_zigzag.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"points": points, "seg_lens": seg_lens,
                   "step_m": STEP}, f)
    print("已写出", out)


if __name__ == "__main__":
    main()
