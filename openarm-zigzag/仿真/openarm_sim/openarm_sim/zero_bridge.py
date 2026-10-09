# -*- coding: utf-8 -*-
"""
zero 文件桥接模块（阶段1）
==========================

职责: 把实机标定产物 zero(JSON) 的"电机反馈系"数据换算到 URDF 关节系，
     为仿真提供机械零位姿态与各关节软限位；并提供双向换算，
     供后续轨迹从仿真导出到实机（URDF→电机）使用。

坐标系三层模型（左臂）:
  电机反馈系 M: 达妙电机反馈的多圈累积读数（实机 zero 文件所处坐标系）。
                标定节点全程记录的都是该系读数，zero_position_rad =
                标定启动时操作员手摆位形的读数（calibration_worker 第 1 步锁定）。
  出厂编码系 F: 官方 MECH_LIM_V1 定义的出厂零位系。
  URDF 关节系 U: v1.urdf 各 revolute joint 的角度（仿真 FK/IK 全程使用）。

换算链:  U = M - s + offset
  s:      每关节常量偏移，由"实测限位区间 ↔ 出厂限位区间 满程对满程"唯一确定。
          各关节限位区间宽度 87°~281° 均小于 360°，且要求两区间对齐重叠，
          因此 s 的解唯一（平移 ±360° 均会导致区间错开，已被数据排除）。
  offset: 出厂系 → URDF 系的固定偏移，左臂 J1=-120°、J2=-90°、其余 0
          （与 rviz_bridge.py 满程映射一致，仿真与实机均已验证）。

实测数据结论（2026-09-02 实机 zero，左臂，check_zero_bridge.py 可复现）:
  - 假设"zero 文件本身就是出厂系"被否定: J1 实测[-252°,29°]与出厂[-80°,200°]完全错位；
  - 常量偏移假设成立: J1/J2/J5/J6/J7 端残差≤0.8°，J3 为 2.5°（碰限位合规过冲，
    各关节普遍有 0.5~2.5° 过冲且方向一致，属正常物理现象）；
  - J4 正向限位比官方提前 12.6°（127.4° vs 140°）: 本台机械臂的真实干涉限位，
    锚定负端（官方负限位 0° 处碰撞点吻合）后保留实测正值；
  - 夹爪开合比官方标称宽 15.4°（指爪柔性），不参与臂关节运动学。

数据源约定（重要）:
  仿真的零位与软限位**唯一来源于实机 zero 文件**（换算到 URDF 系后留安全边距，
  不做任何与官方/URDF 机械限位的取交集裁剪）。官方 MECH_LIM_V1 与 URDF 限位
  仅承担两个辅助角色: (1) 估计偏移 s 的对齐标尺——zero 文件本身没有绝对参考系，
  必须借助满程对满程锚定; (2) 事后核对提示（实测超出机械限位仅告警，不裁剪）。
"""

import json
import math

D2R = math.pi / 180.0
R2D = 180.0 / math.pi

# 官方 MECH_LIM_V1 出厂系限位（度），8 关节 [neg, pos]
FACTORY_LIMITS_DEG = [
    (-80.0, 200.0),   # J1
    (-100.0, 100.0),  # J2
    (-90.0, 90.0),    # J3
    (0.0, 140.0),     # J4
    (-90.0, 90.0),    # J5
    (-45.0, 45.0),    # J6
    (-90.0, 90.0),    # J7
    (-60.0, 0.0),     # 夹爪
]

# v1.urdf 机械限位（度），由 config/v1.urdf 各 revolute joint 的 limit 读出。
# 研究对象 = 面向正面时位于左侧的臂 = URDF openarm_right_*（-y 侧挂载）
URDF_LIMITS_BY_SIDE = {
    "right": [
        (-80.0, 200.0),   # openarm_right_joint1
        (-10.0, 190.0),   # openarm_right_joint2
        (-90.0, 90.0),    # openarm_right_joint3
        (0.0, 140.0),     # openarm_right_joint4
        (-90.0, 90.0),    # openarm_right_joint5
        (-45.0, 45.0),    # openarm_right_joint6
        (-90.0, 90.0),    # openarm_right_joint7
    ],
    "left": [
        (-200.0, 80.0),   # openarm_left_joint1
        (-190.0, 10.0),   # openarm_left_joint2
        (-90.0, 90.0),
        (0.0, 140.0),
        (-90.0, 90.0),
        (-45.0, 45.0),
        (-90.0, 90.0),
    ],
}

# 出厂系 → URDF 系固定偏移（度）。J1/J2 因安装方向镜像，
# 满程对满程唯一确定（与 rviz_bridge.build_mapping 一致）
URDF_OFFSET_BY_SIDE = {
    "right": [0.0, 90.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "left": [-120.0, -90.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}

JOINT_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "grip"]


def urdf_joint_names(side: str):
    """指定臂侧的 7 个 URDF 关节名。"""
    return ["openarm_%s_joint%d" % (side, i) for i in range(1, 8)]

# 偏移估计方式: "center"=区间中点对齐（限位两端均可信）；
# "neg"=负端锚定（负限位与官方吻合，正向端存在真实物理差异，如 J4 干涉、夹爪柔性）
ANCHOR_MODE = {3: "neg", 7: "neg"}   # J4=idx3, grip=idx7

# URDF finger 平移行程（m），夹爪电机角 → 双指开合线性映射（与 rviz_bridge 一致）
FINGER_MAX_M = 0.044


class ZeroBridge:
    """实机 zero 数据 → URDF 关节系的桥接器。

    用法:
        zb = ZeroBridge(json.load(open("zero")))
        q_urdf = zb.motor_to_urdf(q_motor_rad, idx)   # 单关节换算
        lims = zb.urdf_soft_limits_rad                # 7 臂关节软限位 [(min,max)]
        q_home = zb.urdf_home_rad                     # 标定初始位形（URDF 系）
    """

    def __init__(self, zero_data: dict, safety_margin_deg: float = 3.0,
                 target_side: str = "right"):
        if zero_data.get("status") != "ok":
            raise ValueError("zero 文件 status=%r ≠ ok，不能用于仿真"
                             % zero_data.get("status"))
        # 研究对象 = 面向正面时位于左侧的臂 = URDF right（-y 挂载）
        self.side = str(target_side).lower()
        self.side_mismatch = (str(zero_data.get("arm_side", "")).lower()
                              != self.side)
        # 注: zero 文件的 arm_side 是标定时的配置标签（影响碰撞方向），
        # 与 URDF 挂载侧可以不一致，仅提示不阻断
        self.urdf_limits_deg = URDF_LIMITS_BY_SIDE[self.side]
        self.urdf_offset_deg = URDF_OFFSET_BY_SIDE[self.side]

        m_neg = [float(v) for v in zero_data["neg_limits_rad"]]
        m_pos = [float(v) for v in zero_data["pos_limits_rad"]]
        m_zero = [float(v) for v in zero_data["zero_position_rad"]]
        self._m_neg, self._m_pos = m_neg, m_pos   # 供报告显示
        if not (len(m_neg) == len(m_pos) == len(m_zero) == 8):
            raise ValueError("zero 文件限位/零位必须为 8 关节")

        # ---------- 每关节常量偏移 s: 实测限位 ↔ 出厂限位 满程对满程 ----------
        self.s_rad = []
        self._width_warn = []          # 宽度与官方不符的关节（物理差异）
        self._over_urdf = []           # 实测限位超出 URDF 机械限位的关节（仅提示）
        for i in range(8):
            f0, f1 = FACTORY_LIMITS_DEG[i]
            if ANCHOR_MODE.get(i) == "neg":
                # 负端锚定: F = M - s 且 F_neg = 官方负限位 → s = M_neg - F_neg
                s = m_neg[i] - f0 * D2R
            else:
                c_m = 0.5 * (m_neg[i] + m_pos[i])
                c_f = 0.5 * (f0 + f1) * D2R
                s = c_m - c_f
            self.s_rad.append(s)
            w_m = (m_pos[i] - m_neg[i]) * R2D
            w_f = f1 - f0
            if abs(w_m - w_f) > 5.0:
                self._width_warn.append((JOINT_NAMES[i], w_m, w_f))

        # ---------- URDF 系换算 ----------
        # U = M - s + offset
        off = self.urdf_offset_deg
        self.urdf_home_rad = [m_zero[i] - self.s_rad[i]
                              + (off[i] * D2R if i < 7 else 0.0)
                              for i in range(8)]
        # 软限位 = 实测限位(换算到 URDF 系) - 安全边距。
        # 数据源唯一: 实机 zero 文件。官方 MECH_LIM_V1 / URDF 机械限位仅用于
        # 估计偏移 s（对齐标尺）与事后核对，不参与限位数值本身。
        self.marg = safety_margin_deg
        self.urdf_soft_limits_rad = []
        ujoints = urdf_joint_names(self.side)
        for i in range(7):
            lo = m_neg[i] - self.s_rad[i] + off[i] * D2R
            hi = m_pos[i] - self.s_rad[i] + off[i] * D2R
            lo += self.marg * D2R
            hi -= self.marg * D2R
            if lo >= hi:
                raise ValueError(
                    "%s 软限位为空: [%.2f, %.2f]°，请检查 zero 数据"
                    % (ujoints[i], lo * R2D, hi * R2D))
            self.urdf_soft_limits_rad.append((lo, hi))
            # 事后核对（仅提示，不裁剪）: 实测限位超出 URDF 机械限位的量
            u0, u1 = self.urdf_limits_deg[i]
            over_lo = u0 * D2R - (lo - self.marg * D2R)
            over_hi = (hi + self.marg * D2R) - u1 * D2R
            if over_lo > 0.01 or over_hi > 0.01:
                self._over_urdf.append(
                    (ujoints[i], over_lo * R2D, over_hi * R2D))

        # 夹爪实测开合范围（电机系）→ finger 平移映射用
        self._grip_min, self._grip_max = m_neg[7], m_pos[7]
        # zero 文件原始限位/零位（电机系原值）——控制台/滑条直接使用，
        # 保证操作数值与 zero 文件逐位一致（用户要求）
        self.motor_limits_rad = [(m_neg[i], m_pos[i]) for i in range(8)]
        self.zero_position_rad = m_zero

    def motor_limits_deg(self, n: int = 7):
        """前 n 个关节的 zero 文件原值限位（度）。"""
        return [(lo * R2D, hi * R2D) for lo, hi in self.motor_limits_rad[:n]]

    def hanging_motor_deg(self):
        """URDF 全零位形（手臂自然下垂）对应的电机系角度（度）。"""
        return [self.urdf_to_motor(0.0, i) * R2D for i in range(7)]

    # ------------------------------------------------------------------
    # 双向换算
    # ------------------------------------------------------------------

    def motor_to_urdf(self, q_motor: float, idx: int) -> float:
        """电机反馈系 → URDF 关节系（idx 0..6 臂关节）。"""
        return q_motor - self.s_rad[idx] + self.urdf_offset_deg[idx] * D2R

    def urdf_to_motor(self, q_urdf: float, idx: int) -> float:
        """URDF 关节系 → 电机反馈系（实机轨迹导出用）。"""
        return q_urdf + self.s_rad[idx] - self.urdf_offset_deg[idx] * D2R

    def grip_to_finger_m(self, q_grip_motor: float) -> float:
        """夹爪电机角（电机系）→ 双指开合平移量 0~0.044 m。"""
        ratio = (q_grip_motor - self._grip_min) / (self._grip_max - self._grip_min)
        ratio = min(1.0, max(0.0, ratio))
        return ratio * FINGER_MAX_M

    # ------------------------------------------------------------------
    # 报告
    # ------------------------------------------------------------------

    def report_rows(self):
        """生成自检报告行: [(关节, 实测区间, s, 换算出厂区间, 端残差,
        URDF 实测区间, URDF 软限位, 零位URDF角)]（均为度）。"""
        rows = []
        for i, name in enumerate(JOINT_NAMES):
            f0, f1 = FACTORY_LIMITS_DEG[i]
            s = self.s_rad[i]
            # 出厂系 F = 实测 M - s
            cf = ((self._m_neg[i] - s) * R2D, (self._m_pos[i] - s) * R2D)
            res = (cf[0] - f0, cf[1] - f1)
            if i < 7:
                u0, u1 = self.urdf_limits_deg[i]
                cu = (cf[0] + self.urdf_offset_deg[i],
                      cf[1] + self.urdf_offset_deg[i])
                sl = self.urdf_soft_limits_rad[i]
                rows.append((name, cf, res, cu, (u0, u1),
                             (sl[0] * R2D, sl[1] * R2D),
                             self.urdf_home_rad[i] * R2D))
            else:
                rows.append((name, cf, res, None, None, None,
                             self.urdf_home_rad[i] * R2D))
        return rows

    @property
    def width_warnings(self):
        """宽度与官方出厂表不符的关节 [(名称, 实测宽, 官方宽)]。"""
        return list(self._width_warn)

    @property
    def urdf_overshoot_warnings(self):
        """实测限位超出 URDF 机械限位的关节 [(关节名, 负端超出量°, 正端超出量°)]。
        仅为核对提示：软限位仍完全采用 zero 文件实测值，不做裁剪。"""
        return list(self._over_urdf)


def load_zero_file(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
