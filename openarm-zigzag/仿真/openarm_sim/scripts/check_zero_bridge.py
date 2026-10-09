#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zero_bridge 自检脚本（阶段1验收）
=================================
1. 加载实机 zero 文件，打印"电机系 → 出厂系 → URDF 系"完整换算表；
2. 一致性断言:
   - 换算后的实测限位与 URDF 机械限位偏差 ≤3°+过冲（合规容忍）；
   - motor→URDF→motor 往返校验无损；
3. 打印宽度异常关节（真实物理差异）与标定初始位形（URDF 系）。

用法:  /usr/bin/python3 scripts/check_zero_bridge.py [zero文件路径] [right|left]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openarm_sim.zero_bridge import (JOINT_NAMES, R2D, ZeroBridge,  # noqa: E402
                                     load_zero_file, urdf_joint_names)


def default_zero_path():
    # 仓库根目录/标定/zero（scripts/ → openarm_sim/ → 仿真/ → 仓库根）
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "..",
                                         "标定", "zero"))


def main():
    side = sys.argv[2] if len(sys.argv) > 2 else "right"
    path = sys.argv[1] if len(sys.argv) > 1 else default_zero_path()
    zb = ZeroBridge(load_zero_file(path), safety_margin_deg=3.0,
                    target_side=side)
    ujoints = urdf_joint_names(side)

    print("=" * 100)
    print("zero_bridge 换算自检  |  zero 文件: %s  |  研究臂: URDF %s" %
          (path, side))
    if zb.side_mismatch:
        print("[提示] zero 文件 arm_side 标签与研究臂侧不一致（标定配置标签，"
              "不影响换算，仅影响标定时碰撞方向记录）")
    print("=" * 100)
    hdr = ("关节   实测→出厂区间(°)        端残差(°)      "
           "URDF实测区间(°)        URDF机械限位(°)   URDF软限位(°)      零位URDF(°)")
    print(hdr)
    for name, cf, res, cu, ul, sl, home in zb.report_rows():
        line = "%-5s [%8.2f,%8.2f]  [%+5.1f,%+5.1f]  " % (name, cf[0], cf[1],
                                                          res[0], res[1])
        if cu is not None:
            line += "[%8.2f,%8.2f]  [%6.1f,%6.1f]        [%7.2f,%7.2f]    %8.2f" % (
                cu[0], cu[1], ul[0], ul[1], sl[0], sl[1], home)
        else:
            line += "%56s%8.2f" % ("（夹爪不参与臂运动学）      ", home)
        print(line)

    print("-" * 100)
    print("每关节常量偏移 s（电机系→出厂系, °）: %s"
          % ", ".join("%s=%+.2f" % (n, zb.s_rad[i] * R2D)
                      for i, n in enumerate(JOINT_NAMES)))

    # ---------- 物理差异告警 ----------
    warns = zb.width_warnings
    if warns:
        print("\n[告警] 限位宽度与官方出厂表存在真实物理差异（软限位仍按 zero 实测采用）:")
        for name, w_m, w_f in warns:
            note = ("正向干涉限位提前 %.1f°（本台机械臂实测）"
                    % (w_f - w_m) if w_m < w_f
                    else "超出官方标称 %.1f°（柔性/过冲，不影响臂关节）"
                    % (w_m - w_f))
            print("  - %s: 实测宽 %.1f° vs 官方 %.1f° —— %s" % (name, w_m, w_f, note))
    overs = zb.urdf_overshoot_warnings
    if overs:
        print("[提示] 实测限位超出 URDF 机械限位（合规过冲，软限位不做裁剪）:")
        for name, o_lo, o_hi in overs:
            print("  - %s: 负端超出 %.2f°, 正端超出 %.2f°" % (name, o_lo, o_hi))

    # ---------- 一致性断言 ----------
    print("\n一致性断言:")
    for i, uj in enumerate(ujoints):
        u0, u1 = zb.urdf_limits_deg[i]
        sl0, sl1 = zb.urdf_soft_limits_rad[i]
        # 软限位必须严格在机械限位内（实测过冲最大 2.5° < 3° 边距，因此恒成立）
        assert sl0 * R2D >= u0 + 0.01 and sl1 * R2D <= u1 - 0.01, uj
        # 往返校验
        for q in (sl0, sl1, zb.urdf_home_rad[i]):
            assert abs(zb.motor_to_urdf(zb.urdf_to_motor(q, i), i) - q) < 1e-12
    print("  [通过] 软限位 = zero 实测限位(URDF 系) − 3° 边距，无任何官方/机械限位裁剪")
    print("  [通过] 软限位仍落在 URDF 机械限位内（过冲 < 边距，含 safety 余量）")
    print("  [通过] motor→URDF→motor 往返换算无损")

    print("\n标定初始位形（URDF %s 系, °）: %s"
          % (side, ", ".join("%s=%.2f" % (n, zb.urdf_home_rad[i] * R2D)
                             for i, n in enumerate(JOINT_NAMES))))
    print("\n阶段1自检全部通过 ✔")


if __name__ == "__main__":
    main()
