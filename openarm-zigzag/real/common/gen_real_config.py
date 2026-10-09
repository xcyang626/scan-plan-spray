#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真机部署配置生成器(双臂通用, --side 选左/右臂)
==================
从实机标定 zero 文件出发, 一次性生成真机所需的三份产物:

  1. real_safety.yaml            安全层用: 每关节 zero 偏移 d(电机↔URDF 换算)、
                                 软限位、标定初始位形(home)、回位限速
  2. openarm_<side>_real.urdf    真机 URDF: 基于 moveit_sim/<side>_config 单臂
                                 URDF 修改——
                                 - ros2_control 块换 openarm_zero_hw/ZeroOffsetHW
                                   (7 个 d1..d7 偏移参数由本脚本算好嵌入)
                                 - 7 个臂关节的 limit 写入 zero 实测软限位
                                 - 手指两关节改 fixed(真机不控夹爪, 消除状态缺失)
  3. real_safety.yaml 同时被 safety_watchdog.py 使用

坐标系约定(2026-09-04 重锚定, 以 zero 文件为准):
  zero 文件的 zero_position_rad = 标定基准位形(操作员摆好的"零位", 实测即
  垂直下垂位形), 以它为 URDF 零位: d = m_zero, U = M - d, home = 全零。
  软限位 = zero 实测双向限位平移到该系(限位不对称——臂贴立柱安装, J3 向后
  摆 ~7° 即碰柱, 向前可扫 ~178°, 故不可用"行程中点"锚定)。
  J1 的零位方向 = 标定基准位形时大臂朝向(任务工作空间应在此方向, 标定前
  已确认臂面朝桌子)。

用法:
  /usr/bin/python3 gen_real_config.py --side left  [--can can0] [--canfd true]
  /usr/bin/python3 gen_real_config.py --side right --hand off
生成文件写在本目录同级的 real/<side>/config/ (--out-dir 可覆盖)。
"""

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import yaml

D2R = math.pi / 180.0

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))   # openarm实验/

# 通道映射(2026-09-04 左臂二次修正, 推翻当日早间的对调结论):
#   实测(逐电机动作+真机运行): ESC1=肩俯仰, ESC2=肩偏航/侧摆, ESC3=肩滚转,
#   ESC4=肘俯仰, ESC5=腕滚转, ESC6=腕俯仰, ESC7=腕偏航。
#   URDF 零位姿势下关节转轴几何(零运动可复算):
#     joint1 轴=左右水平(俯仰!), joint2 轴=前后水平(绕立柱侧摆),
#     joint3 轴=竖直(滚转), joint4=肘, joint5-7=腕。
#   即 URDF 关节1 本来就是俯仰 —— 当日早间"joint1=偏航需对调"是误判,
#   对调运行时肩部俯仰/侧摆值互换, 抬臂方向偏 ~70°(真机首跑失败根因)。
#   ⇒ 左臂正确映射为一一对应。CHANNEL_MAP[k] = URDF 第 k 关节使用的 ESC 序号。
#
# 各臂差异集中在此表; 每臂的 zero 文件/基底 URDF/方向符号以真机实测为准。
# 注意: 右臂方向符号尚未逐关节实测(bring-up 微动测试确认后回填), 先恒等。
SIDES = {
    "left": {
        "zero_file": os.path.join(_ROOT, "标定", "zero"),
        "base_urdf": os.path.join(
            _ROOT, "仿真", "moveit_sim", "left_config",
            "openarm_left_moveit.urdf"),
        "channel_map": [1, 2, 3, 4, 5, 6, 7],
        # 2026-09-04 真机逐关节对照(move_j1~j7): J1-J6 一致, J7 方向反 ⇒ -1
        "channel_sign": [1, 1, 1, 1, 1, 1, -1],
    },
    "right": {
        "zero_file": os.path.join(_ROOT, "标定", "zero_right"),
        "base_urdf": os.path.join(
            _ROOT, "仿真", "moveit_sim", "right_config",
            "openarm_right_moveit.urdf"),
        # 右臂 7 电机(无夹爪 ESC8)。方向符号 = 电机 raw→URDF 的硬件方向
        # 系数(与"左↔右姿态镜像映射"是两回事, 不可混用!):
        # 2026-09-09 真机阶段 F 微动终裁(move_j1~j7 vs view_live 模型):
        # J1/J7 实物反向 ⇒ 取反; J2~J6 一致 ⇒ +1。与左臂终裁同款(J7=-1)。
        # 初值[-1,1,1,1,1,1,1]中 J1(安装朝向推断)与 J7(限位跨度比对)
        # 均被微动测试推翻——限位法对近对称/镜像关节不可靠, 以微动为准。
        "channel_map": [1, 2, 3, 4, 5, 6, 7],
        "channel_sign": [1, 1, 1, 1, 1, 1, -1],
    },
}

SAFETY_MARGIN_DEG = 3.0     # 软限位安全边距(自实测机械限位向内收)
HOME_PAD_DEG = 2.0          # home 距软限位最小余量(home 必须是 MoveIt 合法
                            # 目标); d=m_zero 后 home=全零, 一般不会触发


def esc_to_urdf(seq, channel_map):
    """把按 ESC 序号排列的 7 元列表按通道映射重排为 URDF 关节序。"""
    return [seq[c - 1] for c in channel_map]


def compute_d(zero, prev_d, channel_map, channel_sign):
    """决定映射 d:
    - 仅限位文件(limits-only: 没做 set_zero, 电机系未变) → d = sign ⊙
      zero_position_rad(零位基准位形读数, 与首次标定同口径);
    - sidecar 命中(同一份 zero 已换算过) → 直接用存档结果(幂等);
    - 首次换算: d = prev_d − zero_target_pre_setzero_rad
      (set_zero 使新电机系 = 旧电机系 − pre, 支持任意位形作零位);
    - 无 pre 记录(旧版文件) → d = sign ⊙ zero_position_rad
      (zero_position = 零位基准位形的电机读数; 驱动 urdf = sign*motor − d
      要求零位位形读数为 URDF 零 ⇒ d_i = sign_i * z_i。仅当零位位形即
      期望的 URDF 零位(如垂直下垂)时成立)。"""
    z_u = esc_to_urdf([float(v) for v in zero["zero_position_rad"][:7]],
                      channel_map)
    if (not zero.get("set_zero_done", True)
            or "limits-only" in str(zero.get("calib_version", ""))):
        print("d 来源: sign ⊙ zero_position_rad(仅限位文件, 电机零点未动,"
              "不参与链式换算)")
        return [channel_sign[i] * z_u[i] for i in range(7)]
    pre = zero.get("zero_target_pre_setzero_rad")
    if prev_d is not None and pre is not None and len(pre) >= 7:
        print("d 来源: 上次映射 − sign⊙set_zero前零位读数(零位=你摆放的位形)")
        pre_u = esc_to_urdf([float(v) for v in pre], channel_map)
        return [prev_d[i] - channel_sign[i] * pre_u[i] for i in range(7)]
    print("d 来源: sign ⊙ zero_position_rad(旧版文件, 零位基准=URDF 零位)")
    return [channel_sign[i] * z_u[i] for i in range(7)]


def build_from_d(d_src, zero, channel_map, channel_sign, joint_names):
    """由 d 生成 (d_rad, soft, home):
    软限位 = 实测双向限位平移到 urdf 系(内收 SAFETY_MARGIN_DEG);
    home = 零位位形在 urdf 系的角度(= −d, 距软限位不足时夹取)。"""
    m_neg = [float(v) for v in esc_to_urdf(
        [float(v) for v in zero["neg_limits_rad"]], channel_map)]
    m_pos = [float(v) for v in esc_to_urdf(
        [float(v) for v in zero["pos_limits_rad"]], channel_map)]
    d_rad, soft, home = [], [], []
    for i in range(7):
        d = d_src[i]
        d_rad.append(d)
        mn, mp = m_neg[i], m_pos[i]
        if channel_sign[i] < 0:
            # 方向镜像关节: urdf = -motor - d, 限位区间随之翻转
            mn, mp = -mp, -mn
        lo = (mn - d) + SAFETY_MARGIN_DEG * D2R
        hi = (mp - d) - SAFETY_MARGIN_DEG * D2R
        soft.append((lo, hi))
        # home = 零位基准位形在 urdf 系的坐标 = 0(d 的定义保证),
        # 距软限位余量不足时夹取, 保证 home 是 MoveIt 合法目标
        h = 0.0
        pad = HOME_PAD_DEG * D2R
        hc = min(max(h, lo + pad), hi - pad)
        if abs(hc - h) > 1e-9:
            print("注: %s 零位位形距软限位余量不足, home 夹取为 %.2f°"
                  % (joint_names[i], hc / D2R))
        home.append(hc)
    return d_rad, soft, home


def write_safety_yaml(path, d_rad, soft, home, joint_names):
    lines = [
        "# 真机安全配置(由 gen_real_config.py 从标定 zero 文件生成, 勿手改数值)",
        "# d_rad: 电机系→URDF 系综合偏移, urdf = motor - d",
        "zero_offsets_rad:",
    ]
    for n, d in zip(joint_names, d_rad):
        lines.append("  %s: %.9f" % (n, d))
    lines.append("soft_limits_rad:")
    for n, (lo, hi) in zip(joint_names, soft):
        lines.append("  %s: [%.9f, %.9f]" % (n, lo, hi))
    lines.append("home_rad: [%s]" % ", ".join("%.9f" % v for v in home))
    lines += [
        "# 安全回位关节速度上限(rad/s)——保守值, 宁慢勿快",
        "return_speed: 0.25",
        "# 异常运动判定阈值(rad/s), 连续 3 帧超限触发急停",
        "max_joint_vel: 4.0",
        "# 软限位触发额外余量(rad)",
        "trip_margin_rad: 0.035",
        "# 触发判定需要连续帧数",
        "trip_consecutive: 5",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("已生成", path)


def write_real_urdf(path, base_urdf, d_rad, soft, can_iface, can_fd, kp, kd,
                    prefix, channel_map, channel_sign, mpc_topic, mpc_enable,
                    hand_hold=True, hand_kp=10.0, hand_kd=0.9):
    joint_names = ["openarm_%s_joint%d" % (prefix, i) for i in range(1, 8)]
    finger_names = ("openarm_%s_finger_joint1" % prefix,
                    "openarm_%s_finger_joint2" % prefix)
    tree = ET.parse(base_urdf)
    root = tree.getroot()

    # 1. 臂关节 limit 写入 zero 软限位(velocity/effort 沿用原值)
    for j in root.findall("joint"):
        if j.get("name") in joint_names:
            lim = j.find("limit")
            lo, hi = soft[joint_names.index(j.get("name"))]
            lim.set("lower", "%.9f" % lo)
            lim.set("upper", "%.9f" % hi)

    # 2. 手指关节改 fixed(真机不控夹爪; 保留 links 供显示, 消除状态缺失)
    for j in root.findall("joint"):
        if j.get("name") in finger_names:
            j.set("type", "fixed")
            mim = j.find("mimic")
            if mim is not None:
                j.remove(mim)
            lim = j.find("limit")
            if lim is not None:
                j.remove(lim)
            ax = j.find("axis")
            if ax is not None:
                j.remove(ax)

    # 3. 替换 ros2_control 块
    for rc in root.findall("ros2_control"):
        root.remove(rc)
    rc = ET.Element("ros2_control",
                    {"name": "OpenArm%sRealSystem" % prefix.capitalize(),
                     "type": "system"})
    hw = ET.SubElement(rc, "hardware")
    ET.SubElement(hw, "plugin").text = "openarm_zero_hw/ZeroOffsetHW"
    # mpc_enable/mpc_topic 由 --scheme 决定(两方案完全分离, 便于对比选型):
    #   mpc(默认) = OpenArmMpcController 插件接管 JTC 槽位, 注入通道关
    #               (见 docs/MPC控制器插件_线路一改造设计.md);
    #   rl        = 纯强化学习 rl_exec.py 流式位置指令(无 MPC 计算),
    #               注入通道开 + 独立话题, 控制器槽位应为 JTC(回滚一行)。
    for k, v in [("can_interface", can_iface), ("can_fd", can_fd),
                 ("arm_prefix", prefix + "_"),
                 ("hand", "true" if hand_hold else "false"),
                 ("mpc_enable", mpc_enable), ("mpc_topic", mpc_topic),
                 ("channel_map", ",".join(str(c) for c in channel_map)),
                 ("channel_sign",
                  ",".join(str(s) for s in channel_sign))]:
        ET.SubElement(hw, "param", {"name": k}).text = v
    if hand_hold:
        # ESC8 夹爪使能后原地锁定, 增益默认=标定 grip_hold_kp/kd
        ET.SubElement(hw, "param", {"name": "hand_kp"}).text = "%.2f" % hand_kp
        ET.SubElement(hw, "param", {"name": "hand_kd"}).text = "%.2f" % hand_kd
    # 双保险: 同时写 kp/kd 逗号列表, 兼容按列表读增益的驱动版本
    ET.SubElement(hw, "param", {"name": "kp"}).text = \
        ",".join("%.4f" % v for v in kp)
    ET.SubElement(hw, "param", {"name": "kd"}).text = \
        ",".join("%.4f" % v for v in kd)
    for i in range(7):
        ET.SubElement(hw, "param",
                      {"name": "kp%d" % (i + 1)}).text = "%.4f" % kp[i]
        ET.SubElement(hw, "param",
                      {"name": "kd%d" % (i + 1)}).text = "%.4f" % kd[i]
        ET.SubElement(hw, "param",
                      {"name": "d%d" % (i + 1)}).text = "%.9f" % d_rad[i]
    for n in joint_names:
        je = ET.SubElement(rc, "joint", {"name": n})
        ET.SubElement(je, "command_interface", {"name": "position"})
        for si in ("position", "velocity", "effort"):
            ET.SubElement(je, "state_interface", {"name": si})
    root.append(rc)

    tree.write(path, encoding="unicode", xml_declaration=True)
    print("已生成", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="left", choices=sorted(SIDES),
                    help="目标臂: left(默认)/right")
    ap.add_argument("--can", default="can0", help="SocketCAN 接口名")
    ap.add_argument("--canfd", default="true", choices=["true", "false"],
                    help="CAN-FD 开关(按你的 CAN 模块能力定)")
    ap.add_argument("--kp", default="120,70,70,90,30,30,30",
                    help="7 关节 MIT kp(2026-09-04: J5-J7 由 10 提到 30,"
                         " kp=10 的腕部在竖直零位下顶不住接触/线束阻力,"
                         " 会下垂 20° 导致 JTC 容差中止)")
    ap.add_argument("--kd", default="3.6,3.0,2.0,2.2,1.5,1.5,1.2",
                    help="7 关节 MIT kd(随 kp 提高相应加强阻尼)")
    ap.add_argument("--hand", default="hold", choices=["off", "hold"],
                    help="ESC8 夹爪: hold=control 层使能后原地锁定(默认), "
                         "off=完全不接触 ESC8(右臂无夹爪必选)")
    ap.add_argument("--hand-kp", default="10.0",
                    help="夹爪锁定 kp(默认=标定 grip_hold_kp)")
    ap.add_argument("--hand-kd", default="0.9",
                    help="夹爪锁定 kd(默认=标定 grip_hold_kd)")
    ap.add_argument("--scheme", default="mpc", choices=["mpc", "rl"],
                    help="指令方案(两方案互斥, 便于对比选型): "
                         "mpc=OpenArmMpcController 插件接管 JTC 槽位, "
                         "注入通道关闭(默认, 线路一); "
                         "rl=纯强化学习 rl_exec 流式指令, 注入通道开启"
                         "+独立话题 /<side>_rl_position_commands。"
                         "切换后需重启 control 层, 并跑 verify_deploy --scheme")
    ap.add_argument("--out-dir", default=None,
                    help="输出目录(默认 real/<side>/config)")
    args = ap.parse_args()
    side = args.side
    S = SIDES[side]
    prefix = side
    channel_map = list(S["channel_map"])
    channel_sign = list(S["channel_sign"])
    joint_names = ["openarm_%s_joint%d" % (prefix, i) for i in range(1, 8)]
    mpc_topic = ("/%s_rl_position_commands" % prefix
                 if args.scheme == "rl"
                 else "/%s_mpc_position_commands" % prefix)
    mpc_enable = "true" if args.scheme == "rl" else "false"
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else \
        os.path.abspath(os.path.join(_HERE, "..", side, "config"))
    os.makedirs(out_dir, exist_ok=True)

    kp = [float(v) for v in args.kp.split(",")]
    kd = [float(v) for v in args.kd.split(",")]
    assert len(kp) == 7 and len(kd) == 7

    with open(S["zero_file"], encoding="utf-8") as f:
        zero = json.load(f)
    if zero.get("status") != "ok":
        sys.exit("zero 文件 status != ok, 拒绝生成真机配置")
    if str(zero.get("arm_side", "")).lower() != side:
        print("警告: zero 文件 arm_side=%r 与目标侧 %s 不一致, 请人工确认"
              % (zero.get("arm_side"), side))

    # 上次映射 d(用于通用零位换算 d = d_prev − set_zero前零位读数)。
    # 存于 sidecar(d_prev_sidecar.json): 记录该 d 对应的 zero 时间戳,
    # 同一份 zero 文件重复运行 gen 时直接复用结果(幂等, 防重复叠加平移)。
    sidecar_path = os.path.join(out_dir, "d_prev_sidecar.json")
    sidecar = None
    if os.path.isfile(sidecar_path):
        try:
            with open(sidecar_path, encoding="utf-8") as f:
                sidecar = json.load(f)
        except (OSError, ValueError):
            sidecar = None
    if sidecar and sidecar.get("zero_ts") == zero.get("timestamp"):
        print("d_prev: sidecar 命中(该 zero 已换算过, 本次结果幂等)")
        d_src = [float(v) for v in sidecar["d"]]
    else:
        prev_yaml = os.path.join(out_dir, "real_safety.yaml")
        prev_d = None
        if os.path.isfile(prev_yaml):
            try:
                with open(prev_yaml, encoding="utf-8") as f:
                    _prev = yaml.safe_load(f)
                _zmap = _prev.get("zero_offsets_rad", {})
                prev_d = [float(_zmap[n]) for n in joint_names]
            except (OSError, ValueError, KeyError, TypeError):
                prev_d = None
                print("警告: 现有 real_safety.yaml 缺 zero_offsets_rad, "
                      "退回旧版换算")
        d_src = compute_d(zero, prev_d, channel_map, channel_sign)

    d_rad, soft, home = build_from_d(d_src, zero, channel_map,
                                     channel_sign, joint_names)

    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump({"zero_ts": zero.get("timestamp"),
                   "d": [round(v, 9) for v in d_rad]}, f, indent=2)

    print("==== %s 臂 偏移/限位核对(度) ====" % side)
    for i, n in enumerate(joint_names):
        print("%s: d=%9.3f  软限位=[%8.2f, %8.2f]  标定初始位形=%8.2f"
              % (n, d_rad[i] / D2R,
                 soft[i][0] / D2R, soft[i][1] / D2R, home[i] / D2R))
    print("方案=%s | mpc_enable=%s | 指令话题=%s"
          % (args.scheme, mpc_enable, mpc_topic))
    print("通道映射=%s 通道方向=%s" % (channel_map, channel_sign))

    write_safety_yaml(os.path.join(out_dir, "real_safety.yaml"),
                      d_rad, soft, home, joint_names)
    write_real_urdf(
        os.path.join(out_dir, "openarm_%s_real.urdf" % prefix),
        S["base_urdf"], d_rad, soft, args.can, args.canfd, kp, kd,
        prefix, channel_map, channel_sign, mpc_topic, mpc_enable,
        hand_hold=(args.hand == "hold"),
        hand_kp=float(args.hand_kp), hand_kd=float(args.hand_kd))
    print("夹爪 ESC8: %s (hand_kp=%s hand_kd=%s)"
          % ("使能并原地锁定" if args.hand == "hold" else "不接触",
             args.hand_kp, args.hand_kd))


if __name__ == "__main__":
    main()
