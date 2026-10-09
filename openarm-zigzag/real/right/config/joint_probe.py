#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单关节单向验证共用模块(被 move_j1.py ~ move_j7.py 调用)。

行为(2026-09-09 改版):
  从当前位置朝一个方向动 deg 度(慢速) → 停住**保持** → 人工对照
  终端 B(view_live RViz 模型)与终端 B'(check_states)看方向/幅度 →
  **按 Ctrl+C 自动慢速回 home 再退出**(回程中再按 Ctrl+C = 就地停住)。

用法(经 move_jN.py): /usr/bin/python3 move_j3.py [度数] [--speed 0.08]

安全: 目标钳位在软限位内 2.5°(大于 watchdog 2° 触发余量), 超界拒绝;
限位/home 实时读 real_safety.yaml; /joint_states 按关节名取
(2026-09-09 教训: 按位切片会被乱序关节错位)。执行期间人不离臂,
物理急停是最后防线。
"""

import argparse
import math
import os
import sys
import threading
import time

import rclpy
import yaml
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

_HERE = os.path.dirname(os.path.abspath(__file__))
SAFETY = os.path.join(_HERE, "real_safety.yaml")
ACTION = "/right_joint_trajectory_controller/follow_joint_trajectory"
NAMES = ["openarm_right_joint%d" % i for i in range(1, 8)]

# 各关节语义(2026-09-04 修正: joint1=俯仰, joint2=侧摆)
DESC = {1: "肩俯仰(+后仰 / -前抬)",
        2: "肩侧摆(绕立柱, +一侧 / -另一侧)",
        3: "大臂滚转(自转)",
        4: "肘(只做正方向, 负向贴限位)",
        5: "腕滚转",
        6: "腕俯仰",
        7: "腕偏航"}
DEFAULT_DEG = {1: 5.0, 2: 4.0, 3: 10.0, 4: 10.0, 5: 10.0, 6: 10.0, 7: 10.0}
MARGIN = math.radians(2.5)   # 钳位余量, 大于 watchdog 的 2° 触发余量
SPEED = 0.10                 # rad/s, 慢速便于观察
D2R = math.pi / 180.0


def build_traj(q0, q1, speed):
    """两点轨迹(JTC 自行线性插值), 按最大行程限速 speed(rad/s)。"""
    worst = max(abs(q1[i] - q0[i]) for i in range(7))
    dt = max(worst / speed, 1.5)
    jt = JointTrajectory()
    jt.joint_names = NAMES
    for t, pos in ((0.0, q0), (dt, q1)):
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in pos]
        p.time_from_start.sec = int(t)
        p.time_from_start.nanosec = int(round((t - int(t)) * 1e9))
        jt.points.append(p)
    return jt


def send_goal(act, jt, timeout):
    """发目标并等结果。Ctrl+C → 取消目标(电机保持)→ 返回 "cancelled"。"""
    if not act.wait_for_server(timeout_sec=5.0):
        sys.exit("JTC action 不可用 —— control 层的 JTC 起了吗?")
    goal = FollowJointTrajectory.Goal()
    goal.trajectory = jt
    gh = None
    try:
        fut = act.send_goal_async(goal)
        t0 = time.monotonic()
        while not fut.done() and time.monotonic() - t0 < 10.0:
            time.sleep(0.05)
        if not fut.done():
            sys.exit("发目标超时")
        gh = fut.result()
        if not gh.accepted:
            sys.exit("目标被 JTC 拒收(JTC 忙/起始偏差?)")
        rf = gh.get_result_async()
        t0 = time.monotonic()
        while not rf.done() and time.monotonic() - t0 < timeout:
            time.sleep(0.1)
        if not rf.done():
            return gh, None
        return gh, rf.result().status
    except KeyboardInterrupt:
        print("\n[Ctrl+C] 取消当前目标, 电机保持当前位置 ...")
        if gh is not None:
            gh.cancel_goal_async()
        time.sleep(1.0)          # 等电机停稳
        return gh, "cancelled"


def read_q(box, settle=0.5):
    """等一段沉降后取最新关节读数(度未转, rad)。"""
    time.sleep(settle)
    t0 = time.monotonic()
    while box["q"] is None and time.monotonic() - t0 < 5.0:
        time.sleep(0.05)
    if box["q"] is None:
        sys.exit("读不到 /joint_states —— 终端 A 的 control 层在跑吗?")
    return list(box["q"])


def go_home(act, box, home, speed):
    """慢速回 home 并打印残余偏差。返回 False=中途被再次 Ctrl+C。"""
    q_now = read_q(box)
    worst = max(abs(q_now[i] - home[i]) for i in range(7))
    print("自动回 home: 最大行程 %.1f°, 限速 %.2f rad/s ..."
          % (worst / D2R, speed))
    gh, res = send_goal(act, build_traj(q_now, home, speed), 300.0)
    if res == "cancelled":
        print("[中断] 回家被取消, 臂停在当前位置。重新执行本命令可继续。")
        return False
    if res != 4:
        print("⚠ 回家轨迹未正常完成(status=%s)——读数核对口径:"
              " check_states.py" % res)
        return False
    q_end = read_q(box, settle=1.0)
    resid = max(abs(q_end[i] - home[i]) for i in range(7))
    print("已回 home, 最大残余偏差 %.2f°(应接近 0)" % (resid / D2R))
    return True


def probe(joint, argv):
    ap = argparse.ArgumentParser(
        prog="move_j%d.py" % joint,
        description="J%d(%s) 单向微动: 动→保持→Ctrl+C 自动回 home"
                    % (joint, DESC[joint]))
    ap.add_argument("deg", nargs="?", type=float, default=DEFAULT_DEG[joint],
                    help="微动幅度(度), 正=URDF 正方向, 默认 %.1f"
                         % DEFAULT_DEG[joint])
    ap.add_argument("--speed", type=float, default=SPEED,
                    help="关节限速(rad/s), 默认 %.2f" % SPEED)
    args = ap.parse_args(argv[1:])
    deg, speed = args.deg, args.speed

    with open(SAFETY, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    lim = cfg["soft_limits_rad"]
    if isinstance(lim, dict):
        lo, hi = lim[NAMES[joint - 1]]
    else:
        lo, hi = lim[joint - 1]
    home = list(cfg["home_rad"])

    # 关闭 rclpy 内置 SIGINT 处理: 它会在 Ctrl+C 时直接关掉 context,
    # 与"Ctrl+C → 自动回 home"流程竞争(context 失效后无法再发轨迹)。
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Node("move_j%d" % joint)
    box = {"q": None}

    def on_js(m):
        try:
            box["q"] = [float(m.position[m.name.index(n)]) for n in NAMES]
        except (ValueError, IndexError):
            pass    # 消息缺臂关节(如其它发布者), 不更新

    node.create_subscription(JointState, "/joint_states", on_js, 10)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    act = ActionClient(node, FollowJointTrajectory, ACTION)

    try:
        q = read_q(box, settle=0.2)
        tgt = q[joint - 1] + math.radians(deg)
        if tgt < lo + MARGIN or tgt > hi - MARGIN:
            sys.exit("拒绝: J%d 目标 %.2f° 超出安全范围 [%.2f, %.2f]°\n"
                     "(软限位 %s, 保留 2.5° watchdog 余量)"
                     % (joint, math.degrees(tgt), math.degrees(lo + MARGIN),
                        math.degrees(hi - MARGIN),
                        "[%.2f, %.2f]" % (math.degrees(lo),
                                          math.degrees(hi))))

        print("=" * 62)
        print("J%d (%s) 即将 %+.1f°, 限速 %.2f rad/s, 到位后保持"
              % (joint, DESC[joint], deg, speed))
        print("对照: 终端 B view_live 的 RViz 模型(转向/幅度) + "
              "终端 B' check_states(数值)")
        print("=" * 62)

        gh, res = send_goal(act, build_traj(q, q[:joint - 1] + [tgt] +
                                            q[joint:], speed), 120.0)
        if res == "cancelled":
            go_home(act, box, home, speed)      # 动作中途被打断 → 直接回家
            return
        ok = res == 4
        now_q = read_q(box, settle=1.0)
        print("=" * 62)
        print("J%d: 命令 %+.1f° | %s | 读数变化 %+.2f° (起点 %.2f°)"
              % (joint, deg, "已停住保持" if ok else
                 "失败(status=%s)" % res,
                 math.degrees(now_q[joint - 1] - q[joint - 1]),
                 math.degrees(q[joint - 1])))
        if ok:
            print("保持位形中 ... 记录方向/幅度后按 Ctrl+C → 自动回 home")
            print("=" * 62)
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("\n[Ctrl+C] 结束保持")
        else:
            print("动作未正常完成, 转入自动回 home")
        go_home(act, box, home, speed)
    except KeyboardInterrupt:
        # 落在读数/间隙里的 Ctrl+C(保持循环与 send_goal 内已各自接住):
        # 就地退出, 电机保持当前位置。
        print("\n[Ctrl+C] 退出, 臂保持当前位置")
    finally:
        executor.remove_node(node)
        executor.shutdown()
        spin.join(timeout=2.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    pass    # 本模块被 move_j1.py ~ move_j7.py import 调用
