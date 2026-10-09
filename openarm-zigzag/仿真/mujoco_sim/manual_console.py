#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MuJoCo 交互控制台（zero 文件原值限位）
======================================
场景: 右臂（胶囊连杆）+ 2m 半透明墙 + 0.3m TCP 工作平面 + TCP 球；
控制: 终端控制台直接操作（数值与限位 = zero 文件原值，电机反馈系），
      内部换算到 URDF 关节系 → MuJoCo 位置伺服。

用法:
    conda activate mujoco
    python ~/桌面/openarm实验/仿真/mujoco_sim/manual_console.py

命令: 1..7 选关节 | w/+ 正步进 | s/- 负步进 | = <deg> 绝对设定
      step <deg> 改步长 | 0 当前关节回 0 | z 回零位 | g <mm> 夹爪 | q 退出
"""

import math
import os
import sys
import threading
import time

_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.abspath(os.path.join(_SIM_DIR, "..", "openarm_sim"))
_REPO_ROOT = os.path.abspath(os.path.join(_SIM_DIR, "..", ".."))
sys.path.insert(0, _SIM_DIR)
sys.path.insert(0, _PKG_DIR)

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402

from openarm_sim.kinematics import UrdfChain  # noqa: E402
from openarm_sim.term_console import TerminalConsole  # noqa: E402
from openarm_sim.zero_bridge import ZeroBridge, load_zero_file  # noqa: E402

XML_PATH = os.path.join(_SIM_DIR, "openarm_right.xml")

# ---------- 数据 ----------
if not os.path.exists(XML_PATH):
    from make_mjcf import build
    build()
bridge = ZeroBridge(load_zero_file(os.path.join(_REPO_ROOT, "标定", "zero")),
                    target_side="right")
chain = UrdfChain(os.path.abspath(os.path.join(_PKG_DIR, "config", "v1.urdf")),
                  tip_link="openarm_right_hand_tcp")

limits_deg = bridge.motor_limits_deg(7)
initial_deg = bridge.hanging_motor_deg()
home_deg = [v * 180.0 / math.pi for v in bridge.zero_position_rad[:7]]

# ---------- 共享控制状态 ----------
_lock = threading.Lock()
_shared = {"q_urdf": [0.0] * 7,
           "finger": bridge.grip_to_finger_m(bridge.zero_position_rad[7]),
           "last_cmd": 0.0}


def _set_shared(q_motor_rad, finger_m):
    q_urdf = [bridge.motor_to_urdf(q, i) for i, q in enumerate(q_motor_rad)]
    with _lock:
        _shared["q_urdf"] = q_urdf
        _shared["finger"] = float(finger_m)
        _shared["last_cmd"] = time.monotonic()


def console_thread():
    def on_change():
        _set_shared(cons.q_rad(), cons.finger_m())

    cons = TerminalConsole(
        limits_deg, on_change=on_change,
        get_tcp_text=lambda: _shared.get("tcp", "TCP: --"),
        initial_deg=initial_deg, home_deg=home_deg)
    cons.run()
    os._exit(0)   # 控制台退出 → 结束整个仿真


threading.Thread(target=console_thread, daemon=True).start()

# ---------- MuJoCo ----------
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
# ctrl 布局: [0..6]=J1..J7, [7..8]=双指（与 make_mjcf.py 的生成顺序一致）

print("=" * 60)
print("MuJoCo 交互控制台就绪（研究臂: openarm_right_*）")
print("=" * 60)

# ---------- 主循环（物理 500Hz，渲染 125Hz，实时节流）----------
PHYS_DT = model.opt.timestep            # 0.002
STEPS_PER_SYNC = 4                      # 500/4 = 125Hz 显示
last_print = 0.0
next_tick = time.monotonic()
step_in_frame = 0
_set_shared([v * math.pi / 180.0 for v in initial_deg],
            bridge.grip_to_finger_m(bridge.zero_position_rad[7]))

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        with _lock:
            q_urdf = list(_shared["q_urdf"])
            finger = _shared["finger"]
        data.ctrl[0:7] = q_urdf
        data.ctrl[7] = finger
        data.ctrl[8] = finger
        mujoco.mj_step(model, data)
        step_in_frame += 1

        if step_in_frame >= STEPS_PER_SYNC:
            step_in_frame = 0
            viewer.sync()
            _, p = chain.fk(q_urdf)
            _shared["tcp"] = ("TCP: x=%+.3f  y=%+.3f  z=%+.3f m"
                              % (p[0], p[1], p[2]))
            # 仅在控制活跃期（最后一次命令后 1s）打印实际关节角
            now = time.monotonic()
            if now - _shared["last_cmd"] < 1.0 and now - last_print > 0.3:
                last_print = now
                adr = [model.jnt_qposadr[mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT,
                    "openarm_right_joint%d" % (k + 1))] for k in range(7)]
                actual = data.qpos[adr] * 180.0 / math.pi
                print("  实际(°): %s"
                      % " ".join("J%d[%+7.1f]" % (k + 1, v)
                                 for k, v in enumerate(actual)))

        # 实时节流
        next_tick += PHYS_DT
        sleep_s = next_tick - time.monotonic()
        if sleep_s > 0.0:
            time.sleep(sleep_s)
        else:
            next_tick = time.monotonic()
