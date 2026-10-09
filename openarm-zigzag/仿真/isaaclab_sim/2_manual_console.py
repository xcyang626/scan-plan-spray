#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2_manual_console.py —— Isaac Lab 交互控制台（zero 文件原值限位）
================================================================
场景: 右臂（研究臂）+ 地面 + 2m 半透明墙 + 0.3m TCP 工作平面；
控制: 终端控制台直接操作（数值与限位 = zero 文件原值，电机反馈系），
      内部换算到 URDF 关节系驱动 Isaac Sim 位置目标。

用法:
    cd ~/IsaacLab && ./isaaclab.sh -p \
        ~/桌面/openarm实验/仿真/isaaclab_sim/2_manual_console.py

命令: 1..7 选关节 | w/+ 正步进 | s/- 负步进 | = <deg> 绝对设定
      step <deg> 改步长 | 0 当前关节回 0 | z 回零位 | g <mm> 夹爪 | q 退出
"""

import argparse
import math
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import USD_PATH, load_bridge, load_chain  # noqa: E402

parser = argparse.ArgumentParser(description="OpenArm Isaac Lab 交互控制台")
from isaaclab.app import AppLauncher  # noqa: E402
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

from openarm_sim.term_console import TerminalConsole  # noqa: E402

D2R = math.pi / 180.0

# ---------- 数据 ----------
bridge = load_bridge()
chain = load_chain()
limits_deg = bridge.motor_limits_deg(7)          # zero 文件原值（电机系）
initial_deg = bridge.hanging_motor_deg()         # 悬垂位形（电机系）
home_deg = [v * 180.0 / math.pi
            for v in bridge.zero_position_rad[:7]]   # 标定手摆零位

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
    # 控制台退出 → 关闭仿真
    simulation_app.close()


threading.Thread(target=console_thread, daemon=True).start()

# ---------- 场景 ----------
sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0)
sim = SimulationContext(sim_cfg)
sim.set_camera_view(eye=(2.6, -2.2, 1.3), target=(0.35, 0.0, 0.55))

sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=2500.0).func(
    "/World/light", sim_utils.DomeLightCfg(intensity=2500.0))

# 墙: 基座前方 2m 半透明竖直平面
wall = sim_utils.CuboidCfg(
    size=(0.03, 2.0, 1.8),
    visual_material=sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.85, 0.85, 0.9), opacity=0.55))
wall.func("/World/scene_wall", wall, translation=(2.015, 0.0, 0.9))

# TCP 工作平面: 0.3m（与墙平行）
plane = sim_utils.CuboidCfg(
    size=(0.004, 1.0, 1.0),
    visual_material=sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.2, 0.5, 1.0), opacity=0.25))
plane.func("/World/scene_tcp_plane", plane, translation=(0.3, 0.0, 0.75))

# ---------- 机械臂 ----------
robot_cfg = ArticulationCfg(
    prim_path="/World/OpenArm",
    spawn=sim_utils.UsdFileCfg(usd_path=USD_PATH),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={"openarm_right_joint[1-7]": 0.0,
                   "openarm_right_finger_joint[12]": 0.0}),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["openarm_right_joint[1-7]"],
            effort_limit_sim={"openarm_right_joint[1-2]": 40.0,
                              "openarm_right_joint[3-4]": 27.0,
                              "openarm_right_joint[5-7]": 7.0},
            velocity_limit_sim={"openarm_right_joint[1-2]": 16.75,
                                "openarm_right_joint[3-4]": 5.45,
                                "openarm_right_joint[5-7]": 20.94},
            stiffness={"openarm_right_joint[1-2]": 300.0,
                       "openarm_right_joint[3-4]": 150.0,
                       "openarm_right_joint[5-7]": 60.0},
            damping={"openarm_right_joint[1-2]": 5.0,
                     "openarm_right_joint[3-4]": 4.0,
                     "openarm_right_joint[5-7]": 2.0},
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["openarm_right_finger_joint[12]"],
            effort_limit_sim=100.0, velocity_limit_sim=1.0,
            stiffness=50.0, damping=1.0,
        ),
    },
)
robot = Articulation(robot_cfg)

sim.reset()
robot.reset()

# 关节索引（URDF 关节名 → 机器人内部顺序）
jidx = [robot.find_joints(name)[0][0] for name in chain.joint_names]
fidx = robot.find_joints("openarm_right_finger_joint[12]")[0]

_set_shared([0.0] * 7, _shared["finger"])
print("=" * 60)
print("Isaac Lab 交互控制台就绪（研究臂: openarm_right_*）")
print("在终端按控制台命令操作机械臂")
print("=" * 60)

# ---------- 主循环（实时节流 120Hz；敲命令后 1s 内显示实际关节角）----------
DT = 1.0 / 120.0
step_cnt = 0
last_print = 0.0
next_tick = time.monotonic()
while simulation_app.is_running():
    with _lock:
        q_urdf = list(_shared["q_urdf"])
        finger = _shared["finger"]
    targets = torch.zeros(robot.num_joints,
                          device=robot.device, dtype=torch.float32)
    for k, qi in enumerate(jidx):
        targets[qi] = float(q_urdf[k])
    targets[fidx] = float(finger)
    robot.set_joint_position_target(targets)
    # TCP 读数（numpy FK，控制台实时显示）
    _, p = chain.fk(q_urdf)
    _shared["tcp"] = ("TCP: x=%+.3f  y=%+.3f  z=%+.3f m"
                      % (p[0], p[1], p[2]))
    # render=True: 每步刷新视口（默认 step 不渲染，画面会卡在第一帧）
    sim.step(render=True)
    step_cnt += 1
    # 仅在控制活跃期（最后一次命令后 1s）打印实际关节角，避免刷屏
    now = time.monotonic()
    if now - _shared["last_cmd"] < 1.0 and now - last_print > 0.3:
        last_print = now
        actual = robot.data.joint_pos[0][jidx].cpu().numpy() * 180.0 / math.pi
        print("  实际(°): %s"
              % " ".join("J%d[%+7.1f]" % (k + 1, v)
                         for k, v in enumerate(actual)))
    # 实时节流: 保持 120Hz 步进
    next_tick += DT
    sleep_s = next_tick - time.monotonic()
    if sleep_s > 0.0:
        time.sleep(sleep_s)
    else:
        next_tick = time.monotonic()   # 落后了就重新对齐
