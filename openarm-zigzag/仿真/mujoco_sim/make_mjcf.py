#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MuJoCo MJCF 生成器: 从官方 v1.urdf 运动学数据生成单右臂 MJCF
============================================================
- 连杆外观用胶囊体/盒体近似（无外部 mesh 依赖，加载即开）；
- 关节 range = zero 文件实测限位换算到 URDF 系；
- 位置伺服（position actuator）驱动，ctrlrange 与关节限位一致；
- 场景: 2m 半透明墙 + 0.3m TCP 工作平面 + 地面；TCP 球为 link7 上的实体 geom。

用法: /usr/bin/python3 make_mjcf.py   （纯 stdlib + numpy，产出 openarm_right.xml）
"""

import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.abspath(os.path.join(_SIM_DIR, "..", "openarm_sim"))
_REPO_ROOT = os.path.abspath(os.path.join(_SIM_DIR, "..", ".."))
sys.path.insert(0, _PKG_DIR)

from openarm_sim.zero_bridge import ZeroBridge, load_zero_file  # noqa: E402

V1_URDF = os.path.abspath(os.path.join(_PKG_DIR, "config", "v1.urdf"))
MJCF_PATH = os.path.join(_SIM_DIR, "openarm_right.xml")
TIP = "openarm_right_hand_tcp"

WALL_X = 2.0     # 墙距基座水平距离
PLANE_X = 0.30   # TCP 工作平面

D2R = math.pi / 180.0


def rpy_to_quat(rpy):
    """URDF rpy（R=Rz(y)Ry(p)Rx(r)）→ MuJoCo quat (w,x,y,z)。"""
    r, p, y = rpy
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return (cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy)


def fmt(v):
    return " ".join("%.6g" % float(x) for x in v)


def build():
    bridge = ZeroBridge(load_zero_file(os.path.join(_REPO_ROOT, "标定", "zero")),
                        target_side="right")

    # ---- 解析 URDF, 取 world → tip 运动链（含 fixed 关节）----
    root = ET.parse(V1_URDF).getroot()
    child_joint = {}
    for j in root.findall("joint"):
        child_joint[j.find("child").get("link")] = {
            "name": j.get("name"), "type": j.get("type"),
            "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"),
            "xyz": [float(v) for v in j.find("origin").get("xyz", "0 0 0").split()],
            "rpy": [float(v) for v in j.find("origin").get("rpy", "0 0 0").split()],
            "axis": [float(v) for v in (j.find("axis").get("xyz", "0 0 1")
                                        if j.find("axis") is not None
                                        else "0 0 1").split()],
        }
    chain = []
    link = TIP
    while link != "world":
        j = child_joint[link]
        chain.append(j)
        link = j["parent"]
    chain.reverse()

    # 各 body 的子 body（画胶囊连线用）
    children = {}
    for j in chain:
        children.setdefault(j["parent"], []).append(j)

    # 关节限位（URDF 系, rad）: zero 实测值换算
    jlim = {}
    for j in chain:
        if j["type"] == "revolute":
            i = int(j["name"].replace("openarm_right_joint", "")) - 1
            lo = bridge.motor_to_urdf(bridge.motor_limits_rad[i][0], i)
            hi = bridge.motor_to_urdf(bridge.motor_limits_rad[i][1], i)
            jlim[j["name"]] = (lo, hi)

    # 胶囊半径（按连杆名）
    def radius(name):
        if "link1" in name or "link2" in name:
            return 0.045
        if "link3" in name or "link4" in name:
            return 0.040
        return 0.030

    kp = {"openarm_right_joint1": 300.0, "openarm_right_joint2": 300.0,
          "openarm_right_joint3": 150.0, "openarm_right_joint4": 150.0,
          "openarm_right_joint5": 60.0, "openarm_right_joint6": 60.0,
          "openarm_right_joint7": 60.0}
    forcerange = {"openarm_right_joint1": 40.0, "openarm_right_joint2": 40.0,
                  "openarm_right_joint3": 27.0, "openarm_right_joint4": 27.0,
                  "openarm_right_joint5": 7.0, "openarm_right_joint6": 7.0,
                  "openarm_right_joint7": 7.0}

    # ---- 递归生成 body 树 ----
    def body_elem(j, indent):
        pad = "  " * indent
        quat = rpy_to_quat(j["rpy"])
        s = '%s<body name="%s" pos="%s" quat="%s">\n' % (
            pad, j["child"], fmt(j["xyz"]), fmt(quat))
        # 可动关节
        if j["type"] in ("revolute", "prismatic"):
            s += '%s  <joint name="%s" type="%s" axis="%s" damping="1.0"' % (
                pad, j["name"], "hinge" if j["type"] == "revolute" else "slide",
                fmt(j["axis"]))
            if j["type"] == "revolute":
                lo, hi = jlim[j["name"]]
                s += ' range="%s" limited="true"' % fmt([lo, hi])
            else:
                s += ' range="0 0.044" limited="true"'
            s += "/>\n"
        # 外观: 指爪 / TCP / 连杆胶囊
        cname = j["child"]
        if "finger" in cname:
            side = -1.0 if "joint1" in j["name"] else 1.0
            s += ('%s  <geom type="box" size="0.006 0.008 0.045" '
                  'pos="%s" rgba="0.25 0.28 0.32 1" '
                  'contype="0" conaffinity="0"/>\n'
                  % (pad, fmt([0.0, side * 0.02, 0.035])))
        elif "hand_tcp" in cname:
            pass  # 由 link7 画 TCP 球
        else:
            # 连杆胶囊: 本体原点 → 各子体原点
            for cj in children.get(cname, []):
                to = np.array(cj["xyz"])
                if np.linalg.norm(to) < 1e-6:
                    continue
                r = 0.02 if "link7" in cname else radius(cname)
                s += ('%s  <geom type="capsule" fromto="0 0 0 %s" '
                      'size="%s" rgba="0.55 0.58 0.62 1" '
                      'contype="0" conaffinity="0"/>\n'
                      % (pad, fmt(to), r))
            # link7: TCP 球 + 指尖基座 + 夹爪双指（slide 关节）
            if "link7" in cname:
                s += ('%s  <geom type="sphere" size="0.015" pos="0 0 0.12" '
                      'rgba="0.1 0.9 0.9 0.7" contype="0" conaffinity="0"/>\n'
                      '%s  <site name="tcp" pos="0 0 0.12" size="0.006" '
                      'rgba="0.1 0.9 0.9 1"/>\n'
                      '%s  <geom type="cylinder" size="0.025 0.03" pos="0 0 0.04" '
                      'rgba="0.3 0.32 0.35 1" contype="0" conaffinity="0"/>\n')
                for jname, axis, side in (
                        ("openarm_right_finger_joint1", "0 -1 0", -1.0),
                        ("openarm_right_finger_joint2", "0 1 0", 1.0)):
                    s += ('%s  <body name="finger_%s" pos="0 0 0.1025">\n'
                          '%s    <joint name="%s" type="slide" axis="%s" '
                          'range="0 0.044" limited="true" damping="0.5"/>\n'
                          '%s    <geom type="box" size="0.006 0.008 0.045" '
                          'pos="%s" rgba="0.25 0.28 0.32 1" '
                          'contype="0" conaffinity="0"/>\n'
                          '%s  </body>\n'
                          % (pad, jname, pad, jname, axis, pad,
                             fmt([0.0, side * 0.02, 0.035]), pad))
        # 子 body
        for cj in children.get(cname, []):
            s += body_elem(cj, indent + 1)
        s += "%s</body>\n" % pad
        return s

    # ---- 组装 MJCF ----
    body0 = {"name": "openarm_body_link0", "type": "fixed",
             "parent": "world", "child": "openarm_body_link0",
             "xyz": [0, 0, 0], "rpy": [0, 0, 0]}
    xml = []
    xml.append('<mujoco model="openarm_right">')
    xml.append('  <compiler angle="radian"/>')
    xml.append('  <option timestep="0.002" integrator="implicitfast"/>')
    xml.append('  <visual><headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/>'
               '  <map znear="0.01"/></visual>')
    xml.append('  <asset>')
    xml.append('    <texture name="grid" type="2d" builtin="checker" '
               'rgb1="0.25 0.27 0.3" rgb2="0.32 0.34 0.38" width="300" height="300"/>')
    xml.append('    <material name="mat_grid" texture="grid" texrepeat="6 6" '
               'reflectance="0.1"/>')
    xml.append('  </asset>')
    xml.append('  <worldbody>')
    xml.append('    <light pos="1.5 -1.5 2.5" dir="-1 1 -1.2" directional="true"/>')
    xml.append('    <geom name="floor" type="plane" size="3 3 0.1" '
               'material="mat_grid"/>')
    # 墙: x = %.1f 半透明
    xml.append('    <geom name="wall" type="box" size="0.015 1.0 0.9" '
               'pos="%s" rgba="0.85 0.85 0.9 0.4" contype="0" conaffinity="0"/>'
               % fmt([WALL_X + 0.015, 0, 0.9]))
    # TCP 工作平面: x = 0.3
    xml.append('    <geom name="tcp_plane" type="box" size="0.002 0.5 0.5" '
               'pos="%s" rgba="0.2 0.5 1.0 0.25" contype="0" conaffinity="0"/>'
               % fmt([PLANE_X, 0, 0.75]))
    # 基座柱体
    xml.append('    <geom name="torso" type="box" size="0.07 0.09 0.35" '
               'pos="0 -0.02 0.35" rgba="0.4 0.42 0.46 1" contype="0" conaffinity="0"/>')
    # 右臂链（body0 → right_link0 为 fixed）
    xml.append(body_elem(body0, 1).rstrip())
    xml.append('  </worldbody>')
    # 位置伺服
    xml.append('  <actuator>')
    for j in chain:
        if j["type"] == "revolute":
            lo, hi = jlim[j["name"]]
            xml.append('    <position name="act_%s" joint="%s" kp="%s" '
                       'ctrlrange="%s" forcerange="%s"/>'
                       % (j["name"], j["name"], kp[j["name"]],
                          fmt([lo - 0.01, hi + 0.01]),
                          fmt([-forcerange[j["name"]], forcerange[j["name"]]])))
    for jname in ("openarm_right_finger_joint1", "openarm_right_finger_joint2"):
        xml.append('    <position name="act_%s" joint="%s" kp="200" '
                   'ctrlrange="0 0.044" forcerange="-100 100"/>' % (jname, jname))
    xml.append('  </actuator>')
    xml.append('</mujoco>')

    with open(MJCF_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(xml))
    print("MJCF 已生成:", MJCF_PATH)
    return MJCF_PATH


if __name__ == "__main__":
    build()
