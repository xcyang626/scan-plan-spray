#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sync_mjcf_inertia.py —— 把权威参数(惯性+关节限位)注入 MJCF(幂等)
======================================================================
背景(2026-09-16 审计): make_mjcf.py 不写 <inertial>, 质量来自胶囊体/盒体
几何默认密度, 与 URDF 逐 link 偏差最大 3.5×(link2 0.28 vs 0.84, link7
0.47 vs 0.13) → MuJoCo 孪生(rl_validate/pd_screen/verify_deploy 回放)的
重力/动力学失真。本脚本以 real/right/config/openarm_right_real.urdf
(≡训练 USD 源, 权威)为准, 给每个臂 link body 注入 <inertial>。

v2(同日): 关节限位同步 real_safety.yaml 的 soft_limits_rad —— 旧 MJCF
range 是过期标定口径(J4 上限 2.224 vs 现行 2.414), 孪生会把合法参考
顶死在旧限位上(J4@68.8s 卡 1.3s 的"换向滞后"实为孪生限位伪影)。

- URDF inertial origin(rpy≠0)旋转: I_body = R·I·Rᵀ → fullinertia;
- 三角不等式不满足(指尖小连杆)时特征值补齐(同 MuJoCo balanceinertia);
- 幂等: 已有 <inertial> 的 body 替换之; URDF 没有的 body(如 hand_tcp
  site 挂点)不动。几何 geom 不动(仅外观/碰撞), 惯性显式覆盖后
  MuJoCo 不再用 geom 密度推质量。

用法(mujoco 环境):  python sync_mjcf_inertia.py   # 处理两份 XML
"""

import os
import re
import xml.etree.ElementTree as ET

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.abspath(os.path.join(_HERE, "..", "..", "real", "right",
                                    "config", "openarm_right_real.urdf"))
SAFETY = os.path.abspath(os.path.join(_HERE, "..", "..", "real", "right",
                                      "config", "real_safety.yaml"))
TARGETS = [os.path.join(_HERE, "openarm_right.xml"),
           os.path.join(_HERE, "openarm_right_rl.xml")]


def parse_soft_limits(path):
    """real_safety.yaml → {joint_name: (lo, hi)}(soft_limits_rad 节)。"""
    txt = open(path, encoding="utf-8").read()
    out = {}
    for m in re.finditer(
            r"(openarm_right_joint\d):\s*\[\s*([-\d.eE+]+)\s*,"
            r"\s*([-\d.eE+]+)\s*\]", txt):
        out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return out


def rpy_mat(rpy):
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0],
                   [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def parse_urdf_inertials(path):
    """link name → (mass, com(3,), I_body(3,3)) — 张量已旋进 link 系。"""
    root = ET.parse(path).getroot()
    out = {}
    for link in root.iter("link"):
        ine = link.find("inertial")
        if ine is None:
            continue
        m = float(ine.find("mass").get("value"))
        o = ine.find("origin")
        com = np.array([float(x) for x in o.get("xyz", "0 0 0").split()])
        rpy = [float(x) for x in o.get("rpy", "0 0 0").split()]
        a = ine.find("inertia").attrib
        I = np.array([
            [float(a["ixx"]), float(a.get("ixy", 0)), float(a.get("ixz", 0))],
            [float(a.get("ixy", 0)), float(a["iyy"]), float(a.get("iyz", 0))],
            [float(a.get("ixz", 0)), float(a.get("iyz", 0)), float(a["izz"])],
        ])
        R = rpy_mat(rpy)
        out[link.get("name")] = (m, com, R @ I @ R.T)
    return out


def balance_fix(I, name):
    """三角不等式 A+B≥C 不满足时特征值补齐(同 MuJoCo balanceinertia)。"""
    w, V = np.linalg.eigh(I)
    if w[0] + w[1] < w[2]:
        d = (w[2] - w[0] - w[1]) / 2.0 + 1e-9
        w = w + d
        print("  [balance] %s 特征值补齐 %+.2e" % (name, d))
        return V @ np.diag(w) @ V.T
    return I


def sync(xml_path, urdf_ine, soft_limits):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    n_fix = n_rng = 0
    for body in root.iter("body"):
        name = body.get("name", "")
        if name in urdf_ine:
            m, com, I = urdf_ine[name]
            I = balance_fix(I, name)
            fi = [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]
            for old in list(body.findall("inertial")):
                body.remove(old)
            ine = ET.Element("inertial")
            ine.set("pos", " ".join("%.6g" % v for v in com))
            ine.set("mass", "%.6g" % m)
            ine.set("fullinertia", " ".join("%.6g" % v for v in fi))
            body.insert(0, ine)
            n_fix += 1
    for jnt in root.iter("joint"):
        name = jnt.get("name", "")
        if name in soft_limits:
            lo, hi = soft_limits[name]
            jnt.set("range", "%.6f %.6f" % (lo, hi))
            n_rng += 1
    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return n_fix, n_rng


def main():
    urdf_ine = parse_urdf_inertials(URDF)
    soft = parse_soft_limits(SAFETY)
    print("URDF 惯性参数: %d links | 现行软限位: %d 关节" % (
        len(urdf_ine), len(soft)))
    for x in TARGETS:
        if not os.path.isfile(x):
            print("跳过(不存在): %s" % x)
            continue
        n, r = sync(x, urdf_ine, soft)
        print("%s: 注入 %d 个 <inertial>, 同步 %d 个关节限位"
              % (os.path.basename(x), n, r))
    print("完成 —— 孪生质量+限位已对齐权威(URDF + real_safety.yaml)")


if __name__ == "__main__":
    main()
