# -*- coding: utf-8 -*-
"""
URDF 正运动学模块（阶段2 确认方向用，阶段3 工作区间、阶段4 IK 复用）
====================================================================

从 v1.urdf 解析 world → openarm_left_hand_tcp 的运动链，纯 numpy 实现 FK:
    T_tcp(q) = Π (origin_i · Rot(axis_i, q_i))

坐标系约定:
  - world 系 = openarm_body_link0 系（二者重合），body +x = 机械臂正前方、
    +y = 左侧、+z = 竖直向上（左臂挂载在 +y 侧，符合 ROS 常规）；
  - 关节角为 URDF 关节系（zero_bridge 换算后的系），q = [J1..J7]；
  - TCP 默认取 hand_tcp（与 link7 同原点）沿夹爪接近轴(+z_link7)偏移 tcp_offset_m，
    近似指尖接触点，实际值在 RViz 中人工校准。
"""

import math
import xml.etree.ElementTree as ET

import numpy as np

D2R = math.pi / 180.0


def _rpy_to_R(rpy):
    """URDF origin rpy(固定轴 XYZ 欧拉角) → 旋转矩阵。"""
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _axis_rot(axis, q):
    """绕单位轴 axis 旋转 q（Rodrigues 公式）。"""
    x, y, z = axis
    c, s = math.cos(q), math.sin(q)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


class UrdfChain:
    """world → tip_link 运动链 FK。"""

    def __init__(self, urdf_path: str,
                 tip_link: str = "openarm_right_hand_tcp",
                 root_link: str = "world"):
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        child_to_joint = {}
        for j in root.findall("joint"):
            jtype = j.get("type")
            parent = j.find("parent").get("link")
            child = j.find("child").get("link")
            org = j.find("origin")
            xyz = np.array([float(v) for v in
                            (org.get("xyz", "0 0 0")).split()])
            rpy = [float(v) for v in (org.get("rpy", "0 0 0")).split()]
            ax = j.find("axis")
            axis = np.array([float(v) for v in
                             (ax.get("xyz", "0 0 1")).split()]) \
                if ax is not None else np.array([0.0, 0.0, 1.0])
            child_to_joint[child] = {
                "name": j.get("name"), "type": jtype,
                "parent": parent, "child": child,
                # 静态部分: 平移 + rpy 旋转
                "T0": _rpy_to_R(rpy), "p0": xyz, "axis": axis,
            }

        # 从 tip 沿 parent 回溯到 root
        chain = []
        link = tip_link
        while link != root_link:
            if link not in child_to_joint:
                raise ValueError("URDF 链断裂: 找不到 link %r 的父关节" % link)
            j = child_to_joint[link]
            chain.append(j)
            link = j["parent"]
        chain.reverse()
        self._chain = chain
        # 可动（revolute）关节名与序号，供 IK 使用
        self.joint_names = [j["name"] for j in chain if j["type"] == "revolute"]

    # ------------------------------------------------------------------

    def fk(self, q, tcp_offset_m: float = 0.0):
        """正运动学。q: 7 关节角（rad，按 joint_names 顺序）。
        返回 (R, p): TCP 旋转矩阵 3x3 与位置 3，world 系。
        tcp_offset_m: 沿 link7 +z（夹爪接近轴）的指尖偏移。"""
        q = list(q)
        T_R = np.eye(3)
        T_p = np.zeros(3)
        qi = 0
        z_axis = np.array([0.0, 0.0, 1.0])
        last_R = None
        for j in self._chain:
            T_p = T_p + T_R @ j["p0"]
            T_R = T_R @ j["T0"]
            if j["type"] == "revolute":
                T_R = T_R @ _axis_rot(j["axis"], q[qi])
                qi += 1
            last_R = T_R
        # 指尖偏移: 沿最后一段 link7 的 +z（接近轴）
        if tcp_offset_m != 0.0:
            T_p = T_p + last_R @ z_axis * tcp_offset_m
        return T_R, T_p

    def num_joints(self) -> int:
        return len(self.joint_names)
