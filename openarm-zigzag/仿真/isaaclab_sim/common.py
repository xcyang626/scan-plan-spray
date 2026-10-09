# -*- coding: utf-8 -*-
"""
Isaac Lab 仿真共享模块
======================
- 路径常量与 zero_bridge/kinematics 复用（纯 numpy，无 ROS 依赖）；
- make_arm_urdf(side): 从官方 v1.urdf 生成单臂 URDF
  （删除另一侧臂链、mesh package:// 路径改写为绝对路径，供 Isaac Sim 导入器使用）。

研究对象 = 左臂（根据 zero 文件关节限位与 arm_side=left 判定）。
"""

import os
import sys
import xml.etree.ElementTree as ET

# 复用 ROS2 包里的纯 numpy 模块（zero_bridge / kinematics）
_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.abspath(os.path.join(_SIM_DIR, "..", "openarm_sim"))
_REPO_ROOT = os.path.abspath(os.path.join(_SIM_DIR, "..", ".."))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from openarm_sim.kinematics import UrdfChain          # noqa: E402
from openarm_sim.zero_bridge import (ZeroBridge,      # noqa: E402
                                     load_zero_file)

V1_URDF = os.path.abspath(os.path.join(_PKG_DIR, "config", "v1.urdf"))
ZERO_FILE = os.path.join(_REPO_ROOT, "标定", "zero")

# 官方 openarm_description 包源码目录（mesh 绝对路径前缀）
_OPENARM_DESC_SRC = os.path.abspath(os.path.join(
    os.path.expanduser("~"), "桌面", "openarm", "openarm_ros2_ws", "src",
    "openarm_description"))

SIDE = "left"                       # 研究臂侧（用户按关节限位判定为左臂）
ARM_URDF = os.path.join(_SIM_DIR, "openarm_%s.urdf" % SIDE)
USD_DIR = _SIM_DIR
USD_PATH = os.path.join(_SIM_DIR, "openarm_%s.usd" % SIDE)
TIP_LINK = "openarm_%s_hand_tcp" % SIDE


def load_bridge() -> ZeroBridge:
    return ZeroBridge(load_zero_file(ZERO_FILE), target_side=SIDE)


def load_chain(urdf_path: str = ARM_URDF) -> UrdfChain:
    return UrdfChain(urdf_path, tip_link=TIP_LINK)


def make_arm_urdf(side: str = SIDE, force: bool = False) -> str:
    """从 v1.urdf 生成单臂 URDF（已存在且非 force 则直接返回）。
    - 删除另一侧臂链（链接/关节/指爪）；
    - mesh 路径 package://openarm_description/... → 绝对路径（导入器可解析）。
    """
    urdf_path = os.path.join(_SIM_DIR, "openarm_%s.urdf" % side)
    drop = "openarm_right" if side == "left" else "openarm_left"
    if os.path.exists(urdf_path) and not force:
        return urdf_path
    tree = ET.parse(V1_URDF)
    root = tree.getroot()

    for elem in list(root):
        if elem.tag not in ("link", "joint"):
            continue
        name = elem.get("name", "")
        if drop in name:
            root.remove(elem)
            continue
        for mesh in elem.iter("mesh"):
            fn = mesh.get("filename", "")
            if fn.startswith("package://openarm_description/"):
                mesh.set("filename",
                         os.path.join(_OPENARM_DESC_SRC,
                                      fn[len("package://openarm_description/"):]))

    ET.ElementTree(root).write(urdf_path, encoding="utf-8", xml_declaration=True)
    return urdf_path


if __name__ == "__main__":
    path = make_arm_urdf(force=True)
    print("生成:", path)
    # 验证: FK 与双臂 URDF 完全一致
    chain = load_chain()
    zb = load_bridge()
    R, p = chain.fk([0.0] * 7)
    print("全零位 TCP:", [round(v, 4) for v in p],
          "(左臂期望 [0, 0.1535, 0.262])")
    print("关节:", chain.joint_names)
