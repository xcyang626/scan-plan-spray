#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_rl_usd.py —— 生成 RL 训练用 USD: 夹爪关节改为 fixed(彻底不动)
=====================================================================
与真机一致性: 真机 ESC8 夹爪"使能后原地锁定", 不参与轨迹跟踪。
仿真里把 finger prismatic 关节改成 fixed → 转换器 merge 进 link7,
零自由度、无 PD 刚度需求(高刚度 PD 在 200Hz 下会数值发散)。

用法: cd ~/IsaacLab && ./isaaclab.sh -p <此脚本>
产出: rl/openarm_right_rl.usd
"""

import os
import sys
import xml.etree.ElementTree as ET

_HERE = os.path.dirname(os.path.abspath(__file__))
_ISA = os.path.abspath(os.path.join(_HERE, ".."))
SRC_URDF = os.path.join(_ISA, "openarm_right.urdf")
DST_URDF = os.path.join(_HERE, "openarm_right_rl.urdf")

from isaaclab.app import AppLauncher  # noqa: E402

parser_app = __import__("argparse").ArgumentParser()
AppLauncher.add_app_launcher_args(parser_app)
args = parser_app.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402


def main():
    tree = ET.parse(SRC_URDF)
    root = tree.getroot()
    n_fixed = 0
    for j in root.iter("joint"):
        if "finger" in j.get("name", ""):
            j.set("type", "fixed")
            for tag in ("axis", "limit", "mimic"):
                for e in list(j.findall(tag)):
                    j.remove(e)
            n_fixed += 1
    ET.ElementTree(root).write(DST_URDF, encoding="utf-8",
                               xml_declaration=True)
    print("[1/2] 夹爪关节改 fixed: %d 个 → %s" % (n_fixed, DST_URDF))

    cfg = UrdfConverterCfg(
        asset_path=DST_URDF,
        usd_dir=_HERE,
        usd_file_name="openarm_right_rl",
        fix_base=True,
        merge_fixed_joints=True,          # 夹爪 fixed → 并入 link7
        force_usd_conversion=True,
        make_instanceable=True,
        self_collision=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force", target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=200.0, damping=5.0)),
    )
    conv = UrdfConverter(cfg)
    print("[2/2] USD: %s" % conv.usd_path)
    simulation_app.close()


if __name__ == "__main__":
    main()
