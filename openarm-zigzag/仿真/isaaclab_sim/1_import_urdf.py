#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1_import_urdf.py —— OpenArm 右臂 URDF → USD（只需运行一次）
===========================================================
用法:
    cd ~/IsaacLab && ./isaaclab.sh -p \
        ~/桌面/openarm实验/仿真/isaaclab_sim/1_import_urdf.py

产出: 仿真/isaaclab_sim/openarm_right.usd
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ARM_URDF, USD_DIR, USD_PATH  # noqa: E402

parser = argparse.ArgumentParser(description="OpenArm 右臂 URDF 导入 Isaac Sim")
parser.add_argument("--force", action="store_true", help="覆盖已有 USD 重新导入")
AppLauncherArgs = argparse.ArgumentParser
# AppLauncher 参数（默认 headless，导入无需视口）
from isaaclab.app import AppLauncher  # noqa: E402
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

usd_path = USD_PATH
if os.path.exists(usd_path) and not args.force:
    print("USD 已存在: %s（--force 可强制重导）" % usd_path)
    simulation_app.close()
    raise SystemExit(0)

cfg = UrdfConverterCfg(
    asset_path=ARM_URDF,
    usd_dir=USD_DIR,
    usd_file_name=os.path.splitext(os.path.basename(usd_path))[0],
    fix_base=True,                            # 基座固定
    merge_fixed_joints=True,                  # 合并 fixed 关节（body/link0 等）
    force_usd_conversion=True,
    make_instanceable=True,
    convert_mimic_joints_to_normal_joints=True,   # 夹爪 mimic → 双独立指关节
    self_collision=False,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=200.0, damping=5.0),
    ),
)
converter = UrdfConverter(cfg)
print("=" * 60)
print("导入完成: %s" % converter.usd_path)
print("=" * 60)
simulation_app.close()
