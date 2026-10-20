# -*- coding: utf-8 -*-
"""步骤8: 天工3 上身轨迹 -> 官方 wrc_trajectory_executor 的 trajectory.hdf5 格式.

执行器(天工自带, Cython 编译)读取通道(strings 反查):
  master/arm_left_position_align/data    (N,7) 左臂 rad
  master/arm_right_position_align/data   (N,7) 右臂 rad
  puppet/waist_position_align/data       (N,3) 腰 yaw/roll/pitch
  puppet/head_position_align/data        (N,2) 头
  每通道配套 timestamp (N,) 与 is_intervene (N, bool)
  master/end_effector_*_position_align   夹爪(恒定 0=不动)
"到工作初始位"与"停止恢复"由执行器 config 的 prepare/restore 插值段完成 —— 因此
输入用 NO_BLEND 原始轨迹(07 已含限速重定时与限位钳制).
用法: python 08_tiangong_hdf5_export.py <upper.csv> <out.hdf5>
"""
import sys

import h5py
import numpy as np

# 官方样例 trajectory.hdf5 的首帧: 左臂自然下垂位/头习惯位(恒定, 不参与喷涂)
ARM_L_REST = [0.02, 0.153, 0.025, -0.261, -0.164, 0.017, 0.075]
HEAD_REST = [0.0, 0.4]


def build(upper_csv, out_h5):
    a = np.loadtxt(upper_csv, delimiter=",", skiprows=1)
    hdr = open(upper_csv, encoding="utf8").readline().strip().split(",")
    t = a[:, 0]
    cols = {n: i for i, n in enumerate(hdr)}
    waist = a[:, [cols["waist_yaw_joint"], cols["waist_roll_joint"],
                  cols["waist_pitch_joint"]]]
    arm_r = a[:, [cols[n] for n in
                  ("shoulder_pitch_r_joint", "shoulder_roll_r_joint",
                   "shoulder_yaw_r_joint", "elbow_pitch_r_joint",
                   "elbow_yaw_r_joint", "wrist_pitch_r_joint",
                   "wrist_roll_r_joint")]]
    n = len(t)
    arm_l = np.tile(ARM_L_REST, (n, 1))
    head = np.tile(HEAD_REST, (n, 1))
    print(f"[hdf5] {n} 帧 {t[-1]:.1f}s | 右臂范围 "
          f"{arm_r.min():+.2f}~{arm_r.max():+.2f} | 腰 |{waist.min():+.2f}|max")

    def put(g, data):
        d = g.create_group(f"{g.name.rsplit('/',1)[-1]}") if False else g
    with h5py.File(out_h5, "w") as f:
        def chan(prefix, name, data):
            g = f.create_group(f"{prefix}/{name}")
            g.create_dataset("data", data=data)
            g.create_dataset("timestamp", data=t)
            g.create_dataset("is_intervene", data=np.zeros(n, bool))

        chan("master", "arm_left_position_align", arm_l)
        chan("master", "arm_right_position_align", arm_r)
        chan("master", "waist_position_align", waist[:, :1])     # master 腰为 (N,1) 口径
        chan("puppet", "waist_position_align", waist)
        chan("puppet", "head_position_align", head)
        chan("master", "end_effector_left_position_align", np.zeros(n))
        chan("master", "end_effector_right_position_align", np.zeros(n))
        f.create_group("metadata")
    print(f"[hdf5] 输出 {out_h5}")


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
