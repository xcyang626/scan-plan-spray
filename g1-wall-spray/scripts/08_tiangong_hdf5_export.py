# -*- coding: utf-8 -*-
"""步骤8: 天工3 上身轨迹 -> 官方 wrc_trajectory_executor 的 trajectory.hdf5 格式.

执行器(天工自带, Cython 编译)读取通道(strings 反查):
  master/arm_left_position_align/data    (N,7) 左臂 rad
  master/arm_right_position_align/data   (N,7) 右臂 rad
  master/waist_position_align/data       (N,3) 腰 yaw/roll/pitch
  puppet/waist_position_align/data       (N,3) 腰 yaw/roll/pitch(同一份腰数据, 形状一致)
  puppet/head_position_align/data        (N,2) 头
  每通道配套 timestamp (N,) 与 is_intervene (N, bool)
  master/end_effector_*_position_align   夹爪(恒定 0=不动)
"到工作初始位"与"停止恢复"由执行器 config 的 prepare/restore 插值段完成 —— 因此
输入用 NO_BLEND 原始轨迹(07 已含限速重定时与限位钳制).

骨盆高度(下蹲)通道说明:
  执行器的通道集里**没有骨盆高度通道**(官方样例 trajectory.hdf5 只有
  腰/头/双臂/夹爪), 因此骨盆高不能塞进执行通道 —— 强行加一个执行器不认的
  通道只会"看似有数据、实际不下发". 这里把 upper.csv 的 pelvis_z_m 原样写进
  metadata/pelvis_z_m, 并在 metadata/notes 里显式标注"骨盆高未纳入执行通道",
  同时在终端打印提醒(天工自带控制器走 /hric/robot/stand_cmd 的 linear.z,
  范围 0.6~0.98m; 或走整腿位置流). 不让它无声丢失.

用法: python 08_tiangong_hdf5_export.py <upper.csv> <out.hdf5>
"""
import sys

import h5py
import numpy as np

# 官方样例 trajectory.hdf5 的首帧: 左臂自然下垂位/头习惯位(恒定, 不参与喷涂)
ARM_L_REST = [0.02, 0.153, 0.025, -0.261, -0.164, 0.017, 0.075]
# 头习惯位: 必须落在 head_pitch_joint 的 URDF 限位 (-0.8727, 0.2618) 内,
# 旧值 0.4 越界 0.138 rad(执行器侧会被限位钳制/直接报错)
HEAD_REST = [0.0, 0.26]

NOTES = ("骨盆高(pelvis_z_m)未纳入执行通道: 官方 trajectory.hdf5 的通道集只有 "
         "腰/头/双臂/夹爪, 执行器不认骨盆高通道; 数值另存于 metadata/pelvis_z_m. "
         "天工自带控制器下蹲请走 /hric/robot/stand_cmd 的 twist.linear.z "
         "(范围 0.6~0.98m) 或整腿位置流, 否则底部几行会整体偏高.")


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
    # 骨盆高(下蹲指令): 执行器无对应通道 -> 存 metadata 并标注
    pz = a[:, cols["pelvis_z_m"]] if "pelvis_z_m" in cols else None
    print(f"[hdf5] {n} 帧 {t[-1]:.1f}s | 右臂范围 "
          f"{arm_r.min():+.2f}~{arm_r.max():+.2f} | 腰 |{waist.min():+.2f}|max")
    if pz is not None:
        print(f"[hdf5] 骨盆高 {pz.min():.3f}~{pz.max():.3f}m —— **未纳入执行通道**, "
              f"已写入 metadata/pelvis_z_m(见 notes)")

    with h5py.File(out_h5, "w") as f:
        def chan(prefix, name, data):
            g = f.create_group(f"{prefix}/{name}")
            g.create_dataset("data", data=data)
            g.create_dataset("timestamp", data=t)
            g.create_dataset("is_intervene", data=np.zeros(n, bool))

        chan("master", "arm_left_position_align", arm_l)
        chan("master", "arm_right_position_align", arm_r)
        chan("master", "waist_position_align", waist)        # 腰: master/puppet 同形状 (N,3)
        chan("puppet", "waist_position_align", waist)
        chan("puppet", "head_position_align", head)
        chan("master", "end_effector_left_position_align", np.zeros(n))
        chan("master", "end_effector_right_position_align", np.zeros(n))
        meta = f.create_group("metadata")
        if pz is not None:
            meta.create_dataset("pelvis_z_m", data=pz)
        meta.create_dataset("notes", data=NOTES)
    print(f"[hdf5] 输出 {out_h5}")


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
