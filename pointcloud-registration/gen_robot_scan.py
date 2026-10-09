# -*- coding: utf-8 -*-
"""重新生成"机器人扫描"风格的黑白点云，覆盖 舱室点云_原始扫描黑白.ply。

与原有黑白图（与语义参考图同源同坐标、同为 502 万点）不同，
新生成的黑白图模拟机器人真实建图结果：
  - 采样密度更稀疏（点数与原来不同，约 120 万点）
  - 坐标转换到机器人局部坐标系（先平移至原点再施加轻微扫描方向偏差），
    与语义参考图坐标系明显不同，故必须以配准回归到语义图坐标系
  - 叠加入测量噪声（σ=0.5mm）模拟扫描误差
  - 灰度强度与黑白预览一致的确定性伪强度

同时把生成时所用的真实刚体变换保存到 gt_robot_scan.npy，
供配准模块校验"恢复的变换 vs 真值"。
"""
import os, importlib.util
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "rl", os.path.join(BASE, "register_and_label.py"))
rl = importlib.util.module_from_spec(spec); spec.loader.exec_module(rl)

SRC   = os.path.join(BASE, "舱室点云", "舱室装配体点云.ply")
OUT   = os.path.join(BASE, "舱室点云", "舱室点云_原始扫描黑白.ply")
GTPATH = os.path.join(BASE, "gt_robot_scan.npy")

rng = np.random.default_rng(2024)

# 1) 稀疏采样：点数与原来明显不同
xyz, rec, rgb = rl.load_ply(SRC)
n_src = len(xyz)
sel = rl.voxel_subsample(xyz, 4.0, None, 7)          # 4mm 体素保留细节
TARGET = 1_200_000
if len(sel) > TARGET:
    sel = rng.choice(sel, TARGET, replace=False)
sel = np.sort(sel)
P = xyz[sel]
print("源点数", n_src, "-> 采样点数", len(P))

# 2) 叠加测量噪声(0.5mm)
P = P + rng.normal(0.0, 0.5, P.shape)

# 3) 刚体变换到机器人局部坐标系：去中心(参考坐标约 2e5 附近) → 旋转 → 平移
center = xyz.mean(0)
ax = np.deg2rad(np.array([3.0, -2.0, 4.0]))
R = rl.exp_so3([ax[0],0,0]) @ rl.exp_so3([0,ax[1],0]) @ rl.exp_so3([0,0,ax[2]])
t = np.array([300.0, -200.0, 150.0])
S_robot = (P - center) @ R.T + t
print("旋转(°)", np.round(np.degrees(ax),2), " 平移(mm)", t)
print("机器人坐标范围 X[%.0f,%.0f] Y[%.0f,%.0f] Z[%.0f,%.0f]"
      % (S_robot[:,0].min(), S_robot[:,0].max(), S_robot[:,1].min(),
         S_robot[:,1].max(), S_robot[:,2].min(), S_robot[:,2].max()))

# 4) 灰度强度 + 写全覆盖原黑白图
base_i = 200
inten = base_i + rng.integers(-55, 30, len(S_robot))
inten = np.clip(inten, 35, 255).astype(np.uint8)
gray = np.tile(inten[:, None], (1, 3))
rl.write_ply(OUT, S_robot, gray, rec["part_id"][sel])

# 5) 保存真实变换(参考→机器人, 4x4)
GT = np.eye(4); GT[:3,:3] = R
T_robot = np.eye(4); T_robot[:3,3] = - center      # 先平移去中心
GT = GT @ T_robot
GT[:3,3] = GT[:3,3] + t
np.save(GTPATH, GT)
print("已保存真实变换:", GTPATH)
print("完成")