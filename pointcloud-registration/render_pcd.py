# -*- coding: utf-8 -*-
"""渲染舱室点云PLY为多视角PNG预览图。"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# 优先使用 Windows 中文字体，避免标题中文乱码
zh = next((f for f in fm.findSystemFonts(fontpaths=None, fontext="ttf")
           if any(k in f for k in ("yahei", "msyh", "simhei", "simsun"))), None)
if zh:
    matplotlib.rcParams["font.sans-serif"] = [fm.FontProperties(fname=zh).get_name()]
matplotlib.rcParams["axes.unicode_minus"] = False

PLY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "舱室点云", "舱室装配体点云.ply")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "舱室点云", "舱室点云预览.png")
MAX_PLOT = 300_000  # 绘图点数上限（过多会卡）

# 读取 binary PLY
with open(PLY, "rb") as f:
    header = b""
    while True:
        line = f.readline()
        header += line
        if line.strip() == b"end_header":
            break
    body = f.read()
dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("part_id", "<u2"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
rec = np.frombuffer(body, dtype=dt)
xyz = np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float64)
rgb = np.stack([rec["r"], rec["g"], rec["b"]], axis=1)
rgb_n = rgb / 255.0

# 用均匀间隔采样降低点数，保证覆盖面
if len(xyz) > MAX_PLOT:
    idx = np.linspace(0, len(xyz) - 1, MAX_PLOT).astype(np.int64)
    xyz, rgb_n = xyz[idx], rgb_n[idx]

def ax_style(ax, az, el, title):
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_box_aspect((np.ptp(xyz[:,0]), np.ptp(xyz[:,1]), np.ptp(xyz[:,2])))
    ax.set_xlabel("X"), ax.set_ylabel("Y"), ax.set_zlabel("Z")
    ax.view_init(elev=el, azim=az)

fig = plt.figure(figsize=(18, 12))
# 1) 等轴三维视图
ax = fig.add_subplot(2, 2, 1, projection="3d")
ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=rgb_n, s=0.4, depthshade=False)
ax_style(ax, 30, 25, "三维俯射视角 (az=30, el=25)")
# 2) 正视(沿X/墙面方向)
ax = fig.add_subplot(2, 2, 2, projection="3d")
ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=rgb_n, s=0.4, depthshade=False)
ax_style(ax, 0, 0, "正视投影 (az=0, el=0)")
# 3) 俯视图 XY
ax = fig.add_subplot(2, 2, 3, projection="3d")
ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=rgb_n, s=0.4, depthshade=False)
ax_style(ax, -90, 90, "俯视图 XY (az=-90, el=90)")
# 4) 侧视图 YZ
ax = fig.add_subplot(2, 2, 4, projection="3d")
ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=rgb_n, s=0.4, depthshade=False)
ax_style(ax, 90, 0, "侧视图 YZ (az=90, el=0)")
plt.tight_layout()
plt.savefig(OUT, dpi=110)
print("已生成:", OUT)
print("点数(采样后):", len(xyz))