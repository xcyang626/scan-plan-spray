"""图: 舱室语义点云总览 + 示例墙(左墙)3D点云图"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

W = r"c:\Users\Yangon\Desktop\舱室绝缘喷涂人形机器人技术方案\喷涂点位划分\work"
FIG = r"c:\Users\Yangon\Desktop\舱室绝缘喷涂人形机器人技术方案\喷涂点位划分\figs"
import os
os.makedirs(FIG, exist_ok=True)

xyz = np.load(W + r"\xyz.npy") / 1000.0
rgb = np.load(W + r"\rgb.npy")
ORIGIN = np.array([209073.0, 4380.0, 57558.0]) / 1000.0
xyz = xyz - ORIGIN

def mask_of(c):
    return np.all(rgb == np.array(c, dtype=np.uint8), axis=1)

rng = np.random.default_rng(7)

# ---------- 图1: 舱室语义点云总览 ----------
fig = plt.figure(figsize=(11, 4.6))
ax = fig.add_subplot(111, projection="3d")

specs = [
    ((150, 220, 150), "墙面(需喷涂)", 120000, (0.55, 0.80, 0.55), 0.05),
    ((0, 110, 255),   "加强筋(需喷涂)", 60000, (0.0, 0.45, 1.0), 0.5),
    ((0, 0, 0),       "障碍物(不喷涂)", 50000, (0.25, 0.25, 0.25), 0.08),
    ((255, 230, 0),   "窗(不喷涂)", 12000, (1.0, 0.90, 0.0), 0.9),
    ((178, 64, 222),  "门(不喷涂)", 4132, (0.70, 0.25, 0.87), 0.9),
]
for c, label, nmax, col, a in specs:
    m = mask_of(c)
    p = xyz[m]
    if len(p) > nmax:
        p = p[rng.choice(len(p), nmax, replace=False)]
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=1.2, c=[col], alpha=a,
               depthshade=False, label=f"{label} ({m.sum()//1000}k点)", linewidths=0)

# 灰色地面
gm = mask_of((128, 128, 128))
p = xyz[gm][::12]
ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=0.8, c=(0.78, 0.78, 0.78), alpha=0.25, depthshade=False, linewidths=0)

ax.set_xlabel("x (m)", fontsize=9, labelpad=1)
ax.set_ylabel("y (m)", fontsize=9, labelpad=1)
ax.set_zlabel("z (m)", fontsize=9, labelpad=1)
ax.set_title("舱室语义点云地图（绿=墙面 蓝=加强筋 黑=障碍物 黄=窗 紫=门）", fontsize=12)
ax.view_init(elev=32, azim=-58)
ax.set_box_aspect((3.23, 5.06, 3.92))
ax.tick_params(labelsize=8)
ax.legend(loc="upper left", fontsize=8, framealpha=0.9, markerscale=6)
ax.set_zlim(0, 3.92)
plt.tight_layout()
plt.savefig(FIG + r"\fig_scene.png", dpi=200)
plt.close()
print("fig_scene.png done")

# ---------- 图2: 左墙3D点云(斜视) + 墙面坐标系 ----------
# 左墙 x≈0, y∈[0,5.06], z∈[0,3.92]; 凸出物 x>0
fig = plt.figure(figsize=(11, 4.8))
ax = fig.add_subplot(111, projection="3d")

def wall_points(c, tol=0.35, nmax=200000):
    m = mask_of(c) & (np.abs(xyz[:, 0]) < tol)
    p = xyz[m]
    if len(p) > nmax:
        p = p[rng.choice(len(p), nmax, replace=False)]
    return p

p = wall_points((150, 220, 150), 0.05, 130000)
ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=1.0, c=(0.62, 0.85, 0.62), alpha=0.30, depthshade=False, linewidths=0)
p = wall_points((0, 110, 255), 0.15, 60000)
ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=2.2, c=(0.0, 0.42, 1.0), alpha=0.9, depthshade=False, linewidths=0)
p = wall_points((255, 230, 0), 0.20, 12000)
ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=2.2, c=(1.0, 0.88, 0.0), alpha=0.95, depthshade=False, linewidths=0)
p = wall_points((0, 0, 0), 0.60, 40000)
ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=1.4, c=(0.25, 0.25, 0.25), alpha=0.30, depthshade=False, linewidths=0)

# 2D投影坐标系 (u=y, v=z) 画于墙面左下角
ax.plot([0.05, 0.05], [0.20, 1.05], [0.15, 0.15], "k-", lw=1.8)
ax.plot([0.05, 0.05], [0.20, 0.20], [0.15, 1.00], "k-", lw=1.8)
ax.text(0.05, 1.15, 0.15, "u(y)", fontsize=10, style="italic")
ax.text(0.05, 0.20, 1.08, "v(z)", fontsize=10, style="italic")

ax.set_xlabel("x (m) 墙面法向", fontsize=9, labelpad=1)
ax.set_ylabel("y (m) 墙面横向", fontsize=9, labelpad=1)
ax.set_zlabel("z (m) 高度", fontsize=9, labelpad=1)
ax.set_title("示例墙面（左墙）三维语义点云（内侧面视角）：墙面(绿)·加强筋3根(蓝)·窗(黄)·贴墙障碍物(黑)", fontsize=11.5)
ax.view_init(elev=10, azim=14)   # 相机位于舱室内侧(+x), 正对墙面内表面
ax.set_box_aspect((1.2, 5.06, 3.92))
ax.set_xlim(-0.05, 0.75)
ax.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig(FIG + r"\fig_wall3d.png", dpi=200)
plt.close()
print("fig_wall3d.png done")
