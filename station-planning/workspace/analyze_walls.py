import os
import sys
from pathlib import Path

import numpy as np

# 数据目录: 优先环境变量 STATION_DATA, 未设时默认脚本同级 "数据" 目录(仓库不携带点云缓存)
W = Path(os.environ.get("STATION_DATA", Path(__file__).resolve().parent / "数据"))
for f in ("xyz.npy", "rgb.npy"):
    if not (W / f).exists():
        sys.exit(f"[路径] 缺少数据文件 {W / f}: 请将点云缓存(xyz.npy/rgb.npy)放入该目录, "
                 f"或设置环境变量 STATION_DATA 指向数据目录")

xyz = np.load(W / "xyz.npy") / 1000.0  # -> meters
rgb = np.load(W / "rgb.npy")

# 局部坐标系（m）：原点=(209073, 4380, 57558)mm 即左墙-前墙-地面交点
ORIGIN = np.array([209073.0, 4380.0, 57558.0]) / 1000.0
xyz = xyz - ORIGIN

def mask_of(c):
    return np.all(rgb == np.array(c, dtype=np.uint8), axis=1)

wall = xyz[mask_of((150, 220, 150))]
stiff = xyz[mask_of((0, 110, 255))]
obst = xyz[mask_of((0, 0, 0))]
win = xyz[mask_of((255, 230, 0))]
door = xyz[mask_of((178, 64, 222))]

print("room inner size: X(0..%.2f) Y(0..%.2f) Z(0..%.2f)" % (
    wall[:, 0].max(), wall[:, 1].max(), wall[:, 2].max()))

# 分离平面：厚度阈值
def near(v, val, tol):
    return np.abs(v - val) < tol

# 左墙 X~0, 右墙 X~?, 前墙 Y~0, 后墙 Y~?, 天花板 Z~?
print("\n=== wall planes (green) ===")
for nm, ax, val, tol in [("left X=0", 0, 0.0, 0.06), ("front Y=0", 1, 0.0, 0.06),
                          ("back Y~?", 1, wall[:, 1].max(), 0.10), ("ceil Z~?", 2, wall[:, 2].max(), 0.08)]:
    m = near(wall[:, ax], val, tol)
    p = wall[m]
    if len(p) == 0:
        print(nm, ": none"); continue
    print(f"{nm}: n={len(p)}, extent: ax{[i for i in range(3) if i != ax]} range "
          f"[{p[:, [i for i in range(3) if i != ax]].min(axis=0).round(2)}, "
          f"{p[:, [i for i in range(3) if i != ax]].max(axis=0).round(2)}]")

# 右墙检测：X直方图峰在 3.15 和 3.23
for xval in [3.15, 3.2, 3.228]:
    m = near(wall[:, 0], xval, 0.05)
    print(f"right wall X~{xval}: n={m.sum()}")

print("\n=== stiffener (blue) plane assignment ===")
# 贴墙的加强筋：按其质心最近平面分组
planes = {"left(X=0)": (0, 0.0), "right(X~3.23)": (0, 3.228), "front(Y=0)": (1, 0.0),
          "back(Y~5.03)": (1, 5.032), "ceil(Z~3.92)": (2, 3.918)}
# 聚类加强筋: 体素化后做真连通域(scipy.ndimage.label, 6-连通)
from scipy import ndimage

def cluster(p, voxel=0.08, min_pts=300):
    """体素连通域聚类: 点云按 voxel 体素化 -> 6-连通连通域 -> 逐点回填标签,
    仅保留点数 >= min_pts 的连通域 (1..N), 噪声点标签为 0"""
    gi = np.floor(p / voxel).astype(int)
    gi -= gi.min(axis=0)                 # 平移到正象限, 缩小网格
    dims = gi.max(axis=0) + 1
    occ = np.zeros(dims, dtype=bool)
    occ[gi[:, 0], gi[:, 1], gi[:, 2]] = True
    lab_grid, n_lab = ndimage.label(occ)  # 默认结构元 = 6-连通
    vlab = lab_grid[gi[:, 0], gi[:, 1], gi[:, 2]]
    labels = np.zeros(len(p), dtype=int)
    lab = 0
    for v in range(1, n_lab + 1):
        m = vlab == v
        if m.sum() >= min_pts:
            lab += 1
            labels[m] = lab
    return labels

lab = cluster(stiff)
print("stiffener clusters:", lab.max())
for l in range(1, lab.max() + 1):
    p = stiff[lab == l]
    c = p.mean(axis=0)
    dists = {nm: abs(c[ax] - val) for nm, (ax, val) in planes.items()}
    near_plane = min(dists, key=dists.get)
    print(f"cluster {l}: n={len(p)}, centroid=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f}), "
          f"bbox=[{p.min(axis=0).round(2)},{p.max(axis=0).round(2)}] -> {near_plane} (d={dists[near_plane]:.2f})")

print("\n=== window (yellow) ===")
if len(win):
    sub = win[::5]
    for nm, (ax, val) in planes.items():
        m = np.abs(sub[:, ax] - val) < 0.08
        if m.sum() > 10:
            p = sub[m]
            others = [i for i in range(3) if i != ax]
            print(f"window on {nm}: n~{m.sum()*5}, extent "
                  f"[{p[:, others].min(axis=0).round(2)},{p[:, others].max(axis=0).round(2)}]")

print("\n=== door (purple) ===")
print("door bbox:", door.min(axis=0).round(2), door.max(axis=0).round(2), "n=", len(door))

print("\n=== obstacles (black) clusters ===")
lab_o = cluster(obst, voxel=0.15, min_pts=500)
print("obstacle clusters:", lab_o.max())
for l in range(1, lab_o.max() + 1):
    p = obst[lab_o == l]
    print(f"obst {l}: n={len(p)}, bbox=[{p.min(axis=0).round(2)},{p.max(axis=0).round(2)}]")
