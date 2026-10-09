import numpy as np

W = r"c:\Users\Yangon\Desktop\舱室绝缘喷涂人形机器人技术方案\喷涂点位划分\work"
xyz = np.load(W + r"\xyz.npy") / 1000.0  # -> meters
rgb = np.load(W + r"\rgb.npy")

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
# 聚类加强筋：先按DBSCAN式简单网格聚类
from scipy import ndimage
# 用体素降采样聚类：对stiff点做连通域
def cluster(p, voxel=0.08, min_pts=300):
    gi = np.floor(p / voxel).astype(int)
    dims = gi.max(axis=0) + 2
    keys = (gi[:, 0] * dims[1] + gi[:, 1]) * dims[2] + gi[:, 2]
    order = np.argsort(keys)
    ks = keys[order]
    uniq, start = np.unique(ks, return_index=True)
    labels = np.zeros(len(p), dtype=int)
    lab = 0
    segs = np.split(order, start[1:])
    for s in segs:
        if len(s) >= min_pts:
            lab += 1
            labels[s] = lab
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
print("obstacle clusters:", lab_o)
for l in range(1, lab_o + 1):
    p = obst[lab_o == l]
    print(f"obst {l}: n={len(p)}, bbox=[{p.min(axis=0).round(2)},{p.max(axis=0).round(2)}]")
