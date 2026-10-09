import numpy as np

W = r"c:\Users\Yangon\Desktop\舱室绝缘喷涂人形机器人技术方案\喷涂点位划分\work"
xyz = np.load(W + r"\xyz.npy") / 1000.0
rgb = np.load(W + r"\rgb.npy")
ORIGIN = np.array([209073.0, 4380.0, 57558.0]) / 1000.0
xyz = xyz - ORIGIN

def mask_of(c):
    return np.all(rgb == np.array(c, dtype=np.uint8), axis=1)

stiff = xyz[mask_of((0, 110, 255))]

# 四面墙定义: (name, axis_normal, plane_val, axis_u, u_range)
walls = [
    ("left",  0, 0.00, 1, (0.00, 5.06)),
    ("right", 0, 3.228, 1, (0.00, 5.06)),
    ("front", 1, 0.00, 0, (0.00, 3.23)),
    ("back",  1, 5.032, 0, (0.00, 3.23)),
]

TOL = 0.12  # 筋凸出墙面约8cm
for name, axn, val, axu, urange in walls:
    m = np.abs(stiff[:, axn] - val) < TOL
    p = stiff[m]
    if len(p) == 0:
        print(f"--- {name} wall: no stiffener"); continue
    u = p[:, axu]
    # 直方图找筋位置（bin=2cm）
    hist, edges = np.histogram(u, bins=np.arange(urange[0], urange[1] + 0.02, 0.02))
    peaks = []
    in_peak = False
    for i, h in enumerate(hist):
        if h > 50 and not in_peak:
            in_peak = True; s = edges[i]
        elif h <= 50 and in_peak:
            in_peak = False; peaks.append((s, edges[i]))
    if in_peak: peaks.append((s, edges[-1]))
    print(f"--- {name} wall: n={len(p)}, stiffener bands (u):")
    for s, e in peaks:
        mm = (u >= s) & (u < e)
        pp = p[mm]
        zmax = pp[:, 2].max(); zmin = pp[:, 2].min()
        protr = pp[:, axn].max() - val if axn == 0 or axn == 1 else 0
        print(f"    u=[{s:.2f},{e:.2f}] center={(s+e)/2:.2f} width={e-s:.2f} "
              f"z=[{zmin:.2f},{zmax:.2f}] n={mm.sum()}")

# 筋的横截面尺寸（以左墙一根为例）
m = (np.abs(stiff[:, 0]) < TOL) & (np.abs(stiff[:, 1] - 0.85) < 0.1)
if m.sum():
    p = stiff[m]
    print("\nleft-wall stiffener@y=0.85 cross-section: x=[%.3f,%.3f] y=[%.3f,%.3f] z=[%.2f,%.2f]" % (
        p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max(),
        p[:, 2].min(), p[:, 2].max()))

# 贴墙障碍物（黑色，距墙<0.6m）
obst = xyz[mask_of((0, 0, 0))]
print("\n=== obstacles near walls (dist<0.6m) ===")
for name, axn, val, axu, urange in walls:
    d = obst[:, axn] - val
    side = 1 if name in ("left", "front") else -1
    m = (d * side > 0) & (np.abs(d) < 0.6)
    p = obst[m]
    if len(p) < 200: continue
    u = p[:, axu]
    print(f"{name}: n={len(p)}, u=[{u.min():.2f},{u.max():.2f}], z=[{p[:,2].min():.2f},{p[:,2].max():.2f}], "
          f"depth=[{np.abs(d[m]).min():.2f},{np.abs(d[m]).max():.2f}]")
    # 障碍物聚类（u方向）
    hist, edges = np.histogram(u, bins=np.arange(urange[0], urange[1] + 0.1, 0.1))
    segs = []
    in_seg = False
    for i, h in enumerate(hist):
        if h > 100 and not in_seg:
            in_seg = True; s = edges[i]
        elif h <= 100 and in_seg:
            in_seg = False; segs.append((s, edges[i]))
    if in_seg: segs.append((s, edges[-1]))
    for s, e in segs:
        mm = (u >= s) & (u < e)
        pp = p[mm]
        if len(pp) < 300: continue
        print(f"    obst u=[{s:.1f},{e:.1f}] z_max={pp[:,2].max():.2f} n={len(pp)}")
