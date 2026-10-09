# -*- coding: utf-8 -*-
"""分析舱室语义点云: 找出各墙、窗户、加强筋的精确位置"""
import numpy as np, os, sys

PLY = "/home/yang/桌面/轨迹规划/cabin_solution/点云.py/舱室点云/舱室点云_语义标注.ply"

def load_ply(path):
    with open(path, "rb") as f:
        header = b""
        while True:
            line = f.readline(); header += line
            if line.strip() == b"end_header": break
        body = f.read()
    n = int([l for l in header.decode().splitlines()
             if l.startswith("element vertex")][0].split()[-1])
    dt = np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("part_id","<u2"),
                   ("r","u1"),("g","u1"),("b","u1")])
    rec = np.frombuffer(body, dtype=dt, count=n)
    xyz = np.stack([rec["x"], rec["y"], rec["z"]], 1).astype(np.float64)
    rgb = np.stack([rec["r"], rec["g"], rec["b"]], 1).astype(np.int32)
    return xyz, rgb

C = dict(ground=(128,128,128), wall=(150,220,150), door=(178,64,222),
         window=(255,230,0), rib=(0,110,255), other=(0,0,0))

xyz, rgb = load_ply(PLY)
print("总点数:", len(xyz))
print("坐标范围: x[%.0f %.0f] y[%.0f %.0f] z[%.0f %.0f]" % (
    xyz[:,0].min(), xyz[:,0].max(), xyz[:,1].min(), xyz[:,1].max(), xyz[:,2].min(), xyz[:,2].max()))

labels = {}
for k, c in C.items():
    m = (rgb[:,0]==c[0])&(rgb[:,1]==c[1])&(rgb[:,2]==c[2])
    labels[k] = m
    if m.sum():
        p = xyz[m]
        print(f"{k:8s} n={m.sum():8d}  x[{p[:,0].min():.0f},{p[:,0].max():.0f}] "
              f"y[{p[:,1].min():.0f},{p[:,1].max():.0f}] z[{p[:,2].min():.0f},{p[:,2].max():.0f}]")

# 地面高度
gz = xyz[labels['ground'],2]
print("\n地面 z 中位数: %.1f, P5: %.1f, P95: %.1f" % (np.median(gz), np.percentile(gz,5), np.percentile(gz,95)))

# 对每个竖直墙面(法向±x / ±y), 统计墙点、窗点、肋点
def wall_report(name, mask_pts, mask_lbl):
    if mask_pts.sum() < 1000: return
    p = xyz[mask_pts]
    # 平面常数
    if 'X' in name: c = np.median(p[:,0])
    else: c = np.median(p[:,1])
    print(f"\n== 墙 {name}: 平面常数 {c:.1f}, 墙点 {mask_pts.sum()}")
    for lbl in ['window','rib','door']:
        q = xyz[mask_lbl & labels[lbl]]
        if len(q) < 50: continue
        print(f"   {lbl:7s} n={len(q):7d} x[{q[:,0].min():.0f},{q[:,0].max():.0f}] "
              f"y[{q[:,1].min():.0f},{q[:,1].max():.0f}] z[{q[:,2].min():.0f},{q[:,2].max():.0f}]")

xm = np.abs(rgb[:,0]-150)<40
is_wall = labels['wall']
wall_report("x-min (x≈209000)", is_wall & (xyz[:,0] < 209150), is_wall)
wall_report("x-max (x≈212500)", is_wall & (xyz[:,0] > 212350), is_wall)
wall_report("y-min (y≈4000)",   is_wall & (xyz[:,1] < 4150),   is_wall)
wall_report("y-max (y≈9400)",   is_wall & (xyz[:,1] > 9250),   is_wall)

# 窗/肋的整体分布(不区分墙)
for lbl in ['window','rib','door']:
    p = xyz[labels[lbl]]
    if len(p)==0: continue
    print(f"\n{lbl} 全部点聚类粗看:")
    # 按 x 分两组(两面墙)
    for side, m in [("x<210700", p[:,0]<210700), ("x>=210700", p[:,0]>=210700)]:
        if m.sum()<50: continue
        q=p[m]
        print(f"  [{side}] n={m.sum()} x[{q[:,0].min():.0f},{q[:,0].max():.0f}] "
              f"y[{q[:,1].min():.0f},{q[:,1].max():.0f}] z[{q[:,2].min():.0f},{q[:,2].max():.0f}]")
