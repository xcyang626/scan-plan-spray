# -*- coding: utf-8 -*-
"""详细分析: 每面墙的窗户/加强筋在墙面坐标系(横向u, 高度h)中的分布"""
import numpy as np

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
FLOOR = 57563.0
labels = {}
for k, c in C.items():
    labels[k] = (rgb[:,0]==c[0])&(rgb[:,1]==c[1])&(rgb[:,2]==c[2])

def clusters_1d(vals, gap=100.0):
    """1D 聚类, 返回 [(lo,hi,n), ...]"""
    v = np.sort(vals); out=[]; s=v[0]; p=v[0]; n=1
    for x in v[1:]:
        if x - p > gap:
            out.append((s,p,n)); s=x; n=1
        else: n+=1
        p=x
    out.append((s,p,n)); return out

def analyze_wall(tag, sel, u_axis):
    """sel: bool mask of points near this wall plane; u_axis: 0->use y as horizontal, 1->use x"""
    p = xyz[sel]
    if len(p) < 500: return
    u = p[:,1] if u_axis=='y' else p[:,0]
    h = p[:,2] - FLOOR
    print(f"\n===== 墙 {tag}: {len(p)} 点, u[{u.min():.0f},{u.max():.0f}] h[{h.min():.0f},{h.max():.0f}]")
    for lbl in ['window','rib','door']:
        m = labels[lbl] & sel
        if m.sum() < 50: continue
        q = xyz[m]
        uu = q[:,1] if u_axis=='y' else q[:,0]
        hh = q[:,2] - FLOOR
        print(f"  {lbl}: n={m.sum()}")
        for lo,hi,n in clusters_1d(uu, gap=150):
            mm = (uu>=lo)&(uu<=hi)
            print(f"     u[{lo:7.1f},{hi:7.1f}] w={hi-lo:6.1f}  h[{hh[mm].min():6.1f},{hh[mm].max():6.1f}] n={mm.sum()}")
        # 肋的厚度方向分布
        if lbl=='rib':
            print(f"     厚度方向范围: {('x' if u_axis=='y' else 'y')} [{q[:,0].min() if u_axis=='y' else q[:,1].min():.0f},{q[:,0].max() if u_axis=='y' else q[:,1].max():.0f}]")

# x-max 墙 (x≈212300): 门+大窗
m = (xyz[:,0] > 212250) & (labels['wall']|labels['window']|labels['door']|labels['rib'])
analyze_wall("x-max (内面 x≈212300, 法向-x, 从室内看横向=y)", m, 'y')
# 肋需要单独看(可能凸出)
m2 = labels['rib'] & (xyz[:,0] > 212150)
analyze_wall("x-max 墙肋(凸出部分, x>212150)", m2, 'y')

# x-min 墙
m3 = (xyz[:,0] < 209250) & (labels['wall']|labels['window']|labels['rib'])
analyze_wall("x-min (x≈209080)", m3, 'y')

# y-min / y-max 墙
m4 = (xyz[:,1] < 4550) & (labels['wall']|labels['window']|labels['rib'])
analyze_wall("y-min (y≈4380)", m4, 'x')
m5 = (xyz[:,1] > 9250) & (labels['wall']|labels['window']|labels['rib'])
analyze_wall("y-max (y≈9430)", m5, 'x')
