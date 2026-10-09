# -*- coding: utf-8 -*-
"""分析RIBS各零件横截面形状：区分 L型角钢加强筋 与 方形/圆形管子。"""
import numpy as np
BASE = r"c:\Users\Yangon\Desktop\舱室绝缘喷涂人形机器人技术方案\点云.py"
PLY = BASE + r"\舱室点云\舱室装配体点云.ply"
RIBS = [232,233,234,235,236,237,238,239,240,241,242]

with open(PLY,"rb") as f:
    while f.readline().strip()!=b"end_header": pass
    body=f.read()
dt=np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("part_id","<u2"),
             ("r","u1"),("g","u1"),("b","u1")])
rec=np.frombuffer(body,dtype=dt)
xyz=np.stack([rec["x"],rec["y"],rec["z"]],axis=1)
pid=rec["part_id"]

def cross_section(pts, z0, grid=2.0):
    """取 z0 起 ±半厚切片, 投影XY平面, 返回占据掩码。"""
    s=pts[(pts[:,2]>=z0-10)&(pts[:,2]<=z0+10)]
    if len(s)<5: return None
    u0,u1=s[:,0].min(),s[:,0].max(); v0,v1=s[:,1].min(),s[:,1].max()
    iu=np.clip(((s[:,0]-u0)/grid).astype(int),0,9999)
    iv=np.clip(((s[:,1]-v0)/grid).astype(int),0,9999)
    occ=np.zeros((iu.max()+1,iv.max()+1),bool); occ[iu,iv]=True
    return occ

print(f"{'id':>4} {'点数':>7} {'高mm':>7}  距顶1m处截面: 尺寸mm 填充比")
for p in RIBS:
    pts=xyz[pid==p]; h=pts[:,2].max()-pts[:,2].min()
    ztop=pts[:,2].max()
    occ=cross_section(pts, ztop-1000)   # 顶部往下1m的截面
    tag='无点'
    if occ is not None:
        R,C=np.nonzero(occ)
        r0,r1,c0,c1=R.min(),R.max(),C.min(),C.max()
        w=(r1-r0+1)*2.0; d=(c1-c0+1)*2.0
        box=(r1-r0+1)*(c1-c0+1)
        fill=occ.sum()/max(box,1)
        tag=f"{w:.0f}×{d:.0f}mm 填充比={fill:.2f}"
    print(f"{p:>4} {pts.shape[0]:>7} {h:7.0f}  {tag}")