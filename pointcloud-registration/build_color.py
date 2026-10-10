# -*- coding: utf-8 -*-
"""舱室点云语义着色(门窗边界版)：
  地面=灰, 四面墙=浅绿, 门开洞边界5mm=紫, 窗开洞边界5mm=黄,
  加强筋(L型)=蓝, 其他=黑。
门/窗不再整洞填充，而是给开洞边缘~5mm一圈的墙体真实点着色，供后续识别。
"""
import numpy as np
from collections import deque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

BASE = r"c:\Users\Yangon\Desktop\舱室绝缘喷涂人形机器人技术方案\点云.py"
PLY = BASE + r"\舱室点云\舱室装配体点云.ply"
OUTPLY = BASE + r"\舱室点云\舱室点云_语义标注.ply"
OUTPNG = BASE + r"\舱室点云\舱室点云_语义标注.png"
MAXPLOT = 400_000

GROUND = [0]
WALLS = [264, 287, 275, 276]
RIBS = [234, 235, 236, 237, 238, 239, 240, 241, 242]   # 232为方形空心管,不是L型角钢加强筋
C = dict(ground=(128,128,128), wall=(150,220,150), door=(178,64,222),
         window=(255,230,0), rib=(0,110,255), other=(0,0,0))
RING = 10.0         # 门窗边缘外扩着色带厚度(mm, 向外扩1cm)

with open(PLY, "rb") as f:
    while f.readline().strip() != b"end_header":
        pass
    body = f.read()
dt = np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("part_id","<u2"),
               ("r","u1"),("g","u1"),("b","u1")])
rec = np.frombuffer(body, dtype=dt)
xyz = np.stack([rec["x"],rec["y"],rec["z"]],axis=1)
pid = rec["part_id"]

def conn_label(mask):
    H,W=mask.shape; lab=np.zeros_like(mask,dtype=np.int32); cur=0
    for i in range(H):
        for j in range(W):
            if mask[i,j] and not lab[i,j]:
                cur+=1; lab[i,j]=cur; dq=deque([(i,j)])
                while dq:
                    x,y=dq.popleft()
                    for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)):
                        nx,ny=x+dx,y+dy
                        if 0<=nx<H and 0<=ny<W and mask[nx,ny] and not lab[nx,ny]:
                            lab[nx,ny]=cur; dq.append((nx,ny))
    return lab,cur

GRID=500
def find_openings(p, axis):
    m=xyz[pid==p]; lo=m.min(0); hi=m.max(0)
    axs=[i for i in range(3) if i!=axis]
    U=m[:,axs[0]]; V=m[:,axs[1]]
    ## histogram2d(U,V): 结果矩阵 H 行=U, 列=V
    occ,_,_=np.histogram2d(U,V,bins=GRID); mat=occ>0
    empty=~mat; lab,_=conn_label(empty)
    border=set(np.unique(np.concatenate([lab[0],lab[-1],lab[:,0],lab[:,-1]])))-{0}
    inner_lbl=set(np.unique(lab))-border; inner_lbl.discard(0)
    hl,nh=conn_label(np.isin(lab,list(inner_lbl)))
    cell=(U.max()-U.min())/GRID*(V.max()-V.min())/GRID
    res=[]
    for h in range(1,nh+1):
        rows,cols=np.where(hl==h)          # rows=U轴索引, cols=V轴索引
        r0,r1,c0,c1=rows.min(),rows.max(),cols.min(),cols.max()
        if ((r1-r0+1)*(c1-c0+1))*cell < 40000: continue
        # 正确映射：U 用行索引 rows, V 用列索引 cols
        u0=U.min()+(r0+0.5)*(U.max()-U.min())/GRID
        u1=U.min()+(r1+0.5)*(U.max()-U.min())/GRID
        v0=V.min()+(c0+0.5)*(V.max()-V.min())/GRID
        v1=V.min()+(c1+0.5)*(V.max()-V.min())/GRID
        res.append((u0,u1,v0,v1))
    return res, axs, lo, hi

# ---- 汇总所有门窗开洞：axis(固定轴) fixed(墙平面坐标) u0,u1,v0,v1 wall cat ----
floor_top = float(xyz[pid==0, 2].max())           # 地面顶面Z(落地基准)
openings=[]
for p,axis in [(264,0),(287,0),(275,1),(276,1)]:
    holes, axs, lo, hi = find_openings(p,axis)
    for (u0,u1,v0,v1) in holes:
        openings.append((axis, lo[axis]+1, u0,u1, v0,v1, p))
print(f"地面顶面Z={floor_top:.0f}  开洞总数={len(openings)}  各洞底部离地(mm): " +
      ", ".join(f"{o[4]-floor_top:.0f}({'XYZ'[o[0]]}墙{o[6]})" for o in openings))
if not openings:
    # 开洞列表为空：后续 min(openings) 必然 ValueError，直接中文报错退出
    raise SystemExit("[错误] 未检出任何开洞, 请检查墙面 part_id 配置与开洞检测参数")
# 门=底部离地低于阈值的全部落地开洞；其余=窗。
# 原实现只取"底部离地最近"的单一洞为门, 多门场景会漏标；
# 阈值 450mm 与 register_and_label.detect_door_frame 的 tol_floor(门触地容差)口径一致
DOOR_FLOOR_TOL = 450.0
filled = [(*o, ("door" if abs(o[4]-floor_top) <= DOOR_FLOOR_TOL else "window"))
          for o in openings]

# ---- 原有点基础着色 ----
color=np.zeros((len(pid),3),dtype=np.uint8); color[:]=C["other"]
for p in GROUND: color[pid==p]=C["ground"]
for p in WALLS:  color[pid==p]=C["wall"]
for p in RIBS:   color[pid==p]=C["rib"]

# ---- 门窗边界 轮廓距离法 ----
# 先用点拟合一圈洞口轮廓(多边形折线)，再计算墙点到轮廓的真实欧氏距离，≤RING着色。
PY = 2.0                     # 网格每格 = 2mm(用于稳健识别洞口空洞；轮廓距离为点级真实mm)
MARG = 25.0                  # 空洞四周外扩范围(mm)

def _neighbor4(mask):
    """返回与 mask 相邻的周围1像素(非mask)掩码(即原mask外扩一圈)。"""
    pad = np.zeros((mask.shape[0]+2, mask.shape[1]+2), bool)
    pad[1:-1,1:-1] = mask
    d = np.zeros_like(pad, bool)
    d[:-1,:] |= pad[1:,:]        # 下
    d[1:,:]  |= pad[:-1,:]       # 上
    d[:,:-1] |= pad[:,1:]        # 右
    d[:,1:]  |= pad[:,:-1]       # 左
    return d[1:-1,1:-1] & ~mask

def boundary_points(wall_part, axis, u0,u1,v0,v1):
    """提取门窗轮廓坐标→闭合多边形→对每个墙体点算到各边(连续线段)的欧氏距离，
    ≤RING(向外扩)即染色。用10mm粗网格识别空洞(稀疏点云下墙面可连成闭围)、
    点到线段距离点级连续，同一圈的点均匀染色。"""
    MARGY=200.0          # 空洞识别外扩(mm)：保证洞四周有墙体闭围
    PYB=10.0             # 空洞识别网格(mm)
    sel = pid == wall_part
    m  = xyz[sel]
    axs = [i for i in range(3) if i != axis]
    U = m[:,axs[0]].astype(np.float64); V = m[:,axs[1]].astype(np.float64)
    U0,U1 = u0-MARGY, u1+MARGY
    V0,V1 = v0-MARGY, v1+MARGY
    inb = (U>=U0)&(U<=U1)&(V>=V0)&(V<=V1)
    arr_idx = np.nonzero(sel)[0][inb]
    if len(arr_idx)==0:
        return arr_idx
    nU=int(np.ceil((U1-U0)/PYB))+1; nV=int(np.ceil((V1-V0)/PYB))+1
    iu=np.clip(((U[inb]-U0)/PYB).astype(np.int32),0,nU-1)
    iv=np.clip(((V[inb]-V0)/PYB).astype(np.int32),0,nV-1)
    occ=np.zeros((nU,nV),bool); occ[iu,iv]=True
    empty=~occ
    from collections import deque as _dq
    lab=np.zeros_like(empty,np.int32); cur=0
    for i in range(nU):
        for j in range(nV):
            if empty[i,j] and lab[i,j]==0:
                cur+=1; lab[i,j]=cur; q=_dq([(i,j)])
                while q:
                    a,b=q.popleft()
                    for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)):
                        na,nb=a+dx,b+dy
                        if 0<=na<nU and 0<=nb<nV and empty[na,nb] and lab[na,nb]==0:
                            lab[na,nb]=cur; q.append((na,nb))
    bc=set(np.unique(np.concatenate([lab[0],lab[-1],lab[:,0],lab[:,-1]]))); bc.discard(0)
    internal = empty & ~np.isin(lab, list(bc))     # 洞口内部空洞
    hlb,hc = _blob(internal)
    if hc==0: return arr_idx[np.zeros(len(U[inb]),bool)]
    main = hlb == (np.bincount(hlb.ravel())[1:].argmax()+1)
    adj=np.zeros_like(internal)
    adj[1:,:]|=main[:-1,:]; adj[:-1,:]|=main[1:,:]
    adj[:,1:]|=main[:,:-1]; adj[:,:-1]|=main[:,1:]
    ring = occ & adj
    cy,cx = np.nonzero(ring)
    pts = np.column_stack([U0+(cy+0.5)*PYB, V0+(cx+0.5)*PYB])
    if len(pts)<3: return arr_idx[np.zeros(len(U[inb]),bool)]
    # 绕质心角度排序成闭合多边形(门窗孔为凸形状)
    cen=pts.mean(0)
    pts=pts[np.argsort(np.arctan2(pts[:,1]-cen[1], pts[:,0]-cen[0]))]
    A=pts; B=np.vstack([pts[1:],pts[0]])
    AB=B-A; AB2=np.einsum("ij,ij->i",AB,AB); mS=len(A)
    QU=U[inb]; QV=V[inb]; nq=len(QU); keep=np.zeros(nq,bool); BLK=800
    for s in range(0,nq,BLK):
        e=min(s+BLK,nq); K=e-s
        q=np.stack([QU[s:e],QV[s:e]],1)
        qA=q[:,None,:]-A[None,:,:]
        t=np.clip(np.einsum("kmi,mi->km",qA,AB)/AB2[None,:],0,1)
        proj=A[None,:,:]+t[...,None]*AB[None,:,:]
        dd=np.sqrt(((q[:,None,:]-proj)**2).sum(2)).min(1)
        keep[s:e]=dd<=RING
    return arr_idx[keep]

def _blob(mask):
    H,W=mask.shape; lab=np.zeros_like(mask,np.int32); cur=0
    for i in range(H):
        for j in range(W):
            if mask[i,j] and lab[i,j]==0:
                cur+=1; lab[i,j]=cur; dq=[(i,j)]
                while dq:
                    a,b=dq.pop()
                    for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)):
                        na,nb=a+dx,b+dy
                        if 0<=na<H and 0<=nb<W and mask[na,nb] and lab[na,nb]==0:
                            lab[na,nb]=cur; dq.append((na,nb))
    return lab,cur

MARGTH = 40.0   # 抓取“该墙体平面”点的轴坐标容差(mm)
def gen_band_ann(axis, fixed, u0,u1,v0,v1, cat):
    """在洞口带宽(RING)内铺一圈连续标注点(每网格格心一点)，保证边界外观看成闭环。"""
    col = C[cat]; pidmark = 1001 if cat=="door" else 1002
    axs=[i for i in range(3) if i!=axis]
    m=xyz[np.abs(xyz[:,axis]-fixed)<=MARGTH]
    U=m[:,axs[0]].astype(np.float64); V=m[:,axs[1]].astype(np.float64)
    U0,U1=u0-MARG,u1+MARG; V0,V1=v0-MARG,v1+MARG
    inb=(U>=U0)&(U<=U1)&(V>=V0)&(V<=V1)
    nU=int(np.ceil((U1-U0)/PY))+1; nV=int(np.ceil((V1-V0)/PY))+1
    iu=np.clip(((U[inb]-U0)/PY).astype(np.int32),0,nU-1)
    iv=np.clip(((V[inb]-V0)/PY).astype(np.int32),0,nV-1)
    occ=np.zeros((nU,nV),bool); occ[iu,iv]=True
    empty=~occ
    from collections import deque as _dq
    lab=np.zeros_like(empty,np.int32); cur=0
    for i in range(nU):
        for j in range(nV):
            if empty[i,j] and lab[i,j]==0:
                cur+=1; lab[i,j]=cur; q=_dq([(i,j)])
                while q:
                    a,b=q.popleft()
                    for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)):
                        na,nb=a+dx,b+dy
                        if 0<=na<nU and 0<=nb<nV and empty[na,nb] and lab[na,nb]==0:
                            lab[na,nb]=cur; q.append((na,nb))
    bc=set(np.unique(np.concatenate([lab[0],lab[-1],lab[:,0],lab[:,-1]]))); bc.discard(0)
    internal = empty & ~np.isin(lab, list(bc))
    # BFS 到空洞的距离(全平面格, 允许穿过材料/背景), 网格快捷格距离
    dist=np.full((nU,nV),1e9); dq=_dq()
    for i in range(nU):
        for j in range(nV):
            if internal[i,j]: dist[i,j]=0; dq.append((i,j,0))
    while dq:
        x,y,d=dq.popleft()
        for dx in(-1,0,1):
            for dy in(-1,0,1):
                if dx==0 and dy==0: continue
                nx,ny=x+dx,y+dy
                if (0<=nx<nU and 0<=ny<nV and not internal[nx,ny]
                        and dist[nx,ny] > d+1):
                    dist[nx,ny]=d+1; dq.append((nx,ny,d+1))
    # 沿洞口轮廓的整个环带(距空洞0<RING mm内)均匀铺点, 保证连续闭环
    band = (dist>0) & (dist*PY <= RING)
    rr,cc = np.nonzero(band)                    # 行=U,列=V
    coords=np.zeros((len(rr),3),np.float32)
    coords[:,axis]=fixed
    coords[:,axs[0]]=U0+(rr+0.5)*PY
    coords[:,axs[1]]=V0+(cc+0.5)*PY
    return coords, col, pidmark

stats={}
for (axis, fixed, u0,u1,v0,v1, wall, cat) in filled:
    idx = boundary_points(wall, axis, u0,u1,v0,v1)
    color[idx] = C[cat]
    stats[cat] = stats.get(cat,0)+len(idx)
print("开洞:", len(filled))
for (axis,fixed,u0,u1,v0,v1,wall,cat) in filled:
    print(f"  cat={cat}: 墙id={wall} 固定轴{'XYZ'[axis]}={fixed:.0f} "
          f"{'XY'[axis] if 0 else 'YZ'}区间[{u0:.0f},{u1:.0f}] 高[{v0:.0f},{v1:.0f}]")
print("边界着色点数: 门=%d 窗=%d"%(stats.get('door',0), stats.get('window',0)))
print("原有点分类: 地面%d 墙%d 筋%d 其他%d"%(
    np.isin(pid,GROUND).sum(),np.isin(pid,WALLS).sum(),
    np.isin(pid,RIBS).sum(),(~np.isin(pid,GROUND+WALLS+RIBS)).sum()))

# ---- 写出(仅原始点 + 门窗边缘真实点染色) ----
n=len(pid)
hdr=("ply\nformat binary_little_endian 1.0\n"
     f"element vertex {n}\n"
     "property float x\nproperty float y\nproperty float z\n"
     "property ushort part_id\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
     "element face 0\nproperty list uchar int vertex_indices\nend_header\n")
out=np.zeros(n,dtype=[("x","<f4"),("y","<f4"),("z","<f4"),("part_id","<u2"),
    ("r","u1"),("g","u1"),("b","u1")])
out["x"]=rec["x"]; out["y"]=rec["y"]; out["z"]=rec["z"]
out["part_id"]=rec["part_id"]; out["r"]=color[:,0]; out["g"]=color[:,1]; out["b"]=color[:,2]
with open(OUTPLY,"wb") as f: f.write(hdr.encode("latin1"));
with open(OUTPLY,"ab") as f: f.write(out.tobytes())
print("已写出:", OUTPLY, "总点", n)

# ---- 预览 ----
zh=next((x for x in fm.findSystemFonts(None,"ttf") if any(k in x for k in("yahei","simhei","msyh"))),None)
matplotlib.rcParams["font.sans-serif"]=[fm.FontProperties(fname=zh).get_name()] if zh else []
matplotlib.rcParams["axes.unicode_minus"]=False
idx=np.linspace(0,n-1,MAXPLOT).astype(np.int64) if n>MAXPLOT else np.arange(n)
cc=color[idx]/255.0
fig=plt.figure(figsize=(18,12))
views=[("三维透视",30,25),("正视(看X面)",0,0),("俯视XY",-90,90),("侧视YZ",90,0)]
for i,(t,az,el) in enumerate(views,1):
    ax=fig.add_subplot(2,2,i,projection="3d")
    ax.scatter(xyz[idx,0],xyz[idx,1],xyz[idx,2],c=cc,s=0.5,depthshade=False)
    ax.set_title(t,fontsize=13); ax.view_init(elev=el,azim=az)
    ax.set_box_aspect((np.ptp(xyz[idx,0]),np.ptp(xyz[idx,1]),np.ptp(xyz[idx,2])))
plt.tight_layout(); plt.savefig(OUTPNG,dpi=110)
print("预览:", OUTPNG)