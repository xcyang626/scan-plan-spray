# -*- coding: utf-8 -*-
"""黑白机器人扫描点云 与 手持设备语义地图 的三维配准 + 语义标签传播。

输入
----
  M_ref : 舱室点云_语义标注.ply    （语义参考地图，已含 ground/wall/door/window/rib/other）
  S_ref : 舱室点云_原始扫描黑白.ply（机器人待标注黑白点云）

流程
----
  0) 预处理：体素降采样、PCA法线/曲率估计、法向一致性取向（面朝房间内部）。
  1) 粗配准（对齐附件《点云配准》方法，适配房间对象——关键点 = 四个墙角 + 门框）：
       关键点检测：舱室是长方体，结构地标即为四个墙角(其底/顶共 8 个非共面端点)与门框
       (门洞四角)。· 四个墙角由墙体平面常数 ±x/±y(中位) + 地面 z0 + 墙顶 ztop 给出；
       · 门框在语义图直接用门标签包围盒，黑白图用墙面底部贯穿孔洞几何检测。
       几何一致性匹配三维距离三元组 → Kabsch(SVD) 求初始变换；另用地面平面+质心初始化，
       二者经稠密点云最近邻重叠度择优。门偏置 → 消除墙角折叠/对称歧义。
  2) 精配准（附件式自适应距离阈值点到面ICP）：
       每轮 ε_adp=μ_d+0.1σ_d 误匹配剔除 + 点到面 + 李代数增量 + 残差/内点双收敛判据。
  3) 语义标签传播：S 变换回参考系后用体素哈希最近邻迁移 M 的语义标签。
  4) 输出 舱室地图机器人扫描版本.ply（原黑白点云的语义标签地图，不覆盖原始黑白图）。

注：文件中另保留一组“对照路径”函数（附件式通用方法：SI 关键点/FPFH/几何一致性/
  RANSAC/定阈值点到面ICP），当前主管线 main() 未调用，仅用于与房间适配主管线对照。

依赖：仅 numpy + matplotlib。
"""
import os, heapq
from collections import deque
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
CLOUD = os.path.join(BASE, "舱室点云")
M_REF = os.path.join(CLOUD, "舱室点云_语义标注.ply")
S_REF = os.path.join(CLOUD, "舱室点云_原始扫描黑白.ply")
OUT   = os.path.join(CLOUD, "舱室地图机器人扫描版本.ply")
OUTPNG= os.path.join(CLOUD, "舱室地图机器人扫描版本.png")

C = dict(ground=(128,128,128), wall=(150,220,150), door=(178,64,222),
         window=(255,230,0), rib=(0,110,255), other=(0,0,0))
LABEL_ID = {k:i for i,k in enumerate(C)}

# ---------------------------------------------------------------------- IO
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
    return xyz, rec, rgb

def write_ply(path, xyz, rgb, part_id):
    n = len(xyz)
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           f"element vertex {n}\n"
           "property float x\nproperty float y\nproperty float z\n"
           "property ushort part_id\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\n"
           "element face 0\nproperty list uchar int vertex_indices\nend_header\n")
    out = np.empty(n, dtype=[("x","<f4"),("y","<f4"),("z","<f4"),("part_id","<u2"),
                             ("r","u1"),("g","u1"),("b","u1")])
    out["x"], out["y"], out["z"] = xyz[:,0], xyz[:,1], xyz[:,2]
    out["part_id"] = (part_id if part_id is not None
                      else np.zeros(n,np.uint16)).astype(np.uint16)
    out["r"], out["g"], out["b"] = rgb[:,0], rgb[:,1], rgb[:,2]
    with open(path, "wb") as f:
        f.write(hdr.encode("latin1")); f.write(out.tobytes())
    print("[写出]", path, "点数", n)

# ------------------------------------------------------------------ KDTree(kNN)
class KDTree:
    """轻量 k-d 树，支持批量 k-近邻搜索。"""
    def __init__(self, pts, leaf_size=16):
        self.pts = np.ascontiguousarray(pts, np.float64)
        self.leaf_size = leaf_size
        self.n = len(self.pts)
        self.perm = np.arange(self.n)
        self.LO = []; self.HI = []; self.LC = []; self.RC = []
        self.BMIN = []; self.BMAX = []
        self._build(0, self.n, 0)

    def _build(self, lo, hi, depth):
        node = len(self.LO)
        pts = self.pts[self.perm[lo:hi]]
        self.LO.append(lo); self.HI.append(hi)
        self.BMIN.append(pts.min(0)); self.BMAX.append(pts.max(0))
        self.LC.append(-1); self.RC.append(-1)
        if hi - lo <= self.leaf_size:
            return node
        axis = depth % 3
        seg = self.pts[self.perm[lo:hi], axis]
        mid = (lo + hi) // 2
        kth = np.argpartition(seg, mid - lo) + lo
        self.perm[lo:hi] = self.perm[kth]
        self.LC[node] = self._build(lo, mid, depth + 1)
        self.RC[node] = self._build(mid, hi, depth + 1)
        return node

    def _dbox(self, node, p):
        mn, mx = self.BMIN[node], self.BMAX[node]
        d = 0.0
        for a in range(3):
            if p[a] < mn[a]: d += (mn[a] - p[a]) ** 2
            elif p[a] > mx[a]: d += (p[a] - mx[a]) ** 2
        return d

    def query(self, q, k=1):
        q = np.atleast_2d(np.asarray(q, np.float64))
        out_i = np.empty((len(q), k), np.int64)
        out_d = np.empty((len(q), k), np.float64)
        for a in range(len(q)):
            p = q[a]; heap = []
            self._search(0, p, k, heap)
            heap.sort(key=lambda e: -e[0])
            for jj in range(min(k, len(heap))):
                nd, ix = heap[jj]; out_d[a,jj] = -nd; out_i[a,jj] = ix
        return out_i, out_d

    def _search(self, node, p, k, heap):
        dbox = self._dbox(node, p)
        worst = -heap[0][0] if len(heap) == k else 1e300
        if dbox > worst: return
        lo, hi = self.LO[node], self.HI[node]
        if self.RC[node] < 0:                       # 叶
            d2 = np.sum((self.pts[self.perm[lo:hi]] - p) ** 2, 1)
            for t in range(len(d2)):
                nd = d2[t]; ix = self.perm[lo + t]
                if len(heap) < k:
                    heapq.heappush(heap, (-nd, ix))
                elif nd < -heap[0][0]:
                    heapq.heapreplace(heap, (-nd, ix))
            return
        l, r = self.LC[node], self.RC[node]
        dl = self._dbox(l, p); dr = self._dbox(r, p)
        first, second = (l, r) if dl <= dr else (r, l)
        self._search(first, p, k, heap)
        self._search(second, p, k, heap)

# ------------------------------------------------------------------ 工具
def voxel_subsample(xyz, leaf, max_n=None, seed=0):
    cell = np.floor(xyz / leaf).astype(np.int64)
    # 体素键直接按三维格坐标整行去重：原 c0*1e6+c1*1e3+c2 混合键在任一轴
    # 跨度≥1000 格时发生哈希碰撞，会把不同体素的点静默合并丢点
    # （gen_robot_scan leaf=4mm 时 y 轴可跨约 1262 格，必然触发）
    _, first = np.unique(cell, axis=0, return_index=True)
    sel = first.astype(np.int64)
    if max_n and len(sel) > max_n:
        sel = np.random.default_rng(seed).choice(sel, max_n, replace=False)
    return np.sort(sel)

def estimate_normals(xyz, k=17):
    kdt = KDTree(xyz); n = len(xyz)
    normals = np.empty((n,3)); curv = np.empty(n)
    for i in range(0, n, 2048):
        j = slice(i, min(i+2048, n))
        idx, _ = kdt.query(xyz[j], k)
        c = xyz[idx] - xyz[idx].mean(1, keepdims=True)
        cov = np.einsum("nji,njk->nik", c, c) / np.clip(idx.shape[1]-1, 2, None)
        w, v = np.linalg.eigh(cov)
        normals[j] = v[:,:,0]; curv[j] = w[:,0]/np.clip(w.sum(1),1e-12,None)
    return normals, curv

def orient_toward_center(xyz, normals):
    cen = xyz.mean(0)
    flip = np.einsum("ij,ij->i", cen[None,:]-xyz, normals) < 0
    normals[flip] *= -1
    return normals

# ===================== 对照路径：附件式关键点检测（当前主管线未调用） =====================
def _shape_index(xyz, normals, idx):
    """附件式形状指数(Shape Index)：对邻域做局部二次曲面拟合 z=f(x,y)，
    由Hessian特征值得主曲率κ1,κ2，SI=1/2-(1/π)arctan((κ1+κ2)/(κ1-κ2))。
    房间适配：平面区域SI≈0.5，门/窗/加强筋边缘等结构点SI显著偏离0.5。"""
    n = len(xyz)
    SI = np.empty(n)
    ref = np.where((np.abs(normals[:,0]) < 0.9)[:,None], [1.0,0.0,0.0], [0.0,1.0,0.0])  # 每点切向参考轴
    for i in range(0, n, 1024):
        s = slice(i, min(i+1024, n)); b = idx[s]
        Ni = normals[s]
        xA = np.cross(ref[s], Ni); xA /= np.clip(np.linalg.norm(xA,axis=1)[:,None], 1e-9, None)
        yA = np.cross(Ni, xA)
        P = xyz[b] - xyz[s][:,None,:]                       # (b,k,3) 相对坐标
        x = np.einsum("bkd,bd->bk", P, xA)
        y = np.einsum("bkd,bd->bk", P, yA)
        z = np.einsum("bkd,bd->bk", P, Ni)
        A = np.stack([np.ones_like(x), x, y, x*x, x*y, y*y], axis=2)   # (b,k,6)
        AtA = np.einsum("bki,bkj->bij", A, A) + 1e-8*np.eye(6)
        Atb = np.einsum("bki,bk->bi", A, z)
        coef = np.linalg.solve(AtA, Atb[...,None])[...,0]  # (b,6): a0,a1,a2,a3,a4,a5
        # 二次项系数为 x²、xy、y²，即 coef 第 3/4/5 列（原代码取 2/3/4 列，
        # 把线性 y 当二次项、丢了 y² 项，导致 Hessian 错位）
        a3, a4, a5 = coef[:,3], coef[:,4], coef[:,5]
        tr = 2*a3 + 2*a5; det = 4*a3*a5 - a4*a4
        disc = np.abs(tr*tr - 4*det)
        kp = 0.5*(tr + np.sqrt(disc)); km = 0.5*(tr - np.sqrt(disc))
        ratio = (kp + km) / np.clip(kp - km, 1e-9, None)
        SI[s] = 0.5 - 1.0/np.pi*np.arctan(ratio)
    return SI

def select_keypoints(xyz, normals, curv, frac=0.10, min_dist=150.0, seed=1,
                     kn=18, alpha=0.25, beta=0.25):
    """【对照路径，当前主管线未调用】附件式关键点检测（Shape Index判据，适配房间场景）。
    注意：此处只用 (1±α)μ 阈值筛选，未实现附件式 4-29 的“SI≥邻域最大值”局部极值条件。
    保留局部SI相对邻域均值μ显著偏大/偏小的结构点（棱线、窗框、门框、加强筋等），
    并剔除位于点云边界附近支持不足的候选点（附件式边界剔除），最后空间去重。"""
    kdt = KDTree(xyz); n = len(xyz)
    idx, _ = kdt.query(xyz, kn)
    SI = _shape_index(xyz, normals, idx)
    mu = SI[idx].mean(1)
    hi = (1.0 + alpha) * mu; lo = (1.0 - beta) * mu
    cand = np.nonzero((SI >= hi) | (SI <= lo))[0]
    dist = np.linalg.norm(xyz[idx] - xyz[:,None,:], axis=2)
    rmed = np.median(dist, 1)                     # 局部特征尺度（第kn近邻距离）
    support = (dist <= 2.0*rmed[:,None]).mean(1)  # 有效支撑占比
    cand = cand[support[cand] > 0.6]              # 剔除边界/支撑不足点
    if len(cand) > 50000:
        order = np.argsort(-np.abs(SI[cand] - 0.5)); cand = cand[order[:50000]]
    rng = np.random.default_rng(seed); rng.shuffle(cand)
    sel = []; grid = {}; r2 = min_dist * min_dist
    for i in cand:
        key = tuple(np.floor(xyz[i]/min_dist).astype(np.int64))
        hit = False
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for dz in (-1,0,1):
                    nb = grid.get((key[0]+dx, key[1]+dy, key[2]+dz))
                    if nb is not None:
                        for jy in nb:
                            if np.dot(xyz[i]-xyz[jy], xyz[i]-xyz[jy]) < r2:
                                hit = True; break
                    if hit: break
                if hit: break
            if hit: break
        if not hit:
            grid.setdefault(key, []).append(i); sel.append(i)
    return np.array(sel, np.int64)

def select_structure_keypoints(xyz, normals, min_dist=150.0, kn=28, thr2=0.12,
                               maxpts=9000, seed=1):
    """【对照路径，当前主管线未调用】结构化关键点：舱室墙角/墙面交线 + 门(窗)框边缘（面向室内场景的配准关键点）。
    房间是长方体结构，法线主轴方向只有±x/±y/±z三类：
    - 墙角/墙面交线：邻域内法线主轴类别≥两类且第二占比达阈值(如相邻墙 x∩z 交线、
      三面交汇的房间角 x∩y∩z)的点即为墙角类关键点。
    - 门(窗)框：沿主导主轴方向(墙面法向)邻域深度离散度显著偏大的墙内边缘点。
    二者合并并做空间去重；对源/目标点云均几何检测，保证关键点对称一一对应。"""
    kdt = KDTree(xyz); n = len(xyz)
    cat = np.argmax(np.abs(normals), 1)                 # 法线主轴类别 0/1/2
    n2 = normals * normals
    dom = n2[np.arange(n), cat]                         # 主导分量占比(主轴明确度)
    idx, _ = kdt.query(xyz, kn)
    coord = xyz[np.arange(n), cat]
    sd_ax = (coord[idx] - coord[:,None]).std(1)         # 沿主导轴深度离散度
    med_sd = np.median(sd_ax)
    crn_pts, ege_pts = [], []
    for i in range(0, n, 2048):
        s = slice(i, min(i+2048, n)); cc = cat[idx[s]]; dd = dom[s]; et = sd_ax[s]
        c0 = (cc==0).sum(1); c1 = (cc==1).sum(1); c2 = (cc==2).sum(1)
        tot = c0+c1+c2; tot = np.where(tot>0, tot, 1)
        Pt = np.sort(np.stack([c0/tot, c1/tot, c2/tot], 1), 1)[:,::-1]
        corner = np.nonzero((Pt[:,0] < 0.97) & (Pt[:,1] >= thr2))[0]   # 墙角/交线
        edge   = np.nonzero((dd >= 0.85) & (et >= 3.0*med_sd))[0]      # 门(窗)框边缘
        crn_pts.append(i + corner); ege_pts.append(i + edge)
    cand = np.concatenate(crn_pts + ege_pts)
    if len(cand) > 20000:                             # 约束去重成本
        if len(crn_pts) and len(crn_pts[0]):
            cand = np.concatenate(crn_pts) if len(crn_pts) else cand
            extra = np.concatenate(ege_pts)
            keep = 20000 - len(cand)
            if keep > 0 and len(extra) > keep:
                extra = np.random.default_rng(0).choice(extra, keep, replace=False)
            cand = np.concatenate([cand, extra])
        else:
            cand = np.random.default_rng(0).choice(cand, 20000, replace=False)
    rng = np.random.default_rng(seed); rng.shuffle(cand)
    sel = []; grid = {}; r2 = min_dist * min_dist
    for i in cand:
        key = tuple(np.floor(xyz[i]/min_dist).astype(np.int64))
        hit = False
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for dz in (-1,0,1):
                    nb = grid.get((key[0]+dx, key[1]+dy, key[2]+dz))
                    if nb is not None:
                        for jy in nb:
                            if np.dot(xyz[i]-xyz[jy], xyz[i]-xyz[jy]) < r2:
                                hit = True; break
                    if hit: break
                if hit: break
            if hit: break
        if not hit:
            grid.setdefault(key, []).append(i); sel.append(i)
    sel = np.array(sel, np.int64)
    return sel[:maxpts]

def detect_room_corners(xyz, normals=None, leaf=25.0, k=8, seed=1):
    """房间墙角地标检测（舱室适配的关键点：墙面平面法，对应"四个墙角"）。
    长方体舱室由 ±x/±y 四面大平面墙围成。将法向取向朝房间内部后，按主导法向 ±x/±y
    分类得四面墙，各取其平面常数(墙点坐标中位数) → 4 个墙角 (x,y)；再取地面高度 z0
    (最低水平密面) 与墙顶高度 ztop(竖直墙点的高分位) → 4×2=8 个非共面墙角柱端点。
    - 非共面端点集完整约束刚体 6 自由度，且两幅点云用同一准则几何检测、一一对应；
    - 相比全局 PCA 包围盒角点(受极端杂点/密度拉扯)，墙面平面常数由数千点中位稳健给出；
    - 舱内障碍(围栏/操作台)均在房间墙面内侧，不改变墙平面常数。"""
    Ssub = xyz[voxel_subsample(xyz, leaf, 80000, 3)]
    if normals is None:
        normals, _ = estimate_normals(Ssub, 17)
    nrm = orient_toward_center(Ssub, normals.copy())
    Hz = np.abs(nrm[:,2]) > 0.85; hyp = Ssub[Hz]
    if len(hyp) < 500: return None
    h, e = np.histogram(hyp[:,2], bins=40)
    z0 = 0.5*(e[np.argmax(h)] + e[np.argmax(h)+1])
    vert = np.abs(nrm[:,2]) < 0.4; vp = Ssub[vert]; vn = nrm[vert]
    if len(vp) < 2000: return None
    ztop = np.percentile(vp[:,2], 99.0)
    nx, ny = vn[:,0], vn[:,1]
    def wc(mask, axis):
        p = vp[mask]
        return np.median(p[:,axis]) if p.size > 150 else None
    xlo, xhi = wc(nx < -0.7, 0), wc(nx > 0.7, 0)
    ylo, yhi = wc(ny < -0.7, 1), wc(ny > 0.7, 1)
    if None in (xlo, xhi, ylo, yhi): return None
    out = []
    for xc in (xlo, xhi):
        for yc in (ylo, yhi):
            out.append([xc, yc, z0]); out.append([xc, yc, ztop])
    return np.array(out, float)

def detect_door_frame(xyz, normals=None, leaf=25.0, seed=1,
                      door_mask=None, GX=200, GY=240, min_cells=40,
                      tol_floor=450.0, min_h=1200.0, min_w=400.0, max_w=1800.0):
    """门框关键点检测（舱室适配的关键点补充：墙壁内部贯通的孔洞 = 门洞）。

    与四个墙角互补：长方体舱室的墙角仅提供尺寸信息，而门通常偏置在某一面墙上，
    是不对称的结构地标，可消除墙角折叠(对角重排)/对称歧义，使两面点云的关键点
    一一对应唯一化。

    检测思路：仿照附件『孔洞』思想，把墙面点投影到 (沿墙方向, 高度) 栅格，做连通域
    空洞分析；门 = 底部触达地面(z≈z0)、高度与宽度在门窗合理范围、且不贯穿到顶/墙端的
    内部孔洞。返回该门洞左下/右下/左上/右上 4 个角点的 3D 坐标。

    - door_mask : 若提供(语义地图的门标签布尔数组)，则对该面墙上属于门的点直接取
      沿墙-高度包围盒作为门框，替代/补充栅格空洞检测(语义图门点稀疏时更稳)。
    - 关键点顺序固定为 (沿墙小,底)→(沿墙大,底)→(沿墙小,顶)→(沿墙大,顶)，
      因此 S/M 两侧以同一准则检测、一一对应。"""
    Ssub = xyz[voxel_subsample(xyz, leaf, 80000, seed)]
    if normals is None:
        normals, _ = estimate_normals(Ssub, 17)
    nrm = orient_toward_center(Ssub, normals.copy())

    Hz = np.abs(nrm[:,2]) > 0.85; hyp = Ssub[Hz]
    if len(hyp) < 500: return None
    h, e = np.histogram(hyp[:,2], bins=40)
    z0 = 0.5*(e[np.argmax(h)] + e[np.argmax(h)+1])

    vert = np.abs(nrm[:,2]) < 0.4; vp = Ssub[vert]; vn = nrm[vert]
    if len(vp) < 2000: return None
    nx, ny = vn[:,0], vn[:,1]
    def wall_spec(mask, axis):
        p = vp[mask]
        if p.size < 150: return None
        nw = vn[mask].mean(0); nw /= np.linalg.norm(nw)
        t = np.cross([0.,0.,1.], nw); tn = np.linalg.norm(t)
        if tn < 0.2: return None
        t = t/tn
        d0 = float(np.median(np.einsum("ij,j->i", p, nw)))   # 带符号平面常数 p·n=d0
        return nw, t, d0
    walls = [("xm", nx<-0.7, 0), ("xp", nx>0.7, 0),
             ("ym", ny<-0.7, 1), ("yp", ny>0.7, 1)]

    label_box = None   # (alo, ahi, zlo, zhi, nw, t, d0) —— 语义门标签框或栅格空洞
    if door_mask is not None and np.any(door_mask):
        dpts = xyz[door_mask]
    else:
        dpts = np.empty((0,3))

    # (a) 语义门标签：直接对门点拟合平面并取(沿墙,高度)包围盒 —— 门点本身即门洞所在平面，
    #     不再吸附到四面墙，避免门洞偏置较远时把门框错标到相邻墙面上
    if len(dpts):
        dmu = dpts.mean(0)
        _, _, Vh = np.linalg.svd(dpts - dmu, full_matrices=False)
        nw = Vh[2]; nw /= np.linalg.norm(nw)
        if nw[2] > 0: nw = -nw
        t = np.cross([0., 0., 1.], nw); tn = np.linalg.norm(t)
        if tn > 0.2:
            t = t / tn
            d0 = float(np.einsum("ij,j->i", dpts, nw).mean())
            al = dpts @ t; zz = dpts[:, 2]
            alo, ahi = al.min(), al.max(); zlo, zhi = zz.min(), zz.max()
            hgt, wdt = zhi - zlo, ahi - alo
            if wdt >= 250 and hgt >= 400:
                label_box = (alo, ahi, zlo, zhi, nw, t, d0)

    # (b) 无语义标签时：栅格空洞检测(门 = 底部触地、尺寸合理的内部孔洞)
    if label_box is None:
        for tag, m, axis in walls:
            sp = wall_spec(m, axis)
            if sp is None: continue
            nw, t, d0 = sp
            j = np.nonzero(m)[0]; wp = vp[j]; along = wp@t
            ay = np.percentile(along,[0.5,99.5]); az = np.percentile(wp[:,2],[0.5,99.5]); az[0]=min(az[0], z0-300)
            xs = np.linspace(ay[0],ay[1],GX+1); zs = np.linspace(az[0],az[1],GY+1)
            occ = np.zeros((GY,GX),bool)
            xi = np.clip(np.searchsorted(xs,along,side='right')-1,0,GX-1)
            yi = np.clip(np.searchsorted(zs,wp[:,2],side='right')-1,0,GY-1)
            occ[yi,xi]=True
            Hs,W=occ.shape; vis=np.zeros_like(occ,bool)
            for yy in range(Hs):
                for xx in range(W):
                    if occ[yy,xx] or vis[yy,xx]: continue
                    q=deque([(yy,xx)]); vis[yy,xx]=True; cells=[]
                    while q:
                        cy,cx=q.popleft(); cells.append((cy,cx))
                        for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                            nyr,nxr=cy+dy,cx+dx
                            if 0<=nyr<Hs and 0<=nxr<W and not occ[nyr,nxr] and not vis[nyr,nxr]:
                                vis[nyr,nxr]=True; q.append((nyr,nxr))
                    if len(cells)<min_cells: continue
                    ys=[c[0] for c in cells]; xss=[c[1] for c in cells]
                    if max(xss)>=W-1 or max(ys)>=Hs-1: continue      # 贯穿到墙端或顶 → 非门
                    zlo = az[0]+ min(ys)/(GY)*(az[1]-az[0])
                    zhi = az[0]+(max(ys)+1)/(GY)*(az[1]-az[0])
                    alo = ay[0]+ min(xss)/(GX)*(ay[1]-ay[0])
                    ahi = ay[0]+(max(xss)+1)/(GX)*(ay[1]-ay[0])
                    hgt=zhi-zlo; wdt=ahi-alo; fg=zlo-z0
                    if fg < tol_floor and hgt>=min_h and wdt>=min_w:
                        if label_box is None or wdt*hgt > (label_box[1]-label_box[0])*(label_box[3]-label_box[2]):
                            label_box = (alo,ahi,zlo,zhi,nw,t,d0)

    if label_box is None: return None
    alo,ahi,zlo,zhi,nw,t,d0 = label_box
    # 尺寸合理性再确认：门须是"宽≤门宽上限、高≥门高下限"的矩形
    if (ahi-alo) < min_w or (ahi-alo) > max_w or (zhi-zlo) < 800:
        return None
    def mk(a, zz): return d0*nw + t*a + np.array([0.,0.,zz])
    return np.array([mk(alo,zlo), mk(ahi,zlo), mk(alo,zhi), mk(ahi,zhi)], float)

def select_by_overlap(cands, Ssub, Msub, th=180.0, leaf=30.0, seed=0, maxv=40000):
    """从粗配准候选变换中按稠密点云最近邻重叠度择优，消除墙角折叠/对称歧义。
    对每个候选 (R,t)：
      - 用体素降采样减小代价；
      - 计算变换后源点在 M 点云最近邻距离 < th 的比例(重叠度)；
    返回重叠度最高者的 (R,t)。"""
    ids = voxel_subsample(Ssub, leaf, maxv, seed); ps = Ssub[ids]
    dm = voxel_subsample(Msub, leaf, maxv, seed+1); pm = Msub[dm]
    kdt = KDTree(pm)
    best = (None, -1.0)
    for R, t in cands:
        q = apply(R, t, ps)
        _, qd = kdt.query(q, 1)
        frac = float((qd[:,0] < th).mean())
        if frac > best[1]:
            best = ((R, t), frac)
    return best[0] if best[0] is not None else cands[0]

def floor_init(X, Y, leaf=40.0, seed=5, maxv=30000, band=200.0):
    """地面平面 + 质心初始化（舱室粗配准的稳健候选）。
    提取两侧点云最密的水平(地面)层及其质心：t = M地面质心 - S地面质心。
    舱室机器人扫描相对语义图通常仅存在约5°平面内旋转，故 R 取单位阵，剩余小旋转
    交由精配准自适应ICP收敛；若地面法向非严格竖直，再用最小旋转把 S 地面法向转到 M。
    返回 (R, t)。"""
    def floor_part(P):
        sub = P[voxel_subsample(P, leaf, maxv, seed)]
        nrm, _ = estimate_normals(sub, 17)
        nrm = orient_toward_center(sub, nrm.copy())
        Hz = np.abs(nrm[:,2]) > 0.85
        hidx = np.nonzero(Hz)[0]                 # 水平面点在 sub 中的原始索引
        hyp = sub[hidx]
        if len(hyp) < 200: return None, None
        h, e = np.histogram(hyp[:,2], bins=40)
        z0 = 0.5*(e[np.argmax(h)] + e[np.argmax(h)+1])
        band_m = np.abs(hyp[:,2]-z0) < band
        gr = hyp[band_m]
        # 地面法向只对 z0 附近地面带(gr)取均值：Hz 同时含地面与天花两层水平面，
        # 对全部水平点取均值会法向相互抵消，得到噪声方向
        n = nrm[hidx[band_m]].mean(0); n /= np.linalg.norm(n)
        return gr.mean(0), n
    fs, ns = floor_part(X); fm, nm = floor_part(Y)
    if fs is None or fm is None:
        # 地面初值不可用（至少一侧地面提取失败）：回退整体质心差平移（单位旋转），
        # 剩余位姿差交由精配准自适应ICP收敛
        print("[警告] 地面提取失败, 地面初值不可用, 回退整体质心差平移")
        return np.eye(3), Y.mean(0) - X.mean(0)
    axis = np.cross(ns, nm); ang = np.arccos(np.clip(np.dot(ns, nm), -1, 1))
    R = np.eye(3)
    if np.linalg.norm(axis) > 1e-6:
        R = exp_so3(axis/np.linalg.norm(axis) * ang)
    t = fm - R @ fs
    return R, t

def corner_registration(CS, CM, tol_ratio=0.02, flat=400.0, th=400.0):
    """地标配准：附件粗配准'关键点+几何一致性+鲁棒估计'的房间适配。
    房间八个主成分包围盒墙角在两侧点云近似一一对应；用三维距离几何一致性匹配三角形，
    对全等三角形以 Kabsch(SVD) 求解刚体变换，按全体角点内点数享受留内点一致者。
    因墙角存在对称折叠歧义，返回所有达到最优内点数的候选变换，供后续按稠密重叠度择优。"""
    nS, nM = len(CS), len(CM)
    if nS < 3 or nM < 3:
        return [(np.eye(3), np.zeros(3))], -1
    DS = np.sqrt(((CS[:,None,:]-CS[None,:,:])**2).sum(2))
    DM = np.sqrt(((CM[:,None,:]-CM[None,:,:])**2).sum(2))
    best_n = -1; found = {}                        # 转换签名 -> (R,t,nin)
    import itertools
    def order_vertices(D, tri):
        a, b, c = tri
        sig = [(tuple(sorted((D[a,b], D[a,c]))), a),
               (tuple(sorted((D[a,b], D[b,c]))), b),
               (tuple(sorted((D[a,c], D[b,c]))), c)]
        sig.sort(key=lambda x: x[0])
        return [s[1] for s in sig]
    idxS = list(itertools.combinations(range(nS), 3))
    idxM = list(itertools.combinations(range(nM), 3))
    edges = lambda A, tri: sorted((A[tri[0],tri[1]],A[tri[0],tri[2]],A[tri[1],tri[2]]))
    for tS in idxS:
        eS = edges(DS, tS); oS = order_vertices(DS, tS)
        for tM in idxM:
            eM = edges(DM, tM)
            if all(abs(a-b) <= tol_ratio*max(a,b) + flat for a,b in zip(eS,eM)):
                oM = order_vertices(DM, tM)
                R, t = svd_transform(CS[oS], CM[oM])
                q = CS@R.T + t
                d = ((q[:,None,:]-CM[None,:,:])**2).sum(2).min(1)
                nin = int((d < th).sum())
                # 先相对“当前最优”判定再收录：达到 max(最优-1,3) 才收录，收录同时更新最优。
                # （原实现先 best_n=nin 再判定恒真、末尾过滤分支永假，属死代码）
                # 初值 best_n=-1：首个候选仅需 nin>=3 即收录
                if nin >= max(best_n - 1, 3):
                    best_n = max(best_n, nin)
                    key = (np.round(R,3).tobytes(), np.round(t,0).tobytes())
                    if key not in found or found[key][2] < nin:
                        found[key] = (R, t, nin)
    # 兜底：按全体角点内点数排序，取前 8 个候选变换
    cands = [(R, t) for (R, t, _) in sorted(found.values(),
             key=lambda v: v[2], reverse=True)[:8]]
    return cands if cands else [(np.eye(3), np.zeros(3))], best_n

# ------------------------------------------------------------------ 配准
def exp_so3(w):
    th = np.linalg.norm(w)
    if th < 1e-12: return np.eye(3)
    K = np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])
    return np.eye(3) + np.sin(th)/th*K + ((1-np.cos(th))/th**2)*(K@K)

def svd_transform(src, dst):
    cs, cd = src.mean(0), dst.mean(0)
    H = (src-cs).T @ (dst-cd)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0: R[:, -1] *= -1
    return R, cd - R @ cs

def apply(R, t, P):
    return P @ R.T + t

# ============ 对照路径：附件式 FPFH 特征 + 几何一致性 + RANSAC 粗配准（当前主管线未调用） ============
def fpfh_descriptors(xyz, normals, kp, kn=20):
    kdt = KDTree(xyz)
    nb, _ = kdt.query(xyz[kp], kn)
    out = []
    for a, p in enumerate(kp):
        n0 = normals[p]; rel = xyz[nb[a]] - xyz[p]; dist = np.linalg.norm(rel, axis=1)
        nq = normals[nb[a]]
        u = np.broadcast_to(n0[None,:], (len(dist),3))
        v = np.cross(u, rel); vn = v/np.clip(np.linalg.norm(v,axis=1)[:,None],1e-9,None)
        w = np.cross(u, v); wn = w/np.clip(np.linalg.norm(w,axis=1)[:,None],1e-9,None)
        alpha = np.einsum("ij,ij->i", vn, nq)
        phi   = np.einsum("ij,ij->i", u, rel)/np.clip(dist,1e-9,None)
        theta = np.arctan2(np.einsum("ij,ij->i", wn, nq), np.einsum("ij,ij->i", u, nq))
        hist = np.zeros(33)
        for bid, vals in enumerate((alpha, phi, theta)):
            t = np.clip((vals/np.pi + 1.0)/2.0*10, 0, 10).astype(np.int32)
            hist[bid*11:(bid+1)*11] = np.bincount(t, minlength=11)
        hist /= max(hist.sum(), 1)
        out.append(hist)
    return np.stack(out)

def match_features(d1, d2, ratio=0.8):
    """【对照路径，当前主管线未调用】FPFH 最近邻比值检验匹配。"""
    corr = []
    for i in range(len(d1)):
        dd = np.sum((d2-d1[i])**2, 1)
        o2 = np.argpartition(dd, 2)[:2]; o2 = o2[np.argsort(dd[o2])]
        if dd[o2[0]] < ratio*dd[o2[1]]:
            corr.append((i, int(o2[0])))
    return np.array(corr, np.int64) if corr else np.empty((0,2),np.int64)

def gc_filter(corr, ps_s, ps_m, gamma=0.20, keep_ratio=0.6, seed=0):
    """【对照路径，当前主管线未调用】几何一致性筛选：保留与多数对应满足刚性距离一致性的对应（附件式）。
    返回筛选后的对应子集。corr 行对应源/目标关键点坐标 ps_s / ps_m(与corr同序)。"""
    nc = len(corr)
    if nc <= 8: return corr, nc
    Ds = np.sqrt(np.sum((ps_s[:,None,:]-ps_s[None,:,:])**2, 2))
    Dm = np.sqrt(np.sum((ps_m[:,None,:]-ps_m[None,:,:])**2, 2))
    scale = np.maximum(Ds, Dm) + 1e-9
    cons = np.abs(Ds - Dm) <= gamma * scale     # nc x nc, 对称
    np.fill_diagonal(cons, False)
    support = cons.sum(1)                        # 每个对应的支持数
    th = np.percentile(support, (1-keep_ratio)*100)
    good = np.nonzero(support >= th)[0]
    # 在 good 上保留连通主体（并查集）
    parent = np.arange(len(good))
    def find(x):
        r = x
        while parent[r] != r: r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r
    g2 = {gi: i for i, gi in enumerate(good)}
    for gi in good:
        neighbors = np.nonzero(cons[gi])[0]
        base = find(g2[gi])
        for gj in neighbors:
            if gj in g2 and find(g2[gj]) != base:
                parent[find(g2[gj])] = base
    comp = {}
    for i, gi in enumerate(good): comp.setdefault(find(i), []).append(gi)
    best = max(comp.values(), key=len)
    return corr[np.array(sorted(best), np.int64)], len(best)

def ransac_coarse(corr, xyz_s, xyz_m, kp_s, kp_m, iters=4000, inlier_th=0.06):
    """【对照路径，当前主管线未调用】附件式 RANSAC 粗配准（配合 fpfh/match_features/gc_filter）。"""
    ps = xyz_s[kp_s[corr[:,0]]]; pm = xyz_m[kp_m[corr[:,1]]]
    N = len(corr)
    if N < 4:
        return np.eye(3), np.zeros(3), 0, N
    best_in = -1; best_R = np.eye(3); best_t = np.zeros(3)
    rng = np.random.default_rng(2)
    for _ in range(iters):
        sel = rng.choice(N, 3, replace=False)
        R, t = svd_transform(ps[sel], pm[sel])
        d = np.linalg.norm(apply(R, t, ps) - pm, axis=1)
        inl = int((d < inlier_th).sum())
        if inl > best_in:
            best_in = inl; best_R = R.copy(); best_t = t.copy()
    if best_in >= 4:
        d = np.linalg.norm(apply(best_R, best_t, ps) - pm, axis=1)
        mask = d < inlier_th
        best_R, best_t = svd_transform(ps[mask], pm[mask])
    return best_R, best_t, best_in, N

def point_to_plane_icp(src, dst, dst_n, R, t, max_iter=50, tol=1e-7, max_dist=0.05):
    """【对照路径，当前主管线未调用】定阈值点到面 ICP（主管线用 point_to_plane_icp_adaptive）。"""
    kdt = KDTree(dst); prev = 1e9
    for it in range(max_iter):
        q = apply(R, t, src)
        qi, qd = kdt.query(q, 1); qi = qi[:,0]
        mask = qd[:,0] < max_dist
        if mask.sum() < 10: break
        p1 = q[mask]; q1 = dst[qi[mask]]; n1 = dst_n[qi[mask]]
        err = np.einsum("ij,ij->i", p1-q1, n1)
        A = np.hstack([np.cross(p1, n1), n1])
        try:
            dxi = np.linalg.solve(A.T@A + 1e-9*np.eye(6), -A.T@err)
        except np.linalg.LinAlgError:
            break
        dR = exp_so3(dxi[:3]); R = dR@R; t = dR@t + dxi[3:]
        e = float(np.abs(err).mean())
        if abs(prev - e) < tol: 
            Q = apply(R, t, src); qi,_ = kdt.query(Q,1)
            return R, t, float(np.sqrt(((Q-dst[qi[:,0]])**2).sum(1).mean()))
        prev = e
    Q = apply(R, t, src); qi,_ = kdt.query(Q,1)
    return R, t, float(np.sqrt(((Q-dst[qi[:,0]])**2).sum(1).mean()))

def point_to_plane_icp_adaptive(src, dst, dst_n, R, t, max_iter=60,
                                eps_rmse=1e-3, eps_inlier=1e-3, cap=150.0):
    """附件式精配准：自适应距离阈值误匹配去除 + 点到面 + 李代数增量 + 双重收敛判据。
    - 误匹配去除：每轮按对应点距离分布计算自适应阈值 ε_adp=μ_d+0.1σ_d（附件4-46），
      仅保留距离<ε_adp的内点参与位姿估计；cap为安全上界，避免初期阈值过大。
    - 位姿增量：点到面残差 + 李代数(δω,δt)线性最小二乘正规方程，指数映射更新。
    - 收敛判据：残差稳定性δ_rmse≤ε_rmse 且 内点比例稳定性δ_inlier≤ε_inlier（附件4-61），
      或达到最大迭代次数。数据单位为 mm，ε_rmse=1e-3 即残差变化小于 0.001mm 视为稳定
      （原 1e-5 对应 0.00001mm，mm 尺度下几乎不可能达到，双判据永不触发）。
    房间适配：对以平面为主的室内场景，自适应阈值天然对地面/墙面的大范围对应保持宽容，
    随收敛自动收紧，抑制非重叠区域误匹配。"""
    kdt = KDTree(dst)
    N = len(src)
    prev_rmse = 1e9; prev_inl = -1.0
    for it in range(max_iter):
        Q = apply(R, t, src)
        qi, qd = kdt.query(Q, 1); qi = qi[:,0]; qd = qd[:,0]
        mu, sd = qd.mean(), qd.std()
        th = min(mu + 0.1*sd, cap)            # 自适应距离阈值(附件4-46)
        mask = qd < th
        nin = int(mask.sum()); inl_ratio = nin / max(N, 1)
        if nin < 30: break
        p1 = Q[mask]; q1 = dst[qi[mask]]; n1 = dst_n[qi[mask]]
        r = np.einsum("ij,ij->i", p1 - q1, n1)            # 点到面残差
        rmse = float(np.sqrt((r*r).mean()))
        A = np.hstack([np.cross(p1, n1), n1])             # 雅可比(1×6)
        try:
            dxi = np.linalg.solve(A.T@A + 1e-9*np.eye(6), -A.T@r)
        except np.linalg.LinAlgError:
            break
        dR = exp_so3(dxi[:3]); R = dR@R; t = dR@t + dxi[3:]
        if abs(prev_rmse - rmse) < eps_rmse and abs(prev_inl - inl_ratio) < eps_inlier:
            break
        prev_rmse = rmse; prev_inl = inl_ratio
    Q = apply(R, t, src); qi,_ = kdt.query(Q, 1)
    return R, t, float(np.sqrt(((Q - dst[qi[:,0]])**2).sum(1).mean()))

# ------------------------------------------------------------------ 语义传播(体素哈希)
def label_transfer(xyz_s, xyz_m, label_m, leaf=0.02, thresh=0.02):
    OFFS = 32768
    def pack(c):
        return (c[:,0]+OFFS)*1000000 + (c[:,1]+OFFS)*1000 + (c[:,2]+OFFS)
    cm = np.floor(xyz_m/leaf).astype(np.int64); km = pack(cm)
    order = np.argsort(km)
    ksm = km[order]; xsm = xyz_m[order]; lm = label_m[order]
    uniq_m, start = np.unique(ksm, return_index=True)
    start = np.append(start, len(ksm))
    cs = np.floor(xyz_s/leaf).astype(np.int64)
    offs = [(dx,dy,dz) for dx in(-1,0,1) for dy in(-1,0,1) for dz in(-1,0,1)]
    n = len(xyz_s); lab = np.zeros(n, np.int32)
    for i in range(0, n, 4096):
        j = slice(i, min(i+4096, n))
        xs = xyz_s[j]; best_d = np.full(len(xs), 1e30); best_l = np.zeros(len(xs), np.int32)
        for (dx,dy,dz) in offs:
            cc = cs[j].copy()
            cc[:,0] += dx; cc[:,1] += dy; cc[:,2] += dz
            keys = pack(cc)
            u, inv = np.unique(keys, return_inverse=True)
            # searchsorted(side='left') 返回首个 >=u 的位置：pos==0 也可能是命中
            # （恰为排序后第一个体素键），原 pos>0 会把该体素邻域的点全部丢标签
            pos = np.searchsorted(uniq_m, u)
            found = (pos < len(uniq_m)) & (pos >= 0) & (uniq_m[np.clip(pos,0,len(uniq_m)-1)] == u)
            for ui in np.nonzero(found)[0]:
                k = int(pos[ui])
                qm = inv == ui; lo = start[k]; hi = start[k+1]
                if hi <= lo: continue
                cand = xsm[lo:hi]
                dmat = np.sqrt(np.sum((cand[None,:,:]-xs[qm][:,None,:])**2, 2))
                am = np.argmin(dmat, 1); dmin = dmat[np.arange(len(am)), am]
                bm = dmin < best_d[qm]
                best_l[qm] = np.where(bm, lm[lo+am], best_l[qm])
                best_d[qm] = np.where(bm, dmin, best_d[qm])
        lab[j] = np.where(best_d <= thresh, best_l, 0)
    return lab

# ------------------------------------------------------------------ 主流程
def main():
    print("[1/6] 读取语义参考图与黑白图")
    M_xyz, M_rec, M_rgb = load_ply(M_REF)
    S_xyz, S_rec, S_rgb = load_ply(S_REF)
    M_lab = np.zeros(len(M_xyz), np.int32)
    for k, c in C.items():
        m = (M_rgb[:,0]==c[0])&(M_rgb[:,1]==c[1])&(M_rgb[:,2]==c[2])
        M_lab[m] = LABEL_ID[k]
    print("  M标签:", {k:int((M_lab==v).sum()) for k,v in LABEL_ID.items()})

    leaf = 10.0                       # 配准降采样体素(mm)
    ds_s = voxel_subsample(S_xyz, leaf, 140000, 10)
    ds_m = voxel_subsample(M_xyz, leaf, 140000, 11)
    Ss = S_xyz[ds_s]; Ms = M_xyz[ds_m]
    print(f"[2/6] 配准降采样 S={len(Ss)} M={len(Ms)}")

    # 黑白图已由 gen_robot_scan.py 变换到机器人局部坐标系，无需再人为施加偏差
    S_robot = S_xyz
    Ssub = S_robot[ds_s]

    print("[4/6] 粗配准（房间适配：地面平面 + 质心初始化，墙角地标报告）")
    # 附件粗配准"关键点+特征+几何一致性+RANSAC"的舱室适配：长方体房间对称、通用特征描述子
    # 区分度差，改用结构地标——检测两侧点云一一对应的墙角，经三维距离几何一致性匹配
    # 三元组、Kabsch求解、全体角点内点数择优，完成粗配准。此处同时用地面平面+质心初始化，
    # 二者经稠密重叠度择优，再交自适应ICP解决约5°的平面内旋转。
    CS = detect_room_corners(Ssub, None); CM = detect_room_corners(Ms, None)
    print("  检测墙角地标 S=%s M=%s"%((len(CS) if CS is not None else None),(len(CM) if CM is not None else None)))
    cands, ncorner = corner_registration(CS, CM) if (CS is not None and CM is not None) else ([(np.eye(3),np.zeros(3))], -1)
    cands = cands + [floor_init(Ssub, Ms)]          # 合并地面平面初始化候选
    # 门框关键点：语义图 M 直接用门标签取门洞四角(可靠)；黑白图 S 用墙面底部贯通孔洞几何检测。
    # 两侧均得合理门框时，用门洞四角求解一门框候选加入候选集(门偏置 → 消除墙角折叠/对称歧义)。
    DS = detect_door_frame(Ssub, None)
    dmask = M_lab[ds_m] == LABEL_ID["door"]      # M 已降采样索引上的门标签
    DM = detect_door_frame(Ms, None, door_mask=dmask)
    print("  门框关键点 S=%s M=%s"%(None if DS is None else DS.shape,
                                    None if DM is None else DM.shape))
    if DS is not None and DM is not None:
        d_R, d_t = svd_transform(DS, DM)
        cands = cands + [(d_R, d_t)]
        print("  已加入门框候选(消歧义)")
    Rc, tc = select_by_overlap(cands, Ssub, Ms, th=250.0)   # 稠密重叠度择优，消除墙角对称歧义
    # 墙角检测失败(CS/CM 为 None 或空)时不输出内点统计，避免 min(len,len) 崩溃
    if CS is not None and CM is not None and len(CS) and len(CM):
        print("  粗配准候选 %d 个, 墙角内点 %d/%d"%(len(cands), ncorner, min(len(CS), len(CM))))
    else:
        print("  粗配准候选 %d 个, 墙角地标未检出, 跳过墙角内点统计"%len(cands))
    # 真值：参考系(S_REF) → 机器人系由 gt_robot_scan.npy 记录；配准目标是把机器人系映射回参考系，
    # 故真变换为 GT 之逆。
    GT = np.load(os.path.join(BASE, "gt_robot_scan.npy"))
    GTi = np.linalg.inv(GT); Rt, tt = GTi[:3,:3], GTi[:3,3]
    ec = np.linalg.norm(apply(Rc, tc, Ssub) - apply(Rt, tt, Ssub), axis=1).mean()
    print("  粗配准 vs 真值误差 %.2f mm"% ec)

    print("[5/6] 精配准(附件式自适应阈值点到面ICP)")
    # 附件精配准『源点云降采样→收敛后恢复高密度』：用子集做位姿求精，既快又稳
    nrm_m, _ = estimate_normals(Ms)
    nrm_m = orient_toward_center(Ms, nrm_m)
    icp_idx = voxel_subsample(Ssub, 20.0, 30000, 12)
    Sicp = Ssub[icp_idx]
    Rf, tf, res = point_to_plane_icp_adaptive(Sicp, Ms, nrm_m, Rc, tc, cap=600.0)
    ef = np.linalg.norm(apply(Rf, tf, Ssub) - apply(Rt, tt, Ssub), axis=1).mean()
    print("  精配准平均残差 %.2f mm, vs 真值 %.2f mm"%(res, ef))

    print("[6/6] 语义标签传播 + 保存")
    S_align = apply(Rf, tf, S_robot)
    lab = label_transfer(S_align, M_xyz, M_lab, leaf=15.0, thresh=15.0)
    out_rgb = np.zeros((len(S_xyz),3), np.int32)
    for k, v in LABEL_ID.items(): out_rgb[lab==v] = C[k]
    print("  标签划分:", {k:int((lab==v).sum()) for k,v in LABEL_ID.items()})
    write_ply(OUT, S_align, np.asarray(out_rgb,np.uint8), S_rec["part_id"])
    rms = np.sqrt(np.mean(np.sum((S_align-S_xyz)**2,1)))
    print("  对齐 vs 原始坐标 RMS %.2f mm"% rms)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, matplotlib.font_manager as fm
    zh = next((x for x in fm.findSystemFonts(None,"ttf") if any(
        k in x.lower() for k in ("yahei","simhei","msyh","notosanscjk","notoserifcjk","wenquanyi","wqy"))), None)
    if zh: matplotlib.rcParams["font.sans-serif"]=[fm.FontProperties(fname=zh).get_name()]
    matplotlib.rcParams["axes.unicode_minus"]=False
    n=len(S_xyz); maxplot=400000
    idx=np.linspace(0,n-1,maxplot).astype(np.int64) if n>maxplot else np.arange(n)
    fig=plt.figure(figsize=(18,12))
    for i,(t,az,el) in enumerate([("三维透视",30,25),("正视(看X面)",0,0),("俯视XY",-90,90),("侧视YZ",90,0)],1):
        ax=fig.add_subplot(2,2,i,projection="3d")
        ax.scatter(S_align[idx,0],S_align[idx,1],S_align[idx,2],c=out_rgb[idx]/255.0,s=0.5,depthshade=False)
        ax.set_title(t); ax.view_init(elev=el,azim=az)
        ax.set_box_aspect((np.ptp(S_align[idx,0]),np.ptp(S_align[idx,1]),np.ptp(S_align[idx,2])))
    plt.tight_layout(); plt.savefig(OUTPNG,dpi=110); print("[预览]",OUTPNG)
    print("[完成]")

if __name__ == "__main__":
    main()