# -*- coding: utf-8 -*-
"""可靠提取四面墙开洞矩形：正确轴向 + 打印占据率自证。"""
import numpy as np
from collections import deque

PLY = r"c:\Users\Yangon\Desktop\舱室绝缘喷涂人形机器人技术方案\点云.py\舱室点云\舱室装配体点云.ply"
with open(PLY, "rb") as f:
    while f.readline().strip() != b"end_header":
        pass
    body = f.read()
dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("part_id", "<u2"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
r = np.frombuffer(body, dtype=dt)
xyz = np.stack([r["x"], r["y"], r["z"]], axis=1)
pid = r["part_id"]

GRID = 500

def conn_label(mask):
    H, W = mask.shape; lab = np.zeros_like(mask, dtype=np.int32); cur = 0
    for i in range(H):
        for j in range(W):
            if mask[i, j] and not lab[i, j]:
                cur += 1; lab[i, j] = cur; dq = deque([(i, j)])
                while dq:
                    x, y = dq.popleft()
                    for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
                        nx, ny = x+dx, y+dy
                        if 0 <= nx < H and 0 <= ny < W and mask[nx, ny] and not lab[nx, ny]:
                            lab[nx, ny] = cur; dq.append((nx, ny))
    return lab, cur

def opening(p, axis, name):
    m = xyz[pid == p]; lo = m.min(0); hi = m.max(0)
    axs = [i for i in range(3) if i != axis]     # 面内两轴
    U = m[:, axs[0]]; V = m[:, axs[1]]
    occ, _, _ = np.histogram2d(U, V, bins=GRID)  # 行=U, 列=V
    mat = occ > 0
    print(f"== 墙{name} id={p} 面内轴={label(axs[0])}{label(axs[1])}(emb) bbox "
          f"{label(axs[0])}:{U.min():.0f}~{U.max():.0f} {label(axs[1])}:{V.min():.0f}~{V.max():.0f} "
          f"厚:{hi[axis]-lo[axis]:.0f} 占据率:{mat.mean():.2f}", flush=True)
    empty = ~mat
    lab, _ = conn_label(empty)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:,0], lab[:,-1]]))) - {0}
    inner_lbl = set(np.unique(lab)) - border
    inner_lbl.discard(0)                    # 0=实体, 剔除
    inner = np.isin(lab, list(inner_lbl))
    hl, nh = conn_label(inner)
    cell_area = (U.max()-U.min())/GRID * (V.max()-V.min())/GRID   # 单格mm²
    found = []
    for h in range(1, nh+1):
        ys,xs = np.where(hl==h); y0,y1,x0,x1 = ys.min(),ys.max(),xs.min(),xs.max()
        cells = (hl[y0:y1+1,x0:x1+1]==h).sum()
        w=(x1-x0+1)*(U.max()-U.min())/GRID; ht=(y1-y0+1)*(V.max()-V.min())/GRID
        if cells * cell_area < 40000:      # 过滤噪声小洞 (<40k mm²)
            continue
        cor = hl[y0:y1+1,x0:x1+1]==h
        fc = sum([not cor[0,0], not cor[0,-1], not cor[-1,0], not cor[-1,-1]])
        u0=U.min()+(x0+0.5)*(U.max()-U.min())/GRID; u1=U.min()+(x1+0.5)*(U.max()-U.min())/GRID
        v0=V.min()+(y0+0.5)*(V.max()-V.min())/GRID; v1=V.min()+(y1+0.5)*(V.max()-V.min())/GRID
        kind = "门(方角)" if fc>=3 else "窗(圆角)" if fc<=1 else f"混合(角{fc}/4)"
        found.append((w, ht, v0, v1, fc, kind, u0, u1))
    found.sort(reverse=True)
    for w, ht, v0, v1, fc, kind, u0, u1 in found:
        print(f"  开洞: {w:.0f}x{ht:.0f}mm 下边Z={v0:.0f} 上边Z={v1:.0f} "
              f"{label(axs[0])}[{u0:.0f},{u1:.0f}] -> {kind}", flush=True)

def label(a):
    return {0:"X",1:"Y",2:"Z"}[a]

for p, axis, name in [(264,0,"X面右"),(287,0,"X面左"),(275,1,"Y面前"),(276,1,"Y面后")]:
    opening(p, axis, name)