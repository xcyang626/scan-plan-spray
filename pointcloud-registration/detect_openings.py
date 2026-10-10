# -*- coding: utf-8 -*-
"""在四面墙上检测开洞组件，报告洞的包围盒、尺寸，并判断方角(门)还是圆角(窗)。"""
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

walls = {264: "X面(右05212)", 287: "X面(左09082)", 275: "Y面(前04380)", 276: "Y面(后09429)"}
GRID = 300

def labels_4conn(mask):
    """4联通标记，返回同尺寸int标记，0=背景。"""
    H, W = mask.shape
    lab = np.zeros((H, W), dtype=np.int32)
    cur = 0
    for i in range(H):
        for j in range(W):
            if mask[i, j] and not lab[i, j]:
                cur += 1
                lab[i, j] = cur
                dq = deque([(i, j)])
                while dq:
                    x, y = dq.popleft()
                    for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
                        nx, ny = x+dx, y+dy
                        if 0 <= nx < H and 0 <= ny < W and mask[nx, ny] and not lab[nx, ny]:
                            lab[nx, ny] = cur
                            dq.append((nx, ny))
    return lab, cur

def find_holes(p, axis, name):
    m = xyz[pid == p]
    axs = [i for i in range(3) if i != axis]
    U = m[:, axs[0]]; V = m[:, axs[1]]      # U 水平、V 高度
    bu = np.linspace(U.min(), U.max(), GRID+1)
    bv = np.linspace(V.min(), V.max(), GRID+1)
    occ, _, _ = np.histogram2d(V, U, bins=[bv, bu])
    mat = occ > 0                            # True=材料
    empty = ~mat
    # 去掉与边框连通的外部空白（只留内部洞）
    lab, _ = labels_4conn(empty)
    border_lbls = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:,0], lab[:,-1]]))) - {0}
    hole_mask = np.isin(lab, list(set(np.unique(lab)) - border_lbls))
    hl, nh = labels_4conn(hole_mask)
    holes = []
    for h in range(1, nh+1):
        ys, xs = np.where(hl == h)
        y0,y1,x0,x1 = ys.min(), ys.max(), xs.min(), xs.max()
        hmm = hl[y0:y1+1, x0:x1+1] == h
        hch = (hmm == False)  # 洞bbox内非本洞=材料
        # 角是否被材料占据(圆角) —— 洞bbox四角
        corners = [hch[0,0], hch[0,-1], hch[-1,0], hch[-1,-1]]
        filled_corners = sum(corners)  # 1=该角为材料 → 洞未填满bbox → 圆角
        w = (x1-x0+1)*(U.max()-U.min())/GRID
        ht = (y1-y0+1)*(V.max()-V.min())/GRID
        holes.append((h, w, ht, filled_corners,
                      U.min()+(x0+0.5)*(U.max()-U.min())/GRID,
                      V.min()+(y0+0.5)*(V.max()-V.min())/GRID,
                      (x0,x1,y0,y1)))
    print(f"\n=== 墙{name} 面内 宽{U.max()-U.min():.0f} 高{V.max()-V.min():.0f}, 开洞数={len(holes)} ===")
    for (h, w, ht, fc, cu, cv, bb) in sorted(holes, key=lambda a:-a[1]*a[2]):
        # 方向修正：洞bbox四角被材料占据(fc大)说明洞是圆角(窗)，洞填满bbox(fc小)才是方角(门)
        # （与 detect_door_window.py 的面板角填充方向相反：面板角有材料=方角，洞角有材料=圆角）
        kind = "圆角(疑似窗)" if fc >= 3 else ("方角(疑似门)" if fc <= 1 else "混合角")
        print(f"  洞#{h}: {w:.0f}mm宽 x {ht:.0f}mm高, 角填充(filled_corners)={fc}/4 -> {kind}, 中心(U={cu:.0f}.., 高V={cv:.0f}..)")

for p, name in walls.items():
    find_holes(p, 0 if p in (264,287) else 1, name)