# -*- coding: utf-8 -*-
"""列出所有贴墙竖直面板，检测其四方角 vs 圆角，用于区分 门/窗口/加强筋。"""
import numpy as np

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

walls = {"x0": 209082, "x1": 212292, "y0": 4380, "y1": 9429}
def wall_dq(c):
    return min(abs(c[0]-walls["x0"]), abs(c[0]-walls["x1"]),
               abs(c[1]-walls["y0"]), abs(c[1]-walls["y1"]))

def corner_roundness(m):
    """在面板局部平面内，量化四个角的圆角程度。返回 (最长边u, 最短边v, 4角填充率均值)。"""
    # 用PCA取面内两主轴
    c = m.mean(0); q = m - c
    cov = q.T @ q / len(m)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    e0 = v[:, order[0]]  # 面内长轴
    e1 = v[:, order[1]]  # 面内短轴
    u = q @ e0; wu = np.ptp(u)
    vv = q @ e1; wv = np.ptp(vv)
    if wu < wv:
        e0, e1 = e1, e0; u, vv = vv, u; wu, wv = wv, wu
    # 四个角：取u,v各自两端 ±R (R=短边20%)的正方形，统计是否存在点→角填充
    R = 0.20 * wv
    fills = []
    for su in (-1, 1):
        for sv in (-1, 1):
            lo0, hi0 = (u.min() if su < 0 else u.max() - R), (u.min() + R if su < 0 else u.max())
            lo1, hi1 = (vv.min() if sv < 0 else vv.max() - R), (vv.min() + R if sv < 0 else vv.max())
            sel = (u >= lo0) & (u <= hi0) & (vv >= lo1) & (vv <= hi1)
            fills.append(sel.mean())
    return wu, wv, float(np.mean(fills)), [float(x) for x in fills]

print(f"{'id':<4}{'n':<8}{'中心':<34}{'尺寸(x*y*z)':<18}{'贴墙距':<7}{'u*v':<10}{'角填充':<8}判定")
for p in np.unique(pid):
    m = xyz[pid == p]; lo = m.min(0); hs = m.max(0) - lo
    if hs.min() > 30 or hs[2] < 200:
        continue
    c = m.mean(0); dq = wall_dq(c)
    if dq > 400:
        continue
    wu, wv, mf, fl = corner_roundness(m)
    # 圆角判定：角填充显著<0.5 → 圆角；近似1 → 方角
    verdict = "圆角(窗?)" if mf < 0.65 else ("方角(门/筋?)" if mf > 0.92 else "规则角")
    cc = str([int(x) for x in c]); dd = str([int(x) for x in hs])
    print(f"{p:<4}{len(m):<8}{cc:<34}{dd:<18}{dq:<7.0f}"
          f"{wu:>4.0f}x{wv:<5.0f}{mf:<8.3f}{verdict}")