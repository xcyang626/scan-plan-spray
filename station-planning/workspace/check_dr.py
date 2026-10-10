"""验证: 调整墙面站位距墙距离 D_R 后, 原始墙面分组顺序是否无交叉
   站位表见 stations.py 单一数据源 (墙角站位不随 D_R 内收)"""
import numpy as np

from stations import ORDER_RAW, STATIONS, CORNERS, RX, RY

def build(d_r):
    def get_xy(name):
        if name in CORNERS:
            return CORNERS[name]
        for wall, lst in STATIONS.items():
            for s in lst:
                if s[0] == name:
                    _, u, _ = s
                    if wall == "left":
                        return d_r, u
                    if wall == "right":
                        return RX - d_r, u
                    if wall == "front":
                        return u, d_r
                    return u, RY - d_r
        raise KeyError(name)
    return get_xy

def orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

def on_seg(a, b, p):
    return (min(a[0], b[0]) - 1e-9 <= p[0] <= max(a[0], b[0]) + 1e-9 and
            min(a[1], b[1]) - 1e-9 <= p[1] <= max(a[1], b[1]) + 1e-9)

def intersect(p1, p2, p3, p4):
    d1 = orient(p3, p4, p1); d2 = orient(p3, p4, p2)
    d3 = orient(p1, p2, p3); d4 = orient(p1, p2, p4)
    if ((d1 > 1e-9 and d2 < -1e-9) or (d1 < -1e-9 and d2 > 1e-9)) and \
       ((d3 > 1e-9 and d4 < -1e-9) or (d3 < -1e-9 and d4 > 1e-9)):
        return True
    if abs(d1) < 1e-9 and on_seg(p3, p4, p1): return True
    if abs(d2) < 1e-9 and on_seg(p3, p4, p2): return True
    if abs(d3) < 1e-9 and on_seg(p1, p2, p3): return True
    if abs(d4) < 1e-9 and on_seg(p1, p2, p4): return True
    return False

def count_cross(order, get_xy):
    pts = [get_xy(n) for n in order]
    seg = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    c = 0; pairs = []
    for i in range(len(seg)):
        for j in range(i + 1, len(seg)):
            if j == i + 1:
                continue
            if intersect(seg[i][0], seg[i][1], seg[j][0], seg[j][1]):
                c += 1; pairs.append((order[i], order[i+1], order[j], order[j+1]))
    return c, pairs

order = ORDER_RAW

for d_r in [0.80, 0.70, 0.60, 0.58, 0.55, 0.50]:
    c, pairs = count_cross(order, build(d_r))
    print(f"D_R={d_r:.2f}: 交叉数={c}")
    for p in pairs:
        print("   ", p[0], "->", p[1], " 与 ", p[2], "->", p[3])
