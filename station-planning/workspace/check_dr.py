"""验证: 调整墙面站位距墙距离 D_R 后, 原始墙面分组顺序是否无交叉"""
import numpy as np

RX, RY = 3.228, 5.058

def build(D_R):
    stations = {
        "right": [("R2", 1.29, "stiff"), ("R3", 1.89, "stiff"), ("R4", 2.29, "stiff"),
                  ("R5", 2.73, "stiff"), ("R6", 3.13, "stiff"), ("R7", 3.57, "stiff"),
                  ("R8", 3.97, "stiff"), ("R9", 4.41, "stiff")],
        "back": [("B1", 0.60, "normal"), ("B2", 1.40, "normal"),
                 ("B3", 2.01, "stiff"), ("B4", 2.61, "stiff")],
        "left": [("L1", 0.60, "stiff"), ("L2", 1.05, "stiff"), ("L3", 1.45, "stiff"),
                 ("L4", 1.89, "stiff"), ("L5", 2.29, "stiff"), ("L6", 2.73, "stiff"),
                 ("L7", 3.595, "normal"), ("L8", 4.15, "normal")],
        "front": [("F2", 0.60, "normal"), ("F3", 1.40, "normal"),
                  ("F4", 2.01, "stiff"), ("F5", 2.61, "stiff")],
    }
    corners = {"C_FL": (0.40, 0.40), "C_FR": (RX - 0.40, 0.40),
               "C_BL": (0.40, RY - 0.40), "C_BR": (RX - 0.40, RY - 0.40)}
    def station_xy(wall, s):
        _, u, _ = s
        if wall == "left":   return D_R, u
        if wall == "right":  return RX - D_R, u
        if wall == "front":  return u, D_R
        return u, RY - D_R
    def get_xy(name):
        if name in corners:
            return corners[name]
        for wall, lst in stations.items():
            for s in lst:
                if s[0] == name:
                    return station_xy(wall, s)
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

order = ["R2","R3","R4","R5","R6","R7","R8","R9","C_BR",
         "B4","B3","B2","B1","C_BL",
         "L8","L7","L6","L5","L4","L3","L2","L1","C_FL",
         "F2","F3","F4","F5","C_FR"]

for D_R in [0.80, 0.70, 0.60, 0.58, 0.55, 0.50]:
    get_xy = build(D_R)
    c, pairs = count_cross(order, get_xy)
    print(f"D_R={D_R:.2f}: 交叉数={c}")
    for p in pairs:
        print("   ", p[0], "->", p[1], " 与 ", p[2], "->", p[3])
