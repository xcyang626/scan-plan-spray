"""检测走位连线交叉: 遍历所有非相邻线段对, 判断是否相交"""
import numpy as np

RX, RY = 3.228, 5.058
D_R = 0.80

stations = {
    "right": [("R2", 1.29, "stiff"), ("R3", 1.89, "stiff"), ("R4", 2.29, "stiff"),
              ("R5", 2.73, "stiff"), ("R6", 3.13, "stiff"), ("R7", 3.57, "stiff"),
              ("R8", 3.97, "stiff"), ("R9", 4.41, "stiff")],
    "back": [("B1", 0.60, "normal"), ("B2", 1.40, "normal"),
             ("B3", 2.01, "stiff"), ("B4", 2.61, "stiff")],
    "left": [("L1", 0.60, "stiff"), ("L2", 1.05, "stiff"), ("L3", 1.45, "stiff"),
             ("L4", 1.89, "stiff"), ("L5", 2.29, "stiff"), ("L6", 2.73, "stiff"),
             ("L7", 3.53, "normal"), ("L8", 4.15, "normal")],
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

# 墙角局部交换顺序 (与 fig_topview.py 一致): 墙角站位插入墙面中间消除交叉
order = ["R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "B4", "C_BR",
         "B3", "B2", "C_BL", "B1",
         "L8", "L7", "L6", "L5", "L4", "L3", "L2", "L1", "F2", "C_FL",
         "F3", "F4", "F5", "C_FR"]

pts = [get_xy(n) for n in order]
seg = [(pts[i], pts[i + 1], i, i + 1) for i in range(len(pts) - 1)]

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

print("=== 交叉检测 (非相邻线段) ===")
cross = 0
for i in range(len(seg)):
    for j in range(i + 1, len(seg)):
        if j == i + 1:
            continue  # 相邻线段共享端点, 不算交叉
        p1, p2, a, b = seg[i]
        p3, p4, c, d = seg[j]
        if intersect(p1, p2, p3, p4):
            cross += 1
            print(f"  交叉: {order[a]}→{order[b]} (seg{i}) 与 {order[c]}→{order[d]} (seg{j})")
print("总交叉数:", cross)

# 走位长度
pl = np.array(pts)
open_len = sum(np.linalg.norm(pl[i + 1] - pl[i]) for i in range(len(pl) - 1))
loop_len = open_len + np.linalg.norm(pl[-1] - pl[0])
print(f"开放走位={open_len:.2f} m, 闭合环线={loop_len:.2f} m")
