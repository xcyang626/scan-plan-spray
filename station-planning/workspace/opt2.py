"""对开放走位应用 2-opt, 检查交叉是否消除, 输出新顺序"""
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

def count_cross(order):
    pts = [get_xy(n) for n in order]
    seg = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    c = 0; pairs = []
    for i in range(len(seg)):
        for j in range(i + 1, len(seg)):
            if j == i + 1:
                continue
            if intersect(seg[i][0], seg[i][1], seg[j][0], seg[j][1]):
                c += 1; pairs.append((i, j, order[i], order[i+1], order[j], order[j+1]))
    return c, pairs

def path_len(order):
    pts = [get_xy(n) for n in order]
    return sum(np.linalg.norm(np.array(pts[i+1]) - np.array(pts[i])) for i in range(len(pts)-1))

order0 = ["R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "C_BR",
          "B4", "B3", "B2", "B1", "C_BL",
          "L8", "L7", "L6", "L5", "L4", "L3", "L2", "L1", "C_FL",
          "F2", "F3", "F4", "F5", "C_FR"]

c0, p0 = count_cross(order0)
print("原始顺序交叉数:", c0)
for p in p0:
    print("   ", p[2], "->", p[3], " 与 ", p[4], "->", p[5])
print("原始长度: %.2f m" % path_len(order0))

# 2-opt (开放路径): 反转 i+1..j 段
order = list(order0)
improved = True
while improved:
    improved = False
    n = len(order)
    for i in range(n - 1):
        for j in range(i + 2, n):
            # 边 (i,i+1) 与 (j,j+1) (j+1 < n)
            if j + 1 >= n:
                continue
            a, b = get_xy(order[i]), get_xy(order[i+1])
            c, d = get_xy(order[j]), get_xy(order[j+1])
            old = np.linalg.norm(np.array(b)-np.array(a)) + np.linalg.norm(np.array(d)-np.array(c))
            new = np.linalg.norm(np.array(c)-np.array(a)) + np.linalg.norm(np.array(d)-np.array(b))
            if new < old - 1e-9:
                order[i+1:j+1] = reversed(order[i+1:j+1])
                improved = True

c1, p1 = count_cross(order)
print("\n2-opt 后交叉数:", c1)
for p in p1:
    print("   ", p[2], "->", p[3], " 与 ", p[4], "->", p[5])
print("2-opt 后长度: %.2f m" % path_len(order))
print("\n2-opt 顺序:")
print(" -> ".join(order))
