"""验证候选顺序的交叉数与长度: 站位表见 stations.py 单一数据源
   候选: 原始(墙面分组) / 局部交换(墙角微调, 采纳序) / 2-opt"""
import numpy as np

from stations import ORDER, ORDER_RAW, get_xy

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
                c += 1; pairs.append((order[i], order[i+1], order[j], order[j+1]))
    return c, pairs

def path_len(order):
    pts = [get_xy(n) for n in order]
    return sum(np.linalg.norm(np.array(pts[i+1]) - np.array(pts[i])) for i in range(len(pts)-1))

candidates = {
    # 原始墙面分组: 各墙走完后经墙角站位转入下一墙 (D_R=0.80 有 3 处交叉)
    "原始(墙面分组)": ORDER_RAW,
    # 局部交换: 墙角站位插入墙面中间消除交叉 (B4→C_BR→B3, B2→C_BL→B1, F2→C_FL→F3)
    "局部交换(墙角微调)": ORDER,
    # 2-opt 结果 (由 opt2.py 对 ORDER_RAW 迭代得到, 此处静态列出供对比)
    "2-opt": ["R2","R3","R4","R5","R6","R7","R8","B4","C_BR","R9","B3","B2","C_BL","B1",
              "L8","L7","L6","L5","L4","L3","L2","F2","C_FL","L1","F3","F4","F5","C_FR"],
}

for name, order in candidates.items():
    c, pairs = count_cross(order)
    print(f"=== {name} ===")
    print(f"  交叉数: {c}, 长度: {path_len(order):.2f} m")
    for p in pairs:
        print("   ", p[0], "->", p[1], " 与 ", p[2], "->", p[3])
    print()
