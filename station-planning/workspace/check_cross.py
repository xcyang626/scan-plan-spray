"""检测走位连线交叉: 遍历所有非相邻线段对, 判断是否相交
   站位表与作业顺序(局部交换序)见 stations.py 单一数据源"""
import numpy as np

from stations import ORDER, get_xy

order = ORDER

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
