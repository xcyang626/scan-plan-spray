"""验证新覆盖带规则: 互不重叠、相接铺满整面墙、每条覆盖带不超过站位左0.20m右0.60m(可小于)
   边界 = 相邻站位中点, 截断至两站位可达范围交集 (>=右站位u-0.20, <=左站位u+0.60)"""
LW = 5.06
CORNER_W = 0.40
st = [('C_FL', 0.40, 'corner'), ('L1', 0.60, 'stiff'), ('L2', 1.05, 'stiff'),
      ('L3', 1.45, 'stiff'), ('L4', 1.89, 'stiff'), ('L5', 2.29, 'stiff'),
      ('L6', 2.73, 'stiff'), ('L7', 3.53, 'normal'), ('L8', 4.15, 'normal'),
      ('C_BL', 4.66, 'corner')]
ribs = [0.85, 1.69, 2.53]
win = (3.49, 4.10)
us = [s[1] for s in st]

# 边界: 墙端0 与 墙角带边缘0.40/4.66 固定; 墙面站位对(L1..L8)取中点截断
bs = [0.00, CORNER_W]
for i in range(1, len(st) - 2):   # L1|L2 ... L7|L8
    a, b = us[i], us[i + 1]
    mid = (a + b) / 2
    lo = b - 0.20
    hi = a + 0.60
    v = min(max(mid, lo), hi)
    bs.append(round(v, 4))
    print(f"  b {st[i][0]}|{st[i+1][0]}: mid={mid:.3f} 区间[{lo:.2f},{hi:.2f}] -> {v:.2f}")
bs += [LW - CORNER_W, LW]
print("boundaries:", bs)

ok = True
print()
print("站位    u     覆盖带         宽    可达范围[u-0.2,u+0.6]  校验")
zones = []
for i, (nm, u, tp) in enumerate(st):
    z0, z1 = bs[i], bs[i + 1]
    zones.append((nm, tp, z0, z1, u))
    lim_lo, lim_hi = u - 0.20, u + 0.60
    if tp == 'corner':
        c = "墙角责任带(固定)"
    else:
        inside = (z0 >= lim_lo - 1e-9) and (z1 <= lim_hi + 1e-9)
        c = "OK" if inside else "超限!"
        ok = ok and inside
    print(f"{nm:6s} {u:5.2f} [{z0:.2f},{z1:.2f}] {z1-z0:5.2f}  [{lim_lo:.2f},{lim_hi:.2f}]  {c}")

full = True
for i in range(len(zones) - 1):
    if abs(zones[i][3] - zones[i + 1][2]) > 1e-9:
        full = False
        print("相邻覆盖带不相接:", zones[i][0], zones[i + 1][0])
print("覆盖带相接(无重叠):", full)
print("首尾覆盖: [%.2f, %.2f] 墙宽 %.2f -> 全覆盖:" % (zones[0][2], zones[-1][3], LW),
      abs(zones[0][2]) < 1e-9 and abs(zones[-1][3] - LW) < 1e-9)
print("总宽:", round(sum(z[3] - z[2] for z in zones), 4))

for r in ribs:
    zs = [z[0] for z in zones if z[2] - 1e-9 <= r <= z[3] + 1e-9]
    print(f"筋@{r}: 位于 {zs} 的覆盖带边界处")
print(f"窗[{win[0]},{win[1]}] 由L7[{zones[7][2]:.2f},{zones[7][3]:.2f}]+L8[{zones[8][2]:.2f},{zones[8][3]:.2f}]完全覆盖:",
      win[0] >= zones[7][2] and win[1] <= zones[8][3])

walls = {
    "left":  [0.60, 1.05, 1.45, 1.89, 2.29, 2.73, 3.53, 4.15],
    "right": [1.29, 1.89, 2.29, 2.73, 3.13, 3.57, 3.97, 4.41],
    "back":  [0.60, 1.40, 2.01, 2.61],
    "front": [0.60, 1.40, 2.01, 2.61]}
for w, arr in walls.items():
    gaps = [round(arr[i + 1] - arr[i], 2) for i in range(len(arr) - 1)]
    print(f"{w}墙 站位间距: {gaps} 最大={max(gaps)} (<=0.80: {max(gaps) <= 0.800001})")
print("左墙所有覆盖带不超限:", ok)
