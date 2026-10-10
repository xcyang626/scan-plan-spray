"""验证覆盖带规则(脚本现行: 中点截断, 互不重叠相接): 站位表见 stations.py 单一数据源
   判据: 每条覆盖带不超过站位左0.20m右0.60m; 相邻覆盖带相接铺满整墙; 筋位于带边界;
   窗带由相邻站位完全覆盖; 各墙站位间距 <= 0.80"""
from stations import (STATIONS, WALL_LEN, RIBS, WINDOWS,
                      BAND_LEFT, BAND_RIGHT, BAND_WIDTH, CORNER_W, wall_zones)

LW = WALL_LEN["left"]

# 左墙站位序列(含两端墙角责任带站位): C_FL + L1..L8 + C_BL
zones = [("C_FL", CORNER_W, 0.00, CORNER_W, "corner")]
zones += [(nm, u, z0, z1, "wall") for nm, u, z0, z1 in wall_zones("left")]
zones.append(("C_BL", LW - CORNER_W, LW - CORNER_W, LW, "corner"))

print("站位    u     覆盖带         宽    可达范围[u-0.2,u+0.6]  校验")
ok = True
for nm, u, z0, z1, kind in zones:
    lim_lo, lim_hi = u - BAND_LEFT, u + BAND_RIGHT
    if kind == "corner":
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

for r in RIBS["left"]:
    zs = [z[0] for z in zones if z[2] - 1e-9 <= r <= z[3] + 1e-9]
    print(f"筋@{r}: 位于 {zs} 的覆盖带边界处")
zl = {z[0]: z for z in zones}
w0, w1 = WINDOWS["left"][0]
print(f"窗[{w0},{w1}] 由L7[{zl['L7'][2]:.2f},{zl['L7'][3]:.2f}]+L8[{zl['L8'][2]:.2f},{zl['L8'][3]:.2f}]完全覆盖:",
      w0 >= zl["L7"][2] and w1 <= zl["L8"][3])

for wall, lst in STATIONS.items():
    arr = [s[1] for s in lst]
    gaps = [round(arr[i + 1] - arr[i], 2) for i in range(len(arr) - 1)]
    print(f"{wall}墙 站位间距: {gaps} 最大={max(gaps)} (<=0.80: {max(gaps) <= BAND_WIDTH + 1e-6})")
print("左墙所有覆盖带不超限:", ok)
