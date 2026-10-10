import os
import sys
from pathlib import Path

import numpy as np

# 数据目录: 优先环境变量 STATION_DATA, 未设时默认脚本同级 "数据" 目录(仓库不携带点云缓存)
W = Path(os.environ.get("STATION_DATA", Path(__file__).resolve().parent / "数据"))
for f in ("xyz.npy", "rgb.npy"):
    if not (W / f).exists():
        sys.exit(f"[路径] 缺少数据文件 {W / f}: 请将点云缓存(xyz.npy/rgb.npy)放入该目录, "
                 f"或设置环境变量 STATION_DATA 指向数据目录")

xyz = np.load(W / "xyz.npy") / 1000.0   # mm -> m (统一单位为米)
rgb = np.load(W / "rgb.npy")

# 语义分组（按颜色）
def mask_of(c):
    return np.all(rgb == np.array(c, dtype=np.uint8), axis=1)

groups = {
    "wall_green": (150, 220, 150),
    "stiffener_blue": (0, 110, 255),
    "obstacle_black": (0, 0, 0),
    "window_yellow": (255, 230, 0),
    "door_purple": (178, 64, 222),
    "gray_other": (128, 128, 128),
}

for name, c in groups.items():
    m = mask_of(c)
    p = xyz[m]
    if len(p) == 0:
        print(f"{name}: n=0  min/max: none (该颜色无点)")
        continue
    print(f"{name}: n={m.sum()}")
    print(f"   min={p.min(axis=0).round(3)}  max={p.max(axis=0).round(3)}")

# 地面高度（灰色点）
gm = mask_of(groups["gray_other"])
gz = xyz[gm, 2]
if len(gz) == 0:
    print("\ngray z range: none (灰色组无点)")
else:
    print("\ngray z range: %.3f ~ %.3f m -> floor" % (gz.min(), gz.max()))

# 墙面绿色点：分析平面结构。用RANSAC思路太重，先看在X/Y上的直方图分布
wm = mask_of(groups["wall_green"])
wp = xyz[wm]
# 降采样后统计
sub = wp[::20]
print("\nwall green subsample:", len(sub))
for ax, nm in [(0, "X"), (1, "Y")]:
    hist, edges = np.histogram(sub[:, ax], bins=40)
    print(f"\n{nm} histogram (wall, m):")
    for h, e in zip(hist, edges):
        if h > 0:
            print(f"  {e:10.3f}: {'#' * int(h / max(hist) * 60)} {h}")

# z分布
hist, edges = np.histogram(sub[:, 2], bins=30)
print("\nZ histogram (wall, m):")
for h, e in zip(hist, edges):
    if h > 0:
        print(f"  {e:10.3f}: {'#' * int(h / max(hist) * 60)} {h}")
