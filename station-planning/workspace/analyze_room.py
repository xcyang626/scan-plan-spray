import numpy as np

W = r"c:\Users\Yangon\Desktop\舱室绝缘喷涂人形机器人技术方案\喷涂点位划分\work"
xyz = np.load(W + r"\xyz.npy")
rgb = np.load(W + r"\rgb.npy")

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
    print(f"{name}: n={m.sum()}")
    print(f"   min={p.min(axis=0).round(1)}  max={p.max(axis=0).round(1)}")

# 地面高度（灰色点）
gm = mask_of(groups["gray_other"])
gz = xyz[gm, 2]
print("\ngray z range:", gz.min(), gz.max(), "-> floor")

# 墙面绿色点：分析平面结构。用RANSAC思路太重，先看在X/Y上的直方图分布
wm = mask_of(groups["wall_green"])
wp = xyz[wm]
# 降采样后统计
sub = wp[::20]
print("\nwall green subsample:", len(sub))
for ax, nm in [(0, "X"), (1, "Y")]:
    hist, edges = np.histogram(sub[:, ax], bins=40)
    print(f"\n{nm} histogram (wall):")
    for h, e in zip(hist, edges):
        if h > 0:
            print(f"  {e:10.0f}: {'#' * int(h / max(hist) * 60)} {h}")

# z分布
hist, edges = np.histogram(sub[:, 2], bins=30)
print("\nZ histogram (wall):")
for h, e in zip(hist, edges):
    if h > 0:
        print(f"  {e:10.0f}: {'#' * int(h / max(hist) * 60)} {h}")
