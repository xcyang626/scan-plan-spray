"""图: 整个舱室 工位点位俯视图 (四面墙全部站位, 含作业顺序编号与走位连线)
   站位表/墙角/作业顺序均取自 stations.py 单一数据源:
   墙角两墙站位合并为一个(对角内收, 距两墙各40cm, 45°面向墙角);
   作业顺序 = 局部交换序(墙角站位插入墙面中间消除走位交叉), 共28站位"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from stations import (RX, RY, D_R, STATIONS, CORNERS, ORDER,
                      RIBS, WINDOWS, DOOR, get_xy, get_dir, get_type)

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 数据/图片目录: 优先环境变量, 未设时默认脚本同级 "数据" 目录与仓库 figures 目录
W = Path(os.environ.get("STATION_DATA", Path(__file__).resolve().parent / "数据"))
FIG = Path(os.environ.get("STATION_FIGS", Path(__file__).resolve().parent.parent / "figures"))
for f in ("xyz.npy", "rgb.npy"):
    if not (W / f).exists():
        sys.exit(f"[路径] 缺少数据文件 {W / f}: 请将点云缓存(xyz.npy/rgb.npy)放入该目录, "
                 f"或设置环境变量 STATION_DATA 指向数据目录")

xyz = np.load(W / "xyz.npy") / 1000.0
rgb = np.load(W / "rgb.npy")
ORIGIN = np.array([209073.0, 4380.0, 57558.0]) / 1000.0
xyz = xyz - ORIGIN

def mask_of(c):
    return np.all(rgb == np.array(c, dtype=np.uint8), axis=1)

rng = np.random.default_rng(11)

type_color = {"corner": "#C0392B", "stiff": "#E67E22", "normal": "#2471A3"}

order = ORDER

# 走位长度
pts = np.array([get_xy(n) for n in order])
open_len = sum(np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1))
loop_len = open_len + np.linalg.norm(pts[-1] - pts[0])

fig, ax = plt.subplots(figsize=(9.0, 11.6))

# 障碍物俯视投影 (z 0.1~2.0)
obst = xyz[mask_of((0, 0, 0))]
m = (obst[:, 2] > 0.10) & (obst[:, 2] < 2.0)
p = obst[m]
if len(p) > 80000:
    p = p[rng.choice(len(p), 80000, replace=False)]
ax.scatter(p[:, 0], p[:, 1], s=0.9, c="#34495E", alpha=0.28, linewidths=0, zorder=2)

# 加强筋 (墙上蓝色块)
for su in RIBS["left"]:
    ax.add_patch(Rectangle((-0.06, su - 0.03), 0.06, 0.06, fc="#1F6FEB", ec="none", zorder=5))
for su in RIBS["right"]:
    ax.add_patch(Rectangle((RX, su - 0.03), 0.06, 0.06, fc="#1F6FEB", ec="none", zorder=5))
for su in RIBS["front"]:
    ax.add_patch(Rectangle((su - 0.03, -0.06), 0.06, 0.06, fc="#1F6FEB", ec="none", zorder=5))
for su in RIBS["back"]:
    ax.add_patch(Rectangle((su - 0.03, RY), 0.06, 0.06, fc="#1F6FEB", ec="none", zorder=5))

# 窗 (黄色段, 画在墙外侧)
for w0, w1 in WINDOWS["left"]:
    ax.add_patch(Rectangle((-0.10, w0), 0.10, w1 - w0, fc="#F4D03F", ec="#B7950B", lw=0.8, zorder=5))
for w0, w1 in WINDOWS["right"]:
    ax.add_patch(Rectangle((RX, w0), 0.10, w1 - w0, fc="#F4D03F", ec="#B7950B", lw=0.8, zorder=5))
for w0, w1 in WINDOWS["front"]:
    ax.add_patch(Rectangle((w0, -0.10), w1 - w0, 0.10, fc="#F4D03F", ec="#B7950B", lw=0.8, zorder=5))
for w0, w1 in WINDOWS["back"]:
    ax.add_patch(Rectangle((w0, RY), w1 - w0, 0.10, fc="#F4D03F", ec="#B7950B", lw=0.8, zorder=5))
# 门 (紫色段)
ax.add_patch(Rectangle((RX, DOOR[0]), 0.10, DOOR[1] - DOOR[0], fc="#B250C8", ec="#7D3C98", lw=0.8, zorder=5))
ax.text(RX + 0.17, (DOOR[0] + DOOR[1]) / 2 + 0.30, "门", fontsize=10, color="#7D3C98",
        rotation=90, va="center", zorder=6)

# 墙体
ax.add_patch(Rectangle((-0.06, -0.06), RX + 0.12, RY + 0.12, fill=False, ec="#2C3E50", lw=3.0, zorder=6))

# ===== 走位连线 (作业顺序) =====
path = [get_xy(n) for n in order]
ax.plot([q[0] for q in path], [q[1] for q in path], color="#95A5A6", ls="--",
        lw=1.3, alpha=0.85, zorder=6.5)

# ===== 站位点 (箭头指向目标墙面/墙角 + 顺序编号) =====
for i, name in enumerate(order, start=1):
    c = type_color[get_type(name)]
    x, y = get_xy(name)
    dx, dy = get_dir(name)
    # 朝向箭头: 从站位点指向墙面/墙角
    ax.annotate("", xy=(x + dx * 0.42, y + dy * 0.42), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=1.7, mutation_scale=14), zorder=7)
    ax.plot(x, y, marker="o", ms=7.5, mfc=c, mec="white", mew=1.0, zorder=8)
    # 双行标签: 顺序号(主) + 站位名 (紧贴点位房间内侧, 与朝向箭头反向)
    if name in CORNERS:
        # 墙角站位: 标签向房间内部偏移
        off = (dx * -0.30, dy * -0.30)
    elif name[0] == "L":
        off = (0.18, 0)
    elif name[0] == "R":
        off = (-0.20, 0)
    elif name[0] == "F":
        off = (0, 0.20)
    else:
        off = (0, -0.22)
    ax.text(x + off[0], y + off[1], f"{i}\n{name}", fontsize=8.4, fontweight="bold",
            color=c, ha="center", va="center", zorder=9, linespacing=1.0,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=c, lw=0.8, alpha=0.92))

# ===== 进门 / 出门方向 =====
ax.annotate("", xy=(2.92, 1.16), xytext=(3.90, 0.95),
            arrowprops=dict(arrowstyle="-|>", color="#7D3C98", lw=2.2, mutation_scale=16), zorder=7)
ax.text(3.55, 1.32, "进门方向", fontsize=9.5, color="#7D3C98", ha="center", fontweight="bold", zorder=9)
_lx, _ly = get_xy(order[-1])
ax.annotate("", xy=(RX + 0.12, 0.82), xytext=(_lx, _ly),
            arrowprops=dict(arrowstyle="-|>", color="#7F8C8D", lw=1.8, mutation_scale=13,
                            linestyle="dashed"), zorder=7)
ax.text((_lx + RX + 0.12) / 2, (_ly + 0.82) / 2 - 0.18, "出门方向(经门离开)", fontsize=9, color="#7F8C8D", ha="center", zorder=9)

# 尺寸标注
ax.annotate("", xy=(RX, -0.30), xytext=(0, -0.30),
            arrowprops=dict(arrowstyle="<->", color="k", lw=1.0))
ax.text(RX / 2, -0.42, "3.23 m", ha="center", fontsize=9)
ax.annotate("", xy=(-0.30, RY), xytext=(-0.30, 0),
            arrowprops=dict(arrowstyle="<->", color="k", lw=1.0))
ax.text(-0.44, RY / 2, "5.06 m", va="center", rotation=90, fontsize=9)
ax.text(RX / 2, RY + 0.24, f"墙面站位距墙 d_r = {D_R:.2f} m；墙角合并站位距两墙各 0.40 m（箭头=机器人正对墙面/墙角）；圆点编号=喷涂作业顺序",
        ha="center", fontsize=10, color="#34495E")
ax.text(RX / 2, RY + 0.44, "作业顺序：进门后先喷右墙 → 逆时针绕行（1→28，动线无交叉）",
        ha="center", fontsize=10, color="#7D3C98")

from matplotlib.lines import Line2D
handles = [
    Line2D([], [], marker="o", ls="", ms=8, mfc="#C0392B", mec="white", label="墙角合并站位 (4个,距两墙各40cm,45°面向墙角)"),
    Line2D([], [], marker="o", ls="", ms=8, mfc="#E67E22", mec="white", label="加强筋定点 (18个,负责墙面)"),
    Line2D([], [], marker="o", ls="", ms=8, mfc="#2471A3", mec="white", label="普通站位 (6个,负责墙面)"),
    Line2D([], [], color="#95A5A6", ls="--", label="走位顺序连线(1→28)"),
    Line2D([], [], marker="s", ls="", ms=8, mfc="#1F6FEB", label="加强筋(墙)"),
    Line2D([], [], marker="s", ls="", ms=8, mfc="#F4D03F", label="窗"),
    Line2D([], [], marker="s", ls="", ms=8, mfc="#B250C8", label="门"),
    Line2D([], [], marker="s", ls="", ms=8, mfc="#34495E", alpha=0.4, label="障碍物(z<2m投影)"),
]
ax.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, framealpha=0.92,
          bbox_to_anchor=(0.5, -0.075))

ax.set_xlim(-0.75, RX + 0.85)
ax.set_ylim(-0.75, RY + 0.60)
ax.set_xlabel("x (m)", fontsize=11)
ax.set_ylabel("y (m)", fontsize=11)
ax.set_title("舱室喷涂工位点位总览（俯视图，共28个站位点，编号为作业顺序）", fontsize=13)
ax.set_aspect("equal")
ax.grid(alpha=0.25, ls=":")
plt.tight_layout()
plt.savefig(FIG / "fig_topview.png", dpi=200, bbox_inches="tight")
plt.close()
print("fig_topview done")

# 输出顺序表 (供文档使用)
cnt = {"corner": 0, "stiff": 0, "normal": 0}
for n in order:
    cnt[get_type(n)] += 1
print("total stations:", len(order), cnt)
print("走位长度: 闭合环线=%.2f m, 开放走位=%.2f m" % (loop_len, open_len))
print("\n作业顺序表:")
for i, n in enumerate(order, start=1):
    x, y = get_xy(n)
    print(f"  #{i:2d} {n:4s} type={get_type(n):6s} xy=({x:.2f},{y:.2f})")
