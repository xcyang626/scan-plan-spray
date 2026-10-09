"""图: 示例墙(左墙) 点云2D投影 + 梯形分解法区域划分 + 站位点规划(含作业顺序)
   覆盖策略: 墙角站位取距墙端40cm处(相邻两墙墙角站位合并为一个, 取中心点, 45°面向墙角),
   仅负责墙角0~0.40m责任带; 墙面站位覆盖带互不重叠、相接铺满整面墙(含窗带);
   每条覆盖带严格不超过站位点左侧0.20m、右侧0.60m(可小于该范围)"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

W = r"c:\Users\Yangon\Desktop\舱室绝缘喷涂人形机器人技术方案\喷涂点位划分\work"
FIG = r"c:\Users\Yangon\Desktop\舱室绝缘喷涂人形机器人技术方案\喷涂点位划分\figs"

xyz = np.load(W + r"\xyz.npy") / 1000.0
rgb = np.load(W + r"\rgb.npy")
ORIGIN = np.array([209073.0, 4380.0, 57558.0]) / 1000.0
xyz = xyz - ORIGIN

def mask_of(c):
    return np.all(rgb == np.array(c, dtype=np.uint8), axis=1)

rng = np.random.default_rng(3)

# ===== 左墙(u=y, v=z) 作业带 z 0~2.0 =====
LW = 5.06
H_WORK = 2.0
stiffeners = [0.85, 1.69, 2.53]
win = (3.49, 4.10, 0.99, 1.89)   # u0,u1,z0,z1

# 覆盖带: 互不重叠、相接铺满整面墙(0~5.06m, 含窗带);
# 每条覆盖带严格不超过站位点左侧0.20m、右侧0.60m(可小于该范围);
# 相邻站位覆盖带边界 = 两站位中点, 截断至可达范围交集(>=右站位u-0.20, <=左站位u+0.60)
# 墙角规则: 相邻两面墙各取距墙端40cm处墙角站位, 合并为一个(取中心点, 距两墙各40cm, 45°面向墙角);
#           墙角站位仅负责墙角带(0~0.40m), 不按左0.2右0.6计算
# 加强筋定点站位规则: 筋左点优先距筋40cm、筋右点优先距筋20cm; 若与相邻站位太近(<0.4m)则向筋靠拢;
# 筋左点覆盖带止于筋中心线、筋右点覆盖带自筋中心线起(筋中心线恰为公共边界, 筋两侧分别喷涂)
cov = [
    ("C_FL", "corner", 0.00, 0.40, 0.40),    # 前墙角合并站位(45°面向墙角, 仅负责墙角带)
    ("L1", "stiff",  0.40, 0.85, 0.60),    # 筋1左定点(覆盖带止于筋1中心线)
    ("L2", "stiff",  0.85, 1.25, 1.05),    # 筋1右定点(覆盖带自筋1中心线起)
    ("L3", "stiff",  1.25, 1.69, 1.45),    # 筋2左定点
    ("L4", "stiff",  1.69, 2.09, 1.89),    # 筋2右定点
    ("L5", "stiff",  2.09, 2.53, 2.29),    # 筋3左定点
    ("L6", "stiff",  2.53, 3.33, 2.73),    # 筋3右定点(覆盖带恰为完整可达范围)
    ("L7", "normal", 3.33, 3.95, 3.53),    # 窗左补齐(覆盖窗带左半)
    ("L8", "normal", 3.95, 4.66, 4.15),    # 窗右补齐(覆盖窗带右半)
    ("C_BL", "corner", 4.66, 5.06, 4.66),  # 后墙角合并站位(45°面向墙角, 仅负责墙角带)
]
CORNER_W = 0.40   # 墙角责任带宽度(距墙端)

# 梯形分解区域: 切割线 = 筋中心线 + 窗边界
regions = [  # (u0, u1, 名称, 绿色深浅交替)
    (0.00, 0.85, "区域1", 0),
    (0.85, 1.69, "区域2", 1),
    (1.69, 2.53, "区域3", 0),
    (2.53, 3.49, "区域4", 1),
    (3.49, 4.10, "区域5", 1),
    (4.10, 5.06, "区域6", 0),
]
GREEN_A = "#7DCEA0"   # 需喷涂墙面 绿(深)
GREEN_B = "#ABEBC6"   # 需喷涂墙面 绿(浅)
RED_C   = "#E74C3C"   # 墙角责任带 红
BLUE_S  = "#1F6FEB"   # 加强筋 蓝
YELLOW_W = "#F4D03F"  # 窗 黄

fig, ax = plt.subplots(figsize=(13.4, 6.0))

# ===== 1. 需喷涂墙面: 绿色梯形分解区域 (深浅交替) =====
for (u0, u1, nm, gi) in regions:
    c = GREEN_A if gi == 0 else GREEN_B
    ax.add_patch(Rectangle((u0, 0), u1 - u0, H_WORK, fc=c, alpha=0.55, ec="none", zorder=1))
    ax.text((u0 + u1) / 2, 1.80, nm, ha="center", va="center", fontsize=11.5,
            fontweight="bold", color="#145A32", zorder=8,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#145A32", lw=1.0, alpha=0.92))

# 区域5窗带仍为绿色(计入覆盖范围), 窗体黄色遮挡其上(下绘)

# ===== 2. 墙面点云纹理 (弱化) =====
wall = xyz[mask_of((150, 220, 150))]
m = (np.abs(wall[:, 0]) < 0.06) & (wall[:, 2] < 2.2)
p = wall[m]
if len(p) > 60000:
    p = p[rng.choice(len(p), 60000, replace=False)]
ax.scatter(p[:, 1], p[:, 2], s=0.8, c="#1E8449", alpha=0.18, linewidths=0, zorder=2)

obst = xyz[mask_of((0, 0, 0))]
m = (np.abs(obst[:, 0]) < 0.55) & (obst[:, 2] < 2.2) & (obst[:, 1] < LW)
p = obst[m]
if len(p) > 40000:
    p = p[rng.choice(len(p), 40000, replace=False)]
ax.scatter(p[:, 1], p[:, 2], s=1.2, c="#1C2833", alpha=0.25, linewidths=0, zorder=3)

# ===== 3. 墙角责任带: 红色 (仅墙角, 不负责相邻墙面) =====
for u0 in (0.0, 4.66):
    ax.add_patch(Rectangle((u0, 0), CORNER_W, H_WORK, fc=RED_C, ec="#922B21", lw=1.6,
                           alpha=0.45, hatch="///", zorder=5))
ax.text(0.20, 1.62, "前墙角\n责任带(红)\n由C_FL负责\n(45°面向墙角)", ha="center", va="center", fontsize=8.5,
        fontweight="bold", color="#922B21", zorder=7)
ax.text(4.86, 1.62, "后墙角\n责任带(红)\n由C_BL负责\n(45°面向墙角)", ha="center", va="center", fontsize=8.5,
        fontweight="bold", color="#922B21", zorder=7)

# ===== 4. 加强筋: 蓝色竖条 =====
for su in stiffeners:
    ax.add_patch(Rectangle((su - 0.025, 0), 0.05, H_WORK, fc=BLUE_S, ec="#154360",
                           lw=1.0, alpha=0.95, zorder=6))

# ===== 5. 窗: 黄色 (计入覆盖范围, 具体喷涂方式另行考虑) =====
ax.add_patch(Rectangle((win[0], win[2]), win[1] - win[0], win[3] - win[2],
                       fc=YELLOW_W, ec="#B7950B", lw=1.4, alpha=0.9, zorder=6))
ax.text((win[0] + win[1]) / 2, (win[2] + win[3]) / 2, "窗\n(计入覆盖)", ha="center",
        va="center", fontsize=9.5, fontweight="bold", color="#6E2C00", zorder=7)

# ===== 6. 梯形分解竖直切割线 =====
cuts = stiffeners + [win[0], win[1]]
for cu in cuts:
    ax.axvline(cu, color="black", ls="--", lw=2.0, alpha=0.85, zorder=7)
cut_lab = [(0.85, "筋1"), (1.69, "筋2"), (2.53, "筋3"), (3.49, "窗左"), (4.10, "窗右")]
for cu, nm in cut_lab:
    ax.text(cu, 2.06, nm, ha="center", va="bottom", fontsize=8.8, color="black",
            fontweight="bold", zorder=8)

# ===== 7. 垂直喷涂带 (站+蹲) / 倾斜补足带 =====
ax.axhspan(0.33, 1.72, color="#2E86C1", alpha=0.08, zorder=0)
ax.axhline(0.33, color="#2E86C1", ls=":", lw=1.2, alpha=0.8, zorder=4)
ax.axhline(1.72, color="#2E86C1", ls=":", lw=1.2, alpha=0.8, zorder=4)
ax.text(2.60, 1.03, "垂直喷涂带 z:0.33~1.72m（站姿+蹲姿）", fontsize=9, color="#1B4F72",
        zorder=8, ha="center",
        bbox=dict(boxstyle="round", fc="white", ec="#2E86C1", alpha=0.85))
ax.text(2.60, 0.14, "倾斜喷涂补足带(喷枪倾角≤23°)", fontsize=8.5, color="#A04000",
        zorder=8, ha="center",
        bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.7))
ax.text(2.60, 1.90, "倾斜喷涂补足带", fontsize=8.5, color="#A04000",
        zorder=8, ha="center",
        bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.7))

# ===== 8. 站位点覆盖带 (顶部, 互不重叠正好覆盖整面墙含窗带) =====
type_style = {
    "corner": ("#C0392B", "角落站位"),
    "stiff": ("#E67E22", "加强筋定点"),
    "normal": ("#2471A3", "普通站位"),
}
Y_BAR = 2.34
for name, tp, u0, u1, su in cov:
    color, _ = type_style[tp]
    ls = "--" if tp == "corner" else "-"
    alpha = 0.30 if tp == "corner" else 0.55
    ax.add_patch(Rectangle((u0, Y_BAR), u1 - u0, 0.14, fc=color, alpha=alpha, ec=color,
                           lw=1.2, ls=ls, zorder=6))
    ax.plot([su, su], [0, Y_BAR], color=color, ls=":", lw=1.1, alpha=0.7, zorder=6)
    ax.plot(su, Y_BAR + 0.07, marker="v", ms=8, color=color, zorder=7)
    if tp == "corner":
        # 墙角站位: 沿墙指向墙角(3D中45°面向墙角)
        dx = -1 if su < LW / 2 else 1
        ax.annotate("", xy=(su + dx * 0.30, Y_BAR + 0.07), xytext=(su, Y_BAR + 0.07),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.7, mutation_scale=13), zorder=7)
        ax.text(su + dx * 0.30, Y_BAR + 0.22, "45°", ha="center", fontsize=8, color=color,
                fontweight="bold", zorder=7)
    ax.text(su, Y_BAR + 0.40, f"{name}", ha="center", fontsize=9.5,
            fontweight="bold", color=color, zorder=7,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, lw=0.7, alpha=0.92))
    ax.text(su, Y_BAR - 0.07, f"u={su:.2f}", ha="center", va="top", fontsize=7.5,
            color="#555", zorder=7)

# 覆盖带边界坐标标注 (顶部, 各覆盖带左边界=站位u-0.20)
for b in [0.00, 0.40, 0.85, 1.25, 1.69, 2.09, 2.53, 3.33, 3.95, 4.66, 5.06]:
    ax.plot([b, b], [Y_BAR - 0.02, Y_BAR + 0.16], color="#555", lw=0.8, zorder=7)
    ax.text(b, Y_BAR + 0.24, f"{b:.2f}", ha="center", va="bottom", fontsize=6.8,
            color="#555", zorder=7)

ax.set_xlim(-0.08, 5.14)
ax.set_ylim(0, 3.00)
ax.set_xlabel("墙面横向 u = y (m)", fontsize=11)
ax.set_ylabel("墙面高度 z (m)", fontsize=11)
ax.grid(alpha=0.2, ls=":")

from matplotlib.lines import Line2D
handles = [
    Line2D([], [], marker="s", ls="", ms=9, mfc=GREEN_A, mec="#145A32", label="需喷涂墙面·梯形区域(绿)"),
    Line2D([], [], marker="s", ls="", ms=9, mfc=RED_C, alpha=0.55, mec="#922B21", label="墙角责任带(红,仅墙角)"),
    Line2D([], [], marker="s", ls="", ms=9, mfc=BLUE_S, label="加强筋(蓝,需喷涂)"),
    Line2D([], [], marker="s", ls="", ms=9, mfc=YELLOW_W, mec="#B7950B", label="窗(黄,计入覆盖范围)"),
    Line2D([], [], marker="s", ls="", ms=9, mfc="#1C2833", alpha=0.4, label="障碍物投影(黑)"),
    Line2D([], [], color="black", ls="--", lw=2, label="梯形分解竖直切割线"),
    Line2D([], [], marker="s", ls="", ms=9, mfc="#C0392B", alpha=0.6, label="墙角合并站位覆盖带(仅墙角,45°面向墙角)"),
    Line2D([], [], marker="s", ls="--", ms=9, mfc="#C0392B", alpha=0.3, label="C_FL/C_BL(距墙端40cm)仅墙角带"),
    Line2D([], [], marker="s", ls="", ms=9, mfc="#E67E22", alpha=0.6, label="加强筋定点覆盖带(负责墙面)"),
    Line2D([], [], marker="s", ls="", ms=9, mfc="#2471A3", alpha=0.6, label="普通站位覆盖带(负责墙面)"),
]
ax.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.4, framealpha=0.92,
          bbox_to_anchor=(0.5, -0.32))
plt.tight_layout()
plt.savefig(FIG + r"\fig_bcd.png", dpi=200, bbox_inches="tight")
plt.close()
print("fig_bcd done")

# 输出站位表 (供文档使用)
print("\n站位点表 (覆盖带互不重叠相接铺满整面墙, 每条不超过站位左0.2右0.6):")
for name, tp, u0, u1, su in cov:
    lim = f"[{su-0.20:.2f},{su+0.60:.2f}]"
    print(f"  {name}: 类型={tp}, 站位u={su:.3f}, 覆盖u=[{u0:.2f},{u1:.2f}], 宽={u1-u0:.2f}, 可达范围{lim}")
print(f"  墙宽 {LW:.2f} m, 覆盖带相接铺满整面墙无重叠无空隙")
