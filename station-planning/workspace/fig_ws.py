"""图: 单工位工作范围 (腰部扭转+蹲姿对比 / 站+蹲可达区与10°覆盖区)
   采用填充区域+覆盖带矩形, 不使用点云散点, 保证清晰可读"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import ndimage

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

W = r"c:\Users\Yangon\Desktop\舱室绝缘喷涂人形机器人技术方案\喷涂点位划分\work"
FIG = r"c:\Users\Yangon\Desktop\舱室绝缘喷涂人形机器人技术方案\喷涂点位划分\figs"

D_G, D_R = 0.40, 0.80
THETA = 10.0
R_SPRAY = D_G * np.tan(np.deg2rad(THETA / 2))   # 0.035m
COV_L, COV_R = 0.20, 0.60   # 单工位规划覆盖: 左0.2m 右0.6m
Z0, Z1 = 0.33, 1.72         # 垂直喷涂带

def grid_mask(yz, res=0.02):
    xe = np.arange(yz[:, 0].min(), yz[:, 0].max() + res, res)
    ze = np.arange(yz[:, 1].min(), yz[:, 1].max() + res, res)
    H, _, _ = np.histogram2d(yz[:, 0], yz[:, 1], bins=[xe, ze])
    M = H > 0
    M = ndimage.binary_closing(M, np.ones((2, 2)))
    return M, xe, ze

def fill_region(ax, yz, color, alpha=0.32, lw=1.4, res=0.02, zorder=3):
    M, xe, ze = grid_mask(yz, res)
    Xc, Zc = np.meshgrid(xe[:-1], ze[:-1], indexing="ij")
    ax.contourf(Xc, Zc, M, levels=[0.5, 1.5], colors=[color], alpha=alpha, zorder=zorder)
    ax.contour(Xc, Zc, M, levels=[0.5], colors=[color], linewidths=lw, zorder=zorder + 0.5)
    return M

# =============== 图1: 五种工作范围对比 (腰部扭转 + 蹲姿) ===============
sets = [
    ("s5",    "无扭转 [-5°, 5°]",               "#E67E22"),
    ("s30",   "站姿·合理扭转 [-30°, 30°]",       "#E74C3C"),
    ("s30sq", "站+蹲·扭转±30° (骨盆0.45~0.96m)", "#8E44AD"),
    ("s90",   "站姿·大扭转 [-90°, 90°]",         "#2980B9"),
    ("full",  "站姿·URDF全范围 [-150°, 185°]",   "#7F8C8D"),
]
fig, ax = plt.subplots(figsize=(8.6, 6.4))
stats = {}
for key, label, color in sets:
    yz = np.load(W + rf"\ws_{key}.npy")
    M, xe, ze = grid_mask(yz)
    area = M.sum() * 0.02 * 0.02
    stats[key] = (area, yz[:, 0].min(), yz[:, 0].max(), yz[:, 1].min(), yz[:, 1].max())
    fill_region(ax, yz, color, alpha=0.30, lw=1.3)
    ax.text(yz[:, 0].max() + 0.02, (yz[:, 1].min() + yz[:, 1].max()) / 2,
            f"{label}\n面积≈{area:.2f} m²", fontsize=8.4, color=color, va="center",
            fontweight="bold", zorder=6)

# 推荐方案 s30sq 高亮描边
yz = np.load(W + r"\ws_s30sq.npy")
fill_region(ax, yz, "#8E44AD", alpha=0.0, lw=2.6, res=0.02)
ax.text(0.02, 0.05, "推荐：站+蹲·扭转±30°", transform=ax.transAxes, fontsize=10.5,
        color="#8E44AD", fontweight="bold",
        bbox=dict(boxstyle="round", fc="white", ec="#8E44AD", alpha=0.92))

# 规划覆盖带参考 (左0.2 右0.6)
ax.add_patch(Rectangle((-COV_L, Z0), COV_L + COV_R, Z1 - Z0, fill=False,
                       ec="#1A5276", lw=1.8, ls="--", zorder=7))
ax.text(-COV_L + 0.02, Z1 - 0.06, "单工位规划覆盖\n(-0.2m~0.6m)", fontsize=8.5,
        color="#1A5276", va="top", zorder=8)

ax.axvline(0, color="k", ls="--", lw=1.0, alpha=0.7)
ax.text(0.02, 0.06, "机器人中心线 y=0", fontsize=9, rotation=90, va="bottom")
ax.set_xlabel("墙面横向 y (m)  (正=右)", fontsize=11)
ax.set_ylabel("墙面高度 z (m)", fontsize=11)
ax.set_title("右臂腕部垂直墙面可达区（距墙40cm平面）— 腰部扭转与蹲姿对比", fontsize=12)
ax.set_xlim(-0.95, 1.15)
ax.set_ylim(0, 2.1)
ax.grid(alpha=0.3, ls=":")
plt.tight_layout()
plt.savefig(FIG + r"\fig_ws_compare.png", dpi=200)
plt.close()
print("fig_ws_compare done")
for k, v in stats.items():
    print(f"  {k}: area={v[0]:.2f}, y=[{v[1]:.2f},{v[2]:.2f}], z=[{v[3]:.2f},{v[4]:.2f}]")

# =============== 图2: 站+蹲可达区 (分色) + 10°喷涂覆盖区 ===============
yzp = np.load(W + r"\ws_s30sq.npy")     # (y, z, pelvis)
stand = yzp[yzp[:, 2] >= 0.94][:, :2]
squat = yzp[yzp[:, 2] < 0.94][:, :2]

M, xe, ze = grid_mask(yzp[:, :2], 0.02)
M = ndimage.binary_closing(M, np.ones((2, 2)))
area_reach = M.sum() * 0.02 * 0.02
M1, _, _ = grid_mask(yzp[:, :2], 0.01)
M1 = ndimage.binary_closing(M1, np.ones((3, 3)))
Mcov = ndimage.binary_dilation(M1, ndimage.generate_binary_structure(2, 1),
                               iterations=int(np.ceil(R_SPRAY / 0.01)))
area_cov = Mcov.sum() * 1e-4
print(f"reach={area_reach:.2f} m², cov={area_cov:.2f} m²")

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6))

# 左图: 站姿/蹲姿可达区 (填充区域)
ax = axes[0]
fill_region(ax, squat, "#2471A3", alpha=0.35, lw=1.2, res=0.02)
fill_region(ax, stand, "#C0392B", alpha=0.30, lw=1.2, res=0.02)
ax.axvline(0, color="gray", ls="--", lw=1.0, alpha=0.7)
ax.set_xlabel("墙面横向 y (m)  (正=右)", fontsize=11)
ax.set_ylabel("墙面高度 z (m)", fontsize=11)
ax.set_title(f"腕部垂直可达区（腰扭转±30°，站姿+蹲姿并集）\n面积≈{area_reach:.2f} m²，z:0.33~1.72m", fontsize=11)
y0, y1 = yzp[:, 0].min(), yzp[:, 0].max()
z0, z1 = 0.33, 1.72
ax.text(0.03, 0.05, f"宽{y1-y0:.2f}m × 高{z1-z0:.2f}m\n蹲姿使下界由0.82m降至0.33m",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round", fc="white", ec="#C0392B", alpha=0.9))
from matplotlib.lines import Line2D
handles = [
    Line2D([], [], marker="s", ls="", ms=9, mfc="#2471A3", label="蹲姿可达 (骨盆0.45~0.96m)"),
    Line2D([], [], marker="s", ls="", ms=9, mfc="#C0392B", label="站姿可达 (骨盆0.96m)"),
]
ax.legend(handles=handles, fontsize=9, loc="upper right", framealpha=0.92)

# 右图: 可达区 + 单工位规划覆盖带
ax = axes[1]
fill_region(ax, yzp[:, :2], "#00838F", alpha=0.30, lw=1.2, res=0.02)
ax.add_patch(Rectangle((-COV_L, Z0), COV_L + COV_R, Z1 - Z0,
                       fc="#F4D03F", alpha=0.30, ec="#B7950B", lw=1.8, zorder=6))
ax.text(0.0, (Z0 + Z1) / 2, f"单工位规划覆盖\n{COV_L:.1f}m左 ~ {COV_R:.1f}m右\n宽{COV_L+COV_R:.1f}m × 高{Z1-Z0:.2f}m",
        ha="center", va="center", fontsize=9.5, color="#7D6608", fontweight="bold", zorder=7,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#B7950B", lw=1.0, alpha=0.94))
ax.axvline(0, color="gray", ls="--", lw=1.0, alpha=0.7)
ax.set_xlabel("墙面横向 y (m)  (正=右)", fontsize=11)
ax.set_ylabel("墙面高度 z (m)", fontsize=11)
ax.set_title(f"单工位喷涂覆盖区\n(10°扇形, 距墙40cm, 站+蹲, 腰±30°)  面积≈{area_cov:.2f} m²", fontsize=11)
ax.text(0.03, 0.05, f"喷涂半径 R = d_g·tan(θ/2) = {R_SPRAY:.3f} m\n(θ=10°, d_g=0.40m)\n规划覆盖带(-0.2~0.6m)在可达区内",
        transform=ax.transAxes, fontsize=9.5,
        bbox=dict(boxstyle="round", fc="white", ec="#00838F", alpha=0.9))
for ax in axes:
    ax.set_xlim(-0.6, 0.95)
    ax.set_ylim(0, 2.05)
    ax.grid(alpha=0.3, ls=":")
    ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig(FIG + r"\fig_ws_coverage.png", dpi=200)
plt.close()
print("fig_ws_coverage done")
