"""图: 喷涂工位规划算法 总体设计框图 (分层架构, 参考图1样式)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

FIG = r"c:\Users\Yangon\Desktop\舱室绝缘喷涂人形机器人技术方案\喷涂点位划分\figs"

fig, ax = plt.subplots(figsize=(12.4, 8.6))
ax.set_xlim(0, 12.4)
ax.set_ylim(0, 8.6)
ax.axis("off")

LX0, LX1 = 1.75, 12.05   # 层背景范围
MX = 1.75                # 层名区右边界

def layer(y0, y1, color, name):
    ax.add_patch(FancyBboxPatch((LX0, y0), LX1 - LX0, y1 - y0,
                 boxstyle="round,pad=0.04", fc=color, ec="#78909C", lw=1.1, alpha=0.55, zorder=1))
    ax.text(0.82, (y0 + y1) / 2, name, ha="center", va="center", fontsize=12.5,
            fontweight="bold", color="#37474F", rotation=90, zorder=3)

def module(x, y, w, h, lines, fs=9.2, fc="white", ec="#546E7A", bold=False, dashed_group=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                 fc=fc, ec=ec, lw=1.3, zorder=3))
    ax.text(x + w / 2, y + h / 2, "\n".join(lines), ha="center", va="center",
            fontsize=fs, zorder=4, fontweight="bold" if bold else "normal", color="#212121",
            linespacing=1.35)
    return (x, y, w, h)

def arrow(x0, y0, x1, y1, label=None, dx=0.12, color="#455A64", lw=1.8, fs=8.6, ha="center"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                 arrowstyle="-|>", mutation_scale=15, color=color, lw=lw, zorder=5))
    if label:
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2, label, fontsize=fs, color=color,
                ha=ha, va="center", zorder=6, linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))

def harrow(x0, x1, y, color="#455A64"):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                 mutation_scale=11, color=color, lw=1.5, zorder=5))

# ==================== 层1: 数据输入层 ====================
y1a, y1b = 7.42, 8.28
layer(y1a, y1b, "#CFD8DC", "数 据 输 入 层")
m11 = module(2.15, 7.56, 2.95, 0.58, ["舱室三维语义点云地图", "(sensor_msgs/PointCloud2)"])
m12 = module(5.55, 7.56, 2.55, 0.58, ["天宫3 URDF", "运动学模型"])
m13 = module(8.55, 7.56, 3.15, 0.58, ["喷涂工艺参数", "θ=10°, d_g=0.40m, d_r=0.80m, H=0~2m"])

# ==================== 层2: 感知处理层 ====================
y2a, y2b = 5.98, 6.98
layer(y2a, y2b, "#FFE0B2", "感 知 处 理 层")
m21 = module(2.15, 6.20, 2.85, 0.58, ["语义特征提取", "(墙面/加强筋/门/窗/障碍物)"])
m22 = module(5.55, 6.20, 2.85, 0.58, ["墙面分离与", "平面参数化(法向/边界)"])
m23 = module(8.85, 6.20, 2.75, 0.58, ["三维→二维投影", "(u横向, v高度)平面"])
harrow(5.0, 5.55, 6.49); harrow(8.40, 8.85, 6.49)

# ==================== 层3: 工作空间分析层 ====================
y3a, y3b = 4.44, 5.54
layer(y3a, y3b, "#C8E6C9", "工 作 空 间 分 析 层")
m31 = module(2.15, 4.66, 1.95, 0.58, ["URDF正运动学", "(腰部3+手臂4关节)"])
m32 = module(4.45, 4.66, 1.95, 0.58, ["蒙特卡洛采样", "(1500万随机样本)"])
m33 = module(6.75, 4.66, 2.05, 0.58, ["腕部垂直可达区", "(距墙d_g平面, α-shape)"])
m34 = module(9.10, 4.66, 2.10, 0.58, ["10°扇形喷涂覆盖区", "R=d_g·tan(θ/2)=3.5cm"])
harrow(4.10, 4.45, 4.95); harrow(6.40, 6.75, 4.95); harrow(8.80, 9.10, 4.95)
module(9.10, 4.44, 2.10, 0.0, [""])  # spacer
m35 = module(2.15, 4.44, 3.30, 0.0, [""])  # spacer

# ==================== 层4: 工位规划层 ====================
y4a, y4b = 2.90, 4.10
layer(y4a, y4b, "#BBDEFB", "工 位 规 划 层")
m41 = module(2.15, 3.36, 2.35, 0.58, ["BCD区域划分", "(切割线:筋中心/门窗边界)"])
m42 = module(4.85, 3.36, 2.35, 0.58, ["特征定点规划", "(墙角40cm·筋左40/右20cm)"])
m43 = module(7.55, 3.36, 2.05, 0.58, ["子区域划分", "(按W_cov=0.9m分列)"])
m44 = module(9.85, 3.36, 1.85, 0.58, ["站位点生成", "与避障校验"])
harrow(4.50, 4.85, 3.65); harrow(7.20, 7.55, 3.65); harrow(9.60, 9.85, 3.65)

# ==================== 层5: 输出接口层 ====================
y5a, y5b = 1.42, 2.42
layer(y5a, y5b, "#E1BEE7", "输 出 接 口 层")
m51 = module(2.15, 1.64, 2.95, 0.58, ["/spray_planning/workstations", "(geometry_msgs/PoseArray) 工位点位"])
m52 = module(5.55, 1.64, 2.95, 0.58, ["/spray_planning/regions", "(visualization_msgs/MarkerArray) 区域结果"])
m53 = module(8.95, 1.64, 2.85, 0.58, ["/spray_planning/status", "(std_msgs/String) 规划状态"])

# ==================== 层间箭头 ====================
arrow(3.6, y1a, 3.6, y2b, "语义点云\n(RGB语义标注)")
arrow(6.8, y2a, 6.8, y3b + 0.06, "墙面2D投影(u,v)\n筋/窗/门位置")
arrow(7.9, y3a, 7.9, y4b + 0.06, "单工位覆盖参数\n(W_cov=0.9m, 左0.25/右0.65m)")
arrow(3.6, y4a, 3.6, y5b + 0.06, "站位点序列\n(坐标/朝向/类型/负责区域)")
# 层1 -> 层3 右侧绕行 (URDF + 工艺参数)
ax.add_patch(FancyArrowPatch((10.1, y1a), (11.6, y1a - 0.02), arrowstyle="-",
             color="#8D6E63", lw=1.6, zorder=5))
ax.plot([11.6, 11.6], [y1a, 4.95], color="#8D6E63", lw=1.6, zorder=5)
ax.add_patch(FancyArrowPatch((11.6, 4.95), (11.28, 4.95), arrowstyle="-|>",
             mutation_scale=14, color="#8D6E63", lw=1.6, zorder=5))
ax.text(11.72, 6.3, "URDF模型\n工艺参数\n(θ, d_g, d_r, H)", fontsize=8.6, color="#8D6E63",
        ha="center", va="center", rotation=90, zorder=6,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))
# 层1 -> 层4 筋定点规则
ax.plot([1.45, 1.45], [7.7, 3.65], color="#8D6E63", lw=1.6, ls="--", zorder=2)
ax.add_patch(FancyArrowPatch((1.45, 3.65), (1.85, 3.65), arrowstyle="-|>",
             mutation_scale=14, color="#8D6E63", lw=1.6, zorder=5))
ax.text(1.18, 5.8, "定点规则\n(墙角/筋左40/右20cm)", fontsize=8.2, color="#8D6E63",
        ha="center", va="center", rotation=90, zorder=6,
        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.85))

# 输出 -> 下游
ax.add_patch(FancyArrowPatch((10.5, 2.06), (12.0, 2.06), arrowstyle="-|>",
             mutation_scale=15, color="#455A64", lw=1.8, zorder=5))
ax.text(12.08, 2.06, "下游:\n路径规划\n喷涂轨迹\n规划", fontsize=8.8, ha="center", va="center",
        color="#37474F", zorder=6, linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.2", fc="#ECEFF1", ec="#90A4AE", lw=1.0))

ax.set_title("喷涂工位规划算法总体设计框图", fontsize=15, fontweight="bold", pad=14)
plt.tight_layout()
plt.savefig(FIG + r"\fig_architecture.png", dpi=200, bbox_inches="tight")
plt.close()
print("fig_architecture done")
