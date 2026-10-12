# -*- coding: utf-8 -*-
"""生成仓库总览图 assets/figures/overview.png（可复现，无手工编辑）。

两条线：G1 人形机器人喷涂线（上，蓝）与 OpenArm 七轴机械臂控制线（下，橙）。
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- 中文字体（显式注册，matplotlib 默认扫描不含 opentype 目录） ----
CAND = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
]
for p in CAND:
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
for f in font_manager.findSystemFonts():
    name = os.path.basename(f).lower()
    if any(k in name for k in ("notosanscjk", "notoserifcjk", "wenquanyi", "wqy")):
        font_manager.fontManager.addfont(f)
zh = None
for f in font_manager.fontManager.ttflist:
    if "Noto Sans CJK" in f.name:
        zh = f.name          # ttc 多面孔，SC 缺失时 JP 面孔同样覆盖全部汉字
        if "SC" in f.name:
            break
if zh is None:
    raise SystemExit("未找到中文字体，请检查系统字体安装")
plt.rcParams["font.sans-serif"] = [zh, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BLUE, ORANGE, GRAY = "#1d4ed8", "#c2410c", "#6b7280"

fig, ax = plt.subplots(figsize=(16.5, 7.2), dpi=150)
ax.set_xlim(0, 100)
ax.set_ylim(0, 37)
ax.axis("off")

def box(x, y, w, h, title, sub, color, fs=12.5):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.35,rounding_size=0.9",
        fc="white", ec=color, lw=2.0))
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
            fontsize=fs, color="#111827", fontweight="bold")
    ax.text(x + w / 2, y + h * 0.24, sub, ha="center", va="center",
            fontsize=8.8, color=GRAY)

def arrow(x0, x1, y, color=GRAY):
    ax.add_patch(FancyArrowPatch(
        (x0, y), (x1, y), arrowstyle="-|>", mutation_scale=16,
        lw=1.8, color=color))

def lane_label(x, y, text, color):
    ax.text(x, y, text, fontsize=13, color=color, fontweight="bold",
            ha="left", va="center")

# ================= 上层：G1 人形机器人喷涂线 =================
yA = 27
lane_label(1.5, yA + 8.0, "人形机器人喷涂线 · 宇树 G1（点云 → 规划 → 全身控制）", BLUE)
boxesA = [
    ("舱室语义点云", "500 万点 · 6 类标签"),
    ("特征匹配配准", "墙角/门框地标 + 自适应 ICP\n对齐 RMSE 0.34 mm"),
    ("工位 / 站位点规划", "BCD 区域划分\n10×2 m 墙 → 12 站位点"),
    ("之字形轨迹 + 逐点 IK", "等宽 307 点\n307/307 收敛 · 夹角≥60°"),
    ("网页工作站", "单文件前后端\nthree.js 3D 预览"),
    ("GR00T 全身控制", "SONIC 平衡 + 上肢 50Hz\n仿真/实机一键切换"),
]
wA, pitchA = 14.3, 16.5            # 6 框：1.5 + 5×16.5 + 14.3 + 0.35 ≈ 98.2
for i, (t, s) in enumerate(boxesA):
    xA = 1.5 + pitchA * i
    box(xA, yA, wA, 6.2, t, s, BLUE)
    if i < len(boxesA) - 1:
        arrow(xA + wA + 0.45, xA + wA + 1.95, yA + 3.1, BLUE)

# ================= 下层：OpenArm 七轴机械臂控制线 =================
yB = 8
lane_label(1.5, yB + 8.0, "机械臂走线控制线 · OpenArm 七轴（标定 → 仿真 → 真机）", ORANGE)
boxesB = [
    ("零点标定", "纯 Python CAN-FD\nvcan0 无硬件可跑"),
    ("仿真", "MoveIt2 + 自研数值 IK\n1237 点稠密轨迹"),
    ("真机执行", "MPC 插件 / RL 策略\nIsaac Lab 2048 env"),
    ("ILC 前馈修正", "真机数据离线学习\n积分器峰值 ↓2~5×"),
    ("全程验收", "143.6 s 零急停\n重复性 RMS 0.04°"),
]
wB, pitchB = 17.6, 19.8            # 5 框：1.5 + 4×19.8 + 17.6 + 0.35 ≈ 98.1
for i, (t, s) in enumerate(boxesB):
    xB = 1.5 + pitchB * i
    box(xB, yB, wB, 6.2, t, s, ORANGE)
    if i < len(boxesB) - 1:
        arrow(xB + wB + 0.45, xB + wB + 1.95, yB + 3.1, ORANGE)

ax.text(50, 2.2, "同一类壁面作业任务的两条路线：人形机器人负责舱室大范围喷涂，机械臂负责高频走线的“快与稳”",
        ha="center", fontsize=10.5, color=GRAY, style="italic")

out = os.path.join(HERE, "figures", "overview.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.tight_layout()
plt.savefig(out, bbox_inches="tight", facecolor="white")
print("写出", out)
