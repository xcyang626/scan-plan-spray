# -*- coding: utf-8 -*-
"""边走边喷实验效果分析: 骨盆路径/朝向漂移/枪尖世界轨迹/墙面覆盖.

用法: ~/miniconda3/envs/isaaclab/bin/python analyze_walk_spray.py [log.npz] \
          [--meta <trajectory_meta.json>]
墙距/喷涂带/行高默认从 trajectory_meta.json 读(与 03 规划同源, 站位标称
d_robot_nominal / uniform_stroke_u / row_heights), 读不到才回退到脚本内置
默认值(控制台会标注"未读到 meta，使用默认")。meta 来源优先级:
  --meta 参数 > 环境变量 G1W_META > G1W_OUT/G1W_RUN_DIR 目录 > 日志目录上溯。
输出: walk_spray/fig_walk_spray_analysis.png + 控制台统计
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts", "exec"))
from arm_task_servo import fk_jac1, quat2rotm   # noqa: E402

# 内置兜底默认值(某一次规划的硬编码值; 只在读不到 meta 时使用)
WALL_X_DEFAULT = 0.74
BAND_DEFAULT = (0.107, 0.617)      # 规划喷涂带(静止站立时的墙面 u 范围)
ROWS_DEFAULT = (0.75, 1.45)        # 站姿行高度范围
GUN_LEN_DEFAULT = 0.12

_ap = argparse.ArgumentParser(description="边走边喷日志分析")
_ap.add_argument("log", nargs="?",
                 default=os.path.join(HERE, "log", "walk_spray.npz"),
                 help="跟踪日志 npz(默认 walk_spray/log/walk_spray.npz)")
_ap.add_argument("--meta", default=None,
                 help="trajectory_meta.json 路径(墙距/喷涂带/行高来源)")
_a = _ap.parse_args()
LOG = _a.log


def _find_meta(explicit=None, log_path=None):
    """定位 trajectory_meta.json: --meta > G1W_META > G1W_OUT/G1W_RUN_DIR 目录
    > 日志所在目录向上找. 找不到返回 None."""
    cands = []
    if explicit:
        cands.append(explicit)
    env = os.environ.get("G1W_META")
    if env:
        cands.append(env)
    for k in ("G1W_OUT", "G1W_RUN_DIR"):
        d = os.environ.get(k)
        if d:
            cands.append(os.path.join(d, "trajectory_meta.json"))
    for c in cands:
        if os.path.isfile(c):
            return c
    d = os.path.dirname(os.path.abspath(log_path or HERE))
    for _ in range(4):                       # 日志一般在 <run>/walk_spray/log/ 下
        c = os.path.join(d, "trajectory_meta.json")
        if os.path.isfile(c):
            return c
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return None


def _load_meta(path):
    """读 meta; 失败返回 (None, 原因)."""
    if not path:
        return None, "未读到 meta，使用默认"
    try:
        with open(path, encoding="utf8") as f:
            return json.load(f), path
    except Exception as e:
        return None, f"meta 读取失败({e})，使用默认"


def _band_pair(meta, keys):
    """从 meta 取 [u0,u1] 喷涂带; 取不到返回 None."""
    for k in keys:
        v = (meta or {}).get(k)
        try:
            if not v or len(v) < 2:
                continue
            return float(v[0]), float(v[1])
        except (TypeError, IndexError, KeyError, ValueError):
            continue
    return None


def _row_range(meta):
    """从 meta 取行高度范围(优先站姿区; 没有就取全部行的 min/max)."""
    rp = (meta or {}).get("row_params")
    if rp:
        stand = [float(r["h"]) for r in rp
                 if isinstance(r, dict) and r.get("zone", "stand") == "stand"
                 and r.get("h") is not None]
        if stand:
            return min(stand), max(stand)
    rh = (meta or {}).get("row_heights")
    if rh:
        try:
            hs = [float(v) for v in rh]
            if hs:
                return min(hs), max(hs)
        except (TypeError, ValueError):
            pass
    return None


_meta, _meta_src = _load_meta(_find_meta(_a.meta, LOG))
_default_note = "未读到 meta，使用默认"

# 墙距: 环境变量 G1W_D_ROBOT > meta.d_robot_nominal > 默认 0.74
_env_d = os.environ.get("G1W_D_ROBOT")
WALL_X = WALL_X_DEFAULT
_wall_src = _default_note
if _env_d:
    try:
        WALL_X = float(_env_d)
        _wall_src = f"环境变量 G1W_D_ROBOT={_env_d}"
    except ValueError:
        _wall_src = f"G1W_D_ROBOT={_env_d!r} 非法，{_default_note}"
elif _meta and _meta.get("d_robot_nominal") is not None:
    try:
        WALL_X = float(_meta["d_robot_nominal"])
        _wall_src = "meta.d_robot_nominal"
    except (TypeError, ValueError):
        _wall_src = _default_note
BAND = _band_pair(_meta, ("uniform_stroke_u", "stroke_u")) or BAND_DEFAULT
_band_src = ("meta.uniform_stroke_u" if _meta and _meta.get("uniform_stroke_u")
             else ("meta.stroke_u" if _meta and _meta.get("stroke_u")
                   else _default_note))
_rows = _row_range(_meta)
ROWS = _rows or ROWS_DEFAULT
_rows_src = "meta.row_params/row_heights" if _rows else _default_note
GUN_LEN = float((_meta or {}).get("gun_len") or GUN_LEN_DEFAULT)

if not os.path.exists(LOG):
    print(f"[错误] 日志文件不存在: {LOG}")
    sys.exit(1)
z = np.load(LOG)
for _k in ("t", "qpos"):
    if _k not in getattr(z, "files", []):
        print(f"[错误] 日志 {LOG} 缺 '{_k}' 数组 —— 不是跟踪日志或已损坏")
        sys.exit(1)
t, qp = z["t"], z["qpos"]
if len(t) == 0 or len(qp) == 0:
    print(f"[错误] 日志 {LOG} 没有有效帧(0 帧) —— 机器人可能摔倒/未播完, "
          f"请重跑 walk_spray_runner.py 后再分析")
    sys.exit(1)
if len(t) != len(qp):
    print(f"[警告] 日志帧数不一致(t {len(t)} / qpos {len(qp)}), 按最小长度截断")
    n = min(len(t), len(qp))
    t, qp = t[:n], qp[:n]
T = float(t[-1] - t[0])
if not np.isfinite(T) or T <= 0:
    print(f"[错误] 日志时间戳退化(T={T:.6f}s, 共 {len(t)} 帧) —— 无法计算速度统计, "
          f"请检查日志完整性")
    sys.exit(1)
print(f"日志 {len(t)} 帧, t [{t[0]:.1f}, {t[-1]:.1f}]s")
print(f"[参数] 墙距 {WALL_X:.3f}m ({_wall_src}) | 喷涂带 u "
      f"[{BAND[0]:.3f}, {BAND[1]:.3f}] ({_band_src}) | 行高 "
      f"[{ROWS[0]:.2f}, {ROWS[1]:.2f}] ({_rows_src}) | 枪长 {GUN_LEN:.3f}m"
      f" | meta: {_meta_src}")

def yaw_of(q):                  # (w,x,y,z)
    return np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]),
                      1 - 2 * (q[2] ** 2 + q[3] ** 2))

yaw = np.degrees(np.array([yaw_of(qp[k, 3:7]) for k in range(len(qp))]))
# 枪尖 + 枪轴: 每帧只做一次 FK(腕心位置 p 与枪轴 a 同一个 FK 结果)
tip = np.zeros((len(qp), 3))
axs = np.zeros((len(qp), 3))
for k in range(len(qp)):
    q10 = np.concatenate([qp[k, 19:22], qp[k, 29:36]])
    p_rel, a_rel, _, _ = fk_jac1(q10)
    Rb = quat2rotm(qp[k, 3:7])
    axs[k] = Rb @ a_rel
    tip[k] = Rb @ p_rel + qp[k, 0:3] + GUN_LEN * axs[k]

# 朝向展开(-180..180 -> 连续)
yaw_c = np.unwrap(np.radians(yaw))
yaw_deg = np.degrees(yaw_c)

# 墙面覆盖: 枪轴朝 +x 且离墙不太远才算有效喷射
toward = axs[:, 0] > 0.3
wall_y, wall_z, wall_t = [], [], []
for k in range(len(qp)):
    if not toward[k] or not np.isfinite(tip[k]).all():
        continue
    dx = WALL_X - tip[k, 0]
    if dx <= 0 or dx > 1.5:      # 枪在墙后或离墙太远
        continue
    s = dx / axs[k, 0]
    wall_y.append(tip[k, 1] + s * axs[k, 1])
    wall_z.append(tip[k, 2] + s * axs[k, 2])
    wall_t.append(t[k])
wall_y = np.array(wall_y); wall_z = np.array(wall_z); wall_t = np.array(wall_t)

# ---- 统计 ----
dy = qp[-1, 1] - qp[0, 1]
print(f"骨盆: 起点 ({qp[0,0]:.2f},{qp[0,1]:.2f}) 终点 ({qp[-1,0]:.2f},{qp[-1,1]:.2f})")
print(f"位移: dx={qp[-1,0]-qp[0,0]:+.2f}m dy={dy:+.2f}m |v|={(np.hypot(qp[-1,0]-qp[0,0], dy)/T):.3f}m/s "
      f"(指令 0.1 前向x/左向y)")
print(f"yaw: 起 {yaw_deg[0]:+.1f}° 终 {yaw_deg[-1]:+.1f}° 最大|漂移| {np.abs(yaw_deg-yaw_deg[0]).max():.1f}°")
print(f"骨盆 z: min {qp[:,2].min():.3f} mean {qp[:,2].mean():.3f} (站立≈0.77, <0.35 视为摔倒)")
print(f"墙面有效喷射点 {len(wall_y)}/{len(qp)}; 墙面 y 覆盖 [{wall_y.min() if len(wall_y) else 0:.2f},"
      f"{wall_y.max() if len(wall_y) else 0:.2f}] (静止时应对齐带 {BAND})")

# ---- 图 ----
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
a1 = axes[0, 0]
a1.plot(qp[:, 1], qp[:, 0], "-", lw=1.2, label="骨盆")
a1.plot(tip[:, 1], tip[:, 0], "-", lw=0.8, alpha=0.8, label="枪尖")
a1.plot(qp[0, 1], qp[0, 0], "g^", ms=9, label="起点")
a1.plot(qp[-1, 1], qp[-1, 0], "rv", ms=9, label="终点")
a1.axvline(0, color="k", lw=0.5)
a1.set_xlabel("y 左向 (m)"); a1.set_ylabel("x 前向 (m)")
a1.set_title(f"俯视轨迹 (指令: 向左 0.1 m/s)"); a1.legend(); a1.axis("equal"); a1.grid(alpha=0.3)

a2 = axes[0, 1]
a2.plot(t, yaw_deg, lw=1.2)
a2.set_xlabel("t (s)"); a2.set_ylabel("yaw (°, 连续)")
a2.set_title(f"基座朝向漂移 (终 {yaw_deg[-1]:+.0f}°)"); a2.grid(alpha=0.3)

a3 = axes[1, 0]
sc = a3.scatter(wall_y, wall_z, c=wall_t, s=2, cmap="viridis")
a3.add_patch(plt.Rectangle((BAND[0], ROWS[0]), BAND[1]-BAND[0], ROWS[1]-ROWS[0],
                           fill=False, ec="r", ls="--", lw=1.5,
                           label="静止喷涂规划带"))
a3.set_xlabel("墙面 y (m)"); a3.set_ylabel("墙面高度 z (m)")
a3.set_title(f"枪尖墙面落点 (n={len(wall_y)}, 颜色=时间)")
a3.legend(); a3.grid(alpha=0.3)
plt.colorbar(sc, ax=a3, label="t (s)")

a4 = axes[1, 1]
a4.plot(t, qp[:, 2], lw=1.0)
a4.set_xlabel("t (s)"); a4.set_ylabel("骨盆 z (m)")
a4.set_title(f"骨盆高度 (min {qp[:,2].min():.2f})"); a4.grid(alpha=0.3)

fig.suptitle("边走边喷实验: 站立喷涂动作 + 左移 0.1 m/s (无世界系修正)", fontsize=13)
fig.tight_layout()
tag = os.path.splitext(os.path.basename(LOG))[0]
out = os.path.join(HERE, f"fig_{tag}_analysis.png")
fig.savefig(out, dpi=110)
print("图已存", out)
