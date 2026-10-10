# -*- coding: utf-8 -*-
"""步骤5b: 参考动作时间序列总览图"""
import numpy as np, os, csv, re
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from common import setup_cn_font
setup_cn_font()
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import glob as _glob
_outbase = os.environ.get("G1W_OUT", os.path.join(BASE, "output"))


def _ver_key(p):
    # 版本目录自然排序键: 按 v 后数字排(v2<v5<v10), 字母序会把 v10 排在 v5 前
    # (同号多 run 目录再按目录名字母序, 口径与 05_export_ufo_motion.py 一致)
    d = os.path.basename(os.path.dirname(p))
    m = re.match(r"v(\d+)", d)
    return (int(m.group(1)) if m else -1, d)


_csvs = sorted(_glob.glob(os.path.join(_outbase, "ufo_motion", "v*",
                                       "wall_spray_motion_qpos.csv")), key=_ver_key)
if not _csvs:
    raise SystemExit("[错误] 未找到任何 wall_spray_motion_qpos.csv "
                     "(先运行 05_export_ufo_motion.py, 或检查 G1W_OUT)")
rows = list(csv.reader(open(_csvs[-1])))
hdr = rows[0]
COL = {n: i for i, n in enumerate(hdr)}
def col(name):
    out = []
    for r in rows[1:]:
        try:
            out.append(float(r[COL[name]]))
        except ValueError:
            out.append(np.nan)
    return np.array(out)
def colstr(name):
    return [r[COL[name]] for r in rows[1:]]
t = col("t_s"); phase = colstr("phase"); seg = colstr("seg_class")

fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
ax = axes[0]
ax.plot(t, col("root_tz"), "k-", lw=1.6, label="root_tz 骨盆高度")
ax.plot(t, col("root_tz") + col("wrist_z"), "b-", lw=1.2, label="腕心高度(离地)")
ax.axhline(0.15, color="r", ls=":", lw=1)
ax.text(1, 0.17, "安全下限0.15m", color="r", fontsize=8)
ax.set_ylabel("高度 (m)"); ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)
ax.set_title("G1 墙面喷涂参考动作时间序列 (30Hz, qpos 36维, 匀速0.05m/s) — 保持→过渡→跟踪→回零位→蹲姿跟踪→回位")

ax = axes[1]
for jn, c in [("right_shoulder_pitch", "#22aa44"), ("right_elbow", "#dd8800"),
              ("right_shoulder_yaw", "#8855cc"), ("waist_yaw", "#2288aa"),
              ("right_knee", "#c46a1f")]:
    ax.plot(t, col(jn), lw=1.1, color=c, label=jn)
ax.set_ylabel("关节角 (rad)"); ax.legend(loc="upper left", fontsize=8, ncol=3); ax.grid(alpha=0.3)

ax = axes[2]
# 腕心速度(NaN 段即非跟踪帧: 置 0, 防止 NaN 沿 diff 传播拉断曲线)
wz = col("wrist_z"); wy = col("wrist_y")
ok = ~np.isnan(wz)
spd = np.zeros_like(t)
dok = ok[1:] & ok[:-1]
_dwy, _dwz = np.diff(wy), np.diff(wz)
spd[1:] = np.where(dok, np.hypot(np.nan_to_num(_dwy), np.nan_to_num(_dwz)), 0.0) * 30
ax.plot(t, spd, "-", color="#555", lw=0.8)
for c, col in [("green", "#22aa44"), ("red", "#dd2222"), ("yellow", "#eebb00")]:
    m = [i for i, s in enumerate(seg) if s == c]
    if m:
        ax.plot([t[i] for i in m], [spd[i] for i in m], ".", color=col, ms=2,
                label={"green": "喷涂0.05m/s", "red": "穿障碍0.05m/s",
                       "yellow": "换行0.05m/s"}[c])
ax.set_ylabel("腕心速度 (m/s)"); ax.set_xlabel("t (s)"); ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3); ax.set_ylim(0, 0.45)
plt.tight_layout()
plt.savefig(os.path.join(_outbase, "ufo_motion", "fig_motion_timeseries.png"), dpi=120)
print("[预览] fig_motion_timeseries.png")
