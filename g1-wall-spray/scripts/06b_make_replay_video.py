# -*- coding: utf-8 -*-
"""步骤6b: 把 06 渲染的回放帧合成为 mp4 + 九宫格总览图.
用法: conda isaaclab 环境(需要 imageio/cv2), 在 06_replay_render.py 之后运行.
输出: output/ufo_motion/<ver>/replay_motion.mp4, fig_replay_contactsheet.png
"""
import numpy as np
import os, glob, re
import imageio
from PIL import Image, ImageDraw

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ver_key(name):
    # 版本目录自然排序键: 按 v 后数字排(v2<v5<v10), 字母序会把 v10 排在 v5 前
    m = re.match(r"v(\d+)", name)
    return int(m.group(1)) if m else -1


_vdirs = [os.path.basename(d) for d in glob.glob(os.path.join(BASE, "output/ufo_motion/v*"))]
VER = os.environ.get("MOTION_VER") or (max(_vdirs, key=lambda d: (_ver_key(d), d))
                                       if _vdirs else "v1")
FR = os.path.join(BASE, "output", "ufo_motion", VER, "replay_frames")
OUT = os.path.join(BASE, "output", "ufo_motion", VER)

side = sorted(glob.glob(os.path.join(FR, "side_*.png")))
front = sorted(glob.glob(os.path.join(FR, "front_*.png")))
assert side, f"无回放帧, 先运行 06_replay_render.py: {FR}"
print(f"[帧] side {len(side)}, front {len(front)}")

# ---- mp4: 侧视(喷涂指向关系最清楚) 15fps ----
frames = []
for p in side:
    im = Image.open(p)
    im = im.resize((im.width // 2, im.height // 2), Image.LANCZOS)
    frames.append(np.asarray(im))
mp4 = os.path.join(OUT, "replay_motion.mp4")
imageio.mimwrite(mp4, frames, fps=15, quality=8, macro_block_size=8)
print(f"[写出] replay_motion.mp4 {len(frames)} 帧 @15fps")

# ---- 九宫格总览: 均匀抽 24 帧, 6x4, side+front 上下并排 ----
idx = np.linspace(0, len(side) - 1, 12).astype(int)
sw, sh = Image.open(side[0]).size
fw, fh = Image.open(front[0]).size
tw, th = sw // 3, sh // 3
tf_w, tf_h = fw // 3, fh // 3
cols, rows = 4, 3
sheet = Image.new("RGB", (cols * tw + (cols + 1) * 6, rows * (th + tf_h + 26) + 6), "white")
d = ImageDraw.Draw(sheet)
for k, fi in enumerate(idx):
    r, c = divmod(k, cols)
    x = 6 + c * (tw + 6)
    y = 6 + r * (th + tf_h + 26)
    im_s = Image.open(side[fi]).resize((tw, th), Image.LANCZOS)
    im_f = Image.open(front[fi]).resize((tf_w, tf_h), Image.LANCZOS)
    sheet.paste(im_s, (x, y + 18))
    sheet.paste(im_f, (x + (tw - tf_w) // 2, y + 18 + th + 4))
    d.text((x + 4, y + 3), f"t={fi * 2 / 30:.1f}s", fill="black")
cs = os.path.join(OUT, "fig_replay_contactsheet.png")
sheet.save(cs)
print(f"[写出] fig_replay_contactsheet.png ({cols}x{rows}, 12 时刻)")
