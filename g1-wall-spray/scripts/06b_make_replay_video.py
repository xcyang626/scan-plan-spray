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


_OUT06 = os.environ.get("G1W_OUT", os.path.join(BASE, "output"))
_vdirs = [os.path.basename(d) for d in glob.glob(os.path.join(_OUT06, "ufo_motion/v*"))]
VER = os.environ.get("MOTION_VER") or (max(_vdirs, key=lambda d: (_ver_key(d), d))
                                       if _vdirs else "v1")
FR = os.path.join(_OUT06, "ufo_motion", VER, "replay_frames")
OUT = os.path.join(_OUT06, "ufo_motion", VER)

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

# ---- 总览条: 均匀抽 12 帧, 4x3, side|front 左右并排(横幅单元格) ----
# 注: 单元格内 side/front 若上下叠会把整图拼成竖版(882x1044), 与 README 里
# 横版场景图并排时高矮悬殊留大片空白 —— 左右并排后整图约 3:1 横幅条
idx = np.linspace(0, len(side) - 1, 12).astype(int)
sw, sh = Image.open(side[0]).size
fw, fh = Image.open(front[0]).size
tw, th = sw // 3, sh // 3
tf_w, tf_h = fw // 3, fh // 3
cols, rows = 4, 3
cw = tw + tf_w + 4                       # 单元格宽 = side|front 横排
ch = max(th, tf_h) + 26
sheet = Image.new("RGB", (cols * cw + (cols + 1) * 6, rows * ch + 6), "white")
d = ImageDraw.Draw(sheet)
for k, fi in enumerate(idx):
    r, c = divmod(k, cols)
    x = 6 + c * (cw + 6)
    y = 6 + r * ch
    im_s = Image.open(side[fi]).resize((tw, th), Image.LANCZOS)
    im_f = Image.open(front[fi]).resize((tf_w, tf_h), Image.LANCZOS)
    sheet.paste(im_s, (x, y + 20))
    sheet.paste(im_f, (x + tw + 4, y + 20))
    d.text((x + 4, y + 4), f"t={fi * 2 / 30:.1f}s", fill="black")
cs = os.path.join(OUT, "fig_replay_contactsheet.png")
sheet.save(cs)
print(f"[写出] fig_replay_contactsheet.png ({cols}x{rows}, 12 时刻, 横幅条)")
