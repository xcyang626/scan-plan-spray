# -*- coding: utf-8 -*-
"""步骤6: 在 MuJoCo(g1-mode_13_15.xml) 中回放参考动作, 渲染帧序列.
用法: conda mujoco 环境, MUJOCO_GL=glfw DISPLAY=:0
输出: output/ufo_motion/replay_frames/*.png (每2帧1张, 15fps)
"""
import numpy as np
import mujoco
import os, glob, re

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE06 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VER_DIR = os.path.join(os.environ.get("G1W_OUT", os.path.join(BASE06, "output")),
                        "ufo_motion", os.environ.get("MOTION_VER") or "")
try:
    import json as _json
    _REG = _json.load(open(os.path.join(BASE06, "station", "robot_registry.json"),
                           encoding="utf8"))
    _ROBOT = "g1"
    _meta_p = os.path.join(_VER_DIR, "motion_meta.json")
    if os.path.exists(_meta_p):
        _ROBOT = _json.load(open(_meta_p, encoding="utf8")).get("robot", "g1")
    _m = _REG[_ROBOT].get("mjcf") or _REG["g1"]["mjcf"]
    _m = os.path.expanduser(_m)        # registry 允许 ~/.cache/... 写法
    MJCF = _m if os.path.isabs(_m) else os.path.join(BASE06, _m)
except Exception:
    _ROBOT = "g1"
    MJCF = os.path.join(BASE06, "models", "g1_mjcf", "g1-mode_13_15.xml")
print("回放机器人:", _ROBOT, "场景:", MJCF)


def _ver_key(name):
    # 版本目录自然排序键: 按 v 后数字排(v2<v5<v10), 字母序会把 v10 排在 v5 前
    m = re.match(r"v(\d+)", name)
    return int(m.group(1)) if m else -1


_OUT06 = os.environ.get("G1W_OUT", os.path.join(BASE, "output"))
_vdirs = [os.path.basename(d) for d in glob.glob(os.path.join(_OUT06, "ufo_motion/v*"))]
VER = os.environ.get("MOTION_VER") or (max(_vdirs, key=lambda d: (_ver_key(d), d))
                                       if _vdirs else "v1")
FR = os.path.join(_OUT06, "ufo_motion", VER, "replay_frames")
os.makedirs(FR, exist_ok=True)

model = mujoco.MjModel.from_xml_path(MJCF)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, 480, 640)

# 动作文件由 VER 拼出(不再独立取 sorted 最新, 避免与显式 MOTION_VER 错位)
_npz = os.path.join(_OUT06, "ufo_motion", VER, "wall_spray_motion.npz")
assert os.path.exists(_npz), f"动作文件不存在: {_npz} (先运行 05, 或用 MOTION_VER 指定版本)"
motion = np.load(_npz)
print("回放动作:", _npz)
qpos = motion["qpos"].astype(float)
T = len(qpos)
print(f"回放 {T} 帧, 每2帧渲染1张")

def set_cam(azimuth, elevation, distance, lookat):
    c = mujoco.MjvCamera()
    c.azimuth, c.elevation, c.distance = azimuth, elevation, distance
    c.lookat[:] = lookat
    return c

if _ROBOT == "tiangong3":      # 天工更高(骨盆~1.0m), 拉远抬视点
    cam_side = set_cam(90, 6, 3.4, (0.15, 0.0, 0.80))
    cam_front = set_cam(180, 9, 3.4, (0.2, 0.0, 0.80))
else:
    cam_side = set_cam(90, 5, 2.6, (0.15, 0.0, 0.55))     # 侧视(机器人面向+x)
    cam_front = set_cam(180, 8, 2.6, (0.2, 0.0, 0.55))    # 斜前视

n_out = 0
for i in range(0, T, 2):
    data.qpos[:] = qpos[i]
    mujoco.mj_forward(model, data)
    for cam, tag in ((cam_side, "side"), (cam_front, "front")):
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        from PIL import Image
        Image.fromarray(img).save(os.path.join(FR, f"{tag}_{n_out:05d}.png"))
    n_out += 1
    if n_out % 100 == 0:
        print(f"{n_out} 帧已渲染", flush=True)
print(f"[完成] {n_out} 组帧 -> {FR}")
