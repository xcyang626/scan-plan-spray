# -*- coding: utf-8 -*-
"""步骤6: 在 MuJoCo(场景模型取自机器人登记表) 中回放参考动作, 渲染帧序列.

场景机器人由**动作版本目录内的 motion_meta.json** 决定(不再是"MOTION_VER 拼个路径,
取不到就静默用 G1"): 动作版本与场景机器人必须来自同一版本, 否则会把天工3的动作放进
G1 场景(关节维/墙距/相机全错)而且毫无提示. 版本目录取不到 motion_meta.json 直接报错退出.
用法: conda mujoco 环境, MUJOCO_GL=glfw DISPLAY=:0
输出: output/ufo_motion/<ver>/replay_frames/*.png (每2帧1张, 15fps)
"""
import numpy as np
import mujoco
import os, glob, re, json
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE06 = BASE
_OUT06 = os.environ.get("G1W_OUT", os.path.join(BASE06, "output"))
_MOTION = "wall_spray_motion.npz"


def _ver_key(name):
    # 版本目录自然排序键: 按 v 后数字排(v2<v5<v10), 字母序会把 v10 排在 v5 前
    m = re.match(r"v(\d+)", name)
    return int(m.group(1)) if m else -1


# ---- 1. 动作版本: 动作文件所在的版本目录(与场景机器人同源, 后面只认这一个) ----
# 版本目录不限 "v*" 前缀: 天工的版本目录叫 t3_v1(见 05_export_tiangong), 只匹配
# "v*" 会让天工动作永远探测不到(只能显式传 MOTION_VER).
_vdirs = [os.path.basename(d) for d in glob.glob(os.path.join(_OUT06, "ufo_motion/*"))
          if os.path.isdir(d)]
_vhave = [d for d in _vdirs
          if os.path.exists(os.path.join(_OUT06, "ufo_motion", d, _MOTION))]
if os.environ.get("MOTION_VER"):
    VER = os.environ["MOTION_VER"]
else:
    # 自动探测: 取**确实含动作文件**的最新版本(避免选到只有中间产物的空版本目录)
    if not _vhave:
        raise SystemExit(f"[回放] {os.path.join(_OUT06, 'ufo_motion')} 下没有含 "
                         f"{_MOTION} 的版本目录(先运行 05 生成动作)")
    VER = max(_vhave, key=lambda d: (_ver_key(d), d))
VER_DIR = os.path.join(_OUT06, "ufo_motion", VER)
_npz = os.path.join(VER_DIR, _MOTION)
if not os.path.exists(_npz):
    raise SystemExit(f"[回放] 动作文件不存在: {_npz} (MOTION_VER={VER}, "
                     f"先运行 05, 或用 MOTION_VER 指定版本)")

# ---- 2. 场景机器人: 由该版本目录的 motion_meta.json 决定 ----
_meta_p = os.path.join(VER_DIR, "motion_meta.json")
if not os.path.exists(_meta_p):
    raise SystemExit(f"[回放] 错误: {_meta_p} 不存在, 无法确定动作对应的机器人型号; "
                     f"拒绝静默按 G1 场景回放(用 MOTION_VER 指定正确的动作版本)")
with open(_meta_p, encoding="utf8") as _f:
    _meta = json.load(_f)
try:
    with open(os.path.join(BASE06, "station", "robot_registry.json"), encoding="utf8") as _f:
        _REG = json.load(_f)
except Exception as _e:
    raise SystemExit(f"[回放] 错误: 读取机器人登记表失败({_e})")

with np.load(_npz) as _mz:
    qpos = _mz["qpos"].astype(float)

_robot = _meta.get("robot")
_robot_src = "motion_meta.json"
if not _robot:
    # G1 导出器(05_export_ufo_motion)未写 robot 字段: 用 npz 的 qpos 维度与登记表
    # dof.total_qpos 比对推断; 推不出唯一结果就报错(不猜, 更不静默用 G1)
    _cand = [k for k, v in _REG.items()
             if isinstance(v, dict) and v.get("dof", {}).get("total_qpos") == qpos.shape[1]]
    if len(_cand) == 1:
        _robot, _robot_src = _cand[0], f"npz qpos 维度 {qpos.shape[1]} 推断"
if not _robot:
    raise SystemExit(f"[回放] 错误: {_meta_p} 无 robot 字段, 且动作 qpos 维度"
                     f"({qpos.shape[1]})无法与登记表唯一对应; 拒绝静默用 G1 场景回放")
if _robot not in _REG:
    raise SystemExit(f"[回放] 错误: 登记表 station/robot_registry.json 没有 '{_robot}' 条目")

# ---- 3. 场景模型: 登记表 mjcf(环境变量 G1W_MJCF 可覆盖), 不再回退 G1 MJCF ----
_m = os.environ.get("G1W_MJCF") or _REG[_robot].get("mjcf")
if not _m:
    raise SystemExit(f"[回放] 错误: 登记表 '{_robot}' 没有 mjcf 字段, 无法加载场景")
_m = os.path.expanduser(_m)        # registry 允许 ~/.cache/... 写法
MJCF = _m if os.path.isabs(_m) else os.path.join(BASE06, _m)
if not os.path.exists(MJCF):
    raise SystemExit(f"[回放] 错误: 场景模型不存在: {MJCF}")
print(f"[回放] 场景机器人={_robot}(来源 {_robot_src}), 动作版本={VER}, 场景: {MJCF}")

FR = os.path.join(VER_DIR, "replay_frames")
os.makedirs(FR, exist_ok=True)

model = mujoco.MjModel.from_xml_path(MJCF)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, 480, 640)

print("回放动作:", _npz)
T = len(qpos)
print(f"回放 {T} 帧, 每2帧渲染1张")


def set_cam(azimuth, elevation, distance, lookat):
    c = mujoco.MjvCamera()
    c.azimuth, c.elevation, c.distance = azimuth, elevation, distance
    c.lookat[:] = lookat
    return c


if _robot == "tiangong3":      # 天工更高(骨盆~1.0m), 拉远抬视点
    cam_side = set_cam(90, 6, 3.4, (0.15, 0.0, 0.80))
    cam_front = set_cam(180, 9, 3.4, (0.2, 0.0, 0.80))
else:
    cam_side = set_cam(90, 5, 2.6, (0.15, 0.0, 0.55))     # 侧视(机器人面向+x)
    cam_front = set_cam(180, 8, 2.6, (0.2, 0.0, 0.55))    # 斜前视

n_out = 0
try:
    for i in range(0, T, 2):
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(model, data)
        for cam, tag in ((cam_side, "side"), (cam_front, "front")):
            renderer.update_scene(data, camera=cam)
            img = renderer.render()
            Image.fromarray(img).save(os.path.join(FR, f"{tag}_{n_out:05d}.png"))
        n_out += 1
        if n_out % 100 == 0:
            print(f"{n_out} 帧已渲染", flush=True)
finally:
    renderer.close()
print(f"[完成] {n_out} 组帧 -> {FR}")
