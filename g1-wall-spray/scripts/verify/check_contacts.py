# -*- coding: utf-8 -*-
"""动作自碰撞检查: 右臂链 vs 双腿链 逐帧 MuJoCo 接触检测(02 工作区不建模自碰撞).

深蹲(V2 蹲姿骨盆地板下探)时手臂下前伸最易与大腿干涉, 本脚本对导出动作逐帧
(步长可调)做 mj_forward 接触检测: 报告臂-腿接触帧与穿透深度.
深穿透(<-5mm)exit 1, 否则 exit 0(浅接触/无接触).

用法(conda mujoco): python verify/check_contacts.py <motion.npz> [--step 5]
"""
import argparse
import json
import os
import sys

import numpy as np
import mujoco

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "scripts"))


def body_class(name):
    n = name.lower()
    if any(k in n for k in ("shoulder", "elbow", "wrist")) and "right" in n:
        return "arm_r"
    if any(k in n for k in ("hip", "knee", "ankle", "calf", "thigh", "foot")):
        return "leg"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("motion")
    ap.add_argument("--step", type=int, default=5)
    a = ap.parse_args()
    _reg = json.load(open(os.path.join(BASE, "station", "robot_registry.json"),
                          encoding="utf8"))
    mjcf = _reg["g1"]["mjcf"]
    mjcf = os.path.expanduser(mjcf)    # registry 允许 ~/.cache/... 写法
    mjcf = mjcf if os.path.isabs(mjcf) else os.path.join(BASE, mjcf)
    model = mujoco.MjModel.from_xml_path(mjcf)
    data = mujoco.MjData(model)
    q = np.load(a.motion)["qpos"].astype(float)
    bcls = [body_class(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                        model.geom_bodyid[g]) or "")
            for g in range(model.ngeom)]
    hits, worst = [], 0.0
    for i in range(0, len(q), a.step):
        data.qpos[:] = q[i]
        mujoco.mj_forward(model, data)
        for c in data.contact[:data.ncon]:
            ba, bb = bcls[c.geom1], bcls[c.geom2]
            if {ba, bb} == {"arm_r", "leg"}:
                hits.append((i, c.dist,
                             mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
                             mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)))
                worst = min(worst, float(c.dist))
    n_f = len(range(0, len(q), a.step))
    print(f"[自碰撞] 检查 {n_f} 帧(步长{a.step}): 臂-腿接触 {len(hits)} 处, "
          f"最深穿透 {abs(worst)*1000:.1f}mm")
    for h in hits[:8]:
        print(f"  帧{h[0]}: {h[2]} <-> {h[3]} 穿透 {abs(h[1])*1000:.1f}mm")
    bad = worst < -0.005
    print(f"[自碰撞] {'FAIL: 存在深穿透(>5mm), 深蹲地板过深或臂姿干涉' if bad else 'PASS'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
