#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_grav_ff.py —— 沿参考轨迹生成重力前馈表 grav_ff_right.npz
======================================================================
内容: gff[i] = qfrc_bias(q_ref[i], qd_ref[i]) [Nm], 50Hz 行与 ref 帧一一同
idx 对应(重力 + 科氏; 参考速度 ~0.4rad/s, 科氏占比极小)。URDF/右臂关节序。
qfrc_bias 不含 armature 项(armature 只进质量阵), 与转子折算无关。

动力学口径(v2, 2026-09-16): **真机描述 URDF 为权威**(与官方
openarm_description 同源, 且与训练 USD 的 URDF 逐项一致 —— 已审计)。
v1 曾用 openarm_right_rl.xml(MJCF), 但其质量是 make_mjcf.py 的
胶囊体/盒体近似(与 URDF 最大差 3.5×, link2/link7), 已废弃。
URDF 直接喂 MuJoCo 会因 package:// mesh 路径失败, 故剥离
visual/collision 只留 inertial+joint 后临时编译(动力学不受影响)。

数据流(唯一权威源, 2026-09-16 重力前馈方案):
  ref_right.npz + real/config/openarm_right_real.urdf → 本脚本 → grav_ff_right.npz
     ├→ 训练 zigzag_env(mit_actuator._ff, 逐 env 缩放 DR)
     ├→ rl_validate.py --grav-ff
     ├→ rl_exec.py --grav-ff(注入消息 → zero_offset_hw → MIT t_ff)
     └→ pd_screen.py(内部 import 同一 compute_grav_ff)
⚠ 参考轨迹重新生成后必须重跑本脚本(verify_deploy.py 审计行数/T_total 配对)。
⚠ MuJoCo 验证链(rl_validate/pd_screen 回放)的 MJCF 物理仍是近似质量,
  模拟中 FF 与 MJCF 重力的残余差 = 保守余量; MJCF 质量同步 URDF 属
  独立任务(牵动全部既有 MuJoCo 基线与 sim2real 系数, 未做)。

用法(mujoco conda 环境):
    python make_grav_ff.py          # 默认输出 ../isaaclab_sim/rl/grav_ff_right.npz
    python make_grav_ff.py --out real/right/rl_track/grav_ff_right.npz  # 部署副本
"""

import argparse
import json
import os
import tempfile
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(_HERE, "openarm_right_rl.xml")
REF = os.path.abspath(os.path.join(_HERE, "..", "isaaclab_sim", "rl",
                                   "ref_right.npz"))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
URDF = os.path.join(_REPO, "real", "right", "config",
                    "openarm_right_real.urdf")
OUT = os.path.abspath(os.path.join(_HERE, "..", "isaaclab_sim", "rl",
                                   "grav_ff_right.npz"))


def strip_geometry(urdf_path):
    """剥离 visual/collision(留 inertial/joint/transmission) → 临时文件。
    MuJoCo 只需要运动学+惯性张量; mesh 引用(package://)无法解析。
    注入 balanceinertia: 指尖小连杆的 URDF 惯量不满足三角不等式
    (官方 CAD 数据原样), 只影响指尖自身动力学, 对臂关节重力可忽略。"""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    n_geo = 0
    for link in root.iter("link"):
        for tag in ("visual", "collision"):
            for g in list(link.findall(tag)):
                link.remove(g)
                n_geo += 1
    mj_ext = root.find("mujoco")
    if mj_ext is None:
        mj_ext = ET.SubElement(root, "mujoco")
        compiler = ET.SubElement(mj_ext, "compiler")
        compiler.set("balanceinertia", "true")
    tmp = tempfile.NamedTemporaryFile(
        suffix=".urdf", delete=False, mode="wb")
    tree.write(tmp, encoding="utf-8", xml_declaration=True)
    tmp.close()
    return tmp.name, n_geo


def compute_grav_ff(model_source, q_ref, qd_ref, fric_visc=0.2,
                    fric_coul=0.3, v_lin=0.05):
    """逐帧前馈 [Nm] = 重力+科氏 + 摩擦换向预补偿 —— 各消费方共用实现。
    model_source: URDF(权威, 自动剥 mesh)或 MJCF 路径。
    摩擦预补偿(v3): τ_fric = b·q̇_ref + c·tanh(q̇_ref/v_lin), 系数取训练
    env 摩擦 DR 区间中值(0.05~0.4 / 0~0.6)—— 速度参考变号时提前把力矩
    顶过摩擦带, 直击 pd_screen 定位的换向滞后主源(J2@59s/J4@69s);
    残余失配由训练侧摩擦 DR 覆盖。"""
    use_urdf = model_source.lower().endswith(".urdf")
    if use_urdf:
        stripped, n_geo = strip_geometry(model_source)
        try:
            model = mujoco.MjModel.from_xml_path(stripped)
        finally:
            os.unlink(stripped)
    else:
        model = mujoco.MjModel.from_xml_path(model_source)
    data = mujoco.MjData(model)
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                              "openarm_right_joint%d" % i)
            for i in range(1, 8)]
    assert all(j >= 0 for j in jids), "URDF 缺臂关节? 检查 joint 命名"
    fq = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                            "openarm_right_finger_joint%d" % f)
          for f in (1, 2)]
    qadr = np.array([model.jnt_qposadr[j] for j in jids])
    fadr = np.array([model.jnt_qposadr[j] for j in fq])
    vadr = np.array([model.jnt_dofadr[j] for j in jids])
    gff = np.empty((len(q_ref), 7))
    for i in range(len(q_ref)):
        data.qpos[qadr] = q_ref[i]
        data.qpos[fadr] = 0.02
        data.qvel[vadr] = qd_ref[i]
        mujoco.mj_forward(model, data)
        gff[i] = data.qfrc_bias[vadr] \
            + fric_visc * qd_ref[i] \
            + fric_coul * np.tanh(qd_ref[i] / v_lin)
    return gff


def main():
    ap = argparse.ArgumentParser(
        description="沿参考轨迹生成重力前馈表 grav_ff_right.npz")
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--model", default=URDF,
                    help="动力学模型源(默认真机描述 URDF=权威; "
                         "传 MJCF 则用其近似质量, 仅调试用)")
    ap.add_argument("--fric-visc", type=float, default=0.2,
                    help="粘性摩擦预补偿系数 [Nm·s/rad](默认 DR 区间中值)")
    ap.add_argument("--fric-coul", type=float, default=0.3,
                    help="库仑摩擦预补偿系数 [Nm](默认 DR 区间中值; 0=关)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    d = np.load(args.ref, allow_pickle=True)
    q_ref, qd_ref = d["q_ref"], d["qd_ref"]
    meta = d["meta_json"].item()
    if isinstance(meta, (bytes, str)):
        meta = json.loads(meta)

    gff = compute_grav_ff(args.model, q_ref, qd_ref,
                          fric_visc=args.fric_visc,
                          fric_coul=args.fric_coul)
    out_meta = dict(
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        tool="make_grav_ff.py(v3 URDF+fric-precomp, 2026-09-16)",
        ref=os.path.abspath(args.ref),
        model=os.path.abspath(args.model),
        dynamics="real URDF description (== training USD source)",
        fric_precomp=dict(visc=args.fric_visc, coul=args.fric_coul),
        T_total=float(meta["T_total"]), hz=50.0, unit="Nm",
        frame="URDF/right joint1..7",
        note="重力+科氏 + b·q̇_ref + c·tanh(q̇_ref/v_lin); 与 ref 帧同 idx; "
             "不含 armature; 消费方: zigzag_env/rl_validate/rl_exec",
    )
    np.savez(args.out, gff=gff.astype(np.float64),
             meta_json=json.dumps(out_meta, ensure_ascii=False))
    a = np.abs(gff)
    print("重力前馈已写出: %s (%d 行)" % (args.out, len(gff)))
    print("逐关节 |gff| 范围 [min, max] Nm 与均值:")
    for j in range(7):
        print("  J%d: [%6.2f, %6.2f]  mean %6.2f" % (
            j + 1, a[:, j].min(), a[:, j].max(), a[:, j].mean()))
    print("部署副本: cp %s ../../real/right/rl_track/grav_ff_right.npz"
          % os.path.basename(args.out))
    if args.model.lower().endswith(".xml"):
        print("⚠ 使用了 MJCF(近似质量) —— 正式产物必须用默认 URDF 口径")


if __name__ == "__main__":
    main()
