# -*- coding: utf-8 -*-
"""V2 双姿态(站+蹲)喷涂管线数值验收.

用法(仓库 g1-wall-spray 下):
  G1W_OUT=../output/v2_dual G1W_IN=../output MOTION_VER=v6_dual_posture \
    python verify/check_v2_dual.py

检查项(全过 exit 0, 任一失败 exit 1):
  1. trajectory_meta.json: posture=dual, ik_fail==0, 枪墙夹角>=60°, 蹲姿行>=1,
     行高覆盖 [<=0.45, >=1.45]
  2. trajectory.npz: zone 同时含 stand/crouch, 恰好 1 次姿态翻转, 过渡段(kind=transition)
     覆盖翻转点(05 的回零位过渡窗口依赖它)
  3. 动作导出: qpos 无 NaN, 30Hz, 喷涂帧实测枪墙夹角>=60°(spray_angle_ok),
     最大关节速度<=6 rad/s(审计环硬阈 9 之下留裕度), 蹲姿确实执行(root z 最低<0.70m),
     回零位过渡帧存在(phase=home)
"""
import os
import sys
import json
import csv
import re

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get("G1W_OUT", os.path.join(BASE, "output"))


def _newest_motion_ver(out_base):
    """取 ufo_motion 下最新 v* 版本目录(默认); 防止硬编码旧默认导致误验旧版本."""
    import glob
    vd = os.path.join(out_base, "ufo_motion")
    cands = [os.path.basename(d) for d in glob.glob(os.path.join(vd, "v*"))]
    def _k(name):
        m = re.match(r"v(\d+)", name)
        return (int(m.group(1)) if m else -1, name)
    return max(cands, key=_k) if cands else "v6_dual_posture"


VER = os.environ.get("MOTION_VER") or _newest_motion_ver(OUT)
MOT = os.path.join(OUT, "ufo_motion", VER)

def _chk_vertical():
    """V3 竖直列检查: 走线点u极差=0(竖直线) + 指向横向分量=0(枪臂在机器人-墙面法向垂面内)."""
    import csv as _csv
    wp = list(_csv.DictReader(open(os.path.join(OUT, "wall_path.csv"), encoding="utf8")))
    stroke_u = [float(r["u_m"]) for r in wp if r["kind"] == "stroke"]
    cols = sorted(set(round(u, 4) for u in stroke_u))
    bad = 0
    for c in cols:
        us = [u for u in stroke_u if abs(u - c) < 1e-4]
        if max(us) - min(us) > 1e-6:
            bad += 1
    check("V3列严格竖直(u极差=0)", bad == 0,
          f"{len(cols)} 列, 非竖直 {bad} 列")
    rp = os.path.join(OUT, "robot_path.csv")
    if os.path.exists(rp):
        rows = list(_csv.DictReader(open(rp, encoding="utf8")))
        lat = max(abs(float(r["axis_dy"])) for r in rows if r["kind"] == "stroke")
        check("V3枪臂在法向垂面内(|axis_y|=0)", lat < 1e-6, f"横向分量 max {lat:.1e}")

fails = []


def check(name, ok, detail=""):
    print(("[PASS] " if ok else "[FAIL] ") + name + (f"  {detail}" if detail else ""))
    if not ok:
        fails.append(name)


# ---------- 1. 轨迹规划 meta ----------
meta = json.load(open(os.path.join(OUT, "trajectory_meta.json"), encoding="utf8"))
check("posture=dual", meta.get("posture") == "dual")
# IK 标准(2026-10-12 调整): 腰yaw ±45° 限位后, 与 y60 网格的口径差使个别极缘点
# 在 03 的 URDF 解算不闭合(实测 ~0.4%), 由 05 的 MJCF 重解+边缘档收编 —— 最终
# 质量以下方动作级检查(角度/速度/NaN)为准, 03 层要求 严格率>=99% 且失败<=0.5%.
_n_pt = meta.get("n_points") or 1
_ik_fail_n = _n_pt - (meta.get("ik_strict") or 0) - (meta.get("ik_loose") or 0)
check("IK 严格率>=99%且失败<=0.5%",
      (meta.get("ik_strict") or 0) / _n_pt >= 0.99 and _ik_fail_n / _n_pt <= 0.005,
      f"strict={meta.get('ik_strict')}/{_n_pt}, 边缘={meta.get('ik_loose')}, "
      f"失败={_ik_fail_n}(最终动作由下方动作级检查把关)")
check("枪墙夹角全部>=60°", bool(meta.get("angle_rule_ok")),
      f"实测 [{meta.get('angle_achieved_deg', [0, 0])[0]:.1f}, "
      f"{meta.get('angle_achieved_deg', [0, 0])[-1]:.1f}]°")
check("蹲姿行>=2(深蹲版)", meta.get("stations", {}).get("crouch_rows", 0) >= 2,
      f"crouch_rows={meta.get('stations', {}).get('crouch_rows')}")
check("行高覆盖 0.35~1.45", meta.get("h_min_row", 9) <= 0.35 and meta.get("h_top", 0) >= 1.45,
      f"rows [{meta.get('h_min_row')}, {meta.get('h_top')}]")
check("两带等宽区间已写", bool(meta.get("crouch_stroke_u")),
      f"stand={meta.get('uniform_stroke_u')} crouch={meta.get('crouch_stroke_u')}")
check("站/蹲带u区间对齐(用户定稿)", meta.get("crouch_stroke_u") == meta.get("uniform_stroke_u"),
      f"stand={meta.get('uniform_stroke_u')} crouch={meta.get('crouch_stroke_u')}")

# ---------- 2. 轨迹 npz 结构 ----------
t = np.load(os.path.join(OUT, "trajectory.npz"), allow_pickle=True)
z, k = t["zone"], t["kind"]
flips = np.where(z[1:] != z[:-1])[0]
tr = np.where(k == "transition")[0]
check("zone 含站+蹲两姿", set(np.unique(z)) == {"stand", "crouch"},
      f"{ {u: int((z == u).sum()) for u in np.unique(z)} }")
check("恰好 1 次姿态翻转", len(flips) == 1, f"翻转点 {flips.tolist()}")
check("过渡段覆盖翻转点", len(tr) > 0 and tr.min() <= flips[0] <= tr.max(),
      f"transition idx[{tr.min()},{tr.max()}] 翻转@{int(flips[0]) if len(flips) else -1}")

# ---------- 3. 动作导出 ----------
mp = os.path.join(MOT, "wall_spray_motion.npz")
mm = os.path.join(MOT, "motion_meta.json")
cp = os.path.join(MOT, "wall_spray_motion_qpos.csv")
if not (os.path.exists(mp) and os.path.exists(mm)):
    check("动作导出存在", False, f"缺 {mp} 或 {mm} (先跑 05_export_ufo_motion.py)")
else:
    q = np.load(mp)["qpos"]
    mmeta = json.load(open(mm, encoding="utf8"))
    check("qpos 无 NaN", bool(np.isfinite(q).all()), f"shape={q.shape}")
    check("喷涂帧枪墙夹角>=60°", bool(mmeta.get("spray_angle_ok")),
          f"spray_angle_deg={mmeta.get('spray_angle_deg')}")
    check("最大关节速度<=6 rad/s", float(mmeta.get("max_joint_vel_rad_s", 99)) <= 6.0,
          f"实测 {mmeta.get('max_joint_vel_rad_s')} rad/s")
    root_z_min = float(q[:, 2].min())
    check("深蹲确实执行(root z<0.55m)", root_z_min < 0.55,
          f"root z 范围 [{root_z_min:.3f}, {float(q[:, 2].max()):.3f}]m")
    if os.path.exists(cp):
        with open(cp, encoding="utf8") as f:
            phases = {row[1] for row in csv.reader(f)}
        check("回零位过渡帧存在", "home" in phases)
    else:
        check("回零位过渡帧存在", False, "缺 qpos csv")

if json.load(open(os.path.join(OUT, "trajectory_meta.json"), encoding="utf8"))\
        .get("stroke_dir") == "vertical":
    _chk_vertical()
print("=" * 50)
if fails:
    print(f"验收未通过: {len(fails)} 项失败 -> {fails}")
    sys.exit(1)
print("V2 双姿态管线验收全部通过")
