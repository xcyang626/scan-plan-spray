#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""speed_matrix.py —— 实时调速的孪生验证矩阵(离线, 不接触机器人)
================================================================================
问题: rl_exec 新增的 /rl/speed_scale 实时调速(参考时间轴 0.5~1.5×), 策略
只在 143.6s 原始时序上训练过 —— 变速是时序分布外, 能不能稳?
方法: 孪生回放(ilc_check_real 同律, 无故障) × 倍率矩阵
  {0.5, 0.75, 1.0, 1.25, 1.5}×, qd_scale/vref 同步缩放(与 rl_exec 部署律
  一致), 报告 跟踪/力矩/积分器蓄能/带宽 指标随倍率的退化曲线。
判读: mean/max 随倍率平滑变化(无拐点)、|τ|不触钳位、0.5~2Hz 不爆 ——
  则该倍率范围可用; 出现拐点则以拐点内为操作范围。

用法(mujoco conda 环境):
    python speed_matrix.py                # 默认矩阵, 基线前馈表
    python speed_matrix.py --scales 0.75,1.0,1.25
输出: speed_matrix_report.json
⚠ 纯离线; 真机变须经 rl_track/README.md 分段阶梯。
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from rl_policy import Policy                                     # noqa: E402
from ilc_check_real import IC, band_p2p, run_replay              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="0.5,0.75,1.0,1.25,1.5")
    ap.add_argument("--ff", default=os.path.join(_HERE, "grav_ff_right.npz"))
    ap.add_argument("--actor", default=None,
                    help="候选 actor npz(默认部署副本 actor_right.npz)")
    ap.add_argument("--out", default=os.path.join(
        _HERE, "speed_matrix_report.json"))
    args = ap.parse_args()
    scales = [float(x) for x in args.scales.split(",")]

    pol = Policy(actor_path=os.path.abspath(args.actor)
                 if args.actor else None)
    gff = np.load(args.ff)["gff"][:pol.n]
    out = []
    for s in scales:
        t0 = time.time()
        rec = run_replay(pol, gff, None, None, fault=False, speed=s)
        err = np.degrees(rec["qref"] - rec["q"])
        integ_pk = np.abs(rec["integ"]).max(axis=0)
        tau_pk = np.abs(rec["tau"]).max(axis=0)
        r = dict(
            scale=s, frames=int(len(err)),
            err_mean=float(np.abs(err).mean()),
            err_max=float(np.abs(err).max()),
            band_p2p_deg=band_p2p(np.radians(err), 0.1, 0.5).tolist(),
            integ_peak=integ_pk.tolist(), tau_peak=tau_pk.tolist(),
            wall_s=round(time.time() - t0, 1))
        out.append(r)
        print("%4.2f×: mean %.2f°  max %.2f°  0.1-0.5Hz p2p J2 %.2f°  "
              "|τ| %.1fNm  integ %.1f  (%.0fs)"
              % (s, r["err_mean"], r["err_max"],
                 r["band_p2p_deg"][1], max(tau_pk), max(integ_pk),
                 r["wall_s"]))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dict(scales=out), f, ensure_ascii=False, indent=1)
    print("已写出 %s" % args.out)


if __name__ == "__main__":
    main()
