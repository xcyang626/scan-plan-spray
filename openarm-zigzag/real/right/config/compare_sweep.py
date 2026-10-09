#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比扫查日志: 仿真 vs 实机
============================
读取 joint_sweep_sim.py 与 joint_sweep.py 落盘的 JSON(默认取各自
sweep_*_latest.json), 逐关节输出: 指令幅度、两侧达成值、回位残差、
耦合(动 J 时其余关节的最大窜动)、实机−仿真差。

用法:
  /usr/bin/python3 compare_sweep.py                    # 用两份 latest
  /usr/bin/python3 compare_sweep.py sim.json real.json

⚠ 结论解读(重要):
  - 数值能查出: 达成幅度差、回位残差(零位漂移/跟踪差)、耦合窜动。
  - 数值查不出: channel_sign 方向错误 —— 命令与反馈经同一个(错)符号
    换算后自洽。**方向以肉眼对照 RViz 为准**(阶段 F 终裁口径)。
"""

import json
import math
import os
import sys

D2R = math.pi / 180.0
_SIM = os.path.expanduser(
    "~/桌面/openarm实验/仿真/moveit_sim/sweep_sim_latest.json")
_REAL = os.path.expanduser(
    "~/桌面/openarm实验/real/right/config/sweep_real_latest.json")


def load(p, tag):
    if not os.path.isfile(p):
        raise SystemExit("缺少%s日志: %s" % (tag, p))
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def per_joint(d):
    """从日志提取每关节: target, achieved, resid, coupling(度)。"""
    out = {}
    q0 = d["q_start"]
    for seg in d["segments"]:
        j = seg["joint"]
        i = j - 1
        go, back = seg["q_go_avg"], seg["q_back_avg"]
        # 耦合: 动 J 期间, 其余 6 关节相对基位的最大偏移
        coup = 0.0
        for k in range(7):
            if k == i:
                continue
            coup = max(coup, abs(go[k] - q0[k]) / D2R)
        resid = max(abs(back[k] - q0[k]) for k in range(7)) / D2R
        out[j] = dict(target=seg["target_deg"],
                      got=(go[i] - q0[i]) / D2R,
                      resid=resid, coup=coup)
    return out


def main():
    sim_p = sys.argv[1] if len(sys.argv) > 1 else _SIM
    real_p = sys.argv[2] if len(sys.argv) > 2 else _REAL
    sim, real = load(sim_p, "仿真"), load(real_p, "实机")

    if abs(sim.get("deg", 5) - real.get("deg", 5)) > 1e-9:
        print("⚠ 两侧 --deg 不一致(仿 %.1f / 实 %.1f), 对比无意义!"
              % (sim["deg"], real["deg"]))
        return
    if abs(sim.get("speed_rad_s", 0.1) - real.get("speed_rad_s", 0.1)) > 1e-9:
        print("⚠ 两侧 --speed 不一致(仿 %.2f / 实 %.2f), 达成值对比打折"
              % (sim["speed_rad_s"], real["speed_rad_s"]))

    s, r = per_joint(sim), per_joint(real)
    common = sorted(set(s) & set(r))
    if not common:
        raise SystemExit("两侧没有共同的已完成关节段")

    hdr = ("J", "指令", "仿达成", "仿残差", "仿耦合",
           "实达成", "实残差", "实耦合", "达成差(实-仿)")
    print("%-4s" % hdr[0] + "%8s" * 8 % hdr[1:])
    for j in common:
        row = (j, s[j]["target"],
               s[j]["got"], s[j]["resid"], s[j]["coup"],
               r[j]["got"], r[j]["resid"], r[j]["coup"],
               r[j]["got"] - s[j]["got"])
        print(("J%-2d %5.1f " % (row[0], row[1])) +
              " ".join("%+7.2f" % v for v in row[2:]))

    worst = max(common, key=lambda j: abs(r[j]["got"] - s[j]["got"]))
    print("\n达成幅度最大差: J%d %.2f° (仿 %+.2f / 实 %+.2f)"
          % (worst, r[worst]["got"] - s[worst]["got"],
             s[worst]["got"], r[worst]["got"]))
    bad_resid = [j for j in common if r[j]["resid"] > 1.0]
    if bad_resid:
        print("⚠ 实机回位残差 >1°: %s —— 零位漂移或积分器未收敛, "
              "先查再继续" % bad_resid)
    bad_coup = [j for j in common if r[j]["coup"] > 1.0]
    if bad_coup:
        print("⚠ 实机耦合窜动 >1°: %s —— 动该关节时其它关节在动, "
              "检查机械/读数" % bad_coup)
    print("\n提醒: 以上数值查不出方向符号错误(读数链自洽)。")
    print("      每关节物理转向是否与 RViz 一致, 以你肉眼记录为准(阶段 F)。")


if __name__ == "__main__":
    main()
