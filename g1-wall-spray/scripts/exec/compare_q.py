# -*- coding: utf-8 -*-
"""对比守护进程同步记录: 指令上肢17(SONIC序) vs 机器人实测上肢17。

用途: 一次执行就能定位 —— ①关节左右是否换位 ②各关节零位/符号偏移(基准问题)
③腰关节是否被跟踪 ④哪些关节跟踪差。纯 numpy, 任意环境可跑。

用法:
  python compare_q.py <记录.npz>            # 记录 = 交接命令.sh 记录停止 的产物
输出: 逐关节表 + 整体判定(左右换位/腰跟踪/偏移)
"""
from __future__ import annotations

import sys

import numpy as np

# SONIC 交错序关节名(与官方 send_upper_body_trajectory.py 一致)
SONIC_NAMES = [
    "腰yaw", "腰roll", "腰pitch",
    "左肩pitch", "右肩pitch", "左肩roll", "右肩roll",
    "左肩yaw", "右肩yaw", "左肘", "右肘",
    "左腕roll", "右腕roll", "左腕pitch", "右腕pitch", "左腕yaw", "右腕yaw",
]
# SONIC 序里 左/右 关节的互换重排(0-2 腰不动, 3↔4, 5↔6, ... 成对交换)
LR_SWAP = list(range(3))
for a, b in ((3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)):
    LR_SWAP += [b, a]


def nearest_align(t_from, q_from, t_to):
    """把 q_from(t_from) 对齐到 t_to 的每个时刻(最近邻索引)."""
    idx = np.searchsorted(t_from, t_to).clip(1, len(t_from) - 1)
    left = np.abs(t_from[idx - 1] - t_to)
    right = np.abs(t_from[idx] - t_to)
    idx = idx - (left < right)
    return q_from[idx]


def analyze(path):
    z = np.load(path, allow_pickle=False)
    st, sq = z["sent_t"], z["sent_upper17"]        # 指令(50Hz)
    mt, mq = z["meas_t"], z["meas_upper17"]        # 实测(g1_debug)
    if not len(st) or not len(mt):
        print(f"记录不完整: 指令 {len(st)} 帧, 实测 {len(mt)} 帧")
        return 1
    dur = min(st[-1], mt[-1]) - max(st[0], mt[0])
    print(f"记录: 指令 {len(st)} 帧({st[-1]-st[0]:.1f}s), "
          f"实测 {len(mt)} 帧({mt[-1]-mt[0]:.1f}s), 重叠 {dur:.1f}s\n")
    # 时间窗取重叠段
    m = (mt >= st[0]) & (mt <= st[-1])
    mt2, mq2 = mt[m], mq[m]
    cmd = nearest_align(st, sq, mt2)               # 对齐后的指令

    err = mq2 - cmd
    print(f"{'关节':<10}{'指令幅度':>10}{'实测幅度':>10}{'均值偏移°':>11}"
          f"{'均值|误差|°':>12}{'跟踪率':>8}")
    swap_scores = []
    for i, name in enumerate(SONIC_NAMES):
        c, r, e = np.degrees(cmd[:, i]), np.degrees(mq2[:, i]), np.degrees(err[:, i])
        amp_c, amp_r = c.max() - c.min(), r.max() - r.min()
        track = (amp_r / amp_c) if amp_c > 3.0 else float("nan")   # 指令幅度<3°不评
        print(f"{name:<10}{amp_c:10.1f}{amp_r:10.1f}{e.mean():11.2f}"
              f"{np.abs(e).mean():12.2f}"
              f"{'' if np.isnan(track) else f'{track*100:6.0f}%'}")
        swap_scores.append(np.abs(np.degrees(
            mq2[:, LR_SWAP[i]] - cmd[:, i])).mean())
    # 判定
    orig = np.abs(np.degrees(err)).mean(axis=0)
    swap = np.asarray(swap_scores)
    arm = slice(3, 17)
    print("\n===== 判定 =====")
    print(f"手臂整体平均误差: 原序 {orig[arm].mean():.1f}° / "
          f"左右互换 {swap[arm].mean():.1f}°"
          f"  -> {'⚠ 疑似左右换位!' if swap[arm].mean() < orig[arm].mean() * 0.7 else '无换位迹象'}")
    big_off = [(SONIC_NAMES[i], orig[i]) for i in range(17)
               if abs(np.degrees(err[:, i]).mean()) > 10]
    if big_off:
        print("⚠ 大偏移关节(疑似零位/符号问题): " +
              ", ".join(f"{n} {o:.0f}°" for n, o in big_off))
    else:
        print("零位偏移: 无(各关节均值偏移 <10°)")
    for i in range(3):
        c = np.degrees(cmd[:, i]); r = np.degrees(mq2[:, i])
        if c.max() - c.min() > 10:
            t = (r.max() - r.min()) / (c.max() - c.min())
            print(f"腰跟踪: {SONIC_NAMES[i]} 指令幅度 {c.max()-c.min():.0f}°, "
                  f"实测 {r.max()-r.min():.0f}° ({t*100:.0f}%)"
                  + ("  ⚠ 未被跟踪" if t < 0.3 else ""))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(analyze(sys.argv[1]))
