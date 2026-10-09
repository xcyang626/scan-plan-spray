#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tune_metrics.py —— PD 整定测量工具(黑匣子 100Hz CSV)
======================================================
子命令:
  ring   <csv> --joint N --t0 S --t1 S      阶跃 ring-down → 估计 f_n 与 ζ
  ripple <csv> --t0 S --t1 S [--joints 1,2] 慢速速度波动占比(需 --speed)
  droop  <rl_exec.csv>                      静态/慢速下垂(用 q vs qref)

用法示例(见 docs/PD参数整定流程.md §5-§6):
  python tune_metrics.py ring  blackbox_0914_1039.csv --joint 2 --t0 100 --t1 103
  python tune_metrics.py ripple blackbox_0914_1039.csv --t0 1 --t1 11 --speed 0.11
  python tune_metrics.py droop rl_exec_20260914_103648.csv
"""

import argparse
import math
import sys

import numpy as np


def load(csv_path, align_csv=None):
    """读黑匣子 CSV。align_csv 给出时, 用其文件名里的起始时刻做墙钟对齐
    (t=0 对齐到那次执行的起点) —— 否则 t=0 是文件首帧, 窗口会错位。"""
    B = np.genfromtxt(csv_path, delimiter=',', usecols=(1, 2)
                      + tuple(range(3, 23)))
    t = B[:, 0] + B[:, 1] / 1e9
    if align_csv:
        import os
        import time as _time
        base = os.path.basename(align_csv).replace('rl_exec_', '')
        date_s, hms = base.split('_')
        h2 = hms.split('.')[0]
        t0 = _time.mktime(_time.strptime(
            "%s-%s-%s %s:%s:%s" % (date_s[0:4], date_s[4:6], date_s[6:8],
                                   h2[0:2], h2[2:4], h2[4:6]),
            "%Y-%m-%d %H:%M:%S"))
        t = t - t0
    else:
        t = t - t[0]
    return t, B[:, 2:9], B[:, 9:16], B[:, 16:23]


def cmd_ring(args):
    t, Q, QD, _ = load(args.csv, args.align)
    m = (t >= args.t0) & (t <= args.t1)
    x = Q[m, args.joint - 1]
    x = x - x.mean()
    dt = float(np.median(np.diff(t[m])))
    # 过零峰法: 找相邻同向峰对, 用对数衰减率估 ζ
    peaks = []
    for i in range(1, len(x) - 1):
        if x[i] > x[i - 1] and x[i] >= x[i + 1] and abs(x[i]) > x.std() * 0.5:
            peaks.append((i, x[i]))
    if len(peaks) < 3:
        print("[ring] 峰不足(%d) —— 微动幅度加大或窗口调准" % len(peaks))
        return
    # 取主振荡方向的连续峰
    p0, p1 = peaks[0], peaks[1]
    n_period = (p1[0] - p0[0]) * dt
    fn = 1.0 / (2 * n_period) if False else 1.0 / n_period / 2.0
    # 对数衰减率: 相邻同号峰幅值比(自动符号配对)
    same = [(peaks[i], peaks[i + 2]) for i in range(len(peaks) - 2)
            if np.sign(peaks[i][1]) == np.sign(peaks[i + 2][1])]
    deltas = [math.log(abs(a[1] / b[1])) for a, b in same
              if abs(b[1]) > 1e-9 and abs(a[1]) > abs(b[1])]
    if not deltas:
        print("[ring] 无法配峰(可能过阻尼) —— ζ 已 ≥0.7, 无需再加 kd")
        return
    delta = float(np.median(deltas)) * (n_period / (2 * n_period) * 2) \
        if False else float(np.median(deltas))
    zeta = delta / math.sqrt(4 * math.pi ** 2 + delta ** 2)
    print("[ring] J%d: f_osc ≈ %.2f Hz, δ=%.2f → ζ ≈ %.2f "
          "(目标 0.45-0.70; 峰数 %d)" % (args.joint, fn, delta, zeta,
                                         len(peaks)))


def cmd_ripple(args):
    t, Q, QD, _ = load(args.csv, args.align)
    m = (t >= args.t0) & (t <= args.t1)
    dt = float(np.median(np.diff(t[m])))
    qd = np.gradient(Q[m], t[m], axis=0)
    joints = ([int(j) - 1 for j in args.joints.split(',')]
              if args.joints else range(7))
    w = max(int(1.5 / dt), 5)
    print("[ripple] 窗口 %.1f-%.1fs, 参考速度 %.2f rad/s" % (
        args.t0, args.t1, args.speed))
    for j in joints:
        res = qd[:, j] - np.convolve(qd[:, j], np.ones(w) / w, mode='same')
        rms = float(res.std())
        print("  J%d: 残余速度 RMS %.4f rad/s, 占参考 %.0f%% "
              "(门限 30%%)" % (j + 1, rms, 100 * rms / max(args.speed, 0.01)))


def cmd_droop(args):
    rows = np.genfromtxt(args.csv, delimiter=',', names=True)
    t = rows['t']
    E = np.degrees(np.stack([rows['q%d' % j] - rows['qref%d' % j]
                             for j in range(1, 8)], axis=1))
    w = max(int(2.0 / 0.1), 5)
    print("[droop] 按关节的低频(2s 均值)误差 = 下垂/滞后分量 (门限 J2≤2° 其余≤1.5°):")
    for j in range(7):
        ma = np.convolve(E[:, j], np.ones(w) / w, mode='same')
        print("  J%d: mean %.2f° | 低频 mean %.2f°, 范围 [%.1f, %.1f] | "
              "高频 std %.2f°" % (j + 1, E[:, j].mean(), ma.mean(),
                                  ma.min(), ma.max(),
                                  (E[:, j] - ma).std()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['ring', 'ripple', 'droop'])
    ap.add_argument('csv')
    ap.add_argument('--joint', type=int, default=2)
    ap.add_argument('--joints', default=None)
    ap.add_argument('--t0', type=float, default=0.0)
    ap.add_argument('--t1', type=float, default=None)
    ap.add_argument('--speed', type=float, default=0.1)
    ap.add_argument('--align', default=None,
                    help='rl_exec CSV 文件名 → 按其起始时刻墙钟对齐')
    args = ap.parse_args()
    if args.cmd == 'ring':
        cmd_ring(args)
    elif args.cmd == 'ripple':
        cmd_ripple(args)
    else:
        cmd_droop(args)


if __name__ == '__main__':
    sys.exit(main())
