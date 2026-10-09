#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""增益一致性验收: mpc_gain_dump(C++) 输出 vs mpc_core.PreviewMPC(Python)
================================================================
用法:
  ./build/openarm_mpc_controller/mpc_gain_dump > /tmp/k_cpp.txt
  /usr/bin/python3 check_gain_parity.py /tmp/k_cpp.txt
判定: 每关节 max|ΔK| < 1e-9 (闭式解经由不同 BLAS, 1e-12 量级为正常)。
"""

import sys

import numpy as np

from mpc_core import PreviewMPC


def main():
    if len(sys.argv) < 2:
        sys.exit("用法: check_gain_parity.py <mpc_gain_dump 输出文件>")
    cpp = {}
    header = {}
    with open(sys.argv[1], encoding="utf-8") as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "N":
                # 行格式: N <N> dt <dt> w_p <w_p> w_v <w_v> w_a <w_a>
                header = {"N": int(tok[1]), "dt": float(tok[3]),
                          "w_p": float(tok[5]), "w_v": float(tok[7]),
                          "w_a": float(tok[9])}
            elif tok[0] == "K":
                cpp[int(tok[1])] = np.array([float(v) for v in tok[2:]])
    if not cpp or not header:
        sys.exit("输入文件不含增益(须为 mpc_gain_dump 原始输出)")

    mpc = PreviewMPC(7, N=header["N"], dt=header["dt"], w_p=header["w_p"],
                     w_v=header["w_v"], w_a=header["w_a"])
    print("参数: N=%d dt=%g w=(%g,%g,%g)"
          % (header["N"], header["dt"], header["w_p"], header["w_v"],
             header["w_a"]))
    worst = 0.0
    for j in range(7):
        if j not in cpp:
            sys.exit("缺 J%d 增益行" % (j + 1))
        k_py = np.asarray(mpc.K[j]).ravel()
        k_cpp = cpp[j]
        if k_py.shape != k_cpp.shape:
            sys.exit("J%d 维度不符: py %s vs cpp %s"
                     % (j + 1, k_py.shape, k_cpp.shape))
        d = float(np.max(np.abs(k_py - k_cpp)))
        worst = max(worst, d)
        print("J%d: max|ΔK| = %.3e" % (j + 1, d))
    if worst < 1e-9:
        print("✔ 一致性通过 (worst %.3e < 1e-9)" % worst)
    else:
        sys.exit("✘ 一致性失败 (worst %.3e ≥ 1e-9)" % worst)


if __name__ == "__main__":
    main()
