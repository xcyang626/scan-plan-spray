#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只读检查各 ESC 电机的控制模式与限参寄存器(不使能、不运动)。

用法(control 层/标定节点/所有用 CAN 的程序都要先关闭):
  /usr/bin/python3 check_esc.py            # 只读检查
  /usr/bin/python3 check_esc.py --fix      # 把非 MIT 模式的电机写回 MIT 模式
"""
import os
import sys
import time

# dm_canfd 在标定包源码目录下, 以本脚本为锚点相对定位(项目整体移动后仍可用)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "openarm_calibration", "openarm_calibration"))
import dm_canfd as dm  # noqa: E402

FIX = "--fix" in sys.argv
# 打开 can0(需 root); 第二个参数 True = CAN-FD 模式
bus = dm.CanFdBus("can0", True)


def query(esc_id, rid, wait=0.15):
    """查询单个 ESC 的参数寄存器(rid), 返回参数值; 超时返回 None。"""
    cid, data = dm.build_query_param(esc_id, rid)
    bus.send(cid, data)
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        got = bus.recv(0.005)
        if got and got[0] == esc_id + 0x10:
            parsed = dm.parse_param(got[1])
            if parsed and parsed[0] == rid:
                return parsed[1]
    return None


def write_param(esc_id, rid, value):
    """写单个 ESC 的参数寄存器(--fix 模式专用)。"""
    cid, data = dm.build_write_param(esc_id, rid, value)
    bus.send(cid, data)
    time.sleep(0.05)


print("ESC  CTRL_MODE  PMAX   VMAX   TMAX")
results = {}
# 逐个查询 ESC1-7(夹爪 ESC8 不参与控制, 不查)
for esc in range(1, 8):
    mode = query(esc, dm.RID_CTRL_MODE)
    p = query(esc, dm.RID_PMAX)
    v = query(esc, dm.RID_VMAX)
    t = query(esc, dm.RID_TMAX)
    results[esc] = (mode, p, v, t)
    mark = ""
    if mode is not None and mode != dm.CTRL_MODE_MIT:
        mark = "  ← 非 MIT!"
    print(" %d     %s        %s   %s   %s%s"
          % (esc, mode, p, v, t, mark))

if FIX:
    print("\n--fix: 将所有非 MIT 电机写回 MIT 模式...")
    for esc, (mode, *_rest) in results.items():
        if mode is not None and mode != dm.CTRL_MODE_MIT:
            write_param(esc, dm.RID_CTRL_MODE, dm.CTRL_MODE_MIT)
            print("  ESC %d → MIT" % esc)
    print("完成。注意: 部分固件需要断电重上电后模式才完全生效。")
else:
    print("\n若有非 MIT 通道, 可运行: /usr/bin/python3 check_esc.py --fix")
