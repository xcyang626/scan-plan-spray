# -*- coding: utf-8 -*-
"""
终端交互控制台（阶段2）
========================
在终端里直接操作研究臂（openarm_right_*）:
  - **数值与限位 = zero 文件原值（电机反馈系）**，与标定文件逐位一致，
    内部自动换算到 URDF 系驱动 RViz；
  - 每次操作后实时打印全部关节角 + TCP 位置（ROS 线程刷新）；
  - 命令简单直观，终端即控制台。

命令:
  1..7 / j1..j7   选择关节          w / +   正向步进
  s / -           负向步进          = <deg> 绝对设定当前关节
  step <deg>      修改步长(默认5°)  0       当前关节回 0
  z               全部回零位(=zero文件零位)   g <mm>  夹爪开合
  q               退出
"""

import math
import sys

D2R = math.pi / 180.0
R2D = 180.0 / math.pi


class TerminalConsole:

    def __init__(self, soft_limits_deg, on_change,
                 get_tcp_text=lambda: "TCP: --", step_deg=5.0,
                 initial_deg=None, home_deg=None):
        """soft_limits_deg: [(min°,max°)]×7（zero 文件原值，电机系）；
        initial_deg: 初始关节角（度，缺省全 0 夹限位内）；
        home_deg: "z 回零位"的目标（度，= zero 文件 zero_position）；
        on_change() 每次变更后回调；get_tcp_text(): 最新 TCP 描述。"""
        self.limits = soft_limits_deg
        self.on_change = on_change
        self.get_tcp_text = get_tcp_text
        self.step = step_deg
        self.sel = 0                          # 当前选中关节 (0..6)
        if initial_deg is not None:
            self.q_deg = [float(v) for v in initial_deg]
        else:
            self.q_deg = []
            for lo, hi in soft_limits_deg:    # 初始: 0 在限位内取 0，否则取区间中点
                self.q_deg.append(0.0 if lo <= 0.0 <= hi
                                  else round(0.5 * (lo + hi), 1))
        for i in range(len(self.q_deg)):      # 初始值同样夹进限位
            self._clamp(i)
        self.home_deg = ([float(v) for v in home_deg]
                         if home_deg is not None else [0.0] * 7)
        self.finger_mm = 0.0

    # ------------------------------------------------------------------

    def _clamp(self, idx):
        lo, hi = self.limits[idx]
        self.q_deg[idx] = min(hi, max(lo, self.q_deg[idx]))

    def _print_state(self):
        parts = []
        for i, q in enumerate(self.q_deg):
            mark = ">" if i == self.sel else " "
            parts.append("%sJ%d[%+7.1f]" % (mark, i + 1, q))
        lo, hi = self.limits[self.sel]
        print("  ".join(parts))
        print("  夹爪[%5.1fmm] 选中 J%d 限位[%+.1f°, %+.1f°] 步长%.1f°  %s"
              % (self.finger_mm, self.sel + 1, lo, hi, self.step,
                 self.get_tcp_text()))

    def _changed(self):
        self._print_state()
        if self.on_change:
            self.on_change()

    # ------------------------------------------------------------------

    def q_rad(self):
        return [q * D2R for q in self.q_deg]

    def finger_m(self):
        return self.finger_mm / 1000.0

    def run(self):
        print("=" * 78)
        print("OpenArm 终端控制台（数值与限位 = zero 文件原值，电机反馈系）")
        print("命令: 1..7 选关节 | w/+ 正步进 | s/- 负步进 | = <deg> 绝对设定")
        print("      step <deg> 改步长 | 0 当前关节回零 | z 全部回零 | g <mm> 夹爪 | q 退出")
        print("=" * 78)
        self._changed()
        while True:
            try:
                line = input("cmd> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                self._changed()
                continue
            low = line.lower()
            if low in ("q", "quit", "exit"):
                break
            try:
                if low.startswith("j") and low[1:].isdigit():
                    self.sel = int(low[1:]) - 1
                elif low.isdigit() and 1 <= int(low) <= 7:
                    self.sel = int(low) - 1
                elif low in ("w", "+"):
                    self.q_deg[self.sel] += self.step
                    self._clamp(self.sel)
                elif low in ("s", "-"):
                    self.q_deg[self.sel] -= self.step
                    self._clamp(self.sel)
                elif low.startswith("=") or low.startswith("go "):
                    val = float(line.split()[1])
                    self.q_deg[self.sel] = val
                    self._clamp(self.sel)
                elif low.startswith("step"):
                    self.step = abs(float(line.split()[1]))
                elif low == "0":
                    self.q_deg[self.sel] = 0.0
                    self._clamp(self.sel)
                elif low == "z":
                    self.q_deg = [float(v) for v in self.home_deg]
                    self.finger_mm = 0.0
                elif low.startswith("g"):
                    self.finger_mm = min(44.0,
                                         max(0.0, float(line.split()[1])))
                else:
                    print("未知命令: %s" % line)
                    continue
            except (IndexError, ValueError):
                print("命令格式错误: %s" % line)
                continue
            self._changed()
        print("控制台退出")
