# -*- coding: utf-8 -*-
"""
关节滑条控制面板（tkinter，主线程运行）
========================================
7 关节滑条（范围 = zero 文件软限位，URDF 系，度）+ 夹爪开合滑条（mm）+
回零按钮 + TCP 实时位置显示。滑条变化即通过 on_change 回调上报，
由 scene_node 在 ROS 线程发布 /urdf_joint_states，实现 RViz 实时联动。
"""

import math
import tkinter as tk

D2R = math.pi / 180.0
R2D = 180.0 / math.pi


class JointSliderPanel:

    def __init__(self, soft_limits_deg, finger_max_m=0.044, on_change=None,
                 initial_deg=None):
        """soft_limits_deg: [(min°,max°)] × 7（zero 文件原值，电机系）；
        initial_deg: 初始关节角（度，缺省 0 在限位内取 0 否则区间中点）；
        on_change() 滑条变动时回调。"""
        self.on_change = on_change
        self.finger_max_mm = finger_max_m * 1000.0
        if initial_deg is None:
            initial_deg = [0.0 if lo <= 0.0 <= hi else round(0.5 * (lo + hi), 1)
                           for lo, hi in soft_limits_deg]
        self.initial_deg = initial_deg

        self.root = tk.Tk()
        self.root.title("OpenArm 关节控制（限位 = zero 文件原值，电机系）")
        # 置顶防被 RViz 全屏窗口遮挡
        self.root.attributes("-topmost", True)
        self.root.lift()
        self.root.focus_force()

        self.scales = []
        self.labels = []
        for i, (lo, hi) in enumerate(soft_limits_deg):
            row = tk.Frame(self.root)
            row.pack(fill=tk.X, padx=10, pady=3)
            tk.Label(row, text="J%d" % (i + 1), width=3,
                     font=("TkDefaultFont", 11, "bold")).pack(side=tk.LEFT)
            s = tk.Scale(row, from_=round(lo, 1), to=round(hi, 1),
                         resolution=0.1, orient=tk.HORIZONTAL, length=430,
                         command=lambda _v, idx=i: self._on_slider(idx))
            s.set(round(float(initial_deg[i]), 1))
            s.pack(side=tk.LEFT, fill=tk.X, expand=True)
            val = tk.Label(row, text="%8.1f°" % s.get(), width=9,
                           font=("Courier", 10))
            val.pack(side=tk.LEFT)
            self.scales.append(s)
            self.labels.append(val)

        # 夹爪开合（mm）
        row = tk.Frame(self.root)
        row.pack(fill=tk.X, padx=10, pady=3)
        tk.Label(row, text="夹爪", width=3,
                 font=("TkDefaultFont", 11, "bold")).pack(side=tk.LEFT)
        self.finger_scale = tk.Scale(row, from_=0.0, to=self.finger_max_mm,
                                     resolution=0.5, orient=tk.HORIZONTAL,
                                     length=430,
                                     command=lambda _v: self._on_slider(-1))
        self.finger_scale.set(0.0)
        self.finger_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.finger_label = tk.Label(row, text="%6.1f mm" % 0.0, width=9,
                                     font=("Courier", 10))
        self.finger_label.pack(side=tk.LEFT)

        # 底部: 回零按钮 + TCP 读数
        bottom = tk.Frame(self.root)
        bottom.pack(fill=tk.X, padx=10, pady=6)
        tk.Button(bottom, text="全部回零", width=10,
                  command=self.reset_zero).pack(side=tk.LEFT)
        self.tcp_label = tk.Label(bottom, text="TCP: --", width=46,
                                  font=("Courier", 9), anchor="w")
        self.tcp_label.pack(side=tk.LEFT, padx=12)

        # TCP 读数轮询刷新（ROS 线程写入 self.tcp_text）
        self.tcp_text = "TCP: --"
        self.root.after(200, self._poll_tcp)

    # ------------------------------------------------------------------

    def _on_slider(self, idx):
        if idx >= 0:
            self.labels[idx].config(text="%8.1f°" % self.scales[idx].get())
        else:
            self.finger_label.config(
                text="%6.1f mm" % self.finger_scale.get())
        if self.on_change:
            self.on_change()

    def reset_zero(self):
        for i, s in enumerate(self.scales):
            s.set(round(float(self.initial_deg[i]), 1))
        self.finger_scale.set(0.0)
        if self.on_change:
            self.on_change()

    def _poll_tcp(self):
        self.tcp_label.config(text=self.tcp_text)
        self.root.after(200, self._poll_tcp)

    # ------------------------------------------------------------------

    def q_rad(self):
        """当前 7 关节角（rad）。"""
        return [s.get() * D2R for s in self.scales]

    def finger_m(self):
        """当前夹爪开合（m）。"""
        return self.finger_scale.get() / 1000.0

    def run(self):
        self.root.mainloop()
