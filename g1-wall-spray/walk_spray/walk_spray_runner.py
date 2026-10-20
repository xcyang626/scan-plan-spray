# -*- coding: utf-8 -*-
"""边走边喷实验 runner (walk-while-spraying, 2026-10-13)。

与原站立喷涂(groot_spray_runner)的唯一区别: 播放动作期间给 WBC 持续注入
向左(+y)行走速度 loco_cmd=[0, walk_v, 0](>0.05 自动切 walk 策略),
手臂/腰动作原样叠加, **不做任何世界系修正**:
  - 任务伺服(task_servo)关闭 —— 手臂就是规划关节角的原始注入;
  - 偏航保持保留 —— 只锁基座朝向(走直线的前提), 不修正手臂;
  - 播放结束速度归零, 机器人站住。

本文件是独立实验入口, 复用原 runner 的全部机制(import, 不复制不修改),
不污染原方案; 状态文件重定向到本目录, 不与工作站页面互串。

用法:
  ~/桌面/轨迹规划/GR00T-WholeBodyControl/.venv_sim_cpu/bin/python walk_spray_runner.py \
      --motion <stand段动作.npz> --walk-v 0.1 --headless \
      [--stand 6] [--duration 300] [--out log/walk_spray.npz]

效果分析(枪尖墙面覆盖/骨盆路径)见同目录 analyze_walk_spray.py。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXEC = os.path.join(os.path.dirname(_HERE), "scripts", "exec")
sys.path.insert(0, _EXEC)
import groot_spray_runner as G   # noqa: E402  (复用原机制, 不改其源码)

# 状态文件重定向: 不写工作站的 station/robot_status.json, 只落本目录
G.STATUS_F = os.path.join(_HERE, "status.json")


class WalkSprayController(G.SprayController):
    """播放期叠加左移速度; 其余行为与原 SprayController 完全一致。

    walk_mode:
      const —— 恒定 +walk_v (第一次实验);
      sync  —— 腕心横移方向同步: 手臂往左喷就往左走, 往右喷往右走,
               行间竖直过渡段停住 (2026-10-13 用户方案: 往返抵消不走远)。"""

    def __init__(self, inner, walk_v=0.1, walk_mode="const", **kw):
        kw.setdefault("task_servo", False)   # 实验口径: 无世界系手臂修正
        super().__init__(inner, **kw)
        self.walk_v = float(walk_v)
        self.walk_mode = walk_mode
        self.stroke_sign = (self._precompute_stroke_signs(self.motion)
                            if (walk_mode == "sync" and self.motion is not None)
                            else None)
        if self.stroke_sign is not None:
            n_pos = int((self.stroke_sign > 0.5).sum())
            n_neg = int((self.stroke_sign < -0.5).sum())
            print(f"[walk-spray] sync 模式: 腕横移方向序列就绪 "
                  f"(+左 {n_pos} 帧 / -右 {n_neg} 帧 / 停 {len(self.stroke_sign)-n_pos-n_neg} 帧)",
                  flush=True)

    @staticmethod
    def _precompute_stroke_signs(motion):
        """每帧腕心(基座系)y 横移方向: +1/-1/0(行间过渡或停)。
        5 帧采样 FK -> 数值微分 -> 平滑 -> 滞回三态化 -> 插值回每帧。"""
        from arm_task_servo import fk_jac1
        idx = np.arange(0, len(motion), 5)
        ys = []
        for k in idx:
            q10 = np.concatenate([motion[k, 19:22], motion[k, 29:36]])
            p, _, _, _ = fk_jac1(q10)
            ys.append(float(p[1]))
        ys = np.array(ys)
        v = np.gradient(ys) * (30.0 / 5.0)          # m/s (30Hz 折算)
        # 3 点中值平滑去毛刺
        vs = v.copy()
        for i in range(1, len(v) - 1):
            vs[i] = np.median(v[i - 1:i + 2])
        sign = np.zeros(len(vs))
        s = 0.0
        for i, vi in enumerate(vs):
            if vi > 0.015:
                s = 1.0
            elif vi < -0.015:
                s = -1.0
            elif abs(vi) < 0.008:
                s = 0.0
            sign[i] = s
        return np.interp(np.arange(len(motion)), idx, sign)

    def _walk_cmd_y(self):
        """当前拍该给的 loco_cmd[1]。"""
        if not self.playing or self.walk_v <= 0:
            return 0.0
        if self.walk_mode == "sync" and self.stroke_sign is not None:
            f = self.t_play() * G.TRAJ_FPS
            i = int(np.clip(f, 0, len(self.stroke_sign) - 1))
            return float(self.walk_v * self.stroke_sign[i])
        return float(self.walk_v)

    def _rl_step(self):
        # 每个控制拍(50Hz)刷新 loco_cmd: 播放期=行走指令, 其余=0(防播完继续走)
        c = self.c
        with c.cmd_lock:
            c.control_dict["loco_cmd"][0] = 0.0
            c.control_dict["loco_cmd"][1] = self._walk_cmd_y()
        super()._rl_step()


def main():
    ap = argparse.ArgumentParser(description="边走边喷实验(站立动作 + 恒定左移)")
    ap.add_argument("--motion", required=True, help="动作 npz (qpos T×36 @30Hz, 建议站姿段)")
    ap.add_argument("--walk-v", type=float, default=0.1, help="左移速度 m/s (默认 0.1)")
    ap.add_argument("--walk-mode", choices=["const", "sync"], default="const",
                    help="const=恒定向左 | sync=跟随腕横移方向往返(不走远)")
    ap.add_argument("--stand", type=float, default=6.0, help="起播前站立秒数")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--out", default=os.path.join(_HERE, "log", "walk_spray.npz"),
                    help="跟踪日志输出 npz")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    inner, _ = G.build_controller(headless=a.headless)
    motion = G.load_motion(a.motion)
    ctl = WalkSprayController(inner, walk_v=a.walk_v, walk_mode=a.walk_mode,
                              motion=motion, stand_s=a.stand, log_path=a.out)
    print(f"[walk-spray] 边走边喷实验: {len(motion)} 帧 ({len(motion)/30:.1f}s), "
          f"模式 {a.walk_mode} 速度 {a.walk_v} m/s, 任务伺服=关, 偏航保持=开", flush=True)
    print(f"[walk-spray] 预计左移距离 ≈ {a.walk_v * len(motion)/30:.2f} m", flush=True)
    t0 = time.time()
    ctl.run(headless=a.headless, duration=a.duration)
    print(f"[walk-spray] 结束, 总耗时 {time.time()-t0:.1f}s, 日志 {a.out}", flush=True)
    # 骨盆漂移摘要(从最后一帧日志快速看)
    try:
        q = np.asarray(ctl.log and [x[1] for x in ctl.log] or [])
        if len(q):
            print(f"[walk-spray] 骨盆: y {q[0,1]:+.3f} -> {q[-1,1]:+.3f} m "
                  f"(漂移 {q[-1,1]-q[0,1]:+.3f} m), z_min {q[:,2].min():.3f} m",
                  flush=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
