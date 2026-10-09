# -*- coding: utf-8 -*-
"""GR00T-WBC 喷涂执行 runner (在 GR00T 官方 Decoupled WBC 仿真里跑我们的轨迹).

原理:
  GR00T 仿真 `GearWbcController` 中, 下肢 12 关节 + 腰 3 关节由 RL 策略平衡/驱动,
  双臂 14 关节为 PD 保持零位。我们把规划好的喷涂轨迹 (腰3 + 右臂7, 30Hz) 注入:
    - 腰 3 关节: 覆盖 RL 输出 (target_dof_pos[12:15])
    - 右臂 7 关节: 替换手臂 PD 目标 (14 维向量后 7 维)
  左臂保持零位, 下肢仍由 RL 平衡 —— 机器人在站姿平衡下, 腰+右臂精确跟随墙面轨迹。

已验证 (2026-10-08): GR00T 模型 g1_gear_wbc.xml 与规划模型关节顺序完全一致 (nq=36),
  同一组关节角复现设计喷枪位姿误差 <=4mm(腕心) / <=1°(枪轴), 故轨迹可直接注入。

用法:
  # 自动模式: 站立 T_STAND 秒后自动播放
  python groot_spray_runner.py --motion <motion.npz> [--stand 6] [--headless]

  # 触发模式(工作站用): 站立等待, 出现 trigger 文件后播放其中指定的动作
  python groot_spray_runner.py --trigger station/spray_trigger.json [--headless]

  # 无界面自检(不启动 viewer, 播放完整轨迹并输出统计)
  python groot_spray_runner.py --motion <motion.npz> --headless --stand 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

GROOT_ROOT = os.environ.get("GROOT_ROOT", "/home/yang/桌面/轨迹规划/GR00T-WholeBodyControl")
sys.path.insert(0, GROOT_ROOT)

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATUS_F = os.path.join(BASE, "station", "robot_status.json")

# GR00T 仿真模型关节布局 (已验证与规划模型一致)
WAIST_QPOS = slice(19, 22)      # 腰 yaw/roll/pitch
RARM_QPOS = slice(29, 36)       # 右臂 7
NUM_ACTIONS = 15                # RL 控制: 腿12 + 腰3
ARM_TARGET_IDX = slice(7, 14)   # 手臂 PD 目标 14 维向量的后 7 维 = 右臂
TRAJ_FPS = 30.0


def write_status(**kw):
    try:
        st = {"ts": time.time()}
        st.update(kw)
        with open(STATUS_F, "w", encoding="utf8") as f:
            json.dump(st, f, ensure_ascii=False)
    except Exception:
        pass


def load_motion(path):
    z = np.load(path)
    q = z["qpos"] if "qpos" in z.files else z[z.files[0]]
    q = np.asarray(q, dtype=np.float64)
    assert q.ndim == 2 and q.shape[1] == 36, f"动作维度异常: {q.shape}"
    return q


def build_controller(headless: bool):
    from decoupled_wbc.sim2mujoco.scripts.run_mujoco_gear_wbc import GearWbcController
    cfg = os.path.join(GROOT_ROOT, "decoupled_wbc", "sim2mujoco",
                       "resources", "robots", "g1")
    return GearWbcController(cfg, enable_keyboard=False), GearWbcController


class SprayController:
    """包装 GR00T 控制器: 注入腰+右臂轨迹目标 (不修改官方仓库文件)."""

    def __init__(self, inner, motion=None, stand_s=6.0, trigger=None,
                 arm_kp_scale=4.0, arm_kd_scale=6.0, waist_kp_scale=1.5,
                 waist_kd_scale=2.0, loop=False, log_path=None):
        self.c = inner
        self.motion = motion
        self.stand_s = float(stand_s)
        self.trigger = trigger
        self.loop = loop
        self.arm_kp_scale = float(arm_kp_scale)
        self.arm_kd_scale = float(arm_kd_scale)
        self.waist_kp_scale = float(waist_kp_scale)
        self.waist_kd_scale = float(waist_kd_scale)
        self.playing = False
        self.t_play0 = None
        self.done = False
        self.runs = 0
        self._trig_fp = None
        self._orig_pd = inner.pd_control
        inner.pd_control = self._pd
        # 站立阶段: 记录 RL 输出的腰部目标, 不覆盖
        self.freeze_waist = np.zeros(3)
        self.log_path = log_path
        self.log = []          # 播放期: (t_play, qpos, cmd_q10)

    # ---- 轨迹采样 (30Hz, 线性插值) ----
    def sample(self, t, with_vel=False):
        if self.motion is None:
            return None
        f = t * TRAJ_FPS
        i = int(np.floor(f))
        if i >= len(self.motion) - 1:
            i = len(self.motion) - 2
            a = 1.0
        else:
            a = f - i
        i = max(0, i)
        q = (1 - a) * self.motion[i] + a * self.motion[i + 1]
        if not with_vel:
            return q
        v = (self.motion[i + 1] - self.motion[i]) * TRAJ_FPS   # 速度前馈 (rad/s)
        return q, v

    # ---- 注入点: 拦截 PD 调用 ----
    def _pd(self, target_q, q, kp, target_dq, dq, kd):
        n_arm = self.c.n_joints - self.c.config["num_actions"]     # 14
        if target_q.shape[0] == n_arm:                             # 手臂调用
            sv = self.sample(self.t_play(), True) if self.playing else None
            kpa = kp * self.arm_kp_scale
            kda = kd * self.arm_kd_scale
            if sv is not None:
                s, v = sv
                tgt = np.zeros(n_arm, dtype=np.float64)
                tgt[ARM_TARGET_IDX] = s[RARM_QPOS]                 # 右臂位置目标
                tdq = np.zeros(n_arm, dtype=np.float64)
                tdq[ARM_TARGET_IDX] = v[RARM_QPOS]                 # 右臂速度前馈
                return self._orig_pd(tgt.astype(np.float32), q, kpa,
                                     tdq.astype(np.float32), dq, kda)
            return self._orig_pd(target_q, q, kpa, target_dq, dq, kda)
        if target_q.shape[0] == self.c.config["num_actions"]:      # 腿+腰调用
            t = np.array(target_q, dtype=np.float64, copy=True)
            tdq = np.array(target_dq, dtype=np.float64, copy=True)
            kpw = kp.copy() if isinstance(kp, np.ndarray) else kp
            kdw = kd.copy() if isinstance(kd, np.ndarray) else kd
            sv = self.sample(self.t_play(), True) if self.playing else None
            if sv is not None:
                s, v = sv
                t[12:15] = s[WAIST_QPOS]                           # 腰位置目标
                tdq[12:15] = v[WAIST_QPOS]                         # 腰速度前馈
                if isinstance(kpw, np.ndarray):
                    kpw[12:15] = kpw[12:15] * self.waist_kp_scale
                    kdw[12:15] = kdw[12:15] * self.waist_kd_scale
            else:
                t[12:15] = self.freeze_waist                       # 站立: 用 RL 原值
            return self._orig_pd(t.astype(np.float32), q, kpw,
                                 tdq.astype(np.float32), dq, kdw)
        return self._orig_pd(target_q, q, kp, target_dq, dq, kd)

    def t_play(self):
        if self.t_play0 is None:
            return 0.0
        return time.time() - self.t_play0

    # ---- 触发/计时 ----
    def maybe_start(self):
        # 触发模式下 motion 初始为 None(等触发文件给出), 不能因此提前返回
        if self.playing or (self.motion is None and not self.trigger):
            return
        el = time.time() - self.t0
        if self.trigger:
            if os.path.exists(self.trigger):
                try:
                    raw = open(self.trigger, encoding="utf8").read()
                    info = json.loads(raw)
                    mp = info.get("motion")
                    if mp and os.path.exists(mp):
                        self.motion = load_motion(mp)
                        self._trig_fp = raw          # 消费指纹
                        self.playing = True
                        self.t_play0 = time.time()
                        print(f"[spray] 收到触发, 播放 {mp} ({len(self.motion)} 帧)", flush=True)
                        write_status(phase="playing", motion=mp,
                                     frames=int(len(self.motion)))
                except Exception as e:
                    print(f"[spray] 触发解析失败: {e}", flush=True)
        elif el >= self.stand_s:
            self.playing = True
            self.t_play0 = time.time()
            print(f"[spray] 站立 {el:.1f}s 结束, 开始播放轨迹", flush=True)
            write_status(phase="playing", frames=int(len(self.motion)))

    def run(self, headless=False, duration=None):
        import mujoco
        import mujoco.viewer
        import contextlib

        self.t0 = time.time()
        c = self.c
        run_duration = duration if duration is not None else 1e9
        viewer_ctx = (contextlib.nullcontext(None) if headless
                      else mujoco.viewer.launch_passive(c.model, c.data))
        write_status(phase="standing", motion=None)
        print("[spray] 站立中 (RL 平衡), 等待执行…" if self.trigger
              else f"[spray] 站立 {self.stand_s}s 后播放", flush=True)
        with viewer_ctx as viewer:
            start = time.time()
            last_report = 0.0
            _closed_logged = False
            while time.time() - start < run_duration:
                # 窗口被关闭不终止任务: 转入无头继续(触发模式下机器人仍在运行)
                if viewer is not None and not viewer.is_running():
                    if not _closed_logged:
                        print("[spray] viewer 窗口已关闭, 转入无头继续运行", flush=True)
                        _closed_logged = True
                    viewer = None
                step_start = time.time()
                self.maybe_start()

                # 腿+腰 PD (RL 输出; 腰部被轨迹覆盖)
                leg_tau = self._pd(
                    c.target_dof_pos, c.data.qpos[7:7 + c.config["num_actions"]],
                    c.config["kps"], np.zeros_like(c.config["kps"]),
                    c.data.qvel[6:6 + c.config["num_actions"]], c.config["kds"])
                c.data.ctrl[:c.config["num_actions"]] = leg_tau

                # 双臂 PD (右臂被轨迹覆盖)
                arm_tau = self._pd(
                    np.zeros(c.n_joints - c.config["num_actions"], dtype=np.float32),
                    c.data.qpos[7 + c.config["num_actions"]:7 + c.n_joints],
                    np.full(c.n_joints - c.config["num_actions"], 100.0),
                    np.zeros(c.n_joints - c.config["num_actions"]),
                    c.data.qvel[6 + c.config["num_actions"]:6 + c.n_joints],
                    np.full(c.n_joints - c.config["num_actions"], 0.5))
                c.data.ctrl[c.config["num_actions"]:] = arm_tau

                mujoco.mj_step(c.model, c.data)
                c.counter += 1

                # RL 推理 (与官方一致)
                if c.counter % c.config["control_decimation"] == 0:
                    with c.cmd_lock:
                        cur = c.control_dict
                    single_obs, _ = c.compute_observation(
                        c.data, c.config, c.action, cur, c.n_joints)
                    c.obs_history.append(single_obs)
                    for i, ho in enumerate(c.obs_history):
                        c.obs[i * c.single_obs_dim:(i + 1) * c.single_obs_dim] = ho
                    if np.linalg.norm(np.array(cur["loco_cmd"])) <= 0.05:
                        c.action = c.policy(c.obs).squeeze(0)
                    else:
                        c.action = c.walk_policy(c.obs).squeeze(0)
                    c.target_dof_pos = (c.action * c.config["action_scale"]
                                        + c.config["default_angles"])
                    if not self.playing:
                        self.freeze_waist = np.array(c.target_dof_pos[12:15], copy=True)

                # 跟踪日志 (播放期)
                if self.log_path and self.playing:
                    s_ = self.sample(self.t_play())
                    if s_ is not None:
                        self.log.append((self.t_play(), c.data.qpos.copy(),
                                         np.concatenate([s_[WAIST_QPOS], s_[RARM_QPOS]])))

                # 状态上报 + 结束判定
                now = time.time()
                if now - last_report > 0.5:
                    last_report = now
                    ph = "playing" if self.playing else "standing"
                    prog = (self.t_play() / (len(self.motion) / TRAJ_FPS)
                            if (self.playing and self.motion is not None) else 0.0)
                    write_status(phase=ph, progress=round(float(min(prog, 1.0)), 3),
                                 pelvis=round(float(c.data.qpos[2]), 3))
                if self.playing and self.motion is not None:
                    if self.t_play() > len(self.motion) / TRAJ_FPS + 0.5:
                        if self.loop:
                            self.t_play0 = time.time()
                        elif self.trigger:
                            # 触发模式: 回到待命, 删触发文件, 等待下一次执行
                            print("[spray] 轨迹播放完成, 回到待命", flush=True)
                            write_status(phase="standing", progress=1.0, done_runs=
                                         int(self.runs) + 1)
                            self.runs += 1
                            self.playing = False
                            self.t_play0 = None
                            self.motion = None
                            try:
                                # 仅当触发文件未被新执行覆盖时才删除
                                if (not hasattr(self, "_trig_fp") or
                                        open(self.trigger, encoding="utf8").read() == self._trig_fp):
                                    os.remove(self.trigger)
                                else:
                                    print("[spray] 检测到新触发, 继续执行", flush=True)
                            except Exception:
                                pass
                        else:
                            print("[spray] 轨迹播放完成", flush=True)
                            write_status(phase="done", progress=1.0)
                            self.done = True
                            if headless:
                                break

                if viewer is not None:
                    viewer.sync()
                else:
                    time.sleep(max(0, c.model.opt.timestep - (time.time() - step_start)))
        if self.log_path and self.log:
            t = np.array([x[0] for x in self.log])
            q = np.array([x[1] for x in self.log])
            cq = np.array([x[2] for x in self.log])
            np.savez(self.log_path, t=t, qpos=q, cmd_q10=cq)
            print(f"[spray] 跟踪日志已存 {self.log_path} ({len(t)} 帧)", flush=True)
        return self.done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", default=None, help="动作 npz (qpos T×36 @30Hz)")
    ap.add_argument("--trigger", default=None, help="触发文件; 出现后按其中 motion 播放")
    ap.add_argument("--stand", type=float, default=6.0, help="站立等待秒数(自动模式)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--arm-kp-scale", type=float, default=4.0)
    ap.add_argument("--arm-kd-scale", type=float, default=6.0)
    ap.add_argument("--waist-kp-scale", type=float, default=1.5)
    ap.add_argument("--waist-kd-scale", type=float, default=2.0)
    ap.add_argument("--log", default=None, help="播放期跟踪日志 npz")
    a = ap.parse_args()
    if not a.motion and not a.trigger:
        ap.error("需要 --motion 或 --trigger")
    try:
        open(os.path.join(BASE, "station", "robot.pid"), "w").write(str(os.getpid()))
    except Exception:
        pass
    motion = load_motion(a.motion) if a.motion else None
    if motion is not None:
        print(f"[spray] 动作 {a.motion}: {motion.shape}", flush=True)
    inner, _ = build_controller(a.headless)
    sc = SprayController(inner, motion=motion, stand_s=a.stand, trigger=a.trigger,
                         arm_kp_scale=a.arm_kp_scale, arm_kd_scale=a.arm_kd_scale,
                         waist_kp_scale=a.waist_kp_scale, waist_kd_scale=a.waist_kd_scale,
                         loop=a.loop, log_path=a.log)
    ok = sc.run(headless=a.headless, duration=a.duration)
    if a.headless:
        print("[spray] 自检结束, 播放完成" if ok else "[spray] 自检结束")


if __name__ == "__main__":
    main()
