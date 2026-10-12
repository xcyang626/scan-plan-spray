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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_task_servo import (ArmTaskServo, JLO as C_JLO, JHI as C_JHI,   # noqa: E402
                            fk_jac1, precompute_motion_targets)

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATUS_F = os.path.join(BASE, "station", "robot_status.json")

# GR00T 仿真模型关节布局 (已验证与规划模型一致)
WAIST_QPOS = slice(19, 22)      # 腰 yaw/roll/pitch
RARM_QPOS = slice(29, 36)       # 右臂 7
NUM_ACTIONS = 15                # RL 控制: 腿12 + 腰3
ARM_TARGET_IDX = slice(7, 14)   # 手臂 PD 目标 14 维向量的后 7 维 = 右臂
TRAJ_FPS = 30.0
# 左臂镜像联动(V3 稳定性修复, 2026-10-12): 单右臂横甩的角动量反作用会把底座带转
# (实测 yaw 漂到 +130~165°, 落点/覆盖全毁). 左臂做右臂的矢状面镜像运动
# (绕x/z轴关节翻符号, 绕y同号), 反作用力矩一阶抵消. G1W_MIRROR_LEFT=0 关闭.
MIRROR_LEFT = os.environ.get("G1W_MIRROR_LEFT", "0") != "0"  # 默认关: 符号约定未推导正确(实测摔倒), 开启前需按URDF逐关节验证
MIRROR_SIGN = np.array([1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0])


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


def build_controller(headless: bool, gait: bool = False):
    """gait=False: 平衡控制器(legacy, 腿+腰 RL 平衡). gait=True: gait 变体
    (control_dict 带 height_cmd/rpy_cmd/freq_cmd 通道) —— 官方 gait 脚本有三个坑,
    此处从外部绕过, 不改官方仓库文件:
      1) load_config 引用模块级全局 CONFIG_PATH(只在官方 __main__ 里赋值) -> 构造前设置;
      2) load_onnx_policy 硬编码 cuda:0, CPU venv 直接崩 -> 无 CUDA 时替换 CPU 加载器;
      3) __init__ 无条件起 pynput 键盘线程 -> 换哑实现(本 runner 无键盘交互)."""
    if not gait:
        from decoupled_wbc.sim2mujoco.scripts.run_mujoco_gear_wbc import GearWbcController
        cfg = os.path.join(GROOT_ROOT, "decoupled_wbc", "sim2mujoco",
                           "resources", "robots", "g1")
        return GearWbcController(cfg, enable_keyboard=False), None
    import importlib
    mod = importlib.import_module(
        "decoupled_wbc.sim2mujoco.scripts.run_mujoco_gear_wbc_gait")
    cfg = os.path.join(GROOT_ROOT, "decoupled_wbc", "sim2mujoco",
                       "resources", "robots", "g1")
    mod.CONFIG_PATH = cfg                                 # 坑1: 模块级全局

    class _NoKb:                                          # 坑3: 哑键盘监听
        def __init__(self, *a, **k):
            pass

        def start(self):
            return self

    mod.pkb.Listener = _NoKb
    try:
        import torch
        _has_cuda = torch.cuda.is_available()
    except Exception:
        _has_cuda = False
    if not _has_cuda:                                     # 坑2: CPU 策略加载器
        def _cpu_policy(self, path):
            import onnxruntime as ort
            sess = ort.InferenceSession(path)

            def run(input_tensor):
                arr = input_tensor.cpu().numpy() if hasattr(input_tensor, "cpu") \
                    else np.asarray(input_tensor, np.float32)
                outs = sess.run(None, {sess.get_inputs()[0].name: arr.astype(np.float32)})
                return torch.tensor(outs[0])
            return run
        mod.GearWbcController.load_onnx_policy = _cpu_policy
    return mod.GearWbcController(cfg), mod


class SprayController:
    """包装 GR00T 控制器: 注入腰+右臂轨迹目标 (不修改官方仓库文件).

    V2 蹲姿支持(--legs motion): 播放期双腿 12 关节改为按动作 PD 跟踪(腰/右臂照旧
    覆盖), 整机跟踪规划姿态 -> 蹲姿段(骨盆0.63)可执行; 站立待命期仍由 RL 平衡.
    V2 实验(--gait): 换 gait 控制器, 播放期把动作骨盆高写入 height_cmd 通道,
    由策略下蹲(官方 gait 脚本观测拼装未验证, 实测为准)."""

    def __init__(self, inner, motion=None, stand_s=6.0, trigger=None,
                 arm_kp_scale=4.0, arm_kd_scale=6.0, waist_kp_scale=1.5,
                 waist_kd_scale=2.0, loop=False, log_path=None,
                 legs="policy", gait=False, height_track=False,
                 task_servo=True, yaw_hold=True, motion_path=None):
        self.c = inner
        self.motion = motion
        self.stand_s = float(stand_s)
        self.trigger = trigger
        self.loop = loop
        self.arm_kp_scale = float(arm_kp_scale)
        self.arm_kd_scale = float(arm_kd_scale)
        self.waist_kp_scale = float(waist_kp_scale)
        self.waist_kd_scale = float(waist_kd_scale)
        self.legs = legs                    # policy(legacy) | motion(V2 运动学腿跟踪)
        self.gait = gait
        self.height_track = height_track or gait   # V2: 动作骨盆高 -> height_cmd
        self.fallen = False                 # 播放期摔倒(骨盆塌陷)标志
        self.pz_min = 9.9                   # 播放期骨盆最低(蹲姿执行证据)
        self.pz_err_max = 0.0               # 播放期 |实际骨盆 - 动作骨盆| 最大值
        # V2.1d 任务空间伺服: 按实际底座位姿闭环补偿腰+臂注入(吃掉底座 yaw/pitch/xy
        # 漂移; 2026-10-10 诊断: 开环注入下蹲姿行落点低 0.3~0.47m, 墙面 0.34~0.61 漏喷)
        self.task_servo = task_servo and not gait
        self.yaw_hold = yaw_hold and not gait
        self.srv = ArmTaskServo()
        self.tgt_P = self.tgt_A = self.tgt_H = None   # 动作帧规划世界系目标(预算)
        self._fall_z = 0.35                            # 摔倒阈值(_prep_targets 按动作自适应)
        self.playing = False
        self.t_play0 = None
        self.trig_speed = 1.0      # 触发文件携带的速度倍率(安全限幅 [0.05,1.5])
        self.done = False
        self.runs = 0
        self._trig_fp = None
        self._orig_pd = inner.pd_control
        inner.pd_control = self._pd
        # 站立阶段: 记录 RL 输出的腰部目标, 不覆盖
        self.freeze_waist = np.zeros(3)
        self.log_path = log_path
        self.log = []          # 播放期: (t_play, qpos, cmd_q10)
        self._motion_path = motion_path
        self._prep_targets()   # __init__ 路径(自动模式)预算目标; 触发模式在装载时再算

    def _prep_targets(self):
        """预算动作每帧的规划世界系 (腕心p*, 枪轴a*, 落点h*) + 动作感知摔倒阈值."""
        if self.motion is None:
            self.tgt_P = self.tgt_A = self.tgt_H = None
            return
        wall_x = float(os.environ.get("G1W_D_ROBOT", "0") or 0.74)
        if not os.environ.get("G1W_D_ROBOT"):       # 站位取动作所在 run 的 meta(更准)
            try:
                _meta = os.path.join(os.path.dirname(os.path.dirname(
                    os.path.dirname(self._motion_path))), "trajectory_meta.json")
                wall_x = float(json.load(open(_meta, encoding="utf8"))
                               .get("d_robot_nominal", wall_x))
            except Exception:
                pass
        self.tgt_P, self.tgt_A, self.tgt_H = precompute_motion_targets(
            self.motion, wall_x=wall_x)
        # 摔倒阈值动作感知: 深蹲动作(骨盆参考最低 0.36)下固定 0.35 只剩 9mm 余量会误报
        self._fall_z = min(0.35, float(self.motion[:, 2].min()) - 0.08)

    def _q10_offset(self):
        """任务伺服补偿量(未播放/未启用时为 0): 快环 dq + 慢环下垂微调 dq_slow."""
        if not (self.task_servo and self.playing):
            return np.zeros(10)
        return self.srv.dq + self.srv.dq_slow

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
                off = self._q10_offset()[3:10]
                tgt = np.zeros(n_arm, dtype=np.float64)
                rt = np.clip(                                     # 右臂+伺服补偿, 限位内
                    s[RARM_QPOS] + off, C_JLO[3:], C_JHI[3:])
                tgt[ARM_TARGET_IDX] = rt
                tdq = np.zeros(n_arm, dtype=np.float64)
                rv = v[RARM_QPOS]
                tdq[ARM_TARGET_IDX] = rv                          # 右臂速度前馈
                if MIRROR_LEFT:                                   # 左臂镜像联动
                    tgt[:7] = rt * MIRROR_SIGN
                    tdq[:7] = rv * MIRROR_SIGN
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
                off = self._q10_offset()[:3]
                t[12:15] = np.clip(                             # 腰+伺服补偿, 限位内
                    s[WAIST_QPOS] + off, C_JLO[:3], C_JHI[:3])
                tdq[12:15] = v[WAIST_QPOS]                         # 腰速度前馈
                if self.legs == "motion":                          # V2: 双腿按动作跟踪
                    t[0:12] = s[7:19]                              # qpos[7:19]=左右腿各6
                    tdq[0:12] = v[7:19]
                if isinstance(kpw, np.ndarray):
                    kpw[12:15] = kpw[12:15] * self.waist_kp_scale
                    kdw[12:15] = kdw[12:15] * self.waist_kd_scale
            else:
                t[12:15] = self.freeze_waist                       # 站立: 用 RL 原值
            return self._orig_pd(t.astype(np.float32), q, kpw,
                                 tdq.astype(np.float32), dq, kdw)
        return self._orig_pd(target_q, q, kp, target_dq, dq, kd)

    def _rl_step(self):
        """RL 推理一步. V2 --height: 播放期把动作骨盆高写进 control_dict['height_cmd']
        (平衡策略观测 command[3], 官方键盘 1/2 即此通道), 待命期复位 yaml 缺省.
        gait 变体走 torch 张量观测(官方 gait run 循环口径); 平衡控制器吃 numpy."""
        c = self.c
        if self.height_track:
            with c.cmd_lock:
                if self.playing:
                    s_ = self.sample(self.t_play())
                    if s_ is not None:
                        ref = float(np.clip(s_[2], 0.30, 0.85))
                        # 外环纠偏: 策略带臂负载时蹲不到指令深度(实测指令0.42只到0.53),
                        # 用骨盆实测反馈按比例加深指令, 纠偏限幅±0.10 防过驱
                        pz_m = float(c.data.qpos[2])
                        corr = min(max(1.8 * (ref - pz_m), -0.10), 0.10)
                        c.control_dict["height_cmd"] = float(
                            np.clip(ref + corr, 0.28, 0.85))
                else:
                    c.control_dict["height_cmd"] = float(c.config["height_cmd"])
        if self.gait:
            import torch
            with c.cmd_lock:
                cur = c.control_dict
            single_obs, _ = c.compute_observation(c.data, c.config, c.action,
                                                  cur, c.n_joints)
            c.obs_history.append(single_obs)
            for i, ho in enumerate(c.obs_history):
                c.obs[i * c.single_obs_dim:(i + 1) * c.single_obs_dim] = ho
            out = c.policy(torch.from_numpy(c.obs).unsqueeze(0))
            try:
                out = out.cpu().detach().numpy()
            except AttributeError:
                out = np.asarray(out)
            c.action = out.squeeze(0)
        else:
            with c.cmd_lock:
                cur = c.control_dict
            single_obs, _ = c.compute_observation(c.data, c.config, c.action,
                                                  cur, c.n_joints)
            c.obs_history.append(single_obs)
            for i, ho in enumerate(c.obs_history):
                c.obs[i * c.single_obs_dim:(i + 1) * c.single_obs_dim] = ho
            if np.linalg.norm(np.array(cur["loco_cmd"])) <= 0.05:
                c.action = c.policy(c.obs).squeeze(0)
            else:
                c.action = c.walk_policy(c.obs).squeeze(0)
        c.target_dof_pos = (c.action * c.config["action_scale"]
                            + c.config["default_angles"])
        # ---- 偏航保持: 默认关. 实测(2026-10-10)平衡策略对 loco_cmd[2] 小速率指令
        # 不稳定(正反馈, 机器人被转到 +110~127°), 仅任务伺服时 0-40s 落点中位差
        # 2.5~5cm; 朝向锁定留给 V2.2 对墙对齐模块(官方 delta_heading 语义). ----
        # ---- 任务空间伺服步(50Hz): 按实测底座位姿 DLS 补偿腰+臂 ----
        if self.task_servo and self.playing and self.motion is not None \
                and self.tgt_P is not None:
            self._task_servo_step()
        if self.yaw_hold:
            # 符号实验确认(2026-10-12): +0.04 指令 -> Δyaw +4.1°/6s, 方向正确;
            # v5 版失控是"yaw保持 + 未禁腰yaw的伺服"双重作用, v6 伺服已禁腰yaw,
            # 重新默认开启. 限幅 0.045 < 0.05(官方阈值, 超过切 walk 策略, 喷涂中禁切)
            q = c.data.qpos[3:7]
            yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]),
                             1 - 2 * (q[2] ** 2 + q[3] ** 2))
            with c.cmd_lock:
                c.control_dict["loco_cmd"][2] = float(
                    np.clip(-2.0 * yaw, -0.045, 0.045)) \
                    if abs(yaw) > np.deg2rad(2.0) else 0.0

    def _task_servo_step(self):
        """50Hz: 目标=动作帧规划世界系 FK(预算插值), 实测=当前 qpos 的底座+腰臂 FK."""
        c = self.c
        f = self.t_play() * TRAJ_FPS
        i = int(np.clip(f, 0, len(self.motion) - 2))
        a = float(np.clip(f - i, 0.0, 1.0))
        q10_m = np.concatenate([
            (1 - a) * self.motion[i, 19:22] + a * self.motion[i + 1, 19:22],
            (1 - a) * self.motion[i, 29:36] + a * self.motion[i + 1, 29:36]])
        p_star = (1 - a) * self.tgt_P[i] + a * self.tgt_P[i + 1]
        a_star = (1 - a) * self.tgt_A[i] + a * self.tgt_A[i + 1]
        n = np.linalg.norm(a_star)
        if n < 1e-6:
            return
        a_star = a_star / n
        q10_meas = np.concatenate([c.data.qpos[19:22], c.data.qpos[29:36]])
        self.srv.step(q10_m, p_star, a_star, c.data.qpos[0:3], c.data.qpos[3:7],
                      q10_meas)   # q10_meas 只进慢速微调环(0.5s低增益, 收PD下垂)

    def t_play(self):
        if self.t_play0 is None:
            return 0.0
        return (time.time() - self.t_play0) * self.trig_speed

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
                        sp = info.get("speed", 1.0)
                        try:
                            sp = float(sp)
                        except (TypeError, ValueError):
                            sp = 1.0
                        # 与站点/daemon 同口径的安全限幅; NaN/inf 归一为实时
                        self.trig_speed = (min(max(sp, 0.05), 1.5)
                                           if np.isfinite(sp) else 1.0)
                        self.motion = load_motion(mp)
                        self._trig_fp = raw          # 消费指纹
                        self._motion_path = mp
                        self.pz_min, self.pz_err_max, self.fallen = 9.9, 0.0, False
                        self._prep_targets()
                        self.srv.reset()
                        if (self.motion[:, 2].min() < 0.72 and self.legs == "policy"
                                and not self.height_track):
                            print("[spray] ⚠ 触发的动作含蹲姿段但本进程未启用蹲姿通道"
                                  "(缺 --height / --legs motion) — 蹲姿段骨盆不会下降",
                                  flush=True)
                        self.playing = True
                        self.t_play0 = time.time()
                        print(f"[spray] 收到触发, 播放 {mp} ({len(self.motion)} 帧, "
                              f"速度×{self.trig_speed})", flush=True)
                        write_status(phase="playing", motion=mp,
                                     frames=int(len(self.motion)))
                except Exception as e:
                    print(f"[spray] 触发解析失败: {e}", flush=True)
        elif el >= self.stand_s:
            self.playing = True
            self.t_play0 = time.time()
            self.srv.reset()                        # 自动模式起播: 伺服补偿清零
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

                # RL 推理 (与官方一致; gait 模式内含 height_cmd 注入)
                if c.counter % c.config["control_decimation"] == 0:
                    self._rl_step()
                    if not self.playing:
                        self.freeze_waist = np.array(c.target_dof_pos[12:15], copy=True)

                # 播放期蹲姿执行指标 + 摔倒检测(骨盆塌陷到 0.35m 以下视为失控)
                if self.playing:
                    pz_now = float(c.data.qpos[2])
                    self.pz_min = min(self.pz_min, pz_now)
                    s_ = self.sample(self.t_play())
                    if s_ is not None:
                        self.pz_err_max = max(self.pz_err_max, abs(pz_now - float(s_[2])))
                    if pz_now < self._fall_z or not np.isfinite(pz_now):
                        self.fallen = True
                        print(f"[spray] ⚠ 检测到摔倒/失控 (骨盆 z={pz_now:.3f}), 停止播放",
                              flush=True)
                        write_status(phase="fallen", pelvis=round(pz_now, 3))
                        self.playing = False
                        break

                # 跟踪日志 (播放期; cmd 记录修正后命令, 便于事后算落点误差)
                if self.log_path and self.playing:
                    s_ = self.sample(self.t_play())
                    if s_ is not None:
                        self.log.append((self.t_play(), c.data.qpos.copy(),
                                         np.concatenate([s_[WAIST_QPOS],
                                                         s_[RARM_QPOS]])
                                         + self._q10_offset()))

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
                            imp = self._impact_stats()
                            print("[spray] 轨迹播放完成, 回到待命", flush=True)
                            if imp:
                                print(f"[spray] 落点误差(实际-规划, 含底座漂移): "
                                      f"mean {imp['mean']:+.3f}m, p95 {imp['p95']:.3f}m, "
                                      f"max {imp['max']:.3f}m (n={imp['n']}帧; "
                                      f"伺服 {self.task_servo})", flush=True)
                            print(f"[spray] 蹲姿执行摘要: 骨盆最低 "
                                  f"{self.pz_min if self.pz_min < 9 else float('nan'):.3f}m, "
                                  f"|骨盆-动作|最大 {self.pz_err_max:.3f}m, 摔倒 {self.fallen}",
                                  flush=True)
                            write_status(phase="standing", progress=1.0, done_runs=
                                         int(self.runs) + 1,
                                         pelvis_min=(round(self.pz_min, 3)
                                                     if self.pz_min < 9 else None),
                                         pelvis_err_max=round(self.pz_err_max, 3),
                                         fallen=self.fallen, impact_err=imp)
                            self.log = []           # 每轮统计后清空, 下一轮重新累计
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
        imp = self._impact_stats()
        print(f"[spray] 蹲姿执行摘要: 骨盆最低 {self.pz_min if self.pz_min < 9 else float('nan'):.3f}m, "
              f"|骨盆-动作|最大 {self.pz_err_max:.3f}m, 摔倒 {self.fallen} "
              f"(legs={self.legs}, gait={self.gait})", flush=True)
        if imp:
            print(f"[spray] 落点误差(实际-规划, 含底座漂移): mean {imp['mean']:+.3f}m, "
                  f"p95 {imp['p95']:.3f}m, max {imp['max']:.3f}m (n={imp['n']}帧; "
                  f"伺服 {self.task_servo}, 偏航保持 {self.yaw_hold})", flush=True)
        write_status(phase=("fallen" if self.fallen else
                            ("done" if self.done else "standing")),
                     pelvis_min=round(self.pz_min, 3) if self.pz_min < 9 else None,
                     pelvis_err_max=round(self.pz_err_max, 3), fallen=self.fallen,
                     impact_err=imp)
        return self.done

    def _impact_stats(self):
        """运行末从跟踪日志算 落点高度误差(实际-规划): 任务空间的最终指标.
        实际落点 = 实测qpos(底座+腰臂) FK 射线交墙; 规划落点 = 预算 tgt_H 插值."""
        if not self.log or self.tgt_H is None:
            return None
        wall_x = float(os.environ.get("G1W_D_ROBOT", "0") or 0.74)
        ts = np.array([x[0] for x in self.log])
        qp = np.array([x[1] for x in self.log])
        f = ts * TRAJ_FPS
        i0 = np.clip(np.floor(f).astype(int), 0, len(self.tgt_H) - 2)
        a = np.clip(f - i0, 0.0, 1.0)
        h_star = (1 - a) * self.tgt_H[i0] + a * self.tgt_H[i0 + 1]
        from arm_task_servo import quat2rotm
        dh = []
        for k in range(0, len(qp), 3):            # 抽稀(60Hz)加速统计
            if not np.isfinite(h_star[k]):
                continue
            q10 = np.concatenate([qp[k, 19:22], qp[k, 29:36]])
            p_rel, a_rel, _, _ = fk_jac1(q10)
            Rb = quat2rotm(qp[k, 3:7])
            p_w = Rb @ p_rel + qp[k, 0:3]
            a_w = Rb @ a_rel
            if a_w[0] > 0.3:
                dh.append(p_w[2] + (wall_x - p_w[0]) * a_w[2] / a_w[0] - h_star[k])
        if len(dh) < 50:
            return None
        dh = np.array(dh)
        return dict(n=int(len(dh)), mean=round(float(dh.mean()), 4),
                    p95=round(float(np.percentile(np.abs(dh), 95)), 4),
                    max=round(float(np.abs(dh).max()), 4))


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
    ap.add_argument("--legs", choices=("policy", "motion"), default="policy",
                    help="播放期腿驱动: policy=RL平衡(legacy) / "
                         "motion=按动作PD跟踪(运动学蹲姿, 起播有跳变风险, 备选)")
    ap.add_argument("--height", action="store_true",
                    help="V2 蹲姿主通道: 播放期动作骨盆高写入 height_cmd, "
                         "平衡策略下蹲(官方键盘 1/2 同通道)")
    ap.add_argument("--no-task-servo", action="store_true",
                    help="关闭任务空间伺服(按实际底座位姿闭环补偿腰+臂; 默认开)")
    ap.add_argument("--no-yaw-hold", action="store_true",
                    help="关闭底座偏航保持(默认开; 符号实验已验证, 限幅<0.05不切walk)")
    ap.add_argument("--gait", action="store_true",
                    help="V2实验: 用 gait 控制器 + height_cmd(观测拼装未验证)")
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
        print(f"[spray] 动作 {a.motion}: {motion.shape}"
              + (f" (含蹲姿段, 骨盆最低 {motion[:, 2].min():.3f}m)"
                 if motion[:, 2].min() < 0.72 else ""), flush=True)
        if (motion[:, 2].min() < 0.72 and a.legs == "policy"
                and not a.height and not a.gait):
            print("[spray] ⚠ 该动作含蹲姿段但未启用蹲姿通道 —— 蹲姿段骨盆不会下降; "
                  "双姿态动作请加 --height(策略下蹲)或 --legs motion(运动学跟踪)", flush=True)
    inner, _ = build_controller(a.headless, gait=a.gait)
    sc = SprayController(inner, motion=motion, stand_s=a.stand, trigger=a.trigger,
                         arm_kp_scale=a.arm_kp_scale, arm_kd_scale=a.arm_kd_scale,
                         waist_kp_scale=a.waist_kp_scale, waist_kd_scale=a.waist_kd_scale,
                         loop=a.loop, log_path=a.log, legs=a.legs, gait=a.gait,
                         height_track=a.height,
                         task_servo=not a.no_task_servo, yaw_hold=not a.no_yaw_hold,
                         motion_path=a.motion)
    ok = sc.run(headless=a.headless, duration=a.duration)
    if a.headless:
        print("[spray] 自检结束, 播放完成" if ok else "[spray] 自检结束")


if __name__ == "__main__":
    main()
