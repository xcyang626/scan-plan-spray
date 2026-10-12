# -*- coding: utf-8 -*-
"""GR00T 执行守护进程 —— 常驻的官方协议发布端（唯一 ZMQ 5556 持有者）。

为什么需要它（连接/控制稳定性修复，2026-10-09）：
  旧实现里每次 命令/动作流 都新建 PUB socket 并 bind 5556，存在三个问题：
    1) 并发 bind 冲突 —— 动作流执行中按【停止/急停】，命令子进程 bind 失败，
       命令丢失（robot.log 中两次 "Address already in use" 崩溃即此类争抢）；
    2) PUB/SUB 慢加入 —— 新 bind 的 PUB 在订阅方重连完成前发送的消息直接丢弃；
    3) 多个发送端生命周期混乱，残留进程占端口导致新部署起不来。
  守护进程全程独占 5556（部署 C++ 端是 SUB + connect 到本机），长连接保持：
  命令即时可达、动作流不与其他发送端抢端口、站点退出（stdin EOF）即释放端口。

通信（全部 JSON 行，UTF-8，一行一条）：
  A) stdin/stdout —— 站点(10_station_app.py)持有本进程时使用
  B) 本地 TCP 127.0.0.1:daemon_port —— estop.py / 交接命令.sh / 人工调试使用
     （连上发一行请求，读一行应答，关闭连接即可）

请求: {"id":n, "cmd": "..."}   应答: {"id":n, "ok":true/false, ...}
  cmd:
    ping                                -> {"ok":true,"uptime_s":..}
    command   kind,start|start_stream|stop|planner_on|planner_off
                                          repeat=3          -> 官方 command 话题
    stream    motion=<npz>, mode=pose|planner, rate=30      -> 动作流(后台线程)
    cancel                               -> 取消当前动作流
事件(无 id, 仅 stdout): {"event":"stream_progress"|"stream_done"|
                          "stream_cancelled"|"stream_error"|"fatal", ...}
"""
from __future__ import annotations

import json
import math
import os
import socket
import sys
import threading
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(_HERE))
GROOT_ROOT = os.environ.get(
    "GROOT_ROOT", "/home/yang/桌面/轨迹规划/GR00T-WholeBodyControl")
if GROOT_ROOT not in sys.path:
    sys.path.insert(0, GROOT_ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# GR00T 官方发送端模块 —— 本文件必须在 GR00T venv 下运行
import groot_bridge as gb  # noqa: E402


def _env_json():
    try:
        _d = json.load(open(os.path.join(BASE, "station", "env.json"),
                            encoding="utf8"))
        try:      # 本机覆盖层(不入库), 与 10_station_app 同机制
            _d.update(json.load(open(
                os.path.join(BASE, "station", "env.local.json"),
                encoding="utf8")))
        except Exception:
            pass
        return _d
    except Exception:
        return {}


ZMQ_PORT = int(os.environ.get("G1W_ZMQ_PORT",
                              _env_json().get("zmq_port", 5556)))
# ZMQ 发布端绑定地址: 优先环境变量显式覆盖, 其次 env.json 的 zmq_host
# (约定必须 localhost), 无配置时默认仅本机 —— 不再默认绑 '*'(全网卡可达)
ZMQ_HOST = (os.environ.get("G1W_ZMQ_HOST")
            or _env_json().get("zmq_host") or "127.0.0.1")
DAEMON_PORT = int(os.environ.get("G1W_DAEMON_PORT",
                                 _env_json().get("daemon_port", 5558)))

# 实机安全限幅: 速度倍率上下限(越界一律 clamp 到此区间并 emit 日志, 不静默)
SPEED_MIN, SPEED_MAX = 0.05, 1.5

# 官方安全保持站姿 = v7_arm_sdk_safe.csv 首帧(SONIC 交错序):
# 腰 0,0,0; 左肩pitch 0.2, 左肩roll 0.2, 左肘 0.6; 其余 0
DEFAULT_HOLD_POSE = [0.0, 0.0, 0.0, 0.2, 0.0, 0.2, 0.0, 0.0, 0.0,
                     0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# 从 29 维 body_q_measured(mujoco序: 腿12+腰3+左臂7+右臂7) 取上肢 17(SONIC交错序)
# —— 官方 pico_manager_thread_server.FeedbackReader._get_upper_body_joint_indices() 原样
UPPER17_FROM_BODY29 = [12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25,
                       19, 26, 20, 27, 21, 28]

# ---------------- V2 蹲姿通道: 腿关节反解骨盆高度 ----------------
# 与 05_export_ufo_motion.leg_ik_planar 同一套平面几何的正向反解:
#   knee -> r = sqrt(L1²+L2²-2·L1·L2·cos(π-knee)); pelvis_h = r + 0.1027 + ANKLE_Z
# (hip 的 b 角不影响 r, 无需知道髋角; 忽略 roll/yaw/脚背俯仰, 作高度估计足够)
_L1, _L2, _ANKLE_Z, _HIP_OFF = 0.34051, 0.31757, 0.038, 0.1027


def pelvis_h_from_q29(q29):
    """q29(mujoco序) -> 骨盆高度估计(m). 双腿取均值; 无效角度返回 None."""
    try:
        kn = [float(q29[3]), float(q29[9])]          # 左/右 knee(腿序: pitch,roll,yaw,knee,...)
        if not all(np.isfinite(kn)):
            return None
        hs = []
        for k in kn:
            g = np.pi - k                             # leg_ik_planar 的 g
            if not (0.02 < g < np.pi - 0.02):
                return None
            r2 = _L1 ** 2 + _L2 ** 2 - 2 * _L1 * _L2 * np.cos(g)
            hs.append(np.sqrt(max(r2, 1e-4)) + _HIP_OFF + _ANKLE_Z)
        return float(np.mean(hs))
    except Exception:
        return None


# ---------------- V2.1g 真机任务空间慢环: 走线水平度(腕心高度)修正 ----------------
# 仿真 v6 伺服的慢环移植: 真机反馈(q29)只有关节角无底座位姿 —— 站/蹲时底座近似
# 竖直朝墙(SONIC 平衡), 以 腿IK反解骨盆高 + 实测上肢关节FK 合成腕心世界位姿,
# 对 [腕心位置; 枪轴] 残差做 10Hz 低增益积分修正(叠加进发送的上肢17目标).
# 关节权限与仿真 v6 教训一致: 腰 yaw 禁用(会与底座偏航正反馈), 腰 roll/pitch 0.3,
# 主力右臂; 修正只加进最终注入命令, 不进快环(此处本就只有慢环).
try:
    from arm_task_servo import (fk_jac1 as _fk_jac1, JLO as _C_JLO,
                                JHI as _C_JHI)
    _SRV_OK = True
except Exception:                                    # 模块缺失时优雅降级
    _SRV_OK = False

# q10(mujoco序: 腰3+右臂7) -> SONIC17 序的索引: 腰 0..2; 右臂 body29 22..28
# (SONIC 交错序 = [腰3, Lsh_p, Rsh_p, Lsh_r, Rsh_r, ...], 见 UPPER17_FROM_BODY29)
_Q10_TO_SONIC17 = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16]
_TRIM_COL = np.array([0.0, 0.3, 0.3, 1, 1, 1, 1, 1, 1, 1])   # 腰yaw禁用


def wrist_world_from_q29(q29, h_pelvis):
    """实测 q29 + 骨盆高 -> (腕心世界估计(3,), 枪轴世界估计(3,)).
    底座假设: 竖直、朝 +x(墙面法向), 位置 [0,0,h_pelvis]."""
    q10 = np.concatenate([q29[12:15], q29[22:29]]).astype(float)
    p_rel, a_rel, _, _ = _fk_jac1(q10)
    return p_rel + np.array([0.0, 0.0, h_pelvis]), a_rel


class FeedbackReader:
    """订阅部署调试输出(5557/g1_debug)读 body_q_measured —— 机器人实际关节角.
    用途: ①保持实测姿态(零瞬态启动) ②执行时与指令同步记录, 定位映射/跟踪问题."""

    def __init__(self, host="localhost", port=5557):
        import zmq
        import msgpack
        self._zmq_mod = zmq
        self._msgpack = msgpack
        self.ctx = zmq.Context()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "g1_debug")
        self.sub.setsockopt(zmq.CONFLATE, 1)          # 只留最新一帧
        self.sub.connect(f"tcp://{host}:{port}")
        self.q29 = None
        self.ts = 0.0
        self.lock = threading.Lock()
        self.on_sample = None                         # 回调(q29, ts) —— 记录器挂这里
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        zmq = self._zmq_mod
        while True:
            try:
                if self.sub.poll(timeout=200):
                    data = self.sub.recv(zmq.NOBLOCK)[len("g1_debug"):]
                    m = self._msgpack.unpackb(data, raw=False)
                    q = m.get("body_q_measured")
                    if q is not None and len(q) == 29:
                        now = time.time()
                        with self.lock:
                            self.q29 = np.asarray(q, float)
                            self.ts = now
                        if self.on_sample:
                            try:
                                self.on_sample(self.q29, now)
                            except Exception:
                                pass
            except Exception:
                time.sleep(0.5)

    def upper17_sonic(self, max_wait=2.0, fresh_s=1.0):
        """最新实测上肢17(SONIC序); 无新鲜反馈时最多等 max_wait 秒, 仍无则 (None, None)."""
        t0 = time.time()
        while time.time() - t0 < max_wait:
            with self.lock:
                if self.q29 is not None and time.time() - self.ts < fresh_s:
                    return self.q29[UPPER17_FROM_BODY29].copy(), self.q29.copy()
            time.sleep(0.05)
        return None, None


def emit(obj):
    """协议输出: 仅 stdout, 一行一条 JSON."""
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


class Daemon:
    """独占 ZMQ PUB 的常驻发布端。所有 socket 发送经 send_lock 串行(zmq 线程不安全)。"""

    def __init__(self, host="127.0.0.1", port=5556):
        self.host, self.port = host, port
        self.pub = None
        self.send_lock = threading.Lock()
        self.stream_thread = None
        self.cancel_evt = threading.Event()
        self.streaming = False
        self.holding = False                      # 轨迹已结束, 正在持续保持末姿
        self.hold_position = None                 # 最近保持姿态(SONIC 序), blend 重跑用
        self._gen = 0                             # 流代数: 旧线程不得清掉新线程的状态
        self.feedback = None                      # FeedbackReader(懒创建, 读 5557)
        self._rec = None                          # 记录器状态(指令+实测同步记录)
        self._rec_lock = threading.Lock()
        self.t0 = time.time()

    def _ensure_feedback(self):
        if self.feedback is None:
            try:
                self.feedback = FeedbackReader(
                    host="localhost",
                    port=int(_env_json().get("debug_port", 5557)))
            except Exception as ex:
                emit({"event": "fatal", "err": f"反馈订阅创建失败: {ex}"})
        return self.feedback

    # ---------- 启动 ----------
    def bind_pub(self, retries=10, delay=0.5):
        """bind ZMQ PUB(唯一 5556 持有者); 被残留进程占用时重试后放弃."""
        import zmq
        last = None
        for _ in range(retries):
            try:
                self.pub = gb.Publisher(self.host, self.port)
                return True
            except zmq.ZMQError as ex:
                last = ex
                time.sleep(delay)
        emit({"event": "fatal", "err": f"ZMQ bind {self.host}:{self.port} 失败: {last}"
              f" (残留进程占用? 清理后重试)"})
        return False

    # ---------- 官方 command 话题 ----------
    CMD_MAP = {
        "start":       dict(start=True,  stop=False, planner=True),
        "start_stream": dict(start=True, stop=False, planner=False),
        "stop":        dict(start=False, stop=True,  planner=False),
        "planner_on":  dict(start=False, stop=False, planner=True),
        "planner_off": dict(start=False, stop=False, planner=False),
    }

    def send_command(self, kind, repeat=3):
        kw = self.CMD_MAP.get(kind)
        if kw is None:
            return {"ok": False, "err": f"未知 command: {kind}"}
        for _ in range(int(repeat)):
            with self.send_lock:
                self.pub.command(**kw)
            time.sleep(0.10)          # 官方节奏: 3 次 × 0.10s
        return {"ok": True, "kind": kind}

    # ---------- 官方上肢动作流 (send_upper_body_trajectory.py 对齐) ----------
    @staticmethod
    def _sample(times, values, t):
        """官方线性插值采样."""
        if t <= times[0]:
            return values[0].copy()
        if t >= times[-1]:
            return values[-1].copy()
        hi = int(np.searchsorted(times, t, side="right"))
        lo = hi - 1
        alpha = (t - times[lo]) / (times[hi] - times[lo])
        return (1.0 - alpha) * values[lo] + alpha * values[hi]

    def _send_upper(self, pos17, vel17, height=-1.0, mode=None):
        with self.send_lock:
            self.pub.planner_upper_body(pos17, upper_vel=vel17,
                                         height=float(height), mode=mode)
        with self._rec_lock:                       # 记录器: 记下每一条指令帧
            if self._rec is not None:
                self._rec["sent_t"].append(time.time())
                self._rec["sent_q"].append(np.asarray(pos17, float))
                self._rec["sent_v"].append(np.asarray(vel17, float))

    # ---------- 同步记录器: 指令(守护进程发的) + 实测(557 反馈) ----------
    def record_start(self):
        with self._rec_lock:
            if self._rec is not None:
                return {"ok": False, "err": "已在记录中"}
            self._rec = {"sent_t": [], "sent_q": [], "sent_v": [],
                         "meas_t": [], "meas_q": []}
            t0 = time.time()
        fb = self._ensure_feedback()
        if fb is not None:
            def _hook(q29, ts):
                with self._rec_lock:
                    if self._rec is not None:
                        self._rec["meas_t"].append(ts)
                        self._rec["meas_q"].append(np.asarray(q29, float))
            fb.on_sample = _hook
        emit({"event": "record_started",
              "note": "同步记录中: 指令帧(50Hz) + 实测帧(g1_debug)"})
        return {"ok": True, "t0": t0}

    def record_stop(self, path=None):
        fb = self.feedback
        if fb is not None:
            fb.on_sample = None
        with self._rec_lock:
            rec, self._rec = self._rec, None
        if rec is None:
            return {"ok": False, "err": "没有进行中的记录"}
        if path is None:
            path = os.path.join(BASE, "output",
                                "q_record_" + time.strftime("%H%M%S") + ".npz")
        sent_q = (np.asarray(rec["sent_q"]) if rec["sent_q"] else
                  np.zeros((0, 17)))
        meas_q = (np.asarray(rec["meas_q"]) if rec["meas_q"] else
                  np.zeros((0, 29)))
        np.savez(path,
                 sent_t=np.asarray(rec["sent_t"]), sent_upper17=sent_q,
                 sent_vel=np.asarray(rec["sent_v"]),
                 meas_t=np.asarray(rec["meas_t"]), meas_q29=meas_q,
                 meas_upper17=(meas_q[:, UPPER17_FROM_BODY29]
                               if len(meas_q) else np.zeros((0, 17))))
        emit({"event": "record_stopped", "path": path,
              "sent_frames": len(sent_q), "meas_frames": len(meas_q)})
        return {"ok": True, "path": path, "sent_frames": len(sent_q),
                "meas_frames": len(meas_q)}

    def _do_stream(self, motion, rate, speed, blend_s, mjcf, gen):
        """官方算法: 50Hz 时间采样 + 梯度速度 + 结束保持末姿 + 取消时 5×hold。
        全程只发 planner 话题(上肢 17 关节, SONIC 交错序) —— 下肢归 SONIC 平衡策略,
        绝不发 planner=False(会切模式触发 TriggerSafetyReset)。
        V2 蹲姿通道(数据驱动): 动作 root_z 最低 <0.72 时逐帧携带 planner height
        (期望骨盆高, 与上肢目标同一条消息, C++ 端进策略观测); 骨盆高用 5557 实测
        q29 经腿部平面几何反解, 外环比例纠偏(与仿真 runner --height 同口径:
        K=1.8, 纠偏±0.10, 指令限幅[0.28,0.85]) —— 带臂负载蹲不深时自动加深指令。
        V2.1h 分段门控: 高度/模式/外环只在蹲姿段(参考 z<0.70)发送, 站姿段
        height=-1+mode=None —— IDLE 不消费 target_height, 站姿段闭环的负纠偏
        会把 h_cmd 压过蹲阈值再弹回, IDLE<->SQUAT 模式极限环即实机"上下摇晃"
        (2026-10-12 首跑实测, 24 次翻转/400 条重规划)。模式滞回改挂参考高度:
        <0.66 进蹲 / >=0.70 回站, 纠偏只调蹲深不再影响模式切换。"""
        try:
            u_hw = gb.qpos_upper17_hw(motion, mjcf)     # (T,17) 硬件序(官方§8方法解析)
            pos = gb.to_sonic_order(u_hw)               # (T,17) SONIC 交错序
            n = len(pos)
            times = np.arange(n) / 30.0                 # 动作 30Hz 导出(官方: index/30)
            vel = np.gradient(pos, times, axis=0)
            duration = float(times[-1] / max(speed, SPEED_MIN))   # 实机安全限幅: 防 speed<=0 除零
            # V2: 蹲姿通道(数据驱动, 与 05 的 DUAL 判据同源)
            try:
                _mq = np.asarray(np.load(motion)["qpos"], float)
                z_root = _mq[:, 2]
                height_on = bool(z_root.min() < 0.72)
            except Exception:
                _mq, z_root, height_on = None, None, False
            # V2.1g: 任务空间慢环目标(腕心世界 P*/枪轴 A*, 规划=底座竖直@z_root)
            tgtP = tgtA = None
            if _SRV_OK and _mq is not None:
                try:
                    from arm_task_servo import precompute_motion_targets
                    tgtP, tgtA, _ = precompute_motion_targets(_mq)
                except Exception:
                    tgtP = tgtA = None
            trim10 = np.zeros(10)
            off17_last = np.zeros(17)
            h_ref = float(np.clip(z_root[0], 0.30, 0.85)) if height_on else -1.0
            h_cmd = h_ref
            h_corr = 0.0
            # 蹲姿模式(带滞回): SONIC planner 只在蹲姿系模式消费 target_height,
            # IDLE 下发高度机器人不会蹲(2026-10-12 实机实测) —— 下蹲段切 IDEL_SQUAT。
            # V2.1h: 滞回挂参考高度而非纠偏后的指令(见 _do_stream docstring)
            squat_mode = False
            seg_squat = False           # 蹲姿段门控(参考 z<0.70), 每帧更新
            fb = self._ensure_feedback() if height_on else None
            st = {"h_meas_min": 9.9, "h_err_max": 0.0, "up_err_sum": 0.0,
                  "up_err_max": 0.0, "n_up": 0,
                  "wz_err_sum": 0.0, "wz_err_max": 0.0, "n_wz": 0,
                  "ang_min": 90.0, "ang_sum": 0.0, "n_ang": 0}
            emit({"event": "stream_meta", "frames": n,
                  "duration_s": round(duration, 2), "rate": rate, "speed": speed,
                  "height_channel": height_on})
            # 官方启用序列: command(start=True, planner=True)×3。
            # 已启动时 start 为空操作(C++ 只在 !operator_state.start 时处理),
            # planner=True 保持 PLANNER 模式不切换 → 无安全重置、无二次斜坡。
            with self.send_lock:
                for _ in range(3):
                    self.pub.command(start=True, stop=False, planner=True)

            period = 1.0 / float(rate)
            next_tick = time.perf_counter()
            # 首次执行直接跑(官方); 从保持态重跑则 1s smoothstep blend(官方 B 语义)
            blend_from = self.hold_position if self.hold_position is not None else pos[0]
            h_blend_from = (float(np.clip(pelvis_h_from_q29(fb.q29), 0.30, 0.85))
                            if (height_on and fb is not None and fb.q29 is not None
                                and pelvis_h_from_q29(fb.q29)) else h_ref)
            phase = "blend" if self.hold_position is not None else "run"
            blend_start = time.perf_counter()
            run_start = time.perf_counter()
            hold_position = pos[-1].copy()
            last_prog = time.perf_counter()
            tick = 0
            while True:
                if self.cancel_evt.is_set() or self._gen != gen:
                    break
                now = time.perf_counter()
                if phase == "run":
                    elapsed = now - run_start
                    t_traj = min(times[-1], elapsed * speed)
                    p = self._sample(times, pos, t_traj)
                    v = self._sample(times, vel, t_traj) * speed
                    if height_on:
                        h_ref = float(np.clip(
                            self._sample(times, z_root, t_traj), 0.30, 0.85))
                    if elapsed >= duration:
                        hold_position = p.copy()
                        self.hold_position = hold_position   # 立即公布(重跑 blend 要用)
                        phase = "hold"
                        self.holding = True
                        emit({"event": "stream_done", "frames": n,
                              "note": "保持末姿中(SONIC 平衡站立)",
                              "height_channel": height_on,
                              "pelvis_min": (round(st["h_meas_min"], 3)
                                             if st["h_meas_min"] < 9 else None),
                              "pelvis_err_max": round(st["h_err_max"], 3),
                              "upper_err_mean": (round(st["up_err_sum"] /
                                                        max(st["n_up"], 1), 4)),
                              "upper_err_max": round(st["up_err_max"], 4),
                              "wrist_z_err_mean": round(st["wz_err_sum"] /
                                                        max(st["n_wz"], 1), 4),
                              "wrist_z_err_max": round(st["wz_err_max"], 4),
                              "gun_angle_off_mean": round(
                                  st["ang_sum"] / max(st["n_ang"], 1), 1),
                              "gun_angle_off_max": round(
                                  90.0 - st["ang_min"]
                                  if st["n_ang"] else 0.0, 1)})
                elif phase == "blend":
                    alpha = min(1.0, (now - blend_start) / blend_s)
                    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                    p = (1.0 - alpha) * blend_from + alpha * pos[0]
                    v = np.zeros(17)
                    if height_on:                      # 高度同步 blend: 实测 -> 首帧参考
                        h_ref = float(np.clip(
                            (1.0 - alpha) * h_blend_from + alpha * z_root[0], 0.30, 0.85))
                    if alpha >= 1.0:
                        phase = "run"
                        run_start = now
                        emit({"event": "stream_run", "note": "blend 完成, 开始轨迹"})
                else:
                    p = hold_position.copy()
                    v = np.zeros(17)
                # 蹲姿外环(10Hz): 实测骨盆高反馈纠偏(实测-参考差放大加深指令)
                # 任务空间慢环(10Hz, V2.1g): 腕心/枪轴残差 -> 上肢17 修正(走线水平度)
                if tick % 5 == 0:
                    _fresh = (fb is not None and fb.q29 is not None
                              and time.time() - fb.ts < 0.5)
                    q29m = np.asarray(fb.q29, float) if _fresh else None
                    h_m = pelvis_h_from_q29(q29m) if q29m is not None else None
                    u_m = q29m[UPPER17_FROM_BODY29] if q29m is not None else None
                    if u_m is not None:                # 上肢关节空间跟踪(SONIC序)
                        e_j = float(np.linalg.norm(u_m - (p + off17_last)))
                        st["up_err_sum"] += e_j; st["n_up"] += 1
                        st["up_err_max"] = max(st["up_err_max"], e_j)
                    if height_on:
                        if h_m is not None:      # 统计全程留痕(站姿段偏差也有诊断价值)
                            st["h_meas_min"] = min(st["h_meas_min"], h_m)
                            st["h_err_max"] = max(st["h_err_max"], abs(h_m - h_ref))
                        if seg_squat:            # 外环只在蹲姿段闭环加深指令;
                            if h_m is not None:  # 站姿段策略不消费高度, 闭环只会
                                h_corr = min(max(1.8 * (h_ref - h_m), -0.10), 0.10)
                            else:
                                h_corr *= 0.9    # 反馈丢失: 纠偏缓慢归零
                            h_cmd = float(np.clip(h_ref + h_corr, 0.28, 0.85))
                    # —— 任务空间慢环: 残差在实测构型上线性化, 积分修正(权限同仿真 v6)
                    if q29m is not None and h_m is not None and tgtP is not None:
                        t_now = (min(times[-1], (now - run_start) * speed)
                                 if phase == "run"
                                 else (0.0 if phase == "blend" else times[-1]))
                        p_star = self._sample(times, tgtP, t_now)
                        a_star = self._sample(times, tgtA, t_now)
                        _n = np.linalg.norm(a_star)
                        if _n > 1e-6:
                            p_w, a_w = wrist_world_from_q29(q29m, h_m)
                            st["wz_err_sum"] += abs(p_w[2] - p_star[2])
                            st["wz_err_max"] = max(st["wz_err_max"],
                                                   abs(p_w[2] - p_star[2]))
                            st["n_wz"] += 1
                            _an = np.degrees(np.arccos(np.clip(
                                a_w[0] / max(np.linalg.norm(a_w), 1e-9), -1, 1)))
                            st["ang_min"] = min(st["ang_min"], _an)
                            st["ang_sum"] += _an; st["n_ang"] += 1
                            q10m = np.concatenate([q29m[12:15], q29m[22:29]])
                            _, _, Jp, Ja = _fk_jac1(q10m)
                            J = np.vstack([Jp, Ja]) * _TRIM_COL
                            e = np.concatenate([p_star - p_w,
                                                0.5 * (a_star / _n - a_w)]) * 0.15
                            dq = J.T @ np.linalg.solve(
                                J @ J.T + 0.01 * np.eye(6), e)
                            np.clip(dq, -0.02, 0.02, out=dq)
                            trim10 = np.clip(trim10 + dq, -0.15, 0.15)
                off17 = np.zeros(17)
                off17[_Q10_TO_SONIC17] = trim10
                off17_last = off17
                # V2.1h 分段门控: 蹲姿段(参考 z<0.70)才发 height/mode, 站姿段
                # height=-1 + mode=None(与纯站姿动作同通道); 模式滞回挂参考
                # (0.66 进 / 0.70 出), 纠偏只调蹲深 —— 消灭站姿段模式极限环
                _seg = bool(height_on and h_ref < 0.70)
                if _seg != seg_squat:
                    seg_squat = _seg
                    emit({"event": "height_seg",
                          "seg": ("squat" if _seg else "stand"),
                          "h_ref": round(float(h_ref), 3)})
                if _seg:
                    squat_mode = squat_mode or h_ref < 0.66
                else:
                    squat_mode = False
                    h_corr = 0.0
                self._send_upper(p + off17, v, height=(h_cmd if _seg else -1.0),
                                 mode=(4 if (_seg and squat_mode) else None))
                if phase == "run" and now - last_prog >= 2.0:
                    emit({"event": "stream_progress", "k": int(t_traj * 30),
                          "n": n, "frac": round(t_traj / max(times[-1], 1e-9), 3)})
                    last_prog = now
                tick += 1
                next_tick += period
                sleep_s = next_tick - time.perf_counter()
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
                else:
                    next_tick = time.perf_counter()
            # 官方退出路径: 连发 5 帧保持目标(不为零速突降), 随后靠 planner >1s
            # 静默自动回 IDLE; 全程不发 stop
            self.hold_position = hold_position
            if self.cancel_evt.is_set():
                for _ in range(5):
                    self._send_upper(hold_position + off17_last, np.zeros(17),
                                     height=(h_cmd if seg_squat else -1.0))
                    time.sleep(0.02)
                emit({"event": "stream_cancelled", "note": "已按官方退出序列保持末姿"})
        except Exception as ex:
            emit({"event": "stream_error", "err": f"{type(ex).__name__}: {ex}"})
        finally:
            if self._gen == gen:
                self.streaming = False
                self.holding = False

    def start_stream(self, motion, rate=50.0, speed=1.0, blend_s=1.0, mjcf=None):
        """rate=50Hz/speed/blend 均为官方默认语义; speed=1.0 实时(动作本身 30Hz 导出)。"""
        speed = float(speed)
        if not math.isfinite(speed):      # NaN/inf 会穿透 min/max 夹取, 归一为实时
            emit({"event": "speed_clamped", "requested": speed, "applied": 1.0})
            speed = 1.0
        if not (SPEED_MIN <= speed <= SPEED_MAX):      # 实机安全限幅
            clamped = min(max(speed, SPEED_MIN), SPEED_MAX)
            emit({"event": "speed_clamped", "requested": speed,
                  "applied": clamped})                 # 记日志, 不静默
            speed = clamped
        if self.streaming and not self.holding:
            return {"ok": False, "err": "busy",
                    "note": "轨迹执行中, 请等本轮完成/进入保持, 或先停止"}
        if not os.path.exists(motion):
            return {"ok": False, "err": f"动作文件不存在: {motion}"}
        if self.streaming and self.holding:
            # 保持态让位给新轨迹(官方 B 重跑): 先递增 _gen 让旧线程在下个
            # tick(≤20ms)自行退出, 再等它结束 —— 避免两个线程同时发帧
            self._gen += 1
            if self.stream_thread is not None:
                self.stream_thread.join(timeout=2.0)
        self.cancel_evt.clear()
        self._gen += 1
        gen = self._gen
        self.streaming = True
        self.holding = False
        self.stream_thread = threading.Thread(
            target=self._do_stream,
            args=(motion, float(rate), float(speed), float(blend_s), mjcf, gen),
            daemon=True)
        self.stream_thread.start()
        return {"ok": True, "rate": rate, "speed": speed,
                "note": "官方上肢通道(17关节 SONIC序): 轨迹→保持末姿; "
                        "完成经 stream_done 事件通知"}

    def cancel_stream(self):
        self.cancel_evt.set()
        return {"ok": True, "cancelled": self.streaming}

    # ---------- 保持流(官方终端2语义: start 后持续发安全站姿) ----------
    def start_hold(self, pose17=None, motion=None, mjcf=None, from_robot=True):
        """50Hz 持续发送指定上肢姿态, 直到被新轨迹替换(blend)或取消。
        默认 from_robot=True: 保持**机器人当前实测姿态**(经 5557/g1_debug 读
        body_q_measured) —— 零瞬态(启动不会猛拉手臂), 且对任何关节序/左右映射
        争议免疫(保持的是测回来的值, 自洽)。无反馈时回退官方 v7 首帧常量。
        作用: 消除 ZMQ 空窗, 防 C++ 回退回放 reference 动作(舞蹈/踢腿)。"""
        src = "官方安全站姿(v7 首帧)"
        if self.streaming and not self.holding:
            return {"ok": False, "err": "busy", "note": "轨迹执行中"}
        if pose17 is None:
            if from_robot:
                fb = self._ensure_feedback()
                p17, _q29 = (fb.upper17_sonic(max_wait=2.0) if fb else (None, None))
                if p17 is not None:
                    pose17 = p17
                    src = f"机器人实测姿态({time.strftime('%H:%M:%S')})"
                else:
                    emit({"event": "hold_note",
                          "note": "未收到 5557/g1_debug 反馈, 回退官方站姿常量"})
            if pose17 is None:
                if motion:
                    u = gb.qpos_upper17_hw(motion, mjcf)
                    pose17 = gb.to_sonic_order(u)[0]
                    src = "动作首帧"
                else:
                    pose17 = DEFAULT_HOLD_POSE
        p = np.asarray(pose17, float)
        assert p.shape == (17,), f"保持姿态需 17 维(SONIC 序), 收到 {p.shape}"
        if self.streaming and self.holding:
            self._gen += 1
            if self.stream_thread is not None:
                self.stream_thread.join(timeout=2.0)
        self.cancel_evt.clear()
        self._gen += 1
        gen = self._gen
        self.hold_position = p.copy()
        self.streaming = True
        self.holding = True

        def _hold_loop():
            period, nxt = 1.0 / 50.0, time.perf_counter()
            try:
                while not self.cancel_evt.is_set() and self._gen == gen:
                    self._send_upper(p, np.zeros(17))
                    nxt += period
                    s = nxt - time.perf_counter()
                    if s > 0:
                        time.sleep(s)
                    else:
                        nxt = time.perf_counter()
                if self.cancel_evt.is_set():
                    for _ in range(5):
                        self._send_upper(p, np.zeros(17))
                        time.sleep(0.02)
                    emit({"event": "hold_cancelled",
                          "note": "已按官方退出序列结束保持(回 IDLE 平衡站立)"})
            except Exception as ex:
                emit({"event": "stream_error", "err": f"hold: {ex}"})
            finally:
                if self._gen == gen:
                    self.streaming = False
                    self.holding = False

        self.stream_thread = threading.Thread(target=_hold_loop, daemon=True)
        self.stream_thread.start()
        emit({"event": "hold_started",
              "note": f"保持流已启动(50Hz, 来源: {src})"})
        return {"ok": True, "note": f"保持流已启动(来源: {src})"}

    # ---------- 请求分发 ----------
    def handle(self, req):
        cmd = req.get("cmd")
        if cmd == "ping":
            return {"ok": True, "uptime_s": round(time.time() - self.t0, 1),
                    "zmq": f"{self.host}:{self.port}",
                    "streaming": self.streaming, "holding": self.holding}
        if cmd == "command":
            return self.send_command(req.get("kind"),
                                     req.get("repeat", 3))
        if cmd == "stream":
            return self.start_stream(req.get("motion"),
                                     float(req.get("rate", 50.0)),
                                     float(req.get("speed", 1.0)),
                                     float(req.get("blend_s", 1.0)),
                                     req.get("mjcf"))
        if cmd == "hold":
            return self.start_hold(req.get("pose"), req.get("motion"),
                                   req.get("mjcf"),
                                   from_robot=bool(req.get("from_robot", True)))
        if cmd == "q_measured":
            fb = self._ensure_feedback()
            p17, q29 = (fb.upper17_sonic(max_wait=0.5) if fb else (None, None))
            return {"ok": p17 is not None, "upper17_sonic": (p17.tolist() if p17 is not None else None),
                    "q29": (q29.tolist() if q29 is not None else None)}
        if cmd == "record_start":
            return self.record_start()
        if cmd == "record_stop":
            return self.record_stop(req.get("path"))
        if cmd == "cancel":
            return self.cancel_stream()
        return {"ok": False, "err": f"未知 cmd: {cmd}"}


def _tcp_server(dmn):
    """本地 TCP 控制口(estop.py / CLI 用): 一行请求一行应答."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind(("127.0.0.1", DAEMON_PORT))
        srv.listen(4)
    except OSError as ex:
        emit({"event": "fatal",
              "err": f"控制口 127.0.0.1:{DAEMON_PORT} 绑定失败: {ex}"})
        return
    while True:
        try:
            conn, _ = srv.accept()
        except OSError:
            return
        threading.Thread(target=_tcp_serve, args=(conn, dmn),
                         daemon=True).start()


def _tcp_serve(conn, dmn):
    try:
        conn.settimeout(10)
        buf = b""
        while b"\n" not in buf:
            chunk = conn.recv(4096)
            if not chunk:
                return
            buf += chunk
        req = json.loads(buf.split(b"\n", 1)[0].decode("utf8", "ignore"))
        conn.sendall(json.dumps(dmn.handle(req), ensure_ascii=False).encode()
                     + b"\n")
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def main():
    dmn = Daemon(host=ZMQ_HOST, port=ZMQ_PORT)
    if not dmn.bind_pub():
        sys.exit(2)
    emit({"event": "ready", "zmq": f"{ZMQ_HOST}:{ZMQ_PORT}",
          "control": f"127.0.0.1:{DAEMON_PORT}"})
    threading.Thread(target=_tcp_server, args=(dmn,), daemon=True).start()
    # stdin 请求循环: 站点退出(EOF)即结束, 释放 ZMQ 端口
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            emit({"ok": False, "err": "bad-json"})
            continue
        try:
            resp = dmn.handle(req)
        except Exception as ex:
            resp = {"ok": False, "err": f"{type(ex).__name__}: {ex}"}
        resp.setdefault("id", req.get("id"))
        emit(resp)
    # stdin EOF -> 站点已退出 -> 跟随退出
    try:
        with dmn.send_lock:
            dmn.pub.close()
    except Exception:
        pass
    sys.exit(0)


if __name__ == "__main__":
    main()
