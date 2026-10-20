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
from collections import deque

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(_HERE))
GROOT_ROOT = os.environ.get(
    "GROOT_ROOT", os.path.expanduser("~/桌面/轨迹规划/GR00T-WholeBodyControl"))
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


# ---------------- 腿式里程计(2026-10-19: SONIC 行走闭环位移反馈) ----------------
# G1 低层控制模式固件不提供里程计(g1_debug 无位移字段, C++ 无速度估计), 但 5557
# 调试流有关节编码器+IMU —— 腿式里程计: 支撑脚世界位置不动, 骨盆位移 =
# -支撑脚在体坐标下的位移(按 IMU yaw 旋到世界系), 逐帧积分。比开环时间积分
# 准一个量级, 行走指令据此闭环(走-量-停, 欠补过退)。
def _rot_axis(ax, a):
    c, s = math.cos(a), math.sin(a)
    if ax == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if ax == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _quat2rot(q4):
    """wxyz 四元数 -> 旋转矩阵。"""
    w, x, y, z = (float(v) for v in q4)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# G1 腿链逐级规格(自 MJCF g1-mode_13_15.xml 逐项核对, 2026-10-19): 每级 =
# (关节轴, 相对父级杆件偏移, 固定姿态四元数 wxyz|None)。链内有 ±10° 固定
# 转角与横向杆偏置, 简单 L1/L2 两连杆 FK 的位移差分误差可到 4cm(实测对模型
# 验证), 必须带全链。
_LEG_CHAIN_L = [
    ("y", (0.0, 0.064452, -0.1027), None),                       # hip_pitch_link
    ("x", (0.0, 0.052, -0.030465), (0.99617869, 0.0, -0.08733857, 0.0)),
    ("z", (0.025001, 0.0, -0.12412), None),                      # hip_yaw_link
    ("y", (-0.078273, 0.0021489, -0.17734), (0.99617869, 0.0, 0.08733857, 0.0)),
    ("y", (0.0, -0.00009445, -0.30001), None),                   # ankle_pitch_link
    ("x", (0.0, 0.0, -0.017558), None),                          # ankle_roll_link
]
_LEG_CHAIN_R = [                                                  # 右腿=y 镜像
    ("y", (0.0, -0.064452, -0.1027), None),
    ("x", (0.0, -0.052, -0.030465), (0.99617869, 0.0, -0.08733857, 0.0)),
    ("z", (0.025001, 0.0, -0.12412), None),
    ("y", (-0.078273, -0.0021489, -0.17734), (0.99617869, 0.0, 0.08733857, 0.0)),
    ("y", (0.0, 0.00009445, -0.30001), None),
    ("x", (0.0, 0.0, -0.017558), None),
]


def foot_in_base(qleg, side="L"):
    """单腿 6 关节(mujoco序) -> 脚心在骨盆系下的位置(踝 roll link 下 38mm
    估计点)。髋相对骨盆常值偏移已在链首, Δ 差分与积分都用同一函数, 自洽。"""
    chain = _LEG_CHAIN_L if side == "L" else _LEG_CHAIN_R
    R = np.eye(3)
    p = np.zeros(3)
    for i, (ax, off, q4) in enumerate(chain):
        p = p + R @ np.asarray(off)
        if q4 is not None:
            R = R @ _quat2rot(q4)
        R = R @ _rot_axis(ax, float(qleg[i]))
    return p + R @ np.array([0.0, 0.0, -_ANKLE_Z])


def _yaw_from_quat(wxyz):
    """IMU 四元数(w,x,y,z) -> 偏航角 rad(行走近乎朝墙, yaw≈0)。"""
    try:
        w, x, y, z = (float(v) for v in wxyz)
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    except Exception:
        return 0.0


# ---------------- 遥测环形缓冲(2026-10-19: 出问题后能回看"之前的数值") ----------------
# 常驻 50Hz×120s 环形缓冲(里程计/骨盆高/支撑脚/腿关节/yaw) + 指令与任务事件
# 打点; 异常(取消/错误/行走停滞)或手动触发时 dump 最近 120s -> npz+events.json
# (纯 numpy/json, 零额外依赖), 站点侧收到事件后自动分析出图并给参数建议。
_TELE_HZ, _TELE_SPAN_S = 50.0, 120.0
_TELE_CAP = int(_TELE_HZ * _TELE_SPAN_S)


class Telemetry:
    def __init__(self, cap=_TELE_CAP):
        self.t = np.zeros(cap)
        self.odom = np.zeros((cap, 2))
        self.pelvis_h = np.full(cap, np.nan)
        self.yaw = np.zeros(cap)
        self.leg_q = np.zeros((cap, 12))
        self.sup = np.zeros(cap, dtype=np.int8)       # bit0=左支撑 bit1=右支撑
        self.head, self.count, self.cap = 0, 0, cap
        self.events = deque(maxlen=4000)              # (time.time(), kind, dict)
        self._lock = threading.Lock()

    def sample(self, ts, odom, pelvis_h, yaw, leg_q, sup_mask):
        with self._lock:
            i = self.head
            self.t[i] = ts
            self.odom[i] = odom[:2]
            self.pelvis_h[i] = pelvis_h if pelvis_h is not None else np.nan
            self.yaw[i] = yaw
            self.leg_q[i] = leg_q
            self.sup[i] = sup_mask
            self.head = (i + 1) % self.cap
            self.count = min(self.count + 1, self.cap)

    def event(self, kind, **kw):
        with self._lock:
            self.events.append((time.time(), kind, kw))

    def dump(self, reason):
        """落盘最近 _TELE_CAP 帧 + 事件表; 返回 (npz路径, json路径)。"""
        out_dir = os.path.join(BASE, "output", "telemetry")
        os.makedirs(out_dir, exist_ok=True)
        name = "tele_" + time.strftime("%Y%m%d_%H%M%S")
        with self._lock:
            if self.count:
                idx = (np.arange(self.count) + self.head - self.count) % self.cap
                data = dict(t=self.t[idx].copy(), odom_x=self.odom[idx, 0].copy(),
                            odom_y=self.odom[idx, 1].copy(),
                            pelvis_h=self.pelvis_h[idx].copy(),
                            yaw=self.yaw[idx].copy(), leg_q=self.leg_q[idx].copy(),
                            sup=self.sup[idx].copy())
            else:
                data = dict(t=np.zeros(0), odom_x=np.zeros(0), odom_y=np.zeros(0),
                            pelvis_h=np.zeros(0), yaw=np.zeros(0),
                            leg_q=np.zeros((0, 12)), sup=np.zeros(0, np.int8))
            evs = list(self.events)
        npz_p = os.path.join(out_dir, name + ".npz")
        json_p = os.path.join(out_dir, name + ".json")
        np.savez(npz_p, **data)
        json.dump(evs, open(json_p, "w", encoding="utf8"), ensure_ascii=False)
        return npz_p, json_p


_ODOM_LIFT_M = 0.03          # 抬脚 >3cm 视为摆动腿(非支撑)
_ODOM_STEP_CAP_M = 0.05      # 单帧位移上限(防解码毛刺跳变)


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

    def __init__(self, host="localhost", port=5557, telemetry=None):
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
        # 腿式里程计(世界系 xy, m; 行走闭环用): 支撑脚不动 -> Δbase = -Δfoot
        self.odom = np.zeros(2)                       # [x, y] 世界系(y+=左)
        self.odom_n = 0                               # 已积分样本数(置信度)
        self._odom_prev = None                        # (feet[2], sup(set), yaw)
        self.telemetry = telemetry                    # 遥测采样钩子(可选)
        threading.Thread(target=self._loop, daemon=True).start()

    def odom_reset(self):
        with self.lock:
            self.odom[:] = 0.0
            self._odom_prev = None

    def _odom_step(self, q, quat):
        """一帧里程计积分: 调用于持有 self.lock 的 _loop 内。"""
        try:
            feet = (foot_in_base(q[0:6], "L"), foot_in_base(q[6:12], "R"))
            zmin = min(f[2] for f in feet)
            sup = [i for i, f in enumerate(feet) if f[2] < zmin + _ODOM_LIFT_M]
            yaw = _yaw_from_quat(quat) if quat else 0.0
            self._last_sup_mask = (1 if 0 in sup else 0) | (2 if 1 in sup else 0)
            self._last_yaw = yaw
            cy, sy = math.cos(yaw), math.sin(yaw)
            prev = self._odom_prev
            if prev is not None:
                pfeet, psup = prev[0], prev[1]
                ds = []
                for i in sup:
                    if i in psup:                     # 连续支撑才可信(切脚瞬态跳过)
                        d = -(feet[i][:2] - pfeet[i][:2])
                        if np.hypot(*d) < _ODOM_STEP_CAP_M:
                            ds.append(np.array([cy * d[0] - sy * d[1],
                                                sy * d[0] + cy * d[1]]))
                if ds:
                    self.odom += np.mean(ds, axis=0)
                    self.odom_n += 1
            self._odom_prev = (feet, sup, yaw)
        except Exception:
            self._odom_prev = None                    # 坏帧: 重建支撑基准

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
                            self._odom_step(self.q29, m.get("base_quat"))
                            if self.telemetry is not None:
                                self.telemetry.sample(
                                    now, self.odom, pelvis_h_from_q29(self.q29),
                                    getattr(self, "_last_yaw", 0.0),
                                    self.q29[0:12],
                                    getattr(self, "_last_sup_mask", 0))
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
        self.telemetry = Telemetry()              # 常驻环形遥测(异常时 dump)
        self._rec = None                          # 记录器状态(指令+实测同步记录)
        self._rec_lock = threading.Lock()
        self.t0 = time.time()

    def _ensure_feedback(self):
        if self.feedback is None:
            try:
                self.feedback = FeedbackReader(
                    host="localhost",
                    port=int(_env_json().get("debug_port", 5557)),
                    telemetry=self.telemetry)
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

    def _do_stream(self, motion, rate, speed, blend_s, mjcf, gen, once=False):
        """官方算法: 50Hz 时间采样 + 梯度速度 + 结束保持末姿 + 取消时 5×hold。
        全程只发 planner 话题(上肢 17 关节, SONIC 交错序) —— 下肢归 SONIC 平衡策略,
        绝不发 planner=False(会切模式触发 TriggerSafetyReset)。
        once=True: 单站播完(官方 5×hold 收尾)即返回 —— 多工位接力线程内联调用,
        不进入无限保持(保持期由接力状态机接管行走/下一站)。
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
                        if once:                   # 接力单站: 收尾即让出通道
                            for _ in range(5):
                                self._send_upper(hold_position + off17_last,
                                                 np.zeros(17))
                                time.sleep(0.02)
                            break
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
            try:
                npz_p, json_p = self.telemetry.dump("stream_error")
                emit({"event": "telemetry_dumped", "reason": "stream_error",
                      "npz": npz_p, "json": json_p})
            except Exception:
                pass
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
        self.telemetry.event("cancel", streaming=self.streaming)
        try:                                        # 急停/手动停止: 自动留现场
            npz_p, json_p = self.telemetry.dump("cancel")
            emit({"event": "telemetry_dumped", "reason": "cancel",
                  "npz": npz_p, "json": json_p})
        except Exception:
            pass
        return {"ok": True, "cancelled": self.streaming}

    # ---------- SONIC 行走(2026-10-19 实机多工位): planner 运动指令 ----------
    def _walk(self, dy, v, gen, settle_s=2.5, t0_margin=0.8, sign=1.0):
        """SONIC 位移(2026-10-19 定稿: 腿式里程计闭环)。脉冲式 走-量-停:
        每段命令 ≤BURST_S 即停, 停稳后读里程计, 距目标欠 TOL 以上续走、过冲
        反向退 —— 不再依赖"时间×速度"开环估算。里程计不可用(无 5557 反馈/
        积分样本不足)时回退开环(t0_margin 步态死区裕量)。失效安全不变:
        线程死→断流→部署端 1s 收不到 planner 消息自停。"""
        v = min(max(float(v), 0.2), 0.8)            # SLOW_WALK 官方速度域
        fb = self._ensure_feedback()
        _odom_ok = bool(fb is not None and fb.odom_n >= 10)
        t0 = time.perf_counter()
        tmo = float(dy) / v * 4.0 + 25.0            # 总超时(闭环含停稳测量)

        def _send_mv(on, dir=1.0):
            with self.send_lock:
                self.pub.planner_locomotion(
                    (gb.MODE_SLOW_WALK if on else gb.MODE_IDLE),
                    ((0.0, float(sign) * dir, 0.0) if on else (0.0, 0.0, 0.0)),
                    (1.0, 0.0, 0.0), speed=(v if on else -1.0))

        def _idle5():
            for _ in range(5):
                _send_mv(False)
                time.sleep(0.033)

        def _progress():
            """沿目标方向的已走距离(m, 里程计世界 y×sign)。"""
            with fb.lock:
                return float(sign * fb.odom[1])

        emit({"event": "walk_started", "dy": round(float(dy), 3), "v": v,
              "mode": ("odom-closed-loop" if _odom_ok else "open-loop-fallback"),
              "t0_margin": (0.0 if _odom_ok else round(float(t0_margin), 2)),
              "est_s": round(float(dy) / v
                             + (0.0 if _odom_ok else max(0.0, float(t0_margin))), 1)})
        _tele = self.telemetry
        _tele.event("walk_start", dy=round(float(dy), 4), v=v,
                    mode=("odom" if _odom_ok else "open"), sign=float(sign))
        _dump_reason = None
        try:
            if _odom_ok:
                fb.odom_reset()
                BURST_S, GATE_M, TOL_M = 0.7, 0.05, 0.035
                _stall, _last = 0, 0.0
                while not self.cancel_evt.is_set() and self._gen == gen:
                    p = _progress()
                    if abs(dy - p) <= TOL_M:
                        break
                    d = 1.0 if p < dy else -1.0      # 欠走正向 / 过冲反向退
                    _tele.event("cmd_mv_on", dir=float(sign * d), v=v,
                                p=round(p, 4))
                    tb = time.perf_counter()
                    nxt = tb
                    while (time.perf_counter() - tb) < BURST_S:
                        if self.cancel_evt.is_set() or self._gen != gen:
                            break
                        p = _progress()
                        if (d > 0 and p >= dy - GATE_M) or \
                           (d < 0 and p <= dy + GATE_M):
                            break                    # 提前收臂, 给减速留距离
                        _send_mv(True, d)
                        nxt += 1.0 / 30.0
                        s = nxt - time.perf_counter()
                        time.sleep(s if s > 0 else 0.0)
                    _idle5()
                    _tele.event("cmd_mv_off", p=round(_progress(), 4))
                    time.sleep(0.8)                  # 停稳再量(里程计含惯性段)
                    p2 = _progress()
                    if abs(p2 - _last) < 0.01:
                        _stall += 1                  # 走不动(卡住/托底)
                    else:
                        _stall = 0
                    _last = p2
                    if _stall >= 3:
                        _dump_reason = "walk_stall"
                        break
                    if (time.perf_counter() - t0) > tmo:
                        _dump_reason = "walk_timeout"
                        break
                walked = _progress()
                measured = True
            else:
                t_walk = max(0.0, float(t0_margin)) + float(dy) / v
                nxt = time.perf_counter()
                while (not self.cancel_evt.is_set() and self._gen == gen):
                    el = time.perf_counter() - t0
                    if el >= t_walk or el > tmo:
                        break
                    _send_mv(True)
                    nxt += 1.0 / 30.0
                    s = nxt - time.perf_counter()
                    time.sleep(s if s > 0 else 0.0)
                walked = min(v * max(0.0, (time.perf_counter() - t0)
                                     - max(0.0, float(t0_margin))), float(dy))
                measured = False
            _cancelled = self.cancel_evt.is_set()
            _idle5()
            _tele.event("walk_done", dy=round(float(dy), 4),
                        walked=round(float(walked), 4), measured=measured,
                        cancelled=_cancelled,
                        held_s=round(time.perf_counter() - t0, 2))
            emit({"event": "walk_done", "dy": round(float(dy), 3),
                  "walked_est": round(float(walked), 3), "measured": measured,
                  "cancelled": _cancelled,
                  "held_s": round(time.perf_counter() - t0, 2)})
            if _dump_reason:                     # 走不动/超时: 留现场供回看分析
                npz_p, json_p = _tele.dump(_dump_reason)
                emit({"event": "telemetry_dumped", "reason": _dump_reason,
                      "npz": npz_p, "json": json_p})
            if settle_s > 0 and not _cancelled and self._gen == gen:
                emit({"event": "walk_settle", "settle_s": settle_s})
                time.sleep(settle_s)
        except Exception as ex:
            emit({"event": "stream_error", "err": f"walk: {ex}"})
            try:
                npz_p, json_p = _tele.dump("walk_exception")
                emit({"event": "telemetry_dumped", "reason": "walk_exception",
                      "npz": npz_p, "json": json_p})
            except Exception:
                pass

    def _busy_guard(self):
        """接力/行走与轨迹流互斥: 执行中(且非保持态)拒绝新任务。"""
        if self.streaming and not self.holding:
            return {"ok": False, "err": "busy",
                    "note": "任务执行中, 请等本轮完成或先停止"}
        if self.streaming and self.holding:         # 保持态让位(官方 B 重跑语义)
            self._gen += 1
            if self.stream_thread is not None:
                self.stream_thread.join(timeout=2.0)
        return None

    def start_walk(self, dy, v=0.25, settle_s=2.5, t0_margin=0.8):
        """单段行走(测试原语): 独立验证 SONIC 位移指令的实机跟踪(量出实际位移
        可反算步态死区校准 t0_margin), 也供页面【补走】修正站位。dy<0=右移。"""
        _sign = -1.0 if float(dy) < 0 else 1.0
        busy = self._busy_guard()
        if busy:
            return busy
        self.cancel_evt.clear()
        self._gen += 1
        gen = self._gen
        self.streaming = True
        self.holding = False

        def _th():
            try:
                self._walk(abs(float(dy)), v, gen, settle_s, t0_margin,
                           sign=_sign)
            finally:
                if self._gen == gen:
                    self.streaming = False
                    self.holding = False
        self.stream_thread = threading.Thread(target=_th, daemon=True)
        self.stream_thread.start()
        return {"ok": True, "dy": dy, "v": v, "settle_s": settle_s,
                "note": "SONIC 单段位移(SLOW_WALK, 开环时间积分+步态死区裕量)"}

    def start_relay(self, motion, walk_list, speed=1.0, walk_v=0.25,
                    settle_s=2.5, rate=50.0, blend_s=1.0, mjcf=None, band=None,
                    walk_t0=0.8):
        """多工位接力(实机): 每站 _do_stream(once) → 停稳 → SONIC 左移 walk_list[k]
        → 停稳 → 下一站; 全部完成 relay_done。行走后用 5557 实测上肢作下站
        blend 起点(从实际姿态平滑进轨迹, 不从陈旧保持姿猛拉)。"""
        if not os.path.exists(motion):
            return {"ok": False, "err": f"动作文件不存在: {motion}"}
        wl = [float(v) for v in (walk_list or [])]
        if not wl or any(not (0.05 < d < 1.5) for d in wl):
            return {"ok": False, "err": f"步距列表异常(每段需 0.05~1.5m): {wl}"}
        busy = self._busy_guard()
        if busy:
            return busy
        self.cancel_evt.clear()
        self._gen += 1
        gen = self._gen
        self.streaming = True
        self.holding = False
        n = 1 + len(wl)

        def _th():
            try:
                for k in range(1, n + 1):
                    if self.cancel_evt.is_set() or self._gen != gen:
                        break
                    emit({"event": "relay_station", "station_idx": k,
                          "stations_total": n, "band": band,
                          "walk_dy_planned": (wl[k - 1] if k > 1 else wl[0])})
                    self._do_stream(motion, float(rate), float(speed),
                                    float(blend_s), mjcf, gen, once=True)
                    self.streaming = True           # _do_stream finally 会清, 接力续占
                    if self.cancel_evt.is_set() or self._gen != gen:
                        break
                    if k < n:
                        emit({"event": "relay_phase", "phase": "station_done",
                              "station_idx": k, "stations_total": n})
                        time.sleep(settle_s)        # 喷完先停稳
                        if self.cancel_evt.is_set() or self._gen != gen:
                            break
                        self._walk(wl[k - 1], walk_v, gen, settle_s=settle_s,
                                   t0_margin=walk_t0)
                        self.streaming = True
                        if self.cancel_evt.is_set() or self._gen != gen:
                            break
                        fb = self._ensure_feedback()
                        p17, _q = (fb.upper17_sonic(max_wait=1.0) if fb
                                   else (None, None))
                        self.hold_position = (np.asarray(p17, float)
                                              if p17 is not None else None)
                if not self.cancel_evt.is_set() and self._gen == gen:
                    emit({"event": "relay_done", "stations_total": n})
            finally:
                if self._gen == gen:
                    self.streaming = False
                    self.holding = False
        self.stream_thread = threading.Thread(target=_th, daemon=True)
        self.stream_thread.start()
        return {"ok": True, "stations": n, "walk_list": wl,
                "walk_v": walk_v, "settle_s": settle_s,
                "note": f"实机多工位接力: {n} 站喷涂 + SONIC 行走"
                        f"({len(wl)} 段左移, 开环位移, 人工监护随时急停)"}

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
        if cmd == "telemetry_dump":            # 手动抓取现场(最近 120s)
            try:
                npz_p, json_p = self.telemetry.dump("manual")
                emit({"event": "telemetry_dumped", "reason": "manual",
                      "npz": npz_p, "json": json_p})
                return {"ok": True, "npz": npz_p, "json": json_p,
                        "samples": self.telemetry.count}
            except Exception as ex:
                return {"ok": False, "err": f"dump 失败: {ex}"}
        if cmd == "walk":                 # 单段行走(SONIC 位移指令实机测试原语)
            return self.start_walk(float(req.get("dy", 0.3)),
                                   float(req.get("v", 0.25)),
                                   float(req.get("settle_s", 2.5)),
                                   float(req.get("t0_margin", 0.8)))
        if cmd == "relay":                # 多工位接力: 每站 stream + SONIC 行走
            return self.start_relay(req.get("motion"),
                                    req.get("walk_list") or [],
                                    float(req.get("speed", 1.0)),
                                    float(req.get("walk_v", 0.25)),
                                    float(req.get("settle_s", 2.5)),
                                    float(req.get("rate", 50.0)),
                                    float(req.get("blend_s", 1.0)),
                                    req.get("mjcf"),
                                    req.get("band"),
                                    float(req.get("walk_t0", 0.8)))
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
