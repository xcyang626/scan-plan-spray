# -*- coding: utf-8 -*-
"""rl_policy.py —— 部署侧策略数学(纯 numpy, 无 ROS/仿真依赖)
=============================================================
单一事实来源: rl_exec.py(真机) 与 verify_deploy.py(离线验证) 共用本模块,
保证"验证过的代码 = 上机的代码"。

观测 67 维(与 IsaacLab 训练环境逐项一致):
  [0:7]   q 归一化      (q - q_mid)/q_half
  [7:14]  qd / 10
  [14:21] tanh(上一动作)
  [21:28] q_ref(t) 归一化
  [28:56] q_ref(t+2/4/8/16) 归一化     ← 4×7 预览
  [56:63] qd_ref(t) / 10
  [63]    t / N
  [64:67] 相位 one-hot (转移出/走线/转移回)

指令: q_cmd = clamp(q_ref(t) + ACTION_SCALE·tanh(a), 软限位 ∓ CLEARANCE)
"""

import json
import math
import os
import re

import numpy as np

POLICY_HZ = 50
# 逐关节残差幅值 α = 0.25·τ_rated/kp(论文 2508.08241 §S3;
# 与训练侧 zigzag_env.action_scale 必须一致 —— 已按论文公式基线重训后生效)
ACTION_SCALE = np.array([0.0417, 0.05, 0.0964, 0.0534,
                         0.0583, 0.0583, 0.0583])
CLEARANCE = math.radians(3.0)
PREVIEW_STEPS = (2, 4, 8, 16)      # ×20ms = +0.04/0.08/0.16/0.32s
OBS_DIM = 67

_RL_TRACK = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SAFETY = os.path.abspath(os.path.join(
    _RL_TRACK, "..", "config", "real_safety.yaml"))
_DEFAULT_REF = os.path.join(_RL_TRACK, "ref_right.npz")
_DEFAULT_ACTOR = os.path.join(_RL_TRACK, "actor_right.npz")

# 兜底常量(real_safety.yaml 不可用时); 权威来源始终是 yaml
_FALLBACK_LO = [-0.3490, -0.1336, -1.4069, 0.0505, -1.5189, -0.2543, -1.5049]
_FALLBACK_HI = [4.4420, 3.2685, 1.6237, 2.4200, 1.5265, 0.9463, 1.5661]
_FALLBACK_HOME = [0.0, 0.0, 0.0, 0.085366463, 0.0, 0.0, 0.0]


def _parse_safety_yaml(text):
    """极简解析 real_safety.yaml(避免依赖 pyyaml —— 部署关键路径)。
    只取 soft_limits_rad 的 7 组区间与 home_rad。"""
    names = ["openarm_right_joint%d" % i for i in range(1, 8)]
    lo, hi = [], []
    for n in names:
        m = re.search(r"%s:\s*\[\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\]"
                      % re.escape(n), text)
        if not m:
            raise ValueError("soft_limits_rad 缺 %s" % n)
        lo.append(float(m.group(1)))
        hi.append(float(m.group(2)))
    m = re.search(r"home_rad:\s*\[([^\]]+)\]", text)
    if not m:
        raise ValueError("缺 home_rad")
    home = [float(v) for v in m.group(1).split(",")]
    if len(home) != 7:
        raise ValueError("home_rad 长度 %d != 7" % len(home))
    return np.array(lo), np.array(hi), np.array(home)


def load_limits(yaml_path=None):
    """从 real_safety.yaml 读软限位与 home(单一权威来源)。
    返回 (lo[7], hi[7], home[7], source_str)。"""
    p = yaml_path or _DEFAULT_SAFETY
    try:
        with open(p, encoding="utf-8") as f:
            text = f.read()
        lo, hi, home = _parse_safety_yaml(text)
        return lo, hi, home, os.path.basename(p)
    except Exception as e:  # noqa: BLE001
        print("[警告] 解析 %s 失败 (%s), 用内置兜底限位" % (p, e))
        return (np.array(_FALLBACK_LO), np.array(_FALLBACK_HI),
                np.array(_FALLBACK_HOME), "内置兜底")


class StreamClock:
    """参考时间轴时钟: 支持转移段加速(冲出摩擦带, 抖动治理)。
    pos 为浮点参考帧索引; 每个策略步前进 rate(idx) 帧。
    rate = 1 + (boost-1)·f(idx): f 在转移相位为 1、走线相位为 0,
    相位界 ±blend 帧内余弦过渡 —— 时间轴加速本身无阶跃。"""

    def __init__(self, pol, boost=1.0, blend_s=1.0):
        self.pol = pol
        self.boost = float(boost)
        self.blend = max(int(blend_s / pol.dt), 1)
        self.pos = 0.0
        self.scale = 1.0               # 实时全局速度倍率(操作员输入, 0923)

    def set_speed(self, s):
        """实时设置全局速度倍率(0=暂停保持)。上限/变化率限幅由调用方负责。"""
        self.scale = max(float(s), 0.0)

    def f_transfer(self, idx):
        b1, b2, bl = self.pol.i_out, self.pol.i_back, self.blend
        idx = float(idx)
        if idx < b1:
            return 1.0
        if idx < b1 + bl:
            return 1.0 - (idx - b1) / bl
        if idx < b2 - bl:
            return 0.0
        if idx < b2:
            return (idx - (b2 - bl)) / bl
        return 1.0

    def rate(self, idx):
        return (1.0 + (self.boost - 1.0) * self.f_transfer(idx)) * self.scale

    def advance(self):
        """前进一个策略步, 返回当前帧 idx。"""
        self.pos += self.rate(self.pos)
        return int(min(self.pos, self.pol.n - 1))

    @property
    def done(self):
        return self.pos >= self.pol.n - 1


class Policy:
    """参考轨迹 + actor 权重 → 50Hz 位置指令(部署与验证共用)。"""

    def __init__(self, actor_path=None, ref_path=None, yaml_path=None,
                 lp_freq=5.0, fade_s=1.5):
        """lp_freq: 策略残差一阶低通截止 [Hz](0=关)。真机转移段(0.12 rad/s,
        摩擦带内)出现 ~11Hz hunting, 残差低通压制策略 50Hz 输出对该模态的
        喂励; 参考运动带宽 ≤3Hz, 5Hz 截距对跟踪影响 <12%。
        fade_s: 流启动后残差线性渐入时长 [s](0=关), 消除起动手感冲击。"""
        self.actor_path = actor_path or _DEFAULT_ACTOR
        self.ref_path = ref_path or _DEFAULT_REF
        self.lp_freq = float(lp_freq)
        self.fade_s = float(fade_s)
        self._res_f = None
        self._k = 0
        self.lo, self.hi, self.home, self.lim_src = load_limits(yaml_path)
        self.q_mid = (self.lo + self.hi) / 2
        self.q_half = (self.hi - self.lo) / 2

        d = np.load(self.ref_path, allow_pickle=True)
        self.q_ref = np.asarray(d["q_ref"], dtype=np.float64)
        self.qd_ref = np.asarray(d["qd_ref"], dtype=np.float64)
        self.n = len(self.q_ref)
        meta = d["meta_json"]
        if hasattr(meta, "item"):
            meta = meta.item()
        if isinstance(meta, (bytes, str)):
            meta = json.loads(meta)
        self.meta = meta
        self.dt = 1.0 / float(meta["policy_hz"])
        self.i_out = int(float(meta["t_phase_out"]) / self.dt)
        self.i_back = int(float(meta["t_phase_weave_end"]) / self.dt)
        self.home_ref = np.array(meta["home_rad"])
        self.phase_labels = ("transfer_out", "weave", "transfer_back")

        p = dict(np.load(self.actor_path))
        self.activation = str(p.get("activation", b"elu"))
        # 层号可能非连续(train.py 保原索引 W0/W2/W4/W6; export_actor.py 重排
        # 为 W0..W3) → 统一重排为连续索引, 前向按顺序执行
        ws = sorted((k for k in p if re.fullmatch(r"W\d+", str(k))),
                    key=lambda k: int(str(k)[1:]))
        if not ws:
            raise ValueError("actor npz 中未找到 W* 层: %s" % list(p.keys()))
        self.params = {}
        for i, wk in enumerate(ws):
            self.params["W%d" % i] = p[wk]
            self.params["b%d" % i] = p["b" + str(wk)[1:]]
        self.n_layers = len(ws)
        self.layer_dims = [tuple(self.params["W%d" % i].shape)
                           for i in range(self.n_layers)]

    # ------------------------------------------------------------------
    def phase_of(self, idx):
        if idx < self.i_out:
            return 0
        return 1 if idx < self.i_back else 2

    def phase_onehot(self, idx):
        ph = self.phase_of(idx)
        v = [0.0, 0.0, 0.0]
        v[ph] = 1.0
        return v

    def obs(self, q, qd, prev_a, idx, qd_scale=1.0):
        """装配 67 维观测(布局见模块 docstring)。
        qd_scale: 时间轴加速时实际参考速度 = rate·qd_ref, 观测同步缩放,
        保持"观测描述真实运动"的训练一致性。"""
        idx = int(min(max(idx, 0), self.n - 1))
        q = np.asarray(q, dtype=np.float64)
        qd = np.asarray(qd, dtype=np.float64)
        parts = [
            (q - self.q_mid) / self.q_half,
            qd / 10.0,
            np.tanh(prev_a),
            (self.q_ref[idx] - self.q_mid) / self.q_half,
        ]
        for k in PREVIEW_STEPS:
            parts.append((self.q_ref[min(idx + k, self.n - 1)] - self.q_mid)
                         / self.q_half)
        parts.append(self.qd_ref[idx] * qd_scale / 10.0)
        parts.append([idx / self.n])
        parts.append(self.phase_onehot(idx))
        return np.concatenate(parts)

    def forward(self, obs):
        """ELU MLP 前向(确定性均值动作, 不含分布采样)。"""
        h = np.asarray(obs, dtype=np.float64)
        for i in range(self.n_layers):
            h = h @ self.params["W%d" % i].T + self.params["b%d" % i]
            if i < self.n_layers - 1:
                if self.activation.startswith("elu"):
                    h = np.where(h > 0, h, np.expm1(h))
                elif self.activation.startswith("relu"):
                    h = np.maximum(h, 0.0)
        return h

    def action(self, q, qd, prev_a, idx, qd_scale=1.0):
        return np.clip(self.forward(self.obs(q, qd, prev_a, idx,
                                             qd_scale=qd_scale)),
                       -3.0, 3.0)

    def raw_command(self, q, qd, prev_a, idx):
        """未做限位/速率钳位的理论指令(供验证范围)。"""
        a = np.tanh(self.action(q, qd, prev_a, idx))
        idx = int(min(max(idx, 0), self.n - 1))
        return self.q_ref[idx] + ACTION_SCALE * a

    def reset_stream(self):
        """每段指令流开始时调用一次: 清空滤波器状态与渐入计数。"""
        self._res_f = None
        self._k = 0

    def command(self, q, qd, prev_a, idx, qd_scale=1.0):
        """最终指令 = 参考 + 滤波后残差, 钳位到 软限位 ∓ CLEARANCE。
        残差处理链: tanh(a)·scale → 一阶低通(lp_freq) → 启动渐入(fade_s)。"""
        idx = int(min(max(idx, 0), self.n - 1))
        a = self.action(q, qd, prev_a, idx, qd_scale=qd_scale)
        res = ACTION_SCALE * np.tanh(a)
        if self.lp_freq > 0:
            alpha = 1.0 - math.exp(-2.0 * math.pi * self.lp_freq * self.dt)
            self._res_f = (res.copy() if self._res_f is None
                           else self._res_f + alpha * (res - self._res_f))
            res = self._res_f
        if self.fade_s > 0:
            res = res * min(1.0, (self._k * self.dt) / self.fade_s)
            self._k += 1
        return np.clip(self.q_ref[idx] + res,
                       self.lo + CLEARANCE, self.hi - CLEARANCE)

    def summary(self):
        return dict(
            ref_frames=self.n, dt=self.dt,
            T_total=float(self.meta["T_total"]),
            i_out=self.i_out, i_back=self.i_back,
            obs_dim=int(self.params["W0"].shape[1]),
            layers=self.n_layers,
            layer_dims=self.layer_dims,
            action_scale=ACTION_SCALE.tolist(),
            clearance_deg=math.degrees(CLEARANCE),
            lim_source=self.lim_src,
            cmd_lo=(self.lo + CLEARANCE).tolist(),
            cmd_hi=(self.hi - CLEARANCE).tolist(),
        )
