# -*- coding: utf-8 -*-
"""OpenArm 右臂之字形走线 RL 环境(DirectRLEnv) + 配置
====================================================
任务: 50Hz 位置残差策略跟踪 ref_right.npz 参考轨迹(速度层级已内含),
奖励 = 跟踪 + 速度剖面跟随 + 平滑/稳定/限位惩罚
(见 docs/RL轨迹提速_奖励与时间设计.md §5)。
执行器 = DmMITActuator(与真机 zero_offset_hw 控制律一致)。

时序(与 DirectRLEnv.step 一致):
  pre_physics(t): 用 t_idx 计算 q_cmd → 物理推进(200Hz, 4 步)
  dones(t):      终止判定后 t_idx += 1
  rewards(t):    用 t_idx-1 对齐当前状态
  observations:  面向下一步决策(t_idx 已前移)
"""

import math
import os
from collections.abc import Sequence

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, UsdFileCfg, \
    spawn_ground_plane
from isaaclab.utils import configclass

from .mit_actuator import DmMITActuatorCfg

_HERE = os.path.dirname(os.path.abspath(__file__))       # rl/openarm_zigzag
_RL = os.path.abspath(os.path.join(_HERE, ".."))         # rl
_USD_DIR = os.path.abspath(os.path.join(_RL, ".."))      # isaaclab_sim
# RL 专用 USD: 夹爪 prismatic 关节已改 fixed 并并入 link7(零自由度),
# 对应真机 ESC8 夹爪使能后原地锁定; 高刚度 PD 夹爪在 200Hz 下会数值发散
USD_PATH = os.path.join(_RL, "openarm_right_rl.usd")
REF_PATH = os.path.join(_RL, "ref_right.npz")

ARM_JOINTS = ["openarm_right_joint%d" % i for i in range(1, 8)]
# 真机安全配置镜像(real_safety.yaml 单一权威来源; 部署时以 yaml 为准)
_SAFETY = os.path.abspath(os.path.join(
    _HERE, "..", "..", "..", "..", "real", "right", "config",
    "real_safety.yaml"))


def _read_limits():
    """软限位/home 从 real_safety.yaml 读(单一权威来源)。
    曾因硬编码副本与重标定后的真机配置冲突而导致部署失败。"""
    import re
    txt = open(_SAFETY, encoding="utf-8").read()
    lim = []
    for i in range(1, 8):
        m = re.search(r"openarm_right_joint%d:\s*\[\s*([-\d.eE+]+)\s*,"
                      r"\s*([-\d.eE+]+)\s*\]" % i, txt)
        lim.append([float(m.group(1)), float(m.group(2))])
    h = re.search(r"home_rad:\s*\[([^\]]+)\]", txt)
    home = [float(v) for v in h.group(1).split(",")]
    return lim, home


SOFT_LIMITS, HOME_RAD = _read_limits()
# ---- 执行器参数: 论文公式 v2(含负载修正, 2026-09-15) ----
# 公式: kp = I·ω², kd = 2Iζω; I = k_g²·I_rotor + I_load,标称位形
# (用户查证: 论文只用折算转子惯量的前提是"转子惯量占轴惯量主导"——
#  高减速比执行器(G1, kg≥40)成立; 我们的准直驱关节 kg=9/10 时
#  转子占比仅 1-4%, 必须并入负载, ζ 回到 0.5-0.7 正常区间)
# 实测印证: kg=40 的 J3/J4 用论文值改善; kg=9/10 的 J1/J2/J5-7 崩
KP = [120.0, 100.0, 70.0, 126.3, 30.0, 30.0, 30.0]
KD = [7.4, 12.2, 7.5, 8.04, 1.5, 1.5, 1.5]
# effort_limit 用额定(20/20/27/27/7³, v2 设计"额定预算由策略侧执行"):
# 2026-09-16 对照实验——峰值钳位(54/28/10)下高频力矩内容直达物理,
# 腕部 qd 频繁超 vel_lim, 行为惩罚(action_smooth/vel_lim/action_rate)
# 从开局就淹没探索信号(reward −220 起、std 涨到 1.25 发散); 额定钳位
# 实为振幅限制器(低通), 首训开局 +975 健康的实证。TAU_PEAK 仅留档。
TAU_PEAK = [54.0, 54.0, 28.0, 28.0, 10.0, 10.0, 10.0]
# 力矩预算: J1/J2=额定20, J3/J4=峰值27(J4 实测瞬态 12Nm>额定9),
# 腕部=峰值7; 硬件寄存器 tmax 保持 54/28/10 不动(物理余量)
TAU_MAX = [20.0, 20.0, 27.0, 27.0, 7.0, 7.0, 7.0]
# 力矩预算 v3(2026-09-17, 操作员裁定): 额定×1.3 为可接受短时预算,
# 不超过硬件 tmax 寄存器(J3/J4 1.3×额定=35.1>28 → 按物理 28 封顶)。
# w_torque 归一仍用 TAU_MAX(额定): 100~130% 段受二次软惩罚,
# 引导策略优先额定域、必要时可用足 1.3 域。
TAU_BUDGET = [min(1.3 * r, p) for r, p in zip(TAU_MAX, TAU_PEAK)]
# 折算转子惯量(物理量): I = k_g²·I_rotor, 按电机型号逐关节。
# 腕部下限 0.005: 200Hz 显式积分要求 I ≥ kd·dt/2(腕 kd=1.5 → 0.00375),
# 物理真值 0.0018 低于该下限会高频发散 —— 仿真器约束, 非物理修改。
ARMATURE = [0.0122, 0.0122, 0.032, 0.032, 0.005, 0.005, 0.005]


@configclass
class OpenArmZigzagEnvCfg(DirectRLEnvCfg):
    # env: 200Hz 物理 ×4 = 50Hz 策略
    decimation = 4
    episode_length_s = 12.0            # 随机窗口长度 [s](600 步)
    action_space = 7
    observation_space = 67
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)

    # replicate_physics 必须为 True: 本版本 GPU 管线下 False 会触发
    # PhysX fabric device assert; 执行器增益随机化是 Python 侧缓冲, 不受影响
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048, env_spacing=3.0, replicate_physics=True,
        clone_in_fabric=True)

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UsdFileCfg(usd_path=USD_PATH),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={**{n: 0.0 for n in ARM_JOINTS},
                       "openarm_right_joint4": HOME_RAD[3]},
        ),
        actuators={
            "arm": DmMITActuatorCfg(
                joint_names_expr=["openarm_right_joint[1-7]"],
                stiffness={n: v for n, v in zip(ARM_JOINTS, KP)},
                damping={n: v for n, v in zip(ARM_JOINTS, KD)},
                effort_limit={n: v for n, v in zip(ARM_JOINTS, TAU_BUDGET)},
                # 折算转子惯量(物理量): I = k_g²·I_rotor, 按电机型号逐关节
                armature={n: v for n, v in zip(ARM_JOINTS, ARMATURE)},
                ki=[12.0, 12.0, 12.0, 10.0, 8.0, 8.0, 8.0],
                integ_clamp=[10.0, 10.0, 6.0, 8.0, 3.0, 3.0, 3.0],
                sim_dt=1 / 200,
            ),
        },
    )

    # ---- 参考轨迹 ----
    ref_path: str = REF_PATH
    window_s: float = 12.0             # episode 窗口 [s]
    init_noise_q: float = 0.0131       # 重置关节噪声 std [rad](0.75°)
    init_noise_qd: float = 0.05        # 重置关节速度噪声 std [rad/s]

    # ---- 动作 ----
    # 逐关节残差幅值 α = 0.25·τ_budget/kp(论文 §S3: 残差最多动用 25% 力矩预算)
    action_scale: tuple = (0.0417, 0.05, 0.0964, 0.0534,
                           0.0583, 0.0583, 0.0583)
    clearance_rad: float = 0.0524      # 指令软限位内缩 [rad](3°)

    # ---- 观测 ----
    preview_steps: tuple = (2, 4, 8, 16)   # 预览(+0.04/0.08/0.16/0.32s)

    # ---- 奖励权重(设计文档 §5) ----
    # σ 按实测误差量级设定(重力下垂 ~8°=0.14rad): σ 太小 → 指数核饱和,
    # 策略拿不到"减小误差"的梯度(实测 σ=0.03 时 e=0.15 处奖励≈1e-6)
    w_track: float = 2.0
    sigma_track: float = 0.06          # 3.4°(收紧: 精度目标 2° 级)
    w_vel: float = 0.6
    sigma_vel: float = 0.40            # rad/s
    w_bonus: float = 2.0
    w_action_rate: float = 0.10        # ×2(2026-09-17 反粘滑包: 换向段猛修
                                       # 在真实摩擦下激发极限环)
    w_action_smooth: float = 0.10      # ‖a_t − 2a_{t-1} + a_{t-2}‖²(动作空间, ×2)
    w_acc: float = 0.01
    acc_std: float = 100.0             # 实测 qdd_rms 20~70 rad/s²
    w_jerk: float = 0.0                # 废弃: 50Hz 二阶差分噪声被平方放大
    jerk_std: float = 2000.0
    w_torque: float = 0.0015
    w_vel_lim: float = 0.5
    vel_lim: float = 2.0
    w_limit: float = 0.5
    limit_margin: float = 0.1
    w_fail: float = 1.0
    fail_err: float = 0.30             # 17° 终止阈(静态重力下垂~8°, 真机安全
                                       # 由软限位/watchdog/estop 兜底)
    # 尾部加权(2026-09-16, 安全优先: 大偏离问责平方级增长; 与 fail_err
    # 硬终止同向。2026-09-17 曾被置 0 做对照实验(v7/v9), 现恢复至验证过
    # 的 11-47-10 checkpoint 同款配置)
    w_tail: float = 30.0
    e_tail0: float = 0.035             # 尾部起征点 2°(rad)
    w_growth: float = 2.0              # 苗头抑制: 已越 2° 且仍在长的线性问责

    # ---- 域随机化 ----
    dr_kp: float = 0.2                 # ±比例
    dr_kd: float = 0.2
    dr_ki: float = 0.3
    dr_effort: float = 0.1
    dr_armature: tuple = (0.0, 0.1)    # [kg·m²]
    dr_friction: tuple = (0.0, 0.2)
    # 关节摩擦 DR(执行器内注入; 论文 §S3 同类目, GPU 安全实现)
    fric_visc_range: tuple = (0.05, 0.6)   # 粘性 [Nm·s/rad](2026-09-17 反粘滑包:
    # 真机 35~55s 确定性粘滑振, 仿真摩擦太温和是根因, 区间加宽覆盖)
    fric_coul_range: tuple = (0.0, 1.0)    # 库仑 [Nm](上探真实静摩擦)
    # 重力前馈(make_grav_ff.py v2, 真机 URDF 权威动力学=训练 USD 同源):
    # 空=自动找 rl/grav_ff_right.npz, 缺文件自动关闭; 模型≈真机 → DR 围绕
    # 1.0 对称(残余误差来自负载差异/装配公差, 不再假设欠补偿偏置)
    grav_ff_file: str = ""
    grav_ff_scale_range: tuple = (0.8, 1.2)
    use_vel_ref: bool = True
    """v_des=参考速度(真机 rl_exec --vref 成对同步, 部署默认开)。
    2026-09-17 曾被置 False 做对照实验(v7/v9/v10, 其动作 std 发散记录
    见 git 历史/交接), 现恢复至验证过的 11-47-10 checkpoint 同款配置:
    该配置下 verify_deploy 全绿(mean 2.88°/max 6.08°)且真机 60s 跟踪
    与孪生预测吻合(3.02°/6.03°)。"""
    obs_noise_q: float = 0.002         # [rad]
    obs_noise_qd: float = 0.02         # [rad/s]
    delay_prob: float = 0.45           # 动作延迟 1 步概率(反粘滑包目标值;
                                       # 课程化时由 train.py 从起始值 ramp)
    debug_terms: bool = False          # 逐项奖励记录(诊断用)

    # ---- 反粘滑课程化(2026-09-17, 稳步训练法) ----
    # 方差项(摩擦 DR/延迟/力矩预算/熵)从"验证过的稳定配置"在 curr_iters
    # 内线性 ramp 到目标值: 策略先在已验证环境学会跟踪, 再逐步获得鲁棒
    # 性 —— 避免开局即在最大方差下训练导致的震荡/发散。train.py 逐块
    # 按 f=min(1, 迭代/curr_iters) 插值以下起始字段; 0=关课程。
    curr_iters: int = 300
    curr_fric_visc_start: tuple = (0.05, 0.4)
    curr_fric_coul_start: tuple = (0.0, 0.6)
    curr_delay_start: float = 0.3
    curr_budget_frac_start: float = 0.0   # 0=额定预算(验证态)→1=v3 预算

    # ---- 时间轴速度 DR(2026-09-23, "任意速度"需求) ----
    # 每回合抽速度倍率 s: 参考每步前进 s 帧(小数进位), qd_ref(观测/v_des/
    # 速度跟随奖励)与预览帧距全部 ×s —— 观测语义保持时间一致, 策略任务
    # = "同样感知, 更快/更慢地做"。部署侧对应 rl_exec /rl/speed_scale
    # (rl_policy obs 预览已同步 ×qd_scale, 训练↔部署逐项一致)。
    dr_speed_range: tuple = (0.75, 1.25)
    curr_speed_start: tuple = (1.0, 1.0)  # 课程起点(恒速)→ramp 到目标


@configclass
class OpenArmZigzagAgentCfg:
    """rsl_rl PPO 配置(train.py 以 to_dict() 传给 OnPolicyRunner)。"""
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 48        # ~4s / rollout
    max_iterations: int = 3000
    save_interval: int = 100
    experiment_name: str = "openarm_zigzag"
    run_name: str = ""
    actor_hidden: tuple = (256, 256, 128)
    critic_hidden: tuple = (256, 256, 128)
    lr: float = 3.0e-4
    entropy: float = 0.005


class OpenArmZigzagEnv(DirectRLEnv):
    cfg: OpenArmZigzagEnvCfg

    def __init__(self, cfg: OpenArmZigzagEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ---- 关节索引(顺序必须 = joint1..joint7) ----
        arm_ids, arm_names = self.robot.find_joints(
            "openarm_right_joint[1-7]")
        assert arm_names == ARM_JOINTS, "关节顺序异常: %s" % arm_names
        self._arm_ids = arm_ids
        # 夹爪已 fixed → 无夹爪关节; 全关节数应为 7(防御性断言)
        assert self.robot.num_joints == 7, (
            "USD 含 %d 个关节(应为 7, 夹爪应已 fixed 并入 link7)"
            % self.robot.num_joints)
        self._num_joints = self.robot.num_joints

        # ---- 软限位/尺度 ----
        lo = torch.tensor([l[0] for l in SOFT_LIMITS], device=self.device)
        hi = torch.tensor([h[1] for h in SOFT_LIMITS], device=self.device)
        self._q_mid = (lo + hi) / 2
        self._q_half = (hi - lo) / 2
        self._cmd_lo = (lo + self.cfg.clearance_rad)[None]
        self._cmd_hi = (hi - self.cfg.clearance_rad)[None]
        self._tau_max = torch.tensor(TAU_MAX, device=self.device)[None]

        # ---- 参考轨迹 ----
        d = np.load(self.cfg.ref_path, allow_pickle=True)
        self._q_ref = torch.tensor(np.asarray(d["q_ref"]),
                                   dtype=torch.float32,
                                   device=self.device)        # (N,7)
        self._qd_ref = torch.tensor(np.asarray(d["qd_ref"]),
                                    dtype=torch.float32,
                                    device=self.device)
        self._ref_len = len(self._q_ref)
        meta = json_meta(d["meta_json"])
        if isinstance(meta, str):
            import json as _json
            meta = _json.loads(meta)
        self._window = int(self.cfg.window_s / self.step_dt)
        assert self._window < self._ref_len
        self._ph_out = int(float(meta["t_phase_out"]) / self.step_dt)
        self._ph_back = int(float(meta["t_phase_weave_end"]) / self.step_dt)

        # ---- 重力前馈(与部署同一份 grav_ff_right.npz) ----
        ff_path = self.cfg.grav_ff_file or os.path.join(_RL,
                                                        "grav_ff_right.npz")
        if os.path.isfile(ff_path):
            dff = np.load(ff_path, allow_pickle=True)
            gff = np.asarray(dff["gff"], dtype=np.float32)
            assert len(gff) >= self._ref_len, \
                "grav_ff 行数(%d) < 参考帧数(%d), 请重跑 make_grav_ff.py" \
                % (len(gff), self._ref_len)
            self._gff = torch.tensor(gff[:self._ref_len],
                                     device=self.device)
            self._ff_scale = torch.ones(self.num_envs, 1,
                                        device=self.device)
            print("[zigzag_env] 重力前馈: %s (%d 行, DR 缩放 %.2f~%.2f)"
                  % (os.path.basename(ff_path), len(gff),
                     *self.cfg.grav_ff_scale_range))
        else:
            self._gff = None
            print("[zigzag_env] 重力前馈: 未找到 %s → 关闭" % ff_path)

        # ---- 状态缓冲 ----
        self._t_idx = torch.zeros(self.num_envs, dtype=torch.long,
                                  device=self.device)
        # 时间轴速度 DR: 每 env 速度倍率 + 小数进位累加器
        self._spd = torch.ones(self.num_envs, 1, device=self.device)
        self._spd_frac = torch.zeros(self.num_envs, 1, device=self.device)
        self._prev_a = torch.zeros(self.num_envs, 7, device=self.device)
        self._a_rate_ref = torch.zeros(self.num_envs, 7, device=self.device)
        self._a_rate_ref2 = torch.zeros(self.num_envs, 7, device=self.device)
        self._prev_qd = torch.zeros(self.num_envs, 7, device=self.device)
        self._prev_qdd = torch.zeros(self.num_envs, 7, device=self.device)
        self._prev_e_max = torch.zeros(self.num_envs, device=self.device)
        self._q_cmd = torch.zeros(self.num_envs, 7, device=self.device)
        self._last_raw = torch.zeros(self.num_envs, 7, device=self.device)

        self._arm_act = self.robot.actuators["arm"]
        self._dbg = {}

    # ------------------------------------------------------------------
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # 无地面: 默认 Grid USD 走 NVIDIA 云端(网络抖动致二训无法启动),
        # 本地静态网格又被 spawn_ground_plane 的碰撞体检查拦下;
        # 臂固定于台座且无坠落自由度, 地面纯视觉 —— 直接不生成。
        # 需要可视地面时用带碰撞的完整资产重开。
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0,
                                           color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        raw = actions.clamp(-3.0, 3.0)
        # 动作延迟随机化: 依概率复用上一步动作(模拟通信/执行延迟)
        flip = torch.rand(self.num_envs, 1, device=self.device) \
            < self.cfg.delay_prob
        delayed = torch.where(flip, self._prev_a, raw)
        a = torch.tanh(delayed)
        q_ref_t = self._q_ref[self._t_idx]
        act_scale = torch.tensor(self.cfg.action_scale, device=self.device)
        self._q_cmd = torch.clamp(
            q_ref_t + act_scale * a,
            self._cmd_lo, self._cmd_hi)
        self._prev_a = raw.detach()
        self._last_raw = raw.detach()
        # 重力前馈写入执行器(与真机 rl_exec → MIT t_ff 同源同 idx)
        if self._gff is not None:
            self._arm_act._ensure(self.num_envs)
            self._arm_act._ff[:] = self._gff[self._t_idx] * self._ff_scale

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._q_cmd,
                                             joint_ids=self._arm_ids)
        # v_des = 参考速度×速度倍率(与真机 rl_exec MIT 速度通道同源同缩放)
        if self.cfg.use_vel_ref:
            self.robot.set_joint_velocity_target(
                self._qd_ref[self._t_idx] * self._spd, joint_ids=self._arm_ids)

    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple:
        q = self.robot.data.joint_pos[:, self._arm_ids]
        e = (q - self._q_ref[self._t_idx]).abs().max(dim=-1).values
        terminated = e > self.cfg.fail_err
        # 时间轴推进(速度 DR): 平均每步 _spd 帧, 小数进位保留
        self._spd_frac += self._spd
        _step = torch.floor(self._spd_frac)
        self._spd_frac = self._spd_frac - _step
        self._t_idx += _step[:, 0].to(torch.long)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        dt = self.step_dt
        q = self.robot.data.joint_pos[:, self._arm_ids]
        qd = self.robot.data.joint_vel[:, self._arm_ids]
        q_ref = self._q_ref[self._t_idx - 1]
        qd_ref = self._qd_ref[self._t_idx - 1] * self._spd  # 速度跟随目标×s
        tau = self.robot.data.applied_torque[:, self._arm_ids]

        e2 = ((q - q_ref) ** 2).sum(dim=-1)
        ev2 = ((qd - qd_ref) ** 2).sum(dim=-1)
        r = c.w_track * torch.exp(-e2 / (2 * c.sigma_track ** 2))
        r = r + c.w_vel * torch.exp(-ev2 / (2 * c.sigma_vel ** 2))

        # 平滑/稳定惩罚
        qdd = (qd - self._prev_qd) / dt
        jerk = (qdd - self._prev_qdd) / dt
        dr = self._last_raw - self._a_rate_ref
        ddr = self._last_raw - 2.0 * self._a_rate_ref + self._a_rate_ref2
        r = r - c.w_action_rate * (dr ** 2).sum(dim=-1)
        r = r - c.w_action_smooth * (ddr ** 2).sum(dim=-1)
        r = r - c.w_acc * ((qdd / c.acc_std) ** 2).sum(dim=-1)
        r = r - c.w_jerk * ((jerk / c.jerk_std) ** 2).sum(dim=-1)
        r = r - c.w_torque * ((tau / self._tau_max) ** 2).sum(dim=-1)
        r = r - c.w_vel_lim * (
            torch.relu(qd.abs() - c.vel_lim) ** 2).sum(dim=-1)
        margin = torch.minimum(q - self._cmd_lo[0], self._cmd_hi[0] - q)
        r = r - c.w_limit * (
            torch.relu(c.limit_margin - margin) ** 2).sum(dim=-1)
        # 尾部加权 + 苗头抑制(归一化到 fail_err 尺度): v7 教训 —— 未归一化
        # 时 w_tail=30 的惩罚早期百倍于跟踪奖励, 价值函数拟合失败致奖励发散;
        # 归一化后单关节最坏 ~1, 与主奖励同量级, 只改梯度指向不改稳定性
        e_abs = (q - q_ref).abs()
        tail = torch.relu(e_abs - c.e_tail0) / self.cfg.fail_err
        r = r - c.w_tail * (tail ** 2).sum(dim=-1)
        e_max = e_abs.max(dim=-1).values
        growth = torch.relu((e_max - self._prev_e_max) / dt / c.e_tail0)
        r = r - c.w_growth * (tail.max(dim=-1).values * growth)
        self._prev_e_max = e_max.detach()

        if c.debug_terms:
            def _m(x):
                return float(x.mean())
            self._dbg = dict(
                track=_m(c.w_track * torch.exp(
                    -e2 / (2 * c.sigma_track ** 2))),
                vel=_m(c.w_vel * torch.exp(
                    -ev2 / (2 * c.sigma_vel ** 2))),
                action_rate=_m(-c.w_action_rate * (dr ** 2).sum(-1)),
                action_smooth=_m(-c.w_action_smooth * (ddr ** 2).sum(-1)),
                acc=_m(-c.w_acc * ((qdd / c.acc_std) ** 2).sum(-1)),
                jerk=_m(-c.w_jerk * ((jerk / c.jerk_std) ** 2).sum(-1)),
                torque=_m(-c.w_torque * ((tau / self._tau_max) ** 2).sum(-1)),
                vel_lim=_m(-c.w_vel_lim * (
                    torch.relu(qd.abs() - c.vel_lim) ** 2).sum(-1)),
                tail=_m(-c.w_tail * (torch.relu(
                    (q - q_ref).abs() - c.e_tail0) ** 2).sum(-1)),
                limit=_m(-c.w_limit * (
                    torch.relu(c.limit_margin - margin) ** 2).sum(-1)),
                qdd_rms=float(qdd.abs().mean()),
                jerk_rms=float(jerk.abs().mean()),
                fail=float(self.reset_terminated.float().mean()),
            )

        # 失败终止 / 窗口完成质量
        r = torch.where(self.reset_terminated, r - c.w_fail, r)
        bonus = c.w_bonus * torch.exp(-e2 / (2 * c.sigma_track ** 2))
        r = torch.where(self.reset_time_outs & ~self.reset_terminated,
                        r + bonus, r)

        self._a_rate_ref2 = self._a_rate_ref
        self._a_rate_ref = self._last_raw.detach()
        self._prev_qd = qd.detach()
        self._prev_qdd = qdd.detach()
        return r

    def _get_observations(self) -> dict:
        q = self.robot.data.joint_pos[:, self._arm_ids]
        qd = self.robot.data.joint_vel[:, self._arm_ids]
        t = self._t_idx.clamp(max=self._ref_len - 1)
        spd = self._spd[:, 0]                      # (N,)
        parts = [
            (q - self._q_mid) / self._q_half,
            qd / 10.0,
            torch.tanh(self._last_raw),
            (self._q_ref[t] - self._q_mid) / self._q_half,
        ]
        # 预览帧距 ×s: 时间一致的时间前瞻(与部署侧 rl_policy.obs 同口径)
        ks = torch.tensor(self.cfg.preview_steps, device=self.device,
                          dtype=torch.float32)[None, :] * spd[:, None]
        for i, k in enumerate(self.cfg.preview_steps):
            tk = (t + ks[:, i].to(torch.long)).clamp(max=self._ref_len - 1)
            parts.append((self._q_ref[tk] - self._q_mid) / self._q_half)
        parts.append(self._qd_ref[t] * self._spd / 10.0)
        parts.append((t.to(torch.float32) / self._ref_len)[:, None])
        phase = torch.ones(self.num_envs, 3, device=self.device)
        phase[:, :] = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        phase[t < self._ph_out, 0] = 1.0
        phase[(t >= self._ph_out) & (t < self._ph_back), 0] = 0.0
        phase[(t >= self._ph_out) & (t < self._ph_back), 1] = 1.0
        phase[t >= self._ph_back, 1] = 0.0
        phase[t >= self._ph_back, 2] = 1.0
        parts.append(phase)
        obs = torch.cat(parts, dim=-1)
        # 观测噪声(限位归一后加性)
        obs[:, 0:7] += (self.cfg.obs_noise_q / self._q_half) \
            * torch.randn(self.num_envs, 7, device=self.device)
        obs[:, 7:14] += (self.cfg.obs_noise_qd / 10.0) \
            * torch.randn(self.num_envs, 7, device=self.device)
        return {"policy": obs}

    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int]):
        if env_ids is None or len(env_ids) == 0:
            return
        super()._reset_idx(env_ids)
        env_ids_t = torch.as_tensor(env_ids, device=self.device)
        n = len(env_ids)
        # 时间轴速度 DR: 先抽本回合倍率(课程区间), 窗口起点随其收缩
        spd_lo, spd_hi = getattr(self.cfg, "dr_speed_range", (1.0, 1.0))
        self._spd[env_ids_t] = torch.empty(
            n, 1, device=self.device).uniform_(spd_lo, spd_hi)
        self._spd_frac[env_ids_t] = 0.0
        # 随机窗口起点(避开静止首段); 窗口覆盖帧数 = 600×s, 上界随之收缩
        n0 = int(0.5 / self.step_dt)
        win = int(self.cfg.window_s / self.step_dt)
        hi_j = (self._ref_len - 2 - win * self._spd[env_ids_t][:, 0]) \
            .to(torch.long).clamp(min=n0 + 1)
        t0 = n0 + ((hi_j - n0).to(torch.float32)
                   * torch.rand(n, device=self.device)).to(torch.long)
        self._t_idx[env_ids_t] = t0
        # 初态 = 参考初态 + 噪声
        q0 = self._q_ref[t0] + self.cfg.init_noise_q * torch.randn(
            n, 7, device=self.device)
        qd0 = self._qd_ref[t0] + self.cfg.init_noise_qd * torch.randn(
            n, 7, device=self.device)
        # GPU 管线对 env_ids×joint_ids 组合索引写入会触发 device assert,
        # 组装全关节状态后按 env 索引整行写入
        q_full = torch.zeros(n, self._num_joints, device=self.device)
        qd_full = torch.zeros(n, self._num_joints, device=self.device)
        q_full[:, self._arm_ids] = q0
        self.robot.write_joint_state_to_sim(q_full, qd_full, env_ids=env_ids)
        # 注: armature/摩擦的逐环境写入在 GPU 管线会触发 fabric device assert,
        # 故模型层随机化只保留执行器增益/力矩限(下方), 关节阻尼用 USD 值 1.0。
        # 执行器: 积分器清零 + 增益随机化
        self._arm_act.reset(env_ids)
        # 基类增益缓冲同样按克隆前 num_envs=1 分配 → 先扩容再随机化
        if self._arm_act.stiffness.shape[0] < n:
            dev = self.device
            self._arm_act.stiffness = torch.tensor(
                KP, device=dev)[None].repeat(n, 1)
            self._arm_act.damping = torch.tensor(
                KD, device=dev)[None].repeat(n, 1)
            _frac = float(getattr(self.cfg, "curr_budget_frac", 1.0))
            _budget = [r + (b - r) * _frac
                       for r, b in zip(TAU_MAX, TAU_BUDGET)]
            self._arm_act.effort_limit = torch.tensor(
                _budget, device=dev)[None].repeat(n, 1)
            self._arm_act._fric_b = torch.full(
                (n, 7), self.cfg.fric_visc_mid, device=dev)
            self._arm_act._fric_c = torch.full(
                (n, 7), self.cfg.fric_coul_mid, device=dev)
        s = torch.rand(n, 1, device=self.device)
        s_kp = 1.0 + (2 * s - 1) * self.cfg.dr_kp
        s_kd = 1.0 + (2 * torch.rand(n, 1, device=self.device) - 1) \
            * self.cfg.dr_kd
        s_ki = 1.0 + (2 * torch.rand(n, 1, device=self.device) - 1) \
            * self.cfg.dr_ki
        s_ef = 1.0 + (2 * torch.rand(n, 1, device=self.device) - 1) \
            * self.cfg.dr_effort
        base_kp = torch.tensor(KP, device=self.device)[None]
        base_kd = torch.tensor(KD, device=self.device)[None]
        # 课程化力矩预算: frac=0 额定(验证态) → 1=v3 预算(train.py ramp)
        _frac = float(getattr(self.cfg, "curr_budget_frac", 1.0))
        _budget = [r + (b - r) * _frac for r, b in zip(TAU_MAX, TAU_BUDGET)]
        base_ef = torch.tensor(_budget, device=self.device)[None]
        self._arm_act.stiffness[env_ids_t] = base_kp * s_kp
        self._arm_act.damping[env_ids_t] = base_kd * s_kd
        self._arm_act.ki[env_ids_t] = self._arm_act._ki_base[None] * s_ki
        self._arm_act.integ_clamp[env_ids_t] = \
            self._arm_act._ic_base[None] * torch.clamp(s_ki, 0.5, 1.5)
        self._arm_act.effort_limit[env_ids_t] = base_ef * s_ef
        # 重力前馈逐 env 缩放随机化(孪生→真机重力模型误差)
        if self._gff is not None:
            self._ff_scale[env_ids_t] = self._ff_scale.new_empty(
                n, 1).uniform_(*self.cfg.grav_ff_scale_range)
        # 关节摩擦 DR(执行器内): 粘性 + 库仑
        fb = torch.empty(n, 1, device=self.device).uniform_(
            *self.cfg.fric_visc_range)
        fc = torch.empty(n, 1, device=self.device).uniform_(
            *self.cfg.fric_coul_range)
        self._arm_act._fric_b[env_ids_t] = fb
        self._arm_act._fric_c[env_ids_t] = fc
        # 状态缓冲复位
        self._prev_a[env_ids_t] = 0.0
        self._a_rate_ref[env_ids_t] = 0.0
        self._a_rate_ref2[env_ids_t] = 0.0
        self._prev_qd[env_ids_t] = qd0
        self._prev_qdd[env_ids_t] = 0.0
        self._prev_e_max[env_ids_t] = 0.0
        self._last_raw[env_ids_t] = 0.0
        self._q_cmd[env_ids_t] = q0


def json_meta(v):
    """np.load 的 meta_json(0-d ndarray) → dict/str。"""
    if hasattr(v, "item"):
        v = v.item()
    return v
