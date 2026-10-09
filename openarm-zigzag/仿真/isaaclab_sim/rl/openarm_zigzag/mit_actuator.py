# -*- coding: utf-8 -*-
"""达妙 MIT 执行器复刻(IsaacLab 显式执行器模型)
==============================================
复刻 real/common/ws/src/openarm_zero_hw write() 的控制律:
  tau = kp*(p_des - p) + kd*(0 - v) + I
  err_f = alpha*err + (1-alpha)*err_f        (EMA 平滑)
  I += ki * err_f * dt,  钳位 ±integ_clamp;  |err_f|>err_clear 时清零
差异: 真机在电机坐标系(s=±1)做误差, s=-1 的 J7 等价于 URDF 系负反馈,
统一用 URDF 系; 真机 100Hz EMA α=0.3, 此处按物理时间常数换算 α(dt)。
"""

import torch
from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass


class DmMITActuator(IdealPDActuator):
    """理想 PD + 软件积分前馈(与 zero_offset_hw 一致)。"""

    def __init__(self, cfg: "DmMITActuatorCfg", joint_names, joint_ids,
                 num_envs, device, **kwargs):
        super().__init__(cfg, joint_names, joint_ids, num_envs, device,
                         **kwargs)
        self._n = self.stiffness.shape[1]
        self._ki_base = torch.tensor(cfg.ki, dtype=torch.float32,
                                     device=device)
        self._ic_base = torch.tensor(cfg.integ_clamp, dtype=torch.float32,
                                     device=device)
        # 注意: 执行器构造发生在场景克隆之前(num_envs=1), 逐环境缓冲必须
        # 惰性扩容 —— 由 _ensure() 在 compute/reset 时按实际批量分配
        self._alpha = 1.0 - float(torch.exp(torch.tensor(
            -cfg.sim_dt / cfg.ema_tau)))
        self._err_clear = cfg.err_clear
        # 关节摩擦 DR(注入执行器力矩, GPU 管线安全; 等效 PhysX 关节摩擦)
        self._fric_b = torch.full((1, self._n), cfg.fric_visc_mid, device=device)
        self._fric_c = torch.full((1, self._n), cfg.fric_coul_mid, device=device)
        self._v_lin = cfg.fric_v_lin
        self._integ = torch.zeros(1, self._n, device=device)
        self._err_f = torch.zeros(1, self._n, device=device)
        # 重力前馈缓冲 [Nm](环境每步写入 gff[t_idx]·逐 env 缩放; 与真机
        # rl_exec → zero_offset_hw MIT t_ff 通道同源, 见 make_grav_ff.py)
        self._ff = torch.zeros(1, self._n, device=device)
        self.ki = self._ki_base.repeat(1, 1)
        self.integ_clamp = self._ic_base.repeat(1, 1)

    def _ensure(self, n: int):
        if self._integ.shape[0] < n:
            dev = self._integ.device
            self._integ = torch.zeros(n, self._n, device=dev)
            self._err_f = torch.zeros(n, self._n, device=dev)
            self.ki = self._ki_base.repeat(n, 1)
            self.integ_clamp = self._ic_base.repeat(n, 1)
            self._fric_b = self._fric_b[:1].repeat(n, 1)
            self._fric_c = self._fric_c[:1].repeat(n, 1)
            self._ff = self._ff[:1].repeat(n, 1)

    def compute(self, control_action, joint_pos: torch.Tensor,
                joint_vel: torch.Tensor):
        self._ensure(joint_pos.shape[0])
        # 期望速度未下发(纯位置指令) → 0, 与真机 vel_commands_=0 一致
        if control_action.joint_velocities is not None:
            e_v = control_action.joint_velocities - joint_vel
        else:
            e_v = -joint_vel
        e_p = control_action.joint_positions - joint_pos
        # EMA 平滑误差; 大偏差清零(换段/中止防积分饱和)
        self._err_f.mul_(1.0 - self._alpha).add_(e_p * self._alpha)
        big = self._err_f.abs() > self._err_clear
        if bool(big.any()):
            self._integ[big] = 0.0
            self._err_f[big] = 0.0
        self._integ += self.ki * self._err_f * self.cfg.sim_dt
        self._integ.clamp_(-self.integ_clamp, self.integ_clamp)
        # 关节摩擦(逐环境随机): 粘性 b·v + 平滑库仑 c·tanh(v/v_lin)
        tau_fric = -(self._fric_b * joint_vel
                     + self._fric_c * torch.tanh(joint_vel / self._v_lin))
        self.computed_effort = (self.stiffness * e_p + self.damping * e_v
                                + self._integ + tau_fric + self._ff)
        self.applied_effort = self._clip_effort(self.computed_effort)
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    def reset(self, env_ids=None):
        if env_ids is None:
            self._integ.zero_()
            self._err_f.zero_()
            self.ki[:] = self._ki_base
            self.integ_clamp[:] = self._ic_base
            return
        # 摩擦系数由环境在随机化时写入, reset 不覆盖
        ids = torch.as_tensor(env_ids, device=self._integ.device).reshape(-1)
        if ids.numel() == 0:
            return
        self._ensure(int(ids.max().item()) + 1)
        self._integ[ids] = 0.0
        self._err_f[ids] = 0.0
        self.ki[ids] = self._ki_base
        self.integ_clamp[ids] = self._ic_base


@configclass
class DmMITActuatorCfg(IdealPDActuatorCfg):
    """DM-MIT(位置+积分前馈)执行器配置。"""

    class_type: type = DmMITActuator
    ki: list = None
    """积分增益 [rad/(s·Nm)] —— 真机 ki=[12,12,12,10,8,8,8]"""
    integ_clamp: list = None
    """积分钳位 [Nm] —— 真机 ±[10,10,6,8,3,3,3]"""
    ema_tau: float = 0.0281
    """误差 EMA 时间常数 [s](100Hz 时 α≈0.3)"""
    err_clear: float = 0.5
    """误差清零阈值 [rad]"""
    sim_dt: float = 0.005
    """物理步长 [s](积分器步进)"""
    fric_visc_mid: float = 0.2
    """粘性摩擦中值 [Nm·s/rad](重置时按环境 cfg 的范围随机化)"""
    fric_coul_mid: float = 0.3
    """库仑摩擦中值 [Nm]"""
    fric_v_lin: float = 0.02
    """库仑平滑线性区速度 [rad/s](0.05→0.02: 更尖锐 breakaway 逼近真机
    粘滑特性, 2026-09-17 反粘滑包)"""
