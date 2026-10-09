#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train.py —— OpenArm 右臂之字形 RL 训练(rsl_rl PPO)
====================================================
用法(isaaclab conda 环境):
    cd 仿真/isaaclab_sim/rl
    python train.py --headless                       # 默认 2048 envs
    python train.py --headless --num_envs 4096 --max_iterations 5000
    python train.py --headless --checkpoint logs/.../model_xxx.pt
产出: logs/rsl_rl/openarm_zigzag/<时间戳>/{model_*.pt, ..., actor_final.npz}
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="OpenArm zigzag RL 训练")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iterations", type=int, default=3000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", type=str, default=None,
                    help="续训 checkpoint 路径")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True                       # 训练一律无头
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- sim 启动后再导入环境侧模块 ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils import configclass  # noqa: E402
from isaaclab_rl.rsl_rl import (RslRlMLPModelCfg,  # noqa: E402
                                RslRlOnPolicyRunnerCfg,
                                RslRlPpoAlgorithmCfg, RslRlVecEnvWrapper)
from openarm_zigzag.zigzag_env import (OpenArmZigzagAgentCfg,  # noqa: E402
                                       OpenArmZigzagEnv,
                                       OpenArmZigzagEnvCfg)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ⚠ 注意: 本文件的 AgentCfg 是**实际生效**的 PPO 配置(OnPolicyRunner 用它);
# zigzag_env.py 里的 OpenArmZigzagAgentCfg 为早期声明、已不参与运行 ——
# 改超参(学习率/熵/γ/网络)改本文件, 不要改那边。
# ⚠ schedule 必须保持 "fixed": adaptive 会在策略收敛后 KL 走低时反复 ×1.5
# 放大学习率 → 把策略推出好盆地, 造成"见顶后系统性衰减"(2026-09-17 两次复现)。
@configclass
class AgentCfg(RslRlOnPolicyRunnerCfg):
    seed: int = args.seed
    device: str = "cuda:0"
    num_steps_per_env: int = 48            # 4s / rollout
    max_iterations: int = args.max_iterations
    save_interval: int = 100
    experiment_name: str = "openarm_zigzag"
    run_name: str = ""
    obs_groups: dict = {"actor": ["policy"], "critic": ["policy"]}
    actor: RslRlMLPModelCfg = RslRlMLPModelCfg(
        class_name="MLPModel", hidden_dims=[256, 256, 128],
        activation="elu", obs_normalization=False)
    critic: RslRlMLPModelCfg = RslRlMLPModelCfg(
        class_name="MLPModel", hidden_dims=[256, 256, 128],
        activation="elu", obs_normalization=False)
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        num_learning_epochs=5, num_mini_batches=4,
        # schedule fixed(2026-09-17): adaptive 在策略收敛后 KL 持续低于目标
        # → 学习率反复 ×1.5 放大 → 把策略推出好盆地, 两次独立复现的
        # "见顶后系统性衰减"主因。固定 LR + checkpoint 验证选取。
        learning_rate=3.0e-4, schedule="fixed",
        gamma=0.999, lam=0.95, entropy_coef=0.005,
        desired_kl=0.01, max_grad_norm=1.0,
        value_loss_coef=1.0, use_clipped_value_loss=True,
        clip_param=0.2)


def export_actor_npz(runner: OnPolicyRunner, out_path: str) -> None:
    """把 actor MLP 导出为纯 numpy 权重(npz), 供 MuJoCo 验证与真机使用。
    只导 MLP 干线(确定性均值输出), 分布参数(std)不导出。
    层键形如 mlp.<i>.weight (i = 0,2,4,6, 中间夹激活层)。"""
    import re
    import numpy as np
    sd = runner.alg.actor.state_dict()
    layers = {}
    for k, v in sd.items():
        m = re.fullmatch(r"mlp\.(\d+)\.(weight|bias)", k)
        if m:
            tag = "W" if m.group(2) == "weight" else "b"
            layers["%s%s" % (tag, m.group(1))] = v.detach().cpu().numpy()
    n_layers = len([k for k in layers if k.startswith("W")])
    np.savez(out_path, activation=b"elu", **layers)
    print("[导出] actor → %s (%d 层, dims %s)" % (
        out_path, n_layers,
        [tuple(layers["W%s" % i].shape)
         for i in sorted({k[1:] for k in layers if k.startswith("W")},
                         key=int)]))


def main():
    env_cfg = OpenArmZigzagEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed

    agent_cfg = AgentCfg()
    from datetime import datetime
    log_dir = os.path.abspath(os.path.join(
        "logs", "rsl_rl", agent_cfg.experiment_name,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    os.makedirs(log_dir, exist_ok=True)

    env = OpenArmZigzagEnv(env_cfg)
    env = RslRlVecEnvWrapper(env)

    cfg_dict = agent_cfg.to_dict()
    # IsaacLab 源码版 rsl_rl cfg 与 pip rsl-rl-lib 5.4.2 之间有键名差异:
    # 新版 'stochastic' → 5.4.2 需要显式 distribution_cfg(GaussianDistribution)
    import copy
    for k, stoch in (("actor", True), ("critic", False)):
        dd = {kk: copy.deepcopy(vv) for kk, vv in cfg_dict[k].items()
              if kk in ("class_name", "hidden_dims", "activation",
                        "obs_normalization", "distribution_cfg")}
        want_stoch = bool(cfg_dict[k].get("stochastic", stoch)) or stoch
        if want_stoch and not dd.get("distribution_cfg"):
            # init_std=0.3: 冷启动时动作噪声 ≈±4.3°(tanh 后 ×0.25rad),
            # 叠加重力下垂 ~8° 仍低于 17° 失败阈; std=1.0 会一上来就失败
            dd["distribution_cfg"] = {"class_name": "GaussianDistribution",
                                      "init_std": 0.3,
                                      "std_type": "scalar",
                                      "learn_std": True}
        if not want_stoch:
            dd.pop("distribution_cfg", None)
        cfg_dict[k] = dd

    runner = OnPolicyRunner(env, cfg_dict, log_dir=log_dir,
                            device=args.device)
    if args.checkpoint:
        runner.load(args.checkpoint)
        print("[续训] %s" % args.checkpoint)

    try:
        # 分段 learn: 每 50 迭代打印逐项奖励(诊断)并推进课程(稳步训练法:
        # 方差项从验证过的稳定配置线性 ramp 到反粘滑目标, 熵同步退火)。
        # 本版 rsl_rl learn() 返回 None, 按块数循环。
        env_u = env.unwrapped
        env_u.cfg.debug_terms = True
        chunk = 50
        ci = float(getattr(env_u.cfg, "curr_iters", 0) or 0)
        done = 0
        while done < args.max_iterations:
            n = min(chunk, args.max_iterations - done)
            if ci > 0:
                f = min(1.0, done / ci)

                def lp(a, b):
                    return (a[0] + (b[0] - a[0]) * f,
                            a[1] + (b[1] - a[1]) * f)
                env_u.cfg.fric_visc_range = lp(
                    env_u.cfg.curr_fric_visc_start, (0.05, 0.6))
                env_u.cfg.fric_coul_range = lp(
                    env_u.cfg.curr_fric_coul_start, (0.0, 1.0))
                env_u.cfg.delay_prob = env_u.cfg.curr_delay_start + \
                    (0.45 - env_u.cfg.curr_delay_start) * f
                env_u.cfg.curr_budget_frac = \
                    env_u.cfg.curr_budget_frac_start + \
                    (1.0 - env_u.cfg.curr_budget_frac_start) * f
                env_u.cfg.dr_speed_range = lp(
                    env_u.cfg.curr_speed_start, (0.75, 1.25))
                runner.alg.entropy_coef = 0.005 + (0.001 - 0.005) * f
                print("[curr] f=%.2f visc=%s coul=%s delay=%.2f "
                      "budget_frac=%.2f speed=%s entropy=%.4f"
                      % (f,
                         tuple(round(v, 3) for v in
                               env_u.cfg.fric_visc_range),
                         tuple(round(v, 3) for v in
                               env_u.cfg.fric_coul_range),
                         env_u.cfg.delay_prob,
                         env_u.cfg.curr_budget_frac,
                         tuple(round(v, 2) for v in
                               env_u.cfg.dr_speed_range),
                         runner.alg.entropy_coef), flush=True)
            runner.learn(n)
            done += n
            dbg = getattr(env_u, "_dbg", None)
            if dbg:
                print("[dbg] %s" % " ".join(
                    "%s=%.3f" % (k, v) for k, v in sorted(dbg.items())),
                    flush=True)
    finally:
        runner.save(os.path.join(log_dir, "model_final.pt"))
        export_actor_npz(runner, os.path.join(log_dir, "actor_final.npz"))
    simulation_app.close()


if __name__ == "__main__":
    main()
