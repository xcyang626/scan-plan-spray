#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""play.py —— 加载训练 checkpoint 滚参考轨迹, 出评估指标与图
============================================================
用法:
    python play.py --checkpoint logs/rsl_rl/openarm_zigzag/<run>/model_final.pt
    python play.py --checkpoint ... --num_episodes 5      # 多窗口统计
    python play.py --render                               # 开视口(可选)
产出: <checkpoint目录>/play_report.npz + play_report.png, 终端打印:
    跟踪误差(关节/TCP) / 实测各相位用时 / 速度 HF 抖动(对比基线) / 平滑指标
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="OpenArm zigzag RL 评估")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=3)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--full", action="store_true",
                    help="整段评估(窗口=156s, 接近全 mission)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils import configclass  # noqa: E402
from isaaclab_rl.rsl_rl import (RslRlMLPModelCfg,  # noqa: E402
                                RslRlOnPolicyRunnerCfg,
                                RslRlPpoAlgorithmCfg, RslRlVecEnvWrapper)
from openarm_zigzag.zigzag_env import (  # noqa: E402
    ARM_JOINTS, SOFT_LIMITS, OpenArmZigzagEnv, OpenArmZigzagEnvCfg)

# FK 用(纯 numpy, 无 ROS 依赖)
_ISA = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ISA)
sys.path.insert(0, os.path.join(_ISA, "..", "openarm_sim"))
from openarm_sim.kinematics import UrdfChain  # noqa: E402


@configclass
class AgentCfg(RslRlOnPolicyRunnerCfg):
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 48
    max_iterations: int = 1
    save_interval: int = 1000
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
        class_name="PPO", num_learning_epochs=5, num_mini_batches=4,
        learning_rate=3.0e-4, schedule="adaptive", gamma=0.999, lam=0.95,
        entropy_coef=0.005, desired_kl=0.01, max_grad_norm=1.0,
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2)


def main():
    env_cfg = OpenArmZigzagEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    if args.full:
        # 窗口自动跟随参考总长(留 1s 余量; 新标定参考 150.5s)
        import json as _json
        _d = np.load(env_cfg.ref_path, allow_pickle=True)
        _m = _d["meta_json"].item()
        _T = float(_json.loads(_m)["T_total"]) if isinstance(_m, str) \
            else float(_m["T_total"])
        env_cfg.window_s = _T - 1.0
        env_cfg.episode_length_s = _T - 1.0
    env_cfg.init_noise_q = 0.0
    env_cfg.init_noise_qd = 0.0
    env_cfg.delay_prob = 0.0
    env_cfg.obs_noise_q = 0.0
    env_cfg.obs_noise_qd = 0.0
    env = OpenArmZigzagEnv(env_cfg)
    env = RslRlVecEnvWrapper(env)

    import copy
    cfg_dict = AgentCfg().to_dict()
    for k in ("actor", "critic"):
        dd = {kk: copy.deepcopy(vv) for kk, vv in cfg_dict[k].items()
              if kk in ("class_name", "hidden_dims", "activation",
                        "obs_normalization", "distribution_cfg")}
        if k == "actor" and not dd.get("distribution_cfg"):
            dd["distribution_cfg"] = {"class_name": "GaussianDistribution",
                                      "init_std": 0.3, "std_type": "scalar",
                                      "learn_std": True}
        if k == "critic":
            dd.pop("distribution_cfg", None)
        cfg_dict[k] = dd
    runner = OnPolicyRunner(env, cfg_dict, log_dir=None, device="cuda:0")
    runner.load(os.path.abspath(args.checkpoint))
    policy = runner.get_inference_policy(device="cuda:0")

    # 参考/FK(评估口径)
    d = np.load(env_cfg.ref_path, allow_pickle=True)
    q_ref = np.asarray(d["q_ref"])
    phase_ref = np.asarray(d["phase"])
    meta = d["meta_json"].item() if hasattr(d["meta_json"], "item") \
        else d["meta_json"]
    if isinstance(meta, (bytes, str)):
        import json as _json
        meta = _json.loads(meta)
    chain = UrdfChain(os.path.join(_ISA, "openarm_right.urdf"),
                      tip_link="openarm_right_hand_tcp")
    tcp_off = float(meta["tcp_off"])

    N = args.num_envs
    rec_q, rec_qref, rec_qd, rec_t, rec_ph, rec_a = [], [], [], [], [], []
    ep_terminated = 0
    obs = env.get_observations()
    for step in range(int(env.max_episode_length)):
        with torch.no_grad():
            act = policy(obs)
        obs, rew, dones, infos = env.step(act)
        rec_q.append(env.unwrapped.robot.data.joint_pos[
            :, env.unwrapped._arm_ids].cpu().numpy().copy())
        rec_qd.append(env.unwrapped.robot.data.joint_vel[
            :, env.unwrapped._arm_ids].cpu().numpy().copy())
        t_idx = (env.unwrapped._t_idx - 1).clamp(min=0).cpu().numpy()
        rec_qref.append(q_ref[t_idx].copy())
        rec_t.append(t_idx.copy())
        rec_ph.append(phase_ref[t_idx].copy())
        rec_a.append(act.cpu().numpy().copy())
        if bool(dones.any()):
            to = infos.get("time_outs")
            n_to = int(np.asarray(to.cpu()).sum()) if to is not None else 0
            ep_terminated += int(dones.sum()) - n_to
        if (step % 100) == 0:
            u2 = env.unwrapped
            qn = u2.robot.data.joint_pos[:, u2._arm_ids]
            qdn = u2.robot.data.joint_vel[:, u2._arm_ids]
            import math as _m
            print("step %d/%d t=%.1fs |q|ref-err %.2fdeg |qd|max %.2f rew %.3f"
                  % (step, env.max_episode_length, step * env.unwrapped.step_dt,
                     float((qn - u2._q_ref[u2._t_idx.clamp(max=u2._ref_len-1)]).abs().max()) * 57.3,
                     float(qdn.abs().max()), float(rew.mean())),
                  flush=True)
            if not bool(torch.isfinite(qn).all()) or not bool(torch.isfinite(qdn).all()):
                print("[play] NaN/Inf 检出 @step", step, flush=True)
                break

    Q = np.stack(rec_q)              # (T, N, 7)
    Qref = np.stack(rec_qref)
    QD = np.stack(rec_qd)
    A = np.stack(rec_a)
    PH = np.stack(rec_ph)
    T = np.stack(rec_t)
    dt = env.unwrapped.step_dt

    err = Q - Qref
    err_rms = np.sqrt((err ** 2).mean())
    err_max = np.abs(err).max()
    # TCP 误差(FK)
    tcp = np.array([chain.fk(qq, tcp_off)[1] for qq in Q[:, 0]])
    tcp_ref = np.array([chain.fk(qq, tcp_off)[1] for qq in Qref[:, 0]])
    tcp_err = np.linalg.norm(tcp - tcp_ref, axis=1)
    # 平滑指标: 速度 HF sigma(减 0.5s 滑动均值), 与 JTC 基线 0.043/0.053 对比
    def hf_sigma(x):
        w = int(0.5 / dt)
        ma = np.convolve(x, np.ones(w) / w, mode="same")
        return float((x - ma).std())
    hf = {("J%d" % (j + 1)): hf_sigma(QD[:, 0, j]) for j in range(2)}
    # 实测相位覆盖
    phases = PH[:, 0]
    out = dict(
        err_rms=err_rms, err_max=float(err_max),
        tcp_err_mean=float(tcp_err.mean()), tcp_err_max=float(tcp_err.max()),
        hf_J1=hf["J1"], hf_J2=hf["J2"],
        action_rms=float(np.sqrt((A ** 2).mean())),
        phase_covered=[float((phases == p).mean()) for p in range(3)],
        window_s=float(env.max_episode_length * dt),
    )
    print("== 评估 ==")
    print("关节跟踪误差 RMS %.3f° / max %.2f°" % (
        out["err_rms"] * 57.2958, out["err_max"] * 57.2958))
    print("TCP 误差 mean %.2fmm / max %.2fmm" % (
        out["tcp_err_mean"] * 1000, out["tcp_err_max"] * 1000))
    print("速度 HF σ: J1 %.4f J2 %.4f rad/s (JTC 基线 0.043/0.053)"
          % (out["hf_J1"], out["hf_J2"]))
    print("动作 RMS %.3f | 窗口 %.1fs | 失败终止 %d 次" % (
        out["action_rms"], out["window_s"], ep_terminated))

    out_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    np.savez_compressed(os.path.join(out_dir, "play_report.npz"),
                        Q=Q[:, 0], Qref=Qref[:, 0], QD=QD[:, 0], PH=PH[:, 0],
                        T=T[:, 0], A=A[:, 0], tcp=tcp, tcp_ref=tcp_ref,
                        meta=np.array(meta))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.arange(len(Q)) * dt
        fig, ax = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
        for j in range(7):
            ax[0].plot(t, np.degrees(Q[:, 0, j]), lw=0.7)
            ax[0].plot(t, np.degrees(Qref[:, 0, j]), lw=0.7, ls="--",
                       alpha=0.4)
        ax[0].set_ylabel("q [deg] 实线=策略 虚线=参考")
        for j in range(7):
            ax[1].plot(t, err[:, 0, j] * 57.2958, lw=0.6)
        ax[1].set_ylabel("跟踪误差 [deg]")
        ax[2].plot(t[1:], np.linalg.norm(np.diff(Q[:, 0], axis=0), axis=1)
                   / dt, lw=0.5)
        ax[2].set_ylabel("|qd| [rad/s]")
        ax[2].set_xlabel("t [s]")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "play_report.png"), dpi=130)
        print("已写出", os.path.join(out_dir, "play_report.png"))
    except Exception as e:  # noqa: BLE001
        print("绘图失败(不影响指标):", e)
    simulation_app.close()


if __name__ == "__main__":
    main()
