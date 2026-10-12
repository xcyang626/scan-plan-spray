# -*- coding: utf-8 -*-
"""平衡策略下蹲极限扫描: height_cmd 从 0.74 缓降到 0.30, 记录骨盆实际跟踪.

目的: 确定 GR00T 平衡策略(height_cmd 通道)能稳定保持的最低骨盆高度 —— V2 蹲姿
喷涂带向下扩到 0m 的第一步(策略地板). 不注入任何手臂/腰部轨迹, 纯策略站立下蹲.

用法(conda: GR00T .venv_sim_cpu):
  python exec/groot_height_sweep.py [--down 0.30] [--rate 0.02] [--headless]
输出: 每 0.05m 指令档的骨盆实测(均值/最小/是否保持), 结算安全地板建议值.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

GROOT_ROOT = os.environ.get("GROOT_ROOT", "/home/yang/桌面/轨迹规划/GR00T-WholeBodyControl")
sys.path.insert(0, GROOT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from groot_spray_runner import build_controller  # noqa: E402  (平衡控制器构造)

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--down", type=float, default=0.30, help="扫描最低指令(m)")
    ap.add_argument("--rate", type=float, default=0.02, help="下降速率(m/s)")
    ap.add_argument("--hold", type=float, default=3.0, help="起始/结束保持(s)")
    a = ap.parse_args()
    import mujoco

    c, _ = build_controller(headless=True)
    h0 = float(c.config["height_cmd"])
    log = []
    fallen = False
    t0 = time.time()

    def cmd_at(t):
        if t < a.hold:
            return h0
        t2 = t - a.hold
        if t2 < (h0 - a.down) / a.rate:
            return h0 - a.rate * t2
        return a.down

    start = time.time()
    while time.time() - start < 200:
        t = time.time() - t0
        with c.cmd_lock:
            c.control_dict["height_cmd"] = cmd_at(t)
            cur = dict(c.control_dict)
        # 腿+腰 PD(策略输出), 双臂零位 PD —— 与 runner 站立期一致
        leg_tau = c.pd_control(
            c.target_dof_pos, c.data.qpos[7:7 + c.config["num_actions"]],
            c.config["kps"], np.zeros_like(c.config["kps"]),
            c.data.qvel[6:6 + c.config["num_actions"]], c.config["kds"])
        c.data.ctrl[:c.config["num_actions"]] = leg_tau
        arm_tau = c.pd_control(
            np.zeros(c.n_joints - c.config["num_actions"], dtype=np.float32),
            c.data.qpos[7 + c.config["num_actions"]:7 + c.n_joints],
            np.full(c.n_joints - c.config["num_actions"], 100.0),
            np.zeros(c.n_joints - c.config["num_actions"]),
            c.data.qvel[6 + c.config["num_actions"]:6 + c.n_joints],
            np.full(c.n_joints - c.config["num_actions"], 0.5))
        c.data.ctrl[c.config["num_actions"]:] = arm_tau
        mujoco.mj_step(c.model, c.data)
        c.counter += 1
        if c.counter % c.config["control_decimation"] == 0:
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
        pz = float(c.data.qpos[2])
        log.append((t, cur["height_cmd"], pz))
        if pz < 0.28 or not np.isfinite(pz):
            fallen = True
            print(f"[sweep] ⚠ 失稳 @t={t:.1f}s cmd={cur['height_cmd']:.2f} 骨盆={pz:.3f}",
                  flush=True)
            break
        if t > a.hold + (h0 - a.down) / a.rate + a.hold:
            break
        time.sleep(max(0, c.model.opt.timestep - (time.time() - start) % c.model.opt.timestep)
                   if False else 0)
    # 分档统计: 每 0.05m 指令档的骨盆实测
    arr = np.array(log)
    print(f"[sweep] 结束: {'失稳' if fallen else '完整'}, 共 {len(arr)} 步", flush=True)
    print(f"{'指令档':>6} {'骨盆均值':>8} {'骨盆最小':>8} {'样本':>5} 跟踪差", flush=True)
    ok_floor = a.down
    for lo in np.arange(round(h0, 2), round(a.down - 1e-9, 2) - 0.05, -0.05):
        m = (arr[:, 1] >= lo - 0.025) & (arr[:, 1] < lo + 0.025)
        if m.sum() < 20:
            continue
        pz_m, pz_min = arr[m, 2].mean(), arr[m, 2].min()
        dev = abs(pz_m - arr[m, 1].mean())
        keep = "保持" if dev < 0.08 else "跟不上!"
        if dev < 0.08:
            ok_floor = min(ok_floor, float(lo))
        print(f"{lo:6.2f} {pz_m:8.3f} {pz_min:8.3f} {int(m.sum()):5d} "
              f"{dev*1000:5.0f}mm {keep}", flush=True)
    print(f"[sweep] 结论: 策略可稳定保持的最低指令档 ≈ {ok_floor:.2f}m "
          f"(建议工作地板 = 该值+0.05 裕量; {'本次失稳, 以上为失稳前数据' if fallen else '未失稳'})",
          flush=True)
    np.save("/tmp/height_sweep.npy", arr)
    json.dump(dict(fallen=fallen, ok_floor=ok_floor, h0=h0, down=a.down),
              open(os.path.join(BASE, "output", "height_sweep.json"), "w",
                   encoding="utf8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
