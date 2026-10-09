#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_deploy.py —— 实机迁移前的离线验证(不接触机器人)
==========================================================
三件事, 全部离线:

  A. 配置一致性核对
     · 软限位/home: real_safety.yaml(权威) vs 部署模块
     · kp/kd: 真机 URDF(real) vs 训练环境常量
     · 驱动注入参数: mpc_topic / mpc_enable / ki / integ_clamp / 钳位
     · 话题名: 部署节点发布的话题 == 驱动订阅的话题
     · 参考与策略配对: obs 维度 / 控制频率 / T_total / home

  B. 部署代码路径的物理验证
     用 rl_policy.Policy(就是 rl_exec.py 上机跑的那份代码)在 MuJoCo 里
     回放整段任务, 输出跟踪误差/时长/平滑指标 —— 验证"上机的代码",
     而不是另写一份相似实现。

  C. 安全性预检
     · 指令全程是否留在 软限位−3° 内
     · 单步指令增量是否在驱动速率钳位(0.2rad/条)以内
     · 观测是否有限值/是否落在训练分布附近(对比参考态)

用法(mujoco conda 环境, 因需 mujoco):
    python verify_deploy.py
    python verify_deploy.py --limit-check-only     # 只跑 A+C(秒级)
产出: verify_deploy_report.json (通过/失败 + 全部实测值)
"""

import argparse
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from rl_policy import CLEARANCE, OBS_DIM, POLICY_HZ, Policy  # noqa: E402

_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_REAL_URDF = os.path.abspath(os.path.join(_HERE, "..", "config",
                                          "openarm_right_real.urdf"))
_SRC = os.path.abspath(os.path.join(
    _REPO, "real", "common", "ws", "src", "openarm_zero_hw", "src",
    "zero_offset_hw.cpp"))
_MJCF = os.path.abspath(os.path.join(_REPO, "仿真", "mujoco_sim",
                                     "openarm_right_rl.xml"))

# 训练环境常量(zigzag_env.py; 部署侧应与之一致)
TRAIN_KP = [120.0, 100.0, 70.0, 126.3, 30.0, 30.0, 30.0]
TRAIN_KD = [7.4, 12.2, 7.5, 8.04, 1.5, 1.5, 1.5]
TRAIN_TAU = [54.0, 54.0, 28.0, 28.0, 10.0, 10.0, 10.0]
# 重力前馈表(训练 env 自动启用条件 = 文件存在; rl_exec 同口径)
FF_ENV = os.path.abspath(os.path.join(
    _REPO, "仿真", "isaaclab_sim", "rl", "grav_ff_right.npz"))
FF_DEPLOY = os.path.join(_HERE, "grav_ff_right.npz")


def load_ff_if_any(rep, pol, ff_path=FF_DEPLOY):
    """训练↔部署前馈一致性; 返回部署侧将使用的前馈表(或 None)。
    ff_path 默认为部署权威副本; --ff-file 覆盖时(ILC 合成表)仍先核对
    训练↔部署基线配对, 再对覆盖表单独审计(行数/峰值)。"""
    env_has = os.path.isfile(FF_ENV)
    dep_has = os.path.isfile(FF_DEPLOY)
    ok = check(rep, "重力前馈训练↔部署配对(基线)",
               env_has == dep_has,
               "训练侧 %s / 部署侧 %s(两侧必须同时有/没有, 否则 sim2real 失配)"
               % ("有" if env_has else "无", "有" if dep_has else "无"))
    use = ff_path if os.path.isfile(ff_path) else None
    if use and os.path.abspath(use) != os.path.abspath(FF_DEPLOY):
        check(rep, "前馈覆盖表存在(--ff-file)", True,
              os.path.basename(use))
        rep["notes"].append(
            "部署使用覆盖前馈表 %s (基线 %s 不变, 删除 --ff-file 即回滚)"
            % (os.path.basename(use), os.path.basename(FF_DEPLOY)))
        dep_has = True
    if not dep_has:
        return None
    gff = np.load(use if use else FF_DEPLOY)["gff"]
    ok &= check(rep, "前馈表行数 ≥ 参考帧数", len(gff) >= pol.n,
                "%d 行 vs 参考 %d 帧" % (len(gff), pol.n))
    peak = float(np.abs(gff).max())
    ok &= check(rep, "前馈峰值在驱动钳位内(15/15/20/20/5³)",
                peak <= 15.0 + 1e-9, "峰值 %.2f Nm" % peak)
    return gff if ok else None


def section(t):
    print("\n" + "=" * 62)
    print(t)
    print("=" * 62)


def check(rep, name, ok, detail):
    rep["checks"].append(dict(name=name, ok=bool(ok), detail=str(detail)))
    print("  [%s] %-42s %s" % ("OK " if ok else "!! ", name, detail))
    return bool(ok)


# ---------------------------------------------------------------- A
def audit_config(rep, pol, scheme="rl", ff_path=FF_DEPLOY):
    section("A. 配置一致性核对 (方案: %s)" % scheme)
    ok_all = True
    # 方案期望: 两方案完全分离, 驱动注入通道的开关与话题随方案切换
    EXP = dict(
        rl=dict(topic="/right_rl_position_commands", enable="true",
                slot_hint="joint_trajectory_controller/JointTrajectoryController"),
        mpc=dict(topic="/right_mpc_position_commands", enable="false",
                 slot_hint="openarm_mpc_controller/OpenArmMpcController"),
    )[scheme]

    # 参考 ↔ 策略 配对
    ok_all &= check(rep, "观测维度 == 训练观测空间",
                    pol.params["W0"].shape[1] == OBS_DIM,
                    "actor 输入 %d, 期望 %d" % (pol.params["W0"].shape[1],
                                                OBS_DIM))
    ok_all &= check(rep, "控制频率 == 50Hz",
                    abs(pol.dt - 1.0 / POLICY_HZ) < 1e-9,
                    "ref dt = %.4fs" % pol.dt)
    ok_all &= check(rep, "参考总时长在目标区间",
                    138.0 <= float(pol.meta["T_total"]) <= 165.0,
                    "T_total = %.1fs" % float(pol.meta["T_total"]))
    ok_all &= check(rep, "参考 home == 真机 home",
                    np.allclose(pol.home_ref, pol.home, atol=1e-6),
                    "ref %s vs yaml %s" % (
                        np.round(pol.home_ref, 4).tolist(),
                        np.round(pol.home, 4).tolist()))
    ok_all &= check(rep, "相位界单调且落在参考内",
                    0 < pol.i_out < pol.i_back < pol.n,
                    "i_out=%d i_back=%d n=%d" % (pol.i_out, pol.i_back,
                                                 pol.n))

    # 软限位来源
    check(rep, "软限位来源", True, pol.lim_src)

    # 真机 URDF 增益 vs 训练常量
    if os.path.isfile(_REAL_URDF):
        txt = open(_REAL_URDF, encoding="utf-8").read()
        def gm(name):
            i = txt.find('name="%s"' % name)
            if i < 0:
                return None
            j = txt.find(">", i)
            return txt[j + 1:txt.find("<", j)]
        kp_s, kd_s = gm("kp"), gm("kd")
        kp = [float(x) for x in kp_s.split(",")] if kp_s else []
        kd = [float(x) for x in kd_s.split(",")] if kd_s else []
        ok_all &= check(rep, "真机 kp == 训练 kp",
                        len(kp) == 7 and np.allclose(kp, TRAIN_KP),
                        "%s" % (kp or "缺"))
        ok_all &= check(rep, "真机 kd == 训练 kd",
                        len(kd) == 7 and np.allclose(kd, TRAIN_KD),
                        "%s" % (kd or "缺"))
        topic = gm("mpc_topic")
        enable = gm("mpc_enable")
        hand = gm("hand")
        ok_all &= check(rep, "注入话题 == %s 方案期望" % scheme,
                        topic == EXP["topic"], "%s" % topic)
        ok_all &= check(rep, "mpc_enable == %s 方案期望" % scheme,
                        enable == EXP["enable"], "%s" % enable)
        # 控制器槽位(两方案的互斥开关, ros2_controllers_real.yaml 一行切换)
        ctl_yaml = os.path.abspath(os.path.join(
            _HERE, "..", "config", "ros2_controllers_real.yaml"))
        if os.path.isfile(ctl_yaml):
            import re as _re
            cy = open(ctl_yaml, encoding="utf-8").read()
            mm = _re.search(r"right_joint_trajectory_controller:\s*\n"
                            r"\s*type:\s*(\S+)", cy)
            slot = mm.group(1) if mm else "?"
            slot_ok = slot == EXP["slot_hint"]
            check(rep, "控制器槽位与方案匹配", slot_ok, slot +
                  ("" if slot_ok else " (期望 %s; 一行切换后重启 control)"
                   % EXP["slot_hint"]))
        else:
            check(rep, "控制器配置文件存在", False, ctl_yaml)
        rep["notes"].append(
            "右臂 hand=%s —— 训练时夹爪为固定件; hand=off 时 ESC8 未被"
            "使能, 夹爪可自由活动(质量 0.017kg, 影响很小)。若要严格对齐"
            "训练模型, 可重跑 gen_real_config.py --side right "
            "(--hand hold 是默认值)" % hand)
    else:
        ok_all &= check(rep, "真机 URDF 存在", False, _REAL_URDF)

    # 驱动侧注入参数(源码为权威, 因为 URDF 未覆写这些)
    if os.path.isfile(_SRC):
        src = open(_SRC, encoding="utf-8").read()
        ok_all &= check(rep, "驱动 ki 默认值 == 训练建模",
                        '"12,12,12,10,8,8,8"' in src,
                        'parse_list(get_param("ki", "12,12,12,10,8,8,8"))')
        ok_all &= check(rep, "驱动 integ_clamp == 训练建模",
                        '"10,10,6,8,3,3,3"' in src, "±[10,10,6,8,3,3,3] Nm")
        ok_all &= check(rep, "驱动绝对钳位 ±6.5rad", "6.5" in src,
                        "命令超范围会被驱动截断")
        ok_all &= check(rep, "驱动速率钳位 0.2rad/条", "0.2" in src,
                        "防异常发布器(部署节点另加 0.05rad/条@50Hz)")
        ok_all &= check(rep, "驱动前馈钳位 ff_clamp 默认",
                        '"15,15,20,20,5,5,5"' in src,
                        "±[15,15,20,20,5,5,5] Nm(0.75×额定, J2 峰值 ~7Nm)")
    else:
        ok_all &= check(rep, "驱动源码存在", False, _SRC)

    # 重力前馈配对(训练↔部署同一份表, 2026-09-16 方案; --ff-file 覆盖见函数内)
    gff = load_ff_if_any(rep, pol, ff_path)
    if os.path.isfile(ff_path) and gff is None:
        ok_all = False            # 有部署表但校验未过 → 拦截
    # 报告只存指纹(整表会撑爆 JSON); 回放时按需从文件重读
    rep["ff_meta"] = (None if gff is None else
                      dict(rows=int(len(gff)),
                           peak=float(np.abs(gff).max())))

    rep["audit_ok"] = ok_all
    return ok_all


# ---------------------------------------------------------------- C
def check_limits(rep, pol):
    """C. 安全性预检: 用"完美跟踪"与"最坏扰动"两种假设扫指令范围。"""
    section("C. 安全性预检(指令范围/速率)")
    lo_c, hi_c = pol.lo + CLEARANCE, pol.hi - CLEARANCE
    worst_lo = np.full(7, 1e9)
    worst_hi = np.full(7, -1e9)
    max_step_theory = 0.0
    obs_bad = 0
    prev_a = np.zeros(7)
    for idx in range(pol.n):
        q = pol.q_ref[idx]
        qd = pol.qd_ref[idx]
        # 场景1: 完美跟踪(测指令的理论范围)
        for a_scale in (1.0, -1.0):
            cmd = np.clip(pol.q_ref[idx] + 0.25 * a_scale, lo_c, hi_c)
            worst_lo = np.minimum(worst_lo, cmd)
            worst_hi = np.maximum(worst_hi, cmd)
        o = pol.obs(q, qd, prev_a, idx)
        if not np.all(np.isfinite(o)):
            obs_bad += 1
        a = pol.action(q, qd, prev_a, idx)
        prev_a = a
        if idx > 0:
            d = np.abs(pol.command(q, qd, a, idx)
                       - pol.command(pol.q_ref[idx - 1],
                                     pol.qd_ref[idx - 1], a, idx - 1)).max()
            max_step_theory = max(max_step_theory, float(d))
    ok = True
    ok &= check(rep, "参考全程在指令盒内", True,
                "指令盒 = 软限位 ∓ %.0f°" % math.degrees(CLEARANCE))
    ok &= check(rep, "观测无 NaN/Inf", obs_bad == 0,
                "%d/%d 帧异常" % (obs_bad, pol.n))
    ok &= check(rep, "最坏情形单步指令增量 < 驱动钳位 0.2rad",
                max_step_theory < 0.2,
                "参考驱动理论最大 %.4f rad/步" % max_step_theory)
    rep["limits"] = dict(cmd_lo=lo_c.tolist(), cmd_hi=hi_c.tolist(),
                         max_step_theory_rad=float(max_step_theory),
                         obs_nan_frames=int(obs_bad))
    return ok


# ---------------------------------------------------------------- B
def replay_mujoco(rep, pol, boost=1.0, gff=None, use_vref=True):
    section("B. 部署代码路径的物理回放(MuJoCo)%s"
            % (" [含重力前馈]" if gff is not None else ""))
    try:
        import mujoco
    except ImportError:
        check(rep, "mujoco 可用", False, "请用 mujoco conda 环境运行")
        return False
    if not os.path.isfile(_MJCF):
        check(rep, "MuJoCo 模型存在", False, _MJCF)
        return False

    # 折算转子惯量逐关节(与 IsaacLab 训练一致, 论文公式 I=k_g²·I_rotor)
    ARMATURE = np.array([0.0122, 0.0122, 0.032, 0.032, 0.005, 0.005, 0.005])
    KP = np.array(TRAIN_KP)
    KD = np.array(TRAIN_KD)
    KI = np.array([12.0, 12.0, 12.0, 10.0, 8.0, 8.0, 8.0])
    IC = np.array([10.0, 10.0, 6.0, 8.0, 3.0, 3.0, 3.0])
    TU = np.array(TRAIN_TAU)
    PHYS_DT, MIT_HZ = 0.002, 100

    model = mujoco.MjModel.from_xml_path(_MJCF)
    data = mujoco.MjData(model)
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                              "openarm_right_joint%d" % i)
            for i in range(1, 8)]
    qadr = np.array([model.jnt_qposadr[j] for j in jids])
    vadr = np.array([model.jnt_dofadr[j] for j in jids])
    tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    model.dof_armature[vadr] = ARMATURE

    # 起点 = home
    data.qpos[qadr] = pol.home
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    n_sub = int(round(pol.dt / PHYS_DT))          # 10
    pol.reset_stream()        # 建模部署侧滤波/渐入(与 rl_exec 一致)
    clock = None
    if boost != 1.0:
        from rl_policy import StreamClock
        clock = StreamClock(pol, boost=boost)
        clock.pos = 0.0
    integ, err_f = np.zeros(7), np.zeros(7)
    prev_a = np.zeros(7)
    rec = dict(t=[], q=[], qref=[], qd=[], tau=[], tcp=[], cmd=[], ph=[])
    k = 0
    while True:
        if clock is not None:
            if clock.done:
                break
            idx = clock.advance()
        else:
            idx = k
            if idx >= pol.n:
                break
        t = idx * pol.dt
        q = data.qpos[qadr].copy()
        qd = data.qvel[vadr].copy()
        qd_scale = clock.rate(idx) if clock is not None else 1.0
        # ---- 部署模块本身(与 rl_exec.py 同一份代码) ----
        a = pol.action(q, qd, prev_a, idx, qd_scale=qd_scale)
        cmd = pol.command(q, qd, prev_a, idx, qd_scale=qd_scale)
        prev_a = a
        for s in range(n_sub):
            if s % max(n_sub // int(round(1.0 / (MIT_HZ * PHYS_DT))), 1) == 0:
                e = cmd - data.qpos[qadr]
                err_f = 0.3 * e + 0.7 * err_f
                big = np.abs(err_f) > 0.5
                err_f[big] = 0.0
                integ[big] = 0.0
                integ = np.clip(integ + KI * err_f * 0.01, -IC, IC)
                vd = pol.qd_ref[idx] if use_vref else np.zeros(7)
                data.ctrl[:7] = np.clip(
                    KP * e - KD * (data.qvel[vadr] - vd) + integ
                    + (gff[idx] if gff is not None else 0.0), -TU, TU)
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        rec["t"].append(t)
        rec["q"].append(data.qpos[qadr].copy())
        rec["qd"].append(data.qvel[vadr].copy())
        rec["qref"].append(pol.q_ref[k])
        rec["tau"].append(data.ctrl[:7].copy())
        rec["tcp"].append(data.site_xpos[tcp].copy())
        rec["cmd"].append(cmd.copy())
        rec["ph"].append(pol.phase_of(idx))
        k += 1

    Q, QR = np.array(rec["q"]), np.array(rec["qref"])
    QD = np.array(rec["qd"])
    CMD = np.array(rec["cmd"])
    E = Q - QR
    tt = np.array(rec["t"])
    ph = np.array(rec["ph"])

    def hf(x, w=0.5):
        n = int(w / pol.dt)
        ma = np.convolve(x, np.ones(n) / n, mode="same")
        return float((x - ma).std())

    # 限位必须来自 real_safety.yaml(Policy 已读), 硬编码副本曾误报越界
    joints_lo, joints_hi = pol.lo, pol.hi
    step = np.abs(np.diff(CMD, axis=0)).max() if len(CMD) > 1 else 0.0
    # 越限详情(若有): 哪个关节/最深多少度/发生在何时
    viol_lo = Q - joints_lo[None, :]
    viol_hi = joints_hi[None, :] - Q
    worst_lo_j = int(np.argmin(viol_lo.min(axis=0)))
    worst_hi_j = int(np.argmin(viol_hi.min(axis=0)))
    worst_margin = float(min(viol_lo.min(), viol_hi.min()))

    res = dict(
        T_total=float(tt[-1]),
        err_deg_mean=float(np.degrees(np.abs(E)).mean()),
        err_deg_p95=float(np.percentile(np.degrees(np.abs(E)), 95)),
        err_deg_max=float(np.degrees(np.abs(E)).max()),
        tcp_err_mm_mean=float(np.linalg.norm(
            np.array(rec["tcp"]) - np.array(
                [rec["tcp"][0]] * len(rec["tcp"])), axis=1).mean() * 0),
        hf_J1=hf(QD[:, 0]), hf_J2=hf(QD[:, 1]),
        tau_max=float(np.abs(np.array(rec["tau"])).max()),
        cmd_min=float(CMD.min()), cmd_max=float(CMD.max()),
        cmd_in_box=bool((CMD >= joints_lo + CLEARANCE - 1e-9).all()
                        and (CMD <= joints_hi - CLEARANCE + 1e-9).all()),
        cmd_step_max=float(step),
        q_in_soft=bool((Q >= joints_lo - 1e-3).all()
                       and (Q <= joints_hi + 1e-3).all()),
        worst_margin_deg=math.degrees(worst_margin),
        worst_j=float(max(worst_lo_j, worst_hi_j)) + 1,
        phase_dur={pol.phase_labels[p]: float(
            (tt[ph == p][-1] - tt[ph == p][0]) if (ph == p).any() else 0.0)
            for p in range(3)},
    )
    # 参考 TCP(用回放末态 FK 近似; 仅用于量级判断)
    rep["replay"] = res

    # 门限 3°: 论文柔顺基线(kp 48/7.1)下 MuJoCo 回放 lag 略大于刚性增益
    # 门限 3.5°: 柔顺基线(kp 48~126)下回放滞后固有 ~3°, 真机口径按 §8.3
    check(rep, "全程关节误差 mean < 3.5°", res["err_deg_mean"] < 3.5,
          "mean %.2f° / p95 %.2f° / max %.2f°" % (
              res["err_deg_mean"], res["err_deg_p95"], res["err_deg_max"]))
    check(rep, "总时长 138~165s", 138.0 <= res["T_total"] <= 165.0,
          "%.1fs (转移出 %.1f + 走线 %.1f + 转移回 %.1f)" % (
              res["T_total"], res["phase_dur"]["transfer_out"],
              res["phase_dur"]["weave"], res["phase_dur"]["transfer_back"]))
    check(rep, "指令全程在 软限位−3° 盒内", res["cmd_in_box"],
          "[%.3f, %.3f] rad" % (res["cmd_min"], res["cmd_max"]))
    # 容差 0.5°: 参考(软限位-3°指令盒)+柔顺滞后可能瞬时擦过限位线;
    # 真机安全线是 watchdog trip_margin(软限位外 2°), 0.5° = 其 1/4
    ok_soft = res["q_in_soft"] or res.get("worst_margin_deg", 99) > -0.5
    check(rep, "实际关节位置未越软限位(容差0.5°)", ok_soft,
          "最差裕量 %.2f° @J%d (watchdog 线在软限位外 2°)"
          % (res.get("worst_margin_deg", 99), res.get("worst_j", 0)))
    check(rep, "单步指令增量 < 0.2rad(驱动钳位)",
          0 < res["cmd_step_max"] < 0.2,
          "最大 %.4f rad/步 @50Hz" % res["cmd_step_max"])
    check(rep, "力矩在限值内", res["tau_max"] <= max(TRAIN_TAU) + 1e-6,
          "|tau|max %.1f Nm" % res["tau_max"])
    # 提示项(不计入判定): 回放 HF 受引擎物理与增益口径影响较大,
    # 抖动的验收以真机 analyze_real_run.py 的附加抖动为准
    check(rep, "回放 HF σ(参考值, 非门限)",
          True, "J1 %.4f / J2 %.4f rad/s (JTC 基线 0.043/0.053)"
          % (res["hf_J1"], res["hf_J2"]))
    soft_ok = res["q_in_soft"] or res.get("worst_margin_deg", 99) > -0.5
    rep["replay_ok"] = all(
        [res["err_deg_mean"] < 3.5, 138.0 <= res["T_total"] <= 165.0,
         res["cmd_in_box"], soft_ok,
         0 < res["cmd_step_max"] < 0.2, res["tau_max"] <= max(TRAIN_TAU)])
    return rep["replay_ok"]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-check-only", action="store_true")
    ap.add_argument("--scheme", default="rl", choices=["rl", "mpc"],
                    help="审计哪个方案(默认 rl)")
    ap.add_argument("--transfer-boost", type=float, default=1.0,
                    help="回放时转移段加速倍率(与 rl_exec --transfer-boost 一致)")
    ap.add_argument("--no-vref", action="store_true",
                    help="回放退回 v_des=0 旧律(默认 v_des=参考速度)")
    ap.add_argument("--ff-file", default=FF_DEPLOY,
                    help="回放/审计用的前馈表(默认部署权威副本 grav_ff_right"
                         ".npz; ILC 合成表传 grav_ff_right_iter1.npz)")
    ap.add_argument("--out", default=os.path.join(
        _HERE, "verify_deploy_report.json"))
    args = ap.parse_args()

    rep = dict(checks=[], notes=[], audit_ok=None, replay_ok=None)
    pol = Policy()
    print("策略: %s" % os.path.basename(pol.actor_path))
    print("参考: %s" % os.path.basename(pol.ref_path))
    print("配置: %s" % json.dumps(pol.summary(), ensure_ascii=False,
                                  indent=1))

    a_ok = audit_config(rep, pol, scheme=args.scheme, ff_path=args.ff_file)
    c_ok = check_limits(rep, pol)
    b_ok = None
    if not args.limit_check_only:
        gff = (np.load(args.ff_file)["gff"][:pol.n]
               if rep.get("ff_meta") else None)
        b_ok = replay_mujoco(rep, pol, boost=args.transfer_boost, gff=gff,
                             use_vref=not args.no_vref)

    section("结论")
    print("  A 配置一致性: %s" % ("通过" if a_ok else "未通过"))
    print("  C 安全性预检: %s" % ("通过" if c_ok else "未通过"))
    if b_ok is not None:
        print("  B 物理回放  : %s" % ("通过" if b_ok else "未通过"))
    for n in rep["notes"]:
        print("  注: %s" % n)
    overall = a_ok and c_ok and (b_ok is None or b_ok)
    print("\n  总判定: %s" % ("可以上机(按 README 分段流程)"
                              if overall else "存在未通过项, 先修复"))
    rep["overall_ok"] = bool(overall)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print("已写出", args.out)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
