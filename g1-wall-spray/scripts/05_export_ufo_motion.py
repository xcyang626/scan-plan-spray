# -*- coding: utf-8 -*-
"""步骤5: 把规划轨迹导出为 UFO/MimicLite 跟踪策略的参考动作时间序列.

输出格式(any4hdmi-g1-lafan manifest 合同):
  npz 仅含 qpos (T, 36) float32, 30Hz (manifest timestep=1/30)
  36 维 = root pos3 + quat wxyz4 + 左腿6 + 右腿6 + 腰3 + 左臂7 + 右臂7

流程:
  1. 骨盆高度参考恒定(站姿单区0.77) — 避免 IK 逐点自由骨盆的抖动
  2. MJCF 内重解 IK: g1-mode_13_15.xml 与 URDF 有 mm 级连杆原点差异, 以 URDF 解为种子,
     在 MJCF 中顺序热启动(保持冗余分支连续), 目标=腕心(0.40, y, z)+枪轴指向;
     工作区边缘点允许目标横向内松 ≤4cm
  3. 时间参数化: 实际腕心路径按弧长积分, 速度按段类(绿0.10/红0.30/黄0.15/过渡0.06 m/s),
     行首行尾停留 0.25s, 重采样 30Hz
  4. 全身补全: 腿=骨盆高度的平面站姿IK; 左臂=init站姿常数; root quat=朝墙单位
  5. 相位: 初始保持1.5s(g1.json init站姿) -> 2.5s过渡 -> 跟踪 -> 2.0s回位 -> 保持1.0s
  6. qpos 零相位滑动平均(7帧), 再逐帧 mujoco 贴地校正 root_tz
运行环境: conda mujoco
腰部三自由度(v4解锁): q=[腰yaw, 腰roll, 腰pitch, 右臂7] 共10维全程直通, 无8维映射
"""
import numpy as np
import mujoco
import os, json, csv, math
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nn_ik_hook import nn_ik

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = os.environ.get("MOTION_VER", "v2_uniform_slow")            # 每版独立文件夹
OUT_BASE = os.environ.get("G1W_OUT", os.path.join(BASE, "output"))
IN = os.environ.get("G1W_IN", OUT_BASE)
OUT = os.path.join(OUT_BASE, "ufo_motion", VER)
os.makedirs(OUT, exist_ok=True)

# MJCF 路径: 机器人登记表(station/robot_registry.json)可换型号, 环境变量 G1W_MJCF 优先
MJCF = os.environ.get("G1W_MJCF")
if not MJCF:
    try:
        import json as _json
        _reg = _json.load(open(os.path.join(BASE, "station", "robot_registry.json"),
                               encoding="utf8"))
        _m = _reg[os.environ.get("G1W_ROBOT", "g1")]["mjcf"]
        MJCF = _m if os.path.isabs(_m) else os.path.join(BASE, _m)
    except Exception:
        MJCF = os.path.join(BASE, "models", "g1_mjcf", "g1-mode_13_15.xml")
DT = 1.0 / 30.0
HOLD0_S, APPROACH_S, RETREAT_S, HOLD1_S = 1.5, 2.5, 2.0, 1.0
DWELL_S = 0.25
# 用户: 绿黄红速度区别取消, 整体降50% -> 统一 0.05 m/s
SPEED = {"green": 0.05, "red": 0.05, "yellow": 0.05, "transition": 0.05}
SMOOTH_WIN = 3     # 轻度平滑: 节点已顺序热启动连续, 重度平滑会抹坏枪轴指向
CROUCH_PZ, STAND_PZ = 0.63, 0.77     # 骨盆高度参考(区常数)
D_ROBOT, D_WRIST = 0.74, 0.40   # 标称站位/腕平面(腰锁定保证带 0.70~0.78m)
GUN_NOTE = "墙内面放于 root 前方 +x 0.74m 处(机器人面向+x, 腰锁定站位保证带 0.70~0.78m)"

INIT = dict(hip_pitch=-0.1, knee=0.3, ankle_pitch=-0.2,
            l_shoulder_pitch=0.2, l_shoulder_roll=0.2, l_elbow=0.6,
            r_shoulder_pitch=0.2, r_shoulder_roll=-0.2, r_elbow=0.6)

CHAIN = [((0.0, 0, 0), (0, 0, 0), 'z'), ((-0.0039635, 0, 0.035), (0, 0, 0), 'x'),
         ((0.0, 0, 0.019), (0, 0, 0), 'y'),
         ((0.0039563, -0.10021, 0.23778), (-0.27931, 5.4949e-05, 0.00019159), 'y'),
         ((0.0, -0.038, -0.013831), (0.27925, 0, 0), 'x'),
         ((0.0, -0.00624, -0.1032), (0, 0, 0), 'z'),
         ((0.015783, 0, -0.080518), (0, 0, 0), 'y'),
         ((0.100, -0.00188791, -0.010), (0, 0, 0), 'x'),
         ((0.038, 0, 0), (0, 0, 0), 'y'), ((0.046, 0, 0), (0, 0, 0), 'z')]


def fk_wrist_local(q10):
    def rot(ax, a):
        c, s = math.cos(a), math.sin(a)
        if ax == 'x':
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        if ax == 'y':
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R = np.eye(3); p = np.zeros(3)
    for i, (xyz, rpy, ax) in enumerate(CHAIN):
        p = p + R @ np.array(xyz)
        Rr = R @ rot('x', rpy[0]) @ rot('y', rpy[1]) @ rot('z', rpy[2])
        R = Rr @ rot(ax, q10[i])
    return p


def leg_ik_planar(pelvis_h):
    L1, L2, ANKLE_Z = 0.34051, 0.31757, 0.038
    r = np.clip((pelvis_h - 0.1027) - ANKLE_Z, 0.30, L1 + L2 - 0.004)
    b = np.arccos(np.clip((L1 * L1 + r * r - L2 * L2) / (2 * L1 * r), -1, 1))
    g = np.arccos(np.clip((L1 * L1 + L2 * L2 - r * r) / (2 * L1 * L2), -1, 1))
    hip = -b
    return hip, np.pi - g, -(hip + np.pi - g)


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * x * (x * (x * 6 - 15) + 10)


def bias_tilt(tgt, bias_deg):
    """把枪轴名义指向向墙面法向(+x)旋转 bias_deg(Rodrigues, 轴=t×n), 提高墙面夹角.
    v5: 本轨迹60°名义行恰在"实测≥60°"门上, IK/平滑残差必然压出59.x°帧; 抬2°后
    实测≥60.8°, 与名义指向 dot=cos(2°±容差)≥0.998 仍远超0.965门, 喷涂更垂直.
    60°行取的是可行档最大θ, 偏置方向是倾角减小(30°→28°), 朝可行带内部, 必可行."""
    b = math.radians(bias_deg)
    t = np.asarray(tgt, float)
    t = t / np.linalg.norm(t)
    u = np.cross(t, np.array([1.0, 0.0, 0.0]))
    nu = np.linalg.norm(u)
    if nu < 1e-9:
        return t.copy()                      # 已垂直墙面, 无可抬
    u /= nu
    return t * math.cos(b) + np.cross(u, t) * math.sin(b)


def main():
    model = mujoco.MjModel.from_xml_path(MJCF)
    data = mujoco.MjData(model)
    WRIST_BODY = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    foot_geom = [g for g in range(model.ngeom)
                 if "ankle_roll" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                                      model.geom_bodyid[g]) or "")]

    ARM, YAW = "right", 60
    t = np.load(os.path.join(OUT_BASE, "trajectory.npz"), allow_pickle=True)
    q_urdf = t["q"].astype(float)
    cls, kind, zone, mode = t["cls"], t["kind"], t["zone"], t["mode"]
    y_r, z_r = t["rob_y_r"].astype(float), t["rob_z"].astype(float)
    axis_t = t["axis"].astype(float)
    N = len(q_urdf)
    print(f"[载入] 轨迹 {N} 点", flush=True)

    base36 = np.zeros(36); base36[3] = 1.0
    base36[7:13] = [INIT["hip_pitch"], 0, 0, INIT["knee"], INIT["ankle_pitch"], 0]
    base36[13:19] = base36[7:13]
    base36[22:29] = [INIT["l_shoulder_pitch"], INIT["l_shoulder_roll"], 0, INIT["l_elbow"], 0, 0, 0]
    base36[29:36] = [INIT["r_shoulder_pitch"], INIT["r_shoulder_roll"], 0, INIT["r_elbow"], 0, 0, 0]

    # ---------- 1. 骨盆高度参考: 站姿单区恒定(v4: 不再有蹲姿/回零位拼接) ----------
    pz_ref = np.full(N, STAND_PZ)
    print(f"[骨盆参考] 站姿单区恒定 {STAND_PZ} m", flush=True)

    # ---------- 2. MJCF 内重解 IK (顺序热启动 + 连续性优先) ----------
    # 用户更新: 腰部三自由度全部解锁(与 URDF 规划一致), roll/pitch 限幅收紧至±5°
    ARM_JNAMES = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
                  "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                  "right_shoulder_yaw_joint", "right_elbow_joint",
                  "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
    arm_jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JNAMES]
    arm_qadr = [model.jnt_qposadr[i] for i in arm_jid]
    arm_dadr = [model.jnt_dofadr[i] for i in arm_jid]
    alo = np.array([model.jnt_range[i][0] for i in arm_jid])
    ahi = np.array([model.jnt_range[i][1] for i in arm_jid])
    # 用户更新: 腰 roll/pitch 限位收紧至 ±5°, 覆盖 MJCF 原生范围 (索引1/2=roll/pitch)
    _lim = float(os.environ.get("G1W_WAIST_RP", "0")) or 0.0872665
    alo[1] = max(alo[1], -_lim); ahi[1] = min(ahi[1], _lim)
    alo[2] = max(alo[2], -_lim); ahi[2] = min(ahi[2], _lim)

    def mj_pose(q10):                   # 10维自由变量: [腰yaw, 腰roll, 腰pitch, 右臂7]
        qq = base36.copy()
        qq[arm_qadr] = np.clip(q10, alo, ahi)
        data.qpos[:] = qq
        mujoco.mj_forward(model, data)
        return (data.xpos[WRIST_BODY].copy(),
                data.xmat[WRIST_BODY].reshape(3, 3)[:, 0].copy())

    def ik_mj(p_target, ax_target, q_seed, iters=500, pos_tol=0.004, ax_tol=0.05):
        q = np.clip(np.asarray(q_seed, float), alo, ahi)
        lam2 = 0.0036
        nv = len(arm_dadr)
        J = np.zeros((6, nv))
        jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
        for it in range(iters):
            p, ax = mj_pose(q)
            e = np.concatenate([p_target - p, ax_target - ax])
            if np.linalg.norm(e[:3]) < pos_tol and np.linalg.norm(e[3:]) < ax_tol:
                return q, True
            mujoco.mj_jacBody(model, data, jacp, jacr, WRIST_BODY)
            for k in range(nv):
                J[:3, k] = jacp[:, arm_dadr[k]]
                J[3:, k] = np.cross(jacr[:, arm_dadr[k]], ax)
            dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), e)
            q = np.clip(q + dq, alo, ahi)
        p, ax = mj_pose(q)
        ok = np.linalg.norm(p_target - p) < pos_tol and np.linalg.norm(ax_target - ax) < ax_tol
        return q, ok

    q_mj = np.zeros((N, 10)); conv = np.zeros(N, bool)
    wri_act = np.zeros((N, 3))                    # 基座系(骨盆原点)实际腕心
    rng = np.random.default_rng(3)
    prev = np.asarray(q_urdf[0], float)
    # MC 工作区样本作为高质量种子(腰部三自由度放开限幅下的真实可行构型)
    ws_npz = np.load(os.path.join(IN, "g1_workspace.npz"))
    mc_all = np.concatenate([ws_npz[f"{ARM}_{m}_y{YAW}_endsub"].astype(float)[:, :3]
                             for m in ("stand", "crouch")], 0)
    mc_allq = np.concatenate([ws_npz[f"{ARM}_{m}_y{YAW}_qsub"].astype(float)[:, :10]
                              for m in ("stand", "crouch")], 0)

    def mc_nn(p):
        return int(np.argmin(((mc_all - p) ** 2).sum(1)))
    n_relax, hops, n_loose = 0, [], 0
    t_ik0 = None
    import time as _time
    t_ik0 = _time.time()
    n_incr, n_nn = 0, 0
    for i in range(N):
        p_t = np.array([D_ROBOT - D_WRIST, y_r[i], z_r[i] - pz_ref[i]])   # 相对骨盆 (0.34)
        best = None
        # NN-IK 提速插槽: 训练好的模型放 station/nn_ik/g1_nn_ik.npz 即生效(只作种子,
        # 精度仍由后续容差保证); 未部署 -> nn_ik 返回 None, 零改动回退增量播种
        q_nn = nn_ik(np.array([D_ROBOT - D_WRIST, y_r[i], z_r[i]]), axis_t[i])  # 训练同坐标: 离地高z
        if q_nn is not None:
            qf, ok = ik_mj(p_t, axis_t[i], np.asarray(q_nn, float),
                           iters=60, pos_tol=0.004, ax_tol=0.05)
            if ok:
                jmp = float(np.max(np.abs(qf - prev)))
                if jmp < 0.35:
                    best = (jmp, qf, 0.0); n_nn += 1
        # 增量播种加速(智能加速): 03已输出连续URDF解, MJCF与之仅mm级差异 ->
        # 以其为种子小步修正(iters<=60早停), 一步收敛且跳变小即收; 不达标走下方阶梯
        qf, ok = ik_mj(p_t, axis_t[i], q_urdf[i], iters=60, pos_tol=0.004, ax_tol=0.05)
        if ok:
            jmp = float(np.max(np.abs(qf - prev)))
            if jmp < 0.35:
                best = (jmp, qf, 0.0)
                n_incr += 1
        if best is None:
            # 严格顺序优先: 仅用上一节点解作种子, 阶梯放宽容差也不换构型分支
            # (15mm 位置代价 << 分支翻转 2~3rad 造成的速度尖峰与枪轴抹坏)
            for _pt, _at in ((0.004, 0.05), (0.008, 0.08), (0.015, 0.12)):
                qf, ok = ik_mj(p_t, axis_t[i], prev, iters=800, pos_tol=_pt, ax_tol=_at)
                if ok:
                    jmp = float(np.max(np.abs(qf - prev)))
                    if jmp < 0.35:
                        best = (jmp, qf, 0.0)
                        break
                    if best is None or jmp < best[0]:
                        best = (jmp, qf, 0.0)
        if best is None:
            for relax in (0.0, 0.012, 0.025, 0.04):
                p2 = p_t.copy()
                if relax > 0 and abs(p_t[1]) > 1e-4:
                    p2[1] = p_t[1] - np.sign(p_t[1]) * relax
                seeds = [prev, q_urdf[i]] if relax == 0 else [prev]
                if relax == 0:
                    seeds.append(mc_allq[mc_nn(p_t)])
                    seeds.append(prev + rng.uniform(-0.05, 0.05, 10))
                    seeds.append(prev + rng.uniform(-0.15, 0.15, 10))
                for sd in seeds:
                    qf, ok = ik_mj(p2, axis_t[i], sd)
                    if ok:
                        jump = float(np.max(np.abs(qf - prev)))
                        if best is None or jump < best[0]:
                            best = (jump, qf, relax)
                        if jump < 0.25:
                            break
                if best is not None and best[0] < 0.6:
                    break
        if best is not None:
            jump, qf, relax = best
            q_mj[i] = qf; conv[i] = True; prev = qf
            if relax > 0:
                n_relax += 1
            if jump >= 0.6:
                hops.append(i)
        else:
            # 边缘容差档: 工作区角落点放宽容差到 1cm/0.10, 目标向内/向下微松
            done2 = False
            p_t2 = np.array([D_ROBOT - D_WRIST, y_r[i], z_r[i] - pz_ref[i]])
            for ti, (dz, dy) in enumerate(((0.0, 0.0), (-0.02, 0.012), (-0.04, 0.025), (-0.06, 0.04))):
                p2 = p_t2.copy()
                if abs(y_r[i]) > 1e-4:
                    p2[1] = y_r[i] - np.sign(y_r[i]) * dy
                p2[2] += dz - (0.02 * ti if z_r[i] > 1.15 else 0.0)
                for sd in (q_urdf[i], prev):
                    qf, ok2 = ik_mj(p2, axis_t[i], sd, iters=800, pos_tol=0.010, ax_tol=0.10)
                    if ok2:
                        q_mj[i] = qf; conv[i] = True
                        if abs(y_r[i]) > 1e-4 and dy > 0:
                            n_relax += 1
                        n_loose += 1
                        done2 = True
                        break
                if done2:
                    break
            if not done2:
                p_now, ax_now = mj_pose(prev)
                print(f"[MJCF-IK] 点{i}({kind[i]}/{cls[i]}) 全档失败: "
                      f"pos差 {np.linalg.norm(p_now - p_t2)*1000:.0f}mm", flush=True)
            prev = q_mj[i]
        wp, _ = mj_pose(q_mj[i])
        wri_act[i] = wp
        if i % 100 == 0:
            print(f"[MJCF-IK] {i}/{N} 收敛 {conv.sum()}", flush=True)
    print(f"[MJCF-IK] 收敛 {conv.sum()}/{N}, 增量播种命中 {n_incr}/{N}, NN种子命中 {n_nn} "
          f"({100.0 * n_incr / N:.0f}%), 边缘微松 {n_relax} 点, 容差档 {n_loose} 点, "
          f"分支跳变 {len(hops)} 处, 耗时 {_time.time() - t_ik0:.1f}s (旧版~400s)", flush=True)

    # ---------- 2.5 节点级连续性整流: 分支跳变处用"前一节点解"为种子重解 ----------
    # (主IK循环本身顺序热启动; 跳变只发生在重试换种子的孤立点, 原地重解即可恢复连续,
    #  从源头消除速度尖峰, 避免后级重度平滑抹坏枪轴)
    for sweep in range(4):
        dqn = np.abs(np.diff(q_mj, axis=0)).max(1)
        bad_k = np.where(dqn > 0.6)[0]      # 只追分支翻转(2~3.5rad); 0.1~0.2rad是臂伸展
        if not len(bad_k):                   # 2cm腕移的正常构型漂移(帧速~0.4rad/s), 不修
            break
        n_rep_k = 0
        for b in bad_k:
            if zone[b] != zone[b + 1]:
                continue            # 站<->蹲区接缝: 时间轴由回零位停顿覆盖, 不修
            i = int(b) + 1
            p_t_k = np.array([D_ROBOT - D_WRIST, y_r[i], z_r[i] - pz_ref[i]])
            # 连接段(yellow->yellow)节点: 允许宽松容差档(25mm/0.35, 优于连接段标准
            # 25mm/0.88), 优先从源头消除分支翻转; 喷涂节点仍只许严格档
            yel = cls[i] == "yellow" and cls[b] == "yellow"
            tols = ((0.004, 0.05), (0.010, 0.10), (0.025, 0.35)) if yel \
                else ((0.004, 0.05),)
            for pt, at in tols:
                done_b = False
                for relax in (0.0, 0.015, 0.03):
                    p2 = p_t_k.copy()
                    if relax > 0 and abs(p_t_k[1]) > 1e-4:
                        p2[1] = p_t_k[1] - np.sign(p_t_k[1]) * relax
                    qf, ok = ik_mj(p2, axis_t[i], q_mj[i - 1], iters=500,
                                   pos_tol=pt, ax_tol=at)
                    if ok:
                        q_mj[i] = qf
                        n_rep_k += 1
                        done_b = True
                        break
                if done_b:
                    break
        print(f"[节点整流·{sweep + 1}] 节点跳变 {len(bad_k)} 处, 重解 {n_rep_k} 节点", flush=True)
    if n_loose:
        print(f"[MJCF-IK] 容差档点腕位误差 <=10mm/轴0.10 (喷涂半径≥50mm, 影响可忽略)", flush=True)

    # 实际腕心(世界系, z=离地高)
    rob_act_world = wri_act.copy()
    rob_act_world[:, 2] += pz_ref

    data.qpos[:] = base36
    mujoco.mj_forward(model, data)
    INIT_PELVIS = base36[2] - min(data.geom_xpos[g][2] for g in foot_geom)
    print(f"[init站姿] 贴地骨盆高度 {INIT_PELVIS:.4f} m")

    # ---------- 3. 时间参数化 ----------
    ds = np.linalg.norm(np.diff(rob_act_world, axis=0), axis=1)
    v = np.array([SPEED["transition"] if kind[i + 1] == "transition"
                  else SPEED["yellow"] if cls[i + 1] == "yellow"
                  else SPEED[cls[i + 1]] for i in range(N - 1)])
    v = np.maximum(v, 1e-3)
    dwell = np.zeros(N)
    for i in range(N):
        if kind[i] == "stroke" and (i == 0 or kind[i - 1] != "stroke"):
            dwell[i] += DWELL_S / 2
        if kind[i] == "stroke" and (i == N - 1 or kind[i + 1] != "stroke"):
            dwell[i] += DWELL_S / 2
    tt = np.zeros(N)
    for i in range(1, N):
        tt[i] = tt[i - 1] + ds[i - 1] / v[i - 1] + dwell[i]
    T_track = tt[-1] + dwell[-1]
    print(f"[时间] 跟踪段 {T_track:.1f}s (路径长 {ds.sum():.2f}m)", flush=True)

    n_hold0, n_app = int(HOLD0_S / DT), int(APPROACH_S / DT)
    n_ret, n_hold1 = int(RETREAT_S / DT), int(HOLD1_S / DT)
    n_trk = int(np.ceil(T_track / DT)) + 1
    T = n_hold0 + n_app + n_trk + n_ret + n_hold1
    print(f"[帧数] {T} 帧 ({T * DT:.1f}s @30Hz)", flush=True)

    tq = np.arange(n_trk) * DT
    trk_q = np.column_stack([np.interp(tq, tt, q_mj[:, k]) for k in range(10)])
    trk_pz = np.interp(tq, tt, pz_ref)
    trk_wrist_world = np.column_stack([np.interp(tq, tt, rob_act_world[:, c]) for c in range(3)])
    trk_idx = np.clip(np.searchsorted(tt, tq), 0, N - 1)
    trk_ax = axis_t[trk_idx].astype(float)             # 每帧目标枪轴(供5.6修复用)

    def qvec_full(q10, pz):
        """q10 = [腰yaw, 腰roll, 腰pitch, 右臂7]: v36[19:22]=q10[0:3], v36[29:36]=q10[3:10]"""
        v36 = base36.copy()
        hip, knee, ankle = leg_ik_planar(pz)
        v36[7:13] = [hip, 0, 0, knee, ankle, 0]
        v36[13:19] = v36[7:13]
        v36[2] = pz
        v36[19:22] = q10[0:3]          # waist yaw/roll/pitch (三自由度全解锁)
        v36[29:36] = q10[3:10]
        return v36

    q_init_vec = qvec_full(np.zeros(10), INIT_PELVIS)
    q_first = qvec_full(trk_q[0], trk_pz[0])
    q_last = qvec_full(trk_q[-1], trk_pz[-1])

    # ---------- 3.5 站->蹲过渡段: 回零位停顿 (等价于按 i 回默认姿态) ----------
    trk_cls = cls[trk_idx].astype(object)
    trk_kind = kind[trk_idx].astype(object)
    # zone 翻转处 (站->蹲接缝, 即原 transition 桥梁位置)
    _zones = zone[trk_idx].astype(object)
    tf = np.where(_zones[1:] != _zones[:-1])[0] + 1
    if len(tf):
        f0, f1 = int(tf.min()), int(tf.max())
        n_b, n_h = 21, 36                    # 0.7s 收臂 + 1.2s 零位停顿 + 0.7s 伸向蹲姿起点
        iA, iB = max(f0 - 1, 0), min(f1 + 1, n_trk - 1)
        qA, qB = trk_q[iA], trk_q[iB]
        pzA, pzB = trk_pz[iA], trk_pz[iB]
        wA, wB = trk_wrist_world[iA].copy(), trk_wrist_world[iB].copy()
        home_arm = np.zeros(10)              # 零位: 腰三自由度+右臂全零
        home_w = np.array([0.20, -0.149, INIT_PELVIS + 0.085])
        b1 = smoothstep(np.arange(1, n_b + 1) / n_b)[:, None]
        b2 = smoothstep(np.arange(n_b) / n_b)[:, None]
        home_q = np.vstack([qA + (home_arm - qA) * b1,
                            np.repeat(home_arm[None], n_h, 0),
                            home_arm + (qB - home_arm) * b2])
        home_pz = np.concatenate([pzA + (INIT_PELVIS - pzA) * b1[:, 0],
                                  np.full(n_h, INIT_PELVIS),
                                  INIT_PELVIS + (pzB - INIT_PELVIS) * b2[:, 0]])
        home_w = np.vstack([wA + (home_w - wA) * b1,
                            np.repeat(home_w[None], n_h, 0),
                            home_w + (wB - home_w) * b2])
        nb = n_b * 2 + n_h
        w_all = np.linspace(0.0, 1.0, nb)[:, None]
        home_ax = trk_ax[iA] + (trk_ax[iB] - trk_ax[iA]) * w_all
        trk_q = np.vstack([trk_q[:f0], home_q, trk_q[f1 + 1:]])
        trk_pz = np.concatenate([trk_pz[:f0], home_pz, trk_pz[f1 + 1:]])
        trk_wrist_world = np.vstack([trk_wrist_world[:f0], home_w, trk_wrist_world[f1 + 1:]])
        trk_ax = np.vstack([trk_ax[:f0], home_ax, trk_ax[f1 + 1:]])
        trk_cls = np.concatenate([trk_cls[:f0], np.full(nb, "yellow", object), trk_cls[f1 + 1:]])
        trk_kind = np.concatenate([trk_kind[:f0], np.full(nb, "home", object), trk_kind[f1 + 1:]])
        n_trk = len(trk_q)
        print(f"[回零位] 站->蹲过渡插入回家停顿 {nb} 帧 ({nb * DT:.1f}s)", flush=True)

    # ---------- 4. 组装相位 ----------
    n_trk = len(trk_q)                       # 回零位拼接后已更新
    T = n_hold0 + n_app + n_trk + n_ret + n_hold1
    qpos = np.zeros((T, 36))
    phase = np.zeros(T, dtype=object)
    qpos[:n_hold0] = q_init_vec; phase[:n_hold0] = "hold_init"
    for i in range(n_app):
        a = smoothstep((i + 1) / n_app)
        qpos[n_hold0 + i] = (1 - a) * q_init_vec + a * q_first
        phase[n_hold0 + i] = "approach"
    k0 = n_hold0 + n_app
    for i in range(n_trk):
        qpos[k0 + i] = qvec_full(trk_q[i], trk_pz[i])
        phase[k0 + i] = ("home" if trk_kind[i] == "home"
                         else "track_" + str(trk_cls[i]))
    k1 = k0 + n_trk
    for i in range(n_ret):
        a = smoothstep(i / n_ret)
        qpos[k1 + i] = (1 - a) * q_last + a * q_init_vec
        phase[k1 + i] = "retreat"
    qpos[k1 + n_ret:] = q_init_vec; phase[k1 + n_ret:] = "hold_init"

    # ---------- 4.5 连续性修复: 分支跳变窗口线性插值 ----------
    _hops = np.where(np.max(np.abs(np.diff(q_mj, axis=0)), axis=1) > 0.20)[0]
    if len(_hops):
        used = set()
        for h in _hops:
            if h in used or h + 1 in used:
                continue
            i0 = h
            while i0 - 1 in _hops:
                i0 -= 1
            i1 = h + 1
            while i1 + 1 in _hops:
                i1 += 1
            if i1 - i0 > 40:            # 窗口过长不处理(目标本身不同)
                continue
            qa, qb = q_mj[i0], q_mj[i1]
            n = i1 - i0
            for k, i in enumerate(range(i0 + 1, i1)):
                a = smoothstep((k + 1) / n)
                cand = qa * (1 - a) + qb * a
                p_c, ax_c = mj_pose(np.clip(cand, alo, ahi))
                p_t2 = np.array([D_ROBOT - D_WRIST, y_r[i], z_r[i] - pz_ref[i]])
                if (np.linalg.norm(p_c - p_t2) < 0.03
                        and np.dot(ax_c, axis_t[i]) > 0.95):
                    q_mj[i] = np.clip(cand, alo, ahi)
                    used.add(i)
                else:
                    used.add(i)             # 无法修复也标记, 避免重复处理
        print(f"[连续性] 分支跳变窗口 {len(_hops)} 处, 已线性插值修复", flush=True)

    # ---------- 5. 平滑 + 贴地校正 ----------
    np.savez(os.path.join(OUT, "debug_knots.npz"), q_mj=q_mj, conv=conv,
             y_r=y_r, z_r=z_r, axis_t=axis_t, pz_ref=pz_ref,
             cls=cls, kind=kind)                       # 调试: 节点解与目标(离线分析用)
    win = SMOOTH_WIN; pad = win // 2
    ker7 = np.ones(win) / win
    qs = np.vstack([np.repeat(qpos[:1], pad, 0), qpos, np.repeat(qpos[-1:], pad, 0)])
    qpos_s = np.vstack([np.convolve(qs[:, j], ker7, "valid") for j in range(36)]).T
    qpos_s[:n_hold0] = q_init_vec
    qpos_s[k1 + n_ret:] = q_init_vec
    for i in range(T):
        data.qpos[:] = qpos_s[i]
        mujoco.mj_forward(model, data)
        qpos_s[i, 2] -= min(data.geom_xpos[g][2] for g in foot_geom)

    # 尖峰局部重平滑: 对关节速度>6rad/s 的帧簇, 在±20帧窗口内对该关节做重度平滑
    for _pass in range(3):
        dqx = np.abs(np.diff(qpos_s[:, 7:], axis=0)) / DT
        bad = np.where(dqx.max(1) > 9.0)[0]
        if len(bad) == 0:
            break
        clusters = []
        b0 = bad[0]; b1 = bad[0]
        for b in bad[1:]:
            if b - b1 <= 3:
                b1 = b
            else:
                clusters.append((b0, b1)); b0 = b1 = b
        clusters.append((b0, b1))
        for c0, c1 in clusters:
            w0, w1 = max(0, c0 - 20), min(len(qpos_s), c1 + 22)
            for j in range(7, 36):
                seg = qpos_s[w0:w1, j]
                qpos_s[w0:w1, j] = np.convolve(
                    np.pad(seg, 12, mode="edge"), np.ones(25) / 25, "valid")
    dq = np.abs(np.diff(qpos_s[:, 7:], axis=0)) / DT   # 含腰三自由度与左臂(常数)
    jnames = []
    for s in ("left", "right"):
        jnames += [f"{s}_hip_pitch", f"{s}_hip_roll", f"{s}_hip_yaw",
                   f"{s}_knee", f"{s}_ankle_pitch", f"{s}_ankle_roll"]
    jnames += ["waist_yaw", "waist_roll", "waist_pitch"]
    for s in ("left", "right"):
        jnames += [f"{s}_shoulder_pitch", f"{s}_shoulder_roll", f"{s}_shoulder_yaw",
                   f"{s}_elbow", f"{s}_wrist_roll", f"{s}_wrist_pitch", f"{s}_wrist_yaw"]
    worst = np.unravel_index(np.argmax(dq), dq.shape)
    print(f"[平滑+贴地] 完成; 最大关节速度 {dq.max():.2f} rad/s ({jnames[worst[1]]} "
          f"@t={worst[0] * DT:.1f}s)", flush=True)

    # ---------- 5.5 枪轴修复: 平滑/尖峰重平滑破坏指向的帧, 顺序热启动重解 ----------
    # 目标 = 名义插值目标(腕心位置用跟踪段名义路径, 轴向用该帧轨迹指向),
    # 从上一个好帧的解顺序热启动穿过坏块, 保证修复段自身连续.
    def ik_polish(seed10, p_t, ax_t, iters=600, dot_ok=0.98, pos_ok=0.006, ax_early=None):
        # pos_ok/dot_ok = 最终验收档: 喷涂帧默认(6mm/0.98); 连接段宽松档(25mm/0.88)
        # 早停判据随验收档走: 宽松调用时达到连接段标准即收, 否则会放着已达标解
        # 继续朝严格4mm硬收敛跑飞(分支逃逸), 反而验收失败
        if pos_ok < 0.02:                    # 喷涂档: 早停=严格(4mm/轴0.05)
            p_early, a_early = 0.004, 0.05
        else:                                # 连接段档: 早停=20mm/轴0.45(dot>0.90)
            p_early, a_early = 0.020, 0.45
        if ax_early is not None:             # v5边界行偏置修复: 收敛更紧(0.008rad≈0.46°)
            a_early = ax_early
        q = np.clip(np.asarray(seed10, float), alo, ahi)
        jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
        J = np.zeros((6, len(arm_dadr)))
        lam2 = 0.0036                              # 与 ik_mj 一致(轴向不加权: 加权会发散)
        for it in range(iters):
            qq = qpos_s_template.copy()
            qq[19:22] = q[0:3]; qq[29:36] = q[3:10]
            data.qpos[:] = qq
            mujoco.mj_forward(model, data)
            p = data.xpos[WRIST_BODY].copy()
            ax = data.xmat[WRIST_BODY].reshape(3, 3)[:, 0].copy()
            e = np.concatenate([(p_t - p), (ax_t - ax)])
            if np.linalg.norm(e[:3]) < p_early and np.linalg.norm(e[3:]) < a_early:
                return q, True
            mujoco.mj_jacBody(model, data, jacp, jacr, WRIST_BODY)
            for k in range(len(arm_dadr)):
                J[:3, k] = jacp[:, arm_dadr[k]]
                J[3:, k] = np.cross(jacr[:, arm_dadr[k]], ax)
            dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), e)
            q = np.clip(q + dq, alo, ahi)
        qq = qpos_s_template.copy()
        qq[19:22] = q[0:3]; qq[29:36] = q[3:10]
        data.qpos[:] = qq
        mujoco.mj_forward(model, data)
        ax = data.xmat[WRIST_BODY].reshape(3, 3)[:, 0]
        ok = (np.linalg.norm(data.xpos[WRIST_BODY] - p_t) < pos_ok
              and float(np.dot(ax, ax_t)) > dot_ok)
        return q, ok

    qpos_s_template = qpos_s[k0].copy()      # 模板: 腿/骨盆/root 取自平滑帧, 仅腰三自由度+右臂可变
    Q10C = [19, 20, 21] + list(range(29, 36))   # 36维qpos中的自由变量索引

    def q10_of(i):
        return qpos_s[i, Q10C].copy()

    def set_q10(i, q10):
        qpos_s[i, Q10C] = q10

    def repair_sweep(dot_thresh_spray=0.965):
        """修复与名义偏差过大的帧, 判据分两类(v4):
        喷涂帧(track_green/red): 与名义指向 dot<0.965 需修复, 修复后 dot>0.98 且位置<6mm
          (不看绝对角度: 名义角由轨迹构造保证 60~90° 离散, 60° 行不得被'修复'抬高);
        连接帧(track_yellow, 不喷涂): 实际腕心位置误差≤25mm 且 dot>0.88 即合格,
          否则按宽松验收档(pos_ok=0.025, dot_ok=0.88)重解.
        种子=名义构型trk_q + 上一修复解; 目标可横向内松."""
        n_fix = ok_fix = 0
        fixed_idx = []
        prev10 = None
        for i in range(k0, k1):
            if not str(phase[i]).startswith("track_") or phase[i] == "home":
                continue
            j = i - k0
            spray = str(phase[i]) in ("track_green", "track_red")
            qpos_s_template[:] = qpos_s[i]       # 该帧的腿/骨盆/root 平滑值
            data.qpos[:] = qpos_s[i]
            mujoco.mj_forward(model, data)
            ax = data.xmat[WRIST_BODY].reshape(3, 3)[:, 0].copy()
            p_now = data.xpos[WRIST_BODY].copy()
            tgt = trk_ax[j] / np.linalg.norm(trk_ax[j])
            p_t0 = trk_wrist_world[j].copy()     # 世界系名义腕心(root xy=0, 与data.xpos同系)
            dotv = float(np.dot(ax, tgt))
            q10 = q10_of(i)
            if spray:
                if dotv > dot_thresh_spray:
                    prev10 = q10
                    continue
            elif np.linalg.norm(p_now - p_t0) <= 0.025 and dotv > 0.88:
                prev10 = q10
                continue
            got = None
            node_q = q_mj[trk_idx[j]] if j < len(trk_idx) else trk_q[j]
            p_pv = np.array([D_ROBOT - D_WRIST, p_t0[1], p_t0[2] - trk_pz[j]])
            extra_seeds = [node_q, mc_allq[mc_nn(p_pv)]]   # 节点IK解 + MC最近邻种子
            for relax in (0.0, 0.015, 0.03):     # 边缘松弛档: 目标横向内松(与主IK一致)
                p_t = p_t0.copy()
                if relax > 0 and abs(p_t[1]) > 1e-4:
                    p_t[1] -= np.sign(p_t[1]) * relax
                seeds = ([prev10, trk_q[j]] + extra_seeds if prev10 is not None
                         else [trk_q[j], q10] + extra_seeds)
                for sd in seeds:
                    if spray:
                        qf, ok2 = ik_polish(sd, p_t, tgt)
                    else:
                        qf, ok2 = ik_polish(sd, p_t, tgt, pos_ok=0.025, dot_ok=0.88)
                    if ok2:
                        got = qf
                        break
                if got is not None:
                    break
            n_fix += 1
            if got is not None:
                set_q10(i, got)
                ok_fix += 1
                fixed_idx.append(i)
                prev10 = got
        return fixed_idx, n_fix, ok_fix

    fixed_idx, n_fix, ok_fix = repair_sweep(0.975)
    print(f"[枪轴修复] 检出指向偏差帧 {n_fix}, 修复成功 {ok_fix} "
          f"(失败帧保持平滑解)", flush=True)

    # 修复块边界 smoothstep 混合: 消除修复解与平滑解接缝处的速度阶跃
    if fixed_idx:
        runs = np.split(np.array(fixed_idx), np.where(np.diff(fixed_idx) > 1)[0] + 1)
        NB = 6
        for run in runs:
            q_rep_l, q_rep_r = q10_of(int(run[0])), q10_of(int(run[-1]))
            for k in range(1, NB + 1):
                w = ((NB + 1 - k) / (NB + 1)) ** 1.5      # 越靠近修复段权重越大
                for i, q_rep in ((int(run[0]) - k, q_rep_l), (int(run[-1]) + k, q_rep_r)):
                    if not (k0 <= i < k1):
                        continue
                    ph = str(phase[i])
                    if not ph.startswith("track_") or ph == "home":
                        continue
                    set_q10(i, q_rep * w + q10_of(i) * (1.0 - w))
        # 兜底二扫: 混合可能造成的新偏差(阈值=喷涂标准0.965)
        fixed2, n2, ok2n = repair_sweep(0.965)
        print(f"[枪轴修复·二扫] 检出 {n2}, 修复 {ok2n}", flush=True)

    # 连续性整流: 相邻帧腰三自由度+右臂关节差>阈值时, 以前一帧解为种子重解该帧.
    # 相邻帧名义目标仅差~2mm, 连续解必然存在(与主MJCF-IK顺序热启动同原理).
    def rectify_pass():
        n_rep_total = 0
        for sweep in range(3):
            dq8 = np.abs(np.diff(qpos_s[:, Q10C], axis=0)).max(1)
            bad = np.where(dq8 > 0.25)[0]   # 帧级尖峰阈值(~7.5rad/s); 节点间正常漂移~0.4rad/s
            if not len(bad):
                break
            n_rep = 0
            for b in bad:
                for i in (int(b) + 1, int(b)):     # 优先整流后一帧, 前一帧兜底
                    if not (k0 <= i < k1):
                        continue
                    ph = str(phase[i])
                    if not ph.startswith("track_"):    # home/hold 帧不可整流(插值块端点)
                        continue
                    j = i - k0
                    qpos_s_template[:] = qpos_s[i]
                    tgt = trk_ax[j] / np.linalg.norm(trk_ax[j])
                    p_t0 = trk_wrist_world[j].copy()
                    prev_trk = i - 1 if (i - 1 >= k0 and str(phase[i - 1]).startswith("track_")
                                         and phase[i - 1] != "home") else i + 1
                    seed = q10_of(prev_trk) if k0 <= prev_trk < k1 else q10_of(i)
                    node_q = q_mj[trk_idx[j]] if j < len(trk_idx) else trk_q[j]
                    done = False
                    yel_i = ph == "track_yellow"       # 连接段帧: 宽松验收(25mm/0.88)
                    for relax in (0.0, 0.015, 0.03):   # 边缘松弛档: 目标横向内松
                        p_t = p_t0.copy()
                        if relax > 0 and abs(p_t[1]) > 1e-4:
                            p_t[1] -= np.sign(p_t[1]) * relax
                        for sd in (seed, node_q, trk_q[j]):   # 种子: 邻帧解/节点IK解/名义构型
                            if yel_i:
                                qf, ok3 = ik_polish(sd, p_t, tgt, iters=600,
                                                    dot_ok=0.88, pos_ok=0.025)
                            else:
                                qf, ok3 = ik_polish(sd, p_t, tgt, iters=600)
                            if ok3:
                                set_q10(i, qf)
                                n_rep += 1
                                done = True
                                break
                        if done:
                            break
            n_rep_total += n_rep
            print(f"[连续性整流·{sweep + 1}] 大跳变 {len(bad)} 处, 重解 {n_rep} 帧", flush=True)
            if n_rep == 0:
                break
        return n_rep_total

    rectify_pass()
    dq_post = np.abs(np.diff(qpos_s[:, 7:], axis=0)) / DT
    print(f"[枪轴修复后] 最大关节速度 {dq_post.max():.2f} rad/s", flush=True)

    # ---------- 5.65 审计驱动修复环: 实测墙角<60° 或 帧速>9rad/s 则持续重解 ----------
    # (只加大修复通道迭代与种子, 轨迹目标不变; 顺序热启动保证修复段自身连续)
    def measured_track_stats():
        """逐 track 帧实测: 墙面夹角 + 与名义指向 dot + 类别(green/red/yellow)"""
        ii = [i for i in range(k0, k1) if str(phase[i]).startswith("track_")]
        ang = np.zeros(len(ii)); dotv = np.zeros(len(ii))
        cl = np.array([str(phase[i])[6:] for i in ii], object)
        for r, i in enumerate(ii):
            data.qpos[:] = qpos_s[i]
            mujoco.mj_forward(model, data)
            ax = data.xmat[WRIST_BODY].reshape(3, 3)[:, 0]
            ang[r] = 90.0 - np.degrees(np.arccos(np.clip(ax[0], -1, 1)))
            tgt = trk_ax[i - k0] / np.linalg.norm(trk_ax[i - k0])
            dotv[r] = float(np.dot(ax, tgt))
        return np.array(ii), ang, dotv, cl

    def repair_frames(targets, dot_ok=0.975):
        """定向重解给定帧. v5: 种子=当前帧解优先(同分支微调, 从机制上防分支翻转),
        其后邻帧修复解/名义构型/节点IK解/MC最近邻.
        边界行(名义墙角<63°, 本轨迹60°行)用向墙面法向偏置2°的目标轴(名义恰在≥60°门上,
        残差必然压出59.x°帧; 偏置后实测≥60.8°且与名义dot≥cos(2°+容差)≈0.998>0.965),
        收敛早停0.008rad, 验收=实测墙角≥60.8° 且 与名义dot>0.968; 高名义角帧按原档.
        连续性守卫: 修复解与当前构型最大关节差>0.20rad(6rad/s)视为分支翻转, 拒绝."""
        n_ok = 0
        prev10 = None
        for i in targets:
            j = i - k0
            qpos_s_template[:] = qpos_s[i]
            tgt = trk_ax[j] / np.linalg.norm(trk_ax[j])
            ang_nom = 90.0 - np.degrees(np.arccos(np.clip(tgt[0], -1, 1)))
            edge_row = ang_nom < 63.0
            tgt_r = bias_tilt(tgt, 2.0) if edge_row else tgt
            p_t0 = trk_wrist_world[j].copy()
            cur = q10_of(i)
            if prev10 is None:
                prev10 = cur
            node_q = q_mj[trk_idx[j]] if j < len(trk_idx) else trk_q[j]
            p_pv = np.array([D_ROBOT - D_WRIST, p_t0[1], p_t0[2] - trk_pz[j]])
            seeds = [cur, prev10, node_q, mc_allq[mc_nn(p_pv)], trk_q[j]]
            got = None
            for relax in (0.0, 0.015, 0.03):
                p_t = p_t0.copy()
                if relax > 0 and abs(p_t[1]) > 1e-4:
                    p_t[1] -= np.sign(p_t[1]) * relax
                for sd in seeds:
                    qf, ok2 = ik_polish(sd, p_t, tgt_r, iters=600,
                                        dot_ok=0.9998 if edge_row else dot_ok,
                                        ax_early=0.008 if edge_row else None)
                    if not ok2:
                        continue
                    if edge_row:             # 偏置修复帧: 按最终门+裕度实测验收
                        qq2 = qpos_s_template.copy()
                        qq2[19:22] = qf[0:3]
                        qq2[29:36] = qf[3:10]
                        data.qpos[:] = qq2
                        mujoco.mj_forward(model, data)
                        axf = data.xmat[WRIST_BODY].reshape(3, 3)[:, 0]
                        angf = 90.0 - np.degrees(np.arccos(np.clip(axf[0], -1, 1)))
                        ok2 = (angf >= 60.8 and float(np.dot(axf, tgt)) > 0.968)
                    if ok2 and float(np.max(np.abs(qf - cur))) > 0.20:
                        ok2 = False          # 分支翻转解: 拒绝(保连续性/帧速)
                    if ok2:
                        got = qf
                        break
                if got is not None:
                    break
            if got is not None:
                set_q10(i, got)
                prev10 = got
                n_ok += 1
            else:
                prev10 = cur
        return n_ok

    def spread_flip():
        """单帧分支翻转摊宽: 翻转对两侧构型线性插值为种子, 逐帧重投影
        (位置严/轴向松), 每帧实测墙角≥61° 才接受; 任一帧失败则整体回退."""
        dqx = np.abs(np.diff(qpos_s[:, Q10C], axis=0)).max(1)
        cand = [int(b) for b in np.where(dqx > 0.25)[0]
                if k0 <= b < k1 and str(phase[b]).startswith("track_")
                and str(phase[b + 1]).startswith("track_")]
        if not cand:
            return False
        snap = qpos_s.copy()
        for b in cand:
            done_w = False
            for w in (8, 4, 2):                    # 摊宽半宽(帧)
                lo, hi = b - w + 1, b + w
                if lo < k0 or hi >= k1:
                    continue
                idxs = list(range(lo, hi + 1))
                if any(not str(phase[i]).startswith("track_") or phase[i] == "home"
                       for i in idxs):
                    continue
                qA = qpos_s[b, Q10C].copy()
                qB = qpos_s[b + 1, Q10C].copy()
                ok_all = True
                for n_, i in enumerate(idxs):
                    t = (n_ + 1) / (len(idxs) + 1)
                    ci = qA + (qB - qA) * smoothstep(t)
                    j = i - k0
                    spray_i = str(phase[i]) in ("track_green", "track_red")
                    qpos_s_template[:] = qpos_s[i]
                    tgt = trk_ax[j] / np.linalg.norm(trk_ax[j])
                    p_t = trk_wrist_world[j].copy()
                    got = None
                    if spray_i:      # 喷涂帧: 严验收
                        tries = ((ci, 0.90), (ci, 0.94), (qpos_s[i, Q10C], 0.985))
                    else:            # 连接段帧: 宽松验收(25mm/0.88)
                        tries = ((ci, 0.90), (qpos_s[i, Q10C], 0.90))
                    for sd, dko in tries:
                        if spray_i:
                            qf, ok2 = ik_polish(sd, p_t, tgt, iters=800, dot_ok=dko)
                        else:
                            qf, ok2 = ik_polish(sd, p_t, tgt, iters=800, dot_ok=dko,
                                                pos_ok=0.025)
                        if ok2:
                            got = qf
                            break
                    if got is None:
                        ok_all = False
                        break
                    set_q10(i, got)
                    data.qpos[:] = qpos_s[i]
                    mujoco.mj_forward(model, data)
                    axv = data.xmat[WRIST_BODY].reshape(3, 3)[:, 0]
                    if spray_i:      # 喷涂帧: 实测墙角≥61°
                        if 90.0 - np.degrees(np.arccos(np.clip(axv[0], -1, 1))) < 61.0:
                            ok_all = False
                            break
                    else:            # 连接段帧: 位置≤25mm 且 dot>0.88
                        pv = data.xpos[WRIST_BODY]
                        tv = trk_ax[j] / np.linalg.norm(trk_ax[j])
                        if (np.linalg.norm(pv - p_t) > 0.025
                                or float(np.dot(axv, tv)) <= 0.88):
                            ok_all = False
                            break
                if ok_all:
                    done_w = True
                    break
            if not done_w:
                qpos_s[:] = snap                   # 回退, 保持原状
                return False
        return True

    def spread_yellow():
        """连接段(track_yellow, 不喷涂)残留分支翻转: 在该连接段整个时间窗内做
        关节空间 smoothstep 摊宽 — 段两端构型锚定不动, 中间逐帧 smoothstep 插值.
        该翻转是解流形两片之间的真实重构(分支A在其死亡节点连25mm管都不可行,
        管内连续路径已两种方法验证不存在), 中段插值帧腕心横向偏差最大~0.2m,
        但枪轴与名义指向 dot≥0.98; 末端审计只对 green/red 设硬阈值, yellow 按
        连接段标准(25mm/0.88)评估并如实写入meta(仅报告, 不设为通过性门).
        重投影只在"跟得住坡道"(结果接近插值值)时采用, 防止跳片回吸成尖峰."""
        dqx = np.abs(np.diff(qpos_s[:, Q10C], axis=0)).max(1)
        cand = [int(b) for b in np.where(dqx > 0.25)[0]
                if k0 <= b < k1 and str(phase[b]) == "track_yellow"
                and str(phase[b + 1]) == "track_yellow"]
        if not cand:
            return 0
        segs, seen = [], set()
        for b in cand:                       # 同段多翻转去重, 每段只摊一次
            s0 = b
            while s0 - 1 >= k0 and str(phase[s0 - 1]) == "track_yellow":
                s0 -= 1
            if s0 in seen:
                continue
            seen.add(s0)
            s1 = b + 1
            while s1 + 1 < k1 and str(phase[s1 + 1]) == "track_yellow":
                s1 += 1
            segs.append((s0, s1))
        n_done = 0
        for s0, s1 in segs:
            print(f"[连接段摊宽] 处理段 帧{s0}~{s1} ({s1 - s0}帧)", flush=True)
            snap_seg = qpos_s[s0:s1 + 1].copy()
            qA = qpos_s[s0, Q10C].copy()
            qB = qpos_s[s1, Q10C].copy()
            n_seg = s1 - s0
            n_tube = n_out = 0
            ok_all = True
            prev_acc = qA.copy()             # 上一已收帧(连续性锚)
            for k, i in enumerate(range(s0 + 1, s1)):
                ci = np.clip(qA + (qB - qA) * smoothstep((k + 1) / n_seg), alo, ahi)
                j = i - k0
                tgt = trk_ax[j] / np.linalg.norm(trk_ax[j])
                p_t = trk_wrist_world[j].copy()
                # 纯插值帧的连接段标准评估(腕心位置/枪轴dot)
                qpos_s_template[:] = qpos_s[i]
                cand_q = qpos_s[i].copy()
                cand_q[Q10C] = ci
                data.qpos[:] = cand_q
                mujoco.mj_forward(model, data)
                pv = data.xpos[WRIST_BODY].copy()
                av = data.xmat[WRIST_BODY].reshape(3, 3)[:, 0].copy()
                in_tube = (np.linalg.norm(pv - p_t) <= 0.025
                           and float(np.dot(av, tgt)) > 0.88)
                acc = ci
                if not in_tube:
                    # 出管: ci种子宽松重投影; 仅当结果同时贴近插值值与上一已收帧
                    # (未跳片、不产生新的帧间跳变)才采用, 否则保持坡道插值(如实报告)
                    qf, ok2 = ik_polish(ci, p_t, tgt, iters=400,
                                        dot_ok=0.88, pos_ok=0.025)
                    if (ok2 and np.max(np.abs(qf - ci)) <= 0.15
                            and np.max(np.abs(qf - prev_acc)) <= 0.20):
                        acc = qf
                    else:
                        n_out += 1     # 保持坡道插值(如实报告), 不跳片
                if in_tube:
                    n_tube += 1
                set_q10(i, acc)
                prev_acc = acc.copy()
            # 摊宽后整段复查: 相邻帧关节增量(应≤~0.15rad)与两端接缝
            seg_dq = np.abs(np.diff(qpos_s[s0:s1 + 1, Q10C], axis=0)).max()
            seam0 = float(np.abs(qpos_s[s0, Q10C] - qpos_s[s0 - 1, Q10C]).max()) \
                if s0 - 1 >= k0 else 0.0
            seam1 = float(np.abs(qpos_s[s1 + 1, Q10C] - qpos_s[s1, Q10C]).max()) \
                if s1 + 1 < k1 else 0.0
            if seg_dq > 0.30 or seam0 > 0.30 or seam1 > 0.30:
                qpos_s[s0:s1 + 1] = snap_seg          # 异常回退
                print(f"[连接段摊宽] 段{s0}~{s1} 摊宽后仍有跳变(段内{seg_dq:.2f} "
                      f"接缝{seam0:.2f}/{seam1:.2f}) → 回退", flush=True)
                continue
            n_done += 1
            print(f"[连接段摊宽] 段{s0}~{s1} 整窗smoothstep摊宽完成: 段内最大帧增 "
                  f"{seg_dq:.3f} rad ({seg_dq * DT:.1f} rad/s), 接缝 {seam0:.3f}/{seam1:.3f} rad; "
                  f"插值帧连接段标准(25mm/0.88) 达标 {n_tube} 帧, 出管(纯坡道, 轴dot仍>0.88) "
                  f"{n_out} 帧", flush=True)
        if n_done:
            print(f"[连接段摊宽] {len(segs)} 段含翻转, 摊宽成功 {n_done} 段", flush=True)
        return n_done

    n_spread = 0
    for _round in range(8):
        ii_r, ang_r, dot_r, cls_r = measured_track_stats()
        vmax = float(np.abs(np.diff(qpos_s[:, 7:], axis=0)).max() / DT)
        spr_m = np.array([c in ("green", "red") for c in cls_r], bool)
        if ang_r[spr_m].min() >= 60.0 and dot_r[spr_m].min() > 0.965 and vmax <= 9.0:
            break
        # 违规只看喷涂帧(green/red): 角度<60.8°或dot<0.968(门限60°/0.965留裕度);
        # 不再用62°——60°名义行永远到不了62°, 只会整行反复重解制造分支翻转; yellow仅报告
        viol = [int(ii_r[r]) for r in range(len(ii_r))
                if spr_m[r] and (ang_r[r] < 60.8 or dot_r[r] < 0.968)]
        n_ok = repair_frames(viol) if viol else 0
        n_yel = 0
        if vmax > 9.0:                       # 连接段残留翻转: 整窗摊宽优先(不碰喷涂帧)
            n_yel = spread_yellow()
        n_sp, sp_txt = 0, "—"
        if not viol and vmax > 9.0 and n_spread < 3:
            n_spread += 1
            n_sp = 1 if spread_flip() else 0
            sp_txt = "成功" if n_sp else "失败"
        n_rect = rectify_pass()
        print(f"[审计修复环·{_round + 1}] 喷涂违规帧(<60.8°或dot<0.968) {len(viol)}, "
              f"修复 {n_ok}, 连接段摊宽 {n_yel} 段, 摊宽 {sp_txt}, 整流 {n_rect}; "
              f"环前喷涂最小角 {ang_r[spr_m].min():.1f}°, 喷涂最小dot "
              f"{dot_r[spr_m].min():.4f}, 帧速峰 {vmax:.1f} rad/s", flush=True)
        if n_ok == 0 and n_rect == 0 and n_sp == 0 and n_yel == 0:
            break
    # 回位块重锚定: 平滑/审计修复只改跟踪段, 回位块仍从修复前的 q_last 出发,
    # 末端帧被修复时会在 跟踪->回位 接缝产生单帧速度阶跃. 以最终跟踪末帧的
    # q10(腰三自由度+右臂)重锚定回位块(smoothstep 端点导数为0, 不引入新尖峰);
    # root/腿保持已贴地校正的原值不动.
    qA_final = qpos_s[k1 - 1, Q10C].copy()
    for i in range(n_ret):
        a = smoothstep(i / n_ret)
        qpos_s[k1 + i, Q10C] = (1 - a) * qA_final + a * q_init_vec[Q10C]
    dq_final = np.abs(np.diff(qpos_s[:, 7:], axis=0)) / DT
    print(f"[最终] 最大关节速度 {dq_final.max():.2f} rad/s", flush=True)

    # ---------- 5.7 末端指向审计(喷涂质量): 平滑后实测枪轴与墙面法向夹角 ----------
    # root quat 恒 identity -> 墙面法向=世界+x; 审计所有跟踪帧(track_*)但分两类:
    # 喷涂帧(green/red)严格要求 ≥60° 且与名义指向 dot>0.965; 连接段(yellow)不喷涂,
    # 分开统计仅报告(硬阈值不适用).
    ii_trk = np.array([i for i in range(k0, k1) if str(phase[i]).startswith("track_")], int)
    ang_wall = np.zeros(len(ii_trk)); dot_nom = np.zeros(len(ii_trk))
    pos_err = np.zeros(len(ii_trk))
    cls_trk = np.array([str(phase[i])[6:] for i in ii_trk], object)
    for r, i in enumerate(ii_trk):
        data.qpos[:] = qpos_s[i]
        mujoco.mj_forward(model, data)
        ax = data.xmat[WRIST_BODY].reshape(3, 3)[:, 0]
        ang_wall[r] = 90.0 - np.degrees(np.arccos(np.clip(ax[0], -1, 1)))
        tgt = trk_ax[i - k0] / np.linalg.norm(trk_ax[i - k0])
        dot_nom[r] = float(np.dot(ax, tgt))
        pos_err[r] = np.linalg.norm(data.xpos[WRIST_BODY] - trk_wrist_world[i - k0])
    m_spr = np.array([c in ("green", "red") for c in cls_trk], bool)
    m_yel = ~m_spr
    ang_spr, dot_spr = ang_wall[m_spr], dot_nom[m_spr]
    spray_ok = bool(ang_spr.min() >= 60.0 and dot_spr.min() > 0.965)
    print(f"[末端指向审计] 喷涂帧(green/red) {int(m_spr.sum())} 实测墙面夹角 "
          f"[{ang_spr.min():.1f}, {ang_spr.max():.1f}]°, 名义指向dot "
          f"[{dot_spr.min():.4f}, {dot_spr.max():.4f}] (要求≥60°且dot>0.965, "
          f"{'全部满足' if spray_ok else '存在违规!'})", flush=True)
    if m_yel.any():
        n_y25 = int((pos_err[m_yel] <= 0.025).sum())
        print(f"[连接段统计·不喷涂] yellow {int(m_yel.sum())} 帧夹角 "
              f"[{ang_wall[m_yel].min():.1f}, {ang_wall[m_yel].max():.1f}]°, "
              f"dot [{dot_nom[m_yel].min():.4f}, {dot_nom[m_yel].max():.4f}], "
              f"腕心位置误差max {pos_err[m_yel].max() * 1000:.0f}mm "
              f"(≤25mm {n_y25}/{int(m_yel.sum())} 帧; 仅报告不设硬阈值, "
              f"摊宽过渡帧出管为解流形两片间重构的固有代价)", flush=True)

    # ---------- 6. 输出 ----------
    np.savez(os.path.join(OUT, "wall_spray_motion.npz"), qpos=qpos_s.astype(np.float32))
    print(f"[写出] wall_spray_motion.npz  qpos{qpos_s.shape} float32 @30Hz", flush=True)

    trk_wrist_out = trk_wrist_world.copy()
    trk_wrist_out[:, 2] -= trk_pz          # 根相对
    meta = dict(
        format="MimicLite any4hdmi-g1-lafan manifest v2: npz{qpos:(T,36) float32}, 30Hz",
        duration_s=round(T * DT, 2), frames=T,
        phases=dict(hold_init_s=HOLD0_S, approach_s=APPROACH_S, track_s=round(T_track, 2),
                    retreat_s=RETREAT_S, hold_end_s=HOLD1_S),
        speed_mps=SPEED, dwell_s=DWELL_S, path_len_m=round(float(ds.sum()), 2),
        pelvis_ref=dict(crouch=CROUCH_PZ, stand=STAND_PZ,
                        range=[round(float(qpos_s[:, 2].min()), 3),
                               round(float(qpos_s[:, 2].max()), 3)]),
        wrist_z_above_ground_m=[round(float(trk_wrist_world[:, 2].min()), 3),
                                round(float(trk_wrist_world[:, 2].max()), 3)],
        mjcf_ik=(f"{int(conv.sum())}/{N} 收敛(位置<4mm, 轴<0.05), 容差档(≤10mm) {n_loose} 点, "
                 f"边缘微松 {n_relax} 点, 跳变 {len(hops)}"),
        spray_angle_deg=[round(float(ang_spr.min()), 1), round(float(ang_spr.max()), 1)],
        spray_angle_ok=spray_ok,
        spray_dot_nominal=[round(float(dot_spr.min()), 4), round(float(dot_spr.max()), 4)],
        spray_frames=int(m_spr.sum()),
        yellow_connect_stats=dict(
            frames=int(m_yel.sum()),
            wall_angle_deg=[round(float(ang_wall[m_yel].min()), 1),
                            round(float(ang_wall[m_yel].max()), 1)] if m_yel.any() else None,
            dot_nominal=[round(float(dot_nom[m_yel].min()), 4),
                         round(float(dot_nom[m_yel].max()), 4)] if m_yel.any() else None,
            wrist_pos_err_max_m=round(float(pos_err[m_yel].max()), 3) if m_yel.any() else None,
            frames_within_25mm=n_y25 if m_yel.any() else None,
            note=("连接段不喷涂, 仅报告不设硬阈值; 修复判据 位置≤25mm 且 dot>0.88. "
                  "残留分支翻转为解流形两片间重构(管内无连续路径), 已按规格在该连接段"
                  "整个时间窗做关节空间smoothstep摊宽, 过渡帧腕心出管为固有代价, "
                  "枪轴dot仍>0.88; 末端审计硬阈值只针对green/red")),
        max_joint_vel_rad_s=round(float(dq_final.max()), 2),
        init_pelvis_m=round(float(INIT_PELVIS), 4),
        wall_placement=GUN_NOTE,
        root="xy=0, quat=identity(面向+x), z=骨盆参考+逐帧贴地校正",
        csv_note="wrist_x/y/z 为骨盆系(根相对)坐标, 米; 腕心离地高 = root_tz + wrist_z",
        usage="npz 放入 mimiclite/motions/any4hdmi-g1-lafan/, 项目 cfg 指定该动作",
    )
    json.dump(meta, open(os.path.join(OUT, "motion_meta.json"), "w", encoding="utf8"),
              ensure_ascii=False, indent=2)

    with open(os.path.join(OUT, "wall_spray_motion_qpos.csv"), "w", newline="",
              encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "phase", "root_tx", "root_ty", "root_tz", "root_qw",
                    "root_qx", "root_qy", "root_qz"] + jnames +
                   ["wrist_x", "wrist_y", "wrist_z", "seg_class"])
        for i in range(T):
            wrist = trk_wrist_out[i - k0] if k0 <= i < k1 else (np.nan, np.nan, np.nan)
            w.writerow([round(i * DT, 4), phase[i]] +
                       [f"{x:.5f}" for x in qpos_s[i]] +
                       [f"{x:.4f}" for x in wrist] +
                       [phase[i][6:] if phase[i].startswith("track_")
                        else ("home" if phase[i] == "home" else "-")])
    print(f"[写出] wall_spray_motion_qpos.csv ({T} 行)", flush=True)
    print("[完成]")


if __name__ == "__main__":
    main()
