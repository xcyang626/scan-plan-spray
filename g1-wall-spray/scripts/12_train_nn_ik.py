# -*- coding: utf-8 -*-
"""步骤12: 训练 per-robot NN-IK 提速模型 -> station/nn_ik/<robot>_nn_ik.npz.

流程:
  1. 数据: 在喷涂工作区(腕平面x 0.30~0.38, y ±0.36, z_rel -0.02~0.48, 纯俯仰指向
     φ ±32°)采样 6 维输入 [p_rel,d], 用 03 的 batch_ik(MC种子, DLS) 解出 10 维关节
     (腰3+臂7) —— 与规划同分布、同容差, 局部单支(邻域目标邻接播种);
  2. 模型: torch MLP 6->256->256->10 (tanh, 输出层线性, 输出做均值方差标准化);
  3. 验证: 在当前真实轨迹(output/trajectory.npz 的 307 目标)上对比
     NN种子 vs MC种子 的收敛率@iters∈{10,30,60};
  4. 部署: station/nn_ik/<robot>_nn_ik.npz (03/05 的 nn_ik() 插槽自动加载;
     每个机器人独立一份, 名字=station/robot_registry.json 的条目名)。
用法:
  python 12_train_nn_ik.py --robot g1 --n 60000 --epochs 150
"""
import os, sys, json, argparse, time, importlib.util
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# per-robot 数据采样配置(legacy 6维小模型): 腕平面带x / 横向带y(右臂可达不对称) /
# 腕心离地z带 / qsub种子过滤带okp / 站姿种子骨盆pz / 工作区npz文件名(与ensure_workspace同名)
# pb_bands: DLS 骨盆带按目标z分段(双姿态机器人): 低段蹲带高段站带, 单值单解防回归坍缩
# (t3 带内实测: stand y -0.60..+0.35, z 0.90..1.83; 蹲带 pz 0.43 起 h0.2 行可行)
ROBOT_CFG = {
    "g1":        dict(x=(0.30, 0.38), y=(-0.36, 0.36), z=(0.72, 1.28),
                      okp=(0.25, 0.42), pz=0.78, ws="g1_workspace.npz"),
    "tiangong3": dict(x=(0.46, 0.54), y=(-0.45, 0.32), z=(0.30, 1.60),
                      okp=(0.41, 0.59), pz=0.98, ws="tiangong_workspace.npz",
                      pb_bands=[(0.85, (0.53, 0.75)), (9.99, (0.96, 1.005))]),   # 蹲带下限=05平面模型膝限位地板
}


def load_plan03():
    spec = importlib.util.spec_from_file_location(
        "plan03", os.path.join(BASE, "scripts", "03_plan_trajectories.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["plan03"] = m
    spec.loader.exec_module(m)          # 仅常量/FK/batch_ik, main 有 __main__ 保护
    return m


def gen_dataset(P03, n, rng, cfg):
    """采样喷涂工作区目标 -> DLS 真解 (输入6维, 输出10维+pz). 采样带按 ROBOT_CFG.
    双姿态机器人(pb_bands)按目标z分段独立采集: 每带自己的单支流形链(蹲低站高),
    跨带共用一条链会让高段解被低段跳支过滤丢弃, 回归被蹲姿支主导."""
    from scipy.spatial import cKDTree
    ws = np.load(os.environ.get("G1W_WS", os.path.join(BASE, "output", cfg["ws"])), allow_pickle=True)
    qsub = np.concatenate([ws[f"right_{m}_y60_qsub"].astype(float)
                           for m in ("stand", "crouch")], 0)      # (M,11)
    endsub = np.concatenate([ws[f"right_{m}_y60_endsub"].astype(float)
                             for m in ("stand", "crouch")], 0)    # (M,4)
    okp = (endsub[:, 0] > cfg["okp"][0]) & (endsub[:, 0] < cfg["okp"][1])   # 腕平面带内
    qsub, pos = qsub[okp], endsub[okp, :3]
    tree = cKDTree(pos)
    pb_bands = cfg.get("pb_bands") or [(9.99, None)]
    # 每带的目标 z 区间 = (上一带 zmax, 本带 zmax) ∩ cfg["z"]
    segs, z_prev = [], cfg["z"][0]
    for zmax, pb in pb_bands:
        lo = max(cfg["z"][0], z_prev); hi = min(cfg["z"][1], zmax)
        if hi > lo:
            segs.append((int(n * (hi - lo) / (cfg["z"][1] - cfg["z"][0])), (lo, hi), pb))
        z_prev = max(z_prev, zmax)

    P_all, D_all, Q_all, PZ_all = [], [], [], []
    B = 2000
    t0 = time.time()
    for n_seg, z_rng, pb_band in segs:
        x = rng.uniform(*cfg["x"], n_seg)
        y = rng.uniform(*cfg["y"], n_seg)
        z = rng.uniform(*z_rng, n_seg)
        phi = np.radians(rng.uniform(-32.0, 32.0, n_seg))
        p_star = np.column_stack([x, y, z])
        d_star = np.column_stack([np.cos(phi), np.zeros(n_seg), np.sin(phi)])
        sp = np.lexsort((y, x, z))                # 空间排序(主键z): 沿高度连续爬升,
        p_star, d_star = p_star[sp], d_star[sp]   # 臂伸展连续变化 -> 单支流形更纯
        Q = np.zeros((n_seg, 10)); PZ = np.zeros(n_seg); OK = np.zeros(n_seg, bool)
        seed_q, seed_pz, last_q = None, None, None
        for s in range(0, n_seg, B):
            e = min(s + B, n_seg)
            if seed_q is None:
                j = tree.query(p_star[s:e], k=1)[1]
                seed_q = qsub[j][:, :10].copy(); seed_pz = qsub[j][:, 10].copy()
            else:
                seed_q = np.repeat(last_q[None], e - s, 0)
                seed_pz = np.full(e - s, (pb_band[0] + pb_band[1]) / 2 if pb_band else cfg["pz"])
            pb = None
            if pb_band:
                pb = np.tile(np.array(pb_band, float), (e - s, 1))
            qf, pf, e1, e2, cv = P03.batch_ik(seed_q.copy(), seed_pz.copy(),
                                              p_star[s:e], d_star[s:e], "stand",
                                              iters=220, pb=pb)
            for k in range(e - s):
                if not cv[k]:
                    continue
                if last_q is not None and np.abs(qf[k] - last_q).max() > 0.6:
                    continue                   # 跳支: 丢弃, 维持单支流形
                Q[s + k], PZ[s + k], OK[s + k] = qf[k], pf[k], True
                last_q = qf[k]
        P_all.append(p_star[OK]); D_all.append(d_star[OK])
        Q_all.append(Q[OK]); PZ_all.append(PZ[OK])
        print(f"[数据] 段z{z_rng} pz带{pb_band}: 单支保留 {OK.sum()}/{n_seg} "
              f"({time.time()-t0:.0f}s)")
    P = np.vstack(P_all); D = np.vstack(D_all); Q = np.vstack(Q_all)
    PZ = np.concatenate(PZ_all)
    print(f"[数据] 合计单支流形保留 {len(P)}/{n} ({time.time()-t0:.0f}s)")
    return P, D, Q, PZ


def train(X, Q, epochs, dev):
    import torch
    import torch.nn as nn
    qm, qs = Q.mean(0), Q.std(0) + 1e-6
    xm, xs = X.mean(0), X.std(0) + 1e-6
    Y = (Q - qm) / qs
    xt = torch.tensor((X - xm) / xs, dtype=torch.float32, device=dev)
    yt = torch.tensor(Y, dtype=torch.float32, device=dev)
    net = nn.Sequential(nn.Linear(6, 256), nn.Tanh(),
                        nn.Linear(256, 256), nn.Tanh(),
                        nn.Linear(256, 10)).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ds = torch.utils.data.TensorDataset(xt, yt)
    dl = torch.utils.data.DataLoader(ds, batch_size=4096, shuffle=True, drop_last=False)
    hist = []
    for ep in range(epochs):
        tot, nb = 0.0, 0
        for xb, yb in dl:
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(xb), yb)
            loss.backward(); opt.step()
            tot += float(loss); nb += 1
        sch.step(); hist.append(tot / nb)
        if (ep + 1) % 25 == 0 or ep == 0:
            print(f"[训练] epoch {ep+1}/{epochs} mse {tot/nb:.4f}")
    net.eval()
    return net, (qm.astype(np.float32), qs.astype(np.float32),
                 xm.astype(np.float32), xs.astype(np.float32)), hist


def predict(net, X, qm, qs, xm, xs, dev):
    import torch
    with torch.no_grad():
        y = net(torch.tensor((X - xm) / xs, dtype=torch.float32,
                             device=dev)).cpu().numpy()
    return y * qs + qm


def evaluate(P03, net, qm, qs, xm, xs, dev, cfg):
    """真实轨迹目标(G1W_TRAJ 可指定): NN种子 vs MC种子 收敛率@小iters (提速证据)."""
    try:
        tz = np.load(os.environ.get("G1W_TRAJ",
                                    os.path.join(BASE, "output", "trajectory.npz")),
                     allow_pickle=True)
        d = tz["axis"].astype(float)
        if cfg is ROBOT_CFG["g1"]:
            p_rel = np.column_stack([                  # 旧口径: 世界系 rob_xyz 换算
                tz["rob_xyz"][:, 0] - (P03.X_FACE - P03.D_ROBOT),
                tz["rob_y_r"].astype(float),
                tz["rob_z"].astype(float)])
        else:
            p_rel = np.column_stack([                  # 与03同约定: x=腕平面, z=离地高
                np.full(len(d), P03.D_ROBOT - P03.D_WRIST),
                tz["rob_y_r"].astype(float),
                tz["rob_z"].astype(float)])
        assert len(p_rel) == len(d) > 10
    except Exception as e:
        print(f"[评估] trajectory.npz 不可用({e}), 跳过真实轨迹评估"); return None
    ws = np.load(os.environ.get("G1W_WS", os.path.join(BASE, "output", cfg["ws"])), allow_pickle=True)
    qsub = np.concatenate([ws[f"right_{m}_y60_qsub"].astype(float)
                           for m in ("stand", "crouch")], 0)
    from scipy.spatial import cKDTree
    res = {}
    pb_bands = cfg.get("pb_bands")          # dual 轨迹: 蹲/站段目标各用对应骨盆带评估
    q_nn = predict(net, np.column_stack([p_rel, d]), qm, qs, xm, xs, dev)
    for it in (10, 30, 60):
        if pb_bands:
            # 逐样本骨盆带+带中点种子(z<带的分界用蹲带, 否则站带)
            pb = np.array([next(b for zm, b in pb_bands if z <= zm) for z in p_rel[:, 2]])
            pzs = pb.mean(1)
        else:
            pb, pzs = None, np.full(len(q_nn), cfg["pz"])
        _, _, _, _, cv_nn = P03.batch_ik(q_nn.copy(), pzs, p_rel, d, "stand",
                                         iters=it, pb=pb)
        sel = np.random.default_rng(0).choice(len(qsub), len(p_rel))
        _, _, _, _, cv_mc = P03.batch_ik(qsub[sel][:, :10].copy(), pzs,
                                         p_rel, d, "stand", iters=it, pb=pb)
        res[it] = (float(cv_nn.mean()), float(cv_mc.mean()))
        print(f"[评估] iters={it}: NN种子收敛率 {cv_nn.mean():.1%} vs 随机MC种子 {cv_mc.mean():.1%}")
    return res


def gen_dataset_universal(P03, n, rng):
    """V2 通用大模型数据: 全腰限档(5/10/15/20°) × 站/蹲/过渡三姿态分组采集.
    输入 8 维 = [x,y,z(离地), d(3), pelvis(采样骨盆), rp(腰限rad)]; 解用窄骨盆带 pb
    求出 -> 输出与条件输入一致(同目标不同姿态给不同解, 网络按条件区分)."""
    from scipy.spatial import cKDTree
    ws = np.load(os.environ.get("G1W_WS", os.path.join(BASE, "output", "g1_workspace.npz")),
                 allow_pickle=True)
    sub = {m: (ws[f"right_{m}_y60_qsub"].astype(float),
               ws[f"right_{m}_y60_endsub"].astype(float)) for m in ("stand", "crouch")}
    okp = {m: (e[:, 0] > 0.25) & (e[:, 0] < 0.42) for m, (q, e) in sub.items()}
    trees = {m: cKDTree(sub[m][1][okp[m], :3]) for m in sub}

    RPS = (5.0, 10.0, 15.0, 20.0)
    # V2.1e: 姿态组采样带取**机器人运动学极限**(与 G1W_CROUCH_LO 等配置解耦) ——
    # 蹲姿地板 0.30 = 膝关节几何可达下界(策略 height_cmd 限幅 0.28 内的安全带),
    # 不再跟随当前规划配置: 换蹲姿地板/站位/腰限档都不需要重训(一次训练全域覆盖)
    _clo = 0.30
    GROUPS = (("stand", 0.35, (0.765, 0.795), (0.70, 1.32), "stand"),
              ("crouch", 0.40, (_clo, 0.700), (max(0.12, _clo - 0.18), 1.10), "crouch"),
              ("mid", 0.25, (max(0.50, _clo + 0.20), 0.760), (0.45, 1.20), "stand"))
    _jlo0 = P03.JLO.copy(); _jhi0 = P03.JHI.copy()      # 原腰限(±5°)备份
    X8, Q10 = [], []
    t0 = time.time()
    for gname, frac, pzb, zb, seed_m in GROUPS:
        n_g = int(n * frac / len(RPS))
        for rp_deg in RPS:
            rp = np.deg2rad(rp_deg)
            P03.JLO[1] = P03.JLO[2] = -rp               # 该批的腰 roll/pitch 限位
            P03.JHI[1] = P03.JHI[2] = +rp
            x = rng.uniform(0.30, 0.38, n_g)
            y = rng.uniform(-0.36, 0.36, n_g)
            z = rng.uniform(*zb, n_g)
            pz = rng.uniform(*pzb, n_g)
            phi = np.radians(rng.uniform(-32.0, 32.0, n_g))
            p_star = np.column_stack([x, y, z])
            d_star = np.column_stack([np.cos(phi), np.zeros(n_g), np.sin(phi)])
            sp = np.lexsort((z, y, x))                  # 空间排序: 单支流形
            p_star, d_star, pz = p_star[sp], d_star[sp], pz[sp]
            pb = np.column_stack([pz - 0.012, pz + 0.012])
            qs, es = sub[seed_m]
            Q = np.zeros((n_g, 10)); OK = np.zeros(n_g, bool)
            B = 2000
            last_q = None
            for s in range(0, n_g, B):
                e2 = min(s + B, n_g)
                if last_q is None:
                    j = trees[seed_m].query(p_star[s:e2], k=1)[1]
                    seed_q = qs[okp[seed_m]][j][:, :10].copy()
                    seed_pz = pz[s:e2].copy()
                else:
                    seed_q = np.repeat(last_q[None], e2 - s, 0)
                    seed_pz = pz[s:e2].copy()
                qf, _, _, _, cv = P03.batch_ik(seed_q.copy(), seed_pz.copy(),
                                               p_star[s:e2], d_star[s:e2], "stand",
                                               iters=220, pb=pb[s:e2])
                for k in range(e2 - s):
                    if not cv[k]:
                        continue
                    if last_q is not None and np.abs(qf[k] - last_q).max() > 0.75:
                        continue
                    Q[s + k] = qf[k]; OK[s + k] = True; last_q = qf[k]
            keep = OK.sum()
            cond = np.column_stack([np.full(keep, rp)])
            X8.append(np.column_stack([p_star[OK], d_star[OK], pz[OK], cond[:, 0]]))
            Q10.append(Q[OK])
            print(f"[数据] {gname:6s} rp±{rp_deg:2.0f}°: 单支保留 {keep}/{n_g} "
                  f"({time.time()-t0:.0f}s)")
    P03.JLO[:], P03.JHI[:] = _jlo0, _jhi0               # 还原(评估用缺省±5°)
    return np.vstack(X8), np.vstack(Q10)


def train_universal(X, Q, epochs, dev):
    """V2 通用大模型: 8 -> 512 -> 512 -> 512 -> 10 (tanh, 输出标准化折叠)."""
    import torch
    import torch.nn as nn
    qm, qs = Q.mean(0), Q.std(0) + 1e-6
    xm, xs = X.mean(0), X.std(0) + 1e-6
    yt = torch.tensor((Q - qm) / qs, dtype=torch.float32, device=dev)
    xt = torch.tensor((X - xm) / xs, dtype=torch.float32, device=dev)
    net = nn.Sequential(nn.Linear(8, 512), nn.Tanh(),
                        nn.Linear(512, 512), nn.Tanh(),
                        nn.Linear(512, 512), nn.Tanh(),
                        nn.Linear(512, 10)).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ds = torch.utils.data.TensorDataset(xt, yt)
    dl = torch.utils.data.DataLoader(ds, batch_size=8192, shuffle=True, drop_last=False)
    for ep in range(epochs):
        tot, nb = 0.0, 0
        for xb, yb in dl:
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(xb), yb)
            loss.backward(); opt.step()
            tot += float(loss); nb += 1
        sch.step()
        if (ep + 1) % 25 == 0 or ep == 0:
            print(f"[训练] epoch {ep+1}/{epochs} mse {tot/nb:.5f}")
    net.eval()
    return net, (qm.astype(np.float32), qs.astype(np.float32),
                 xm.astype(np.float32), xs.astype(np.float32))


def evaluate_universal(P03, net, qm, qs, xm, xs, dev):
    """在当前轨迹(优先 V2 双轨迹)上按点条件评估: NN种子 vs MC种子 收敛率@小iters.
    V2.1d 增补压力档: 骨盆 0.30/0.80 的合成目标带(当前规划未用, 验证'一次训练'
    的全域覆盖 —— 蹲姿地板再下探/站位抬高都不需要重训)."""
    import torch
    from scipy.spatial import cKDTree
    cand = [os.path.join(BASE, "output", p, "trajectory.npz")
            for p in ("v2_dual", "")] + [os.path.join(BASE, "output", "trajectory.npz")]
    tp = next((c for c in cand if os.path.exists(c)), None)
    if tp is None:
        print("[评估] 无 trajectory.npz, 跳过"); return None
    t = np.load(tp, allow_pickle=True)
    d = t["axis"].astype(float)
    p_rel = np.column_stack([np.full(len(d), P03.D_ROBOT - P03.D_WRIST),
                             t["rob_y_r"].astype(float), t["rob_z"].astype(float)])
    zone = t["zone"].astype(str) if "zone" in t.files else np.array(["stand"] * len(d))
    kind = t["kind"].astype(str)
    pb = np.array([P03.PELVIS_RANGE["trans"] if k == "transition"
                   else P03.PELVIS_RANGE[z] for k, z in zip(kind, zone)], float)
    pz_cond = t["pelvis"].astype(float)                 # 条件=规划解的骨盆
    rp = np.full(len(d), _default_rp())
    ws = np.load(os.environ.get("G1W_WS", os.path.join(BASE, "output", "g1_workspace.npz")),
                 allow_pickle=True)
    qsub_all = np.vstack([ws[f"right_{m}_y60_qsub"].astype(float)
                          for m in ("stand", "crouch")])

    def _run(tag, p_r, dd, pb_, pz_, rp_):
        with torch.no_grad():
            xin = np.column_stack([p_r, dd, pz_, rp_])
            q_nn = (net(torch.tensor((xin - xm) / xs, dtype=torch.float32,
                                     device=dev)).cpu().numpy() * qs + qm)
        sel = np.random.default_rng(0).choice(len(qsub_all), len(dd))
        res = {}
        for it in (10, 30, 60):
            _, _, _, _, cv_nn = P03.batch_ik(q_nn.copy(), pz_.copy(), p_r, dd,
                                             "stand", iters=it, pb=pb_)
            _, _, _, _, cv_mc = P03.batch_ik(qsub_all[sel][:, :10].copy(), pz_.copy(),
                                             p_r, dd, "stand", iters=it, pb=pb_)
            res[it] = (float(cv_nn.mean()), float(cv_mc.mean()))
            print(f"[评估] {tag} iters={it}: NN(条件)种子收敛率 {cv_nn.mean():.1%} "
                  f"vs 随机MC种子 {cv_mc.mean():.1%}")
        return res

    ev = {"real": _run(os.path.basename(os.path.dirname(tp)), p_rel, d, pb,
                       pz_cond, rp)}
    # ---- 压力档: 规划从未用过的骨盆带(0.30-0.34 / 0.78-0.82) ----
    rng = np.random.default_rng(11)
    for tag, zb in (("压蹲0.30~0.34", (0.30, 0.34)), ("压站0.78~0.82", (0.78, 0.82))):
        m = 120
        x = rng.uniform(0.30, 0.38, m); y = rng.uniform(-0.30, 0.30, m)
        pz2 = rng.uniform(*zb, m)
        z2 = np.clip(pz2 + rng.uniform(-0.20, 0.45, m), 0.12, 1.35)
        phi = np.radians(rng.uniform(-32, 32, m))
        p2 = np.column_stack([x, y, z2 - pz2 * 0])       # z=离地高(与真实轨迹同约定)
        p2[:, 2] = z2
        d2 = np.column_stack([np.cos(phi), np.zeros(m), np.sin(phi)])
        pb2 = np.column_stack([pz2 - 0.012, pz2 + 0.012])
        ev[tag] = _run(tag, p2, d2, pb2, pz2.copy(),
                       np.full(m, np.deg2rad(10.0)))
    return ev


def _default_rp():
    return float(os.environ.get("G1W_WAIST_RP", "0")) or 0.0872665


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="g1")
    ap.add_argument("--waist-deg", type=int, default=0,
                    help="腰限档位(5/10/15/20); 0=通用文件名")
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--universal", action="store_true",
                    help="V2 通用大模型: 8维条件输入(骨盆+腰限), 一次覆盖全档×站/蹲/过渡")
    a = ap.parse_args()
    dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    P03 = load_plan03()
    rng = np.random.default_rng(7)
    if a.universal:
        if a.robot != "g1":
            sys.exit("--universal 目前仅支持 g1(采样带按G1运动学全域硬编码); "
                     "其他机器人用 legacy 6维小模型")
        # V2.1d 通用大模型: 构型空间采样(默认, 全域一次覆盖) / 任务空间采样(旧法)
        print(f"[训练] 设备 {dev}, 通用大模型(运动学全域采样) {a.robot}")
        X8, Q = gen_dataset_universal(P03, a.n, rng)
        net, (qm, qs, xm, xs) = train_universal(X8, Q, a.epochs, dev)
        ev = evaluate_universal(P03, net, qm, qs, xm, xs, dev)
        out_dir = os.path.join(BASE, "station", "nn_ik")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"{a.robot}_all_nn_ik.npz")
        import torch
        # 隐层 W1..Wk(tanh) + 输出层 Wo; 输入/输出标准化折叠进首末层(与 hook 公式一致)
        lin = [m for m in net.modules() if isinstance(m, torch.nn.Linear)]
        Ws = [l.weight.detach().cpu().numpy().T.astype(np.float32) for l in lin]
        bs = [l.bias.detach().cpu().numpy().astype(np.float32) for l in lin]
        W1 = (Ws[0] / xs[:, None]).astype(np.float32)
        b1 = (bs[0] - (xm / xs).astype(np.float32) @ Ws[0]).astype(np.float32)
        Wo = (Ws[-1] * qs[None, :]).astype(np.float32)
        bo = (bs[-1] * qs + qm).astype(np.float32)
        arrs = dict(W1=W1, b1=b1)
        for i in range(1, len(Ws) - 1):
            arrs[f"W{i+1}"] = Ws[i]
            arrs[f"b{i+1}"] = bs[i]
        arrs["Wo"], arrs["bo"] = Wo, bo
        tmp = out + ".tmp.npz"
        np.savez_compressed(tmp, **arrs)
        os.replace(tmp, out)
        stats = dict(robot=a.robot, universal=True,
                     n_data=int(len(Q)), epochs=a.epochs,
                     coverage=("腰限±5/10/15/20° × 站(0.765~0.795)/蹲(0.30~0.70 运动学"
                               "全域)/过渡(0.50~0.76), 与 G1W_CROUCH_LO 等配置解耦; "
                               f"输入8维=[p(3,世界z),d(3),pelvis,rp], 隐层{len(Ws)-1}×512; "
                               "换蹲姿地板/站位/腰限档无需重训"),
                     eval=ev, model=out)
        json.dump(stats, open(out.replace(".npz", "_stats.json"), "w", encoding="utf8"),
                  ensure_ascii=False, indent=1)
        print(f"[部署] {out}\n[部署] 统计 {out.replace('.npz','_stats.json')}")
        return
    cfg = ROBOT_CFG.get(a.robot)
    if cfg is None:
        cfg = ROBOT_CFG["g1"]
        print(f"[警告] 未知机器人 {a.robot}, 采样带回退 g1 配置")
    print(f"[训练] 设备 {dev}, 机器人 {a.robot}")
    X, D, Q, PZ = gen_dataset(P03, a.n, rng, cfg)
    net, (qm, qs, xm, xs), _hist = train(np.column_stack([X, D]), Q, a.epochs, dev)
    ev = evaluate(P03, net, qm, qs, xm, xs, dev, cfg)

    out_dir = os.path.join(BASE, "station", "nn_ik")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir,
                       f"{a.robot}_rp{a.waist_deg}_nn_ik.npz" if a.waist_deg
                       else f"{a.robot}_nn_ik.npz")
    import torch
    l1, l2, l3 = net[0], net[2], net[4]
    W1 = l1.weight.detach().cpu().numpy().T.astype(np.float32)   # (6,256)
    W2 = l2.weight.detach().cpu().numpy().T.astype(np.float32)   # (256,256)
    Wo = l3.weight.detach().cpu().numpy().T.astype(np.float32)   # (256,10)
    b1 = (l1.bias.detach().cpu().numpy().astype(np.float32)
          - (xm / xs).astype(np.float32) @ W1)   # 输入标准化折叠: x@W1'+b1' = (x-xm)/xs@W1+b1
    W1 = (W1 / xs[:, None]).astype(np.float32)
    b2 = l2.bias.detach().cpu().numpy().astype(np.float32)
    bo = (l3.bias.detach().cpu().numpy() * qs + qm).astype(np.float32)
    Wo = (Wo * qs[None, :]).astype(np.float32)   # 标准化折叠: q 原值直接输出
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, W1=W1, b1=b1, W2=W2, b2=b2, Wo=Wo, bo=bo)
    os.replace(tmp, out)            # 原子替换: 并发读取(03/05 hook)永不读半截文件
    stats = dict(robot=a.robot, n_data=int(len(Q)), epochs=a.epochs,
                 eval=ev, model=out)
    json.dump(stats, open(out.replace(".npz", "_stats.json"), "w", encoding="utf8"),
              ensure_ascii=False, indent=1)
    print(f"[部署] {out}\n[部署] 统计 {out.replace('.npz','_stats.json')}")


if __name__ == "__main__":
    main()
