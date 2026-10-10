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


def load_plan03():
    spec = importlib.util.spec_from_file_location(
        "plan03", os.path.join(BASE, "scripts", "03_plan_trajectories.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["plan03"] = m
    spec.loader.exec_module(m)          # 仅常量/FK/batch_ik, main 有 __main__ 保护
    return m


def gen_dataset(P03, n, rng):
    """采样喷涂工作区目标 -> DLS 真解 (输入6维, 输出10维+pz)."""
    from scipy.spatial import cKDTree
    ws = np.load(os.environ.get("G1W_WS", os.path.join(BASE, "output", "g1_workspace.npz")), allow_pickle=True)
    qsub = np.concatenate([ws[f"right_{m}_y60_qsub"].astype(float)
                           for m in ("stand", "crouch")], 0)      # (M,11)
    endsub = np.concatenate([ws[f"right_{m}_y60_endsub"].astype(float)
                             for m in ("stand", "crouch")], 0)    # (M,4)
    okp = (endsub[:, 0] > 0.25) & (endsub[:, 0] < 0.42)           # 腕平面带内
    qsub, pos = qsub[okp], endsub[okp, :3]
    tree = cKDTree(pos)

    x = rng.uniform(0.30, 0.38, n)
    y = rng.uniform(-0.36, 0.36, n)
    z = rng.uniform(0.72, 1.28, n)   # 腕心离地高(与03同约定; 骨盆高由DLS自优化)
    phi = np.radians(rng.uniform(-32.0, 32.0, n))
    p_star = np.column_stack([x, y, z])
    d_star = np.column_stack([np.cos(phi), np.zeros(n), np.sin(phi)])
    sp = np.lexsort((z, y, x))                     # 空间排序: 邻接目标=同支邻域(x主键,补上x)
    p_star, d_star = p_star[sp], d_star[sp]

    # 单支流形采集: 邻接播种 + 连续性过滤(与上一保留解差<0.6rad) -> 数据落在
    # 同一支解流形片上, 消除多支平均导致的回归坍缩
    Q = np.zeros((n, 10)); PZ = np.zeros(n); OK = np.zeros(n, bool)
    B = 2000
    seed_q, seed_pz, last_q = None, None, None
    t0 = time.time()
    for s in range(0, n, B):
        e = min(s + B, n)
        if seed_q is None:
            j = tree.query(p_star[s:e], k=1)[1]
            seed_q = qsub[j][:, :10].copy(); seed_pz = qsub[j][:, 10].copy()
        else:
            seed_q = np.repeat(last_q[None], e - s, 0)
            seed_pz = np.full(e - s, 0.78)
        qf, pf, e1, e2, cv = P03.batch_ik(seed_q.copy(), seed_pz.copy(),
                                          p_star[s:e], d_star[s:e], "stand", iters=220)
        for k in range(e - s):
            if not cv[k]:
                continue
            if last_q is not None and np.abs(qf[k] - last_q).max() > 0.6:
                continue                       # 跳支: 丢弃, 维持单支流形
            Q[s + k], PZ[s + k], OK[s + k] = qf[k], pf[k], True
            last_q = qf[k]
    print(f"[数据] 单支流形保留 {OK.sum()}/{n} ({time.time()-t0:.0f}s)")
    return p_star[OK], d_star[OK], Q[OK], PZ[OK]


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


def evaluate(P03, net, qm, qs, xm, xs, dev):
    """真实 307 轨迹目标: NN种子 vs MC种子 收敛率@小iters (提速证据)."""
    try:
        tz = np.load(os.path.join(BASE, "output", "trajectory.npz"), allow_pickle=True)
        d = tz["axis"].astype(float)
        p_rel = np.column_stack([                      # 与03同约定: 腕心离地高
            tz["rob_xyz"][:, 0] - (P03.X_FACE - P03.D_ROBOT),
            tz["rob_y_r"].astype(float),
            tz["rob_z"].astype(float)])
        assert len(p_rel) == len(d) > 10
    except Exception as e:
        print(f"[评估] trajectory.npz 不可用({e}), 跳过真实轨迹评估"); return None
    ws = np.load(os.environ.get("G1W_WS", os.path.join(BASE, "output", "g1_workspace.npz")), allow_pickle=True)
    qsub = np.concatenate([ws[f"right_{m}_y60_qsub"].astype(float)
                           for m in ("stand", "crouch")], 0)
    from scipy.spatial import cKDTree
    res = {}
    q_nn = predict(net, np.column_stack([p_rel, d]), qm, qs, xm, xs, dev)
    for it in (10, 30, 60):
        _, _, _, _, cv_nn = P03.batch_ik(q_nn.copy(), np.full(len(q_nn), 0.78),
                                         p_rel, d, "stand", iters=it)
        _, _, _, _, cv_mc = P03.batch_ik(qsub[np.random.default_rng(0).choice(
            len(qsub), len(p_rel))][:, :10].copy(),
            np.full(len(p_rel), 0.78), p_rel, d, "stand", iters=it)
        res[it] = (float(cv_nn.mean()), float(cv_mc.mean()))
        print(f"[评估] iters={it}: NN种子收敛率 {cv_nn.mean():.1%} vs 随机MC种子 {cv_mc.mean():.1%}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="g1")
    ap.add_argument("--waist-deg", type=int, default=0,
                    help="腰限档位(5/10/15/20); 0=通用文件名")
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--epochs", type=int, default=150)
    a = ap.parse_args()
    dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"[训练] 设备 {dev}, 机器人 {a.robot}")
    P03 = load_plan03()
    rng = np.random.default_rng(7)
    X, D, Q, PZ = gen_dataset(P03, a.n, rng)
    net, (qm, qs, xm, xs), _hist = train(np.column_stack([X, D]), Q, a.epochs, dev)
    ev = evaluate(P03, net, qm, qs, xm, xs, dev)

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
