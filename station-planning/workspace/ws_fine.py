"""工作空间精细采样 + 10°扇形覆盖区计算，保存绘图数据"""
import numpy as np
import sys
sys.path.insert(0, r"c:\Users\Yangon\Desktop\舱室绝缘喷涂人形机器人技术方案\喷涂点位划分\work")
from ws_calc import fk_batch, J, L_FA, D_G, PELVIS_Z
from scipy import ndimage

rng = np.random.default_rng(2026)
COS_TOL = 0.92   # 小臂与墙面法向夹角 <= 23° 视为垂直喷涂
PLANE_TOL = 0.06

def sample_batch(n):
    lo = np.array([j[4] for j in J]); hi = np.array([j[5] for j in J])
    q = rng.uniform(lo, hi, size=(n, 7))
    return q

def collect(yaw_range, total=15_000_000, batch=1_500_000, max_keep=400_000):
    lo = np.array([j[4] for j in J]); hi = np.array([j[5] for j in J])
    out = []
    n_done = 0
    while n_done < total and sum(len(o) for o in out) < max_keep:
        q = sample_batch(batch)
        q[:, 0] = np.clip(q[:, 0], yaw_range[0], yaw_range[1])
        p_elbow, R_elbow = fk_batch(q)
        fa = -R_elbow[:, :, 2]
        wrist = p_elbow + L_FA * fa
        m = (fa[:, 0] > COS_TOL) & (np.abs(wrist[:, 0] - D_G) < PLANE_TOL)
        if m.any():
            out.append(wrist[m])
        n_done += batch
    w = np.concatenate(out)
    # 降采样用于绘图
    if len(w) > 120000:
        w = w[rng.choice(len(w), 120000, replace=False)]
    return w

def grid(yz, res=0.02):
    x0, x1 = yz[:, 0].min(), yz[:, 0].max()
    z0, z1 = yz[:, 1].min(), yz[:, 1].max()
    xe = np.arange(x0, x1 + res, res); ze = np.arange(z0, z1 + res, res)
    H, _, _ = np.histogram2d(yz[:, 0], yz[:, 1], bins=[xe, ze])
    return (H > 0), xe, ze

yaw_sets = {
    "s5":   (-np.deg2rad(5), np.deg2rad(5)),
    "s30":  (-np.deg2rad(30), np.deg2rad(30)),
    "s90":  (-np.deg2rad(90), np.deg2rad(90)),
    "full": (-2.618, 3.229),
}

W = r"c:\Users\Yangon\Desktop\舱室绝缘喷涂人形机器人技术方案\喷涂点位划分\work"
summary = {}
for nm, yr in yaw_sets.items():
    w = collect(yr)
    yz = np.stack([-w[:, 1], w[:, 2]], axis=1)   # 墙面坐标 y(正=右) = -y_ros
    G, xe, ze = grid(yz)
    # 形态学闭运算去噪
    G = ndimage.binary_closing(G, np.ones((2, 2)))
    area = G.sum() * 0.02 * 0.02
    y0, y1 = yz[:, 0].min(), yz[:, 0].max()
    z0, z1 = yz[:, 1].min(), yz[:, 1].max()
    # 最大内接矩形 (近似: 每行最大连续True)
    best_w = best_h = best_area = 0
    for i in range(G.shape[0]):
        row = G[i]
        # 最长连续段
        d = np.diff(np.concatenate([[0], row.view(np.int8), [0]]))
        starts = np.where(d == 1)[0]; ends = np.where(d == -1)[0]
        if len(starts):
            j0, j1 = starts[np.argmax(ends - starts)], 0
            j1 = ends[np.argmax(ends - starts)]
            h = (j1 - j0) * 0.02; ww = 0  # placeholder
    summary[nm] = dict(area=area, y0=y0, y1=y1, z0=z0, z1=z1, n=len(yz))
    print(f"{nm}: n={len(yz)}, area={area:.2f} m², y=[{y0:.2f},{y1:.2f}], z=[{z0:.2f},{z1:.2f}]")
    np.save(W + rf"\ws_{nm}.npy", yz)

# ---- 选定 ±30°: 计算 10° 扇形喷涂覆盖区 ----
THETA = np.deg2rad(10)
R_SPRAY = D_G * np.tan(THETA / 2)     # 0.035m
yz30 = np.load(W + r"\ws_s30.npy")
G, xe, ze = grid(yz30, 0.01)
G = ndimage.binary_closing(G, np.ones((2, 2)))
# 膨胀 R_SPRAY -> 覆盖区
k = int(np.ceil(R_SPRAY / 0.01))
Gcov = ndimage.binary_dilation(G, ndimage.generate_binary_structure(2, 1), iterations=k)
area_cov = Gcov.sum() * 0.0001
area_reach = G.sum() * 0.0001
print(f"\n±30°: reach area={area_reach:.2f} m², spray(10°) coverage area={area_cov:.2f} m², R={R_SPRAY:.3f} m")
print(f"coverage y=[{xe[0]-R_SPRAY:.2f},{xe[-1]+R_SPRAY:.2f}], z=[{ze[0]-R_SPRAY:.2f},{ze[-1]+R_SPRAY:.2f}]")

# 包络矩形(覆盖区)
ys = np.where(Gcov.any(axis=1))[0]; zs = np.where(Gcov.any(axis=0))[0]
print(f"coverage bbox: y=[{xe[ys[0]]:.2f},{xe[ys[-1]]:.2f}] z=[{ze[zs[0]]:.2f},{ze[zs[-1]]:.2f}]")

np.save(W + r"\ws_summary.npy", summary, allow_pickle=True)
with open(W + r"\ws_stats.txt", "w") as f:
    for k2, v in summary.items():
        f.write(f"{k2}: {v}\n")
    f.write(f"R_SPRAY={R_SPRAY:.4f}, area_reach={area_reach:.3f}, area_cov={area_cov:.3f}\n")
    f.write(f"cov_y=[{xe[ys[0]]:.3f},{xe[ys[-1]]:.3f}], cov_z=[{ze[zs[0]]:.3f},{ze[zs[-1]]:.3f}]\n")
