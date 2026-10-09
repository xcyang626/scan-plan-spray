"""蹲姿(骨盆高度变化)+腰部±30°: 腕部垂直可达区与10°覆盖区计算"""
import numpy as np
import sys
sys.path.insert(0, r"c:\Users\Yangon\Desktop\舱室绝缘喷涂人形机器人技术方案\喷涂点位划分\work")
from ws_calc import J, L_FA, D_G, rotx, roty, rotz
from scipy import ndimage

rng = np.random.default_rng(2027)
COS_TOL = 0.92
PLANE_TOL = 0.06
H_MIN, H_MAX = 0.45, 0.96      # 蹲姿骨盆高度工程范围 (URDF腿FK, 足底贴地)
YAW = (-np.deg2rad(30), np.deg2rad(30))

def fk_batch2(q, pelvis_z):
    N = q.shape[0]
    R = np.tile(np.eye(3), (N, 1, 1))
    p = np.stack([np.zeros(N), np.zeros(N), pelvis_z], axis=1)
    for i, (nm, xyz, rpy, ax, lo, hi) in enumerate(J):
        p = p + R @ np.array(xyz)
        Rr = R @ rotx(rpy[0]) @ roty(rpy[1]) @ rotz(rpy[2])
        a = q[:, i]
        Ri = np.zeros((N, 3, 3))
        if ax == 'x':
            Ri[:, 0, 0] = 1.0
            Ri[:, 1, 1] = np.cos(a); Ri[:, 1, 2] = -np.sin(a)
            Ri[:, 2, 1] = np.sin(a); Ri[:, 2, 2] = np.cos(a)
        elif ax == 'y':
            Ri[:, 0, 0] = np.cos(a); Ri[:, 0, 2] = np.sin(a); Ri[:, 1, 1] = 1.0
            Ri[:, 2, 0] = -np.sin(a); Ri[:, 2, 2] = np.cos(a)
        else:
            Ri[:, 0, 0] = np.cos(a); Ri[:, 0, 1] = -np.sin(a)
            Ri[:, 1, 0] = np.sin(a); Ri[:, 1, 1] = np.cos(a); Ri[:, 2, 2] = 1.0
        R = np.einsum('nij,njk->nik', Rr, Ri)
    return p, R

lo = np.array([j[4] for j in J]); hi = np.array([j[5] for j in J])
out = []
n_done = 0
TOTAL, BATCH, MAX_KEEP = 15_000_000, 1_500_000, 400_000
while n_done < TOTAL and sum(len(o) for o in out) < MAX_KEEP:
    q = rng.uniform(lo, hi, size=(BATCH, 7))
    q[:, 0] = np.clip(q[:, 0], YAW[0], YAW[1])
    # 混合采样: 一半站姿(骨盆0.96), 一半蹲姿(骨盆0.45~0.96)
    pz = rng.uniform(H_MIN, H_MAX, BATCH)
    pz[: BATCH // 2] = H_MAX
    p_elbow, R_elbow = fk_batch2(q, pz)
    fa = -R_elbow[:, :, 2]
    wrist = p_elbow + L_FA * fa
    m = (fa[:, 0] > COS_TOL) & (np.abs(wrist[:, 0] - D_G) < PLANE_TOL)
    if m.any():
        w = wrist[m]
        out.append(np.stack([-w[:, 1], w[:, 2], pz[m]], axis=1))
    n_done += BATCH
    print(f"  {n_done/1e6:.0f}M done, kept={sum(len(o) for o in out)}")

yzp = np.concatenate(out)
if len(yzp) > 150000:
    yzp = yzp[rng.choice(len(yzp), 150000, replace=False)]
np.save(r"c:\Users\Yangon\Desktop\舱室绝缘喷涂人形机器人技术方案\喷涂点位划分\work\ws_s30sq.npy", yzp)

def grid(yz, res=0.02):
    x0, x1 = yz[:, 0].min(), yz[:, 0].max()
    z0, z1 = yz[:, 1].min(), yz[:, 1].max()
    xe = np.arange(x0, x1 + res, res); ze = np.arange(z0, z1 + res, res)
    H, _, _ = np.histogram2d(yz[:, 0], yz[:, 1], bins=[xe, ze])
    return (H > 0), xe, ze

G, xe, ze = grid(yzp[:, :2])
G = ndimage.binary_closing(G, np.ones((2, 2)))
area = G.sum() * 0.0004
print(f"\n±30°+蹲姿(骨盆{H_MIN}~{H_MAX}m): n={len(yzp)}, area≈{area:.2f} m²")
print(f"  y=[{yzp[:,0].min():.2f},{yzp[:,0].max():.2f}], z=[{yzp[:,1].min():.2f},{yzp[:,1].max():.2f}]")

# 站姿子集 (骨盆>=0.94)
stand = yzp[yzp[:, 2] >= 0.94]
print(f"  站姿子集: n={len(stand)}, z=[{stand[:,1].min():.2f},{stand[:,1].max():.2f}]")
# 蹲姿子集 (骨盆<=0.60)
sq = yzp[yzp[:, 2] <= 0.60]
print(f"  深蹲子集(骨盆<=0.60): n={len(sq)}, z=[{sq[:,1].min():.2f},{sq[:,1].max():.2f}]")

# 10°扇形覆盖区
THETA = np.deg2rad(10)
R_SPRAY = D_G * np.tan(THETA / 2)
G2, xe2, ze2 = grid(yzp[:, :2], 0.01)
G2 = ndimage.binary_closing(G2, np.ones((2, 2)))
k = int(np.ceil(R_SPRAY / 0.01))
Gcov = ndimage.binary_dilation(G2, ndimage.generate_binary_structure(2, 1), iterations=k)
area_cov = Gcov.sum() * 1e-4
area_reach = G2.sum() * 1e-4
ys = np.where(Gcov.any(axis=1))[0]; zs = np.where(Gcov.any(axis=0))[0]
print(f"  reach(0.01)={area_reach:.2f} m², coverage(10°)={area_cov:.2f} m², R={R_SPRAY:.3f} m")
print(f"  coverage bbox: y=[{xe2[ys[0]]-0.005:.2f},{xe2[ys[-1]]+0.005:.2f}] "
      f"z=[{ze2[zs[0]]-0.005:.2f},{ze2[zs[-1]]+0.005:.2f}]")
