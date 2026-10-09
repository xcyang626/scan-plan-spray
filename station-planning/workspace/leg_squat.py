"""天宫3 URDF 左腿FK: 足底贴地约束下可达骨盆高度范围(蹲姿)"""
import numpy as np

def rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def roty(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

# 左腿链 (origin_xyz, origin_rpy, axis, lower, upper)
LEG = [
    ("hip_pitch",   (-0.00010305, 0.076815, -0.11847), (-0.17453, 0, 0.0013415), 'y', -3.0543261909900763,  2.792526803190927),
    ("hip_roll",    (0,           0.073992, -0.026966), (0.17453, 0, 0),          'x', -0.41887902047863906, 2.6179938779914944),
    ("hip_yaw",     (0,           0,        -0.0785),   (0, 0, 0),                'z', -1.3962634015954636,  4.4505895925855405),
    ("knee_pitch",  (-0.012927,   0,        -0.30599),  (0, 0.018418, 0),         'y', -0.08726646259971647, 2.5307274153917776),
    ("ankle_pitch", (0.0073681,   0,        -0.39993),  (0, 0, 0),                'y', -1.1868238913561442,  0.5235987755982988),
    ("ankle_roll",  (0,           0,         0),        (0, 0, 0),                'x', -0.5235987755982988,  0.5235987755982988),
]
AX = {'x': rotx, 'y': roty, 'z': rotz}
PELVIS_Z_STAND = 0.96

def leg_fk(q):
    R = np.eye(3); p = np.zeros(3)
    for (nm, xyz, rpy, ax, lo, hi), qi in zip(LEG, q):
        p = p + R @ np.array(xyz)
        Rr = R @ rotx(rpy[0]) @ roty(rpy[1]) @ rotz(rpy[2])
        R = Rr @ AX[ax](qi)
    return p, R   # 踝关节位置(骨盆系), 踝姿态

# 直立标定: 全0关节 → 踝关节在骨盆下方 dz; 足底贴地时踝离地高度 = 0.96 + p_stand[2]
p0, R0 = leg_fk(np.zeros(6))
ankle_ground = PELVIS_Z_STAND + p0[2]
print(f"直立: 踝相对骨盆 z = {p0[2]:.4f} m, 踝离地 ≈ {ankle_ground:.4f} m")
print(f"足平姿态校验(直立): 踝姿态pitch分量 ≈ {np.arcsin(-R0[0,2]):.4f} rad")

# 采样: 矢状面蹲姿 (hip_pitch, knee_pitch 变化, roll/yaw=0, 足底贴地)
rng = np.random.default_rng(11)
N = 2_000_000
hip = rng.uniform(-3.0543, 2.7925, N)
knee = rng.uniform(-0.0873, 2.5307, N)
ankle = -(hip + knee)                       # 足底贴地
lo_a, hi_a = -1.1868, 0.5236
ok = (ankle >= lo_a) & (ankle <= hi_a)
print(f"足底贴地可行样本: {ok.sum()} / {N} ({ok.mean()*100:.1f}%)")

hs = []
for h, k in zip(hip[ok][:200000], knee[ok][:200000]):
    p, R = leg_fk(np.array([h, 0, 0, k, -(h+k), 0]))
    hs.append(ankle_ground - p[2])          # 骨盆离地高度
hs = np.array(hs)
# 足平约束下足姿态应为水平: 校验
p_chk, R_chk = leg_fk(np.array([1.0, 0, 0, 1.0, -2.0, 0]))
print(f"校验(h=1.0,k=1.0): 足姿态pitch ≈ {np.arcsin(-R_chk[0,2]):.4f} rad (应≈0)")
print(f"骨盆高度范围: [{hs.min():.3f}, {hs.max():.3f}] m")
for pct in [1, 5, 10, 25, 50]:
    print(f"  P{pct:02d}: {np.percentile(hs, pct):.3f} m")
# 深蹲姿态示例
i_min = hs.argmin()
print(f"最深蹲姿: hip={hip[ok][:200000][i_min]:.3f} rad, knee={knee[ok][:200000][i_min]:.3f} rad, "
      f"pelvis={hs.min():.3f} m")
