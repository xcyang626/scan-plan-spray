"""天宫3 URDF 右臂 FK + 蒙特卡洛采样：腕部垂直墙面可达范围计算"""
import numpy as np

rng = np.random.default_rng(42)

def rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

def roty(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

# 关节: (name, origin_xyz, origin_rpy, axis, lower, upper)  右臂链
J = [
    ("waist_yaw",   (0, 0, 0),          (0, 0, 0.0013415), 'z', -2.618, 3.229),
    ("waist_roll",  (0, 0, 0.083796),   (0, 0, 0),         'x', -0.436, 0.436),
    ("waist_pitch", (0, 0, 0.078424),   (0, 0, 0),         'y', -0.524, 0.873),
    ("sh_pitch",    (0, -0.14552, 0.18664), (-0.087266, 0, 0), 'y', -2.845, 2.845),
    ("sh_roll",     (0, -0.0678, 0),    (0, 0, 0),         'x', -3.334, 0.192),
    ("sh_yaw",      (0, 0, -0.1004),    (0, 0, 0),         'z', -2.845, 2.845),
    ("elbow_pitch", (0.02, 0, -0.1596), (0, 0, 0),         'y', -2.548, 0.192),
]
AX = {'x': rotx, 'y': roty, 'z': rotz}
PELVIS_Z = 0.96   # 直立时骨盆高度 (m)
L_FA = 0.25       # 肘关节 -> 腕部(喷枪安装) 小臂有效长 (m)
D_G = 0.40        # 腕部距墙
D_R = 0.80        # 机器人中心距墙

def fk_batch(q):  # q: (N,7)
    N = q.shape[0]
    R = np.tile(np.eye(3), (N, 1, 1))
    p = np.tile(np.array([0.0, 0.0, PELVIS_Z]), (N, 1))
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
    return p, R  # 肘关节位置, 肘姿态

def sample_workspace(yaw_range, n=4_000_000, cos_tol=0.92, plane_tol=0.08):
    """采样并过滤: 小臂方向(肘系-z) 垂直墙面(+x, 夹角<=23°即方向余弦>=0.92), 腕部位于 x≈D_G 平面
    机器人坐标系: x指向前方墙面, y左, z上 (ROS)。墙在 x=D_R。
    """
    lo = np.array([j[4] for j in J]); hi = np.array([j[5] for j in J])
    q = rng.uniform(lo, hi, size=(n, 7))
    q[:, 0] = np.clip(q[:, 0], yaw_range[0], yaw_range[1])  # 腰部扭转范围
    p_elbow, R_elbow = fk_batch(q)
    fa_dir = -R_elbow[:, :, 2]        # 小臂方向 = 肘系 -z
    wrist = p_elbow + L_FA * fa_dir   # 腕部(喷枪安装位置)
    # 喷枪垂直墙面: 小臂方向 ≈ +x (朝墙)
    m_perp = fa_dir[:, 0] > cos_tol
    m_plane = np.abs(wrist[:, 0] - D_G) < plane_tol
    m = m_perp & m_plane
    return wrist[m]

def alpha_area(pts_yz, res=0.02):
    """occupancy 网格面积"""
    if len(pts_yz) < 100:
        return 0.0, None
    H, xe, ye = np.histogram2d(pts_yz[:, 0], pts_yz[:, 1],
                               bins=[np.arange(pts_yz[:, 0].min(), pts_yz[:, 0].max() + res, res),
                                     np.arange(pts_yz[:, 1].min(), pts_yz[:, 1].max() + res, res)])
    area = (H > 0).sum() * res * res
    return area, (H, xe, ye)

if __name__ == "__main__":
    yaw_sets = {
        "无扭转[-5,5]":   (-np.deg2rad(5), np.deg2rad(5)),
        "合理扭转[-30,30]": (-np.deg2rad(30), np.deg2rad(30)),
        "大扭转[-90,90]":  (-np.deg2rad(90), np.deg2rad(90)),
        "URDF全范围":      (-2.618, 3.229),
    }
    for nm, yr in yaw_sets.items():
        w = sample_workspace(yr, n=3_000_000)
        # 墙面坐标: y_wall = -y_ros (正=机器人右手侧/墙右侧)
        yz = np.stack([-w[:, 1], w[:, 2]], axis=1)
        area, _ = alpha_area(yz, 0.02)
        print(f"{nm}: kept={len(w)}, area≈{area:.2f} m², "
              f"y=[{yz[:,0].min():.2f},{yz[:,0].max():.2f}], z=[{yz[:,1].min():.2f},{yz[:,1].max():.2f}]")
    # 注: 旧的 ws_results.npy 死输出已删除(无下游脚本消费该文件)
