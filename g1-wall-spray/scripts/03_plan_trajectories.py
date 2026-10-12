# -*- coding: utf-8 -*-
"""步骤3: 生成墙面之字形喷涂轨迹 + 机器人(G1)之字形轨迹, 并逐点IK验证.

设计(站姿单一 + 等宽完整之字形 + 腕心水平线 + 离散角度, 用户定稿):
  1. 只保留站姿(骨盆 0.77~0.79m, 腰roll/pitch ±0.0873(±5°) 与步骤2一致), 无蹲/站分区
  2. 行高序列: 行距0.10m; 最低行 = (z_bot - 0.40*tan30°) 向上取到 k*PITCH+PITCH/2 网格;
     最高行 1.45m; hs_seq 从低到高
  3. 等宽 + 离散行角 + 腕心水平线 联合求解(核心):
       - 行角 θ 只取 {90,80,70,60}° 之一, 每行为行常数, 可行前提下取最大
       - 腕心每行走严格水平直线: z_w = h + dir*0.40*tan(90°-θ) 为行常数,
         dir=+1 腕在墙点上方(枪下倾,用tiltd网格), dir=-1 腕在下方(枪上倾,用tiltu网格),
         θ=90 时 z_w=h(perp网格); y_r=u-u_robot 与墙面行同区间同步扫描(腕/墙同宽);
         指向恒为纯俯仰(左右分量=0)
       - 可行性: 候选水平线段 [u_lo,u_hi](两端内收0.02) 上每0.01m 的 (y_r,z_w)
         都落在对应网格(站姿 right_stand_y60_perp/tiltd/tiltu)内
       - 统一宽度 W 从0.70起以0.01递减, 中心u_c=0.36(区间clip到[0.005,0.715]),
         取首个使每一行 h 都存在至少一个可行(θ,dir) 的 W; 每行再取其可行档中最大θ
  4. 腕心轨迹 = 干净标准之字形: 行内水平直线(z_w常数), 行端竖直U型连接段
     (y停行端, z_w(i)->z_w(i+1)线性, 对应墙点同u竖直移动, 连接段角度在两行角之间
     线性插值, 仍纯俯仰, 倾角≤30°); 禁止任何逐点网格投影/边界跟踪
  5. 工作区域保证: 站位离墙 0.70~0.78m(定位误差) -> 三腕平面(0.30/0.34/0.38)保证区(步骤2);
     每个轨迹点再用批量阻尼最小二乘IK(腰3+臂7+骨盆高)验证(MC种子+热启动波次+容差档)
  6. 解序列连续性整流: 全部收敛后逐点顺序重解(前解作种子, 差<0.35rad才替换, 2遍),
     仍>0.6rad的相邻对用线性插值构型作种子再试; FK夹角不达标回退整流前解
"""
import numpy as np
import os, json, csv
from scipy import ndimage
from scipy.spatial import cKDTree
from nn_ik_hook import nn_ik

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("G1W_OUT", os.path.join(BASE, "output"))
_WRP = float(os.environ.get("G1W_WAIST_RP", "0"))
IN = os.environ.get("G1W_IN", OUT)      # 只读静态输入(工作区网格/舱室场景); 缺省=主output

# ---------------- 参数 ----------------
X_FACE = 212.292          # 墙内面 (m, 世界坐标)
FLOOR_Z = 57.563          # 地面 (m)
U0 = 7.789                # 段起点世界y (m)  (7789mm)
D_ROBOT = 0.74            # 机器人离墙标称距离(腰锁定后保证带 0.70~0.78m)
D_WRIST = 0.40            # 腕心距墙 (末端前一关节)
PITCH = 0.10              # 之字形行距
POINT_SPACING = 0.02      # 轨迹点间距
STROKE_U = (0.005, 0.715) # 喷涂横向扫描范围 (最右止于加强筋左侧面0.720前)
H_TOP = 1.45              # 最高行 (m)
TILT_MAX = 30.0           # 倾角上限: 行角离散{90,80,70,60}°=>倾角≤30°(与墙面夹角≥60°), 连接段插值亦不超
Z_MIN_ROBOT = 0.15        # 机器人轨迹最低安全高度
GUN_LEN = 0.12            # 喷枪长度(腕心->喷嘴)
ARM = "right"; YAW = 60

# ---- 工作站配置注入 (10_station_app.py / 11_cloud_scan.py 产出; 缺省=舱室默认) ----
# G1W_CFG json 可覆盖: d_robot, wall_x, floor_z, u0, stroke_u, h_top, pitch, posture
_cfgp = os.environ.get("G1W_CFG", "")
_POSTURE = "stand"
_CROUCH_TOP = None
if _cfgp and os.path.exists(_cfgp):
    _c = json.load(open(_cfgp, encoding="utf8"))
    if "wall_x" not in _c:
        # cfg 存在却缺 wall_x: 墙面x会被静默替换成站位距离(世界坐标全错), 直接报错退出
        raise SystemExit(f"[配置] 错误: G1W_CFG({_cfgp}) 缺 wall_x 键(点云机器人系墙距), "
                         f"拒绝回退默认值; 请检查工作站/云扫描产出的 cfg")
    D_ROBOT = float(_c.get("d_robot", D_ROBOT))
    X_FACE = float(_c["wall_x"])                    # 点云机器人系: 墙距=wall_x
    FLOOR_Z = float(_c.get("floor_z", 0.0))
    U0 = float(_c.get("u0", 0.0))
    if _c.get("stroke_u"):
        STROKE_U = tuple(float(v) for v in _c["stroke_u"])
    if _c.get("h_top") is not None:
        H_TOP = float(_c["h_top"])
    if _c.get("pitch") is not None:
        PITCH = float(_c["pitch"])
    _POSTURE = str(_c.get("posture", "stand"))
    if _c.get("crouch_top"):
        _CROUCH_TOP = float(_c["crouch_top"])
    print(f"[配置] 工作站配置已加载 {_cfgp}: d_robot={D_ROBOT}, wall_x={X_FACE}, "
          f"stroke_u=({STROKE_U[0]:.3f},{STROKE_U[1]:.3f}), h_top={H_TOP}, pitch={PITCH}, "
          f"posture={_POSTURE}")

# V2 双姿态(2026-10): 缺省 stand = v5 站姿单一模式, 行为零改动;
# dual = 站姿先喷上半(0.55~H_TOP) -> 回零位过渡 -> 蹲姿喷下半(物理下限~0.45), 用户定稿顺序.
# 环境变量 G1W_POSTURE 优先于 G1W_CFG 的 posture 键.
POSTURE = os.environ.get("G1W_POSTURE", "") or _POSTURE
if POSTURE not in ("stand", "dual"):
    raise SystemExit(f"[配置] posture 只支持 stand/dual, 收到 {POSTURE!r}")
# V2 蹲姿带上边界(用户定稿 2026-10-10: 0.55/0.65 两行归蹲姿, 站姿带只喷 0.75~H_TOP,
# 减少姿态切换且站姿带底缘不再依赖极限俯仰档); None=自动(站姿最低可行行以下归蹲姿, 首版行为)
CROUCH_TOP = float(os.environ.get("G1W_CROUCH_TOP", "0") or 0) or _CROUCH_TOP or None
# V3 走线方向(用户定稿 2026-10-12): vertical=竖直列 —— 喷涂线平行于"机器人-墙面法向"
# 的垂面(枪臂恒在该垂面内, 俯仰±30°, 60/70/80/90 离散角被认可, 横向分量恒0);
# 横向只在列间跳转(kind=hop 非喷涂)。horizontal=旧水平行(回退用)。
STROKE_DIR = (os.environ.get("G1W_STROKE_DIR", "horizontal") or "horizontal").strip().lower()
COL_PITCH_MAX = 0.15        # 列距上限(喷幅横向覆盖)

# 腕平面 x 越界检查(启动时): 02 工作区网格只覆盖三腕平面(0.30/0.34/0.38)保证带,
# x=D_ROBOT-D_WRIST 超出 [0.30,0.38]±0.03 时网格可能不覆盖该平面, 打警告(不退出)
_wx_chk = D_ROBOT - D_WRIST
if not (0.30 - 0.03 <= _wx_chk <= 0.38 + 0.03):
    print(f"[警告] 腕平面 x=D_ROBOT-D_WRIST={_wx_chk:.3f}m 超出保证带 [0.30,0.38]±0.03 "
          f"(d_robot={D_ROBOT}); 工作区网格可能不覆盖该平面, 可行区判定或失真")

# ---------------- G1 FK (与步骤2一致, 官方 URDF g1_29dof_nohand) ----------------
# 腰部三自由度: yaw放开(±150°), roll/pitch 收紧至 ±5°=±0.0873(与步骤2一致)
WAIST = [
    ((0.0, 0.0, 0.0), (0, 0, 0), 'z', -2.618, 2.618),
    ((-0.0039635, 0, 0.035), (0, 0, 0), 'x',
     -(_WRP if _WRP > 0 else 0.0872665), (_WRP if _WRP > 0 else 0.0872665)),
    ((0.0, 0.0, 0.019), (0, 0, 0), 'y',
     -(_WRP if _WRP > 0 else 0.0872665), (_WRP if _WRP > 0 else 0.0872665)),
]
ARM_R = [
    ((0.0039563, -0.10021, 0.23778), (-0.27931, 5.4949e-05, 0.00019159), 'y', -3.0892, 2.6704),
    ((0.0, -0.038, -0.013831), (0.27925, 0, 0), 'x', -2.2515, 1.5882),
    ((0.0, -0.00624, -0.1032), (0, 0, 0), 'z', -2.618, 2.618),
    ((0.015783, 0, -0.080518), (0, 0, 0), 'y', -1.0472, 2.0944),
    ((0.100, -0.00188791, -0.010), (0, 0, 0), 'x', -1.9722, 1.9722),
    ((0.038, 0, 0), (0, 0, 0), 'y', -1.6144, 1.6144),
    ((0.046, 0, 0), (0, 0, 0), 'z', -1.6144, 1.6144),
]
CHAIN = WAIST + ARM_R
JLO = np.array([j[3] for j in CHAIN]); JHI = np.array([j[4] for j in CHAIN])
# V3 稳定性(2026-10-12 实测教训): 腰yaw URDF 全限位 ±150° 会让 IK 冗余解持续拧腰
# (实测爬到 +150°, 躯干拧转把底座拖转 +130°, 落点/覆盖全毁). 工作区网格本按
# ±30/60° 腰yaw档建立, 冗余该由手臂消化 —— 腰yaw硬限 ±35°(档位+裕量).
_WYAW_MAX = np.deg2rad(float(os.environ.get("G1W_WAIST_YAW_MAX", "45")))
JLO[0] = -_WYAW_MAX; JHI[0] = _WYAW_MAX
PELVIS_RANGE = {"stand": (0.77, 0.79)}          # 站姿(默认, v5 口径)
PELVIS_RANGE.update({                            # V2 双姿态增补:
    "crouch": (float(os.environ.get("G1W_CROUCH_LO", "0")) or 0.60, 0.70),  # 蹲姿地板=02 同源(实测可稳到0.29)
    "trans": (float(os.environ.get("G1W_CROUCH_LO", "0")) or 0.60, 0.79)})  # 站<->蹲过渡段(骨盆自由穿越)


def rotx(a):
    c, s = np.cos(a), np.sin(a)
    R = np.zeros((len(a), 3, 3))
    R[:, 0, 0] = 1; R[:, 1, 1] = c; R[:, 1, 2] = -s
    R[:, 2, 1] = s; R[:, 2, 2] = c
    return R


def roty(a):
    c, s = np.cos(a), np.sin(a)
    R = np.zeros((len(a), 3, 3))
    R[:, 0, 0] = c; R[:, 0, 2] = s; R[:, 1, 1] = 1
    R[:, 2, 0] = -s; R[:, 2, 2] = c
    return R


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    R = np.zeros((len(a), 3, 3))
    R[:, 0, 0] = c; R[:, 0, 1] = -s; R[:, 1, 0] = s; R[:, 1, 1] = c; R[:, 2, 2] = 1
    return R


AXF = {'x': rotx, 'y': roty, 'z': rotz}


def fk_wrist(q, pz):
    """q:(N,10), pz:(N,) -> 腕心位置(N,3), 枪轴(N,3)  (机器人基座系, x朝墙)"""
    N = q.shape[0]
    R = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
    p = np.broadcast_to(np.array([0.0, 0.0, 0.0]), (N, 3)).copy()
    p[:, 2] = pz
    for i, (xyz, rpy, ax, lo, hi) in enumerate(CHAIN):
        p = p + np.einsum('nij,j->ni', R, np.array(xyz))
        Rr = R @ (rotx(np.full(N, rpy[0])) @ roty(np.full(N, rpy[1])) @ rotz(np.full(N, rpy[2])))
        R = np.einsum('nij,njk->nik', Rr, AXF[ax](q[:, i]))
    return p, R[:, :, 0]


def fk_jac(q, pz):
    """一次FK同时返回 腕心(N,3), 枪轴(N,3), 位置雅可比(N,3,10), 轴雅可比(N,3,10).
    解析式: d p/dq_i = a_i x (p_end - o_i);  d axis/dq_i = a_i x axis
    其中 a_i = 第i关节轴向量(基系), o_i = 第i关节原点."""
    N = q.shape[0]
    R = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
    p = np.broadcast_to(np.array([0.0, 0.0, 0.0]), (N, 3)).copy()
    p[:, 2] = pz
    e_ax = {'x': [1., 0., 0.], 'y': [0., 1., 0.], 'z': [0., 0., 1.]}
    orgs = np.empty((N, 10, 3)); axv = np.empty((N, 10, 3))
    for i, (xyz, rpy, ax, lo, hi) in enumerate(CHAIN):
        p = p + np.einsum('nij,j->ni', R, np.array(xyz))
        Rr = R @ (rotx(np.full(N, rpy[0])) @ roty(np.full(N, rpy[1])) @ rotz(np.full(N, rpy[2])))
        axv[:, i] = np.einsum('nij,j->ni', Rr, np.array(e_ax[ax]))
        orgs[:, i] = p
        R = np.einsum('nij,njk->nik', Rr, AXF[ax](q[:, i]))
    axis = R[:, :, 0]
    rel = p[:, None, :] - orgs                       # (N,10,3)
    Jp = np.cross(axv, rel)                          # (N,10,3)
    Ja = np.cross(axv, axis[:, None, :])
    return p, axis, Jp.transpose(0, 2, 1), Ja.transpose(0, 2, 1)


# ---------------- 批量 DLS IK ----------------
def batch_ik(q0, pz0, p_star, d_star, mode, iters=500, pos_tol=0.004, ax_tol=0.03, pb=None):
    """阻尼最小二乘: 腕心->p_star, 枪轴->d_star. 变量 = q(10)+骨盆高(1).
    纯DLS(无零空间种子拉力: 拉力会在误差大时把解拖离目标并冻结).
    pb=(N,2) 每点骨盆界限(V2 双姿态按点分 stand/crouch/trans); None=按 mode 标量界限(legacy).
    返回 (q, pz, pos_err, axis_err, converged)"""
    N = len(q0)
    X = np.column_stack([q0, pz0]).astype(float)
    if pb is not None:
        pba = np.asarray(pb, float)                     # (N,2) 每点骨盆下/上限
        lo2 = np.column_stack([np.tile(JLO, (N, 1)), pba[:, 0:1]])   # (N,11)
        hi2 = np.column_stack([np.tile(JHI, (N, 1)), pba[:, 1:2]])
    else:
        plo, phi = PELVIS_RANGE[mode]
        lo1 = np.append(JLO, plo); hi1 = np.append(JHI, phi)
    lam = 0.08
    done = np.zeros(N, bool)
    for it in range(iters):
        act = np.where(~done)[0]
        if len(act) == 0:
            break
        Xa = X[act]
        p_cur, ax_cur, Jp, Ja = fk_jac(Xa[:, :10], Xa[:, 10])
        e = np.concatenate([p_star[act] - p_cur, d_star[act] - ax_cur], 1)   # (na,6)
        J = np.empty((len(act), 6, 11))
        J[:, :3, :10] = Jp
        J[:, 3:, :10] = Ja
        J[:, :, 10] = 0.0
        J[:, 2, 10] = 1.0                          # 骨盆高度仅平移 z
        JJt = np.einsum('nij,nkj->nik', J, J)
        A = JJt + (lam ** 2) * np.eye(6)[None]
        y = np.linalg.solve(A, e[..., None])[..., 0]        # (na,6)
        dq = np.einsum('nij,ni->nj', J, y)                  # J^T y = DLS 步
        if pb is not None:
            X[act] = np.clip(Xa + dq, lo2[act], hi2[act])
        else:
            X[act] = np.clip(Xa + dq, lo1[None, :], hi1[None, :])
        p_cur, ax_cur = fk_wrist(X[act, :10], X[act, 10])
        pe = np.linalg.norm(p_star[act] - p_cur, axis=1)
        ae = np.linalg.norm(d_star[act] - ax_cur, axis=1)
        ok = (pe < pos_tol) & (ae < ax_tol)
        done[act[ok]] = True
        lam = max(0.01, lam * 0.99)
    p_fin, ax_fin = fk_wrist(X[:, :10], X[:, 10])
    pe = np.linalg.norm(p_star - p_fin, axis=1)
    ae = np.linalg.norm(d_star - ax_fin, axis=1)
    conv = (pe < pos_tol) & (ae < ax_tol)
    return X[:, :10], X[:, 10], pe, ae, conv


# ---------------- 主流程 ----------------
def main():
    from common import setup_cn_font
    setup_cn_font()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = json.load(open(os.path.join(IN, "wall_section_meta.json"), encoding="utf8"))
    ws = np.load(os.path.join(IN, "g1_workspace.npz"))
    ys_g, zs_g = ws["ys"], ws["zs"]

    # ---------------- 站姿单一模式: 网格与IK种子 ----------------
    DUAL = POSTURE == "dual"
    g_perp = ws[f"{ARM}_stand_y{YAW}_perp"]          # 垂直指向(<=15°)保证区 (θ=90档)
    g_td = ws[f"{ARM}_stand_y{YAW}_tiltd"]           # 下倾(<=30°)可达 (dir=+1档)
    g_tu = ws[f"{ARM}_stand_y{YAW}_tiltu"]           # 上倾(<=30°)可达 (dir=-1档)
    qsub = ws[f"{ARM}_stand_y{YAW}_qsub"].astype(float)      # (N,11) 末列=骨盆高
    esub = ws[f"{ARM}_stand_y{YAW}_endsub"].astype(float)
    tree_stand = cKDTree(esub[:, :3])
    tree_all = tree_stand                            # 兜底树(仅站姿, 与stand一致)
    qsub_all = qsub
    if DUAL:                                         # V2: 蹲姿网格与种子(02 已算好 *_crouch_* 键)
        g_perp_c = ws[f"{ARM}_crouch_y{YAW}_perp"]
        g_td_c = ws[f"{ARM}_crouch_y{YAW}_tiltd"]
        g_tu_c = ws[f"{ARM}_crouch_y{YAW}_tiltu"]
        qsub_c = ws[f"{ARM}_crouch_y{YAW}_qsub"].astype(float)
        esub_c = ws[f"{ARM}_crouch_y{YAW}_endsub"].astype(float)
        tree_all = cKDTree(np.vstack([esub[:, :3], esub_c[:, :3]]))
        qsub_all = np.vstack([qsub, qsub_c])
        print("[V2双姿态] 蹲姿网格/IK种子已加载(骨盆 0.60~0.70), 执行序: "
              "站姿喷上半 -> 回零位过渡 -> 蹲姿喷下半")

    # 保证区(垂直) y 范围(站姿)
    yrows = np.where(g_perp.any(1))[0]
    Y_LO, Y_HI = float(ys_g[yrows.min()]), float(ys_g[yrows.max()])
    # 下倾(<=30°)可达的最低腕心高度 -> 可喷下限
    td_cols = np.where(g_td.any(0))[0]
    z_bot_grid = float(zs_g[td_cols.min()]) if len(td_cols) else 0.30
    Z_BOT = max(Z_MIN_ROBOT, z_bot_grid)
    H_MIN_SPRAY = Z_BOT - D_WRIST * np.tan(np.deg2rad(TILT_MAX))
    # 行高序列: 最低行向上取到 k*PITCH + PITCH/2 网格, 最高行 H_TOP, 从低到高
    k_min = int(np.ceil((H_MIN_SPRAY - PITCH / 2) / PITCH - 1e-9))
    k_min = max(k_min, 1)          # 下限钳位: 可喷下限过低时防负行高(轨迹入地)
    H_MIN_ROW = k_min * PITCH + PITCH / 2
    hs_seq = list(np.arange(H_MIN_ROW, H_TOP + 1e-9, PITCH))
    if not hs_seq:
        raise RuntimeError(f"行高序列为空: 最低行 {H_MIN_ROW:.2f}m > 最高行 {H_TOP:.2f}m "
                           f"(H_MIN_SPRAY={H_MIN_SPRAY:.3f}, 请检查 Z_BOT/H_TOP 配置)")
    _stand_all = list(hs_seq)                        # 未切分前的完整站姿行表
    if DUAL and CROUCH_TOP:
        hs_seq = [h for h in hs_seq if h > CROUCH_TOP + 1e-9]
        if not hs_seq:
            raise RuntimeError(f"crouch_top={CROUCH_TOP} 过高: 站姿带无剩余行")
        moved = [f"{h:.2f}" for h in _stand_all if h <= CROUCH_TOP + 1e-9]
        print(f"[V2双姿态] 蹲姿上边界 {CROUCH_TOP:.2f}m(用户定稿): 站姿带底缘 {moved} "
              f"归入蹲姿带 -> 站姿 {min(hs_seq):.2f}~{H_TOP:.2f}m 共{len(hs_seq)}行")
    zone_rows = ["stand"] * len(hs_seq)             # 执行序=高度升序(v5 口径)
    H_MIN_SPRAY_MIN = H_MIN_SPRAY
    Z_BOT_C = H_MIN_SPRAY_C = None
    crouch_cands = []
    if DUAL:
        # V2: 蹲姿带下倾最低腕心 -> 理论可喷下限; 行候选=站姿最低行以下全部网格行,
        # 真正保留哪些行由求解段的网格探测决定(理论下限只作参考口径)
        td_cols_c = np.where(g_td_c.any(0))[0]
        z_bot_c = float(zs_g[td_cols_c.min()]) if len(td_cols_c) else np.nan
        Z_BOT_C = max(Z_MIN_ROBOT, z_bot_c)
        H_MIN_SPRAY_C = Z_BOT_C - D_WRIST * np.tan(np.deg2rad(TILT_MAX))
        crouch_cands = sorted(set(
            [round(k * PITCH + PITCH / 2, 2) for k in range(1, k_min)]
            + ([round(h, 2) for h in _stand_all if h <= CROUCH_TOP + 1e-9]
               if CROUCH_TOP else [])))
        print(f"[V2双姿态] 蹲姿下倾最低腕心z={z_bot_c:.3f} -> 理论可喷下限 {H_MIN_SPRAY_C:.3f}m "
              f"(=Z_BOT_C {Z_BOT_C:.3f} - 0.40*tan30°); 行候选 {crouch_cands}(以网格探测为准)")
    print(f"[保证区] 站姿 y[{Y_LO:+.2f},{Y_HI:+.2f}], 下倾(<=30°)最低腕心z={z_bot_grid:.3f} "
          f"-> Z_BOT={Z_BOT:.3f}")
    print(f"[喷涂质量] 末端指向与墙面夹角须>=60°(倾角<=30°): 站姿可喷下限 h>={H_MIN_SPRAY:.3f}m "
          f"(= Z_BOT {Z_BOT:.3f} - 0.40*tan30°)")
    print(f"[行高] 站姿最低行 H_MIN_ROW={H_MIN_ROW:.2f}m (=可喷下限向上取到 k*{PITCH}+{PITCH/2} 网格), "
          f"最高行 {H_TOP:.2f}m, 行距 {PITCH}m, 站姿带 {len(hs_seq)} 行: "
          + " ".join(f"{h:.2f}" for h in hs_seq))

    # 机器人横向站位: 工作区中心对准喷涂带中心
    u_center = 0.5 * (STROKE_U[0] + STROKE_U[1])
    u_robot = u_center - 0.5 * (Y_LO + Y_HI)
    print(f"[站位] 机器人横向 u_robot={u_robot:.3f}m (世界y={U0 + u_robot:.3f}m), "
          f"离墙 {D_ROBOT}m (误差范围0.70~0.78)")

    # ---------------- D: 等宽 + 离散行角 + 腕心水平线 联合求解 ----------------
    # 行角θ∈{90,80,70,60}°(行常数, 可行前提下取最大); 腕心每行走严格水平线:
    #   z_w = h + dir*0.40*tan(90°-θ) 为行常数; dir=+1 腕在墙点上方(枪下倾,tiltd网格),
    #   dir=-1 腕在下方(枪上倾,tiltu网格); θ=90 时 z_w=h(perp网格); 指向纯俯仰。
    ANG_OPTS = [(90, +1), (80, +1), (80, -1), (70, +1), (70, -1), (60, +1), (60, -1)]

    def grid_for(theta, dirn, zmode="stand"):
        if zmode == "crouch":                       # V2: 蹲姿网格
            if theta == 90:
                return g_perp_c
            return g_td_c if dirn > 0 else g_tu_c
        if theta == 90:
            return g_perp
        return g_td if dirn > 0 else g_tu

    def zw_of(h, theta, dirn):
        return h + dirn * D_WRIST * np.tan(np.deg2rad(90.0 - theta))

    def iy_arr(y_vals):
        # 返回未夹紧的网格索引; 越界由 seg_feasible 判不可行(夹到边缘格会把网格外
        # 坐标误判成可行)
        y_vals = np.asarray(y_vals, float)
        return np.round((y_vals - ys_g[0]) / (ys_g[1] - ys_g[0])).astype(int)

    def iz_of(z_val):
        iz = int(np.round((z_val - zs_g[0]) / (zs_g[1] - zs_g[0])))
        return int(np.clip(iz, 0, len(zs_g) - 1))

    def seg_feasible(h, theta, dirn, a, b, zmode="stand"):
        """水平线段 y_r=u-u_robot (u∈[a,b] 步0.01) 的每个 (y_r,z_w) 都落在对应网格内?"""
        us = np.arange(a, b + 1e-9, 0.01)
        g = grid_for(theta, dirn, zmode)
        iy = iy_arr(us - u_robot)
        if (iy < 0).any() or (iy >= len(ys_g)).any():
            return False                       # 网格外直接不可行(不夹边缘格)
        return bool(g[iy, iz_of(zw_of(h, theta, dirn))].all())

    U_C = 0.5 * (STROKE_U[0] + STROKE_U[1])             # 统一区间中心(默认(0.005,0.715)->0.36)

    def solve_band(rows_h, zmode, tag, fixed=None):
        """单姿态带等宽求解: 带内统一u区间(中心U_C, clip到STROKE_U, 两端内收0.02m),
        z_w 随行高严格递增(数值保护间隔0.005m), 行角θ取可行最大档(回溯).
        V2: 蹲姿带以站姿带区间为准(fixed) —— 两带带宽不一致墙面覆盖对不齐(用户定稿),
        站姿带是窄的一方, 对齐只收窄蹲姿带. 返回 (W, u_lo, u_hi, assign)."""
        W = 0.70
        while True:
            if fixed is not None:
                ulo, uhi = fixed[0], fixed[1]
            else:
                raw_lo = max(STROKE_U[0], U_C - W / 2)  # 区间clip到[0.005,0.715]
                raw_hi = min(STROKE_U[1], U_C + W / 2)
                ulo, uhi = raw_lo + 0.04, raw_hi - 0.04  # 两端各内收0.04m(腰yaw限位后极缘点需更多余量)
            feas = []                                   # 每行可行(θ,dir)列表(ANG_OPTS已θ降序)
            ok = True
            for h in rows_h:
                opts = [(t, d) for t, d in ANG_OPTS if seg_feasible(h, t, d, ulo, uhi, zmode)]
                if not opts:
                    ok = False
                    break
                feas.append(opts)
            assign = None
            if ok:
                # 叠层单调(用户定稿): 带内 z_w 随行高严格递增, θ尽量大 -> 回溯分配
                n_r = len(feas)
                assign = [None] * n_r

                def bt(i, z_prev):
                    if i == n_r:
                        return True
                    for theta, dirn in feas[i]:
                        z = zw_of(rows_h[i], theta, dirn)
                        # 竖直列模式不要求 z_w 递增: 那是水平行叠层防碰的约束; 列内
                        # 沿高度连续扫过整段, 顶段上仰时腕心自然回落(wrist上限), 无害
                        if i == 0 or STROKE_DIR == "vertical" \
                                or z > z_prev + 0.005 - 1e-9:
                            assign[i] = (theta, dirn)
                            if bt(i + 1, z):
                                return True
                            assign[i] = None
                    return False

                if bt(0, -9.9):
                    break
                assign = None
            if fixed is not None:
                for h in rows_h:
                    opts = [(t, d, round(zw_of(h, t, d), 3)) for t, d in ANG_OPTS
                            if seg_feasible(h, t, d, max(STROKE_U[0], U_C - 0.1),
                                            min(STROKE_U[1], U_C + 0.1), zmode)]
                    print(f"[诊断] h={h:.2f}({zmode}) 可行档(W~0.2): {opts}")
                raise RuntimeError(f"等宽求解失败({tag}): 固定区间[{fixed[0]:.3f},{fixed[1]:.3f}]"
                                   f"下存在无可行(θ,dir)或z_w不递增的行!")
            W -= 0.01
            if W < 0.05:
                for h in rows_h:
                    opts = [(t, d, round(zw_of(h, t, d), 3)) for t, d in ANG_OPTS
                            if seg_feasible(h, t, d, max(STROKE_U[0], U_C - 0.1),
                                            min(STROKE_U[1], U_C + 0.1), zmode)]
                    print(f"[诊断] h={h:.2f}({zmode}) 可行档(W~0.2): {opts}")
                raise RuntimeError(f"等宽联合求解失败({tag}): "
                                   f"无W使每行存在可行(θ,dir)且z_w严格递增!")
        return round(uhi - ulo + 0.04, 2), round(ulo, 3), round(uhi, 3), list(assign)

    if STROKE_DIR == "vertical":
        # V3 竖直列: 阶梯加密 0.05m; 站姿底从 0.70 起探(低端不可行逐档抬高),
        # 蹲姿顶=站姿底+2cm(两带重叠防水平漏带: 列内连续, 带间是水平缝必须重叠)
        PV = 0.05
        _a = max(STROKE_U[0], U_C - 0.1); _b = min(STROKE_U[1], U_C + 0.1)
        def _probe(h, zm):
            return any(seg_feasible(h, t, d, _a, _b, zm) for t, d in ANG_OPTS)
        _vs = [round(float(h), 2) for h in np.arange(0.70, H_TOP + 1e-9, PV)]
        while _vs and not _probe(_vs[0], "stand"):
            _vs.pop(0)
        hs_seq = _vs
        zone_rows = ["stand"] * len(hs_seq)
        if DUAL:
            _ct = round(hs_seq[0] + 0.06, 2)
            _vc = [round(float(h), 2) for h in np.arange(0.25, _ct + 1e-9, PV)]
            while _vc and not _probe(_vc[-1], "crouch"):
                _vc.pop()
            crouch_cands = _vc
        print(f"[V3竖直列] 阶梯pitch {PV}m: 站姿 {hs_seq[0]:.2f}~{hs_seq[-1]:.2f}"
              + (f" / 蹲姿 {crouch_cands[0]:.2f}~{crouch_cands[-1]:.2f}"
                 f"(带重叠 {crouch_cands[-1] - hs_seq[0]:.2f}m 防漏带)" if DUAL else ""))
    W, u_lo, u_hi, assign_s = solve_band(hs_seq, "stand", "站姿带")
    row_params = []
    for h, (theta, dirn) in zip(hs_seq, assign_s):
        row_params.append(dict(h=float(h), z_w=float(zw_of(h, theta, dirn)),
                               angle_deg=int(theta), dir=int(dirn), zone="stand",
                               u_lo=u_lo, u_hi=u_hi))
    u_lo_c = u_hi_c = None
    if DUAL:
        # V2(用户定稿): 蹲姿带行候选按站姿区间探测, 直接取站姿区间对齐求解
        # (带宽不一致墙面覆盖对不齐; 蹲姿网格更宽, 对齐只收窄蹲姿带, 站姿带不变)
        hs_crouch = [h for h in crouch_cands
                     if any(seg_feasible(h, t, d, u_lo, u_hi, "crouch") for t, d in ANG_OPTS)]
        if not hs_crouch:
            raise RuntimeError(f"V2双姿态: 蹲姿网格在站姿区间[{u_lo:.3f},{u_hi:.3f}]下无可行行, "
                               "请检查 02 蹲姿工作区/站位配置")
        Wc, u_lo_c, u_hi_c, assign_c = solve_band(hs_crouch, "crouch", "蹲姿带", fixed=(u_lo, u_hi))
        for h, (theta, dirn) in zip(hs_crouch, assign_c):
            row_params.append(dict(h=float(h), z_w=float(zw_of(h, theta, dirn)),
                                   angle_deg=int(theta), dir=int(dirn), zone="crouch",
                                   u_lo=u_lo_c, u_hi=u_hi_c))
        # ---- 底部延伸带(2026-10-12 用户: "骨盆都到0.35了怎么还喷不到下面") ----
        # 直接 IK 探测表明 60°下倾 + 骨盆0.30 时落点可到 ~0.10m; 旧蹲姿网格的
        # 下倾腕心下限(Z_BOT_C~0.58)把行地板保守卡在0.35. 延伸行绕开网格、
        # 逐行 IK 探宽(底部两侧不可达 -> 延伸行比主带窄, 主带结构不动).
        _EXT_FLOOR = float(os.environ.get("G1W_EXT_FLOOR", "0.35"))  # 默认关(=蹲姿地板): 弯腰延伸行有分支翻转, 用户定稿先不喷最下面
        if _EXT_FLOOR < min(hs_crouch) - 1e-9:
            _pz_ext = 0.30                       # 延伸行骨盆(策略 height_cmd 下限带内)
            from scipy.spatial import cKDTree as _cKDTree
            _ws_npz = np.load(os.environ.get("G1W_WS", os.path.join(IN, "g1_workspace.npz")),
                              allow_pickle=True)
            _qsub_ext = np.vstack([_ws_npz[f"right_{_m}_y60_qsub"].astype(float)
                                   for _m in ("stand", "crouch")])
            _tree_ext = _cKDTree(np.vstack([
                _ws_npz[f"right_{_m}_y60_endsub"].astype(float)[:, :3]
                for _m in ("stand", "crouch")]))
            # 延伸行骨盆 0.30 < G1W_CROUCH_LO: 放宽蹲姿带下界(点级 pb 用)
            PELVIS_RANGE["crouch"] = (min(0.30, PELVIS_RANGE["crouch"][0]),
                                      PELVIS_RANGE["crouch"][1])
            _ext_rows = []
            for _h in np.arange(_EXT_FLOOR, min(hs_crouch) - 1e-9, PITCH):
                _h = round(float(_h), 2)
                _zw = _h + D_WRIST * np.tan(np.deg2rad(TILT_MAX))   # 60°下倾
                _got = None
                for _W in (0.40, 0.35, 0.30, 0.25, 0.20, 0.15):     # 先宽后窄
                    if _got is not None:
                        break
                    _a = max(STROKE_U[0], U_C - _W / 2) + 0.02   # 与主带同款两端内收
                    _b = min(STROKE_U[1], U_C + _W / 2) - 0.02
                    if _b - _a < _W - 0.05:
                        continue
                    _us = np.arange(_a, _b + 1e-9, 0.02)
                    _ps = np.column_stack([np.full(len(_us), D_ROBOT - D_WRIST),
                                           _us - u_robot, np.full(len(_us), _zw)])
                    _d = np.tile(np.array([np.cos(np.deg2rad(-TILT_MAX)), 0.0,
                                           np.sin(np.deg2rad(-TILT_MAX))]),
                                 (len(_us), 1))
                    _j = _tree_ext.query(_ps)[1]
                    _sq = np.vstack([_qsub_ext[_k][:10] for _k in _j])
                    for _pz in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55,
                                0.60):                        # 逐行试骨盆档(弯腰档:骨盆可高)
                        _pb = np.column_stack([np.full(len(_us), _pz - 0.012),
                                               np.full(len(_us), _pz + 0.012)])
                        _, _, _, _, _cv = batch_ik(_sq.copy(), np.full(len(_us), _pz),
                                                   _ps, _d, "stand", iters=600, pb=_pb)
                        if _cv.all():
                            _got = dict(h=_h, z_w=float(_zw), angle_deg=60, dir=1,
                                        zone="crouch", u_lo=round(_a, 3),
                                        u_hi=round(_b, 3), pz=_pz)
                            break
                if _got is not None:
                    _got["pz_band"] = (_got["pz"] - 0.05, _got["pz"] + 0.05)
                    _ext_rows.append(_got)
            if _ext_rows:
                print(f"[底部延伸] {len(_ext_rows)} 行 " +
                      " ".join(f"{r['h']:.2f}(宽{r['u_hi']-r['u_lo']:.2f})"
                               for r in _ext_rows))
                # 插到蹲姿行之前(z_w 升序执行: 深蹲先喷最底, 沿伸行上行接主带)
                _n_st = sum(1 for r in row_params if r["zone"] == "stand")
                row_params[_n_st:_n_st] = _ext_rows
                hs_crouch = [r["h"] for r in _ext_rows] + hs_crouch
        hs_seq = hs_seq + hs_crouch
        zone_rows = zone_rows + ["crouch"] * len(hs_crouch)
        H_MIN_SPRAY_MIN = min(H_MIN_SPRAY, H_MIN_SPRAY_C)
        print(f"[联合求解·蹲姿带] 与站姿带同区间[{u_lo_c:.3f},{u_hi_c:.3f}](两带对齐, 用户定稿); "
              f"蹲姿行 " + " ".join(f"{h:.2f}" for h in hs_crouch))
    if STROKE_DIR != "vertical":
        assert all(b["z_w"] - a["z_w"] >= 0.005 - 1e-6
                   for a, b in zip(row_params, row_params[1:])
                   if b["zone"] == a["zone"]), \
            "同姿态带内 z_w 未严格递增!"
    print(f"[联合求解·站姿带] 统一宽度 W={W:.2f}m (中心u_c={U_C}, clip{STROKE_U}, 两端内收0.02m) "
          f"-> 统一区间 [{u_lo:.3f},{u_hi:.3f}] (宽 {u_hi - u_lo:.3f}m; z_w叠层严格递增)")
    for rp in row_params:
        tag = "垂直" if rp["angle_deg"] == 90 else ("枪下倾" if rp["dir"] > 0 else "枪上倾")
        print(f"[行参数] h={rp['h']:.2f}m({rp['zone'][0]}) z_w={rp['z_w']:.3f}m "
              f"θ={rp['angle_deg']:2d}° dir={rp['dir']:+d} ({tag}) u[{rp['u_lo']:.3f},{rp['u_hi']:.3f}]")

    # ---------------- 障碍物(红色穿越区) ----------------
    # 点云模式: cfg 提供障碍(window 开口 + 凸出物箱) -> 覆盖舱内硬编码几何
    cloud_obst = None
    if _cfgp and os.path.exists(_cfgp):
        cloud_obst = json.load(open(_cfgp, encoding="utf8")).get("obstacles")
    sec = np.load(os.path.join(IN, "wall_section.npz"))
    pts, lab = sec["points"].astype(float), sec["label"]
    LBL = meta["labels"]
    obstacles = []
    if cloud_obst is not None:
        win_c = next((o for o in cloud_obst if o["kind"].startswith("window")), None)
        win_box = ([win_c["u"][0] - 0.03, win_c["u"][1] + 0.03],
                   [win_c["h"][0] - 0.03, win_c["h"][1] + 0.03]) if win_c \
            else ([-1, -1], [-1, -1])                     # 无开口检测 -> 不设窗区
        obstacles.append(dict(name="window", u=win_box[0], h=win_box[1]))
        NU, NH = 100, 200
        pipe_occ = np.zeros((NH, NU), bool)
        for o in cloud_obst:
            if o["depth"] <= 0.05 and not o["kind"].startswith("window"):
                continue            # 窗口开口(depth=0)也标穿越; 浅凸出忽略
            i0 = np.clip((o["u"][0] - 0.03) / 0.01, 0, NU - 1).astype(int)
            i1 = np.clip((o["u"][1] + 0.03) / 0.01, 0, NU - 1).astype(int)
            h0 = np.clip((o["h"][0] - 0.03) / 0.01, 0, NH - 1).astype(int)
            h1 = np.clip((o["h"][1] + 0.03) / 0.01, 0, NH - 1).astype(int)
            pipe_occ[h0:h1 + 1, i0:i1 + 1] = True
        print(f"[障碍区] 点云模式: {len(cloud_obst)} 项 -> "
              f"穿越区 {int(pipe_occ.sum())} 格 (~{pipe_occ.sum()*1e-4:.3f} m²)")
    else:
        wb = meta["window"]
        win_box = [wb["u"][0] - 0.03, wb["u"][1] + 0.03], [wb["h"][0] - 0.03, wb["h"][1] + 0.03]
        obstacles.append(dict(name="window", u=win_box[0], h=win_box[1]))
        # 管道/凸出物(other, 离墙 d>2cm): 占据栅格 1cm + 膨胀3cm -> 点级红色判定
        om = (lab == LBL["other"]) & (pts[:, 2] > 0.04) & (pts[:, 2] < 0.30)
        P = pts[om][:, :2]
        pipe_occ = None
        if len(P) > 50:
            NU, NH = 100, 200          # u 0~1.0m, h 0~2.0m, 1cm
            gi = np.clip((P[:, 0] / 0.01).astype(int), 0, NU - 1)
            gh = np.clip((P[:, 1] / 0.01).astype(int), 0, NH - 1)
            occ = np.zeros((NH, NU), bool)
            occ[gh, gi] = True
            occ = ndimage.binary_opening(occ, structure=np.ones((3, 3), bool))  # 去孤立噪点
            pipe_occ = ndimage.binary_dilation(occ, structure=np.ones((7, 7), bool))  # ~3cm
            n_cells = int(pipe_occ.sum())
            print(f"[障碍区] window u{win_box[0]} h{win_box[1]}")
            print(f"[障碍区] 管道占据栅格膨胀后 {n_cells} 格 (~{n_cells*1e-4:.3f} m²)")

    def in_obstacle(u, h):
        if win_box[0][0] <= u <= win_box[0][1] and win_box[1][0] <= h <= win_box[1][1]:
            return True
        if pipe_occ is not None:
            iu = int(round(u / 0.01)); ih = int(round(h / 0.01))
            if 0 <= ih < pipe_occ.shape[0] and 0 <= iu < pipe_occ.shape[1]:
                return bool(pipe_occ[ih, iu])
        return False

    # ---- 点云模式: 背景散点换成用户点云(u,h), 画图障碍框用点云检出 ----
    CLOUD_PLOT = cloud_obst is not None
    plot_boxes = cloud_obst if CLOUD_PLOT else obstacles
    if CLOUD_PLOT:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "cloud_scan", os.path.join(BASE, "scripts", "11_cloud_scan.py"))
        _cs = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_cs)
        _cp = _cs.load_cloud(json.load(open(_cfgp, encoding="utf8"))["cloud"])
        if len(_cp) > 80000:
            _cp = _cp[np.random.default_rng(0).choice(len(_cp), 80000, replace=False)]
        pts = np.column_stack([_cp[:, 1], _cp[:, 2]])      # (u,h)
        lab = np.zeros(len(pts), int)

    # ---------------- E: 腕心干净标准之字形(行内水平线 + 行端竖直U型连接) ----------------
    # 行内: 墙点u扫描, 腕心 y_r=u-u_robot 同步扫描, z_w 行常数 -> 严格水平直线;
    # 连接段: u=行端常数(墙点竖直移动), y_r 停行端, z_w(i)->z_w(i+1) 线性,
    #         指向俯仰角在两行之间线性插值(仍纯俯仰, |φ|≤30° 由构造保证)。
    # 无任何逐点网格投影/边界跟踪。

    def n_samples(a, b):
        return max(2, int(np.ceil(abs(b - a) / POINT_SPACING)) + 1)

    rows = []      # (u, h, kind, cls, y_r, z_r, phi, zone)  phi=带符号俯仰角(rad), 纯俯仰指向
    u_prev_end = None                                 # 上一行末端u(V2接缝过渡的u线性起点)
    if STROKE_DIR == "vertical":
        # V3 竖直列(用户定稿 2026-10-12): 喷涂线=竖直线, 平行于"机器人-墙面法向"垂面;
        # 枪臂恒在该垂面内(纯俯仰φ, 横向分量=0), 60/70/80/90 档沿高度阶梯取角并插值;
        # 横向只在列间跳转(kind=hop, 非喷涂黄段); 列距<=0.15m(喷幅横向覆盖)。
        _lad = {(rp["zone"], round(rp["h"], 2)): rp for rp in row_params}

        def lad_at(zn_, h_):
            """阶梯插值: 返回该高度处的 (phi, z_w) —— 相邻档线性, 横向分量恒0."""
            _hs = sorted(k[1] for k in _lad if k[0] == zn_)
            h_ = float(np.clip(h_, _hs[0], _hs[-1]))
            i = int(np.clip(np.searchsorted(_hs, h_), 1, len(_hs) - 1))
            h0, h1 = _hs[i - 1], _hs[i]
            t = 0.0 if h1 == h0 else (h_ - h0) / (h1 - h0)
            a0, a1 = _lad[(zn_, h0)], _lad[(zn_, h1)]
            f0 = -a0["dir"] * np.deg2rad(90.0 - a0["angle_deg"])
            f1 = -a1["dir"] * np.deg2rad(90.0 - a1["angle_deg"])
            return f0 + (f1 - f0) * t, a0["z_w"] + (a1["z_w"] - a0["z_w"]) * t

        _full_lo, _full_hi = u_lo - 0.02, u_hi + 0.02
        _K = max(2, int(np.ceil((_full_hi - _full_lo) / COL_PITCH_MAX)))
        _pc = (_full_hi - _full_lo) / _K
        cols_u = [round(_full_lo + _pc * (k + 0.5), 3) for k in range(_K)]
        zones_seq = ["stand"] + (["crouch"] if DUAL else [])
        for _zi, _zn in enumerate(zones_seq):
            _band = [rp for rp in row_params if rp["zone"] == _zn]
            _h_lo = min(rp["h"] for rp in _band)
            _h_hi = max(rp["h"] for rp in _band)
            for _k, _cu in enumerate(cols_u):
                _up = (_k % 2 == 0)
                _ha, _hb = (_h_lo, _h_hi) if _up else (_h_hi, _h_lo)
                if rows:                            # 列间跳转(非喷涂): 上一列末端 -> 本列首端
                    # hop 高度=本列起点高度(_ha): 与上一列末端同高(zigzag 交替保证),
                    # 修 bug: 旧代码误用终点高度 _hb, hop 与列首之间凭空多出整段竖直瞬移
                    _pu, _ph = rows[-1][0], rows[-1][1]
                    _ph_t = _ha
                    for _uq in np.linspace(_pu, _cu, n_samples(_pu, _cu))[1:-1]:
                        _fq, _zw = lad_at(_zn, _ph_t)
                        rows.append((float(_uq), float(_ph_t), "hop", "yellow",
                                     float(_uq) - u_robot, _zw, _fq, _zn))
                for _hq in np.linspace(_ha, _hb, n_samples(_ha, _hb)):
                    _hq = float(_hq)
                    _fq, _zw = lad_at(_zn, _hq)
                    rows.append((_cu, _hq, "stroke",
                                 "red" if in_obstacle(_cu, _hq) else "green",
                                 _cu - u_robot, _zw, _fq, _zn))
            if _zi == 0 and DUAL and rows:
                # 姿态翻转过渡(kind=transition): 05 在 zone 翻转处整段替换为回零位过渡
                _pu, _ph = rows[-1][0], rows[-1][1]
                _zn2 = zones_seq[1]
                _b2 = [rp for rp in row_params if rp["zone"] == _zn2]
                _h2 = min(rp["h"] for rp in _b2)    # 蹲姿第一列(k=0)自底起
                for _t, (_uq, _hq) in enumerate(zip(
                        np.linspace(_pu, cols_u[0], 12), np.linspace(_ph, _h2, 12))):
                    _zq = "stand" if 2 * (_t + 1) <= 12 else "crouch"
                    # 站/蹲阶梯值连续混合(w=smoothstep): 避免硬切换造成 zw/φ 阶跃
                    _w = (_t + 1) / 12.0
                    _w = _w * _w * (3.0 - 2.0 * _w)
                    _fs, _zs = lad_at("stand", float(_hq))
                    _fc, _zc = lad_at("crouch", float(_hq))
                    _fq = _fs + (_fc - _fs) * _w
                    _zw = _zs + (_zc - _zs) * _w
                    rows.append((float(_uq), float(_hq), "transition", "yellow",
                                 float(_uq) - u_robot, _zw, _fq, _zq))
        print(f"[V3竖直列] {_K} 列 u={['%.3f' % c for c in cols_u]} 列距{_pc:.3f}m; "
              f"带序 {'->'.join(zones_seq)}; 列内竖直走线+列间hop(非喷涂)"
              + ("; 带间回零位过渡(transition)" if DUAL else ""))
    else:
        for r, rp in enumerate(row_params):
            h, zw, theta, dirn = rp["h"], rp["z_w"], rp["angle_deg"], rp["dir"]
            phi = -dirn * np.deg2rad(90.0 - theta)          # 行常数: 指向与墙面法向的带符号俯仰角
            u_a, u_b = (rp["u_lo"], rp["u_hi"]) if r % 2 == 0 else (rp["u_hi"], rp["u_lo"])
            us = np.linspace(u_a, u_b, n_samples(u_a, u_b))
            if r > 0:                                       # 行端竖直U型连接段
                rp0 = row_params[r - 1]
                phi0 = -rp0["dir"] * np.deg2rad(90.0 - rp0["angle_deg"])
                uc = us[0]                                  # = 上一行u_b: 行端常数
                y_rc = uc - u_robot
                hh = np.linspace(rp0["h"], h, n_samples(rp0["h"], h))[1:]
                seam = rp["zone"] != rp0["zone"]            # V2 姿态接缝(stand->crouch)
                for k, hq in enumerate(hh):
                    t = (k + 1) / len(hh)
                    # V2 接缝段 kind="transition": 05 在 zone 翻转处整段替换为"收臂->零位停顿
                    # ->伸出+下蹲"回零位过渡(用户定稿); 两带u区间不同 -> u也线性过渡;
                    # zone 上半 stand/下半 crouch 供翻转检测, IK 骨盆界限取 trans(0.60~0.79)
                    zq = "stand" if 2 * (k + 1) <= len(hh) else "crouch"
                    uq = (float(u_prev_end + (uc - u_prev_end) * t) if seam else float(uc))
                    rows.append((uq, float(hq), "transition" if seam else "connector",
                                 "yellow",                 # 连接/过渡段强制标黄: 不做障碍红判定,
                                 uq - u_robot,             # 下游按非喷涂段处理(角度规范豁免)
                                 rp0["z_w"] + (zw - rp0["z_w"]) * t,     # z_w 线性
                                 phi0 + (phi - phi0) * t,               # 俯仰角线性插值
                                 (zq if seam else rp["zone"])))
            for u in us:                                    # 行内水平走线点
                rows.append((float(u), h, "stroke",
                             "red" if in_obstacle(u, h) else "green",
                             u - u_robot, zw, phi, rp["zone"]))
            u_prev_end = float(us[-1])
    print(f"[之字形] {len(hs_seq)} 行, 站姿带u区间[{u_lo:.3f},{u_hi:.3f}]"
          + (f", 蹲姿带u区间[{u_lo_c:.3f},{u_hi_c:.3f}](两带各自等宽, 共用中心)"
             if DUAL else "(统一u区间)")
          + ", 行方向交替, 行端竖直U型连接, 腕心行内水平(z_w行常数)"
          + (f"; V2执行序: 站姿带{row_params[0]['h']:.2f}~"
             f"{max(rp['h'] for rp in row_params if rp['zone'] == 'stand'):.2f}m -> "
             f"回零位过渡 -> 蹲姿带"
             f"{min(rp['h'] for rp in row_params if rp['zone'] == 'crouch'):.2f}~"
             f"{max(rp['h'] for rp in row_params if rp['zone'] == 'crouch'):.2f}m" if DUAL else ""))

    u_w = np.array([r_[0] for r_ in rows])
    h_w = np.array([r_[1] for r_ in rows])
    kind_w = np.array([r_[2] for r_ in rows])
    cls_w = np.array([r_[3] for r_ in rows])
    y_r = np.array([r_[4] for r_ in rows])
    z_r = np.array([r_[5] for r_ in rows])
    phi_w = np.array([r_[6] for r_ in rows])
    zone_w = np.array([r_[7] for r_ in rows])
    # V2 每点骨盆界限: 过渡段 trans(自由穿越), 其余按 zone; stand 模式 None(走 mode 标量, legacy)
    _EXT_PZ = {round(r["h"], 2): r["pz_band"] for r in row_params if "pz_band" in r}
    pt_pb = (np.array([
        PELVIS_RANGE["trans"] if k == "transition" else
        (_EXT_PZ.get(round(float(h), 2), PELVIS_RANGE[z])
         if (z == "crouch" and "pz_band" in {} or True) else PELVIS_RANGE[z])
        for k, z, h in zip(kind_w, zone_w, h_w)], float) if DUAL else None)

    # ---------------- 机器人轨迹映射: 腕心位置 + 纯俯仰指向 ----------------
    # x: 腕平面 x_w = D_ROBOT - D_WRIST; 指向 d = (cosφ, 0, sinφ): 左右分量恒=0
    wall_xyz = np.stack([np.full(len(u_w), X_FACE), U0 + u_w, FLOOR_Z + h_w], 1)
    wri_xyz = np.stack([np.full(len(u_w), X_FACE - D_WRIST),
                        U0 + u_robot + y_r, FLOOR_Z + z_r], 1)
    d_star = np.stack([np.cos(phi_w), np.zeros_like(phi_w), np.sin(phi_w)], 1)
    tilt_deg = np.degrees(np.abs(phi_w))           # 指令倾角(与墙面法向夹角)
    assert tilt_deg.max() <= TILT_MAX + 1e-9, "指向倾角超出30°(离散行角/插值构造保证≤30°)"
    perp_ok = tilt_deg <= 0.5                      # 实际近垂直(90°)的点
    n_perp_cmd = int(((phi_w == 0.0) & (kind_w == "stroke")).sum())
    ang_rows = {}
    for rp in row_params:
        k = f"{rp['angle_deg']}°"
        ang_rows[k] = ang_rows.get(k, 0) + 1
    print(f"[映射] 走线点 {int((kind_w == 'stroke').sum())} (垂直指向行 "
          f"{sum(1 for q in row_params if q['angle_deg'] == 90)} 行, 共{n_perp_cmd}点), "
          f"连接段 {int((kind_w == 'connector').sum())}点; 行角分布 {ang_rows}")
    print(f"[映射] 指令倾角 max {tilt_deg.max():.1f}° (行角离散+插值, 构造保证≤{TILT_MAX:.0f}°), "
          f"指向横向分量恒=0")
    print(f"[机器人轨迹z] 最低 {z_r.min():.3f}m (安全要求>=0.15), 最高 {z_r.max():.3f}m")

    # ---------------- 快速IK: 锚点求解 + 样条插值 + 增量修正 (智能加速) ----------------
    # 逐点多种子IK+逐点整流(旧版~2min)替换为: ~90个锚点顺序热启动真IK(自然连续)
    # -> 全序列三次样条插值(连续性由构造保证) -> 全点FK校验驱动的增量修正(小步DLS).
    import time as _time
    _t0 = _time.time()
    p_star_rel = np.column_stack([np.full(len(u_w), D_ROBOT - D_WRIST), y_r, z_r])
    N = len(u_w)

    # 1) 锚点: 每条走线行均匀6个(含两端) + 每个连接段首尾
    anchor_set = set()
    for rp in row_params:
        mrow = np.where((kind_w == "stroke") & (np.abs(h_w - rp["h"]) < 1e-9))[0]
        if len(mrow):
            for k in np.linspace(0, len(mrow) - 1, 4).astype(int):
                anchor_set.add(int(mrow[k]))
    kd = np.where(kind_w == "connector")[0]
    for seg in (np.split(kd, np.where(np.diff(kd) > 1)[0] + 1) if len(kd) else []):
        anchor_set.add(int(seg[0])); anchor_set.add(int(seg[-1]))
    anchor_idx = np.array(sorted(anchor_set), int)
    n_anchor = len(anchor_idx)

    # 2) 锚点顺序求解(热启动链; 收敛候选中取与上一锚点差最小者 => 锚点链连续)
    q_ik = np.zeros((N, 10)); pz_ik = np.zeros(N)
    rng_a = np.random.default_rng(11)
    _, j_mc = tree_all.query(p_star_rel[anchor_idx], k=2)
    prev_q, prev_pz, n_anchor_ok = None, None, 0
    for r, i in enumerate(anchor_idx):
        # 种子分梯队批量并行: 一梯队(NN+热种子, 小iters快收敛) -> 二梯队(MC+噪声, 大iters)
        # NN-IK 条件输入(V2 通用大模型): 骨盆=该点姿态带中点, 腰限=当前档位
        q_nn = nn_ik(p_star_rel[i], d_star[i],
                     pelvis=(float(pt_pb[i].mean()) if DUAL else 0.78),
                     rp=(_WRP if _WRP > 0 else 0.0872665))
        t1, t2 = [], []
        if q_nn is not None:
            t1.append((np.asarray(q_nn, float), 0.78))
        if prev_q is not None:
            t1.append((prev_q.copy(), prev_pz))
        t2.append((qsub_all[j_mc[r, 0]][:10].astype(float).copy(),
                   float(qsub_all[j_mc[r, 0]][10])))
        if prev_q is not None:
            t2.append((prev_q + rng_a.uniform(-0.15, 0.15, 10), prev_pz))
        best = None
        for tier, it in ((t1, 180), (t2, 700)):
            if not tier:
                continue
            qs = np.array([sq for sq, _ in tier])
            pzs = np.array([sp for _, sp in tier])
            qf, pf, e1, e2, cv = batch_ik(qs.copy(), pzs,
                                          np.repeat(p_star_rel[i:i + 1], len(tier), 0),
                                          np.repeat(d_star[i:i + 1], len(tier), 0),
                                          "stand", iters=it,
                                          pb=(np.repeat(pt_pb[i][None], len(tier), 0)
                                              if DUAL else None))
            for k in np.where(cv)[0]:
                jump = float(np.abs(qf[k] - prev_q).max()) if prev_q is not None else 0.0
                if best is None or jump < best[0]:
                    best = (jump, qf[k].copy(), float(pf[k]))
            if best is not None and best[0] < 0.35:
                break
        if best is not None:
            prev_q, prev_pz = best[1], best[2]
            q_ik[i], pz_ik[i] = best[1], best[2]
            n_anchor_ok += 1
    print(f"[IK·锚点] {n_anchor_ok}/{n_anchor} 锚点收敛 ({_time.time() - _t0:.1f}s)", flush=True)

    # 3) 全序列三次样条插值(锚点含行端与连接段端点 => 跨行/连接段自然过渡)
    from scipy.interpolate import CubicSpline
    ok_a = anchor_idx[np.abs(q_ik[anchor_idx]).sum(1) > 0]
    if len(ok_a) < 2:
        raise RuntimeError(f"锚点IK成功数不足: {len(ok_a)}/{n_anchor} "
                           f"(失败锚点无法构成样条, 请检查工作区网格/站位配置)")
    _i0a, _i1a = int(ok_a[0]), int(ok_a[-1])
    _xa = np.arange(N)
    if _i0a > 0:        # 首锚点失败: 前段用线性插值+端点夹紧(样条外推会振荡)
        for k in range(10):
            q_ik[:_i0a, k] = np.interp(_xa[:_i0a], ok_a, q_ik[ok_a, k])
    if _i1a < N - 1:    # 末锚点失败: 尾段同样线性+夹紧
        for k in range(10):
            q_ik[_i1a + 1:, k] = np.interp(_xa[_i1a + 1:], ok_a, q_ik[ok_a, k])
    cs_q = CubicSpline(ok_a, q_ik[ok_a], axis=0)
    q_ik[_i0a:_i1a + 1] = cs_q(_xa[_i0a:_i1a + 1])    # 中段(锚点覆盖区)用样条
    pz_ik[:] = np.interp(_xa, ok_a, pz_ik[ok_a])
    qd_pre = np.abs(np.diff(q_ik, axis=0)).max(1)          # 修正前(样条)相邻差

    # 4) 全点FK校验 + 增量修正(小步DLS): 位置>4mm / 轴差>0.03 / 走线点角度下限保护
    ang_cmd = 90.0 - np.degrees(np.abs(phi_w))              # 每点指令角
    stroke_m = kind_w == "stroke"
    n_corr_tot = 0
    for rd in range(3):
        p_c, ax_c = fk_wrist(q_ik, pz_ik)
        pe = np.linalg.norm(p_star_rel - p_c, axis=1)
        ae = np.linalg.norm(d_star - ax_c, axis=1)
        ang_c = 90.0 - np.degrees(np.arccos(np.clip(ax_c[:, 0], -1, 1)))
        need = (pe > 0.004) | (ae > 0.03) | (stroke_m & (ang_c < np.maximum(60.3, ang_cmd - 0.5)))
        idx = np.where(need)[0]
        if not len(idx):
            break
        qf, pf, e1, e2, cv = batch_ik(q_ik[idx], pz_ik[idx], p_star_rel[idx], d_star[idx],
                                      "stand", iters=150, pos_tol=0.004, ax_tol=0.03,
                                      pb=(pt_pb[idx] if DUAL else None))
        q_ik[idx], pz_ik[idx] = qf, pf
        n_corr_tot += len(idx)
        print(f"[IK·修正] 第{rd + 1}轮 增量修正 {len(idx)} 点 ({_time.time() - _t0:.1f}s)", flush=True)
    # 兜底: 3轮后仍未达标点全力重解; 种子=当前解邻域阶梯(保分支)优先, MC末位;
    # 逐点连续性守卫(与相邻解差<0.6rad才接受)
    p_c, ax_c = fk_wrist(q_ik, pz_ik)
    pe = np.linalg.norm(p_star_rel - p_c, axis=1)
    ae = np.linalg.norm(d_star - ax_c, axis=1)
    ang_c = 90.0 - np.degrees(np.arccos(np.clip(ax_c[:, 0], -1, 1)))
    bad = np.where((pe > 0.008) | (ae > 0.05) | (stroke_m & (ang_c < 60.0)))[0]
    rng_f = np.random.default_rng(23)
    if len(bad):
        d2, j2 = tree_all.query(p_star_rel[bad], k=2)
        n_resc = 0
        for si in range(5):
            p_now, ax_now = fk_wrist(q_ik, pz_ik)
            ang_now = 90.0 - np.degrees(np.arccos(np.clip(ax_now[bad, 0], -1, 1)))
            still = bad[(np.linalg.norm(p_star_rel[bad] - p_now[bad], axis=1) > 0.008)
                        | (stroke_m[bad] & (ang_now < 60.0))]
            if not len(still):
                break
            if si < 3:      # 热种子+噪声阶梯(同分支微扰)
                sig = 0.05 * (si + 1)
                q0 = np.clip(q_ik[still] + rng_f.normal(0, sig, (len(still), 10)), JLO, JHI)
                pz0 = pz_ik[still].copy()
            else:           # MC末位种子
                sel = np.isin(bad, still)
                q0 = qsub_all[j2[sel, si - 3]][:, :10].copy()
                pz0 = qsub_all[j2[sel, si - 3]][:, 10].copy()
            qf, pf, e1, e2, cv = batch_ik(q0, pz0, p_star_rel[still], d_star[still],
                                          "stand", iters=1200,
                                          pb=(pt_pb[still] if DUAL else None))
            _, ax_r = fk_wrist(qf, pf)
            ang_r = 90.0 - np.degrees(np.arccos(np.clip(ax_r[:, 0], -1, 1)))
            for r, i in enumerate(still):
                if not cv[r]:
                    continue
                if stroke_m[i] and ang_r[r] < 60.0 - 1e-9:
                    continue
                j_lo = float(np.abs(qf[r] - q_ik[i - 1]).max()) if i > 0 else 0.0
                j_hi = float(np.abs(q_ik[i + 1] - qf[r]).max()) if i + 1 < N else 0.0
                if max(j_lo, j_hi) < 0.6:
                    q_ik[i], pz_ik[i] = qf[r], float(pf[r])
                    n_resc += 1
        print(f"[IK·兜底] 顽固点 {len(bad)} 热种子重解接纳 {n_resc} "
              f"({_time.time() - _t0:.1f}s)", flush=True)

    def bias_pass():
        # 4.5) 60°行实现偏置兜底(函数化: 支延续后可再跑一轮): 实测<60.05°走线点, 目标轴朝垂直偏+0.6°小步重解;
        # 逐点连续性守卫(与当前相邻解差<0.6rad才接受), 多轮迭代逐步接纳
        n_bias = 0
        for _b in range(6):
            p_c, ax_c = fk_wrist(q_ik, pz_ik)
            ang_c = 90.0 - np.degrees(np.arccos(np.clip(ax_c[:, 0], -1, 1)))
            low60 = np.where(stroke_m & (ang_c < 60.05))[0]
            if not len(low60):
                break
            d_b = d_star[low60].copy()
            tb = np.deg2rad(1.0)
            d_b[:, 0] = np.cos(np.arccos(np.clip(d_star[low60, 0], -1, 1)) - tb)
            d_b[:, 2] = np.sign(d_star[low60, 2]) * np.sqrt(np.maximum(
                0.0, 1.0 - d_b[:, 0] ** 2))
            qf, pf, e1, e2, cv = batch_ik(q_ik[low60], pz_ik[low60], p_star_rel[low60],
                                          d_b, "stand", iters=90, pos_tol=0.004, ax_tol=0.02,
                                          pb=(pt_pb[low60] if DUAL else None))
            _, ax_b = fk_wrist(qf, pf)
            ang_b = 90.0 - np.degrees(np.arccos(np.clip(ax_b[:, 0], -1, 1)))
            acc = 0
            for r, i in enumerate(low60):
                if not (cv[r] and ang_b[r] >= 60.0 - 1e-9):
                    continue
                j_lo = float(np.abs(qf[r] - q_ik[i - 1]).max()) if i > 0 else 0.0
                j_hi = float(np.abs(q_ik[i + 1] - qf[r]).max()) if i + 1 < N else 0.0
                if max(j_lo, j_hi) < 0.6:
                    q_ik[i], pz_ik[i] = qf[r], float(pf[r])
                    acc += 1
            n_bias += acc
            if acc == 0:
                break
        print(f"[IK·偏置] 60°行压线点偏置重解累计接纳 {n_bias}", flush=True)
        return n_bias

    n_bias = bias_pass()

    # 5) 连续性核验(样条+小步修正 => 天然连续); 残余坏对用插值构型种子局部重解
    def pair_fix():
        n = 0
        for pp in range(N - 1):
            if np.abs(q_ik[pp + 1] - q_ik[pp]).max() <= 0.6:
                continue
            if DUAL and zone_w[pp] != zone_w[pp + 1]:
                continue            # V2 姿态接缝: 时间轴由05回零位过渡覆盖, 不修
            qm = 0.5 * (q_ik[pp] + q_ik[pp + 1]); pzm = 0.5 * (pz_ik[pp] + pz_ik[pp + 1])
            for tgt in (pp + 1, pp):
                q_pre = q_ik[tgt - 1] if tgt > 0 else qm
                qf, pf, e1, e2, cv = batch_ik(qm[None].copy(), np.array([pzm]),
                                              p_star_rel[tgt:tgt + 1], d_star[tgt:tgt + 1],
                                              "stand", iters=800,
                                              pb=(pt_pb[tgt:tgt + 1] if DUAL else None))
                if cv[0] and np.abs(qf[0] - q_pre).max() < 0.6:
                    q_ik[tgt], pz_ik[tgt] = qf[0], float(pf[0])
                    n += 1
                    break
        return n
    n_pairfix = pair_fix()

    # 支延续修复: 剩余>0.6rad跳变 => 以连续侧支为种子正向/反向重解整段,
    # 消除行内解流形支切换(±5°腰限下流形变窄时会出现), 失败则整段回退原解
    def branch_continue():
        n = 0
        for _ in range(3):
            qd = np.abs(np.diff(q_ik, axis=0)).max(1)
            bad = np.where(qd > 0.6)[0]
            if DUAL:      # V2: 姿态接缝跳变合法(骨盆/构型切换, 05回零位过渡覆盖), 排除
                bad = np.array([b for b in bad if zone_w[b] == zone_w[b + 1]], int)
            if not len(bad):
                break
            p = int(bad[0])
            snap = (q_ik.copy(), pz_ik.copy())
            done = False
            for k in range(p + 1, len(q_ik)):          # 正向: 低侧支延续到链尾
                qf, pf, e1, e2, cv = batch_ik(
                    q_ik[k - 1][None].copy(), np.array([pz_ik[k - 1]]),
                    p_star_rel[k:k + 1], d_star[k:k + 1], "stand", iters=400,
                    pb=(pt_pb[k:k + 1] if DUAL else None))
                if cv[0] and np.abs(qf[0] - q_ik[k - 1]).max() < 0.35:
                    q_ik[k], pz_ik[k] = qf[0], pf[0]; n += 1
                else:
                    break
            if np.abs(np.diff(q_ik, axis=0)).max() <= 0.6:
                done = True
            if not done:                                # 反向: 高侧支重解到链头
                q_ik[:], pz_ik[:] = snap
                for k in range(p, -1, -1):
                    qf, pf, e1, e2, cv = batch_ik(
                        q_ik[k + 1][None].copy(), np.array([pz_ik[k + 1]]),
                        p_star_rel[k:k + 1], d_star[k:k + 1], "stand", iters=400,
                        pb=(pt_pb[k:k + 1] if DUAL else None))
                    if cv[0] and np.abs(qf[0] - q_ik[k + 1]).max() < 0.35:
                        q_ik[k], pz_ik[k] = qf[0], pf[0]; n += 1
                    else:
                        break
                if np.abs(np.diff(q_ik, axis=0)).max() > 0.6:
                    q_ik[:], pz_ik[:] = snap            # 双向都失败: 回退
                    break
        return n

    n_branch = branch_continue()
    n_bias2 = bias_pass()          # 支延续可能覆盖压线偏置解 -> 再补一轮
    qd_post = np.abs(np.diff(q_ik, axis=0)).max(1)
    q_contig_max = float(qd_post.max())
    runtime_ik = _time.time() - _t0
    print(f"[IK·连续] 坏对修复 {n_pairfix} 处, 支延续 {n_branch} 点, "
          f"相邻最大关节差 {q_contig_max:.3f} rad "
          f"({int((qd_post > 0.6).sum())} 处>0.6)")

    # 收敛统计(口径: 位置<4mm且轴<0.03=严格档; <8mm=边缘档; 角度下限见终检)
    p_c, ax_c = fk_wrist(q_ik, pz_ik)
    pe = np.linalg.norm(p_star_rel - p_c, axis=1)
    ae = np.linalg.norm(d_star - ax_c, axis=1)
    conv_strict = (pe < 0.004) & (ae < 0.03)
    conv = (pe < 0.008) & (ae < 0.03)
    conv_loose = conv & ~conv_strict
    n_rect, n_interp, n_angfix = 0, n_pairfix, n_bias   # meta兼容(旧整流计数)
    revert_idx = np.array([], int)
    print(f"[IK] 锚点{n_anchor}/收敛{n_anchor_ok}, 增量修正{n_corr_tot}点, 严格档{int(conv_strict.sum())}, "
          f"边缘档{int(conv_loose.sum())}, 未收敛{int((~conv).sum())} / {N}; "
          f"IK总耗时 {runtime_ik:.1f}s (旧版逐点IK+整流约120s)")
    # ---------------- 喷涂质量终检: 对最终关节解FK实测枪轴与墙面夹角 ----------------
    _, ax_ach = fk_wrist(q_ik, pz_ik)
    ang_wall = 90.0 - np.degrees(np.arccos(np.clip(ax_ach[:, 0], -1, 1)))
    stroke_m2 = kind_w == "stroke"      # 角度规范只约束横向走线(连接段豁免, 用户定稿)
    angle_rule_ok = bool((ang_wall[stroke_m2] >= 60.0).all())
    ax_dot = np.clip(np.einsum('ij,ij->i', ax_ach, d_star), -1, 1)
    print(f"[喷涂质量终检] 关节解FK实测墙面夹角 [{ang_wall.min():.1f}, {ang_wall.max():.1f}]° "
          f"(要求>=60°, {'全部满足' if angle_rule_ok else '存在违规!'}), "
          f"与目标指向偏差 max {np.degrees(np.arccos(ax_dot.min())):.2f}°")

    # ---------------- F: 自检 ----------------
    # (c)两模式共用: 指向无横向分量(枪臂恒在"机器人-墙面"法向垂面内, 用户定稿)
    lat_cmd = float(np.abs(d_star[:, 1]).max())
    lat_fk = float(np.abs(ax_ach[:, 1]).max())
    if STROKE_DIR == "vertical":
        # V3 竖直列自检: (a)列内走线点u极差=0(严格竖直线) (b)列覆盖(高度极差与点数)
        uptp_cols, angvar_cols = [], []
        for cu in (cols_u if STROKE_DIR == "vertical" else []):
            mcol = (kind_w == "stroke") & (np.abs(u_w - cu) < 1e-6)
            if not mcol.any():
                continue
            uptp_cols.append(float(np.ptp(u_w[mcol])))
            angvar_cols.append(float(np.var(ang_wall[mcol])))
        uptp_max = max(uptp_cols) if uptp_cols else 9.9
        angvar_max_col = max(angvar_cols) if angvar_cols else 9.9
        wrist_horiz_ok = bool(uptp_max == 0.0 and lat_cmd == 0.0)
        print(f"[自检a·V3] 各列走线点u极差 max={uptp_max:.1e} m (要求=0: 严格竖直线), "
              f"共 {len(uptp_cols)} 列")
        print(f"[自检b·V3] 各列实测墙面夹角方差 max={angvar_max_col:.2e} deg² "
              f"(列内沿高度阶梯变角, 仅作参考)")
        print(f"[自检c] 指向横向分量: 指令 max={lat_cmd:.1e} (要求=0: 枪臂在法向垂面内), "
              f"FK实测 max={lat_fk:.4f} (轴向容差0.03内)")
        if not wrist_horiz_ok:
            print("[自检] 竖直列/垂面内指向自检未通过!")
        zptp_max, zstd_max, angvar_max = uptp_max, 0.0, angvar_max_col
    else:
        # 水平行自检: (a)行内腕心水平 (b)行角一致
        zstd_rows, zptp_rows, angvar_rows = [], [], []
        for rp in row_params:
            mrow = (kind_w == "stroke") & (np.abs(h_w - rp["h"]) < 1e-9)
            zptp_rows.append(float(np.ptp(z_r[mrow])))          # (a) 走线点腕心z极差(max-min)
            zstd_rows.append(float(np.std(z_r[mrow])))          #     标准差(float均值舍入~1e-16)
            angvar_rows.append(float(np.var(ang_wall[mrow])))   # (b) 走线点实测夹角方差
        zptp_max = max(zptp_rows)
        zstd_max, angvar_max = max(zstd_rows), max(angvar_rows)
        wrist_horiz_ok = bool(zptp_max == 0.0 and angvar_max < 4.0 and lat_cmd == 0.0)
        print(f"[自检a] 各行走线点腕心z极差(max-min) max={zptp_max:.1e} m (要求=0: 严格水平线), "
              f"标准差 max={zstd_max:.1e} m (np.mean浮点舍入量级)")
        print(f"[自检b] 各行实测墙面夹角方差 max={angvar_max:.2e} deg² (要求≈0: 行角一致, <4.0)")
        print(f"[自检c] 指向横向分量: 指令 max={lat_cmd:.1e} (要求=0: 纯俯仰), "
              f"FK实测 max={lat_fk:.4f} (轴向容差0.03内)")
        if not wrist_horiz_ok:
            print("[自检] 腕心水平线/行角一致性自检未通过!")

    # ---------------- 保存 ----------------
    with open(os.path.join(OUT, "wall_path.csv"), "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["seq", "zone", "kind", "class", "u_m", "h_m",
                    "world_x_m", "world_y_m", "world_z_m"])
        for i in range(len(u_w)):
            w.writerow([i, zone_w[i], kind_w[i], cls_w[i], f"{u_w[i]:.4f}", f"{h_w[i]:.4f}",
                        f"{wall_xyz[i,0]:.4f}", f"{wall_xyz[i,1]:.4f}", f"{wall_xyz[i,2]:.4f}"])
    with open(os.path.join(OUT, "robot_path.csv"), "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["seq", "zone", "kind", "class", "mode", "perp", "wall_u_m", "wall_h_m",
                    "wrist_x_m", "wrist_y_m", "wrist_z_m", "axis_dx", "axis_dy", "axis_dz",
                    "tilt_deg", "ik_ok", "pelvis_z_m"] + [f"q{i+1}" for i in range(10)])
        for i in range(len(u_w)):
            w.writerow([i, zone_w[i], kind_w[i], cls_w[i], zone_w[i], int(perp_ok[i]),
                        f"{u_w[i]:.4f}", f"{h_w[i]:.4f}",
                        f"{wri_xyz[i,0]:.4f}", f"{wri_xyz[i,1]:.4f}", f"{wri_xyz[i,2]:.4f}",
                        f"{d_star[i,0]:.4f}", f"{d_star[i,1]:.4f}", f"{d_star[i,2]:.4f}",
                        f"{tilt_deg[i]:.2f}", int(conv[i]), f"{pz_ik[i]:.3f}"] +
                       [f"{v:.4f}" for v in q_ik[i]])
    def write_traj_ply(path):
        n = len(u_w)
        rgb = {"green": (34, 170, 68), "red": (221, 34, 34), "yellow": (238, 187, 0)}
        hdr = ("ply\nformat binary_little_endian 1.0\n"
               f"element vertex {2 * n}\n"
               "property float x\nproperty float y\nproperty float z\n"
               "property uchar red\nproperty uchar green\nproperty uchar blue\n"
               "end_header\n")
        rows_out = []
        for i in range(n):
            rows_out.append((wall_xyz[i], rgb[cls_w[i]]))
            rows_out.append((wri_xyz[i], rgb[cls_w[i]]))
        out = np.empty(2 * n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                     ("r", "u1"), ("g", "u1"), ("b", "u1")])
        for k, (p3, c3) in enumerate(rows_out):
            out[k] = (p3[0], p3[1], p3[2], c3[0], c3[1], c3[2])
        with open(path, "wb") as f:
            f.write(hdr.encode("latin1")); f.write(out.tobytes())
    write_traj_ply(os.path.join(OUT, "trajectory_points.ply"))

    np.savez_compressed(os.path.join(OUT, "trajectory.npz"),
                        wall_u=u_w, wall_h=h_w, wall_xyz=wall_xyz, kind=kind_w, cls=cls_w,
                        zone=zone_w, rob_xyz=wri_xyz, rob_y_r=y_r, rob_z=z_r, perp=perp_ok,
                        mode=zone_w, axis=d_star, tilt=tilt_deg,
                        ik_ok=conv, ik_strict=conv_strict, q=q_ik,
                        pelvis=pz_ik, tilt_ach=90.0 - ang_wall, angle_wall=ang_wall,
                        wall_pts_sec=sec["points"], wall_lab=sec["label"],
                        u_robot=u_robot,
                        uniform_stroke_u=np.array([u_lo, u_hi]),
                        row_heights=np.array(hs_seq),
                        row_params=np.array(
                            [(rp["h"], rp["z_w"], rp["angle_deg"], rp["dir"],
                              rp["zone"], rp["u_lo"], rp["u_hi"])
                             for rp in row_params],
                            dtype=[("h", "f8"), ("z_w", "f8"),
                                   ("angle_deg", "i4"), ("dir", "i4"),
                                   ("zone", "U8"), ("u_lo", "f8"), ("u_hi", "f8")]))
    traj_meta = dict(
        arm=ARM, waist_yaw_range_deg=YAW, d_robot_nominal=D_ROBOT, d_wrist=D_WRIST,
        gun_len=GUN_LEN, u_robot=u_robot, pitch=PITCH, point_spacing=POINT_SPACING,
        stroke_u=list(STROKE_U),
        uniform_stroke_u=[float(u_lo), float(u_hi)],
        crouch_stroke_u=([float(u_lo_c), float(u_hi_c)] if DUAL else None),
        row_heights=[round(float(h), 2) for h in hs_seq],
        row_params=[dict(h=round(rp["h"], 2), z_w=round(rp["z_w"], 4),
                         angle_deg=rp["angle_deg"], dir=rp["dir"], zone=rp["zone"])
                    for rp in row_params],
        posture=POSTURE,
        stroke_dir=STROKE_DIR,
        **({"columns": [round(c, 3) for c in cols_u],
            "col_pitch": round(float(_pc), 4)} if STROKE_DIR == "vertical" else {}),
        **({"crouch_top": float(CROUCH_TOP)} if CROUCH_TOP else {}),
        h_min_row=float(min(hs_seq)), h_top=float(H_TOP),
        z_bot=float(Z_BOT_C if DUAL else Z_BOT),
        h_min_sprayed=float(H_MIN_SPRAY_MIN),
        Y_LO=Y_LO, Y_HI=Y_HI,
        tilt_max_deg=TILT_MAX, tilt_max_actual=float(tilt_deg.max()),
        row_angle_hist=ang_rows,
        angle_rule_deg=[60.0, 90.0],
        angle_achieved_deg=[float(ang_wall.min()), float(ang_wall.max())],
        angle_rule_ok=angle_rule_ok,
        stroke_geom_ok=bool(wrist_horiz_ok),          # 竖直:列u极差=0&横向分量0 / 水平:行z极差0
        n_perp_cmd=n_perp_cmd,
        z_min_robot_traj=float(z_r.min()), z_max_robot_traj=float(z_r.max()),
        obstacles=obstacles,
        ik_strict=int(conv_strict.sum()), ik_loose=int(conv_loose.sum()),
        ik_fail=int((~conv).sum()), n_points=len(u_w),
        q_contig_max=q_contig_max,
        checks=dict(wrist_z_ptp_max=float(zptp_max), wrist_z_std_max=float(zstd_max),
                    row_angle_var_max=float(angvar_max),
                    cmd_lateral_max=float(lat_cmd), fk_lateral_max=float(lat_fk),
                    wrist_horiz_ok=wrist_horiz_ok),
        rectify=dict(before_max=float(qd_pre.max()), after_max=q_contig_max,
                     before_jumps_gt06=int((qd_pre > 0.6).sum()),
                     after_jumps_gt06=int((qd_post > 0.6).sum()),
                     n_seq_replaced=int(n_rect), n_interp_fixed=int(n_interp),
                     n_angle_sel_replaced=int(n_angfix),
                     n_angle_revert=int(len(revert_idx))),
        n_green=int((cls_w == "green").sum()), n_red=int((cls_w == "red").sum()),
        n_yellow=int((cls_w == "yellow").sum()),
        stations=dict(
            base=dict(x=X_FACE - D_ROBOT, y=U0 + u_robot),
            stand_pelvis=list(PELVIS_RANGE["stand"]),
            **(dict(crouch_pelvis=list(PELVIS_RANGE["crouch"]),
                    zone_split_h=float(H_MIN_ROW),
                    crouch_rows=int(sum(1 for z in zone_rows if z == "crouch")),
                    h_min_sprayed_stand=float(H_MIN_SPRAY))
               if DUAL else {})),
        notes=dict(
            保证区="站姿单一模式(骨盆0.77~0.79m), 腰roll/pitch±0.0873(±5°)与步骤2一致; "
                  "站位0.70~0.78m三腕平面(0.30/0.34/0.38)保证区(perp/tiltd/tiltu)求交",
            等宽之字形=f"等宽+离散行角+腕心水平线联合求解: 行角θ∈{{90,80,70,60}}°(行常数,取最大档), "
                      f"腕心每行走严格水平线 z_w=h+dir*0.40*tan(90°-θ) (dir=+1枪下倾/tiltd, "
                      f"dir=-1枪上倾/tiltu, θ=90垂直/perp), y_r=u-u_robot 腕/墙同宽同步扫描, "
                      f"指向纯俯仰(横向分量=0); 统一宽度W从0.70递减0.01(中心u_c=0.36, "
                      f"clip[0.005,0.715], 两端内收0.02m), 首个使每行存在可行(θ,dir)的W="
                      f"{W:.2f}m -> 统一区间[{u_lo:.3f},{u_hi:.3f}]m; 行方向交替, 行端竖直U型连接",
            喷涂质量="连接段角度在两行角之间线性插值(仍纯俯仰, ≤30°); "
                     "关节解FK实测夹角"
                     f"[{ang_wall.min():.1f},{ang_wall.max():.1f}]°"
                     f"({'全部>=60°' if angle_rule_ok else '存在违规'}); "
                     f"自检: 行内腕心z标准差max={zstd_max:.1e}(=0), "
                     f"行实测夹角方差max={angvar_max:.2e}deg²(≈0), 指向横向分量=0; "
                     f"IK严格档{int(conv_strict.sum())}/边缘档{int(conv_loose.sum())}/"
                     f"失败{int((~conv).sum())}",
            安全="机器人轨迹最低点>=0.15m"),
    )
    json.dump(traj_meta, open(os.path.join(OUT, "trajectory_meta.json"), "w", encoding="utf8"),
              ensure_ascii=False, indent=2)

    # ---------------- 2D 图1: 墙面轨迹 ----------------
    SINGLE = "#1f7ac2"          # 统一走线颜色(用户: 不再区分绿黄红)
    colors = dict(green=SINGLE, red=SINGLE, yellow=SINGLE)
    lab_c = {0: "#c9c9c9", 1: "#d8efd8", 2: "#e3c2ee", 3: "#fff3b0", 4: "#bcd8f7", 5: "#f0f0f0"}
    lab_n = ({0: "你的点云"} if CLOUD_PLOT else
             {0: "地面", 1: "墙", 2: "门框", 3: "窗", 4: "加强筋", 5: "管道/其他"})
    fig, ax = plt.subplots(figsize=(13, 15))
    for k in np.unique(lab):
        m = lab == k
        if m.any():
            ax.scatter(pts[m, 0], pts[m, 1], s=1.2,
                       c="#9fb8c8" if CLOUD_PLOT else lab_c[k],
                       label=lab_n.get(k, "点云"), zorder=1)
    if DUAL:
        # V2(用户定稿): 站/蹲走线分色 —— 蓝=站姿(上半), 绿=蹲姿(下半), 灰虚线=回零位过渡段
        C_ST, C_CR, C_TR = "#1f7ac2", "#1e9e6a", "#8a8a8a"
        m_st = (kind_w != "transition") & (zone_w == "stand")
        m_cr = (kind_w != "transition") & (zone_w == "crouch")
        m_tr = kind_w == "transition"
        ax.plot(u_w[m_st], h_w[m_st], "-", color=C_ST, lw=2.2, zorder=3,
                solid_capstyle="round", label="站姿喷涂走线(上半 0.55~1.45m)")
        ax.plot(u_w[m_cr], h_w[m_cr], "-", color=C_CR, lw=2.2, zorder=3,
                solid_capstyle="round", label="蹲姿喷涂走线(下半 0.45m)")
        ax.plot(u_w[m_tr], h_w[m_tr], "--", color=C_TR, lw=1.6, zorder=3,
                label="回零位过渡段(05替换为收臂/停顿/下蹲)")
        for m, c in (((kind_w == "stroke") & (zone_w == "stand"), C_ST),
                     ((kind_w == "stroke") & (zone_w == "crouch"), C_CR)):
            ax.plot(u_w[m], h_w[m], "o", ms=2.5, color=c, zorder=3)
    else:
        ax.plot(u_w, h_w, "-", color=colors["green"], lw=2.2, zorder=3,
                solid_capstyle="round", label="之字形喷涂走线(行距0.1m, 匀速)")
        ax.plot(u_w[kind_w == "stroke"], h_w[kind_w == "stroke"], "o", ms=2.5,
                color=colors["green"], zorder=3)
    for oi, o in enumerate(plot_boxes):
        ax.add_patch(plt.Rectangle((o["u"][0], o["h"][0]), o["u"][1] - o["u"][0],
                                   o["h"][1] - o["h"][0], fill=False,
                                   ec="#e07b00" if CLOUD_PLOT else "#999999",
                                   ls="-" if CLOUD_PLOT else "--", lw=1.4, zorder=4,
                                   label=("点云检出障碍" if CLOUD_PLOT else "障碍范围(信息标注)")
                                   if oi == 0 else None))
    ax.axvspan(u_lo, u_hi, color="#1f7ac2", alpha=0.07,
               label=(f"两带统一u区间 [{u_lo:.2f},{u_hi:.2f}]m (站/蹲对齐)" if DUAL
                      else f"统一u区间 [{u_lo:.2f},{u_hi:.2f}]m (等宽完整之字形)"))
    ax.axhline(H_MIN_SPRAY_MIN, color="darkred", ls=":", lw=1.6)
    ax.text(0.40, H_MIN_SPRAY_MIN - 0.035, f"可喷下限 {H_MIN_SPRAY_MIN:.2f}m (与墙面夹角>=60°约束)",
            color="darkred", fontsize=10, ha="center")
    if CLOUD_PLOT:
        ax.set_xlim(pts[:, 0].min() - 0.06, pts[:, 0].max() + 0.06)
        ax.set_ylim(min(-0.05, pts[:, 1].min()), pts[:, 1].max() + 0.08)
    else:
        ax.axvline(0.720, color="darkorange", ls="--", lw=1.5)
        ax.text(0.731, 0.72, "加强筋(右边界)", rotation=90, va="center",
                color="darkorange", fontsize=13, weight="bold")
        ax.set_xlim(-0.07, 0.90); ax.set_ylim(-0.07, 1.60)
    ax.set_aspect("equal")
    ax.set_xlabel("u 沿墙向右 (m)"); ax.set_ylabel("h 距地高度 (m)")
    ax.set_title(f"G1 墙面之字形喷涂轨迹 ({'站+蹲双姿态' if DUAL else '站姿'}, 行距{PITCH}m, "
                 f"{min(hs_seq):.2f}→{H_TOP:.2f}m, "
                 f"统一u区间[{u_lo:.2f},{u_hi:.2f}], 夹角60°~90°)")
    ax.legend(loc="lower right", markerscale=4, fontsize=9); ax.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_wall_path_2d.png"), dpi=130); plt.close()

    # ---------------- 2D 图2: 机器人轨迹 + 指向 ----------------
    u_rob = u_robot + y_r      # 腕心在墙面坐标系中的横向位置
    fig, ax = plt.subplots(figsize=(13, 15))
    if CLOUD_PLOT:
        ax.scatter(pts[:, 0], pts[:, 1], s=1.0, c="#dbe6ee", zorder=0)
    else:
        m = lab == 1
        ax.scatter(pts[m, 0], pts[m, 1], s=1.0, c="#e8f2e8", zorder=0)
    ax.scatter(u_w[::4], h_w[::4], s=2.5, c="#aaaaaa", zorder=1, label="墙面轨迹点(对应参考)")
    if DUAL:
        # V2: 腕心走线同口径站/蹲分色
        C_ST, C_CR, C_TR = "#1f7ac2", "#1e9e6a", "#8a8a8a"
        m_st = (kind_w != "transition") & (zone_w == "stand")
        m_cr = (kind_w != "transition") & (zone_w == "crouch")
        m_tr = kind_w == "transition"
        ax.plot(u_rob[m_st], z_r[m_st], "-", color=C_ST, lw=2.0, zorder=3,
                solid_capstyle="round", label="站姿腕心走线(距墙0.40m)")
        ax.plot(u_rob[m_cr], z_r[m_cr], "-", color=C_CR, lw=2.0, zorder=3,
                solid_capstyle="round", label="蹲姿腕心走线(骨盆参考0.63m)")
        ax.plot(u_rob[m_tr], z_r[m_tr], "--", color=C_TR, lw=1.5, zorder=3,
                label="回零位过渡段")
    else:
        ax.plot(u_rob, z_r, "-", color=colors["green"], lw=2.0, zorder=3,
                solid_capstyle="round", label="机器人腕心走线(距墙0.40m)")
    idx = np.linspace(0, len(u_w) - 1, 150).astype(int)
    for i in idx:
        if not conv[i]:
            continue
        ax.plot([u_rob[i], u_w[i]], [z_r[i], h_w[i]], "-",
                color=SINGLE, lw=0.6, alpha=0.45, zorder=2)
    ax.axhline(Z_MIN_ROBOT, color="r", ls=":", lw=1.6)
    ax.text(0.60, Z_MIN_ROBOT + 0.008, "机器人轨迹安全下限 0.15m", color="r", fontsize=9)
    ax.axhline(H_MIN_SPRAY_MIN, color="darkred", ls=":", lw=1.6)
    ax.text(0.62, H_MIN_SPRAY_MIN - 0.035, f"可喷下限 {H_MIN_SPRAY_MIN:.2f}m (夹角>=60°)",
            color="darkred", fontsize=9, ha="center")
    ax.axhspan(0.755, 0.775, color="orchid", alpha=0.25)
    ax.text(0.60, 0.785, "站姿骨盆 0.77m", fontsize=8, color="purple")
    if DUAL:
        ax.axhspan(0.60, 0.70, color="orchid", alpha=0.18)
        ax.text(0.60, 0.585, "蹲姿骨盆带 0.60~0.70m (导出参考0.63)", fontsize=8, color="purple")
    ax.axvline(u_lo, color="#1f7ac2", ls="--", lw=1.2)
    ax.axvline(u_hi, color="#1f7ac2", ls="--", lw=1.2)
    ax.annotate("等宽完整之字形(腕心水平线,行角离散{90,80,70,60}°)",
                xy=(0.5 * (u_lo + u_hi), 1.25), xytext=(0.02, 1.52), fontsize=11,
                color="darkred", weight="bold",
                arrowprops=dict(arrowstyle="->", color="darkred"))
    if CLOUD_PLOT:
        ax.set_xlim(pts[:, 0].min() - 0.10, pts[:, 0].max() + 0.10)
        ax.set_ylim(min(-0.07, pts[:, 1].min()), pts[:, 1].max() + 0.10)
    else:
        ax.set_xlim(-0.10, 0.90); ax.set_ylim(-0.07, 1.60)
    ax.set_aspect("equal")
    ax.set_xlabel("u 沿墙向右 (m)  (腕心每行走水平线, 行角离散{90,80,70,60}°, 指向纯俯仰)")
    ax.set_ylabel("z 高度 (m)")
    ax.set_title(f"G1 机器人(腕心,距墙0.40m)等宽完整之字形(统一u区间"
                 f"[{u_lo:.2f},{u_hi:.2f}]) — {'站+蹲双姿态(先站后蹲)' if DUAL else '站姿'}, "
                 f"站位离墙0.70~0.78m")
    ax.legend(loc="lower left", markerscale=4, fontsize=9); ax.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_robot_path_2d.png"), dpi=130)
    plt.close()
    print("[预览] fig_wall_path_2d.png / fig_robot_path_2d.png")


if __name__ == "__main__":
    main()
