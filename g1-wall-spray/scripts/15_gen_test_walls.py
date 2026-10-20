# -*- coding: utf-8 -*-
"""步骤15: 生成多工位测试墙点云(2026-10-15, 旁路新增 —— 仿真/实机标定用).

按机器人系约定造墙: x=墙距(默认0.74, 与站位标称一致), y=横向u(0..W), z=0..H。
障碍两种:
  window  开口(挖洞): 删除矩形区域的墙面点 —— 11 会检出 window(开口)
  box     凸出物:     在墙面前方 depth(默认0.10m) 生成一块盒面点云

预设(与 2026-10-15 多工位验证配套; uploads/ 下):
  1m: 窄墙+窗洞(2站)   2m: 凸出盒(约5站)   3m: 凸出盒(7站, 已有同规格)
  4m: 窗洞+凸出盒(约9站)  5m: 纯平墙规模测试(约11站)

用法(isaaclab 解释器):
  python 15_gen_test_walls.py --preset 5m                 # 预设
  python 15_gen_test_walls.py --width 6 --height 2.2 \
      --obstacle window:2.5,3.1,0.9,1.3 --obstacle box:4.2,4.5,1.0,1.4,0.12 \
      --out ../uploads/wall_6m_custom.ply                 # 自定义
"""
import os
import argparse
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRESETS = {
    "1m": dict(width=1.0, height=2.0,
               obstacles=[("window", 0.30, 0.60, 0.90, 1.30, 0.0)]),
    "2m": dict(width=2.0, height=2.0,
               obstacles=[("box", 0.95, 1.20, 0.95, 1.35, 0.10)]),
    "3m": dict(width=3.0, height=2.0,
               obstacles=[("box", 1.40, 1.62, 0.90, 1.32, 0.10)]),
    "4m": dict(width=4.0, height=2.0,
               obstacles=[("window", 1.20, 1.70, 0.90, 1.30, 0.0),
                          ("box", 2.60, 2.82, 1.00, 1.40, 0.10)]),
    "5m": dict(width=5.0, height=2.0, obstacles=[]),
}


def gen_wall(width, height, obstacles, dist=0.74, seed=7):
    """墙面网格点 + 障碍(开口删点 / 凸出盒加点), 返回 (N,3) float32."""
    rng = np.random.default_rng(seed)
    ys = np.arange(0.0, width, 0.02)
    zs = np.arange(0.0, height, 0.025)
    Y, Z = np.meshgrid(ys, zs)
    keep = np.ones(Y.shape, bool)
    pts_extra = []
    for ob in obstacles:
        kind, u0, u1, z0, z1 = ob[0], ob[1], ob[2], ob[3], ob[4]
        depth = ob[5] if len(ob) > 5 else 0.10
        m = (Y >= u0) & (Y <= u1) & (Z >= z0) & (Z <= z1)
        if kind == "window":
            keep &= ~m                        # 开口: 删墙面点
        elif kind == "box":                   # 凸出盒: 盒正面+侧缘点
            oy, oz = np.meshgrid(np.arange(u0, u1 + 1e-9, 0.02),
                                 np.arange(z0, z1 + 1e-9, 0.02))
            pts_extra.append(np.column_stack([
                np.full(oy.size, dist - depth), oy.ravel(), oz.ravel()]))
    X = np.full_like(Y, dist)
    pts = np.column_stack([X.ravel()[keep.ravel()],
                           Y.ravel()[keep.ravel()]
                           + rng.normal(0, 0.002, int(keep.sum())),
                           Z.ravel()[keep.ravel()]])
    pts[:, 0] += rng.normal(0, 0.003, len(pts))
    pts[:, 2] += rng.normal(0, 0.003, len(pts))
    for pe in pts_extra:
        pe = pe + rng.normal(0, 0.003, pe.shape)
        pts = np.vstack([pts, pe])
    return pts.astype(np.float32)


def save_ply(path, pts):
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {len(pts)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "end_header\n").encode())
        f.write(pts.astype("<f4").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), help="预设尺寸")
    ap.add_argument("--width", type=float, help="自定义墙宽(m)")
    ap.add_argument("--height", type=float, default=2.0)
    ap.add_argument("--dist", type=float, default=0.74, help="墙距(机器人系 x)")
    ap.add_argument("--obstacle", action="append", default=[],
                    help="障碍, 可多次: window:u0,u1,z0,z1 | box:u0,u1,z0,z1[,depth]")
    ap.add_argument("--out", help="输出 ply 路径(默认 uploads/wall_<W>m_test.ply)")
    a = ap.parse_args()

    if a.preset:
        p = PRESETS[a.preset]
        width, height, obstacles = p["width"], p["height"], [
            tuple(o) for o in p["obstacles"]]
        out = a.out or os.path.join(BASE, "uploads",
                                    f"wall_{a.preset}_test.ply")
    else:
        if not a.width:
            ap.error("需要 --preset 或 --width")
        width, height = a.width, a.height
        obstacles = []
        for s in a.obstacle:
            parts = s.split(":")
            vals = [float(v) for v in parts[1].split(",")]
            obstacles.append((parts[0], *vals))
        out = a.out or os.path.join(BASE, "uploads",
                                    f"wall_{width:g}m_test.ply")
    pts = gen_wall(width, height, obstacles, dist=a.dist)
    save_ply(out, pts)
    obs_txt = ("; ".join(f"{o[0]} u[{o[1]},{o[2]}] h[{o[3]},{o[4]}]"
                         + (f" 深{o[5]}" if len(o) > 5 and o[5] else "")
                         for o in obstacles) or "无")
    print(f"生成 {out}: {len(pts)} 点, 墙 {width}x{height}m @x={a.dist}, 障碍: {obs_txt}")


if __name__ == "__main__":
    main()
