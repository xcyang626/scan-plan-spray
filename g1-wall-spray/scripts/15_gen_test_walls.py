# -*- coding: utf-8 -*-
"""步骤15: 生成多工位测试墙点云(2026-10-15, 旁路新增 —— 仿真/实机标定用).

按机器人系约定造墙: x=墙距(默认0.74, 与站位标称一致), y=横向u(0..W), z=0..H。
障碍两种:
  window  开口(挖洞): 删除矩形区域的墙面点 —— 11 会检出 window(开口)
  box     凸出物:     在墙面前方 depth(默认0.10m) 生成一块盒面点云

预设(与 2026-10-15 多工位验证配套; uploads/ 下):
  1m: 窄墙+窗洞(2站)   2m: 凸出盒(约5站)   3m: 凸出盒(7站, 已有同规格)
  4m: 窗洞+凸出盒(约9站)  5m: 纯平墙规模测试(约11站)

文件名带宽度/高度/障碍摘要(同宽不同障碍/高度不会写进同一个文件); 目标文件
已存在时默认不覆盖(自动加序号后缀), 要覆盖请显式加 --overwrite。

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

OB_KINDS = ("window", "box")
_FMT_HINT = ("障碍格式: window:u0,u1,z0,z1 (开口) | "
             "box:u0,u1,z0,z1[,depth] (凸出物, depth 默认 0.10m)")


def _obstacle_arg(s):
    """--obstacle 解析成 (kind,u0,u1,z0,z1[,depth]); 格式错就明确报错
    (拼写错误/个数不对/非数字都不能静默忽略)."""
    if ":" not in s:
        raise argparse.ArgumentTypeError(
            f"障碍写法错误: {s!r} —— {_FMT_HINT}")
    kind, _, rest = s.partition(":")
    kind = kind.strip().lower()
    if kind not in OB_KINDS:
        raise argparse.ArgumentTypeError(
            f"障碍类型未知: {kind!r} —— 只支持 {'/'.join(OB_KINDS)}; {_FMT_HINT}")
    parts = [p.strip() for p in rest.split(",") if p.strip() != ""]
    if kind == "window" and len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"window 障碍要 4 个数(u0,u1,z0,z1), 收到 {len(parts)} 个: {s!r} —— {_FMT_HINT}")
    if kind == "box" and len(parts) not in (4, 5):
        raise argparse.ArgumentTypeError(
            f"box 障碍要 4 或 5 个数(u0,u1,z0,z1[,depth]), 收到 {len(parts)} 个: "
            f"{s!r} —— {_FMT_HINT}")
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"障碍参数不是数字: {s!r} —— {_FMT_HINT}")
    if not all(np.isfinite(v) for v in vals):
        raise argparse.ArgumentTypeError(f"障碍参数含非有限值: {s!r}")
    u0, u1, z0, z1 = vals[:4]
    if u1 <= u0 or z1 <= z0:
        raise argparse.ArgumentTypeError(
            f"障碍范围非法(要求 u0<u1 且 z0<z1): {s!r}")
    if kind == "window":
        return (kind, u0, u1, z0, z1)
    depth = vals[4] if len(vals) > 4 else 0.10
    if depth <= 0:
        raise argparse.ArgumentTypeError(f"box 凸出深度需 >0: {s!r}")
    return (kind, u0, u1, z0, z1, depth)


def _obs_summary(obstacles):
    """障碍摘要(进文件名): 无障碍 -> noobs; 否则 win0.3-0.6_box0.95-1.2 ..."""
    if not obstacles:
        return "noobs"
    return "_".join(f"{o[0][:3]}{o[1]:g}-{o[2]:g}" for o in obstacles)


def _wall_name(width, height, obstacles, tag=""):
    """默认文件名: 宽x高+障碍摘要(同宽不同障碍/高度不会写进同一个文件).
    预设用 tag(预设名已含宽度), 自定义用实际宽高。"""
    obs = _obs_summary(obstacles)
    if tag:
        return f"wall_{tag}_x{height:g}m_{obs}.ply"
    return f"wall_{width:g}m_x{height:g}m_{obs}.ply"


def _no_overwrite_path(path, overwrite=False):
    """已存在不静默覆盖: 除非 --overwrite, 否则加序号后缀(wall.ply -> wall_1.ply)."""
    if overwrite or not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    i = 1
    while os.path.exists(f"{root}_{i}{ext}"):
        i += 1
    return f"{root}_{i}{ext}"


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
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)          # --out 指向不存在的目录也能写
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {len(pts)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "end_header\n").encode())
        f.write(pts.astype("<f4").tobytes())


def main():
    ap = argparse.ArgumentParser(description="生成多工位测试墙点云")
    ap.add_argument("--preset", choices=sorted(PRESETS), help="预设尺寸")
    ap.add_argument("--width", type=float, help="自定义墙宽(m)")
    ap.add_argument("--height", type=float, default=None,
                    help="墙高(m); 预设时给出会覆盖预设高度(不给=用预设/默认2.0)")
    ap.add_argument("--dist", type=float, default=0.74, help="墙距(机器人系 x)")
    ap.add_argument("--obstacle", action="append", default=[], type=_obstacle_arg,
                    help="障碍, 可多次: window:u0,u1,z0,z1 | box:u0,u1,z0,z1[,depth]")
    ap.add_argument("--seed", type=int, default=7, help="随机种子(默认 7)")
    ap.add_argument("--out", help="输出 ply 路径(默认 uploads/wall_<W>x<H>_<障碍摘要>.ply)")
    ap.add_argument("--overwrite", action="store_true",
                    help="目标文件已存在时直接覆盖(默认不覆盖, 自动加序号后缀)")
    a = ap.parse_args()

    if a.preset:
        p = PRESETS[a.preset]
        width, height = p["width"], p["height"]
        if a.height is not None and abs(a.height - height) > 1e-9:
            print(f"[提示] 预设 {a.preset} 的高度 {height}m 被 --height "
                  f"{a.height}m 覆盖")
            height = a.height
        obstacles = [tuple(o) for o in p["obstacles"]]
        out = a.out or os.path.join(BASE, "uploads",
                                    _wall_name(width, height, obstacles,
                                               tag=a.preset))
    else:
        if not a.width:
            ap.error("需要 --preset 或 --width")
        if a.width <= 0:
            ap.error(f"--width 需 >0, 收到 {a.width}")
        width = a.width
        if a.height is None:
            height = 2.0
            print("[提示] 未指定 --height, 使用默认 2.0m")
        else:
            height = a.height
            if height <= 0:
                ap.error(f"--height 需 >0, 收到 {height}")
        obstacles = list(a.obstacle)
        out = a.out or os.path.join(BASE, "uploads",
                                    _wall_name(width, height, obstacles))
    if a.dist <= 0:
        ap.error(f"--dist 需 >0, 收到 {a.dist}")
    pts = gen_wall(width, height, obstacles, dist=a.dist, seed=a.seed)
    out = _no_overwrite_path(out, overwrite=a.overwrite)
    save_ply(out, pts)
    obs_txt = ("; ".join(f"{o[0]} u[{o[1]},{o[2]}] h[{o[3]},{o[4]}]"
                         + (f" 深{o[5]}" if len(o) > 5 and o[5] else "")
                         for o in obstacles) or "无")
    print(f"生成 {out}: {len(pts)} 点, 墙 {width}x{height}m @x={a.dist}, "
          f"障碍: {obs_txt}, seed={a.seed}")


if __name__ == "__main__":
    main()
