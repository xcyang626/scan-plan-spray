# -*- coding: utf-8 -*-
"""步骤1: 从舱室语义点云中提取喷涂墙面段.

要求(与工位规划一致):
  - 宽 ~0.8m, 高 0~1.5m (从地面起)
  - 段内包含窗户(喷涂时红色快速穿过)
  - 加强筋(rib)位于最右边
选择: x-max 墙 (内面 x≈212292mm, 法向朝室内 -x), 横向 u = 世界 y (向右增大)
  - 切片 u ∈ [7789, 8589]mm (0.80m), h = z-57563 ∈ [0, 1500]mm
  - 大窗 u[6364,8393] h[478,2108] -> 段内部分 u[7789,8393] h[478,1500]
  - 加强筋 D u[8509,8589] (凸出125mm) 正好在段的最右边

输出:
  output/wall_section.ply      段内点云(段局部坐标, 米)
  output/wall_section.npz      段内点云数组 + 标签
  output/wall_section_meta.json几何元数据
  output/fig_wall_section.png  段正视图(标签着色)
"""
import numpy as np
import json, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLY_IN = "/home/yang/桌面/轨迹规划/cabin_solution/点云.py/舱室点云/舱室点云_语义标注.ply"
OUT = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)

# ---- 参数 ----
U0, U1 = 7789.0, 8589.0          # 段横向范围 (世界y, mm)  宽0.80m
H0, H1 = 0.0, 1500.0             # 段高度范围 (地面起, mm)
X_FACE = 212292.0                # 墙内面 (mm)
X_MIN, X_MAX = 212120.0, 212360.0  # 提取厚度范围(含加强筋凸出125mm)
FLOOR_Z = 57563.0

C = dict(ground=(128, 128, 128), wall=(150, 220, 150), door=(178, 64, 222),
         window=(255, 230, 0), rib=(0, 110, 255), other=(0, 0, 0))
LABEL_ID = {k: i for i, k in enumerate(C.items())}
LBL_NAMES = list(C.keys())


def load_ply(path):
    with open(path, "rb") as f:
        header = b""
        while True:
            line = f.readline(); header += line
            if line.strip() == b"end_header":
                break
        body = f.read()
    n = int([l for l in header.decode().splitlines()
             if l.startswith("element vertex")][0].split()[-1])
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("part_id", "<u2"),
                   ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec = np.frombuffer(body, dtype=dt, count=n)
    xyz = np.stack([rec["x"], rec["y"], rec["z"]], 1).astype(np.float64)
    rgb = np.stack([rec["r"], rec["g"], rec["b"]], 1).astype(np.int32)
    return xyz, rgb


def write_ply(path, xyz, rgb, labels):
    n = len(xyz)
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           f"element vertex {n}\n"
           "property float x\nproperty float y\nproperty float z\n"
           "property uchar label\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\n"
           "end_header\n")
    out = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                             ("label", "u1"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out["label"] = labels.astype(np.uint8)
    out["r"], out["g"], out["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(hdr.encode("latin1")); f.write(out.tobytes())
    print("[写出]", path, n, "点")


def main():
    xyz, rgb = load_ply(PLY_IN)
    lbl = np.full(len(xyz), -1, np.int32)
    for i, k in enumerate(LBL_NAMES):
        c = C[k]
        m = (rgb[:, 0] == c[0]) & (rgb[:, 1] == c[1]) & (rgb[:, 2] == c[2])
        lbl[m] = i

    # ---- 段内选择 ----
    sel = ((xyz[:, 1] >= U0) & (xyz[:, 1] <= U1) &
           (xyz[:, 2] >= FLOOR_Z + H0 - 10) & (xyz[:, 2] <= FLOOR_Z + H1 + 10) &
           (xyz[:, 0] >= X_MIN) & (xyz[:, 0] <= X_MAX))
    P = xyz[sel].copy()
    L = lbl[sel].copy()
    print(f"段内点数: {len(P)}")
    for i, k in enumerate(LBL_NAMES):
        n = int((L == i).sum())
        if n:
            q = P[L == i]
            print(f"  {k:8s} n={n:7d}  x[{q[:,0].min():.0f},{q[:,0].max():.0f}] "
                  f"y[{q[:,1].min():.0f},{q[:,1].max():.0f}] z[{q[:,2].min():.0f},{q[:,2].max():.0f}]")

    # ---- 段局部坐标 (米): u'=y-U0 向右, h=z-FLOOR 向上, d=X_FACE-x 向室内为正 ----
    loc = np.stack([(P[:, 1] - U0) / 1000.0,
                    (P[:, 2] - FLOOR_Z) / 1000.0,
                    (X_FACE - P[:, 0]) / 1000.0], 1)   # (u, h, d)

    # ---- 窗户/加强筋元数据 (段坐标) ----
    def bbox(mask):
        if mask.sum() < 5:
            return None
        q = loc[mask]
        return dict(u=[float(q[:, 0].min()), float(q[:, 0].max())],
                    h=[float(q[:, 1].min()), float(q[:, 1].max())],
                    d=[float(q[:, 2].min()), float(q[:, 2].max())],
                    n=int(mask.sum()))

    i_win = LBL_NAMES.index("window")
    i_rib = LBL_NAMES.index("rib")
    win_bb = bbox(L == i_win)
    rib_bb = bbox(L == i_rib)
    print("窗户bbox(段坐标):", win_bb)
    print("加强筋bbox(段坐标):", rib_bb)

    meta = dict(
        wall="x-max 墙 (内面 x=212292mm, 法向朝室内 -x), 横向u=世界y(向右增大)",
        floor_z_mm=FLOOR_Z,
        u_range_mm=[U0, U1], width_m=(U1 - U0) / 1000.0,
        h_range_m=[0.0, 1.5],
        world=dict(u_axis="world y", x_face=X_FACE),
        window=win_bb, rib=rib_bb,
        rib_world_y_mm=[8509, 8589],
        window_world=dict(y=[6364, 8393], z=[58041, 59671]),
        labels={k: i for i, k in enumerate(LBL_NAMES)},
    )
    with open(os.path.join(OUT, "wall_section_meta.json"), "w", encoding="utf8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    write_ply(os.path.join(OUT, "wall_section.ply"), loc, rgb[sel], L)
    np.savez_compressed(os.path.join(OUT, "wall_section.npz"),
                        points=loc.astype(np.float32), label=L.astype(np.int32))

    # ---- 正视图 PNG ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common import setup_cn_font
    setup_cn_font()
    plt.figure(figsize=(10, 16))
    ax = plt.gca()
    order = np.argsort(L)  # 后画的标签在上层
    for i in np.unique(L[order]):
        m = L == i
        c = np.array(C[LBL_NAMES[i]]) / 255.0 if i >= 0 else (0.5, 0.5, 0.5)
        name = LBL_NAMES[i] if i >= 0 else "?"
        ax.scatter(loc[m, 0], loc[m, 1], s=1.5, c=[c], label=f"{name} ({m.sum()})")
    if win_bb:
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((win_bb["u"][0], win_bb["h"][0]),
                               win_bb["u"][1] - win_bb["u"][0],
                               win_bb["h"][1] - win_bb["h"][0],
                               fill=False, ec="red", lw=1.5, ls="--"))
    if rib_bb:
        ax.axvline(rib_bb["u"][0], color="orange", lw=2, ls="--")
    ax.set_xlim(-0.05, 0.85); ax.set_ylim(-0.05, 1.55)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("u (m)  向右"); ax.set_ylabel("h (m)  距地面高度")
    ax.set_title("提取的墙面段 0.80m × 1.5m (左: 窗户, 最右: 加强筋)")
    ax.legend(loc="upper left", markerscale=6, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_wall_section.png"), dpi=130)
    print("[预览]", os.path.join(OUT, "fig_wall_section.png"))


if __name__ == "__main__":
    main()
