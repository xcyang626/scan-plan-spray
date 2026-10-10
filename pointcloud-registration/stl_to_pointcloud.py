# -*- coding: utf-8 -*-
"""
将 SolidWorks 导出的整个装配体的二进制 STL 零件批量转换为一张点云。

- 遍历 STL 文件夹内所有 .STL 文件（每个文件对应装配体的一个零件）
- 每个零件在三角网格表面按面积均匀采样点（面积占比分配点数）
- 合并为单一点云，每个点带零件 ID 和按零件着色的 RGB
- 输出为二进制 PLY（可用 CloudCompare / Open3D / MeshLab 打开）
"""
import os
import re
import glob
import numpy as np

# ------------------------- 配置 -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STL_DIR = os.path.join(BASE_DIR, "模型stl")
OUT_DIR = os.path.join(BASE_DIR, "舱室点云")
OUT_PLY = os.path.join(OUT_DIR, "舱室装配体点云.ply")

TOTAL_POINTS = 5_000_000   # 整张点云的总点数（按零件面积比例分配）
MIN_POINTS_PER_PART = 200  # 每个零件最小采样点数，避免小零件被丢弃
POINTS_PER_BATCH = 200_000 # 单零件采样分块大小，控制内存
RANDOM_SEED = 42
# ------------------------- 配置结束 -------------------------


def read_binary_stl(path):
    """读取 (二进制优先，自动回退 ASCII) STL，返回顶点数组 [F,3,3]。"""
    with open(path, "rb") as f:
        data = f.read()
    cnt = np.frombuffer(data[80:84], dtype="<u4")
    if len(cnt) == 1 and cnt[0] * 50 + 84 == len(data):
        # 二进制 STL：80字节头 + uint32 面数 + 每面50字节(法向量12 + 3顶点36 + 属性2)
        # 用结构化 dtype 一次性向量化读取（每面恰 50 字节、字段紧凑无对齐填充），
        # 替代原逐面 Python 循环，结果逐位一致
        dt_stl = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")])
        assert dt_stl.itemsize == 50
        nrec = (len(data) - 84) // 50
        rec = np.frombuffer(data, dtype=dt_stl, count=nrec, offset=84)
        return rec["v"]  # 丢弃法向量，仅取三个顶点
    # ASCII STL 回退（原实现对 findall 的三元组逐个 float() 会 TypeError，直接整体转数组）
    text = data.decode("ascii", errors="ignore")
    pts = np.array(re.findall(r"vertex\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)", text),
                   dtype=np.float64)
    return pts.reshape(-1, 3, 3)


def area_weighted_sample(verts, n_points):
    """在三角形网格 [F,3,3] 上按面积均匀采样 n_points 个点，返回 [N,3]。可一次穿过所有点。"""
    if n_points <= 0 or len(verts) == 0:
        return np.empty((0, 3), dtype=np.float64)
    v0, v1, v2 = verts[:, 0], verts[:, 1], verts[:, 2]
    cross = np.cross(v1 - v0, v2 - v0)
    area = np.linalg.norm(cross, axis=1)
    total = area.sum()
    if total == 0:
        return np.empty((0, 3), dtype=np.float64)
    # 使用采样-拒绝，直接一次性完成面积加权采样
    out = np.empty((n_points, 3), dtype=np.float64)
    idx_base = 0
    while idx_base < n_points:
        m = min(POINTS_PER_BATCH, n_points - idx_base)
        f = np.random.choice(len(area), size=m, p=area / total)
        u1 = np.sqrt(np.random.uniform(0, 1, m))
        u2 = np.random.uniform(0, 1, m)
        out[idx_base:idx_base + m] = (
            (1 - u1)[:, None] * v0[f]
            + (u1 * (1 - u2))[:, None] * v1[f]
            + (u1 * u2)[:, None] * v2[f]
        )
        idx_base += m
    return out


def part_colors(n_parts):
    """用黄金角HSV旋转为每个零件生成视觉上高区分度且唯一的RGB。"""
    hue = np.arange(n_parts) * 0.6180339887498949025  # 黄金角，避免相邻色相近
    hue = (hue - np.floor(hue)) * 360.0
    # sat/val 在相邻零件间微微错开，进一步提高区分度
    band = (np.arange(n_parts) // 7) % 2
    sat = np.where(band == 0, 0.85, 0.55)
    val = np.where(band == 0, 0.95, 0.75)
    h = hue / 60.0
    x = 1.0 - np.abs(h % 2 - 1.0)
    # 标准 HSV→RGB 分段：激活段取 1、上升/下降段取 x、非激活段取 0
    # （原实现在非激活段也取 x：R 的 h∈[2,4)、G 的 h∈[4,6)、B 的 h∈[0,2)
    #   均应为 0，导致纯绿偏黄、紫色偏灰等色偏）
    c = (val * sat)[:, None] * np.stack([
        np.where((h < 1) | (h >= 5), 1.0, np.where((h < 2) | (h >= 4), x, 0.0)),  # R
        np.where((h >= 1) & (h < 3), 1.0, np.where(h < 4, x, 0.0)),               # G
        np.where((h >= 3) & (h < 5), 1.0, np.where(h >= 2, x, 0.0)),              # B
    ], axis=1)
    m = (val * (1.0 - sat))[:, None]                                # m = V - C (C = V·S 为纯色标量)
    rgb = c + m
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def write_binary_ply(path, xyz, part_id, rgb):
    """写出二进制 PLY，带 x,y,z, part_id(uint16), r,g,b(uchar) 属性。"""
    n = len(xyz)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property ushort part_id\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    # 结构化数组保证内存布局与 PLY 逐顶点属性顺序一致
    if n == 0:
        with open(path, "wb", encoding="latin1") as fh:
            fh.write(header.encode("latin1"))
        return
    rec = np.empty(n, dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("part_id", "<u2"), ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["part_id"] = part_id.astype(np.uint16)
    rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as fh:
        fh.write(header.encode("latin1"))
        fh.write(rec.tobytes())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(RANDOM_SEED)
    stl_files = sorted(glob.glob(os.path.join(STL_DIR, "*.STL")))
    if not stl_files:
        print(f"未在 {STL_DIR} 找到 .STL 文件")
        return
    print(f"共 {len(stl_files)} 个零件 STL")

    # 第一遍：读取并统计每个零件的面积与顶点，据此分配点数
    metas = []
    total_area = 0.0
    for fp in stl_files:
        verts = read_binary_stl(fp)
        area = 0.5 * np.linalg.norm(
            np.cross(verts[:, 1] - verts[:, 0], verts[:, 2] - verts[:, 0]), axis=1
        ).sum()
        metas.append({"path": fp, "verts": verts, "area": area})
        total_area += area

    # 按面积占比分配总点数，并设置每零件最小点数
    counts = [max(MIN_POINTS_PER_PART, round(TOTAL_POINTS * (m["area"] / total_area)))
              for m in metas]

    # 第二遍：采样合并 —— 按 counts 前缀和预分配总空间后逐段填充，
    # 替代原循环 vstack 的反复整块拷贝（O(n²) 拷贝 → O(n) 填充）
    colors = part_colors(len(metas))
    starts = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)  # 各零件段起点(前缀和)
    all_xyz = np.empty((int(starts[-1]), 3), dtype=np.float64)
    all_id = np.empty(int(starts[-1]), dtype=np.uint16)
    all_rgb = np.empty((int(starts[-1]), 3), dtype=np.uint8)
    off = 0
    for i, (meta, cnt) in enumerate(zip(metas, counts)):
        pts = area_weighted_sample(meta["verts"], cnt)
        all_xyz[off:off + len(pts)] = pts
        all_id[off:off + len(pts)] = i
        all_rgb[off:off + len(pts)] = colors[i]
        off += len(pts)
        print(f"[{i + 1}/{len(metas)}] {os.path.basename(meta['path'])}  "
              f"面积={meta['area']:.1f}mm2 点数={cnt}")
    # 退化零件(面积0)返回点数可能少于分配数：截去未填充尾部，保持三数组等长
    all_xyz, all_id, all_rgb = all_xyz[:off], all_id[:off], all_rgb[:off]

    print(f"\n点云总点数: {len(all_xyz)}")
    lo, hi = all_xyz.min(axis=0), all_xyz.max(axis=0)
    print("坐标范围 X: %.2f ~ %.2f, Y: %.2f ~ %.2f, Z: %.2f ~ %.2f"
          % (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))

    write_binary_ply(OUT_PLY, all_xyz, all_id, all_rgb)
    print(f"已输出: {OUT_PLY}")
    print(f"零件数: {len(stl_files)}，唯一零件ID数: {np.unique(all_id).size}")


if __name__ == "__main__":
    main()