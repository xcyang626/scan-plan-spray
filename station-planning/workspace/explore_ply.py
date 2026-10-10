"""读取语义标注 ply(字段: x,y,z float32 + part_id uint16 + r,g,b uint8),
   导出 xyz/rgb/pid 三个 npy 缓存供 analyze_*/fig_* 脚本使用"""
import os
import sys
from pathlib import Path

import numpy as np

# 数据目录: 优先环境变量 STATION_DATA, 未设时默认脚本同级 "数据" 目录;
# 源 ply: 优先环境变量 STATION_PLY, 未设时默认数据目录下的 舱室点云_语义标注.ply
# 注: 仓库内 pointcloud-registration/舱室点云/ 中的 ply 字段名为 red/green/blue
# 且仅 30 万点, 与本脚本期望的 r/g/b 字段(约 502 万点原扫描)不同, 不能直接替代。
W = Path(os.environ.get("STATION_DATA", Path(__file__).resolve().parent / "数据"))
PLY = Path(os.environ.get("STATION_PLY", W / "舱室点云_语义标注.ply"))
if not PLY.exists():
    sys.exit(f"[路径] 缺少语义点云文件 {PLY}: 请将 舱室点云_语义标注.ply (字段 x,y,z,part_id,r,g,b) "
             f"放入 {W}, 或设置环境变量 STATION_PLY 指向该文件")

with open(PLY, "rb") as f:
    header = b""
    while b"end_header" not in header:
        header += f.readline()
    n = int([l for l in header.split(b"\n") if l.startswith(b"element vertex")][0].split()[2])
    # dtype: x,y,z float32; part_id uint16; r,g,b uint8
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("part_id", "<u2"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    data = np.fromfile(f, dtype=dt, count=n)

print("total points:", len(data))
xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)
rgb = np.stack([data["r"], data["g"], data["b"]], axis=1)
pid = data["part_id"]

print("bbox min:", xyz.min(axis=0).round(3))
print("bbox max:", xyz.max(axis=0).round(3))

# unique colors
colors, counts = np.unique(rgb, axis=0, return_counts=True)
order = np.argsort(-counts)
print("\nunique colors (r,g,b) count:")
for i in order[:30]:
    print(tuple(colors[i]), counts[i])

print("\npart_id values:", np.unique(pid))
for v in np.unique(pid):
    m = pid == v
    print(f"part_id={v}: n={m.sum()}, bbox=({xyz[m].min(axis=0).round(2)}, {xyz[m].max(axis=0).round(2)})")

W.mkdir(parents=True, exist_ok=True)
np.save(W / "xyz.npy", xyz.astype(np.float32))
np.save(W / "rgb.npy", rgb.astype(np.uint8))
np.save(W / "pid.npy", pid)
print("\nsaved npy caches ->", W)
