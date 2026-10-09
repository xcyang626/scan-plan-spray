import numpy as np
import struct

PLY = r"c:\Users\Yangon\.trae-cn\attachments\6a87c1a056617ae4ad10f9da\d292b523-42b2-4ce1-b436-36ef35603d7f_7154ca79-a3f1-44b1-9f01-ee3b3416691a_舱室点云_语义标注.ply"

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

np.save(r"c:\Users\Yangon\Desktop\舱室绝缘喷涂人形机器人技术方案\喷涂点位划分\work\xyz.npy", xyz.astype(np.float32))
np.save(r"c:\Users\Yangon\Desktop\舱室绝缘喷涂人形机器人技术方案\喷涂点位划分\work\rgb.npy", rgb.astype(np.uint8))
np.save(r"c:\Users\Yangon\Desktop\舱室绝缘喷涂人形机器人技术方案\喷涂点位划分\work\pid.npy", pid)
print("\nsaved npy caches")
