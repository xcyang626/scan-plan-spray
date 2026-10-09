# -*- coding: utf-8 -*-
"""生成降采样样例数据，让本模块开箱即可运行。

原始全量点云约 500 万点、合计 >100 MB，体积过大不入库。本脚本从原始工程
生成一对降采样样例（覆盖 舟室点云/ 下同名文件）：

  - 舱室点云_语义标注.ply     语义参考地图 M（体素降采样，逐点保留语义颜色）
  - 舱室点云_原始扫描黑白.ply  机器人黑白扫描 S（对装配体施加已知刚体变换
    到机器人局部坐标系 + 0.5 mm 测量噪声，模拟真实建图）
  - gt_robot_scan.npy         生成时使用的真值 4x4 变换（供配准精度校验）

用法：
    python make_sample_data.py --src /path/to/原始工程/点云.py
"""
import os
import argparse
import importlib.util

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "rl", os.path.join(HERE, "register_and_label.py"))
rl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="原始工程 点云.py 目录（内含 舱室点云/）")
    ap.add_argument("--leaf", type=float, default=12.0,
                    help="降采样体素边长 mm（默认 12）")
    ap.add_argument("--max-m", type=int, default=300000,
                    help="语义图样例点数上限")
    ap.add_argument("--max-s", type=int, default=200000,
                    help="机器人扫描样例点数上限")
    args = ap.parse_args()

    cloud = os.path.join(args.src, "舱室点云")
    rng = np.random.default_rng(2026)
    out_cloud = os.path.join(HERE, "舱室点云")

    # ---- M：语义参考图（保留颜色语义标签） ----
    print("[1/3] 读取语义参考图并降采样")
    M_xyz, M_rec, M_rgb = rl.load_ply(
        os.path.join(cloud, "舱室点云_语义标注.ply"))
    selM = rl.voxel_subsample(M_xyz, args.leaf, args.max_m, 21)
    rl.write_ply(os.path.join(out_cloud, "舱室点云_语义标注.ply"),
                 M_xyz[selM], M_rgb[selM], M_rec["part_id"][selM])

    # ---- S：由装配体生成机器人局部坐标系的黑白扫描 ----
    print("[2/3] 读取装配体点云，施加已知刚体变换 + 噪声")
    A_xyz, A_rec, _ = rl.load_ply(
        os.path.join(cloud, "舱室装配体点云.ply"))
    selA = rl.voxel_subsample(A_xyz, args.leaf, args.max_s, 22)
    P = A_xyz[selA] + rng.normal(0.0, 0.5, (len(selA), 3))

    center = A_xyz.mean(0)
    ax = np.deg2rad(np.array([3.0, -2.0, 4.0]))
    R = (rl.exp_so3([ax[0], 0, 0]) @ rl.exp_so3([0, ax[1], 0])
         @ rl.exp_so3([0, 0, ax[2]]))
    t = np.array([300.0, -200.0, 150.0])
    S_robot = (P - center) @ R.T + t

    inten = np.clip(200 + rng.integers(-55, 30, len(S_robot)),
                    35, 255).astype(np.uint8)
    gray = np.tile(inten[:, None], (1, 3))
    rl.write_ply(os.path.join(out_cloud, "舱室点云_原始扫描黑白.ply"),
                 S_robot, gray, A_rec["part_id"][selA])

    # ---- 真值变换（参考系 → 机器人系） ----
    print("[3/3] 保存真值变换")
    GT = np.eye(4)
    GT[:3, :3] = R
    T0 = np.eye(4)
    T0[:3, 3] = -center
    GT = GT @ T0
    GT[:3, 3] = GT[:3, 3] + t
    np.save(os.path.join(HERE, "gt_robot_scan.npy"), GT)

    print("样例生成完成：M=%d 点，S=%d 点（leaf=%.0fmm）"
          % (len(selM), len(selA), args.leaf))


if __name__ == "__main__":
    main()
