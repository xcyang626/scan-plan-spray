# -*- coding: utf-8 -*-
"""步骤11: 点云输入 -> 墙面平面拟合 / 距离核验 / 喷涂范围 / 凸出障碍检测.

输入(机器人系: x前 y左 z上, 单位m):
  .ply (ascii/binary_little_endian) / .pcd (ascii/binary) / .npy / .txt / .csv
用法:
  python 11_cloud_scan.py --cloud <file> [--dist 0.74] [--out output/cloud_report.json]

输出:
  cloud_report.json: {distance, wall_u:[..], wall_h:[..], stroke_u:[..], h_top, obstacles[], warns[]}
  cloud_report.png : 俯视 + 正视(含喷涂带) 两联图, 供工作站检查页显示
配置约定: stroke_u 取 无凸出障碍(深度>0.05m)的最大无遮挡区间(两端内收0.02m),
  保证等宽之字形落在可达且无遮挡的墙面上。
"""
import os, sys, json, struct, argparse
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------- 点云读取 ----------------
def _ply_points(path):
    with open(path, 'rb') as f:
        head, got, n = [], False, None
        while not got:
            line = f.readline()
            if not line:
                raise ValueError('PLY 头不完整')
            s = line.decode('ascii', 'ignore').strip()
            head.append(s)
            if s.startswith('element vertex'):
                n = int(s.split()[-1])
            if s == 'end_header':
                got = True
        if n is None:
            raise ValueError('PLY 头缺少 element vertex 行, 无法确定点数')
        fmt = next((l for l in head if l.startswith('format')), 'ascii')
        props, bin_types = [], {'float': 'f4', 'double': 'f8', 'uchar': 'u1',
                                'int': 'i4', 'uint': 'u4', 'short': 'i2', 'ushort': 'u2',
                                'char': 'i1', 'float16': 'f2'}
        ncol = 0
        for l in head[head.index('end_header') - 200 if False else 0:]:
            pass
        for l in head:
            p = l.split()
            if p[:1] == ['property'] and len(p) == 3 and p[1] in bin_types:
                props.append((p[1], p[2])); ncol += 1
            elif p[:1] == ['property'] and len(p) == 3 and p[1] == 'list':
                raise ValueError('暂不支持含 list 属性的 PLY')
        if 'ascii' in fmt:
            data = np.loadtxt(open(path, 'rb'), skiprows=len(head), max_rows=n, ndmin=2)
        else:
            # 按 fmt 声明选端序: binary_big_endian 必须按大端解, 不能一律当小端
            endian = '>' if 'binary_big_endian' in fmt else '<'
            dtype = np.dtype([(name, endian + bin_types[t]) for t, name in props])
            raw = np.frombuffer(f.read(dtype.itemsize * n), dtype=dtype)
            data = np.column_stack([raw[name] for _, name in props]).astype(np.float64)
    return data[:, :3]


def _pcd_points(path):
    with open(path, 'rb') as f:
        head, got = [], False
        while not got:
            line = f.readline().decode('ascii', 'ignore').strip()
            head.append(line)
            if line.startswith('POINTS'):
                n = int(line.split()[-1])
            if line.startswith('DATA'):
                fmt = line.split()[-1]; got = True
        # 注意 [1:]: FIELDS/TYPE/SIZE 行首是关键字本身, 旧写法 [2:] 会丢掉第一个字段名
        cols = next((l.split()[1:] for l in head if l.startswith('FIELDS')), ['x', 'y', 'z'])
        types = next((l.split()[1:] for l in head if l.startswith('TYPE')), None)
        sizes = next((l.split()[1:] for l in head if l.startswith('SIZE')), None)
        xi, yi, zi = cols.index('x'), cols.index('y'), cols.index('z')
        if fmt == 'ascii':
            data = np.loadtxt(open(path, 'rb'), skiprows=len(head), max_rows=n, ndmin=2)
        else:
            # PCD 按 TYPE×SIZE 组合映射 dtype(F浮点/I有符号/U无符号);
            # 旧写法 'F'+'8' 会拼出非法 <f48 导致 numpy 拒绝解析
            _pcd_dt = {('F', '2'): 'f2', ('F', '4'): 'f4', ('F', '8'): 'f8',
                       ('I', '1'): 'i1', ('I', '2'): 'i2', ('I', '4'): 'i4', ('I', '8'): 'i8',
                       ('U', '1'): 'u1', ('U', '2'): 'u2', ('U', '4'): 'u4', ('U', '8'): 'u8'}
            dt = []
            for name, t, s in zip(cols, types, sizes):
                if (t, s) not in _pcd_dt:
                    raise ValueError(f'暂不支持的 PCD 字段类型 TYPE {t} SIZE {s}')
                dt.append((name, '<' + _pcd_dt[(t, s)]))
            raw = np.frombuffer(f.read(np.dtype(dt).itemsize * n), dtype=np.dtype(dt))
            data = np.column_stack([raw[cols[xi]], raw[cols[yi]], raw[cols[zi]]]).astype(np.float64)
    return data[:, :3]


def load_cloud(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.ply':
        pts = _ply_points(path)
    elif ext == '.pcd':
        pts = _pcd_points(path)
    elif ext == '.npy':
        pts = np.load(path, allow_pickle=False).astype(np.float64)[:, :3]
    else:  # txt / csv
        pts = np.loadtxt(path, delimiter=',' if ext == '.csv' else None,
                         max_rows=2_000_000, ndmin=2)[:, :3]
    pts = pts[np.isfinite(pts).all(1)]
    return pts


# ---------------- 墙面拟合与障碍 ----------------
def analyze(pts, dist_nominal=None, rib_depth=0.05, inset=0.02):
    warns = []
    from scipy import ndimage
    if len(pts) < 500:
        raise ValueError(f'点云过少({len(pts)}点), 需>=500')
    if len(pts) > 500_000:                       # 大云子采样加速
        idx = np.random.default_rng(0).choice(len(pts), 500_000, replace=False)
        pts = pts[idx]

    # 1) 墙面 = x 直方图最强峰(机器人面向+x), 取峰两侧 2cm 板内点
    xs = pts[:, 0]
    bins = np.arange(xs.min(), xs.max() + 0.01, 0.01)
    hist, edges = np.histogram(xs, bins=bins)
    k = int(np.argmax(hist))
    x_peak = 0.5 * (edges[k] + edges[k + 1])
    inw = np.abs(xs - x_peak) <= 0.02
    if inw.sum() < 0.10 * len(pts):
        warns.append(f'墙面内点占比仅{inw.mean():.0%}(<10%), 点云可能未对准墙或含大量杂点')
    if inw.sum() < 100:
        raise ValueError('未找到墙面平面(x向强峰), 请确认点云为机器人系(x前,y左,z上)')
    wall = pts[inw]
    distance = float(wall[:, 0].mean())
    if dist_nominal is not None and abs(distance - dist_nominal) > 0.10:
        warns.append(f'点云实测墙面距离 {distance:.3f}m 与输入 {dist_nominal:.3f}m 相差>10cm')

    # 2) 墙面范围 (0.5% 分位去离群)
    u_lo, u_hi = np.percentile(wall[:, 1], [0.5, 99.5])
    h_lo, h_hi = np.percentile(wall[:, 2], [0.5, 99.5])
    wall_u, wall_h = [float(u_lo), float(u_hi)], [float(h_lo), float(h_hi)]

    # 3) 凸出障碍: 比墙面深 rib_depth 以上的点 -> 5cm 体素连通域
    pro = pts[pts[:, 0] < distance - rib_depth]
    obstacles = []
    if len(pro) > 50:
        v = 0.05
        ijk = np.floor((pro - [pro[:, 0].min(), pro[:, 1].min(), pro[:, 2].min()]) / v).astype(int)
        ijk = np.clip(ijk, 0, 3000)
        dims = ijk.max(0) + 1
        occ = np.zeros(dims, dtype=bool)
        occ[tuple(ijk.T)] = True
        from scipy import ndimage
        lab, nl = ndimage.label(occ, structure=np.ones((3, 3, 3)))
        for ci in range(1, nl + 1):
            sel = lab[tuple(ijk.T)] == ci
            if sel.sum() < 30:
                continue
            q = pro[sel]
            u0, u1 = q[:, 1].min(), q[:, 1].max()
            h0, h1 = q[:, 2].min(), q[:, 2].max()
            depth = float(distance - q[:, 0].min())
            us, hs = u1 - u0, h1 - h0
            kind = ('rib(立筋)' if hs > 0.8 and us < 0.5 else
                    'beam/pipe(横管)' if us > 0.5 and hs < 0.5 else 'protrusion(凸出物)')
            obstacles.append(dict(kind=kind, u=[float(u0), float(u1)],
                                  h=[float(h0), float(h1)], depth=round(depth, 3)))

    # 4) 窗口开口 = 墙面内点的空洞 (内部大空连通域); 管道等凸出物 -> 03 标绿穿越段
    cell = 0.025        # 2.5cm 格: 5cm 格下开口边缘格被窗框点占用, 0.2x0.4m 开口仅剩 3x7=21 格,
                        # 既达不到面积阈值又使框偏小; 细格后框偏差 <=1 格(2.5cm)
    gu = np.arange(u_lo, u_hi + cell, cell)
    gh = np.arange(h_lo, h_hi + cell, cell)
    occ = np.zeros((len(gh), len(gu)), bool)
    occ[np.clip(np.searchsorted(gh, wall[:, 2]) - 1, 0, len(gh) - 1),
        np.clip(np.searchsorted(gu, wall[:, 1]) - 1, 0, len(gu) - 1)] = True
    from scipy import ndimage
    lab, nl = ndimage.label(~occ)                # 直接标空连通域, 不做 closing(避免开口与外部空域粘连导致框变胖)
    window = None
    best = 0
    for ci in range(1, nl + 1):
        sel = lab == ci
        if sel.sum() < 36:                       # 面积阈值 >=36 格
            continue
        ih, iu = np.where(sel)
        if (iu.min() == 0 or iu.max() == len(gu) - 1
                or ih.min() == 0 or ih.max() == len(gh) - 1):
            continue                             # 与 extent 边界格相邻的是外部空域, 不算内部开口
        if sel.sum() <= best:                    # 取最大的内部空连通域
            continue
        best = int(sel.sum())
        # 框 = 占用格单元索引的 min/max; 右/上边界 +1 后夹取, 避免索引越界回绕成负数
        b_u = [float(gu[iu.min()]), float(gu[min(int(iu.max()) + 1, len(gu) - 1)])]
        b_h = [float(gh[ih.min()]), float(gh[min(int(ih.max()) + 1, len(gh) - 1)])]
        window = dict(kind='window(开口)', u=b_u, h=b_h, depth=0.0)

    # 5) stroke_u: 立筋(全高凸出)与舱室默认一致 -> 截在筋面内侧; 其余障碍穿越通过
    pad = 0.02
    u_a, u_b = u_lo + inset, u_hi - inset
    ribs = [o for o in obstacles if o['kind'].startswith('rib') and o['depth'] > rib_depth]
    for o in ribs:
        cu = 0.5 * (o['u'][0] + o['u'][1])
        if cu > 0.5 * (u_lo + u_hi):             # 右侧筋 -> 左边界截到筋左面
            u_b = min(u_b, o['u'][0] - pad)
        else:
            u_a = max(u_a, o['u'][1] + pad)
    if u_b - u_a < 0.15:
        raise ValueError('剔除立筋后墙面无≥0.15m可用横向区间')
    stroke_u = [float(u_a), float(u_b)]
    if obstacles:
        warns.append(f"凸出障碍 {len(obstacles)} 处: "
                     + '; '.join(f"{o['kind']} u{o['u']} h{o['h']} 深{o['depth']}m"
                                 for o in obstacles[:6]))

    out = dict(distance=round(distance, 4), wall_u=wall_u, wall_h=wall_h,
               stroke_u=stroke_u, h_top=float(min(1.45, wall_h[1] - 0.05)),
               obstacles=obstacles + ([window] if window else []),
               warns=warns, n_pts=int(len(pts)))
    return out


# ---------------- 报告图 ----------------
def plot_report(pts, rep, png):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import setup_cn_font
    setup_cn_font()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    if len(pts) > 200_000:
        pts = pts[np.random.default_rng(1).choice(len(pts), 200_000, replace=False)]
    dist = rep['distance']
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.2), dpi=115)
    fig.patch.set_facecolor('#101216')

    # ---- 左: 侧视剖面 (机器人->墙, 看"墙在哪/什么凸出来") ----
    ax = axs[0]
    ax.set_facecolor('#16181d')
    ax.scatter(pts[:, 0], pts[:, 2], s=1.2, c=pts[:, 1], cmap='coolwarm', alpha=.55, lw=0)
    ax.axvline(dist, color='#ff5555', lw=2.2, label=f'墙面拟合平面 x={dist:.3f}m')
    ax.axvline(0, color='#4f8ef7', lw=2, label='机器人站位 x=0')
    ax.axvline(dist - 0.40, color='#f5a623', lw=1.6, ls='--', label='腕心平面 (距墙0.40m)')
    for o in rep['obstacles']:
        if o['depth'] <= 0.02:
            continue
        xm = dist - o['depth']
        ax.annotate(f"{o['kind']} 深{o['depth']:.2f}m",
                    xy=(xm, float(np.clip(np.median([o['h'][0], o['h'][1]]), 0, 1.45))),
                    xytext=(xm - 0.30, min(o['h'][0] + 0.55, 1.30)),
                    color='#ffb347', fontsize=10, weight='bold',
                    arrowprops=dict(arrowstyle='->', color='#ffb347'))
    ax.set_xlabel('x 前向 (m)', color='#e8edf3', fontsize=12)
    ax.set_ylabel('h 高度 (m)', color='#e8edf3', fontsize=12)
    ax.set_title('侧视剖面: 墙面距离与凸出物', color='#e8edf3', fontsize=14, weight='bold', pad=10)
    ax.legend(facecolor='#1d2026', edgecolor='#33363e', labelcolor='#e8edf3',
              fontsize=10, loc='lower right')
    ax.tick_params(colors='#a8b3c1')
    for sp in ax.spines.values():
        sp.set_color('#33363e')

    # ---- 右: 正视 (u,h) 按深度着色, 喷涂带+障碍框 ----
    ax = axs[1]
    ax.set_facecolor('#16181d')
    sc = ax.scatter(pts[:, 1], pts[:, 2], s=1.2, c=dist - pts[:, 0],
                    cmap='inferno', alpha=.75, lw=0)
    cb = fig.colorbar(sc, ax=ax, pad=0.015, fraction=0.045)
    cb.set_label('凸出深度 (m)', color='#e8edf3', fontsize=11)
    cb.ax.yaxis.set_tick_params(color='#a8b3c1')
    cb.outline.set_edgecolor('#33363e')
    a, b = rep['stroke_u']
    ax.axvspan(a, b, color='#29d391', alpha=0.10,
               label=f"喷涂带 [{a:.3f},{b:.3f}] (等宽之字落位)")
    for o in rep['obstacles']:
        ax.add_patch(plt.Rectangle((o['u'][0], o['h'][0]), o['u'][1] - o['u'][0],
                                   o['h'][1] - o['h'][0], fill=False, ec='#ffb347', lw=1.8))
        ax.text(o['u'][0], o['h'][1] + 0.02, f"{o['kind']}", color='#ffb347',
                fontsize=10, weight='bold')
    ax.set_xlabel('u 沿墙横向 (m)', color='#e8edf3', fontsize=12)
    ax.set_ylabel('h 高度 (m)', color='#e8edf3', fontsize=12)
    ax.set_title('正视: 喷涂带落位与障碍检测', color='#e8edf3', fontsize=14, weight='bold', pad=10)
    ax.legend(facecolor='#1d2026', edgecolor='#33363e', labelcolor='#e8edf3',
              fontsize=10, loc='lower left')
    ax.tick_params(colors='#a8b3c1')
    for sp in ax.spines.values():
        sp.set_color('#33363e')

    fig.suptitle(f"点云检测报告 — 实测墙面距离 {dist:.3f} m · {len(pts)} 点",
                 color='#f2f5f9', fontsize=16, weight='bold', y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(png, dpi=115, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cloud', required=True)
    ap.add_argument('--dist', type=float, default=None)
    ap.add_argument('--out', default=os.path.join(BASE, 'output', 'cloud_report.json'))
    ap.add_argument('--png', default=None)
    a = ap.parse_args()
    pts = load_cloud(a.cloud)
    rep = analyze(pts, a.dist)
    rep['cloud_file'] = os.path.abspath(a.cloud)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rep, open(a.out, 'w', encoding='utf8'), ensure_ascii=False, indent=1)
    png = a.png or a.out.replace('.json', '.png')
    plot_report(pts, rep, png)
    rep['png'] = png
    print(json.dumps({k: v for k, v in rep.items() if k != 'png'},
                     ensure_ascii=False, indent=1))
    print(f"[点云] 报告 {a.out}\n[点云] 图 {png}")


if __name__ == '__main__':
    main()
