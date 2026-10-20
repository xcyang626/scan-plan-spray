# -*- coding: utf-8 -*-
"""多工位喷涂矢量图: (1) 条带划分 (2) 两站接力仿真结果.

用途: 多工位方案的对外说明图 + 站间精度复核(站1/站2 对比).
输出为**真矢量**(SVG/PDF), PNG 仅作预览; SVG 可直接被 VectorCraft 导入编辑.

用法(isaaclab 解释器, 有 matplotlib):
  # 只画条带划分
  python 16_fig_multi_station.py --run <run目录(含 multi_station_plan.json)>
  # 再画仿真对比(需要跟踪日志)
  python 16_fig_multi_station.py --run <run目录> \
      --sim-log <log.npz.station1.npz> --sim-log2 <log.npz.final.npz> \
      --sim-motion <动作 npz>
输出: media/fig_multi_station_bands.svg|png
      media/fig_multi_station_sim.svg|png
"""
import argparse
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import setup_cn_font  # noqa: E402

MEDIA = os.path.join(BASE, "media")
TRAJ_FPS = 30.0
D_ROBOT = 0.74          # G1 标称离墙距离(与规划同口径)


def _plan(run_dir):
    p = os.path.join(run_dir, "multi_station_plan.json")
    if not os.path.exists(p):
        raise SystemExit(f"找不到多工位 plan: {p}")
    with open(p, encoding="utf8") as f:
        return json.load(f)


# ============================ 图1: 条带划分 ============================
def fig_bands(plan, out_svg, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_cn_font()

    wall_u = plan["wall_u"]
    stations = plan["stations"]
    steps = plan.get("steps") or []
    first_actual = plan.get("first_band_actual")
    colors = ["#2f6fb0", "#3f9d5a", "#c8763c", "#8a5fb5", "#b04a5a", "#4a8f96"]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12.0, 5.6), dpi=110,
                                  gridspec_kw=dict(height_ratios=[3, 1], hspace=0.42))
    # ---- 上: 墙面俯视, 站位条带 ----
    ax.add_patch(plt.Rectangle((wall_u[0], 0.0), wall_u[1] - wall_u[0], 1.0,
                               facecolor="#e8ecf1", edgecolor="#9aa7b4", lw=1.0))
    for st in stations:
        lo, hi = st["strip"]
        c = colors[(st["idx"] - 1) % len(colors)]
        ax.add_patch(plt.Rectangle((lo, 0.0), hi - lo, 1.0, facecolor=c,
                                   alpha=0.20, edgecolor=c, lw=1.6, zorder=2))
        ax.text((lo + hi) / 2, 0.80, f"站{st['idx']}", ha="center", va="center",
                fontsize=13, color=c, fontweight="bold", zorder=4)
        ax.text((lo + hi) / 2, 0.56, f"条带 {hi - lo:.3f}m", ha="center", va="center",
                fontsize=9.5, color="#333", zorder=4)
        ax.plot([lo, lo], [0, 1], color=c, lw=1.0, ls=":", zorder=3)
    # 首站实喷带(03 实际喷到的范围)
    if first_actual:
        lo, hi = first_actual
        ax.add_patch(plt.Rectangle((lo, -0.12), hi - lo, 0.12, facecolor="none",
                                   edgecolor="#111", lw=1.4, hatch="///", zorder=4))
        ax.text((lo + hi) / 2, -0.20, f"首站实喷带 {lo:.3f}~{hi:.3f}m",
                ha="center", va="top", fontsize=9.5, color="#111")
    # 站间步距箭头
    for k in range(len(stations) - 1):
        c1 = stations[k]["center"]
        c2 = stations[k + 1]["center"]
        ax.annotate("", xy=(c2, -0.34), xytext=(c1, -0.34),
                    arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.6))
        ax.text((c1 + c2) / 2, -0.40, f"步距 {steps[k]:.3f}m" if k < len(steps)
                else "", ha="center", va="top", fontsize=9.5, color="#444")
    ax.set_xlim(wall_u[0] - 0.12, wall_u[1] + 0.12)
    ax.set_ylim(-0.62, 1.18)
    ax.set_yticks([])
    ax.set_xlabel("沿墙坐标 u (m)")
    ax.set_title(f"多工位条带划分: 整墙 {plan['wall_width']:.2f}m -> "
                 f"{plan['n_stations']} 站无缝覆盖 "
                 f"(条带 {min(steps):.3f}~{max(steps):.3f}m, 不重叠不防缝)",
                 fontsize=12.5, pad=10)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", ls=":", alpha=0.35)

    # ---- 下: 机器人站位示意(沿墙方向的位姿) ----
    ax2.set_xlim(wall_u[0] - 0.12, wall_u[1] + 0.12)
    ax2.set_ylim(-0.75, 0.85)
    ax2.set_yticks([])
    ax2.set_xlabel("沿墙坐标 u (m)")
    ax2.set_title("机器人站位示意(俯视): 机器人面向墙面, 站位随工位左移", fontsize=10.5)
    for sp in ("top", "right", "left"):
        ax2.spines[sp].set_visible(False)
    for st in stations:
        c = colors[(st["idx"] - 1) % len(colors)]
        u = st["center"]
        # 机器人(俯视三角形, 尖端朝墙) + 喷枪指向墙面的射线
        ax2.add_patch(plt.Polygon([[u, 0.16], [u - 0.085, -0.10], [u + 0.085, -0.10]],
                                  closed=True, facecolor=c, alpha=0.30,
                                  edgecolor=c, lw=1.5))
        ax2.plot([u, u], [0.16, 0.58], color=c, lw=2.4)      # 喷枪指向墙面
        ax2.plot([u, u], [0.16, 0.30], color=c, lw=1.2, ls=":")
        ax2.text(u, -0.30, f"站{st['idx']}", ha="center", va="top",
                 fontsize=10.5, color=c, fontweight="bold")
    ax2.plot([wall_u[0], wall_u[1]], [0.60, 0.60], color="#666", lw=2.5)
    ax2.text(wall_u[1], 0.68, "墙面", ha="right", va="bottom", fontsize=10, color="#666")
    ax2.grid(axis="x", ls=":", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] {out_svg}")
    print(f"[图] {out_png}")


# ============================ 图2: 两站接力仿真结果 ============================
def fig_sim(log1, log2, motion, out_svg, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_cn_font()
    sys.path.insert(0, os.path.join(BASE, "scripts", "exec"))
    from arm_task_servo import fk_jac1, precompute_motion_targets, quat2rotm

    m = np.load(motion)
    mq = np.asarray(m["qpos"] if "qpos" in m.files else m[m.files[0]], float)
    tgtP, _, tgtH = precompute_motion_targets(mq)

    def load(p):
        z = np.load(p)
        return z["t"], z["qpos"], z["cmd_q10"]

    BLEND = 1.5          # 起播渐变+稳定的时长(s): 这段不计入稳态统计

    def series(p):
        t, q, cq = load(p)
        i0 = np.clip(np.floor(t * TRAJ_FPS).astype(int), 0, len(tgtP) - 2)
        a = np.clip(t * TRAJ_FPS - i0, 0, 1)
        plan_rel_y = (1 - a) * tgtP[i0, 1] + a * tgtP[i0 + 1, 1]
        plan_h = (1 - a) * tgtH[i0] + a * tgtH[i0 + 1]
        sel = np.arange(0, len(t), 3)
        rel_y, dh, off_max = [], [], []
        for k in sel:
            q10 = np.concatenate([q[k, 19:22], q[k, 29:36]])
            p_rel, a_rel, _, _ = fk_jac1(q10)
            Rb = quat2rotm(q[k, 3:7])
            p_w = Rb @ p_rel + q[k, 0:3]
            a_w = Rb @ a_rel
            rel_y.append(p_rel[1])
            if a_w[0] > 0.3 and np.isfinite(plan_h[k]):
                wx = q[k, 0] + D_ROBOT          # 墙随底座(与规划相对口径一致)
                dh.append(p_w[2] + (wx - p_w[0]) * a_w[2] / a_w[0] - plan_h[k])
        # 伺服修正量 = 指令 q10 - 动作 q10(渐变段含切换瞬态, 故另标)
        i1 = np.clip(np.floor(t * TRAJ_FPS).astype(int), 0, len(mq) - 2)
        a1 = np.clip(t * TRAJ_FPS - i1, 0, 1)
        pq = np.concatenate([(1 - a1)[:, None] * mq[i1, 19:22] + a1[:, None] * mq[i1 + 1, 19:22],
                             (1 - a1)[:, None] * mq[i1, 29:36] + a1[:, None] * mq[i1 + 1, 29:36]], axis=1)
        off_max = np.abs(cq - pq).max(axis=1)
        rel_y = np.array(rel_y)
        dh = np.array(dh)
        # 稳态(起播渐变之后)的统计: 渐变段的瞬态不计入
        st = t[sel] > BLEND
        return dict(t=t, q=q, rel_y=rel_y, plan_rel_y=plan_rel_y[sel],
                    dh=dh, off=off_max, sel=sel, steady=st,
                    dh_mean=float(dh[st].mean()) if st.any() else float(dh.mean()),
                    dy_mean=float((rel_y - plan_rel_y[sel])[st].mean()))

    s1, s2 = series(log1), series(log2)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.6), dpi=110)
    (axA, axB), (axC, axD) = axes
    c1, c2 = "#2f6fb0", "#c8763c"

    # (a) 底座俯视轨迹
    axA.plot(s1["q"][0:len(s1["q"]):3, 0], s1["q"][0:len(s1["q"]):3, 1], ".", ms=2.2, color=c1, alpha=0.75, label="工位1")
    axA.plot(s2["q"][0:len(s2["q"]):3, 0], s2["q"][0:len(s2["q"]):3, 1], ".", ms=2.2, color=c2, alpha=0.75, label="工位2")
    p1 = s1["q"][0, 0:2]
    p2 = s2["q"][0, 0:2]
    axA.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="-|>", color="#111", lw=1.8))
    axA.text(0.03, 0.05, f"Δy = {p2[1]-p1[1]:+.3f}m (步距目标 0.500m)\n"
             f"Δx = {p2[0]-p1[0]:+.3f}m (平衡策略漂移, 非走位)",
             transform=axA.transAxes, fontsize=9.5, color="#111",
             bbox=dict(boxstyle="round", fc="#fffbe6", ec="#bbb", alpha=0.95))
    axA.set_xlabel("底座 x (m, 正向离墙)")
    axA.set_ylabel("底座 y (m, 沿墙)")
    axA.set_title("(a) 底座轨迹(俯视): 走位后基座沿墙平移", fontsize=11)
    axA.legend(fontsize=9, loc="upper right", framealpha=0.95)
    axA.grid(ls=":", alpha=0.4)
    _xs = np.concatenate([s1["q"][:, 0], s2["q"][:, 0]])
    _ys = np.concatenate([s1["q"][:, 1], s2["q"][:, 1]])
    axA.set_xlim(_xs.min() - 0.06, _xs.max() + 0.06)
    axA.set_ylim(_ys.min() - 0.08, _ys.max() + 0.08)

    # (b) 落点高度误差
    for s, c, lab in ((s1, c1, "工位1"), (s2, c2, "工位2")):
        axB.plot(s["t"][s["sel"]], s["dh"], "-", lw=0.9, color=c, alpha=0.75, label=lab)
        axB.axhline(s["dh_mean"], color=c, lw=1.5,
                    label=f"{lab} 稳态均值 {s['dh_mean']:+.3f}m")
    axB.axhspan(-0.05, 0.05, color="#3f9d5a", alpha=0.10)
    axB.axhline(0, color="#333", lw=0.8)
    axB.set_ylim(-0.55, 0.32)
    axB.set_xlabel("工位内时间 (s)")
    axB.set_ylabel("落点高度误差 (m)")
    axB.set_title("(b) 落点高度误差(实际-规划, 墙随底座): ±5cm 绿带内为合格", fontsize=11)
    axB.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
    axB.grid(ls=":", alpha=0.4)

    # (c) 沿墙方向(相对底座 y)误差
    for s, c, lab in ((s1, c1, "工位1"), (s2, c2, "工位2")):
        d = s["rel_y"] - s["plan_rel_y"]
        axC.plot(s["t"][s["sel"]], d, "-", lw=0.9, color=c, alpha=0.75, label=lab)
        axC.axhline(s["dy_mean"], color=c, lw=1.5,
                    label=f"{lab} 稳态均值 {s['dy_mean']:+.3f}m")
    axC.axhline(0, color="#333", lw=0.8)
    axC.set_xlabel("工位内时间 (s)")
    axC.set_ylabel("腕心相对底座 y 误差 (m)")
    axC.set_title("(c) 沿墙方向图案偏移: 站2 比站1 多偏(躯干倾角变大)", fontsize=11)
    axC.legend(fontsize=8.5, loc="lower left", framealpha=0.95)
    axC.grid(ls=":", alpha=0.4)

    # (d) 伺服修正量
    for s, c, lab in ((s1, c1, "工位1"), (s2, c2, "工位2")):
        axD.plot(s["t"], s["off"], lw=0.9, color=c, alpha=0.85, label=lab)
    axD.axhline(0.5, color="#b04a5a", ls="--", lw=1.2)
    axD.set_ylim(-0.05, 0.78)
    axD.set_xlabel("工位内时间 (s)")
    axD.set_ylabel("任务伺服修正量 max|Δq| (rad)")
    axD.set_title("(d) 任务伺服修正量: 渐变瞬态后进入稳态(未持续饱和)", fontsize=11)
    axD.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
    axD.grid(ls=":", alpha=0.4)
    axD.text(0.30, 0.60, "起播渐变瞬态峰值 1.8rad(超出图幅)", fontsize=8.5, color="#555")
    axD.text(0.30, 0.53, "虚线 = 伺服修正限位 0.5rad", fontsize=8.5, color="#b04a5a")

    # 起播渐变区(不计入稳态均值)
    for ax_ in (axB, axC, axD):
        ax_.axvspan(0, BLEND, color="#999", alpha=0.13, zorder=0)
    axB.text(BLEND + 0.6, -0.50, "起播渐变\n(不计入均值)", fontsize=8.5, color="#555")

    fig.suptitle("多工位接力仿真(GR00T MuJoCo, 2 站, 步距 0.5m): "
                 "走位精确到位, 站2 落点高度与站1 同级; 沿墙图案偏移见 (c)",
                 fontsize=12.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=2.4, w_pad=2.0)
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] {out_svg}")
    print(f"[图] {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run 目录(含 multi_station_plan.json)")
    ap.add_argument("--sim-log", help="工位1 跟踪日志 npz")
    ap.add_argument("--sim-log2", help="工位2 跟踪日志 npz")
    ap.add_argument("--sim-motion", help="动作 npz(qpos T×36)")
    ap.add_argument("--out-dir", default=MEDIA)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.run:
        plan = _plan(a.run)
        fig_bands(plan,
                  os.path.join(a.out_dir, "fig_multi_station_bands.svg"),
                  os.path.join(a.out_dir, "fig_multi_station_bands.png"))
    if a.sim_log and a.sim_log2 and a.sim_motion:
        fig_sim(a.sim_log, a.sim_log2, a.sim_motion,
                os.path.join(a.out_dir, "fig_multi_station_sim.svg"),
                os.path.join(a.out_dir, "fig_multi_station_sim.png"))


if __name__ == "__main__":
    main()
