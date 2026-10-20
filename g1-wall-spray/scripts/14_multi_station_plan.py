# -*- coding: utf-8 -*-
"""步骤14: 多工位全覆盖规划(2026-10-15 旁路新增; 当日 V3 无缝定稿).

把"双工位接力"推广为 N 工位, 目标 = **尽量少的点位喷完一整面点云墙**(用户
2026-10-15 定稿): 条带**无缝拼接**、每站喷**自己条带的宽度**(带宽独立贴合,
不重叠; 防缝是以后出了问题再考虑的事), 单站宽度 >= 0.30m。

    w_max = 最大等宽带(阶段A 整墙求得)
    N     = ceil(W / w_max)       最少站数(无缝下的数学下界)
    s     = W / N                 每站条带宽 = 该站动作带宽(单一动作)
    首站规划带 = 条带±0.045 外扩(补偿 03 两端内收 0.04+5mm), 实喷≈条带宽

站 i 条带 [U0+(i-1)s, U0+i·s]; 执行 = 同一动作 + 每站左移 s
(runner walk_list 兼容逐站步距)。障碍(窗洞/凸条)在行内由 03/11 口径处理,
条带划分以整墙 wall_u 为准。

用法(isaaclab 解释器, 站点自动调用):
  python 14_multi_station_plan.py --run <run_dir> [--smin 0.30] [--fig-only] [--update]
输入: <run>/cfg.json + trajectory_meta.json (+ cloud_report.json, 有则画点云背景)
输出: <run>/multi_station_plan.json + fig_multi_station.png(全景规划图)
"""
import os, sys, json, math, argparse
import importlib.util

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STRIP_COLORS = ["#2f6fdb", "#e07b2a", "#2fa36b", "#c23a7a", "#8a6ad2",
                "#0f9bb0", "#b0922e", "#d24f4f", "#5f8f3a", "#7a5fd0",
                "#35bfc0", "#c46a2e"]

# 实机 relay(exec/groot_daemon.py)接受的步距区间: 0.05 < d < 1.5
STEP_MIN, STEP_MAX = 0.05, 1.5


def _band_pair(su, name):
    """[u0,u1] 校验: 必须是两个数字, 否则给清晰报错(不要抛裸 TypeError)."""
    if su is None:
        raise ValueError(f"{name} 缺失 —— 需要 [u0,u1] 两个数字")
    try:
        lo, hi = float(su[0]), float(su[1])
    except (TypeError, IndexError, KeyError, ValueError):
        raise ValueError(f"{name} 格式不对: {su!r} —— 需要 [u0,u1] 两个数字")
    return lo, hi


def _check_step(d, what="步距"):
    """步距范围校验(实机 relay 口径: 0.05 < d < 1.5)."""
    if not (STEP_MIN < d < STEP_MAX):
        raise ValueError(
            f"{what} {d:.3f}m 超出实机允许区间 ({STEP_MIN}~{STEP_MAX}m; "
            f"exec/groot_daemon.py 拒绝该步距) —— 请检查墙宽与站数划分")
    return d


def _load_json(path):
    with open(path, encoding="utf8") as f:
        return json.load(f)


def _save_json(path, obj):
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def _load_cloud_pts(cloud_path):
    """借 11_cloud_scan 的加载器读点云(支持 ply/pcd 等); 失败返回 None."""
    try:
        spec = importlib.util.spec_from_file_location(
            "C11", os.path.join(BASE, "scripts", "11_cloud_scan.py"))
        c11 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(c11)
        return c11.load_cloud(cloud_path)
    except Exception:
        return None


def plan_strips(wall_u, w_band, s_min=0.30):
    """无缝条带划分(用户定稿 2026-10-15 V3: 不重叠、不防缝 —— 防缝是以后
    出了问题再考虑的事; 每站喷**自己条带的宽度**, 带宽各自独立贴合).

    模型: 整墙 [U0,U1] 切成 N 段互不重叠的条带, 每段宽度 = 该站动作带宽:
      每站 <= w_max(最大等宽带)  ->  N = ceil(W / w_max) 为最少站数
      均分 s = W/N 即达到该下界, 且每站条带宽 = 动作带宽 = s(单一动作)
      校验 s >= 0.30(用户下限); 墙过窄导致无解时明确报错。
    返回 (N, steps[], stations[])。"""
    if wall_u is None:
        raise ValueError("墙范围 wall_u 缺失 —— 请重新上传点云跑 11_cloud_scan.py")
    try:
        u0, u1 = float(wall_u[0]), float(wall_u[1])
    except (TypeError, IndexError, KeyError, ValueError):
        raise ValueError(f"墙范围 wall_u 格式不对: {wall_u!r} —— 需要 [u0,u1] 两个数字")
    W = u1 - u0
    if W <= 0:
        raise ValueError(f"墙宽异常: wall_u=[{u0:.3f},{u1:.3f}] 宽 {W:.3f}m <= 0 —— "
                         f"请检查点云/11_cloud_scan 的墙面范围")
    if w_band is None:
        raise ValueError("最大等宽带缺失 —— 03 未成功或 trajectory_meta 无 "
                         "uniform_stroke_u")
    w_band = float(w_band)
    if not (w_band > 0):
        raise ValueError(f"最大等宽带异常: {w_band:.3f}m <= 0 —— 03 未成功或 "
                         f"trajectory_meta 的 uniform_stroke_u 不对, 请重跑 03")
    if W <= w_band + 1e-9:                    # 单站覆盖
        steps = [round(W, 4)]
    else:
        n = max(2, math.ceil(W / w_band - 1e-9))
        s = W / n
        if s < s_min - 1e-9:
            raise ValueError(
                f"无缝划分无解: 墙宽 {W:.3f}m 按最大带宽 {w_band:.3f}m 需 {n} 站, "
                f"均分条带 {s:.3f}m < 单站最小 {s_min}m —— 请上传更宽的墙, "
                f"或放宽腰限增大带宽(当前该墙宽下无缝与单站>= {s_min} 不可兼得)")
        steps = [round(s, 4)] * n
        steps[-1] = round(W - sum(steps[:-1]), 4)      # 数值闭合(吸收舍入)
    if min(steps) < s_min - 5e-3:
        raise ValueError(f"条带宽 {min(steps):.3f}m < 单站最小 {s_min}m")
    for s in steps:                           # 实机 relay 步距区间保护
        _check_step(s, "条带宽度(=逐站步距)")
    stations, acc = [], u0
    for i, s in enumerate(steps):
        stations.append(dict(
            idx=i + 1,
            strip=[round(acc, 4), round(acc + s, 4)],   # 该站喷涂范围=条带(墙 u 坐标)
            center=round(acc + s / 2, 4)))
        acc += s
    return len(steps), steps, stations


def make_plan(run_dir, s_min=0.30):
    cfg = _load_json(os.path.join(run_dir, "cfg.json"))
    meta = _load_json(os.path.join(run_dir, "trajectory_meta.json"))
    su = meta.get("uniform_stroke_u") or meta.get("stroke_u")
    if not su:
        raise ValueError("trajectory_meta 缺 uniform_stroke_u —— 先跑 03")
    su_lo, su_hi = _band_pair(su, "trajectory_meta.uniform_stroke_u")
    w_band = su_hi - su_lo
    if not (w_band > 0):
        raise ValueError(f"最大等宽带异常: uniform_stroke_u=[{su_lo:.3f},{su_hi:.3f}] "
                         f"宽 {w_band:.3f}m <= 0 —— 请重跑 03")
    # 墙范围: 有点云用整墙 wall_u(多工位覆盖对象=完整点云)
    rep_p = os.path.join(run_dir, "cloud_report.json")
    if not (os.path.exists(rep_p) and cfg.get("cloud")):
        raise ValueError("多工位全覆盖需要点云(wall_u) —— 请上传点云后重新规划")
    rep = _load_json(rep_p)
    wall_u = rep.get("wall_u") or rep.get("stroke_u")
    if wall_u is None:
        raise ValueError("cloud_report.json 缺 wall_u/stroke_u(整墙横向范围) —— "
                         "请重新上传点云跑 11_cloud_scan.py")
    n, steps, stations = plan_strips(wall_u, w_band, s_min=s_min)
    # 带宽贴合(V3 用户定稿: 每站喷自己条带的宽度, 不重叠不防缝):
    # 首站规划带 = 条带±0.045 外扩(补偿 03 两端内收 0.04 + 5mm 数值余量),
    # 03 内收后实喷 ≈ 条带宽 —— 全景图条带宽 = 之字形图带宽, 两图一致。
    s0 = stations[0]["strip"]
    plan = dict(
        mode="multi-station-fullcover", robot=cfg.get("robot"),
        wall_u=[round(float(wall_u[0]), 4), round(float(wall_u[1]), 4)],
        wall_width=round(float(wall_u[1]) - float(wall_u[0]), 4),
        band_width=round(w_band, 4),          # 最大等宽带(阶段A, 上限参考)
        n_stations=n, steps=[round(s, 4) for s in steps],
        step_s=round(max(steps), 4), s_min=s_min,
        first_plan_band=[round(s0[0] - 0.045, 4), round(s0[1] + 0.045, 4)],
        stations=stations,
        exec_hint=dict(stations=n, walk_list=[round(s, 4) for s in steps[:-1]],
                       # 首站条带(与 stations[0]["strip"] 同口径; 不再写阶段A的
                       # 整墙等宽带 —— 未跑 --update 时消费端拿到的必须是首站带)
                       band=[round(s0[0], 4), round(s0[1], 4)],
                       used_by="仿真+实机(10_station_app.py 的多工位触发都消费本文件)"),
        note=f"最少 {n} 站无缝覆盖整墙 {round(sum(steps),3)}m: 每站喷自己条带 "
             f"{min(steps):.3f}~{max(steps):.3f}m(带宽独立贴合, 不重叠; "
             f"防缝后续按需引入)")
    _save_json(os.path.join(run_dir, "multi_station_plan.json"), plan)
    return plan


def _load_stroke_path(run_dir):
    """读首站之字形走线(wall_path.csv): 返回 [(u,h,kind), ...] 按 seq 保序."""
    p = os.path.join(run_dir, "wall_path.csv")
    if not os.path.exists(p):
        return None
    import csv as _csv
    try:
        with open(p, encoding="utf8") as f:
            rows = list(_csv.DictReader(f))
        return [(float(r["u_m"]), float(r["h_m"]), r["kind"]) for r in rows]
    except Exception:
        return None


def _plot_zigzags(ax, plan, path, h_top):
    """把首站之字形按执行形态平移复制到每个条带(站 k = 首站轨迹 + 累计步距).

    绿实线=喷涂走线(stroke), 淡黄细线=换行/过渡连接段 —— 每站怎么喷一眼可见。"""
    if not path:
        return False
    # 按 kind 连续性切段(相邻点连线; stroke 段与连接段分开画)
    segs, cur = [], [path[0]]
    for a, b in zip(path, path[1:]):
        if a[2] == b[2]:
            cur.append(b)
        else:
            segs.append(cur)
            cur = [b]
    segs.append(cur)
    base0 = plan["stations"][0]["strip"][0]
    for st in plan["stations"]:
        off = st["strip"][0] - base0
        for seg in segs:
            xs = [p[0] + off for p in seg]
            hs = [p[1] for p in seg]
            if seg[0][2] == "stroke":
                ax.plot(xs, hs, color="#1d9e52", lw=1.1, alpha=0.9, zorder=4)
            else:
                ax.plot(xs, hs, color="#d9b23c", lw=0.6, alpha=0.45, zorder=3)
    return True


def plot_plan(run_dir, plan):
    """全景规划图: 点云背景 + N 条带(含各站之字形走线) + 走位箭头链 + 标注."""
    cfg = _load_json(os.path.join(run_dir, "cfg.json"))
    from common import setup_cn_font
    setup_cn_font()
    u0, u1 = plan["wall_u"]
    pts = _load_cloud_pts(cfg["cloud"]) if cfg.get("cloud") else None
    if pts is not None and len(pts) == 0:
        pts = None                      # 空点云等同未提供, 避免下面 h_top 引用未赋值的 p
    fig, ax = plt.subplots(figsize=(13, 5.2), dpi=110)
    if pts is not None and len(pts):
        sel = np.random.default_rng(0).choice(len(pts),
                                              min(len(pts), 20000), replace=False)
        p = np.asarray(pts)[sel]
        ax.scatter(p[:, 1], p[:, 2], s=0.6, c="#9fb6d8", alpha=0.35, lw=0,
                   label=f"点云 ({len(pts)} 点)")
    h_top = 2.0 if pts is None else min(2.0, float(np.percentile(p[:, 2], 99)) + 0.1)
    h_bot = 0.0
    # 条带: 名义边界竖线 + 着色带 + 站号
    for st in plan["stations"]:
        lo, hi = st["strip"]
        c = STRIP_COLORS[(st["idx"] - 1) % len(STRIP_COLORS)]
        ax.axvspan(lo, hi, color=c, alpha=0.14, lw=0)
        ax.axvline(lo, color=c, lw=1.0, ls="--", alpha=0.55)
        ax.text((lo + hi) / 2, h_top * 0.955, f"站{st['idx']}",
                ha="center", va="top", fontsize=12, color=c, fontweight="bold")
        ax.text((lo + hi) / 2, h_top * 0.905,
                f"{hi - lo:.2f}m", ha="center", va="top", fontsize=8.5, color=c)
    ax.axvline(u1, color=STRIP_COLORS[(plan["n_stations"] - 1) % len(STRIP_COLORS)],
               lw=1.0, ls="--", alpha=0.55)
    # 各站之字形走线(执行真实形态: 首站轨迹平移复制; 绿=喷涂行, 淡黄=换行连接)
    _has_zz = _plot_zigzags(ax, plan, _load_stroke_path(run_dir), h_top)
    # 障碍框(11 检出)
    rep_p = os.path.join(run_dir, "cloud_report.json")
    if os.path.exists(rep_p):
        rep = _load_json(rep_p)
        for o in rep.get("obstacles", []):
            ou, oh = o["u"], o["h"]
            ax.add_patch(plt.Rectangle((ou[0], oh[0]), ou[1] - ou[0], oh[1] - oh[0],
                                       fill=False, ec="#e5484d", lw=1.2, ls=":"))
    # 走位箭头链(墙脚下方): 站中心 → 下一站中心
    yw = -0.16
    for a, b in zip(plan["stations"], plan["stations"][1:]):
        ax.annotate("", xy=(b["center"], yw), xytext=(a["center"], yw),
                    arrowprops=dict(arrowstyle="-|>", color="#0d99f2", lw=2.0))
    ax.scatter([st["center"] for st in plan["stations"]],
               [yw] * len(plan["stations"]), marker="s", s=46, c="#0d99f2",
               zorder=5)
    ax.text(plan["stations"][0]["center"], yw - 0.10,
            f"机器人走位: 站1 → 每站喷完左移(步距=条带宽) → 站{plan['n_stations']}",
            ha="left", fontsize=10, color="#0d99f2")
    ax.set_xlim(u0 - 0.25, u1 + 0.25)
    ax.set_ylim(-0.42, h_top * 1.02)
    ax.set_xlabel("墙面横向 u (m)")
    ax.set_ylabel("高度 h (m)")
    ax.set_title(f"多工位全覆盖规划 — 最少 {plan['n_stations']} 站无缝覆盖整墙 "
                 f"{plan['wall_width']:.2f}m (每站喷自己条带 "
                 f"{min(plan['steps']):.2f}~{max(plan['steps']):.2f}m, 带宽独立贴合, "
                 f"单站≥{plan.get('s_min', 0.30)}m, 不重叠)")
    ax.grid(alpha=0.18, lw=0.4)
    # 完整图例: 图上每个视觉元素都要有对应项(条带/之字形/连接线/障碍/点云)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    _handles = [Patch(facecolor="#7f9cc9", alpha=0.25, label="工位条带(每站喷涂范围)"),
                Line2D([], [], color="#1d9e52", lw=1.6, label="喷涂之字形走线"),
                Line2D([], [], color="#d9b23c", lw=1.0, alpha=0.7, label="换行/过渡连接段"),
                Line2D([], [], color="#e5484d", lw=1.2, ls=":", label="点云检出障碍")]
    if pts is not None and len(pts):
        _handles.append(Line2D([], [], marker=".", ls="", color="#9fb6d8",
                               alpha=0.6, label=f"点云 ({len(pts)} 点)"))
    ax.legend(handles=_handles, loc="lower right", fontsize=9, framealpha=0.9)
    out = os.path.join(run_dir, "fig_multi_station.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("[预览]", out)
    return out


def update_after_strip_run(run_dir, s_min=0.30):
    """阶段B(首站条带贴合重跑 03)之后: 校验实喷带宽与条带一致.

    无缝模型: 条带外扩 0.045/边, 03 两端内收 0.04/边 -> 设计上实喷带比条带宽
    +0.010m(每边 +0.005); 再叠加 03 把带边界 round 到 3 位小数(<=0.001m).
    故判据取 0.0125(设计余量 + 舍入 + 少量裕度), 用 0.010 会有一半的合法结果
    落在公差外被判失败。

    中心位置判据(2026-10-20 加): 只比宽度不够 —— 阶段A(整墙)与阶段B(首站带)
    的 u 系不同(03 按请求带重锚), 阶段B 方案若喷到墙上别处(甚至墙外)而宽度
    恰好对上, 宽度校验照样放行, 无缝拼接直接报废。故再比中心: 偏差超过
    容差(条带宽的 5%, 且不超过 5cm)报错, 并给出实际中心与期望中心。"""
    plan = _load_json(os.path.join(run_dir, "multi_station_plan.json"))
    meta = _load_json(os.path.join(run_dir, "trajectory_meta.json"))
    su = meta.get("uniform_stroke_u") or meta.get("stroke_u")
    if not su:
        raise ValueError(f"{run_dir}/trajectory_meta.json 缺 uniform_stroke_u/stroke_u, "
                         f"无法校验首站实喷带宽 —— 请先跑完 03")
    su_lo, su_hi = _band_pair(su, "trajectory_meta.uniform_stroke_u")
    w2 = su_hi - su_lo
    s0 = float(plan["steps"][0])
    if abs(w2 - s0) > 0.0125:
        raise ValueError(f"首站实喷带宽 {w2:.3f}m 与条带宽 {s0:.3f}m 偏差 "
                         f"{abs(w2 - s0) * 100:.1f}cm 超过 12.5mm "
                         f"(无缝拼接会被破坏) —— 请放宽腰限/调整站位后重试")
    # 中心位置: 期望=首站条带中心, 实际=阶段B 实喷带中心
    stations = plan.get("stations") or []
    if not stations:
        raise ValueError(f"{run_dir}/multi_station_plan.json 缺 stations —— "
                         f"请先跑 14(阶段A)生成条带划分")
    exp_c = float(stations[0].get("center", 0.5 * (stations[0]["strip"][0]
                                                    + stations[0]["strip"][1])))
    act_c = 0.5 * (su_lo + su_hi)
    tol = min(0.05 * s0, 0.05)               # 条带宽的 5%, 且不超过 5cm
    if abs(act_c - exp_c) > tol:
        raise ValueError(
            f"首站实喷带中心 {act_c:.3f}m 与规划条带中心 {exp_c:.3f}m 偏差 "
            f"{abs(act_c - exp_c) * 100:.1f}cm 超过容差 {tol * 100:.1f}cm —— "
            f"阶段B 的 03 必须按首站条带(first_plan_band="
            f"{plan.get('first_plan_band')})重跑, 否则喷到墙上别处/墙外, "
            f"无缝拼接被破坏")
    plan.update(
        band_width=round(w2, 4),
        first_band_actual=[round(su_lo, 4), round(su_hi, 4)],
        note=f"最少 {plan['n_stations']} 站无缝覆盖整墙 {plan['wall_width']:.2f}m: "
             f"每站喷自己条带 {min(plan['steps']):.3f}~{max(plan['steps']):.3f}m "
             f"(带宽独立贴合, 实喷 {w2:.3f}m, 不重叠; 防缝后续按需引入)")
    plan["exec_hint"]["band"] = plan["first_band_actual"]
    plan["exec_hint"].pop("sim_only", None)     # 旧版产物里的过时标注
    plan["exec_hint"]["used_by"] = "仿真+实机(10_station_app.py 的多工位触发都消费本文件)"
    _save_json(os.path.join(run_dir, "multi_station_plan.json"), plan)
    return plan          # 出图由 main() 统一做, 避免 --update 渲染两遍


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run 目录(含 cfg/trajectory_meta)")
    ap.add_argument("--smin", type=float, default=0.30,
                    help="单站名义喷涂宽度下限(m, 用户规范 >=0.30)")
    ap.add_argument("--fig-only", action="store_true",
                    help="只按已有 plan.json 重画图(调试用)")
    ap.add_argument("--update", action="store_true",
                    help="首站满带重跑 03 后: 用实际带宽重算划分并校验(阶段B)")
    a = ap.parse_args()
    if a.update:
        plan = update_after_strip_run(a.run, a.smin)
    elif a.fig_only:
        plan = _load_json(os.path.join(a.run, "multi_station_plan.json"))
    else:
        plan = make_plan(a.run, a.smin)
    plot_plan(a.run, plan)
    print(f"[多工位] 最少 {plan['n_stations']} 站无缝, 每站条带宽=动作带宽 "
          f"{min(plan['steps']):.3f}~{max(plan['steps']):.3f}m, "
          f"墙宽 {plan['wall_width']}m")
    print(f"[多工位] 首站规划带(外扩补偿内收) {plan['first_plan_band']}; "
          f"逐站步距 walk_list={plan['exec_hint']['walk_list']}")


if __name__ == "__main__":
    main()
