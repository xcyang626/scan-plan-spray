# -*- coding: utf-8 -*-
"""步骤10: 喷涂轨迹工作站 (模块化: 轨迹输出 + 动作输出 + ROS topic; 执行接口预留).

启动:  python scripts/10_station_app.py [--port 8090] [--host 127.0.0.1]   (解释器见 station/env.json)
       默认只绑 127.0.0.1(仅本机可访问); 需局域网访问时显式 --host 0.0.0.0
界面:  http://127.0.0.1:8090   (浏览器弹出; 生成轨迹完成自动弹出检查页)

流程(全部异步后台, 界面实时日志, 反应秒级):
  [① 生成轨迹] 上传点云(可选)+输入站位距离 -> 11点云解析 -> 03快速IK(锚点+样条+NN种子)
               -> 04三维图  ->  自动弹出【待检查】页(轨迹图+指标+点云报告)
  [② 确认生成动作] -> 05增量MJCF-IK(NN种子) -> 05b时序图 -> (可选)ROS topic 输出
  [③ 导出执行包] -> UFO 部署zip + ROS topics
  [执行] POST /api/exec  —— 模块化预留接口, 本期返回 501, 执行模块后续接入
"""
import os, sys, json, time, argparse, threading, subprocess, zipfile, shutil, math
import signal
import atexit
import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(BASE, "station", "run")          # 兼容旧文件
RUNS = os.path.join(BASE, "runs")                    # 每次运行独立目录的根
os.makedirs(RUNS, exist_ok=True)
UP = os.path.join(BASE, "uploads")
os.makedirs(RUN, exist_ok=True); os.makedirs(UP, exist_ok=True)
# /api/upload 请求体上限(点云文件足够; 超限 413, 防内存被一次请求占满)
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
_ENV_JSON = {}
try:
    _ENV_JSON = json.load(open(os.path.join(BASE, "station", "env.json"),
                               encoding="utf8"))
    # 本机覆盖层: station/env.local.json 不入库(.gitignore)。跟踪版 env.json
    # 只放通用占位(无本机路径/内网配置), 换机部署 = 复制为 env.local.json 填值
    try:
        _ENV_JSON.update(json.load(open(
            os.path.join(BASE, "station", "env.local.json"), encoding="utf8")))
    except Exception:
        pass
except Exception:
    pass
ISA_PY = (os.environ.get("G1W_ISA_PY") or _ENV_JSON.get("isaac_py")
          or os.path.expanduser("~/miniconda3/envs/isaaclab/bin/python"))
MUJ_PY = (os.environ.get("G1W_MUJ_PY") or _ENV_JSON.get("mujoco_py")
          or os.path.expanduser("~/miniconda3/envs/mujoco/bin/python"))


# ---- GR00T-WBC 机器人执行 (启动策略/自动执行) ----
GROOT_ROOT = (os.environ.get("G1W_GROOT_ROOT") or _ENV_JSON.get("groot_root")
              or os.path.expanduser("~/桌面/轨迹规划/GR00T-WholeBodyControl"))
GROOT_PY = (os.environ.get("G1W_GROOT_PY") or _ENV_JSON.get("groot_python")
            or os.path.join(GROOT_ROOT, ".venv_sim_cpu", "bin", "python"))
ROBOT_RUNNER = os.path.join(BASE, "scripts", "exec", "groot_spray_runner.py")
TRIGGER_F = os.path.join(BASE, "station", "spray_trigger.json")
ROBOT_STATUS_F = os.path.join(BASE, "station", "robot_status.json")
ROBOT_PID_F = os.path.join(BASE, "station", "robot.pid")
ROBOT_LOG = os.path.join(BASE, "output", "robot.log")
ROBOT_STDIN_FIFO = os.path.join(BASE, "station", "robot_stdin.fifo")
PUB_PID_F = os.path.join(BASE, "station", "zmq_pub.pid")
_ROBOT_STDIN_FD = None      # 常驻 FIFO 写端(供急停注入按键)


AUTO_START_ROBOT = str(_ENV_JSON.get("auto_start_robot", "true")).lower() != "false"
# 动作生成后是否直接下发给机器人执行 —— 默认关: 生成后停在"动作就绪",
# 由操作者复核(检查页/指标/速度选择)后手动点【执行动作】
AUTO_EXEC_AFTER_MOTION = str(_ENV_JSON.get("auto_exec", "false")).lower() == "true"
# 执行速度倍率(默认; 页面下拉可覆盖). 0.5 = 半速复核
EXEC_SPEED = float(_ENV_JSON.get("exec_speed", 1.0))
# 实机安全限幅: 执行速度倍率上下限(越界一律夹取; groot_daemon.py 端同样限幅)
EXEC_SPEED_MIN, EXEC_SPEED_MAX = 0.05, 1.5
# 蹲姿地板(骨盆高度下界)统一口径 —— train_nn_ik / stage_plan / stage_motion
# 必须取同一个值: 该值编进工作区缓存目录名(g1_rp5_c35 / g1_rp5_c60), 口径
# 不一致会把工作区分裂成两套, 站上发布的 NN 模型(按 c60 训练)与 dual 规划
# 用的 c35 工作区对不上, 种子落在模型训练分布之外(实机蹲不下去/站不住)。
G1_CROUCH_LO_DUAL = 0.35     # G1 双姿态(站+蹲)的蹲姿下界
G1_CROUCH_LO_STAND = 0.60    # G1 纯站姿(不下蹲)时的下界
T3_CROUCH_LO = 0.43           # 天工3 蹲姿地板 = 腿几何下限
# 官方安全保持站姿(v7 首帧, SONIC 交错序) —— 启动机器人后持续保持, 防 reference 回放
HOLD_POSE_UPPER17 = _ENV_JSON.get("hold_pose_upper17_sonic",
                                  [0.0, 0.0, 0.0, 0.2, 0.0, 0.2, 0.0, 0.0, 0.0,
                                   0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


class _GrootDaemon:
    """常驻官方协议发布端代理 —— 站点与 scripts/exec/groot_daemon.py 之间的
    stdin/stdout JSON 行协议。守护进程独占 ZMQ 5556(部署 C++ 端 connect 过来),
    命令/动作流全部经它发送: 无 bind 冲突、无 PUB/SUB 慢加入丢消息。"""

    def __init__(self):
        self.proc = None
        self.lock = threading.Lock()
        self.wlock = threading.Lock()   # 写锁: id 分配 + stdin 写入必须整体原子
        self._id = 0
        self._wait = {}          # id -> [Event, response]

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def _reader(self):
        p = self.proc
        try:
            for line in p.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    m = json.loads(line)
                except Exception:
                    continue
                rid = m.get("id")
                _w = self._wait.get(rid) if rid is not None else None
                if _w is not None:       # 超时已被回收的 id 直接忽略(不抛 KeyError 中断读线程)
                    _w[1] = m
                    _w[0].set()
                elif m.get("event"):
                    _daemon_event(m)
        except Exception:
            pass

    def ensure(self):
        """确保守护进程在运行(懒启动, 失败返回 False 走子进程兜底)."""
        with self.lock:
            if self.alive():
                return True
            script = os.path.join(BASE, "scripts", "exec", "groot_daemon.py")
            if not os.path.exists(GROOT_PY) or not os.path.exists(script):
                log(f"⚠ 执行守护进程文件缺失: {script}")
                return False
            try:
                lg = open(os.path.join(BASE, "output", "groot_daemon.log"), "a")
                self.proc = subprocess.Popen(
                    [GROOT_PY, script], cwd=GROOT_ROOT,
                    env={**os.environ, "GROOT_ROOT": GROOT_ROOT,
                         "G1W_ZMQ_HOST": ZMQ_HOST,
                         "G1W_ZMQ_PORT": str(ZMQ_PORT),
                         "G1W_DAEMON_PORT": str(DAEMON_PORT)},
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=lg,
                    text=True, bufsize=1)
            except Exception as ex:
                log(f"⚠ 执行守护进程启动失败: {ex}")
                return False
            threading.Thread(target=self._reader, daemon=True).start()
        r = self.call("ping", timeout=10)
        if r.get("ok"):
            log("★ 执行守护进程已就绪 (官方协议发布端, 独占 ZMQ 端口)")
            return True
        log(f"⚠ 执行守护进程无响应: {r.get('err')} (命令走子进程兜底)")
        return False

    def call(self, cmd, timeout=15.0, **kw):
        """发请求并等应答; 守护进程不在运行则直接失败(不自动拉起, 避免递归)."""
        if not self.alive():
            return {"ok": False, "err": "daemon-not-running"}
        ev = threading.Event()
        # 写锁内完成 id 分配 + 写 + flush: /api/walk、/api/telemetry/dump 等并发
        # 调用若在锁外写, 两个线程的半行会交错进管道, daemon 解析错乱
        with self.wlock:
            self._id += 1
            rid = self._id
            self._wait[rid] = [ev, None]
            try:
                self.proc.stdin.write(json.dumps({"id": rid, "cmd": cmd, **kw}) + "\n")
                self.proc.stdin.flush()
            except Exception as ex:
                self._wait.pop(rid, None)
                return {"ok": False, "err": f"write-failed: {ex}"}
        if not ev.wait(timeout):
            self._wait.pop(rid, None)
            return {"ok": False, "err": f"timeout-{timeout:.0f}s"}
        return self._wait.pop(rid)[1] or {"ok": False, "err": "no-response"}


_DAEMON = _GrootDaemon()

# 实机多工位接力进度(write_status_phase 每次覆盖全文件, 故集中存这里随写随带)
_RELAY = {"station_idx": 1, "stations_total": 0, "band": None,
          "walk_dy_planned": 0.0, "walk_dy_done": 0.0}
_RELAY_ACTIVE = False
_RELAY_LOCK = threading.Lock()   # _RELAY 由守护事件线程与 HTTP 工作线程并发读写


def _relay_update(**kw):
    """_RELAY 整体加锁更新(混写会拼出跨来源的脏快照)."""
    with _RELAY_LOCK:
        _RELAY.update(kw)


def _relay_snapshot():
    """_RELAY 的一致性快照(整份拷贝, 不与并发写入交错)."""
    with _RELAY_LOCK:
        return dict(_RELAY)


def _analyze_telemetry(npz_p, json_p):
    """现场报告(2026-10-19 监测系统): daemon dump 的环形遥测 -> 行走窗口逐段
    分析(实测位移/步态死区/稳态速度/脉冲数) + 双联图(里程计+指令阴影 / 骨盆高
    +支撑脚) + 参数建议(写入 JOB 供页面【应用建议】落 calib.json)。
    在后台线程运行, 不阻塞守护进程事件流。"""
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        z = np.load(npz_p)
        evs = json.load(open(json_p, encoding="utf8"))
        t, oy, ph, sup = (z["t"], z["odom_y"], z["pelvis_h"], z["sup"])
        wins = []
        for i, (et, kind, kw) in enumerate(evs):
            if kind != "walk_start":
                continue
            t0e, dy = et, float(kw.get("dy") or 0.0)
            t1e = t0e + 20.0
            for et2, k2, _ in evs[i:]:
                if k2 == "walk_done" and et2 >= t0e:
                    t1e = et2
                    break
            m = (t >= t0e - 0.2) & (t <= t1e + 0.2)
            if m.sum() < 10:
                continue
            tw, yw = t[m], oy[m] - oy[m][0]
            ons = [e[0] for e in evs[i:]
                   if e[1] == "cmd_mv_on" and t0e <= e[0] <= t1e]
            offs = [e[0] for e in evs[i:]
                    if e[1] == "cmd_mv_off" and t0e <= e[0] <= t1e]
            # 死区: 首个指令 -> 里程计净位移首次超 2cm
            dead = None
            if ons and tw.size:
                y0v = float(np.interp(ons[0], tw, yw))
                mm = (tw >= ons[0]) & (np.abs(yw - y0v) > 0.02)
                if mm.any():
                    dead = float(tw[mm][0] - ons[0])
            moved = float(yw[-1]) if tw.size else 0.0
            vs = 0.0
            if dead is not None and (t1e - ons[0] - dead) > 0.3:
                vs = abs(moved - 0.02) / (t1e - ons[0] - dead)
            wins.append(dict(dy=dy, walked=round(moved, 3),
                             deadzone_s=(round(dead, 2) if dead is not None
                                         else None),
                             v_obs=round(vs, 3), bursts=len(ons),
                             t0=t0e, t1=t1e, y0=float(oy[m][0])))
        sug = {}
        dzs = [w["deadzone_s"] for w in wins if w["deadzone_s"]]
        if dzs:
            sug["walk_t0"] = round(max(0.2, float(np.median(dzs))), 2)
        vs_ = [w["v_obs"] for w in wins if w["v_obs"] > 0.05]
        if vs_:
            sug["walk_v"] = round(min(0.8, max(0.2, float(np.median(vs_)))), 2)
        # 出图: 上=里程计(指令区间阴影+目标线), 下=骨盆高+支撑脚
        png_p = npz_p.replace(".npz", ".png")
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
        for w in wins:
            mm = (t >= w["t0"] - 0.5) & (t <= w["t1"] + 0.5)
            if mm.sum() < 5:
                continue
            tr = t[mm] - w["t0"]
            a1.plot(tr, oy[mm] - w["y0"], lw=1.6,
                    label=f"目标{w['dy']:.3f} 实测{w['walked']:+.3f}m"
                          f" 死区{w['deadzone_s']}s {w['bursts']}段")
            a1.axhline(w["dy"], color="#1d9e52", ls=":", lw=1.0)
            for j, (e, k2, kw2) in enumerate(evs):
                if k2 == "cmd_mv_on" and w["t0"] <= e <= w["t1"]:
                    off = next((e2[0] for e2 in evs[j:]
                                if e2[1] == "cmd_mv_off" and e2[0] >= e), e + 0.2)
                    a1.axvspan(e - w["t0"], off - w["t0"], color="#f5b942",
                               alpha=0.18)
        a1.set_ylabel("里程计横向位移 (m)")
        a1.legend(fontsize=8.5, loc="best"); a1.grid(alpha=0.25)
        a2.plot(t - (wins[0]["t0"] if wins else t[0]), ph, color="#7f5af0", lw=1.2)
        a2.set_ylabel("骨盆高估计 (m)"); a2.grid(alpha=0.25)
        a2b = a2.twinx()
        a2b.step(t - (wins[0]["t0"] if wins else t[0]), sup, where="mid",
                 color="#444", lw=0.8, alpha=0.6)
        a2b.set_ylabel("支撑脚(1=左 2=右 3=双)", fontsize=9)
        a2.set_xlabel("时间 (s, 自首个行走窗口)")
        fig.suptitle("现场报告 — 里程计/指令时序 与 骨盆/支撑脚 "
                     f"({os.path.basename(npz_p)})")
        fig.tight_layout()
        fig.savefig(png_p, dpi=120)
        plt.close(fig)
        JOB["metrics"]["tele"] = {"png": png_p, "npz": npz_p,
                                  "windows": [{k: w[k] for k in
                                               ("dy", "walked", "deadzone_s",
                                                "v_obs", "bursts")}
                                               for w in wins],
                                  "suggest": sug}
        for w in wins:
            log(f"◈ 现场报告·行走段: 目标 {w['dy']:.3f}m 实测 {w['walked']:+.3f}m, "
                f"步态死区 {w['deadzone_s']}s, 稳态速度 {w['v_obs']}m/s, "
                f"{w['bursts']} 段脉冲")
        if sug:
            log(f"◈ 数据建议: " + ", ".join(f"{k}={v}" for k, v in sug.items())
                + " — 页面可一键【应用建议参数】(写入 station/calib.json)")
        log(f"◈ 现场报告图: {png_p}")
    except Exception as ex:
        log(f"⚠ 现场报告分析失败: {ex}")


def _daemon_event(m):
    """守护进程事件 -> 站点日志 + 机器人状态文件."""
    global _RELAY_ACTIVE
    ev = m.get("event")
    if ev == "stream_meta":
        log(f"▶ 官方上肢流: {m.get('frames')} 帧 / {m.get('duration_s')}s "
            f"@{m.get('rate')}Hz (speed×{m.get('speed')})"
            + (" [蹲姿通道: planner height + 骨盆闭环]" if m.get("height_channel") else ""))
    elif ev == "relay_station":                    # 实机多工位: 第 k 站开始喷涂
        _RELAY_ACTIVE = True
        _relay_update(station_idx=int(m.get("station_idx", 1)),
                      stations_total=int(m.get("stations_total", 0)),
                      band=m.get("band"),
                      walk_dy_planned=float(m.get("walk_dy_planned") or 0.0),
                      walk_dy_done=0.0)
        _snap = _relay_snapshot()
        _n, _k = _snap["stations_total"], _snap["station_idx"]
        log(f"▶ 实机接力 第 {_k}/{_n} 站喷涂开始"
            + (f" (下一站左移 {_snap['walk_dy_planned']:.3f}m)" if _k < _n else " (末站)"))
        write_status_phase("playing", progress=0.0, **_snap)
    elif ev == "relay_phase" and m.get("phase") == "station_done":
        _relay_update(station_idx=int(m.get("station_idx", 1)),
                      stations_total=int(m.get("stations_total", 0)))
        _snap = _relay_snapshot()
        _k, _n = _snap["station_idx"], _snap["stations_total"]
        log(f"★ 第 {_k}/{_n} 站喷涂完成 — 停稳后 SONIC 左移 "
            f"{_snap['walk_dy_planned']:.3f}m")
        write_status_phase("station_done", **_snap)
    elif ev == "walk_started":
        _relay_update(walk_dy_planned=float(m.get("dy") or 0.0), walk_dy_done=0.0)
        log(f"▶ SONIC 行走: 左移 {m.get('dy')}m @ {m.get('v')}m/s "
            + ("[腿式里程计闭环: 走-量-停, 欠补过退]" if m.get("mode") == "odom-closed-loop"
               else f"[⚠ 无位移反馈, 回退开环: 命令 ~{m.get('est_s')}s 含裕量 "
                    f"{m.get('t0_margin')}s]")
            + ", 人工监护, 随时可急停)")
        write_status_phase("walking", **_relay_snapshot())
    elif ev == "walk_done":
        _relay_update(walk_dy_done=float(m.get("walked_est") or 0.0))
        log(f"★ 行走完成({'里程计实测' if m.get('measured') else '开环估算'}): "
            f"{m.get('walked_est')}m / 目标 {m.get('dy')}m, 用时 {m.get('held_s')}s"
            + ("" if m.get("measured") else " — 建议按标尺用【补走】修正")
            + (" (被取消)" if m.get("cancelled") else ""))
        write_status_phase("walking", **_relay_snapshot())
    elif ev == "walk_settle":
        write_status_phase("settle", **_relay_snapshot())
    elif ev == "relay_done":
        _snap = _relay_snapshot()
        log(f"★ 实机多工位接力全部完成: {_snap['stations_total']} 站已喷")
        write_status_phase("standing", **_snap)
        _RELAY_ACTIVE = False
    elif ev == "stream_run":
        log("blend 过渡完成, 轨迹开始")
    elif ev == "stream_progress":
        n, k = m.get("n") or 1, m.get("k") or 0
        write_status_phase("playing", progress=(k / n if n else 0.0),
                           **(_relay_snapshot() if _RELAY_ACTIVE else {}))
    elif ev == "stream_done":
        if not _RELAY_ACTIVE:                      # 接力中站点完成由 relay_phase 接管
            write_status_phase("done", progress=1.0)
        _hc = " [蹲姿通道]"
        if m.get("height_channel") or m.get("pelvis_err_max") is not None:
            log(f"★ 轨迹执行完成: {m.get('frames')} 帧已全部下发, "
                f"{m.get('note') or '保持末姿'}"
                f"{_hc} 骨盆最低 {m.get('pelvis_min')}m / |骨盆-参考|最大 "
                f"{m.get('pelvis_err_max')}m / 腕心高度偏差 max "
                f"{(m.get('wrist_z_err_max') or 0)*100:.1f}cm"
                f"(走线水平度) / 枪偏离法向 max "
                f"{m.get('gun_angle_off_max')}°")
        else:
            log(f"★ 轨迹执行完成: {m.get('frames')} 帧已全部下发, "
                f"{m.get('note') or '保持末姿'}")
    elif ev == "stream_cancelled":
        _RELAY_ACTIVE = False                      # 接力随取消一并终止
        write_status_phase("soft_stopped")
        log(f"动作流已取消 ({m.get('note') or '官方退出序列'})")
    elif ev == "hold_started":
        log("★ " + str(m.get("note") or "保持流已启动"))
    elif ev == "hold_cancelled":
        log("保持流已结束 (" + str(m.get("note") or "") + ")")
    elif ev == "stream_error":
        log(f"⚠ 动作流异常: {m.get('err')}")
    elif ev == "speed_clamped":
        log(f"⚠ 速度倍率 {m.get('requested')} 超出安全限幅, "
            f"已夹取为 {m.get('applied')} (实机安全限幅 [{EXEC_SPEED_MIN}, {EXEC_SPEED_MAX}])")
    elif ev == "fatal":
        log(f"⚠ 执行守护进程致命错误: {m.get('err')}")
    elif ev == "telemetry_dumped":
        # 现场已落盘(急停/错误/行走停滞/手动): 后台线程分析出图, 不堵事件流
        log(f"◈ 遥测现场已保存(原因: {m.get('reason')}) — 最近 120s: "
            f"{os.path.basename(m.get('npz') or '')}")
        if m.get("npz") and m.get("json"):
            threading.Thread(target=_analyze_telemetry,
                             args=(m["npz"], m["json"]), daemon=True).start()


def groot_command(kind):
    """经 GR00T 官方 command 话题下发控制命令。
    首选常驻守护进程(独占 ZMQ 端口, 即时可达); 失败回退子进程直发。"""
    if _DAEMON.ensure():
        r = _DAEMON.call("command", kind=kind, timeout=15)
        if r.get("ok"):
            log(f"GR00T command({kind}) 已下发 (守护进程)")
            return True
        log(f"GR00T command({kind}) 守护进程失败: {r.get('err')} → 子进程兜底")
    try:
        bridge = os.path.join(BASE, "scripts", "exec", "groot_bridge.py")
        r = subprocess.run([GROOT_PY, bridge, "--send-command", kind,
                            "--host", ZMQ_HOST, "--port", str(ZMQ_PORT)],
                           cwd=GROOT_ROOT, capture_output=True, text=True, timeout=30)
        ok = r.returncode == 0
        log(f"GR00T command({kind}) {'已下发' if ok else '失败: ' + (r.stderr or '')[-160:]}")
        return ok
    except Exception as ex:
        log(f"GR00T command({kind}) 异常: {ex}")
        return False


def _wait_deploy_ready(timeout=90.0, offset=0):
    """等部署进程打印 'Init Done' (就绪标志).
    offset: 只认本次启动后新写入的日志 —— 避免上次会话残留的 Init Done
    让 start 命令过早下发(PUB/SUB 尚未连接, 消息会被丢弃)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with open(ROBOT_LOG, "rb") as f:
                if offset > 0:
                    f.seek(offset)
                tail = f.read()[-4000:].decode("utf8", "ignore")
            if "Init Done" in tail:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def send_key(ch):
    """向部署进程 stdin(FIFO) 注入按键(等价终端按键)."""
    try:
        fd = os.open(ROBOT_STDIN_FIFO, os.O_WRONLY | os.O_NONBLOCK)
        try:
            os.write(fd, ch.encode())
        finally:
            os.close(fd)
        return True
    except Exception as ex:
        log(f"  ⚠ 注入按键 {ch!r} 失败: {ex}")
        return False


_STATUS_LOCK = threading.Lock()     # robot_status.json 多线程(HTTP/守护事件/watchdog)共用


def write_status_phase(phase, progress=None, **kw):
    """原子写状态文件: 临时文件 + os.replace —— 并发写不会留下半截 JSON.
    kw 附加字段(实机多工位接力的 station_idx/band 等)随 phase 一起落盘."""
    try:
        st = {"phase": phase, "ts": time.time()}
        if progress is not None:
            st["progress"] = progress
        st.update(kw)
        with _STATUS_LOCK:
            tmp = ROBOT_STATUS_F + ".tmp"
            with open(tmp, "w", encoding="utf8") as f:
                json.dump(st, f)
            os.replace(tmp, ROBOT_STATUS_F)
    except Exception:
        pass


def _robot_stdin_fd():
    """建立并持有 FIFO 读写端: 作为部署进程 stdin, 同时供急停写入按键."""
    global _ROBOT_STDIN_FD
    if _ROBOT_STDIN_FD is None:
        try:
            if os.path.exists(ROBOT_STDIN_FIFO):
                os.remove(ROBOT_STDIN_FIFO)
            os.mkfifo(ROBOT_STDIN_FIFO)
        except FileExistsError:
            pass
        _ROBOT_STDIN_FD = os.open(ROBOT_STDIN_FIFO, os.O_RDWR | os.O_NONBLOCK)
    return _ROBOT_STDIN_FD
ROBOT_MODE = (os.environ.get("G1W_ROBOT_MODE") or _ENV_JSON.get("robot_mode") or "sim").lower()
# SONIC 行走步态死区裕量(s): 蹲/站→SLOW_WALK 起身+重心转移不产生位移(2026-10-19
# 首跑实测 0.33m@0.25m/s 只动了一下)。闭环行走已不依赖它, 仅作无 5557 反馈时的
# 开环回退参数; 数值由现场报告的数据建议一键写入 station/calib.json 覆盖。
_WALK_T0 = float(os.environ.get("G1W_WALK_T0", "0.8") or 0.8)
_WALK_V = float(os.environ.get("G1W_WALK_V", "0.25") or 0.25)
try:      # 监测系统闭环: 现场报告建议 -> calib.json -> 下一次执行生效
    _calib = json.load(open(os.path.join(BASE, "station", "calib.json"),
                            encoding="utf8"))
    _WALK_T0 = float(_calib.get("walk_t0", _WALK_T0))
    _WALK_V = float(_calib.get("walk_v", _WALK_V))
except Exception:
    pass
GROOT_DEPLOY_DIR = os.path.join(GROOT_ROOT, "gear_sonic_deploy")
GROOT_DEPLOY_BIN = os.path.join(GROOT_DEPLOY_DIR, "target", "release", "g1_deploy_onnx_ref")
# 实机部署资产 (官方低延迟 SONIC 栈)
GROOT_DECODER = "policy/low_latency/model_decoder.onnx"
GROOT_ENCODER = "policy/low_latency/model_encoder.onnx"
GROOT_OBS_CFG = "policy/low_latency/observation_config.yaml"
GROOT_PLANNER = "planner/target_vel/V2/planner_sonic.onnx"
GROOT_MOTION_DIR = "reference/example"
GROOT_LD_LIBS = [".deps/TensorRT/lib", ".deps/onnxruntime/lib",
                 ".venv_sim_cpu/lib/python3.10/site-packages/nvidia/cuda_runtime/lib",
                 "gear_sonic_deploy/thirdparty/unitree_sdk2/thirdparty/lib/x86_64",
                 "/usr/local/cuda/lib64"]
ZMQ_PORT = int(_ENV_JSON.get("zmq_port", 5556))
# ZMQ 发布端绑定地址: 约定必须 localhost(env.json zmq_host), 无配置时默认仅本机,
# 不再默认 '*'(全网卡可达); 需局域网访问时在 env.json 显式指定
ZMQ_HOST = _ENV_JSON.get("zmq_host", "127.0.0.1")
DAEMON_PORT = int(_ENV_JSON.get("daemon_port", 5558))
ROBOT_IFACE = _ENV_JSON.get("robot_iface", "enp8s0")
# 输入类型: gamepad_manager=手柄(推荐, F2软急停/Select退出/F1切ZMQ流) | zmq=纯键盘
# GR00T 原生输入类型: zmq_manager = 支持 planner 话题(上肢通道) + command 话题(start/stop)
ROBOT_INPUT = _ENV_JSON.get("robot_input_type", "zmq_manager")
# 策略接管混合时长(秒): 启动控制瞬间从实测姿态 smoothstep 过渡到策略输出,
# 消除网页启动时的剧烈抖动(官方为硬切换); 0=回官方行为. 2026-10-13 键盘流程实测有效后移植
TAKEOVER_BLEND_S = float(_ENV_JSON.get("takeover_blend_s", 2.0))


_ROBOT_PROC = None


def _alive(pid):
    """进程是否存活(排除僵尸)."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            st = f.read().decode(errors="ignore").rsplit(")", 1)[1].split()[0]
        return st != "Z"
    except Exception:
        return False


def _robot_cmd_ok(pid):
    """cmdline 是否确为我们的 runner/部署进程 —— argv[0] 必须是解释器或部署二进制.
    (pgrep -f 匹配整条 cmdline, 会把恰好包含同名子串的测试 shell/grep 命令一起
    匹配到: 曾经把 bash 包装进程认领成机器人, 状态/急停全指向错误进程)"""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            argv = [a for a in f.read().decode(errors="ignore").split("\0") if a]
    except Exception:
        return False
    if not argv:
        return False
    a0 = os.path.basename(argv[0])
    if a0 == "g1_deploy_onnx_ref":
        return True
    if not a0.startswith(("python", "Python")):   # sim runner: [python, .../groot_spray_runner.py, ...]
        return False
    return any("groot_spray_runner" in a for a in argv[1:])


def _robot_pid():
    """稳健存活判定: 排除僵尸进程与 PID 复用 (校验 cmdline 确为我们的 runner)."""
    global _ROBOT_PROC
    if _ROBOT_PROC is not None:
        try:
            _ROBOT_PROC.poll()      # 回收子进程, 避免僵尸
        except Exception:
            pass
    try:
        pid = int(open(ROBOT_PID_F).read().strip())
    except Exception:
        pid = None
    if pid is None or not os.path.exists(f"/proc/{pid}") or not _robot_cmd_ok(pid):
        # PID 文件丢失/失效/被复用 -> 按进程名兜底(自愈): 避免孤儿部署占端口后再启一个
        for pat in ("g1_deploy_onnx_ref", "groot_spray_runner"):
            try:
                r = subprocess.run(["pgrep", "-f", pat], capture_output=True,
                                   text=True, timeout=5)
                cand = [int(x) for x in r.stdout.split() if x.isdigit()]
            except Exception:
                cand = []
            cand = [c for c in cand if c != os.getpid() and _alive(c)
                    and _robot_cmd_ok(c)]
            if cand:
                pid = cand[0]
                try:
                    open(ROBOT_PID_F, "w").write(str(pid))   # 认领并补记 PID
                except Exception:
                    pass
                log(f"检测到已在运行的机器人进程 pid={pid} (已认领)")
                return pid
        return None
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            st = f.read().decode(errors="ignore").rsplit(")", 1)[1].split()[0]
        if st == "Z":                     # 僵尸: 已退出未回收
            return None
        return pid
    except Exception:
        return None


def _robot_status():
    """状态以**实时进程检测**为准, 状态文件只补充 phase/progress(不得覆盖 running)。"""
    if _T3_READY and _robot_pid() is None:
        return {"running": True, "mode": "tiangong3-planning", "daemon": False,
                "robot": "tiangong3", "phase": "ready", "progress": 0.0}
    live = _robot_pid() is not None
    # robot=会话身份: 界面按钮/文案据此判断, 不能用 JOB 的产物机器人(切换后会错位)
    st = {"running": live, "mode": ROBOT_MODE, "daemon": _DAEMON.alive(),
          "robot": (_SESSION_ROBOT or "g1") if live else None}
    try:
        f = json.load(open(ROBOT_STATUS_F, encoding="utf8"))
        for k in ("phase", "progress", "pelvis", "motion", "frames", "done_runs",
                  "pelvis_min", "pelvis_err_max", "fallen", "impact_err",
                  "station_idx", "stations_total", "walk_dy_planned",
                  "walk_dy_done", "band", "path"):
            if k in f:
                st[k] = f[k]
    except Exception:
        pass
    if not live:                        # 进程已退出 -> 不采信文件里的残留 phase
        st["phase"] = "stopped"
        st["progress"] = 0.0
    st.setdefault("phase", "stopped" if not live else "starting")
    return st


def kill_deploy(soft_wait=8.0):
    """可靠终止部署: 官方 command(stop) → 等优雅退出 → 超时升级 SIGTERM/SIGKILL.
    返回 True=进程已消失. (旧版在站点进程内 import estop→官方模块, mujoco 环境
    缺 msgpack 会静默失败, 残留部署占端口导致新部署 'Address already in use' 崩溃)"""
    groot_command("stop")                       # 官方急停: 阻尼指令 + 优雅退出
    pid = _robot_pid()
    if not pid:
        return True
    t0 = time.time()
    while time.time() - t0 < soft_wait and _robot_pid():
        time.sleep(0.5)
    for sig, wait in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 3.0)):
        pid = _robot_pid()
        if not pid:
            break
        try:
            os.kill(pid, sig)
        except Exception:
            pass
        t0 = time.time()
        while time.time() - t0 < wait and _robot_pid():
            time.sleep(0.5)
    try:      # 清兜底路径遗留的发布器(官方脚本/旧发布端, 可能占 ZMQ 端口)
        subprocess.run(["pkill", "-f", "send_upper_body_trajector[y]"], timeout=5)
        subprocess.run(["pkill", "-f", "groot_bridge"], timeout=5)
    except Exception:
        pass
    return _robot_pid() is None


_T3_READY = False     # 天工3 虚拟就绪标志(规划纯运动学, 无需策略进程)
_SESSION_ROBOT = None  # 当前机器人会话身份(g1|tiangong3): 界面按钮/状态文案跟随
                       # 实际启动的机器人, 而非 JOB 里上次产物的机器人(两者会不一致)


def _robot_pid_or_virtual():
    """_robot_pid 的天工感知版: 天工虚拟就绪时也返回真值(用负数区分)."""
    return -1 if _T3_READY else _robot_pid()


def start_robot(posture="stand", robot=None):
    """启动机器人: 拉起 GR00T-WBC 仿真+策略, 机器人回到初始站姿并等待执行.
    posture=dual 时 sim 分支给 runner 追加 --height(蹲姿执行通道: 动作骨盆高
    写入 height_cmd, 由平衡策略下蹲) —— 执行通道与回放都能看到完整蹲姿.
    robot=tiangong3: 规划/动作/回放为纯运动学, 虚拟就绪(不起进程, 不占端口)."""
    global ESTOP, _ROBOT_HEIGHT, _T3_READY, _SESSION_ROBOT
    ESTOP = False               # 新会话起点: 解除急停对 stage 线程的封锁
    _ROBOT_HEIGHT = True     # sim 启动永远预挂 --height(见下方 sim 分支说明)
    # 启动机器人 = 新会话起点: 上次运行(可能是另一台机器人)的检查/就绪状态与
    # 产物一并作废 —— 否则界面会把旧产物的"动作就绪"当成当前会话的可执行对象
    # (_persist 自身也要拿 LOCK, 必须移出 with 块 —— threading.Lock 不可重入)
    _reset = False
    with LOCK:
        if JOB["state"] in ("inspect", "ready", "error"):
            JOB.update(state="idle", error=None, metrics={}, artifacts={},
                       motion_ver=None, run_dir=None)
            _reset = True
    if _reset:
        _persist()
    if robot == "tiangong3":
        if _robot_pid():        # 切换机器人 = 结束上一会话: 停掉 G1 部署进程
            log("切换到天工3 → 先停止当前 G1 部署进程…")
            kill_deploy()
        _T3_READY = True
        _SESSION_ROBOT = "tiangong3"
        log("★ 天工3 规划会话就绪(纯运动学, 无策略进程) — 可直接 ① 生成轨迹; "
            "执行通道未接入, 成品看【打开仿真回放】")
        write_status_phase("ready")
        return {"ok": True, "already": False, "mode": "tiangong3-planning"}
    _T3_READY = False           # 启动 G1 = 结束天工虚拟会话(防幽灵"运行中")
    _SESSION_ROBOT = "g1"
    # 停掉可能在跑的动作流(否则机器人停在喷涂姿态, 看起来"没回初始位姿")
    try:
        subprocess.run(["pkill", "-f", "send_upper_body_trajector[y]"], timeout=5)
        subprocess.run(["pkill", "-f", "groot_bridge"], timeout=5)
    except Exception:
        pass
    if _robot_pid():
        # 已在运行: 为保证"点启动机器人 => 回到初始位姿", 走一次完整重启
        # (GR00T 的 start 只在 WAIT_FOR_CONTROL 生效, 重复发送不会重跑 INIT 斜坡;
        #  只有重启进程才会经 INIT 斜坡把机器人送回默认站姿)
        log("机器人已在运行 → 执行完整重启以回到初始位姿…")
        kill_deploy()
        log("  旧进程已结束, 重新启动…")
    if not os.path.exists(GROOT_PY):
        raise RuntimeError(f"GR00T 解释器不存在: {GROOT_PY} (改 station/env.json 的 groot_python)")
    if not os.path.exists(ROBOT_RUNNER):
        raise RuntimeError(f"runner 缺失: {ROBOT_RUNNER}")
    try:
        os.remove(TRIGGER_F)          # 清掉旧触发
    except Exception:
        pass
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
           "HOME": os.environ.get("HOME", os.path.expanduser("~")),
           "DISPLAY": os.environ.get("DISPLAY") or _ENV_JSON.get("display") or ":0",
           "GROOT_ROOT": GROOT_ROOT,
           "LANG": os.environ.get("LANG", "C.UTF-8"),
           "LIBGL_ALWAYS_SOFTWARE": os.environ.get("LIBGL_ALWAYS_SOFTWARE", "1"),
           "__GLX_VENDOR_LIBRARY_NAME": os.environ.get("__GLX_VENDOR_LIBRARY_NAME", "mesa"),
           "MESA_GL_VERSION_OVERRIDE": os.environ.get("MESA_GL_VERSION_OVERRIDE", "3.3")}
    os.makedirs(os.path.dirname(ROBOT_LOG), exist_ok=True)
    try:      # 日志轮转: 超 20MB 归档, 防长期追加导致就绪检测变慢
        if os.path.getsize(ROBOT_LOG) > 20 * 1024 * 1024:
            os.replace(ROBOT_LOG, ROBOT_LOG + ".old")
    except Exception:
        pass
    logf = open(ROBOT_LOG, "a")
    log_offset = os.path.getsize(ROBOT_LOG)   # 只认本次启动后新写入的部分
    global _ROBOT_PROC
    if ROBOT_MODE == "real":
        # 顺序关键: 先让守护进程 bind 5556, 再启动部署(部署 SUB connect 过来),
        # 全程无慢加入 —— start 命令不会被丢
        if not _DAEMON.ensure():
            log("⚠ 守护进程未就绪, 命令/动作流将走子进程兜底(稳定性降级)")
        # 实机: C++ SONIC 部署(经 DDS 连物理 G1) + ZMQ 动作推流端
        # 直接调二进制: 官方 deploy.sh 依赖 just(本机未装), 且需交互确认
        if not os.path.exists(GROOT_DEPLOY_BIN):
            raise RuntimeError(f"部署二进制不存在: {GROOT_DEPLOY_BIN} (需先构建)")
        for _f in (GROOT_DECODER, GROOT_ENCODER, GROOT_OBS_CFG, GROOT_PLANNER):
            if not os.path.exists(os.path.join(GROOT_DEPLOY_DIR, _f)):
                raise RuntimeError(f"缺少部署资产: {_f}")
        env2 = dict(env)
        _libs = os.pathsep.join(os.path.join(GROOT_ROOT, p) for p in GROOT_LD_LIBS)
        env2["LD_LIBRARY_PATH"] = _libs + os.pathsep + env2.get("LD_LIBRARY_PATH", "")
        cmd = [GROOT_DEPLOY_BIN, ROBOT_IFACE, GROOT_DECODER, GROOT_MOTION_DIR,
               "--obs-config", GROOT_OBS_CFG,
               "--encoder-file", GROOT_ENCODER,
               "--planner-file", GROOT_PLANNER,
               "--input-type", ROBOT_INPUT, "--output-type", "zmq",
               "--zmq-host", ZMQ_HOST, "--zmq-port", str(ZMQ_PORT),
               "--zmq-topic", "pose"]
        if TAKEOVER_BLEND_S > 0:
            cmd += ["--takeover-blend-s", f"{TAKEOVER_BLEND_S:g}"]
        log(f"实机模式: 启动 SONIC 部署 (网卡 {ROBOT_IFACE}, 输入 {ROBOT_INPUT}, "
            f"zmq {ZMQ_HOST}:{ZMQ_PORT})"
            + (f", 接管混合 {TAKEOVER_BLEND_S:g}s" if TAKEOVER_BLEND_S > 0 else ""))
        if ROBOT_INPUT == "gamepad_manager":
            log("  手柄: Start=启动控制  F1=切换到ZMQ流模式  F2=软急停(变软)  Select=退出")
        else:
            log("  键盘: ] 启动控制, ENTER 切 ZMQ 流, O 软急停, X 退出")
        _ROBOT_PROC = subprocess.Popen(cmd, cwd=GROOT_DEPLOY_DIR, env=env2,
                                       stdout=logf, stderr=subprocess.STDOUT,
                                       stdin=_robot_stdin_fd(),   # FIFO: 支持急停注入 'O'
                                       start_new_session=True)
        try:
            _pidf = os.path.join(BASE, "station", "robot.pid")
            with open(_pidf, "w") as _f:
                _f.write(str(_ROBOT_PROC.pid))     # 关键: 供状态/急停定位部署进程
            log(f"  部署进程 pid={_ROBOT_PROC.pid} (已记录)")
        except Exception as _e:
            log(f"  ⚠ 记录部署 PID 失败: {_e}")
        time.sleep(3)
        log("真机模式: 就绪 (动作在执行时经 GR00T 官方 planner 话题下发)")
        # 自动化: 部署就绪后自动 启动控制(s) + 启用ZMQ流(z), 无需人工按手柄
        if AUTO_START_ROBOT:
            def _auto_start():
                ok = _wait_deploy_ready(timeout=90, offset=log_offset)
                if not ok:
                    log("⚠ 部署未在预期时间内就绪, 自动启动控制跳过 (可手柄 Start)")
                    return
                time.sleep(0.5)
                # GR00T 原生: command(start=True, planner=True)
                #  -> 启动控制(机器人回到初始站姿) 且 进入 planner 模式(接收 upper_body_position)
                if groot_command("start"):
                    log("★ 已通过 GR00T command 启动控制 + planner 模式 → 机器人回初始站姿")
                    # 官方终端2语义: start 后立即持续保持上肢站姿(50Hz)。
                    # 默认保持**机器人当前实测姿态**(5557/g1_debug 反馈, 零瞬态,
                    # 对关节序/左右映射免疫); 无反馈时守护进程回退官方 v7 首帧。
                    # 若无保持流, C++ 在 planner 静默时会回退回放 reference/
                    # example 里的参考动作(舞蹈/踢腿) —— 即实测的"剧烈运动"
                    _DAEMON.ensure()
                    r = _DAEMON.call("hold", timeout=15)
                    log(f"★ 保持流: {r.get('note') or '已启动'}"
                        if r.get("ok") else f"⚠ 保持流启动失败: {r.get('err')}")
                write_status_phase("standing")
            threading.Thread(target=_auto_start, daemon=True).start()
    else:
        # sim 启动永远预挂蹲姿通道: 对站姿动作无影响(height_cmd 播放期才生效且跟随
        # 动作骨盆, 站姿动作即恒 ~0.758; 待命期自动回 yaml 默认) —— 从机制上消灭
        # "启动时刻姿态选择决定能否下蹲"这一整类失败(2026-10-10 两次实测踩中)
        # --log: 每次执行自动留跟踪日志(落点误差诊断的数据源, 每次运行覆盖)
        _cmd = [GROOT_PY, ROBOT_RUNNER, "--trigger", TRIGGER_F, "--height",
                "--log", os.path.join(BASE, "output", "exec_last.npz")]
        log("▶ 蹲姿执行通道已预挂(--height): 站姿动作不受影响, 双姿态动作可直接下蹲")
        _ROBOT_PROC = subprocess.Popen(
            _cmd,
            cwd=GROOT_ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
    write_status_phase("starting")        # 原子写(与各线程并发写共用一把锁)
    log("★ 机器人启动中 (GR00T-WBC 仿真+策略), 等待回到初始站姿…")
    return {"ok": True, "already": False, "mode": ROBOT_MODE}


def stop_robot():
    """停止机器人 = **立即全身变软**(阻尼保护态), 不退出进程.
    说明: 退出进程会让机器人失去指令源(会报警/缓慢归位);
          保持进程持续下发 kp=0,kd=8 阻尼指令, 才是真正的"变软".
    另外: 取消进行中的规划/动作流水线, 清空本轮任务结果(界面复位)。
    彻底结束请用 exit_robot() / 网页【退出程序】。"""
    global ESTOP, _T3_READY, _SESSION_ROBOT
    ESTOP = False               # 完整复位: 解除急停对 stage 线程的封锁
    _T3_READY = False           # 天工虚拟会话一并结束
    _SESSION_ROBOT = None
    # 停止 = GR00T 原生急停: command(stop=True)
    #   官方行为: 下发零力矩阻尼指令(kp=0, kd=8) -> 全身变软, 然后优雅停止
    if _DAEMON.alive():
        _DAEMON.call("cancel", timeout=5)     # 停止残留动作流(不再发帧)
    groot_command("stop")
    try:
        os.remove(TRIGGER_F)
    except Exception:
        pass
    # 取消进行中的流水线(规划/动作)
    _cancel_proc()
    # 停止后: 机器人进程仍在(阻尼保护态), 只清空本轮任务结果
    with LOCK:
        JOB.update(state="idle", error=None, metrics={}, artifacts={},
                   motion_ver=None, run_dir=None, stage="", ros_mode=None)
    time.sleep(1.5)                       # 等原生急停走完(阻尼指令 -> 优雅退出)
    write_status_phase("soft_stopped")
    _persist()
    alive = _robot_pid() is not None
    log("★ 已停止: GR00T 原生急停已下发(零力矩阻尼=全身变软)"
        + (", 部署进程已优雅退出" if not alive else ""))
    return {"ok": True, "soft": True, "running": alive}


# 官方上肢 17 关节列序(与 send_upper_body_trajectory.py HARDWARE_ORDER 一致;
# 不能在站点进程 import groot_bridge —— 它模块级 import 官方发送端, 依赖 msgpack)
_UPPER17_HW_ORDER = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def _npz_to_upper_csv(motion_npz, out_csv):
    """动作 npz -> 官方上肢 CSV(README §8: mujoco jnt_qposadr 按名字解析)."""
    import numpy as np, csv as _csv, mujoco
    m = mujoco.MjModel.from_xml_path(
        REGISTRY.get(JOB.get("robot", "g1"), {}).get("mjcf", ""))
    q = np.asarray(np.load(motion_npz)["qpos"], float)
    idx = [int(m.jnt_qposadr[m.joint(n).id]) for n in _UPPER17_HW_ORDER]
    u = q[:, idx]
    with open(out_csv, "w", newline="", encoding="utf8") as f:
        w = _csv.writer(f)
        w.writerow(["time_s", *_UPPER17_HW_ORDER])
        for i, row in enumerate(u):
            w.writerow([f"{i / 30.0:.6f}", *[f"{v:.9f}" for v in row]])
    return out_csv


def _motion_path_allowed(mp):
    """motion 白名单: 只允许 本次 run 目录(state 里的 run_dir) 与 BASE/output 之内
    的路径 —— realpath 解析后做 base+os.sep 前缀判断(防 ../ 与同前缀目录绕过)."""
    rp = os.path.realpath(mp)
    for base in (JOB.get("run_dir"), os.path.join(BASE, "output")):
        if not base:
            continue
        rb = os.path.realpath(base)
        if rp == rb or rp.startswith(rb + os.sep):
            return True
    return False


def exec_spray(motion_path, run_dir=None, speed=None, stations=1):
    """手动执行 —— 完全对齐官方 send_upper_body_trajectory.py:
    planner 话题 17 维上肢(SONIC 交错序) @50Hz + 梯度速度 + 结束保持末姿;
    下肢由 SONIC 平衡策略控制(不发腿)。speed 可选半速复核(0.5)。
    首选常驻守护进程, 失败回退官方脚本本体。
    sim 模式走触发文件通道(MuJoCo runner 不订阅 ZMQ)。
    stations>1: 多工位接力 —— 喷①→停稳→左移一个步距→喷②…(sim=MuJoCo 闭环走位;
    real=SONIC 行走指令开环走位, 2026-10-19 起支持)。"""
    if JOB.get("robot") == "tiangong3":
        return {"err": "天工3 执行通道未接入(无策略栈) — 请用【打开仿真回放】查看成品"}
    if not _robot_pid():
        return {"err": "机器人未启动 — 请先点【启动机器人】"}
    if not motion_path or not os.path.exists(motion_path):
        return {"err": f"动作文件不存在: {motion_path}"}
    if stations == "auto":
        pass               # 多工位全覆盖: 稍后从 multi_station_plan.json 解析站数
    else:
        try:
            stations = max(1, int(stations or 1))
        except (TypeError, ValueError):
            stations = 1
    # 2026-10-19 用户定稿: 多工位上实机放开 —— 站间行走用 SONIC 运动指令
    # (planner movement/speed, SLOW_WALK 域 0.2~0.8m/s)。位移无世界反馈, 按
    # 时间×速度开环, 全程人工监护 + 随时急停; 计划解析与下发在下方 real 段。
    # V2 双姿态执行防护(兜底): sim 启动现已永远预挂 --height, 此守卫只防手撸 CLI/
    # 异常路径; trajectory_meta 在 run_dir(动作路径需上溯三层, 旧代码少算一层从未生效)
    try:
        _rd = JOB.get("run_dir") or os.path.dirname(
            os.path.dirname(os.path.dirname(motion_path)))
        _tmeta = os.path.join(_rd, "trajectory_meta.json")
        _is_dual = (os.path.exists(_tmeta)
                    and json.load(open(_tmeta, encoding="utf8")).get("posture") == "dual")
    except Exception:
        _is_dual = False
    if _is_dual and not _ROBOT_HEIGHT:
        return {"err": "该动作含蹲姿段, 但机器人未以蹲姿通道启动 — 请把喷涂姿态选为"
                       "「站+蹲双姿态」后重新点【启动机器人】, 再执行"}
    if _is_dual:
        log("▶ 双姿态动作: 蹲姿执行通道已启用"
            + (" (planner height + 骨盆闭环, SONIC 策略下蹲)" if ROBOT_MODE == "real"
               else " (骨盆高 -> height_cmd, 策略下蹲)"))
    try:
        speed = EXEC_SPEED if speed is None else float(speed)
    except (TypeError, ValueError):
        return {"err": f"速度倍率格式错误: {speed!r}"}
    if not math.isfinite(speed):   # NaN/inf 会穿透 min/max 夹取, 归一为默认倍率
        log(f"⚠ 速度倍率非有限值 {speed!r}, 已按默认 ×{EXEC_SPEED} 执行")
        speed = EXEC_SPEED
    if not (EXEC_SPEED_MIN <= speed <= EXEC_SPEED_MAX):   # 实机安全限幅
        clamped = min(max(speed, EXEC_SPEED_MIN), EXEC_SPEED_MAX)
        log(f"⚠ 速度倍率 {speed} 超出安全限幅 "
            f"[{EXEC_SPEED_MIN}, {EXEC_SPEED_MAX}], 已夹取为 {clamped}")
        speed = clamped
    if ROBOT_MODE != "real":
        # sim 通道: MuJoCo runner(--trigger) 轮询触发文件, 不订阅 ZMQ ——
        # 按 groot_spray_runner.py 约定写 JSON(字段 motion=动作 npz 路径)
        trig = {"motion": motion_path, "speed": speed}
        if stations == "auto":
            # 多工位全覆盖: 站数/站距/首站带由 14 的条带划分给出
            # (整段包 try: plan 缺 exec_hint/steps 或 steps 为空时不能抛 KeyError/
            #  ValueError 冲出 do_POST(连接被中断), 一律回 400 错误信息)
            try:
                _rd2 = run_dir or JOB.get("run_dir") or os.path.dirname(
                    os.path.dirname(os.path.dirname(motion_path)))
                _p14 = json.load(open(os.path.join(_rd2, "multi_station_plan.json"),
                                      encoding="utf8"))
                _n = int(_p14["n_stations"])
                _steps = [float(v) for v in (_p14["steps"] or [])]
                if not _steps:
                    raise ValueError("steps(站距)为空")
                _wl = [round(float(v), 3) for v in
                       (_p14["exec_hint"].get("walk_list") or [])]
                if len(_wl) != _n - 1:      # 步距列表长度不符: 退回等距(不静默丢弃逐站步距)
                    _wl = [round(max(_steps), 3)] * (_n - 1)
                band = _p14.get("first_band_actual") or _p14["exec_hint"]["band"]
                trig.update(stations=_n,
                            walk_dy=round(float(max(_steps)), 3),
                            walk_list=_wl,
                            walk_v=0.25, settle_s=2.5, band=band)
                log(f"▶ 多工位接力(仿真): 最少 {_n} 站覆盖整墙 "
                    f"{_p14['wall_width']:.2f}m, 智能站距 "
                    f"{min(_steps):.3f}~{max(_steps):.3f}m"
                    f"({len(_wl)} 段), 左移速度 0.25 m/s")
            except Exception as ex:
                return {"err": f"多工位执行需要 multi_station_plan.json(先以"
                               f"多工位模式完成规划), 且字段须完整"
                               f"(n_stations/steps/exec_hint): {ex}"}
        elif stations > 1:
            band = None                     # 带宽/范围从 trajectory_meta 读
            try:
                _rd2 = run_dir or JOB.get("run_dir") or os.path.dirname(
                    os.path.dirname(os.path.dirname(motion_path)))
                _meta = json.load(open(os.path.join(_rd2, "trajectory_meta.json"),
                                       encoding="utf8"))
                su = _meta.get("uniform_stroke_u") or _meta.get("stroke_u")
                band = [round(float(su[0]), 3), round(float(su[1]), 3)]
            except Exception as ex:
                return {"err": f"双工位需要 trajectory_meta 的带宽(uniform_stroke_u): {ex}"}
            if not band or band[1] - band[0] < 0.05:
                return {"err": f"带宽信息异常: {band!r}"}
            trig.update(stations=stations, walk_dy=round(band[1] - band[0], 3),
                        walk_v=0.25, settle_s=2.5, band=band)
            log(f"▶ 双工位接力(仿真): 带宽 {band[1]-band[0]:.3f}m "
                f"(墙面 u {band[0]}~{band[1]}), 左移速度 0.25 m/s")
        try:
            os.makedirs(os.path.dirname(TRIGGER_F), exist_ok=True)
            tmp = TRIGGER_F + ".tmp"     # 原子替换, 防 runner 读到半截文件
            with open(tmp, "w", encoding="utf8") as f:
                json.dump(trig, f)
            os.replace(tmp, TRIGGER_F)
        except Exception as ex:
            log(f"✘ 触发文件写入失败: {ex}")
            return {"err": f"触发文件写入失败: {ex}"}
        write_status_phase("executing")
        _n_st = trig.get("stations", 1)
        log(f"★ 已写执行触发(仿真通道, MuJoCo 播放, 速度×{speed}"
            + (f", 接力×{_n_st}站)" if _n_st and _n_st > 1 else ")")
            + f": {os.path.basename(motion_path)}")
        return {"ok": True, "motion": motion_path, "mode": "sim-trigger",
                "speed": speed, "stations": _n_st}
    mjcf = REGISTRY.get(JOB.get("robot", "g1"), {}).get("mjcf")
    if stations != 1:
        # ---- 实机多工位接力(2026-10-19): 每站上肢流 → 停稳 → SONIC 左移 → 下一站
        # 步距来源: auto=14 条带划分 walk_list; 双工位=带宽均分。行走在守护进程
        # relay 状态机内(开环时间积分, 1s 断流自停兜底), 页面 mission 实时跟进度。
        try:
            _rd3 = run_dir or JOB.get("run_dir") or os.path.dirname(
                os.path.dirname(os.path.dirname(motion_path)))
            if stations == "auto":
                _p14 = json.load(open(os.path.join(_rd3, "multi_station_plan.json"),
                                      encoding="utf8"))
                _n = int(_p14["n_stations"])
                _wl = [round(float(v), 3) for v in
                       (_p14["exec_hint"].get("walk_list") or [])]
                if len(_wl) != _n - 1:
                    _wl = [round(float(max(_p14["steps"])), 3)] * (_n - 1)
                _band = _p14.get("first_band_actual") or _p14["exec_hint"]["band"]
            else:
                _meta = json.load(open(os.path.join(_rd3, "trajectory_meta.json"),
                                       encoding="utf8"))
                _su = _meta.get("uniform_stroke_u") or _meta.get("stroke_u")
                _band = [round(float(_su[0]), 3), round(float(_su[1]), 3)]
                _n = int(stations)
                _wl = [round(_band[1] - _band[0], 3)] * (_n - 1)
        except Exception as ex:
            return {"err": f"多工位执行需要规划产物(multi_station_plan/轨迹meta): {ex}"}
        if not _band or _band[1] - _band[0] < 0.05:
            return {"err": f"带宽信息异常: {_band!r}"}
        if not _DAEMON.ensure():
            return {"err": "执行守护进程不可用 — 实机多工位接力依赖常驻守护进程"
                           "(重试一次, 仍失败则重启机器人)"}
        r = _DAEMON.call("relay", motion=motion_path, walk_list=_wl,
                         speed=speed, walk_v=_WALK_V, settle_s=2.5,
                         mjcf=mjcf, band=_band, walk_t0=_WALK_T0, timeout=20)
        if not r.get("ok"):
            if r.get("err") == "busy":
                return {"err": "任务执行中 — 请等本轮完成/进入保持, 或先停止"}
            return {"err": f"接力下发失败: {r.get('err')}"}
        _relay_update(station_idx=1, stations_total=_n, band=_band,
                      walk_dy_planned=_wl[0], walk_dy_done=0.0)
        write_status_phase("playing", progress=0.0, **_relay_snapshot())
        log(f"★ 实机多工位接力已启动: {_n} 站喷涂 + SONIC 行走"
            f"({len(_wl)} 段左移 {_wl[0]:.3f}~{_wl[-1]:.3f}m, 0.25m/s, "
            f"开环位移无反馈 — 全程人工监护, 随时【急停】)")
        return {"ok": True, "motion": motion_path, "mode": "gr00t-relay-sonic",
                "speed": speed, "stations": _n}
    if _DAEMON.ensure():
        r = _DAEMON.call("stream", motion=motion_path, rate=50, speed=speed,
                         mjcf=mjcf, timeout=20)
        if r.get("ok"):
            write_status_phase("executing")            # 界面显示"执行中"
            log(f"★ 已下发执行(官方上肢通道 17 关节@50Hz, 速度×{speed}, "
                f"结束保持末姿): {os.path.basename(os.path.dirname(motion_path))}")
            return {"ok": True, "motion": motion_path,
                    "mode": "gr00t-upperbody-official", "speed": speed}
        if r.get("err") == "busy":
            return {"err": "轨迹执行中 — 请等本轮完成/进入保持, 或先停止"}
        log(f"⚠ 守护进程下发失败({r.get('err')}), 回退官方脚本")
    # 兜底: 转官方 CSV 后直接跑官方发布器(与官方双终端等价)
    try:
        csv_path = os.path.splitext(motion_path)[0] + "_upper17.csv"
        _npz_to_upper_csv(motion_path, csv_path)
        official = os.path.join(GROOT_ROOT, "gear_sonic_deploy", "scripts",
                                "send_upper_body_trajectory.py")
        lg = open(os.path.join(BASE, "output", "spray_stream.log"), "a")
        subprocess.Popen([GROOT_PY, official, "--yes", "--csv", csv_path,
                          "--rate", "50", "--speed", str(speed)],
                         cwd=GROOT_ROOT, env={**os.environ, "GROOT_ROOT": GROOT_ROOT},
                         stdout=lg, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
        write_status_phase("executing")
        log(f"★ 已下发执行(官方脚本兜底, 速度×{speed})")
        return {"ok": True, "motion": motion_path,
                "mode": "official-script-fallback", "speed": speed}
    except Exception as ex:
        log(f"✘ 执行下发失败: {ex}")
        return {"err": f"执行下发失败: {ex}"}


def open_replay():
    """打开(必要时先生成) MuJoCo 仿真回放视频 —— 未接机器人时查看喷涂效果的主通道.
    双姿态动作的蹲姿段只在回放里完整呈现(执行通道腿由 RL 平衡, 不下蹲)."""
    with LOCK:
        if JOB.get("replay_busy"):
            return {"err": "回放生成中, 请稍候(完成后自动打开播放器)"}
        run_dir, ver = JOB.get("run_dir"), JOB.get("motion_ver")
        if not run_dir or not ver or not os.path.exists(
                os.path.join(run_dir, "ufo_motion", ver, "wall_spray_motion.npz")):
            return {"err": "尚无已生成的动作 — 请先完成轨迹与动作生成"}
        mp4 = os.path.join(run_dir, "ufo_motion", ver, "replay_motion.mp4")
        if os.path.exists(mp4):
            subprocess.Popen(["xdg-open", mp4], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            log(f"★ 已打开仿真回放: {mp4}")
            return {"ok": True, "opened": True}
        JOB["replay_busy"] = True
    threading.Thread(target=_render_replay, args=(run_dir, ver, mp4),
                     daemon=True).start()
    log("▶ 仿真回放渲染中(06 渲染 + 06b 合成, 约2~4分钟), 完成后自动打开播放器")
    return {"ok": True, "rendering": True}


def _render_replay(run_dir, ver, mp4):
    """后台渲染回放并打开; 06 用 mujoco 环境(egl 无头渲染), 06b 用 isaaclab 环境合成."""
    try:
        run_stage("06 回放渲染(MuJoCo)",
                  [MUJ_PY, "06_replay_render.py"],
                  env={"G1W_OUT": run_dir, "MOTION_VER": ver,
                       "MUJOCO_GL": "egl",
                       "DISPLAY": os.environ.get("DISPLAY", ":0")}, timeout=1800)
        run_stage("06b 回放合成(mp4+九宫格)",
                  [ISA_PY, "06b_make_replay_video.py"],
                  env={"G1W_OUT": run_dir, "MOTION_VER": ver}, timeout=600)
        subprocess.Popen(["xdg-open", mp4], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        log(f"★ 仿真回放已生成并打开: {mp4}")
    except Exception as ex:
        log(f"✘ 仿真回放生成失败: {ex}")
    finally:
        with LOCK:
            JOB["replay_busy"] = False


def _p(v):
    """registry 路径解析: 先展开 ~, 绝对原样, 相对则基于工程根(打包后随包走)."""
    v = os.path.expanduser(v) if v else v
    return v if (not v or os.path.isabs(v)) else os.path.join(BASE, v)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REGISTRY = json.load(open(os.path.join(BASE, "station", "robot_registry.json"),
                          encoding="utf8"))
for _r in REGISTRY.values():                     # 模型路径相对化(打包可移植)
    if isinstance(_r, dict):
        # 环境变量覆盖 registry 里的本机绝对路径(换机免改运行数据文件):
        #   G1W_MJCF_DIR  = mjcf xml 所在目录   G1W_MODEL_DIR = urdf 模型目录
        # 未设置时原样保留 registry 登记值作缺省回退(不改 robot_registry.json 本身)
        _mdir_env = os.environ.get("G1W_MODEL_DIR")
        if _mdir_env and _r.get("model_dir"):
            _r["model_dir"] = _mdir_env
        _mjcf_env = os.environ.get("G1W_MJCF_DIR")
        if _mjcf_env and _r.get("mjcf"):
            _r["mjcf"] = os.path.join(_mjcf_env, os.path.basename(_r["mjcf"]))
        if _r.get("mjcf"):
            _r["mjcf"] = _p(_r["mjcf"])
        if _r.get("model_dir"):
            _r["model_dir"] = _p(_r["model_dir"])

JOB = {"state": "idle", "stage": "", "t0": None, "log": [], "error": None,
       "metrics": {}, "artifacts": {}, "motion_ver": None, "robot": "g1",
       "ros_mode": None, "posture": "stand", "replay_busy": False,
       # 下面几个键一次性登记好: 后台线程写它们时不再"新增键",
       # /api/status 锁内 dict(JOB) 不会撞 "dictionary changed size during iteration"
       "run_dir": None, "waist_deg": None, "stations": None, "nn_training": False}
LOCK = threading.Lock()
PROC = None
ESTOP = False          # 急停封锁标志: /api/estop 置位; start_robot/stop_robot/退出 清除
_ROBOT_HEIGHT = False  # V2: 机器人是否以蹲姿通道(--height)启动(start_robot 设置)


STATE_F = os.path.join(BASE, "station", "run", "last_job.json")


def _persist():
    """站点重启后仍能对最近一次运行执行 导出执行包 等操作."""
    try:
        with LOCK:
            st = {k: v for k, v in JOB.items() if k != "log"}
        json.dump(st, open(STATE_F, "w", encoding="utf8"),
                  ensure_ascii=False, default=str)
    except Exception:
        pass


def log(s):
    with LOCK:
        JOB["log"].append(f"{datetime.datetime.now().strftime('%H:%M:%S')} {s}")
        JOB["log"] = JOB["log"][-400:]


def run_stage(name, cmd, env=None, timeout=600):
    """前台子进程(工作线程内), 实时日志; 失败抛异常终止任务.
    子环境白名单净化: 防止站点进程的 CONDA/PYTHON 变量串到另一 conda 环境的子解释器."""
    global PROC
    log(f"▶ {name}")
    e = {k: os.environ[k] for k in
         ("PATH", "HOME", "LANG", "LC_ALL", "TERM", "PYTHONUTF8",
          "PYTHONIOENCODING") if k in os.environ}
    e.setdefault("LANG", "C.UTF-8")
    e.update(env or {})
    p = subprocess.Popen(cmd, cwd=os.path.join(BASE, "scripts"), env=e,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    with LOCK:                 # PROC 全局句柄: 赋值纳入锁(防 /api/train_nn 与 /api/plan 并发覆盖)
        PROC = p
    for line in p.stdout:
        log("  " + line.rstrip())
    try:
        rc = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # 超时先杀再等回收, 否则子进程变孤儿继续占资源/写文件
        p.kill()
        p.wait()
        with LOCK:
            PROC = None
        raise RuntimeError(f"{name} 超时 (>{timeout}s), 已终止子进程")
    with LOCK:
        PROC = None
    if rc != 0:
        raise RuntimeError(f"{name} 失败 (exit {rc})")
    log(f"✔ {name}")


def _cancel_proc():
    """终止进行中的规划/动作子进程(terminate→kill) —— stop_robot 与 /api/estop 共用."""
    with LOCK:
        p = PROC
    try:
        if p is not None and p.poll() is None:
            p.terminate(); time.sleep(0.3)
            if p.poll() is None:
                p.kill()
            log("已取消进行中的任务")
    except Exception:
        pass


def _estop_check(where):
    """stage 线程关键步骤处的急停检查: 已急停则提前退出
    (不改 state —— 急停处理已置 idle, 不能再被改回 inspect/ready)."""
    if ESTOP:
        log(f"⚠ 急停: 已中止 {where}")
        return True
    return False


def stage_plan(cfg):
    """任务1: (点云解析)-> 03轨迹 -> 04三维图.
    注: state=planning 已由 /api/plan 在锁内置位(防快速双击并发起两个规划线程)."""
    t0 = time.time()
    try:
        run_dir = os.path.join(RUNS, "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(run_dir, exist_ok=True)
        log(f"运行目录 {run_dir}")
        crouch_lo = (REGISTRY["tiangong3"].get("crouch_lo", T3_CROUCH_LO)
                     if cfg["robot"] == "tiangong3" else
                     (G1_CROUCH_LO_DUAL if cfg.get("posture") == "dual"
                      else G1_CROUCH_LO_STAND))  # t3 蹲姿地板=腿几何下限; G1 dual 0.35
        ws_dir, ws_new = ensure_workspace(cfg["robot"], cfg["waist_deg"], crouch_lo)
        with LOCK:               # JOB 键写入纳入锁: 后台线程插键会让 /api/status
            JOB["run_dir"] = run_dir   # 锁内 dict(JOB) 撞 "dictionary changed size"
            JOB["waist_deg"] = cfg["waist_deg"]
            JOB["posture"] = cfg.get("posture", "stand")
        if cfg.get("posture") == "dual":
            log(f"▶ V2 双姿态模式(蹲姿地板 {crouch_lo}m): 站姿喷上半 -> 回零位过渡 "
                f"-> 蹲姿喷下半")
        if ws_new:
            log(f"腰限 ±{cfg['waist_deg']:.0f}°/蹲姿地板 {crouch_lo}m: 首次使用, "
                f"重算工作区(约3分钟)…")
            run_stage(f"02 工作区重算(腰±{cfg['waist_deg']:.0f}°, 蹲姿地板{crouch_lo})",
                      [ISA_PY, ("02_tiangong_workspace.py"
                                if cfg["robot"] == "tiangong3" else "02_g1_workspace.py")],
                      env={"G1W_OUT": ws_dir,
                           "G1W_WAIST_RP": str(round(cfg["waist_deg"] * math.pi / 180, 6)),
                           "G1W_CROUCH_LO": str(crouch_lo)},
                      timeout=900)
        if _estop_check("轨迹生成(点云解析前)"):
            return
        env = {"G1W_CFG": os.path.join(run_dir, "cfg.json"),
               "G1W_OUT": run_dir, "G1W_IN": ws_dir,
               "G1W_ROBOT": cfg["robot"], "G1W_MJCF": REGISTRY[cfg["robot"]]["mjcf"],
               "G1W_WAIST_RP": str(round(cfg["waist_deg"] * math.pi / 180, 6)),
               "G1W_WAIST_DEG": str(int(round(cfg["waist_deg"]))),
               "G1W_CROUCH_LO": str(crouch_lo)}
        if cfg.get("cloud"):
            run_stage("点云解析(墙面/距离/障碍)",
                      [ISA_PY, "11_cloud_scan.py", "--cloud", cfg["cloud"],
                       "--dist", str(cfg["dist"]),
                       "--out", os.path.join(run_dir, "cloud_report.json")], env)
            rep = json.load(open(os.path.join(run_dir, "cloud_report.json"), encoding="utf8"))
            cfg["wall_x"] = rep["distance"]            # 点云实测墙面距离
            cfg["stroke_u"] = rep["stroke_u"]          # 点云驱动: 喷涂带/最高行/障碍进 cfg
            cfg["wall_u_full"] = rep.get("wall_u")     # 整墙范围(多工位全覆盖的划分基准)
            cfg["h_top"] = rep["h_top"]
            cfg["obstacles"] = rep["obstacles"]
            JOB["metrics"]["cloud"] = {k: rep[k] for k in
                                       ("distance", "wall_u", "wall_h", "stroke_u",
                                        "h_top", "obstacles", "warns")}
            JOB["artifacts"]["cloud_png"] = True
        cfg["d_robot"] = cfg["dist"]                   # 无点云: 标称距离即墙面距离
        if "wall_x" not in cfg:
            cfg["wall_x"] = cfg["dist"]
        if cfg.get("posture") == "dual":
            # t3 分界 0.80(扫描定优: 站带0.75窄行归蹲后等宽 0.39->0.55); G1 维持 V2 定稿 0.65
            cfg.setdefault("crouch_top", 0.80 if cfg["robot"] == "tiangong3" else 0.65)
        # ---- 多工位全覆盖(2026-10-15 旁路新增, stations="auto"; 旧单/双工位链路不变) ----
        # 两段式: A) 以整墙为带跑 03 求最大等宽带 -> 14 条带划分(N=ceil(W/(w-2ov)),
        # 站距 s=W/N 刚好, 重叠 ov 防缝); B) cfg.stroke_u 改首站带重跑 03 -> 14
        # 用实际带宽校验防缝并出全景规划图。执行=同一动作+N 站左移(runner 现成)。
        _multi = (cfg.get("stations") == "auto")
        if _multi:
            if not cfg.get("cloud") or not cfg.get("wall_u_full"):
                raise RuntimeError("多工位全覆盖需要点云(整墙范围 wall_u) — 请先上传点云")
            if cfg["robot"] != "g1":
                raise RuntimeError("多工位全覆盖当前仅支持 G1(执行通道限制)")
            cfg["stroke_u"] = cfg["wall_u_full"]       # 阶段A: 整墙为带, 求最大等宽带
        json.dump(cfg, open(os.path.join(run_dir, "cfg.json"), "w", encoding="utf8"),
                  ensure_ascii=False, indent=1)
        if _estop_check("轨迹生成(03 规划前)"):
            return

        def _run03_equal_width(stage_name, env_extra=None):
            """03 + 外层等宽闭环(用户定稿 2026-10-15 完全等宽): 走线点有 IK 失败
            说明该带宽下存在解不动的点(旧行为直接丢点 -> 行截断/漏喷), 按带宽收紧
            3cm 重解, 直到主管线全点收敛(以真实求解结果为最终裁决), 最多 3 轮。"""
            _wmax = None
            for _att in range(4):
                _env03 = dict(env)
                _env03.update(env_extra or {})
                if _wmax is not None:
                    _env03["G1W_WMAX"] = f"{_wmax:.2f}"
                run_stage(stage_name, [ISA_PY, "03_plan_trajectories.py"], _env03)
                _meta = json.load(open(os.path.join(run_dir, "trajectory_meta.json"),
                                       encoding="utf8"))
                _sfail = int(_meta.get("stroke_ik_fail") or 0)
                _bw = _meta.get("band_width")
                if _sfail <= 0 or _bw is None:
                    return _meta
                if _att == 3 or (_bw - 0.03) < 0.30:
                    log(f"⚠ 仍有 {_sfail} 个走线点未收敛(带宽已到 {_bw}m) — "
                        f"请检查站位/腰限/点云")
                    return _meta
                _wmax = round(_bw - 0.03, 2)
                log(f"▶ 等宽闭环: 带宽 {_bw}m 有 {_sfail} 个走线点不可达, "
                    f"收紧到 {_wmax}m 重解(第 {_att + 2} 次)")
            return None

        _meta = _run03_equal_width(
            "03 轨迹规划(锚点+样条+NN种子快速IK)",
            env_extra=({"G1W_HMIN_FLOOR": "0.35"} if _multi else None))
        if _multi and _meta is not None:
            run_stage("14 多工位条带划分(整墙全覆盖)",
                      [ISA_PY, "14_multi_station_plan.py", "--run", run_dir], env)
            _p14 = json.load(open(os.path.join(run_dir, "multi_station_plan.json"),
                                  encoding="utf8"))
            log(f"★ 多工位划分: 最少 {_p14['n_stations']} 站无缝覆盖整墙 "
                f"{_p14['wall_width']:.2f}m (每站喷自己条带 "
                f"{min(_p14['steps']):.3f}~{max(_p14['steps']):.3f}m, 带宽独立贴合, "
                f"单站≥{_p14.get('s_min', 0.30)}m, 不重叠)")
            cfg["stroke_u"] = _p14["first_plan_band"]  # 阶段B: 只按首站带出动作
            json.dump(cfg, open(os.path.join(run_dir, "cfg.json"), "w", encoding="utf8"),
                      ensure_ascii=False, indent=1)
            # 蹲姿地板钳 0.35: 深蹲行全宽能力弱会拖窄带宽, 多工位优先墙宽覆盖
            _meta = _run03_equal_width("03 首站带动能重规划(多工位)",
                                       env_extra={"G1W_HMIN_FLOOR": "0.35"})
            run_stage("14 多工位终版校验+全景图",
                      [ISA_PY, "14_multi_station_plan.py", "--run", run_dir,
                       "--update"], env)
            _p14 = json.load(open(os.path.join(run_dir, "multi_station_plan.json"),
                                  encoding="utf8"))
            JOB["metrics"]["multi"] = {k: _p14[k] for k in
                                       ("n_stations", "steps", "step_s", "wall_width",
                                        "band_width", "s_min")}
        if _estop_check("轨迹生成(04 可视化前)"):
            return
        run_stage("04 三维可视化", [ISA_PY, "04_visualize_3d.py"], env)
        if _estop_check("轨迹生成收尾"):
            return
        tj = json.load(open(os.path.join(run_dir, "trajectory_meta.json"),
                            encoding="utf8"))
        JOB["metrics"]["traj"] = _traj_summary(tj)
        JOB["state"] = "inspect"                       # 待检查
        JOB["artifacts"]["plan_sec"] = round(time.time() - t0, 1)
        _persist()
        log(f"★ 轨迹已生成, 请在检查页确认 (耗时 {time.time()-t0:.1f}s)")
    except Exception as ex:
        if ESTOP:                       # 急停导致的中止: 不覆盖急停后的 idle 状态
            log(f"✘ 轨迹生成已因急停中止: {ex}")
            return
        JOB.update(state="error", error=str(ex)); log(f"✘ {ex}")


def stage_motion(cfg):
    """任务2: 05动作导出 -> 05b时序图 -> (可选)ROS topics.
    注: state=motion 已由 /api/approve 在锁内置位(防快速双击并发起两个动作线程)."""
    t0 = time.time()
    try:
        ver = f"v{'6' if cfg.get('posture') == 'dual' else '5'}_run{datetime.datetime.now().strftime('%H%M%S')}"
        with LOCK:               # JOB 键写入纳入锁(同 stage_plan)
            JOB["motion_ver"] = ver
        run_dir = JOB.get("run_dir") or RUN
        # 蹲姿地板与 plan 同源(G1_CROUCH_LO_*), 防止缓存目录按不同地板分裂
        crouch_lo = (REGISTRY.get(cfg["robot"], {}).get("crouch_lo", T3_CROUCH_LO)
                     if cfg.get("robot") == "tiangong3" else
                     (G1_CROUCH_LO_DUAL if cfg.get("posture") == "dual"
                      else G1_CROUCH_LO_STAND))
        _ws_dir, _ = ensure_workspace(cfg["robot"], cfg.get("waist_deg", 5.0), crouch_lo)
        if _estop_check("动作生成(05 导出前)"):
            return
        env = {"MOTION_VER": ver, "G1W_ROBOT": cfg["robot"],
               "G1W_MJCF": REGISTRY[cfg["robot"]]["mjcf"],
               "G1W_CFG": os.path.join(run_dir, "cfg.json"),
               "G1W_OUT": run_dir, "G1W_IN": _ws_dir,
               "G1W_WAIST_RP": str(round(cfg.get("waist_deg", 5.0) * math.pi / 180, 6)),
               "G1W_WAIST_DEG": str(int(round(cfg.get("waist_deg", 5.0)))),
               "G1W_CROUCH_LO": str(crouch_lo)}
        if cfg.get("robot") == "tiangong3":
            # 天工: 05-lite 直组装(腰臂样条+腿平面IK); 05b/基座补偿为 G1 专属, 跳过
            run_stage("05 天工动作导出(直组装+腿平面IK)",
                      [ISA_PY, "05_export_tiangong.py"], env)
        else:
            run_stage("05 动作导出(增量MJCF-IK+NN种子)", [MUJ_PY, "05_export_ufo_motion.py"], env)
            if _estop_check("动作生成(05b 时序图前)"):
                return
            run_stage("05b 时间序列图", [ISA_PY, "05b_plot_motion.py"], env)
            if _estop_check("动作生成(基座补偿验证前)"):
                return
            run_stage("基座扰动补偿验证(世界系锁定)",
                      [ISA_PY, "13_base_compensation.py",
                       "--traj", os.path.join(run_dir, "trajectory.npz"),
                       "--out", os.path.join(run_dir, "base_comp_report")], env)
        mdir = os.path.join(run_dir, "ufo_motion", ver)
        mj = json.load(open(os.path.join(mdir, "motion_meta.json"), encoding="utf8"))
        JOB["metrics"]["motion"] = _motion_summary(mj)
        JOB["metrics"]["motion"]["dir"] = mdir
        try:
            bc = json.load(open(os.path.join(run_dir, "base_comp_report.json"),
                                encoding="utf8"))
            JOB["metrics"]["base_comp"] = dict(
                nc_mm=bc["无补偿"]["世界腕心误差max_mm"],
                lock_mm=bc["世界系锁定"]["世界腕心误差max_mm"],
                ang_min=bc["世界系锁定"]["墙面夹角min_deg"],
                waist_ok=bc["世界系锁定"]["腰限位保持"],
                vel_add=bc["世界系锁定"]["附加关节速度max_rad_s"],
                ik=bc["世界系锁定"]["IK收敛"])
        except Exception as ex:
            log(f"基座补偿指标读取失败: {ex}")
        if cfg.get("ros"):
            # ROS 输出桥要读 npz(需要 numpy), 站点主进程是系统 python(无 numpy) ->
            # 子进程用 isaaclab 解释器跑, 结果落 ros_emit_result.json 回读
            res_f = os.path.join(run_dir, "ros_emit_result.json")
            try:
                run_stage("ROS topic 输出",
                          [ISA_PY, "ros_bridge.py", "--robot", cfg["robot"],
                           "--traj", run_dir, "--motion", mdir, "--out", run_dir],
                          env, timeout=300)
                r = json.load(open(res_f, encoding="utf8"))
                JOB["ros_mode"] = r["mode"]
                JOB["artifacts"]["ros"] = r
                log(f"★ ROS topics 输出完成: {r['mode']}")
            except Exception as ex:
                log(f"⚠ ROS 输出失败(不影响动作使用): {ex}")
        json.dump(dict(robot=cfg.get("robot"), dist=cfg.get("dist"),
                       pitch=cfg.get("pitch"), cloud=cfg.get("cloud"),
                       waist_deg=cfg.get("waist_deg"), stations=cfg.get("stations"),
                       motion_ver=ver, run_dir=run_dir,
                       metrics=JOB.get("metrics")),
                  open(os.path.join(run_dir, "run_manifest.json"), "w", encoding="utf8"),
                  ensure_ascii=False, indent=1, default=str)
        if _estop_check("动作打包"):
            return
        pkg = package()                      # 动作+打包一体: 完成即出执行包(manifest 已就位)
        JOB["artifacts"]["package"] = pkg
        # 动作就绪后直接下发给机器人执行(无需手动点“自动执行”)
        if AUTO_EXEC_AFTER_MOTION and _robot_pid():
            mv_npz = os.path.join(mdir, "wall_spray_motion.npz")
            r = exec_spray(mv_npz, run_dir,
                           stations=(cfg.get("stations") or JOB.get("stations")))
            if r.get("ok"):
                log("★ 动作已自动下发给机器人执行")
            else:
                log(f"自动执行未生效: {r.get('err')}")
        if _estop_check("动作生成收尾"):
            return
        JOB["state"] = "ready"
        JOB["artifacts"]["motion_sec"] = round(time.time() - t0, 1)
        _persist()
        log(f"★ 动作已生成 {ver} (耗时 {time.time()-t0:.1f}s)")
    except Exception as ex:
        if ESTOP:                       # 急停导致的中止: 不覆盖急停后的 idle 状态
            log(f"✘ 动作生成已因急停中止: {ex}")
            return
        JOB.update(state="error", error=str(ex)); log(f"✘ {ex}")


def _traj_summary(tj):
    rows = tj.get("row_params", [])
    return dict(points=tj.get("n_points"), rows=len(rows),
                z_w=[round(r["z_w"], 3) for r in rows],
                stroke_dir=tj.get("stroke_dir", "horizontal"),
                columns=len(tj.get("columns") or []),
                stroke_geom_ok=tj.get("stroke_geom_ok"),
                zones=[r.get("zone") or "stand" for r in rows],
                angles=[r["angle_deg"] for r in rows],
                stroke_u=tj.get("uniform_stroke_u"),
                posture=tj.get("posture", "stand"),
                crouch_rows=((tj.get("stations") or {}).get("crouch_rows", 0)),
                h_min_row=tj.get("h_min_row"),
                angle_range=([round(v, 1) for v in tj["angle_achieved_deg"]]
                             if tj.get("angle_achieved_deg") else None),
                angle_ok=tj.get("angle_rule_ok"),
                ik=f'{tj.get("ik_strict")}/{tj.get("n_points")} 严格档',
                contig=tj.get("q_contig_max"))


def _motion_summary(mj):
    return dict(frames=mj.get("frames"), duration_s=mj.get("duration_s"),
                max_vel=mj.get("max_joint_vel_rad_s"),
                spray_angle=mj.get("spray_angle_deg"), spray_ok=mj.get("spray_angle_ok"),
                mjcf_ik=mj.get("mjcf_ik"))


# ---------------- HTTP ----------------
class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        b = body if isinstance(body, bytes) else json.dumps(
            body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            self._send(200, PAGE_HTML.encode(), "text/html; charset=utf-8")
        elif u.path == "/inspect":
            self._send(200, INSPECT_HTML.encode(), "text/html; charset=utf-8")
        elif u.path == "/api/status":
            with LOCK:
                st = dict(JOB); st["log"] = JOB["log"][-30:]
            st["nn_training"] = bool(JOB.get("nn_training"))
            st["nn"] = _nn_status()
            st["figs"] = _figs()
            st["robot"] = _robot_status()
            st["robot_name"] = JOB.get("robot") or "g1"   # 当前产物的机器人(徽章事实来源)
            self._send(200, st)
        elif u.path == "/api/init":
            self._send(200, dict(
                robots={k: {"name": v["name"],
                            # /model/<robot>/<file>: URDFLoader 请求 mesh 时不带任何
                            # query 参数, 机器人信息只能编码进路径 —— 否则 mesh 归属
                            # 回退到 JOB 当前机器人, 换机器人后 mesh 全 404/串模型
                            "urdf": "/model/" + k + "/" + v.get("urdf_file", ""),
                            "waist_deg": round(np_deg(v.get(
                                "waist_roll_pitch_limit_rad", 0.349))),
                            "workspace_only": bool(v.get("workspace_only")),
                            "exec_ready": not bool(v.get("plan_no_policy")),
                            # 姿态选项: registry 显式声明优先, 缺省按有无策略栈推断
                            "postures": (v.get("postures")
                                         or (["stand"] if v.get("plan_no_policy")
                                             else ["stand", "dual"])),
                            "stance_band": v.get("stance_band_m", [0.70, 0.78]),
                            "ws_fig": ("/fig/" + v["workspace_fig"]
                                       if v.get("workspace_fig") else None),
                            "ws_stats": ("/fig/" + v["workspace_stats"]
                                         if v.get("workspace_stats") else None)}
                        for k, v in REGISTRY.items() if not k.startswith("_")},
                nn=_nn_status(), ros_live=_ros_live(),
                figs=_figs()))
        elif u.path.startswith("/fig/"):
            self._file(os.path.join(BASE, "output", u.path[5:]))
        elif u.path.startswith("/run/"):
            self._file(os.path.join(JOB.get("run_dir") or RUN, u.path[5:]))
        elif u.path.startswith("/static/"):
            self._file(os.path.join(BASE, "station", "static", u.path[8:]))
        elif u.path.startswith("/model/"):
            self._model_file(u.path[7:], parse_qs(u.query).get("robot", [None])[0])
        else:
            self._send(404, {"err": "not found"})

    def do_POST(self):
        global ESTOP, _T3_READY, _SESSION_ROBOT, _WALK_T0, _WALK_V
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        if u.path == "/api/upload":
            # 上传大小上限: 点云文件实际远小于此, 超限直接拒绝(防恶意/误传占满内存)
            if n > MAX_UPLOAD_BYTES:
                self.close_connection = True      # 丢弃未读完的请求体
                return self._send(413, {"err": f"文件过大: {n} 字节 "
                                              f"(上限 {MAX_UPLOAD_BYTES} 字节)"})
            fn = self._upload(n)
            return self._send(200, {"file": fn})
        body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        if u.path == "/api/plan":
            err = self._validate(body)
            if err:
                return self._send(400, {"err": err})
            cfg = dict(robot=body.get("robot", "g1"), dist=float(body["dist"]),
                       pitch=float(body.get("pitch", 0.10)),
                       waist_deg=float(body.get("waist_deg", 5.0)),
                       posture=("dual" if body.get("posture") == "dual" else "stand"),
                       h_top=body.get("h_top"), cloud=body.get("cloud"),
                       ros=bool(body.get("ros")))
            try:    # 工位模式: 1/2=固定站数, "auto"=多工位全覆盖(按点云条带划分)
                cfg["stations"] = ("auto" if body.get("stations") == "auto"
                                   else max(1, min(8, int(body.get("stations", 1)
                                                           or 1))))
            except (TypeError, ValueError):
                cfg["stations"] = 1
            with LOCK:
                if ESTOP:   # 急停悬置: 拒新任务, 否则 state 卡 planning 且后台白跑
                    return self._send(409, {"err": "急停未复位 — 请先点【停止】或【启动机器人】"})
                if JOB["state"] in ("planning", "motion"):
                    return self._send(409, {"err": "任务进行中"})
                # 锁内立即置 planning: 双击并发请求会在此被 409 拦下
                # (旧实现在新线程里才置位, 存在竞态窗口)
                JOB.update(robot=cfg["robot"], metrics={}, artifacts={},
                           state="planning", error=None, stations=cfg["stations"])
            threading.Thread(target=stage_plan, args=(cfg,), daemon=True).start()
            self._send(200, {"ok": True})
        elif u.path == "/api/approve":
            _r_ap = REGISTRY.get(JOB.get("robot", "g1"), {})
            if not _robot_pid() and not _r_ap.get("plan_no_policy"):
                return self._send(400, {"err": "机器人未启动 — 请先点【启动机器人】"})
            with LOCK:
                if ESTOP:   # 急停悬置: 拒新任务, 否则 state 卡 motion 且后台白跑
                    return self._send(409, {"err": "急停未复位 — 请先点【停止】或【启动机器人】"})
                if JOB["state"] != "inspect":
                    return self._send(409, {"err": f"当前状态 {JOB['state']}, 需先完成轨迹生成"})
                cfg = json.load(open(os.path.join(JOB.get("run_dir") or RUN,
                                                  "cfg.json"), encoding="utf8"))
                # 锁内立即置 motion: 双击并发请求会在此被 409 拦下
                # (旧实现在新线程里才置位, 存在竞态窗口)
                JOB.update(state="motion", error=None)
            threading.Thread(target=stage_motion, args=(cfg,), daemon=True).start()
            self._send(200, {"ok": True})
        elif u.path == "/api/open_replay":
            return self._send(200, open_replay())
        elif u.path == "/api/package":
            r = package()
            self._send(200, r)
        elif u.path == "/api/train_nn":
            robot = body.get("robot") or JOB.get("robot", "g1")
            if not isinstance(REGISTRY.get(robot), dict):   # REGISTRY 含 "_说明" 等非 dict 键
                return self._send(400, {"err": f"未知机器人 {robot}"})
            wd = float(body.get("waist_deg", JOB.get("waist_deg", 5.0)))
            if wd not in (5.0, 10.0, 15.0, 20.0):
                return self._send(400, {"err": f"腰限档位仅支持 5/10/15/20°, 收到 {wd}°"})
            with LOCK:
                if JOB.get("nn_training"):
                    return self._send(409, {"err": "该机器人模型训练进行中"})
                # 锁内立即置位: 双击并发请求会在此被 409 拦下
                # (旧实现在新线程里才置位, 存在竞态窗口 —— 与 /api/plan 同款问题)
                JOB["nn_training"] = True
            log(f"▶ NN-IK 重训启动: {robot} (按当前限位的工作区)")

            def _train():
                try:
                    # 蹲姿地板与 stage_plan/stage_motion **完全同源**: 训练工作区必须
                    # 与规划/执行用的工作区一致, 否则缓存目录分裂(c60 vs c35),
                    # NN 的种子落在训练分布之外; 02 脚本也按机器人选(否则 t3 会跑
                    # G1 工作区). 姿态取当前任务(JOB["posture"]), 与规划同一次
                    # 选择 —— 纯站姿时同样用 0.60, 不能固定成 dual 的 0.35.
                    _posture = JOB.get("posture", "stand")
                    crouch_lo = (REGISTRY.get(robot, {}).get("crouch_lo", T3_CROUCH_LO)
                                 if robot == "tiangong3" else
                                 (G1_CROUCH_LO_DUAL if _posture == "dual"
                                  else G1_CROUCH_LO_STAND))
                    ws_dir, ws_new = ensure_workspace(robot, wd, crouch_lo)
                    if ws_new:
                        log(f"腰限 ±{wd:.0f}°: 首次使用, 先重算工作区(约3分钟)…")
                        run_stage(f"02 工作区重算(腰±{wd:.0f}°)",
                                  [ISA_PY, ("02_tiangong_workspace.py"
                                            if robot == "tiangong3" else "02_g1_workspace.py")],
                                  env={"G1W_OUT": ws_dir,
                                       "G1W_WAIST_RP": str(round(wd * math.pi / 180, 6)),
                                       "G1W_CROUCH_LO": str(crouch_lo)},
                                  timeout=900)
                    run_stage(f"NN-IK 训练({robot}, 腰±{wd:.0f}°工作区)",
                              [ISA_PY, "12_train_nn_ik.py", "--robot", robot,
                               "--waist-deg", str(int(wd)),
                               "--n", "40000", "--epochs", "150"],
                              env=_rp_env(robot, wd, ws_dir), timeout=1200)
                    log(f"★ NN-IK 重训完成(腰±{wd:.0f}°), 模型已生效 (station/nn_ik/)")
                except Exception as ex:
                    log(f"✘ NN-IK 训练失败: {ex}")
                finally:
                    with LOCK:       # JOB 键写入纳入锁(同 stage_plan)
                        JOB["nn_training"] = False
            threading.Thread(target=_train, daemon=True).start()
            self._send(200, {"ok": True, "robot": robot})
        elif u.path == "/api/robot/start":
            try:
                return self._send(200, start_robot(
                    posture=("dual" if body.get("posture") == "dual" else "stand"),
                    robot=body.get("robot")))
            except Exception as ex:
                return self._send(500, {"err": str(ex)})
        elif u.path == "/api/estop":
            # 急停 = 官方 command(stop) → 全身变软。经守护进程即时下发,
            # 不再进程内 import estop(mujoco 环境缺 msgpack, 旧实现必失败)。
            ESTOP = True       # 封锁后台 stage 线程: 不再把 state 改回 inspect/ready
            _T3_READY = False  # 急停同时结束天工虚拟会话(如有)
            _SESSION_ROBOT = None
            try:
                if _DAEMON.alive():
                    _DAEMON.call("cancel", timeout=5)   # 停掉动作流/保持流
                done = groot_command("stop")
                if not done:                   # ZMQ 通道失败 → 官方 'O' 键兜底
                    done = send_key("O")
                # 取消进行中的规划/动作子进程(与 stop_robot 相同的 terminate+kill),
                # 并删掉待执行的仿真触发文件(急停后不得再开始播放)
                _cancel_proc()
                try:
                    os.remove(TRIGGER_F)
                except Exception:
                    pass
                # 送达校验: 官方 stop 语义 = 阻尼指令 + 部署进程优雅退出;
                # ZMQ PUB 无订阅者时"发送成功"实为丢弃 —— 必须以进程退出来确认
                if ROBOT_MODE != "real":
                    # sim 通道: runner 不订阅 ZMQ 也无 stdin FIFO, 进程退出确认
                    # 不适用 —— 以上面处置的落实为准(触发文件已删, 不会再开始新播放)
                    confirmed = not os.path.exists(TRIGGER_F)
                else:
                    confirmed = _robot_pid() is None
                    t0 = time.time()
                    while not confirmed and time.time() - t0 < 3.0:
                        time.sleep(0.2)
                        confirmed = _robot_pid() is None
                if ROBOT_MODE != "real":
                    log("★★ 网页软急停已触发(仿真通道): 执行触发已删除, "
                        "进行中的播放将在本轮轨迹结束后停止")
                else:
                    log("★★ 网页软急停已触发并确认 (机器人全身变软/阻尼保护态)"
                        if confirmed else
                        "网页软急停: ⚠ 未确认送达(部署进程仍在运行), 请按手柄急停")
                with LOCK:
                    JOB.update(state="idle", error=None, metrics={},
                               artifacts={}, motion_ver=None, run_dir=None)
                write_status_phase("soft_stopped")
                # 软急停只改状态, 保留 robot.pid —— 之后还要靠它执行"退出程序"
                if ROBOT_MODE != "real":
                    note = ("仿真急停: 已删除执行触发" if confirmed
                            else "⚠ 仿真急停未落实(触发文件仍在), 请重试或退出程序")
                else:
                    note = ("机器人已变软(阻尼保护态); 程序仍在运行" if confirmed
                            else "⚠ 急停未确认送达(部署进程未退出) —— 请立即按手柄急停!")
                return self._send(200, {"ok": True, "soft": True,
                                        "estopped": done, "confirmed": confirmed,
                                        "note": note})
            except Exception as ex:
                return self._send(500, {"err": f"急停执行异常: {ex}"})
        elif u.path == "/api/robot/restart":
            try:
                kill_deploy()                  # 官方 stop + 可靠回收(不再依赖 in-process estop)
                time.sleep(1.0)
                # robot 跟随请求体(缺省 G1): 天工会话恢复时不许误拉 G1 部署
                start_robot(robot=body.get("robot"))
                return self._send(200, {"ok": True, "restarted": True})
            except Exception as ex:
                return self._send(500, {"err": f"重启异常: {ex}"})
        elif u.path == "/api/robot/exit":
            try:
                ESTOP = False       # 全部退出: 解除急停封锁
                _T3_READY = False   # 天工虚拟会话一并结束
                _SESSION_ROBOT = None
                done = kill_deploy()           # 官方 stop → 优雅退出 → 超时强杀
                log("★★ 已退出程序 (部署进程已结束)" if done else "退出: 部署进程未能确认结束(见日志)")
                with LOCK:
                    JOB.update(state="idle", error=None, metrics={},
                               artifacts={}, motion_ver=None, run_dir=None)
                for f in (ROBOT_STATUS_F, ROBOT_PID_F, PUB_PID_F):
                    try:
                        os.remove(f)
                    except Exception:
                        pass
                return self._send(200, {"ok": True, "exited": done})
            except Exception as ex:
                return self._send(500, {"err": f"退出执行异常: {ex}"})
        elif u.path == "/api/robot/stop":
            return self._send(200, stop_robot())
        elif u.path == "/api/exec":
            # 手动执行: 把最近一次生成的动作下发给运行中的机器人(可带 speed 倍率;
            # body.motion 可指定动作 npz 覆盖 —— 供官方 v7 对照等调试用)
            mdir = JOB.get("metrics", {}).get("motion", {}).get("dir")
            mv = JOB.get("motion_ver")
            mp = body.get("motion")
            if not (mp and os.path.exists(mp)):
                mp = None
                if mdir and mv:
                    cand = os.path.join(mdir, "wall_spray_motion.npz")
                    if os.path.exists(cand):
                        mp = cand
            # motion 白名单: 只允许 本次 run 目录 与 output/ 之内的路径
            # (接口无鉴权, 不能放任意路径下发)
            if mp and not _motion_path_allowed(mp):
                log(f"⚠ 拒绝执行白名单外的动作路径: {mp}")
                return self._send(400, {"err": f"动作路径不在允许范围"
                                            f"(本次 run 目录/output): {mp}"})
            r = exec_spray(mp, JOB.get("run_dir"), speed=body.get("speed"),
                           stations=(body.get("stations") or JOB.get("stations")))
            self._send(200 if r.get("ok") else 400, r)
        elif u.path == "/api/walk":
            # 手动补走/校准: SONIC 单段位移(daemon walk 原语)。dy>0 左移 +y,
            # dy<0 右移; 量出实际位移可反算步态死区(G1W_WALK_T0)校准接力。
            if ROBOT_MODE != "real":
                return self._send(400, {"err": "补走仅实机(SONIC 运动指令通道)"})
            if not _robot_pid():
                return self._send(400, {"err": "机器人未启动 — 请先点【启动机器人】"})
            try:
                dy = float(body.get("dy", 0.05))
                v = float(body.get("v", _WALK_V))
            except (TypeError, ValueError):
                return self._send(400, {"err": f"补走参数格式错误: {body!r}"})
            if not (-0.5 <= dy <= 0.5) or not math.isfinite(dy):
                return self._send(400, {"err": f"补走位移需在 ±0.5m 内, 收到 {dy}m"})
            v = min(max(v if math.isfinite(v) else _WALK_V, 0.2), 0.8)
            if not _DAEMON.ensure():
                return self._send(400, {"err": "执行守护进程不可用 — 重试或重启机器人"})
            r = _DAEMON.call("walk", dy=dy, v=v, settle_s=1.5,
                             t0_margin=_WALK_T0, timeout=10)
            if r.get("ok"):
                log(f"▶ 手动补走: {'左移' if dy >= 0 else '右移'} {abs(dy):.3f}m"
                    f" @ {v}m/s (含步态死区裕量 {_WALK_T0:.1f}s, 人工监护)")
                self._send(200, r)
            else:
                self._send(400, r)
        elif u.path == "/api/telemetry/dump":
            # 手动抓取现场: dump 最近 120s 遥测, daemon 事件回来后自动分析出图
            if ROBOT_MODE != "real":
                return self._send(400, {"err": "遥测仅实机(5557 数据流)"})
            if not _robot_pid():
                return self._send(400, {"err": "机器人未启动 — 先【启动机器人】"})
            if not _DAEMON.ensure():
                return self._send(400, {"err": "执行守护进程不可用"})
            r = _DAEMON.call("telemetry_dump", timeout=10)
            self._send(200 if r.get("ok") else 400, r)
        elif u.path == "/api/telemetry/apply":
            # 应用现场报告的数据建议 -> station/calib.json(下次执行生效)
            try:
                sug = json.load(open(os.path.join(BASE, "station",
                                                  "calib.json"),
                                     encoding="utf8"))
            except Exception:
                sug = {}
            # 只认本次请求提供的键: 先做类型校验(非数字/非有限值直接 400),
            # 守卫不能看合并后的 calib.json —— 否则只要 calib.json 里已有
            # walk_t0, 空/垃圾请求体也会被放行并重写文件
            _req = {}
            for _k in ("walk_t0", "walk_v"):
                if body.get(_k) is None:
                    continue
                try:
                    _v = float(body[_k])
                except (TypeError, ValueError):
                    return self._send(400, {"err": f"{_k} 必须是数字, 收到 {body[_k]!r}"})
                if not math.isfinite(_v):
                    return self._send(400, {"err": f"{_k} 必须是有限数值, 收到 {body[_k]!r}"})
                _req[_k] = round(_v, 2)
            if not _req:
                return self._send(400, {"err": "无建议参数(请求未提供 walk_t0/walk_v)"})
            sug.update(_req)
            sug.setdefault("_说明", "现场报告数据建议自动写入: 下次执行生效")
            json.dump(sug, open(os.path.join(BASE, "station", "calib.json"),
                                "w", encoding="utf8"),
                      ensure_ascii=False, indent=1)
            if "walk_t0" in sug:
                try:                    # calib.json 被人手改坏时不至于 500
                    _WALK_T0 = float(sug["walk_t0"])
                except (TypeError, ValueError):
                    pass
            if "walk_v" in sug:
                try:
                    _WALK_V = float(sug["walk_v"])
                except (TypeError, ValueError):
                    pass
            log(f"★ 已按现场报告数据应用: " + ", ".join(
                f"{k}={sug[k]}" for k in ("walk_t0", "walk_v") if k in sug)
                + " (写入 station/calib.json)")
            self._send(200, {"ok": True, "calib": sug})
        else:
            self._send(404, {"err": "not found"})

    def _validate(self, b):
        r0 = REGISTRY.get(b.get("robot", "g1"))
        # 未知机器人先判: 否则 _no_policy=False 会先命中"机器人未启动",
        # "未知机器人"分支永远走不到(REGISTRY 里还有 "_说明" 等非 dict 键)
        if not isinstance(r0, dict):
            return f"未知机器人 {b.get('robot')}"
        _no_policy = bool(r0.get("plan_no_policy"))
        if not _robot_pid() and not _no_policy:
            return "机器人未启动 — 请先点【启动机器人】(策略就位后才能规划)"
        r = r0
        wd = float(b.get("waist_deg", 5))
        if wd not in (5.0, 10.0, 15.0, 20.0):
            return f"腰限档位仅支持 5/10/15/20°, 收到 {wd}°"
        if b.get("posture") not in (None, "stand", "dual"):
            return f"喷涂姿态仅支持 stand/dual, 收到 {b.get('posture')!r}"
        _posts = r0.get("postures") or (["stand"] if _no_policy else ["stand", "dual"])
        if b.get("posture") == "dual" and "dual" not in _posts:
            return f"{r0['name']} 未开放双姿态 — 请用站姿单一模式"
        d = float(b.get("dist", 0))
        lo, hi = r["stance_band_m"]
        if not (lo - 1e-9 <= d <= hi + 1e-9):
            return f"站位距离 {d:.2f}m 超出 {r['name']} 站姿可达带 [{lo},{hi}]m"
        c = b.get("cloud")
        if c and not os.path.exists(c):
            return f"点云文件不存在: {c}"
        if b.get("stations") == "auto":
            if b.get("robot", "g1") != "g1":
                return "多工位全覆盖当前仅支持 G1"
            if not c:
                return "多工位全覆盖需要点云(墙宽) — 请先上传点云再规划"
        return None

    def _upload(self, n):
        ctype = self.headers.get("Content-Type", "")
        raw = self.rfile.read(n)
        fn = None
        if "boundary=" in ctype:
            bnd = ctype.split("boundary=")[1].encode()
            parts = raw.split(bnd)
            for p in parts:
                if b"filename=" in p:
                    head, _, data = p.partition(b"\r\n\r\n")
                    fn = head.split(b"filename=")[1].split(b'"')[1].decode()
                    data = data.rsplit(b"\r\n", 1)[0]
                    if len(data) < 1024:
                        log(f"⚠ 上传文件过小({len(data)}B), 已拒绝: {fn}")
                        return None
                    if data[:3] != b"ply" and not fn.lower().endswith((".npy", ".txt", ".csv", ".pcd")):
                        log(f"⚠ 上传文件不像点云(头={data[:16]!r}), 已拒绝: {fn}")
                        return None
                    dst = os.path.join(UP, os.path.basename(fn) or "cloud.ply")
                    open(dst, "wb").write(data)
                    break
        return dst if fn else None

    def _model_file(self, rel, robot=None):
        """机器人模型静态文件: 白名单=registry登记的 model_dir, 防目录穿越.
        路径形如 /model/<robot>/<file>(机器人编码在首段, mesh 请求可靠解析归属);
        兼容旧形式 /model/<file>?robot=<key>(query), 最后才回退 JOB 当前机器人.
        机器人必须是 registry 里真正的 dict 条目(REGISTRY 还有 "_说明" 等说明键,
        命中会取到字符串 -> info["model_dir"] 抛 TypeError -> 500);
        未知机器人一律 400, 不能静默换成 JOB 的机器人(否则请求不存在的机器人
        却返回别的机器人的 mesh)."""
        import urllib.parse
        rel = urllib.parse.unquote(rel)
        parts = rel.split("/", 1)              # 新形式: <robot>/<file>
        _seg = parts[0] if len(parts) == 2 else None
        if _seg is not None and _seg in REGISTRY:
            robot, rel = parts[0], parts[1]
        if robot is not None:
            info = REGISTRY.get(robot)
            if not isinstance(info, dict):
                return self._send(400, {"err": f"未知机器人 {robot}"})
        else:
            # 无机器人线索: 回退 JOB 当前机器人(再退第一个登记条目)
            info = REGISTRY.get(JOB.get("robot", "g1"))
            if not isinstance(info, dict):
                info = next(v for k, v in REGISTRY.items() if not k.startswith("_"))
        base = os.path.realpath(info["model_dir"])
        p = os.path.realpath(os.path.join(base, rel))
        if not p.startswith(base + os.sep) and p != base:
            return self._send(403, {"err": "forbidden"})
        if not os.path.exists(p):
            if _seg is not None and robot is None:
                # 首段不是登记的机器人名, 且按回退机器人也找不到文件 -> 未知机器人
                return self._send(400, {"err": f"未知机器人 {_seg}"})
            return self._send(404, {"err": "no file"})
        ct = {".urdf": "application/xml", ".xml": "application/xml",
              ".STL": "application/octet-stream", ".stl": "application/octet-stream",
              ".dae": "model/vnd.collada+xml", ".json": "application/json",
              ".png": "image/png"}.get(os.path.splitext(p)[1], "application/octet-stream")
        self._send(200, open(p, "rb").read(), ct)

    def _file(self, p):
        """静态产物白名单: realpath 解析后做 base+os.sep 前缀判断
        (防 ../ 与同前缀目录绕过, 写法与 _motion_path_allowed 一致)."""
        rp = os.path.realpath(p)
        for base in (os.path.join(BASE, "output"), RUN,
                     os.path.join(BASE, "station", "static"),
                     os.path.join(BASE, "runs")):
            rb = os.path.realpath(base)
            if rp == rb or rp.startswith(rb + os.sep):
                p = rp
                break
        else:
            return self._send(404, {"err": "no file"})
        if not os.path.exists(p):
            return self._send(404, {"err": "no file"})
        ext = os.path.splitext(p)[1]
        ct = {".png": "image/png", ".json": "application/json",
              ".csv": "text/csv", ".md": "text/markdown"}.get(ext, "application/octet-stream")
        self._send(200, open(p, "rb").read(), ct)


def np_deg(x):
    import math
    return math.degrees(x or 0.349)


WS_CACHE = os.path.join(BASE, "output", "ws_cache")


def ensure_workspace(robot, waist_deg, crouch_lo=0.60):
    """按(腰限角度, 蹲姿地板)缓存工作区; 首次遇到该组合重算 02(约3分钟), 之后秒级命中.
    V2 双姿态: 蹲姿地板 0.40(实测 height_sweep 策略可稳到 0.29, 留臂负载裕量).
    返回 (ws_dir, created)."""
    d = os.path.join(WS_CACHE,
                     f"{robot}_rp{int(waist_deg)}_c{int(round(crouch_lo * 100))}")
    os.makedirs(d, exist_ok=True)
    _ws_npz = ("tiangong_workspace.npz" if robot == "tiangong3"
               else "g1_workspace.npz")
    if not os.path.exists(os.path.join(d, _ws_npz)):
        for f in ("wall_section.npz", "wall_section_meta.json"):
            src = os.path.join(BASE, "output", f)
            dst = os.path.join(d, f)
            if os.path.exists(src) and not os.path.exists(dst):
                os.symlink(src, dst)          # 舱室静态场景与腰限无关
        return d, True
    return d, False


def _rp_env(robot, waist_deg, ws_dir=None):
    import math
    rad = round(float(waist_deg) * math.pi / 180, 6)
    e = {"G1W_ROBOT": robot, "G1W_WAIST_RP": str(rad),
         "G1W_WAIST_DEG": str(int(round(float(waist_deg))))}
    if ws_dir:
        _ws = ("tiangong_workspace.npz" if robot == "tiangong3"
               else "g1_workspace.npz")
        e["G1W_WS"] = os.path.join(ws_dir, _ws)
        e["G1W_IN"] = ws_dir
    return e


def _nn_status():
    d = os.path.join(BASE, "station", "nn_ik")
    out = {}
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith("_nn_ik.npz"):
                r = f[:-len("_nn_ik.npz")]
                sj = os.path.join(d, f.replace(".npz", "_stats.json"))
                st = json.load(open(sj, encoding="utf8")) if os.path.exists(sj) else {}
                out[r] = {"deployed": True, "n_data": st.get("n_data"),
                          "eval": st.get("eval"), "model": f}
    return out


def _ros_live():
    try:
        import rospy            # noqa
        return True
    except Exception:
        return False


def _figs():
    rd = JOB.get("run_dir")
    out = []
    for f in ("fig_multi_station.png", "fig_wall_path_2d.png", "fig_robot_path_2d.png",
              "fig_3d_scene.png", "cloud_report.png", "base_comp_report.png"):
        if rd and os.path.exists(os.path.join(rd, f)):
            out.append({"name": f, "url": "/run/" + f})
        elif os.path.exists(os.path.join(BASE, "output", f)):
            out.append({"name": f, "url": "/fig/" + f})
        elif os.path.exists(os.path.join(RUN, f)):
            out.append({"name": f, "url": "/run/" + f})
    return out


def package():
    # t3 等新条目可不登记 motion_ver(动作版本只由当次会话产生), 缺省回退 g1 的旧值
    ver = JOB.get("motion_ver") or REGISTRY.get(
        JOB.get("robot", "g1"), {}).get("motion_ver") or "v5"
    rd = JOB.get("run_dir") or RUN
    mdir = os.path.join(rd, "ufo_motion", ver)
    stamp = datetime.datetime.now().strftime("%H%M%S")
    dname = f"station_run_{stamp}"
    dst = os.path.join(BASE, "deploy", dname)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(os.path.join(dst, "motion"), exist_ok=True)
    os.makedirs(os.path.join(dst, "trajectory"), exist_ok=True)
    os.makedirs(os.path.join(dst, "figures"), exist_ok=True)
    for f in ("wall_spray_motion.npz", "wall_spray_motion_qpos.csv",
              "motion_meta.json", "replay_motion.mp4", "fig_replay_contactsheet.png",
              "tiangong3_motion_real.csv", "tiangong3_motion_real.json",
              "tiangong3_motion_upper.csv"):  # t3 真机指令流(全身版+上身通道版)
        p = os.path.join(mdir, f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(dst, "motion", f))
    for f in ("wall_path.csv", "robot_path.csv", "trajectory_meta.json",
              "trajectory_points.ply", "run_manifest.json"):
        p = os.path.join(rd, f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(dst, "trajectory", f))
    for it in _figs():
        src = os.path.join(rd, it["name"]) if it["url"].startswith("/run/") \
            else os.path.join(BASE, "output", it["name"])
        if os.path.exists(src):
            shutil.copy(src, os.path.join(dst, "figures", it["name"]))
    # 世界系锁定执行端: 实时补偿节点 + 验证报告 + 接线说明
    exec_src = os.path.join(BASE, "scripts", "exec", "base_lock_node.py")
    if os.path.exists(exec_src):
        os.makedirs(os.path.join(dst, "exec"), exist_ok=True)
        shutil.copy(exec_src, os.path.join(dst, "exec", "base_lock_node.py"))
    for f in ("base_comp_report.json", "base_comp_report.png"):
        pth = os.path.join(rd, f)
        if os.path.exists(pth):
            shutil.copy(pth, os.path.join(dst, "trajectory", f))
    open(os.path.join(dst, "执行说明-世界系锁定.md"), "w", encoding="utf8").write(
        "# 世界系锁定执行(默认方案)\n\n"
        "喷涂目标锁定在世界系(trajectory/wall_path.csv)。执行时实时读基座位姿,\n"
        "反投影到当前基座系后由上身IK跟踪 —— 机器人轻微移动时腕/枪不随基座偏。\n\n"
        "1. 补偿节点: `exec/base_lock_node.py`\n"
        "   - 订阅 `/odom`(G1 状态估计, 基座位姿)\n"
        "   - 30Hz 发布 `/{robot}_spray/wrist_target_base`(基座系腕心目标+枪轴四元数)\n"
        "   - 超可达带(>6cm)发布 `comp_status=HOLD`, 上层应停喷\n"
        "   - 自测: `python exec/base_lock_node.py --selftest`\n"
        "2. 验证报告: trajectory/base_comp_report.{json,png}\n"
        "   (平移±3cm/偏航±1.5°/倾±0.6° 扰动: 腕心误差无补偿→锁定, 夹角≥60°, 腰±5°)\n"
        "3. motion/wall_spray_motion.npz 为标称轨迹(UFO/MimicLite 预热/回放用);\n"
        "   真机喷涂以补偿节点输出为准。\n")
    ros = JOB.get("artifacts", {}).get("ros")
    if ros and ros.get("dir"):
        shutil.copytree(ros["dir"], os.path.join(dst, "ros_topics"),
                        dirs_exist_ok=True)
    elif rd and os.path.isdir(os.path.join(rd, "ros_topics")):
        shutil.copytree(os.path.join(rd, "ros_topics"),
                        os.path.join(dst, "ros_topics"), dirs_exist_ok=True)
    z = shutil.make_archive(os.path.join(BASE, "deploy", dname), "zip", dst)
    log(f"★ 执行包 {z}")
    return {"zip": z, "dir": dst, "motion_ver": ver,
            "exec_api": "POST /api/exec {robot, motion_ver} (预留)"}


# ---------------- 页面 ----------------
PAGE_HTML = """<!doctype html><html><head><meta charset=utf-8><title>G1 喷涂轨迹工作站</title>
<style>
:root{--bg0:#0a0b10;--bg1:#101218;--bg2:#161923;--line:#23252e;--line2:#34374a;
--text:#f2f4f8;--text2:#a9b1bf;--text3:#6e7687;--brand:#0070f3;--brand2:#0060d0;
--violet:#5e6ad2;--teal:#14b8a6;
--ok:#29d391;--warn:#f5a623;--err:#e5484d;--ease:cubic-bezier(.2,.8,.2,1)}
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei","Noto Sans CJK SC",sans-serif;
margin:0;background:var(--bg0);color:var(--text);font-size:14px;line-height:1.55;-webkit-font-smoothing:antialiased;
overflow-x:hidden}
body::before{content:"";position:fixed;inset:-30%;z-index:-2;pointer-events:none;
background:
radial-gradient(26% 22% at 16% 12%,rgba(0,112,243,.14),rgba(0,112,243,0) 70%),
radial-gradient(24% 24% at 86% 14%,rgba(94,106,210,.12),rgba(94,106,210,0) 70%),
radial-gradient(24% 26% at 74% 88%,rgba(20,184,166,.08),rgba(20,184,166,0) 70%),
radial-gradient(20% 20% at 24% 88%,rgba(94,106,210,.07),rgba(94,106,210,0) 70%);
will-change:transform;
animation:drift 34s ease-in-out infinite alternate}
body::after{content:"";position:fixed;inset:0;z-index:-1;pointer-events:none;opacity:.5;
background:linear-gradient(rgba(255,255,255,.022) 1px,transparent 1px),
linear-gradient(90deg,rgba(255,255,255,.022) 1px,transparent 1px);
background-size:44px 44px;
mask-image:radial-gradient(70% 60% at 50% 30%,#000 30%,transparent 100%)}
@keyframes drift{0%{transform:translate3d(-2%,-1.5%,0) scale(1) rotate(0deg)}
50%{transform:translate3d(1.5%,2.5%,0) scale(1.06) rotate(2deg)}
100%{transform:translate3d(3%,1%,0) scale(1.1) rotate(-1.5deg)}}

@keyframes textflow{from{background-position:0% 50%}to{background-position:200% 50%}}
@keyframes borderflow{from{background-position:0% 50%}to{background-position:300% 50%}}
header{padding:0 22px;height:52px;display:flex;gap:14px;align-items:center;background:rgba(10,10,10,.8);backdrop-filter:blur(10px);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:10}
h1{font-size:16.5px;font-weight:700;margin:0;letter-spacing:.3px;
background:linear-gradient(90deg,#e8f0ff,#8ec2ff,var(--violet),#7ee8d0,#e8f0ff);
background-size:200% 100%;-webkit-background-clip:text;background-clip:text;color:transparent;
animation:textflow 7s linear infinite}
.branddot{width:8px;height:8px;border-radius:99px;background:var(--brand);box-shadow:0 0 10px var(--brand)}
#bg3d{position:fixed;inset:0;z-index:-1;pointer-events:none;opacity:.6}
#steps{display:flex;gap:26px;padding:12px 24px 0;max-width:1500px;margin:0 auto;width:100%}
.step{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--text3);transition:color .2s}
.step+.step::before{content:"";width:34px;height:2px;margin-right:14px;border-radius:2px;
background:linear-gradient(90deg,rgba(255,255,255,.10),rgba(255,255,255,.22),rgba(255,255,255,.10));
background-size:200% 100%;animation:borderflow 3s linear infinite}
.step .dot{width:10px;height:10px;border-radius:99px;background:#232326;transition:background .2s}
.step.active{color:var(--warn)} .step.active .dot{background:var(--warn);animation:pulse 1.2s infinite}
.step.done{color:var(--ok)} .step.done .dot{background:var(--ok)}
main{display:grid;grid-template-columns:400px 1fr;gap:16px;padding:14px 22px 28px;max-width:1500px;margin:auto}
header{flex-wrap:wrap;height:auto;min-height:52px;padding:8px 22px}
header .badge{margin-top:2px;margin-bottom:2px}
#ws_view img{max-height:210px;width:100%;object-fit:contain;border-radius:8px;background:#fff}
.card{position:relative;background:rgba(19,21,30,.62);
border:1px solid rgba(255,255,255,.085);border-radius:16px;padding:18px;
box-shadow:0 12px 40px rgba(0,0,0,.45),0 1px 0 rgba(255,255,255,.05) inset;
animation:rise .45s var(--ease) both;transition:box-shadow .25s,transform .25s var(--ease)}
.card::before{content:"";position:absolute;inset:0 0 auto;height:1px;border-radius:16px 16px 0 0;
background:linear-gradient(90deg,transparent,rgba(255,255,255,.14),transparent)}
.card::after{content:"";position:absolute;inset:-1px;border-radius:17px;padding:1.2px;pointer-events:none;
background:linear-gradient(120deg,rgba(0,112,243,.6),rgba(94,106,210,.45),rgba(20,184,166,.45),rgba(0,112,243,.6));
background-size:300% 300%;animation:borderflow 4s linear infinite;
-webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);
-webkit-mask-composite:xor;mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);
mask-composite:exclude;opacity:0;transition:opacity .3s var(--ease)}
.card:hover{box-shadow:0 18px 54px rgba(0,0,0,.5),0 0 30px rgba(0,112,243,.12);
transform:translateY(-2px)}
.card:hover::after{opacity:1}
.card-title{display:flex;align-items:center;gap:8px;font-size:14px;font-weight:650;letter-spacing:.3px}
.card-title .ticon{width:7px;height:7px;border-radius:99px;background:var(--brand);
box-shadow:0 0 12px var(--brand);animation:pulse 2.4s infinite}
header{background:rgba(10,10,12,.62);border-bottom:1px solid rgba(255,255,255,.06)}
.card:nth-child(2){animation-delay:.06s}
label{display:block;margin:10px 0 4px;font-size:12.5px;color:var(--text2)}
input,select{width:100%;padding:8px 10px;background:var(--bg2);border:1px solid var(--line);border-radius:8px;color:var(--text);font-size:13px;transition:border-color .15s,box-shadow .15s}
input:focus,select:focus{outline:0;border-color:var(--brand);box-shadow:0 0 0 3px rgba(0,112,243,.18)}
button{width:100%;margin-top:10px;padding:11px;border:0;border-radius:8px;font-size:14.5px;font-weight:600;cursor:pointer;color:#fff;
transition:transform .15s var(--ease),box-shadow .15s,filter .15s}
.sec{margin:14px 0 2px;font-size:11px;font-weight:700;letter-spacing:1.2px;color:var(--text3);
text-transform:uppercase;border-bottom:1px solid var(--line);padding-bottom:4px}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.chk{display:flex;align-items:center;gap:8px;margin:12px 0 0;font-size:13px;color:var(--text2);cursor:pointer}
.chk input{width:auto}
.chip{display:inline-flex;align-items:center;justify-content:center;min-width:30px;height:19px;
padding:0 5px;margin-left:4px;border-radius:6px;font-size:11px;font-weight:600;
background:rgba(255,255,255,.05);color:var(--text3);border:1px solid var(--line)}
.chip.on{color:#0b1510;background:var(--ok);border-color:transparent}
.chip.cur{outline:2px solid rgba(41,211,145,.55);outline-offset:1px}
#logbox summary{cursor:pointer;color:var(--text2);font-size:12.5px;padding:2px 0 6px;user-select:none}
#loglog{max-height:340px;overflow-y:auto;scrollbar-width:thin;white-space:pre-wrap;word-break:break-all}
#b1{position:relative;overflow:hidden;padding:14px;font-size:15.5px;
background:linear-gradient(180deg,#1478f8,#0060d0);
box-shadow:0 1px 0 rgba(255,255,255,.15) inset,0 4px 18px rgba(0,112,243,.4)}
#b1::after{content:"";position:absolute;inset:0;pointer-events:none;
background:linear-gradient(115deg,transparent 30%,rgba(255,255,255,.28) 48%,transparent 62%);
background-size:220% 100%;animation:shine 3.2s linear infinite}
#b1:hover{filter:brightness(1.1);transform:translateY(-1px)}
#b2{background:linear-gradient(180deg,#14b8c4,#0e7490);box-shadow:0 1px 0 rgba(255,255,255,.14) inset,0 4px 18px rgba(14,116,144,.35)}
#b2:hover{filter:brightness(1.12);transform:translateY(-1px)}
.go{background:linear-gradient(180deg,#34e39b,#0fa968);
box-shadow:0 1px 0 rgba(255,255,255,.2) inset,0 6px 26px rgba(41,211,145,.4)}
.go:hover{filter:brightness(1.1);transform:translateY(-1px)}
.stop{background:linear-gradient(180deg,#f2585f,#c22f38);
box-shadow:0 1px 0 rgba(255,255,255,.18) inset,0 6px 26px rgba(229,72,77,.42)}
.stop:hover{filter:brightness(1.1);transform:translateY(-1px)}
/* 禁用态统一置灰: 覆盖 id 级彩色背景(优先级) */
#b0:disabled,#b1:disabled,#b2:disabled,#bApprove:disabled,#b3:disabled,
button:disabled{background:#1a1d22 !important;color:#5b6b7e !important;
box-shadow:none !important;filter:none !important;transform:none !important;
border-color:#262a31 !important;cursor:not-allowed !important}
#b0:disabled::after,#b1:disabled::after,#b2:disabled::after{display:none !important}
/* 急停: 顶部常驻, 始终可点 */
#estopbar{position:fixed;top:0;left:0;right:0;z-index:99;height:0;overflow:visible}
#estopExit{position:fixed;top:10px;right:240px;z-index:100;width:auto;padding:12px 16px;
margin:0;font-size:13px;font-weight:700;color:#fff;background:linear-gradient(180deg,#7f1d1d,#5b1414);
border:1px solid #a33;border-radius:10px;cursor:pointer}
#estopExit:hover{filter:brightness(1.15)}
#estop{position:fixed;top:10px;right:18px;z-index:100;width:auto;padding:12px 22px;
margin:0;font-size:15px;font-weight:800;letter-spacing:1px;color:#fff;
background:linear-gradient(180deg,#ff4d4d,#c81e1e);border:2px solid #ff8080;
border-radius:10px;box-shadow:0 6px 28px rgba(229,72,77,.6),0 0 0 3px rgba(229,72,77,.22);
animation:estopPulse 1.8s ease-in-out infinite;cursor:pointer}
#estop:hover{filter:brightness(1.12);transform:translateY(-1px);animation:none}
#estop:active{transform:scale(.97)}
@keyframes estopPulse{0%,100%{box-shadow:0 6px 28px rgba(229,72,77,.5),0 0 0 3px rgba(229,72,77,.18)}
50%{box-shadow:0 6px 34px rgba(229,72,77,.8),0 0 0 6px rgba(229,72,77,.3)}}
#estop .dot{display:inline-block;width:9px;height:9px;border-radius:99px;background:#fff;
margin-right:8px;vertical-align:1px;animation:pulse 1s infinite}
#estop.soft{background:linear-gradient(180deg,#ffb020,#e08a00);border-color:#ffd070;
box-shadow:0 6px 28px rgba(255,176,32,.5);animation:none}
#b2.go{position:relative;overflow:hidden;
background:linear-gradient(180deg,#34e39b,#0fa968);
box-shadow:0 1px 0 rgba(255,255,255,.2) inset,0 6px 26px rgba(41,211,145,.4)}
#b2.go::after{content:"";position:absolute;inset:0;pointer-events:none;
background:linear-gradient(115deg,transparent 32%,rgba(255,255,255,.3) 48%,transparent 64%);
background-size:220% 100%;animation:shine 3.6s linear infinite}
.toast{position:fixed;right:22px;bottom:22px;z-index:98;max-width:460px;
background:rgba(18,20,26,.9);backdrop-filter:blur(10px);border:1px solid rgba(245,166,35,.5);
border-radius:12px;padding:12px 16px;font-size:13px;color:#f2f4f8;
box-shadow:0 12px 40px rgba(0,0,0,.5);animation:rise .3s var(--ease) both}
#b3,.ghost{background:rgba(255,255,255,.045);border:1px solid rgba(255,255,255,.12);color:var(--text);backdrop-filter:blur(8px);font-size:13px;padding:9px}
.ghost:hover{border-color:rgba(255,255,255,.22);transform:translateY(-1px)}
#b3:hover{transform:translateY(-1px);border-color:#3f3f46}
button:disabled{background:#17171a;color:#555;box-shadow:none;transform:none;cursor:not-allowed;filter:none}
#bApprove::after{content:"";position:absolute;inset:0;pointer-events:none;
background:linear-gradient(115deg,transparent 32%,rgba(255,255,255,.3) 48%,transparent 64%);
background-size:220% 100%;animation:shine 3.6s linear infinite}
#bApprove:hover{filter:brightness(1.1);transform:translateY(-1px);box-shadow:0 6px 30px rgba(41,211,145,.55)}
button:active:not(:disabled){transform:translateY(0) scale(.995)}
.btn-busy{background-size:200% 100%;animation:shine 1.4s linear infinite}
pre{background:var(--bg2);border:1px solid var(--line);border-radius:10px;padding:10px;height:280px;overflow:auto;
font-family:"JetBrains Mono","SF Mono",Consolas,"Courier New",monospace;font-size:11.5px;line-height:1.5;white-space:pre-wrap;color:#c8ccd4}
.badge{display:inline-flex;align-items:center;padding:2px 10px;border-radius:99px;font-size:12px;background:var(--bg2);border:1px solid var(--line);margin-right:6px;color:var(--text2)}
.ok{color:var(--ok)} .warn{color:var(--warn)} .err{color:var(--err)}
#state{font-size:13.5px;font-weight:600;margin:4px 0 2px;display:flex;align-items:center;gap:8px}
.busy-dot{width:8px;height:8px;border-radius:99px;background:var(--warn);animation:pulse 1.2s infinite;display:inline-block}
.runline{font-size:11.5px;color:var(--text3);margin-bottom:4px}
.runline code{font-family:Consolas,monospace;color:var(--text2)}
table{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:10px}
td,th{border:1px solid var(--line);padding:5px 9px;text-align:left}
th{color:var(--text2);font-weight:500;background:var(--bg2)}
.num{font-variant-numeric:tabular-nums;font-feature-settings:"tnum"}
.sumrow{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 4px}
.sumitem{background:var(--bg2);border:1px solid var(--line);border-radius:10px;padding:8px 13px;font-size:12.5px;color:var(--text2);display:flex;align-items:center;gap:7px;animation:rise .35s var(--ease) both}
.sumitem b{font-size:14.5px;color:var(--text);font-weight:650}
.figgrid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px}
.figcell{border-radius:10px;overflow:hidden;background:#fff;border:1px solid var(--line);cursor:zoom-in;position:relative;
transition:transform .18s var(--ease),box-shadow .18s,border-color .18s;animation:rise .35s var(--ease) both}
.figcell:hover{transform:translateY(-4px);border-color:rgba(0,112,243,.55);
box-shadow:0 14px 34px rgba(0,0,0,.5),0 0 22px rgba(0,112,243,.18)}
.figcell img{width:100%;height:290px;object-fit:contain;display:block;opacity:0;transition:opacity .4s ease}
.figcell img.ready{opacity:1}
.figcell .zoom{position:absolute;right:8px;bottom:34px;width:28px;height:28px;border-radius:8px;background:rgba(10,10,10,.78);
display:flex;align-items:center;justify-content:center;opacity:0;transition:opacity .18s}
.figcell:hover .zoom{opacity:1}
.figcap{font-size:13.5px;color:#e9eef6;font-weight:650;letter-spacing:.4px;padding:10px 12px;text-align:center;
background:linear-gradient(180deg,rgba(255,255,255,.055),rgba(255,255,255,.015));border-top:1px solid rgba(255,255,255,.10);
text-shadow:0 1px 8px rgba(0,112,243,.25)}
#lightbox{display:none;position:fixed;inset:0;background:rgba(4,5,8,.72);backdrop-filter:blur(14px);
z-index:99;align-items:center;justify-content:center;cursor:zoom-out;animation:lbIn .25s ease-out}
#lightbox img{max-width:96vw;max-height:96vh;border-radius:10px;animation:lbImg .2s ease-out}
.badge-ok,.badge-bad{display:inline-flex;align-items:center;justify-content:center;width:17px;height:17px;border-radius:99px;flex:0 0 17px}
.badge-ok{background:var(--ok)}
.badge-ok::after{content:"";width:8px;height:4px;border-left:2px solid #04120b;border-bottom:2px solid #04120b;transform:rotate(-45deg) translate(0,-1px)}
.badge-bad{background:var(--err)}
.badge-bad::after{content:"";width:9px;height:9px;background:
linear-gradient(45deg,transparent 42%,#2a0a0d 42%,#2a0a0d 58%,transparent 58%),
linear-gradient(-45deg,transparent 42%,#2a0a0d 42%,#2a0a0d 58%,transparent 58%)}
#rview{position:relative;margin-top:8px;border:1px solid var(--line);border-radius:10px;background:linear-gradient(#131316,#0c0c0e);height:250px;overflow:hidden}
#rcanvas{width:100%;height:100%;display:block}
#rhint{position:absolute;left:10px;bottom:8px}
.small{font-size:12px;color:var(--text3)}
details summary{cursor:pointer;color:var(--text2)}
@keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
@keyframes pulse{0%,100%{opacity:.5}50%{opacity:1}}
@keyframes shine{from{background-position:-200% 0}to{background-position:200% 0}}
@keyframes lbIn{from{opacity:0}to{opacity:1}}
@keyframes lbImg{from{opacity:0;transform:scale(.96)}to{opacity:1;transform:scale(1)}}
::-webkit-scrollbar{width:10px;height:10px}
::-webkit-scrollbar-thumb{background:#26262a;border-radius:99px;border:3px solid var(--bg1)}
::-webkit-scrollbar-track{background:transparent}
</style></head><body>
<header><span class=branddot></span><h1>喷涂轨迹工作站</h1><span id=robot_badge class=badge>机器人: -</span>
<span id=nn_badge class=badge>NN-IK: -</span><span id=ros_badge class=badge>ROS: -</span></header>
<canvas id=bg3d></canvas>
<button id=estopExit onclick=doExitRobot() title="退出程序(结束部署进程)">退出程序</button>
<button id=estop onclick=doEstop() title="软急停: 机器人全身变软(阻尼保护), 程序继续运行">
<span class=dot></span>急停 SOFT STOP</button>
<div id=steps></div>
<script src="/static/three.min.js"></script>
<script src="/static/STLLoader.js"></script>
<script src="/static/URDFLoader.min.js"></script>
<script src="/static/OrbitControls.js"></script>
<main>
<div class=card>
  <div class=card-title><span class=ticon></span>① 输入</div>
  <div class=sec>机器人</div>
  <button onclick=doStartRobot() id=b0 class=go style="padding:14px;font-size:15.5px">启动机器人</button>
  <div id=robot_stat class=small style="margin-top:6px">机器人未启动</div>
  <select id=robot onchange="onRobotChange(this.value)"></select>
  <div id=ws_view style="display:none;margin-top:10px"></div>
  <div id=rview><canvas id=rcanvas></canvas><div id=rhint class=small>3D 模型: 拖动旋转 / 滚轮缩放</div></div>
  <div class=sec>作业参数</div>
  <div id=gated_inputs>
  <div class=row2>
    <div><label>站位离墙 (m)</label>
      <input id=dist type=number step=0.01 min=0.70 max=1.00 value=0.74 onchange=markStale() title="G1 标称 0.74 / 天工标称 0.90"></div>
    <div><label>行距 (m)</label>
      <input id=pitch type=number step=0.01 value=0.10 onchange=markStale()></div>
  </div>
  <div class=row2 style="margin-top:10px">
    <div><label>腰 roll/pitch 限位档位 (即选即用)</label>
      <select id=waist onchange="markStale();renderChips()"></select></div>
    <div><label>喷涂姿态 (V2 新增)</label>
      <select id=posture onchange="markStale();localStorage.setItem('spray_posture',this.value)">
        <option value=stand>站姿单一 (v5 · 0.55~1.45m)</option>
        <option value=dual>站+蹲双姿态 (V2 · ≈0.20~1.5m · 先站后蹲)</option>
      </select></div>
  </div>
  <div id=crouch_hint class=small style="margin-top:4px;color:var(--warn)">双姿态执行需以本模式点【启动机器人】(自动启用蹲姿通道: 骨盆高→height_cmd, 策略下蹲)</div>
  <div id=station_wrap>
  <label style="margin-top:8px">工位模式 (执行时可随时切换)</label>
  <select id=station_mode onchange="localStorage.setItem('spray_stations',this.value)" title="带宽由等宽保证自动收窄; 双工位=喷完工位①后朝向不变左移一个带宽, 接着喷工位②; 多工位=按点云墙宽自动划分 N 站全覆盖(需点云; 实机=SONIC 行走接力, 开环位移人工监护)">
    <option value=1>单工位 · 一条等宽带喷完</option>
    <option value=2>双工位接力 · 喷①→左移一带宽→喷②</option>
    <option value=auto>多工位全覆盖 · 按点云自动 N 站(需点云)</option>
  </select>
  <div class=small style="margin-top:4px">等宽带宽度由"每行全程可达"自动求解(完全等宽, 不截断行); 更宽墙面选双工位, 或用点云+多工位全自动划分 N 站接力覆盖整墙</div>
  </div>
  <div class=sec>场景输入</div>
  <label>点云图 (可选 · 墙面拟合+障碍自动避让)</label>
  <input id=cloud type=file accept=.ply,.pcd,.npy,.txt,.csv>
  <div id=cloud_name class=small></div>
  <label class=chk><input id=ros type=checkbox checked> 以 ROS topic 格式输出</label>
  </div>
  <div id=gate_hint class=small style="margin-top:8px;color:var(--warn)">请先点上方【启动机器人】—— 策略就位后方可输入点云与规划</div>
  <div id=b1_hint class=small style="margin-top:8px;color:var(--warn)">请先启动机器人</div>
  <button onclick=doPlan() id=b1 disabled style="padding:13px;font-size:15px">生成轨迹</button>

</div>
<div class=card>
  <div class=card-title><span class=ticon></span>运行状态</div>
  <div id=mission_view style="display:none;margin:2px 0 10px"></div>
  <div id=state>待机</div>
  <details id=logbox><summary>运行日志</summary><pre id=loglog></pre></details>
  <div id=metrics></div>
  <div id=insp_btns style="display:none;margin-top:12px">
    <button id=bApprove onclick=doApprove() style="margin-top:0;position:relative;overflow:hidden;background:linear-gradient(180deg,#34e39b,#0fa968);box-shadow:0 1px 0 rgba(255,255,255,.2) inset,0 6px 26px rgba(41,211,145,.4);font-size:15.5px;padding:13px">✓ 检查无误 → 生成动作</button>
    <button id=bExec onclick=doExec() style="display:none;margin-top:0;position:relative;overflow:hidden;font-size:15.5px;padding:13px">▶ 执行动作</button>
    <select id=exec_speed style="display:none;margin-top:8px" title="执行速度倍率: 复核时可半速">
      <option value=1 selected>速度 1.0× (实时)</option>
      <option value=0.5>速度 0.5× (半速复核)</option>
      <option value=0.25>速度 0.25× (慢速)</option>
    </select>
    <button id=bReplay onclick=doOpenReplay() style="display:none;margin-top:8px;font-size:15px;padding:12px"
      title="生成(如缺)并打开 MuJoCo 仿真回放视频 —— 未接机器人时查看喷涂效果的主通道, 含双姿态蹲姿段">▶ 打开仿真回放 (MuJoCo · 含蹲姿)</button>
    <div id=walk_fix style="display:none;margin-top:8px" title="SONIC 行走闭环(腿式里程计), 走-量-停; 补走=手动单段位移兜底; 抓取现场=保存最近120s遥测并自动分析">
      <span class=small>补走</span>
      <input id=walk_fix_dy type=number step=0.01 value=0.05 style="width:64px" title="补走位移(m), 正=左移 负=右移">
      <span class=small>m</span>
      <button id=bWalk onclick=doWalkFix() style="font-size:14px;padding:8px 14px"
        title="SONIC 单段位移(里程计闭环) — 走完停稳约3秒; 人工监护随时急停">◀/▶ 补走</button>
      <button id=bTele onclick=doTeleDump() style="font-size:14px;padding:8px 14px"
        title="抓取最近120s遥测现场(里程计/骨盆/支撑脚+全部指令时序), 自动分析出图并给参数建议">◉ 抓取现场</button>
    </div>
    <div id=tele_sug style="display:none;margin-top:8px;font-size:13px"></div>
  </div>
  <div id=inspect_sec></div>
</div>
</main>
<script>
let init={}, upFile=null, tPlan=0;
function $(i){return document.getElementById(i)}
async function boot(){
  init = await (await fetch('/api/init')).json();
  $('robot').innerHTML = Object.entries(init.robots).map(([k,v])=>`<option value=${k}>${v.name}</option>`).join('');
  $('waist').innerHTML = [5,10,15,20].map(d=>`<option value=${d}>±${d}°</option>`).join('');
  $('waist').value='5';
  { const sp=localStorage.getItem('spray_posture'); if(sp) $('posture').value=sp; } // 刷新不丢姿态选择
  { const sn=localStorage.getItem('spray_stations'); if(sn) $('station_mode').value=sn; } // 工位模式同理
  loadRobotModel($('robot').value); renderChips(); syncRobotForm($('robot').value);
  $('ros_badge').textContent = 'ROS: ' + (init.ros_live?'rospy 实发':'topic 记录文件');
}
// ---- 机器人切换总入口: 3D/工作区/NN chips/表单参数(dist带·姿态选项·按钮文案) 全部联动 ----
function onRobotChange(name){
  loadRobotModel(name); markStale(); renderChips(); renderWorkspace(name);
  syncRobotForm(name);
}
function syncRobotForm(name){
  const ri=(init.robots||{})[name]||{};
  // 站位输入框: min/max/默认值跟随该机器人站位带; 当前值出带时换回带中点
  const d=$('dist');
  if(d){ const sb=ri.stance_band||[0.70,0.78], v=+d.value;
    d.min=sb[0]; d.max=sb[1];
    d.title=`${ri.name||name} 站位带 ${sb[0]}~${sb[1]}m (带中点 ${((sb[0]+sb[1])/2).toFixed(2)})`;
    if(v<sb[0]||v>sb[1]||!v) d.value=((sb[0]+sb[1])/2).toFixed(2);
  }
  // 姿态选项: 蹲姿段未开放的机器人隐藏 dual
  const ps=$('posture'), allowed=ri.postures||['stand','dual'];
  if(ps){ [...ps.options].forEach(o=>o.style.display=allowed.includes(o.value)?'':'none');
    if(!allowed.includes(ps.value)){ ps.value='stand'; localStorage.setItem('spray_posture','stand'); } }
  const ch=$('crouch_hint');        // 蹲姿提示只对开放双姿态的机器人显示
  if(ch) ch.style.display = allowed.includes('dual')? '' : 'none';
  // 工位模式: 无执行通道的机器人(规划专用)用不到, 隐藏
  const sw=$('station_wrap');
  if(sw) sw.style.display = ri.exec_ready===false? 'none' : '';
  // 徽章 + 回放按钮文案: 输出通道跟机器人走
  const rb=$('robot_badge');
  if(rb) rb.textContent='机器人: '+(ri.name||name)
    +(ri.waist_deg!=null?` · 腰 roll/pitch ±${ri.waist_deg}°`:'')
    +(ri.exec_ready===false?' · 规划专用(无执行通道)':'');
}
$('cloud').onchange = async e=>{
  const f=e.target.files[0]; if(!f)return;
  const fd=new FormData();
  const body=await f.arrayBuffer();
  const bnd='----b'+Math.random().toString(36).slice(2);
  const pre=`--${bnd}\\r\\nContent-Disposition: form-data; name="file"; filename="${f.name}"\\r\\nContent-Type: application/octet-stream\\r\\n\\r\\n`;
  const end=`\\r\\n--${bnd}--\\r\\n`;
  const r=await fetch('/api/upload',{method:'POST',headers:{'Content-Type':'multipart/form-data; boundary='+bnd},
    body:new Blob([new TextEncoder().encode(pre), body, new TextEncoder().encode(end)])});
  const j=await r.json(); upFile=j.file||null;
  // 后端契约: 点云可选 —— 上传被拒时不显示"已上传: null"误导文案
  $('cloud_name').textContent = j.file? ('已上传: '+j.file) : '上传失败, 未使用点云(可选, 不影响生成)';
  if(!j.file) upFile=null;
  markStale();
  { const b=document.getElementById('b1'); if(b) b.disabled=false;   // 选完即可点(门控由轮询校正)
    const h=document.getElementById('b1_hint'); if(h) h.style.display='none'; }
};
let _nnCache={};
function renderChips(nn){
  if(nn) _nnCache=nn;
  const rb=$('robot')?.value||'g1', cur=+(document.getElementById('waist')?.value||5);
  $('nn_badge').innerHTML = 'NN-IK '+[5,10,15,20].map(d=>{
    // nn_ik_hook 加载优先级: <robot>_all(通用大模型, 覆盖全部腰限档) -> <robot>_rp<deg>;
    // 只看 _rp 会把已部署 _all 的机器人(如 g1)全部误报成灰
    const ok=_nnCache[rb+'_rp'+d]||_nnCache[rb+'_all'];
    return `<span class="chip${ok&&ok.deployed?' on':''}${cur===d?' cur':''}">${d}°</span>`;
  }).join('');
}
async function doPlan(){
  if(upFile==null && $('cloud').files.length){ /*未上传完*/ }
  tPlan=performance.now();
  const r=await fetch('/api/plan',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({robot:$('robot').value,dist:+$('dist').value,pitch:+$('pitch').value,waist_deg:+$('waist').value,posture:$('posture').value,stations:(($('station_mode')||{}).value||1),cloud:upFile,ros:$('ros').checked})});
  const j=await r.json(); if(j.err) alert(j.err);
}
function countUp(el){
  const t=el.textContent; const m=t.match(/^[+-]?[0-9]+([.][0-9]+)?$/); if(!m) return;
  const v=parseFloat(t); const dec=(t.split('.')[1]||'').length; const t0=performance.now();
  (function f(now){const p=Math.min((now-t0)/700,1);const e=1-Math.pow(1-p,3);
    el.textContent=(v*e).toFixed(dec);
    if(p<1) requestAnimationFrame(f); else el.textContent=t;})(t0);
}
function buildInspect(s){
  const fmap=Object.fromEntries(((s.figs&&s.figs.length?s.figs:init.figs)||[]).map(x=>[x.name,x.url]));
  const t=s.metrics.traj, m=s.metrics.motion||null, c=s.metrics.cloud||null, mu=s.metrics.multi||null;
  const w=t.stroke_u? (t.stroke_u[1]-t.stroke_u[0]).toFixed(3):'-';
  // z_w 单调按“带”分段校验: 站姿带内递增 -> 换带(蹲姿从头喷) -> 蹲姿带内递增。
  // 双姿态跨带回落是设计使然, 整条全局递增的旧判据会误报“叠层异常”。
  const zw=t.z_w||[]; let zn=t.zones||[];
  if(zw.length&&zn.length!==zw.length){
    // 旧摘要无逐行 zone: z_w 显著回落(>0.05m)处视为站->蹲换带, 其后均为蹲带
    // (执行序为先整段站姿带再整段蹲姿带, 回落只出现在换带首行)
    let band='stand';
    zn=zw.map((v,i)=>{ if(i>0&&v<zw[i-1]-0.05) band='crouch'; return band; });
  }
  let mono=true;
  for(let i=1;i<zw.length;i++){
    if(zn[i]&&zn[i]!==zn[i-1]) continue;          // 带边界: 允许回落
    if(!(zw[i]>zw[i-1])){mono=false;break;}
  }
  const isV=t.stroke_dir==='vertical';
  const bands={}; (zn.length?zn:['stand']).forEach(z=>bands[z]=(bands[z]||0)+1);
  const caps={'fig_multi_station.png':'多工位全覆盖规划(整墙条带划分+走位)','base_comp_report.png':'基座扰动补偿验证(世界系锁定)','fig_wall_path_2d.png':'墙面之字形(首站带 · 含点云背景/障碍框)','fig_robot_path_2d.png':'腕心轨迹与指向(点云背景)','fig_3d_scene.png':'3D 全景','cloud_report.png':'你的点云报告'};
  const names=['fig_multi_station.png','fig_wall_path_2d.png','fig_robot_path_2d.png','fig_3d_scene.png','base_comp_report.png'];
  if(c) names.push('cloud_report.png');
  const v=Date.now();
  const sum=[`<div class=sumitem><b>${t.points??'-'}</b> 点 / ${t.rows??'-'} 行</div>`,
    `<div class=sumitem>行角 <b>${(t.angles||[]).join('/')}</b></div>`,
    `<div class=sumitem>等宽 <b>${w} m</b> (u ${t.stroke_u?t.stroke_u.map(x=>x.toFixed(3)).join('~'):''})</div>`,
    `<div class=sumitem>实测夹角 <b>${t.angle_range?t.angle_range.map(x=>x.toFixed(1)).join('~'):'-'}°</b> ${t.angle_ok?'<span class=badge-ok></span>':'<span class=badge-bad></span>'}</div>`,
    (isV
      ? `<div class=sumitem>竖直列 <b>${t.columns??'-'} 列</b> u极差=0${t.stroke_geom_ok?'<span class=badge-ok></span>':'<span class=badge-bad></span>'}</div>`
      : `<div class=sumitem>z_w 带内叠层 <b>${mono?'单调':'异常'}</b>${mono?'<span class=badge-ok></span>':'<span class=badge-bad></span>'}</div>`)];
  if(m&&m.frames) sum.push(`<div class=sumitem>动作 <b>${m.frames}</b> 帧 / ${m.duration_s}s / 峰速 ${m.max_vel} rad/s</div>`);
  sum.splice(1,0,`<div class=sumitem>姿态 <b>${t.posture==='dual'?'站+蹲双姿态':'站姿单一'}</b>${t.posture==='dual'?` (蹲带 ${t.crouch_rows??'-'} 行 · 行下限 ${t.h_min_row!=null?t.h_min_row.toFixed(2):'-'}m)`:''}</div>`);
  if(mu){ const ms=mu.steps||[mu.step_s];
    sum.splice(2,0,`<div class=sumitem>全覆盖 <b>${mu.n_stations} 站</b>(最少) 无缝拼接 · 每站条带=动作带宽 ${Math.min(...ms).toFixed(2)}~${Math.max(...ms).toFixed(2)}m · ${mu.wall_width}m 整墙</div>`); }
  const b=s.metrics.base_comp;
  if(b) sum.push(`<div class=sumitem>基座扰动锁定 <b>${b.nc_mm}→${b.lock_mm}mm</b> ${b.lock_mm<=8?'<span class=badge-ok></span>':'<span class=badge-bad></span>'}</div>`,
                 `<div class=sumitem>补偿后夹角 <b>${b.ang_min}°</b> / 腰限位 ${b.waist_ok?'保持':'超限!'}</div>`);
  $('inspect_sec').innerHTML='<h3 style="margin:14px 0 2px">轨迹检查</h3>'
    +`<div class=sumrow>${sum.join('')}</div>`
    +`<div class=figgrid>${names.filter(n=>fmap[n]).map(n=>
        `<div><div class=figcell onclick="showLb('${fmap[n]}?v=${v}')"><img src="${fmap[n]}?v=${v}"><div class=zoom><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ededed" stroke-width="2.2"><circle cx="11" cy="11" r="7"/><line x1="16.5" y1="16.5" x2="21" y2="21"/></svg></div></div><div class=figcap>${caps[n]||n}</div></div>`).join('')}</div>`
    +(c?`<details style="margin-top:8px"><summary class=small>点云检测明细</summary><pre style="height:auto;max-height:200px">${JSON.stringify(c,null,1)}</pre></details>`:'');
  $('insp_btns').style.display='block';
  document.querySelectorAll('#inspect_sec .sumitem b').forEach(countUp);
  document.querySelectorAll('#inspect_sec img').forEach(im=>{
    if(im.complete) im.classList.add('ready'); else im.onload=()=>im.classList.add('ready');
  });
}
function showLb(src){
  const lb=document.createElement('div'); lb.id='lightbox';
  lb.innerHTML=`<img src="${src}">`;
  lb.onclick=()=>lb.remove(); document.body.appendChild(lb);
  lb.style.display='flex';
}
let _rTimer=0;
function loadRobotModel(name){
  const info=(init.robots||{})[name]; if(!info||!info.urdf) return;
  clearTimeout(_rTimer); _rTimer=setTimeout(()=>_loadUrdf(info.urdf+(info.urdf.includes('?')?'&':'?')+'robot='+encodeURIComponent(name)),50);
}
async function renderWorkspace(name){
  const wv=$('ws_view'); if(!wv) return;
  const info=(init.robots||{})[name]||{};
  // 站位默认值归 syncRobotForm 管(按站位带夹取); 这里不再重复设置
  if(!info.ws_fig){ wv.style.display='none'; wv.innerHTML=''; return; }
  let rows='';
  try{
    const st=await (await fetch(info.ws_stats)).json();
    const pick=k=>st[k]? `腰±${k.endsWith('y60')?'60':'30'}°${k.includes('stand')?'站':'蹲'}:
      y ${st[k].y_ext[0].toFixed(2)}~${st[k].y_ext[1].toFixed(2)}m,
      z实心 ${st[k].z_solid[0].toFixed(2)}~${st[k].z_solid[1].toFixed(2)}m,
      段带内z ${st[k].z_band[0].toFixed(2)}~${st[k].z_band[1].toFixed(2)}m`:'';
    rows=['right_stand_y30','right_crouch_y30','right_stand_y60','right_crouch_y60']
      .map(k=>st[k]?`<div class=small>· ${pick(k)}</div>`:'').join('');
  }catch(e){ rows='<div class=small style="color:var(--warn)">统计读取失败(可能尚未计算)</div>'; }
  wv.style.display='block';
  wv.innerHTML=`<div class="sec">可达范围 (右臂, 垂直保证区)</div>
    <img src="${info.ws_fig}" style="width:100%;border-radius:8px;margin-top:6px" alt="workspace">
    ${rows}
    <div class=small style="color:var(--warn);margin-top:4px">${
      info.workspace_only? '⚠ 该机器人当前仅接入可达范围与 3D 预览, 轨迹规划未开放' : ''}</div>`;
}
function _loadUrdf(url){
  const cv=document.getElementById('rcanvas');
  try{ if(window._rrend){window._rrend.dispose(); window._rrend=null;}
       if(window._rscene){window._rscene.traverse(o=>{if(o.geometry)o.geometry.dispose(); if(o.material)o.material.dispose();});}
  }catch(e){}
  const rd=new THREE.WebGLRenderer({canvas:cv,antialias:true,alpha:true});
  window._rrend=rd; rd.setPixelRatio(Math.min(devicePixelRatio,2));
  const sc=new THREE.Scene(); window._rscene=sc;
  sc.add(new THREE.HemisphereLight(0xffffff,0x223344,1.1));
  const dl=new THREE.DirectionalLight(0xffffff,0.9); dl.position.set(2,3,2); sc.add(dl);
  const cam=new THREE.PerspectiveCamera(40,1,0.05,50); cam.position.set(1.6,1.1,1.8);
  const ctr=new THREE.OrbitControls(cam,cv); ctr.target.set(0,0.75,0);
  ctr.enableDamping=true; window._rctr=ctr;
  const hint=document.getElementById('rhint'); hint.textContent='加载模型中…';
  new URDFLoader().load(url, robot=>{
    robot.rotation.x=-Math.PI/2;                      // urdf z-up -> three y-up
    sc.add(robot);
    const box=new THREE.Box3().setFromObject(robot);
    const c=box.getCenter(new THREE.Vector3()), s=box.getSize(new THREE.Vector3());
    robot.position.set(-c.x,-c.y,-c.z);               // 居中后再抬到原点上方
    robot.position.y=-c.y;
    ctr.target.set(0,0,0); cam.position.set(1.2,0.6,1.6);
    hint.textContent='3D 模型: 拖动旋转 / 滚轮缩放';
    (function loop(){ if(window._rrend!==rd) return;
      requestAnimationFrame(loop); ctr.update(); rd.render(sc,cam); })();
    function fit(){ const w=cv.clientWidth,h=cv.clientHeight;
      rd.setSize(w,h,false); cam.aspect=w/h; cam.updateProjectionMatrix(); }
    if(window._rfit) window.removeEventListener('resize', window._rfit); // 切换模型先移除旧监听, 防泄漏堆积
    window._rfit=fit;
    fit(); window.addEventListener('resize',fit);
  }, undefined, err=>{ hint.textContent='模型加载失败: '+err; });
}
async function doApprove(){
  const r=await fetch('/api/approve',{method:'POST'}); const j=await r.json(); if(j.err) alert(j.err);
}
async function doEstop(){
  const j=await (await fetch('/api/estop',{method:'POST'})).json().catch(()=>({err:'请求失败'}));
  if(j.err){ toast(j.err); return; }
  _stale=true; _inspShown=0; _sig=null;
  $('inspect_sec').innerHTML=''; $('insp_btns').style.display='none'; $('metrics').innerHTML='';
  const be=$('bExec'); if(be) be.style.display='none';
  const b0=$('b0'); if(b0){ b0.className='go'; b0.textContent='启动机器人'; b0.onclick=doStartRobot; }
  const gi=$('gated_inputs');
  if(gi){ gi.style.opacity='0.45'; gi.style.pointerEvents='none';
    gi.querySelectorAll('input,select,button').forEach(el=>el.disabled=true); }
  const es=$('estop'); if(es) es.classList.add('soft');
  $('robot_stat').innerHTML='<span class=warn>已软急停 — 机器人全身变软(阻尼保护态), 程序仍在运行。恢复请点【退出程序】后重启</span>';
  toast(j.confirmed? '★ 已软急停: 机器人变软(阻尼保护态), 程序仍在运行'
    : (j.estopped? '⚠ 急停已发送但未确认送达 —— 请立即按手柄急停'
                 : '已软急停 (当前无运行中的机器人)'));
}
async function doRestartRobot(){
  toast('正在重启机器人(退出→重新启动, 约 15s)…');
  const j=await (await fetch('/api/robot/restart',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({robot:($('robot')||{}).value||'g1'})})).json().catch(()=>({err:'请求失败'}));
  if(j.err) toast(j.err); else toast('机器人重启中, 稍候回到初始站姿');
}
async function doExitRobot(){
  const j=await (await fetch('/api/robot/exit',{method:'POST'})).json().catch(()=>({err:'请求失败'}));
  if(j.err){ toast(j.err); return; }
  const es=$('estop'); if(es) es.classList.remove('soft');
  const b0=$('b0'); if(b0){ b0.className='go'; b0.textContent='启动机器人'; b0.onclick=doStartRobot; }
  $('robot_stat').innerHTML='<span>已退出程序 — 请重新【启动机器人】</span>';
  toast(j.exited? '已退出程序 (部署进程已结束)' : '退出: 当前无运行中的进程');
}
async function doStartRobot(){
  const j=await (await fetch('/api/robot/start',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({posture:($('posture')||{}).value||'stand',
                         robot:($('robot')||{}).value||'g1'})})).json();
  if(j.err){ toast(j.err); return; }
  // 启动机器人 = 新会话起点: 清空旧结果与作废标记, 界面回到干净初始态
  _stale=false; _inspShown=0; _sig=null;
  $('inspect_sec').innerHTML=''; $('insp_btns').style.display='none';
  $('metrics').innerHTML='';
  const es2=$('estop'); if(es2) es2.classList.remove('soft');
  toast(j.mode==='tiangong3-planning'
    ? '天工3 规划会话就绪 — 执行通道未接入, 成品用【打开仿真回放】查看'
    : (j.already?'机器人已在运行':'机器人启动中, 稍候回到初始站姿'));
}
async function doStopRobot(){
  const j=await (await fetch('/api/robot/stop',{method:'POST'})).json();
  // 立即复位界面(不等待下一次轮询)
  _stale=true; _inspShown=0; _sig=null;
  $('inspect_sec').innerHTML=''; $('insp_btns').style.display='none';
  $('metrics').innerHTML=''; $('steps') && renderSteps('idle','cur');
  const b1=$('b1'); if(b1) b1.disabled=true;
  const gi=$('gated_inputs');
  if(gi){ gi.style.opacity='0.45'; gi.style.pointerEvents='none';
    gi.querySelectorAll('input,select,button').forEach(el=>el.disabled=true); }
  const gh=$('gate_hint'); if(gh) gh.style.display='block';
  { const ib2=$('insp_btns'); if(ib2) ib2.style.display='none';
    const be2=$('bExec'); if(be2) be2.style.display='none'; }
  const b0x=$('b0'); if(b0x){ b0x.className='go'; b0x.textContent='启动机器人';
    b0x.onclick=doStartRobot; b0x.disabled=false; }
  $('robot_stat').innerHTML='<span>机器人已停止 — 已回归初始状态</span>';
  toast('已停止机器人, 全部任务中止并复位');
}
async function doExec(){
  try{
    const spd=(document.getElementById('exec_speed')||{}).value||1;
    const stnRaw=(document.getElementById('station_mode')||{}).value||1;
    const stn=(stnRaw==='auto')? 'auto' : Math.max(1, +stnRaw||1);
    const r=await fetch('/api/exec',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({speed:+spd, stations:stn})});
    const j=await r.json();
    if(j.err) toast(j.err);
    else if(j.mode==='sim-trigger') toast(stn==='auto'? `多工位已触发 — ${j.stations} 站接力覆盖整墙, 看上方俯视图`
                                              : (stn>1? `${stn} 工位已触发 — 喷①→左移→喷②, 看上方俯视图`
                                              : `已触发仿真执行(×${spd}) — MuJoCo 按轨迹播放`));
    else if(j.mode==='gr00t-relay-sonic') toast(`实机多工位接力已启动 — ${j.stations} 站喷涂 + SONIC 行走(开环位移, 人工监护随时急停), 看上方俯视图`);
    else toast(`已下发执行(×${spd}) — 机器人开始按轨迹喷涂`);
  }catch(e){ toast('执行调用失败: '+e); }
}
async function doWalkFix(){
  try{
    const dy=parseFloat((document.getElementById('walk_fix_dy')||{}).value);
    if(!isFinite(dy)||Math.abs(dy)<1e-6){ toast('补走位移不能为 0'); return; }
    const r=await fetch('/api/walk',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({dy:dy})});
    const j=await r.json();
    if(j.err) toast(j.err);
    else toast(`补走 ${dy>0?'左移':'右移'} ${Math.abs(dy).toFixed(2)}m 已下发 — SONIC 行走+停稳约3秒, 对标尺`);
  }catch(e){ toast('补走调用失败: '+e); }
}
async function doTeleDump(){
  try{
    const r=await fetch('/api/telemetry/dump',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    const j=await r.json();
    if(j.err) toast(j.err);
    else toast('现场已保存 — 正在后台分析出图, 结果看运行日志与本区');
  }catch(e){ toast('抓取现场失败: '+e); }
}
async function doTeleApply(){
  try{
    const sug=(window._teleSuggest||{});
    const r=await fetch('/api/telemetry/apply',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(sug)});
    const j=await r.json();
    if(j.err) toast(j.err);
    else { toast('已应用建议参数(calib.json), 下次执行生效'); const el=$('tele_sug'); if(el) el.style.display='none'; }
  }catch(e){ toast('应用建议失败: '+e); }
}
async function doOpenReplay(){
  try{
    const r=await fetch('/api/open_replay',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    const j=await r.json();
    if(j.err) toast(j.err);
    else if(j.rendering) toast('回放渲染中(约2~4分钟), 完成自动打开播放器 — 进度看运行日志');
    else toast('已打开仿真回放');
  }catch(e){ toast('打开仿真回放失败: '+e); }
}
function toast(msg){
  const t=document.createElement('div'); t.className='toast'; t.textContent=msg;
  document.body.appendChild(t); setTimeout(()=>t.remove(),5000);
}
async function doPackage(){
  const j=await (await fetch('/api/package',{method:'POST'})).json();
  $('metrics').innerHTML += `<div class=card>执行包: <code>${j.zip}</code> (动作 ${j.motion_ver})</div>`;
}
let _stale=false;
function markStale(){
  _stale=true; _inspShown=0; _sig=null;
  $('inspect_sec').innerHTML='';
  $('insp_btns').style.display='none';
  $('metrics').innerHTML='';
}
function renderSteps(st, rstep){
  const order=[['启动机器人','robot'],['生成轨迹','planning'],['轨迹检查','inspect'],['动作并打包','motion'],['复核→执行','ready']];
  const idx=order.findIndex(o=>o[1]===st);
  $('steps').innerHTML=order.map((o,i)=>{
    let cls='step';
    if(i===0 && rstep){ cls += (rstep==='cur'?'':' '+rstep); }
    else if(st==='error'){ if(i<=1) cls+=' done'; if(i===2) cls+=' active'; }
    else if(idx>i) cls+=' done';
    else if(idx===i) cls+=' active';
    return `<div class="${cls}"><span class="dot"></span>${o[0]}</div>`;
  }).join('');
}
// ---- 双工位接力任务俯视图(顶视): 墙面在上, 两站带宽 + 机器人行走轨迹 + 完成填色 ----
function renderMission(rb){
  const mv=$('mission_view'); if(!mv) return;
  const n=rb.stations_total||0, ph=rb.phase||'';
  const active=(n>1)&&!!rb.band&&(ph==='playing' ||   // 实机接力无 path 轨迹也显示
      ['station_done','walking','settle'].includes(ph) ||
      (ph==='standing'&&(rb.walk_dy_done||0)>0));
  if(!active){ mv.style.display='none'; mv.innerHTML=''; return; }
  mv.style.display='block';
  const b=rb.band, u0=b[0], u1=b[1], dy=rb.walk_dy_planned||(u1-u0);
  const path=rb.path||[];
  const W=600,H=310, dWall=0.74, dMax=0.95, mrg=0.22;
  let ys=[u0,u1+(n-1)*dy]; for(const p of path){ ys.push(p[2]); }
  const yMin=Math.min(...ys)-mrg, yMax=Math.max(...ys)+mrg;
  const sx=y=> 34+(y-yMin)/(yMax-yMin)*(W-54);
  const sy=d=> H-36-(d/dMax)*(H-72);           // 深度0(机器人)在下, 墙在上
  const idx=rb.station_idx||1;
  let done_k=k=> (k<idx)|| (k===idx&&['station_done','walking','settle'].includes(ph))
                      || (ph==='standing'&&(rb.walk_dy_done||0)>0&&k===n);
  let bandTxt='', arrowTxt='';  for(let k=1;k<=n;k++){
    const lo=u0+(k-1)*dy, hi=u1+(k-1)*dy;
    const cls=done_k(k)? '#22c98a' : (k===idx&&ph==='playing'? '#f5b942' : '#5a6472');
    bandTxt+=`<rect x="${sx(lo)}" y="${sy(dWall)}" width="${sx(hi)-sx(lo)}" height="${sy(dWall-0.12)-sy(dWall)}" fill="${cls}" fill-opacity="0.85" rx="2"/>`+
             `<text x="${(sx(lo)+sx(hi))/2}" y="${sy(dWall)-7}" fill="#cfd8e3" font-size="12" text-anchor="middle">工位${k}${done_k(k)?' ✓':''}</text>`;
  }
  arrowTxt=`<line x1="${sx(u0+(u1-u0)/2)}" y1="${H-26}" x2="${sx(u0+(u1-u0)/2+dy)}" y2="${H-26}" stroke="#8fd3ff" stroke-width="2" marker-end="url(#arrw)"/>`+
           `<text x="${sx(u0+(u1-u0)/2+dy/2)}" y="${H-32}" fill="#8fd3ff" font-size="12" text-anchor="middle">左移 ${dy.toFixed(2)}m/站 ${(rb.walk_dy_done||0).toFixed(2)}/${dy.toFixed(2)}m</text>`;
  let poly='';
  if(path.length>1){
    poly=`<polyline points="${path.map(p=>sx(p[2]).toFixed(1)+','+sy(p[1]).toFixed(1)).join(' ')}" fill="none" stroke="#8fd3ff" stroke-width="2.2" stroke-linejoin="round" stroke-opacity="0.9"/>`;
    const lp=path[path.length-1];
    poly+=`<circle cx="${sx(lp[2])}" cy="${sy(lp[1])}" r="6" fill="#8fd3ff" stroke="#0b1220" stroke-width="1.5"/>`;
  }
  const phTxt={playing:`工位${idx} 喷涂中`, station_done:`工位${idx} 完成`, walking:'左移中…', settle:'停稳中…', standing:'任务完成'}[ph]||ph;
  mv.innerHTML=`<div class=card-title style="font-size:14px">${n>2?`${n} 工位接力俯视图`:'双工位接力俯视图'} <span class=small>— ${phTxt} (${idx}/${n})</span></div>
  <svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:${W}px;background:#0e1626;border-radius:8px">
    <defs><marker id="arrw" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="#8fd3ff"/></marker></defs>
    <line x1="${sx(yMin)}" y1="${sy(dWall)}" x2="${sx(yMax)}" y2="${sy(dWall)}" stroke="#7f8b9c" stroke-width="3"/>
    <text x="${34}" y="${sy(dWall)+18}" fill="#7f8b9c" font-size="12">墙面</text>
    ${bandTxt}${arrowTxt}${poly}
  </svg>
  <div class=small style="margin-top:4px"><span style="color:#22c98a">■</span>已喷涂 <span style="color:#f5b942">■</span>喷涂中 <span style="color:#5a6472">■</span>待喷涂 <span style="color:#8fd3ff">●</span>机器人(蓝线=行走轨迹)</div>`;
}
setInterval(async()=>{
  const r=await fetch('/api/status').catch(()=>null);   // 网络 reject 也捕获(重启/断连)
  const s=r?await r.json().catch(()=>null):null;
  if(!s) return;                        // 请求失败(重启/断连): 跳过本轮, 下轮重试
  const busy=s.state==='planning'||s.state==='motion';
  if(s.state==='planning') _stale=false;
  const eff=(_stale&&['inspect','ready','error'].includes(s.state))?'idle':s.state;
  // 日志框: 失败时自动展开一次; 其余时间绝不强制开合(用户手动展开的不许被轮询合上)
  const lb=document.getElementById('logbox');
  if(lb){ if(s.state==='error'){ lb.open=true; window._logAuto=true; }
          else if(window._logAuto && s.state!=='error'){ window._logAuto=false; } }
  const jrobot=s.robot_name||$('robot').value;      // 当前产物机器人(事实来源)
  $('robot_badge').textContent =
    '机器人: '+((init.robots||{})[jrobot]||{}).name+' · 生效腰限 ±'+(s.waist_deg??5)+'°'
    +(s.posture==='dual'?' · 站+蹲双姿态':'')
    +(jrobot!==$('robot').value?' · ⚠ 已切换选择器, 重新生成轨迹后生效':'');
  $('state').innerHTML = {idle:_stale?'待机 — 输入已变更，请重新 ① 生成轨迹':'待机',
    planning:'生成轨迹中',inspect:'<span class=warn>待检查</span> — 在下方轨迹检查区确认',
    motion:'生成动作中',ready:'<span class=ok>动作就绪</span> — 复核无误后选速度点【执行动作】',
    error:'<span class=err>失败: '+s.error+'</span>'}[eff]||s.state;
  if(busy) $('state').innerHTML+='<span class=busy-dot></span>';
  $('b1').classList.toggle('btn-busy', s.state==='planning');
  { const be=$('bExec'); if(be) be.classList.toggle('btn-busy', s.state==='motion'); }
  const sig=s.state+'|'+JSON.stringify(s.metrics||{})+'|'+(s.run_dir||'');
  const sigChanged = window._sig!==sig;
  if(sigChanged) window._sig=sig;
  const rb=(s.robot||{});     // 会话机器人状态(会话身份, 非产物机器人)
  {   // 机器人相关 UI: 每次轮询都刷新(轻量), 保证门槛/按钮即时跟随
    const rph=rb.phase||(rb.running?'starting':'stopped');
    renderMission(rb);
  window._robotPhase = rb.running? rph : 'stopped';
  // 文案分机器人: 跟随**会话身份**(rb.robot, 实际启动的机器人)而非产物机器人 jrobot ——
  // 天工产物残留时启动 G1, 按钮/文案不得仍按天工显示
  const isT3 = (rb.mode==='tiangong3-planning') || (rb.robot==='tiangong3');
  const rmap = isT3
    ? {stopped:'天工3 规划会话未开启', starting:'会话开启中…',
       ready:'<span class=ok>天工3 规划会话就绪</span> <span class=small>(纯运动学 · 无策略进程 · 成品看【打开仿真回放】)</span>',
       playing:'回放渲染中…'}
    : {stopped:'机器人未启动', starting:'机器人启动中…', standing:'机器人已就位 (初始站姿, 待执行)',
       executing:'★ 正在执行喷涂动作(上肢跟随)…', soft_stopped:'已停止(GR00T 原生急停, 机器人变软)',
       playing:`喷涂执行中 ${Math.round((rb.progress||0)*100)}%`, done:'本轮执行完成',
       // 双工位接力阶段(runner write_status): 无映射时会裸显英文 phase
       station_done:'<span class=ok>工位喷涂完成</span> — 停稳后左移下一工位',
       walking:'★ 双工位接力: 左移中…', settle:'双工位接力: 停稳中…',
       fallen:'<span class=err>⚠ 行走段摔倒/失控, 任务已终止 — 请急停并检查</span>'};
  const rcol={stopped:'', starting:'warn', standing:'ok', ready:'ok', playing:'warn', done:'ok',
              station_done:'ok', walking:'warn', settle:'warn', fallen:'err'}[rph]||'';
  $('robot_stat').innerHTML = `<span class=${rcol}>${rmap[rph]||rph}</span>`+
    (rb.done_runs?` <span class=small>已完成 ${rb.done_runs} 轮</span>`:'')+
    (rb.impact_err?` <span class=small>落点误差 均值${(rb.impact_err.mean*100).toFixed(1)}cm / p95 ${(rb.impact_err.p95*100).toFixed(1)}cm${Math.abs(rb.impact_err.p95)>0.08?' ⚠':''}</span>`:'');
  const rstep=rb.running? ((rph==='playing'||rph==='executing')?'active':'done') : 'cur';
  renderSteps(eff, rstep);
  // —— 流程门槛: 机器人未启动时, 禁止选文件/改参数/生成轨迹 ——
  const gateOpen=!!rb.running;
  const gi=$('gated_inputs');
  if(gi){ gi.style.opacity=gateOpen?'1':'0.45'; gi.style.pointerEvents=gateOpen?'auto':'none';
    gi.querySelectorAll('input,select,button').forEach(el=>el.disabled=!gateOpen); }
  const gh=$('gate_hint'); if(gh) gh.style.display=gateOpen?'none':'block';
  if(!gateOpen && (eff==='idle'))          // 未启动: 提示先启动机器人(而非"输入已变更")
    $('state').innerHTML='待机 — 请先点上方【启动机器人】';
  const b1=$('b1'); if(b1) b1.disabled=!gateOpen || ['planning','motion'].includes(s.state);
  const ibg=$('insp_btns'); if(ibg && !gateOpen) ibg.style.display='none';
  { const beg=$('bExec'); if(beg && !gateOpen) beg.disabled=true; }
  renderChips(s.nn);
  const b0=$('b0');
  if(b0){ b0.disabled=!!rb.running;
    const soft=(rph==='soft_stopped');
    if(rb.running && soft){
      b0.className='stop'; b0.textContent='重新启动机器人 (恢复控制)';
      b0.onclick=doRestartRobot;
    } else if(rb.running && isT3){
      b0.className='stop'; b0.textContent='结束天工规划会话';
      b0.onclick=doStopRobot;
    } else if(rb.running){
      b0.className='stop'; b0.textContent='停止机器人 (立即全身变软)';
      b0.onclick=doStopRobot;
    } else {
      b0.className='go'; b0.textContent='启动机器人';
      b0.onclick=doStartRobot;
    }
    b0.disabled=false; }
  }
  if(sigChanged && s.metrics && s.metrics.traj){ window._needMetrics=true; }
  if(window._needMetrics && !_stale){ window._needMetrics=false; renderMetrics(s); }
  if(_stale && $('metrics').innerHTML) $('metrics').innerHTML='';
  const ib=$('insp_btns');
  if(ib) ib.style.display = ((s.state==='inspect'||s.state==='ready')&&!_stale) ? 'block':'none';
  const ll=$('loglog');
  if(ll){ // 贴底时才自动跟滚(用户上翻阅读时不抢滚动); 失败时强制滚到底露错误
    const stick = ll.scrollTop + ll.clientHeight >= ll.scrollHeight - 30;
    ll.textContent = (s.log||[]).join('\\n');
    if(stick || s.state==='error') ll.scrollTop = ll.scrollHeight; }
  // 点云为可选输入(后端契约): 未上传点云也可生成轨迹, 仅要求机器人已启动
  const ready2go = (s.robot||{}).running
                   && !['planning','motion'].includes(s.state);
  $('b1').disabled = !ready2go;
  { const h=$('b1_hint');
    if(h){
      h.style.display = ready2go? 'none':'block';
      h.textContent = !(s.robot||{}).running? '请先启动机器人' : '';
    } }
  // 右栏按钮: 待检查时显示"检查无误→生成动作", 动作就绪时切换为"执行动作"+速度选择
  const bEx=$('bExec'), bAp=$('bApprove'), bSp=$('exec_speed');
  if(bEx && bAp){
    const ready=(s.state==='ready') && !_stale;
    const t3 = rb.robot==='tiangong3';   // 跟随会话身份(非产物机器人 jrobot):
                                        // "G1 策略栈 + 天工3 规划产物" 也要能执行
    bAp.style.display = (!ready && s.state==='inspect' && !_stale)? 'block':'none';
    bEx.style.display = (ready && !t3)? 'block':'none';
    if(bSp) bSp.style.display = (ready && !t3)? 'block':'none';
    { // 实机多/双工位: 显示手动补走(SONIC 开环行走欠走时的标尺修正)
      const wf=$('walk_fix');
      if(wf) wf.style.display = (ready && !t3 && (s.robot||{}).mode==='real'
        && (s.robot||{}).running
        && String((document.getElementById('station_mode')||{}).value)!=='1')? 'block':'none';
      // 监测系统: 现场报告摘要 + 数据建议一键应用
      const ts=$('tele_sug');
      if(ts){
        const tele=(s.metrics||{}).tele;
        if(tele && (tele.windows||[]).length){
          window._teleSuggest=tele.suggest||{};
          const ws=tele.windows.map(w=>`目标${(+w.dy).toFixed(3)}/实测${(+w.walked).toFixed(3)}m·死区${w.deadzone_s??'-'}s·${w.bursts}段`).join(' | ');
          const sg=Object.keys(window._teleSuggest||{}).length
            ? ` <button onclick=doTeleApply() style="font-size:13px;padding:6px 12px">应用建议参数(${Object.entries(window._teleSuggest).map(([k,v])=>k+'='+v).join(', ')})</button>` : '';
          ts.innerHTML=`<div class=card-title style="font-size:13px">◉ 现场报告 <span class=small>${ws}</span></div>`
            +`<div class=small style="margin-top:2px"><a href="/fig/telemetry/${tele.png.split('/').pop()}" target="_blank" style="color:#8fd3ff">查看报告图</a>${sg}</div>`;
          ts.style.display='block';
        } else { ts.style.display='none'; window._teleSuggest=null; }
      } }
    { const bRp=$('bReplay'); if(bRp){ bRp.style.display = ready? 'block':'none';
      // 回放内容跟产物机器人走(motion_meta.robot 驱动 06 场景), 文案必须一致
      if(t3){ bRp.textContent='▶ 打开仿真回放 (MuJoCo · 天工3 场景)';
              bRp.title='生成(如缺)并打开天工3 场景的 MuJoCo 回放视频'; }
      else { bRp.textContent='▶ 打开仿真回放 (MuJoCo · 含蹲姿)';
             bRp.title='生成(如缺)并打开 MuJoCo 仿真回放视频 —— 未接机器人时查看喷涂效果的主通道, 含双姿态蹲姿段'; } } }
    if(ready && t3){                       // 天工: 无执行通道, 回放为主出口
      bEx.style.display='block'; bEx.disabled=true; bEx.className='';
      bEx.textContent='✕ 执行通道未接入 (天工3 无策略栈) — 请用【打开仿真回放】查看成品';
    } else if(ready){
      const rok=(s.robot||{}).running, rplaying=(s.robot||{}).phase==='playing';
      const rbusy=rplaying||['walking','settle','station_done'].includes((s.robot||{}).phase);
      bEx.disabled=(!rok)||rbusy;
      if(bSp) bSp.disabled=(!rok)||rbusy;
      bEx.className=(rok&&!rbusy)?'go':'';
      bEx.textContent = rbusy? '任务执行中…' : (rok? '▶ 执行动作' : '▶ 执行动作 (需先启动机器人)');
    }
  }
  if((s.state==='inspect'||s.state==='ready') && !window._inspShown && !_stale){
    window._inspShown=1; buildInspect(s);
  }
  if(s.state==='planning'){
    window._inspShown=0; $('inspect_sec').innerHTML=''; $('insp_btns').style.display='none';
  }
  if(s.run_dir && window._rl!==s.run_dir){
    window._rl=s.run_dir;
    const el=document.getElementById('runline'); if(el) el.remove();
    $('state').insertAdjacentHTML('afterend',
      `<div class=runline id=runline>运行目录 <code>${s.run_dir.split('/').pop()}</code></div>`);
  }
},250);

function renderMetrics(s){
  const t=(s.metrics||{}).traj, m=(s.metrics||{}).motion||{};
  if(!t){ $('metrics').innerHTML=''; return; }
  $('metrics').innerHTML = `<table><tr><th>点数</th><td>${t.points??'-'}</td>
    <th>行数</th><td>${t.rows??'-'}</td></tr>
    <tr><th>行角档</th><td>${(t.angles||[]).join('/')}</td><th>等宽u</th><td>${t.stroke_u?JSON.stringify(t.stroke_u):'-'}</td></tr>
    <tr><th>z_w叠层</th><td colspan=3>${(t.z_w||[]).join(' → ')}</td></tr>
    <tr><th>夹角实测</th><td>${t.angle_range?JSON.stringify(t.angle_range):'-'}</td><th>达标</th><td>${t.angle_ok}</td></tr>
    ${m.frames?`<tr><th>动作帧</th><td>${m.frames} (${m.duration_s}s)</td><th>最大关节速度</th><td>${m.max_vel} rad/s</td></tr>
    <tr><th>喷涂夹角</th><td>${JSON.stringify(m.spray_angle)}</td><th>MJCF-IK</th><td>${m.mjcf_ik}</td></tr>`:''}
    </table>`;
}

// ---- 机器人幽灵动态背景: G1 线框慢旋转 + 上升粒子雾 ----
let _bgInited=false;
function init3DBG(){
  if(_bgInited || !(init.robots||{})[$('robot').value]) return;
  _bgInited=true;
  const cv=document.getElementById('bg3d');
  const rd=new THREE.WebGLRenderer({canvas:cv,antialias:true,alpha:true});
  rd.setPixelRatio(Math.min(devicePixelRatio,1.5));
  const sc=new THREE.Scene();
  const cam=new THREE.PerspectiveCamera(35,innerWidth/innerHeight,.1,60);
  cam.position.set(0,.9,4.6);
  let robot=null;
  new URDFLoader().load(init.robots[$('robot').value].urdf, r=>{
    r.rotation.x=-Math.PI/2;
    const box=new THREE.Box3().setFromObject(r);
    const c=box.getCenter(new THREE.Vector3()), sz=box.getSize(new THREE.Vector3());
    r.position.set(-c.x,-c.y,-c.z);
    r.position.y=-c.y-0.15;
    const k=2.9/Math.max(sz.y,sz.x,sz.z);
    r.scale.setScalar(k);
    r.traverse(o=>{ if(o.isMesh) o.material=new THREE.MeshBasicMaterial(
      {wireframe:true,color:0x4a7dff,transparent:true,opacity:0.15,depthWrite:false}); });
    sc.add(r); robot=r;
  });
  const PN=260, pos=new Float32Array(PN*3), spd=new Float32Array(PN);
  for(let i=0;i<PN;i++){
    pos[i*3]=(Math.random()-.5)*7; pos[i*3+1]=(Math.random()-.5)*4.4; pos[i*3+2]=(Math.random()-.5)*3-1;
    spd[i]=.0018+Math.random()*.004;
  }
  const pg=new THREE.BufferGeometry();
  pg.setAttribute('position',new THREE.BufferAttribute(pos,3));
  const pts=new THREE.Points(pg,new THREE.PointsMaterial(
    {color:0x6ea8ff,size:.022,transparent:true,opacity:.4,depthWrite:false,
     blending:THREE.AdditiveBlending}));
  sc.add(pts);
  function fit(){ rd.setSize(innerWidth,innerHeight,false); cam.aspect=innerWidth/innerHeight; cam.updateProjectionMatrix(); }
  fit(); addEventListener('resize',fit);
  let _skip=false;                      // 限30fps: 隔帧渲染, 负载减半(视觉无差别)
  (function loop(){
    requestAnimationFrame(loop);
    if(document.hidden) return;
    _skip=!_skip; if(_skip) return;
    if(robot) robot.rotation.y+=0.0032;
    const arr=pg.attributes.position.array;
    for(let i=0;i<PN;i++){ arr[i*3+1]+=spd[i]*2; if(arr[i*3+1]>2.4) arr[i*3+1]=-2.4; }
    pg.attributes.position.needsUpdate=true;
    rd.render(sc,cam);
  })();
}
const _oldBoot=boot;
boot=async function(){ await _oldBoot(); init3DBG(); };
boot();

window.addEventListener('error',e=>{ window.__err=e.message; });
</script></body></html>"""

INSPECT_HTML = """<!doctype html><html><head><meta charset=utf-8><title>轨迹检查</title>
<style>body{font-family:system-ui,'Noto Sans CJK SC';background:#10151c;color:#dde3ea;margin:0;padding:16px}
h2{margin:4px 0 10px} img{max-width:100%;border-radius:8px;border:1px solid #232c39;margin:6px 0}
.small{color:#9fb0c3;font-size:12px} button{padding:9px 18px;border:0;border-radius:8px;background:#0e7490;color:#fff;cursor:pointer;font-size:14px}
</style></head><body>
<h2>轨迹检查 <span class=small>(① 生成轨迹的产物 — 确认无误后回主页面点 [② 确认生成动作])</span></h2>
<div id=figs></div>
<div class=small>说明: 红色=喷涂横线(角度规范适用), 黄色=竖直换行连接段(免角度规范);
腕心走线为等宽之字形、行内水平、z_w 自下而上严格叠层; 点云模式下喷涂带已自动避开凸出障碍。</div>
<script>
fetch('/api/init').then(r=>r.json()).then(async init=>{
  const s=await (await fetch('/api/status')).json();
  const div=document.getElementById('figs');
  const names=['fig_wall_path_2d.png','fig_robot_path_2d.png','fig_3d_scene.png'];
  if(s.metrics&&s.metrics.cloud) names.push('cloud_report.png');
  // init.figs 元素是 {name,url} 对象: 按名字匹配取 url(includes 判对象恒 false)
  const urlOf=n=>{ const f=(init.figs||[]).find(x=>x.name===n); return f? f.url : '/run/'+n; };
  div.innerHTML = names.map(n=>`<h3>${n}</h3><img src="${urlOf(n)}">`).join('')
    + (s.metrics&&s.metrics.cloud?`<h3>点云报告</h3><pre>${JSON.stringify(s.metrics.cloud,null,1)}</pre>`:'');
});
</script></body></html>"""


def _deploy_watchdog():
    """部署进程死亡监控: 崩溃/急停/断连退出时, 立刻向界面点破并给出原因线索.
    (此前部署静默死亡, 界面长时间停留在"启动中/就绪", 表现为"连接不稳定")"""
    was_live, dead_polls = False, 0
    while True:
        time.sleep(2.0)
        live = _robot_pid() is not None
        if live:
            was_live, dead_polls = True, 0
            continue
        if not was_live:
            continue
        dead_polls += 1
        if dead_polls < 2:            # 连续两轮(4s)确认死亡, 避免重启间隙误报
            continue
        was_live, dead_polls = False, 0
        write_status_phase("stopped")
        log("⚠ 机器人部署进程已退出 (急停/退出/崩溃/与机器人断连)")
        try:
            with open(ROBOT_LOG, "rb") as f:
                tail = f.read()[-1500:].decode("utf8", "ignore")
            if "Address already in use" in tail:
                log("  原因: ZMQ 端口被残留进程占用 → 运行 清进程 后重新启动")
            elif "Lost LowState" in tail:
                log("  原因: 与机器人断连 (官方 CheckSafety 安全停止) → 检查网线/机器人上电")
            elif "still faulted" in tail:
                log("  原因: 电机故障(0x40000000), 指令被忽略 → 机器人断电重启后重试")
        except Exception:
            pass


def _exit_cleanup():
    """站点退出时的轻量清理(不做重活):
    ① 部署进程 start_new_session=True 脱离会话, 站点退出不会带走它 —— 显式终止;
    ② 守护进程关 stdin(EOF 即其退出语义)并做一次非阻塞回收, 防僵尸.多次调用安全."""
    try:
        pid = _robot_pid()
        if pid:
            os.kill(pid, signal.SIGTERM)
            time.sleep(0.5)
            pid = _robot_pid()
            if pid:
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass
    except Exception:
        pass
    if _ROBOT_PROC is not None:
        try:
            _ROBOT_PROC.poll()          # 回收部署子进程, 防僵尸
        except Exception:
            pass
    try:
        p = _DAEMON.proc
        if p is not None:
            try:
                p.stdin.close()         # stdin EOF -> 守护进程按设计自行退出
            except Exception:
                pass
            try:
                os.waitpid(p.pid, os.WNOHANG)   # 非阻塞回收, 防僵尸
            except Exception:
                pass
    except Exception:
        pass


def _on_exit_signal(signum, frame):
    """SIGINT/SIGTERM -> 干净退出(atexit 里跑 _exit_cleanup)."""
    raise SystemExit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1",
                    help="HTTP 绑定地址, 默认 127.0.0.1(仅本机, 接口无鉴权); "
                         "局域网访问需显式指定 0.0.0.0")
    a = ap.parse_args()
    # 退出清理: Ctrl-C / kill / 正常返回 都会触发(钩子里只做轻量回收)
    atexit.register(_exit_cleanup)
    signal.signal(signal.SIGINT, _on_exit_signal)
    signal.signal(signal.SIGTERM, _on_exit_signal)
    try:
        st = json.load(open(STATE_F, encoding="utf8"))
        JOB.update({k: st[k] for k in ("state", "robot", "motion_ver", "run_dir",
                                       "metrics", "artifacts", "ros_mode")
                    if k in st})
        if JOB.get("run_dir"):
            log("已恢复上次运行: " + JOB["run_dir"])
    except Exception:
        pass
    log("工作站启动: http://%s:%d" % (a.host, a.port))
    threading.Thread(target=_deploy_watchdog, daemon=True).start()
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
