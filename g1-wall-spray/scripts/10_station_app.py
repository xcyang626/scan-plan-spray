# -*- coding: utf-8 -*-
"""步骤10: 喷涂轨迹工作站 (模块化: 轨迹输出 + 动作输出 + ROS topic; 执行接口预留).

启动:  python scripts/10_station_app.py [--port 8090]   (解释器见 station/env.json)
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
import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(BASE, "station", "run")          # 兼容旧文件
RUNS = os.path.join(BASE, "runs")                    # 每次运行独立目录的根
os.makedirs(RUNS, exist_ok=True)
UP = os.path.join(BASE, "uploads")
os.makedirs(RUN, exist_ok=True); os.makedirs(UP, exist_ok=True)
_ENV_JSON = {}
try:
    _ENV_JSON = json.load(open(os.path.join(BASE, "station", "env.json"),
                               encoding="utf8"))
except Exception:
    pass
ISA_PY = (os.environ.get("G1W_ISA_PY") or _ENV_JSON.get("isaac_py")
          or "/home/yang/miniconda3/envs/isaaclab/bin/python")
MUJ_PY = (os.environ.get("G1W_MUJ_PY") or _ENV_JSON.get("mujoco_py")
          or "/home/yang/miniconda3/envs/mujoco/bin/python")


# ---- GR00T-WBC 机器人执行 (启动策略/自动执行) ----
GROOT_ROOT = (os.environ.get("G1W_GROOT_ROOT") or _ENV_JSON.get("groot_root")
              or "/home/yang/桌面/轨迹规划/GR00T-WholeBodyControl")
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
                if rid is not None and rid in self._wait:
                    self._wait[rid][1] = m
                    self._wait[rid][0].set()
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
        with self.lock:
            self._id += 1
            rid = self._id
        ev = threading.Event()
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


def _daemon_event(m):
    """守护进程事件 -> 站点日志 + 机器人状态文件."""
    ev = m.get("event")
    if ev == "stream_meta":
        log(f"▶ 官方上肢流: {m.get('frames')} 帧 / {m.get('duration_s')}s "
            f"@{m.get('rate')}Hz (speed×{m.get('speed')})")
    elif ev == "stream_run":
        log("blend 过渡完成, 轨迹开始")
    elif ev == "stream_progress":
        n, k = m.get("n") or 1, m.get("k") or 0
        write_status_phase("playing", progress=(k / n if n else 0.0))
    elif ev == "stream_done":
        write_status_phase("done", progress=1.0)
        log(f"★ 轨迹执行完成: {m.get('frames')} 帧已全部下发, "
            f"{m.get('note') or '保持末姿'}")
    elif ev == "stream_cancelled":
        write_status_phase("soft_stopped")
        log(f"动作流已取消 ({m.get('note') or '官方退出序列'})")
    elif ev == "hold_started":
        log("★ " + str(m.get("note") or "保持流已启动"))
    elif ev == "hold_cancelled":
        log("保持流已结束 (" + str(m.get("note") or "") + ")")
    elif ev == "stream_error":
        log(f"⚠ 动作流异常: {m.get('err')}")
    elif ev == "fatal":
        log(f"⚠ 执行守护进程致命错误: {m.get('err')}")


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
                            "--host", "*", "--port", str(ZMQ_PORT)],
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


def write_status_phase(phase, progress=None):
    try:
        st = {"phase": phase, "ts": time.time()}
        if progress is not None:
            st["progress"] = progress
        json.dump(st, open(ROBOT_STATUS_F, "w", encoding="utf8"))
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
ZMQ_HOST = _ENV_JSON.get("zmq_host", "*")
DAEMON_PORT = int(_ENV_JSON.get("daemon_port", 5558))
ROBOT_IFACE = _ENV_JSON.get("robot_iface", "enp8s0")
# 输入类型: gamepad_manager=手柄(推荐, F2软急停/Select退出/F1切ZMQ流) | zmq=纯键盘
# GR00T 原生输入类型: zmq_manager = 支持 planner 话题(上肢通道) + command 话题(start/stop)
ROBOT_INPUT = _ENV_JSON.get("robot_input_type", "zmq_manager")


_ROBOT_PROC = None


def _alive(pid):
    """进程是否存活(排除僵尸)."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            st = f.read().decode(errors="ignore").rsplit(")", 1)[1].split()[0]
        return st != "Z"
    except Exception:
        return False


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
    if pid is None or not os.path.exists(f"/proc/{pid}"):
        # PID 文件丢失/失效 -> 按进程名兜底(自愈): 避免孤儿部署占端口后再启一个
        for pat in ("g1_deploy_onnx_ref", "groot_spray_runner"):
            try:
                r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
                cand = [int(x) for x in r.stdout.split() if x.isdigit()]
            except Exception:
                cand = []
            cand = [c for c in cand if c != os.getpid() and _alive(c)]
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
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cl = f.read().decode(errors="ignore")
        if ("groot_spray_runner" not in cl
                and "g1_deploy_onnx_ref" not in cl):
            return None                   # PID 被复用
        return pid
    except Exception:
        return None


def _robot_status():
    """状态以**实时进程检测**为准, 状态文件只补充 phase/progress(不得覆盖 running)。"""
    live = _robot_pid() is not None
    st = {"running": live, "mode": ROBOT_MODE, "daemon": _DAEMON.alive()}
    try:
        f = json.load(open(ROBOT_STATUS_F, encoding="utf8"))
        for k in ("phase", "progress", "pelvis", "motion", "frames", "done_runs"):
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


def start_robot():
    """启动机器人: 拉起 GR00T-WBC 仿真+策略, 机器人回到初始站姿并等待执行."""
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
           "HOME": os.environ.get("HOME", "/home/yang"),
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
        log(f"实机模式: 启动 SONIC 部署 (网卡 {ROBOT_IFACE}, 输入 {ROBOT_INPUT}, "
            f"zmq {ZMQ_HOST}:{ZMQ_PORT})")
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
        _ROBOT_PROC = subprocess.Popen(
            [GROOT_PY, ROBOT_RUNNER, "--trigger", TRIGGER_F],
            cwd=GROOT_ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
    write_status_file = {"phase": "starting", "ts": time.time()}
    json.dump(write_status_file, open(ROBOT_STATUS_F, "w", encoding="utf8"))
    log("★ 机器人启动中 (GR00T-WBC 仿真+策略), 等待回到初始站姿…")
    return {"ok": True, "already": False, "mode": ROBOT_MODE}


def stop_robot():
    """停止机器人 = **立即全身变软**(阻尼保护态), 不退出进程.
    说明: 退出进程会让机器人失去指令源(会报警/缓慢归位);
          保持进程持续下发 kp=0,kd=8 阻尼指令, 才是真正的"变软".
    另外: 取消进行中的规划/动作流水线, 清空本轮任务结果(界面复位)。
    彻底结束请用 exit_robot() / 网页【退出程序】。"""
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
    try:
        if PROC is not None and PROC.poll() is None:
            PROC.terminate(); time.sleep(0.3)
            if PROC.poll() is None:
                PROC.kill()
            log("已取消进行中的任务")
    except Exception:
        pass
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


def exec_spray(motion_path, run_dir=None, speed=None):
    """手动执行 —— 完全对齐官方 send_upper_body_trajectory.py:
    planner 话题 17 维上肢(SONIC 交错序) @50Hz + 梯度速度 + 结束保持末姿;
    下肢由 SONIC 平衡策略控制(不发腿)。speed 可选半速复核(0.5)。
    首选常驻守护进程, 失败回退官方脚本本体。"""
    if not _robot_pid():
        return {"err": "机器人未启动 — 请先点【启动机器人】"}
    if not motion_path or not os.path.exists(motion_path):
        return {"err": f"动作文件不存在: {motion_path}"}
    speed = float(speed or EXEC_SPEED)
    mjcf = REGISTRY.get(JOB.get("robot", "g1"), {}).get("mjcf")
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


def _p(v):
    """registry 路径解析: 绝对原样, 相对则基于工程根(打包后随包走)."""
    return v if (not v or os.path.isabs(v)) else os.path.join(BASE, v)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REGISTRY = json.load(open(os.path.join(BASE, "station", "robot_registry.json"),
                          encoding="utf8"))
for _r in REGISTRY.values():                     # 模型路径相对化(打包可移植)
    if isinstance(_r, dict):
        if _r.get("mjcf"):
            _r["mjcf"] = _p(_r["mjcf"])
        if _r.get("model_dir"):
            _r["model_dir"] = _p(_r["model_dir"])

JOB = {"state": "idle", "stage": "", "t0": None, "log": [], "error": None,
       "metrics": {}, "artifacts": {}, "motion_ver": None, "robot": "g1",
       "ros_mode": None}
LOCK = threading.Lock()
PROC = None


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
    PROC = p
    for line in p.stdout:
        log("  " + line.rstrip())
    rc = p.wait(timeout=timeout)
    PROC = None
    if rc != 0:
        raise RuntimeError(f"{name} 失败 (exit {rc})")
    log(f"✔ {name}")


def stage_plan(cfg):
    """任务1: (点云解析)-> 03轨迹 -> 04三维图."""
    JOB.update(state="planning", error=None)
    t0 = time.time()
    try:
        run_dir = os.path.join(RUNS, "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(run_dir, exist_ok=True)
        JOB["run_dir"] = run_dir
        log(f"运行目录 {run_dir}")
        ws_dir, ws_new = ensure_workspace(cfg["robot"], cfg["waist_deg"])
        JOB["waist_deg"] = cfg["waist_deg"]
        if ws_new:
            log(f"腰限 ±{cfg['waist_deg']:.0f}°: 首次使用, 重算工作区(约3分钟)…")
            run_stage(f"02 工作区重算(腰±{cfg['waist_deg']:.0f}°)",
                      [ISA_PY, "02_g1_workspace.py"],
                      env={"G1W_OUT": ws_dir,
                           "G1W_WAIST_RP": str(round(cfg["waist_deg"] * math.pi / 180, 6))},
                      timeout=900)
        env = {"G1W_CFG": os.path.join(run_dir, "cfg.json"),
               "G1W_OUT": run_dir, "G1W_IN": ws_dir,
               "G1W_ROBOT": cfg["robot"], "G1W_MJCF": REGISTRY[cfg["robot"]]["mjcf"],
               "G1W_WAIST_RP": str(round(cfg["waist_deg"] * math.pi / 180, 6)),
               "G1W_WAIST_DEG": str(int(round(cfg["waist_deg"])))}
        if cfg.get("cloud"):
            run_stage("点云解析(墙面/距离/障碍)",
                      [ISA_PY, "11_cloud_scan.py", "--cloud", cfg["cloud"],
                       "--dist", str(cfg["dist"]),
                       "--out", os.path.join(run_dir, "cloud_report.json")], env)
            rep = json.load(open(os.path.join(run_dir, "cloud_report.json"), encoding="utf8"))
            cfg["wall_x"] = rep["distance"]            # 点云实测墙面距离
            cfg["stroke_u"] = rep["stroke_u"]          # 点云驱动: 喷涂带/最高行/障碍进 cfg
            cfg["h_top"] = rep["h_top"]
            cfg["obstacles"] = rep["obstacles"]
            JOB["metrics"]["cloud"] = {k: rep[k] for k in
                                       ("distance", "wall_u", "wall_h", "stroke_u",
                                        "h_top", "obstacles", "warns")}
            JOB["artifacts"]["cloud_png"] = True
        cfg["d_robot"] = cfg["dist"]                   # 无点云: 标称距离即墙面距离
        if "wall_x" not in cfg:
            cfg["wall_x"] = cfg["dist"]
        json.dump(cfg, open(os.path.join(run_dir, "cfg.json"), "w", encoding="utf8"),
                  ensure_ascii=False, indent=1)
        run_stage("03 轨迹规划(锚点+样条+NN种子快速IK)",
                  [ISA_PY, "03_plan_trajectories.py"], env)
        run_stage("04 三维可视化", [ISA_PY, "04_visualize_3d.py"], env)
        tj = json.load(open(os.path.join(run_dir, "trajectory_meta.json"),
                            encoding="utf8"))
        JOB["metrics"]["traj"] = _traj_summary(tj)
        JOB["state"] = "inspect"                       # 待检查
        JOB["artifacts"]["plan_sec"] = round(time.time() - t0, 1)
        _persist()
        log(f"★ 轨迹已生成, 请在检查页确认 (耗时 {time.time()-t0:.1f}s)")
    except Exception as ex:
        JOB.update(state="error", error=str(ex)); log(f"✘ {ex}")


def stage_motion(cfg):
    """任务2: 05动作导出 -> 05b时序图 -> (可选)ROS topics."""
    JOB.update(state="motion", error=None)
    t0 = time.time()
    try:
        ver = f"v5_run{datetime.datetime.now().strftime('%H%M%S')}"
        JOB["motion_ver"] = ver
        run_dir = JOB.get("run_dir") or RUN
        _ws_dir, _ = ensure_workspace(cfg["robot"], cfg.get("waist_deg", 5.0))
        env = {"MOTION_VER": ver, "G1W_ROBOT": cfg["robot"],
               "G1W_MJCF": REGISTRY[cfg["robot"]]["mjcf"],
               "G1W_CFG": os.path.join(run_dir, "cfg.json"),
               "G1W_OUT": run_dir, "G1W_IN": _ws_dir,
               "G1W_WAIST_RP": str(round(cfg.get("waist_deg", 5.0) * math.pi / 180, 6)),
               "G1W_WAIST_DEG": str(int(round(cfg.get("waist_deg", 5.0))))}
        run_stage("05 动作导出(增量MJCF-IK+NN种子)", [MUJ_PY, "05_export_ufo_motion.py"], env)
        run_stage("05b 时间序列图", [ISA_PY, "05b_plot_motion.py"], env)
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
            from ros_bridge import emit_run
            r = emit_run(cfg["robot"], traj_dir=run_dir,
                         motion_dir=mdir, out_base=run_dir)
            JOB["ros_mode"] = r["mode"]
            JOB["artifacts"]["ros"] = r
            log(f"★ ROS topics 输出完成: {r['mode']}")
        json.dump(dict(robot=cfg.get("robot"), dist=cfg.get("dist"),
                       pitch=cfg.get("pitch"), cloud=cfg.get("cloud"),
                       waist_deg=cfg.get("waist_deg"),
                       motion_ver=ver, run_dir=run_dir,
                       metrics=JOB.get("metrics")),
                  open(os.path.join(run_dir, "run_manifest.json"), "w", encoding="utf8"),
                  ensure_ascii=False, indent=1, default=str)
        pkg = package()                      # 动作+打包一体: 完成即出执行包(manifest 已就位)
        JOB["artifacts"]["package"] = pkg
        # 动作就绪后直接下发给机器人执行(无需手动点“自动执行”)
        if AUTO_EXEC_AFTER_MOTION and _robot_pid():
            mv_npz = os.path.join(mdir, "wall_spray_motion.npz")
            r = exec_spray(mv_npz, run_dir)
            if r.get("ok"):
                log("★ 动作已自动下发给机器人执行")
            else:
                log(f"自动执行未生效: {r.get('err')}")
        JOB["state"] = "ready"
        JOB["artifacts"]["motion_sec"] = round(time.time() - t0, 1)
        _persist()
        log(f"★ 动作已生成 {ver} (耗时 {time.time()-t0:.1f}s)")
    except Exception as ex:
        JOB.update(state="error", error=str(ex)); log(f"✘ {ex}")


def _traj_summary(tj):
    rows = tj.get("row_params", [])
    return dict(points=tj.get("n_points"), rows=len(rows),
                z_w=[round(r["z_w"], 3) for r in rows],
                angles=[r["angle_deg"] for r in rows],
                stroke_u=tj.get("uniform_stroke_u"),
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
            self._send(200, st)
        elif u.path == "/api/init":
            self._send(200, dict(
                robots={k: {"name": v["name"],
                            "urdf": "/model/" + v.get("urdf_file", ""),
                            "waist_deg": round(np_deg(v.get(
                                "waist_roll_pitch_limit_rad", 0.349)))}
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
            self._model_file(u.path[7:])
        else:
            self._send(404, {"err": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        if u.path == "/api/upload":
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
                       h_top=body.get("h_top"), cloud=body.get("cloud"),
                       ros=bool(body.get("ros")))
            with LOCK:
                if JOB["state"] in ("planning", "motion"):
                    return self._send(409, {"err": "任务进行中"})
                JOB.update(robot=cfg["robot"], metrics={}, artifacts={})
            threading.Thread(target=stage_plan, args=(cfg,), daemon=True).start()
            self._send(200, {"ok": True})
        elif u.path == "/api/approve":
            if not _robot_pid():
                return self._send(400, {"err": "机器人未启动 — 请先点【启动机器人】"})
            with LOCK:
                if JOB["state"] != "inspect":
                    return self._send(409, {"err": f"当前状态 {JOB['state']}, 需先完成轨迹生成"})
                cfg = json.load(open(os.path.join(JOB.get("run_dir") or RUN,
                                                  "cfg.json"), encoding="utf8"))
            threading.Thread(target=stage_motion, args=(cfg,), daemon=True).start()
            self._send(200, {"ok": True})
        elif u.path == "/api/package":
            r = package()
            self._send(200, r)
        elif u.path == "/api/train_nn":
            robot = body.get("robot") or JOB.get("robot", "g1")
            if robot not in REGISTRY:
                return self._send(400, {"err": f"未知机器人 {robot}"})
            if JOB.get("nn_training"):
                return self._send(409, {"err": "该机器人模型训练进行中"})
            log(f"▶ NN-IK 重训启动: {robot} (按当前限位的工作区)")

            wd = float(body.get("waist_deg", JOB.get("waist_deg", 5.0)))
            if wd not in (5.0, 10.0, 15.0, 20.0):
                return self._send(400, {"err": f"腰限档位仅支持 5/10/15/20°, 收到 {wd}°"})

            def _train():
                JOB["nn_training"] = True
                try:
                    ws_dir, ws_new = ensure_workspace(robot, wd)
                    if ws_new:
                        log(f"腰限 ±{wd:.0f}°: 首次使用, 先重算工作区(约3分钟)…")
                        run_stage(f"02 工作区重算(腰±{wd:.0f}°)",
                                  [ISA_PY, "02_g1_workspace.py"],
                                  env={"G1W_OUT": ws_dir,
                                       "G1W_WAIST_RP": str(round(wd * math.pi / 180, 6))},
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
                    JOB["nn_training"] = False
            threading.Thread(target=_train, daemon=True).start()
            self._send(200, {"ok": True, "robot": robot})
        elif u.path == "/api/robot/start":
            try:
                return self._send(200, start_robot())
            except Exception as ex:
                return self._send(500, {"err": str(ex)})
        elif u.path == "/api/estop":
            # 急停 = 官方 command(stop) → 全身变软。经守护进程即时下发,
            # 不再进程内 import estop(mujoco 环境缺 msgpack, 旧实现必失败)。
            try:
                if _DAEMON.alive():
                    _DAEMON.call("cancel", timeout=5)   # 停掉动作流/保持流
                done = groot_command("stop")
                if not done:                   # ZMQ 通道失败 → 官方 'O' 键兜底
                    done = send_key("O")
                log("★★ 网页软急停已触发 (机器人全身变软/阻尼保护态)"
                    if done else "网页软急停: 下发通道均失败, 请按手柄急停")
                with LOCK:
                    JOB.update(state="idle", error=None, metrics={},
                               artifacts={}, motion_ver=None, run_dir=None)
                write_status_phase("soft_stopped")
                # 软急停只改状态, 保留 robot.pid —— 之后还要靠它执行"退出程序"
                return self._send(200, {"ok": True, "soft": True, "estopped": done,
                                        "note": "机器人已变软(阻尼保护态); 程序仍在运行"})
            except Exception as ex:
                return self._send(500, {"err": f"急停执行异常: {ex}"})
        elif u.path == "/api/robot/restart":
            try:
                kill_deploy()                  # 官方 stop + 可靠回收(不再依赖 in-process estop)
                time.sleep(1.0)
                start_robot()                  # 再启动(自动 Start+启用流)
                return self._send(200, {"ok": True, "restarted": True})
            except Exception as ex:
                return self._send(500, {"err": f"重启异常: {ex}"})
        elif u.path == "/api/robot/exit":
            try:
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
            r = exec_spray(mp, JOB.get("run_dir"), speed=body.get("speed"))
            self._send(200 if r.get("ok") else 400, r)
        else:
            self._send(404, {"err": "not found"})

    def _validate(self, b):
        if not _robot_pid():
            return "机器人未启动 — 请先点【启动机器人】(策略就位后才能规划)"
        r = REGISTRY.get(b.get("robot", "g1"))
        if not r:
            return f"未知机器人 {b.get('robot')}"
        wd = float(b.get("waist_deg", 5))
        if wd not in (5.0, 10.0, 15.0, 20.0):
            return f"腰限档位仅支持 5/10/15/20°, 收到 {wd}°"
        d = float(b.get("dist", 0))
        lo, hi = r["stance_band_m"]
        if not (lo - 1e-9 <= d <= hi + 1e-9):
            return f"站位距离 {d:.2f}m 超出 {r['name']} 站姿可达带 [{lo},{hi}]m"
        c = b.get("cloud")
        if c and not os.path.exists(c):
            return f"点云文件不存在: {c}"
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

    def _model_file(self, rel):
        """机器人模型静态文件: 白名单=registry登记的 model_dir, 防目录穿越."""
        import urllib.parse
        rel = urllib.parse.unquote(rel)
        info = REGISTRY.get(JOB.get("robot", "g1")) or next(
            v for k, v in REGISTRY.items() if not k.startswith("_"))
        base = os.path.realpath(info["model_dir"])
        p = os.path.realpath(os.path.join(base, rel))
        if not p.startswith(base + os.sep) and p != base:
            return self._send(403, {"err": "forbidden"})
        if not os.path.exists(p):
            return self._send(404, {"err": "no file"})
        ct = {".urdf": "application/xml", ".xml": "application/xml",
              ".STL": "application/octet-stream", ".stl": "application/octet-stream",
              ".dae": "model/vnd.collada+xml", ".json": "application/json",
              ".png": "image/png"}.get(os.path.splitext(p)[1], "application/octet-stream")
        self._send(200, open(p, "rb").read(), ct)

    def _file(self, p):
        p = os.path.abspath(p)
        ok = (p.startswith(os.path.join(BASE, "output")) or p.startswith(RUN)
              or p.startswith(os.path.join(BASE, "station", "static"))
              or p.startswith(os.path.join(BASE, "runs")))
        if not ok or not os.path.exists(p):
            return self._send(404, {"err": "no file"})
        ext = os.path.splitext(p)[1]
        ct = {".png": "image/png", ".json": "application/json",
              ".csv": "text/csv", ".md": "text/markdown"}.get(ext, "application/octet-stream")
        self._send(200, open(p, "rb").read(), ct)


def np_deg(x):
    import math
    return math.degrees(x or 0.349)


WS_CACHE = os.path.join(BASE, "output", "ws_cache")


def ensure_workspace(robot, waist_deg):
    """按腰限角度缓存工作区; 首次遇到该角度时重算 02(约3分钟), 之后秒级命中.
    返回 (ws_dir, created)."""
    d = os.path.join(WS_CACHE, f"{robot}_rp{int(waist_deg)}")
    os.makedirs(d, exist_ok=True)
    if not os.path.exists(os.path.join(d, "g1_workspace.npz")):
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
        e["G1W_WS"] = os.path.join(ws_dir, "g1_workspace.npz")
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
    for f in ("fig_wall_path_2d.png", "fig_robot_path_2d.png",
              "fig_3d_scene.png", "cloud_report.png", "base_comp_report.png"):
        if rd and os.path.exists(os.path.join(rd, f)):
            out.append({"name": f, "url": "/run/" + f})
        elif os.path.exists(os.path.join(BASE, "output", f)):
            out.append({"name": f, "url": "/fig/" + f})
        elif os.path.exists(os.path.join(RUN, f)):
            out.append({"name": f, "url": "/run/" + f})
    return out


def package():
    ver = JOB.get("motion_ver") or REGISTRY[JOB["robot"]]["motion_ver"]
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
              "motion_meta.json", "replay_motion.mp4", "fig_replay_contactsheet.png"):
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
main{display:grid;grid-template-columns:360px 1fr;gap:14px;padding:14px 22px 28px;max-width:1500px;margin:auto}
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
  <select id=robot onchange="loadRobotModel(this.value);markStale();renderChips()"></select>
  <div id=rview><canvas id=rcanvas></canvas><div id=rhint class=small>3D 模型: 拖动旋转 / 滚轮缩放</div></div>
  <div class=sec>作业参数</div>
  <div id=gated_inputs>
  <div class=row2>
    <div><label>站位离墙 (m)</label>
      <input id=dist type=number step=0.01 min=0.70 max=0.78 value=0.74 onchange=markStale()></div>
    <div><label>行距 (m)</label>
      <input id=pitch type=number step=0.01 value=0.10 onchange=markStale()></div>
  </div>
  <label>腰 roll/pitch 限位档位 (即选即用)</label>
  <select id=waist onchange="markStale();renderChips()"></select>
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
  loadRobotModel($('robot').value); renderChips();
  const nn = init.nn['g1'];
  $('nn_badge').innerHTML = nn&&nn.deployed ? `NN-IK: g1 已部署 (数据 ${nn.n_data})` : 'NN-IK: 未部署(回退快速IK)';
  $('ros_badge').textContent = 'ROS: ' + (init.ros_live?'rospy 实发':'topic 记录文件');
  const ri=(init.robots||{})[$('robot').value]||{};
  $('robot_badge').textContent = '机器人: '+$('robot').value+
    (ri.waist_deg!=null?` · 腰 roll/pitch ±${ri.waist_deg}°`:'');
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
  const j=await r.json(); upFile=j.file;
  $('cloud_name').textContent='已上传: '+j.file; markStale();
  { const b=document.getElementById('b1'); if(b) b.disabled=false;   // 选完即可点(门控由轮询校正)
    const h=document.getElementById('b1_hint'); if(h) h.style.display='none'; }
};
let _nnCache={};
function renderChips(nn){
  if(nn) _nnCache=nn;
  const rb=$('robot')?.value||'g1', cur=+(document.getElementById('waist')?.value||5);
  $('nn_badge').innerHTML = 'NN-IK '+[5,10,15,20].map(d=>{
    const ok=_nnCache[rb+'_rp'+d];
    return `<span class="chip${ok&&ok.deployed?' on':''}${cur===d?' cur':''}">${d}°</span>`;
  }).join('');
}
async function doPlan(){
  if(upFile==null && $('cloud').files.length){ /*未上传完*/ }
  tPlan=performance.now();
  const r=await fetch('/api/plan',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({robot:$('robot').value,dist:+$('dist').value,pitch:+$('pitch').value,waist_deg:+$('waist').value,cloud:upFile,ros:$('ros').checked})});
  const j=await r.json(); if(j.err) alert(j.err);
}
function countUp(el){
  const t=el.textContent; const m=t.match(/^[+-]?\d+(\.\d+)?$/); if(!m) return;
  const v=parseFloat(t); const dec=(t.split('.')[1]||'').length; const t0=performance.now();
  (function f(now){const p=Math.min((now-t0)/700,1);const e=1-Math.pow(1-p,3);
    el.textContent=(v*e).toFixed(dec);
    if(p<1) requestAnimationFrame(f); else el.textContent=t;})(t0);
}
function buildInspect(s){
  const fmap=Object.fromEntries(((s.figs&&s.figs.length?s.figs:init.figs)||[]).map(x=>[x.name,x.url]));
  const t=s.metrics.traj, m=s.metrics.motion||null, c=s.metrics.cloud||null;
  const w=t.stroke_u? (t.stroke_u[1]-t.stroke_u[0]).toFixed(3):'-';
  const mono=(t.z_w||[]).every((v,i,a)=>i===0||v>a[i-1]);
  const caps={'base_comp_report.png':'基座扰动补偿验证(世界系锁定)','fig_wall_path_2d.png':'墙面之字形(含点云背景/障碍框)','fig_robot_path_2d.png':'腕心轨迹与指向(点云背景)','fig_3d_scene.png':'3D 全景','cloud_report.png':'你的点云报告'};
  const names=['fig_wall_path_2d.png','fig_robot_path_2d.png','fig_3d_scene.png','base_comp_report.png'];
  if(c) names.push('cloud_report.png');
  const v=Date.now();
  const sum=[`<div class=sumitem><b>${t.points??'-'}</b> 点 / ${t.rows??'-'} 行</div>`,
    `<div class=sumitem>行角 <b>${(t.angles||[]).join('/')}</b></div>`,
    `<div class=sumitem>等宽 <b>${w} m</b> (u ${t.stroke_u?t.stroke_u.map(x=>x.toFixed(3)).join('~'):''})</div>`,
    `<div class=sumitem>实测夹角 <b>${t.angle_range?t.angle_range.map(x=>x.toFixed(1)).join('~'):'-'}°</b> ${t.angle_ok?'<span class=badge-ok></span>':'<span class=badge-bad></span>'}</div>`,
    `<div class=sumitem>z_w 叠层 <b>${mono?'单调':'异常'}</b>${mono?'<span class=badge-ok></span>':'<span class=badge-bad></span>'}</div>`];
  if(m&&m.frames) sum.push(`<div class=sumitem>动作 <b>${m.frames}</b> 帧 / ${m.duration_s}s / 峰速 ${m.max_vel} rad/s</div>`);
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
  clearTimeout(_rTimer); _rTimer=setTimeout(()=>_loadUrdf(info.urdf),50);
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
  toast(j.estopped? '★ 已软急停: 机器人变软(阻尼保护态), 程序仍在运行' : '已软急停 (当前无运行中的机器人)');
}
async function doRestartRobot(){
  toast('正在重启机器人(退出→重新启动, 约 15s)…');
  const j=await (await fetch('/api/robot/restart',{method:'POST'})).json().catch(()=>({err:'请求失败'}));
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
  const j=await (await fetch('/api/robot/start',{method:'POST'})).json();
  if(j.err){ toast(j.err); return; }
  // 启动机器人 = 新会话起点: 清空旧结果与作废标记, 界面回到干净初始态
  _stale=false; _inspShown=0; _sig=null;
  $('inspect_sec').innerHTML=''; $('insp_btns').style.display='none';
  $('metrics').innerHTML='';
  const es2=$('estop'); if(es2) es2.classList.remove('soft');
  toast(j.already?'机器人已在运行':'机器人启动中, 稍候回到初始站姿');
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
    const r=await fetch('/api/exec',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({speed:+spd})});
    const j=await r.json();
    if(j.err) toast(j.err); else toast(`已下发执行(×${spd}) — 机器人开始按轨迹喷涂`);
  }catch(e){ toast('执行调用失败: '+e); }
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
setInterval(async()=>{
  const s=await (await fetch('/api/status')).json();
  const busy=s.state==='planning'||s.state==='motion';
  if(s.state==='planning') _stale=false;
  const eff=(_stale&&['inspect','ready','error'].includes(s.state))?'idle':s.state;
  const lb=document.getElementById('logbox'); if(lb) lb.open = s.state==='error';
  if(s.waist_deg!=null) $('robot_badge').textContent =
    '机器人: '+$('robot').value+' · 生效腰限 ±'+s.waist_deg+'°';
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
  {   // 机器人相关 UI: 每次轮询都刷新(轻量), 保证门槛/按钮即时跟随
    const rb=(s.robot||{}); const rph=rb.phase||(rb.running?'starting':'stopped');
  window._robotPhase = rb.running? rph : 'stopped';
  const rmap={stopped:'机器人未启动', starting:'机器人启动中…', standing:'机器人已就位 (初始站姿, 待执行)',
              executing:'★ 正在执行喷涂动作(上肢跟随)…', soft_stopped:'已停止(GR00T 原生急停, 机器人变软)',
              playing:`喷涂执行中 ${Math.round((rb.progress||0)*100)}%`, done:'本轮执行完成'};
  const rcol={stopped:'', starting:'warn', standing:'ok', playing:'warn', done:'ok'}[rph]||'';
  $('robot_stat').innerHTML = `<span class=${rcol}>${rmap[rph]||rph}</span>`+
    (rb.done_runs?` <span class=small>已完成 ${rb.done_runs} 轮</span>`:'');
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
  $('loglog').textContent = (s.log||[]).join('\\n');
  $('loglog').scrollTop = 1e9;
  const cloudReady = !!upFile;                 // 已成功上传点云
  const ready2go = (s.robot||{}).running && cloudReady
                   && !['planning','motion'].includes(s.state);
  $('b1').disabled = !ready2go;
  { const h=$('b1_hint');
    if(h){
      h.style.display = ready2go? 'none':'block';
      h.textContent = !(s.robot||{}).running? '请先启动机器人'
        : (!cloudReady? '请先选择点云文件 (PLY/PCD/NPY/TXT)' : '');
    } }
  // 右栏按钮: 待检查时显示"检查无误→生成动作", 动作就绪时切换为"执行动作"+速度选择
  const bEx=$('bExec'), bAp=$('bApprove'), bSp=$('exec_speed');
  if(bEx && bAp){
    const ready=(s.state==='ready') && !_stale;
    bAp.style.display = (!ready && s.state==='inspect' && !_stale)? 'block':'none';
    bEx.style.display = ready? 'block':'none';
    if(bSp) bSp.style.display = ready? 'block':'none';
    if(ready){
      const rok=(s.robot||{}).running, rplaying=(s.robot||{}).phase==='playing';
      bEx.disabled=(!rok)||rplaying;
      if(bSp) bSp.disabled=(!rok)||rplaying;
      bEx.className=(rok&&!rplaying)?'go':'';
      bEx.textContent = rplaying? '喷涂执行中…' : (rok? '▶ 执行动作' : '▶ 执行动作 (需先启动机器人)');
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
  div.innerHTML = names.map(n=>`<h3>${n}</h3><img src="${init.figs.includes(n)?'/fig/'+n:'/run/'+n}">`).join('')
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    a = ap.parse_args()
    try:
        st = json.load(open(STATE_F, encoding="utf8"))
        JOB.update({k: st[k] for k in ("state", "robot", "motion_ver", "run_dir",
                                       "metrics", "artifacts", "ros_mode")
                    if k in st})
        if JOB.get("run_dir"):
            log("已恢复上次运行: " + JOB["run_dir"])
    except Exception:
        pass
    log("工作站启动: http://127.0.0.1:%d" % a.port)
    threading.Thread(target=_deploy_watchdog, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
