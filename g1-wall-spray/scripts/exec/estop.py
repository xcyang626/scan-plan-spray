# -*- coding: utf-8 -*-
"""急停 —— 封装 GR00T 原生急停行为。

GR00T 原生急停行为（官方实现，见 gear_sonic_deploy）：
  command(stop=True) 或 键盘 O / 手柄 Select
  → operator_state.stop = true
  → 控制线程退出 → CreateDampingCommand() 下发**零力矩阻尼指令**(kp=0, kd=8)
  → 机器人全身变软, 随后进程优雅退出

另外（GR00T 原生、无需我们实现）：
  - 机器人连接丢失: CheckSafety() 检测 LowState >500ms → 安全停止
  - 动作源丢失:     planner 话题 >1s 无消息 → ZMQManager 自动重置为 IDLE
                    (has_upper_body_control_=false, 上肢回到策略默认, 安全站立)

下发通道(按序回退, 2026-10-09 修复 mujoco 环境缺 msgpack 导致的必失败):
  1) 执行守护进程 TCP 控制口(station 启动后常驻, 独占 ZMQ 5556)
  2) GR00T venv 子进程直发官方 command(需部署未占用端口)
  3) 部署进程 stdin FIFO 注入官方 'O' 键

用法:
  python estop.py            # 急停(官方 command stop)
  python estop.py --keys     # 备用: 直接向部署进程 stdin 注入官方 'O'
  python estop.py --temp     # 报告电机温度(官方 F)
  python estop.py --reinit   # 重置基座朝向(官方 I)
"""
from __future__ import annotations
import argparse, json, os, socket, subprocess, sys, time

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GROOT_ROOT = os.environ.get("GROOT_ROOT", "/home/yang/桌面/轨迹规划/GR00T-WholeBodyControl")
FIFO = os.path.join(BASE, "station", "robot_stdin.fifo")
STATUS_F = os.path.join(BASE, "station", "robot_status.json")
BRIDGE = os.path.join(BASE, "scripts", "exec", "groot_bridge.py")


def _cfg(key, default):
    try:
        return json.load(open(os.path.join(BASE, "station", "env.json"),
                              encoding="utf8")).get(key, default)
    except Exception:
        return default


def _groot_py():
    p = _cfg("groot_python", os.path.join(GROOT_ROOT, ".venv_sim_cpu/bin/python"))
    return p if os.path.exists(p) else sys.executable


def daemon_call(req, timeout=4.0):
    """经执行守护进程的本地 TCP 控制口发请求(一行应答)."""
    port = int(_cfg("daemon_port", 5558))
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        s.settimeout(timeout)
        s.sendall(json.dumps(req).encode() + b"\n")
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
        s.close()
        return json.loads(buf.split(b"\n", 1)[0].decode("utf8", "ignore"))
    except Exception:
        return None


def _send_command_subprocess(kind="stop"):
    """兜底: GR00T venv 子进程直发官方 command(msgpack 只在 venv 里有)."""
    try:
        r = subprocess.run(
            [_groot_py(), BRIDGE, "--send-command", kind,
             "--host", "*", "--port", str(int(_cfg("zmq_port", 5556)))],
            cwd=GROOT_ROOT, capture_output=True, text=True, timeout=30)
        return r.returncode == 0
    except Exception:
        return False


def estop(verbose=True):
    """GR00T 原生急停: 经官方 command 话题下发 stop=True."""
    # 通道1: 常驻守护进程(独占 ZMQ 端口, 站点在运行时它就在)
    r = daemon_call({"cmd": "command", "kind": "stop"})
    if r and r.get("ok"):
        _mark_estopped()
        if verbose:
            print("[estop] 已经守护进程下发 GR00T 官方急停 command(stop=True): "
                  "机器人将收到零力矩阻尼指令(全身变软)后优雅停止")
        return True
    # 通道2: GR00T venv 子进程直发
    if _send_command_subprocess("stop"):
        _mark_estopped()
        if verbose:
            print("[estop] 已经子进程下发 GR00T 官方急停 command(stop=True): "
                  "机器人将收到零力矩阻尼指令(全身变软)后优雅停止")
        return True
    # 通道3: 部署 stdin FIFO 注入官方 'O' 键
    if send_key("O", verbose=False):
        _mark_estopped()
        if verbose:
            print("[estop] 已注入官方按键 'O' (GR00T 原生急停)")
        return True
    if verbose:
        print("[estop] ✗ 三条通道均失败: 守护进程不在运行、ZMQ 直发失败、"
              "FIFO 不可写 —— 请直接按手柄急停键")
    return False


def _mark_estopped():
    try:
        json.dump({"phase": "soft_stopped", "ts": time.time()},
                  open(STATUS_F, "w", encoding="utf8"))
    except Exception:
        pass


def send_key(ch, verbose=True):
    """备用通道: 直接向部署进程 stdin 注入官方按键."""
    try:
        fd = os.open(FIFO, os.O_WRONLY | os.O_NONBLOCK)
        try:
            os.write(fd, ch.encode())
        finally:
            os.close(fd)
        if verbose:
            print(f"[estop] 已注入官方按键 {ch!r}")
        return True
    except Exception as e:
        if verbose:
            print(f"[estop] 注入失败: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", action="store_true", help="备用: 注入官方 'O' 键")
    ap.add_argument("--temp", action="store_true", help="报告电机温度(官方 F)")
    ap.add_argument("--reinit", action="store_true", help="重置基座朝向(官方 I)")
    a = ap.parse_args()
    if a.temp:
        return 0 if send_key("F") else 2
    if a.reinit:
        return 0 if send_key("I") else 2
    if a.keys:
        return 0 if send_key("O") else 2
    return 0 if estop() else 2


if __name__ == "__main__":
    sys.exit(main())
