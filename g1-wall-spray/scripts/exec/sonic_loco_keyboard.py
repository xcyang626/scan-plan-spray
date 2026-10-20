#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SONIC 全身策略 键盘位移/速度/朝向 测试工具（先仿真、后实机）。

背景: 喷涂链路目前给 SONIC 只发 mode/height/上肢17关节, movement 恒 (0,0,0)、
facing 恒 (1,0,0)、speed 恒 -1。本工具把这些"运动指令"字段接到键盘上,
测试 SONIC 对 x/y 位移、位移速度、朝向输入的响应。

SONIC 官方语义 (zmq_manager 收 planner 话题 -> MovementState, 无位置目标):
  movement = 位移方向单位向量 [x,y,z] (朝向系, 接收端会归一化)
  speed    = 位移速度 m/s (SLOW_WALK=1 模式范围 0.1~0.8, -1=模式默认)
  facing   = 朝向单位向量 [x,y,z] (x=0°, y=+90°)
  部署端 1s 收不到 planner 消息会清缓存(机器人停) —— 本工具持续 30Hz 发布。

按键 (终端单字符、小写, 无需回车; 大写一律忽略, 防方向键转义序列误触发):
  w / s      位移 +x / -x (前进/后退, 按住持续, 松开 0.35s 内归零)
  a / d      位移 +y / -y (左/右平移, 可与 w/s 组合成斜向)
  q / e      朝向 -5° / +5° (限 |θ| ≤ 180°, 即全向 360° 覆盖)
  9 / 0      速度 -0.1 / +0.1 m/s (官方 SLOW_WALK 范围 0.2 ~ 0.8)
  z          停止位移 (回 IDLE)
  r          朝向归零
  x          全部归零 (位移 + 朝向)
  ]          启动控制 (官方 command start, planner 模式)
  o          停止按钮: 急停(全身变软+部署退出)并退出本工具 (stop 连发×3)
  p          打印当前状态
  Ctrl-C     同 o: 发 IDLE + stop×3 后退出

安全设计:
  - 速度用官方语义: SLOW_WALK 钳位 [0.2, 0.8] m/s (默认 0.2), 9/0 步进 ±0.1
    (与官方键盘 handler 同款数值; 更快需换 WALK 模式, 官方对其用模式默认速度);
  - 朝向 |θ| ≤ 180° (默认, 全向可达), 步进 5°;
  - 按住才走: 松键 0.35s 即回 IDLE, 不存在"发完指令忘了停";
  - 发送端一死, 部署端 1s 超时自停; Ctrl-C / o 均补发官方 stop。

运行环境: 依赖 msgpack/zmq/numpy —— 用 GR00T venv:
  ~/桌面/轨迹规划/GR00T-WholeBodyControl/.venv_sim_cpu/bin/python sonic_loco_keyboard.py

仿真联调 (前两个终端在 GR00T-WholeBodyControl/ 下):
  1) ./run_official_sim.sh                                        # MuJoCo 仿真
  2) ./run_official_deploy.sh --input-type zmq_manager \
        --zmq-host localhost --zmq-port 5556                      # SONIC 部署端
  3) 本工具 (用上面 venv 的 python), 按 ] 启动控制,
     等 MuJoCo 里机器人站稳后 w/a/s/d/q/e 小幅测试。
  注意: 仿真默认开"弹力带"吊住机器人(不会走)。两种解除方式:
  在 MuJoCo 窗口按 9 切换; 或把 wbc_configs/g1_29dof_sonic_model12.yaml 的
  ENABLE_ELASTIC_BAND 改 False(需 base_sim.py 的创建逻辑补丁, 已打, 否则崩)。
实机: 部署端同样以 --input-type zmq_manager 启动(网卡用实机口), 本工具默认
  绑 env.json 的 zmq_host(无配置 127.0.0.1):5556; 与 station app 的 ZMQ 流同
  端口, 两者不可同时绑 —— 实机接入时先停 station 流或让本工具独占。
"""
from __future__ import annotations

import math
import os
import select
import sys
import termios
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from groot_bridge import Publisher, DEFAULT_ZMQ_HOST, MODE_IDLE, build_planner_message  # noqa: E402

MODE_SLOW_WALK = 1   # 官方 LocomotionMode::SLOW_WALK
# 官方速度语义 (keyboard_handler 同款): SLOW_WALK 钳位 [0.2, 0.8] m/s, 9/0 步进 ±0.1
SPEED_MIN_OFFICIAL = 0.2

# 语义方向 -> 朝向系单位向量 (x 前, y 左)
DIR_VEC = {"w": (1.0, 0.0), "s": (-1.0, 0.0), "a": (0.0, 1.0), "d": (0.0, -1.0)}


class DriveState:
    """位移方向(按住生效) + 速度标量 + 朝向角, 全部带安全限幅。"""

    def __init__(self, speed, max_speed, speed_step, heading_step_deg, max_heading_deg, hold_s):
        self.speed = float(speed)
        self.max_speed = float(max_speed)
        self.speed_step = float(speed_step)
        self.heading_step_deg = float(heading_step_deg)
        self.max_heading_deg = float(max_heading_deg)
        self.hold_s = float(hold_s)
        self.heading_deg = 0.0
        self.last_seen = {}          # 方向键 -> 最近一次按键时刻(按住=持续到来)

    # ---- 键处理 ----
    def on_key(self, ch, now):
        if ch in DIR_VEC:
            self.last_seen[ch] = now
            return True
        if ch == "q":
            self.heading_deg = max(self.heading_deg - self.heading_step_deg, -self.max_heading_deg)
            return True
        if ch == "e":
            self.heading_deg = min(self.heading_deg + self.heading_step_deg, self.max_heading_deg)
            return True
        if ch == "9":
            self.speed = max(self.speed - self.speed_step, SPEED_MIN_OFFICIAL)
            return True
        if ch == "0":
            self.speed = min(self.speed + self.speed_step, self.max_speed)
            return True
        if ch == "z":
            self.last_seen.clear()
            return True
        if ch == "r":
            self.heading_deg = 0.0
            return True
        if ch == "x":
            self.last_seen.clear()
            self.heading_deg = 0.0
            return True
        return False

    # ---- 每个发布 tick 计算 ----
    def active_dirs(self, now):
        return {k for k, t in self.last_seen.items() if now - t <= self.hold_s}

    def movement_vec(self, now):
        """朝向系方向和 -> 归一化世界向量 (旋转 heading)。无激活方向返回 None。"""
        dirs = self.active_dirs(now)
        if not dirs:
            return None
        dx = sum(DIR_VEC[k][0] for k in dirs)
        dy = sum(DIR_VEC[k][1] for k in dirs)
        h = math.radians(self.heading_deg)
        c, s = math.cos(h), math.sin(h)
        vx, vy = dx * c - dy * s, dx * s + dy * c
        n = math.hypot(vx, vy)
        if n < 1e-6:                       # 对冲组合 (如同时按 w+s)
            return None
        return (vx / n, vy / n, 0.0)

    def facing_vec(self):
        h = math.radians(self.heading_deg)
        return (math.cos(h), math.sin(h), 0.0)


def fmt_state(st: DriveState, now) -> str:
    mv = st.movement_vec(now)
    mv_s = f"({mv[0]:+.2f},{mv[1]:+.2f})" if mv else "  停  "
    mode_s = "SLOW_WALK" if mv else "IDLE"
    return (f"mode={mode_s} movement={mv_s} speed={st.speed:.2f}m/s "
            f"heading={st.heading_deg:+.0f}°")


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="SONIC 键盘位移/速度/朝向测试 (planner 话题, 30Hz 连续发布)")
    ap.add_argument("--host", default=DEFAULT_ZMQ_HOST,
                    help="ZMQ 绑定地址 (默认 env.json zmq_host, 无配置 127.0.0.1)")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--speed", type=float, default=0.2,
                    help="初始速度 m/s (官方 SLOW_WALK 范围 0.2~0.8)")
    ap.add_argument("--max-speed", type=float, default=0.8,
                    help="速度上限 m/s (官方 SLOW_WALK 钳位 0.8)")
    ap.add_argument("--speed-step", type=float, default=0.1,
                    help="9/0 速度步进 m/s (官方 ±0.1)")
    ap.add_argument("--heading-step-deg", type=float, default=5.0)
    ap.add_argument("--max-heading-deg", type=float, default=180.0,
                    help="朝向限幅(±度), 180=全向 360° 覆盖")
    ap.add_argument("--rate", type=float, default=30.0, help="planner 发布频率 Hz")
    ap.add_argument("--hold", type=float, default=0.35,
                    help="松键后多少秒归零位移")
    ap.add_argument("--selftest", action="store_true", help="离线自检报文构造, 不联网")
    a = ap.parse_args()

    # 官方 SLOW_WALK 速度语义: 启动即把初值/上限收进 [0.2, 0.8], 防超范围初值
    # (策略分布外输入); 用户显式传更小上限(首跑收紧)仍尊重, 只压到其上限
    a.max_speed = min(a.max_speed, 0.8)
    lo = min(SPEED_MIN_OFFICIAL, a.max_speed)
    a_speed_clamped = min(max(a.speed, lo), a.max_speed)
    if a_speed_clamped != a.speed:
        a.speed = a_speed_clamped
        print(f"[sonic-loco] 初始速度已收进官方范围 [{lo:.2f}, {a.max_speed:.2f}]: "
              f"{a.speed:.2f} m/s")

    st = DriveState(a.speed, a.max_speed, a.speed_step,
                    a.heading_step_deg, a.max_heading_deg, a.hold)

    if a.selftest:
        m1 = build_planner_message(mode=MODE_SLOW_WALK, movement=[1, 0, 0],
                                   facing=[1, 0, 0], speed=0.15, height=-1.0)
        m2 = build_planner_message(mode=MODE_IDLE, movement=[0, 0, 0],
                                   facing=[1, 0, 0], speed=-1.0, height=-1.0)
        # heading=+90° 时: facing=(0,1), w 的位移向量应为 (0,1)
        st.heading_deg = 90.0
        st.last_seen = {"w": time.monotonic()}
        mv = st.movement_vec(time.monotonic())
        fc = st.facing_vec()
        assert mv and abs(mv[0]) < 1e-9 and abs(mv[1] - 1) < 1e-9, mv
        assert abs(fc[0]) < 1e-9 and abs(fc[1] - 1) < 1e-9, fc
        st.heading_deg = 45.0
        st.last_seen = {"w": time.monotonic(), "a": time.monotonic()}
        mv = st.movement_vec(time.monotonic())   # 45° 前左斜和 -> 归一化 (0,1)
        assert mv and abs(mv[0]) < 1e-9 and abs(mv[1] - 1) < 1e-9, mv
        st.heading_deg = 50.0                     # 超限幅不会发生(键步进已限), 直读校验限幅逻辑
        st.heading_deg = min(st.heading_deg, st.max_heading_deg)
        assert st.heading_deg == 50.0
        # 速度官方钳位: 9 降不下 0.2, 0 (配默认上限) 升不过 0.8
        st.speed, st.max_speed = 0.25, 0.8
        st.on_key("9", 0); st.on_key("9", 0)
        assert st.speed == SPEED_MIN_OFFICIAL, st.speed
        for _ in range(8):
            st.on_key("0", 0)
        assert st.speed == 0.8, st.speed
        # 朝向限幅: e 连按封顶 +180, q 连按封底 -180
        st.heading_deg = 0.0
        for _ in range(40):
            st.on_key("e", 0)
        assert st.heading_deg == 180.0, st.heading_deg
        for _ in range(80):
            st.on_key("q", 0)
        assert st.heading_deg == -180.0, st.heading_deg
        print(f"[自测] planner 走行报文 {len(m1)} 字节 topic={m1[:7]}")
        print(f"[自测] planner 停止报文 {len(m2)} 字节 topic={m2[:7]}")
        print(f"[自测] 朝向旋转/组合归一化/限幅 通过 —— {fmt_state(st, time.monotonic())}")
        return 0

    pub = Publisher(a.host, a.port)
    print(f"[sonic-loco] ZMQ PUB 绑定 tcp://{a.host}:{a.port}")
    print("[sonic-loco] w/s=+x/-x  a/d=+y/-y  q/e=朝向∓{:.0f}°(±{:.0f}°)  "
          "9/0=速度{:+.2f}  z=停  r=朝向零  x=全零  ]=启动  o=停止退出(变软)  Ctrl-C=同o".format(
              a.heading_step_deg, a.max_heading_deg, a.speed_step))

    # 终端裸模式(管道输入时跳过, 便于自动化测试)
    # 注意: python termios.tcgetattr 返回 7 元素列表 [iflag,oflag,cflag,lflag,ispeed,ospeed,cc]
    # (不是 C 结构体), lflag=列表[3], 控制字符=列表[6]
    tty = sys.stdin.isatty()
    old_term = None
    if tty:
        old_term = termios.tcgetattr(sys.stdin)
        nt = termios.tcgetattr(sys.stdin)
        nt[3] &= ~(termios.ICANON | termios.ECHO)
        nt[6][termios.VMIN] = 0
        nt[6][termios.VTIME] = 0
        termios.tcsetattr(sys.stdin, termios.TCSANOW, nt)
        os.set_blocking(sys.stdin.fileno(), False)

    interval = 1.0 / a.rate
    last_line = ""
    next_heartbeat = time.monotonic() + 3.0
    was_moving = False

    def handle_char(ch, now):
        if ch == "]":
            pub.command(start=True, stop=False, planner=True)
            print("\n[sonic-loco] 已下发 command start (planner 模式)")
        elif ch == "o":
            # 停止按钮: 阻尼急停(全身变软+部署退出)后本工具一并退出。
            # stop 命令连发3次再退(LINGER 0 会丢未送达消息, 与 daemon 同做法)
            pub.planner_locomotion(MODE_IDLE, [0, 0, 0], [1, 0, 0])
            st.last_seen.clear()
            for _ in range(3):
                pub.command(start=False, stop=True, planner=False)
                time.sleep(0.15)
            print("\n[sonic-loco] 已下发 急停 command stop ×3 (机器人变软+部署退出), 本工具退出")
            raise SystemExit(0)
        elif ch == "p":
            print(f"\n[sonic-loco] {fmt_state(st, now)}")
        else:
            st.on_key(ch, now)

    try:
        while True:
            t0 = time.monotonic()
            deadline = t0 + interval
            # ---- 读键: select 等到本拍结束, 有键即刻处理 ----
            while True:
                rem = deadline - time.monotonic()
                if rem <= 0:
                    break
                r, _, _ = select.select([sys.stdin], [], [], rem)
                if not r:
                    break
                try:
                    buf = os.read(sys.stdin.fileno(), 64)
                except (BlockingIOError, InterruptedError, OSError):
                    continue
                if not buf:               # 管道 EOF, 不再等键
                    break
                now = time.monotonic()
                for ch in buf.decode("utf-8", errors="ignore"):
                    handle_char(ch, now)
            # ---- 发布 planner ----
            now = time.monotonic()
            mv = st.movement_vec(now)
            moving = mv is not None
            pub.planner_locomotion(
                mode=(MODE_SLOW_WALK if moving else MODE_IDLE),
                movement=(list(mv) if moving else [0.0, 0.0, 0.0]),
                facing=list(st.facing_vec()),
                speed=(st.speed if moving else -1.0))
            if moving != was_moving:
                print(f"\n[sonic-loco] {fmt_state(st, now)}")
            line = fmt_state(st, now)
            if line != last_line and (moving or now >= next_heartbeat):
                print(f"\r[sonic-loco] {line}", end="", flush=True)
                last_line = line
                next_heartbeat = now + 3.0
            was_moving = moving
            # ---- 定拍 ----
            el = time.monotonic() - t0
            if el < interval:
                time.sleep(interval - el)
    except KeyboardInterrupt:
        pub.planner_locomotion(MODE_IDLE, [0, 0, 0], [1, 0, 0])
        for _ in range(3):
            pub.command(start=False, stop=True, planner=False)
            time.sleep(0.15)
        print("\n[sonic-loco] 退出前已发 IDLE + command stop ×3 (机器人变软+部署退出)")
    finally:
        if tty and old_term is not None:
            termios.tcsetattr(sys.stdin, termios.TCSANOW, old_term)
        pub.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
