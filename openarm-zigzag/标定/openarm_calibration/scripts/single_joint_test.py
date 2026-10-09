#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
单关节手动检测工具（诊断某个关节能动性/通信质量）
==================================================
用法:
  sudo python3 single_joint_test.py [joint 1-8] [幅度°] [往返次数]

流程: 使能该关节 → 读使能后状态码 → 小幅往返运动(标定同速步进) → 失能
输出: 实时位置/速度/力矩/状态码，用于判断:
  - 使能后状态码应为 0x01(ENABLED)，否则电机处于故障/未响应
  - 位置应跟随目标（不跟随=通信丢帧或电机堵转故障）
  - 全程无反馈=该电机 CAN 通信断
"""
import math
import socket
import struct
import sys
import time

sys.path.insert(0, "/home/yang/桌面/openarm实验/标定/openarm_calibration")
from openarm_calibration import dm_canfd as dm  # noqa: E402

D2R = math.pi / 180.0

# 实测寄存器限参 (pmax, vmax, tmax)
LIMITS = {1: (12.5, 45.0, 54.0), 2: (12.5, 45.0, 54.0),
          3: (12.5, 20.0, 28.0), 4: (12.5, 20.0, 28.0),
          5: (12.5, 50.0, 10.0), 6: (12.5, 50.0, 10.0),
          7: (12.5, 50.0, 10.0), 8: (12.5, 50.0, 10.0)}
STATE_NAMES = {0x0: "DISABLED", 0x1: "ENABLED", 0x8: "过压", 0x9: "欠压",
               0xA: "过流", 0xB: "MOS过温", 0xC: "线圈过温",
               0xD: "通讯丢失", 0xE: "过载"}


class Bus:
    CAN_RAW_FD_FRAMES = 5

    def __init__(self, iface="can0"):
        self.sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW,
                                  socket.CAN_RAW)
        self.sock.setsockopt(socket.SOL_CAN_RAW,
                             self.CAN_RAW_FD_FRAMES, 1)
        self.sock.bind((iface,))
        self.sock.setblocking(False)

    def send(self, can_id, data):
        frame = (struct.pack("<IBB2x", can_id, len(data), dm.CanFdBus.CANFD_BRS)
                 + data.ljust(64, b"\x00"))
        for _ in range(25):
            try:
                self.sock.send(frame)
                return
            except OSError as e:
                if e.errno == 105:
                    time.sleep(0.002)
                    continue
                raise

    def recv(self, timeout=0.002):
        if timeout > 0:
            self.sock.settimeout(timeout)
            try:
                frame, _ = self.sock.recvfrom(72)
            except socket.timeout:
                return None
        else:
            self.sock.settimeout(0.0)
            try:
                frame, _ = self.sock.recvfrom(72)
            except BlockingIOError:
                return None
        if len(frame) == 72:
            can_id, dlen = struct.unpack("<IB", frame[:5])
        else:
            can_id, dlen = struct.unpack("<IB", frame[:5])
        return can_id & 0x7FF, frame[8:8 + dlen]

    def drain(self, dur=0.004):
        t0 = time.monotonic()
        while time.monotonic() - t0 < dur:
            self.recv(0.001)


def mit(bus, esc, lim, kp, kd, q):
    bus.send(*dm.build_mit(esc, lim, kp, kd, q, 0.0, 0.0))
    bus.drain(0.002)


def read_state(bus, recv_id, lim, debug=False):
    """发 refresh 收状态帧；debug=True 时打印所有原始帧。"""
    bus.send(*dm.build_refresh(recv_id - 0x10))
    t0 = time.monotonic()
    while time.monotonic() - t0 < 0.2:
        got = bus.recv(0.005)
        if got is None:
            continue
        if debug:
            print("  [debug] id=0x%03X payload=%s" % (got[0], got[1].hex()))
        if got[0] == recv_id:
            st = dm.parse_state(got[1], lim)
            if st:
                return st
    return None


def main():
    joint = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    amp_deg = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    cycles = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    # 第 4 参数: 运动方向 +1/-1（默认 -1 负方向，如 J1 正方向有限位时用 -1）
    direction = float(sys.argv[4]) if len(sys.argv) > 4 else -1.0
    if not (1 <= joint <= 8):
        print("关节号 1-8")
        return
    esc = joint
    recv_id = esc + 0x10
    lim = LIMITS[joint]
    amp = amp_deg * D2R

    bus = Bus()
    print("=== 单关节检测: J%d (0x%02X/反馈0x%02X) 幅度%.1f° 往返%d次 ==="
          % (joint, esc, recv_id, amp_deg, cycles))

    # 1. 使能
    bus.send(*dm.build_enable(esc))
    bus.drain(0.05)
    st = read_state(bus, recv_id, lim, debug=True)
    if st is None:
        print("[FAIL] 使能后无反馈帧——该电机 CAN 通信异常")
        return
    print("使能后状态: 0x%02X %s" % (st["state_code"],
                                     STATE_NAMES.get(st["state_code"], "?")))
    if st["state_code"] not in (0x01, esc, recv_id):
        # 此固件应答 byte0 回显 ID: refresh 应答回显 ESC_ID(0x06)、
        # 使能应答回显反馈 ID(0x16)——均属正常；仅 0x8-0xE 为真实故障码
        print("[FAIL] 电机未进入 ENABLED——检查供电/故障码，中止")
        bus.send(*dm.build_disable(esc))
        return

    # 2. 记录初始位置
    st = read_state(bus, recv_id, lim)
    q0 = st["q"]
    print("初始位置: %.4f rad" % q0)

    # 3. 往返运动（与标定 bump_to_limit 完全相同的节奏:
    #    MIT帧 → sleep 5ms → 收一帧反馈，每步 0.2°，10s 超时保护）
    kp, kd = 45.0, 1.2
    step = 0.2 * D2R
    # 单方向往返: 去（direction 方向 amp 度）→ 回（零位）
    targets = [q0 + direction * amp, q0]
    fail = False
    for q_target in targets:
        print("--- 目标 %.4f rad (%+0.1f°) ---" % (q_target,
                                                   (q_target - q0) * 180 / math.pi))
        deadline = time.monotonic() + 10.0
        reached = False
        last_print = 0.0
        direction = 1.0 if q_target > q0 else -1.0
        q_target_step = q0
        st = None
        overshoot = math.radians(10.0)   # 允许目标越程 10° 破静摩擦
        tau_lim = min(20.0, lim[2] * 0.5)
        tau_stall = 0
        while time.monotonic() < deadline:
            q_target_step += direction * step
            if (q_target_step - q_target) * direction >= overshoot:
                print("[FAIL] 目标越程 10° 仍未到位(q=%+.4f, tau=%+.1f)"
                      "——关节受阻" % (st["q"] if st else float('nan'),
                                       st["tau"] if st else 0.0))
                fail = True
                break
            bus.send(*dm.build_mit(esc, lim, kp, kd, q_target_step, 0.0, 0.0))
            time.sleep(0.005)
            got = bus.recv(0.002)
            if got and got[0] == recv_id:
                st = dm.parse_state(got[1], lim)
            now = time.monotonic()
            if st and now - last_print > 0.4:
                print("  q=%+.4f dq=%+.3f tau=%+.3f"
                      % (st["q"], st["dq"], st["tau"]))
                last_print = now
            if st and abs(st["q"] - q_target) < math.radians(1.0):
                print("  到位 (q=%+.4f)" % st["q"])
                reached = True
                break
            if st and abs(st["tau"]) > tau_lim and abs(st["dq"]) < 0.05:
                tau_stall += 1
            else:
                tau_stall = 0
            if tau_stall >= 30:
                print("[FAIL] 堵转保护触发（tau=%+.1f Nm 持续，q=%+.4f）"
                      "——遇硬阻挡" % (st["tau"], st["q"]))
                fail = True
                break
        if not reached:
            print("[FAIL] 10s 内未到达目标（最后 q=%+.4f）"
                  "——该关节不动或被卡" % (st["q"] if st else float('nan')))
            fail = True
            break
    print("=== 检测完成: %s，失能 ===" % ("通过" if not fail else "存在异常"))
    bus.send(*dm.build_disable(esc))
    bus.drain(0.05)


if __name__ == "__main__":
    main()
