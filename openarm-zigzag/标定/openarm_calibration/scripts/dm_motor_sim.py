#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
达妙电机仿真器（在 vcan0 虚拟总线上模拟 8 台电机）
==================================================
用法: sudo /usr/bin/python3 dm_motor_sim.py [iface]

模拟内容:
  - 使能(0xFC)/失能(0xFD)/置零(0xFE) 命令
  - MIT 控制帧: 一阶惯性动力学跟踪目标位置
  - 机械限位: 到限位即堵转(dq=0, tau=kp*err 持续增大)——标定碰撞判定可真实触发
  - refresh(0xCC) 回状态帧; 参数查询(0x33) 回参数应答帧
  - 限位表: 官方 v1 机械限位（度），初始位形可配置

标定节点只需把 can_iface 改为 vcan0 即可全程仿真。
"""
import math
import os
import socket
import struct
import sys
import time

D2R = math.pi / 180.0

# 官方 MECH_LIM_V1（openarm-can-zero-position-calibration，出厂编码系，rad）
MECH_LIM_V1 = {
    "J1": (-80.0, 200.0),
    "J2": (-100.0, 100.0),
    "J3": (-90.0, 90.0),
    "J4": (0.0, 140.0),
    "J5": (-90.0, 90.0),
    "J6": (-45.0, 45.0),
    "J7": (-90.0, 90.0),
    "grip": (-60.0, 0.0),
}

# (esc_id, pmax, vmax, tmax, neg_limit_rad, pos_limit_rad) —— 限位 = 官方 V1 表
def _lim(name):
    lo, hi = MECH_LIM_V1[name]
    return (lo * D2R, hi * D2R)

MOTORS = [
    dict(esc=1, pmax=12.5, vmax=45.0, tmax=54.0, lim=_lim("J1")),
    dict(esc=2, pmax=12.5, vmax=45.0, tmax=54.0, lim=_lim("J2")),
    dict(esc=3, pmax=12.5, vmax=20.0, tmax=28.0, lim=_lim("J3")),
    dict(esc=4, pmax=12.5, vmax=20.0, tmax=28.0, lim=_lim("J4")),
    dict(esc=5, pmax=12.5, vmax=50.0, tmax=10.0, lim=_lim("J5")),
    dict(esc=6, pmax=12.5, vmax=50.0, tmax=10.0, lim=_lim("J6")),
    dict(esc=7, pmax=12.5, vmax=50.0, tmax=10.0, lim=_lim("J7")),
    dict(esc=8, pmax=12.5, vmax=50.0, tmax=10.0, lim=_lim("grip")),
]
# 初始位形（度，各关节处于官方限位中位附近，双向都有行程）
INITIAL_DEG = [60.0, 0.0, 0.0, 70.0, 0.0, 0.0, 0.0, -20.0]

DT = 0.001          # 动力学步长 1ms
VEL_GAIN = 8.0      # err(rad)→速度需求(rad/s) 增益
VEL_MAX = 5.0       # 最大角速度 rad/s（模拟真机速度量级）
TAU_FILT = 0.02     # 力矩响应时间常数

RID_UINT32 = set(range(7, 11)) | set(range(13, 17)) | {35, 36}


def d2u(x, mn, mx, bits):
    x = max(mn, min(x, mx))
    return int(round((x - mn) / (mx - mn) * ((1 << bits) - 1)))


def u2d(x, mn, mx, bits):
    return (x / ((1 << bits) - 1)) * (mx - mn) + mn


class SimMotor:
    def __init__(self, cfg, q0):
        self.cfg = cfg
        self.q = q0
        self.dq = 0.0
        self.tau_fb = 0.0
        self.enabled = False
        self.state_code = 0x00                  # 0=DISABLED 1=ENABLED
        self.q_des = q0
        self.kp = 0.0
        self.kd = 0.0
        self.tau_ff = 0.0
        self.neg_lim, self.pos_lim = cfg["lim"]
        self.pending_replies = []               # (can_id, payload)

    # ---------------- 动力学 ----------------

    def tick(self):
        if not self.enabled:
            self.dq = 0.0
            self.tau_fb = 0.0
            return
        err = self.q_des - self.q
        vel_des = max(-VEL_MAX, min(VEL_GAIN * err, VEL_MAX))
        # 一阶惯性趋近速度需求
        self.dq += (vel_des - self.dq) * min(1.0, DT * 20.0)
        self.q += self.dq * DT
        # 机械限位: 堵转（标定碰撞判定的触发来源）
        at_limit = False
        if self.q < self.neg_lim:
            self.q = self.neg_lim
            self.dq = 0.0
            at_limit = True
        elif self.q > self.pos_lim:
            self.q = self.pos_lim
            self.dq = 0.0
            at_limit = True
        # 力矩反馈 = 控制律输出（限位处随 err 持续增大）
        tau = self.kp * err + self.tau_ff
        if at_limit:
            tau = max(0.5, abs(tau))            # 堵转保持力
        self.tau_fb += (tau - self.tau_fb) * min(1.0, DT / TAU_FILT)
        self.tau_fb = max(-self.cfg["tmax"], min(self.tau_fb,
                                                 self.cfg["tmax"]))

    # ---------------- 帧处理 ----------------

    def state_payload(self):
        c = self.cfg
        qu = d2u(self.q, -c["pmax"], c["pmax"], 16)
        dqu = d2u(self.dq, -c["vmax"], c["vmax"], 12)
        tu = d2u(self.tau_fb, -c["tmax"], c["tmax"], 12)
        return bytes([
            self.state_code,
            (qu >> 8) & 0xFF, qu & 0xFF,
            dqu >> 4,
            ((dqu & 0xF) << 4) | ((tu >> 8) & 0xF),
            tu & 0xFF,
            34, 29])                            # MOS/转子温度(固定)

    def param_payload(self, rid, value):
        if rid in RID_UINT32:
            raw = struct.pack("<I", int(value))
        else:
            raw = struct.pack("<f", float(value))
        return bytes([self.cfg["esc"] & 0xFF, 0x00, 0x33, rid]) + raw

    def on_frame(self, can_id: int, payload: bytes):
        """处理一帧命令，必要时生成应答帧。"""
        esc = self.cfg["esc"]
        if can_id == esc and len(payload) >= 8:
            if payload[:7] == b"\xFF" * 7:
                cmd = payload[7]
                if cmd == 0xFC:                 # 使能
                    self.enabled = True
                    self.state_code = 0x01
                elif cmd == 0xFD:               # 失能
                    self.enabled = False
                    self.state_code = 0x00
                elif cmd == 0xFE:               # 置零
                    self.q = 0.0
                self.pending_replies.append(
                    (esc + 0x10, self.state_payload()))
                return
            # MIT 控制帧
            c = self.cfg
            qu = (payload[0] << 8) | payload[1]
            dqu = (payload[2] << 4) | (payload[3] >> 4)
            kp_u = ((payload[3] & 0xF) << 8) | payload[4]
            kd_u = (payload[5] << 4) | (payload[6] >> 4)
            tu = ((payload[6] & 0xF) << 8) | payload[7]
            self.q_des = u2d(qu, -c["pmax"], c["pmax"], 16)
            self.kp = u2d(kp_u, 0, 500, 12)
            self.kd = u2d(kd_u, 0, 5, 12)
            self.tau_ff = u2d(tu, -c["tmax"], c["tmax"], 12)
            self.pending_replies.append((esc + 0x10, self.state_payload()))
        elif can_id == 0x7FF and len(payload) >= 4:
            if payload[0] != esc:
                return
            cmd = payload[2]
            if cmd == 0xCC:                     # refresh
                self.pending_replies.append(
                    (esc + 0x10, self.state_payload()))
            elif cmd == 0x33:                   # 参数查询
                rid = payload[3]
                value = self._param_value(rid)
                self.pending_replies.append(
                    (esc + 0x10, self.param_payload(rid, value)))

    def _param_value(self, rid):
        c = self.cfg
        table = {7: float(esc_id_of(self)), 8: float(self.cfg["esc"]),
                 10: 1.0,                     # CTRL_MODE=MIT
                 13: 0.0, 14: 0.0,            # hw/sw ver
                 21: c["pmax"], 22: c["vmax"], 23: c["tmax"]}
        return table.get(rid, 0.0)


def esc_id_of(m):
    return m.cfg["esc"]


class Simulator:
    CAN_RAW_FD_FRAMES = 5

    def __init__(self, iface: str):
        self.sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW,
                                  socket.CAN_RAW)
        self.sock.setsockopt(socket.SOL_CAN_RAW,
                             self.CAN_RAW_FD_FRAMES, 1)
        self.sock.bind((iface,))
        self.sock.setblocking(False)
        self.motors = [SimMotor(cfg, INITIAL_DEG[i] * D2R)
                       for i, cfg in enumerate(MOTORS)]

    def send_fd(self, can_id: int, payload: bytes):
        frame = (struct.pack("<IBB2x", can_id, len(payload), 0)
                 + payload.ljust(64, b"\x00"))
        try:
            self.sock.send(frame)
        except OSError:
            pass                                # RX 侧拥塞不影响仿真

    def handle_incoming(self):
        while True:
            try:
                frame, _ = self.sock.recvfrom(72)
            except BlockingIOError:
                return
            if len(frame) == 72:
                can_id = struct.unpack("<I", frame[:4])[0] & 0x7FF
                dlen = frame[4]
                payload = frame[8:8 + dlen]
            else:
                can_id, dlen = struct.unpack("<IB", frame[:5])
                payload = frame[8:8 + dlen]
            for m in self.motors:
                m.on_frame(can_id, payload)
            # 参数应答走 0x7FF→recv_id 上面已在 on_frame 排队

    def dispatch_replies(self):
        for m in self.motors:
            while m.pending_replies:
                cid, payload = m.pending_replies.pop(0)
                self.send_fd(cid, payload)

    def run(self):
        print("达妙电机仿真器运行中: 8 台电机, 限位=官方v1表, "
              "初始位形=%s" % INITIAL_DEG)
        last_tick = time.monotonic()
        try:
            while True:
                self.handle_incoming()
                now = time.monotonic()
                while now - last_tick >= DT:
                    for m in self.motors:
                        m.tick()
                    last_tick += DT
                self.dispatch_replies()
                time.sleep(0.0005)
        except KeyboardInterrupt:
            print("仿真器退出")


def main():
    # 注意: ros2 launch 会向 sys.argv 注入 --ros-args，不能从 argv[1] 取接口名
    iface = os.environ.get("DM_SIM_IFACE", "vcan0")
    try:
        sim = Simulator(iface)
    except OSError as e:
        print("仿真器启动失败: %s" % e)
        print("当前 CAN 接口: ", end="")
        try:
            print(sorted(d for d in os.listdir("/sys/class/net")
                         if d.startswith(("can", "vcan"))))
        except OSError:
            print("未知")
        print("请先创建虚拟总线: sudo modprobe vcan && "
              "sudo ip link add dev %s type vcan && sudo ip link set %s up"
              % (iface, iface))
        sys.exit(1)
    sim.run()


if __name__ == "__main__":
    main()
