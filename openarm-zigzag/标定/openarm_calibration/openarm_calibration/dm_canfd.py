#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
达妙电机 CANFD 驱动层（协议对齐官方 openarm_can 库）
====================================================

帧编解码 1:1 对齐:
  - openarm_can-main/src/openarm/damiao_motor/dm_motor_control.cpp
  - openarm_can-main/src/openarm/damiao_motor/dm_motor_device.cpp
  - openarm_can-main/include/openarm/damiao_motor/dm_motor_constants.hpp

要点:
  - 官方 OpenArm 以 enable_fd=True 运行 → 全部帧走 CANFD(BRS)。
  - 状态反馈帧:  MST_ID = ESC_ID + 0x10, 8 字节
  - refresh 帧:  0x7FF + [ESC_ID 小端2B, 0xCC, 0x00...] → 触发状态反馈
  - 参数查询:    0x7FF + [ESC_ID 小端2B, 0x33, RID, 0x00...]
  - 参数写入:    0x7FF + [ESC_ID 小端2B, 0x55, RID, u32小端4B]
  - 使能/失能/置零: 直接发 ESC_ID + FF*7 + FC/FD/FE

本模块只依赖 Python 标准库（socket/struct），可运行于任意解释器。
"""
from __future__ import annotations

import socket
import struct
import time

# ---------------------------------------------------------------------------
# 型号与兜底限参表（官方 MOTOR_LIMIT_PARAMS）: (pMax, vMax, tMax)
# ---------------------------------------------------------------------------
MOTOR_TYPE_NAMES = ["DM3507", "DM4310", "DM4310_48V", "DM4340", "DM4340_48V",
                    "DM6006", "DM8006", "DM8009", "DM10010L", "DM10010",
                    "DMH3510", "DMH6215", "DMG6220"]
FALLBACK_LIMITS = [(12.5, 50.0, 5.0),   # DM3507
                   (12.5, 30.0, 10.0),  # DM4310
                   (12.5, 50.0, 10.0),  # DM4310_48V
                   (12.5, 10.0, 28.0),  # DM4340
                   (12.5, 45.0, 20.0),  # DM6006
                   (12.5, 45.0, 40.0),  # DM8006
                   (12.5, 45.0, 54.0),  # DM8009
                   (12.5, 25.0, 200.0), (12.5, 20.0, 200.0),
                   (12.5, 280.0, 1.0), (12.5, 45.0, 10.0), (12.5, 45.0, 10.0)]

# 寄存器 ID（官方 RID 表，仅标定所需子集）
RID_MST_ID, RID_ESC_ID, RID_CTRL_MODE = 7, 8, 10
RID_HW_VER, RID_SW_VER, RID_PMAX, RID_VMAX, RID_TMAX = 13, 14, 21, 22, 23
# 官方 is_in_ranges: 这些 RID 返回 uint32，其余为 float
RID_UINT32 = set(range(7, 11)) | set(range(13, 17)) | {35, 36}

CTRL_MODE_MIT = 1

# 状态码 byte0（达妙官方定义）
STATE_CODES = {0x0: "DISABLED", 0x1: "ENABLED", 0x8: "过压", 0x9: "欠压",
               0xA: "过流", 0xB: "MOS过温", 0xC: "线圈过温",
               0xD: "通讯丢失", 0xE: "过载"}


# ---------------------------------------------------------------------------
# 数值编解码（官方 double_to_uint / uint_to_double）
# ---------------------------------------------------------------------------

def d2u(x: float, mn: float, mx: float, bits: int) -> int:
    x = max(mn, min(x, mx))
    return int(round((x - mn) / (mx - mn) * ((1 << bits) - 1)))


def u2d(x: int, mn: float, mx: float, bits: int) -> float:
    return (x / ((1 << bits) - 1)) * (mx - mn) + mn


# ---------------------------------------------------------------------------
# CANFD 总线
# ---------------------------------------------------------------------------

class CanFdBus:
    """SocketCAN CANFD 原始收发（CAN_RAW_FD_FRAMES 必须在 bind 前设置）。"""

    CAN_RAW_FD_FRAMES = 5   # linux/can/raw.h（注意: 5 才是 FD 选项）
    CANFD_BRS = 0x01

    def __init__(self, iface: str, use_fd: bool = True):
        self.use_fd = use_fd
        self.sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        if use_fd:
            self.sock.setsockopt(socket.SOL_CAN_RAW,
                                 self.CAN_RAW_FD_FRAMES, 1)
        self.sock.bind((iface,))
        self.sock.setblocking(False)

    def send(self, can_id: int, data: bytes) -> None:
        """发送一帧（FD 模式下 data 不满 64B 自动填充）。
        TX 队列拥塞(ENBUFS)时自动重试，最多约 50ms。"""
        if self.use_fd:
            flags = self.CANFD_BRS
            frame = (struct.pack("<IBB2x", can_id, len(data), flags)
                     + data.ljust(64, b"\x00"))
        else:
            frame = struct.pack("<IB3x8s", can_id, len(data),
                                data.ljust(8, b"\x00"))
        for _ in range(25):
            try:
                self.sock.send(frame)
                return
            except OSError as e:
                if e.errno == 105:      # ENOBUFS: 发送队列满，稍后重试
                    time.sleep(0.002)
                    continue
                raise
        raise OSError(105, "CAN 发送队列持续拥塞（ENBUFS），"
                           "请重启接口: sudo ip link set can0 down/up")

    def recv(self, timeout: float = 0.0):
        """非阻塞收一帧，返回 (can_id, data bytes) 或 None。自动区分 FD/经典帧。"""
        if timeout > 0:
            self.sock.settimeout(timeout)
            try:
                frame, _ = self.sock.recvfrom(72)
            except socket.timeout:
                return None
        else:
            try:
                frame, _ = self.sock.recvfrom(72)
            except BlockingIOError:
                return None
        if len(frame) == 72:                     # canfd_frame
            can_id, dlen = struct.unpack("<IB", frame[:5])
        else:                                    # 经典 can_frame
            can_id, dlen = struct.unpack("<IB", frame[:5])
        return can_id & 0x7FF, frame[8:8 + dlen]

    def drain(self, duration: float) -> int:
        """持续收帧 duration 秒，交给回调统计（由上层 poll 使用）。返回帧数。"""
        n, t0 = 0, time.monotonic()
        while time.monotonic() - t0 < duration:
            got = self.recv(0.0)
            if got:
                n += 1
        return n

    def close(self):
        self.sock.close()


# ---------------------------------------------------------------------------
# 帧构造（官方 CanPacketEncoder）
# ---------------------------------------------------------------------------

def build_enable(esc_id: int) -> tuple[int, bytes]:
    return esc_id, b"\xFF" * 7 + b"\xFC"


def build_disable(esc_id: int) -> tuple[int, bytes]:
    return esc_id, b"\xFF" * 7 + b"\xFD"


def build_set_zero(esc_id: int) -> tuple[int, bytes]:
    return esc_id, b"\xFF" * 7 + b"\xFE"


def build_refresh(esc_id: int) -> tuple[int, bytes]:
    return 0x7FF, bytes([esc_id & 0xFF, (esc_id >> 8) & 0xFF, 0xCC, 0, 0, 0, 0, 0])


def build_query_param(esc_id: int, rid: int) -> tuple[int, bytes]:
    return 0x7FF, bytes([esc_id & 0xFF, (esc_id >> 8) & 0xFF, 0x33, rid, 0, 0, 0, 0])


def build_write_param(esc_id: int, rid: int, value: int) -> tuple[int, bytes]:
    return 0x7FF, (bytes([esc_id & 0xFF, (esc_id >> 8) & 0xFF, 0x55, rid])
                   + struct.pack("<I", value))


def build_mit(esc_id: int, limits, kp: float, kd: float, q: float, dq: float,
              tau: float) -> tuple[int, bytes]:
    """MIT 控制帧（官方 pack_mit_control_data）。limits=(pMax,vMax,tMax)。"""
    pmax, vmax, tmax = limits
    q_u = d2u(q, -pmax, pmax, 16)
    dq_u = d2u(dq, -vmax, vmax, 12)
    kp_u = d2u(kp, 0, 500, 12)
    kd_u = d2u(kd, 0, 5, 12)
    tau_u = d2u(tau, -tmax, tmax, 12)
    return esc_id, bytes([
        (q_u >> 8) & 0xFF, q_u & 0xFF,
        dq_u >> 4,
        ((dq_u & 0xF) << 4) | ((kp_u >> 8) & 0xF),
        kp_u & 0xFF,
        kd_u >> 4,
        ((kd_u & 0xF) << 4) | ((tau_u >> 8) & 0xF),
        tau_u & 0xFF])


def parse_param(payload: bytes):
    """参数应答帧解析（官方 parse_motor_param_data）。返回 (rid, value) 或 None。"""
    if len(payload) < 8 or payload[2] not in (0x33, 0x55):
        return None
    rid = payload[3]
    raw = bytes(payload[4:8])
    if rid in RID_UINT32:
        return rid, float(struct.unpack("<I", raw)[0])
    return rid, struct.unpack("<f", raw)[0]


def parse_state(payload: bytes, limits):
    """状态反馈帧解析（官方 parse_motor_state_data）。limits=(pMax,vMax,tMax)。"""
    if len(payload) < 8:
        return None
    pmax, vmax, tmax = limits
    q_u = (payload[1] << 8) | payload[2]
    dq_u = (payload[3] << 4) | (payload[4] >> 4)
    tau_u = ((payload[4] & 0xF) << 8) | payload[5]
    return {
        "state_code": payload[0],
        "q": u2d(q_u, -pmax, pmax, 16),
        "dq": u2d(dq_u, -vmax, vmax, 12),
        "tau": u2d(tau_u, -tmax, tmax, 12),
        "t_mos": payload[6],
        "t_rotor": payload[7],
    }
