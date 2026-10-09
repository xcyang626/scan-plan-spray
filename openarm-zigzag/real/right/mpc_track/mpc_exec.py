#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mpc_track 执行节点: MPC 精确跟踪 MoveIt2/dense 参考轨迹
================================================================
架构(路线 2, MoveIt2 + MPC 结合):
  MoveIt2/dense 参考路径 → 参考生成器(时间参数化)
    → 闭式预览 MPC (N=25, dt=10ms, 逐关节双积分模型)
    → /right_mpc_position_commands (100Hz 指令流)
    → 驱动注入通道 → MIT 电机
  JTC 被旁路(其梯形重定时是抖动源); watchdog 语义完整:
    /safety/status != OK 时立即停流 → 驱动回落 JTC 保持。

前置: control 层(A)在跑且臂在 home(±2°)。
用法:
  /usr/bin/python3 mpc_exec.py --ref ../../仿真/moveit_sim/dense_zigzag.json \
      --speed-scale 0.4
"""

import argparse
import json
import math
import os
import sys
import threading
import time

import numpy as np
import rclpy
import yaml
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray, String

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from mpc_core import PreviewMPC          # noqa: E402

NAMES = ["openarm_right_joint%d" % i for i in range(1, 8)]
ISA = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "仿真", "isaaclab_sim"))
PKG = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "仿真", "openarm_sim"))
CMD_TOPIC = "/right_mpc_position_commands"


class MpcExec(Node):

    def __init__(self, cfg):
        super().__init__("mpc_exec")
        self.cfg = cfg
        self.q = None
        self.status = None
        self.stop_reason = None
        self.create_subscription(JointState, "/joint_states",
                                 self._on_js, qos_profile_sensor_data)
        self.create_subscription(String, "/safety/status",
                                 lambda m: setattr(self, "status", m.data),
                                 10)
        self.create_subscription(Bool, "/safety/estop",
                                 lambda m: self._on_estop(m), 10)
        self.pub_cmd = self.create_publisher(Float64MultiArray, CMD_TOPIC, 10)
        self.pub_done = self.create_publisher(Bool, "/mpc/finished", 10)

    def _on_js(self, m):
        self.q = list(m.position[:7])

    def _on_estop(self, m):
        if m.data:
            self.stop_reason = "estop"

    # ------------------------------------------------------------------

    def wait_q(self):
        t0 = time.monotonic()
        while self.q is None:
            if time.monotonic() - t0 > 5.0:
                sys.exit("等不到 /joint_states —— control 层在跑吗?")
            time.sleep(0.05)

    def build_reference(self, pts, speed_scale):
        """参考 = [home] + dense + [home] 的时间参数化(分段速率)。
        返回 (ref采样数组, dt) —— dt = cfg.dt, 采样按时间插值。"""
        home = np.array(self.cfg["home_rad"])
        chain_pts = [home] + [np.array(p) for p in pts] + [home]
        tcp = self.cfg["tcp_speed"] * speed_scale
        v_app = self.cfg["approach_joint_speed"]
        v_ret = self.cfg["return_joint_speed"]
        # 每段时间: 双约束取大 —— 接近/回家按关节速率(慢), 走线按
        # TCP 速率与关节速率(构型翻转段单步 J5 139°, 必须按关节速率
        # 摊慢到数秒, 否则 MPC 追不上参考; 自旋翻转不影响指尖位置)。
        lens, speeds, caps = [], [], []
        n_dense = len(chain_pts) - 2
        for k in range(len(chain_pts) - 1):
            a, b = chain_pts[k], chain_pts[k + 1]
            if 0 < k <= n_dense:      # dense 内部段
                lens.append(float(self.seg_tcp[k - 1]))
                speeds.append(tcp)
                caps.append(self.cfg["weave_joint_speed"])
            else:                     # 接近段(k=0) / 回家段(k=末)
                lens.append(float(np.abs(b - a).max()))
                speeds.append(v_app if k == 0 else v_ret)
                caps.append(0.0)
        T = []
        for k in range(len(lens)):
            t_tcp = lens[k] / speeds[k] if speeds[k] > 0 else 0.0
            t_joint = 0.0
            if caps[k] > 0.0:
                jn = float(np.abs(chain_pts[k + 1] - chain_pts[k]).max())
                t_joint = jn / caps[k]
            T.append(max(t_tcp, t_joint))
        total = sum(T)
        # 按执行周期采样参考
        dt = self.cfg["dt"]
        m = int(total / dt) + 1
        R = np.zeros((m + self.cfg["N"] + 2, 7))   # 含预览余量
        for j in range(len(R)):
            tt = min(j * dt, total)
            acc = 0.0
            for k in range(len(T)):
                if tt <= acc + T[k] or k == len(T) - 1:
                    f = 0.0 if T[k] <= 0 else min(
                        max((tt - acc) / T[k], 0.0), 1.0)
                    R[j] = chain_pts[k] + f * (chain_pts[k + 1] - chain_pts[k])
                    break
                acc += T[k]
        self.get_logger().info(
            "参考: 接近 %.1fs + 走线 %.1fs + 回家 %.1fs = %.0fs"
            % (T[0], sum(T[1:1 + n_dense]), T[-1], total))
        return R, dt, total

    def run(self, ref_path, speed_scale):
        with open(ref_path, encoding="utf-8") as f:
            data = json.load(f)
        pts = data["points"]
        if data.get("pretimed"):
            self.get_logger().warn(
                "smooth_track 预定时轨迹与 MPC 互斥(MPC 自带时间轴), "
                "忽略 pretimed 字段")
        # 走线段 TCP 长度: 用 FK 重算(MPC 参考生成需要)
        sys.path.insert(0, ISA)
        sys.path.insert(0, PKG)
        from common import UrdfChain
        # 右臂链必须显式指定 tip(common.load_chain 写死左臂 tip link)
        chain = UrdfChain(os.path.join(ISA, "openarm_right.urdf"),
                          tip_link="openarm_right_hand_tcp")
        self.seg_tcp = [0.0]
        for i in range(1, len(pts)):
            _, pa = chain.fk(np.array(pts[i - 1]), 0.12)
            _, pb = chain.fk(np.array(pts[i]), 0.12)
            self.seg_tcp.append(float(np.linalg.norm(pb - pa)))

        self.wait_q()
        # 起点校验(home)
        dev = max(abs(self.q[k] - self.cfg["home_rad"][k]) for k in range(7))
        if math.degrees(dev) > self.cfg["hold_tolerance_deg"]:
            self.get_logger().error(
                "臂不在 home(偏差 %.2f°), 先执行 move_small.py --to-home"
                % math.degrees(dev))
            return False

        R, dt, total = self.build_reference(pts, speed_scale)
        N = self.cfg["N"]
        mpc = PreviewMPC(7, N=N, dt=dt, w_p=self.cfg["w_p"],
                         w_v=self.cfg["w_v"], w_a=self.cfg["w_a"],
                         a_max=self.cfg["a_max"])
        lo_c = np.array([self.cfg["soft_lo"][n] for n in NAMES]) \
            + math.radians(self.cfg["clearance_deg"])
        hi_c = np.array([self.cfg["soft_hi"][n] for n in NAMES]) \
            - math.radians(self.cfg["clearance_deg"])

        self.get_logger().info("MPC 流式指令开始(dt=%.0fms, N=%d, "
                               "TCP=%.1fcm/s)..." % (dt * 1000, N,
                                                     self.cfg["tcp_speed"]
                                                     * speed_scale * 100))
        q = list(self.q)
        v = [0.0] * 7
        k = 0
        t0 = time.monotonic()
        next_pub = t0
        max_dev = 0.0
        while True:
            now = time.monotonic()
            if self.stop_reason:
                self.get_logger().error("中止: %s" % self.stop_reason)
                break
            if self.status in ("TRIPPED", "RETURNING"):
                self.stop_reason = "watchdog %s" % self.status
                self.get_logger().error("watchdog 触发(%s), 停流" % self.status)
                break
            if self.q is None or now - t0 > 0:      # 占位: 状态断流保护
                pass
            k += 1
            if k >= len(R) - N:
                self.get_logger().info("参考走完, 停流保持")
                break
            ref_win = R[k:k + N + 1]
            qn, vn = [], []
            for j in range(7):
                a = mpc.step(q[j], v[j], ref_win[:, j], j)
                a = max(-self.cfg["a_max"], min(self.cfg["a_max"], a))
                qn.append(q[j] + dt * v[j] + 0.5 * dt * dt * a)
                vn.append(v[j] + dt * a)
            # 硬限位保护(投影)
            for j in range(7):
                qn[j] = min(max(qn[j], lo_c[j]), hi_c[j])
            dev = max(abs(qn[j] - R[min(k, len(R)-1)][j]) for j in range(7))
            max_dev = max(max_dev, dev)
            msg = Float64MultiArray()
            msg.data = [float(x) for x in qn]
            self.pub_cmd.publish(msg)
            q, v = qn, vn
            # 100Hz 节拍
            next_pub += dt
            sl = next_pub - time.monotonic()
            if sl > 0:
                time.sleep(sl)
            else:
                next_pub = time.monotonic()
        self.get_logger().info(
            "MPC 结束: 最大参考偏差 %.2f° (%s)"
            % (math.degrees(max_dev), self.stop_reason or "正常完成"))
        if self.stop_reason is None:
            self.pub_done.publish(Bool(data=True))
        return self.stop_reason is None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=os.path.join(
        _HERE, "..", "..", "..", "仿真", "moveit_sim", "dense_zigzag_right.json"))
    ap.add_argument("--speed-scale", type=float, default=0.4)
    ap.add_argument("--config", default=os.path.join(_HERE, "mpc_config.json"))
    args = ap.parse_args()
    cfg = json.load(open(args.config, encoding="utf-8"))
    # soft limits 从 real_safety.yaml 读取(单一权威来源)
    syp = os.path.abspath(os.path.join(_HERE, "..", "config",
                                       "real_safety.yaml"))
    with open(syp, encoding="utf-8") as f:
        s = yaml.safe_load(f)
    cfg["home_rad"] = list(s["home_rad"])
    cfg["soft_lo"] = {n: s["soft_limits_rad"][n][0]
                      if isinstance(s["soft_limits_rad"], dict)
                      else s["soft_limits_rad"][i][0]
                      for i, n in enumerate(NAMES)}
    cfg["soft_hi"] = {n: s["soft_limits_rad"][n][1]
                      if isinstance(s["soft_limits_rad"], dict)
                      else s["soft_limits_rad"][i][1]
                      for i, n in enumerate(NAMES)}

    rclpy.init()
    node = MpcExec(cfg)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()
    try:
        time.sleep(0.5)                      # 等 status 首帧
        node.wait_q()
        node.run(os.path.abspath(args.ref), args.speed_scale)
    except KeyboardInterrupt:
        pass
    finally:
        ex.remove_node(node)
        ex.shutdown()
        spin.join(timeout=2.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
