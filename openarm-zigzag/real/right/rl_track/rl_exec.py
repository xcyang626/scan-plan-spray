#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rl_exec.py —— 真机 RL 执行节点(右臂之字形走线)
=================================================
架构: /joint_states(100Hz) → Policy(rl_policy.py, 与离线验证同一份代码)
  → 50Hz 位置指令 q_cmd = clamp(q_ref(t) + 0.25·tanh(a), 软限位∓3°)
  → /right_mpc_position_commands(Float64MultiArray, 驱动注入通道)

观测装配/前向/钳位全部来自 rl_policy.Policy —— 与 verify_deploy.py 的
离线验证**是同一份代码**, 保证"验证过的即上机的"。软限位/home 从
real_safety.yaml 读取(单一权威来源), 不再有硬编码副本。

驱动侧既有防护全部保留: ±6.5 绝对钳位、0.2rad/条速率钳位、0.2s 断流回落
JTC、watchdog/estop。本节点另加 0.05rad/条 @50Hz(=2.5rad/s) 速率保护。

前置: control 层在跑且臂在 home(±2°)。
⚠ 安全铁律: 本脚本只由操作员本人执行; AI 不执行任何真机运动命令。

用法(系统 python3, source 工作区后):
    /usr/bin/python3 rl_exec.py --max-t 15 --log-csv   # 分段验证第 1 步
    /usr/bin/python3 rl_exec.py --log-csv              # 全程
"""

import argparse
import math
import os
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray, String

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from rl_policy import POLICY_HZ, Policy, StreamClock  # noqa: E402

# RL 独立话题(--scheme rl 生成配置后驱动订阅此通道; 与 MPC 方案完全分离)
CMD_TOPIC = "/right_rl_position_commands"
RATE_LIMIT = 0.05          # rad/条 @50Hz = 2.5 rad/s 附加速率保护
HOME_TOL_DEG = 2.0
RETURN_SPEED = 0.10        # 缓速回家峰值关节速度 [rad/s](与参考转移段同级)
HOLD_HZ = 50               # 停留模式的指令发布频率(保持通道新鲜)
FF_FADE_S = 1.0            # 重力前馈淡入/淡出时长 [s](防力矩阶跃)
SPEED_MIN, SPEED_MAX = 0.0, 1.5     # 实时速度倍率硬限(0=暂停保持)
SPEED_RATE = 0.5           # 倍率变化率限幅 [1/s](防参考速度阶跃)


class GravFF:
    """重力前馈通道(与训练 env / make_grav_ff.py v2 同一份表)。
    消息格式(2026-09-16 对齐官方 MIT 五元组): 21 项 = 7 位置 + 7 期望
    速度[rad/s] + 7 前馈[Nm, URDF 系]; 仅前馈时 14 项; 都关 7 项(兼容)。"""

    def __init__(self, path, scale, ref_len):
        self.scale = 0.0
        self.gff = None
        if scale == 0.0:
            return
        if not os.path.isfile(path):
            if scale is None:
                print("[grav_ff] 未找到 %s → 关闭(与训练侧无前馈一致)"
                      % path)
                return
            sys.exit("[grav_ff] 显式 --grav-ff 但缺文件: %s" % path)
        gff = np.load(path)["gff"]
        if len(gff) < ref_len:
            sys.exit("[grav_ff] 行数(%d) < 参考帧数(%d), 重跑 make_grav_ff.py"
                     % (len(gff), ref_len))
        self.gff = np.asarray(gff[:ref_len], dtype=np.float64)
        self.scale = 1.0 if scale is None else float(scale)
        print("[grav_ff] %s ×%.2f (峰值 %.2f Nm @J%d)"
              % (os.path.basename(path), self.scale,
                 float(np.abs(self.gff).max()),
                 int(np.abs(self.gff).max(axis=0).argmax()) + 1))

    @property
    def on(self):
        return self.gff is not None and self.scale > 0

    def row(self, idx, ramp=1.0):
        """idx 处的前馈行 ×缩放×渐变系数。"""
        if not self.on:
            return None
        return self.gff[idx] * (self.scale * ramp)


class RlExec(Node):

    def __init__(self, pol, cmd_topic=CMD_TOPIC, ff=None, vref=True,
                 speed_topic=None, speed_min=SPEED_MIN,
                 speed_max=SPEED_MAX, speed_rate=SPEED_RATE):
        super().__init__("rl_exec")
        self.pol = pol
        self.ff = ff or GravFF(None, 0.0, 0)      # 关闭态占位
        self.vref = vref                           # v_des = q̇_ref·qd_scale
        self.msg_ff = self.ff.on
        self.msg_vel = vref
        self.cmd_topic = cmd_topic
        self._ff_last = np.zeros(7)
        self.q = None
        self.vq = None
        self.status = None
        self.stop_reason = None
        self.q_cmd_prev = None
        # 实时速度倍率(操作员通道, 0923): 目标由话题写入, 实际值在主循环
        # 按变化率限幅逼近(防参考速度阶跃); 0=暂停(指令冻结, 通道保鲜)
        self.speed_min = float(speed_min)
        self.speed_max = float(speed_max)
        self.speed_rate = float(speed_rate)        # [倍率/s]
        self.speed_target = 1.0
        self.speed_cur = 1.0
        self.create_subscription(JointState, "/joint_states", self._on_js,
                                 qos_profile_sensor_data)
        self.create_subscription(String, "/safety/status",
                                 lambda m: setattr(self, "status", m.data),
                                 10)
        self.create_subscription(Bool, "/safety/estop", self._on_estop, 10)
        if speed_topic:
            self.create_subscription(
                Float64, speed_topic, self._on_speed, 10)
        self.pub_cmd = self.create_publisher(Float64MultiArray,
                                             self.cmd_topic, 10)
        self.pub_done = self.create_publisher(Bool, "/rl/finished", 10)

    def _on_speed(self, m):
        if m.data == m.data:                       # NaN 防御
            self.speed_target = float(m.data)

    def _on_js(self, m):
        if len(m.position) < 7:
            return
        self.q = np.array(m.position[:7])
        # 个别驱动不发布速度 → 退化为 0(观测中 qd 权重有限, 不影响安全)
        self.vq = (np.array(m.velocity[:7]) if len(m.velocity) >= 7
                   else np.zeros(7))

    def _on_estop(self, m):
        if m.data:
            self.stop_reason = "estop"

    def wait_q(self):
        t0 = time.monotonic()
        while self.q is None or self.vq is None:
            if time.monotonic() - t0 > 5.0:
                sys.exit("等不到 /joint_states —— control 层在跑吗?")
            time.sleep(0.05)

    # ------------------------------------------------------------------
    def _publish(self, q_cmd, vel=None, ff=None):
        """消息长度固定于节点初始化(21/14/7), 驱动按长度解析。"""
        msg = Float64MultiArray()
        data = [float(x) for x in q_cmd]
        if self.msg_vel or self.msg_ff:
            data += [float(x) for x in (vel if vel is not None
                                        else np.zeros(7))]
            if self.msg_ff:
                data += [float(x) for x in (ff if ff is not None
                                            else np.zeros(7))]
        msg.data = data
        self.pub_cmd.publish(msg)

    def _alive(self):
        """estop/watchdog → 返回 False(立即停止一切流)。"""
        if self.stop_reason:
            self.get_logger().error("中止: %s" % self.stop_reason)
            return False
        if self.status in ("TRIPPED", "RETURNING"):
            self.stop_reason = "watchdog %s" % self.status
            self.get_logger().error("watchdog 触发, 停流")
            return False
        return True

    def stream_return_home(self, reason=""):
        """缓速回家段: 从当前指令位置梯形速度规划插值到 home, 到位后停流。
        必须在停流前执行 —— 直接停流会回落 JTC(其保持目标=home),
        等于从当前位置一步跳回 home, 会全力猛拉(已造成过碰撞)。"""
        pol = self.pol
        self.wait_q()
        q0 = (self.q_cmd_prev.copy() if self.q_cmd_prev is not None
              else self.q.copy())
        q1 = np.clip(pol.home, pol.lo, pol.hi)         # 防御性: home 在盒内
        dq = float(np.abs(q1 - q0).max())
        if dq < 1e-3:
            self.get_logger().info("已在 home 附近, 无需回家段")
            return
        # 梯形速度规划(2026-09-18): 五次曲线起步加速度为零, 长行程前 10s
        # 移动 <0.5° 肉眼不可见, 诱导操作员误按急停(急停的自动回家是激进的,
        # 会以 12rad/s 级扑跳)。梯形 1.5s 内即有可观位移。
        v = RETURN_SPEED
        ta = 1.5                                       # 加/减速段 [s]
        d_acc = 0.5 * v * ta
        if dq < 2 * d_acc:                             # 短行程: 缩小峰值
            v = dq / (2 * ta)
            d_acc = 0.5 * v * ta
        d_cruise = max(dq - 2 * d_acc, 0.0)
        T = 2 * ta + d_cruise / v if v > 1e-9 else 3.0
        self.get_logger().info(
            "缓速回 home(梯形): 最大关节行程 %.1f°, 预计 %.1fs (巡航 %.2f rad/s, "
            "1.5s 后即有可见位移)%s"
            % (math.degrees(dq), T, v, (" (%s)" % reason)
               if reason else ""))
        n = int(T * POLICY_HZ)
        period = 1.0 / POLICY_HZ
        next_t = time.monotonic()
        ff0 = self._ff_last.copy()                    # 回家段前馈淡出
        # 回家段指令日志(独立文件, 2026-09-18): 主循环 CSV 不覆盖回家段,
        # 本次事故(背离 home 的协调跳变)无指令记录无法归因 —— 补上
        try:
            csv_rh = open(time.strftime("rl_exec_%Y%m%d_%H%M%S_return.csv"),
                          "w", encoding="utf-8")
            csv_rh.write("t,q1,q2,q3,q4,q5,q6,q7,ff1,ff2,ff3,ff4,ff5,ff6,ff7\n")
        except Exception:
            csv_rh = None

        def _s_of(tt):
            # 2026-09-22 修复长行程单位 bug: 巡航段边界原写作 ta + d_cruise
            # (d_cruise 是弧度, 被当秒), 长行程在 tt≈3.5s 误入减速分支,
            # td = T−tt 高达 ~20s → s 为大负值 → f 大负值 → 规划点外推到
            # home 反方向数倍行程, 速率限幅将其变成 +0.05rad/帧 连续猛拉
            # (0922 复验回家段"拉的很远"急停事故; 0918"背离 home 的协调
            # 跳变"同因)。正确口径: 巡航结束时刻 = ta + d_cruise/v。
            tt = min(max(tt, 0.0), T)
            t_cruise_end = ta + d_cruise / v if v > 1e-9 else ta
            if tt <= ta:
                return 0.5 * (v / ta) * tt * tt
            if tt <= t_cruise_end:
                return d_acc + v * (tt - ta)
            td = T - tt
            return dq - 0.5 * (v / ta) * td * td

        for k in range(n + int(0.5 * POLICY_HZ)):     # +0.5s 到位保持
            if not self._alive():
                break
            f = _s_of(k * period) / max(dq, 1e-9)
            qc = q0 + (q1 - q0) * f
            qc = np.clip(qc, self.q_cmd_prev - RATE_LIMIT,
                         self.q_cmd_prev + RATE_LIMIT)
            self.q_cmd_prev = qc
            ramp = max(0.0, 1.0 - k * period / FF_FADE_S)
            self._ff_last = ff0 * ramp
            self._publish(qc, np.zeros(7),
                          self._ff_last if self.ff.on else None)
            if csv_rh is not None:
                csv_rh.write("%.3f,%s,%s\n" % (
                    k * period,
                    ",".join("%.5f" % v for v in qc),
                    ",".join("%.4f" % v for v in self._ff_last)))
            next_t += period
            sl = next_t - time.monotonic()
            if sl > 0:
                time.sleep(sl)
            else:
                next_t = time.monotonic()
        if csv_rh is not None:
            csv_rh.close()
        if self.stop_reason or self.status not in (None, "OK"):
            self.get_logger().warn(
                "回家段中止(%s/status=%s): 臂停在当前位置, 驱动将以 "
                "0.25rad/s 滑向 JTC 目标 —— 之后请 reset 并重新 to-home 找平"
                % (self.stop_reason or "-", self.status or "-"))
        else:
            self.get_logger().info("回家段完成, 停流(回落 JTC 零阶跃)")

    def hold_until_interrupt(self):
        """位置保持: 冻结当前指令持续发布(通道保鲜)。
        第一段安全垫 —— 直接退出会回落 JTC/home 并猛拉。"""
        self.get_logger().warn(
            "位置保持中(50Hz 冻结当前指令)。再次 Ctrl+C = 强制退出"
            "(驱动将回落 JTC→home, 臂会猛烈拉回!); 缓速回家请另开终端: "
            "/usr/bin/python3 rl_exec.py --return-home")
        period = 1.0 / HOLD_HZ
        next_t = time.monotonic()
        ff0 = self._ff_last.copy()                    # 保持段前馈淡出
        k = 0
        try:
            while True:
                if not self._alive():
                    break
                ramp = max(0.0, 1.0 - k * period / FF_FADE_S)
                self._ff_last = ff0 * ramp
                self._publish(self.q_cmd_prev, np.zeros(7),
                              self._ff_last if self.ff.on else None)
                k += 1
                next_t += period
                sl = next_t - time.monotonic()
                if sl > 0:
                    time.sleep(sl)
                else:
                    next_t = time.monotonic()
        except KeyboardInterrupt:
            self.get_logger().error("强制退出: 停流, 驱动将回落 JTC(home)! "
                                    "臂会向 home 猛拉 —— 请扶住/远离。")
            self.stop_reason = "manual_hard"

    # ------------------------------------------------------------------
    def run(self, max_t=None, log_csv=False, clock=None):
        pol = self.pol
        n_max = pol.n if max_t is None else min(pol.n, int(max_t / pol.dt))
        self.wait_q()

        # 起点校验: 必须已在 home(用 yaml 的 home, 非参考里的副本)
        dev = float(np.abs(self.q - pol.home).max())
        if math.degrees(dev) > HOME_TOL_DEG:
            self.get_logger().error(
                "臂不在 home(偏差 %.2f° > %.1f°), 先 move_small.py --to-home"
                % (math.degrees(dev), HOME_TOL_DEG))
            return False
        self.get_logger().info(
            "策略就绪: 参考 %d 帧 / %.1fs, 本段执行 %.1fs, 限位来源 %s"
            % (pol.n, float(pol.meta["T_total"]), n_max * pol.dt,
               pol.lim_src))

        csv = None
        if log_csv:
            csv = open(os.path.join(
                _HERE, time.strftime("rl_exec_%Y%m%d_%H%M%S.csv")),
                "w", encoding="utf-8")
            csv.write("t,idx," + ",".join("q%d" % (j + 1) for j in range(7))
                      + "," + ",".join("cmd%d" % (j + 1) for j in range(7))
                      + "," + ",".join("qref%d" % (j + 1)
                                       for j in range(7))
                      + ("," + ",".join("ff%d" % (j + 1) for j in range(7))
                         if self.ff.on else "") + ",spd\n")

        pol.reset_stream()
        self.get_logger().info("RL 指令流开始(50Hz, 残差低通 %.1fHz + "
                               "启动渐入 %.1fs)... Ctrl+C=位置保持"
                               % (pol.lp_freq, pol.fade_s))
        self.get_logger().info(
            "指令话题: %s [%d 项: 位置%s%s]" % (
                self.cmd_topic, 7 + (7 if self.msg_vel else 0)
                + (7 if self.msg_ff else 0),
                "+速度ref" if self.msg_vel else "",
                "+重力前馈" if self.msg_ff else ""))
        self.get_logger().info(
            "实时调速: /rl/speed_scale 范围 [%.1f, %.1f] (0=暂停保持), "
            "变化率限幅 %.1f/s —— 只改参考时间轴, 不动任何安全链"
            % (self.speed_min, self.speed_max, self.speed_rate))
        prev_a = np.zeros(7)
        self.q_cmd_prev = self.q.copy()
        period = 1.0 / POLICY_HZ
        next_t = time.monotonic()
        k = 0
        max_dev = 0.0
        partial = n_max < pol.n
        try:
            while True:
                if not self._alive():
                    break
                # 实时速度: 目标夹限 → 变化率限幅 → 时钟倍率(0=暂停保持)
                spd_t = min(max(self.speed_target,
                                self.speed_min), self.speed_max)
                ds = max(-self.speed_rate * pol.dt,
                         min(self.speed_rate * pol.dt,
                             spd_t - self.speed_cur))
                self.speed_cur += ds
                clock.set_speed(self.speed_cur)
                idx = min(clock.advance(), n_max - 1)
                qd_scale = clock.rate(idx)
                # ---- 与离线验证同一份代码(rl_policy.Policy) ----
                a = pol.action(self.q, self.vq, prev_a, idx,
                               qd_scale=qd_scale)
                q_cmd = pol.command(self.q, self.vq, prev_a, idx,
                                    qd_scale=qd_scale)
                prev_a = a
                # 附加速率保护(驱动侧另有 0.2rad/条)
                q_cmd = np.clip(q_cmd, self.q_cmd_prev - RATE_LIMIT,
                                self.q_cmd_prev + RATE_LIMIT)
                self.q_cmd_prev = q_cmd
                # 重力前馈: 与 q_cmd 同 idx, 启动淡入防力矩阶跃
                ff_row = self.ff.row(idx, min(1.0, k * pol.dt / FF_FADE_S))
                if ff_row is not None:
                    self._ff_last = ff_row
                # v_des = 参考速度×时间轴倍率(官方 MIT 通道参数4)
                vel_row = (pol.qd_ref[idx] * qd_scale
                           if self.vref else None)
                self._publish(q_cmd, vel_row, ff_row)
                max_dev = max(max_dev, float(np.abs(
                    self.q - pol.q_ref[idx]).max()))
                if csv is not None and k % 5 == 0:      # 10Hz 记录
                    csv.write("%f,%d,%s,%s,%s%s,%.3f\n" % (
                        k * pol.dt, idx,
                        ",".join("%.5f" % v for v in self.q),
                        ",".join("%.5f" % v for v in q_cmd),
                        ",".join("%.5f" % v for v in pol.q_ref[idx]),
                        ("," + ",".join("%.3f" % v for v in self._ff_last)
                         if self.ff.on else ""),
                        self.speed_cur))
                k += 1
                # 终止: 全程以参考走完为准(变速下 k 不再等价于帧号);
                # 分段(--max-t)按墙钟时间, 参考提前走完也收
                if clock.done or (partial and k >= n_max):
                    if partial:
                        # 分段结束: 臂离 home 远, 必须缓速回家再停流
                        self.stream_return_home(reason="分段执行结束")
                    else:
                        self.get_logger().warn(
                            "全程未走完(原因=%s): 停流前前馈淡出, 臂停在"
                            "当前位置(驱动将 0.25rad/s 滑向 JTC 目标)"
                            % (self.stop_reason or "-"))
                        ff0 = self._ff_last.copy()
                        for kk in range(int(POLICY_HZ * FF_FADE_S)):
                            if not self._alive():
                                break
                            self._ff_last = ff0 * max(
                                0.0, 1.0 - kk / (POLICY_HZ * FF_FADE_S))
                            self._publish(
                                self.q_cmd_prev, np.zeros(7),
                                self._ff_last if self.ff.on else None)
                            time.sleep(1.0 / POLICY_HZ)
                    break
                next_t += period
                sl = next_t - time.monotonic()
                if sl > 0:
                    time.sleep(sl)
                else:
                    next_t = time.monotonic()
        except KeyboardInterrupt:
            if self.q_cmd_prev is not None:
                self.hold_until_interrupt()   # 保持位置, 防回落猛拉
            else:
                self.stop_reason = "manual"
        finally:
            if csv is not None:
                csv.close()
        self.get_logger().info("结束: 最大跟踪偏差 %.2f° (%s)"
                               % (math.degrees(max_dev),
                                  self.stop_reason or "正常完成"))
        if self.stop_reason is None:
            self.pub_done.publish(Bool(data=True))
        return self.stop_reason is None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", default=os.path.join(_HERE, "actor_right.npz"))
    ap.add_argument("--ref", default=os.path.join(_HERE, "ref_right.npz"))
    ap.add_argument("--safety", default=None,
                    help="real_safety.yaml 路径(默认 real/right/config/)")
    ap.add_argument("--max-t", type=float, default=None,
                    help="只执行前 N 秒(分段验证, 结束自动缓速回 home)")
    ap.add_argument("--return-home", action="store_true",
                    help="挽救模式: 从当前位置缓速回 home 后停流"
                         "(用于保持/中止后的回收)")
    ap.add_argument("--log-csv", action="store_true")
    ap.add_argument("--cmd-topic", default=CMD_TOPIC,
                    help="指令话题(默认 /right_rl_position_commands)")
    ap.add_argument("--lp-freq", type=float, default=5.0,
                    help="策略残差低通截止 [Hz], 0=关(默认 5)")
    ap.add_argument("--fade-s", type=float, default=1.5,
                    help="启动残差渐入时长 [s], 0=关(默认 1.5)")
    ap.add_argument("--transfer-boost", type=float, default=1.0,
                    help="转移段时间轴加速倍率(默认 1=关; 1.5=转移速度 "
                         "×1.5 冲出摩擦带, 抖动治理, 走线段不受影响)")
    ap.add_argument("--grav-ff", nargs="?", const=1.0, default=None,
                    type=float,
                    help="重力前馈比例(不带值=1.0; 默认自动: grav_ff_"
                         "right.npz 存在则开, 与训练侧一致; 0=强制关)")
    ap.add_argument("--grav-ff-file", default=os.path.join(
        _HERE, "grav_ff_right.npz"),
        help="前馈表路径(make_grav_ff.py v2 URDF 权威产物)")
    ap.add_argument("--vref", nargs="?", const=1.0, default=1.0,
                    type=float,
                    help="v_des=参考速度比例(默认 1.0=开, 官方 MIT 速度通道; "
                         "0=退回 v_des=0 旧行为)")
    ap.add_argument("--speed-topic", default="/rl/speed_scale",
                    help="实时速度倍率话题(std_msgs/Float64, 默认开; "
                         "传空串关闭)。1.0=原速, 0=暂停保持, 上限见 --speed-max")
    ap.add_argument("--speed-min", type=float, default=SPEED_MIN)
    ap.add_argument("--speed-max", type=float, default=SPEED_MAX)
    ap.add_argument("--speed-rate", type=float, default=SPEED_RATE,
                    help="倍率变化率限幅 [1/s](默认 0.5, 防参考速度阶跃)")
    args = ap.parse_args()

    pol = Policy(actor_path=os.path.abspath(args.actor),
                 ref_path=os.path.abspath(args.ref),
                 yaml_path=args.safety,
                 lp_freq=args.lp_freq, fade_s=args.fade_s)
    ff = GravFF(args.grav_ff_file, args.grav_ff, pol.n)
    clock = StreamClock(pol, boost=args.transfer_boost)
    s = pol.summary()
    print("策略: %s | 层 %s | obs %d" % (
        os.path.basename(pol.actor_path), s["layer_dims"], s["obs_dim"]))
    print("参考: %s | %.1fs | 限位来源 %s" % (
        os.path.basename(pol.ref_path), s["T_total"], s["lim_source"]))
    print("指令盒: lo %s" % [round(v, 3) for v in s["cmd_lo"]])
    print("        hi %s" % [round(v, 3) for v in s["cmd_hi"]])

    # 禁用 rclpy 内置 SIGINT 处理: 默认 SignalHandlerOptions.ALL 下第一次
    # Ctrl+C 就会关闭 ROS 上下文(spin 线程 ExternalShutdownException),
    # "位置保持"分支里的 publish 全部失效 → 指令流断 → 驱动 0.2s 回落
    # JTC → 猛拉。2026-09-17 真机事故复盘(全程 60s 处中止即此因)。
    rclpy.init(signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)
    node = RlExec(pol, cmd_topic=args.cmd_topic, ff=ff,
                  vref=bool(args.vref), speed_topic=args.speed_topic,
                  speed_min=args.speed_min, speed_max=args.speed_max,
                  speed_rate=args.speed_rate)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()
    try:
        time.sleep(0.5)
        # 起动预检: 仅在安全状态 OK 时开流(RETURNING/TRIPPED 下开流会与
        # 看门狗回位打架, 2026-09-18 事故: RETURNING 中启动 → 全关节猛拽)
        t0 = time.monotonic()
        while node.status is None and time.monotonic() - t0 < 5.0:
            time.sleep(0.1)
        if node.status != "OK":
            sys.exit("安全状态=%s(需稳定 OK)——拒绝开流。TRIPPED: 排查后 "
                     "reset; RETURNING: 等回位完成; 无数据: 终端B在跑吗?"
                     % (node.status or "无数据"))
        if args.return_home:
            node.wait_q()
            node.q_cmd_prev = node.q.copy()
            node.stream_return_home(reason="--return-home 挽救模式")
        else:
            node.run(max_t=args.max_t, log_csv=args.log_csv, clock=clock)
    finally:
        ex.remove_node(node)
        ex.shutdown()
        spin.join(timeout=2.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
