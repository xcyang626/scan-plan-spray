#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenArm 六电机 DM-Device ROS2 控制节点
======================================

背景
----
官方 openarm_hardware 的 ros2_control 插件走 SocketCAN(can0)，而本机硬件是达妙
DM-USB2FDCAN，走 dm-device(libdm_device.so)，系统里没有 can0，所以官方插件
无法驱动这台机械臂。

本节点改用与 test_motors.py 完全相同的 motorbridge(libdm) 栈，直接驱动 6 个
DM8009 电机，并通过标准 ROS2 接口对外交互：

  - 发布  /dm_arm/joint_states
          （sensor_msgs/JointState，电机原始弧度，未做 URDF 零位对齐）
  - 订阅  /dm_arm/joint_command
          （sensor_msgs/JointState，用 position[] 给每关节目标位置(弧度)；
            可选用 velocity[] 给每关节速度限幅，缺省用参数 vlim）
  - 服务  /dm_arm/enable    (std_srvs/Trigger)  上电/使能 + 切 pos_vel 模式
  - 服务  /dm_arm/disable   (std_srvs/Trigger)  失电/下使能

【重要】pos_vel 模式(速度受限位置控制)下发的是「绝对位置(弧度)+速度上限」，
目标位置是电机原始编码器空间，与 URDF 的 joint 坐标系存在零位偏移，请勿直接
用 URDF 限位去夹它，应先读 /dm_arm/joint_states 观察实际初始位置，从当前值
相对小幅移动。

【运行】打开达妙设备需要 root（和 scan/test 一样）：
    ./run_arm_control.sh
等价于：
    sudo -E env HOME=/home/yang /usr/bin/python3 arm_dm_control.py
"""
from __future__ import annotations

import sys

# rclpy 跑在 ROS2 的 python3.12 里，而 motorbridge 装在 conda 的 python3.14。
# motorbridge 是纯 Python + ctypes 加载 libdm_device.so，与解释器版本无关，
# 所以把 conda 的 site-packages 追加到 sys.path 末尾即可复用，无需重装。
try:
    import motorbridge  # noqa: F401
except ImportError:
    _CONDA_SP = "/home/yang/miniconda3/lib/python3.14/site-packages"
    if _CONDA_SP not in sys.path:
        sys.path.append(_CONDA_SP)
    import motorbridge  # noqa: F401

from motorbridge import Controller, Mode  # noqa: E402

import time  # noqa: E402
import argparse  # noqa: E402
import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402


class ArmDMControlNode(Node):
    """基于 motorbridge(libdm) 的 OpenArm 控制节点。"""

    def __init__(self, auto_enable: bool = False):
        super().__init__("arm_dm_control")
        # ---------- 参数 ----------
        self.declare_parameter("dm_device_type", "usb2canfd")
        self.declare_parameter("dm_channel", "0")
        self.declare_parameter("model", "8009")
        self.declare_parameter("motor_ids", "1-6")
        self.declare_parameter("vlim", 0.5)          # 默认速度限幅 (rad/s)
        self.declare_parameter("rate", 50.0)         # 控制/发布频率 (Hz)
        self.declare_parameter("auto_enable", False)  # 是否启动即使能

        dev_type = self.get_parameter("dm_device_type").value
        channel = self.get_parameter("dm_channel").value
        self._model = self.get_parameter("model").value
        self._vlim = self.get_parameter("vlim").value
        rate = self.get_parameter("rate").value
        self._auto_enable = auto_enable or self.get_parameter("auto_enable").value

        self._ids = sorted({i for t in
                            self.get_parameter("motor_ids").value.split(",") for i in
                            (list(range(int(t.split("-")[0]), int(t.split("-")[1]) + 1))
                             if "-" in t else [int(t)])})
        if not self._ids:
            self.get_logger().error("未解析到任何电机 ID，退出")
            raise SystemExit(1)

        # ---------- 打开 dm-device 总线 ----------
        self.get_logger().info(f"打开 DM-Device: type={dev_type} channel={channel}")
        self._ctrl = Controller.from_dm_device(dev_type, channel)

        # 添加电机：发送 ID=mid，反馈 ID=mid+0x10（与 scan/test 一致）
        self._motors = {}
        for mid in self._ids:
            m = self._ctrl.add_damiao_motor(mid, mid + 0x10, self._model)
            self._motors[mid] = m
        self.get_logger().info(
            f"已添加电机: {[f'0x{i:02X}' for i in self._ids]}")

        # 关节名 = openarm_joint1..N（顺序与 CAN ID 1..6 对应）
        self._joint_names = [f"openarm_joint{i}" for i in self._ids]

        # 状态 / 指令缓存
        self._enabled = False
        self._cmd_pos = {mid: None for mid in self._ids}  # mid -> 目标位置
        self._cmd_vel = {mid: None for mid in self._ids}  # mid -> 速度限幅(None用默认)

        # ---------- ROS2 接口 ----------
        self._pub = self.create_publisher(JointState, "dm_arm/joint_states", 10)
        self.create_subscription(JointState, "dm_arm/joint_command",
                                 self._on_joint_command, 10)
        self.create_service(Trigger, "dm_arm/enable", self._on_enable)
        self.create_service(Trigger, "dm_arm/disable", self._on_disable)

        self._timer = self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f"节点就绪: 6 电机, rate={rate}Hz, vlim={self._vlim}")

        if self._auto_enable:
            self._enable_all()

    # ================= 内部工具 =================
    def _read_state(self, mid):
        """请求一次反馈并轮询，返回最新 MotorState 或 None。"""
        m = self._motors[mid]
        last = None
        try:
            m.request_feedback()
            for _ in range(3):
                self._ctrl.poll_feedback_once()
                st = m.get_state()
                if st is not None:
                    last = st
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"读电机 0x{mid:X} 反馈失败: {e}")
        return last

    def _enable_all(self):
        for mid, m in self._motors.items():
            try:
                m.clear_error()
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"clear_error 0x{mid:X}: {e}")
            m.enable()
            m.ensure_mode(Mode.POS_VEL, 1000)  # 切 pos_vel 位置模式
            # 使能后先锁住当前位置，避免从 0 起跳
            st = self._read_state(mid)
            self._cmd_pos[mid] = st.pos if st is not None else self._cmd_pos[mid]
        self._enabled = True
        self.get_logger().info("已使能全部电机 (pos_vel 模式)")

    def _disable_all(self):
        for m in self._motors.values():
            try:
                m.disable()
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"disable {e}")
        self._enabled = False
        self.get_logger().info("已失电全部电机")

    # ================= 回调 =================
    def _on_joint_command(self, msg: JointState):
        """解析 /dm_arm/joint_command 的 position[] -> 各电机目标。"""
        if len(msg.position) != len(self._ids):
            self.get_logger().warn(
                f"joint_command 位置数 {len(msg.position)} != 电机数 {len(self._ids)}")
        pos = msg.position if len(msg.position) == len(self._ids) else \
            list(msg.position) + [None] * (len(self._ids) - len(msg.position))
        vel = msg.velocity if (msg.velocity and len(msg.velocity) == len(self._ids)) \
            else [None] * len(self._ids)
        for idx, mid in enumerate(self._ids):
            if pos[idx] is not None:
                self._cmd_pos[mid] = float(pos[idx])
            if vel[idx] is not None:
                self._cmd_vel[mid] = float(vel[idx])

    def _on_enable(self, req, resp: Trigger.Response):
        self._enable_all()
        resp.success = True
        resp.message = "已使能"
        return resp

    def _on_disable(self, req, resp: Trigger.Response):
        self._disable_all()
        resp.success = True
        resp.message = "已失电"
        return resp

    def _tick(self):
        """控制主循环：读状态 -> 发布 -> 下发 pos_vel 指令。"""
        # 读状态 + 发布 joint_states
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(self._joint_names)
        pos, vel, torq = [], [], []
        for mid in self._ids:
            st = self._read_state(mid)
            if st is not None:
                pos.append(st.pos)
                vel.append(st.vel)
                torq.append(st.torq)
            else:  # 没有新反馈就补一个占位，保持数组长度一致
                pos.append(pos[-1] if pos else 0.0)
                vel.append(0.0)
                torq.append(0.0)
        js.position = pos
        js.velocity = vel
        js.effort = torq
        self._pub.publish(js)

        # 下发指令（pos_vel = 目标位置 + 速度限幅）
        if self._enabled:
            for mid in self._ids:
                target = self._cmd_pos[mid]
                if target is None:
                    continue
                vlim = self._cmd_vel[mid] if self._cmd_vel[mid] is not None \
                    else self._vlim
                try:
                    self._motors[mid].send_pos_vel(target, vlim)
                except Exception as e:  # noqa: BLE001
                    self.get_logger().warn(f"send 0x{mid:X} 失败: {e}")

    def destroy_node(self):
        """退出前保证失电，避免电机保持使能。"""
        try:
            self._disable_all()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._ctrl.close_bus()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._ctrl.close()
        except Exception:  # noqa: BLE001
            pass
        super().destroy_node()


def _read_one(ctrl, m):
    """请求反馈并轮询，返回最新 MotorState 或 None。"""
    last = None
    try:
        m.request_feedback()
        for _ in range(3):
            ctrl.poll_feedback_once()
            st = m.get_state()
            if st is not None:
                last = st
    except Exception:  # noqa: BLE001
        pass
    return last


def _wait_reach(ctrl, m, target, tol, timeout_s, vlim):
    """持续下发 pos_vel 直至到位(timeout)。返回 (min_dist, final_pos)。"""
    steps = int(timeout_s / 0.02)
    min_dist, final_pos = tol + 1.0, None
    for _ in range(steps):
        if m is not None:
            pass
        try:
            m.send_pos_vel(target, vlim)
        except Exception:  # noqa: BLE001
            break
        st = _read_one(ctrl, m)
        if st is not None:
            final_pos = st.pos
            d = abs(st.pos - target)
            min_dist = min(min_dist, d)
            if d <= tol:
                break
        time.sleep(0.02)
    return min_dist, final_pos


def selftest(args):
    """单进程自检：逐个使能 6 个电机并相对小幅移动，验证能否通信/使能/转动。

    所有目标位置都以‘当前实际位置’为基准做相对偏移(±offset)，不从 URDF 限位
    推算绝对角度，避免原始编码器零位与 URDF 不对齐导致越界。全程不需要 ROS2。
    """
    ids = sorted({i for t in args.motor_ids.split(",") for i in
                  (list(range(int(t.split("-")[0]), int(t.split("-")[1]) + 1))
                   if "-" in t else [int(t)])})
    hold = args.dwell
    print("=" * 66)
    print(f" 自检电机: {[f'0x{i:02X}' for i in ids]}")
    print(f" 偏移=±{args.offset}rad  速度={args.vlim}rad/s  保持={hold}s  容差={args.tol}rad")
    print(f" 传输: dm-device type={args.dm_device_type} channel={args.dm_channel}")
    print("=" * 66)

    ctrl = Controller.from_dm_device(args.dm_device_type, args.dm_channel)
    results = []
    try:
        for mid in ids:
            m = ctrl.add_damiao_motor(mid, mid + 0x10, args.model)
            res = {"id": mid, "ok": False, "reason": "", "pos0": None, "enabled": False, "per": []}
            try:
                # 0) 读初始位置，确认通信
                pos0 = st0 = None
                for _ in range(10):
                    st0 = _read_one(ctrl, m)
                    if st0 is not None:
                        pos0 = st0.pos
                        break
                    time.sleep(0.02)
                if pos0 is None:
                    res["reason"] = "无反馈(未上电/断线/ID不对)"
                    results.append(res)
                    try: m.close()
                    except Exception: pass
                    continue
                res["pos0"] = pos0

                # 1) 清错 + 使能，等状态变 ENABLED(0x1)
                try: m.clear_error()
                except Exception as e: print(f"    [warn] clear_error: {e}")
                m.enable()
                for _ in range(50):
                    st = _read_one(ctrl, m)
                    if st is not None and st.status_code == 0x1:
                        res["enabled"] = True
                        break
                    time.sleep(0.05)
                if not res["enabled"]:
                    res["reason"] = "使能未生效"
                    try: m.disable()
                    except Exception: pass
                    results.append(res)
                    try: m.close()
                    except Exception: pass
                    continue

                # 2) 切 pos_vel 模式
                try:
                    m.ensure_mode(Mode.POS_VEL, 1000)
                except Exception as e:
                    res["reason"] = f"ensure_mode 失败: {e}"
                    try: m.disable()
                    except Exception: pass
                    results.append(res)
                    try: m.close()
                    except Exception: pass
                    continue

                # 3) 相对扫动并测量实际位移
                max_move = 0.0
                for off in [0.0, +args.offset, 0.0]:
                    target = res["pos0"] + off
                    _, final = _wait_reach(ctrl, m, target, args.tol, hold, args.vlim)
                    if final is not None:
                        max_move = max(max_move, abs(final - res["pos0"]))
                    done = ("到位" if (final is not None and abs(final - target) <= args.tol) else "未到")
                    res["per"].append(f"{off:+}rad:{done}(目标:{final if final is None else round(final,3)})")
                    time.sleep(0.1)

                # 4) 回起点 + 失电
                try: _wait_reach(ctrl, m, res["pos0"], args.tol, hold, args.vlim)
                except Exception: pass
                try: m.disable()
                except Exception: pass

                # 5) 判定：只要能相对当前移动超过振幅一半就算通过
                res["ok"] = max_move > args.offset * 0.5
                if not res["ok"]:
                    res["reason"] = f"几乎无位移(最大位移={max_move:.3f}rad)"
                results.append(res)
            finally:
                try: m.close()
                except Exception: pass
            print(f"  电机0x{mid:02X}: {res['per']}  {'PASS' if res['ok'] else 'FAIL'}  {res['reason']}")
            time.sleep(0.3)
    finally:
        try: ctrl.close_bus()
        except Exception: pass
        try: ctrl.close()
        except Exception: pass

    print("=" * 66)
    for r in results:
        print(f"  电机0x{r['id']:02X}: {'PASS' if r['ok'] else 'FAIL'}  使能={'y' if r['enabled'] else 'n'}  {r['reason']}")
    print(f"  通过 {sum(1 for r in results if r['ok'])}/{len(results)}")
    print("=" * 66)
    return sum(1 for r in results if r['ok']) == len(results)


def main():
    p = argparse.ArgumentParser(description="OpenArm DM-device 控制/自检")
    p.add_argument("--selftest", action="store_true",
                   help="单进程自检：逐个使能 6 电机并相对移动验证(无需 RclPy/IPC)")
    p.add_argument("--auto-enable", action="store_true",
                   help="broker 启动时即使能全部电机(避免跨进程调 enable 服务的阻塞)")
    p.add_argument("--dm-device-type", default="usb2canfd")
    p.add_argument("--dm-channel", default="0")
    p.add_argument("--model", default="8009")
    p.add_argument("--motor-ids", default="1-6")
    p.add_argument("--offset", type=float, default=0.08)
    p.add_argument("--vlim", type=float, default=0.5)
    p.add_argument("--dwell", type=float, default=1.0)
    p.add_argument("--tol", type=float, default=0.03)
    args = p.parse_args()

    if args.selftest:
        # 单进程自检：不调用 rclpy.init()，因此不会写 ~/.ros/log，
        # 也不需要 DDS，逐电机使能+相对移动，直接打印 PASS/FAIL。
        ok = selftest(args)
        raise SystemExit(0 if ok else 1)

    # broker 模式：真正的 ROS2 节点（发布 joint_states / 订阅 joint_command / 服务 enable/disable）
    rclpy.init()
    node = ArmDMControlNode(auto_enable=args.auto_enable)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()