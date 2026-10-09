#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenArm 零点标定 ROS2 节点
==========================

官方零位版时序(2026-09-04, v2 —— 零位写入电机 flash):
    0. go_zero 服务: 使能后以限速缓慢运动到 zero 文件记录的零位基准位形,
       供人工目视验证"零位"(不碰撞、不写零点)
    1. start 标定: 使能 → 缓慢回零位基准(目标 = 现有 zero 文件
       zero_position_rad; 无文件时以当前位形为零位基准) → 保持
    2. J1 防护摆动 → 逐关节双向 bump 碰撞机械限位
       (夹爪→J7→J6→J5→J4→J3→J2→J1), 每次碰撞后限速回零位基准
    3. 全部关节回零位基准 → set_zero(0xFE, 官方行为): 当前位形写为
       电机零点 → 此后电机系 = 零位系, zero_position_rad ≈ 全零
    4. 限位平移到新系写入 zero 文件 → 保持零位(hold_after_done=true
       时持续刚度保持, 供人工验证; Ctrl-C 失能)
    5. set_zero 位于最后: 之前任何中止/异常都不改动电机 flash 零点,
       旧 zero 文件依然有效

ROS 接口:
  服务  /calibration/start (std_srvs/srv/Trigger)   启动标定（需 allow_motion=true）
  服务  /calibration/abort  (std_srvs/srv/Trigger)   中止标定并失能
  话题  /calibration/joint_states (sensor_msgs/JointState) 50Hz 关节状态
  话题  /calibration/status  (std_msgs/String)     JSON 进度/步骤/位置/错误
"""

import json
import math
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from . import dm_canfd as dm

D2R = math.pi / 180.0
R2D = 180.0 / math.pi

# 官方 JOINT_SIGN（J1/J2 因安装方向取 -1）
JOINT_SIGN = [-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# 官方 MECH_LIM_V1: 每关节 [limit_lower_deg, limit_upper_deg]（度）
MECH_LIM_V1_DEG = {
    "J1": [-80.0, 80.0], "J2": [-115.0, 0.0], "J3": [-90.0, 90.0],
    "J4": [0.0, 135.0], "J5": [-90.0, 0.0], "J6": [-45.0, 0.0],
    "J7": [-90.0, 90.0], "grip": [0.0, 40.0],
}


class DMMotor:
    """单台达妙电机的运行时状态。"""

    def __init__(self, esc_id: int, type_idx: int, limits):
        self.esc_id = esc_id
        self.recv_id = esc_id + 0x10           # 反馈 ID = 发送 ID + 0x10
        self.type_idx = type_idx
        self.limits = limits                   # (pMax, vMax, tMax)
        self.q = 0.0
        self.dq = 0.0
        self.tau = 0.0
        self.state_code = 0x0
        self.t_mos = 0
        self.t_rotor = 0
        self.last_update = 0.0

    def apply_state(self, st: dict):
        self.q = st["q"]
        self.dq = st["dq"]
        self.tau = st["tau"]
        self.state_code = st["state_code"]
        self.t_mos = st["t_mos"]
        self.t_rotor = st["t_rotor"]
        self.last_update = time.monotonic()


class ArmDriver:
    """8 电机集合驱动：封装总线收发与状态解析。
    内部 RLock 保证监控定时器与标定线程并发访问 socket 安全。"""

    def __init__(self, bus: dm.CanFdBus, motors: list[DMMotor], log):
        self.bus = bus
        self.motors = motors
        self.log = log
        self.lock = threading.RLock()

    # ---------- 帧处理 ----------

    def _apply_frame(self, got):
        cid, payload = got
        for m in self.motors:
            if cid == m.recv_id:
                if len(payload) >= 8 and payload[2] in (0x33, 0x55):
                    continue        # 参数应答帧，不是状态帧，跳过
                st = dm.parse_state(payload, m.limits)
                if st:
                    m.apply_state(st)

    def _drain_one(self):
        got = self.bus.recv(0.001)
        if got is not None:
            self._apply_frame(got)

    # ---------- 基础收发 ----------

    def send_recv(self, can_id: int, data: bytes, wait_s: float = 0.005):
        """发送一帧并收取 wait_s 内所有反馈帧，更新电机状态。"""
        with self.lock:
            self.bus.send(can_id, data)
            t0 = time.monotonic()
            while time.monotonic() - t0 < wait_s:
                self._drain_one()

    def poll(self, duration: float):
        """纯收帧 duration 秒。"""
        with self.lock:
            t0 = time.monotonic()
            while time.monotonic() - t0 < duration:
                self._drain_one()

    # ---------- 集合操作 ----------

    def enable_all(self):
        with self.lock:
            for m in self.motors:
                cid, data = dm.build_enable(m.esc_id)
                self.bus.send(cid, data)
            self.poll(0.1)

    def disable_all(self):
        with self.lock:
            for m in self.motors:
                cid, data = dm.build_disable(m.esc_id)
                self.bus.send(cid, data)
            self.poll(0.1)

    def set_zero_all(self):
        with self.lock:
            for m in self.motors:
                cid, data = dm.build_set_zero(m.esc_id)
                self.bus.send(cid, data)
                time.sleep(0.001)
            self.poll(0.1)

    def refresh_all(self):
        with self.lock:
            for m in self.motors:
                cid, data = dm.build_refresh(m.esc_id)
                self.bus.send(cid, data)
                self._drain_one()

    def mit(self, m: DMMotor, kp, kd, q, dq, tau):
        with self.lock:
            cid, data = dm.build_mit(m.esc_id, m.limits, kp, kd, q, dq, tau)
            self.bus.send(cid, data)
            self._drain_one()

    def hold(self, kp_kd_pairs):
        """按官方 _hold_position：以各自刚度锁定当前位置。"""
        for m, (kp, kd) in zip(self.motors, kp_kd_pairs):
            self.mit(m, kp, kd, m.q, 0.0, 0.0)

    def read_limits(self, m: DMMotor):
        """读 PMAX/VMAX/TMAX 寄存器（官方 get_pmax 等）。
        注意: 参数应答帧与状态帧一样都从电机反馈 ID(ESC_ID+0x10)返回。"""
        out = {}
        with self.lock:
            for rid, name in ((dm.RID_PMAX, "P"), (dm.RID_VMAX, "V"),
                              (dm.RID_TMAX, "T")):
                cid, data = dm.build_query_param(m.esc_id, rid)
                self.bus.send(cid, data)
                deadline = time.monotonic() + 0.1
                while time.monotonic() < deadline:
                    got = self.bus.recv(0.005)
                    if got and got[0] == m.recv_id:
                        parsed = dm.parse_param(got[1])
                        if parsed and parsed[0] == rid:
                            out[name] = parsed[1]
                            break
        if len(out) == 3:
            return (out["P"], out["V"], out["T"])
        return None

    def read_ctrl_mode(self, m: DMMotor):
        with self.lock:
            cid, data = dm.build_query_param(m.esc_id, dm.RID_CTRL_MODE)
            self.bus.send(cid, data)
            deadline = time.monotonic() + 0.1
            while time.monotonic() < deadline:
                got = self.bus.recv(0.005)
                if got and got[0] == m.recv_id:
                    parsed = dm.parse_param(got[1])
                    if parsed and parsed[0] == dm.RID_CTRL_MODE:
                        return int(parsed[1])
        return None


class CalibrationNode(Node):
    def __init__(self):
        super().__init__("calibration_node")
        # ---------- 参数 ----------
        self.declare_parameters("", [
            ("can_iface", "can0"), ("use_fd", True),
            ("arm_side", "right"), ("robot_version", "v1"),
            ("motor_ids", [1, 2, 3, 4, 5, 6, 7, 8]),
            ("motor_types", [7, 7, 3, 3, 1, 1, 1, 1]),
            ("joint_names", ["openarm_joint%d" % i for i in range(1, 8)]
             + ["openarm_gripper"]),
            ("allow_motion", False), ("bump_timeout_s", 20.0),
            ("max_torque_scale", 1.0), ("read_limits", True),
            ("fallback_limits",
             [12.5, 50.0, 5.0, 12.5, 30.0, 10.0, 12.5, 50.0, 10.0,
              12.5, 10.0, 28.0, 12.5, 45.0, 20.0, 12.5, 45.0, 20.0,
              12.5, 45.0, 40.0, 12.5, 45.0, 54.0, 12.5, 25.0, 200.0,
              12.5, 20.0, 200.0, 12.5, 280.0, 1.0, 12.5, 45.0, 10.0,
              12.5, 45.0, 10.0]),
            ("hold_kp", [300.0, 300.0, 150.0, 150.0, 40.0, 40.0, 30.0]),
            ("hold_kd", [2.5, 2.5, 2.5, 2.5, 0.8, 0.8, 0.8]),
            ("grip_hold_kp", 10.0), ("grip_hold_kd", 0.9),
            ("bump_kp", 45.0), ("bump_kd", 1.2), ("bump_step_deg", 0.2),
            ("grip_dq_th", 0.3), ("grip_tau_th", 0.3),
            ("j1_dq_th", 0.0125), ("j1_tau_th", 5.0),
            ("default_dq_th", 0.1), ("default_tau_th", 2.0),
            ("j1_bump_kp", 280.0), ("j1_bump_kd", 2.1),
            ("j2_home_kp", 280.0), ("j2_home_kd", 2.0),
            ("interp_kp", 52.0), ("interp_kd", 1.5), ("interp_time_s", 2.0),
            # --- 新流程参数 ---
            ("zero_file_path", "zero"),
            ("j1_protect_deg", 45.0), ("j1_protect_dir", "auto"),
            # J1 防护摆 S 型缓动时长(秒): 重基座关节线性匀速摆会先慢后
            # 突快并在结尾抖动(2026-09-09 实测), 余弦缓动两端零速。
            ("j1_protect_time_s", 6.0),
            ("return_time_s", 3.0),
            ("joint_first_dir", [-1, -1, 1, -1, 1, 1, 1, -1]),
            # --- v2 官方零位版 ---
            ("zero_move_speed_deg", 15.0),
            ("hold_after_done", True),
            # --- 仅限位模式: 跳过 set_zero 与零位改写, 限位以碰撞点
            #     绝对 raw 读数落盘(零位字段原样继承现有 zero 文件) ---
            ("limits_only", False),
            # --- 不做碰撞标定的关节, 逗号分隔 1-based 编号字符串
            #     (如 "1"=跳过 J1; 空串=全不跳)。rclpy 参数不允许空
            #     列表默认值(类型不可推断), 故用字符串。仅限位模式下,
            #     被跳过关节的限位从现有 zero 文件原样继承 ---
            ("no_bump_joints", ""),
            # --- 双臂适配 ---
            # has_gripper=false: 右臂 7 电机(无夹爪 ESC8)。电机表由
            # motor_ids/motor_types/joint_names/joint_first_dir 决定,
            # 碰撞顺序去掉夹爪, zero 文件 7 关节(config/calibration_params_right.yaml)
            ("has_gripper", True),
        ])

        gp = self.get_parameter
        self.iface = gp("can_iface").value
        self.use_fd = gp("use_fd").value
        self.side = gp("arm_side").value.lower()
        self.version = gp("robot_version").value.lower()
        self.allow_motion = gp("allow_motion").value
        self.limits_only = gp("limits_only").value
        nb_raw = str(gp("no_bump_joints").value or "")
        self.no_bump = set(int(x) for x in nb_raw.replace(" ", "").split(",")
                           if x)
        self.bump_timeout = gp("bump_timeout_s").value
        self.tau_scale = gp("max_torque_scale").value
        self.joint_names = list(gp("joint_names").value)
        self.has_gripper = bool(gp("has_gripper").value)
        self.n_motors = 8 if self.has_gripper else 7

        # ---------- 状态 ----------
        self.bus = None
        self.driver = None
        self.calib_thread = None
        self.abort_flag = threading.Event()
        self.error_msg = ""
        self.step_idx = 0
        self.phase = "idle"          # idle/init/bump/interp/zeroing/done/error
        self.total_steps = 12 if self.has_gripper else 11
        self.result_deltas = None
        self.zero_data = None
        self._active_motor = None    # 当前运动关节（异常缓回用）
        self._active_ref = None      # 其参考位置（零位）
        self.probe_state = None      # J1 方向试探: None / "probed"
        self.probe_dir = 0.0
        self._zero_q = None          # 试探时锁定的零位基准
        self.calibrating = False
        self.last_desc = "节点已就绪，等待启动指令"

        # ---------- ROS 接口 ----------
        self.pub_js = self.create_publisher(JointState,
                                            "~/joint_states", 10)
        self.pub_status = self.create_publisher(String, "~/status", 10)
        self.srv_start = self.create_service(Trigger, "~/start",
                                             self.on_start)
        self.srv_go_zero = self.create_service(Trigger, "~/go_zero",
                                               self.on_go_zero)
        self.srv_abort = self.create_service(Trigger, "~/abort",
                                             self.on_abort)
        self.js_timer = self.create_timer(0.02, self.publish_joint_states)
        self.refresh_timer = self.create_timer(0.1, self.refresh_cb)
        self.status_timer = self.create_timer(0.5, self.status_cb)

        self.get_logger().info(
            "零点标定节点就绪 iface=%s side=%s allow_motion=%s "
            "(标定需调用服务 /calibration/start)"
            % (self.iface, self.side, self.allow_motion))
        # 启动即初始化总线（只读监控；失败不阻塞节点，标定时会重试）
        try:
            online = self._init_bus()
            self.report("电机在线 %d/%d，监控已启动" % (online, self.n_motors))
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(
                "总线初始化失败（监控不可用，标定时将重试）: %s" % e)
            self.report("总线初始化失败: %s" % e, error=str(e))

    # ------------------------------------------------------------------
    # 总线初始化（启动时只读初始化，供监控；标定时复用）
    # ------------------------------------------------------------------

    def _init_bus(self):
        """打开 CANFD 总线、构建电机、确认在线（只读，不使能）。"""
        bus = dm.CanFdBus(self.iface, self.use_fd)
        motors = self.build_motors()
        drv = ArmDriver(bus, motors, self.get_logger())
        drv.refresh_all()
        online = sum(1 for m in motors if m.last_update > 0.0)
        if online == 0:
            bus.close()
            raise RuntimeError("%s 上无任何电机应答（检查接口配置/供电/接线）"
                               % self.iface)
        self.bus = bus
        self.driver = drv
        if self.get_parameter("read_limits").value:
            self.query_motor_limits(motors)
        return online

    # ------------------------------------------------------------------
    # 参数辅助
    # ------------------------------------------------------------------

    def build_motors(self):
        """构建电机列表；read_limits=true 时逐台读取寄存器限参。"""
        ids = list(self.get_parameter("motor_ids").value)
        types = list(self.get_parameter("motor_types").value)
        fb = list(self.get_parameter("fallback_limits").value)
        do_read = self.get_parameter("read_limits").value
        motors = []
        for i, (mid, mt) in enumerate(zip(ids, types)):
            lim = (fb[mt * 3], fb[mt * 3 + 1], fb[mt * 3 + 2])
            m = DMMotor(mid, mt, lim)
            motors.append(m)
        return motors

    def query_motor_limits(self, motors):
        """逐台读寄存器限参；失败保留兜底值并告警（实测发现 J3-J4 的
        VMAX=20 与官方表 10 不同，故必须以寄存器实测值为准）。"""
        for m in motors:
            got = self.driver.read_limits(m)
            if got:
                m.limits = got
                self.get_logger().info(
                    "ESC_ID=0x%02X 寄存器限参: PMAX=%.2f VMAX=%.2f TMAX=%.2f"
                    % (m.esc_id, got[0], got[1], got[2]))
            else:
                self.get_logger().warn(
                    "ESC_ID=0x%02X 限参寄存器读取失败，使用兜底表" % m.esc_id)

    # ------------------------------------------------------------------
    # 状态发布
    # ------------------------------------------------------------------

    def publish_joint_states(self):
        if self.driver is None:
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [m.q for m in self.driver.motors]
        msg.velocity = [m.dq for m in self.driver.motors]
        msg.effort = [m.tau for m in self.driver.motors]
        self.pub_js.publish(msg)

    def refresh_cb(self):
        """10Hz 轮询电机反馈（标定进行中由标定线程驱动，跳过）。"""
        if self.driver is not None and not self.calibrating:
            try:
                self.driver.refresh_all()
            except OSError as e:
                self.get_logger().warn("总线读取失败: %s" % e)

    def status_cb(self):
        """2Hz 周期发布状态快照（即使未开始标定也持续可见）。"""
        if self.driver is None:
            return
        self._publish_status()

    def _publish_status(self):
        st = {
            "phase": self.phase, "step": self.step_idx,
            "total_steps": self.total_steps, "desc": self.last_desc,
            "joint": "", "side": self.side,
            "positions_rad": ([m.q for m in self.driver.motors]
                              if self.driver else []),
            "state_codes": ([m.state_code for m in self.driver.motors]
                            if self.driver else []),
            "deltas_rad": self.result_deltas,
            "error": self.error_msg,
        }
        msg = String()
        msg.data = json.dumps(st, ensure_ascii=False)
        self.pub_status.publish(msg)

    def report(self, desc: str, joint: str = "", error: str = ""):
        """发布 JSON 状态（标定步骤/进度/关节位置/错误）并写日志。"""
        self.last_desc = desc
        if error:
            self.error_msg = error
        if self.driver is not None:
            self._publish_status()
        self.get_logger().info("[%d/%d] %s" % (self.step_idx,
                                               self.total_steps, desc))

    def next_step(self, desc, joint=""):
        self.step_idx += 1
        self.report(desc, joint)
        if self.abort_flag.is_set():
            raise RuntimeError("用户中止")

    # ------------------------------------------------------------------
    # 服务回调
    # ------------------------------------------------------------------

    def get_j1_protect_dir(self):
        """J1 防护/试探方向: auto 按臂侧, 否则用显式配置。"""
        pdir_cfg = str(self.get_parameter("j1_protect_dir").value).lower()
        if pdir_cfg == "positive":
            return 1.0
        if pdir_cfg == "negative":
            return -1.0
        return -1.0 if self.side == "right" else 1.0

    def on_start(self, req, resp):
        if self.calib_thread and self.calib_thread.is_alive():
            resp.success, resp.message = False, "标定已在进行中"
            return resp
        if not self.allow_motion:
            resp.success, resp.message = False, (
                "allow_motion=false（安全锁定）。确认安全后以 "
                "allow_motion:=true 重启节点")
            self.get_logger().warn("启动被拒绝：安全锁定未解除")
            return resp
        self.abort_flag.clear()
        self.error_msg = ""
        self.step_idx = 0
        self.result_deltas = None
        self.calib_thread = threading.Thread(
            target=self.calibration_worker, daemon=True)
        self.calib_thread.start()
        resp.success, resp.message = True, "标定已启动"
        return resp

    def on_abort(self, req, resp):
        go_zero_alive = (getattr(self, "_go_zero_thread", None) is not None
                         and self._go_zero_thread.is_alive())
        if ((self.calib_thread and self.calib_thread.is_alive())
                or go_zero_alive):
            self.abort_flag.set()
            resp.success, resp.message = True, "中止信号已发出，电机即将缓慢回位失能"
        elif self.probe_state == "probed":
            # 取消方向试探: 失能全部电机，重置状态
            self.probe_state = None
            self._zero_q = None
            try:
                if self.driver:
                    self.driver.disable_all()
            except Exception:  # noqa: BLE001
                pass
            resp.success, resp.message = True, "已取消试探，全部电机失能"
        else:
            resp.success, resp.message = False, "当前无进行中的标定"
        return resp

    def on_go_zero(self, req, resp):
        """缓慢运动到 zero 文件记录的零位基准位形并保持(供人工目视验证)。
        不碰撞限位、不写电机零点。"""
        if self.calib_thread and self.calib_thread.is_alive():
            resp.success, resp.message = False, "标定进行中, 不能执行回零位"
            return resp
        if getattr(self, "_go_zero_thread", None) is not None \
                and self._go_zero_thread.is_alive():
            resp.success, resp.message = False, "回零位已在进行中"
            return resp
        if not self.allow_motion:
            resp.success, resp.message = False, (
                "allow_motion=false（安全锁定）。确认安全后以 "
                "allow_motion:=true 重启节点")
            self.get_logger().warn("回零位被拒绝：安全锁定未解除")
            return resp
        target = self.load_zero_target()
        if target is None:
            resp.success, resp.message = False, (
                "无有效 zero 文件(status=ok), 没有零位基准可回")
            return resp

        def worker():
            try:
                self.calibrating = True
                self.phase = "go_zero"
                self.abort_flag.clear()
                self.error_msg = ""
                if self.driver is None:
                    self._init_bus()
                for m in self.driver.motors:      # 确保 MIT 模式(同 start)
                    mode = self.driver.read_ctrl_mode(m)
                    if mode is not None and mode != dm.CTRL_MODE_MIT:
                        self.bus.send(*dm.build_write_param(
                            m.esc_id, dm.RID_CTRL_MODE, dm.CTRL_MODE_MIT))
                        time.sleep(0.05)
                self.driver.enable_all()
                self.driver.refresh_all()
                self._zero_q = list(target)
                self.report("缓慢回零位基准(限速 %.0f°/s)..."
                            % float(self.get_parameter(
                                "zero_move_speed_deg").value))
                self.move_all_to_zero(target)
                self.driver.hold(
                    [(self.get_parameter("hold_kp").value[i],
                      self.get_parameter("hold_kd").value[i])
                     for i in range(7)]
                    + ([(self.get_parameter("grip_hold_kp").value,
                         self.get_parameter("grip_hold_kd").value)]
                       if self.has_gripper else []))
                self.phase = "done"
                self.report("已到零位基准并保持——目视验证后 Ctrl-C 失能; "
                            "确认无误可调用 /calibration_node/start 继续标定")
            except Exception as e:  # noqa: BLE001
                self.phase = "error"
                self.error_msg = str(e)
                self.report("回零位失败: %s" % e, error=str(e))
                try:
                    if self.driver:
                        self.driver.disable_all()
                except Exception:  # noqa: BLE001
                    pass
            finally:
                self.calibrating = False

        self._go_zero_thread = threading.Thread(target=worker, daemon=True)
        self._go_zero_thread.start()
        resp.success, resp.message = True, "回零位已启动"
        return resp

    # ------------------------------------------------------------------
    # 官方标定原语（时序与官方脚本 1:1）
    # ------------------------------------------------------------------

    def interpolate(self, m: DMMotor, q_target: float, kp: float, kd: float,
                    interp_time: float | None = None):
        """线性插值到目标位（官方 _interpolate: 500 步）。
        interp_time 缺省取参数 interp_time_s（默认 2.0s）。"""
        if interp_time is None:
            interp_time = float(self.get_parameter("interp_time_s").value)
        q_start = m.q
        steps = 500
        dt = interp_time / steps
        cnt = 0
        for _ in range(steps):
            if self.abort_flag.is_set():
                raise RuntimeError("用户中止")
            q_now = q_start + (q_target - q_start) / interp_time * dt
            self.driver.mit(m, kp, kd, q_now, 0.0, 0.0)
            time.sleep(dt)
            q_start = q_now
            cnt += 1
            if cnt % 20 == 0:          # 定期刷新其他关节保持
                self.hold_others(m.esc_id)
            self.driver.poll(0.01)

    def interpolate_eased(self, m: DMMotor, q_target: float, kp: float,
                          kd: float, duration: float):
        """余弦 S 型缓动到目标位(两端零速): 重关节(基座 J1)线性匀速
        摆动会先慢后突快并在结尾抖动, 缓动曲线消除两端的加速度突变。
        500 步, 带中止与相邻关节保持, 与 interpolate 同框架。"""
        q_start = m.q
        steps = 500
        dt = duration / steps
        for k in range(1, steps + 1):
            if self.abort_flag.is_set():
                raise RuntimeError("用户中止")
            t = k / steps
            ease = 0.5 * (1.0 - math.cos(math.pi * t))   # 两端零速
            q_now = q_start + (q_target - q_start) * ease
            self.driver.mit(m, kp, kd, q_now, 0.0, 0.0)
            time.sleep(dt)
            if k % 20 == 0:
                self.hold_others(m.esc_id)
        self.driver.poll(0.01)

    def hold_others(self, active_esc_id: int):
        """给当前标定关节之外的所有关节发送保持帧（单关节标定时其他关节不动）。
        各关节保持目标:
          - J1: 防护位(45°)，仅在其防护期间(_j1_hold_position 非空)生效
          - 其余: 各自保持点(_hold_q, v2: 已走到的关节=目标, 未走的=当前位置,
            防止对远距离误差施加大力矩阶跃拽伤电机/关节)
        注意: J1 在防护位时绝不能被拉回零位，否则姿态崩塌。"""
        hold_q = getattr(self, "_hold_q", None)
        zero = hold_q if hold_q is not None \
            else getattr(self, "_zero_q", None)
        if zero is None or self.driver is None:
            return
        j1_hold = getattr(self, "_j1_hold_position", None)
        interp_kp = self.get_parameter("interp_kp").value
        interp_kd = self.get_parameter("interp_kd").value
        for i, m in enumerate(self.driver.motors):
            if m.esc_id == active_esc_id:
                continue
            if i == 0 and j1_hold is not None:
                # J1 防护位: 用中等刚度保持（kp300 会在位置滞后时猛拉）
                target = j1_hold
                kp, kd = interp_kp, interp_kd
            else:
                target = zero[i]
                if i < 7:
                    kp = self.get_parameter("hold_kp").value[i]
                    kd = self.get_parameter("hold_kd").value[i]
                else:
                    kp = self.get_parameter("grip_hold_kp").value
                    kd = self.get_parameter("grip_hold_kd").value
            self.driver.mit(m, kp, kd, target, 0.0, 0.0)

    def bump_to_limit(self, m: DMMotor, direction: float, step_rad: float,
                      kp: float, kd: float, dq_th: float, tau_th: float):
        """向 direction(+1/-1) 方向以固定步进逼近机械限位。
        到位判定三条件缺一不可:
          1. 已脱开起点(>2°)——排除碰撞后原地起步的旧反馈误判
          2. 位置连续 10 轮(约 50ms)无进展——真限位的堵转特征，
             排除运动途中瞬时卡顿(dq骤降/tau突升)造成的提前误判
          3. |dq|<dq_th 且 |tau|>tau_th
        返回碰撞点绝对位置（第一/第二限位点）。"""
        q_start = m.q
        q_target = q_start + direction * step_rad
        clear_rad = math.radians(2.0)      # 脱开限位所需行程
        t0 = time.monotonic()
        q_prev = m.q
        stalled = 0
        cnt = 0
        while True:
            if self.abort_flag.is_set():
                raise RuntimeError("用户中止")
            if time.monotonic() - t0 > self.bump_timeout:
                raise RuntimeError(
                    "ESC_ID=0x%02X 碰撞限位超时(%.1fs)，检查机械结构与阈值"
                    % (m.esc_id, self.bump_timeout))
            self.driver.mit(m, kp, kd, q_target, 0.0, 0.0)
            time.sleep(0.005)          # 官方控制间隔
            self.driver.poll(0.002)    # 追加收帧，保证反馈新鲜
            cnt += 1
            if cnt % 10 == 0:          # 50ms 周期刷新其他关节零位保持
                self.hold_others(m.esc_id)
            left_start = abs(m.q - q_start) > clear_rad
            # 停滞计数: 位置几乎无进展（<20% 步长）则累计，有进展清零
            if abs(m.q - q_prev) < step_rad * 0.2:
                stalled += 1
            else:
                stalled = 0
            q_prev = m.q
            if (left_start and stalled >= 10
                    and abs(m.dq) < dq_th and abs(m.tau) > tau_th):
                return m.q
            q_target += direction * step_rad

    def settle_at(self, m: DMMotor, q_target: float, kp: float, kd: float,
                  tol_deg: float = 1.0, timeout_s: float = 3.0):
        """发送保持帧并等待电机收敛到目标位（容差 tol_deg），
        超时未收敛抛异常。解决插值结束后立即检查、电机尚未跟踪到位的问题。"""
        deadline = time.monotonic() + timeout_s
        cnt = 0
        while time.monotonic() < deadline:
            if self.abort_flag.is_set():
                raise RuntimeError("用户中止")
            self.driver.mit(m, kp, kd, q_target, 0.0, 0.0)
            time.sleep(0.05)
            cnt += 1
            if cnt % 1 == 0:           # 每轮刷新其他关节零位保持
                self.hold_others(m.esc_id)
            if abs(m.q - q_target) < math.radians(tol_deg):
                return
        raise RuntimeError(
            "ESC_ID=0x%02X 收敛超时: 目标 %.3f, 当前 %.3f (偏差 %.2f°)"
            % (m.esc_id, q_target, m.q, abs(m.q - q_target) * R2D))

    def return_to_zero(self, m: DMMotor, q_zero: float, name: str):
        """以标定碰撞相同的步进节奏（bump_step_deg/5ms）匀速回零位。
        目标位置累积推进（与 bump_to_limit 同模式）：电机若被限位卡住，
        目标持续推远、力矩增大可自动脱困；带超时保护，不会死循环。
        到达后收敛等待（容差 1°）。"""
        step = self.get_parameter("bump_step_deg").value * D2R
        kp, kd, _, _ = self.joint_gains(name)
        q_start = m.q
        direction = math.copysign(1.0, q_zero - q_start)
        total = abs(q_zero - q_start)
        if total < step:                       # 已在零位附近
            self.settle_at(m, q_zero, kp, kd, tol_deg=1.0, timeout_s=3.0)
            return
        # 超时: 理论速度 step/5ms≈40°/s，留 3 倍余量 + 5s 底值
        timeout = max(5.0, total / step * 0.005 * 3.0)
        t0 = time.monotonic()
        q_target = q_start
        cnt = 0
        tau_stall = 0
        # 允许目标越过终点最多 10°——位置误差持续增大→力矩自动爬升，
        # 破静摩擦（目标 clamp 在终点会封顶力矩，卡死在静摩擦处）
        overshoot = math.radians(10.0)
        # 堵转保护阈值: min(20Nm, TMAX×0.5)——真正的硬阻挡安全停止
        tau_lim = min(20.0, m.limits[2] * 0.5)
        while abs(m.q - q_zero) > step:
            if self.abort_flag.is_set():
                raise RuntimeError("用户中止")
            if time.monotonic() - t0 > timeout:
                raise RuntimeError(
                    "ESC_ID=0x%02X 回零位超时(%.1fs): 目标 %.3f, 当前 %.3f "
                    "(关节被卡住?)" % (m.esc_id, timeout, q_zero, m.q))
            q_target += direction * step
            if (q_target - q_zero) * direction >= overshoot:
                raise RuntimeError(
                    "ESC_ID=0x%02X 目标越程 10° 仍未到位(q=%.3f, tau=%.1f)"
                    "——关节受阻" % (m.esc_id, m.q, m.tau))
            self.driver.mit(m, kp, kd, q_target, 0.0, 0.0)
            time.sleep(0.005)
            self.driver.poll(0.002)
            cnt += 1
            if cnt % 10 == 0:          # 50ms 周期刷新其他关节零位保持
                self.hold_others(m.esc_id)
            # 堵转保护: 力矩超限且位置几乎不动 → 立即停止（不硬推）
            if abs(m.tau) > tau_lim and abs(m.dq) < 0.05 \
                    and abs(m.q - q_target) > step:
                tau_stall += 1
            else:
                tau_stall = 0
            if tau_stall >= 30:        # 约 150ms 受阻
                raise RuntimeError(
                    "ESC_ID=0x%02X 移动受阻(tau=%.1f Nm, q=%.3f)——遇到阻挡，"
                    "已停止硬推" % (m.esc_id, m.tau, m.q))
        # 最后收敛等待（容差 1°）
        self.settle_at(m, q_zero, kp, kd, tol_deg=1.0, timeout_s=3.0)

    def load_existing_limits(self):
        """读取现有 zero 文件的限位(绝对 raw 读数), 供 no_bump 关节继承。
        返回 (pos_list, neg_list) 或 (None, None)。"""
        path = str(self.get_parameter("zero_file_path").value)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            pos = [float(v) for v in data["pos_limits_rad"]]
            neg = [float(v) for v in data["neg_limits_rad"]]
            if len(pos) < self.n_motors or len(neg) < self.n_motors:
                return None, None
            return pos, neg
        except (OSError, ValueError, KeyError):
            return None, None

    def load_zero_target(self):
        """从现有 zero 文件读取零位基准位形(按本机电机数, 当前电机系)。
        文件缺失 / status 非 ok / 数据不足 / 臂侧不符时返回 None。"""
        path = str(self.get_parameter("zero_file_path").value)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        if data.get("status") != "ok":
            return None
        if data.get("arm_side") and \
                str(data["arm_side"]).lower() != self.side:
            self.get_logger().error(
                "zero 文件 arm_side=%s 与本节点 side=%s 不符, 拒绝使用"
                "（双臂并存防呆: 别把另一只臂的 zero 当基准）"
                % (data["arm_side"], self.side))
            return None
        z = data.get("zero_position_rad")
        if not z or len(z) < self.n_motors:
            return None
        return [float(v) for v in z[:self.n_motors]]

    def move_slow(self, idx, target, speed_deg=None):
        """单关节限速运动到目标(当前电机系), 带堵转保护与超时。
        运动期间其它关节由 hold_others 保持。"""
        m = self.driver.motors[idx]
        name = ("J1" if m.esc_id == 1 else
                ("grip" if m.esc_id == 8 else "J%d" % m.esc_id))
        # 用保持刚度(足以带动整臂, J3=150)而非碰撞刚度(45 太软);
        # J1 沿用高刚度, 夹爪用夹爪保持刚度
        hkp = self.get_parameter("hold_kp").value
        hkd = self.get_parameter("hold_kd").value
        if m.esc_id == 1:
            kp = float(self.get_parameter("j1_bump_kp").value)
            kd = float(self.get_parameter("j1_bump_kd").value)
        elif m.esc_id == 8:
            kp = float(self.get_parameter("grip_hold_kp").value)
            kd = float(self.get_parameter("grip_hold_kd").value)
        else:
            kp = float(hkp[m.esc_id - 1])
            kd = float(hkd[m.esc_id - 1])
        if speed_deg is None:
            speed_deg = float(self.get_parameter("zero_move_speed_deg").value)
        speed = max(speed_deg * D2R, 1e-3)
        step = speed * 0.05                       # 50ms 一拍
        tau_lim = min(20.0, m.limits[2] * 0.5)
        timeout = max(10.0, abs(target - m.q) / speed * 3.0)
        t0 = time.monotonic()
        q_prev = m.q
        stall = 0
        cnt = 0
        while abs(m.q - target) > math.radians(0.5):
            if self.abort_flag.is_set():
                raise RuntimeError("用户中止")
            if time.monotonic() - t0 > timeout:
                raise RuntimeError(
                    "ESC_ID=0x%02X 回零位运动超时: 目标 %.3f, 当前 %.3f"
                    % (m.esc_id, target, m.q))
            err = target - m.q
            q_cmd = m.q + max(-step, min(step, err))
            self.driver.mit(m, kp, kd, q_cmd, 0.0, 0.0)
            time.sleep(0.005)
            self.driver.poll(0.002)
            cnt += 1
            if cnt % 10 == 0:
                self.hold_others(m.esc_id)
            if abs(m.q - q_prev) < step * 0.05 \
                    and abs(err) > math.radians(1.0):
                stall += 1
            else:
                stall = 0
            q_prev = m.q
            if stall >= 60:                       # 约 3s 无进展
                raise RuntimeError(
                    "ESC_ID=0x%02X 运动受阻(kp=%.0f, tau=%.1f Nm, q=%.3f,"
                    " 目标=%.3f)——电机未出力或被硬挡"
                    % (m.esc_id, kp, m.tau, m.q, target))
        self.settle_at(m, target, kp, kd)

    def move_all_to_zero(self, zero_q):
        """各关节依次限速回零位基准(先 J2..J7, 再 J1)。
        v2: 未轮到的关节只软保持在当前位置(无力矩阶跃),
        走完的关节把保持点更新到目标。
        夹爪(idx 7)跳过: 真机管线不控制夹爪, 其小间隙常被摩擦卡住
        导致无谓超时; 保持原位即可(无夹爪机型整段不存在)。"""
        self._hold_q = [m.q for m in self.driver.motors]
        for idx in (1, 2, 3, 4, 5, 6, 0):
            name = "J1" if idx == 0 else "J%d" % (idx + 1)
            self.report("%s 缓慢回零位" % name, name)
            self._active_motor = self.driver.motors[idx]
            self._active_ref = zero_q[idx]
            self.move_slow(idx, zero_q[idx])
            self._hold_q[idx] = zero_q[idx]
        if self.has_gripper:
            self._hold_q[7] = self.driver.motors[7].q
            self.report("夹爪跳过回零位(真机不控夹爪), 保持原位", "grip")

    def joint_gains(self, name: str):
        """按关节返回碰撞刚度与到位阈值（J1/夹爪特殊，其余默认）。"""
        kp = self.get_parameter("bump_kp").value
        kd = self.get_parameter("bump_kd").value
        if name == "J1":
            kp = self.get_parameter("j1_bump_kp").value
            kd = self.get_parameter("j1_bump_kd").value
            return (kp, kd, self.get_parameter("j1_dq_th").value,
                    self.get_parameter("j1_tau_th").value)
        if name == "grip":
            return (kp, kd, self.get_parameter("grip_dq_th").value,
                    self.get_parameter("grip_tau_th").value)
        return (kp, kd, self.get_parameter("default_dq_th").value,
                self.get_parameter("default_tau_th").value)

    def calibrate_joint_two_sided(self, idx: int, name: str, q_zero: float):
        """单关节双向限位标定：
        正向碰撞记录第一限位 → 反向碰撞记录第二限位 → 低速平稳回零位。"""
        m = self.driver.motors[idx]
        kp, kd, dq_th, tau_th = self.joint_gains(name)
        step = self.get_parameter("bump_step_deg").value * D2R
        first = float(self.get_parameter("joint_first_dir").value[idx])
        ret_t = self.get_parameter("return_time_s").value
        self._active_motor, self._active_ref = m, q_zero

        self.phase = "bump"
        q_a = self.bump_to_limit(m, first, step, kp, kd, dq_th, tau_th)
        self.report("%s 第一限位 %.3f rad" % (name, q_a), name)
        q_b = self.bump_to_limit(m, -first, step, kp, kd, dq_th, tau_th)
        self.report("%s 第二限位 %.3f rad" % (name, q_b), name)

        self.phase = "interp"
        # 以标定同速匀速回零位 + 收敛验证（每关节结束时必须处于零位）
        for attempt in range(3):
            try:
                self.return_to_zero(m, q_zero, name)
                break
            except RuntimeError as e:
                self.get_logger().warn("%s 回零位失败: %s，重试"
                                       % (name, e))
        else:
            raise RuntimeError("%s 未能回到零位" % name)
        self.report("%s 已回零位" % name, name)
        return (q_a, q_b) if first > 0 else (q_b, q_a)

    def safe_recover(self, err: Exception, zero_q):
        """异常保护：不立即停止——当前运动关节缓慢回到参考位，
        失能全部电机，并把当前状态写入 zero 文件。"""
        was_aborted = self.abort_flag.is_set()
        self.abort_flag.clear()          # 允许回位插值执行完毕
        self.phase = "recover"
        self.get_logger().error("标定异常: %s —— 正在缓慢回位保护" % err)
        self.report("异常: %s，缓慢回位中" % err, error=str(err))
        try:
            if (self.driver and self._active_motor is not None
                    and self._active_ref is not None):
                # 与标定同速匀速回参考位，避免速度突变
                self.return_to_zero(self._active_motor, self._active_ref,
                                    "J1" if self._active_motor.esc_id == 1
                                    else ("grip" if self._active_motor.esc_id == 8
                                          else "J%d" % self._active_motor.esc_id))
                self.get_logger().info("已缓慢回到参考位置")
        except Exception as e2:  # noqa: BLE001
            self.get_logger().error("回位过程失败（电机将失能）: %s" % e2)
        try:
            if self.driver:
                self.driver.disable_all()
        except Exception:  # noqa: BLE001
            pass
        # 记录当前状态: 仅限位模式绝不覆盖主 zero 文件(落盘 .partial)
        try:
            payload = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "aborted" if was_aborted else "error",
                "error": str(err),
                "zero_position_rad": ([round(q, 4) for q in zero_q]
                                      if zero_q else None),
                "positions_at_abort_rad":
                    ([round(m.q, 4) for m in self.driver.motors]
                     if self.driver else []),
            }
            if self.limits_only:
                path = str(self.get_parameter("zero_file_path").value)
                with open(path + ".partial", "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                self.get_logger().info(
                    "仅限位模式: 中断状态写 %s.partial(主 zero 文件未动)"
                    % path)
            else:
                self.write_zero_file(payload)
        except Exception as e3:  # noqa: BLE001
            self.get_logger().error("zero 文件写入失败: %s" % e3)

    def write_zero_file(self, payload: dict):
        """标定数据落盘（zero 文件，JSON 格式）。"""
        path = str(self.get_parameter("zero_file_path").value)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self.get_logger().info("标定数据已写入: %s" % path)

    # ------------------------------------------------------------------
    # 标定主流程（需求版：软件零位基准 + 双向限位 + J1 防护）
    # ------------------------------------------------------------------

    def calibration_worker(self):
        zero_q = None
        limits_pos = None
        limits_neg = None
        try:
            self.phase = "init"
            self.calibrating = True
            # 0. 总线与电机初始化（启动时已初始化则复用）
            if self.driver is None:
                online = self._init_bus()
                self.get_logger().info("总线重连成功，在线 %d 台" % online)
            motors = self.driver.motors
            self.driver.refresh_all()
            for m in motors:                       # 确认在线
                if m.last_update == 0.0:
                    raise RuntimeError("ESC_ID=0x%02X 无反馈，电机不在线"
                                       % m.esc_id)
            for m in motors:                       # 确认 MIT 模式
                mode = self.driver.read_ctrl_mode(m)
                if mode is not None and mode != dm.CTRL_MODE_MIT:
                    self.bus.send(*dm.build_write_param(
                        m.esc_id, dm.RID_CTRL_MODE, dm.CTRL_MODE_MIT))
                    time.sleep(0.05)
            self.report("电机在线确认: %d/%d, 限参与控制模式就绪"
                        % (len(motors), self.n_motors))

            # 1. 使能 → 缓慢回零位基准(官方零位版: 目标取现有 zero 文件
            #    的 zero_position_rad; 无文件时以当前位形为零位基准) → 保持
            self.driver.enable_all()
            self.driver.refresh_all()
            hold = [(self.get_parameter("hold_kp").value[i],
                     self.get_parameter("hold_kd").value[i])
                    for i in range(7)]
            if self.has_gripper:
                hold.append((self.get_parameter("grip_hold_kp").value,
                             self.get_parameter("grip_hold_kd").value))
            self.driver.hold(hold)
            self._zero_q = [round(m.q, 4) for m in motors]   # 兜底: 当前位形
            target = self.load_zero_target()
            if target is not None:
                zero_q = list(target)
                self._zero_q = zero_q      # 供 hold_others/move_slow 使用
                self.next_step("缓慢回零位基准(以现有 zero 文件为目标)...")
                self.move_all_to_zero(zero_q)
                self.driver.hold(hold)
            else:
                zero_q = list(self._zero_q)
            self.next_step("已到达零位基准: %s"
                           % ["%.3f" % v for v in zero_q])

            # 2. J1 防护摆动：其余关节保持，J1 沿已确认方向 S 型缓摆 45°
            self.next_step("J1 防护摆动 %.0f°" %
                           self.get_parameter("j1_protect_deg").value, "J1")
            j1 = motors[0]
            prot_deg = self.get_parameter("j1_protect_deg").value * D2R
            pdir = self.get_j1_protect_dir()   # 与试探方向一致（已人工确认）
            self._active_motor, self._active_ref = j1, zero_q[0]
            self.phase = "interp"
            j1_expect = zero_q[0] + pdir * prot_deg
            # S 型缓动到防护位(两端零速: 起步缓突围静摩擦, 结尾不突停
            # 不抖); 结束后留收敛窗口, 重关节跟踪滞后在此期间消化
            self.interpolate_eased(j1, j1_expect,
                                   self.get_parameter("interp_kp").value,
                                   self.get_parameter("interp_kd").value,
                                   float(self.get_parameter(
                                       "j1_protect_time_s").value))
            try:
                self.settle_at(j1, j1_expect,
                               self.get_parameter("interp_kp").value,
                               self.get_parameter("interp_kd").value,
                               tol_deg=8.0, timeout_s=6.0)
            except RuntimeError:
                pass                            # 收敛慢不致命, 下方阈值兜底
            # J1 跟踪自检: 收敛窗口后偏差仍过大才判定不受控
            if abs(j1.q - j1_expect) > math.radians(15):
                raise RuntimeError(
                    "J1 防护摆到位偏差过大(期望 %.1f°, 实际 %.1f°)——J1 "
                    "不受控: 检查供电/使能; 若刚发生过流锁存, 断电 15s "
                    "后重试" % (math.degrees(j1_expect), math.degrees(j1.q)))
            # J1 防护位锁定：其他关节标定期间 J1 保持 45° 而非零位
            self._j1_hold_position = j1_expect

            # 3. 依次双向限位标定: (夹爪) → J7 → J6 → J5 → J4 → J3 → J2
            limits_pos = [None] * len(motors)
            limits_neg = [None] * len(motors)
            order = ([(7, "grip")] if self.has_gripper else []) + \
                [(6, "J7"), (5, "J6"), (4, "J5"), (3, "J4"), (2, "J3"),
                 (1, "J2")]
            for idx, name in order:
                if (idx + 1) in self.no_bump:
                    self.next_step("%s 跳过碰撞标定(限位继承现有文件)" % name,
                                   name)
                    continue
                self.next_step("%s 双向限位标定" % name, name)
                qp, qn = self.calibrate_joint_two_sided(idx, name,
                                                        zero_q[idx])
                limits_pos[idx], limits_neg[idx] = qp, qn

            # 4. J1 标定(可跳过: J1 基座无机械停挡, "限位"=出线束安全
            #    范围, 碰撞逼近会绞线——2026-09-09 实测 20s 无停挡)。
            if 1 in self.no_bump:
                if not self.limits_only:
                    raise RuntimeError(
                        "no_bump_joints 仅支持 limits_only 模式"
                        "(跳过关节的限位需从现有 zero 文件继承)")
                self.next_step("J1 跳过碰撞标定, 防护位回零位", "J1")
                self._j1_hold_position = None
                self._active_motor, self._active_ref = j1, zero_q[0]
                self.phase = "interp"
                self.return_to_zero(j1, zero_q[0], "J1")
            else:
                self.next_step("J1 防护位回零位", "J1")
                self._j1_hold_position = None   # 防护期结束，J1 保持目标归零位
                self._active_motor, self._active_ref = j1, zero_q[0]
                self.phase = "interp"
                self.return_to_zero(j1, zero_q[0], "J1")
                self.next_step("J1 双向限位标定", "J1")
                qp, qn = self.calibrate_joint_two_sided(0, "J1", zero_q[0])
                limits_pos[0], limits_neg[0] = qp, qn

            # 5. 全部关节缓慢回零位基准(消除碰撞累计漂移)
            self.next_step("全部关节缓慢回零位基准")
            self.move_all_to_zero(zero_q)

            # 5.5 仅限位模式(limits_only=true): 到此为止——不 set_zero、
            #     不改写零位, 限位以碰撞点**绝对 raw 读数**落盘(与
            #     hand-spliced 文件同口径, gen 的 urdf=motor−d 直接可用)。
            #     全程电机 flash 零点不被触碰, 任何时刻中止都无副作用。
            if self.limits_only:
                self.phase = "zeroing"
                self.next_step("仅限位模式: 跳过 set_zero, 保存限位数据")
                miss = [i for i in range(len(motors))
                        if limits_pos[i] is None or limits_neg[i] is None]
                if miss:
                    old_p, old_n = self.load_existing_limits()
                    if old_p is None:
                        raise RuntimeError(
                            "跳过关节 %s 需从现有 zero 文件继承限位, "
                            "但文件读取失败" % [i + 1 for i in miss])
                    for i in miss:
                        limits_pos[i], limits_neg[i] = old_p[i], old_n[i]
                        self.get_logger().info(
                            "J%d 限位继承现有 zero 文件" % (i + 1))
                self.zero_data = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "ok",
                    "arm_side": self.side,
                    "calib_version": "hand-spliced-v1-limits-only",
                    "set_zero_done": False,
                    "_note": "仅限位重标定(不 set_zero): "
                             "zero_position_rad 原样继承现有文件; "
                             "限位=碰撞点绝对 raw 读数(未 set_zero 帧)",
                    "zero_position_rad": [round(v, 4) for v in zero_q],
                    "zero_target_pre_setzero_rad": [round(v, 4)
                                                    for v in zero_q],
                    "pos_limits_rad": [round(q, 4) for q in limits_pos],
                    "neg_limits_rad": [round(q, 4) for q in limits_neg],
                }
                self.write_zero_file(self.zero_data)
                self.result_deltas = self.zero_data["pos_limits_rad"]
                self.phase = "done"
                self.driver.hold(hold)
                if self.get_parameter("hold_after_done").value:
                    self.report("仅限位标定完成(电机零点未动)! 正在保持零位"
                                "位形——目视验证后 Ctrl-C 失能")
                    while rclpy.ok() and not self.abort_flag.is_set():
                        self.driver.hold(hold)
                        time.sleep(0.1)
                    self.report("退出保持, 失能全部电机")
                else:
                    time.sleep(2.0)
                    self.report("仅限位标定完成！zero 文件已更新")
                self.driver.disable_all()
                return

            # 6. 官方零位: set_zero(0xFE) 把当前位形写入电机 flash 零点
            #    此步之后电机系 = 零位系; 此前任何中止都不影响 flash
            self.next_step("set_zero 写入电机零点(官方 0xFE)...")
            self.driver.refresh_all()
            self.driver.set_zero_all()
            time.sleep(0.5)
            self.driver.refresh_all()

            # 7. 保存标定数据: 零位 = set_zero 后实测(≈全零);
            #    限位 = 碰撞点平移到新系(减 set_zero 前零位基准读数)
            self.phase = "zeroing"
            self.next_step("保存标定数据")
            zero_new = [round(m.q, 4) for m in motors]
            self.zero_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "ok",
                "arm_side": self.side,
                "calib_version": "official-zero-v2",
                "zero_position_rad": zero_new,
                # set_zero 前零位基准读数(旧电机系): 上位机据此换算
                # 新映射 d = 上次d − 本向量(支持把任意位形定为零位)
                "zero_target_pre_setzero_rad": [round(v, 4) for v in zero_q],
                "pos_limits_rad": [round(q - s, 4) for q, s
                                   in zip(limits_pos, zero_q)],
                "neg_limits_rad": [round(q - s, 4) for q, s
                                   in zip(limits_neg, zero_q)],
            }
            self.write_zero_file(self.zero_data)
            self.result_deltas = self.zero_data["pos_limits_rad"]
            self._zero_q = zero_new            # 之后保持/回位用新系
            self._active_motor = None          # set_zero 后旧参考位作废,
            self._active_ref = None            # 异常保护只做失能

            # 8. 收尾：保持零位供人工验证(Ctrl-C 或 abort 失能)
            self.phase = "done"
            self.driver.hold(hold)
            if self.get_parameter("hold_after_done").value:
                self.report("标定完成(set_zero 已写入电机)! 正在保持零位"
                            "——请目视验证, 完成后 Ctrl-C 失能")
                while rclpy.ok() and not self.abort_flag.is_set():
                    self.driver.hold(hold)
                    time.sleep(0.1)
                self.report("退出保持, 失能全部电机")
            else:
                time.sleep(2.0)
                self.report("标定完成！zero 文件已保存")
            self.driver.disable_all()
        except Exception as e:  # noqa: BLE001 —— 统一异常保护
            self.phase = "error"
            self.error_msg = str(e)
            self.safe_recover(e, zero_q)
        finally:
            # 最终兜底：确保失能；保留 bus 供监控
            self.calibrating = False
            try:
                if self.driver:
                    self.driver.disable_all()
                    self.get_logger().info("已失能全部电机（安全兜底）")
            except Exception:  # noqa: BLE001
                pass


def main(args=None):
    rclpy.init(args=args)
    node = CalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Ctrl+C 时 launch 可能已 shutdown context，容错处理
        try:
            node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        try:
            rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
