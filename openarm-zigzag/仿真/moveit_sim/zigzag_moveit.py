#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MoveIt 之字形走线执行节点（基于 OpenArm 官方 openarm_bimanual_moveit_config）
============================================================================
前置: 官方 demo 已启动（fake hardware + move_group + RViz）:
    ros2 launch openarm_bimanual_moveit_config demo.launch.py \
        arm_type:=openarm_v1.0 use_fake_hardware:=true

本节点全部使用 MoveIt **服务接口**（不依赖 action）:
  - /plan_kinematic_path   (GetMotionPlan)      规划
  - /execute_trajectory (ExecuteTrajectory action) 执行
  - /compute_cartesian_path (GetCartesianPath)   之字形直线段
  - /apply_planning_scene   (ApplyPlanningScene) 墙体碰撞体
  - /get_planning_scene     (GetPlanningScene)   规划坐标系

流程: 回零位(0.15倍速) → 到之字形起点(内置IK解, 0.2倍速) →
      逐段笛卡尔直线 (TCP 5cm/s 限速, 分数<1 回退关节目标)

用法:  ./run_moveit_zigzag.sh [--dry-run]
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
import tf2_ros
import yaml
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from geometry_msgs.msg import Point, Pose
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import (CollisionObject, Constraints, JointConstraint,
                             MotionPlanRequest, RobotState, RobotTrajectory)
from moveit_msgs.srv import (ApplyPlanningScene, GetCartesianPath,
                             GetMotionPlan, GetPlanningScene)
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import Marker, MarkerArray

TCP_OFFSET = 0.12          # hand_tcp → 指尖 (沿接近轴)
PLANE_X = 0.30
APPROACH_AXIS = np.array([1.0, 0.0, 0.0])
# 以下 4 项为臂相关常量, 默认左臂; main() 按 --side 覆盖(双臂重构 2026-09-08)
LINK = "openarm_left_hand_tcp"
GROUP = "left_arm"
JOINT_NAMES = ["openarm_left_joint%d" % i for i in range(1, 8)]
JTC_ACTION = "/left_joint_trajectory_controller/follow_joint_trajectory"
# 重定时关节速率上限(rad/s): 必须是"实机跟得上的速率", 不能用 URDF
# 理论值(16.75 等=960°/s)——2026-09-04 实测: dense 中存在构型翻转点
# (J5 一步 139°/J7 94°, 5D IK 自旋翻转), 理论限速下这些点只分到 0.29s
# (指令 480°/s), 实机跟不上 → JTC 容差中止。改用保守可执行速率后,
# 翻转摊到数秒内缓慢完成(自旋翻转不影响 TCP), 总时长仅 +12s。
VEL_LIMITS = [0.5, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8]  # rad/s
TCP_SPEED = 0.05           # 之字形 TCP 限速 5cm/s

EC_NAMES = {1: "SUCCESS", -1: "PLANNING_FAILED", -2: "NO_MOTION_PLAN_FOUND",
            -3: "INVALID_MOTION_PLAN", -4: "MOTION_PLAN_INVALIDATED_BY_ENV_CHANGE",
            -5: "TIMED_OUT", -7: "CONTROL_FAILED", -10: "START_STATE_IN_COLLISION",
            -12: "GOAL_IN_COLLISION", -31: "NO_IK_SOLUTION",
            -32: "INVALID_ROBOT_STATE", 99999: "FAILURE(通用)"}


def ec_name(val):
    return EC_NAMES.get(val, "code=%s" % val)


def quat_z_axis_to_x():
    """旋转矩阵 (x_tcp=(0,0,-1), y_tcp=(0,1,0), z_tcp=(1,0,0)) → 四元数 (w,x,y,z)。"""
    m = np.array([[0.0, 0.0, 1.0],
                  [0.0, 1.0, 0.0],
                  [-1.0, 0.0, 0.0]])
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
        w = 0.25 * s
        q = [0.0, 0.0, 0.0, 0.0]
        q[i + 1] = 0.25 * s
        q[j + 1] = (m[j, i] + m[i, j]) / s
        q[k + 1] = (m[k, i] + m[i, k]) / s
        x, y, z = q[1], q[2], q[3]
    return w, x, y, z


class ZigzagMover(Node):

    def __init__(self, ws_path: str, dry_run: bool,
                 real: bool = False, speed_scale: float = 1.0,
                 safety_yaml: str = ""):
        super().__init__("zigzag_moveit")
        self.dry = dry_run
        self.real = real              # 真机模式: 零位动作改用标定初始位形
        self.speed_scale = speed_scale  # 整体时间放慢系数(真机首跑建议 0.5)
        with open(ws_path, encoding="utf-8") as f:
            self.ws = json.load(f)
        self.waypoints = [np.array(wp) for wp in self.ws["zigzag_waypoints"]]
        self.joint_sols = self.ws["zigzag_joint_solutions_rad"]
        self.joint_names = list(JOINT_NAMES)
        self.q_quat = quat_z_axis_to_x()
        self.frame = "world"
        # 真机模式: 从 real_safety.yaml 读标定初始位形作为"家"位形
        # (真机上 URDF 全零位形需要 J1 转过 ~120°, 首跑不安全, 故用标定位形)
        self.home_q = [0.0] * 7
        if real:
            with open(safety_yaml, encoding="utf-8") as f:
                self.home_q = list(yaml.safe_load(f)["home_rad"])

        self.cli_plan = self.create_client(GetMotionPlan,
                                           "/plan_kinematic_path")
        self.cli_cart = self.create_client(GetCartesianPath,
                                           "/compute_cartesian_path")
        self.cli_scene = self.create_client(ApplyPlanningScene,
                                            "/apply_planning_scene")
        self.cli_scene_get = self.create_client(GetPlanningScene,
                                                "/get_planning_scene")
        self.pub_mk = self.create_publisher(MarkerArray, "/zigzag_markers", 10)
        # RViz 实际订阅的是这个话题(2026-09-04 实测), 验证标签发这里
        self.pub_verify = self.create_publisher(MarkerArray,
                                                "/visualization_marker_array",
                                                10)
        self.act_exec = ActionClient(self, ExecuteTrajectory,
                                     "/execute_trajectory")
        # /joint_states 首条消息 = joint_state_broadcaster 已激活,
        # move_group 能看到当前状态(否则执行 ABORT)。TCP 验证采样也用它对时。
        self._js_arrived = threading.Event()
        self._last_js = None
        self.create_subscription(JointState, "/joint_states",
                                 self._on_js, 10)
        # JTC 控制器 action server(broadcaster 之后才激活,
        # move_group 对它连接成功前发轨迹会 ABORT)
        from control_msgs.action import FollowJointTrajectory
        self.act_fjt = ActionClient(self, FollowJointTrajectory, JTC_ACTION)
        # TF 采样: 验证 TCP 是否贴 x=0.30 指尖平面(即 TCP x=0.18)且姿态固定
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listen = tf2_ros.TransformListener(self.tf_buf, self)
        self.tcp_log = []            # [(t, np.array(xyz), np.array(quat wxyz))]
        self.tcp_recording = False
        self._tcp_thread = threading.Thread(target=self._sample_tcp, daemon=True)
        self._tcp_thread.start()

    def _sample_tcp(self):
        """50Hz 采样 hand_tcp 位姿, 用于走线质量统计。"""
        err = 0
        while rclpy.ok():
            if self.tcp_recording:
                try:
                    tf = self.tf_buf.lookup_transform(
                        self.frame, LINK, rclpy.time.Time())
                    p = tf.transform.translation
                    q = tf.transform.rotation
                    self.tcp_log.append((
                        time.monotonic(),
                        np.array([p.x, p.y, p.z]),
                        np.array([q.w, q.x, q.y, q.z])))
                except Exception as e:   # TF 尚未就绪/超时, 静默重试
                    err += 1
                    if err == 1:
                        self.get_logger().warn("TF 采样首错: %s %s"
                                               % (type(e).__name__, e))
            time.sleep(0.02)

    def tcp_report(self):
        """统计指尖相对工作平面(x=0.30)偏差与姿态漂移。

        注意: MoveIt 官方 URDF 的 hand_tcp 在 link7 +0.186m(link→hand 0.1025
        + hand→tcp 0.0835), 而本工程路点/IK 语义的"指尖"= link7 +0.12×接近轴
        (stage3 约定)。故把实测 hand_tcp 沿接近轴回退 (0.186−0.12)=0.066m
        再对 x=0.30 平面求偏差。
        """
        if not self.tcp_log:
            self.get_logger().warn("无 TF 采样数据")
            return
        moveit_tcp_len = 0.1025 + 0.0835   # 官方 URDF link7→hand_tcp 长度
        back = moveit_tcp_len - TCP_OFFSET  # 实测点→指尖 沿轴回退量
        dx, dy, dz = [], [], []
        for _, p, _ in self.tcp_log:
            tip = p - back * APPROACH_AXIS
            dx.append(tip[0] - PLANE_X)
            dy.append(tip[1])
            dz.append(tip[2])
        dx = np.abs(np.array(dx))
        ang = []
        axis_ang = []
        for _, _, q in self.tcp_log:
            d = abs(float(np.dot(q, np.array(self.q_quat))))
            ang.append(math.degrees(2.0 * math.acos(min(d, 1.0))))
            # TCP z 轴(接近轴)方向: 四元数旋转 z 单位向量
            w, x, y, z = q
            zax = np.array([2 * (x * z + w * y),
                            2 * (y * z - w * x),
                            1 - 2 * (x * x + y * y)])
            axis_ang.append(math.degrees(math.acos(max(
                -1.0, min(1.0, float(zax @ APPROACH_AXIS))))))
        ang = np.array(ang)
        axis_ang = np.array(axis_ang)
        self.get_logger().info(
            "TCP 验证(已换算到指尖): 采样 %d | 平面偏差 mean=%.4fm max=%.4fm | "
            "y[%.3f,%.3f] z[%.3f,%.3f] | "
            "姿态漂移(含自旋) mean=%.2f° max=%.2f° | "
            "接近轴偏差 mean=%.2f° max=%.2f°"
            % (len(dx), np.mean(dx), np.max(dx),
               min(dy), max(dy), min(dz), max(dz),
               np.mean(ang), np.max(ang),
               np.mean(axis_ang), np.max(axis_ang)))

    def start_spin(self):
        """后台 executor: 转发 DDS 图事件 + 处理 future 回调（必需）。"""
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True)
        self._spin_thread.start()

    def _wait_future(self, fut, timeout=30.0):
        t0 = time.monotonic()
        while not fut.done():
            if time.monotonic() - t0 > timeout:
                self.get_logger().warn("等待 future 超时 (%.0fs)" % timeout)
                return None
            time.sleep(0.02)
        return fut.result()

    def _ec_log(self, ec):
        val = getattr(ec, "val", None)
        self.get_logger().error("MoveItErrorCodes: %s" % ec_name(val))

    # ------------------------------------------------------------------

    def connect(self):
        self.get_logger().info("等待 MoveIt 服务 ...")
        for c in (self.cli_plan, self.cli_cart,
                  self.cli_scene, self.cli_scene_get):
            c.wait_for_service()
        while not self.act_exec.wait_for_server(timeout_sec=0.5):
            self.get_logger().info("等待 /execute_trajectory action ...")
        # 服务就绪 ≠ 可以动: 等 /joint_states 首条消息(broadcaster 已激活),
        # 否则首段执行会 ABORT (MOTION_PLAN_INVALIDATED_BY_ENV_CHANGE)
        if not self._js_arrived.wait(timeout=30.0):
            self.get_logger().warn("30s 未收到 /joint_states, 继续执行(可能失败)")
        # 再等左臂控制器就绪: server 出现后还要给 move_group 的
        # controller handle 留 2s 建连(实测过早发送 = ABORT/env change)
        if not self.act_fjt.wait_for_server(timeout_sec=30.0):
            self.get_logger().warn("左臂控制器 action 未就绪, 继续执行(可能失败)")
        else:
            time.sleep(2.0)
        sreq = GetPlanningScene.Request()
        res = self._wait_future(self.cli_scene_get.call_async(sreq), 10.0)
        if res is not None and res.scene.robot_state.joint_state.header.frame_id:
            self.frame = res.scene.robot_state.joint_state.header.frame_id
        self.get_logger().info("规划坐标系: %s, 连接完成 ✔" % self.frame)

    # ------------------------------------------------------------------

    def publish_markers(self):
        arr = MarkerArray()
        wipe = Marker()
        wipe.action = Marker.DELETEALL
        arr.markers.append(wipe)
        wall = Marker()
        wall.header.frame_id = self.frame
        wall.ns = "scene"
        wall.id = 1
        wall.type = Marker.CUBE
        wall.pose.position.x = 2.015
        wall.pose.position.z = 0.9
        wall.scale.x, wall.scale.y, wall.scale.z = 0.03, 2.0, 1.8
        wall.color.r, wall.color.g, wall.color.b, wall.color.a = \
            0.85, 0.85, 0.9, 0.4
        arr.markers.append(wall)
        pl = Marker()
        pl.header.frame_id = self.frame
        pl.ns = "scene"
        pl.id = 2
        pl.type = Marker.CUBE
        pl.pose.position.x = PLANE_X
        pl.pose.position.z = 0.75
        pl.scale.x, pl.scale.y, pl.scale.z = 0.002, 1.0, 1.0
        pl.color.r, pl.color.g, pl.color.b, pl.color.a = 0.2, 0.5, 1.0, 0.2
        arr.markers.append(pl)
        line = Marker()
        line.header.frame_id = self.frame
        line.ns = "zigzag"
        line.id = 3
        line.type = Marker.LINE_STRIP
        line.scale.x = 0.004
        line.color.r, line.color.g, line.color.b, line.color.a = \
            1.0, 1.0, 0.1, 1.0
        for wp in self.waypoints:
            line.points.append(Point(x=wp[0], y=wp[1], z=wp[2]))
        arr.markers.append(line)
        self.pub_mk.publish(arr)

    def add_wall_to_scene(self):
        co = CollisionObject()
        co.header.frame_id = self.frame
        co.id = "scene_wall"
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.03, 2.0, 1.8]
        co.primitives.append(box)
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = 2.015, 0.0, 0.9
        pose.orientation.w = 1.0
        co.primitive_poses.append(pose)
        co.operation = CollisionObject.ADD
        req = ApplyPlanningScene.Request()
        req.scene.is_diff = True
        req.scene.world.collision_objects.append(co)
        self._wait_future(self.cli_scene.call_async(req), 10.0)
        self.get_logger().info("墙体碰撞体已加入 PlanningScene")

    # ------------------------------------------------------------------

    def _motion_plan_request(self, q, scaling=0.2):
        c = Constraints()
        for name, val in zip(self.joint_names, q):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(val)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        req = MotionPlanRequest()
        req.group_name = GROUP
        req.num_planning_attempts = 20
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = scaling
        req.max_acceleration_scaling_factor = scaling
        req.goal_constraints.append(c)
        return req

    def _on_js(self, m):
        # 必须按关节名取值, 不能 position[:7] 按位切片: broadcaster 可能
        # 把 finger_joint1 排在 joint1 之前(2026-09-09 仿真实测, 整体错位
        # 一格曾致起点校验全关节假偏差 ~40°)。消息缺任一臂关节则不更新。
        try:
            q = [float(m.position[m.name.index(n)]) for n in self.joint_names]
        except ValueError:
            return
        self._last_js = q
        self._js_arrived.set()

    def execute_trajectory(self, traj: RobotTrajectory) -> bool:
        """ExecuteTrajectory action 执行轨迹, 瞬态失败(如 env change)重试一次。
        重试仍失败时, 自动改用 JTC 直连执行同一轨迹(与 move_small 同通道,
        绕过 move_group 的执行监视; 规划/碰撞检查/验证仍由 MoveIt 完成)。"""
        for attempt in range(2):
            goal = ExecuteTrajectory.Goal()
            goal.trajectory = traj
            gh = self._wait_future(self.act_exec.send_goal_async(goal), 30.0)
            if gh is None or not gh.accepted:
                self.get_logger().error("执行 goal 被拒绝/超时")
                continue
            res = self._wait_future(gh.get_result_async(), 600.0)
            if res is None:
                self.get_logger().error("执行等待超时")
                return False
            if res.status == 4:
                return True
            ec = getattr(res.result, "error_code", None)
            val = getattr(ec, "val", None) if ec is not None else None
            self.get_logger().error("执行失败(尝试%d/2): status=%s %s"
                                    % (attempt + 1, res.status, ec_name(val)))
        self.get_logger().warn(
            "ExecuteTrajectory 两次失败, 改用 JTC 直连执行同一轨迹"
            "(规划仍由 MoveIt 完成, 仅绕过其执行监视)")
        return self.execute_via_jtc(traj)

    def _wait_settled(self, timeout=8.0):
        """等待关节停稳(相邻 0.4s 两次采样差 < 0.3°)。
        前一目标被取消后臂有跟踪滞后, 不停稳就发新目标会被 JTC
        以起始偏差拒收(2026-09-04 真机实测)。"""
        prev = None
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            cur = getattr(self, "_last_js", None)
            if cur is not None and prev is not None and all(
                    abs(b - a) < math.radians(0.3)
                    for a, b in zip(prev, cur)):
                return True
            if cur is not None:
                prev = list(cur)
            time.sleep(0.4)
        return False

    def execute_via_jtc(self, traj: RobotTrajectory) -> bool:
        """JTC 直连执行(与 move_small/go home 同通道)。
        前置一个当前状态点(0.5s), 从实际位形平滑接入轨迹。
        发送前等停稳; 被拒(起始偏差)时停稳后重试一次。"""
        from control_msgs.action import FollowJointTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint
        jt = traj.joint_trajectory
        if not jt.points:
            self.get_logger().error("无轨迹点, 无法 JTC 直连执行")
            return False
        if not self.act_fjt.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("JTC action 不可用, 无法直连执行")
            return False
        for attempt in (1, 2):
            self._wait_settled()
            goal = FollowJointTrajectory.Goal()
            goal.trajectory = jt
            q_now = getattr(self, "_last_js", None)
            if q_now is not None and len(q_now) >= 7:
                p0 = JointTrajectoryPoint()
                p0.positions = [float(v) for v in q_now[:7]]
                p0.time_from_start.nanosec = int(0.5e9)
                # 全部原始点整体 +1.0s, 保证与前置点严格递增(2026-09-04:
                # 只移位 [1:] 且原轨迹首点>0 时时间戳撞车, 被 JTC 拒收)
                prev_t = 0.5
                for pt in goal.trajectory.points:
                    t = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
                    t = max(t + 1.0, prev_t + 0.02)
                    pt.time_from_start.sec = int(t)
                    pt.time_from_start.nanosec = int(round(
                        (t - int(t)) * 1e9))
                    prev_t = t
                goal.trajectory.points = [p0] + list(goal.trajectory.points)
                self.get_logger().info("已前置当前位形点, 从实际位置平滑接入")
            gh = self._wait_future(self.act_fjt.send_goal_async(goal), 30.0)
            if gh is not None and gh.accepted:
                res = self._wait_future(gh.get_result_async(), 3600.0)
                ok = res is not None and res.status == 4
                self.get_logger().info(
                    "JTC 直连执行: %s"
                    % ("成功 ✔" if ok else
                       "失败(status=%s)" % (res.status if res else "无响应")))
                return ok
            self.get_logger().error(
                "JTC 直连目标被拒(起始偏差过大?)(尝试 %d/2)" % attempt)
            time.sleep(2.0)
        return False

    def move_to_joint_goal(self, q, scaling=0.2):
        """规划(服务) → 执行(action)。真机: scaling=1.0, 速度由
        joint_limits_real.yaml 的温和上限直接决定(约 0.1-0.2 rad/s)。"""
        if self.real:
            scaling = 1.0
        req = GetMotionPlan.Request()
        req.motion_plan_request = self._motion_plan_request(q, scaling)
        res = self._wait_future(self.cli_plan.call_async(req), 60.0)
        if res is None:
            self.get_logger().error("规划服务超时")
            return False
        ec = res.motion_plan_response.error_code
        if ec.val != 1:
            self.get_logger().error("规划失败: %s" % ec_name(ec.val))
            return False
        self.get_logger().info("规划成功 (%d 个轨迹点), 执行中 ..."
                               % len(res.motion_plan_response.trajectory
                                     .joint_trajectory.points))
        if self.dry:
            return True
        return self.execute_trajectory(res.motion_plan_response.trajectory)

    # ------------------------------------------------------------------
    # 逐关节方向验证(真机): 与之字形完全相同的生产链路
    # (MoveIt 规划 → ExecuteTrajectory → 失败自动 JTC 兜底)。
    # 每步在 RViz(/visualization_marker_array) 打文字标签,
    # 人工对照: RViz 模型动方向 = 实物动方向, 同向同量才算过。

    def _verify_text(self, text, r=1.0, g=0.9, b=0.1):
        m = Marker()
        m.header.frame_id = self.frame
        m.ns = "verify_joints"
        m.id = 10
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = 0.0
        m.pose.position.y = 0.0
        m.pose.position.z = 1.15
        m.pose.orientation.w = 1.0
        m.scale.z = 0.07
        m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 1.0
        m.text = text
        arr = MarkerArray()
        arr.markers.append(m)
        self.pub_verify.publish(arr)

    def verify_joints(self, deg=5.0):
        """逐关节 ±deg 度方向验证(关节4只做正方向: 负向贴 watchdog
        软限位触发线, 会误触发)。任何一步失败立即中止。"""
        d = math.radians(deg)
        self.get_logger().info("逐关节方向验证开始(deg=%g, 链路=MoveIt 规划→执行)"
                               % deg)
        self._verify_text("逐关节验证开始")
        plan = []
        for i in range(1, 8):
            for s in ([+1] if i == 4 else [+1, -1]):
                plan.append((i, s))
        for k, (i, s) in enumerate(plan, 1):
            self._verify_text(
                "(%d/%d) J%d %+g° 测试中 — 对照: RViz 模型方向 = 实物方向?"
                % (k, len(plan), i, s * deg))
            target = list(self.home_q)
            target[i - 1] += s * d
            if not self.move_to_joint_goal(target):
                self._verify_text("J%d %s 方向执行失败, 验证中止"
                                  % (i, "+" if s > 0 else "-"),
                                  r=1.0, g=0.2, b=0.2)
                self.get_logger().error("J%d %+g° 失败, 验证中止"
                                        % (i, s * deg))
                return False
            time.sleep(1.5)
            self._verify_text("(%d/%d) J%d 回家" % (k, len(plan), i))
            if not self.move_to_joint_goal(list(self.home_q)):
                self._verify_text("回家失败, 验证中止", r=1.0, g=0.2, b=0.2)
                self.get_logger().error("回家失败, 验证中止")
                return False
            time.sleep(1.0)
        self._verify_text("逐关节验证全部通过 ✔ 可以之字首跑",
                          r=0.3, g=1.0, b=0.3)
        self.get_logger().info("逐关节方向验证全部通过")
        return True

    # ------------------------------------------------------------------

    def _retiming(self, traj, seg_len=0.0, seg_lens=None):
        """笛卡尔路径重定时: 关节限速 + TCP 限速 (TCP_SPEED m/s)。

        seg_lens: 每步 TCP 位移数组(稠密模式); 标量 seg_len 为逐段均值。
        时间统一乘 1/speed_scale: 真机首跑用 <1 的系数整体放慢。
        """
        pts = traj.points
        if not pts:
            return traj
        t = 0.0
        for i in range(len(pts)):
            if i > 0:
                dt = 0.0
                for k in range(len(self.joint_names)):
                    v = abs(pts[i].positions[k] - pts[i - 1].positions[k])
                    dt = max(dt, v / VEL_LIMITS[k])
                if seg_lens is not None:
                    dt_tcp = seg_lens[i] / TCP_SPEED
                elif seg_len:
                    dt_tcp = (seg_len / max(len(pts) - 1, 1)) / TCP_SPEED
                else:
                    dt_tcp = 0.0
                t += max(dt, dt_tcp, 0.02) / self.speed_scale
            pts[i].time_from_start.sec = int(t)
            pts[i].time_from_start.nanosec = int((t - int(t)) * 1e9)
        return traj

    def return_to_zero_slow(self):
        """收尾: 缓慢回"家"位形并停住(0.1 倍速)。

        仿真模式家 = URDF 全零位形; 真机模式家 = 标定初始位形
        (真机全零需要 J1 转过约 120°, 首跑不安全)。
        """
        if self.real:
            self.get_logger().info("终点 → 缓慢回标定初始位形 (0.1 倍速) ...")
            ok = self.move_to_joint_goal(self.home_q, scaling=0.1)
            label = "标定初始位形"
        else:
            self.get_logger().info("终点 → 缓慢回到零位 (0.1 倍速) ...")
            ok = self.move_to_joint_goal([0.0] * 7, scaling=0.1)
            label = "零位"
        if ok:
            self.get_logger().info("已回%s, 流程结束 ✔" % label)
            return True
        self.get_logger().error("终点回%s失败" % label)
        return False

    def run_dense(self, dense_path):
        """稠密模式: 离线 IK 关节轨迹直接执行(TCP 直线度由 IK 容差保证)。"""
        from trajectory_msgs.msg import JointTrajectory as JT
        with open(dense_path, encoding="utf-8") as f:
            data = json.load(f)
        pts = data["points"]
        seg_lens = data["seg_lens"]
        # 0/1. 前置动作: 回零位(仿真)或标定初始位形(真机) → 到起点
        self.publish_markers()
        self.add_wall_to_scene()
        self.get_logger().info("稠密轨迹 %d 点, 前置: 回家位形 → 起点"
                               % len(pts))
        if not self.move_to_joint_goal(self.home_q if self.real
                                       else [0.0] * 7, scaling=0.15):
            return False
        if not self.move_to_joint_goal(pts[0], scaling=0.2):
            return False
        # 起点强制校验(2026-09-04): JTC goal_time=0 时不校验终点位置,
        # 曾带 J4 19° 偏差进入走线段 → 走线立即超差中止。不合格就地停止。
        # 等 8s: 让驱动积分器把稳态下垂收敛掉再校验。
        # J3 滚转在竖直位附近接近奇异, 积分收敛显著慢于其它关节。
        time.sleep(8.0)
        q_now = getattr(self, "_last_js", None)
        if q_now is None or len(q_now) < 7:
            self.get_logger().error("读不到 /joint_states, 无法校验起点")
            return False
        dev = [(i + 1, math.degrees(pts[0][i] - q_now[i])) for i in range(7)]
        worst = max(dev, key=lambda t: abs(t[1]))
        self.get_logger().info("起点校验: 最大偏差 %+.2f° (J%d)"
                               % (worst[1], worst[0]))
        if abs(worst[1]) > 1.5:   # >1.5° 才拦截; 1~1.5° 属稳态残差(含 J3 奇异)
            for jn, dv in dev:
                if abs(dv) > 0.3:
                    self.get_logger().error("  J%d 偏差 %+.2f°" % (jn, dv))
            self.get_logger().error(
                "起点偏差 >1°, 拒绝执行走线。先检查该关节为何不到位"
                "(增益/负载/机械打滑)再重试")
            return False
        traj = JT()
        traj.joint_names = self.joint_names
        for q in pts:
            from trajectory_msgs.msg import JointTrajectoryPoint
            jp = JointTrajectoryPoint()
            jp.positions = [float(v) for v in q]
            traj.points.append(jp)
        if data.get("pretimed") and data.get("times"):
            # smooth_track 实验轨: 时间轴已平滑(连续速度曲线),
            # 直接使用, 不再做梯形重定时(梯形台阶=抖动源)
            for i, tm in enumerate(data["times"]):
                traj.points[i].time_from_start.sec = int(tm)
                traj.points[i].time_from_start.nanosec = int(
                    round((tm - int(tm)) * 1e9))
            self.get_logger().info(
                "使用 smooth_track 预定时时间轴(总时长 %.0fs)"
                % data["times"][-1])
        else:
            self._retiming(traj, seg_lens=seg_lens)
        self.tcp_recording = True
        est = traj.points[-1].time_from_start.sec + \
            traj.points[-1].time_from_start.nanosec * 1e-9
        self.get_logger().info("执行稠密轨迹(预计 %.0fs)" % est)
        ok = self.execute_trajectory(
            RobotTrajectory(joint_trajectory=traj))
        self.tcp_recording = False
        self.tcp_report()
        if not ok:
            return False
        self.get_logger().info("之字形走线全部完成 ✔")
        return self.return_to_zero_slow()

    def _cartesian_request(self, wp_a, wp_b, avoid_collisions):
        req = GetCartesianPath.Request()
        req.header.frame_id = self.frame
        req.group_name = GROUP
        req.link_name = LINK
        req.max_step = 0.005
        req.jump_threshold = 0.0
        req.avoid_collisions = avoid_collisions
        for wp in (wp_a, wp_b):
            pose = Pose()
            pose.position.x = wp[0] - TCP_OFFSET * APPROACH_AXIS[0]
            pose.position.y = wp[1] - TCP_OFFSET * APPROACH_AXIS[1]
            pose.position.z = wp[2] - TCP_OFFSET * APPROACH_AXIS[2]
            pose.orientation.w, pose.orientation.x, \
                pose.orientation.y, pose.orientation.z = self.q_quat
            req.waypoints.append(pose)
        res = self._wait_future(self.cli_cart.call_async(req), 60.0)
        frac = res.fraction if res is not None else 0.0
        if res is None or frac < 0.999:
            self.get_logger().warn("笛卡尔分数不足 %.3f (avoid_collisions=%s)"
                                   % (frac, avoid_collisions))
            return None
        return res

    def move_line_cartesian(self, wp_a, wp_b, q_seed=None):
        """一段之字线: hand_tcp 笛卡尔直线（位置=指尖点−TCP_OFFSET×轴）。

        两次尝试: avoid_collisions=True → False(纯 IK 跟踪, fake 硬件下
        自碰撞检查是 KDL 提前停下的常见原因), 均失败返回 None 交由回退。
        """
        for avoid in (True, False):
            res = self._cartesian_request(wp_a, wp_b, avoid)
            if res is not None:
                traj = self._retiming(
                    res.solution.joint_trajectory,
                    seg_len=float(np.linalg.norm(wp_b - wp_a)))
                if self.dry:
                    self.get_logger().info("[dry-run] 段规划成功(%d 点), 跳过执行"
                                           % len(traj.points))
                    return True
                if self.execute_trajectory(RobotTrajectory(
                        joint_trajectory=traj)):
                    return True
                return False  # 规划成功但执行失败, 无需再试规划
        return False

    # ------------------------------------------------------------------

    def run(self):
        self.publish_markers()
        self.add_wall_to_scene()
        n = len(self.waypoints)
        self.get_logger().info("之字形路点 %d 个, 开始执行" % n)
        # 0. 先回零位（全零 = 手臂自然下垂, 0.15 倍速）
        self.get_logger().info("前往零位 (全零位形) ...")
        if not self.move_to_joint_goal([0.0] * 7, scaling=0.15):
            self.get_logger().error("回到零位失败")
            return False
        # 1. 起点: 关节目标（workspace 内置解, 0.2 倍速）
        self.get_logger().info("前往之字形起点 ...")
        if not self.move_to_joint_goal(self.joint_sols[0], scaling=0.2):
            self.get_logger().error("移动到起点失败")
            return False
        self.get_logger().info("已到起点, 之字形分段执行 (TCP 5cm/s)")
        self.tcp_recording = True
        nseg = (n - 1) // 2
        for i in range(nseg):
            a, b = self.waypoints[2 * i], self.waypoints[2 * i + 1]
            ok = self.move_line_cartesian(a, b)
            if not ok:
                self.get_logger().warn("段 %d 笛卡尔失败, 回退关节目标" % i)
                ok = self.move_to_joint_goal(self.joint_sols[2 * i + 1],
                                             scaling=0.2)
            self.get_logger().info("段 %d/%d %s" % (i + 1, nseg,
                                                    "完成" if ok else "失败"))
            if not ok:
                return False
        self.tcp_recording = False
        self.tcp_report()
        self.get_logger().info("之字形走线全部完成 ✔")
        return self.return_to_zero_slow()
        return True


def main():
    global LINK, GROUP, JOINT_NAMES, JTC_ACTION
    _HERE = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="left", choices=["left", "right"],
                    help="目标臂(默认 left)。决定关节名/JTC action/TF link"
                         " 及各路径参数的默认值")
    ap.add_argument("--workspace", default=None,
                    help="工作区 json(默认 left=workspace.json / "
                         "right=workspace_right.json)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--real", action="store_true",
                    help="真机模式: 家位形=标定初始位形(非 URDF 全零)")
    ap.add_argument("--speed-scale", type=float, default=1.0,
                    help="轨迹整体放慢系数(真机首跑建议 0.5)")
    ap.add_argument("--safety-yaml", default=None,
                    help="real_safety.yaml 路径(--real 时读取 home 位形; "
                         "默认 real/<side>/config/real_safety.yaml)")
    ap.add_argument("--dense", default=None,
                    help="稠密 IK 轨迹 json (默认 left=dense_zigzag.json / "
                         "right=dense_zigzag_right.json); 空串禁用")
    ap.add_argument("--verify-joints", action="store_true",
                    help="逐关节方向验证(±5°, J4 只正向), 全部通过后再跑之字")
    ap.add_argument("--verify-deg", type=float, default=5.0,
                    help="逐关节验证的幅度(度)")
    args = ap.parse_args()

    # ---- 按臂覆盖臂相关常量与默认路径 ----
    side = args.side
    LINK = "openarm_%s_hand_tcp" % side
    GROUP = "%s_arm" % side
    JOINT_NAMES = ["openarm_%s_joint%d" % (side, i) for i in range(1, 8)]
    JTC_ACTION = "/%s_joint_trajectory_controller/follow_joint_trajectory" % side
    suffix = "" if side == "left" else "_right"
    if args.workspace is None:
        args.workspace = os.path.join(_HERE, "..", "isaaclab_sim",
                                      "workspace%s.json" % suffix)
    if args.safety_yaml is None:
        # 2026-09-08 双臂重构后 real/config 不存在, 按臂取
        args.safety_yaml = os.path.join(
            _HERE, "..", "..", "real", side, "config", "real_safety.yaml")
    if args.dense is None:
        args.dense = os.path.join(_HERE, "dense_zigzag%s.json" % suffix)

    rclpy.init()
    node = ZigzagMover(args.workspace, args.dry_run, real=args.real,
                       speed_scale=args.speed_scale,
                       safety_yaml=os.path.abspath(args.safety_yaml))
    node.start_spin()
    try:
        node.connect()
        if args.verify_joints:
            node.verify_joints(args.verify_deg)
        elif args.dense and os.path.isfile(args.dense):
            node.run_dense(args.dense)
        else:
            node.run()
    except KeyboardInterrupt:
        pass
    finally:
        # 先停后台 executor 再销毁节点, 否则退出时打印
        # "terminate called without an active exception"(无害但难看)
        node._executor.shutdown()
        node._spin_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
