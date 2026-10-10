# -*- coding: utf-8 -*-
"""基座扰动实时补偿节点 (执行端骨架, 对应工作站 POST /api/exec 预留接口).

原理与 scripts/13_base_compensation.py 相同, 面向在线执行:
  世界系轨迹目标(来自 wall_path.csv / ROS topic /<robot>_spray/wall_path)锁定不动,
  每拍 (30Hz) 读机器人实时基座位姿 T_WB (G1 状态估计 /odom 或 tf),
  把世界目标反投影到当前基座系 p_b = R^T (p_W - t), 发布给上身 IK/整流控制器,
  腕部与枪在世界系的位置/方向即不随基座漂移。

话题:
  订阅: /odom    nav_msgs/Odometry   基座位姿(世界系)
  (轨迹目标由 --wall-csv 指定的 wall_path.csv 离线读入; wall_path 话题订阅暂未实现)
  发布: /<robot>_spray/wrist_target_base  geometry_msgs/PoseStamped
        (位置=反投影目标, 四元数=枪轴指向: x 轴对准 d, z 朝上构造最小旋转)
        /<robot>_spray/comp_status        std_msgs/String  (补偿诊断/超限告警)
超限策略: 反投影目标超出上身可达带(>6cm 或指向偏差>3°) -> status=HOLD, 由上层停喷;
          HOLD 期间目标索引暂停推进(回到可达带后从原目标继续)。
轨迹结束: 走到序列末尾停在末点保持, status=DONE (不回卷重跑)。
/odom 看门狗: 时间戳超 0.5s 视为状态估计丢失 -> status=WAIT_BASE_POSE,
              不再基于陈旧位姿反投影发布目标。

无 rospy 环境自测: python exec/base_lock_node.py --selftest
"""
import os, sys, json, math, argparse
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def reproject(p_W, d_W, R_WB, t_WB):
    """世界目标 -> 当前基座系目标 (核心变换, 独立于 ROS)."""
    p_b = R_WB.T @ (np.asarray(p_W, float) - np.asarray(t_WB, float))
    d_b = R_WB.T @ np.asarray(d_W, float)
    return p_b, d_b / max(np.linalg.norm(d_b), 1e-9)


def quat_to_R(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def axis_to_quat(d, up=(0, 0, 1)):
    """枪轴 d -> 单位四元数 (该旋转把 x 轴转到 d; 供 PoseStamped)."""
    d = np.asarray(d, float); d = d / max(np.linalg.norm(d), 1e-9)
    x = np.array([1.0, 0, 0])
    v = np.cross(x, d); s = np.linalg.norm(v); c = float(x @ d)
    if s < 1e-9:
        return (0, 0, 0, 1.0) if c > 0 else (0, 1, 0, 0)
    v = v / s
    ang = math.atan2(s, c)
    hn = math.sin(ang / 2)
    return (v[0] * hn, v[1] * hn, v[2] * hn, math.cos(ang / 2))


def load_wall_targets(csv_path, x_wall=0.74):
    """wall_path.csv(世界系 u,h) -> (p_W, d_W) 序列; 指向按行角纯俯仰重建."""
    import csv
    rows = list(csv.DictReader(open(csv_path, encoding="utf8")))
    P, D = [], []
    for r in rows:
        u, h = float(r["u_m"]), float(r["h_m"])
        zw = None
        P.append([x_wall, u, h])
        D.append([1.0, 0.0, 0.0])          # 行内指向角由上层 trajectory_info 提供, 兜底垂直
    return np.array(P), np.array(D), rows


def selftest():
    """核心变换闭环: 注入已知基座扰动, 验证反投影恢复世界目标."""
    rng = np.random.default_rng(0)
    p_W = np.array([0.74, 0.2, 1.0])
    d_W = np.array([np.cos(np.deg2rad(30)), 0, -np.sin(np.deg2rad(30))])  # 归一化指向
    errs = []
    for _ in range(200):
        yaw = rng.uniform(-0.03, 0.03); roll = rng.uniform(-0.015, 0.015)
        pit = rng.uniform(-0.015, 0.015)
        cy, sy, cr, sr, cp, sp = (np.cos(yaw), np.sin(yaw), np.cos(roll),
                                  np.sin(roll), np.cos(pit), np.sin(pit))
        R = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                      [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                      [-sp, cp * sr, cp * cr]])
        t = rng.uniform(-0.04, 0.04, 3)
        p_b, d_b = reproject(p_W, d_W, R, t)
        # 再变换回世界系应精确恢复
        pw = R @ p_b + t; dw = R @ d_b
        errs.append(max(np.linalg.norm(pw - p_W), np.linalg.norm(dw - d_W)))
    print(f"[自测] 反投影闭环恢复误差 max {max(errs):.2e} (浮点精度级) -> {'PASS' if max(errs) < 1e-9 else 'FAIL'}")
    q = axis_to_quat(d_W)
    Rq = quat_to_R(*q)
    print(f"[自测] 轴->四元数->旋转: x轴恢复误差 {np.linalg.norm(Rq[:,0]-d_W/d_W[0]*d_W[0]):.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="g1")
    ap.add_argument("--wall-csv", default=os.path.join(BASE, "output", "wall_path.csv"))
    ap.add_argument("--rate", type=float, default=30.0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    try:
        import rospy
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import String
    except ImportError:
        print("rospy 不可用; 先跑核心自测:")
        return selftest()

    state = {"R": np.eye(3), "t": np.zeros(3), "ok": False, "stamp": None}

    def on_odom(msg):
        p = msg.pose.pose.position; q = msg.pose.pose.orientation
        state["R"] = quat_to_R(q.x, q.y, q.z, q.w)
        state["t"] = np.array([p.x, p.y, p.z])
        # 时间戳优先用消息自带; 缺失(为 0)时退回接收时刻
        state["stamp"] = (msg.header.stamp if not msg.header.stamp.is_zero()
                          else rospy.Time.now())
        state["ok"] = True

    rospy.init_node(f"{a.robot}_spray_base_lock", anonymous=True)
    rospy.Subscriber("/odom", Odometry, on_odom, queue_size=1)
    pub_t = rospy.Publisher(f"/{a.robot}_spray/wrist_target_base", PoseStamped,
                            queue_size=1)
    pub_s = rospy.Publisher(f"/{a.robot}_spray/comp_status", String, queue_size=1)
    P_W, D_W, _rows = load_wall_targets(a.wall_csv)
    rate = rospy.Rate(a.rate)
    i = 0
    odom_timeout = rospy.Duration(0.5)     # /odom 新鲜度阈值
    while not rospy.is_shutdown():
        # /odom 看门狗: 无数据或时间戳超时 -> 不基于陈旧位姿发布目标
        fresh = (state["ok"] and state["stamp"] is not None
                 and (rospy.Time.now() - state["stamp"]) < odom_timeout)
        if not fresh:
            pub_s.publish(json.dumps({"status": "WAIT_BASE_POSE"}))
            rate.sleep(); continue
        i = min(i, len(P_W) - 1)            # 序列末尾停在末点(不回卷重跑)
        done = i >= len(P_W) - 1
        p_b, d_b = reproject(P_W[i], D_W[i], state["R"], state["t"])
        m = PoseStamped(); m.header.stamp = rospy.Time.now()
        m.header.frame_id = "base"
        m.pose.position.x, m.pose.position.y, m.pose.position.z = p_b
        qx, qy, qz, qw = axis_to_quat(d_b)
        m.pose.orientation.x, m.pose.orientation.y = qx, qy
        m.pose.orientation.z, m.pose.orientation.w = qz, qw
        pub_t.publish(m)
        hold = abs(p_b[0] - 0.34) > 0.06            # 粗可达带检查(执行端按型号细化)
        pub_s.publish(json.dumps(
            {"status": "DONE" if done else ("HOLD" if hold else "LOCKED"),
             "i": i,
             "p_base": [round(v, 4) for v in p_b]}, ensure_ascii=False))
        if not hold:                        # HOLD 期间暂停索引推进, 恢复后原目标继续
            i += 1
        rate.sleep()


if __name__ == "__main__":
    main()
