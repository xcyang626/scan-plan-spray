#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/joint_states 黑匣子: 全速订阅并写 CSV, 供事后诊断(如 J1 跟踪分析)。
列: t_recv, sec, nanosec, q1..q7(rad), v1..v7(rad/s), e1..e7(力矩),
    c1..c7(注入通道最新位置指令, 可选), cmd_age_s(指令龄)
无消息时静默等待, 启动顺序无关。Ctrl+C 结束。
2026-09-18: 增加注入通道指令记录 —— 归因"臂动了但不知道谁发的指令"
类事故(注入通道任何发布源的行为都会被记录)。
2026-09-22: 落盘改为当前目录 + 时间戳文件名。旧版固定 /tmp/js_blackbox.csv
且 "w" 截断 —— 重启即失/逐次覆盖, 0921 全程首跑的黑匣子因此丢失(ILC
迭代数据只能靠 rl_exec CSV 顶上)。留档节奏不变: 重要试验停录后复制走。"""

import sys
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

# 落盘到当前目录(启动处, runbook 为 real/common/)+时间戳: 不覆盖旧记录,
# 重启不丢。第一个参数仍可覆盖注入指令话题(左臂/MPC 场景)。
OUT = time.strftime("js_blackbox_%Y%m%d_%H%M%S.csv")
# 注入指令通道(默认右臂 RL; 左臂或 MPC 场景用第一个参数覆盖)
CMD_TOPIC = sys.argv[1] if len(sys.argv) > 1 else "/right_rl_position_commands"


def main():
    rclpy.init()
    node = Node("js_blackbox")
    # 行缓冲(line buffering): 每条消息立即刷盘, Ctrl+C 中断不丢已记录行
    fh = open(OUT, "w", buffering=1)

    cmd = {"q": [float("nan")] * 7, "t": None}

    def on_cmd(m):
        cmd["q"] = list(m.data[:7])
        cmd["t"] = time.monotonic()

    def on_js(m):
        # 收到时刻用单调钟: 不受系统对时影响, 适合分析到达节奏与断流
        t = time.monotonic()
        age = ("%.3f" % (t - cmd["t"])) if cmd["t"] is not None else "-1"
        fh.write("%.3f,%d,%d,%s,%s,%s,%s,%s\n" % (
            t, m.header.stamp.sec, m.header.stamp.nanosec,
            ",".join("%.5f" % v for v in m.position[:7]),
            ",".join("%.4f" % v for v in m.velocity[:7]),
            ",".join("%.3f" % v for v in m.effort[:7]),
            ",".join("%.5f" % v for v in cmd["q"][:7]), age))

    # 传感器 QoS(BEST_EFFORT, 浅队列): 与发布方(joint_state_broadcaster)兼容,
    # 且晚启动时不会因深度 0 的队列丢掉首帧
    node.create_subscription(JointState, "/joint_states", on_js,
                             qos_profile_sensor_data)
    from std_msgs.msg import Float64MultiArray
    node.create_subscription(Float64MultiArray, CMD_TOPIC, on_cmd, 10)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    print("js_blackbox: 记录到 %s, 等待 /joint_states + %s ..." % (OUT, CMD_TOPIC))
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        fh.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
