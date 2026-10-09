#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 RViz 显示之字规划路径(world 系), 供真机摆放位置人工核对。
零运动: 只发 visualization_msgs, 不接触任何控制器。

用法: source /opt/ros/jazzy/setup.bash
      /usr/bin/python3 show_zigzag_marker.py
RViz 已有 ZigzagMarkers 显示项(topic /zigzag_markers), 打开即见。
"""

import json
import os

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

_HERE = os.path.dirname(os.path.abspath(__file__))
_WS = os.path.abspath(os.path.join(_HERE, "..", "isaaclab_sim",
                                   "workspace.json"))


def main():
    rclpy.init()
    node = Node("zigzag_marker_pub")
    pub = node.create_publisher(MarkerArray, "/visualization_marker_array", 10)
    ws = json.load(open(_WS, encoding="utf-8"))
    wps = ws["zigzag_waypoints"]

    def mk(mid, mtype, pose=(0.0,), pts=None):
        m = Marker()
        m.header.frame_id = "world"
        m.ns = "zigzag"
        m.id = mid
        m.type = mtype
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.pose.position.x = pose[0]
        m.pose.position.y = pose[1] if len(pose) > 1 else 0.0
        m.pose.position.z = pose[2] if len(pose) > 2 else 0.0
        if pts:
            m.points = pts
        m.color.a = 1.0
        return m

    arr = MarkerArray()
    # 1) 之字折线(黄)
    line = mk(1, Marker.LINE_STRIP)
    line.scale.x = 0.008
    line.color.r, line.color.g, line.color.b = 1.0, 1.0, 0.0
    from geometry_msgs.msg import Point
    for w in wps:
        p = Point()
        p.x, p.y, p.z = w
        line.points.append(p)
    arr.markers.append(line)
    # 2) 工作面板(半透明蓝)
    ys = [w[1] for w in wps]
    zs = [w[2] for w in wps]
    panel = mk(2, Marker.CUBE,
               pose=(wps[0][0], (min(ys) + max(ys)) / 2,
                     (min(zs) + max(zs)) / 2))
    panel.scale.x = 0.005
    panel.scale.y = (max(ys) - min(ys)) + 0.06
    panel.scale.z = (max(zs) - min(zs)) + 0.06
    panel.color.r, panel.color.g, panel.color.b = 0.2, 0.4, 1.0
    panel.color.a = 0.25
    arr.markers.append(panel)
    # 3) 起点/终点文字
    for mid, idx, txt in ((3, 0, "起点"), (4, -1, "终点")):
        t = mk(mid, Marker.TEXT_VIEW_FACING, pose=tuple(wps[idx]))
        t.text = txt
        t.scale.z = 0.05
        t.color.r, t.color.g, t.color.b = 1.0, 0.3, 0.3
        t.pose.position.z += 0.05
        arr.markers.append(t)

    import time
    n = 0
    while rclpy.ok():
        pub.publish(arr)
        n += 1
        if n % 20 == 0:
            node.get_logger().info("标记发布中(2Hz, Ctrl+C 退出)")
        time.sleep(0.5)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
