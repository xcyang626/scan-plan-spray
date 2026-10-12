# -*- coding: utf-8 -*-
"""ROS topic 格式输出桥 (模块化接口, 用户定稿: 输出以 ROS topic 格式; 执行后续接入).

双后端, 接口一致:
  1. rospy 可用  -> 实时发布到 ROS master (latch);
  2. 不可用      -> 落 ROS topic 记录文件 output/ros_topics/<stamp>/ *.jsonl
     (每行 {topic, type, ts, layout, data}; 附 manifest.md 给 ROS 侧订阅者对照).

话题契约 (robot 名作前缀段, 如 /g1_spray/...):
  /<robot>_spray/wall_path        Float32MultiArray  (N,6): seq,u,h,x,y,z (世界系, m)
  /<robot>_spray/robot_path       Float32MultiArray  (N,9): 腕心xyz, 枪轴dxyz, tilt_deg
  /<robot>_spray/trajectory_info  String             (JSON: trajectory_meta 摘要)
  /<robot>_spray/motion_qpos      Float32MultiArray  (T,36): 全身关节@30Hz 扁平
  /<robot>_spray/motion_meta      String             (JSON: motion_meta)
  预留(执行模块接入后启用, 本期不发):
  /<robot>_spray/motion/stream    Float32MultiArray  (36,) @30Hz 逐帧流
  /<robot>_spray/exec/cmd         String             执行指令(暂不实现)

用法:
  站点调用:  from ros_bridge import emit_run;  emit_run(robot, traj_dir, motion_dir, out_base)
  命令行:    python ros_bridge.py --robot g1 [--ver v5_stand_zigzag]  # 用当前 output 导出
"""
import os, json, time, argparse

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TOPICS = {
    "wall_path": ("std_msgs/Float32MultiArray", None),
    "robot_path": ("std_msgs/Float32MultiArray", None),
    "trajectory_info": ("std_msgs/String", None),
    "motion_qpos": ("std_msgs/Float32MultiArray", None),
    "motion_meta": ("std_msgs/String", None),
}


def _rospy():
    try:
        import rospy            # noqa
        from std_msgs.msg import Float32MultiArray, String
        return rospy, Float32MultiArray, String
    except Exception:
        return None, None, None


def _rec_dir(out_base):
    import datetime
    d = os.path.join(out_base, "ros_topics",
                     datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(d, exist_ok=True)
    return d


def _fma(arr):
    a = np_asarr(arr)
    return {"layout": {"dim": list(a.shape)}, "data": a.reshape(-1).tolist()}


def np_asarr(a):
    import numpy as np
    return np.asarray(a, dtype=np.float32)


def emit_run(robot, traj_dir=None, motion_dir=None, out_base=None, publish_live=None):
    """轨迹+动作 -> ROS topics. rospy 可用则实发, 否则写 topic 记录文件.
    返回 dict(mode, files|topics)."""
    import numpy as np
    out_base = out_base or os.path.join(BASE, "output")
    traj_dir = traj_dir or os.path.join(out_base)
    motion_dir = motion_dir or os.path.join(out_base, "ufo_motion",
                                            _latest_motion_ver(out_base))
    rospy, FMA, STR = _rospy()
    live = publish_live if publish_live is not None else (rospy is not None)
    msgs = _collect(robot, traj_dir, motion_dir)

    if live and rospy is not None:
        rospy.init_node(f"{robot}_spray_station", anonymous=True, disable_signals=True)
        pubs = {}
        for name, (typ, _) in TOPICS.items():
            if name in msgs:
                pubs[name] = rospy.Publisher(f"/{robot}_spray/{name}",
                                             FMA if "Float32" in typ else STR,
                                             queue_size=1, latch=True)
        for name, body in msgs.items():
            if "Float32" in TOPICS[name][0]:
                m = FMA(); m.layout.dim = [
                    type("d", (), {"label": "", "size": s, "stride": 0})()
                    for s in body["layout"]["dim"]]
                m.layout.data_offset = 0; m.data = body["data"]
            else:
                m = STR(); m.data = json.dumps(body, ensure_ascii=False)
            pubs[name].publish(m)
            time.sleep(0.05)
        return {"mode": "rospy 实发", "topics": [f"/{robot}_spray/{n}" for n in msgs]}

    d = _rec_dir(out_base)
    files = {}
    for name, body in msgs.items():
        p = os.path.join(d, f"{name}.jsonl")
        with open(p, "w", encoding="utf8") as f:
            f.write(json.dumps({"topic": f"/{robot}_spray/{name}",
                                "type": TOPICS[name][0], "ts": time.time(),
                                **body}, ensure_ascii=False) + "\n")
        files[name] = p
    _manifest(d, robot, files)
    return {"mode": "topic记录文件(rospy不可用)", "dir": d, "files": files}


def _latest_motion_ver(out_base):
    vm = os.path.join(out_base, "ufo_motion")
    cands = [v for v in os.listdir(vm) if os.path.isdir(os.path.join(vm, v))] \
        if os.path.isdir(vm) else []
    return max(cands) if cands else "v5_stand_zigzag"


def _collect(robot, traj_dir, motion_dir):
    import numpy as np
    msgs = {}
    wp = os.path.join(traj_dir, "wall_path.csv")
    rp = os.path.join(traj_dir, "robot_path.csv")
    if os.path.exists(wp):
        import csv
        w = list(csv.DictReader(open(wp, encoding="utf8")))
        msgs["wall_path"] = _fma(np.array([[float(r["seq"]), float(r["u_m"]),
                                            float(r["h_m"]), float(r["world_x_m"]),
                                            float(r["world_y_m"]), float(r["world_z_m"])]
                                           for r in w]))
    if os.path.exists(rp):
        import csv
        w = list(csv.DictReader(open(rp, encoding="utf8")))
        msgs["robot_path"] = _fma(np.array([[float(r["wrist_x_m"]), float(r["wrist_y_m"]),
                                             float(r["wrist_z_m"]), float(r["axis_dx"]),
                                             float(r["axis_dy"]), float(r["axis_dz"]),
                                             float(r["tilt_deg"])] for r in w]))
    tj = os.path.join(traj_dir, "trajectory_meta.json")
    if os.path.exists(tj):
        msgs["trajectory_info"] = json.load(open(tj, encoding="utf8"))
    npz_p = os.path.join(motion_dir, "wall_spray_motion.npz")
    if os.path.exists(npz_p):
        q = np.load(npz_p)["qpos"].astype(np.float32)
        msgs["motion_qpos"] = _fma(q)
    mj = os.path.join(motion_dir, "motion_meta.json")
    if os.path.exists(mj):
        msgs["motion_meta"] = json.load(open(mj, encoding="utf8"))
    return msgs


def _manifest(d, robot, files):
    lines = [f"# ROS topic 记录 ({robot})", "",
             "rospy 不可用时落盘的等价格式; 每行一个消息:",
             '{"topic": "...", "type": "std_msgs/...", "ts": ..., "layout": {...}, "data": [...]}',
             "", "| 文件 | 话题 | 类型 |", "|---|---|---|"]
    for name, p in files.items():
        lines.append(f"| {os.path.basename(p)} | /{robot}_spray/{name} | {TOPICS[name][0]} |")
    lines += ["", "## 执行模块接入(预留)", "",
              "- `/motion/stream` 30Hz 逐帧 Float32MultiArray(36) — 接入后由执行模块订阅",
              "- `/exec/cmd` 执行指令 — 本期不实现",
              "- 实发: 在有 rospy 的环境运行 `python ros_bridge.py --robot " + robot + "`"]
    open(os.path.join(d, "manifest.md"), "w", encoding="utf8").write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="g1")
    ap.add_argument("--traj", default=os.path.join(BASE, "output"))
    ap.add_argument("--motion", default=None)
    ap.add_argument("--out", default=None, help="输出根(记录目录/结果json的父目录)")
    a = ap.parse_args()
    r = emit_run(a.robot, a.traj, a.motion, out_base=a.out)
    if a.out:      # 站点子进程模式: 结果落盘回读(主进程系统python无numpy, 不能import本模块)
        try:
            json.dump(r, open(os.path.join(a.out, "ros_emit_result.json"), "w",
                              encoding="utf8"), ensure_ascii=False, indent=1)
        except Exception:
            pass
    print(json.dumps(r, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
