#!/bin/bash
# 生成可分发的一体化工作站精简包: release/g1_wall_spray_station/ + zip
# 包含: 全部脚本/四档工作区与NN模型/机器人模型(MJCF+URDF)/静态资源/示例点云/详细README
# 不含: 运行产物(runs/, output 产物)、部署历史、开发遗留
set -e
cd "$(dirname "$0")/.."
BASE="$(pwd)"
REL="$BASE/release/g1_wall_spray_station"
MJCF_SRC=/home/yang/.cache/huggingface/hub/models--elijahgalahad--g1_xmls/snapshots/1c1d082f299126b9272e81c04121ff347c97fa8f
URDF_SRC="/home/yang/桌面/遥操/g1_model"

echo "==> 清理并建立目录"
rm -rf "$REL"; mkdir -p "$REL"/{scripts/exec,station/nn_ik,station/static,uploads,output,models}

echo "==> 脚本"
for f in scripts/*.py; do
  case "$(basename $f)" in 00_*|00b_*|01_*) ;;   # 上游舱室点云提取, 依赖本机原始数据, 不入包
    *) cp "$f" "$REL/scripts/" ;; esac
done
cp scripts/run_pipeline.sh "$REL/scripts/"
cp scripts/exec/base_lock_node.py scripts/exec/groot_spray_runner.py \
   scripts/exec/estop.py "$REL/scripts/exec/"
# 下列文件存在才拷(部分为可选组件, 缺失不应让 set -e 中断打包)
cp scripts/exec/groot_daemon.py scripts/exec/groot_bridge.py \
   scripts/exec/compare_q.py "$REL/scripts/exec/" 2>/dev/null \
  || echo "  (提示: groot_daemon/groot_bridge/compare_q 未全部打包)"   # 常驻守护进程/桥/关节对照(实机执行链)
cp scripts/exec/zmq_motion_publisher.py "$REL/scripts/exec/" 2>/dev/null \
  || true   # 该发布器已由 groot_daemon.py 取代, 不存在时跳过

echo "==> 静态资源与模型档位"
cp station/static/* "$REL/station/static/"
cp station/nn_ik/g1_rp*_nn_ik.npz station/nn_ik/g1_rp*_stats.json "$REL/station/nn_ik/"
cp station/nn_ik/g1_nn_ik.npz "$REL/station/nn_ik/"          # 命令行流水线 fallback

echo "==> 机器人模型 (打包内置, 免外部依赖)"
mkdir -p "$REL/models/g1_mjcf/meshes" "$REL/models/g1_urdf"
cp "$MJCF_SRC/g1-mode_13_15.xml" "$REL/models/g1_mjcf/"
cp "$MJCF_SRC/meshes/"*.STL "$REL/models/g1_mjcf/meshes/" 2>/dev/null || cp "$MJCF_SRC/meshes/"* "$REL/models/g1_mjcf/meshes/"
cp "$URDF_SRC/g1_29dof_nohand.urdf" "$REL/models/g1_urdf/"
cp -r "$URDF_SRC/meshes" "$REL/models/g1_urdf/meshes"

echo "==> 静态输入与工作区缓存(四档)"
cp output/wall_section.npz output/wall_section_meta.json "$REL/output/"
cp -r output/ws_cache "$REL/output/ws_cache"

echo "==> 示例点云"
cp uploads/test_wall_simple.ply uploads/test_wall_obstacles.ply uploads/cabin_test.ply "$REL/uploads/" 2>/dev/null || true

echo "==> registry(相对路径) + env.json(示例)"
python3 - <<'PY'
import json, os
rel = "release/g1_wall_spray_station"
reg = json.load(open("station/robot_registry.json", encoding="utf8"))
reg["g1"]["mjcf"] = "models/g1_mjcf/g1-mode_13_15.xml"
reg["g1"]["model_dir"] = "models/g1_urdf"
reg["g1"]["urdf_file"] = "g1_29dof_nohand.urdf"
json.dump(reg, open(os.path.join(rel, "station/robot_registry.json"), "w",
                    encoding="utf8"), ensure_ascii=False, indent=2)
json.dump({"_说明": "解释器与机器人执行路径, 按目标机器修改; 亦可用环境变量覆盖",
           "isaac_py": "/home/yang/miniconda3/envs/isaaclab/bin/python",
           "mujoco_py": "/home/yang/miniconda3/envs/mujoco/bin/python",
           "groot_root": "/home/yang/桌面/轨迹规划/GR00T-WholeBodyControl",
           "groot_python": "/home/yang/桌面/轨迹规划/GR00T-WholeBodyControl/.venv_sim_cpu/bin/python",
           "display": ":0",
           "robot_mode": "sim",
           "robot_iface": "enp131s0",
           "zmq_port": 5556,
           "zmq_host": "127.0.0.1",
           "daemon_port": 5558},
          open(os.path.join(rel, "station/env.json"), "w", encoding="utf8"),
          ensure_ascii=False, indent=1)
PY

echo "==> 清洗 stats 中的本机绝对路径"
python3 - <<'PY2'
import json, glob, os
for f in glob.glob("release/g1_wall_spray_station/station/nn_ik/*_stats.json"):
    d = json.load(open(f, encoding="utf8"))
    if isinstance(d.get("model"), str):
        d["model"] = os.path.basename(d["model"])
    json.dump(d, open(f, "w", encoding="utf8"), ensure_ascii=False, indent=1)
PY2

echo "==> README(随包详细文档)"
cp README_RELEASE.md "$REL/README.md" 2>/dev/null \
  || echo "  (提示: README_RELEASE.md 不存在, 跳过随包 README)"

echo "==> 启动脚本"
cat > "$REL/启动工作站.sh" <<'EOS'
#!/bin/bash
# 一键启动喷涂轨迹工作站 (启动服务 + 自动打开浏览器)
cd "$(dirname "$0")"
mkdir -p output runs
if curl -s -m 1 http://127.0.0.1:8090/api/status >/dev/null 2>&1; then
  echo "工作站已在运行"
else
  PY=$(python3 -c "import json;print(json.load(open('station/env.json'))['mujoco_py'])")
  nohup "$PY" scripts/10_station_app.py --port 8090 > output/station.log 2>&1 &
  sleep 2
fi
(xdg-open http://127.0.0.1:8090/ >/dev/null 2>&1 || python3 -m webbrowser http://127.0.0.1:8090/) &
echo "工作站地址: http://127.0.0.1:8090   (日志: tail -f output/station.log)"
EOS
chmod +x "$REL/启动工作站.sh"

echo "==> 打包"
cd release
rm -f g1_wall_spray_station.zip
zip -qr g1_wall_spray_station.zip g1_wall_spray_station
echo "包内文件数: $(unzip -l g1_wall_spray_station.zip | tail -1 | awk '{print $2}')"
du -sh g1_wall_spray_station.zip
echo "==> 完成: $BASE/release/g1_wall_spray_station.zip"
