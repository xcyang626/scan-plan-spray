#!/bin/bash
# 一键启动喷涂轨迹工作站 (启动服务 + 自动打开浏览器)
# 解释器优先取 MUJOCO_PY 环境变量，否则用 python3，再否则回落本机 mujoco conda
cd "$(dirname "$0")"
mkdir -p output
PY="${MUJOCO_PY:-$(command -v python3)}"
# 本机 mujoco conda 兜底: 仅在未显式指定 MUJOCO_PY 时启用(否则无条件覆盖会吃掉上面的优先级)
if [ -z "${MUJOCO_PY:-}" ] && [ -x "$HOME/miniconda3/envs/mujoco/bin/python" ]; then
  PY="$HOME/miniconda3/envs/mujoco/bin/python"
fi
if curl -s -m 1 http://127.0.0.1:8090/api/status >/dev/null 2>&1; then
  echo "工作站已在运行"
else
  nohup "$PY" scripts/10_station_app.py --port 8090 \
    > output/station.log 2>&1 &
  sleep 2
fi
xdg-open http://127.0.0.1:8090/ >/dev/null 2>&1 &
echo "工作站地址: http://127.0.0.1:8090  (本目录终端可看日志: tail -f output/station.log)"
