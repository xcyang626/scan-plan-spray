#!/usr/bin/env bash
# 一键自动化流水线: 站姿等宽完整之字形喷涂轨迹 全链生成 + UFO 部署打包
# 用法: cd scripts && bash run_pipeline.sh [--skip-02]
# 阶段: 02(工作区,可跳过) -> 03(轨迹,快速IK) -> 04(3D图) -> 05(动作导出,增量IK)
#       -> 05b(时间序列图) -> 06(回放渲染) -> 06b(mp4合成) -> 部署打包
set -euo pipefail
cd "$(dirname "$0")"
BASE="$(cd .. && pwd)"
ISAAC=${G1W_ISA_PY:-$(python3 -c "import json;print(json.load(open('$BASE/station/env.json'))['isaac_py'])" 2>/dev/null || echo /home/yang/miniconda3/envs/isaaclab/bin/python)}
MJ=${G1W_MUJ_PY:-$(python3 -c "import json;print(json.load(open('$BASE/station/env.json'))['mujoco_py'])" 2>/dev/null || echo /home/yang/miniconda3/envs/mujoco/bin/python)}
VER=v5_stand_zigzag
export VER   # 供下方 heredoc(<<'PY' 不展开变量)里的 Python 经 os.environ 读取, 避免再次硬编码版本号
T0=$SECONDS
# 工作区来源: 主 output 有则用(本机), 否则用 5° 档缓存(纯 release 包); 站点流程不受影响
if [ -f "$BASE/output/g1_workspace.npz" ]; then WS_DIR="$BASE/output";
else WS_DIR="$BASE/output/ws_cache/g1_rp5"; fi
export G1W_IN="$WS_DIR" G1W_WAIST_DEG=${G1W_WAIST_DEG:-5} G1W_WAIST_RP=${G1W_WAIST_RP:-0.087266}

stage() {  # stage <名称> <命令...>
  local name="$1"; shift
  echo "==> [$name] 开始 $(date +%H:%M:%S)"
  local t=$SECONDS
  "$@"
  echo "==> [$name] 完成, 耗时 $((SECONDS - t))s"
}

if [ "${1:-}" != "--skip-02" ]; then
  stage "02 工作区(腰±5°)" $ISAAC 02_g1_workspace.py
fi
stage "03 轨迹规划(锚点+样条快速IK)" $ISAAC 03_plan_trajectories.py
stage "04 三维可视化" $ISAAC 04_visualize_3d.py
stage "05 动作导出(增量MJCF-IK)" env MOTION_VER=$VER $MJ 05_export_ufo_motion.py
stage "05b 时间序列图" $ISAAC 05b_plot_motion.py
stage "06 回放渲染" env MOTION_VER=$VER MUJOCO_GL=glfw DISPLAY=${DISPLAY:-:0} $MJ 06_replay_render.py
stage "06b mp4与九宫格合成" env MOTION_VER=$VER $ISAAC 06b_make_replay_video.py

# ---- 部署打包 ----
stage "打包UFO部署产物" bash -c "
  set -e
  D=\"$BASE/deploy/${VER}_ufo\"
  rm -rf \"\$D\" && mkdir -p \"\$D/motion\" \"\$D/trajectory\" \"\$D/figures\"
  cp \"$BASE/output/ufo_motion/$VER/wall_spray_motion.npz\" \"\$D/motion/\"
  cp \"$BASE/output/ufo_motion/$VER/wall_spray_motion_qpos.csv\" \"\$D/motion/\"
  cp \"$BASE/output/ufo_motion/$VER/motion_meta.json\" \"\$D/motion/\"
  cp \"$BASE/output/ufo_motion/$VER/replay_motion.mp4\" \"\$D/motion/\"
  cp \"$BASE/output/ufo_motion/$VER/fig_replay_contactsheet.png\" \"\$D/motion/\"
  cp \"$BASE/output/wall_path.csv\" \"$BASE/output/robot_path.csv\" \"\$D/trajectory/\"
  cp \"$BASE/output/trajectory_meta.json\" \"$BASE/output/trajectory_points.ply\" \"\$D/trajectory/\"
  cp \"$BASE/output/fig_3d_scene.png\" \"$BASE/output/fig_wall_path_2d.png\" \"$BASE/output/fig_robot_path_2d.png\" \"\$D/figures/\"
  cp \"$BASE/output/ufo_motion/fig_motion_timeseries.png\" \"\$D/figures/\"
  cd \"$BASE/deploy\" && rm -f \"g1_wall_spray_${VER}_ufo.zip\"
  zip -qr \"g1_wall_spray_${VER}_ufo.zip\" \"${VER}_ufo\"
  echo 包内文件数: \$(unzip -l \"g1_wall_spray_${VER}_ufo.zip\" | tail -1 | awk '{print \$2}')
"

# 使用说明.md (数字自动读自 meta)
$ISAAC - <<'PY'
import json, datetime, os
VER = os.environ["VER"]
BASE = ".."
t = json.load(open(f"{BASE}/output/trajectory_meta.json"))
m = json.load(open(f"{BASE}/output/ufo_motion/{VER}/motion_meta.json"))
rows = "\n".join(
    f"| {r['h']:.2f} | {r['z_w']:.3f} | {r['angle_deg']}° | {'腕上方/枪下倾' if r['dir'] > 0 else '腕下方/枪上倾'} |"
    for r in t["row_params"])
md = f"""# G1 舱壁喷涂 UFO 部署包 使用说明（{VER}）

生成日期: {datetime.date.today()} | 一键复现: `scripts/run_pipeline.sh`

## 1. 动作合同（UFO / MimicLite）
- `motion/wall_spray_motion.npz`: qpos (T={m['frames']}, 36) float32 @30Hz（{m['duration_s']}s）
- 36维 = root位姿7 + 左腿6 + 右腿6 + 腰3 + 左臂7 + 右臂7（顺序见 motion_meta.json/README）
- 用法: npz 放入 `mimiclite/motions/any4hdmi-g1-lafan/`，项目 cfg 指定该动作

## 2. 场地摆放
- 机器人面向 +x，root xy=0，root quat 恒 identity（朝墙），z 含逐帧贴地校正
- **墙内面放在 root 前方 +x 0.74m**（站位保证带 0.70~0.78m）

## 3. 姿态与喷涂参数
- 腰 roll/pitch 限幅 ±5°（收紧稳定裕度，部署前确认实机腰限位与安全策略允许）
- 全程站姿（骨盆≈0.77m）；墙面 0~0.55m 蹲姿区未覆盖（后续版本）
- 行距 0.10m，统一喷涂带宽 u∈{t['uniform_stroke_u']}（宽 {t['uniform_stroke_u'][1]-t['uniform_stroke_u'][0]:.2f}m）
- 每行角离散 {{90,80,70,60}}°，行常数、纯俯仰指向（左右恒90°）；60°行实现带~2°偏置保证≥60°

| 行高 h (m) | 腕心 z_w (m) | 行角 | 偏置方向 |
|---|---|---|---|
{rows}

## 4. 跟踪注意事项（实机经验）
- viewer 中**必须取消勾选 Elastic Band**（否则机器人被拽飞）
- 策略为 LAFAN 数据训练，站姿贴墙动作跟踪有限；建议先仿真逐帧回放（npz）再上实机
- 质量指标: MJCF-IK {m['mjcf_ik'][:20]}，喷涂帧实测角 {m['spray_angle_deg']}°，最大关节速度 {m['max_joint_vel_rad_s']} rad/s

## 5. 文件清单
- motion/: 参考动作npz + 逐帧csv + meta + 回放mp4 + 九宫格
- trajectory/: 墙面/机器人轨迹csv + trajectory_meta.json + PLY点云
- figures/: 三维场景图 / 墙面轨迹图 / 腕心轨迹图 / 时间序列图
"""
open(f"{BASE}/deploy/{VER}_ufo/使用说明.md", "w", encoding="utf8").write(md)
print("使用说明.md 已生成")
PY
stage "重打包(含使用说明)" bash -c "
  cd $BASE/deploy && rm -f g1_wall_spray_${VER}_ufo.zip
  zip -qr g1_wall_spray_${VER}_ufo.zip ${VER}_ufo
  echo 包内文件数: \$(unzip -l g1_wall_spray_${VER}_ufo.zip | tail -1 | awk '{print \$2}')
"

# ---- 终验 ----
echo "==> [终验] 关键数字"
$ISAAC - <<'PY'
import json, os
VER = os.environ["VER"]
t = json.load(open("../output/trajectory_meta.json"))
m = json.load(open(f"../output/ufo_motion/{VER}/motion_meta.json"))
print(f"轨迹: {t['n_points']}点, IK严格{t['ik_strict']}/边缘{t['ik_loose']}/失败{t['ik_fail']}, "
      f"角度{[round(v,2) for v in t['angle_achieved_deg']]}° ok={t['angle_rule_ok']}, "
      f"统一u区间{t['uniform_stroke_u']}")
zs = [r['z_w'] for r in t['row_params']]
assert all(b > a for a, b in zip(zs, zs[1:])), "z_w未严格递增!"
print(f"z_w严格递增 ✓ {[round(z,3) for z in zs]}")
print(f"动作: {m['frames']}帧, mjcf_ik {m['mjcf_ik'][:30]}, "
      f"喷涂角{m.get('spray_angle_deg')} ok={m.get('spray_angle_ok')}, "
      f"最大关节速度{m.get('max_joint_vel_rad_s')} rad/s")
PY
echo "==== 流水线全部完成, 总耗时 $((SECONDS - T0))s ===="
echo "部署包: $BASE/deploy/g1_wall_spray_${VER}_ufo.zip"
