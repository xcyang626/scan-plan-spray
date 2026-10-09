# mpc_track —— 右臂 MPC 执行轨(移植件)

> **⚠ 2026-09-14 起: 本目录是"线路二"遗留路线, 默认失活。**
> MPC 已插件化接管 JTC 槽位(`openarm_mpc_controller`, "线路一"),
> 真机配置 `mpc_enable=false`——`mpc_exec.py` 即使启动也驱动不了电机
> (话题注入通道无消费者)。新路线见
> `docs/MPC控制器插件_线路一改造设计.md` 与
> `../../common/ws/src/openarm_mpc_controller/README.md`。
> 本目录仍保留: `mpc_core.py`(增益一致性验收基准)、
> `mpc_offline_check.py`(实机前离线复核, 对插件同样代表性成立)、
> `excite_ident.py`/`identify_params.py`(辨识链)。

> 由 `left/mpc_track/`(2026-09-08 双臂重构时)机械替换移植而来,
> 逻辑未动。移植差异仅三处: 关节名 `openarm_right_jointN`、
> 指令话题 `/right_mpc_position_commands`(与驱动 URDF 参数一致)、
> FK 链显式用 `仿真/isaaclab_sim/openarm_right.urdf`(算 TCP 段长)。
> 限位/home 仍从 `../config/real_safety.yaml` 读(右臂标定后生成)。

## ⚠ 上真机前的前置(2026-09-08 晚状态)

1. 右臂 `../config/real_safety.yaml` 已生成(手工拼装 zero_right + 优化);
   **微动测试方向终裁(J3/J5)未做**——若有翻转须重新 gen + 重扫 + 重生成
   路线后再回来;
2. 右臂正式路线**已生成**: `仿真/moveit_sim/dense_zigzag_right.json`
   (31×59cm, 1355 点, 全局优化解, 零重构)——MPC 参考即用它;
3. MPC 离线复核(用本 MPC 核对新路线闭环)仍需在实机前完成。

架构/模型/调参说明见 `left/mpc_track/README.md` 与 `mpc_config.json`
(两边配置相同, 参数窗口一致)。

用法(前置满足后):

```bash
cd ~/桌面/openarm实验/real/right/mpc_track
/usr/bin/python3 mpc_exec.py \
    --ref ../../../仿真/moveit_sim/dense_zigzag_right.json --speed-scale 0.4
```

- 终端 A(右臂 control)应出现 `指令源切换: MPC 流`。
- 中止: Ctrl+C(停流, 臂原地保持) → estop → 物理急停。
