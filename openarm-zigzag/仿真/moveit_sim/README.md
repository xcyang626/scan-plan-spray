# OpenArm 左臂之字形走线 — MoveIt 仿真：使用与实现详解

> 路线定位: 三条仿真路线（ROS2+RViz / MuJoCo / **MoveIt**）中的 MoveIt 官方方案，
> 目标是让左臂在 x=0.30m 竖直平面（与 2m 墙平行）上按 36 路点之字形缓慢扫线（工作矩形 29×53cm，见 §9 变更记录），
> 全程关节处于 zero 实测标定限位内。
>
> 本文合并了原《仿真指南》与实现原理：第 1~3 节讲怎么用，第 4~8 节讲怎么实现、
> 为什么这样实现、以及每个坑是怎么踩出来的。

---

## 1. 快速开始

```bash
cd ~/桌面/openarm实验/仿真/moveit_sim
./start_zigzag_sim.sh
```

RViz 窗口弹出（**只有一条左臂**，无右臂、无 ghost），随后自动执行完整流程。
跑完后 RViz 保持打开，终端按回车则清理所有进程退出。

对照模式:

| 命令 | 效果 |
|---|---|
| `./start_zigzag_sim.sh` | 单臂显示 + 稠密 IK 轨迹（推荐，默认） |
| `./start_zigzag_sim.sh --segment` | 旧逐段 MoveIt 笛卡尔模式（对照，会看到大量关节退化） |
| `./start_zigzag_sim.sh --bimanual` | 官方双臂 demo（对照，画面有两条臂 + ghost） |

RViz 画面元素: 黄色线 = 之字形计划路径；半透明蓝面 = x=0.30m 工作平面；
灰色板 = 2m 墙体碰撞体；蓝色目标球 ghost 已在预设中关闭（见 §4.4）。

## 2. 机械臂执行流程

```
零位(全零位形, 0.15 倍速)
  → 之字形起点(关节目标, 0.2 倍速)
  → 之字形走线(稠密关节轨迹, TCP 限速 5cm/s, 约 115s, 1129 点)
  → 终点缓慢回零(0.1 倍速)
```

前置、走线、收尾全部由 `zigzag_moveit.py` 一个节点完成（服务 + action 接口）。

## 3. 文件地图

```
moveit_sim/
├── start_zigzag_sim.sh      # 一键启动: 清理残留 → 起 demo+RViz → 执行 → 清理
├── zigzag_moveit.py          # 主执行节点(两种模式: --dense 稠密 / 逐段笛卡尔)
├── make_dense_zigzag.py      # 离线稠密 IK 轨迹生成(纯 numpy, 无 ROS 依赖)
├── dense_zigzag.json         # 1129 点关节轨迹 + 每步 TCP 位移(生成产物)
├── run_moveit_zigzag.sh      # 只执行不起 demo(需自己先起 demo)
├── workspace.json            # 阶段3产物(在 ../isaaclab_sim/, 本目录不另存)
└── left_config/              # 左臂 MoveIt 配置(§4.4)
    ├── left_demo.launch.py           # 单臂 demo launch
    ├── openarm_left_moveit.urdf      # 单臂 URDF(官方运动学 + mock 硬件块)
    ├── openarm_left.srdf             # left_arm 组 + 碰撞豁免
    ├── openarm_left.rviz             # RViz 预设(关 ghost)
    ├── kinematics.yaml               # KDL IK
    ├── joint_limits.yaml             # 速度+加速度限位(TOTG 必需)
    ├── moveit_controllers.yaml       # MoveIt 侧控制器管理
    └── ros2_controllers.yaml         # ros2_control 侧 JTC + 状态广播
```

> 右臂(2026-09-11 起)为镜像的一套: `right_config/`(同上 8 个文件,
> 名字改 right)、`start_zigzag_sim_right.sh`、`make_dense_zigzag_right.py`
> → `dense_zigzag_right.json`。`left_config/` 与 `right_config/` 是**两套
> 独立副本**，不是符号链接；改一处不会同步另一处，必须手工对齐。

依赖的既有产物:
- `isaaclab_sim/workspace.json`: 阶段3 工作空间扫描的 36 之字形路点 + 网格关节解
- `openarm_sim/openarm_sim/ik_solver.py`: DlsIK 数值逆解（5D 约束，warm start）
- `isaaclab_sim/common.py`: UrdfChain FK / zero 文件桥接
- `openarm_ros2_ws`: 官方 openarm_description + openarm_bimanual_moveit_config

---

## 4. 实现原理

### 4.1 为什么不用 MoveIt 的笛卡尔路径服务

MoveIt 自带 `/compute_cartesian_path`（GetCartesianPath），按理说直线走线该用它。
实测结果: **17 段里 13 段 fraction < 1**（大量 0.47 左右的平台值），即 KDL 沿直线
中途无解，被迫回退关节目标 → TCP 脱离直线。

原因分析:
- 笛卡尔请求指定了**固定四元数**（位置 + 完整姿态 = 6D 约束）。而本任务对姿态的
  真实要求只有"接近轴指向 +x"（5D：位置 3D + 轴 2D，绕轴自旋自由）。
  对 KDL 来说 6D 是过约束，可行解域被人为压窄，沿直线走几步就进死区。
- 实验过的补救都无效: 传 `start_state` 种子（换个冗余分支，更差）、
  `avoid_collisions=False`（分数完全不变，说明瓶颈是 IK 可行性不是碰撞）。

阶段3 工作空间扫描用的就是 5D 约束（`DlsIK`，位置 + 接近轴，自旋放开），
36 个路点的关节解全部收敛——所以正确路线是**把阶段3 的 IK 语义延续到执行**。

### 4.2 稠密 IK 方案（当前方案）

```
make_dense_zigzag.py (离线, 纯 numpy)
  36 路点连线 → 5mm 步长稠密采样(与 GetCartesianPath 的 max_step 同粒度)
  → 逐点 DlsIK warm start(上一解作种子, 随机重启兜底, 容差 1mm/2°)
  → dense_zigzag.json {1129 个关节点, 每步 TCP 位移}

zigzag_moveit.py --dense (在线)
  回零位 → 起点(关节目标) → 整条 1129 点轨迹重定时 → ExecuteTrajectory 一次执行
  → TF 采样验证 → 终点缓慢回零
```

要点:
- **TCP 直线度由 IK 容差保证**，不依赖 MoveIt 规划。实测全程平面偏差
  mean 0.3mm / max 1.5mm（见 §7）。
- **重定时** (`_retiming`): 每步取 `max(关节位移/关节限速, TCP位移/0.05m/s, 0.02s)`
  作为该步时间。TCP 匀速 5cm/s 为主，拐点处关节限速兜底，0.02s 下限防零间隔。
- 限位用 **zero 标定软限位 ∩ 桥接换算限位**（取更紧）。为什么: 桥接限位偏宽，
  曾让 J5 解到 90.5°，超过 URDF 硬限位 ±90°，执行后下一次规划直接
  `Start state out of bounds` 失败。软限位表来自用户标定结论:
  J1[-197.64,77.64] J2[-187.65,7.65] J3[-89.54,89.54] J4[3,124.43]
  J5[-87.48,87.48] J6[-40.64,40.64] J7[-87.83,87.83]（度）。
- warm start 沿 5D 解分支连续走，绕接近轴的自旋连续漂移（实测最大 63°），
  对贴墙作业无影响；接近轴偏差 max 0.92°。

### 4.3 逐段笛卡尔模式（--segment，仅作对照）

`move_line_cartesian`: 每段 2 路点请求 GetCartesianPath（avoid_collisions
True→False 两次尝试），fraction ≥ 0.999 才执行，否则回退 workspace.json 的
关节解。这就是最初跑通全流程的模式，保留下来用于对比演示"6D 过约束为什么不行"。

### 4.4 单臂显示是怎么来的（left_config/）

官方 `demo.launch.py` 是 bimanual 配置: URDF 天然两条臂，MoveIt RViz 插件默认
又显示**目标位姿 ghost（蓝色实体）**和**规划路径动画 ghost（半透明）**，
画面像好几条臂。left_config 用官方素材搭了一套最小单臂配置:

**URDF** (`openarm_left_moveit.urdf`)，由 `isaaclab_sim/openarm_left.urdf` 生成，
因为那份 URDF 已经过验证与官方运动学逐项一致（基座变换、7 关节 origin/rpy/axis
全部 SAME），三处修改:
1. `hand_tcp` 固接从 link7+0 改为 **link7+0.186**（0.1025+0.0835），对齐官方
   TCP 语义——`tcp_report` 的指尖换算依赖这一点（§4.5）；
2. 插入 `<ros2_control>` 块: `mock_components/GenericSystem`（jazzy 里
   `fake_components` 已更名，且环境里实际注册的插件只有 mock 系），
   7 臂关节 + `openarm_left_finger_joint1` 都要声明——**手指关节不声明就不会
   出现在 /joint_states，MoveIt 等完整状态会超时**（finger_joint2 是 mimic，
   不需要单独声明）;
3. mesh 路径从绝对路径改为 `package://openarm_description/...`（22 处），
   由 ament 索引解析，RViz / move_group 都能加载。

**SRDF** (`openarm_left.srdf`): 只定义 left_arm 7 关节组 + zero 位形。
**碰撞豁免清单必不可少**: 从官方 bimanual SRDF 抄了 left 部分的
`disable_collisions`（相邻链接对）。缺了它，零位下相邻 link 的常态接触会被
`CheckStartStateCollision` 判成 "6 contacts detected" 直接拒绝一切规划。

**launch** (`left_demo.launch.py`): robot_state_publisher + ros2_control_node
+ 两个 spawner + move_group + rviz2，手动构造 move_group 参数。两个关键点:
- OMPL 管线参数（planning_plugins / request_adapters / response_adapters）
  是从运行中的官方 move_group `ros2 param get` 逐项 dump 的实测值
  （jazzy 格式，response_adapters 含 AddTimeOptimalParameterization），
  不猜格式;
- **trajectory_execution 的内容必须平铺到 move_group 参数顶层**
  （`moveit_manage_controllers`、`moveit_simple_controller_manager...`），
  嵌在 `trajectory_execution:` 键下会读到 "No controller_names specified"。

**RViz 预设** (`openarm_left.rviz`，拷官方 moveit.rviz 改 4 处):
`Query Goal State: false`（去蓝色 ghost）、`Loop Animation: false`（去透明
动画）、`Planning Group: left_arm`、加 `/zigzag_markers` 的 MarkerArray 显示
（工作平面/路径线/墙体）。

**ros2_controllers.yaml**: 控制器名固定 `left_joint_trajectory_controller`
（与执行节点的 action 等待路径一致）；**JTC 的 state_interfaces 只能写
position/velocity**——jazzy 的 JTC 不接受 `effort`，写了直接 init 失败。

### 4.5 坐标系与 TCP 语义（重要约定，容易踩坑）

两份 URDF 的 `hand_tcp` 定义不同:
- **官方 MoveIt URDF**: `hand_tcp` = link7 + 0.186m（0.1025+0.0835，夹爪真实尖端）
- **isaaclab URDF（本工程 IK 链）**: `hand_tcp` = link7 原点（零偏移）

而全工程的路点/工作平面语义是阶段3 定下的: **路点 = 指尖位置 = link7 + 0.12m
× 接近轴**（`TCP_OFFSET=0.12`）。所以执行验证时，TF 读到的 hand_tcp 要沿接近轴
回退 `0.186 − 0.12 = 0.066m` 才是"指尖"，再对 x=0.30 平面求偏差。
`zigzag_moveit.py::tcp_report` 已做此换算。这个坑的发现过程: 稠密轨迹执行后
TF 实测平面偏差恒定 0.186m（±0.4mm）——恒定偏移不是执行错误，恰是 TCP 定义
差异，而 ±0.4mm 的波动反而证明了走线平面度极佳。

### 4.6 启动时序问题（为什么节点里有一堆"等待"）

"MoveIt 服务就绪" ≠ "可以执行"，实际要等三层（`connect()` 里逐层等）:
1. `/joint_states` 首条消息到达 = joint_state_broadcaster 已激活;
2. `/left_joint_trajectory_controller/follow_joint_trajectory` action server
   出现 = JTC 已激活;
3. 再留 2s 宽限，等 move_group 的 controller handle 与控制器建连。

少任何一层，首段执行会 `MOTION_PLAN_INVALIDATED_BY_ENV_CHANGE` ABORT
（move_group 校验轨迹起点时拿不到当前关节状态，或发轨迹时控制器 action 未连）。
实测教训: 早期一键脚本在服务就绪后立刻执行，两次全在这步翻车。

### 4.7 官方包上的一处前置修改

`openarm_bimanual_moveit_config/config/openarm_v1.0/joint_limits.yaml` 官方
是 `has_acceleration_limits: false`，而 jazzy 的规划响应适配器
`AddTimeOptimalParameterization`(TOTG) **没有加速度限位就直接 FAILURE**——
这是最初"所有关节规划都报 FAILURE(通用)"的根因。已补丁:
`has_acceleration_limits: true`，加速度 = 最大速度一半。install 目录是源码
目录的符号链接，改一处即两处生效; 备份在同名 `.bak`。
**重新下载/重建该包后若规划全挂，先查这个补丁。**

---

## 5. 验证数据解读

执行结束打印一行 `TCP 验证(已换算到指尖)`:

| 字段 | 含义 | 达标值(本机实测) |
|---|---|---|
| 采样 | TF 采样条数(50Hz) | ~5700（约 115s） |
| 平面偏差 | 指尖到 x=0.30m 平面距离 | mean 0.3mm / max 1.5mm |
| y / z 范围 | 指尖覆盖矩形 | y[0.128,0.418] z[0.43,0.96]，与设计矩形吻合 |
| 姿态漂移(含自旋) | 与固定四元数的全姿态差 | ~63°——**5D IK 的预期行为**，自旋不影响贴墙 |
| 接近轴偏差 | 夹爪指向与 +x 夹角 | mean 0.04° / max 0.92° |

## 6. 全局工程约定（本目录）

- ROS 节点脚本用 `/usr/bin/python3`（系统解释器; conda 的 3.14 缺 yaml，
  一键脚本会自动把 conda 移出 PATH）;
- `ROS_LOG_DIR=/tmp/openarm_sim_ros_log` 必须 export;
- 代码注释中文，关键决策写明原因;
- 残留僵尸进程会互相污染（实测 4 个 robot_state_publisher 并存时，
  control_node 从 DDS 话题收到旧双臂 URDF 把硬件加载成 left+right 两块），
  重启前务必清理——一键脚本已内置（pkill 模式用 `[m]ove_group` 方括号写法
  防止匹配到脚本自身命令行，无方括号的模式会自杀，已实测踩坑）。

## 7. 常见问题速查

| 症状 | 根因 | 处置 |
|---|---|---|
| 所有关节规划 FAILURE(通用) | joint_limits 缺加速度限位（补丁丢失） | 对照 `.bak` 恢复 §4.7 补丁 |
| Start state out of bounds (J5≈90.5°) | 轨迹解越 URDF 硬限位 | 用 `make_dense_zigzag.py` 重新生成（已内置软限位∩） |
| 6 contacts detected, 拒绝规划 | SRDF 缺碰撞豁免 | left_config/openarm_left.srdf 已含 |
| 首段执行 ABORT / env change | 控制器未就绪就发轨迹 | 节点已内置三层等待（§4.6），勿删 |
| JTC init 失败 (effort) | jazzy JTC 不接受 effort 状态接口 | ros2_controllers.yaml 已去掉 |
| hardware 插件不存在 | jazzy 无 fake_components | URDF 已改 mock_components |
| move_group 报 No controller_names | trajectory_execution 未平铺 | launch 已按顶层键传参 |
| RViz 机器人残缺/隐形 | mesh 绝对路径失效 | URDF 已改 package://（22 处） |

## 8. 其他仿真路线入口

```bash
# MuJoCo(conda 环境): make_mjcf.py 生成 MJCF, manual_console.py 手动伺服台
conda activate mujoco
cd ~/桌面/openarm实验/仿真/mujoco_sim

# ROS2+RViz(阶段2): openarm_sim/ 场景 + 交互控制台
cd ~/桌面/openarm实验/仿真/openarm_sim
```

三条路线共用阶段3 的 workspace.json 路点与 zero 标定限位。

## 9. 变更记录

### 2026-09-08 工作区重扫 + 换支防护（轨迹数据全量更新）

- **假洞修复**：原网格扫描每格只重试 1 个种子，DLS 局部极小在可达区
  中间留了假洞（复查 148/148 格均可解），最大内接矩形被挤成 17×51cm。
  扫描加重启兜底 + 邻域解传播修复轮（`stage3_workspace.py`）。
- **换支防护（安全关键）**：可达区内混着两条 IK 构型支，矩形跨支时
  dense 出现 J2 单步 181° 跳变（TCP 插值甩大弧线，真机危险）。
  新增"连续支泛洪"：矩形只在"与起点同支连通"的域内选；
  `make_dense_zigzag.py` 求解拒绝换支解 + 输出前全轨迹复核
  （臂关节 J1-J4 单步 >20° 拒绝写文件）。三层保障，域内无跳变点。
- **限位对齐**：扫描限位改为 real_safety.yaml 软限位内缩 3°（=
  dense/smooth 的真机执行限位），扫描可达 = 执行可达，旧的
  "J6 收口 3.3°"补丁作废。
- **新尺寸**：矩形 29×53cm（y[0.128,0.418] z[0.431,0.961]），36 路点，
  dense 1129 点，臂关节最大单步 12.7°，TCP 总长 5.73m。
  快速重算: `stage3_workspace.py --reuse-occupancy`。
