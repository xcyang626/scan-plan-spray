# OpenArm 零点标定工具（ROS2）

对 OpenArm v1 机械臂（达妙 DM 系列电机）执行**零点标定**的 ROS2 包。标定原理与官方
`openarm_can-zero-position-calibration.py` 完全一致：各关节低速碰撞机械限位，记录双向限位
并写入零点文件 `zero`，供后续控制器使用。

纯 Python 实现（CANFD 协议 1:1 对齐官方 C++ 库），无需编译 C++，拿到手 10 分钟即可跑通仿真。

## 功能特性

- **双向限位标定**：夹爪→J7→…→J2 逐关节低速碰限位，J1 先防护摆动 45° 再标定
- **安全锁定**：默认 `allow_motion=false`，电机绝不运动；确认姿态安全后两步解锁
- **异常兜底**：超时/中止/出错均先低速回位再失能，防止甩臂伤人
- **实时监控**：`/calibration_node/status`（阶段/进度/错误）+ `/calibration_node/joint_states`（50Hz）
- **仿真验证**：内置 vcan0 + 8 电机仿真器，无硬件即可验证全流程，结果与官方限位表 8/8 吻合
- **RViz 3D 可视化**（可选）：URDF 实时显示机械臂姿态

## 硬件前提

- Ubuntu 24.04 + ROS2 Jazzy
- CAN 适配器：**gs_usb 类**（如 candleLight，USB ID `1d50:606f`），CANFD 模式
- 电机：8 台达妙（J1-J2=DM8009，J3-J4=DM4340，J5-J7=DM4310，夹爪=DM4310），
  CAN ID 0x01–0x08，反馈 ID = 发送 ID + 0x10，MIT 模式

> ⚠️ 本工具走标准 SocketCAN（can0）。达妙 DM-USB2FDCAN / DM-LinkX 等**私有 USB 协议适配器
> 不适用**（不会生成 can0 接口），需换 gs_usb 类适配器。

## Quick Start

### 1. 环境准备

```bash
# ROS2 Jazzy（若未安装）：https://docs.ros.org/en/jazzy/index.html
sudo apt install -y python3-colcon-common-extensions can-utils
```

### 2. 编译

把整个文件夹放到任意位置（无需特定工作区），直接运行：

```bash
cd 标定
./openarm_calibration/build.sh
```

> conda 用户无需退出环境：build.sh 已内置 `-DPython3_EXECUTABLE=/usr/bin/python3`
> （ROS2 Jazzy 绑定系统 Python3.12，与 conda Python ABI 不兼容）。

### 3. 配置 CAN 接口（每次插拔适配器后执行）

```bash
./openarm_calibration/scripts/can_up.sh
```

脚本自动完成：can0 1Mbps + 5Mbps FD 配置、接口拉起、禁用 USB 自动挂起（防掉线）。
看到 `can0 UP` 即成功。

### 4. 仿真验证（强烈推荐先做，无硬件即可）

```bash
cd 标定
./openarm_calibration/run_simulation.sh true right
```

另开终端触发标定并观察进度：

```bash
cd 标定
./openarm_calibration/ros2.sh service call /calibration_node/start std_srvs/srv/Trigger
./openarm_calibration/ros2.sh topic echo /calibration_node/status
```

约 5–8 分钟跑完，当前目录生成 `zero` 文件，`status` 应为 `"ok"`。
全部命令**必须通过 `ros2.sh`**（已处理 root/DDS 环境问题，直接敲 `ros2` 会收不到消息）。

### 5. 实机标定（两步启动制，防误触发）

**第 1 步：安全锁定启动（电机绝不运动），观察实时状态**

```bash
cd 标定
./openarm_calibration/run_calibration.sh          # allow_motion=false
```

人工把机械臂摆到接近折叠的初始位形，然后确认话题有数据：

```bash
./openarm_calibration/ros2.sh topic echo /calibration_node/joint_states --once
```

**第 2 步：确认姿态安全后，解锁并触发**

```bash
# Ctrl-C 掉第 1 步，以解锁模式重启（right/left = 标定侧）
./openarm_calibration/run_calibration.sh true right

# 另一个终端触发标定
./openarm_calibration/ros2.sh service call /calibration_node/start std_srvs/srv/Trigger
```

**随时中止**（所有电机立即失能）：

```bash
./openarm_calibration/ros2.sh service call /calibration_node/abort std_srvs/srv/Trigger
```

### 6. 标定结果

当前目录生成 `zero` 文件（JSON）：

```json
{
  "timestamp": "2026-09-01 15:00:00",
  "status": "ok",                  // ok / error / aborted
  "arm_side": "right",
  "zero_position_rad": [...],      // 系统零点（8 关节）
  "pos_limits_rad": [...],         // 各关节正向限位
  "neg_limits_rad": [...]          // 各关节负向限位
}
```

异常/中止时同样落盘（含 `positions_at_abort_rad` 现场位置），方便排查。

## 常用参数（config/calibration_params.yaml）

| 参数 | 默认 | 说明 |
|---|---|---|
| `allow_motion` | false | 安全总闸，false 时 start 服务被拒绝 |
| `arm_side` | right | right/left，决定 J1/J2 碰撞方向 |
| `bump_step_deg` | 0.2 | 逐限位步进（度），调小更柔和更慢 |
| `bump_timeout_s` | 20.0 | 单关节碰撞超时，超时即失能保护 |
| `return_time_s` | 3.0 | 限位→零位低速回程时长 |
| `zero_file_path` | zero | 标定结果输出路径 |

## 常见问题（FAQ）

| 现象 | 原因 | 解决 |
|---|---|---|
| start 返回"安全锁定" | allow_motion=false | 用 `run_calibration.sh true right` 重启 |
| 启动报"ESC_ID=0xXX 无反馈" | 接口没配 FD / 电机没上电 | 重跑 `scripts/can_up.sh`；`lsusb \| grep 1d50` 确认适配器 |
| ros2 命令收不到消息 | 绕过了 ros2.sh，DDS 权限问题 | 一律用 `./openarm_calibration/ros2.sh ...` |
| 接口 "Device or resource busy" | 改参数前没 down | 先 `sudo ip link set can0 down` |
| 某关节"碰撞限位超时" | 机械干涉或阈值不匹配 | 检查该关节可自由运动；把 `bump_step_deg` 降到 0.1 |
| 位置读数跳变 | MIT 限参与固件不一致 | 保持 `read_limits: true`（寄存器实测优先） |

## 安全须知

1. 标定会**真实运动机械臂并碰撞机械限位**（这是官方标定原理），请清空工作空间、扶住底座、远离末端。
2. 首次实机标定建议把 `bump_step_deg` 降到 `0.1` 试跑，观察行为后再调回。
3. J1 标定前会自动摆 45° 防护位，防止碰撞基座——不要在 J1 行程上放障碍物。
4. 任何异常都会自动失能；紧急情况直接用 abort 服务或拔电机电源。

## 目录结构

```text
标定/
├── README.md                        # 本文件
├── zero                             # 实机标定结果示例（左臂，2026-09-02，status=ok）
└── openarm_calibration/             # ROS2 包（整体拷入任意位置即可编译）
    ├── build.sh                     # 一键编译
    ├── run_calibration.sh           # 实机标定一键启动
    ├── run_simulation.sh            # 仿真验证一键启动
    ├── ros2.sh                      # ros2 CLI 包装（root 环境问题已处理）
    ├── package.xml / CMakeLists.txt # ament_cmake 包描述
    ├── config/
    │   ├── calibration_params.yaml  # 全部可调参数
    │   └── v1.urdf                  # RViz 用机械臂模型
    ├── launch/
    │   ├── calibration.launch.py    # 实机启动
    │   └── simulation.launch.py     # 仿真启动（可带 RViz）
    ├── scripts/
    │   ├── can_up.sh                # can0 一键配置
    │   └── dm_motor_sim.py          # 8 电机仿真器（vcan0）
    └── openarm_calibration/         # Python 源码
        ├── dm_canfd.py              # CANFD 驱动层（协议对齐官方库）
        ├── calibration_node.py      # 标定节点（官方时序移植）
        └── rviz_bridge.py           # RViz 姿态桥接
```

## 相关链接

- OpenArm 官方项目：`https://github.com/enactic/openarm/`
- 官方 CAN 库文档：`https://docs.openarm.dev/software/can`
- 本工具的标定时序/限位表/刚度参数与官方 `openarm_can-zero-position-calibration.py` 逐项对应，
  详细对照表见 `openarm_calibration/README.md`

## v2 官方零位版（2026-09-04）

流程升级为"官方零位版"：缓慢回零位基准 → 逐关节双向碰撞限位（每次回零位）
→ 全部关节回零位基准 → **set_zero(0xFE) 把当前位形写入电机 flash 零点** →
保存 zero 文件（零位 ≈ 全零，限位已平移到新系）→ 保持零位供人工验证。

- **go_zero 服务（新增）**：使能后以限速（默认 15°/s）缓慢运动到 zero 文件
  记录的零位基准位形并保持；不碰撞限位、不写零点，用于人工目视验证：

  ```bash
  ./openarm_calibration/ros2.sh service call /calibration_node/go_zero std_srvs/srv/Trigger
  ```

- **start**：完整标定（含 set_zero）。set_zero 位于最后一步——之前任何
  中止/异常都不改动电机 flash，旧 zero 文件依然有效。
- 标定完成后机械臂**保持零位不断能**，目视验证完 Ctrl-C 失能。
- 新参数：`zero_move_speed_deg`（回零限速，默认 15）、
  `hold_after_done`（完成后保持，默认 true）。
- 注意：set_zero 会覆盖电机 flash 旧零点（这正是目的）；之后
  zero_position_rad ≈ 全零，上位机的零位偏移 d ≈ 0。
