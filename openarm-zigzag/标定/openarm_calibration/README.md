# OpenArm 零点标定 ROS2 节点使用说明

基于官方 `openarm_can` 标定时序（`openarm_can-zero-position-calibration.py` v1 机械臂）
独立开发的 ROS2 标定程序。通过 SocketCAN/CANFD 直接驱动达妙电机，无需 libdm_device。

## 一、程序结构

```text
openarm_calibration/
├── package.xml                     # 包描述（ament_cmake）
├── CMakeLists.txt                  # 构建配置
├── config/calibration_params.yaml  # 全部可调参数（波特率外的一切）
├── launch/calibration.launch.py    # 启动文件
├── run_calibration.sh              # 一键启动脚本（root + 环境）
├── scripts/calibration_node        # 可执行入口
└── openarm_calibration/
    ├── dm_canfd.py                 # CANFD 驱动层（协议 1:1 对齐官方库）
    └── calibration_node.py         # 标定节点（官方时序移植）
```

## 二、硬件前提（本机已实测确认）

- 适配器：gs_usb（1d50:606f）接 can0，**CANFD 模式**
- 电机：8 台达妙，ID 0x01–0x08（J1-J2=DM8009，J3-J4=DM4340，J5-J7=DM4310，夹爪=DM4310）
- 反馈 ID = 发送 ID + 0x10；全部 MIT 模式

接口初始化（每次插拔适配器后需要重新执行）：

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000 sample-point 0.75 \
     dbitrate 5000000 fd on dsample-point 0.75 dsjw 2
sudo ip link set can0 up
```

> 注意：gs_usb 固件不支持 `restart-ms` 参数，加上会报
> "Device doesn't support restart from Bus Off"。

## 三、部署与编译

```bash
cd ~/桌面/openarm实验
./openarm_calibration/build.sh        # 内部已指定系统 Python，conda 环境无需退出
chmod +x openarm_calibration/run_calibration.sh
```

> conda 共存说明：ROS2 Jazzy 绑定系统 Python3.12，与 conda python3.14 ABI 不兼容。
> 编译必须用 `-DPython3_EXECUTABLE=/usr/bin/python3`（build.sh 已内置）；
> 运行时 ros2 CLI 的 shebang 为绝对路径 /usr/bin/python3，不受 conda 影响。

## 四、标定流程（两步启动制，防误触发）

**第 1 步：以安全锁定启动（电机绝不运动），观察实时状态**

```bash
cd ~/桌面/openarm实验
chmod +x openarm_calibration/run_calibration.sh
./openarm_calibration/run_calibration.sh          # allow_motion=false
```

另开一个终端（也用 root）查看实时状态：

```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash
sudo -E env HOME=$HOME PATH="$PATH" PYTHONPATH="$PYTHONPATH" \
    /opt/ros/jazzy/bin/ros2 topic echo /calibration_node/status
sudo -E env HOME=$HOME PATH="$PATH" PYTHONPATH="$PYTHONPATH" \
    /opt/ros/jazzy/bin/ros2 topic echo /calibration_node/joint_states --once
```

**第 2 步：确认机械臂姿态安全后，触发标定**

方式 A——重启时解锁：

```bash
./openarm_calibration/run_calibration.sh true right   # true=解锁, right/left=标定侧
sudo -E env HOME=$HOME PATH="$PATH" \
    /opt/ros/jazzy/bin/ros2 service call /calibration_node/start \
    std_srvs/srv/Trigger
```

方式 B——启动后用 launch 参数解锁（同上脚本第 1 参数）。

**标定过程（自动，约 5–8 分钟）**：

1. **零位基准**：使能并锁定操作员手摆的初始位形，读取 8 关节位置作为软件零位基准
2. **J1 防护**：J1 摆动 45°（方向按臂侧自动确定），防止后续标定碰撞基座
3. **双向限位标定**（顺序：夹爪→J7→J6→J5→J4→J3→J2）：
   每个关节 → 正向碰撞记录第一限位 → 反向碰撞记录第二限位 → 低速（3s）平稳回零位
4. **J1 标定**：从防护位精确回零位 → 双向限位标定 → 回零位
5. **数据落盘**：写入 `zero` 文件（JSON）
6. 全部电机失能

**zero 文件内容**（默认路径 `zero`，可用 `zero_file_path` 参数改）：

```json
{
  "timestamp": "2026-09-01 15:00:00",
  "status": "ok",
  "arm_side": "right",
  "zero_position_rad": [...],    // 系统零点（8 关节）
  "pos_limits_rad": [...],       // 各关节正向限位
  "neg_limits_rad": [...]        // 各关节负向限位
}
```

异常/中止时同样写 `zero` 文件：`status: "error"/"aborted"`、`error` 字段、
`positions_at_abort_rad`（异常瞬间的关节位置），并**先缓慢回位再失能**
（不立即断电停止，防止甩臂伤人/撞基座）。

**中止**（随时可用）：

```bash
sudo -E env HOME=$HOME PATH="$PATH" \
    /opt/ros/jazzy/bin/ros2 service call /calibration_node/abort \
    std_srvs/srv/Trigger
```

中止后所有电机立即失能（finally 兜底保证）。

## 五、参数调整（config/calibration_params.yaml）

| 参数 | 默认 | 说明 |
|---|---|---|
| `allow_motion` | false | 安全总闸，false 时 start 服务被拒绝 |
| `arm_side` | right | right/left，决定 J1/J2 碰撞方向与理想零位 |
| `robot_version` | v1 | 限位表版本（当前 v1） |
| `bump_timeout_s` | 20.0 | 单关节碰撞超时，超时报错并失能 |
| `bump_step_deg` | 0.2 | 逐限位步进（度），减小更柔和但更慢 |
| `bump_kp/kd` | 45/1.2 | 碰撞刚度（J1 固定 280/2.1，J2 复位 280/2.0，同官方） |
| `hold_kp/kd` | 官方值 | 使能后初始保持刚度 |
| `read_limits` | true | 启动时从电机寄存器读 PMAX/VMAX/TMAX（推荐保持开启：实测 J3/J4 的 VMAX=20 与官方表不同，影响 MIT 编解码精度） |
| `max_torque_scale` | 1.0 | 预留全局转矩缩放 |
| `zero_file_path` | zero | 标定数据文件路径 |
| `j1_protect_deg` | 45.0 | J1 防护摆动角度（度） |
| `j1_protect_dir` | auto | J1 防护方向 auto/positive/negative |
| `return_time_s` | 3.0 | 限位→零位低速回程时长 |
| `joint_first_dir` | [-1,-1,1,-1,1,1,1,-1] | 各关节首次碰撞方向（J1..J7,夹爪） |

## 六、实时监控话题

| 话题 | 类型 | 内容 |
|---|---|---|
| `/calibration_node/joint_states` | sensor_msgs/JointState | 50Hz 位置/速度/转矩 |
| `/calibration_node/status` | std_msgs/String | JSON：phase(init/bump/interp/zeroing/done/error)、step/total_steps、当前关节、8 关节位置、错误信息、标定结果 deltas |

示例 status 输出：

```json
{"phase": "bump", "step": 5, "total_steps": 24, "desc": "J3 碰撞限位",
 "joint": "J3", "side": "right", "positions_rad": [...], "error": ""}
```

## 七、常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| start 返回"安全锁定" | allow_motion=false | 以 `run_calibration.sh true right` 重启 |
| 启动报"ESC_ID=0xXX 无反馈" | 接口没配 FD / 电机没上电 / 线没接本机 | 按第二节重配 can0；`lsusb | grep 1d50` 确认适配器 |
| 某关节"碰撞限位超时" | 阈值不匹配或机械干涉 | 检查该关节可自由运动；必要时调大 default_tau_th 的反义（减小）、检查 bump_step_deg |
| 位置读数跳变异常 | MIT 限参与电机固件不一致 | 保持 read_limits=true（寄存器实测优先） |
| ros2 命令挂起 | 非 root 运行 CLI（DDS 权限） | 所有 ros2 命令按第四节加 sudo -E env |
| 接口 "Device or resource busy" | 未先 down 就改参数 | 先 `sudo ip link set can0 down` |
| launch 后无输出 | 沙箱/环境变量问题 | 用 run_calibration.sh（已处理 HOME/PYTHONPATH/LD_LIBRARY_PATH） |

## 8.5、仿真验证（推荐实机前必做）

在 vcan0 虚拟总线上完整仿真标定流程，验证状态机/顺序/保持逻辑/异常保护/zero 文件，
**标定节点代码与实机完全一致**，仅换总线：

```bash
cd ~/桌面/openarm实验
./openarm_calibration/build.sh                      # 安装 dm_motor_sim.py
./openarm_calibration/run_simulation.sh true right  # 创建 vcan0 + 仿真器 + 标定节点(已解锁)
```

另开终端触发标定（vcan0 无需真实电机，root 环境由脚本处理）：

```bash
./openarm_calibration/ros2.sh service call /calibration_node/start std_srvs/srv/Trigger
./openarm_calibration/ros2.sh topic echo /calibration_node/status
```

仿真器（scripts/dm_motor_sim.py）模拟：
- 8 台电机的使能/失能/MIT 控制律（一阶惯性动力学）
- **机械限位堵转**：限位直接取自官方 `openarm-can-zero-position-calibration` 的
  `MECH_LIM_V1` 表（出厂编码系），碰撞判定可真实触发
- refresh/参数查询应答（PMAX/VMAX/TMAX/CTRL_MODE=MIT）
- 初始位形 [60,0,0,70,0,0,0,-20]°（各关节限位中位附近，双向均有行程）

**验收对照表（zero 文件 vs 官方 MECH_LIM_V1，rad）**：

| 关节 | 官方负限位 | 官方正限位 |
|---|---|---|
| J1 | -1.396 | +3.491 |
| J2 | -1.745 | +1.745 |
| J3 | -1.571 | +1.571 |
| J4 | 0.000 | +2.443 |
| J5 | -1.571 | +1.571 |
| J6 | -0.785 | +0.785 |
| J7 | -1.571 | +1.571 |
| 夹爪 | -1.047 | 0.000 |

仿真跑完后 `zero` 文件的 `neg_limits_rad`/`pos_limits_rad` 应与上表逐一吻合
（误差 < 0.02 rad）。已通过数学级验证（8/8 一致）。

RViz 映射（rviz_bridge）也由官方限位唯一确定：
left 臂 J1 偏移 -120°、J2 偏移 -90°（满程对满程），其余直映射。

实机切换：把 `config/calibration_params.yaml` 的 `can_iface` 保持 `can0` 即可
（sim_params.yaml 仅仿真用）。

## 八、安全须知

1. 标定会**真实运动机械臂并碰撞机械限位**（这是官方标定原理），请清空机械臂工作空间、扶住底座、远离末端。
2. `bump_timeout_s` 到时或任何异常都会触发全电机失能（finally 兜底）。
3. 写零点（0xFE）发生在失能之后、按官方逻辑在结束位形执行——首次标定建议先人工把机械臂摆到接近折叠位形再启动。
4. 首次运行可把 `bump_step_deg` 降到 0.1 试跑一遍观察行为。

## 九、与官方程序的对应关系

| 本程序 | 官方源（openarm_can-main） |
|---|---|
| `dm_canfd.py` 帧编解码 | `damiao_motor/dm_motor_control.cpp` + `dm_motor_device.cpp` |
| `bump_to_limit` | `openarm_can-zero-position-calibration.py::_bump_to_limit` |
| `interpolate` | `::_interpolate`（500 步线性插值） |
| 主流程顺序/刚度 | `::_run_left_sequence/_run_right_sequence`（v1） |
| 限位表/理想零位/JOINT_SIGN | 同脚本常量 |
