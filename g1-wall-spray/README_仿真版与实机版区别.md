# 仿真版 与 实机版 的区别

本工程（`g1_wall_spray`）同一套代码支持两种执行后端，通过一个配置项切换：

```jsonc
// station/env.json
"robot_mode": "real"   // 实机 G1（当前默认）
"robot_mode": "sim"    // MuJoCo 仿真（备份见 backup/仿真版_20261008/）
```

**仿真版备份位置**：`backup/仿真版_20261008/`
（含完整工程包 `g1_wall_spray_station_仿真版.zip`、仿真执行端 `exec_仿真版/`、仿真配置 `env.json.仿真版`、备份说明）

---

## 一、总览对比

| 维度 | 仿真版（sim） | 实机版（real，当前） |
|---|---|---|
| 机器人本体 | MuJoCo 里的 G1 模型 | 物理 Unitree G1 |
| 策略运行位置 | 本机（Python + ONNX CPU） | 本机 C++ SONIC 部署进程（TensorRT/ONNX） |
| 与机器人通信 | 无（进程内直接仿真） | **DDS**（Unitree SDK2，网卡 `enp131s0`，机器人网段 192.168.123.x） |
| 动作下发方式 | 进程内 PD 目标注入 | **ZMQ Protocol v1** 流式发送（`pose` 话题，端口 5556） |
| 下肢平衡 | GR00T RL 策略（Balance/Walk ONNX，CPU） | SONIC 策略（低延迟 checkpoint，C++ 推理） |
| 上身躯干/手臂 | 我们把轨迹的腰3+右臂7 注入 PD 目标（速度前馈） | 同左，经 ZMQ 流给机器人侧策略跟踪 |
| 启动命令 | `.venv_sim_cpu/bin/python groot_spray_runner.py` | `g1_deploy_onnx_ref <iface> ... --input-type zmq` |
| 可视化 | MuJoCo 窗口（本机） | 机器人本体（真机动作即所见） |
| 是否需要网线/机器人 | 不需要 | **必须**：网线连通 + 机器人上电 + 调试模式 |
| 安全风险 | 无 | 有（真机会动），必须有人守急停 |
| 典型用途 | 开发、回归测试、算法验证 | 实际喷涂作业 |

## 二、文件层面的差异

| 文件 | 仿真版 | 实机版 |
|---|---|---|
| `scripts/exec/groot_spray_runner.py` | **核心**：复用 GR00T `GearWbcController`，在 MuJoCo 里跑 | 不使用 |
| `scripts/exec/zmq_motion_publisher.py` | 不使用（也可用来做协议自测 `--selftest`） | **核心**：按官方 ZMQ v1 协议流式发送 qpos |
| `scripts/exec/base_lock_node.py` | 两版通用：基座扰动补偿（世界系锁定）验证/在线节点 | 同左 |
| `station/env.json` | `robot_mode: sim` | `robot_mode: real` + `robot_iface`/`zmq_host`/`zmq_port` |
| `station/10_station_app.py` | 同一份代码，按 `robot_mode` 分流启动逻辑 | 同左 |
| 规划链（02/03/04/05/11/12/13） | **完全相同**（与执行后端无关） | **完全相同** |

> 关键点：**规划链完全共享**。点云解析、等宽之字形轨迹、离散行角、动作导出、
> 基座扰动补偿验证，两版一模一样；只有"最后怎么把动作交给机器人"这一步不同。

## 三、网页流程（两版一致）

```
① 启动机器人 → ② 选点云 + 生成轨迹 → ③ 检查无误 → 生成动作 → ④ 自动执行
```

- 未启动机器人时，点云/参数/「生成轨迹」全部锁定（前后端双重校验）
- 「停止机器人」= 立即全停并回归初始状态
- 实机版「启动机器人」会拉起 C++ SONIC 部署 + ZMQ 推流端待命

## 四、实机版使用步骤

### 1. 前置（必须全部满足）
```bash
# a) 网线接好，机器人上电并进入调试模式
ip -br addr show enp131s0          # 应显示 UP 且 carrier=1
cat /sys/class/net/enp131s0/carrier # 必须为 1（=1 才有物理链路）
ping 192.168.123.161               # 应能通（G1 默认地址）
```
> 若 `carrier=0` / `Link detected: no`：网线未接通或对端未激活，此时无法控制机器人。

### 2. 启动工作站
```bash
cd g1_wall_spray && ./启动工作站.sh     # 打开 http://127.0.0.1:8090
```
点「启动机器人」→ 后台启动：
- `g1_deploy_onnx_ref enp131s0 policy/low_latency/model_decoder.onnx reference/example
   --input-type zmq --output-type zmq --zmq-host <本机IP> --zmq-port 5556`
- ZMQ 推流端（等待执行触发）

### 3. 机器人侧终端操作（重要，需人工）
部署进程的终端里：
| 按键 | 作用 |
|---|---|
| `]` | 启动控制系统 |
| `ENTER` | 切换到 ZMQ 流模式（开始跟踪我们发的动作） |
| `O` | **急停**（随时准备按） |
| `I` | 重置朝向 |

### 4. 走网页流程
选点云 → 生成轨迹 → 检查无误 → 生成动作 → 「自动执行」→ 机器人按轨迹喷涂。

## 五、切回仿真版

```bash
# 方式一：改配置（推荐）
vim g1_wall_spray/station/env.json     # "robot_mode": "sim"
./启动工作站.sh

# 方式二：直接用备份包
unzip backup/仿真版_20261008/g1_wall_spray_station_仿真版.zip -d /tmp/sim版
cd /tmp/sim版/g1_wall_spray_station && ./启动工作站.sh
```


## 六、急停与安全操作（两级设计）

**设计原则：急停 = 全身变软（阻尼），不是刚性归零；要退出程序需再按一个键。**

官方机制（已核对源码并重建部署二进制）：
- 软急停 → 控制线程持续下发 **零力矩阻尼指令**（`kp=0, kd=8`）：关节零位置刚度但有阻尼，
  机器人随外力可动（"变软"）且不会失控摆动；**程序继续运行**
- 退出 → 走官方优雅退出：先发阻尼指令，再结束进程

### 按键表

| 操作 | 手柄（`robot_input_type=gamepad_manager`，推荐） | 键盘（`robot_input_type=zmq`） |
|---|---|---|
| 启动控制 | **Start** | `]` |
| 切换到 ZMQ 动作流 | **F1**（切换流模式） | `ENTER` |
| **软急停（全身变软）** | **F2** | `O` |
| **退出程序** | **Select** | `X` |
| 重置朝向 | — | `I` |
| 看电机温度 | — | `F` |
| 手柄/键盘模式切换 | — | `@`（手柄）/ `#`（ZMQ） |

### 网页按钮

右上角常驻两个按钮（任何状态都能按）：

| 按钮 | 作用 | 结果 |
|---|---|---|
| **急停 SOFT STOP**（红，脉动） | 机器人全身变软（阻尼保护态） | 程序仍在运行；按钮变橙提示已软停 |
| **退出程序**（深红） | 结束部署进程 | 需重新点【启动机器人】恢复 |

### 命令行 / 值守

```bash
# 软急停（默认）
python scripts/exec/estop.py

# 退出程序
python scripts/exec/estop.py --hard

# 值守热键：按 ESC / q / 空格 立即软急停
python scripts/exec/estop.py --watch

# 任意程序触发：创建该文件即软急停
touch station/estop.trigger     # 需先跑 python estop.py --daemon
```

### 恢复流程

软急停后机器人是软的（无法自己站起来）：
1. 网页点【退出程序】（或手柄 Select）结束部署进程
2. 扶稳机器人（或让机器人重新上电/复位）
3. 网页点【启动机器人】→ Start 启动控制 → F1/ENTER 切 ZMQ 流 → 重新执行

### 安全须知（务必遵守）

1. **值守时手柄放手边**：F2 软急停最快（一次按键）；网页按钮次之；命令行 `estop.py` 兜底
2. 首次实机务必先 `robot_mode=sim` 验证轨迹，或空跑低速
3. 身边留人扶稳：软急停后机器人会因自重下坠/坐倒（这是"变软"的固有含义）
4. 网线掉线/机器人掉电时部署进程会失联 → 立即按 F2/急停并【退出程序】
5. 喷涂作业时注意喷枪与墙面/管道净距

## 七、网线连通性诊断（实机第一步，务必先做）

实机控制的前提是**能收到机器人数据**。按下面顺序排查：

```bash
# 1) 链路是否协商成功
ethtool enp131s0 | grep -E "Speed|Duplex|Link detected"
cat /sys/class/net/enp131s0/carrier      # 必须为 1

# 2) 关键判据：网卡有没有收到过包
cat /sys/class/net/enp131s0/statistics/rx_packets   # 必须 > 0 且持续增长
cat /sys/class/net/enp131s0/statistics/rx_errors

# 3) 机器人是否可达
ping -c 3 192.168.123.164      # G1 机载计算机常见地址
ssh unitree@192.168.123.164
```

### 判读表

| 现象 | 含义 | 处理 |
|---|---|---|
| `Link detected: no` / `carrier=0` | 网线未插好或对端未激活 | 插紧网线；确认机器人上电 |
| **`Speed: 10Mb/s, Duplex: Half`** | **网线只有部分线对通**（正常应为 1000Mb/s Full） | **换一根 Cat5e/Cat6 八芯网线** |
| **`rx_packets = 0`（只发不收）** | 链路起来了但收不到任何数据 | 换线；确认插的是 G1 的调试网口；确认机器人网络接口已启用 |
| `rx_packets` 增长但 ping 不通 | 网络通但 IP 不对 | 扫描网段找实际 IP（`for i in $(seq 1 254); do ping -c1 -W1 192.168.123.$i; done`） |
| ping 通但 ssh 拒绝 | 机器人未开 SSH 服务 | 用机器人本体终端/Unitree App 开启 |
| 部署进程刷 `LowState is not available` | **收不到机器人状态**（DDS 无数据） | 回到上表，先解决链路 |

> 实测记录（2026-10-08）：本机曾出现 `Link detected: yes` 但 `Speed: 10Mb/s Half`、
> `rx_packets=0`、`tx_errors=39` —— 链路协商成功但零收包，典型的**网线线对不全/损坏**特征，
> 需更换网线后重试。

## 八、通用安全提示（补充）

1. **必须有人守在终端旁，手放在 `O` 键上**——这是唯一的软件急停
2. 首次运行建议：先把 `station/env.json` 的 `robot_mode` 保持 `sim` 验证轨迹，
   确认无误再切实机
3. 机器人周围留出活动空间；喷涂作业时注意喷枪与墙面/管道距离
4. 网线断开或机器人掉电时，部署进程会失联——此时应立即按 `O` 并停止网页任务
5. 腰限位档位（±5°/10°/15°/20°）决定喷涂带宽与稳定性，实机首次建议用 ±5°

## 九、常见问题

| 现象 | 原因/处理 |
|---|---|
| 「启动机器人」报"网卡/链路"相关错误 | 网线未接或 `carrier=0`；检查物理连接与机器人上电 |
| 部署进程报找不到机器人 | 机器人未进调试模式，或 IP 不在 192.168.123.x |
| 机器人不动 | 终端里没按 `]` 或没按 `ENTER` 切 ZMQ 流模式 |
| 动作有但机器人滞后 | 检查 ZMQ 是否丢包（`--zmq-verbose`），或降低 `--rate` |
| 想回到仿真验证 | 见第五节 |

