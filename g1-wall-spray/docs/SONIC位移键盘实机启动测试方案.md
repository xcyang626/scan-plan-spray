# SONIC 位移/速度/朝向 键盘控制 —— 实机启动测试方案

首次把 `sonic_loco_keyboard.py`（仿真已闭环验证，见版本迭代记录）接到物理 G1。
本方案按"先通道后组合、先小幅后放大"组织，每阶段有明确验收线，不过线不进下一阶段。

**测试对象**：SONIC 全身策略的 planner 通道——movement（x/y 位移方向）、speed（位移速度）、
facing（朝向），经键盘 30Hz 连续发布（zmq_manager 输入端，1s 静默自动回 IDLE）。

---

## 一、人员、场地、角色

| 角色 | 职责 |
|---|---|
| 扶机器人者 ×1 | 站机器人侧后方，轻扶骨盆两侧（**不发力**，只防摔）；接管成功后按指令松手 |
| 键盘操作者 ×1 | 独占键盘工具终端 + 部署终端；一只手始终悬在 `z`/`o` 上 |
| 红线权限 | 任何人喊"停" = 立即 `o` |

- 场地：机器人置于 ≥2m×2m 空区中央，地面平整干燥；预划边界线，机器人越线即 `z`。
- **喷枪/软管/线缆全部拆除**，纯空载行走测试（上肢无指令，跟策略自然摆动，持物危险）。

## 二、启动顺序（顺序关键，勿颠倒）

> 依据：station app 注释——**发布端必须先 bind 5556 再起部署**（部署 SUB connect 过来，
> 无慢加入，`start` 命令不会被 ZMQ 丢掉）。部署一启动就进 INIT（3s 把机器人从当前位形
> 拉到默认站姿），所以**机器人必须先摆好**。

### 阶段 0：前置检查（不动机器人）

```bash
# 1) 链路：carrier 必须 1，速度必须 1000Mb/s Full（10Mb/s Half = 换线）
cat /sys/class/net/enp131s0/carrier
ethtool enp131s0 | grep -E "Speed|Duplex|Link"
cat /sys/class/net/enp131s0/statistics/rx_packets   # >0 且持续增长
ping -c 3 192.168.123.161                            # G1（若不通试 .164 机载口）
# 2) 无残留进程、5556 端口空闲（station app 不要开）
pgrep -af "g1_deploy_onnx_ref|groot_daemon|sonic_loco"   # 应为空
ss -ltnp | grep 5556                                      # 应为空
```

机器人：上电、进调试模式、**双脚自然站立**（不许吊装、不许瘫姿、不许人架着发力）。

### 阶段 1：起键盘工具（先 bind 5556）

终端 A（键盘工具，首跑收紧：上限 0.4 m/s、朝向 ±45°。不带这些参数时默认即官方语义
0.2~0.8 m/s、±180°全向）：

```bash
cd ~/桌面/scan-plan-spray/g1-wall-spray/scripts/exec
~/桌面/轨迹规划/GR00T-WholeBodyControl/.venv_sim_cpu/bin/python sonic_loco_keyboard.py \
  --host 127.0.0.1 --port 5556 \
  --speed 0.2 --max-speed 0.4 --max-heading-deg 45
```

看到 `ZMQ PUB 绑定 tcp://127.0.0.1:5556` 即可，**先不要按任何键**。

### 阶段 2：起部署进程（含接管混合 2s）

终端 B（部署端。带 `--takeover-blend-s 2.0`；**不传 `--disable-crc-check`**——那是仿真专用）：

```bash
cd ~/桌面/轨迹规划/GR00T-WholeBodyControl/gear_sonic_deploy
export LD_LIBRARY_PATH="/home/yang/桌面/轨迹规划/GR00T-WholeBodyControl/.deps/TensorRT/lib:/home/yang/桌面/轨迹规划/GR00T-WholeBodyControl/.deps/onnxruntime/lib:/home/yang/桌面/轨迹规划/GR00T-WholeBodyControl/.venv_sim_cpu/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/home/yang/桌面/轨迹规划/GR00T-WholeBodyControl/gear_sonic_deploy/thirdparty/unitree_sdk2/thirdparty/lib/x86_64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
./target/release/g1_deploy_onnx_ref enp131s0 \
  policy/low_latency/model_decoder.onnx reference/example \
  --obs-config policy/low_latency/observation_config.yaml \
  --encoder-file policy/low_latency/model_encoder.onnx \
  --planner-file planner/target_vel/V2/planner_sonic.onnx \
  --input-type zmq_manager --output-type zmq \
  --zmq-host localhost --zmq-port 5556 \
  --takeover-blend-s 2.0
```

观察点（终端 B）：
1. INIT 3 秒：机器人自动屈膝站到默认站姿。**此阶段若剧烈抖动** → 判读表 R1。
2. 打印 `Init Done`。扶机器人者此时轻扶即可，机器人应自己站住。
3. `Loop timing - LowState age` 应为个位数 ms 且不增长（DDS 健康）。

### 阶段 3：启动控制（策略接管）

终端 A 按 `]`。观察终端 B：
- `[ZMQManager] Planner enabled` + `motion name is planner_motion`
- **`[G1W-blend] engaged: 2s from measured pose`** → 2 秒内从 0% 到 100% 平滑接管（每秒打一行进度）

**验收线**：接管过程无猛抖；扶者逐步松手后，**空载站稳 ≥10s**，基座无可见晃动、无持续高频抖动。
不过线 → 判读表 R2，处理后从阶段 3 重来。

### 阶段 4：单通道测试（一次一个通道，每步间停稳 ≥3s）

| # | 操作（终端 A） | 预期 | 过线标准 |
|---|---|---|---|
| 4.1 | 按住 `w` 1 秒松开 | 前进约 0.10 m 后停住 | 位移 0.05~0.20m，1s 内停稳，无晃 |
| 4.2 | 按住 `s` 1 秒 | 后退同量 | 同上 |
| 4.3 | 按住 `a` 1 秒 | 左移同量（侧步） | 同上 |
| 4.4 | 按住 `d` 1 秒 | 右移同量 | 同上 |
| 4.5 | `e` 按一下（+5°） | 原地小幅左转 | 转角肉眼 ≈5°，无平移窜动 |
| 4.6 | `q` 按一下（−5°） | 转回 | 回到原朝向 ±3° |
| 4.7 | `0` 按一下（0.10→0.15）再 `w` 1 秒 | 前进速度明显加快 | 位移约 0.15m；`9` 降回 0.10 |

全部过线才进阶段 5。任何一步出现抖动/失稳：立即 `z`（只停位移）；仍抖 → `o`。

### 阶段 5：组合与放大

| # | 操作 | 预期 |
|---|---|---|
| 5.1 | `e`×4（+20°）→ 等 3s → `w` 2 秒 | 转身后朝 20° 方向前进 ~0.2m |
| 5.2 | `w`+`a` 同时按住（斜向）2 秒 | 前左斜向 ~0.25m |
| 5.3 | `w` 连续 3 秒 | 直线 ~0.3m，路径无明显画弧 |
| 5.4 | `0` 升到 0.20 → `w` 2 秒 | 明显更快，仍可控；测毕 `9` 降回 |
| 5.5 | `r` 朝向归零 | 原地缓慢转回初始朝向 |

**放大纪律**：以上全绿后，才允许放开 CLI 限幅做第二轮（当前工具默认已是官方语义：
速度 0.2~0.8 m/s、朝向 ±180° 全向）；任何时候幅值感觉"比仿真里大"就 `z` 回来。

### 阶段 6：关停顺序

1. 终端 A 按 `z` → 确认站稳（≥3s）；
2. 终端 A 按 `o`（**停止按钮**：IDLE + 官方 stop×3 → 机器人全身变软 + 部署优雅退出 + 键盘工具自身退出，一步完成；**扶机器人者此时必须扶住**）。终端 B 的 `O` 等效（只退部署）；
3. `pgrep -af "g1_deploy_onnx_ref|sonic_loco"` 确认两个进程都清干净；
4. 留档：终端 B 日志全文（`tee` 到文件更佳）、终端 A 的状态行输出、异常时的时间点笔记。

---

## 三、异常判读表

| 编号 | 现象 | 最可能原因 | 处置 |
|---|---|---|---|
| R1 | INIT 阶段（阶段 2）剧烈抖动 | 机器人没站稳/瘫姿被拉站姿/有人发力对抗；或实机增益不匹配 | 终端 B 按 `O`；重新摆位再来。两轮仍抖：下次启动加 `--motor-kd-scale 12-14=0.7`（腰三关节先软化） |
| R2 | `]` 接管瞬间猛抖 | 硬切换残留（确认日志有 `[G1W-blend] engaged`；没有=blend 参数没生效，检查命令）；或接管时被扶着歪/朝向异常 | `o`；摆正重新 `]`（zmq start 自动重置 IMU 朝向）。反复抖 → `--takeover-blend-s 3.0` 加长 |
| R3 | 运行中持续高频抖（>1s） | 增益与实机电机批次不匹配；网线质量（10Mb/s Half 前科） | `z` 无效就 `o`；查 `ethtool`；下次加腰 KD 缩放 |
| R4 | 机器人自己开始"舞蹈"（上肢大幅摆） | planner 静默 >1s（键盘工具死了/终端 A 被 Ctrl-Z/断网）没发 stop | 终端 B 立即 `O`。恢复流程见下 |
| R5 | 部署端刷 `LowState is not available` | 网线/机器人掉线 | `O` 退出，查链路（阶段 0 命令重跑） |
| R6 | `Address already in use`（终端 A 或 B） | 5556 被占（station app/残留进程） | 清进程（阶段 0 第 2 条），station app 与本测试**互斥** |

**软急停后恢复**（R1~R4 处理后重测）：扶稳机器人 → 终端 B 确认进程已退（没退则 Ctrl-C）→
按阶段 2 重新起部署（INIT 会重新爬站姿）→ 阶段 3 重新 `]`。

## 四、红线（任一出现，立即 `o`，不犹豫）

1. 任何持续超过 1 秒的抖动；
2. 机器人越出 2m 安全区或明显失稳；
3. 任何人进入机器人 1m 范围；
4. 终端 B 出现 `Safety check failed` / `Lost LowState` / 推理报错；
5. 键盘工具终端 A 无响应/被误关。

## 五、急停通道的真实可用性（重要，防误信）

- **首选：终端 B（部署终端）按 `O`** —— zmq_manager 模式下部署进程仍读 stdin，`O`=官方软急停（零力矩阻尼 + 优雅退出）。**键盘工具卡死时这是唯一软件通道**，键盘操作者请把终端 B 放在手边。
- 次选：终端 A 按 `o`（停止按钮：IDLE + stop×3，机器人变软 + 部署退出 + 工具自身退出，一步到位）。
- **注意**：`estop.py` 的 ZMQ 直发通道需要 bind 5556，键盘工具占用期间会失败，其 FIFO 通道只在 station app 启动的部署上存在——**本测试流程里 estop.py 不可靠，README 里的 `estop.py --watch` 模式实际不存在**（脚本只有 `--keys/--temp/--reinit`）。物理急停/机器人本体急停永远最后兜底。
- 5557 `g1_debug` 流在实机上 `base_quat_measured`（IMU 四元数）是活的，可用于事后复盘朝向；`base_trans_measured` 是固定值，**不能**用来核实位移——实机位移靠场地标记/录像核对。

## 六、与既有系统的关系

- 本测试**绕开 station app**（其 daemon 会占 5556 并挂上肢保持流，与键盘工具冲突）；测完想回喷涂流程：确认本测试进程全清 → 正常开工作站。
- 喷涂的 planner 上肢流（`upper_body_position`）与键盘工具的纯运动流走同一 planner 话题、
  互斥使用；后续若要"边走边喷"，需在 daemon 侧把 movement/speed/facing 并入上肢流——那是下一步的设计题，不在本方案内。
