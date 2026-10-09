# OpenArm 左臂 ROS2 仿真 —— 之字形走线任务 Plan

> 目标：把实机标定结果（`标定/zero`，左臂，2026-09-02）导入 ROS2 仿真，在"基座正前方一堵墙、
> 夹爪垂直于墙、TCP 距基座水平距离 30cm"的约束下计算工作区间，在区间内选一个长方形走之字形，
> 夹爪 TCP 跟踪该轨迹，关节逆解连续、运动平滑稳定。
>
> 形态决议（已确认）：**RViz 运动学仿真**（无物理引擎）；IK 用**自研数值 IK**（纯 numpy）。
> 每个阶段完成后停下确认，再进入下一阶段。
>
> **阶段2 修订（2026-09-02）**：研究对象 = 面向正面时位于左侧的臂 = URDF `openarm_right_*`
> 链（基座有支撑的一面为正面）；墙位于基座前方 **2m**（半透明，URDF link 渲染）；
> TCP 工作平面 = 基座前方 **0.3m**（与墙平行，独立 URDF link）；初始位形 = URDF 全零
> （手臂自然下垂，`initial_pose:=zero_file` 可切回标定手摆位形）。

---

## 总体架构

```
仿真/
├── PLAN.md                        ← 本文件
└── openarm_sim/                   ← ROS2 Python 包（ament_cmake，纯 Python）
    ├── config/
    │   ├── sim_params.yaml        # 场景/路径/IK/平滑 全部可调参数
    │   └── v1.urdf                # 复用标定包的官方 URDF（symlink 或复制）
    ├── launch/
    │   └── zigzag_sim.launch.py   # 一键启动全部节点 + RViz
    ├── openarm_sim/
    │   ├── zero_bridge.py         # 阶段1: zero 文件加载 + 电机系↔URDF系 坐标桥接
    │   ├── kinematics.py          # 阶段3: 纯 numpy FK（从 URDF 解析 joint origin/axis）
    │   ├── ik_solver.py           # 阶段4: DLS 数值 IK + 零空间偏置
    │   ├── trajectory.py          # 阶段5: 时间参数化 + 拐角过渡 + 限幅
    │   ├── workspace.py           # 阶段3: 工作区间扫描脚本节点
    │   └── zigzag_node.py         # 阶段6: 主节点（路径生成→IK→平滑→50Hz 发布）
    └── scripts/
        └── ...
```

数据流：

```
zero 文件 ──► zero_bridge ──► 软限位/零位（URDF 系）
                                  │
kinematics(FK) ──► workspace 扫描 ──► 可达区域 ──► 选长方形 ──► 之字形路径点
                                                                │
zigzag_node: 路径 ──► ik_solver(warm start) ──► trajectory 平滑 ──► /sim/joint_states (50Hz)
                                                                        │
                              rviz_bridge(改造) ──► /urdf_joint_states ──► RViz
                              (URDF + 墙体 Marker + TCP 轨迹线 Marker)
```

---

## 阶段 1：零点数据导入 + 坐标系桥接（zero_bridge.py）

**做什么**
1. 解析 `标定/zero`（JSON）：`zero_position_rad` / `pos_limits_rad` / `neg_limits_rad`（8 关节）。
2. 建立统一的**电机系 ↔ URDF 系**转换层，复用零位桥接实现(`仿真/openarm_sim/zero_bridge.py`；原文引用的 rviz_bridge.py 已在 2026-09-11 整理中删除)
   已验证的满程映射：左臂 J1 偏移 −120°、J2 偏移 −90°，J3–J7 直映射，夹爪 [−60°,0°]→指张 0~44mm。
3. 输出 URDF 系下的：
   - 机械零位角 `q_home_urdf = zero_position + offset`
   - 各关节软限位 `[q_min, q_max]`（供 IK 裁剪）
4. **待验证关键点**：实机 zero 文件 J1 限位 [−4.4024, 0.5068] rad 与官方出厂系 MECH_LIM_V1
   J1 [−80°, 200°] 不吻合，需确认实机 zero 的坐标参考系（出厂编码系 vs 上电相对系）。
   方法：两种假设各做一次换算，把限位区间与 URDF 限位 / MECH_LIM_V1 对照，选出自洽的一套，
   并在 RViz 中把机械零位姿态摆出来人工确认方向正确。

**产出**：`zero_bridge.py` + 一个自检脚本（打印换算表，RViz 摆零位姿态截图确认）。

---

## 阶段 2：场景搭建（墙 + RViz）

**做什么**
1. `zigzag_sim.launch.py`：启动 robot_state_publisher（v1.urdf，mesh 走官方 openarm_description）+
   改造版 rviz_bridge（订阅 `/sim/joint_states`，左臂映射）+ RViz。
2. FK 先行确认 URDF 里"基座正前方"对应哪个世界方向（`openarm_left_link0` 固定于 body
   `rpy=-1.5708 0 0, xyz=0 0.031 0.698`，需实际算一次 TCP 初始朝向），据此定墙的位姿。
3. 墙体 = RViz Cube Marker：竖直平面，位于基座前方 50cm（水平距离，以基座原点起算）；
   另加 TCP 位置球 Marker + 轨迹 LineStrip Marker（走线可视化）。

**产出**：RViz 中能看到机械臂（随手动关节验证桥接）+ 墙 + TCP 标记。

---

## 阶段 3：FK + 工作区间计算（kinematics.py / workspace.py）

**做什么**
1. 纯 numpy FK：启动时从 URDF 解析各 joint 的 origin xyz/rpy、axis，逐级左乘变换；
   TCP 取 `openarm_left_hand_tcp`（注意其与 link7 同原点，实际接触点取指尖平面，
   用 finger 链几何修正一个固定偏移量）。
2. 约束定义（已确认口径）：
   - **TCP 位于基座前方水平距离 50cm 的竖直平面上**（x = 0.5 m 平面，具体轴向待阶段2确认）；
   - **夹爪垂直于墙**：TCP 接近轴对准墙面法向（水平指向基座方向）；
   - 全部关节在阶段1的软限位内。
3. 工作区间扫描：在该竖直平面内网格化（y-z，步长 1cm），对每点用宽松阈值 IK 判可达
   （或反向：关节空间随机采样→FK 落到平面上累积点云，两种取并集更稳）。
   输出可达区域点云图（PNG）+ 边界包络（JSON/YAML）。
4. 在可达区域**内缩安全边距**（建议 5cm）选长方形：默认先定一个 ~20cm×20cm 的候选，
   参数全部进 `sim_params.yaml`（长宽/中心/行距/走线速度可调）。
5. 之字形路径生成：行距默认 3cm，拐点列表输出为路径点序列（拐角处理在阶段5做圆角过渡）。

**产出**：`workspace.svg`（可达区域+选中的长方形+之字形叠加图，现统一归档在
`assets/figures/workspace_<side>.svg`）、路径点 JSON。

---

## 阶段 4：自研数值 IK（ik_solver.py）

**算法**（7 自由度冗余臂，重点：解连续 + 平滑）
- **DLS（阻尼最小二乘）**：`Δq = Jᵀ(JJᵀ + λ²I)⁻¹ e`，λ 抑制奇异位形附近的关节速度尖峰。
- **任务空间**：默认 **5D 约束**——TCP 位置 3D + 接近轴对准墙面法向 2D（只约束"垂直于墙"，
  绕法向的自旋放开），比全 6D 姿态约束更宽松、解空间更大、更平滑；预留 6D 全约束开关。
- **零空间偏置**：`Δq_null = (I − J⁺J) · k · (q_ref − q)`，`q_ref` 取"关节限位中点 +
  上一时刻解的混合"，实现关节限位避让 + 解的时间连续性（防冗余自由度漂移抖动）。
- **warm start**：沿路径逐点求解，上一路径点的解作为初值 → 相邻点解天然连续。
- **限位裁剪 + 收敛判据**：位置误差 < 1mm、轴向角误差 < 1° 即收敛；不收敛点记录并告警。

**验收**：路径全部点 IK 收敛率 100%；相邻点关节解最大跳变 < 0.5°（由路径点密度保证）。

---

## 阶段 5：轨迹平滑与稳定（trajectory.py）

**做什么**
1. **拐角处理**：之字形拐点用二次贝塞尔/圆弧过渡（过渡半径默认 2cm），消除位置阶跃
   导致的关节速度反向尖峰。
2. **时间参数化**：整条轨迹用梯形速度规划（TCP 线速度恒定，默认 5cm/s，可调），
   或五次多项式 S 曲线加减速（更柔，作为选项）。
3. **关节级保障**：按 URDF velocity limit（J1/J2 16.75、J3/J4 5.45、J5–J7 20.94 rad/s）
   限幅；再叠加关节加速度限幅 + 可选一阶低通（截止频率可调），双保险抑制抖动。
4. 重采样到 50Hz 控制周期输出。

**验收指标**：TCP 跟踪误差 < 2mm；无任何关节角速度超限；拐角处关节速度曲线无尖峰（画图确认）。

---

## 阶段 6：集成联调（zigzag_node.py + launch）

**做什么**
1. 主节点串起全链路：加载配置 → 生成路径 → IK → 平滑 → 50Hz 发布 `/sim/joint_states`。
2. 交互/观测：`/zigzag/status` 状态话题（进度/当前点/TCP 误差）、`start`/`pause`/`abort` 服务；
   RViz 同步显示墙体、长方形框、之字形轨迹线、TCP 实时位置。
3. 联调 checklist：
   - [ ] RViz 中 TCP 沿之字形走线，始终垂直墙面，与基座水平距离恒 50cm
   - [ ] 关节运动平滑无跳变、无超限
   - [ ] 调参链路可用：行距/速度/长方形尺寸改 yaml 即生效

**产出**：完整可复现的仿真 demo，一键 `./build.sh && ./run_zigzag_sim.sh`。

---

## 阶段 7（可选扩展）

- 夹爪在走线中周期开合（模拟擦拭/喷涂动作）。
- 多组长方形/多条之字形拼接。
- 轨迹导出为关节序列文件，供后续实机回放桥接（原设想的 arm_dm_control 路线已废弃，脚本存档于 `archive/`，现行真机路线见 `real/`）。

---

## 里程碑与确认点

| 阶段 | 内容 | 确认方式 |
|---|---|---|
| 1 | zero 桥接 + 参考系澄清 | 换算表 + RViz 零位姿态人工确认 |
| 2 | 墙 + RViz 场景 | 截图确认墙位姿与"正前方"方向 |
| 3 | 工作区间 + 长方形 | workspace.svg 人工圈定 |
| 4 | 数值 IK | 收敛率/连续性报告 |
| 5 | 平滑化 | 速度/误差曲线图 |
| 6 | 集成 demo | 全流程 RViz 演示 |

## 已知风险

1. **zero 参考系不确定**（阶段1重点解决，影响软限位正确性）。
2. URDF 的 collision mesh 未加载，纯运动学仿真无自碰检测——用"关节限位硬裁剪 + TCP
   高度下限（防撞基座/地面）"简化防护，如需严格碰撞检测再引入 PyBullet 做预检（可选）。
3. 之字形在平面边缘若接近可达边界，IK 会顶限位变"硬"——靠阶段3的 5cm 安全边距规避。
