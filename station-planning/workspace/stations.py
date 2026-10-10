"""舱室喷涂站位表 —— 单一数据源 (规划/校验脚本共用)

本文件是站位坐标 / 作业顺序 / 覆盖带规则参数的唯一权威定义,
fig_topview / check_candidates / check_cross / check_dr / check_zones / opt2 / solve_stiff
等脚本一律 `from stations import ...`, 禁止再各自硬编码站位数值。

!! 与《喷涂工位规划算法设计方案V2.0》存在系统性差异, 详见
   ../docs/站位表与设计方案V2.0差异说明.md:
   - 筋左定点偏移: 本表 = 筋位置 - 0.24 (左墙 L1/L3/L5, 右墙 R4/R6/R8), PDF = 筋 - 0.40;
   - L8 = 4.15, PDF = 4.33;
   - 墙角站位: 本表为对角内收 (距两墙各 0.40m, 45°面向墙角), PDF 为所在墙上距墙端 0.40m;
   - 作业顺序: 本表为"局部交换"序 (墙角站位插入墙面中间消除交叉), PDF 为墙面分组序
     (经交叉检测, PDF 序在本表及 PDF 自身坐标下均有 3 处走位交叉, 见差异说明);
   - 覆盖带规则: 本表为相邻覆盖带互不重叠、中点截断相接, PDF 为相邻覆盖带重叠衔接。
   结论: 待设计侧重新审定后刷新本表。

坐标约定:
  - 房间内净尺寸 RX(宽,x) × RY(长,y), 原点 = 左墙-前墙-地面交点, x 向右、y 向后、z 向上;
  - 各墙 u 为沿墙坐标: 左/右墙 u=y 从前墙角起, 前/后墙 u=x 从左墙角起;
  - 墙面站位位于距墙 d_r 处 (左墙 x=d_r, 右墙 x=RX-d_r, 前墙 y=d_r, 后墙 y=RY-d_r)。

覆盖带规则 (脚本现行规则):
  - 单工位可达范围 = [u-0.20, u+0.60] (站位左 0.20m、右 0.60m, 宽 0.80m, 可达区右偏非对称);
  - 相邻覆盖带互不重叠、相接铺满整墙: 边界 = 两站位中点, 截断至可达范围交集
    (>= 右站位 u-0.20, <= 左站位 u+0.60);
  - 墙角责任带 = 距墙端 0~0.40m, 由墙角合并站位专责(固定, 不按左0.2右0.6计算);
  - 窗带/门带不喷涂。

局限性声明 (自碰撞未建模): 本站位表的工作空间判据仅基于右臂 7 关节 URDF 正运动学
蒙特卡洛采样(腰扭转±30° + 蹲姿骨盆 0.45~0.96m + 小臂垂直墙面夹角<=23°),
未建模腿-臂 / 双臂越中线等自碰撞约束, 蹲姿相关结论基于理想运动学模型,
实机部署前需补充自碰撞校验或实机验证。
"""

# ===== 房间几何与工艺参数 =====
RX, RY = 3.228, 5.058      # 房间内净尺寸 (m, 点云实测)
D_R = 0.80                 # 机器人中心距墙 (m)
D_G = 0.40                 # 腕部(喷枪安装)距墙 (m)

# ===== 覆盖带规则参数 (中点截断规则) =====
BAND_LEFT, BAND_RIGHT = 0.20, 0.60   # 单工位可达: 站位左侧 0.20m / 右侧 0.60m
BAND_WIDTH = BAND_LEFT + BAND_RIGHT  # 0.80m, 站位间距上限
CORNER_W = 0.40                      # 墙角责任带宽 (距墙端 0~0.40m)
RIB_LEFT_OFFSET, RIB_RIGHT_OFFSET = 0.24, 0.20  # 本表筋定点: 筋左-0.24 / 筋右+0.20 (PDF 为 0.40/0.20)

# ===== 场景特征 (各墙 u 坐标, 点云实测) =====
RIBS = {"left": [0.85, 1.69, 2.53],
        "right": [1.69, 2.53, 3.37, 4.21],
        "front": [2.41], "back": [2.41]}
WINDOWS = {"left": [(3.49, 4.10)], "right": [(1.98, 4.01)],
           "front": [(0.91, 1.51)], "back": [(0.91, 1.51)]}
DOOR = (0.41, 1.23)        # 右墙门带 (不喷涂)

# ===== 墙面站位表: (编号, u, 类型) =====
# 类型: stiff=加强筋定点, normal=普通站位
STATIONS = {
    "right": [("R2", 1.29, "stiff"), ("R3", 1.89, "stiff"), ("R4", 2.29, "stiff"),
              ("R5", 2.73, "stiff"), ("R6", 3.13, "stiff"), ("R7", 3.57, "stiff"),
              ("R8", 3.97, "stiff"), ("R9", 4.41, "stiff")],
    "back": [("B1", 0.60, "normal"), ("B2", 1.40, "normal"),
             ("B3", 2.01, "stiff"), ("B4", 2.61, "stiff")],
    "left": [("L1", 0.60, "stiff"), ("L2", 1.05, "stiff"), ("L3", 1.45, "stiff"),
             ("L4", 1.89, "stiff"), ("L5", 2.29, "stiff"), ("L6", 2.73, "stiff"),
             ("L7", 3.53, "normal"), ("L8", 4.15, "normal")],
    "front": [("F2", 0.60, "normal"), ("F3", 1.40, "normal"),
              ("F4", 2.01, "stiff"), ("F5", 2.61, "stiff")],
}

WALL_LEN = {"left": RY, "right": RY, "front": RX, "back": RX}

# ===== 墙角合并站位 (对角内收: 距两墙各 0.40m, 45°面向墙角) =====
CORNERS = {"C_FL": (0.40, 0.40),            # 前-左墙角
           "C_FR": (RX - 0.40, 0.40),       # 前-右墙角
           "C_BL": (0.40, RY - 0.40),       # 后-左墙角
           "C_BR": (RX - 0.40, RY - 0.40)}  # 后-右墙角
CORNER_DIR = {   # 45°面向墙角的方向向量
    "C_FL": (-0.7071, -0.7071),
    "C_FR": (0.7071, -0.7071),
    "C_BL": (-0.7071, 0.7071),
    "C_BR": (0.7071, 0.7071),
}

# ===== 全局作业顺序 =====
# 采纳序 = "局部交换": 墙角站位插入墙面中间(局部交换)以消除走位交叉
# (B4→C_BR→B3, B2→C_BL→B1, F2→C_FL→F3), 已由 check_cross 验证 0 交叉
ORDER = ["R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "B4", "C_BR",
         "B3", "B2", "C_BL", "B1",
         "L8", "L7", "L6", "L5", "L4", "L3", "L2", "L1", "F2", "C_FL",
         "F3", "F4", "F5", "C_FR"]
# 对照序 = "原始墙面分组": 各墙站位走完后经墙角站位转入下一墙 (D_R=0.80 时有 3 处交叉;
# opt2 的 2-opt 输出的是另一种无交叉序(13.37m), 与 ORDER(局部交换序, 14.11m)不等价,
# 两者差异及未采纳 2-opt 序的原因见 ../docs/站位表与设计方案V2.0差异说明.md §3.4)
ORDER_RAW = ["R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "C_BR",
             "B4", "B3", "B2", "B1", "C_BL",
             "L8", "L7", "L6", "L5", "L4", "L3", "L2", "L1", "C_FL",
             "F2", "F3", "F4", "F5", "C_FR"]


# ===== 查询函数 =====
def station_xy(wall, s, d_r=D_R):
    """墙面站位 (name, u, type) -> 房间坐标 (x, y)"""
    _, u, _ = s
    if wall == "left":
        return d_r, u
    if wall == "right":
        return RX - d_r, u
    if wall == "front":
        return u, d_r
    return u, RY - d_r


def get_wall(name):
    """站位编号(墙面站位或墙角站位) -> 所属墙面 (墙角站位返回 None)"""
    for wall, lst in STATIONS.items():
        for s in lst:
            if s[0] == name:
                return wall
    if name in CORNERS:
        return None
    raise KeyError(name)


def get_type(name):
    """站位编号 -> 类型 (stiff/normal/corner)"""
    if name in CORNERS:
        return "corner"
    for lst in STATIONS.values():
        for s in lst:
            if s[0] == name:
                return s[2]
    raise KeyError(name)


def get_xy(name, d_r=D_R):
    """站位编号 -> 房间坐标 (x, y)"""
    if name in CORNERS:
        return CORNERS[name]
    for wall, lst in STATIONS.items():
        for s in lst:
            if s[0] == name:
                return station_xy(wall, s, d_r)
    raise KeyError(name)


def get_dir(name):
    """站位编号 -> 机器人朝向单位向量 (墙面站位正对墙面, 墙角站位 45°面向墙角)"""
    if name in CORNER_DIR:
        return CORNER_DIR[name]
    wall = get_wall(name)
    return {"left": (-1, 0), "right": (1, 0),
            "front": (0, -1), "back": (0, 1)}[wall]


def wall_zones(wall):
    """墙面站位的中点截断覆盖带 (脚本现行规则)

    返回 [(站位名, u, z0, z1)], 按沿墙 u 排序; 墙角责任带(距墙端 0~CORNER_W)固定,
    墙面站位对边界 = 两站位中点, 截断至可达范围交集(>=右站位u-0.20, <=左站位u+0.60)。
    注: 本规则适用于左/前/后墙; 右墙门带 [0.41,1.23] 与首站位 R2 带之间的地面
    不作相接校验(门带不喷涂), 见差异说明文档。
    """
    lw = WALL_LEN[wall]
    st = sorted(STATIONS[wall], key=lambda s: s[1])
    us = [s[1] for s in st]
    names = [s[0] for s in st]
    bs = [0.00, CORNER_W]
    for i in range(len(st) - 1):
        a, b = us[i], us[i + 1]
        mid = (a + b) / 2
        lo = b - BAND_LEFT
        hi = a + BAND_RIGHT
        bs.append(round(min(max(mid, lo), hi), 4))
    bs += [lw - CORNER_W, lw]
    # 墙面站位 i 的覆盖带 = [bs[i+1], bs[i+2]] (两端 [0,CORNER_W]/[lw-CORNER_W,lw] 为墙角责任带)
    return [(names[i], us[i], bs[i + 1], bs[i + 2]) for i in range(len(st))]
