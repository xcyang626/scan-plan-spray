"""左墙站位 LP 求解 —— 约束按《喷涂工位规划算法设计方案V2.0》§2.3-2.4 编码

规则(以左墙为对象, F1 为前墙角落站位, 负责左墙前端墙角责任带):
  (1) 角落站位 F1: 距墙端 40cm (u=0.40), 负责墙角责任带 [0, 0.40];
  (2) 加强筋定点: 筋左定点 u = 筋中心 - 0.40, 筋右定点 u = 筋中心 + 0.20 (等式约束);
  (3) 普通补齐站位: 列左端(=前一覆盖带右端) + 0.20, 即 L7 = L6+0.80, L8 = L7+0.80
      (等式约束, 覆盖带 [u-0.20, u+0.60] 依次重叠衔接);
  (4) 无洞约束(相邻覆盖带重叠衔接, 宽度按带宽 0.80): 相邻站位间距 <= 0.80;
      角带与 L1 带重叠衔接: L1 - 0.20 <= 0.40; L8 带与 L9 角带重叠衔接: L8 + 0.60 >= L9-0.40;
  (5) 目标: 最小与规则定点位置的偏差。筋定点/补齐规则为 PDF 定义性等式约束,
      解出即为规则表值 —— LP 在此作为"规则 => 表格"的对齐验证。
      (若去掉定点等式仅用最小走位目标, 解会向左塌缩, 无法复现 PDF 表;
       PDF 表值来源于筋定点固定偏移规则而非走位优化。)

历史问题修复:
  - B5: 删除 F1-L1 >= 0.4 最小间距约束 —— PDF 中 F1 角带 [0,0.40] 与 L1 带
    [0.25,1.05] 重叠衔接, 两者不冲突; 旧约束使 PDF 方案在 LP 中不可行,
    旧解 L1=F1+0.4=0.60 正是脚本版坐标的来源;
  - B6: 旧 `add(-P("L1"), 0.25)` 实为 L1 >= -0.25 (无效下界); 按 PDF 表 L1=0.45,
    正确编码是 L1 = 筋1 - 0.40 = 0.45 等式(带 [0.25,1.05] 的唯一站位), 已取代;
  - 删除旧的成对重复约束与各对站位的 >=0.4 间距约束(与重叠衔接规则冲突)。
"""
import numpy as np
from scipy.optimize import linprog

from stations import RY, RIBS, BAND_LEFT, BAND_RIGHT, BAND_WIDTH, CORNER_W

# PDF 规则的筋定点偏移 (左 0.40/右 0.20)。注意: stations.py 现行脚本表的筋左偏移为 0.24,
# 与 PDF 规则不同 —— 本 LP 编码的是 PDF §2.3 规则, 用于验证"PDF 规则 => PDF 表格",
# 两者差异见 ../docs/站位表与设计方案V2.0差异说明.md。
PDF_RIB_LEFT, PDF_RIB_RIGHT = 0.40, 0.20

LW = RY                       # 左墙长
ribs = RIBS["left"]           # 筋中心线 u = 0.85 / 1.69 / 2.53
L9 = LW - CORNER_W            # 左墙后端墙角责任带站位 (PDF: L9, 融合旧B1角站)

names = ["F1", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"]
n = len(names)
idx = {k: i for i, k in enumerate(names)}
N = n + n                     # 前 n 个: 站位 u; 后 n 个: 偏差绝对值

# 规则定点位置 (偏好): F1 角站 0.40; 筋定点 筋±偏移; L7/L8 按 L6 链式补齐
pref = {"F1": CORNER_W,
        "L1": ribs[0] - PDF_RIB_LEFT, "L2": ribs[0] + PDF_RIB_RIGHT,
        "L3": ribs[1] - PDF_RIB_LEFT, "L4": ribs[1] + PDF_RIB_RIGHT,
        "L5": ribs[2] - PDF_RIB_LEFT, "L6": ribs[2] + PDF_RIB_RIGHT}
pref["L7"] = pref["L6"] + BAND_WIDTH          # 3.53 = 3.33(列左端) + 0.20
pref["L8"] = pref["L7"] + BAND_WIDTH          # 4.33 = 4.13(列左端) + 0.20

# PDF §2.3.2 表值 (用于对齐验证)
PDF_TABLE = {"F1": 0.40, "L1": 0.45, "L2": 1.05, "L3": 1.29, "L4": 1.89,
             "L5": 2.13, "L6": 2.73, "L7": 3.53, "L8": 4.33}

c = np.zeros(N)
c[n:] = 1.0                                # 目标: 最小总偏差
c[:n] += 1e-6                              # 次目标: 站位尽量靠左(消多重解)

A, b = [], []


def add(le, rhs):
    A.append(le); b.append(rhs)


def P(k):
    e = np.zeros(N); e[idx[k]] = 1.0
    return e


# 偏差定义: |u - pref| <= d
for k in names:
    i = idx[k]
    e = np.zeros(N); e[i] = 1; e[n + i] = -1; add(e, pref[k])
    e = np.zeros(N); e[i] = -1; e[n + i] = -1; add(e, -pref[k])

# (1) 角站: F1 = 0.40 (距墙端 40cm)
add(P("F1"), CORNER_W); add(-P("F1"), -CORNER_W)

# (2) 筋定点等式: 筋左 = 筋-0.40, 筋右 = 筋+0.20
for r, (lk, rk) in zip(ribs, [("L1", "L2"), ("L3", "L4"), ("L5", "L6")]):
    add(P(lk), r - PDF_RIB_LEFT); add(-P(lk), -(r - PDF_RIB_LEFT))
    add(P(rk), r + PDF_RIB_RIGHT); add(-P(rk), -(r + PDF_RIB_RIGHT))

# (3) 补齐等式: 下一列左端(=前一覆盖带右端) + 0.20
for a, bb in [("L6", "L7"), ("L7", "L8")]:
    add(P(bb) - P(a), BAND_WIDTH); add(P(a) - P(bb), -BAND_WIDTH)

# (4) 无洞(重叠衔接): 相邻站位间距 <= 带宽 0.80; 角带两侧重叠衔接
seq = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"]
for a, bb in zip(seq, seq[1:]):
    add(P(bb) - P(a), BAND_WIDTH)
add(P("L1"), CORNER_W + BAND_LEFT)          # F1 角带右端 0.40 >= L1 带左端 L1-0.20
add(-P("L8"), -(L9 - BAND_RIGHT))              # L9 角带左端 4.66 <= L8 带右端 L8+0.60

res = linprog(c, A_ub=np.array(A), b_ub=np.array(b),
              bounds=[(None, None)] * N, method="highs")
if not res.success:
    raise SystemExit("LP 无解: " + res.message)

x = res.x
pos = {k: x[idx[k]] for k in names}
dev = sum(abs(pos[k] - pref[k]) for k in names)

print("=== 左墙站位 LP 解 (PDF §2.3-2.4 规则) ===")
print(f"总偏差={dev:.4f} (规则等式约束下应为 0)")
ok = True
for k in names:
    d_rib = ""
    if k in ("L1", "L2"):
        d_rib = f"  距筋1={pos[k] - ribs[0]:+.2f}"
    elif k in ("L3", "L4"):
        d_rib = f"  距筋2={pos[k] - ribs[1]:+.2f}"
    elif k in ("L5", "L6"):
        d_rib = f"  距筋3={pos[k] - ribs[2]:+.2f}"
    diff = pos[k] - PDF_TABLE[k]
    ok = ok and abs(diff) <= 0.02
    print(f"  {k}: {pos[k]:.4f}  (PDF表 {PDF_TABLE[k]:.2f}, 差 {diff:+.3f}){d_rib}")

st = [pos[k] for k in seq]
gaps = [st[i + 1] - st[i] for i in range(len(st) - 1)]
print("L1..L8 间距:", " ".join(f"{g:.2f}" for g in gaps),
      f"(全部<=0.80: {all(g <= BAND_WIDTH + 1e-9 for g in gaps)})")
print(f"F1 角带 [0,{CORNER_W:.2f}] 与 L1 带 [{pos['L1'] - BAND_LEFT:.2f},{pos['L1'] + BAND_RIGHT:.2f}] "
      f"重叠衔接: {pos['L1'] - BAND_LEFT <= CORNER_W + 1e-9}")
print(f"L8 带 [{pos['L8'] - BAND_LEFT:.2f},{pos['L8'] + BAND_RIGHT:.2f}] 与 L9 角带 "
      f"[{L9:.2f},{LW:.2f}] 重叠衔接: {pos['L8'] + BAND_RIGHT >= L9 - 1e-9}")
print(f"左墙内走位(L1->L9): {L9 - pos['L1']:.2f} m")
print("\n判定: LP 解", "≈ PDF 表格值(容差±0.02) -> 规则对齐成功" if ok else
      "与 PDF 表格值偏差超限 -> 规则编码需复核")
