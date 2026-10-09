"""对比不同F1取值下的最优站位解 (LP)
F1=0.20: 墙角站位保持现状;  F1=0: 墙角站位靠左(允许L1保持40cm)"""
import numpy as np
from scipy.optimize import linprog

stiff = {1: 0.85, 2: 1.69, 3: 2.53}
pref = {"F1": 0.20, "L1": 0.45, "L2": 1.05, "L3": 1.29, "L4": 1.89, "L5": 2.13, "L6": 2.73}
L7 = 3.595

def solve(F1_fixed=None, tiebreak_left=False):
    names = ["F1", "L1", "L2", "L3", "L4", "L5", "L6"]
    n = len(names)
    idx = {k: i for i, k in enumerate(names)}
    N = n + n
    c = np.zeros(N)
    for i in range(n):
        c[n + i] = 1.0
    if tiebreak_left:
        for i in range(n):
            c[i] += 1e-6  # 次目标: 站位尽量靠左

    A, b = [], []
    def add(le, rhs):
        A.append(le); b.append(rhs)
    def P(k):
        e = np.zeros(N); e[idx[k]] = 1
        return e

    for k in names:
        i = idx[k]
        e = np.zeros(N); e[i] = 1; e[n + i] = -1; add(e, pref[k])
        e = np.zeros(N); e[i] = -1; e[n + i] = -1; add(e, -pref[k])

    pairs = [("F1", "L1"), ("L1", "L2"), ("L2", "L3"), ("L3", "L4"), ("L4", "L5"), ("L5", "L6")]
    for a, c2 in pairs:
        add(P(a) - P(c2), -0.4)

    # 带内
    add(-P("F1"), 0.0); add(P("F1"), 0.40)
    add(-P("L1"), -0.40); add(P("L1"), 0.85)
    add(-P("L2"), -0.85); add(P("L2") - P("L3"), 0.0)
    add(P("L2") - P("L3"), 0.0); add(P("L3"), 1.69)
    add(-P("L4"), -1.69); add(P("L4") - P("L5"), 0.0)
    add(P("L4") - P("L5"), 0.0); add(P("L5"), 2.53)
    add(-P("L6"), -2.53); add(P("L6"), L7)
    # 覆盖
    add(P("F1"), 0.20); add(-P("F1"), 0.20)
    add(P("L1"), 0.60); add(-P("L1"), 0.25)
    add(P("L2"), 1.05); add(P("L3") - P("L2"), 1.2)
    add(P("L3") - P("L2"), 0.4); add(-P("L3"), -1.09)
    add(P("L4"), 1.89); add(P("L5") - P("L4"), 1.2)
    add(P("L5") - P("L4"), 0.4); add(-P("L5"), -1.93)
    add(P("L6"), 2.73); add(-P("L6"), L7 - 1.2)
    # 靠左
    add(P("F1"), 0.20)
    add(P("L1"), 0.625)
    add(3 * P("L2") - P("L3"), 1.7)
    add(3 * P("L3") - P("L2"), 3.38)
    add(3 * P("L4") - P("L5"), 3.38)
    add(3 * P("L5") - P("L4"), 5.06)
    add(3 * P("L6"), 5.06 + L7)
    # 加强筋
    for s, (l, r) in {1: ("L1", "L2"), 2: ("L3", "L4"), 3: ("L5", "L6")}.items():
        for nm in (l, r):
            add(P(nm), stiff[s] + 0.2)
            add(-P(nm), -(stiff[s] - 0.6))

    if F1_fixed is not None:
        e = np.zeros(N); e[idx["F1"]] = 1
        add(e, F1_fixed); add(-e, -F1_fixed)

    A = np.array(A); b = np.array(b)
    res = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * N, method="highs")
    if not res.success:
        return None
    x = res.x
    pos = {k: x[idx[k]] for k in names}
    dev = sum(abs(pos[k] - pref[k]) for k in names)
    return pos, dev

for F1 in (0.20, 0.10, 0.00):
    r = solve(F1_fixed=F1)
    if r:
        pos, dev = r
        print(f"=== F1 = {F1:.2f}  总偏差={dev:.4f} ===")
        for k in ["F1", "L1", "L2", "L3", "L4", "L5", "L6"]:
            d = pos[k] - stiff[1] if k in ("L1", "L2") else \
                pos[k] - stiff[2] if k in ("L3", "L4") else \
                pos[k] - stiff[3] if k in ("L5", "L6") else 0
            print(f"  {k}: {pos[k]:.4f}  距筋={d:+.4f}")
        st = [pos[k] for k in ["F1", "L1", "L2", "L3", "L4", "L5", "L6"]]
        print("  间距:", " ".join(f"{st[i+1]-st[i]:.3f}" for i in range(len(st)-1)))
    else:
        print(f"=== F1 = {F1:.2f}: 无解 ===")
