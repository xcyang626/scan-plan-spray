# -*- coding: utf-8 -*-
"""NN-IK 可插拔后端 (用户: "前面的ik模型可以提速流程, 训练好之后放进去").

用法(03/05 内调用):
    from nn_ik_hook import nn_ik
    q_hint = nn_ik(p_rel, d)     # p_rel(3,) 骨盆系腕心目标, d(3,) 指向单位向量
                                 # 未部署模型 -> 返回 None, 流程零改动回退现有快速IK
部署: 把训练好的模型存为 station/nn_ik/<robot>_nn_ik.npz:
    键: W1,b1,W2,b2,Wo,bo   float32
    输入 6 维: [p_rel(3), d(3)]  ->  输出 10 维: [腰yaw,腰roll,腰pitch, 右臂7] (弧度)
    归一化: 输入原样(m), 输出原样(rad)   激活: tanh, 输出层线性
接入方式: 结果只作为 "种子" 送入 DLS 精解(03 锚点/05 增量), 精度与约束仍由
DLS+容差+限位保证 —— 网络只提速, 不破坏角度/连续性规范。
"""
import os
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cache = {}


def nn_ik(p_rel, d, robot="g1"):
    deg = os.environ.get("G1W_WAIST_DEG", "").strip()
    p = os.path.join(BASE, "station", "nn_ik",
                     f"{robot}_rp{deg}_nn_ik.npz" if deg else f"{robot}_nn_ik.npz")
    if not os.path.exists(p):
        return None
    try:
        m = _cache[p]
    except KeyError:
        try:
            m = dict(np.load(p))          # 训练器可能在重写模型文件, 失败则本点跳过
            _cache[p] = m
        except Exception:
            return None
    x = np.concatenate([np.asarray(p_rel, np.float32).ravel(),
                        np.asarray(d, np.float32).ravel()])
    h = np.tanh(x @ m["W1"] + m["b1"])
    h = np.tanh(h @ m["W2"] + m["b2"])
    return h @ m["Wo"] + m["bo"]
