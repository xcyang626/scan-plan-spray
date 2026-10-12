# -*- coding: utf-8 -*-
"""NN-IK 可插拔后端 (用户: "前面的ik模型可以提速流程, 训练好之后放进去").

用法(03/05 内调用):
    from nn_ik_hook import nn_ik
    q_hint = nn_ik(p_rel, d)                       # 6维输入(legacy, 老模型)
    q_hint = nn_ik(p_rel, d, pelvis=0.63, rp=0.087) # 8维输入(V2通用大模型)
        p_rel(3,) 腕心目标(x,y,z; z=离地高, 与03同约定), d(3,) 指向单位向量,
        pelvis 骨盆参考高(通用模型条件输入; 缺省0.77), rp 腰roll/pitch限位rad(缺省取
        G1W_WAIST_RP 环境变量, 再缺省 ±5°)
        未部署模型 -> 返回 None, 流程零改动回退现有快速IK

部署(优先级从高到低, 命中即用):
  1. station/nn_ik/<robot>_all_nn_ik.npz  —— V2 通用大模型(一次训练覆盖全部腰限档
     ×站/蹲/过渡姿态, 输入8维=[p_rel(3),d(3),pelvis,rp], 隐层任意深度 W1..Wk+Wo)
  2. station/nn_ik/<robot>_rp<G1W_WAIST_DEG>_nn_ik.npz —— 按腰限档的小模型(6维)
  3. station/nn_ik/<robot>_nn_ik.npz —— 通用文件名小模型(6维)
键: W1,b1,W2,b2,...,Wo,bo   float32   激活: tanh, 输出层线性
输出 10 维: [腰yaw,腰roll,腰pitch, 右臂7] (弧度)
接入方式: 结果只作为 "种子" 送入 DLS 精解(03 锚点/05 增量), 精度与约束仍由
DLS+容差+限位保证 —— 网络只提速, 不破坏角度/连续性规范。
"""
import os
import re
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cache = {}

_DEFAULT_RP = 0.0872665          # ±5°(03/02 源码缺省腰 roll/pitch 限位)


def _rp_now():
    try:
        v = float(os.environ.get("G1W_WAIST_RP", "0"))
        return v if v > 0 else _DEFAULT_RP
    except ValueError:
        return _DEFAULT_RP


def nn_ik(p_rel, d, robot="g1", pelvis=None, rp=None):
    deg = os.environ.get("G1W_WAIST_DEG", "").strip()
    nn_dir = os.path.join(BASE, "station", "nn_ik")
    # 优先级: 通用大模型(_all, 8维) -> 按腰限档(6维) -> 通用名(6维)
    cand = [os.path.join(nn_dir, f"{robot}_all_nn_ik.npz")]
    if deg:
        cand.append(os.path.join(nn_dir, f"{robot}_rp{deg}_nn_ik.npz"))
    cand.append(os.path.join(nn_dir, f"{robot}_nn_ik.npz"))
    p = next((c for c in cand if os.path.exists(c)), None)
    if p is None:
        return None
    try:
        # 缓存键含 (mtime, size): 12 重训并原子替换文件后, 长驻进程自动重载新权重
        st = os.stat(p)
        key = (p, st.st_mtime, st.st_size)
    except OSError:
        return None
    try:
        m = _cache[key]
    except KeyError:
        try:
            m = dict(np.load(p))          # 训练器可能在重写模型文件, 失败则本点跳过
            _cache.clear()                # 旧文件键失效, 清掉防堆积
            _cache[key] = m
        except Exception:
            return None
    x = np.concatenate([np.asarray(p_rel, np.float32).ravel(),
                        np.asarray(d, np.float32).ravel()])
    if m["W1"].shape[0] == 8:             # V2 通用模型: 追加 (pelvis, rp) 条件输入
        x = np.concatenate([x, np.float32([
            0.77 if pelvis is None else float(pelvis),
            _rp_now() if rp is None else float(rp)])])
    h = x
    for k in sorted((s for s in m if re.fullmatch(r"W[0-9]+", s)),
                    key=lambda s: int(s[1:])):
        h = np.tanh(h @ m[k] + m["b" + k[1:]])
    return h @ m["Wo"] + m["bo"]
