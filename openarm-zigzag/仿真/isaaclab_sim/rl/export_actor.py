#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""export_actor.py —— 从 rsl_rl checkpoint 导出 actor 为纯 numpy 权重
====================================================================
不启动 Isaac Sim(仅 torch)。产出与 train.py 末尾自动导出相同的格式:
    actor_right.npz: W0,b0,W2,b2,... activation=elu
供 MuJoCo 验证(rl_validate.py)与真机 rl_exec.py 使用(两者都按 W0,W1,...
顺序前向, 层号只作标识)。

用法: python export_actor.py --checkpoint .../model_600.pt --out actor_right.npz
"""

import argparse
import re

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu",
                      weights_only=False)
    sd = None
    for key in ("actor_state_dict", "model_state_dict"):
        if isinstance(ckpt, dict) and key in ckpt:
            sd = ckpt[key]
            break
    if sd is None:
        sd = ckpt
    # 层键形如 mlp.<i>.weight (i=0,2,4,6, 中间夹激活); 也兼容 mlp.layers.<i>
    layers = {}
    for k, v in sd.items():
        m = re.fullmatch(r"(?:.*\.)?mlp\.(?:layers\.)?(\d+)\.(weight|bias)", k)
        if m:
            tag = "W" if m.group(2) == "weight" else "b"
            layers["%s%s" % (tag, m.group(1))] = v.detach().cpu().numpy()
    if not layers:
        print("可用键:", list(sd.keys())[:40])
        raise SystemExit("[致命] checkpoint 中未找到 MLP 参数")
    ws = sorted((k for k in layers if k.startswith("W")),
                key=lambda k: int(k[1:]))
    # 重排为连续序号(前向按顺序用, 层号仅是标识)
    out_layers = {}
    for i, wk in enumerate(ws):
        out_layers["W%d" % i] = layers[wk]
        out_layers["b%d" % i] = layers["b" + wk[1:]]
    np.savez(args.out, activation=b"elu", **out_layers)
    print("已导出 %s (%d 层):" % (args.out, len(ws)))
    for i in range(len(ws)):
        print("  W%d %s" % (i, out_layers["W%d" % i].shape))
    if "distribution.std_param" in sd:
        print("  (分布 std 未导出: 部署用确定性均值动作)")


if __name__ == "__main__":
    main()
