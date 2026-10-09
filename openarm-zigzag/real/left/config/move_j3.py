#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""J3 单向验证: /usr/bin/python3 move_j3.py [+度|-度]  (默认见 joint_probe)"""
import os
import sys
# joint_probe 位于同目录, 需先把本目录加入 import 搜索路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_probe import probe

# 微动逻辑(限位/速度/安全提示)集中在 joint_probe.probe(),
# 本脚本只是入口分发: 关节号 = 文件名里的数字
probe(3, sys.argv)
