# -*- coding: utf-8 -*-
"""公共工具: 中文字体设置"""
import matplotlib
import matplotlib.font_manager as fm

_FONT_SET = False


def setup_cn_font():
    global _FONT_SET
    if _FONT_SET:
        return
    for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/truetype/arphic/ukai.ttc",
              "/usr/share/fonts/truetype/arphic/uming.ttc"]:
        try:
            fm.fontManager.addfont(p)
            matplotlib.rcParams["font.sans-serif"] = [fm.FontProperties(fname=p).get_name()]
            break
        except Exception:
            continue
    matplotlib.rcParams["axes.unicode_minus"] = False
    _FONT_SET = True
