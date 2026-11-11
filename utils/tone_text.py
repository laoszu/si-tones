import re

from matplotlib import font_manager as fm
from pypinyin import Style, lazy_pinyin

from utils.config import cfg


def find_cjk_font():
    fonts = {f.name: f.fname for f in fm.fontManager.ttflist}

    for key in cfg["tones"]["cjk_fonts"]:
        for name, path in fonts.items():
            if key in name:
                return fm.FontProperties(fname=path)

    print("no font with chinese characters found, pass font_path")
    return fm.FontProperties()


def parse(text):
    chars = [c for c in text if "一" <= c <= "鿿"]
    pinyin = lazy_pinyin("".join(chars), style=Style.TONE3, neutral_tone_with_five=True)

    syllables = [re.sub(r"\d", "", p) for p in pinyin]
    tones = [int(p[-1]) if p[-1].isdigit() else 5 for p in pinyin]

    return chars, syllables, tones


def bold_font(font, size):
    font = font.copy()
    font.set_size(size)
    font.set_weight("bold")
    return font
