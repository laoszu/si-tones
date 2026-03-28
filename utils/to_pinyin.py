from pypinyin import pinyin, Style

def to_pinyin(text, tones=True):
    style = Style.TONE if tones else Style.NORMAL
    return " ".join([p[0] for p in pinyin(text, style=style)])