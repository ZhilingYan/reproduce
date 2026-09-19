# -*- coding: utf-8 -*-
"""模型输出文本 → sciworld 动作的解析。

逐行克隆 textcraft_synth/projection.py:14-31(移植对照表 2026-09-19):
合格(valids=1)要求同时有 <thought> 和 <action> 标签;抠出 <action> 内容小写后
交给环境——小写化恰合 scienceworld 官方 step() 的 .lower().strip() sanitize;
不合格时动作截成末 60 字符照样送环境(得到 unknown action 报错),
无效标志交给 η(flat 走 actor 惩罚,递归走优势层 invalid_coef)。
`delegate: ...` / `answer: ...` 都写在 <action> 内,这里不特判
(递归适配器 parse_delegation / 环境 answer noop 各自分流)。"""
import re
from typing import List


def sciworld_projection(actions: List[str]):
    valids = [0] * len(actions)
    for i in range(len(actions)):
        original = actions[i]
        lowered = actions[i].lower()
        start = lowered.find("<action>")
        end = lowered.find("</action>")
        try:
            if start == -1 or end == -1 or end < start:
                actions[i] = lowered[-60:]
                continue
            actions[i] = lowered[start + len("<action>"):end].strip()
            valids[i] = 1
        except Exception:
            actions[i] = lowered[-60:]

        if original.find("<thought>") == -1 or original.find("</thought>") == -1:
            valids[i] = 0
        if re.search(r"[一-鿿]", original):
            valids[i] = 0
    return actions, valids
