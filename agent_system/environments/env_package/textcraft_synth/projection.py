# -*- coding: utf-8 -*-
"""模型输出文本 → textcraft_synth 动作的解析。

合格(valids=1)要求同时有 <thought> 和 <action> 标签——注意用的是 RAO 官方同款的
<thought>(普通文本),不是 SDAR 其他环境的 <think>:后者是 Qwen3 特殊 token,
对去思考版模型(4B-Instruct-2507)有毒(详见 prompts/textcraft_synth.py 头注释)。
抠出 <action> 内容小写后交给环境;不合格时动作截成末 60 字符照样送环境
(会得到 unknown action 报错),并在 ray_trainer 里吃 -0.1 无效动作惩罚。"""
import re
from typing import List


def textcraft_synth_projection(actions: List[str]):
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
