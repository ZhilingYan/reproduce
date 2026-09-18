# -*- coding: utf-8 -*-
"""Search-QA 递归模式的动作抠取:三标签互斥版。

克隆自 env_package/search/projection.py(二标签),规则平移
(RSO_searchqa_data_mapping.md §四:标签集 <search>/<answer>/<delegate>,
三标签互斥、同种多次无效、多段只取首个完整块、首个闭合标签后硬截断):
  抠取:按文本中最先出现的完整块取(search/answer/delegate 任一);没有 → 结果空串。
  无效(valids=0):没有任何完整块;或原文中出现两种不同标签;或同种标签多于一个。
  与 flat 版一致:无效时已抠到的块保留在结果里,valid 标志单独交给 trainer 的 η。
"""
from typing import List, Tuple
import re

_TAGS = ("search", "answer", "delegate")
_BLOCK_RES = {t: re.compile(rf"<{t}>(.*?)</{t}>", re.IGNORECASE | re.DOTALL) for t in _TAGS}
_OPEN_RES = {t: re.compile(rf"<{t}>", re.IGNORECASE) for t in _TAGS}


def _postprocess_action(action: str) -> str:
    """在最先出现的闭合标签处硬截断,防多段拼接(flat 版同款逻辑,扩到三标签)。"""
    cut = None
    for t in _TAGS:
        pos = action.find(f"</{t}>")
        if pos >= 0:
            end = pos + len(f"</{t}>")
            cut = end if cut is None else min(cut, end)
    return action[:cut] if cut is not None else action


def search_rso_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    results: List[str] = []
    valids: List[int] = [1] * len(actions)
    for i, action in enumerate(actions):
        original = action or ""
        trimmed = _postprocess_action(original)

        # ---- 抠取:取文本中最先出现的完整块
        best = None
        for t in _TAGS:
            m = _BLOCK_RES[t].search(trimmed)
            if m and (best is None or m.start() < best[1].start()):
                best = (t, m)
        if best is None:
            results.append("")
            valids[i] = 0
        else:
            t, m = best
            results.append(f"<{t}>{m.group(1).strip()}</{t}>")

        # ---- 有效性(在原文上判)
        counts = {t: len(_OPEN_RES[t].findall(original)) for t in _TAGS}
        n_kinds = sum(1 for c in counts.values() if c > 0)
        if n_kinds > 1 or any(c > 1 for c in counts.values()):
            valids[i] = 0
    return results, valids
