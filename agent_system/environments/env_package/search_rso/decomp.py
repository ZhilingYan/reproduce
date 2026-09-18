# -*- coding: utf-8 -*-
"""Search-QA 的 decomp 库加载与 Φ 追踪器。

设计出处:Ideation/ideas/RSO_searchqa_data_mapping.md §一(2026-09-18 最终版):
  Φ(E) = 还没解决的 retrieval 跳数(compare 尾步不进分母;E 全树共享、只增不减,
  G ≥ 0 恒成立,不搬 TextCraft 的负值/冻结补丁)。
  跳的"解决"只有两个通道,每跳至多记一次:
    (c) 作答承诺:某【子】agent 某 turn 的 <answer> 含该跳 answer;
    (d) 任务承诺:任一节点的 <search> query 或 <delegate> 子任务文本含该跳 answer。
  <think> 自由文本与检索返回内容都不产生 Φ 事件;root <answer> 不进 Φ(归 A_out)。

匹配口径:归一化(小写、非字母数字改空格、压空白)后按【词边界】整词匹配别名集;
别名集在建库时就已归一化(examples/data_preprocess/make_searchrso_data_products.py,
两边的 _norm 必须保持一字不差)。

decomp_store.json 的键是归一化题面;OOD 子集(nq/triviaqa/popqa/hotpotqa/bamboogle)
查不到 → 空 tracker,ΔΦ 恒 0,不影响其它通路。
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional


def norm_text(s: str) -> str:
    """与建库脚本的 norm 一字不差:小写、非字母数字改空格、压空白。"""
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", str(s).lower())).strip()


_STORE_CACHE: Dict[str, dict] = {}


def load_decomp_store(path: str) -> dict:
    """进程内缓存:每个 Ray worker/driver 各加载一次(几十 MB,秒级)。"""
    if path not in _STORE_CACHE:
        with open(path) as f:
            _STORE_CACHE[path] = json.load(f)
        print(f"[search_rso] decomp store loaded: {len(_STORE_CACHE[path])} questions from {path}")
    return _STORE_CACHE[path]


class Hop:
    __slots__ = ("question", "answer", "aliases", "compare", "_res")

    def __init__(self, d: dict):
        self.question = d["q"]
        self.answer = d["ans"]
        self.aliases = list(d.get("aliases") or [])
        self.compare = bool(d.get("compare", False))
        # 词边界整词匹配;别名已归一化,拼成一个 alternation,按跳编译一次
        pats = [re.escape(a) for a in self.aliases if a]
        self._res = re.compile(r"(?<![0-9a-z])(?:" + "|".join(pats) + r")(?![0-9a-z])") \
            if pats else None

    def matches(self, normed_text: str) -> bool:
        return bool(self._res.search(normed_text)) if (self._res and normed_text) else False


class SearchDecompTracker:
    """一棵树一个(键 = root uid,由适配器持有)。E = resolved 下标集合,只增不减。

    resolve_events(text, via, node_uid) 返回本次新解决的跳数(=ΔΦ 的绝对值;
    Φ 单调下降,ΔΦ 语义按 rso_core 的口径记【正的进展量】delta_phi = Φ_before − Φ_after)。
    """

    def __init__(self, hops: Optional[List[dict]]):
        self.hops: List[Hop] = [Hop(h) for h in (hops or [])]
        self.retrieval_idx = [i for i, h in enumerate(self.hops) if not h.compare]
        self.resolved: Dict[int, dict] = {}          # hop_idx → {via, node_uid, text}

    # ------------------------------------------------------------ 状态量
    @property
    def phi(self) -> float:
        return float(len(self.retrieval_idx) - len([i for i in self.resolved
                                                    if i in set(self.retrieval_idx)]))

    @property
    def n_hops(self) -> int:
        return len(self.retrieval_idx)

    # ------------------------------------------------------------ 事件
    def scan(self, text: str, include_resolved: bool = False) -> List[int]:
        """返回文本命中的 retrieval 跳下标(默认只看未解决的)。"""
        normed = norm_text(text)
        if not normed:
            return []
        out = []
        for i in self.retrieval_idx:
            if not include_resolved and i in self.resolved:
                continue
            if self.hops[i].matches(normed):
                out.append(i)
        return out

    def resolve_events(self, text: str, via: str, node_uid: str) -> float:
        """按 (c)/(d) 通道登记解决事件,返回 delta_phi(≥0)。"""
        hits = self.scan(text)
        for i in hits:
            self.resolved[i] = {"via": via, "node_uid": node_uid,
                                "text": str(text)[:200]}
        return float(len(hits))

    # ------------------------------------------------------------ priv 渲染原料
    def unresolved_hops(self) -> List[Hop]:
        return [self.hops[i] for i in self.retrieval_idx if i not in self.resolved]

    def resolved_by_node(self, node_uid: str) -> List[dict]:
        return [{"hop": self.hops[i], **meta} for i, meta in self.resolved.items()
                if meta["node_uid"] == node_uid]
