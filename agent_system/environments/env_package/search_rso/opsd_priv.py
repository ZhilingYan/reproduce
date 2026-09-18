# -*- coding: utf-8 -*-
"""Search-QA 的 priv(节点局部特权文本)渲染 + OPSD 适配器子类。

设计出处:Ideation/ideas/RSO_searchqa_data_mapping.md §三(2026-09-18 用户定稿的三段式):
  [目标]  本节点要回答的问题
  [已知]  本节点已解决跳的依据(已验证的承诺答案;检索段落标题按需)
  [待查]  剩余 question_decomposition 中所有待解决 item 的 "question" + "answer"
禁令:金段落正文不进[待查]。
判序:行开局现算(编排器 _render_top/reset 时调 build_priv,机制与 textcraft 相同)。

编排器只认 build_priv 鸭子类型(orchestrator._priv_enabled 探测);
阶段 1 SearchRecursiveAdapter 没有该方法,行为不变。
"""
from __future__ import annotations

from typing import Dict

from agent_system.environments.env_package.search_rso.decomp import SearchDecompTracker
from agent_system.environments.env_package.search_rso.recursive_adapter import (
    SearchRecursiveAdapter)
from agent_system.recursive.node import Node

_PRIV_HEADER = ("[Privileged decomposition — training-time teacher reference only; "
                "the student never sees this]\n")


def render_priv(node: Node, tracker: SearchDecompTracker | None) -> str:
    lines = [_PRIV_HEADER + f"Goal: {node.goal_text}"]
    if tracker is None or not tracker.n_hops:
        lines.append("Decomposition: (not available for this question)")
        return "\n".join(lines)

    mine = tracker.resolved_by_node(node.uid)
    if mine:
        lines.append("Resolved by this agent so far:")
        for r in mine:
            lines.append(f"  - {r['hop'].question}  =>  {r['hop'].answer}  "
                         f"(committed via {r['via']})")

    remaining = tracker.unresolved_hops()
    if remaining:
        lines.append("Remaining gold sub-questions (with answers; still unresolved):")
        for h in remaining:
            lines.append(f"  - {h.question}  =>  {h.answer}")
    else:
        lines.append("Remaining gold sub-questions: (none — all hops are resolved; "
                     "the final answer should now be given)")
    return "\n".join(lines)


class SearchOPSDAdapter(SearchRecursiveAdapter):
    """阶段 1 search 适配器 + build_priv。useless_goal 标记已在父类 on_node_open 里做。"""

    def build_priv(self, node: Node, env_info: Dict) -> str:
        return render_priv(node, self._tracker(env_info)
                           if not node.is_root else self._trackers.get(node.uid))
