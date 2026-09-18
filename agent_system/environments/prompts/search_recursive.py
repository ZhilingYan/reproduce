# -*- coding: utf-8 -*-
"""Search-QA 递归模式的 prompt 模板:从 flat 的 search.py 派生,不另起炉灶。

派生法照 textcraft_synth_recursive.py 的先例(2026-09-19 用户要求重写):
flat 模板【原文 import】,递归版只做四处最小增量,每处 replace 带命中断言,
flat 的其余每一个字(含推理标签 <think>——遵循 flat 原文继承;Qwen3 特殊 token
风险已另行挂账,是否随模型调整由 flat 口径统一决定)原样保留:
  增量 1  动作清单末尾追加第 (3) 条 delegate(flat 的 (1)(2) 一字不动);
  增量 2  "(do not perform both)" → "(do not perform more than one)"(三选一后原句失真);
  增量 3  带历史版的历史说明句补半句"子 agent 战报跟在 <delegate> 之后";
  增量 4  追加 DELEGATION STRATEGY 五句(textcraft 版逐句平移到 QA 语境;
          textcraft 第 5 句"共享库存"在 QA 无对应物,以"子 agent 全新上下文、
          子问题须自包含"替位——这是子 agent 语义里最近的等价约束)。
子节点模板 = 递归版 header 两处单句替换(角色句 + "Your question:"→"Your sub-question:"),
与 textcraft 的 _HEADER_CHILD 手法相同。

占位符沿用 flat:task_description / step_count / memory_context
(由 search_rso/recursive_adapter.build_observation 填;历史行格式照
memory.py:178 的 "Step {n}:{act} {obs}",与 flat 逐字相同)。
"""
from agent_system.environments.prompts.search import (
    SEARCH_TEMPLATE, SEARCH_TEMPLATE_NO_HIS)

_DELEGATE_OPTION = """(3) If a distinct sub-question must be answered first, you can delegate it to a sub-agent using format: <delegate> the sub-question </delegate>. The sub-agent starts with a fresh context, researches only that sub-question, and reports its answer back to you. For example, <delegate>Who directed the film Move (1970)?</delegate>.
"""

# textcraft _DELEGATION_TIPS(官方 platoon agent.py:228-235)的 QA 平移,逐句对应
_DELEGATION_TIPS = """
DELEGATION STRATEGY:
- For multi-hop questions, it is **highly recommended** to delegate the first unresolved sub-question to a sub-agent
- Break the question into INDEPENDENT sub-questions that can be answered separately
- For questions that are sufficiently complex, it is recommended to recursively delegate; i.e., sub-agents can further delegate to other sub-agents.
- Delegate one sub-question at a time, not everything at once
- Each delegated sub-question must be fully self-contained (name entities explicitly) - the sub-agent starts fresh and cannot see your context
"""


def _must_replace(tpl: str, old: str, new: str) -> str:
    assert old in tpl, f"派生失配:flat 模板里找不到 {old!r}(flat 原文变了?)"
    return tpl.replace(old, new)


def _recursify(tpl: str, has_history: bool) -> str:
    tpl = _must_replace(tpl, "(do not perform both)", "(do not perform more than one)")
    if has_history:
        tpl = _must_replace(
            tpl,
            "returned by the external search engine. History:",
            "returned by the external search engine, and sub-agent reports follow your "
            "<delegate> requests. History:")
    return tpl + _DELEGATE_OPTION + _DELEGATION_TIPS


SEARCH_RECURSIVE_TEMPLATE_NO_HIS = _recursify(SEARCH_TEMPLATE_NO_HIS, has_history=False)
SEARCH_RECURSIVE_TEMPLATE = _recursify(SEARCH_TEMPLATE, has_history=True)


def _childify(tpl: str) -> str:
    tpl = _must_replace(
        tpl,
        "You are an expert agent tasked with answering the given question step-by-step.",
        "You are an expert sub-agent. A parent agent delegated the following sub-question "
        "to you; answer it step-by-step and keep the final answer short and to the point "
        "(an entity, a date, yes/no, ...) - it will be reported back to the parent agent.")
    return _must_replace(tpl, "Your question:", "Your sub-question:")


SEARCH_RECURSIVE_CHILD_TEMPLATE_NO_HIS = _childify(SEARCH_RECURSIVE_TEMPLATE_NO_HIS)
SEARCH_RECURSIVE_CHILD_TEMPLATE = _childify(SEARCH_RECURSIVE_TEMPLATE)
assert SEARCH_RECURSIVE_CHILD_TEMPLATE != SEARCH_RECURSIVE_TEMPLATE
