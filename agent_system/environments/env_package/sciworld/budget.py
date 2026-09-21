# -*- coding: utf-8 -*-
"""root/child 解耦的预算(sciworld 口径)。

2026-09-20 起,解耦逻辑本身已上提到内核 `agent_system/recursive/budget.py`
的 RootChildBudget(textcraft 真环境 OOD 评测也要用,同一份实现只应存在一处)。
本模块保留 sciworld 自己的默认值与类名,行为与此前逐字相同。

依据:10 案例双臂分析——root 30 步腰斩 power-component(57→7),金路径 billed
中位 26 / 最大 178;拍板值(参数对照_textcraft_vs_sciworld.md,方案 A):
root 60 / child 20 / depth 2。

接线点:orchestrator.py:150 `budget.allocate(0)`(root)与 :208
`budget.allocate(top.depth + 1)`(子)——allocate 本来就带 depth 参数,
子类只覆写它,can_delegate/refusal 语义原封不动。
"""
from __future__ import annotations

from agent_system.recursive.budget import RootChildBudget as _KernelRootChildBudget

DEFAULT_ROOT_STEPS = 60
DEFAULT_CHILD_STEPS = 20
DEFAULT_SCIWORLD_MAX_DEPTH = 2


class RootChildBudget(_KernelRootChildBudget):
    """内核同名类的 sciworld 默认值版本(root 60 / child 20 / depth 2)。"""

    def __init__(self, root_steps: int = DEFAULT_ROOT_STEPS,
                 child_steps: int = DEFAULT_CHILD_STEPS,
                 max_depth=DEFAULT_SCIWORLD_MAX_DEPTH):
        super().__init__(root_steps=root_steps, child_steps=child_steps,
                         max_depth=max_depth)

    @classmethod
    def from_config(cls, rao_cfg) -> "RootChildBudget":
        cfg = dict(rao_cfg or {})
        root = cfg.get("root_max_steps", DEFAULT_ROOT_STEPS)
        child = cfg.get("per_agent_max_steps", DEFAULT_CHILD_STEPS)
        depth = cfg["max_depth"] if "max_depth" in cfg else DEFAULT_SCIWORLD_MAX_DEPTH
        return cls(root_steps=int(root), child_steps=int(child),
                   max_depth=None if depth is None else int(depth))
