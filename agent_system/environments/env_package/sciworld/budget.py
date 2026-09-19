# -*- coding: utf-8 -*-
"""root/child 解耦的预算(sciworld 专用;不改 agent_system/recursive/budget.py 本体)。

依据:10 案例双臂分析——root 30 步腰斩 power-component(57→7),金路径 billed
中位 26 / 最大 178;拍板值(参数对照_textcraft_vs_sciworld.md,方案 A):
root 60 / child 20 / depth 2。

接线点:orchestrator.py:150 `budget.allocate(0)`(root)与 :208
`budget.allocate(top.depth + 1)`(子)——allocate 本来就带 depth 参数
(budget.py:131-141 注明"保留是为了将来按深度分配"),子类只覆写它,
can_delegate/refusal 语义原封不动。
"""
from __future__ import annotations

from agent_system.recursive.budget import PerAgentBudget

DEFAULT_ROOT_STEPS = 60
DEFAULT_CHILD_STEPS = 20
DEFAULT_SCIWORLD_MAX_DEPTH = 2


class RootChildBudget(PerAgentBudget):
    def __init__(self, root_steps: int = DEFAULT_ROOT_STEPS,
                 child_steps: int = DEFAULT_CHILD_STEPS,
                 max_depth=DEFAULT_SCIWORLD_MAX_DEPTH):
        super().__init__(per_agent_steps=child_steps, max_depth=max_depth)
        if int(root_steps) <= 0:
            raise ValueError(f"root_steps 必须为正整数,收到 {root_steps}")
        self.root_steps = int(root_steps)

    @classmethod
    def from_config(cls, rao_cfg) -> "RootChildBudget":
        cfg = dict(rao_cfg or {})
        root = cfg.get("root_max_steps", DEFAULT_ROOT_STEPS)
        child = cfg.get("per_agent_max_steps", DEFAULT_CHILD_STEPS)
        depth = cfg["max_depth"] if "max_depth" in cfg else DEFAULT_SCIWORLD_MAX_DEPTH
        return cls(root_steps=int(root), child_steps=int(child),
                   max_depth=None if depth is None else int(depth))

    def allocate(self, depth: int) -> int:
        return self.root_steps if int(depth) == 0 else self.per_agent_steps

    def __repr__(self) -> str:
        return (f"<RootChildBudget root={self.root_steps} child={self.per_agent_steps} "
                f"max_depth={self.max_depth}>")
