# -*- coding: utf-8 -*-
"""`get_goal_progress()` 字符串的解析与两种特权渲染。

字符串格式(scienceworld.py:471;2026-09-19 melt v21 实测样例,
见 代码理解/scienceworld_10case_双臂对比分析.md 立项材料):
    Completed keys: <逗号分隔,可空>
    ----(横线)----
    Sequential Subgoals:
    ----
    0\tfalse\tGoalFind\tfocus on substance
    ...
    ----
    Unordered and Optional Subgoals:
    ----
    0\tfalse\tGoalInRoomWithObject\tbe in same location as lead
    ...

渲染口径(RSO_sciworld_data_mapping.md §2a):
  gt_plan(flat OPSD 全局特权)= 整题 subgoal 骨架:sequential 全列 + optional 压缩;
  priv(递归 OPSD 节点局部)= [目标]/[已达]/[待办]/[提示],截断顺序 目标>待办>已达>提示;
  同模板合并:形如 "activate heater (stove)" 的同前缀项合成一行
  "activate heater (stove/blast furnace/...)"。
禁令:gold_action_sequence 不进任何渲染(金路径非唯一,D5)。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

_ROW_RE = re.compile(r"^(\d+)\t(true|false)\t(\S+)\t(.*)$")
_PAREN_RE = re.compile(r"^(.*?)\s*\((.*?)\)\s*$")


@dataclass
class Subgoal:
    idx: int
    done: bool
    kind: str
    desc: str


@dataclass
class GoalProgress:
    sequential: List[Subgoal] = field(default_factory=list)
    optional: List[Subgoal] = field(default_factory=list)


def parse_goal_progress(text: str) -> GoalProgress:
    gp = GoalProgress()
    section = None
    for raw in (text or "").splitlines():
        line = raw.rstrip()
        if line.startswith("Sequential Subgoals"):
            section = gp.sequential
            continue
        if line.startswith("Unordered and Optional Subgoals"):
            section = gp.optional
            continue
        m = _ROW_RE.match(line)
        if m and section is not None:
            section.append(Subgoal(idx=int(m.group(1)), done=(m.group(2) == "true"),
                                   kind=m.group(3), desc=m.group(4).strip()))
    return gp


def _merge_paren_variants(goals: List[Subgoal]) -> List[str]:
    """同前缀括号变体合并:'activate heater (stove)' + '(oven)' →
    'activate heater (stove/oven)';无括号项原样保序。"""
    out: List[str] = []
    merged: Dict[str, int] = {}          # 前缀 → out 里的下标
    for g in goals:
        m = _PAREN_RE.match(g.desc)
        if not m:
            out.append(g.desc)
            continue
        prefix, variant = m.group(1), m.group(2)
        if prefix in merged:
            out[merged[prefix]] += f"/{variant}"
        else:
            merged[prefix] = len(out)
            out.append(f"{prefix} ({variant}")
    for prefix, k in merged.items():
        out[k] += ")"
    return out


def render_gt_plan(gp: GoalProgress) -> str:
    """flat OPSD 的全局特权(对应 textcraft 的 gt_plan / search 的整题分解渲染):
    静态骨架,不带完成状态(reset 时渲染一次)。"""
    lines = ["Gold subgoal plan for this task (do the ordered steps in order):"]
    for i, g in enumerate(gp.sequential):
        lines.append(f"  {i + 1}. {g.desc}")
    opts = _merge_paren_variants(gp.optional)
    if opts:
        lines.append("Helpful intermediate steps (any order, alternatives grouped):")
        for d in opts:
            lines.append(f"  - {d}")
    return "\n".join(lines)


_PRIV_HEADER = ("[Privileged subgoal plan — training-time teacher reference only; "
                "the student never sees this]\n")
_MAX_OPTIONAL_LINES = 12                 # 截断顺序:目标 > 待办 > 已达 > 提示


def render_priv(goal_text: str, gp: GoalProgress, score: int, peak: int) -> str:
    lines = [_PRIV_HEADER + f"Goal: {goal_text}"]
    todo = [g for g in gp.sequential if not g.done]
    achieved = [g for g in gp.sequential if g.done] + [g for g in gp.optional if g.done]
    if todo:
        lines.append("Remaining ordered subgoals (do these in order):")
        for g in todo:
            lines.append(f"  - {g.desc}")
    else:
        lines.append("Remaining ordered subgoals: (none — the task is complete)")
    if achieved:
        lines.append(f"Achieved so far (score {score}, peak {peak}):")
        for g in achieved:
            lines.append(f"  - {g.desc}")
    opts = _merge_paren_variants([g for g in gp.optional if not g.done])
    if opts:
        lines.append("Optional helpful steps (any order):")
        for d in opts[:_MAX_OPTIONAL_LINES]:
            lines.append(f"  - {d}")
        if len(opts) > _MAX_OPTIONAL_LINES:
            lines.append(f"  - ... ({len(opts) - _MAX_OPTIONAL_LINES} more omitted)")
    return "\n".join(lines)
