# -*- coding: utf-8 -*-
"""ScienceWorld 的 priv(节点局部特权文本)渲染 + OPSD 适配器子类。

设计出处:Ideation/ideas/RSO_sciworld_data_mapping.md §2a(2026-09-19 拍板):
  [目标]  root = 官方 task description;子节点 = 受托子任务文本
  [待办]  Sequential Subgoals 中未完成项按序全文
  [已达]  goal_progress 中已 true 的 subgoal 描述 + 当前分数/峰值
  [提示]  Unordered/Optional 未完成项(同模板合并压缩)
禁令:gold_action_sequence 不进 priv(D5)。判序:行开局现算(编排器机制同 search)。
材料 = 运行时 goal_progress(envs.py 需开 +env.sciworld.fetch_goal_progress=true,
opsd_factory 装配时强制打开)。

编排器只认 build_priv 鸭子类型(orchestrator.py:103);
阶段 1 SciWorldRecursiveAdapter 没有该方法,行为不变。
"""
from __future__ import annotations

from typing import Dict

from agent_system.environments.env_package.sciworld.goal_progress import (
    parse_goal_progress, render_priv)
from agent_system.environments.env_package.sciworld.recursive_adapter import (
    SciWorldRecursiveAdapter)
from agent_system.recursive.node import Node


class SciWorldOPSDAdapter(SciWorldRecursiveAdapter):
    """阶段 1 sciworld 适配器 + build_priv。"""

    def build_priv(self, node: Node, env_info: Dict) -> str:
        tree = self._trees.get(node.uid) if node.is_root else self._tree(env_info)
        if tree is None:
            return ""
        gp = parse_goal_progress(tree.goal_progress)
        return render_priv(node.goal_text, gp, score=tree.score, peak=tree.peak)
