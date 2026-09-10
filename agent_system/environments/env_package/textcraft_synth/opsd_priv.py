# -*- coding: utf-8 -*-
"""RSO+OPSD 的 priv(节点局部特权文本)渲染 + OPSD 适配器子类。

设计出处:Ideation/ideas/RSO_method_design.md §2a(2026-09-10 修订版:K_X 条件化 +
工作清单 + 行开局现算判序);实现对照表:docs/RSO_OPSD_DESIGN.md §三/§四。

priv(X, t) 四段(英文渲染,与环境 prompt 语言一致):
  [目标]      goal_X(node.goal,物品→数量)
  [工作清单]  剩余需求闭包逐项:Nx item(recipe known / unknown;subtree size k)
  [已知配方]  K_X = node.scratch["known_recipes"] 中属于目标静态子 DAG 的条目,完整原文
  [待查]      清单中配方未知的物品,只给名字,不给内容

关键口径:
  * 工作清单的闭包 DP 与 Φ 完全同源:needed_closure_counts 逐行克隆自
    phi.py:35-73 的 needed_closure(632/632 已对拍),唯一改动是返回"产量 Counter"
    而不是集合——priv 要渲染数量,Φ 只要个数。
  * K_X 取节点【私有】笔记本,不取 Φ 的树全局 K(设计 §2a 黑体条款):
    未挣到的配方内容不进 priv,只在 [待查] 给名字。
  * 目标已满足 → 清单显式写 "(empty — already satisfied)",teacher 对冗余委托
    沉默的判据(打分表第 6 行)靠它成立。
  * 状态不可解(叶子缺口补不回来)→ 清单写 "(unavailable ...)",与 Φ 冻结对偶。

本文件不被阶段 1 RSO 的任何代码 import;适配器子类只由 opsd_factory.py 装配。
"""
from __future__ import annotations

import collections
import math
from typing import Dict, Optional

from agent_system.environments.env_package.textcraft_synth.recursive_adapter import (
    TextCraftSynthRecursiveAdapter)
from agent_system.recursive.node import Node


def needed_closure_counts(db, targets: Dict[str, int], inventory: Dict[str, int]):
    """剩余需求闭包,带产量:返回 Counter{物品: 还需生产的数量};不可解返回 None。

    逐行克隆自 phi.py:35-73 needed_closure(它 632/632 对拍过 gold 步数),
    仅把最后的 `return set(produce)` 改成 `return produce`(渲染工作清单要数量)。
    """
    inv = dict(inventory)
    demand = collections.Counter(targets)
    produce = collections.Counter()
    queue = list(targets)
    seen_guard = 0
    while queue:
        seen_guard += 1
        if seen_guard > 200000:
            raise RuntimeError("closure expansion exceeded guard; recipe graph may have a cycle")
        item = queue.pop()
        need = demand[item]
        if need <= 0:
            continue
        have = inv.get(item, 0)
        use = min(have, need)
        inv[item] = have - use
        deficit = need - use
        demand[item] = 0
        if deficit <= 0:
            continue
        rs = db.get_recipes_for_item(item)
        if not rs:                      # 叶子却还缺 → 不可解
            return None
        r = rs[0]
        n_batches = math.ceil(deficit / r.result_count)
        made = n_batches * r.result_count
        produce[item] += made
        inv[item] = inv.get(item, 0) + (made - deficit)
        for ing, per in r.ingredients.items():
            demand[ing] += per * n_batches
            queue.append(ing)
    return produce


def static_subtree_size(db, item: str, cache: Optional[Dict[str, int]] = None) -> int:
    """物品需求 DAG 的静态规模:该物品闭包里可 craft 物品的个数(含自身),记忆化。
    叶子(无配方)规模 0。配方图无环(phi.py 同一前提),DFS 安全。"""
    if cache is None:
        cache = {}
    if item in cache:
        return cache[item]
    rs = db.get_recipes_for_item(item)
    if not rs:
        cache[item] = 0
        return 0
    cache[item] = 0        # 占位防御(理论上无环,占位让意外环退化成小值而非死循环)
    craftable = {item}
    stack = list(rs[0].ingredients)
    seen = set()
    while stack:
        ing = stack.pop()
        if ing in seen:
            continue
        seen.add(ing)
        ing_rs = db.get_recipes_for_item(ing)
        if ing_rs:
            craftable.add(ing)
            stack.extend(ing_rs[0].ingredients)
    cache[item] = len(craftable)
    return cache[item]


def static_goal_dag(db, goal: Dict[str, int]):
    """目标的静态需求子 DAG:从 goal 出发沿配方展开,收集所有可 craft 物品(含 goal 自身
    中可 craft 的)。与库存无关——[已知配方] 段的"属于子 DAG"用它判,已满足但相关的
    配方(如需重造时)仍在 teacher 眼里。叶子不收(没有配方,谈不上"已知配方")。"""
    craftable = set()
    stack = list(goal or {})
    seen = set()
    while stack:
        it = stack.pop()
        if it in seen:
            continue
        seen.add(it)
        rs = db.get_recipes_for_item(it)
        if rs:
            craftable.add(it)
            stack.extend(rs[0].ingredients)
    return craftable


def _known_craft_items(known_lines: Dict[str, str]):
    """K_X 里真正算"已知配方"的物品:笔记本条目以 craft 开头的(base ingredient 条目不算配方)。"""
    return {it for it, line in (known_lines or {}).items()
            if isinstance(line, str) and line.strip().lower().startswith("craft")}


def render_priv(db, goal: Dict[str, int], inventory: Dict[str, int],
                known_lines: Dict[str, str],
                subtree_cache: Optional[Dict[str, int]] = None) -> str:
    """按设计 §2a 渲染 priv 四段。任何输入缺失都渲染出合法文本,绝不抛错穿透 rollout。"""
    goal = dict(goal or {})
    inventory = dict(inventory or {})
    known_lines = dict(known_lines or {})
    known = _known_craft_items(known_lines)
    if subtree_cache is None:
        subtree_cache = {}

    lines = []
    goal_str = ", ".join(f"{n}x {it}" for it, n in goal.items()) or "(none)"
    lines.append(f"Goal: {goal_str}")

    try:
        closure = needed_closure_counts(db, goal, inventory) if goal else collections.Counter()
    except Exception:
        closure = None

    if closure is None:
        lines.append("Work list: (unavailable — this goal cannot be completed from the "
                     "current inventory; required base materials are missing)")
        worklist_items = []
    elif not closure:
        lines.append("Work list: (empty — the goal is already satisfied by the current "
                     "inventory; no further craft or delegation is needed)")
        worklist_items = []
    else:
        worklist_items = sorted(closure)
        lines.append("Work list (what still must be crafted, given the current inventory):")
        for it in worklist_items:
            tag = "recipe known" if it in known else "recipe unknown — look it up first"
            size = static_subtree_size(db, it, subtree_cache)
            lines.append(f"  - {closure[it]}x {it}  ({tag}; subtree size {size})")

    # [已知配方]:K_X ∩ 目标静态子 DAG。用静态子 DAG 而不是剩余清单,是为了已满足但
    # 相关的配方(如需重造时)仍在 teacher 眼里;子 DAG 外的已知配方与本节点目标无关,不进。
    try:
        static_dag = static_goal_dag(db, goal)
    except Exception:
        static_dag = set(worklist_items)
    relevant_known = sorted(known & static_dag)
    if relevant_known:
        lines.append("Known recipes relevant to this goal (from your notebook):")
        for it in relevant_known:
            lines.append(f"  {known_lines[it]}")

    unknown = [it for it in worklist_items if it not in known]
    if unknown:
        lines.append("Recipes still unknown (names only — use get_info to learn them): "
                     + ", ".join(unknown))
    return "\n".join(lines)


class TextCraftSynthOPSDAdapter(TextCraftSynthRecursiveAdapter):
    """阶段 1 递归适配器 + OPSD 两件新事:build_priv(行开局现算的原料)与 useless_goal 标记。

    编排器只认 `build_priv` 这个鸭子类型方法(orchestrator.__init__ 的 _priv_enabled 探测);
    阶段 1 的父类没有它,所以 main_rso 的行为一个字节不变。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._subtree_cache: Dict[str, int] = {}
        self._db_cache = None

    def _db(self):
        if self._db_cache is None:
            from agent_system.environments.env_package.textcraft_synth.synth_core import (
                get_shared_recipe_db)
            self._db_cache = get_shared_recipe_db()
        return self._db_cache

    # ------------------------------------------------------------ priv
    def build_priv(self, node: Node, env_info: Dict) -> str:
        """行开局现算:按【当时】库存与该节点私有 K_X 渲染 priv(设计 §2a 判序,2026-09-10 定稿)。
        调用时机由编排器负责(reset / _render_top,即构造该行 prompt 的时刻)。"""
        goal = node.goal if isinstance(node.goal, dict) else {}
        inv = env_info.get("extra.inventory") or {}
        return render_priv(self._db(), goal, inv,
                           node.scratch.get("known_recipes") or {},
                           subtree_cache=self._subtree_cache)

    # ------------------------------------------------------------ useless_goal
    def on_node_open(self, node: Node, env_info: Dict) -> None:
        """父类逻辑不动;子节点开张时补记 useless_goal:goal 的任一物品不在
        root 剩余闭包(root targets × 当时库存,与 Φ 同一 DP)内 → True。
        rso/useless_goal_rate 指标的原料(RSO_method_design §四;checklist #6)。"""
        super().on_node_open(node, env_info)
        if node.is_root:
            return
        root_uid = str(env_info.get("root_node_uid") or "")
        tracker = self._phi_trackers.get(root_uid)
        goal = node.goal if isinstance(node.goal, dict) else {}
        if tracker is None or not goal:
            return
        try:
            closure = needed_closure_counts(
                self._db(), tracker.targets, env_info.get("extra.inventory") or {})
        except Exception:
            closure = None
        if closure is None:
            return                       # root 已不可解:无从判定,不记(与 Φ 冻结口径一致)
        node.scratch["useless_goal"] = any(it not in closure for it in goal)
