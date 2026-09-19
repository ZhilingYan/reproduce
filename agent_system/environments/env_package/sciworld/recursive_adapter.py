# -*- coding: utf-8 -*-
"""ScienceWorld 的递归适配器:实现 agent_system/recursive/protocol.py 的 RecursiveEnvAdapter。

体例镜像 search_rso/recursive_adapter.py;设计出处
Ideation/ideas/RSO_sciworld_data_mapping.md(2026-09-19 拍板版)。要点:

  goal 载荷 = {"task": str}。root 的题面 = 官方 task description(envs.py 写进
  info["extra.task_desc"]);子节点的题面 = `delegate: <sub-task>` 的子任务文本。

  Φ(全在本适配器;mapping §一,峰值推进口径):
    P = clip 分数峰值(envs.py 维护,经 info["extra.peak"] 下发);
    每个真实 env 轮 delta_phi = P_after − P_before ≥ 0(按"谁执行该步"归属到行);
    委托轮 / noop / answer 汇报轮 P 不变 → delta 0;焦死不回退 P(失败终分为
    定值 −100,Goal.scala:100;G ≥ 0 构造性成立)。

  终止与判分:
    root:环境 done(score 100 / 焦死 / envStepLimit)由内核按 CLOSE_EPISODE_END
    收摊 → evaluate_node 给分 = clip(final, 0, 100)/100(官方口径负分裁零);
    子节点:`answer: <report>` 显式汇报(补丁②)→ goal_met 关闭,报告文本进父观测;
    预算耗尽/卡死由内核与 stuck_in_loop(textcraft 适配器逐字复用)处置。
    子节点 success(诊断用,不进 A_out)= 有汇报,或任期内峰值有推进。

  invalid 回折(mapping §一规则 3):env 回 "No known action matches" 的轮,
  在 on_turn_result 里把 info["is_action_valid"] 压成 0(orchestrator.py:231 先按
  projection 结果 stamp,本钩子在其后、收集器读取之前执行,时序见 orchestrator
  _advance_slot)。noop/answer 轮不罚。
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

import numpy as np

from agent_system.environments.env_package.textcraft_synth.recursive_adapter import (
    stuck_in_loop, DEFAULT_STUCK_THRESHOLD, DEFAULT_STUCK_WINDOW)
from agent_system.environments.prompts.sciworld_recursive import (
    SCIWORLD_RECURSIVE_CHILD_TEMPLATE, SCIWORLD_RECURSIVE_CHILD_TEMPLATE_NO_HIS,
    SCIWORLD_RECURSIVE_TEMPLATE, SCIWORLD_RECURSIVE_TEMPLATE_NO_HIS,
    SUBAGENT_TASK_NOTE)
from agent_system.recursive.node import Node, NodeRecord
from agent_system.recursive.protocol import (
    CLOSE_GOAL_MET, CLOSE_STUCK, DelegationRequest, NodeOutcome)

_DELEGATE_RE = re.compile(r"^\s*delegate\s*:?\s*(.*)$", re.IGNORECASE | re.DOTALL)
# 模型惯性带出的预算尾巴(评测版文法遗产)容错剥离:'... budget 8' → '...'
_BUDGET_TAIL_RE = re.compile(r"[\s,;]*budget\s*[:=]?\s*\d+\s*\.?\s*$", re.IGNORECASE)
_ANSWER_RE = re.compile(r"^\s*answer\s*:?\s*(.*)$", re.IGNORECASE | re.DOTALL)


class _TreeState:
    """树级共享世界状态(共享库存的 sciworld 对应物):以 root uid 为键。"""

    def __init__(self):
        self.task_desc = ""
        self.score = 0
        self.peak = 0
        self.obs = ""
        self.admissible = ""
        self.goal_progress = ""


class SciWorldRecursiveAdapter:

    def __init__(self, config=None, history_length: Optional[int] = None,
                 stuck_threshold: int = DEFAULT_STUCK_THRESHOLD,
                 stuck_window: int = DEFAULT_STUCK_WINDOW):
        hl = history_length
        if hl is None:
            try:
                hl = int(config.env.history_length)
            except Exception:
                hl = 2                               # textcraft/sciworld 口径(参数对照表)
        self.history_length = max(int(hl), 0)
        rao_cfg = {}
        try:
            rao_cfg = dict(config.env.get("rao", {}) or {})
        except Exception:
            pass
        self.stuck_threshold = int(rao_cfg.get("stuck_threshold", stuck_threshold))
        self.stuck_window = int(rao_cfg.get("stuck_window", stuck_window))
        self._trees: Dict[str, _TreeState] = {}
        self._answers: Dict[str, str] = {}           # child uid → 汇报文本
        self._open_peak: Dict[str, tuple] = {}       # child uid → (root_uid, 开张时峰值)

    # ================================================================ 目标
    def root_goal(self, env_info: Dict[str, Any], obs_text: str) -> Dict[str, str]:
        return {"task": str(env_info.get("extra.task_desc") or obs_text)}

    def goal_text(self, goal: Dict[str, str], is_root: bool = True) -> str:
        return str((goal or {}).get("task", "")).strip()

    # ================================================================ 动作侧
    def parse_delegation(self, action_text: str) -> Optional[DelegationRequest]:
        m = _DELEGATE_RE.match((action_text or "").strip())
        if m is None:
            return None
        sub = re.sub(r"\s+", " ", m.group(1)).strip()
        sub = _BUDGET_TAIL_RE.sub("", sub).strip()
        if not sub:
            return None                              # 空委托当普通动作,环境回 unmatched
        return DelegationRequest(goal={"task": sub}, context="", raw_action=action_text)

    def filler_action(self) -> str:
        return ""                                    # envs.py noop:不进 env、不改状态

    # ================================================================ 文本渲染
    def child_report_text(self, rec: NodeRecord) -> str:
        answer = self._answers.get(rec.uid, "")
        root_uid, open_peak = self._open_peak.get(rec.uid, ("", 0))
        tree = self._trees.get(root_uid)
        gain = (tree.peak - open_peak) if tree is not None else 0
        if answer:
            verdict = f"The sub-agent's report: {answer}"
        else:
            verdict = f"The sub-agent ended without a report ({rec.close_reason})."
        return (f"Sub-agent report for '{rec.goal_text}': {verdict}\n"
                f"Score gained during the sub-task: {gain}. "
                f"Budget used by subagent: {rec.turns}/{rec.budget_total} steps.")

    def delegation_refused_text(self, reason: str, detail: str) -> str:
        return f"Cannot launch a subagent. {detail}"

    # ================================================================ 树级状态
    def on_reset_begin(self) -> None:
        self._trees = {}
        self._answers = {}
        self._open_peak = {}

    def _tree(self, env_info: Dict[str, Any]) -> Optional[_TreeState]:
        return self._trees.get(str(env_info.get("root_node_uid") or ""))

    @staticmethod
    def _sync_tree(tree: _TreeState, env_info: Dict[str, Any]) -> None:
        if "extra.score" in env_info:
            tree.score = int(env_info["extra.score"])
        if "extra.peak" in env_info:
            tree.peak = max(tree.peak, int(env_info["extra.peak"]))
        if env_info.get("extra.admissible"):
            tree.admissible = str(env_info["extra.admissible"])
        if env_info.get("extra.goal_progress"):
            tree.goal_progress = str(env_info["extra.goal_progress"])

    # ================================================================ 状态与判定
    def on_node_open(self, node: Node, env_info: Dict[str, Any]) -> None:
        node.scratch.setdefault("context", "")
        if node.is_root:
            # root uid 在 reset 时还没被 stamp 进 info(search 适配器同款处理)
            tree = _TreeState()
            tree.task_desc = str(env_info.get("extra.task_desc") or "")
            tree.obs = str(env_info.get("extra.observation") or "")
            self._sync_tree(tree, env_info)
            self._trees[node.uid] = tree
            env_info["phi_after"] = float(100 - tree.peak)
            return
        tree = self._tree(env_info)
        peak = tree.peak if tree is not None else 0
        self._open_peak[node.uid] = (str(env_info.get("root_node_uid") or ""), peak)

    def on_turn_result(self, node: Node, env_info: Dict[str, Any]) -> None:
        tree = self._tree(env_info)
        if tree is None:
            return
        peak_before = tree.peak
        self._sync_tree(tree, env_info)
        # 最新世界观测进树缓存(子节点开张的首观测取这里;委托轮 env 走 noop 不改状态)
        action = node.history[-1].action if node.history else ""
        result = node.history[-1].result if node.history else ""
        kind = str(env_info.get("extra.last_action_kind") or "")
        if kind == "env":
            tree.obs = result
        # ---- Φ:峰值推进,归属到执行该步的行(mapping §一)
        delta = float(tree.peak - peak_before)
        env_info["delta_phi"] = float(env_info.get("delta_phi", 0.0)) + max(delta, 0.0)
        env_info["phi_after"] = float(100 - tree.peak)
        env_info["phi_frozen"] = False               # sciworld 无冻结语义
        # ---- 子 agent 汇报:记录文本(关闭在 node_outcome 里判)
        m = _ANSWER_RE.match((action or "").strip())
        if m and not node.is_root:
            self._answers[node.uid] = re.sub(r"\s+", " ", m.group(1)).strip()
        # ---- invalid 回折(mapping §一规则 3):unmatched 轮压掉有效标志
        if env_info.get("extra.env_unmatched"):
            env_info["is_action_valid"] = np.array(0)

    def _root_score(self, env_info: Dict[str, Any], node: Node) -> float:
        tree = self._trees.get(node.uid) if node.is_root else self._tree(env_info)
        score = tree.score if tree is not None else int(env_info.get("extra.score", 0))
        return max(min(int(score), 100), 0) / 100.0

    def _child_success(self, node: Node, env_info: Dict[str, Any]) -> float:
        if self._answers.get(node.uid):
            return 1.0
        root_uid, open_peak = self._open_peak.get(node.uid, ("", 0))
        tree = self._trees.get(root_uid)
        return 1.0 if (tree is not None and tree.peak > open_peak) else 0.0

    def node_outcome(self, node: Node, env_info: Dict[str, Any]) -> NodeOutcome:
        if not node.is_root and node.history:
            if _ANSWER_RE.match((node.history[-1].action or "").strip()):
                return NodeOutcome(done=True, success=self._child_success(node, env_info),
                                   reason=CLOSE_GOAL_MET)
        if node.is_root:
            tree = self._trees.get(node.uid)
            if tree is not None and tree.score >= 100:
                env_info["won"] = True
                return NodeOutcome(done=True, success=1.0, reason=CLOSE_GOAL_MET)
        if stuck_in_loop([t.action for t in node.history],
                         self.stuck_threshold, self.stuck_window):
            return NodeOutcome(done=True, success=self.evaluate_node(node, env_info),
                               reason=CLOSE_STUCK)
        return NodeOutcome(done=False)

    def evaluate_node(self, node: Node, env_info: Dict[str, Any]) -> float:
        if node.is_root:
            score = self._root_score(env_info, node)
            env_info["won"] = bool(score >= 1.0)
            return score
        return self._child_success(node, env_info)

    # ================================================================ 观测渲染
    def _current_observation(self, node: Node, is_first_turn: bool,
                             tree: Optional[_TreeState]) -> str:
        world = tree.obs if tree is not None else ""
        if node.is_root:
            lines = [f"Task: {node.goal_text}",
                     f"(You have {node.budget_total} steps of your own; "
                     "sub-agents get their own separate budget.)"]
        else:
            lines = [SUBAGENT_TASK_NOTE, f"Your sub-task: {node.goal_text}",
                     f"(You have {node.budget_total} steps for this sub-task; "
                     "report back with 'answer: ...' before they run out.)"]
            ctx = node.scratch.get("context") or ""
            if ctx:
                lines.append(f"Context provided from parent agent: {ctx}")
        if is_first_turn or not node.history:
            lines.append(f"\nCurrent observation: {world}")
        else:
            lines.append(f"\nResult of your last action: {node.last_result}")
        return "\n".join(lines)

    def build_observation(self, node: Node, is_first_turn: bool) -> str:
        tree = self._trees.get(node.uid) if node.is_root else \
            self._trees.get(self._open_peak.get(node.uid, ("", 0))[0])
        admissible = tree.admissible if tree is not None else ""
        cur = self._current_observation(node, is_first_turn, tree)
        tpl_no_his = SCIWORLD_RECURSIVE_TEMPLATE_NO_HIS if node.is_root \
            else SCIWORLD_RECURSIVE_CHILD_TEMPLATE_NO_HIS
        tpl_his = SCIWORLD_RECURSIVE_TEMPLATE if node.is_root \
            else SCIWORLD_RECURSIVE_CHILD_TEMPLATE
        if is_first_turn or not node.history or self.history_length <= 0:
            return tpl_no_his.format(current_observation=cur, admissible_actions=admissible)
        hist = node.history
        recent = hist[-self.history_length:]
        start = len(hist) - len(recent)
        lines = []
        for j, turn in enumerate(recent):
            # 历史行格式逐字照 memory.py 的 "Step {n}:{act} {obs}\n"(search 适配器同款)
            lines.append(f"Step {start + j + 1}:{turn.action} {turn.result}\n")
        return tpl_his.format(
            current_observation=cur,
            step_count=len(hist),
            history_length=len(recent),
            action_history="\n".join(lines),
            current_step=len(hist) + 1,
            admissible_actions=admissible,
        )
