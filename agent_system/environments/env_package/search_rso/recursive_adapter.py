# -*- coding: utf-8 -*-
"""Search-QA 的递归适配器:实现 agent_system/recursive/protocol.py 的 RecursiveEnvAdapter。

体例镜像 textcraft_synth/recursive_adapter.py;设计出处
Ideation/ideas/RSO_searchqa_data_mapping.md(2026-09-18 最终版)。要点:

  goal 载荷 = {"question": str}。root 的题面来自 env_kwargs(经 envs.py 写进
  info["extra.question"]);子节点的题面 = <delegate> 标签内容。

  Φ 事件(全在本适配器,底层环境与检索返回不参与判定):
    (d) 任一节点 <search> query 含某未解决跳的 answer      → on_turn_result
    (d) 任一节点 <delegate> 子任务文本含某未解决跳的 answer → on_node_open(child)
        (发生在委托轮,编排器在同一 info 上先 stamp 元数据、再调本钩子、
         最后抄 turn_meta——所以这里写 info["delta_phi"] 恰好记到父节点那一行)
    (c) 【子】节点 <answer> 含某未解决跳的 answer           → on_turn_result
    root <answer> 不进 Φ(归 A_out);<think>/检索返回不产生事件。

  终止与判分:
    节点最后一个动作是 <answer> → node_outcome 判 done;
    root success = em_check(答案, targets)(EM 归一化用 skyrl utils 原函数),
    并 stamp info["won"] 供 _process_batch 记成功率;
    子节点 success = 答案命中任一跳 answer(含已解决的)—— 无 decomp 的题
    (OOD)子节点 success 记 0,只作诊断,不进 A_out。
    节点级循环检测逐字复用 textcraft 适配器的 stuck_in_loop(官方 agent.py 口径)。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.search.utils import (
    em_check)
from agent_system.environments.env_package.search_rso.decomp import (
    SearchDecompTracker, load_decomp_store, norm_text)
from agent_system.environments.env_package.textcraft_synth.recursive_adapter import (
    stuck_in_loop, DEFAULT_STUCK_THRESHOLD, DEFAULT_STUCK_WINDOW)
from agent_system.environments.prompts.search_recursive import (
    SEARCH_RECURSIVE_CHILD_TEMPLATE, SEARCH_RECURSIVE_CHILD_TEMPLATE_NO_HIS,
    SEARCH_RECURSIVE_TEMPLATE, SEARCH_RECURSIVE_TEMPLATE_NO_HIS)
from agent_system.recursive.node import Node, NodeRecord
from agent_system.recursive.protocol import CLOSE_GOAL_MET, CLOSE_STUCK, DelegationRequest, NodeOutcome

import re

_DELEGATE_RE = re.compile(r"^\s*<delegate>(.*?)</delegate>\s*$", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.IGNORECASE | re.DOTALL)


class SearchRecursiveAdapter:
    """无状态约定与 textcraft 版一致:节点级状态放 Node.scratch,树级状态(tracker)
    以 root uid 为键挂在适配器上,on_reset_begin 清场。"""

    def __init__(self, config=None, history_length: Optional[int] = None,
                 stuck_threshold: int = DEFAULT_STUCK_THRESHOLD,
                 stuck_window: int = DEFAULT_STUCK_WINDOW):
        hl = history_length
        if hl is None:
            try:
                hl = int(config.env.history_length)
            except Exception:
                hl = 4                               # flat search 的框架默认(run_search_3b.sh)
        self.history_length = max(int(hl), 0)
        rao_cfg = {}
        srch_cfg = {}
        try:
            rao_cfg = dict(config.env.get("rao", {}) or {})
            srch_cfg = dict(config.env.get("search_rso", {}) or {})
        except Exception:
            pass
        self.stuck_threshold = int(rao_cfg.get("stuck_threshold", stuck_threshold))
        self.stuck_window = int(rao_cfg.get("stuck_window", stuck_window))
        decomp_path = srch_cfg.get("decomp_path")
        assert decomp_path, ("[search_rso] 缺 +env.search_rso.decomp_path=decomp_store.json 路径"
                             "(examples/data_preprocess/make_searchrso_data_products.py 生成)")
        self._store = load_decomp_store(str(decomp_path))
        self._trackers: Dict[str, SearchDecompTracker] = {}
        # NodeRecord 不带 history,战报要引用子答案 → 按 uid 留答案表(on_reset_begin 清场)
        self._answers: Dict[str, str] = {}

    # ================================================================ 目标
    def root_goal(self, env_info: Dict[str, Any], obs_text: str) -> Dict[str, str]:
        return {"question": str(env_info.get("extra.question") or obs_text)}

    def goal_text(self, goal: Dict[str, str], is_root: bool = True) -> str:
        # "Your question:" / "Your sub-question:" 前缀在模板里(flat 原文),这里只给裸问题
        return str((goal or {}).get("question", "")).strip()

    # ================================================================ 动作侧
    def parse_delegation(self, action_text: str) -> Optional[DelegationRequest]:
        m = _DELEGATE_RE.match((action_text or "").strip())
        if m is None:
            return None
        sub_q = re.sub(r"\s+", " ", m.group(1)).strip()
        if not sub_q:
            return None                              # 空委托当普通动作,环境回 invalid 观测
        return DelegationRequest(goal={"question": sub_q}, context="", raw_action=action_text)

    def filler_action(self) -> str:
        return ""                                    # envs.py 的 noop:不打检索、不改状态

    # ================================================================ 文本渲染
    def child_report_text(self, rec: NodeRecord) -> str:
        answer = self._answers.get(rec.uid, "")
        if answer:
            verdict = f"The sub-agent's answer: {answer}"
        else:
            verdict = f"The sub-agent FAILED to produce an answer ({rec.close_reason})."
        return (f"Sub-agent report for '{rec.goal_text}': {verdict}\n\n"
                f"Budget used by subagent: {rec.turns}/{rec.budget_total} steps.")

    def delegation_refused_text(self, reason: str, detail: str) -> str:
        return f"Not enough budget to launch subagent. {detail}"

    # ================================================================ 树级 tracker
    def on_reset_begin(self) -> None:
        self._trackers = {}
        self._answers = {}

    def _tracker(self, env_info: Dict[str, Any]) -> Optional[SearchDecompTracker]:
        return self._trackers.get(str(env_info.get("root_node_uid") or ""))

    def _stamp_phi(self, env_info: Dict[str, Any], tracker: SearchDecompTracker,
                   delta: float) -> None:
        env_info["delta_phi"] = float(env_info.get("delta_phi", 0.0)) + float(delta)
        env_info["phi_after"] = float(tracker.phi)
        env_info["phi_frozen"] = False               # search 无冻结语义(mapping §一)

    # ================================================================ 状态与判定
    def on_node_open(self, node: Node, env_info: Dict[str, Any]) -> None:
        node.scratch.setdefault("context", "")
        node.scratch["final_answer"] = None
        if node.is_root:
            entry = self._store.get(norm_text(str(env_info.get("extra.question") or "")))
            tracker = SearchDecompTracker((entry or {}).get("hops"))
            # root uid 在 reset 时还没被 stamp 进 info(_stamp_info 在 on_node_open 之后),
            # 直接用 node.uid——root 的 uid 就是树的 root_node_uid。
            self._trackers[node.uid] = tracker
            env_info["phi_after"] = float(tracker.phi)
            node.scratch["has_decomp"] = tracker.n_hops > 0
            return
        # ---- 子节点开张 = 父的 <delegate> 落地:Φ 事件通道 (d) 之委托承诺。
        # 本方法在委托轮、meta_rows 抄表之前被调用,写 info["delta_phi"] 记到父那一行。
        tracker = self._tracker(env_info)
        if tracker is not None and tracker.n_hops:
            sub_q = str((node.goal or {}).get("question", ""))
            hits_before = tracker.scan(sub_q)
            delta = tracker.resolve_events(sub_q, via="delegate", node_uid=node.parent_uid or "")
            self._stamp_phi(env_info, tracker, delta)
            # 委托与剩余清单是否吻合(开张时判):不吻合 = useless_goal(§四 监控)
            node.scratch["useless_goal"] = (len(hits_before) == 0)

    def on_turn_result(self, node: Node, env_info: Dict[str, Any]) -> None:
        action = node.history[-1].action if node.history else ""
        tracker = self._tracker(env_info)
        m = _SEARCH_RE.search(action or "")
        if m:
            node.scratch.setdefault("queries", []).append(m.group(1).strip())
            if tracker is not None and tracker.n_hops:
                delta = tracker.resolve_events(m.group(1), via="search", node_uid=node.uid)
                self._stamp_phi(env_info, tracker, delta)
            return
        m = _ANSWER_RE.search(action or "")
        if m:
            ans = m.group(1).strip()
            node.scratch["final_answer"] = ans
            self._answers[node.uid] = ans
            if (not node.is_root) and tracker is not None and tracker.n_hops:
                delta = tracker.resolve_events(ans, via="answer", node_uid=node.uid)
                self._stamp_phi(env_info, tracker, delta)

    def _root_em(self, node: Node, env_info: Dict[str, Any]) -> float:
        ans = node.scratch.get("final_answer")
        targets = list(env_info.get("extra.ground_truth_targets") or [])
        if not ans or not targets:
            return 0.0
        return 1.0 if em_check(ans, targets) else 0.0

    def _child_score(self, node: Node, env_info: Dict[str, Any]) -> float:
        ans = node.scratch.get("final_answer")
        if not ans:
            return 0.0
        tracker = self._tracker(env_info)
        if tracker is None or not tracker.n_hops:
            return 0.0
        return 1.0 if tracker.scan(ans, include_resolved=True) else 0.0

    def node_outcome(self, node: Node, env_info: Dict[str, Any]) -> NodeOutcome:
        if node.scratch.get("final_answer") is not None:
            success = self._root_em(node, env_info) if node.is_root \
                else self._child_score(node, env_info)
            if node.is_root:
                env_info["won"] = bool(success >= 1.0)
            return NodeOutcome(done=True, success=success, reason=CLOSE_GOAL_MET)
        if stuck_in_loop([t.action for t in node.history],
                         self.stuck_threshold, self.stuck_window):
            return NodeOutcome(done=True, success=self.evaluate_node(node, env_info),
                               reason=CLOSE_STUCK)
        return NodeOutcome(done=False)

    def evaluate_node(self, node: Node, env_info: Dict[str, Any]) -> float:
        if node.is_root:
            score = self._root_em(node, env_info)
            env_info["won"] = bool(score >= 1.0)
            return score
        return self._child_score(node, env_info)

    # ================================================================ 观测渲染
    def _task_description(self, node: Node) -> str:
        """填模板的 {task_description}(模板自带 "Your question:" 前缀,flat 原文)。
        裸问题 + 递归特有的两行(父 context / 步数预算,对应 textcraft 版的 task 行)。"""
        lines = [node.goal_text]
        ctx = node.scratch.get("context") or ""
        if ctx:
            lines.append(f"Context provided from parent agent: {ctx}")
        lines.append(f"Budget: you have {node.budget_total} steps in total.")
        return "\n".join(lines)

    def build_observation(self, node: Node, is_first_turn: bool) -> str:
        """占位符与历史渲染完全照 flat:SEARCH_TEMPLATE 的 task_description /
        step_count / memory_context;历史行格式逐字照 memory.py:178 的
        "Step {n}:{act} {obs}\\n"(最近 H 轮,含最新一轮——flat 的最新结果就在
        历史末行里,没有单独的"上轮结果"段)。"""
        tpl_no_his = SEARCH_RECURSIVE_TEMPLATE_NO_HIS if node.is_root \
            else SEARCH_RECURSIVE_CHILD_TEMPLATE_NO_HIS
        tpl_his = SEARCH_RECURSIVE_TEMPLATE if node.is_root \
            else SEARCH_RECURSIVE_CHILD_TEMPLATE
        task = self._task_description(node)
        if is_first_turn or not node.history or self.history_length <= 0:
            return tpl_no_his.format(task_description=task)

        hist = node.history
        recent = hist[-self.history_length:]
        start = len(hist) - len(recent)
        lines = []
        for j, turn in enumerate(recent):
            lines.append(f"Step {start + j + 1}:{turn.action} {turn.result}\n")
        return tpl_his.format(
            task_description=task,
            step_count=len(hist),
            memory_context="\n".join(lines),
        )
