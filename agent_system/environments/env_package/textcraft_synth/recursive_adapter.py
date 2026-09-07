# -*- coding: utf-8 -*-
"""TextCraft-Synth 的递归适配器:实现 agent_system/recursive/protocol.py 的 RecursiveEnvAdapter。

这是"环境特有"的那一层。换 benchmark 只需要另写一个这样的类,内核 orchestrator.py 不改。

移植来源(路径相对 Experiment/code_references/platoon/,行号为**本地副本**的行号):
  plugins/textcraft/platoon/textcraft/env.py:334-351   launch_subagent:targets dict → goal 字符串
      :339-340   target_str = ", ".join(f"{count}x {item}") ; goal = "Craft the following items: ..."
      :344-345   有 context 则追加 "\n\nContext provided from parent agent: {context}"
  plugins/textcraft/platoon/textcraft/env.py:499        depth-aware 版签名:无 num_steps
  plugins/textcraft/platoon/textcraft/env.py:596        self._initial_inventory 快照(fork 时拷贝)
  plugins/textcraft/platoon/textcraft/env.py:628-640    evaluate:current − initial >= required(root 沿用)
  plugins/textcraft/platoon/textcraft/env.py:674-700    fork 时把 goal 字符串反解析回 target_items
  platoon/agents/codeact/agent.py:75-76, :90-109        每-agent 的循环检测(threshold=4, window=3,周期模式)
  platoon/agents/codeact/agent.py:111-125               触发后该 agent 自己 finish,只有它结束
  platoon/agents/actions/subagent.py:58-63              委托被拒:拼文本 return,不抛异常
  platoon/agents/actions/subagent.py:89-98              战报:finish/error 文本 + "Budget used by subagent: a/b steps"
  platoon/envs/base.py:36-39                            Task.__str__:"Your Goal: ...\\nBudget: ... steps"
  platoon/agents/codeact/prompt_builder.py:63-79        子 agent 的对话从空开始(上下文隔离)

对应 flat 侧的复用:
  synth_core.py:103-105 / :198-232   文本动作文法(inventory / get_info / craft),本文件只加 delegate
  synth_core.py:146-157              状态块的措辞(配方笔记本 + 库存快照),本文件按【节点】维护同样的东西
  env_manager.py:770-799 + memory.py:170-183   滑窗历史的渲染格式 "Step k:{act} {obs}"
  projection.py                      <action> 抠取,原样复用(注意它会把整行小写)

状态块作用域(用户 2026-08-24 决定:按节点私有,对应官方子 agent 全新对话):
  flat 里状态块由环境维护、每局一份(synth_core._state_block)。递归模式下环境被告知不再拼状态块
  (append_state_block=False),本适配器在每个 Node.scratch 里维护该节点自己查过的配方和自己做过的
  库存快照——信息制度与 flat 完全一致(只重放模型自己挣来的信息),只是作用域从"每局"缩到"每节点"。
  配置项 env.rao.state_block_scope 目前只接受 "node";"tree"(全树共享)留作将来消融,未实现。

===========================================================================================
子目标判定口径(2026-08-25,用户决定,是对官方的【有意偏离】,论文里要声明)
===========================================================================================
官方子节点沿用 TextCraftEnv.evaluate 的净增量口径:子开张时拷贝库存(env.py:596),关闭时要求
current − initial >= required(env.py:632-640,注释原话 "prevents giving credit for items that were
already in starting inventory")。这意味着"需要 3 个、库存已有 2 个、委托 3x"时,子要再做 3 个(总 5 个)。
冒烟 21461096 验证集:7/16 棵树的子节点因此"自认完成(按绝对数量)却判失败 → 无事可做 → 打转"。
而父委托的真实语义是它的配方需要"库存里【有】N 个",是绝对数量;多做的每一份都在吃 root 后面要用
的原料。所以子节点改用:
  子成功 ⇔ 任期内某一刻,库存里每个目标物品的数量 >= 目标数(绝对可用量)。
  开张即判:压栈那一刻就算一次,已满足则 0 步关闭(goal_met,record.preexisting=True),父只付 1 轮。
  功劳归属无歧义:开张时不够、关闭时够了,只可能是这个子及其后代做的(父挂起、兄弟不并行)。
  不可被刷:0 步关闭的子没有训练行;有行的子必然做出了缺口部分;λ=0 时子成败不进父的奖励。
root 的判定不变(净增量;数据集保证初始库存无目标物,等价于绝对量),与 flat / 官方一致。
子节点的目标行措辞与 HEADER 的 Note 句相应改为"库存里有 N 个即可"(prompts/textcraft_synth_recursive.py
的 _HEADER_CHILD),否则与官方那句 "on top of what you already have" 自相矛盾。

===========================================================================================
节点级循环检测(2026-08-25,生命周期下沉的一部分)
===========================================================================================
flat 的循环检测在 synth_core 里、按环境槽计数,分不清动作是树上哪个节点发的,一个子节点打转会把
整棵树杀掉(冒烟验证集 7/16 棵树这样死的)。官方的检测是每个 agent 各自做的:agent.py:90-109 只看
本 agent 的 history,判"周期 ≤ window(3) 的模式重复 ≥ threshold(4) 次",触发后该 agent 自己
finish(:111-125),只有它结束。本适配器在 node_outcome 里逐行镜像那段算法,作用于 node.history;
递归模式下环境级检测由工厂关闭(loop_detection=False)。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from agent_system.environments.env_package.textcraft_synth.synth_core import NOOP_ACTION
from agent_system.environments.prompts.textcraft_synth_recursive import (
    TEXTCRAFT_SYNTH_RECURSIVE_CHILD_TEMPLATE,
    TEXTCRAFT_SYNTH_RECURSIVE_CHILD_TEMPLATE_NO_HIS,
    TEXTCRAFT_SYNTH_RECURSIVE_TEMPLATE,
    TEXTCRAFT_SYNTH_RECURSIVE_TEMPLATE_NO_HIS,
)
from agent_system.recursive.node import Node, NodeRecord
from agent_system.environments.env_package.textcraft_synth.phi import PhiTracker
from agent_system.recursive.protocol import CLOSE_GOAL_MET, CLOSE_STUCK, DelegationRequest, NodeOutcome

# 委托文法。示例(projection 已把整行小写):
#   delegate: craft 4 a0_i1
#   delegate craft 4 a0_i1, 2 a0_i3 | these are for a1_i2
#   delegate: craft 4x a0_i1 and 2x a0_i3
# 冒号可省;数量后可带 x;多目标用逗号或 and 分隔;竖线后是可选 context。
#
# 2026-08-25 冒烟(作业 21459012)暴露的模型真实写法(tips #16):模型把父节点笔记本里的配方行
# "craft N item using ..." 整行抄进委托,还会在 using 配料表里带逗号、在多个 | 段各写一个 craft:
#   delegate: craft 2 t0_i2_10 using 2 t2_i1_12, 2 t1_i1, craft 2 t7_i1 using 2 raw_t0 | needed for t6_i3_16
#   delegate: craft 2 m5_i1 using 1 m4_ore | needed for m5_i2, craft 4 m5_i1_15 using 1 m7_ore | these too
# 解析规则(用户 2026-08-25 决定):
#   1) 第一个竖线之前是目标段:using 子句整段剥离(从 using 延伸到下一个 craft 关键字 / 句尾);
#      按 逗号 / and / craft 关键字 切分,每段必须是 "N item",任一段解析不了 → 整条拒绝(返回 None),
#      绝不静默取子集——旧实现静默跳过坏段,导致 13 例子节点拿到被截短的目标却报 SUCCEEDED;
#   2) 竖线之后【原样】保留为 context——这是官方 launch_subagent(targets, context) 的 context 通道,
#      父把自己查到的配方写在这里传给子(子是全新上下文,父不传就没有),using 绝不能剥、文字绝不能改;
#   3) context 里出现的 "craft N item",只有当 item 不在目标里时才补为目标(覆盖模型把第二个目标写到
#      竖线后的写法,冒烟 4 例);已在目标里的视为父在传配方,不重复计数。
_DELEGATE_PREFIX_RE = re.compile(r"^\s*delegate\s*:?\s*", re.IGNORECASE)
_USING_CLAUSE_RE = re.compile(r"\busing\b.*?(?=\bcraft\b|\||$)", re.DOTALL | re.IGNORECASE)
_SEGMENT_SPLIT_RE = re.compile(r",|\band\b|\bcraft\b", re.IGNORECASE)
_TARGET_RE = re.compile(r"^\s*(\d+)\s*x?\s+([A-Za-z0-9_]+)[\s\.;:!]*$")
_CRAFT_TOKEN_RE = re.compile(r"\bcraft\s+(\d+)\s*x?\s+([A-Za-z0-9_]+)", re.IGNORECASE)

# 官方 agent.py:75-76 的默认值
DEFAULT_STUCK_THRESHOLD = 4
DEFAULT_STUCK_WINDOW = 3


def stuck_in_loop(codes: List[str], threshold: int = DEFAULT_STUCK_THRESHOLD,
                  window: int = DEFAULT_STUCK_WINDOW) -> bool:
    """逐行镜像 platoon/agents/codeact/agent.py:90-109 的 _stuck_in_loop。

    codes 是本 agent 自己的动作文本序列。判据:存在周期 period ∈ [1, window],使得最近
    period × threshold 个动作恰好是同一个长度为 period 的模式反复出现。
    官方 :95 的 max_period = min(window, n // threshold) 保证序列够长才检查。
    """
    n = len(codes)
    if n < threshold:
        return False
    codes = [c.strip() for c in codes]
    max_period = min(window, n // threshold)
    if max_period <= 0:
        return False
    for period in range(1, max_period + 1):
        segment_len = period * threshold
        segment = codes[-segment_len:]
        pattern = segment[:period]
        if all(segment[i] == pattern[i % period] for i in range(segment_len)):
            return True
    return False


class TextCraftSynthRecursiveAdapter:
    """RecursiveEnvAdapter 的 TextCraft-Synth 实现。无状态(节点级状态全放在 Node.scratch)。"""

    def __init__(self, config=None, history_length: Optional[int] = None,
                 state_block_scope: str = "node",
                 stuck_threshold: int = DEFAULT_STUCK_THRESHOLD,
                 stuck_window: int = DEFAULT_STUCK_WINDOW):
        hl = history_length
        if hl is None:
            try:
                hl = int(config.env.history_length)
            except Exception:
                hl = 2                                   # flat 的框架默认值(tips #8)
        self.history_length = max(int(hl), 0)
        rao_cfg = {}
        try:
            rao_cfg = dict(config.env.get("rao", {}) or {})
        except Exception:
            pass
        scope = str(rao_cfg.get("state_block_scope", state_block_scope))
        if scope != "node":
            raise NotImplementedError(
                f"state_block_scope={scope!r} 未实现;当前只支持 'node'(官方口径:子 agent 全新上下文)")
        self.state_block_scope = scope
        # ---- [RSO] Φ 进展追踪(设计出处 RSO_advantage_design.md §三;每棵树一个追踪器,
        # 键 = root 节点 uid。K(已查配方并集)是全树共享的,所以挂在树上而不是节点上。
        # phi_gamma 是配方项权重(Φ = C + γ·|M\K|),默认 1。
        rso_cfg = {}
        try:
            rso_cfg = dict(config.env.get("rso", {}) or {})
        except Exception:
            pass
        self.phi_gamma = float(rso_cfg.get("phi_gamma", 1.0))
        self._phi_trackers = {}
        self._phi_warned = False
        self.stuck_threshold = int(rao_cfg.get("stuck_threshold", stuck_threshold))
        self.stuck_window = int(rao_cfg.get("stuck_window", stuck_window))

    # ================================================================ 目标
    def root_goal(self, env_info: Dict[str, Any], obs_text: str) -> Dict[str, int]:
        """root 的目标来自数据集(官方 env.py:565 读 task.misc["target_items"])。"""
        return dict(env_info["extra.target_items"])

    def goal_text(self, goal: Dict[str, int], is_root: bool = True) -> str:
        """root:env.py:339-340 的格式(数据集里 root 的 goal 字符串就是它)。
        子:措辞改为"库存里有 N 个即可",与子目标的绝对可用量口径一致(见头注释)。"""
        items = ", ".join(f"{n}x {item}" for item, n in goal.items())
        if is_root:
            return "Craft the following items: " + items
        return "Make sure your inventory contains at least: " + items

    # ================================================================ 动作侧
    def parse_delegation(self, action_text: str) -> Optional[DelegationRequest]:
        """认出委托动作。不以 delegate 开头 → None(交给环境当普通动作)。
        目标段有任一片段解析不了 → 也返回 None(环境回 "unknown action",靠观测反馈纠正)。
        官方这一步不存在——Python 解释器就是它的解析器(envs/codeact/env.py:91 exec 代码,
        launch_subagent 是注入 shell 命名空间的函数,:353-354)。规则见文件头。
        """
        text = (action_text or "").strip()
        pm = _DELEGATE_PREFIX_RE.match(text)
        if pm is None:
            return None
        body = text[pm.end():]
        head, sep, tail = body.partition("|")
        context = re.sub(r"\s+", " ", tail).strip(" ,;.") if sep else ""

        head = _USING_CLAUSE_RE.sub(" ", head)             # 规则 1:只对目标段剥离配方尾巴
        goal: Dict[str, int] = {}
        segs = [s for s in _SEGMENT_SPLIT_RE.split(head) if s.strip()]
        if not segs:
            return None
        for s in segs:
            tm = _TARGET_RE.match(s)
            if tm is None:
                return None                                 # 整条拒绝,绝不静默取子集
            n, item = int(tm.group(1)), tm.group(2)
            if n <= 0:
                return None
            goal[item] = goal.get(item, 0) + n

        for cm in _CRAFT_TOKEN_RE.finditer(tail):          # 规则 3
            n, item = int(cm.group(1)), cm.group(2)
            if n > 0 and item not in goal:
                goal[item] = n
        return DelegationRequest(goal=goal, context=context, raw_action=text)

    def filler_action(self) -> str:
        """委托那轮给环境的占位。用 synth_core.NOOP_ACTION:只计步、不动状态、不进循环检测。"""
        return NOOP_ACTION

    # ================================================================ 文本渲染
    def child_report_text(self, rec: NodeRecord) -> str:
        """对应 subagent.py:89-98。官方返回 (finish_message or error_message) + 预算用量。
        我们没有 finish 动作,用成败判定代替 finish_message;出错时附 traceback 的首行
        (官方 error_message 也是回给父的文本);预算行照官方措辞。"""
        if rec.success >= 1.0 and rec.turns == 0:
            verdict = ("SUCCEEDED immediately: the requested items were already available in your "
                       "inventory, so no sub-agent steps were needed.")
        elif rec.success >= 1.0:
            verdict = "SUCCEEDED: the requested items are now available in your inventory."
        else:
            verdict = f"FAILED ({rec.close_reason}): the requested items are not all available."
        text = (f"Sub-agent report for '{rec.goal_text}': {verdict}\n\n"
                f"Budget used by subagent: {rec.turns}/{rec.budget_total} steps.")
        if rec.error:
            first = rec.error.strip().splitlines()[0] if rec.error.strip() else "error"
            text += f"\nSub-agent error: {first}"
        return text

    def delegation_refused_text(self, reason: str, detail: str) -> str:
        """对应 subagent.py:60-62;detail 已含官方 message + guidance 原文(见 budget.py)。"""
        return f"Not enough budget to launch subagent. {detail}"

    # ================================================================ 状态与判定
    def _satisfied(self, node: Node, env_info: Dict[str, Any]) -> bool:
        """root:净增量(env.py:632-640,与 flat 一致);子:绝对可用量(见头注释)。"""
        inv = env_info.get("extra.inventory", {})
        if node.is_root:
            base = node.open_snapshot.get("inventory", {})
            return all(inv.get(k, 0) - base.get(k, 0) >= v for k, v in node.goal.items())
        return all(inv.get(k, 0) >= v for k, v in node.goal.items())

    def on_reset_begin(self) -> None:
        """[RSO] 新一局开始:清掉上一局的 Φ 追踪器(编排器 reset 时调用,可选钩子)。"""
        self._phi_trackers = {}

    def on_node_open(self, node: Node, env_info: Dict[str, Any]) -> None:
        """env.py:596:记开张时的库存(root 判净增量要用)。初始化该节点的私有状态块。
        同时记 preexisting 诊断量:子开张时目标就已满足 = 浪费型委托,编排器紧接着会 0 步关闭它。"""
        node.open_snapshot = {"inventory": dict(env_info.get("extra.inventory", {}))}
        node.scratch.setdefault("context", "")
        node.scratch["known_recipes"] = {}
        node.scratch["inv_snapshot"] = None
        node.scratch["inv_step"] = None
        node.scratch["preexisting"] = (not node.is_root) and self._satisfied(node, env_info)
        # ---- [RSO] root 开张即建该树的 Φ 追踪器(整棵树共用一个 Φ,目标固定为 root 目标)。
        # 防御:单元测试可能用极简 info 调本方法,缺目标/库存时跳过(该树 ΔΦ 恒 0)。
        if node.is_root:
            targets = env_info.get("extra.target_items")
            inv = env_info.get("extra.inventory")
            if targets and inv is not None:
                from agent_system.environments.env_package.textcraft_synth.synth_core import (
                    get_shared_recipe_db)
                tr = PhiTracker(get_shared_recipe_db(), gamma=self.phi_gamma)
                tr.reset(targets, inv)
                self._phi_trackers[node.uid] = tr
                env_info["phi_after"] = float(tr.phi)
            elif not self._phi_warned:
                self._phi_warned = True
                print("[RSO adapter] env_info 缺 extra.target_items/extra.inventory,"
                      "该树不建 Φ 追踪器(ΔΦ 恒 0)。真实训练里不应出现这条。")

    def on_turn_result(self, node: Node, env_info: Dict[str, Any]) -> None:
        """把本轮的结构化结果记进该节点私有状态块。措辞与 synth_core._get_info / _execute 一致。"""
        last = env_info.get("extra.last_get_info")
        if last:
            for entry in last:
                item = entry.get("item")
                recipes = entry.get("recipes") or []
                if recipes:
                    r = recipes[0]                        # synth 每物品只有一个配方
                    ings = ", ".join(f"{c} {n}" for n, c in r["ingredients"].items())
                    node.scratch["known_recipes"][item] = (
                        f"craft {r['result_count']} {item} using {ings}"
                        f"   (depth {entry.get('crafting_depth', -1)})")
                elif entry.get("is_base"):
                    node.scratch["known_recipes"][item] = f"{item}: base ingredient, cannot be crafted"
        if node.history and node.history[-1].action.strip() == "inventory":
            inv = env_info.get("extra.inventory", {})
            node.scratch["inv_snapshot"] = (
                ", ".join(f"{k}: {v}" for k, v in sorted(inv.items())) if inv else "(empty)")
            node.scratch["inv_step"] = node.turns_used

        # ---- [RSO] 真实环境轮之后推进该树的 Φ。root_node_uid 由编排器在调用本方法前
        # stamp 进 info(orchestrator._advance_slot 先 _stamp_info 再 on_turn_result);
        # 委派占位轮不经过本方法,其 delta_phi 由 turn_meta 的缺省 0 承担——语义正确:
        # 那一轮底层环境收到的是 filler,状态没变。
        root_uid = str(env_info.get("root_node_uid") or "")
        tracker = self._phi_trackers.get(root_uid)
        if tracker is not None:
            delta, phi_after, frozen = tracker.update(
                env_info.get("extra.inventory") or {},
                env_info.get("extra.last_get_info"))
            env_info["delta_phi"] = float(delta)
            env_info["phi_after"] = float(phi_after)
            env_info["phi_frozen"] = bool(frozen)

    def node_outcome(self, node: Node, env_info: Dict[str, Any]) -> NodeOutcome:
        """先判目标,再判本节点自己的循环(官方 agent.py:90-118:触发 → 自己 finish → 照常判分)。"""
        if self._satisfied(node, env_info):
            return NodeOutcome(done=True, success=1.0, reason=CLOSE_GOAL_MET)
        if stuck_in_loop([t.action for t in node.history], self.stuck_threshold, self.stuck_window):
            return NodeOutcome(done=True, success=self.evaluate_node(node, env_info), reason=CLOSE_STUCK)
        return NodeOutcome(done=False)

    def evaluate_node(self, node: Node, env_info: Dict[str, Any]) -> float:
        return 1.0 if self._satisfied(node, env_info) else 0.0

    # ================================================================ 观测渲染
    def _task_lines(self, node: Node) -> str:
        """对应官方 Task.__str__(envs/base.py:36-39)+ context 追加(env.py:344-345)。
        格式与 flat 的 root 初始观测(synth_core.reset:143)一致:goal 行 + Budget 行。"""
        lines = [node.goal_text]
        ctx = node.scratch.get("context") or ""
        if ctx:
            lines.append(f"Context provided from parent agent: {ctx}")
        lines.append(f"Budget: you have {node.budget_total} steps in total.")
        return "\n".join(lines)

    def _state_block(self, node: Node) -> str:
        """与 synth_core._state_block 逐字同款,只是数据来自该节点私有的 scratch。"""
        lines: List[str] = []
        if node.scratch.get("inv_snapshot") is not None:
            lines.append(f"Your inventory as of step {node.scratch['inv_step']} "
                         f"(when you last checked): {node.scratch['inv_snapshot']}")
        kr = node.scratch.get("known_recipes") or {}
        if kr:
            lines.append("Recipes you have learned so far (from get_info):")
            for item in sorted(kr):
                lines.append(f"  {kr[item]}")
        return ("\n" + "\n".join(lines)) if lines else ""

    def build_observation(self, node: Node, is_first_turn: bool) -> str:
        """只用 node 自己的 goal / context / history / scratch 渲染——上下文隔离的落实处。
        结构与 flat 的 TextCraftSynthEnvironmentManager.build_text_obs(env_manager.py:770-799)一致;
        子节点用 _HEADER_CHILD 版模板(Note 句描述绝对可用量口径)。"""
        tpl_no_his = TEXTCRAFT_SYNTH_RECURSIVE_TEMPLATE_NO_HIS if node.is_root else TEXTCRAFT_SYNTH_RECURSIVE_CHILD_TEMPLATE_NO_HIS
        tpl_his = TEXTCRAFT_SYNTH_RECURSIVE_TEMPLATE if node.is_root else TEXTCRAFT_SYNTH_RECURSIVE_CHILD_TEMPLATE
        task = self._task_lines(node)
        if is_first_turn or not node.history:
            return tpl_no_his.format(current_observation=task)

        current = task + "\n\nResult of your last action: " + node.last_result + self._state_block(node)
        if self.history_length <= 0:
            return tpl_no_his.format(current_observation=current)

        # 滑窗:镜像 memory.py:170-183 的 "Step k:{act} {obs}",obs 是【动作前】的观测——
        # 第 1 轮的"动作前观测"是任务行,之后是上一轮的结果(与 env_manager.py:741-743 的记账一致)。
        H = self.history_length
        hist = node.history
        recent = hist[-H:]
        start = len(hist) - len(recent)
        lines = []
        for j, turn in enumerate(recent):
            k = start + j
            obs_before = task if k == 0 else hist[k - 1].result
            lines.append(f"Step {k + 1}:{turn.action} {obs_before}\n")
        return tpl_his.format(
            current_observation=current,
            step_count=len(hist),
            history_length=len(recent),
            action_history="\n".join(lines),
            current_step=len(hist) + 1,
        )
