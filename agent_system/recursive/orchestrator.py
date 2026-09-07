# -*- coding: utf-8 -*-
"""递归执行树的 lockstep 状态机。这是 RAO 官方 episode 层在批处理架构下的等价物。

移植来源(路径相对 Experiment/code_references/platoon/,行号为**本地副本**的行号,
本地副本加过读码笔记注释,与上游原版有位移):

  platoon/episode/loop.py:44-115      run_episode:单个 agent 的 reset → while(act → step) 循环
      :64        obs = await env.reset()
      :70-77     while not halt_episode(obs): action = agent.act(obs); obs = env.step(action)
      :79-97     异常留档(见 trace.py)
      :98        finally 归档轨迹
  platoon/episode/loop.py:140-146     halt_episode:三种终止条件
      :141       exhausted_budget = remaining_budget() <= 0      → 本文件按 node.budget_exhausted 判
      :145-146   obs.finished(目标达成)                        → 本文件交适配器 node_outcome 判
      (整局结束由底层环境的 done 驱动)
  platoon/agents/actions/subagent.py:41-100   委托的完整时序
      :41        subtask = task.fork(goal, ...)                  → 建子节点,goal 来自 DelegationRequest
      :50-51     forked_agent / forked_env                       → 子节点 depth+1、parent_uid、fork_step
      :57        reserve_budget(...)  depth-aware 下只查深度     → budget.can_delegate
      :58-63     被拒:把理由拼成文本 return 给模型              → delegation_refused_text 当作父这轮的结果
      :76        traj = await create_task(run_episode(子))      → 压栈,之后各轮由子出招
      :84        release_budget  (depth-aware 下空操作)
      :89-98     算用量、拼战报、返回 finish_message or error_message → 弹栈时 child_report_text 写回父
  plugins/textcraft/platoon/textcraft/env.py:487-524
      recursive executor 记账"本步派了哪些子、各自成败"          → 弹栈时 parent.note_child_result
  platoon/episode/trajectory.py:178   to_dict() 导出全树           → collect_node_records

核心翻转(为什么长得和官方不一样,详见 protocol.py 头注释与 tips #15):
  官方每个 agent 是一个协程,自己 while 循环、自己调模型;递归 = 起一个子协程并 await 它。
  本框架一次批量生成 128 个槽的回复,不存在"某个 agent 单独调模型"的操作。所以官方的
  "每个 agent 一个 while"被翻转成"外层按轮批量推进,内层每个槽记着此刻轮到树上哪个节点"。
  每个槽持有一个节点栈:栈底是 root,栈顶是正在出招的节点。委托 = 压栈,子节点关闭 = 弹栈。
  任意时刻每个槽只有栈顶在行动——这恰好就是官方"父 await 子、串行深度优先"的执行顺序。

lockstep 特有、官方没有的两处(设计文档 §6 已记):
  1. filler 动作:委托那一轮仍要给底层环境发一个不改状态的动作占位(官方父协程直接挂起);
  2. 全局轮数上限:采集循环的 for 轮数(config.env.max_steps)。官方每 agent 各卡 25 步、
     树的总步数无界;lockstep 的 for 循环天然需要一个上限,到了就按 CLOSE_ROLLOUT_END 收摊。

本文件不知道 TextCraft 的存在:所有环境相关的判断都经 protocol.RecursiveEnvAdapter 回调。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.recursive.budget import PerAgentBudget
from agent_system.recursive.node import Node, NodeRecord
from agent_system.recursive.protocol import (
    CLOSE_BUDGET_EXHAUSTED, CLOSE_EPISODE_END, CLOSE_ERROR, CLOSE_OVERFLOW, CLOSE_ROLLOUT_END,
    RecursiveEnvAdapter,
)
from agent_system.recursive import trace
from agent_system.recursive.tree_trace import TreeTraceWriter


# 委托刚发出、子还没回来时,父那一轮暂时记的结果。子关闭后会被战报覆盖。
_PENDING_RESULT = "(delegated; waiting for the sub-agent to finish)"


class RecursiveEnvironmentManager(EnvironmentManagerBase):
    """把任意支持 RecursiveEnvAdapter 的环境变成递归执行树。

    对外接口与 TextCraftSynthEnvironmentManager 完全一致(reset / step / success_evaluator /
    close),采集循环 rollout_loop.py 不需要知道它在跑递归;额外多出:
      * step 返回的 infos 里每槽带节点元数据:node_uid / node_depth / node_goal /
        root_node_uid / is_delegation_turn;
      * collect_node_records():采集循环结束后调,拿到每槽的执行树记录(List[NodeRecord]),
        训练侧据此回填节点成败、算 RAO 优势。

    参数:
        envs         底层批量环境(如 TextCraftSynthEnvs),与 flat 完全相同的实例
        projection_f 模型文本 → 动作文本的解析(与 flat 相同)
        config       全局配置;本类读 config.env.rao.* 和 config.env.max_steps
        adapter      RecursiveEnvAdapter 的实现
        budget       PerAgentBudget;None 则从 config.env.rao 构造
    """

    def __init__(self, envs, projection_f, config,
                 adapter: RecursiveEnvAdapter,
                 budget: Optional[PerAgentBudget] = None,
                 trace_tag: str = "train"):
        super().__init__(envs, projection_f, config)
        if not isinstance(adapter, RecursiveEnvAdapter):
            raise TypeError(f"adapter 缺少 RecursiveEnvAdapter 要求的方法: {type(adapter)}")
        self.adapter = adapter

        rao_cfg = {}
        try:
            rao_cfg = dict(config.env.get("rao", {}) or {})
        except Exception:
            pass
        self.budget = budget or PerAgentBudget.from_config(rao_cfg)
        # prompt 超长守卫(继承自 flat 的 TextCraftSynthEnvironmentManager,tips #12):
        # 超限 = 该 episode 记失败、训练继续,而不是整个 job 崩溃。None 表示关闭守卫。
        self.max_obs_chars: Optional[int] = rao_cfg.get("max_obs_chars", 26000)
        # 完整轨迹记录器(env.rao.trace_dir 为空则关闭)。见 tree_trace.py 头注释。
        self.tracer = TreeTraceWriter(rao_cfg.get("trace_dir", None), tag=trace_tag)
        self.last_prompt: List[str] = []      # 每槽上一轮渲染给模型的 prompt 原文(turn 事件用)

        # 每槽状态,reset 时初始化
        self.stacks: List[List[Node]] = []
        self.episode_done: np.ndarray = np.zeros(0, dtype=bool)
        self.finished_records: List[List[NodeRecord]] = []
        self.last_infos: List[Dict[str, Any]] = []
        self.node_open_round: Dict[str, int] = {}
        self.round: int = 0
        # 轮次元数据(2026-08-24 步骤 5 回溯加):turn_meta[round][slot] = 该槽该轮的节点元数据。
        # 收集器的 gather_rollout_data 按 (slot, step_idx) 查它,把 node_uid 等字段贴到训练行上。
        # 采集循环里 total_batch_list[slot][step_idx] 每轮每槽恰好一条(含已结束的陪跑槽),
        # 所以 step_idx 与这里的 round 下标一一对应;收集器会断言二者长度一致防错位。
        self.turn_meta: List[List[Dict[str, Any]]] = []

    # ================================================================== reset
    def reset(self, kwargs=None):
        """对应 loop.py:64 的 env.reset() 加 loop.py:117-138 的 set_context_vars 建 root 轨迹。"""
        obs, infos = self.envs.reset()
        n = len(obs)
        self.round = 0
        self.stacks = []
        self.episode_done = np.zeros(n, dtype=bool)
        self.finished_records = [[] for _ in range(n)]
        self.last_infos = [dict(info) for info in infos]
        self.node_open_round = {}
        self.turn_meta = []
        self.tracer.begin_reset()
        # [RSO] 可选钩子:适配器若维护跨局状态(如 Φ 追踪器),在这里清场
        hook = getattr(self.adapter, "on_reset_begin", None)
        if callable(hook):
            hook()

        texts: List[str] = []
        for i in range(n):
            info = infos[i]
            goal = self.adapter.root_goal(info, str(obs[i]))
            root = Node(depth=0, goal=goal, goal_text=self.adapter.goal_text(goal, is_root=True),
                        budget=self.budget.allocate(0), parent_uid=None, fork_step=None)
            self.adapter.on_node_open(root, info)       # 快照初始状态(如初始库存)
            self.node_open_round[root.uid] = 0
            self.stacks.append([root])
            texts.append(self.adapter.build_observation(root, is_first_turn=True))
            self._stamp_info(info, root, i, is_delegation_turn=False)
            self.tracer.emit("reset", slot=i, tag=self.tracer.tag,
                             task_id=info.get("extra.task_id"), difficulty=info.get("extra.difficulty"),
                             target_items=info.get("extra.target_items"),
                             initial_inventory=info.get("extra.inventory"))
            self._trace_node_open(i, root)
        self.last_prompt = list(texts)

        return {"text": texts, "image": None, "anchor": list(obs)}, infos

    # =================================================================== step
    def step(self, text_actions: List[str]):
        """一轮 = 官方 loop.py:70-77 那个 while 体在所有槽上同时执行一次。"""
        self.round += 1
        n = len(text_actions)
        actions, valids = self.projection_f(list(text_actions))

        # ---------- 阶段 1:决定每个槽发给底层环境的动作(对应 subagent.py:41-63 的委托前半段)
        env_actions: List[str] = [self.adapter.filler_action()] * n
        acting: List[Optional[Node]] = [None] * n
        pending_child: List[Optional[Node]] = [None] * n
        delegation_result: List[Optional[str]] = [None] * n
        for i in range(n):
            if self.episode_done[i] or not self.stacks[i]:
                continue                                  # 已结束的槽:发 filler 陪跑,结果丢弃
            top = self.stacks[i][-1]
            acting[i] = top
            try:
                req = self.adapter.parse_delegation(actions[i])
            except Exception as e:                        # 解析器自身崩溃也不能炸 batch(P11)
                # 当作普通动作发给环境(环境会回 unknown action),留档不杀节点——解析器崩溃是代码
                # bug,不该让模型的这一步变成整个 agent 的死亡
                top.scratch.setdefault("parser_errors", []).append(trace.format_exception(e, "parse_delegation"))
                print(f"[RAO orchestrator] slot {i} parse_delegation raised {type(e).__name__}: {e}; treated as plain action")
                req = None
            if req is None:
                env_actions[i] = actions[i]               # 普通动作,原样进环境
                continue

            # ---- 委托动作:先问深度(subagent.py:57 → trajectory.py:334-351 只查深度)
            if not self.budget.can_delegate(top.depth):
                reason, detail = self.budget.refusal(top.depth)
                delegation_result[i] = self.adapter.delegation_refused_text(reason, detail)
                # subagent.py:58-63:拒绝理由作为观测返回,不抛异常;这一步照样是父的一步
                continue

            # ---- 深度允许:建子节点(subagent.py:41 task.fork + :50-51 fork agent/env)
            child = Node(
                depth=top.depth + 1,
                goal=req.goal,
                goal_text=self.adapter.goal_text(req.goal, is_root=False),
                budget=self.budget.allocate(top.depth + 1),
                parent_uid=top.uid,
                fork_step=len(top.history),               # trajectory.py:131 fork_step=len(parent.steps)
            )
            child.scratch["context"] = req.context     # 父附带的说明,渲染子的 prompt 时用(env.py:344-345)
            child.scratch["raw_action"] = req.raw_action
            pending_child[i] = child
            delegation_result[i] = _PENDING_RESULT
            # env_actions[i] 保持 filler:委托那一轮底层环境不该被真正操作

        # ---------- 阶段 2:底层环境同步走一步
        next_obs, rewards, dones, infos = self.envs.step(env_actions)
        rewards = to_numpy(rewards).astype(np.float32)
        dones = to_numpy(dones)
        if rewards.ndim == 2:
            rewards = rewards.squeeze(1)
        if dones.ndim == 2:
            dones = dones.squeeze(1)

        # ---------- 阶段 3:记结果、压栈/弹栈、渲染下一轮观测
        out_text: List[str] = [""] * n
        for i in range(n):
            info = infos[i]
            info["is_action_valid"] = to_numpy(valids[i])
            node = acting[i]
            if node is None:                              # 本轮没有行动者(槽已结束或已失败)
                self._stamp_info(info, None, i, is_delegation_turn=False)
                continue
            self.last_infos[i] = dict(info)
            try:
                self._advance_slot(i, node, actions[i], str(next_obs[i]), info,
                                   bool(dones[i]), pending_child[i], delegation_result[i],
                                   rewards, dones, out_text, raw_output=str(text_actions[i]))
            except Exception as e:
                self._stamp_info(info, node, i, is_delegation_turn=False)
                self._fail_node(i, node, e, where="advance_slot", info=info, dones=dones, out_text=out_text)

        self.episode_done = np.logical_or(self.episode_done, dones)
        # 本轮每槽的元数据快照(从 infos 里抄,与写进 info 的完全一致),供收集器按位置回填
        self.turn_meta.append([{
            "node_uid": info.get("node_uid", ""),
            "node_depth": int(info.get("node_depth", -1)),
            "root_node_uid": info.get("root_node_uid", ""),
            "is_delegation_turn": bool(info.get("is_delegation_turn", False)),
            # [RSO] Φ 进展(适配器在 on_turn_result 里写进 info;委派/陪跑轮没写 → 缺省 0,
            # 语义正确:那些轮环境状态没变)。收集器按位置回填成每行的列。
            "delta_phi": float(info.get("delta_phi", 0.0)),
            "phi_frozen": bool(info.get("phi_frozen", False)),
        } for info in infos])
        return {"text": out_text, "image": None, "anchor": list(next_obs)}, rewards, dones, infos

    # ------------------------------------------------------------ 单槽推进
    def _advance_slot(self, i: int, node: Node, action: str, env_result: str,
                      info: Dict[str, Any], done: bool, child: Optional[Node],
                      delegation_result: Optional[str], rewards: np.ndarray,
                      dones: np.ndarray, out_text: List[str], raw_output: str = "") -> None:
        is_delegation_turn = delegation_result is not None
        self._stamp_info(info, node, i, is_delegation_turn=is_delegation_turn)

        # ---- 记这一轮。委托那轮也扣预算(trajectory.py:326-329:每个 step 都算,P5 正解)
        if is_delegation_turn:
            node.record_turn(action, delegation_result, is_delegation=True)
            rewards[i] = 0.0                              # filler 的环境奖励与父无关,归零
        else:
            node.record_turn(action, env_result, is_delegation=False)
            self.adapter.on_turn_result(node, info)   # 让适配器把本轮结构化结果记进该节点私有状态
        self.tracer.emit("turn", slot=i, round=self.round, node_uid=node.uid, depth=node.depth,
                         turn_index=node.turns_used,
                         prompt=self.last_prompt[i] if i < len(self.last_prompt) else "",
                         model_output=raw_output, parsed_action=action,
                         is_delegation=is_delegation_turn, env_result=node.last_result,
                         reward=float(rewards[i]), budget_left_after=node.budget_left,
                         delta_phi=float(info.get("delta_phi", 0.0)),
                         phi_after=info.get("phi_after"))
        if is_delegation_turn:
            if child is not None:                         # subagent.py:76 起子:这里是压栈
                node.add_child(child)
                self.adapter.on_node_open(child, info)    # env.py:594 快照开张时状态
                self.node_open_round[child.uid] = self.round
                self.stacks[i].append(child)
                self._trace_node_open(i, child)
                self.tracer.emit("delegate", slot=i, round=self.round, parent_uid=node.uid,
                                 child_uid=child.uid, goal_text=child.goal_text,
                                 context=child.scratch.get("context", ""))
            else:
                self.tracer.emit("refusal", slot=i, round=self.round, node_uid=node.uid,
                                 text=delegation_result)

        # ---- 整局结束:全树收摊(loop.py:98 finally 归档 + 整局 done)
        if done:
            self._close_slot(i, info, reason=CLOSE_EPISODE_END)
            out_text[i] = ""
            return

        # ---- 每个节点各自的终止判定(halt_episode 的语义按节点执行),可级联;root 终止 = 整局结束
        if self._settle_stack(i, info):
            self.episode_done[i] = True
            dones[i] = True
            out_text[i] = ""
            return

        # ---- 渲染栈顶的私有观测(上下文隔离在适配器 build_observation 里落实)
        if self._render_top(i, info):
            self.episode_done[i] = True
            dones[i] = True
            out_text[i] = "Episode terminated: context length limit exceeded."
            return
        out_text[i] = self.last_prompt[i] if i < len(self.last_prompt) else ""

    def _settle_stack(self, i: int, info: Dict[str, Any]) -> bool:
        """把 halt_episode(loop.py:140-146)的三个条件按【节点】执行,返回整局是否结束。

        非 root:目标达成(适配器 node_outcome,含它自己的循环检测)/ 预算耗尽 → 弹栈回父,
        可级联(父可能在发委托那一步用掉了最后一步预算 → 父也立即关闭 → 祖父收战报……
        官方同样会级联:父 await 子返回后回到 while,halt_episode 查到父预算为 0 就结束)。
        刚压栈、还没行动的子也在这里判一次("开张即判"):目标已满足就 0 步关闭。
        root:只在它是栈顶(没有子在跑)时判;node_outcome 说 done(目标达成 / 自己打转)
        或预算耗尽 → 整局结束。官方 root 的 halt_episode 同样只在它自己行动时检查。"""
        while len(self.stacks[i]) > 1:
            top = self.stacks[i][-1]
            outcome = self.adapter.node_outcome(top, info)
            if outcome.done:
                self._pop_child(i, success=outcome.success, reason=outcome.reason, info=info)
            elif top.budget_exhausted:
                success = self.adapter.evaluate_node(top, info)   # 预算耗尽也按各自口径打分
                self._pop_child(i, success=success, reason=CLOSE_BUDGET_EXHAUSTED, info=info)
            else:
                break
        if not self.stacks[i]:
            return True
        root = self.stacks[i][0]
        if len(self.stacks[i]) == 1:
            outcome = self.adapter.node_outcome(root, info)
            if outcome.done:
                self._close_slot(i, info, reason=CLOSE_EPISODE_END, root_reason=outcome.reason)
                return True
            if root.budget_exhausted:
                # 官方 halt_episode(loop.py:141)对 root 同样生效:root 也只有 25 步(synth_rollout.py:107)
                self._close_slot(i, info, reason=CLOSE_EPISODE_END, root_reason=CLOSE_BUDGET_EXHAUSTED)
                return True
        return False

    def _render_top(self, i: int, info: Dict[str, Any]) -> bool:
        """渲染栈顶的观测写进 self.last_prompt[i],返回整局是否因超长而结束。
        超长守卫按【节点】执行(官方:超限 = 该 agent 的请求报错 = 该 agent 结束):子节点超长 →
        只关它(CLOSE_OVERFLOW,按各自口径判分),父收到战报后重新渲染父;root 超长 → 整局结束。"""
        while self.stacks[i]:
            top = self.stacks[i][-1]
            text = self.adapter.build_observation(top, is_first_turn=(len(top.history) == 0))
            if self.max_obs_chars is None or len(text) <= self.max_obs_chars:
                if i < len(self.last_prompt):
                    self.last_prompt[i] = text            # 下一轮 turn 事件记的就是它
                return False
            info["prompt_overflow"] = True
            if len(self.stacks[i]) > 1:
                success = self.adapter.evaluate_node(top, info)
                self._pop_child(i, success=success, reason=CLOSE_OVERFLOW, info=info)
                continue
            self._close_slot(i, info, reason=CLOSE_EPISODE_END, root_reason=CLOSE_OVERFLOW)
            return True
        return True

    def _pop_child(self, i: int, success: float, reason: str, info: Dict[str, Any]) -> None:
        """子关闭 → 父收战报。对应 subagent.py:84-98 与 env.py:487-524。"""
        child = self.stacks[i].pop()
        child.close(success=success, reason=reason)
        record = child.to_record()
        self.finished_records[i].append(record)
        trace.record_node_span(child, slot=i,
                               opened_round=self.node_open_round.get(child.uid, 0),
                               closed_round=self.round)
        self._trace_node_close(i, child)

        parent = self.stacks[i][-1]
        parent.note_child_result(success)                 # env.py:524 取子的 reward/success 记账
        report = self.adapter.child_report_text(record)   # subagent.py:89-98 拼战报
        self.tracer.emit("child_report", slot=i, round=self.round, child_uid=child.uid,
                         parent_uid=parent.uid, report_text=report,
                         child_success=float(success), child_reason=reason)
        # 战报是父"发起委托那一步"的真正结果(官方:launch_subagent 的返回值就是那一步的 obs)。
        # 父在子运行期间挂起,所以它 history 的最后一轮必然是那次委托,直接覆盖占位文本。
        last = parent.history[-1] if parent.history else None
        if last is not None and last.is_delegation and last.result == _PENDING_RESULT:
            last.result = report
        parent.last_result = report

    def _close_slot(self, i: int, info: Dict[str, Any], reason: str,
                    root_reason: Optional[str] = None) -> None:
        """整局结束或强制收摊:栈内全部未关节点自顶向下按各自口径打分关闭。
        对应 loop.py:98 的 finally(每条轨迹都要归档)。
        reason 给还开着的子节点(它们是被整局结束连累的,统一记 episode_end / rollout_end);
        root_reason 给 root 自己的真实原因(预算耗尽 / 打转 / 超长 / 出错),缺省与 reason 相同。"""
        while self.stacks[i]:
            node = self.stacks[i][-1]
            if len(self.stacks[i]) > 1:
                success = self.adapter.evaluate_node(node, info)
                self._pop_child(i, success=success, reason=reason, info=info)
            else:
                success = self.adapter.evaluate_node(node, info)
                node.close(success=success, reason=root_reason or reason)
                self.finished_records[i].append(node.to_record())
                trace.record_node_span(node, slot=i,
                                       opened_round=self.node_open_round.get(node.uid, 0),
                                       closed_round=self.round)
                self._trace_node_close(i, node)
                self.stacks[i].pop()

    def _fail_node(self, i: int, node: Node, exc: BaseException, where: str,
                   info: Dict[str, Any], dones: np.ndarray, out_text: List[str]) -> None:
        """节点级异常处理。对应 loop.py:79-97:异常只结束【那一个 agent】的 run_episode,
        死因变成文本(error_message)回给父(subagent.py:98),父继续;root 出错才整局结束。
        怎么死都不能炸穿整个 rollout,但死因要留档(traceback 进 NodeRecord.error 与 tree_trace)。"""
        stack = self.stacks[i]
        # 归因给【栈顶】节点:异常发生时正在被处理的就是它(刚压栈的子在判定时炸 → 是子的错,
        # 不是发起委托的父的错);栈空时才落到 acting 节点上
        culprit = stack[-1] if stack else node
        culprit.error = trace.format_exception(exc, where=f"{where} (slot {i}, depth {culprit.depth})")
        print(f"[RAO orchestrator] slot {i} node d{culprit.depth} failed in {where}: {type(exc).__name__}: {exc}")
        if not stack:
            self.episode_done[i] = True
            dones[i] = True
            out_text[i] = ""
            return
        if len(stack) == 1:                               # root 出错 → 整局结束
            self._close_slot(i, info, reason=CLOSE_EPISODE_END, root_reason=CLOSE_ERROR)
            self.episode_done[i] = True
            dones[i] = True
            out_text[i] = ""
            return
        self._pop_child(i, success=0.0, reason=CLOSE_ERROR, info=info)   # 子出错:关子,父收战报
        if self._settle_stack(i, info) or self._render_top(i, info):
            self.episode_done[i] = True
            dones[i] = True
            out_text[i] = ""
            return
        out_text[i] = self.last_prompt[i] if i < len(self.last_prompt) else ""

    # ------------------------------------------------------------ 轨迹记录
    def _trace_node_open(self, i: int, node: Node) -> None:
        self.tracer.emit("node_open", slot=i, round=self.round, node_uid=node.uid,
                         parent_uid=node.parent_uid,
                         tree_id=self.stacks[i][0].uid if self.stacks[i] else node.uid,
                         depth=node.depth, goal_text=node.goal_text,
                         context=node.scratch.get("context", ""), fork_step=node.fork_step,
                         budget_total=node.budget_total,
                         preexisting=bool(node.scratch.get("preexisting", False)))

    def _trace_node_close(self, i: int, node: Node) -> None:
        self.tracer.emit("node_close", slot=i, round=self.round, node_uid=node.uid,
                         depth=node.depth, success=float(node.success), reason=node.close_reason,
                         turns=node.turns_used, error=node.error)

    # ------------------------------------------------------------ 元数据
    def _stamp_info(self, info: Dict[str, Any], node: Optional[Node], slot: int,
                    is_delegation_turn: bool) -> None:
        """往 info 里写本轮的节点元数据,采集器按行带进训练 batch。
        root_node_uid 供训练侧按树找 root(Eq.3 的 LOO 基线要 root 的奖励)。"""
        if node is None:
            info["node_uid"] = ""
            info["node_depth"] = -1
            info["node_goal"] = ""
            info["root_node_uid"] = ""
            info["is_delegation_turn"] = False
            return
        info["node_uid"] = node.uid
        info["node_depth"] = node.depth
        info["node_goal"] = node.goal_text
        info["is_delegation_turn"] = bool(is_delegation_turn)
        stack = self.stacks[slot] if slot < len(self.stacks) else []
        info["root_node_uid"] = stack[0].uid if stack else (node.uid if node.is_root else "")

    # ------------------------------------------------------------ 导出
    def collect_node_records(self) -> List[List[NodeRecord]]:
        """采集循环结束后调用。还开着的槽(全局轮数到了)按 CLOSE_ROLLOUT_END 收摊。
        对应 trajectory.py:178 的 to_dict() 整体导出。"""
        for i in range(len(self.stacks)):
            if self.stacks[i]:
                self._close_slot(i, self.last_infos[i], reason=CLOSE_ROLLOUT_END)
                self.episode_done[i] = True
        self.tracer.end_reset()                           # 本次采集的轨迹文件到此完整
        return self.finished_records

    # ------------------------------------------------------------ 统计
    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        """每槽取最后一个活跃行的 info['won'] 作为整局成败(与 flat 同口径),
        另按难度分桶,并补 RAO 专属统计:该树是否委托过、节点数、最大深度。"""
        for k in reversed(range(len(total_batch_list[batch_idx]))):
            if total_batch_list[batch_idx][k]["active_masks"]:
                info = total_infos[batch_idx][k]
                won = float(info["won"])
                success["success_rate"].append(won)
                diff = info.get("extra.difficulty")
                if diff is not None:
                    success[f"{diff}_success_rate"].append(won)
                    if diff in ("easy", "medium"):
                        success["easymedium_success_rate"].append(won)
                break
        recs = self.finished_records[batch_idx] if batch_idx < len(self.finished_records) else []
        success["rao/nodes_per_tree"].append(float(len(recs)))
        success["rao/max_depth"].append(float(max((r.depth for r in recs), default=0)))
        success["rao/delegating_tree"].append(float(any(r.depth > 0 for r in recs)))
        subs = [r for r in recs if r.depth > 0]
        success["rao/subagent_success_rate"].append(
            float(np.mean([r.success for r in subs])) if subs else 0.0)
        # 浪费型委托的比例(子开张时目标就已满足,0 步关闭)——RL 应把它学低
        success["rao/delegation_preexisting_rate"].append(
            float(np.mean([r.preexisting for r in subs])) if subs else 0.0)
        success["rao/subagent_stuck_rate"].append(
            float(np.mean([r.close_reason == "stuck_in_loop" for r in subs])) if subs else 0.0)
