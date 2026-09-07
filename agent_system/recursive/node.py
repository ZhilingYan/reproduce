# -*- coding: utf-8 -*-
"""执行树的节点数据结构。

移植来源(路径相对 Experiment/code_references/platoon/,行号为**本地副本**的行号,
本地副本里加过读码笔记注释,与 GitHub 上游原版有位移):

  platoon/episode/trajectory.py:29-37     TrajectoryStep  -> 本文件的 Turn
  platoon/episode/trajectory.py:40-43     ParentInfo      -> Node.parent_uid / fork_step
  platoon/episode/trajectory.py:46-63     Trajectory      -> Node
  platoon/episode/trajectory.py:127-137   create_trajectory 里填 fork_step 的那一下
  platoon/episode/trajectory.py:326-332   used_budget_for / remaining_budget_for
                                          (每节点独立预算的口径)
  plugins/textcraft/platoon/textcraft/env.py:499-526
                                          recursive executor 记账子代成败
                                          -> Node.children_uids / children_success
  plugins/textcraft/platoon/textcraft/env.py:594 及 :602-660
                                          _initial_inventory 与 evaluate() 的净增量判定
                                          -> Node.open_snapshot

有意为之的结构差异(已在 docs/RAO_PORT_DESIGN.md 记录):
  官方是"一条 Trajectory 对应一个 agent 的一生",整棵树靠 TrajectoryCollection 里的
  扁平字典加 parent_info 指针连起来。它必须这么做,因为它用 asyncio 协程递归,
  父子跑在不同的执行上下文里,只能靠指针关联。
  我们是 lockstep 批处理:每个 batch 槽在同一时刻只有栈顶一个节点在行动,
  所以直接用"节点对象加一个显式的栈"来表达同一棵树。
  父指针我们照样保留(parent_uid),因为训练侧算深度、找 root 都要用它,
  语义与官方 ParentInfo.id 完全一致。官方 ParentInfo 还记了 fork_step
  (父走到第几步时派出的我),我们同样记录,排查问题时有用。
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Turn:
    """节点历史里的一轮。

    对应 platoon/episode/trajectory.py:29-37 的 TrajectoryStep。官方那个基类刻意留白,
    只有一个 misc 字典,具体字段(reward 等)由各插件的子类自己塞;我们把 lockstep 下
    真正必需的三样东西写成显式字段,其余仍然放 misc。

    is_delegation 标记这一轮是不是委托动作。它有两个用处:一是排查时能一眼看出
    树是在哪一轮分叉的,二是统计"委托动作出现率"这个冒烟测试判据。
    """
    action: str
    result: str
    is_delegation: bool = False
    misc: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeRecord:
    """节点关闭后导出的记录,是这个包交给训练侧的唯一产物。

    字段的选择依据是 RAO 论文的三个公式,每个字段都有明确的消费方:
      success            -> Eq.1 的第一项 s(X),节点自己任务的成败
      children_success   -> Eq.1 的第二项 mean_c s(c) 的原料(直接子代的成功率)
      depth              -> Eq.4 深度反频率加权的分组依据
      parent_uid / uid   -> 训练侧还原树形、定位 root 用;Eq.3 的 LOO 基线需要 root 的奖励
    剩下的 close_reason / turns / budget_used / fork_step / error 不参与训练,
    纯粹为了事后统计和排查(比如"子节点都是怎么结束的""委托是不是全都预算耗尽")。
    """
    uid: str
    parent_uid: Optional[str]
    depth: int
    goal_text: str
    success: float
    children_uids: List[str]
    children_success: List[float]
    close_reason: str
    turns: int
    budget_used: int
    budget_total: int = 0                # 2026-08-24 加:战报要写 "used/total steps"(subagent.py:92-94)
    fork_step: Optional[int] = None
    error: Optional[str] = None
    preexisting: bool = False            # 子开张时目标就已满足(浪费型委托的诊断量;适配器写 scratch["preexisting"])


class Node:
    """执行树上的一个节点,等价于官方的一条 Trajectory(一个 agent 的一生)。

    与 platoon/episode/trajectory.py:46-63 的 Trajectory 字段对照:

        官方                      我们              说明
        id                    ->  uid
        parent_info.id        ->  parent_uid
        parent_info.fork_step ->  fork_step
        task.goal             ->  goal / goal_text  goal 是适配器自定义的载荷
        steps                 ->  history           我们只存文本,不存 token
        reward                ->  success           官方 reward 是逐步累加,
                                                    我们是这个节点任务的成败(0 或 1)
        (官方没有对应字段)      ->  budget_left       官方靠 tracker 现场数 steps 算,
                                                    我们直接在节点上存剩余步数
    """

    __slots__ = ("uid", "parent_uid", "depth", "goal", "goal_text", "fork_step",
                 "budget_left", "budget_total", "turns_used", "history", "last_result",
                 "children_uids", "children_success", "open_snapshot", "scratch",
                 "success", "close_reason", "closed", "error")

    def __init__(self, depth: int, goal: Any, goal_text: str, budget: int,
                 parent_uid: Optional[str] = None, fork_step: Optional[int] = None):
        self.uid: str = str(uuid.uuid4())
        self.parent_uid = parent_uid
        self.depth = int(depth)
        self.goal = goal                      # 适配器自定义的载荷,内核不解读它的内容
        self.goal_text = goal_text            # 渲染好的文字,用于 prompt 和节点记录
        self.fork_step = fork_step            # 父走到第几步时派出的我

        # 预算:每个节点独立持有自己的剩余步数。
        # 这对应 platoon/episode/trajectory.py:326-332 的 DepthAwareStepBudgetTracker:
        # used_budget_for 只数【自己的】steps,remaining = task.max_steps 减去自己的步数,
        # 子代消耗多少完全不影响父节点。
        # 官方 root 的预算同样是 25(见 synth_rollout.py:107 那行显式覆写
        # task.max_steps = per_agent_max_steps),所以这里 root 和子节点一视同仁。
        self.budget_total = int(budget)
        self.budget_left = int(budget)
        self.turns_used = 0

        # 上下文隔离的载体:每个节点只有自己的历史,渲染 prompt 时也只用它。
        # 官方是靠 agent.fork() 造一个没有父消息历史的"失忆分身"来实现的
        # (见 platoon/agents/actions/subagent.py:50 那行 forked_agent = await agent.fork)。
        # 我们因为 vLLM 每轮生成本来就无状态(prompt 进、文本出),
        # 只要渲染 prompt 时只看自己的 history,就等价地实现了同一件事。
        self.history: List[Turn] = []
        self.last_result: str = ""

        # 子代账本。Eq.1 的委托奖励项需要"直接子代的成功率",
        # 论文 §2.2.1 特意说明用成功率而不是成功个数,是为了防止策略靠多生子代刷奖励。
        # 官方的对应实现在 plugins/textcraft/platoon/textcraft/env.py:499-526,
        # 那里 diff 一下 trajectory 集合找出本步新生的子代,再从子的最后一步
        # 取 reward_misc["reward/success"]。我们在编排器里弹栈时直接记,更直接。
        self.children_uids: List[str] = []
        self.children_success: List[float] = []

        # 节点开张时的环境状态快照,由适配器写入(TextCraft-Synth 里是当时的库存)。
        # 用途是算净增量:官方 TextCraftEnv 在 __init__ 里存 self._initial_inventory
        # (env.py:594),evaluate() 里用 current_count - initial_count >= required
        # 判定成功(env.py:602-660)。
        # 这是取代旧实现 _CRAFTED_RE 正则计数的关键——用环境状态判定,不去猜观测文本
        # (旧实现的正则在 synth 环境上必然失配,见审查报告 P3)。
        self.open_snapshot: Dict[str, Any] = {}

        # 适配器的私有杂物袋(2026-08-24 步骤 3 回溯加)。内核不读它。TextCraft-Synth 适配器
        # 用它放:委托附带的 context 文本、该节点私有的配方笔记本、库存快照——
        # 这些都是"按节点隔离"的状态,官方对应物是子 agent 全新对话里逐步积累的消息历史。
        self.scratch: Dict[str, Any] = {}

        self.success: float = 0.0
        self.close_reason: str = ""
        self.closed: bool = False
        self.error: Optional[str] = None      # 内核异常的 traceback 文本(见 trace.py,补 P11)

    # ---------------------------------------------------------------- 属性
    @property
    def is_root(self) -> bool:
        return self.depth == 0

    @property
    def budget_exhausted(self) -> bool:
        """预算是否已经用尽。"""
        return self.budget_left <= 0

    # ---------------------------------------------------------------- 记账
    def record_turn(self, action: str, result: str, is_delegation: bool = False) -> None:
        """记录一轮,并扣掉一步预算。

        扣减的口径对应官方的 used_budget_for = len(traj.steps)
        (platoon/episode/trajectory.py:326-329):**每一步都算**,包括发起委托的那一步。
        官方那边父 agent 调 launch_subagent 时,这次调用本身就是父 trajectory 的一个 step,
        所以委托对发起者不是免费的。

        旧实现在这里错了:它只在非委托动作时才递减预算,于是子节点可以无限发委托而
        自身预算不减(审查报告 P5)。本实现按官方语义改正。
        """
        self.history.append(Turn(action=action, result=result, is_delegation=is_delegation))
        self.last_result = result
        self.turns_used += 1
        self.budget_left -= 1

    def add_child(self, child: "Node") -> None:
        """登记一个直接子代。子代的成败要等它关闭时才由 note_child_result 补上。"""
        self.children_uids.append(child.uid)

    def note_child_result(self, success: float) -> None:
        """子代关闭时回填它的成败,供 Eq.1 的委托奖励项使用。"""
        self.children_success.append(float(success))

    def close(self, success: float, reason: str) -> None:
        """关闭节点。幂等:重复调用不会覆盖第一次的结果。

        幂等是必要的,因为整局结束时编排器会把栈里所有节点一次性收摊,
        而其中某些节点可能已经因为目标达成或预算耗尽被关过了。
        """
        if self.closed:
            return
        self.success = float(success)
        self.close_reason = reason
        self.closed = True

    # ---------------------------------------------------------------- 导出
    def to_record(self) -> NodeRecord:
        return NodeRecord(
            uid=self.uid,
            parent_uid=self.parent_uid,
            depth=self.depth,
            goal_text=self.goal_text,
            success=float(self.success),
            children_uids=list(self.children_uids),
            children_success=list(self.children_success),
            close_reason=self.close_reason,
            turns=self.turns_used,
            budget_used=self.turns_used,
            budget_total=self.budget_total,
            fork_step=self.fork_step,
            error=self.error,
            preexisting=bool(self.scratch.get("preexisting", False)),
        )

    def __repr__(self) -> str:   # 排查用
        state = ("closed:" + self.close_reason) if self.closed else "open"
        return (f"<Node d={self.depth} goal={self.goal_text!r} "
                f"turns={self.turns_used} left={self.budget_left} {state}>")
