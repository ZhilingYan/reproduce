# -*- coding: utf-8 -*-
"""预算语义:每个 agent 独立的步数预算,加上委托深度上限。

移植来源(路径相对 Experiment/code_references/platoon/,行号为**本地副本**的行号,
本地副本里加过读码笔记注释,与 GitHub 上游原版有位移):

  platoon/episode/trajectory.py:297-355   DepthAwareStepBudgetTracker 的全部逻辑
      :306      max_depth: int | None = None   深度上限可以为 None(不限)
      :310-312  _allocated_budget     每条轨迹的额度 = 它自己的 task.max_steps
      :314-322  _trajectory_depth     沿 parent 指针爬到 root 数深度(root = 0)
      :326-329  used_budget_for       只数【自己的】步数,后代的不算
      :331-332  remaining_budget_for  额度减去自己已用的
      :334-351  reserve_budget        只检查深度,【不】从父那里预留步数
      :353-355  release_budget        空实现(因为压根没预留过)
  platoon/episode/trajectory.py:190-203   BudgetExceededError 的 reason 与 guidance 字段
  plugins/textcraft/platoon/textcraft/synth_rollout.py:83   _TEXTCRAFT_SYNTH_MAX_DEPTH = 6
  plugins/textcraft/platoon/textcraft/synth_rollout.py:89   per_agent_max_steps: int = 25
  plugins/textcraft/platoon/textcraft/synth_rollout.py:107  root 的预算也被覆写成同一个值
  plugins/textcraft/platoon/textcraft/synth_rollout.py:124  正式实验装的就是这个 tracker

--------------------------------------------------------------------------
关于这两个数字在官方那边是怎么配的(2026-08-24 查证,决定了我们的做法)
--------------------------------------------------------------------------
官方对这两个参数的待遇不一样,而且有一处实现瑕疵:

1) per_agent_max_steps(25)—— 名义上可配,实际写死。
   配置链路是:yaml 的 workflow_config.rollout_config.max_steps: 25
     → step_wise.py:427-428 把它写进 task.max_steps
     → 但紧接着 synth_rollout.py:107 又执行 task.max_steps = per_agent_max_steps,
       把上一步的值【覆盖掉】。
   而所有调用方(train_areal_synth.py:125、tinker 的训练脚本、推理脚本)调
   run_synth_depth_aware_rollout 时都只传 (task, config) 两个位置参数,从不传
   per_agent_max_steps。所以实际生效的永远是函数签名的默认值 25。
   yaml 里那个 max_steps: 25 对 depth-aware 路径其实是装饰性的(对 linear 路径
   才真正生效,那边写的是 200),因为两处恰好都是 25,平时看不出问题。

2) max_depth(6)—— 完全硬编码,没有任何配置通道。
   _TEXTCRAFT_SYNTH_MAX_DEPTH = 6 定义在 synth_rollout.py:83,同样只作为签名默认值,
   没有调用方传它。要改只能改代码。

我们的做法:**两个参数都从 yaml 配置读**,默认值取官方【实际生效】的数字。
理由是我们要复刻的是官方的逻辑,不是它的实现瑕疵。这样默认行为严格等于官方 default,
而调 yaml 就能改,不用动代码。配置项:
    env.rao.per_agent_max_steps   默认 25
    env.rao.max_depth             默认 6(设为 null 表示不限深度,官方也支持,见 :306)
本模块的常量只作为构造函数的默认参数存在,真正的值由编排器从 config 读出来后传进来。

--------------------------------------------------------------------------
【不移植】platoon/episode/trajectory.py:227-295 的 StepBudgetTracker
--------------------------------------------------------------------------
那是"整棵树共享一个步数池"的语义:派子前要 reserve(max_steps + 1),子跑完 release,
且 used_budget_for 会递归统计全部后代的步数。它不是官方正式实验用的那套——
论文附录 A.4 写明 "For recursive agents, we allow 25 steps for both the root and
sub-agents",配置里 depth_aware: true,装的是 DepthAwareStepBudgetTracker。
按"只复刻 default 路径"的原则,共享池那套不搬。将来若需要,约 60 行可补。

--------------------------------------------------------------------------
与官方的一个实现差异(不影响语义)
--------------------------------------------------------------------------
官方的 tracker 是无状态的:每次要用时现场去 TrajectoryCollection 里数 len(traj.steps)、
沿 parent 指针爬深度。它必须这样,因为 asyncio 下各协程只能通过共享账本通信。
我们把剩余步数直接存在 Node 上(node.budget_left),深度也直接存(node.depth),
所以这个类退化成一组"策略函数",本身不持有任何运行时状态,只持有两个配置参数。
数值口径与官方完全一致:节点每走一步扣一步,子代的消耗不影响父。
"""
from __future__ import annotations

from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# 默认值 = 官方【实际生效】的数字,出处见文件头。
# 这两个常量只用作构造函数的默认参数;正式运行时由编排器从 yaml 配置读值传进来。
# ---------------------------------------------------------------------------
DEFAULT_PER_AGENT_STEPS = 25    # synth_rollout.py:89 的签名默认值
DEFAULT_MAX_DEPTH = 6           # synth_rollout.py:83 的 _TEXTCRAFT_SYNTH_MAX_DEPTH
                                # 论文 §3.1:训练用 6,评测时放宽到 12

# 委托被拒时返回给模型的话术,原文照抄
# platoon/episode/trajectory.py:345-348 的 guidance 字段。
# 关键设计:官方拒绝委托时【不是抛异常给程序】,而是把拒绝理由和建议作为一段文本
# 返回给模型当观测(见 agents/actions/subagent.py:58-63 的 except 分支:捕获
# BudgetExceededError 之后,把 message 和 guidance 拼成 msg 直接 return)。
# 也就是说"你不能再委托了"这件事是靠环境反馈教给模型的,不是靠程序报错。
DEPTH_REFUSAL_REASON = "depth"
DEPTH_REFUSAL_GUIDANCE = (
    "The maximum depth for hierarchical delegation has been reached. "
    "You should perform the task yourself instead of delegating."
)


class PerAgentBudget:
    """每 agent 独立预算,加深度上限。对应官方的 DepthAwareStepBudgetTracker。

    参数:
        per_agent_steps: 每个节点(含 root)各自的步数额度。
                         默认 25 = 官方实际生效值;由 env.rao.per_agent_max_steps 配置。
        max_depth:       委托树的最大深度,root 记为 0。
                         默认 6 = 官方值;由 env.rao.max_depth 配置。
                         传 None 表示不限深度——官方也支持这个取值,
                         见 trajectory.py:306 的 max_depth: int | None = None。
    """

    def __init__(self,
                 per_agent_steps: int = DEFAULT_PER_AGENT_STEPS,
                 max_depth: Optional[int] = DEFAULT_MAX_DEPTH):
        if int(per_agent_steps) <= 0:
            raise ValueError(f"per_agent_steps 必须为正整数,收到 {per_agent_steps}")
        if max_depth is not None and int(max_depth) < 0:
            raise ValueError(f"max_depth 不能为负,收到 {max_depth}")
        self.per_agent_steps = int(per_agent_steps)
        self.max_depth = None if max_depth is None else int(max_depth)

    # ------------------------------------------------------------------ 构造
    @classmethod
    def from_config(cls, rao_cfg) -> "PerAgentBudget":
        """从配置字典构造。rao_cfg 对应 yaml 里的 env.rao 子树。

        读两个键:per_agent_max_steps 和 max_depth。键缺失时用官方默认值。
        max_depth 显式写 null 表示不限深度。
        写成 classmethod 是为了让"配置怎么映射到参数"这件事只有一处定义,
        编排器和测试都走同一个入口。
        """
        cfg = dict(rao_cfg or {})
        steps = cfg.get("per_agent_max_steps", DEFAULT_PER_AGENT_STEPS)
        # 注意 None 是合法值(不限深度),所以不能用 cfg.get(k, default) or default 这种写法
        depth = cfg["max_depth"] if "max_depth" in cfg else DEFAULT_MAX_DEPTH
        return cls(per_agent_steps=int(steps),
                   max_depth=None if depth is None else int(depth))

    # ------------------------------------------------------------------ 额度
    def allocate(self, depth: int) -> int:
        """新节点开张时分配的步数额度。

        对应 platoon/episode/trajectory.py:310-312 的 _allocated_budget:
        额度取自该轨迹自己的 task.max_steps。而 depth-aware 模式下所有轨迹的
        task.max_steps 都被设成了同一个值(子的来自 synth_rollout.py:111 传给 env 的
        subagent_max_steps,root 的来自 :107 那行覆写),所以这里不区分深度,
        一律返回同一个数。
        参数 depth 保留是为了将来可能做"按深度递减预算"的实验,当前不使用。
        """
        return self.per_agent_steps

    # ------------------------------------------------------------------ 判定
    def can_delegate(self, current_depth: int) -> bool:
        """站在 current_depth 的节点上,还能不能再派一个子代。

        对应 platoon/episode/trajectory.py:334-351 的 reserve_budget。
        那个方法名字叫 reserve(预留),但在 depth-aware 实现里它【只检查深度】,
        一步预算都不从父那里扣——官方注释原文是 "subagent steps are not reserved
        from the parent"。判据就是 current_depth + 1 > max_depth 则拒绝。

        举例(max_depth=6):root 在 depth 0,它派的子在 depth 1……
        depth 5 的节点还能派(5+1=6,不大于 6),depth 6 的节点不能派(6+1=7 > 6)。
        所以整棵树最深到 depth 6,连 root 一共 7 层。

        注意这里【不】检查发起者自己还剩多少步。官方同样不检查:depth-aware 的
        reserve_budget 里只有深度那一个判断。发起者的步数是在它执行完这一步之后
        照常扣减的(见 Node.record_turn),扣到 0 就会在下一轮被编排器按预算耗尽关闭。
        """
        if self.max_depth is None:
            return True
        return current_depth + 1 <= self.max_depth

    def refusal(self, current_depth: int) -> Tuple[str, str]:
        """委托被拒时返回 (reason, detail)。

        文本对应 platoon/episode/trajectory.py:341-348 的 BudgetExceededError
        构造参数:message 说明超了什么限制,guidance 给出可执行的建议。
        编排器会把这两段交给适配器渲染成给模型看的观测文本。
        """
        detail = (f"Launching a subagent from depth {current_depth} would exceed "
                  f"the maximum allowed depth of {self.max_depth}.")
        return DEPTH_REFUSAL_REASON, detail + " " + DEPTH_REFUSAL_GUIDANCE

    # ------------------------------------------------------------------ 空操作
    def release(self, amount: int = 0) -> None:
        """空实现,保留只为对齐官方接口。

        对应 platoon/episode/trajectory.py:353-355 的 release_budget,
        官方注释原文:"No-op: subagent steps do not consume parent budget."
        因为 depth-aware 模式压根没预留过任何步数,自然也无从归还。
        (共享池那套 StepBudgetTracker 才需要真正的 reserve/release 配对,
         见 trajectory.py:272-295,我们不移植。)
        """
        return None

    def __repr__(self) -> str:
        return (f"<PerAgentBudget steps={self.per_agent_steps} "
                f"max_depth={self.max_depth}>")
