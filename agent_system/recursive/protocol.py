# -*- coding: utf-8 -*-
"""递归内核与具体环境之间的契约。

本文件不含任何 TextCraft 相关的东西,也不 import torch / verl。内核只认识这里定义的
几个数据类和一个接口;想让某个 benchmark 支持递归委托,只需要实现 RecursiveEnvAdapter
这一个类(TextCraft-Synth 的实现在 env_package/textcraft_synth/recursive_adapter.py)。

这个文件不是从某一段官方代码直译来的。官方 platoon 把这些环节分散在多处,靠 asyncio
协程和 contextvars 隐式串起来;我们把它们收敛成一组显式接口。每个接口对应官方的
哪个行为,见下面各方法的注释。路径相对 Experiment/code_references/platoon/,
行号为**本地副本**的行号(本地副本加过读码笔记,与上游原版有位移):

  plugins/textcraft/platoon/textcraft/env.py:322-351
      launch_subagent(targets, num_steps, context) —— 委托请求长什么样、
      targets 字典怎么渲染成 goal 字符串(:339-345)
  plugins/textcraft/platoon/textcraft/env.py:499
      depth-aware 版的 launch_subagent(self, targets, context="") —— 没有 num_steps 参数
  plugins/textcraft/platoon/textcraft/env.py:594 及 :602-660
      _initial_inventory 快照与 evaluate() 的净增量判定
  platoon/agents/actions/subagent.py:57-63
      委托被拒时把理由拼成文本返回给模型(不抛异常)
  platoon/agents/actions/subagent.py:89-100
      子跑完后拼给父看的战报文本
  platoon/agents/actions/subagent.py:50
      agent.fork() 造失忆分身 —— 子 agent 上下文隔离的实现
  platoon/agents/codeact/prompt_builder.py:63-79
      对话式 prompt 的拼法(系统提示 + 目标 + 逐轮追加)

为什么不直接用官方的 asyncio + contextvars(2026-08-24 与用户确认,详见
recursive_OPSD_调试tips.md #15):官方每个 agent 是独立协程,agent.act() 是一次 HTTP
调用;本框架的模型调用是整批一次(rollout_loop.py:509 的 generate_sequences),
不存在"某个 agent 单独调一次模型"的操作,协程递归没有对应物。所以官方的
while: act -> step 循环被"翻转"成"外层按轮批量生成、内层每槽记着轮到树上哪个节点",
显式栈是同一棵树在批处理架构下的自然表达。复刻的是全部算法逻辑,替换的只是执行引擎。

两处有意的偏差,都来自这个架构差异:
  1. 官方靠 agent.fork() 造一个全新对话来隔离子 agent 的上下文;我们靠
     build_observation 渲染 prompt 时只用该节点自己的 history 来实现等价效果
     (vLLM 每轮生成本来就是无状态的:prompt 进、文本出)。
  2. 官方父协程在委托期间挂起,不与环境交互;我们的 lockstep 要求每个槽每一轮都必须
     给环境一个动作,所以委托那一轮要给环境发一个不改状态的 filler 动作占位,
     其返回结果被丢弃。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:   # 只给类型检查器看,运行时不 import,避免与 node.py 循环引用
    from agent_system.recursive.node import Node, NodeRecord


# ---------------------------------------------------------------------------
# 节点关闭原因。这些字符串会写进 NodeRecord.close_reason,便于事后统计
# "子节点都是怎么结束的"(比如冒烟测试要看委托是不是全都预算耗尽)。
# ---------------------------------------------------------------------------
CLOSE_GOAL_MET = "goal_met"                  # 自己的子目标达成(适配器 node_outcome 判定)
CLOSE_BUDGET_EXHAUSTED = "budget_exhausted"  # 自己的步数预算用完(内核判定)
CLOSE_EPISODE_END = "episode_end"            # 底层环境整局结束,全树被迫收摊
CLOSE_ROLLOUT_END = "rollout_end"            # 采集循环的全局轮数上限到了,强制收摊
CLOSE_ERROR = "error"                        # 内核/适配器异常(见 trace.py);只结束出错的那个节点
CLOSE_STUCK = "stuck_in_loop"                # 节点自己的动作历史出现重复模式(官方 agent.py:90-118 的每-agent 检测)
CLOSE_OVERFLOW = "prompt_overflow"           # 该节点的 prompt 超长(官方:超限=该 agent 的请求报错=该 agent 结束)

# 2026-08-25 生命周期下沉(冒烟 21461096 验证集 7/16 棵树被子节点的槽级循环检测连累后修正):
# 上面每一种关闭原因都只作用于【当前节点】——子节点关闭后父继续;只有 root 关闭才结束整局。
# 槽级(整棵树共享)的只剩两样真正属于"世界"的东西:共享库存,和 lockstep 必需的全局轮数上限。


@dataclass
class DelegationRequest:
    """一次委托请求 = 适配器从动作文本里解析出来的结构化意图。

    对应官方 launch_subagent 的参数(env.py:499 depth-aware 版的签名):
        targets: Dict[str, int]  ->  goal(适配器自定义载荷,内核不看内容)
        context: str             ->  context
    **注意没有 num_steps / budget 字段。** 官方 recursive 变体的签名带 num_steps
    (env.py:322),但正式实验用的 depth-aware 变体把它删掉了(env.py:499),
    prompt 里教模型估预算的那几段也一并删掉(见 tips #13)。预算由配置固定,
    模型不再决定"派多少步"。旧实现的 `delegate: craft N item budget K` 文法带
    budget 参数,语义上对应的是官方没在正式实验里用的 recursive 变体——本实现改正。

    goal 载荷在 TextCraft-Synth 里会是 {物品名: 数量} 的字典,内核只负责原样传给
    子节点,需要渲染成文字时回调适配器的 goal_text。
    raw_action 保存原始动作文本,只用于日志和排查。
    """
    goal: Any
    context: str = ""
    raw_action: str = ""


@dataclass
class NodeOutcome:
    """适配器对"这个节点此刻该不该关闭"的判断结果。

    done=False 时 success / reason 被忽略;内核只在 done=True 时读它们。
    reason 应当是本文件顶部 CLOSE_* 常量之一(适配器一般只会给 CLOSE_GOAL_MET;
    预算耗尽和整局结束由内核自己判定,不经这里)。
    """
    done: bool
    success: float = 0.0
    reason: str = ""


@runtime_checkable
class RecursiveEnvAdapter(Protocol):
    """环境适配器接口。换一个 benchmark 只需要实现这一个类。

    内核在每个 batch 槽、每一轮的调用时序:
        1. 模型产出动作文本
        2. 内核调 parse_delegation:这是委托吗?
           - 是 → 内核先问预算(深度够不够);够就压栈新节点,并把 filler_action()
                  发给底层环境(那一步的环境结果会被丢弃,因为父这一步的"真正结果"
                  要等子跑完后的战报);不够就把 delegation_refused_text 当作
                  父这一步的结果
           - 否 → 内核把动作原样发给底层环境
        3. 底层环境执行,返回观测 / 奖励 / done / info
        4. 内核调 node_outcome 判断栈顶节点是否该关闭
        5. 内核调 build_observation 渲染下一轮给模型看的 prompt

    Protocol + runtime_checkable 意味着鸭子类型:任何带齐这些方法的对象都算适配器,
    不需要继承。这和官方 TrajectoryEventHandler / BudgetTracker 的写法一致
    (platoon/episode/trajectory.py:71-72, :210-211)。
    """

    # ------------------------------------------------------------------ 动作侧
    def parse_delegation(self, action_text: str) -> Optional[DelegationRequest]:
        """从动作文本里认出委托意图。不是委托就返回 None,内核会把它当普通动作发给环境。

        官方没有这一步:launch_subagent 是 Python 函数调用,参数天然是结构化的。
        我们的动作是一行文本,所以必须先解析。这是 lockstep 移植必需的新增环节。
        解析时要对"模型的自然写法"做鲁棒处理(逗号 / 空格 / 引号 / 大小写),
        tips #9 记录过因此踩坑的教训。
        """
        ...

    def filler_action(self) -> str:
        """委托那一轮发给底层环境的占位动作。

        为什么需要它:lockstep 要求每个槽每一轮都给环境一个动作,不能跳过。
        这个动作必须【不改变环境状态】(TextCraft-Synth 里用 'inventory' 查询),
        它的返回结果会被内核丢弃,不进任何节点的 history。
        官方没有对应物——asyncio 下父协程在委托期间直接挂起,不与环境交互。
        """
        ...

    # ------------------------------------------------------------------ 目标
    def root_goal(self, env_info: Dict[str, Any], obs_text: str) -> Any:
        """从环境 reset 返回的 info / 初始观测里取出 root 节点的目标载荷。

        对应官方 root 与子节点目标来源的不对称:子节点的 target_items 是 fork 时从
        goal 字符串反解析出来的(env.py:674-680),而 root 的直接来自数据集塞在
        task.misc 里(env.py:527-528 读 initial_inventory,evaluate :565 读
        task.misc["target_items"])。我们的 synth 环境把这些放在 reset 返回的 info 里,
        由适配器决定取哪个键。返回值的类型与 DelegationRequest.goal 一致。
        """
        ...

    # ------------------------------------------------------------------ 文本渲染
    def goal_text(self, goal: Any, is_root: bool = True) -> str:
        """把 goal 载荷渲染成模型可读的一句话,用于 prompt 和节点记录。

        对应 env.py:339-345:官方把 targets 字典渲染成
        "Craft the following items: 2x stick, 1x oak_planks" 再塞进子任务的 goal。
        is_root 允许适配器给子目标用不同措辞(TextCraft-Synth 的子目标判定口径与 root 不同,
        见 recursive_adapter.py 头注释"子目标判定"一节)。
        """
        ...

    def child_report_text(self, child_record: "NodeRecord") -> str:
        """子节点关闭后,拼给父看的战报文本。

        对应 platoon/agents/actions/subagent.py:89-100:官方返回
        (traj.finish_message or traj.error_message or "") + 预算用量说明。
        这段文本会成为父节点那一轮(发起委托的那一轮)的 result,
        父模型下一步看到的就是它——父永远只见文字,见不到子的轨迹对象。
        """
        ...

    def delegation_refused_text(self, reason: str, detail: str) -> str:
        """委托被拒(深度超限)时给父看的文本。

        对应 platoon/agents/actions/subagent.py:57-63:官方捕获 BudgetExceededError
        后,把 message 和 guidance 拼成一段话【直接 return 给模型】,不抛异常。
        也就是说"你不能再委托了"是靠环境反馈教给模型的,不是靠程序报错。
        reason / detail 由 budget.py 的 PerAgentBudget.refusal 给出,
        detail 里已经含官方原文的 guidance。
        """
        ...

    # ------------------------------------------------------------------ 状态与判定
    def on_node_open(self, node: "Node", env_info: Dict[str, Any]) -> None:
        """节点开张时快照环境状态,写进 node.open_snapshot。

        对应 env.py:594 的 self._initial_inventory = initial_inventory.copy()。
        官方子 env 在 fork 时记下当时的库存,之后 evaluate() 拿它算净增量。
        我们在压栈时调用这个钩子,让适配器把它关心的状态(库存)存到节点上。
        root 节点也会调:它的快照就是任务的初始库存。
        """
        ...

    def on_turn_result(self, node: "Node", env_info: Dict[str, Any]) -> None:
        """节点执行了一个【普通动作】、环境返回之后调用(委托那轮不调,因为环境跑的是 filler)。

        2026-08-24 步骤 3 回溯加。用途:让适配器把这一轮环境给的结构化信息记到该节点的
        私有状态里——TextCraft-Synth 里就是把 info['extra.last_get_info'] 写进这个节点
        自己的配方笔记本、把库存快照记成 "as of step N"。
        官方没有对应钩子,因为官方子 agent 的对话历史天然只含自己的轮次(subagent.py:50 fork
        出的分身从空对话开始);我们要显式维护才能得到同样的"只知道自己查过什么"。
        """
        ...

    def node_outcome(self, node: "Node", env_info: Dict[str, Any]) -> NodeOutcome:
        """判断该节点此刻是否应关闭,以及成败。每一轮环境执行完后调用。

        对应 env.py:602-660 的 evaluate(),核心是 :620-636 那段:对每个目标物品,
        current_count - initial_count >= required_count 才算达成。
        【必须用环境状态(env_info 里的库存)判定,不要去正则解析观测文本】——
        旧实现的 _CRAFTED_RE 在 synth 环境上必然失配,是审查报告 P3 的根源。

        对所有节点调用,包括 root:子节点 done → 弹栈回父;root done → 整局结束
        (对应官方 root 调 finish 或被判 stuck 时它自己的 run_episode 结束)。
        也在节点【刚压栈、还没行动】时调一次("开张即判"):目标已满足就 0 步关闭。
        预算耗尽由内核自己判断(它掌握步数),适配器不必管。
        适配器还应在这里做【本节点自己历史】上的循环检测(官方 agent.py:90-109),
        触发时返回 done=True, reason=CLOSE_STUCK, success=evaluate_node(...)。
        """
        ...

    def evaluate_node(self, node: "Node", env_info: Dict[str, Any]) -> float:
        """强制关闭时给节点打分(整局结束 / 全局轮数耗尽 / 内核异常)。

        与 node_outcome 分开,是因为那时不问"该不该关",只问"到此为止算不算成功"。
        判定口径应与 node_outcome 一致(同样是净增量)。
        """
        ...

    # ------------------------------------------------------------------ 观测渲染
    def build_observation(self, node: "Node", is_first_turn: bool) -> str:
        """渲染该节点【私有视角】的 prompt。

        关键约束:只能用 node 自己的 goal / goal_text / history / last_result,
        不能掺入父节点或兄弟节点的历史。这就是 RAO"子 agent 上下文隔离"在
        lockstep 框架里的等价物——官方靠 agent.fork()(subagent.py:50)造一个
        没有父消息历史的全新对话,我们靠这条渲染规则达到同样效果。

        prompt 的措辞应对齐官方 TextCraftDepthAwarePromptBuilder
        (plugins/textcraft/platoon/textcraft/agent.py:193-260),
        用 <thought> 普通文本标签,不用 <think>(审查报告 P2)。
        is_first_turn=True 时节点还没有任何历史(刚开张),模板可以据此省略历史段。
        """
        ...
