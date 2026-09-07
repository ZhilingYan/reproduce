# -*- coding: utf-8 -*-
"""TextCraft-Synth 递归(RAO)模式的 prompt 模板。

对齐目标:RAO 官方正式实验用的 TextCraftDepthAwarePromptBuilder
    Experiment/code_references/platoon/plugins/textcraft/platoon/textcraft/agent.py:193-260
以及工具说明
    Experiment/code_references/platoon/plugins/textcraft/platoon/textcraft/env.py:353-401
    (synth 分支 :354-352 的 craft/get_info/view_inventory/finish 四段 + :334-351 的 launch_subagent 段)

组装原则:在 flat 的模板(prompts/textcraft_synth.py)基础上【只加委托相关内容】,其余逐字复用,
保证 flat 与 RAO 的 prompt 只差"能不能委托"这一件事——与官方 linear/recursive 两份配置只差
递归开关的做法一致。2026-08-24 用 diff 核实:flat 的 33 行一行不删一字不改,递归版只多 13 行
(菜单第四条 6 行 + DELEGATION STRATEGY 7 行),<think> 出现 0 次。

与官方 depth-aware prompt 的逐段对照:
  官方 agent.py:204-214  HEADER(角色/库存足够/目标数量说明)      → 复用 flat 的 _HEADER,逐句相同
  官方 agent.py:216-226  CRAFTING STRATEGY                         → 复用 flat 的 _TIPS,逐句相同
  官方 agent.py:228-235  DELEGATION STRATEGY(depth-aware 版,已删预算估算)
                                                                  → 本文件 _DELEGATION_TIPS,措辞照搬,
                                                                    删掉两条与代码载体绑定的内容(见下)
  官方 agent.py:236-239  "launch_subagent is async, you MUST use await" + gather 示例
                                                                  → 【删除】文本文法没有 await/gather
  官方 agent.py:243-252  <thought> + <python> 输出格式说明           → 复用 flat 的 _FORMAT(<action> 标签)
  官方 env.py:334-351    launch_subagent(targets, context) 工具说明  → 本文件 _MENU_RECURSIVE 第四条
                                                                    (文本文法版,无 num_steps 参数,
                                                                     对应 env.py:499 depth-aware 签名)

有意删除的两条(载体差异,已记入 docs/RAO_PORT_DESIGN.md §5):
  * agent.py:234 "Items can be delegated in parallel if they don't depend on each other"
    —— 文本文法一轮只能发一个动作,不存在并行委托,保留会误导模型;
  * agent.py:236-239 await / asyncio.gather 那四行。

标签用 <thought> / <action>(普通文本),不用 <think>(Qwen3 特殊 token,对默认模型有毒,
审查报告 P2;flat 模板头注释有实测记录)。这套标签已被三条 flat baseline 的训练验证。

【待验证】"delegate: craft ..." 这一条文法对 4B 模型是否可写出、可解析,尚无直接证据
(旧实现用过同款文法但跑在 <think> 有毒模板上,数据不可信)。步骤 6 冒烟的第一道门槛就是
零样本统计委托动作出现率与解析成功率;若模型写不出,只需调本文件的文法示例与
recursive_adapter.parse_delegation 的正则,其余代码不受影响。
"""
from agent_system.environments.prompts.textcraft_synth import _FORMAT, _HEADER, _MENU, _TIPS

# 第四条动作:委托。文法设计要点:
#   * 多目标:官方 launch_subagent 的 targets 是 dict,可一次派多个物品(env.py:339 的 ", ".join),
#     文本版用逗号分隔;
#   * 可选 context:对应官方 context 参数(env.py:344-345 拼进子的 goal),用 " | " 分隔;
#   * 【没有预算参数】:官方 depth-aware 签名 launch_subagent(targets, context) 无 num_steps
#     (env.py:499),预算由配置固定;旧实现的 "budget K" 是 recursive 变体的语义,已废弃。
_MENU_RECURSIVE = _MENU + """
- delegate: craft <N> <item>[, <M> <item2>, ...] [| <context for the sub-agent>]
    Launch a sub-agent to craft the listed items for you. The sub-agent starts with a
    fresh context, shares your inventory, has its own fixed step budget, and may itself
    delegate further. You are paused until it finishes, then you receive its report.
    Anything it crafts is immediately available in your inventory.
    Example: delegate: craft 4 a0_i1, 2 a0_i3 | these are ingredients for a1_i2"""

# 官方 agent.py:228-235 DELEGATION STRATEGY(depth-aware 版)逐句照搬,除上面说明的两条删除。
_DELEGATION_TIPS = """DELEGATION STRATEGY:
- It is **highly recommended** to delegate crafting of intermediate ingredients
- Break complex tasks into INDEPENDENT subtasks that can be solved separately
- For tasks that are sufficiently complex, it is recommended to recursively delegate; i.e., subagents can further delegate to other subagents.
- Delegate one group of related items at a time, not everything at once
- Delegated tasks share your inventory - results are immediately available"""

# 把 DELEGATION STRATEGY 并进 <TIPS> 块,与官方结构一致(官方 :216-240 两段都在同一个 <TIPS> 里)
_TIPS_RECURSIVE = _TIPS.replace("</TIPS>", "\n" + _DELEGATION_TIPS + "\n</TIPS>")

# ---------------------------------------------------------------------------------------------
# 子节点专用 HEADER(2026-08-25,用户决定:子目标判定改为"绝对可用量",见 recursive_adapter.py 头注释)
# flat/官方的 HEADER 里那句 "craft the requested number on top of what you already have" 描述的是
# 净增量口径,对子节点已不成立(子目标 = 让库存里【有】足够数量,已有的算数),照抄会自相矛盾。
# 除这一句外与 _HEADER 逐句相同。root 仍用 _HEADER(root 口径不变,与 flat / 官方一致)。
# ---------------------------------------------------------------------------------------------
_HEADER_CHILD = _HEADER.replace(
    "Note: If you already have one of the target items in your inventory, you should craft the "
    "requested number of the target on top of what you already have.",
    "Note: Your task is complete as soon as your inventory contains at least the listed amounts "
    "of every target item. Items already in the inventory count toward the target, so only craft "
    "what is missing.",
)
assert _HEADER_CHILD != _HEADER, "flat 的 HEADER 措辞变了,子节点 Note 替换失配,请同步更新"

TEXTCRAFT_SYNTH_RECURSIVE_TEMPLATE_NO_HIS = _HEADER + """

{current_observation}

""" + _MENU_RECURSIVE + """

""" + _TIPS_RECURSIVE + """

""" + _FORMAT + "\n"

TEXTCRAFT_SYNTH_RECURSIVE_TEMPLATE = _HEADER + """

{current_observation}

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}

You are now at step {current_step}.

""" + _MENU_RECURSIVE + """

""" + _TIPS_RECURSIVE + """

""" + _FORMAT + "\n"

# 子节点模板:与上面两份逐字相同,只是 HEADER 换成 _HEADER_CHILD(Note 句描述"绝对可用量"口径)
TEXTCRAFT_SYNTH_RECURSIVE_CHILD_TEMPLATE_NO_HIS = TEXTCRAFT_SYNTH_RECURSIVE_TEMPLATE_NO_HIS.replace(_HEADER, _HEADER_CHILD, 1)
TEXTCRAFT_SYNTH_RECURSIVE_CHILD_TEMPLATE = TEXTCRAFT_SYNTH_RECURSIVE_TEMPLATE.replace(_HEADER, _HEADER_CHILD, 1)
assert TEXTCRAFT_SYNTH_RECURSIVE_CHILD_TEMPLATE != TEXTCRAFT_SYNTH_RECURSIVE_TEMPLATE
