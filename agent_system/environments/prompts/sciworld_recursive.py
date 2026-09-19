# -*- coding: utf-8 -*-
"""ScienceWorld 递归模式 prompt(训练版):从 prompts/sciworld.py 派生。

底本 = code_references/SDAR/agent_system/environments/prompts/sciworld_recursive.py
(评测版,2026-09-19)。训练版三补丁(移植对照表"有意偏离清单",出处
Ideation/ideas/RSO_sciworld_data_mapping.md §二点七 与 10 案例双臂分析):
  补丁① delegate 文法去 budget 参数——训练框架的 DelegationRequest 无预算通道
        (agent_system/recursive/protocol.py:77-81,官方 depth-aware 正式口径:
        预算由配置固定,模型不再决定"派多少步");
  补丁② 子 agent 菜单加 `answer: <report>` 显式汇报动作 + SUBAGENT_TASK_NOTE 改为
        无条件 focus 禁令(melt v21 死于子 agent focus on oven;grow-plant v93
        死于预算耗尽未汇报);
  补丁③ 预算数值不写进 prompt(配置定:root/child 解耦见 env_package/sciworld/budget.py)。
SDAR 侧评测版 prompt 暂不动;训完评测 ckpt 前必须把两边同步(对照表 TODO 4)。
"""
from agent_system.environments.prompts.sciworld import (
    _FORMAT, _HEADER, _MENU, _TIPS,
    SCIWORLD_TEMPLATE, SCIWORLD_TEMPLATE_NO_HIS)

_HEADER_RECURSIVE = _HEADER.replace(
    "The environment contains everything needed to complete the task; though, you may need to find, prepare, or combine objects first.",
    "The environment contains everything needed to complete the task; though, you may need to find, prepare, or combine objects first. "
    "You may also DELEGATE a self-contained sub-task to a sub-agent (another instance of yourself).")
assert _HEADER_RECURSIVE != _HEADER

# 补丁①:无 budget 参数;子 agent 有自己的固定步数预算并会回报
_MENU_RECURSIVE = _MENU + """
- delegate: <sub-task>
    Spawn a sub-agent whose only goal is the stated sub-task,
    e.g. 'delegate: bring a metal pot filled with water to the stove'.
    The sub-agent has its own fixed step budget and acts in the same world:
    any changes it makes (objects moved, devices activated) persist for you.
    It will report back what it found or accomplished."""

# 补丁②:子 agent 专属菜单 = 递归菜单 + 显式汇报动作
_MENU_CHILD = _MENU_RECURSIVE + """
- answer: <what you found or accomplished>
    End your sub-task and report back to your parent agent. Use this as soon as
    your sub-task is done or you have the requested information."""

_DELEGATION_STRATEGY = """Delegation strategy:
- Delegating preparatory sub-tasks (fetching tools, moving objects, setting up devices) is highly recommended for multi-stage procedures.
- Delegate ONE self-contained sub-task at a time; phrase it so it is fully understandable on its own.
- Sub-agents may themselves delegate further if the sub-task is still complex.
- Never delegate the "focus on" step: perform every task-required focus yourself (a wrong focus by anyone ends the episode with failure)."""

_FORMAT_RECURSIVE_BASE = _FORMAT.replace(
    "about your next step in the <thought> </thought> tags",
    "about what to do or delegate next in the <thought> </thought> tags")
assert _FORMAT_RECURSIVE_BASE != _FORMAT
# 示例不对称(照 textcraft_recursive.py 的做法):NO_HIS 用 delegate 示例(补丁①去 budget),
# 带历史版用普通动作示例
_FORMAT_RECURSIVE_NO_HIS = _FORMAT_RECURSIVE_BASE.replace(
    "Example: <thought>The task asks about a living thing, so I should look in the greenhouse.</thought><action>teleport to greenhouse</action>",
    "Example: <thought>I need a pot of water on the stove before boiling; a sub-agent can set that up.</thought><action>delegate: bring a metal pot filled with water to the stove</action>")
assert _FORMAT_RECURSIVE_NO_HIS != _FORMAT_RECURSIVE_BASE


def _assemble(header, menu, fmt, with_history):
    mid = """

{current_observation}

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}

You are now at step {current_step}.

Your admissible actions of the current situation are: [{admissible_actions}].

""" if with_history else """

{current_observation}

Your admissible actions of the current situation are: [{admissible_actions}].

"""
    return header + mid + menu + "\n\n" + _TIPS + "\n\n" + _DELEGATION_STRATEGY + "\n\n" + fmt + "\n"


SCIWORLD_RECURSIVE_TEMPLATE_NO_HIS = _assemble(
    _HEADER_RECURSIVE, _MENU_RECURSIVE, _FORMAT_RECURSIVE_NO_HIS, with_history=False)
SCIWORLD_RECURSIVE_TEMPLATE = _assemble(
    _HEADER_RECURSIVE, _MENU_RECURSIVE, _FORMAT_RECURSIVE_BASE, with_history=True)
SCIWORLD_RECURSIVE_CHILD_TEMPLATE_NO_HIS = _assemble(
    _HEADER_RECURSIVE, _MENU_CHILD, _FORMAT_RECURSIVE_BASE, with_history=False)
SCIWORLD_RECURSIVE_CHILD_TEMPLATE = _assemble(
    _HEADER_RECURSIVE, _MENU_CHILD, _FORMAT_RECURSIVE_BASE, with_history=True)

# 补丁②:无条件 focus 禁令 + answer 汇报规范(评测版是条件禁令,10 案例证明会被无视)
SUBAGENT_TASK_NOTE = (
    "(You are a sub-agent. A parent agent delegated this sub-task to you. "
    "You act in the same world as the parent. Focus ONLY on your delegated sub-task; "
    "the episode-level task is handled by your parent. "
    "NEVER use 'focus on': submitting answers via focus is exclusively your parent's job, "
    "and a wrong focus ends the whole episode for everyone. "
    "When your sub-task is done, or you have the requested information, "
    "report back with 'answer: <your report>'.)"
)
