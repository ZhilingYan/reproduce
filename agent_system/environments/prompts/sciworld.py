# -*- coding: utf-8 -*-
"""ScienceWorld 的 prompt 模板——结构与措辞尽量逐句贴合 TextCraft 训练用的
core_code/RSO/agent_system/environments/prompts/textcraft_synth.py(2026-09-19 用户定稿:
ScienceWorld 作为 TextCraft 的 OOD 基准,prompt 骨架必须与训练时所见同构,
否则 OOD 分数里混入"格式迁移"变量)。

与 textcraft_synth.py 的逐段对应:
  _HEADER   同构:角色句 + "环境里有完成任务所需的一切"(对应库存足够声明)+
            Note 句(textcraft 是目标数量说明;ScienceWorld 换成 focus 警告——
            同为"判定口径"类提示,占同一结构位;focus 焦错对象即负分终止,
            22190938 chemistry-mix 实测 reward=-125)
  {current_observation}  同位:harness 填"任务描述 + 当前观测"
  {admissible_actions}   ScienceWorld 新增槽,紧跟观测:当前状态的合法动作全集
            (get_valid_action_object_combinations();这是 22191244 vs 22190938
            对照实验证明的成绩决定因素)。TextCraft 无此槽——它的动作文法封闭、
            配方信息靠 get_info 挣;ScienceWorld 动作模板 × 对象指称开放,
            合法清单的角色相当于"免费的 get_info",归入状态信息而非菜单。
  历史句    逐字同 textcraft("Prior to this step, ... You are now at step N.")
  _MENU     同构:"You may take exactly ONE of the following actions per step:" +
            逐条动作说明(textcraft 3 条,ScienceWorld 是官方 25 个模板)
  _TIPS     同构:<TIPS> 块,CRAFTING STRATEGY → SCIENCE STRATEGY
  _FORMAT   逐字同 textcraft(~1-3 句 <thought> + 恰一个 <action>),仅示例换域。
推理标签 <thought>:与 textcraft 一致(<think> 是 Qwen3 特殊 token,
textcraft_synth.py 头注释 2026-08-17 诊断 + 探针 22191832 双重实锤)。
"""

_MENU = """You may take exactly ONE of the following actions per step:
- look around / look at OBJ / look in OBJ
    Observe the room, an object, or a container's contents.
- inventory
    View your current inventory.
- task
    Re-read the current task description.
- go OBJ / teleport OBJ
    Move to a door/location, or teleport directly to a location
    (teleport exists only under the "easy" simplification preset used here).
- open OBJ / close OBJ
    Open or close a door or container.
- pick up OBJ / put down OBJ / move OBJ to OBJ
    Take an object, drop it, or move it into/onto something.
- activate OBJ / deactivate OBJ
    Turn a device (stove, sink, ...) on or off.
- pour OBJ in OBJ / dunk OBJ in OBJ / mix OBJ
    Handle liquids and mixtures.
- use OBJ on OBJ
    Use a tool or device on something (e.g. a thermometer on a substance).
- read OBJ
    Read a note or recipe.
- eat OBJ / flush OBJ
    Consume or flush an object.
- focus on OBJ
    Signal that OBJ is the object of interest required by the task.
- wait / wait1
    Let time pass (10 steps / 1 step).
- reset task
    Reset the task to the beginning."""

_TIPS = """<TIPS>
SCIENCE STRATEGY:
- The admissible actions list shows every action that can be executed right now - choose from it
- Focus is a commitment: "focus on OBJ" only on the object(s) the task asks for; focusing on anything else immediately ends the episode with failure
- Some outcomes need time (heating, growing): repeat measurements or wait instead of assuming failure
- Always verify what is present before claiming something is impossible
- Check the room and containers to confirm where task-relevant objects are
- If the target object is not in this room, move to other rooms to find it
</TIPS>"""

_HEADER = """You are an agent in ScienceWorld, a text-based science environment. Your goal is to complete elementary science tasks by interacting with objects and devices.
The environment contains everything needed to complete the task; though, you may need to find, prepare, or combine objects first.

Note: "focus on OBJ" is NOT looking at something - it is how you SUBMIT AN ANSWER. "focus on X" declares "X is the object the task asks about", and a wrong declaration immediately ends the episode with failure. Therefore: first FIND the exact object the task asks about, then focus on it. Never focus on a room, door, container, device, air, or the agent (unless the task explicitly names it)."""

_FORMAT = """You will get multiple steps to complete the task.
For your current step, first briefly reason (~1-3 sentences) about your next step in the <thought> </thought> tags and then output exactly one action in <action> </action> tags.
Example: <thought>The task asks about a living thing, so I should look in the greenhouse.</thought><action>teleport to greenhouse</action>"""

SCIWORLD_TEMPLATE_NO_HIS = _HEADER + """

{current_observation}

Your admissible actions of the current situation are: [{admissible_actions}].

""" + _MENU + """

""" + _TIPS + """

""" + _FORMAT + "\n"

SCIWORLD_TEMPLATE = _HEADER + """

{current_observation}

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}

You are now at step {current_step}.

Your admissible actions of the current situation are: [{admissible_actions}].

""" + _MENU + """

""" + _TIPS + """

""" + _FORMAT + "\n"


