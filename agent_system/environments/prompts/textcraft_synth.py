# -*- coding: utf-8 -*-
"""TextCraft-Synth 的 prompt 模板——措辞尽量逐句对齐 RAO 官方 platoon 的
system prompt(platoon/plugins/textcraft/platoon/textcraft/agent.py)。

与官方的对应关系:
  * 角色/库存足够声明/目标数量说明/TIPS:逐句照搬官方原文;
  * 推理标签用官方同款 <thought></thought>(普通文本)。
    ⚠ 不要改回 SDAR 惯例的 <think></think>——那是 Qwen3 词表里的特殊 token
    (151667/151668),对去思考版模型(Qwen3-4B-Instruct-2507)是分布外输入,
    实测会让模型第一个 token 就输出结束符,整局空转(2026-08-17 诊断实锤);
  * 唯一结构性差异:官方动作是 <python> 代码块(jupyter 执行),我们的 lockstep
    循环用 <action> 文本命令(普通文本标签,无毒)。动作菜单对应官方的
    craft/get_info/view_inventory 三个函数。
"""

_MENU = """You may take exactly ONE of the following actions per step:
- inventory
    View your current inventory.
- get_info item1, item2, ...
    Get recipe information for items (whether craftable, recipes with ingredient
    amounts and result counts, crafting depth, how many you hold).
- craft <N> <item> using <c1> <ingredient1>, <c2> <ingredient2>, ...
    Craft N of <item> (N must be a multiple of the recipe's result count).
    Ingredient amounts must EXACTLY equal per-craft amounts times executions.
    Example: craft 4 a0_i1 using 2 raw_a4"""

# 官方 TIPS 原文照搬
_TIPS = """<TIPS>
CRAFTING STRATEGY:
- Recipes produce fixed quantities per execution - you cannot craft arbitrary amounts
  Example: If a recipe produces 2 items, you can only craft in multiples of 2 (2, 4, 6...)
- Recipe ingredients scale with the number of times you execute it
  Example: Recipe "2 ore -> 2 items" means 2 ore for 1 execution, 4 ore for 2 executions
- Always verify what you have before claiming something is impossible
- Check your inventory and recipe information to confirm ingredient availability
- Calculate carefully: if a recipe uses 2 ingredients to make 2 items, you need exactly 2 ingredients for 2 items
</TIPS>"""

# 官方开头段原文照搬(含目标数量的说明)
_HEADER = """You are an agent in a crafting game. Your goal is to craft items by combining ingredients.
You have access to an inventory of existing ingredients, which are sufficient to craft the target items; though, you may need to craft intermediate ingredients first.

Note: If you already have one of the target items in your inventory, you should craft the requested number of the target on top of what you already have."""

# 输出格式段:对齐官方 include_reasoning 版的措辞,<python> 换成 <action>
_FORMAT = """You will get multiple steps to complete the task.
For your current step, first briefly reason (~1-3 sentences) about your next step in the <thought> </thought> tags and then output exactly one action in <action> </action> tags.
Example: <thought>I need the recipe for a1_i2 first.</thought><action>get_info a1_i2</action>"""

TEXTCRAFT_SYNTH_TEMPLATE_NO_HIS = _HEADER + """

{current_observation}

""" + _MENU + """

""" + _TIPS + """

""" + _FORMAT + "\n"

TEXTCRAFT_SYNTH_TEMPLATE = _HEADER + """

{current_observation}

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}

You are now at step {current_step}.

""" + _MENU + """

""" + _TIPS + """

""" + _FORMAT + "\n"
