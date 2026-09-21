# -*- coding: utf-8 -*-
"""真 TextCraft(ADaPT 官方环境)OOD 评测的 prompt:从 synth prompt 逐字派生。

出处:scripts_rso/eval_textcraft_ood.py 的 prompt 手术段原样迁入 prompts 包,
供 repo 正式 val_only 流程(flat 管理器 + 递归适配器)使用。派生原则:
与 synth 训练 prompt 保持"只差环境事实"的受控差异(2×2 推理网格),每处
替换带 assert,synth 措辞一变立刻在 import 期报错。

四处环境事实差异(见 SDAR/代码理解/textcraft官方_vs_synth_兼容性评估.md):
  ① 库存起始为空,基础原料须 get(synth 库存管够、无 get);
  ② 配方上榜进观测(synth 须 get_info 逐条查询);
  ③ craft 按榜上命令原样执行(synth 是精确量乘法语义);
  ④ tag 家族与干扰配方提示(synth 精确匹配、无干扰)。
递归增量(delegate 菜单 + 委托 TIPS)与 synth 递归逐字相同,仅示例物品换真实名。
"""
from agent_system.environments.prompts.textcraft_synth import (
    _FORMAT, _HEADER, _MENU, _TIPS)
from agent_system.environments.prompts.textcraft_synth_recursive import (
    _DELEGATION_TIPS, _MENU_RECURSIVE)

_HEADER_OOD = _HEADER.replace(
    "You have access to an inventory of existing ingredients, which are sufficient to craft the target items; though, you may need to craft intermediate ingredients first.",
    "Your inventory starts EMPTY. Base ingredients must be fetched with the 'get' action; "
    "craftable items cannot be fetched and must be crafted, possibly via intermediate ingredients. "
    "The Crafting commands listed with the task are sufficient to craft the target.")
assert _HEADER_OOD != _HEADER

_GET_INFO_BLOCK = """- get_info item1, item2, ...
    Get recipe information for items (whether craftable, recipes with ingredient
    amounts and result counts, crafting depth, how many you hold)."""
_GET_BLOCK = """- get <N> <item>
    Fetch N of a base (non-craftable) ingredient into your inventory.
    Example: get 2 dark oak log"""
_CRAFT_BLOCK_SYNTH = """- craft <N> <item> using <c1> <ingredient1>, <c2> <ingredient2>, ...
    Craft N of <item> (N must be a multiple of the recipe's result count).
    Ingredient amounts must EXACTLY equal per-craft amounts times executions.
    Example: craft 4 a0_i1 using 2 raw_a4"""
_CRAFT_BLOCK_OOD = """- craft <N> <item> using <c1> <ingredient1>, <c2> <ingredient2>, ...
    Execute ONE crafting command exactly as listed in the Crafting commands;
    each execution produces the listed amount, repeat the command to craft more.
    Example: craft 4 dark oak planks using 1 dark oak log"""
_MENU_OOD = _MENU.replace(_GET_INFO_BLOCK, _GET_BLOCK).replace(_CRAFT_BLOCK_SYNTH, _CRAFT_BLOCK_OOD)
assert _GET_BLOCK in _MENU_OOD and _CRAFT_BLOCK_OOD in _MENU_OOD

_TIPS_OOD = _TIPS.replace("</TIPS>", """- Some commands accept an item family (e.g. 'using 1 planks' matches any specific planks you hold)
- Only base ingredients can be fetched with get; if get says it cannot find an item, craft it instead
</TIPS>""")
assert _TIPS_OOD != _TIPS

_FORMAT_OOD = _FORMAT.replace(
    "Example: <thought>I need the recipe for a1_i2 first.</thought><action>get_info a1_i2</action>",
    "Example: <thought>I need planks before I can craft the sign.</thought>"
    "<action>craft 4 dark oak planks using 1 dark oak log</action>")
assert _FORMAT_OOD != _FORMAT

# 递归增量:synth 递归菜单的 delegate 条目原文摘出(逐字复用,仅示例换真实物品)
_DELEGATE_BLOCK = _MENU_RECURSIVE[len(_MENU):]
assert _DELEGATE_BLOCK.lstrip().startswith("- delegate:")
_DELEGATE_BLOCK_OOD = _DELEGATE_BLOCK.replace(
    "Example: delegate: craft 4 a0_i1, 2 a0_i3 | these are ingredients for a1_i2",
    "Example: delegate: craft 4 dark oak planks | ingredients for the sign")
assert _DELEGATE_BLOCK_OOD != _DELEGATE_BLOCK
_MENU_OOD_REC = _MENU_OOD + _DELEGATE_BLOCK_OOD
_TIPS_OOD_REC = _TIPS_OOD.replace("</TIPS>", "\n" + _DELEGATION_TIPS + "\n</TIPS>")
_HEADER_OOD_CHILD = _HEADER_OOD.replace(
    "Note: If you already have one of the target items in your inventory, you should craft the "
    "requested number of the target on top of what you already have.",
    "Note: Your task is complete as soon as your inventory contains at least the listed amounts "
    "of every target item. Items already in the inventory count toward the target, so only craft "
    "what is missing.")
assert _HEADER_OOD_CHILD != _HEADER_OOD

_MID_HIS = """

{current_observation}

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}

You are now at step {current_step}.

"""
_MID_NO_HIS = "\n\n{current_observation}\n\n"


def _tpl(header, menu, tips, his):
    return header + (_MID_HIS if his else _MID_NO_HIS) + menu + "\n\n" + tips + "\n\n" + _FORMAT_OOD + "\n"


TEXTCRAFT_OOD_TEMPLATE_NO_HIS = _tpl(_HEADER_OOD, _MENU_OOD, _TIPS_OOD, his=False)
TEXTCRAFT_OOD_TEMPLATE = _tpl(_HEADER_OOD, _MENU_OOD, _TIPS_OOD, his=True)
TEXTCRAFT_OOD_REC_TEMPLATE_NO_HIS = _tpl(_HEADER_OOD, _MENU_OOD_REC, _TIPS_OOD_REC, his=False)
TEXTCRAFT_OOD_REC_TEMPLATE = _tpl(_HEADER_OOD, _MENU_OOD_REC, _TIPS_OOD_REC, his=True)
TEXTCRAFT_OOD_REC_CHILD_TEMPLATE_NO_HIS = _tpl(_HEADER_OOD_CHILD, _MENU_OOD_REC, _TIPS_OOD_REC, his=False)
TEXTCRAFT_OOD_REC_CHILD_TEMPLATE = _tpl(_HEADER_OOD_CHILD, _MENU_OOD_REC, _TIPS_OOD_REC, his=True)

__all__ = [
    "TEXTCRAFT_OOD_TEMPLATE_NO_HIS", "TEXTCRAFT_OOD_TEMPLATE",
    "TEXTCRAFT_OOD_REC_TEMPLATE_NO_HIS", "TEXTCRAFT_OOD_REC_TEMPLATE",
    "TEXTCRAFT_OOD_REC_CHILD_TEMPLATE_NO_HIS", "TEXTCRAFT_OOD_REC_CHILD_TEMPLATE",
]
