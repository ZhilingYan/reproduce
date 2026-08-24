# -*- coding: utf-8 -*-
"""TextCraft-Synth 单环境核心(无 Ray),按 RAO 官方 platoon 实现精确复刻。

复刻来源: code_references/platoon/plugins/textcraft/platoon/textcraft/env.py
设计要点(对应 recursive_OPSD_调试tips.md 的"决定2"):
  * 可选菜单式接口:动作 = inventory / get_info / craft 三种查询-执行命令,
    配方不塞 prompt,模型按需用 get_info 查;
  * 报错信息精确可执行(整除/数量/多余原料/库存不足,措辞与官方一致);
  * 成功判定 = 每个目标物品的【净合成数】(当前-初始)达到要求,与官方相同;
    与官方的一个已记录差异:官方要模型显式调 finish() 才结算,我们的 lockstep
    循环在每步后自动检查(SDAR 各环境的惯例,模型不需要 finish 动作);
  * 循环检测:同一动作连续重复 >=4 次即判死循环提前终止(官方保护机制的移植);
  * OPSD 特权通道:每局的 gold_trajectory 转成动作文本,放进 info['extra.gt_plan']。

动作文法(纯文本,由 projection 从 <action> 标签抠出后传进来):
    inventory
    get_info item1, item2, ...
    craft <N> <target> using <c1> <ing1>, <c2> <ing2>, ...
      语义 = 官方 craft({ing1:c1, ing2:c2}, (target, N)),N 是想要的总产出数,
      必须是配方单次产量的整数倍,原料量必须精确等于 单次用量×执行次数。
"""
import json
import os
import re
from typing import Dict, List, Tuple

from .synth_recipe_generator import SynthRecipeDatabase

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data")


# ---------------------------------------------------------------------------
# 配方库:与官方数据集配套的确定性重建(seed=42, tier=25,见官方 SynthRecipeLoader)
# ---------------------------------------------------------------------------
class SynthRecipeDB:
    def __init__(self, seed: int = 42, items_per_domain_tier: int = 25):
        self._db = SynthRecipeDatabase()
        self._db.generate_all_recipes(seed=seed, items_per_domain_tier=items_per_domain_tier)

    def get_recipes_for_item(self, item: str):
        return self._db.get_recipes_for_item(item)

    def can_craft(self, item: str) -> bool:
        return self._db.can_craft(item)

    def is_base_item(self, item: str) -> bool:
        return self._db.is_base_item(item)

    def get_crafting_depth(self, item: str) -> int:
        try:
            return self._db.get_crafting_depth(item)
        except Exception:
            return -1


_SHARED_DB = None


def get_shared_recipe_db() -> SynthRecipeDB:
    """配方库构建要解析全部合成树,每个进程建一次全局共享即可。"""
    global _SHARED_DB
    if _SHARED_DB is None:
        _SHARED_DB = SynthRecipeDB()
    return _SHARED_DB


# ---------------------------------------------------------------------------
# 任务加载
# ---------------------------------------------------------------------------
def load_tasks(split: str, difficulties: List[str]) -> List[dict]:
    """从 vendored jsonl 读任务,按难度过滤。split ∈ {train, val}。"""
    path = os.path.join(DATA_DIR, f"textcraft_synth_{split}.jsonl")
    tasks = []
    with open(path) as f:
        for line in f:
            t = json.loads(line)
            if t["misc"].get("difficulty") in difficulties:
                tasks.append(t)
    if not tasks:
        raise ValueError(f"no tasks for split={split} difficulties={difficulties}")
    return tasks


def gold_to_actions(gold_trajectory: List[dict]) -> List[str]:
    """把数据集里的 gold 轨迹转成本环境的动作文本(OPSD 特权 gt_plan 用)。

    gold 每步的字段语义(经真实轨迹核对):ingredients = 总用量,
    result_count = 总产出量 → 正好对应 craft <result_count> <item> using <总用量清单>。
    """
    actions = []
    for s in gold_trajectory:
        item = s["target"][0]
        total_out = s["result_count"]
        ings = ", ".join(f"{c} {name}" for name, c in s["ingredients"].items())
        actions.append(f"craft {total_out} {item} using {ings}")
    return actions


# ---------------------------------------------------------------------------
# 单环境
# ---------------------------------------------------------------------------
_CRAFT_RE = re.compile(r"^craft\s+(\d+)\s+(\S+)\s+using\s+(.+)$")
_ING_RE = re.compile(r"^\s*(\d+)\s+(\S+)\s*$")
_GETINFO_RE = re.compile(r"^get_info\s+(.+)$")

LOOP_REPEAT_LIMIT = 4   # 同一动作连续重复达到该次数 → 判死循环终止(官方保护的移植)


class SynthTextCraftEnv:
    def __init__(self, recipe_db: SynthRecipeDB = None):
        self.db = recipe_db or get_shared_recipe_db()
        self._task = None

    # ---------------- reset ----------------
    def reset(self, task: dict, max_steps_override: int = None) -> Tuple[str, dict]:
        """max_steps_override: 覆盖任务自带的步数预算。任务 jsonl 里写的是 75,
        但官方推理/训练实际用 config 覆盖(如 800)——深任务的 gold 本身就超 75 步,
        不覆盖的话按任务预算根本走不完。SDAR 侧应传 config.env.max_steps 进来。"""
        self._task = task
        self.goal_text = task["goal"]
        self.targets: Dict[str, int] = dict(task["misc"]["target_items"])
        self.inventory: Dict[str, int] = dict(task["misc"]["initial_inventory"])
        self.initial_inventory: Dict[str, int] = dict(task["misc"]["initial_inventory"])
        self.max_steps = int(max_steps_override or task.get("max_steps", 75))
        self.steps = 0
        self._last_action = None
        self._repeat_count = 0
        self.won = False
        self.gt_plan = "\n".join(gold_to_actions(task["misc"]["gold_trajectory"]))
        # 状态块机制(2026-08-19 加,medium 全零的修复)。滑窗 history_length=2 会把
        # 10 步前查到的配方冲掉,medium(要记 5-10 个配方)因此全程 0 成功。修复思路
        # 仿 ALFWorld 的"状态自描述",但信息制度严格做 RAO 完整对话历史的【子集】——
        # 只重放模型自己查询挣来的信息,不赠送任何新信息:
        #   * 配方笔记本:本局 get_info 查到过的配方,永久重放,再查同物刷新;
        #   * 库存快照:模型最近一次执行 inventory 动作的结果原样重放,带"as of
        #     step N"时效标注,craft 后不自动刷新——要新鲜数据得再查,inventory
        #     动作因此保持有效(与 RAO 里 view_inventory 快照会过时的行为一致)。
        self.known_recipes: Dict[str, str] = {}
        self.last_inv_snapshot: str = None   # 最近一次 inventory 查询的渲染文本
        self.last_inv_step: int = None       # 查询发生在第几步

        obs = f"{self.goal_text}\nBudget: you have {self.max_steps} steps in total."
        return obs, self._info()

    def _state_block(self) -> str:
        """每步观察末尾附加的状态块:已习得配方 + 最近库存快照。全部是模型
        自己查询过的信息的重放(防滑窗遗忘),没有环境额外赠予。空则返回空串。"""
        lines = []
        if self.last_inv_snapshot is not None:
            lines.append(f"Your inventory as of step {self.last_inv_step} "
                         f"(when you last checked): {self.last_inv_snapshot}")
        if self.known_recipes:
            lines.append("Recipes you have learned so far (from get_info):")
            for item in sorted(self.known_recipes):
                lines.append(f"  {self.known_recipes[item]}")
        return ("\n" + "\n".join(lines)) if lines else ""

    def _info(self) -> dict:
        return {
            "won": self.won,
            "extra.gt_plan": self.gt_plan,
            "extra.difficulty": self._task["misc"].get("difficulty"),
            "extra.max_depth": self._task["misc"].get("max_depth"),
            "extra.task_id": self._task.get("id"),
        }

    # ---------------- step ----------------
    def step(self, action: str) -> Tuple[str, float, bool, dict]:
        action = (action or "").strip()
        self.steps += 1

        # 循环检测(官方保护机制):同一动作连续重复即时止损
        if action == self._last_action:
            self._repeat_count += 1
        else:
            self._repeat_count = 1
            self._last_action = action
        if self._repeat_count >= LOOP_REPEAT_LIMIT:
            obs = ("Loop detected: the same action was repeated "
                   f"{self._repeat_count} times. Episode terminated.")
            return obs, 0.0, True, self._info()

        obs = self._execute(action)

        # 成功判定:每个目标的净合成数(当前-初始)达标(与官方一致)
        if all(
            self.inventory.get(it, 0) - self.initial_inventory.get(it, 0) >= n
            for it, n in self.targets.items()
        ):
            self.won = True
            return obs + "\nAll target items crafted. Task complete!", 1.0, True, self._info()

        done = self.steps >= self.max_steps
        return obs + self._state_block(), 0.0, done, self._info()

    # ---------------- 动作执行 ----------------
    def _execute(self, action: str) -> str:
        if action == "inventory":
            if not self.inventory:
                inv_text = "(empty)"
            else:
                inv_text = ", ".join(f"{k}: {v}" for k, v in sorted(self.inventory.items()))
            # 记快照供状态块重放,直到下一次 inventory 查询才刷新
            self.last_inv_snapshot = inv_text
            self.last_inv_step = self.steps
            return "Inventory: " + inv_text

        m = _GETINFO_RE.match(action)
        if m:
            # 逗号或空格分隔均接受(2026-08-19 修复:medium 诊断 8 题里 7 题因模型
            # 用空格分隔、被整体解析成一个不存在的物品名而循环致死。synth 物品名
            # 不含空格,按 [,\s]+ 切分无歧义;官方是 Python 函数调用无此问题)。
            items = [x.strip().strip("'\"[]") for x in re.split(r"[,\s]+", m.group(1))
                     if x.strip().strip("'\"[]")]
            return self._get_info(items[:20])

        m = _CRAFT_RE.match(action)
        if m:
            target_count = int(m.group(1))
            target_item = m.group(2)
            ingredients: Dict[str, int] = {}
            for part in m.group(3).split(","):
                im = _ING_RE.match(part)
                if not im:
                    return (f"Error: could not parse ingredient '{part.strip()}'. "
                            "Use: craft <N> <item> using <c1> <ing1>, <c2> <ing2>")
                ingredients[im.group(2)] = ingredients.get(im.group(2), 0) + int(im.group(1))
            return self._craft(ingredients, (target_item, target_count))

        return ("Error: unknown action. Valid actions: 'inventory' | "
                "'get_info item1, item2' | 'craft <N> <item> using <c1> <ing1>, <c2> <ing2>'")

    # get_info:与官方返回同构的结构化信息(可查配方/深度/库存)
    def _get_info(self, items: List[str]) -> str:
        out = []
        for item in items:
            entry = {
                "item": item,
                "can_craft": self.db.can_craft(item),
                "is_base": self.db.is_base_item(item),
                "in_inventory": self.inventory.get(item, 0),
                "crafting_depth": self.db.get_crafting_depth(item),
                "recipes": [
                    {"ingredients": dict(r.ingredients), "result_count": r.result_count}
                    for r in self.db.get_recipes_for_item(item)
                ],
            }
            out.append(entry)
            # 写入配方笔记本(模型查过才记;行文用 craft 命令同款语法便于照抄)。
            # synth 生成器每物品只有一个配方,取 [0] 即全部。
            recs = self.db.get_recipes_for_item(item)
            if recs:
                r = recs[0]
                ings = ", ".join(f"{c} {n}" for n, c in r.ingredients.items())
                self.known_recipes[item] = (
                    f"craft {r.result_count} {item} using {ings}"
                    f"   (depth {self.db.get_crafting_depth(item)})")
            elif self.db.is_base_item(item):
                self.known_recipes[item] = f"{item}: base ingredient, cannot be crafted"
        return repr(out)

    # craft:逐条复刻官方错误路径(措辞保持一致,便于和官方行为对表)
    def _craft(self, ingredients: Dict[str, int], target: Tuple[str, int]) -> str:
        target_item, target_count = target
        if target_count <= 0:
            return f"Error: target count must be positive, got {target_count}"

        recipes = self.db.get_recipes_for_item(target_item)
        if not recipes:
            return f"Error: No recipe found for {target_item}"

        errors = []
        for recipe_idx, recipe in enumerate(recipes):
            # 1) 总产出数必须能被单次产量整除
            if target_count % recipe.result_count != 0:
                errors.append(
                    f"Recipe {recipe_idx + 1}: Target count {target_count} is not divisible "
                    f"by recipe result count {recipe.result_count}"
                )
                continue
            num_crafts = target_count // recipe.result_count

            # 2) 每种配方原料必须被提供,且数量精确等于 单次用量×执行次数
            recipe_error = None
            for recipe_ing, per_count in recipe.ingredients.items():
                total_required = per_count * num_crafts
                if recipe_ing not in ingredients:
                    recipe_error = f"Missing ingredient {recipe_ing}. Need {total_required}"
                    break
                if ingredients[recipe_ing] != total_required:
                    recipe_error = (
                        f"Wrong amount of {recipe_ing}. Need {total_required}, "
                        f"provided {ingredients[recipe_ing]}"
                    )
                    break
            if recipe_error:
                errors.append(f"Recipe {recipe_idx + 1}: {recipe_error}")
                continue

            # 3) 不允许提供配方之外的多余原料
            extra = [ing for ing in ingredients if ing not in recipe.ingredients]
            if extra:
                errors.append(
                    f"Recipe {recipe_idx + 1}: Extra ingredients not required: {', '.join(extra)}"
                )
                continue

            # 4) 库存必须足够
            missing = []
            for recipe_ing, per_count in recipe.ingredients.items():
                required = per_count * num_crafts
                available = self.inventory.get(recipe_ing, 0)
                if available < required:
                    missing.append(f"{recipe_ing}: need {required}, have {available}")
            if missing:
                errors.append(
                    f"Recipe {recipe_idx + 1}: Insufficient ingredients in inventory: "
                    + ", ".join(missing)
                )
                continue

            # 5) 通过全部检查:扣原料、加产物
            for recipe_ing, per_count in recipe.ingredients.items():
                amount = per_count * num_crafts
                self.inventory[recipe_ing] -= amount
                if self.inventory[recipe_ing] <= 0:
                    del self.inventory[recipe_ing]
            crafted = recipe.result_count * num_crafts
            self.inventory[target_item] = self.inventory.get(target_item, 0) + crafted
            return f"Successfully crafted {crafted} {target_item}(s)"

        return (f"Error: All {len(recipes)} recipe(s) failed for {target_item}. Errors:\n"
                + "\n".join(errors))
