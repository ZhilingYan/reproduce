"""
Ground-truth crafting-plan extractor for the TextCraft environment.

Given a goal item id and a CraftingTree, recursively expand the *minimum-depth*
recipe DAG down to base / fetchable leaves and return an ordered list of env
actions:

    "get N <leaf>"                       -- for fetchable (non-craftable) leaves
    "craft N <item> using C1 i1, C2 i2"  -- for craftable internal nodes

The plan is emitted in dependency order (leaves first, goal last), so that
feeding the actions sequentially into TextCraft.step() satisfies every recipe's
input requirements before it is crafted.

Design notes
------------
* Minimum-depth recipe selection mirrors CraftingTree.get_min_depth_recipes:
  for each candidate recipe of an item, the recipe depth is
  max_over_inputs(depth(input) + 1); we pick the recipe achieving the item's
  stored min_depth (ties broken by first encountered).
* Multi-output recipes: a recipe may yield `output_count > 1` items per craft.
  We craft ceil(needed / output_count) times, which can over-produce; that is
  fine for the env (surplus stays in inventory).
* Tag inputs: a recipe input may reference a *tag* (e.g. minecraft:planks)
  rather than a concrete item. We resolve the tag to a single canonical
  concrete item (CraftingTree.get_items_with_tags), preferring one that is
  craftable, else fetchable. The env's craft matcher accepts a concrete item
  for a tag slot via item_id_to_tag, so supplying the canonical item works.
  NOTE: after CraftingTree.clean_up_recipes(), tags that have no craftable
  member are rewritten to plain item ids, so most leaf "tags" are already
  concrete fetchable items by the time we see them.
* Cycles: we carry a `visiting` set along each DAG path. If an item reappears
  on its own ancestry path we treat it as a fetchable leaf instead of
  recursing (defensive; CraftingTree already strips cyclic recipes at load).
* Aggregation: identical (item, recipe) craft steps and identical `get` leaves
  are accumulated by required count, then emitted once with the summed count,
  in a topologically valid order.
"""

import os
import sys
from math import ceil
from collections import OrderedDict

# Ensure the vendored TextCraft package (importable as `textcraft`) is on the
# path even when this module is run/imported standalone. This copy lives inside
# the SDAR env_package; the textcraft package is vendored as the sibling
# symlink `textcraft_pkg` -> textcraft_src/textcraft. We expose its parent dir
# (which contains a directory literally named `textcraft`) so `import textcraft`
# resolves to the verified 1.16.5 recipe package.
_HERE = os.path.dirname(os.path.abspath(__file__))
# textcraft_pkg is a symlink to .../textcraft_src/textcraft; its real parent
# contains the `textcraft` package directory.
_SRC = os.path.dirname(os.path.realpath(os.path.join(_HERE, "textcraft_pkg")))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from textcraft.utils import item_id_to_str  # noqa: E402


def _select_min_depth_recipe(tree, item_id):
    """Return the recipe for `item_id` that achieves its minimum depth.

    Returns None if the item has no recipe (i.e. it is a fetchable leaf).
    """
    recipes = tree.itemid_recipes.get(item_id)
    if not recipes:
        return None
    target_depth = tree.get_min_depth(item_id)
    best = None
    best_depth = None
    for recipe in recipes:
        depths = [tree.get_min_depth(inp.item_tag.name) + 1
                  for inp in recipe.input_items]
        rdepth = max(depths) if depths else 1
        if rdepth == target_depth:
            return recipe
        if best_depth is None or rdepth < best_depth:
            best, best_depth = recipe, rdepth
    return best


def _resolve_input_item(tree, item_tag):
    """Resolve a recipe input ItemTag to a concrete item id.

    For a concrete item the id is returned directly. For a tag we pick a
    canonical concrete member, preferring craftable then fetchable.
    Returns the concrete item id (a string) or None if unresolvable.
    """
    # concrete item id present
    if item_tag.item_id is not None and not tree.is_tag(item_tag.item_id):
        return item_tag.item_id
    # it is a tag -> resolve to a canonical member
    tag = item_tag.tag if item_tag.tag is not None else item_tag.item_id
    members = list(tree.get_items_with_tags(tag))
    if not members:
        # no member registered for the tag; fall back to the raw id
        return item_tag.item_id or tag
    # deterministic order, prefer craftable then valid/fetchable
    members.sort()
    for m in members:
        if tree.is_craftable(m):
            return m
    for m in members:
        if tree.is_valid_item(m):
            return m
    return members[0]


def _expand(tree, item_id, needed, visiting, craft_steps, leaf_needs):
    """Recursively accumulate craft steps and leaf fetches for `needed` copies
    of `item_id`.

    craft_steps : OrderedDict keyed by item_id -> dict with
                  {recipe, count(=times to craft), order}
    leaf_needs  : dict item_id -> total count to `get`
    """
    recipe = None
    if item_id not in visiting:
        recipe = _select_min_depth_recipe(tree, item_id)

    if recipe is None:
        # fetchable / base leaf (or cycle guard hit)
        leaf_needs[item_id] = leaf_needs.get(item_id, 0) + needed
        return

    output_count = recipe.output_item.count or 1
    times = ceil(needed / output_count)

    # recurse into inputs first so leaves/sub-crafts are registered before us
    visiting = visiting | {item_id}
    resolved_inputs = []  # (concrete_item_id, per_craft_count)
    for inp in recipe.input_items:
        concrete = _resolve_input_item(tree, inp.item_tag)
        per_craft = inp.count
        resolved_inputs.append((concrete, per_craft))
        _expand(tree, concrete, per_craft * times, visiting,
                craft_steps, leaf_needs)

    # register / accumulate this craft step (after its dependencies)
    if item_id in craft_steps:
        craft_steps[item_id]["times"] += times
    else:
        craft_steps[item_id] = {
            "recipe": recipe,
            "times": times,
            "output_count": output_count,
            "resolved_inputs": resolved_inputs,
            "order": len(craft_steps),
        }


def _craft_action_str(item_id, recipe, resolved_inputs):
    """Build a single 'craft 1 X using ...' action string using concrete items.

    We craft one output unit at a time (count matches the recipe's output
    count), repeated `times` by the caller, to keep matching simple and robust.
    """
    out_count = recipe.output_item.count or 1
    parts = []
    for (concrete, per_craft) in resolved_inputs:
        parts.append("{} {}".format(per_craft, item_id_to_str(concrete)))
    return "craft {} {} using {}".format(
        out_count, item_id_to_str(item_id), ", ".join(parts))


def get_gt_plan(tree, goal_item_id, goal_count=1):
    """Return an ordered list of env action strings that crafts `goal_count`
    of `goal_item_id`, expanding the min-depth recipe DAG.

    Raises ValueError if the goal is not craftable.
    """
    if goal_item_id not in tree.itemid_recipes:
        raise ValueError("Goal {} is not craftable".format(goal_item_id))

    craft_steps = OrderedDict()
    leaf_needs = {}
    _expand(tree, goal_item_id, goal_count, set(), craft_steps, leaf_needs)

    actions = []
    # 1) fetch all leaves first (deterministic order)
    for leaf in sorted(leaf_needs):
        actions.append("get {} {}".format(leaf_needs[leaf], item_id_to_str(leaf)))

    # 2) craft steps in dependency order (insertion order == dependency order
    #    because we register a node only after recursing into its inputs)
    for item_id, step in craft_steps.items():
        action = _craft_action_str(item_id, step["recipe"],
                                   step["resolved_inputs"])
        for _ in range(step["times"]):
            actions.append(action)
    return actions


if __name__ == "__main__":
    from textcraft.crafting_tree import CraftingTree
    tree = CraftingTree(minecraft_dir=os.path.realpath(os.path.join(_HERE, "textcraft_pkg")))
    for goal, _ in list(tree.item_recipes_min_depth(3))[:3]:
        print("GOAL:", goal, "depth", tree.get_min_depth(goal))
        for a in get_gt_plan(tree, goal):
            print("   ", a)
        print()
