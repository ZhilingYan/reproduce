### GENERAL SKILLS ###

Here are some useful general strategies for completing crafting tasks in this environment:

1. Your Starting Inventory Is Sufficient: The base ingredients you hold at the start are guaranteed to be enough to craft the target items (possibly via intermediate crafts). If a craft fails, the cause is a wrong recipe, wrong amounts, or missing intermediates — never an impossible task.

2. Query Recipes Before Acting: Use 'get_info item1, item2, ...' to learn each item's recipe (ingredient amounts per craft and result count per craft). Batch several items into ONE get_info call instead of querying them one at a time; each query costs a step from your budget.

3. Backward-Chain from the Target: Decompose the target item's recipe into its inputs. For each input you do not already hold in sufficient quantity, query its recipe and recurse until every leaf is an ingredient you hold. Only then start crafting.

4. Craft in Dependency Order: Craft intermediate items strictly before the recipes that consume them. A craft fails unless every listed ingredient is already in your inventory with the exact required count.

5. Respect Result-Count Multiples: A recipe produces a fixed number of items per execution. You can only craft in multiples of that result count (a recipe producing 2 can yield 2, 4, 6...). Round your requested amount UP to the next multiple when the needed amount is not itself a multiple.

6. Scale Ingredient Totals Exactly: The ingredient amounts you list must EXACTLY equal per-craft amounts times the number of executions. For 'craft 4 X using ...' where one craft makes 2 X from 2 ore, you must list '4 ore' — no more, no less. Extra or missing amounts make the craft fail.

7. Count What You Already Hold: If some target items are already in your inventory, you must still craft the requested number ON TOP of what you hold — success is measured by NET newly crafted items, not final inventory. Never subtract held targets from the requested amount.

8. Use Inventory to Recover: If unsure what you currently hold (e.g. after a failed craft), issue 'inventory' once, reconcile it against your plan, and resume from the first unmet requirement. Do not spam inventory checks.

9. Read Error Messages Literally: Failure messages state the exact problem (unknown recipe, wrong ingredient amount, missing ingredient, result-count multiple violation). Fix precisely what the message names instead of retrying the same command.

10. One Action per Step, No Filler: Each step executes exactly one 'inventory', 'get_info', or 'craft' action. Plan so that the final craft of the target items is your last action; the episode ends successfully the moment the requested amount has been crafted.
