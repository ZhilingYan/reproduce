# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --------------------- TextCraft --------------------- #
# TextCraft is a Minecraft-style crafting environment. The observation already
# lists the available crafting commands (recipes + distractors) and the goal
# item. The agent must gather base items and craft intermediate items until it
# crafts the goal.
#
# Available actions:
#   get <N> <item>                       -- obtain N of a fetchable base item
#   craft <N> <item> using <c1> <i1>, <c2> <i2>, ...
#                                        -- craft using inputs from inventory
#   inventory                            -- inspect what you currently hold
#
# Items use lowercase names with spaces (e.g. "iron ingot", "oak planks").

TEXTCRAFT_TEMPLATE_NO_HIS = """
You are an expert agent playing TextCraft, a Minecraft-style crafting game.
Your objective is to craft the goal item by gathering base materials and crafting
intermediate items in the correct order.

Environment description (recipes and goal):
{current_observation}

You may take exactly one of the following actions per step:
- get <N> <item>            : obtain N copies of a base (fetchable) item, e.g. 'get 3 stick'
- craft <N> <item> using <c1> <input1>, <c2> <input2>, ... : craft using items you already hold,
  e.g. 'craft 1 wooden pickaxe using 3 oak planks, 2 stick'
- inventory                 : list the items currently in your inventory

Rules:
- You must already possess the required inputs (with sufficient counts) before you can craft.
- Only base/fetchable items can be obtained via 'get'; craftable items must be crafted.
- The episode ends successfully the moment you craft the goal item.

Now take one action for the current step.
First reason step-by-step about what to gather or craft next; this reasoning MUST be
enclosed within <think> </think> tags.
Then output exactly one action, enclosed within <action> </action> tags,
for example: <action>craft 1 stick using 2 oak planks</action>.
"""

TEXTCRAFT_TEMPLATE = """
You are an expert agent playing TextCraft, a Minecraft-style crafting game.
Your objective is to craft the goal item by gathering base materials and crafting
intermediate items in the correct order.

Environment description (recipes and goal):
{current_observation}

Prior to this step, you have already taken {step_count} step(s). Below are the most
recent {history_length} observations and the corresponding actions you took:
{action_history}

You are now at step {current_step}.

You may take exactly one of the following actions per step:
- get <N> <item>            : obtain N copies of a base (fetchable) item, e.g. 'get 3 stick'
- craft <N> <item> using <c1> <input1>, <c2> <input2>, ... : craft using items you already hold,
  e.g. 'craft 1 wooden pickaxe using 3 oak planks, 2 stick'
- inventory                 : list the items currently in your inventory

Rules:
- You must already possess the required inputs (with sufficient counts) before you can craft.
- Only base/fetchable items can be obtained via 'get'; craftable items must be crafted.
- The episode ends successfully the moment you craft the goal item.

Now take one action for the current step.
First reason step-by-step about what to gather or craft next; this reasoning MUST be
enclosed within <think> </think> tags.
Then output exactly one action, enclosed within <action> </action> tags,
for example: <action>craft 1 stick using 2 oak planks</action>.
"""
