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

"""Ray-based vectorized TextCraft environment for SDAR / verl-agent (GiGPO).

Mirrors the webshop template:
  * ``TextCraftWorker``            -- Ray actor hosting ONE ``TextCraft`` env.
  * ``TextCraftMultiProcessEnv``   -- vectorized ``gym.Env`` over
                                       ``env_num * group_n`` workers.
  * ``build_textcraft_envs(...)``  -- factory used by ``make_envs``.

Key SDAR/GRPO invariants implemented here:
  * ITEM-HELD-OUT split: the 419 depth>=2 craftable items are deterministically
    partitioned into a train pool (~80%) and a disjoint test pool (~20%).
    ``is_train`` selects which pool a worker draws its goal from.
  * GRPO grouping: the same goal is repeated across each group of ``group_n``
    consecutive workers (``np.repeat(goal_idx, group_n)``), exactly like
    webshop's session grouping.
  * ``info['won'] = bool(reward == 1)`` on every step.
  * ``info['extra.gt_plan']`` = newline-joined ground-truth crafting plan for
    the episode goal (from the verified ``gt_plan.get_gt_plan``).
  * ``info['extra.goal_item']`` / ``info['extra.goal_depth']`` expose the goal.
  * 5-tuple ``(obs, reward, terminated, truncated, info)`` from TextCraft is
    converted to the ``(obs, reward, done, info)`` contract the manager expects
    (``done = terminated or truncated``).
"""

import os
import sys

import ray
import gym
import numpy as np

# Vendored TextCraft package (symlink `textcraft_pkg` -> textcraft_src/textcraft).
# Its real parent directory contains a package literally named `textcraft`.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_LINK = os.path.join(_HERE, "textcraft_pkg")
_MINECRAFT_DIR = os.path.realpath(_PKG_LINK)          # dir holding recipes/
_TEXTCRAFT_PARENT = os.path.dirname(_MINECRAFT_DIR)   # dir holding `textcraft/`
if _TEXTCRAFT_PARENT not in sys.path:
    sys.path.insert(0, _TEXTCRAFT_PARENT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# -----------------------------------------------------------------------------
# 递归评测的委托轮填充动作哨兵(worker.step 对它短路;flat 路径永远不会发出它)
RECURSIVE_NOOP = "__noop__"

# Item-held-out split ---------------------------------------------------------
# -----------------------------------------------------------------------------

def compute_item_split(minecraft_dir, min_depth=2, test_every=5, tree=None):
    """Deterministically partition all depth>=``min_depth`` craftable items into
    disjoint train / test pools.

    The items are sorted by item id (stable, deterministic) and every
    ``test_every``-th item (index % test_every == 0) is assigned to the test
    pool; the rest form the train pool. With ``test_every=5`` this yields an
    ~80/20 train/test split with NO overlap.

    Returns ``(train_items, test_items)`` where each is a list of item ids
    together with their min depth: ``[(item_id, depth), ...]``.
    """
    if tree is None:
        from textcraft.crafting_tree import CraftingTree
        tree = CraftingTree(minecraft_dir=minecraft_dir)
    items = sorted(tree.item_recipes_min_depth(min_depth), key=lambda x: x[0])
    train_items, test_items = [], []
    for idx, (item, depth) in enumerate(items):
        if idx % test_every == 0:
            test_items.append((item, depth))
        else:
            train_items.append((item, depth))
    return train_items, test_items


# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

class TextCraftWorker:
    """Ray remote actor hosting a single ``TextCraft`` environment.

    A worker owns one env AND its own copy of the split-restricted goal pool.
    ``reset(goal_index)`` selects the ``goal_index``-th item from this worker's
    pool (the caller guarantees the same ``goal_index`` is passed to every
    worker in a GRPO group, so the whole group shares the goal).
    """

    def __init__(self, seed, is_train, env_kwargs):
        env_kwargs = dict(env_kwargs or {})
        minecraft_dir = env_kwargs.get("minecraft_dir", _MINECRAFT_DIR)
        self.max_steps = int(env_kwargs.get("max_steps", 20))
        min_depth = int(env_kwargs.get("min_depth", 2))
        test_every = int(env_kwargs.get("test_every", 5))

        # Lazy import to keep the driver process light and avoid import order
        # issues under Ray.
        from textcraft.env import TextCraft
        from textcraft.crafting_tree import CraftingTree
        import gt_plan  # vendored, self-contained GT plan extractor

        self._minecraft_dir = minecraft_dir
        self.env = TextCraft(minecraft_dir=minecraft_dir)
        # Reuse the env's own crafting tree (avoid rebuilding it 2 more times per
        # worker; rebuilding parses 860 recipe JSONs and made Ray init crawl).
        self._tree = self.env.crafting_tree
        self._get_gt_plan = gt_plan.get_gt_plan

        # Split-restricted pool for this worker (computed from the existing tree).
        train_items, test_items = compute_item_split(
            minecraft_dir, min_depth=min_depth, test_every=test_every, tree=self._tree)
        self._pool = train_items if is_train else test_items
        self.is_train = is_train

        self._step_count = 0
        self._cur_goal = None
        self._cur_depth = None
        self._cur_gt_plan = ""

    def _pool_size(self):
        return len(self._pool)

    def _compute_gt_plan(self, goal_item):
        try:
            actions = self._get_gt_plan(self._tree, goal_item, 1)
            return "\n".join(actions)
        except Exception as e:  # never crash rollout on plan extraction
            return "ERROR: could not compute GT plan for {} ({})".format(
                goal_item, e)

    def reset(self, goal_index):
        """Reset onto the ``goal_index``-th goal in this worker's pool.

        We override TextCraft's own seed->goal selection: TextCraft.reset(seed)
        picks a goal from the FULL item pool, which would break the split. So we
        set the goal explicitly here and re-run the recipe/distractor listing so
        the observation is well-formed.
        """
        import random
        from textcraft.utils import item_id_to_str

        goal_item, goal_depth = self._pool[goal_index % self._pool_size()]
        self._cur_goal = goal_item
        self._cur_depth = int(goal_depth)
        self._step_count = 0

        # Reproduce TextCraft.reset internals but with OUR chosen goal, so the
        # goal is drawn strictly from the split pool.
        self.env.inventory = {}
        random.seed(goal_index)
        self.env.goal = goal_item
        recipes_set = set()
        distractor_set = set()
        max_distractor = 10
        recipes, distractors = self.env.crafting_tree.create_recipe_set(goal_item)
        for recipe in recipes:
            recipes_set.add(recipe.recipe_str)
        for distractor in distractors:
            if distractor.recipe_str not in recipes_set:
                distractor_set.add(distractor.recipe_str)
        recipes_list = list(recipes_set) + random.sample(
            list(distractor_set), min(len(distractor_set), max_distractor))
        random.shuffle(recipes_list)
        obs = "Crafting commands:\n{}\n\nGoal: craft {}.".format(
            "\n".join(recipes_list), item_id_to_str(goal_item))

        self._cur_gt_plan = self._compute_gt_plan(goal_item)
        # 递归适配器契约(与 synth 的 extra.* 同名同义):配方清单 + 目标量 + 库存快照。
        # flat 路径不读这三个键,加了不影响原行为。
        self._cur_commands = list(recipes_list)
        self._cur_target_items = {item_id_to_str(goal_item): 1}

        info = {
            "won": False,
            "extra.gt_plan": self._cur_gt_plan,
            "extra.goal_item": goal_item,
            "extra.goal_depth": self._cur_depth,
            "extra.inventory": self._spaced_inventory(),
            "extra.target_items": dict(self._cur_target_items),
            "extra.commands": list(self._cur_commands),
        }
        return obs, info

    def _spaced_inventory(self):
        """库存字典的空格式物品名版本(synth 适配器按空格名判定目标达成)。"""
        from textcraft.utils import item_id_to_str
        return {item_id_to_str(k): v for k, v in self.env.inventory.items()}

    def step(self, action):
        """Execute one action; convert the 5-tuple to (obs, reward, done, info).

        ``done = terminated or truncated`` where truncation is our own
        ``max_steps`` budget.
        """
        if (action or "").strip() == RECURSIVE_NOOP:
            # 递归编排的委托轮填充动作:不进官方 env、不计步、无副作用
            # (官方 TextCraft 没有无副作用的 noop,喂垃圾会返回 Could not execute 并污染观测)。
            info = {
                "won": False,
                "extra.gt_plan": self._cur_gt_plan,
                "extra.goal_item": self._cur_goal,
                "extra.goal_depth": self._cur_depth,
                "task_score": 0.0,
                "extra.action_failed": False,
                "extra.inventory": self._spaced_inventory(),
                "extra.target_items": dict(self._cur_target_items),
                "extra.commands": list(self._cur_commands),
            }
            return "", 0.0, False, info

        obs, reward, terminated, truncated, _info = self.env.step(action)
        self._step_count += 1
        if self._step_count >= self.max_steps:
            truncated = True
        done = bool(terminated or truncated)

        info = {
            "won": bool(reward == 1),
            "extra.gt_plan": self._cur_gt_plan,
            "extra.goal_item": self._cur_goal,
            "extra.goal_depth": self._cur_depth,
            "task_score": float(reward),
            # 全量测评"无效动作率"的原料;只是多一个 info 键,判分与成败不受影响
            "extra.action_failed": bool(isinstance(obs, str) and obs.strip().startswith("Could not")),
            "extra.inventory": self._spaced_inventory(),
            "extra.target_items": dict(self._cur_target_items),
            "extra.commands": list(self._cur_commands),
        }
        return obs, float(reward), done, info

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Vectorized Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class TextCraftMultiProcessEnv(gym.Env):
    """Vectorized, Ray-based wrapper over ``env_num * group_n`` TextCraft envs.

    ``reset()`` draws ``env_num`` distinct goal indices from the split pool and
    repeats each ``group_n`` times so every GRPO group shares a goal.
    """

    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict = None,
    ) -> None:
        super().__init__()

        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train:
            assert group_n == 1, "eval must use group_n == 1"

        self._rng = np.random.RandomState(seed)

        env_kwargs = dict(env_kwargs or {})
        env_kwargs.setdefault("minecraft_dir", _MINECRAFT_DIR)
        self._env_kwargs = env_kwargs

        # ---- determine pool size (drives goal_idx sampling range) ----
        train_items, test_items = compute_item_split(
            env_kwargs["minecraft_dir"],
            min_depth=int(env_kwargs.get("min_depth", 2)),
            test_every=int(env_kwargs.get("test_every", 5)),
        )
        self.pool_size = len(train_items) if is_train else len(test_items)

        # ---- spin up Ray actors ----
        env_worker = ray.remote(**resources_per_worker)(TextCraftWorker)
        self._workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(
                seed + (i // self.group_n), is_train, self._env_kwargs)
            self._workers.append(worker)

    # ------------------------------------------------------------------
    def step(self, actions):
        if len(actions) != self.num_processes:
            raise ValueError(
                f"Expected {self.num_processes} actions, got {len(actions)}")

        futures = [w.step.remote(a) for w, a in zip(self._workers, actions)]
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        return obs_list, reward_list, done_list, info_list

    def reset(self):
        # Draw env_num distinct goal indices, then repeat each group_n times so
        # every GRPO group of workers shares the same goal.
        size = min(self.env_num, self.pool_size)
        idx = self._rng.choice(self.pool_size, size=size, replace=False)
        if size < self.env_num:  # pool smaller than env_num: allow repeats
            extra = self._rng.choice(
                self.pool_size, size=self.env_num - size, replace=True)
            idx = np.concatenate([idx, extra])
        idx = np.repeat(idx, self.group_n).tolist()

        futures = [w.reset.remote(int(i)) for w, i in zip(self._workers, idx)]
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    # ------------------------------------------------------------------
    def close(self):
        if getattr(self, "_closed", False):
            return
        try:
            ray.get([w.close.remote() for w in self._workers])
        except Exception:
            pass
        for worker in self._workers:
            try:
                ray.kill(worker)
            except Exception:
                pass
        self._closed = True

    def __del__(self):
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_textcraft_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
):
    """Mirror ``build_webshop_envs`` so ``make_envs`` can swap seamlessly."""
    return TextCraftMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )
