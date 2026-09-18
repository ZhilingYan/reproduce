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

import asyncio
import concurrent.futures
from typing import Any, Dict, List, Tuple

import gym
import numpy as np
from omegaconf import DictConfig, ListConfig
from copy import deepcopy 


class SearchMultiProcessEnv(gym.Env):
    """
    - env_num  : Number of groups (logical sharding; keep the parameter for external compatibility)
    - group_n  : Number of environments per group
    - total_envs = env_num * group_n
    """

    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        is_train: bool = True,
        env_config: DictConfig | None = None,
    ) -> None:
        super().__init__()

        from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.search.env import SearchEnv

        self.env_num   = env_num
        self.group_n   = group_n
        self.batch_size = env_num * group_n
        self.is_train  = is_train
        self.max_steps = env_config.max_steps

        self._rng = np.random.RandomState(seed)

        # ---------- Key changes start ----------
        # 1) Normalize search_url into a list
        search_cfg  = env_config.search
        search_urls = search_cfg.search_url
        if not isinstance(search_urls, ListConfig):
            search_urls = [search_urls]

        n_clients = len(search_urls)

        # 2) Assign a URL to each env in a round-robin manner
        self.envs = []
        for idx in range(self.batch_size):
            cfg_i = deepcopy(search_cfg)
            cfg_i.search_url = search_urls[idx % n_clients]
            self.envs.append(SearchEnv(cfg_i))

        max_workers = min(self.batch_size, 256)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

    def _sync_reset(self, env, kwargs):
        extras = {
            "ground_truth": kwargs["ground_truth"],
            "max_turns": self.max_steps,
            "data_source": kwargs.get("data_source", "unknown")
        }
        env.reset(extras)
        obs = kwargs["question"]
        info = {'data_source': kwargs.get("data_source", "unknown")}
        # [2026-09-19 RSO_search 新增] GT 特权通道:把本题标准答案放进 info['extra.gt_plan'],
        # 走 rollout_loop._gtpriv_from_infos 的既有管道铺到每行,供 privileged_source=gt 的
        # teacher 取用(search 域的 gt = golden answers;textcraft 是配方计划,同一通道)。
        # 不开 gt 模式时该列无人消费,flat GRPO 行为不变。
        gt = kwargs.get("ground_truth")
        if isinstance(gt, dict):
            gt = gt.get("target")
        if gt is None:
            targets = []
        elif isinstance(gt, str):
            targets = [gt]
        else:
            targets = [str(x) for x in list(gt)]
        gt_text = ("The ground-truth answer to this question: " + "; ".join(targets)) if targets else None
        # 存到 env 实例上:rollout_loop 每一步都会用 step infos 重取 gt(rollout_loop.py:546),
        # 只在 reset 给会导致后续轮的行缺 gt_plan 列(预检 22179396:64 首轮行 vs 154 总行,
        # collate 不齐直接断言)。step 侧从这里回读。
        env._rso_gt_plan_text = gt_text
        if gt_text:
            info["extra.gt_plan"] = gt_text
        return obs, info
    
    def _sync_step(self, env, action: str):
        out = env.step(action)

        obs = out["observations"]
        obs = "" if len(obs) == 0 else obs[0]["content"].strip()
        reward = out["reward"]
        done = out["done"]

        info = dict(out.get("metadata", {}))
        info["postprocessed_action"] = out.get("postprocessed_action")
        info["won"] = bool(done and reward >= 1.0)
        # [2026-09-19] GT 特权通道:每步 info 都带(理由见 _sync_reset 的注释)
        gt_text = getattr(env, "_rso_gt_plan_text", None)
        if gt_text:
            info["extra.gt_plan"] = gt_text
        return obs, reward, done, info

    def reset(self, kwargs: List[Dict]):
        if len(kwargs) > self.batch_size:
            raise ValueError(f"Got {len(kwargs)} kwarg dicts, but the env was initialised with total_envs={self.batch_size}")

        pad_n = self.batch_size - len(kwargs)
        dummy_kw = {
                    "ground_truth": "",
                    "question": "",
                    "data_source": "unkown",
                }

        padded_kwargs = list(kwargs) + [dummy_kw] * pad_n
        valid_mask = [True] * len(kwargs) + [False] * pad_n

        tasks = [
            self._loop.run_in_executor(self._executor, self._sync_reset, env, kw)
            for env, kw in zip(self.envs, padded_kwargs)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))

        obs_list, info_list = map(list, zip(*results))

        obs_list = [o for o, keep in zip(obs_list, valid_mask) if keep]
        info_list = [i for i, keep in zip(info_list, valid_mask) if keep]

        return obs_list, info_list

    def step(self, actions: List[str]):
        if len(actions) > self.batch_size:
            raise ValueError(f"Got {len(actions)} actions, but the env was initialized with total_envs={self.batch_size}")

        pad_n = self.batch_size - len(actions)
        padded_actions = list(actions) + [""] * pad_n
        valid_mask = [True] * len(actions) + [False] * pad_n

        tasks = [
            self._loop.run_in_executor(self._executor, self._sync_step, env, act)
            for env, act in zip(self.envs, padded_actions)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))

        obs_list, reward_list, done_list, info_list = map(list, zip(*results))

        obs_list = [o for o, keep in zip(obs_list, valid_mask) if keep]
        reward_list = [r for r, keep in zip(reward_list, valid_mask) if keep]
        done_list = [d for d, keep in zip(done_list, valid_mask) if keep]
        info_list = [i for i, keep in zip(info_list, valid_mask) if keep]

        return obs_list, reward_list, done_list, info_list

    def close(self):
        if getattr(self, "_closed", False):
            return
        for env in self.envs:
            env.close()
        self._executor.shutdown(wait=True)
        self._loop.close()
        self._closed = True

    def __del__(self):
        self.close()


def build_search_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
    env_config=None,
):
    return SearchMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
        env_config=env_config,
    )