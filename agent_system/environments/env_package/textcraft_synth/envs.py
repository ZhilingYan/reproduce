# -*- coding: utf-8 -*-
"""TextCraft-Synth 的 Ray 批量封装,结构镜像 env_package/textcraft/envs.py。

分工:synth_core.py 管单环境逻辑(动作/配方/成功判定),本文件只管
"128 个环境怎么并行、GRPO 组怎么共享任务":
  * 每个 Ray worker 进程持有一个 SynthTextCraftEnv + 自己 split 的任务池;
  * reset() 由批量封装抽 env_num 个任务下标,每个重复 group_n 次 →
    同一 GRPO 组的 worker 拿到同一个任务(组内奖励可比的前提);
  * 训练/验证池 = 官方 jsonl 的 train/val 文件 × 配置里的难度过滤
    (与官方对齐:难度切分自带,不需要 item split 那套)。
info 额外字段:extra.gt_plan(OPSD 特权通道)/extra.difficulty/extra.max_depth。
"""
import numpy as np
import ray

from .synth_core import SynthTextCraftEnv, load_tasks, get_shared_recipe_db


class TextCraftSynthWorker:
    """一个 Ray 进程 = 一个环境实例 + 一份任务池。"""

    def __init__(self, seed: int, is_train: bool, env_kwargs: dict):
        # val_split 允许把验证池换成固定子集文件(如 val100 = 每难度 25 题,
        # 抽样种子与 id 清单见 data/val100_ids.json)。2026-08-18 引入:
        # 全量 632 验证 worker 曾把节点内存顶穿(OUT_OF_MEMORY, 209GB>200GB)。
        split = "train" if is_train else env_kwargs.get("val_split", "val")
        difficulties = env_kwargs.get(
            "train_difficulties" if is_train else "val_difficulties")
        self._tasks = load_tasks(split, difficulties)
        self._max_steps = int(env_kwargs.get("max_steps", 75))
        self._env = SynthTextCraftEnv(get_shared_recipe_db())

    def pool_size(self):
        return len(self._tasks)

    def reset(self, task_index: int):
        task = self._tasks[task_index % len(self._tasks)]
        obs, info = self._env.reset(task, max_steps_override=self._max_steps)
        return obs, info

    def step(self, action: str):
        return self._env.step(action)


class TextCraftSynthEnvs:
    """env_num * group_n 个 worker 的批量封装(接口与其他 env_package 一致)。"""

    def __init__(self, seed, env_num, group_n, resources_per_worker,
                 is_train=True, env_kwargs=None):
        if not ray.is_initialized():
            ray.init()
        self.env_num = env_num
        self.group_n = group_n
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train:
            assert group_n == 1, "eval must use group_n == 1"
        self._rng = np.random.RandomState(seed)
        env_kwargs = dict(env_kwargs or {})
        env_kwargs.setdefault("train_difficulties", ["medium"])
        env_kwargs.setdefault("val_difficulties", ["easy", "medium", "hard"])

        worker_cls = ray.remote(**resources_per_worker)(TextCraftSynthWorker)
        self._workers = [
            worker_cls.remote(seed + (i // group_n), is_train, env_kwargs)
            for i in range(self.num_processes)
        ]
        self.pool_size = ray.get(self._workers[0].pool_size.remote())
        # 验证池按固定顺序轮转(每次 reset 换下一批任务),训练池随机抽
        self._val_cursor = 0

    def reset(self):
        if self.is_train:
            size = min(self.env_num, self.pool_size)
            idx = self._rng.choice(self.pool_size, size=size, replace=False)
            if size < self.env_num:
                extra = self._rng.choice(self.pool_size, size=self.env_num - size, replace=True)
                idx = np.concatenate([idx, extra])
        else:
            idx = (np.arange(self.env_num) + self._val_cursor) % self.pool_size
            self._val_cursor = (self._val_cursor + self.env_num) % self.pool_size
        idx = np.repeat(idx, self.group_n).tolist()

        futures = [w.reset.remote(int(i)) for w, i in zip(self._workers, idx)]
        results = ray.get(futures)
        obs_list = [r[0] for r in results]
        info_list = [r[1] for r in results]
        return obs_list, info_list

    def step(self, actions):
        if len(actions) != self.num_processes:
            raise ValueError(f"Expected {self.num_processes} actions, got {len(actions)}")
        futures = [w.step.remote(a) for w, a in zip(self._workers, actions)]
        results = ray.get(futures)
        obs_list = [r[0] for r in results]
        reward_list = [r[1] for r in results]
        done_list = [r[2] for r in results]
        info_list = [r[3] for r in results]
        return obs_list, reward_list, done_list, info_list

    def close(self):
        for w in self._workers:
            ray.kill(w, no_restart=True)


def build_textcraft_synth_envs(seed, env_num, group_n, resources_per_worker,
                               is_train=True, env_kwargs=None):
    return TextCraftSynthEnvs(seed, env_num, group_n, resources_per_worker,
                              is_train=is_train, env_kwargs=env_kwargs)
