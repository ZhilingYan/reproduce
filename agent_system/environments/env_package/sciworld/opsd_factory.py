# -*- coding: utf-8 -*-
"""装配 ScienceWorld 的 RSO+OPSD 递归环境管理器。只从 main_rso_opsd_sciworld.py 调用。

体例镜像 search_rso/opsd_factory.py。sciworld 特有的三处:
  1. 任务由环境自采(textcraft 口径),manager.reset 忽略数据集 kwargs
     (假 parquet 只驱动批次,见 examples/data_preprocess/make_sciworld_dummy_parquet.py);
  2. 底层环境 zero_reward=True(R(τ) 由 rso_core 取 root 行 node_success = 官方分
     负分裁零/100)——step() 之后把"本轮 root 关闭"槽的 reward 补成 root success,
     让 episode 级 reward 指标有意义(search opsd_factory.py:35-44 同款);
  3. 预算 = RootChildBudget(root 60 / child 20 / depth 2,方案 A;
     配置键 env.rao.root_max_steps / per_agent_max_steps / max_depth)。
统计:per-task 桶 + 猝死族单列宏平均 + focus_death_rate + child_zero_gain_rate。
"""
from typing import List

import numpy as np

from agent_system.environments.env_package.sciworld.budget import RootChildBudget
from agent_system.environments.env_package.sciworld.envs import (
    SUDDEN_DEATH_TASKS, build_sciworld_envs)
from agent_system.environments.env_package.sciworld.opsd_priv import SciWorldOPSDAdapter
from agent_system.environments.env_package.sciworld.projection import sciworld_projection
from agent_system.environments.env_package.sciworld.recursive_adapter import (
    SciWorldRecursiveAdapter)
from agent_system.recursive.orchestrator import RecursiveEnvironmentManager


class SciWorldRecursiveEnvironmentManager(RecursiveEnvironmentManager):

    def reset(self, kwargs=None):
        # 任务由环境自采(env_manager.py textcraft 分支同款),数据集 kwargs 忽略
        return super().reset()

    def step(self, text_actions: List[str]):
        n_closed_before = [len(r) for r in self.finished_records]
        obs, rewards, dones, infos = super().step(text_actions)
        # root 本轮关闭的槽:episode reward 补成 root 官方分(纯指标用途,优势不读它)
        for i, recs in enumerate(self.finished_records):
            if len(recs) > n_closed_before[i]:
                root_recs = [r for r in recs[n_closed_before[i]:] if r.depth == 0]
                if root_recs:
                    rewards[i] = float(root_recs[-1].success)
        return obs, rewards, dones, infos

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        super()._process_batch(batch_idx, total_batch_list, total_infos, success)
        recs = self.finished_records[batch_idx] if batch_idx < len(self.finished_records) else []
        roots = [r for r in recs if r.depth == 0]
        subs = [r for r in recs if r.depth > 0]
        for k in reversed(range(len(total_batch_list[batch_idx]))):
            if total_batch_list[batch_idx][k]["active_masks"]:
                info = total_infos[batch_idx][k]
                task = str(info.get("extra.task") or "")
                score = float(roots[-1].success) if roots else 0.0
                if task:
                    success[f"{task}_official"].append(score)
                    bucket = "sudden_death" if task in SUDDEN_DEATH_TASKS else "main"
                    success[f"{bucket}_official"].append(score)
                success["official_score"].append(score)
                success["rso/focus_death_rate"].append(
                    float(bool(info.get("extra.focus_death", False))))
                break
        success["rso/child_zero_gain_rate"].append(
            float(np.mean([float(r.success) == 0.0 for r in subs])) if subs else 0.0)


def _make(config, adapter_cls):
    from omegaconf import OmegaConf, open_dict
    # priv 现算需要每步的 goal_progress:装配时强制打开取数开关
    with open_dict(config):
        if "sciworld" not in config.env:
            config.env.sciworld = OmegaConf.create({})
        config.env.sciworld.fetch_goal_progress = True

    group_n = max(int(config.env.rollout.n), 1)
    _envs = build_sciworld_envs(
        seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n,
        is_train=True, env_config=config.env, zero_reward=True)
    _val_envs = build_sciworld_envs(
        seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1,
        is_train=False, env_config=config.env, zero_reward=True)

    rao_cfg = dict(config.env.get("rao", {}) or {})
    budget = RootChildBudget.from_config(rao_cfg)
    adapter = adapter_cls(config)
    envs = SciWorldRecursiveEnvironmentManager(
        _envs, sciworld_projection, config, adapter=adapter, budget=budget, trace_tag="train")
    val_envs = SciWorldRecursiveEnvironmentManager(
        _val_envs, sciworld_projection, config, adapter=adapter, budget=budget, trace_tag="val")
    print(f"[sciworld_rso] recursive envs: {budget}, history_length={adapter.history_length}, "
          f"trace_dir={rao_cfg.get('trace_dir')}, adapter={adapter_cls.__name__}")
    return envs, val_envs


def make_rso_sciworld_envs(config):
    """阶段 1(纯 RSO,无蒸馏)。"""
    assert "sciworld_rso" in str(config.env.env_name).lower(), \
        f"[sciworld_rso] 工厂只装配 sciworld_rso,收到 env_name={config.env.env_name}"
    return _make(config, SciWorldRecursiveAdapter)


def make_rso_opsd_sciworld_envs(config):
    """RSO+OPSD(适配器带 build_priv,编排器据此打开 priv 通道)。"""
    assert "sciworld_rso" in str(config.env.env_name).lower(), \
        f"[sciworld_rso] 工厂只装配 sciworld_rso,收到 env_name={config.env.env_name}"
    return _make(config, SciWorldOPSDAdapter)
