# -*- coding: utf-8 -*-
"""装配 Search-QA 的 RSO+OPSD 递归环境管理器。只从 main_rso_opsd_search.py 调用。

体例镜像 textcraft_synth/opsd_factory.py。search 特有的两处:
  1. RecursiveEnvironmentManager.reset() 不传 kwargs(textcraft 任务由环境自采),
     而 search 的题目来自数据集 env_kwargs → SearchRecursiveEnvironmentManager
     在 super().reset() 之前把 kwargs 暂存进底层环境(envs.stage_reset_kwargs);
  2. 底层环境永远 reward=0(R(τ) 由 rso_core 取 root 行 node_success)——
     为了让 episode 级 reward 指标也有意义,step() 之后把"本轮 root 关闭"的槽的
     reward 补成 root success。
统计:基类 _process_batch 的 won/难度分桶照用(难度键缺省跳过),追加按
data_source 分桶的成功率与 rso/useless_goal_rate。
"""
from typing import List

import numpy as np

from agent_system.environments.env_package.search_rso.envs import build_search_rso_envs
from agent_system.environments.env_package.search_rso.opsd_priv import SearchOPSDAdapter
from agent_system.environments.env_package.search_rso.projection import search_rso_projection
from agent_system.environments.env_package.search_rso.recursive_adapter import (
    SearchRecursiveAdapter)
from agent_system.recursive.budget import PerAgentBudget
from agent_system.recursive.orchestrator import RecursiveEnvironmentManager


class SearchRecursiveEnvironmentManager(RecursiveEnvironmentManager):

    def reset(self, kwargs=None):
        # kwargs 是 gen_batch 弹出的 env_kwargs,rollout 侧给的是 numpy 对象数组——
        # 不能用 `or` 做真值判断(多元素数组真值歧义,预检 22178471 的死因),显式判 None。
        self.envs.stage_reset_kwargs(list(kwargs) if kwargs is not None else [])
        return super().reset()

    def step(self, text_actions: List[str]):
        n_closed_before = [len(r) for r in self.finished_records]
        obs, rewards, dones, infos = super().step(text_actions)
        # root 本轮关闭的槽:episode reward 补成 root 的 EM(纯指标用途,优势不读它)
        for i, recs in enumerate(self.finished_records):
            if len(recs) > n_closed_before[i]:
                root_recs = [r for r in recs[n_closed_before[i]:] if r.depth == 0]
                if root_recs:
                    rewards[i] = float(root_recs[-1].success)
        return obs, rewards, dones, infos

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        super()._process_batch(batch_idx, total_batch_list, total_infos, success)
        # 按 data_source 分桶的成功率(per-source val EM 的原料,mapping §四)
        for k in reversed(range(len(total_batch_list[batch_idx]))):
            if total_batch_list[batch_idx][k]["active_masks"]:
                info = total_infos[batch_idx][k]
                src = info.get("data_source")
                if src:
                    success[f"{src}_success_rate"].append(float(info.get("won", False)))
                break
        recs = self.finished_records[batch_idx] if batch_idx < len(self.finished_records) else []
        subs = [r for r in recs if r.depth > 0]
        success["rso/useless_goal_rate"].append(
            float(np.mean([bool(getattr(r, "useless_goal", False)) for r in subs])) if subs else 0.0)


def _make(config, adapter_cls):
    group_n = max(int(config.env.rollout.n), 1)
    _envs = build_search_rso_envs(
        seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n,
        is_train=True, env_config=config.env)
    _val_envs = build_search_rso_envs(
        seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1,
        is_train=False, env_config=config.env)

    rao_cfg = dict(config.env.get("rao", {}) or {})
    budget = PerAgentBudget.from_config(rao_cfg)
    adapter = adapter_cls(config)
    envs = SearchRecursiveEnvironmentManager(
        _envs, search_rso_projection, config, adapter=adapter, budget=budget, trace_tag="train")
    val_envs = SearchRecursiveEnvironmentManager(
        _val_envs, search_rso_projection, config, adapter=adapter, budget=budget, trace_tag="val")
    print(f"[search_rso] recursive envs: {budget}, history_length={adapter.history_length}, "
          f"trace_dir={rao_cfg.get('trace_dir')}, adapter={adapter_cls.__name__}")
    return envs, val_envs


def make_rso_search_envs(config):
    """阶段 1(纯 RSO,无蒸馏)。"""
    assert "search_rso" in str(config.env.env_name).lower(), \
        f"[search_rso] 工厂只装配 search_rso,收到 env_name={config.env.env_name}"
    return _make(config, SearchRecursiveAdapter)


def make_rso_opsd_search_envs(config):
    """RSO+OPSD(适配器带 build_priv,编排器据此打开 priv 通道)。"""
    assert "search_rso" in str(config.env.env_name).lower(), \
        f"[search_rso] 工厂只装配 search_rso,收到 env_name={config.env.env_name}"
    return _make(config, SearchOPSDAdapter)
