# -*- coding: utf-8 -*-
"""装配 RSO+OPSD 的递归环境管理器。只从 main_rso_opsd.py 调用。

逐行克隆 recursive_factory.py:61-90 的 make_recursive_synth_envs,仅两处不同:
  1. 适配器换成 TextCraftSynthOPSDAdapter(多 build_priv + useless_goal 标记;
     编排器探测到 build_priv 才开 priv 通道,阶段 1 RSO 不受影响);
  2. 管理器换成 OPSDRecursiveEnvironmentManager(_process_batch 先 super 再追加
     rso/useless_goal_rate 一个统计键,RSO_method_design §四;checklist #6)。
env_kwargs 直接复用 recursive_factory.build_recursive_env_kwargs(同一份纯函数,不复制)。
"""
from functools import partial

from omegaconf import OmegaConf

import numpy as np

from agent_system.environments.env_package.textcraft_synth import (
    build_textcraft_synth_envs, textcraft_synth_projection)
from agent_system.environments.env_package.textcraft_synth.opsd_priv import (
    TextCraftSynthOPSDAdapter)
from agent_system.environments.env_package.textcraft_synth.recursive_factory import (
    build_recursive_env_kwargs)
from agent_system.recursive.budget import PerAgentBudget
from agent_system.recursive.orchestrator import RecursiveEnvironmentManager


class OPSDRecursiveEnvironmentManager(RecursiveEnvironmentManager):
    """基类统计不动,追加 useless_goal_rate(子节点 goal 有物品不在 root 剩余闭包内的占比)。"""

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        super()._process_batch(batch_idx, total_batch_list, total_infos, success)
        recs = self.finished_records[batch_idx] if batch_idx < len(self.finished_records) else []
        subs = [r for r in recs if r.depth > 0]
        success["rso/useless_goal_rate"].append(
            float(np.mean([bool(getattr(r, "useless_goal", False)) for r in subs])) if subs else 0.0)


def make_rso_opsd_synth_envs(config):
    """返回 (envs, val_envs),接口与 make_recursive_synth_envs 相同。"""
    assert "textcraft_synth" in str(config.env.env_name).lower(), \
        f"[RSO+OPSD] 工厂目前只装配 textcraft_synth,收到 env_name={config.env.env_name}"

    group_n = max(int(config.env.rollout.n), 1)
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)
    env_kwargs = build_recursive_env_kwargs(config)

    _envs = build_textcraft_synth_envs(
        seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n,
        is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
    _val_envs = build_textcraft_synth_envs(
        seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1,
        is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
    projection_f = partial(textcraft_synth_projection)

    rao_cfg = dict(config.env.get("rao", {}) or {})
    budget = PerAgentBudget.from_config(rao_cfg)
    adapter = TextCraftSynthOPSDAdapter(config)
    envs = OPSDRecursiveEnvironmentManager(_envs, projection_f, config, adapter=adapter,
                                           budget=budget, trace_tag="train")
    val_envs = OPSDRecursiveEnvironmentManager(_val_envs, projection_f, config, adapter=adapter,
                                               budget=budget, trace_tag="val")
    print(f"[RSO+OPSD] recursive synth envs: {budget}, state_block_scope={adapter.state_block_scope}, "
          f"history_length={adapter.history_length}, trace_dir={rao_cfg.get('trace_dir')}, "
          f"env_max_steps={env_kwargs['max_steps']}, loop_detection={env_kwargs['loop_detection']}, "
          f"priv=行开局现算(build_priv)")
    return envs, val_envs
