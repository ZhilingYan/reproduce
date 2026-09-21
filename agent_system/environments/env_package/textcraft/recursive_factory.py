# -*- coding: utf-8 -*-
"""装配 textcraft_ood(真 TextCraft / ADaPT 官方环境)的递归环境管理器。

逐行镜像 textcraft_synth/recursive_factory.py,差异只有三处:
  1. 底层换 build_textcraft_envs / textcraft_projection(env_manager.py:1149-1163 的 flat 装配同款);
  2. 适配器换 RealTextcraftAdapter(recursive_adapter.py,synth 子类五覆写);
  3. env_kwargs 按真 textcraft worker 的键拼(max_steps/min_depth/test_every;
     synth 的难度/val_split/状态块/循环检测键该 worker 不识)。
用途:textcraft 训练出的 ckpt 在真 TextCraft 官方 test 划分(84 题)上的 OOD 推理,
走 trainer.val_only=True,与训练后全量测评完全同一条代码路径。
"""
from functools import partial

from omegaconf import OmegaConf

from agent_system.environments.env_package.textcraft import (
    build_textcraft_envs, textcraft_projection)
from agent_system.environments.env_package.textcraft.recursive_adapter import (
    RealTextcraftAdapter)
from agent_system.recursive.budget import RootChildBudget
from agent_system.recursive.orchestrator import RecursiveEnvironmentManager

DEFAULT_ENV_MAX_STEPS = 10 ** 6   # 底层 env 步数上限:递归模式下形同不限(全局限制由采集轮数负责)


def build_recursive_textcraft_env_kwargs(config) -> dict:
    """拼底层真 TextCraft workers 的 env_kwargs。纯函数,不碰 Ray。"""
    rao_cfg = dict(config.env.get("rao", {}) or {})
    tc_cfg = getattr(config.env, 'textcraft', None)
    return {
        # 若把 config.env.max_steps 喂给底层 env,它会数单槽步数并返回 done,
        # 把递归的强制收摊错误归因成 episode_end(synth 工厂同注释)
        'max_steps': int(rao_cfg.get('env_max_steps', DEFAULT_ENV_MAX_STEPS)),
        'min_depth': int(getattr(tc_cfg, 'min_depth', 2)) if tc_cfg is not None else 2,
        'test_every': int(getattr(tc_cfg, 'test_every', 5)) if tc_cfg is not None else 5,
    }


def make_recursive_textcraft_envs(config):
    """返回 (envs, val_envs),接口与 make_envs 相同。"""
    name = str(config.env.env_name).lower()
    assert "textcraft_ood" in name, \
        f"[RAO] 本工厂只装配 textcraft_ood(真 TextCraft OOD 评测),收到 env_name={config.env.env_name}"

    group_n = max(int(config.env.rollout.n), 1)
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)
    env_kwargs = build_recursive_textcraft_env_kwargs(config)

    _envs = build_textcraft_envs(
        seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n,
        is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
    _val_envs = build_textcraft_envs(
        seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1,
        is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
    projection_f = partial(textcraft_projection)

    rao_cfg = dict(config.env.get("rao", {}) or {})
    # root 与子解耦:root_max_steps 对齐 flat 的 env.max_steps,否则比的是预算不是方法
    # (未配 root_max_steps 时退回 per_agent_max_steps,与 PerAgentBudget 行为一致)
    budget = RootChildBudget.from_config(rao_cfg)
    adapter = RealTextcraftAdapter(config)
    envs = RecursiveEnvironmentManager(_envs, projection_f, config, adapter=adapter, budget=budget,
                                       trace_tag="train")
    val_envs = RecursiveEnvironmentManager(_val_envs, projection_f, config, adapter=adapter, budget=budget,
                                           trace_tag="val")
    print(f"[RAO] recursive textcraft_ood envs: {budget}, "
          f"stuck_detection={adapter.stuck_detection}, "
          f"history_length={adapter.history_length}, trace_dir={rao_cfg.get('trace_dir')}, "
          f"env_max_steps={env_kwargs['max_steps']}", flush=True)
    return envs, val_envs
