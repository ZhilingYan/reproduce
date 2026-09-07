# -*- coding: utf-8 -*-
"""装配 TextCraft-Synth 的递归环境管理器(训练 + 验证)。只从 main_rao.py 调用,不改 make_envs。

框架先例:agent_system/environments/env_manager.py:1000-1019(flat 的 synth 装配)——
本文件逐行镜像它,只有两处不同:
  1. env_kwargs 多传三个键(见 build_recursive_env_kwargs 的注释):
     append_state_block=False(状态块改由适配器按节点维护)、loop_detection=False(循环检测改按节点)、
     max_steps=不限(全局轮数上限由采集循环负责);
  2. 外面套的是 RecursiveEnvironmentManager(+ TextCraftSynthRecursiveAdapter),不是
     TextCraftSynthEnvironmentManager。
底层 build_textcraft_synth_envs / textcraft_synth_projection 与 flat 完全相同的实例与函数,
所以 flat 与 RAO 用的是同一套环境、同一套任务池、同一套动作解析——只差递归。

官方对照:train_areal_synth.py:124-129 按配置选 rollout_fn,synth_rollout.py:109-117 造 env/agent。

2026-08-26 教训(冒烟 21463170 因 UnboundLocalError 3 分钟即败):本文件依赖 Ray,此前没有 CPU 测试
覆盖;env_kwargs 的拼装现已抽成纯函数 build_recursive_env_kwargs,由 tests/test_recursive_synth_adapter.py
E11 用真实 yaml + 冒烟脚本同款覆盖键验证。
"""
from functools import partial

from omegaconf import OmegaConf

from agent_system.environments.env_package.textcraft_synth import (
    build_textcraft_synth_envs, textcraft_synth_projection)
from agent_system.environments.env_package.textcraft_synth.recursive_adapter import (
    TextCraftSynthRecursiveAdapter)
from agent_system.recursive.budget import PerAgentBudget
from agent_system.recursive.orchestrator import RecursiveEnvironmentManager

DEFAULT_ENV_MAX_STEPS = 10 ** 6   # 底层 env 的步数上限:递归模式下形同不限(全局限制由采集轮数负责)


def build_recursive_env_kwargs(config) -> dict:
    """拼底层 TextCraftSynthEnvs 的 env_kwargs。纯函数,不碰 Ray,可在 CPU 上测试。"""
    rao_cfg = dict(config.env.get("rao", {}) or {})
    ts_cfg = getattr(config.env, 'textcraft_synth', None)

    def _lst(v, default):
        return list(v) if v is not None else default

    # ---- 以下三键与 env_manager.py:1007-1013 逐行相同 ----
    env_kwargs = {
        'train_difficulties': _lst(getattr(ts_cfg, 'train_difficulties', None) if ts_cfg is not None else None, ['medium']),
        'val_difficulties': _lst(getattr(ts_cfg, 'val_difficulties', None) if ts_cfg is not None else None, ['easy', 'medium', 'hard']),
        'val_split': getattr(ts_cfg, 'val_split', 'val') if ts_cfg is not None else 'val',
        # ---- 递归模式的三处差异 ----
        # 状态块改由适配器按【节点】维护(synth_core.reset 注释;官方子 agent 是全新上下文)
        'append_state_block': False,
        # 2026-08-25 生命周期下沉:槽级循环检测分不清动作是哪个节点发的,一个子打转会杀全树
        # (冒烟 21461096:32% 的树这样死);改由适配器按节点检测(官方 agent.py:90-118 语义)
        'loop_detection': False,
        # 底层 env 的步数上限设为不限——官方每 agent 各 25 步、树总步数无界;lockstep 唯一的全局限制是
        # 采集循环的轮数上限(config.env.max_steps),到了按 CLOSE_ROLLOUT_END 收摊、各节点按各自口径判分。
        # 若把 env.max_steps 也喂给底层 env,它会数全槽轮数并返回 done,把强制收摊错误归因成 episode_end。
        'max_steps': int(rao_cfg.get('env_max_steps', DEFAULT_ENV_MAX_STEPS)),
    }
    return env_kwargs


def make_recursive_synth_envs(config):
    """返回 (envs, val_envs),接口与 make_envs 相同。"""
    assert "textcraft_synth" in str(config.env.env_name).lower(), \
        f"[RAO] 递归工厂目前只装配 textcraft_synth,收到 env_name={config.env.env_name}"

    # ---- 以下与 env_manager.py:1000-1017 逐行相同(env_kwargs 见上面的纯函数)----
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

    # ---- 差异 2:套递归管理器 ----
    rao_cfg = dict(config.env.get("rao", {}) or {})
    budget = PerAgentBudget.from_config(rao_cfg)
    adapter = TextCraftSynthRecursiveAdapter(config)
    envs = RecursiveEnvironmentManager(_envs, projection_f, config, adapter=adapter, budget=budget,
                                       trace_tag="train")
    val_envs = RecursiveEnvironmentManager(_val_envs, projection_f, config, adapter=adapter, budget=budget,
                                           trace_tag="val")
    print(f"[RAO] recursive synth envs: {budget}, state_block_scope={adapter.state_block_scope}, "
          f"history_length={adapter.history_length}, trace_dir={rao_cfg.get('trace_dir')}, "
          f"env_max_steps={env_kwargs['max_steps']}, loop_detection={env_kwargs['loop_detection']}")
    return envs, val_envs
