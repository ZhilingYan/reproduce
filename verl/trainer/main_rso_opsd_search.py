# -*- coding: utf-8 -*-
"""RSO + OPSD 的 Search-QA 域训练入口。

从 verl/trainer/main_rso_opsd.py 克隆(2026-09-18,克隆底本家规:以完整入口为底,
只换 search 域必须换的四样;textcraft 入口一个字不动,断言各守各的):
  换 1:环境断言 textcraft_synth → search_rso;补 decomp_path 存在性断言
  换 2:环境工厂 make_rso_opsd_synth_envs → search_rso.opsd_factory.make_rso_opsd_search_envs
  换 3:_KNOWN_OPSD_KEYS 增加 act_tags(act_mask 标签集,search 递归 = search/answer/delegate)
  换 4:val_batch_size 警戒线提示不变(ref+teacher 双前向内存配方同样适用)
其余(KL 断言、守卫①②、装配、收集器)与底本逐字相同。
方法设计出处:Ideation/ideas/RSO_searchqa_data_mapping.md。
"""
import os

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.rso_opsd_ray_trainer import RSOOPSDRayTrainer
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_rso_opsd_search(config)


def run_rso_opsd_search(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = RSOOPSDSearchTaskRunner.remote()
    ray.get(runner.run.remote(config))


def validate_rso_opsd_search_config(config) -> None:
    est = str(config.algorithm.adv_estimator)
    assert est == "rso", f"[RSO+OPSD/search] 优势仍是阶段 1 的 rso,收到 adv_estimator={est!r}"
    assert not bool(config.algorithm.filter_groups.enable), (
        "[RSO+OPSD/search] 不支持 filter_groups(动态采样的多批拼接会破坏收集器的位置回填)")
    assert not bool(config.actor_rollout_ref.actor.get("use_invalid_action_penalty", False)), (
        "[RSO+OPSD/search] 请关闭 actor.use_invalid_action_penalty;无效动作用 +algorithm.rso.invalid_coef 配置")
    assert float(config.actor_rollout_ref.actor.get("entropy_coeff", 0.001)) == 0.0, (
        "[RSO+OPSD/search] 必须 actor.entropy_coeff=0(熵只观测不进梯度)")
    assert bool(config.actor_rollout_ref.actor.get("use_kl_loss", False)), (
        "[RSO+OPSD/search] 必须 actor.use_kl_loss=True(§一点五 底座)")
    kl_coef = float(config.actor_rollout_ref.actor.get("kl_loss_coef", 0.0))
    assert kl_coef == 0.01, f"[RSO+OPSD/search] kl_loss_coef 必须是 0.01,收到 {kl_coef}"
    kl_type = str(config.actor_rollout_ref.actor.get("kl_loss_type", ""))
    assert kl_type == "low_var_kl", f"[RSO+OPSD/search] kl_loss_type 必须是 low_var_kl,收到 {kl_type!r}"
    # ---- 换 1:search 域环境断言
    assert "search_rso" in str(config.env.env_name).lower(), (
        f"[RSO+OPSD/search] 本入口只配 search_rso 环境,收到 env.env_name={config.env.env_name}")
    dp = str(dict(config.env.get("search_rso", {}) or {}).get("decomp_path", ""))
    assert dp and os.path.exists(dp), (
        f"[RSO+OPSD/search] +env.search_rso.decomp_path 缺失或文件不存在:{dp!r}"
        "(examples/data_preprocess/make_searchrso_data_products.py 生成)")
    assert config.actor_rollout_ref.rollout.n == 1, (
        "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. In verl+env, we keep n=1, and achieve GRPO by env.rollout.n")
    opsd = dict(config.algorithm.get("rso_opsd", {}) or {})
    lam = float(opsd.get("lambda_coef", 0.01))
    beta = float(opsd.get("gate_beta", 2.5))
    assert lam > 0, f"[RSO+OPSD/search] lambda_coef 必须大于 0,当前 {lam}"
    assert beta > 0, f"[RSO+OPSD/search] gate_beta 必须大于 0,当前 {beta}"
    tags = list(opsd.get("act_tags", []) or [])
    assert tags == ["search", "answer", "delegate"], (
        f"[RSO+OPSD/search] 必须 +algorithm.rso_opsd.act_tags=[search,answer,delegate]"
        f"(act_mask 标签集,mapping §四),收到 {tags}")
    vbs = int(config.data.val_batch_size)
    if vbs > 200:
        print(f"[RSO+OPSD/search] WARNING data.val_batch_size={vbs} > 200:"
              "ref + teacher 双前向的主机内存配方,注意 val_sub 是 175 行")
    rso_alg = dict(config.algorithm.get("rso", {}) or {})
    print(f"[RSO+OPSD/search] algorithm.rso={rso_alg}  rso_opsd={{lambda_coef:{lam}, "
          f"gate_beta:{beta}, act_tags:{tags}}}")


_KNOWN_RSO_KEYS = {"progress_coef", "progress_clip", "progress_baseline_loo",
                   "invalid_coef", "invalid_gate_min_valid_ratio"}
_KNOWN_OPSD_KEYS = {"gate_beta", "lambda_coef", "prefix_template", "act_tags"}


def validate_rso_opsd_search_wiring(trainer, config) -> None:
    rso_cfg = dict(config.algorithm.get("rso", {}) or {})
    unknown = set(rso_cfg) - _KNOWN_RSO_KEYS
    assert not unknown, f"[RSO+OPSD/search 接线] algorithm.rso 里有未知键 {sorted(unknown)}"
    for k in _KNOWN_RSO_KEYS:
        if k in rso_cfg:
            cfg_v = rso_cfg[k]
            tr_v = trainer.rso_params[k]
            assert type(tr_v)(cfg_v) == tr_v, (
                f"[RSO+OPSD/search 接线] algorithm.rso.{k}={cfg_v} 没有到达 trainer(读到 {tr_v})")
    opsd_cfg = dict(config.algorithm.get("rso_opsd", {}) or {})
    unknown = set(opsd_cfg) - _KNOWN_OPSD_KEYS
    assert not unknown, f"[RSO+OPSD/search 接线] algorithm.rso_opsd 里有未知键 {sorted(unknown)}"
    if "gate_beta" in opsd_cfg:
        assert float(opsd_cfg["gate_beta"]) == trainer.opsd_gate_beta, "[接线] gate_beta 未到达 trainer"
    if "lambda_coef" in opsd_cfg:
        assert float(opsd_cfg["lambda_coef"]) == trainer.opsd_lambda, "[接线] lambda_coef 未到达 trainer"
    if "act_tags" in opsd_cfg:
        assert list(opsd_cfg["act_tags"]) == trainer.opsd_act_tags, "[接线] act_tags 未到达 trainer"
    actor = config.actor_rollout_ref.actor
    assert bool(actor.get("use_rso_opsd_loss", False)) is True, "[接线] actor.use_rso_opsd_loss 未打开"
    assert float(actor.get("rso_opsd_loss_coef", -1)) == trainer.opsd_lambda, "[接线] rso_opsd_loss_coef 不一致"
    assert float(actor.get("rso_opsd_gate_beta", -1)) == trainer.opsd_gate_beta, "[接线] rso_opsd_gate_beta 不一致"
    print("[RSO+OPSD/search 接线] 守卫②通过:rso 组 + rso_opsd 组(含 act_tags)+ actor 树全部对上")


@ray.remote(num_cpus=1)
class RSOOPSDSearchTaskRunner:
    def run(self, config):
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        from omegaconf import open_dict as _open_dict
        _opsd_cfg = config.algorithm.get("rso_opsd", {}) or {}
        with _open_dict(config):
            config.actor_rollout_ref.actor.use_rso_opsd_loss = True
            config.actor_rollout_ref.actor.rso_opsd_loss_coef = float(_opsd_cfg.get("lambda_coef", 0.01))
            config.actor_rollout_ref.actor.rso_opsd_gate_beta = float(_opsd_cfg.get("gate_beta", 2.5))

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        validate_rso_opsd_search_config(config)

        local_path = copy_to_local(config.actor_rollout_ref.model.path,
                                   use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        # ---- 换 2:search 域 OPSD 环境工厂 ----
        from agent_system.environments.env_package.search_rso.opsd_factory import (
            make_rso_opsd_search_envs)
        envs, val_envs = make_rso_opsd_search_envs(config)

        from verl.utils import hf_processor, hf_tokenizer
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge
            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker
            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
        mapping = {Role.ActorRollout: global_pool_id, Role.Critic: global_pool_id}

        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == "episode":
            from agent_system.reward_manager import EpisodeRewardManager
            reward_manager_cls = EpisodeRewardManager
        else:
            raise NotImplementedError
        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from agent_system.multi_turn_rollout.recursive_rollout_loop import RecursiveTrajectoryCollector
        traj_collector = RecursiveTrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = RSOOPSDRayTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
        )
        validate_rso_opsd_search_wiring(trainer, config)
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
