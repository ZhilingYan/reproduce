# -*- coding: utf-8 -*-
"""RAO(Recursive Agent Optimization, arXiv:2605.06639)训练入口。

从 verl/trainer/main_ppo.py 克隆(2026-08-24,以其 :52-257 为准),trainer 仍是 RayPPOTrainer,
只换两处、加四条断言:
  main_ppo.py:96   make_envs(config)               → make_recursive_synth_envs(config)
  main_ppo.py:223  TrajectoryCollector(...)        → RecursiveTrajectoryCollector(...)
  断言:adv_estimator=rao / filter_groups 关 / 无效动作惩罚关 / env 是 textcraft_synth
其余装配代码与 main_ppo.py 逐行相同(去掉了那边的读码笔记注释)。

官方对照:plugins/textcraft/platoon/textcraft/train_scripts/areal/train_areal_synth.py:102-165
  —— 官方也是"同一个 trainer,按 depth_aware 开关换 rollout_fn";我们是"同一个 RayPPOTrainer,
  换环境管理器与收集器"。算法差异完全落在 adv_estimator=rao 那个分支里(ray_trainer.py)。

【不要】像旧实现那样继承 SkillSDRayTrainer:那会把 teacher forward / SDL / SDAR loss 全拖进来,
再靠 bash 传参逐个关闭(审查报告 P8)。本轮只做纯 RAO,与 OPSD 无关。
"""
import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_rao(config)


def run_rao(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = RAOTaskRunner.remote()
    ray.get(runner.run.remote(config))


def validate_rao_config(config) -> None:
    """RAO 专属的配置断言。放在装配之前,报错早于开进程。"""
    est = str(config.algorithm.adv_estimator)
    assert est == "rao", f"[RAO] main_rao 要求 algorithm.adv_estimator=rao,收到 {est!r}"
    assert not bool(config.algorithm.filter_groups.enable), (
        "[RAO] 不支持 algorithm.filter_groups.enable=True(官方 dynamic_sampling: false;"
        "动态采样的多批拼接会破坏收集器的位置回填)")
    # [2026-08-30] 原来这里断言必须关掉无效动作惩罚,理由是"RAO 不读 token_level_rewards,
    # 减在那上面会被静默丢弃"。现在惩罚改成由 rao_core 直接减在行优势上,已经真的生效了,
    # 所以断言改成:打开时必须给出一个大于 0 的系数,免得开了开关却没扣分,又变成一场空。
    if bool(config.actor_rollout_ref.actor.get("use_invalid_action_penalty", False)):
        coef = float(config.actor_rollout_ref.actor.get("invalid_action_penalty_coef", 0.0))
        assert coef > 0, (
            "[RAO] 打开了 use_invalid_action_penalty 就必须给 invalid_action_penalty_coef>0,"
            f"当前是 {coef}")
    assert "textcraft_synth" in str(config.env.env_name).lower(), (
        f"[RAO] 目前只装配了 textcraft_synth 的递归适配器,收到 env.env_name={config.env.env_name}")
    assert config.actor_rollout_ref.rollout.n == 1, (
        "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. In verl+env, we keep n=1, and achieve GRPO by env.rollout.n")
    rao_env = dict(config.env.get("rao", {}) or {})
    rao_alg = dict(config.algorithm.get("rao", {}) or {})
    print(f"[RAO] env.rao={rao_env}  algorithm.rao={rao_alg}  "
          f"(缺省值:per_agent_max_steps=25, max_depth=6, lam=0, leave_one_out=True, depth_level_weighting=True)")


@ray.remote(num_cpus=1)
class RAOTaskRunner:
    def run(self, config):
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        validate_rao_config(config)

        local_path = copy_to_local(config.actor_rollout_ref.model.path,
                                   use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        # ---- 换点 1(main_ppo.py:95-96):递归环境管理器,不改 make_envs ----
        from agent_system.environments.env_package.textcraft_synth.recursive_factory import (
            make_recursive_synth_envs)
        envs, val_envs = make_recursive_synth_envs(config)

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

        # ---- 换点 2(main_ppo.py:222-223):递归收集器 ----
        from agent_system.multi_turn_rollout.recursive_rollout_loop import RecursiveTrajectoryCollector
        traj_collector = RecursiveTrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = RayPPOTrainer(
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
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
