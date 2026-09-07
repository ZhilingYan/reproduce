# -*- coding: utf-8 -*-
"""RSO(我们自己的递归 RL 方法)训练入口。

从 verl/trainer/main_rao.py 克隆(2026-09-06),装配完全相同(递归环境工厂 +
RecursiveTrajectoryCollector + RayPPOTrainer),只换:
  algorithm.adv_estimator=rso(优势走 rso_core.py:A_out + α·A_prog − 门控 η)
  断言集合(见 validate_rso_config)
方法设计出处:Ideation/ideas/{RSO_method_design, RSO_advantage_design, RSO_progress_shaping,
RSO_open_risks}.md;实现对照表:docs/RSO_DESIGN.md。
消融口径:+algorithm.rso.progress_coef=0 即消融表 #3 的"乙"(root 结局广播、无进展项),
同一入口跑 #3 与 #4。
"""
import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_rso(config)


def run_rso(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = RSOTaskRunner.remote()
    ray.get(runner.run.remote(config))


def validate_rso_config(config) -> None:
    """RSO 专属断言。放在装配之前,报错早于开进程。"""
    est = str(config.algorithm.adv_estimator)
    assert est == "rso", f"[RSO] main_rso 要求 algorithm.adv_estimator=rso,收到 {est!r}"
    assert not bool(config.algorithm.filter_groups.enable), (
        "[RSO] 不支持 filter_groups(动态采样的多批拼接会破坏收集器的位置回填)")
    # 无效动作的处置走 algorithm.rso.invalid_coef(默认 0,带门控,见 rso_core),
    # 不用 flat 那套 actor 级惩罚——那条路对 RSO 无效(不读 token_level_rewards)
    # 且会污染 critic/score/mean 的口径。
    assert not bool(config.actor_rollout_ref.actor.get("use_invalid_action_penalty", False)), (
        "[RSO] 请关闭 actor.use_invalid_action_penalty;无效动作用 +algorithm.rso.invalid_coef 配置")
    # 底座硬前提(RSO_open_risks.md §二):熵不进梯度、无 KL。这两条是 rao_v1/v2
    # 两次崩塌换来的教训,RSO 不允许在带熵奖励的底座上跑。
    assert float(config.actor_rollout_ref.actor.get("entropy_coeff", 0.001)) == 0.0, (
        "[RSO] 必须 actor.entropy_coeff=0(熵只观测不进梯度;依据 RSO_open_risks.md §二)")
    assert not bool(config.actor_rollout_ref.actor.get("use_kl_loss", False)), (
        "[RSO] 必须 actor.use_kl_loss=False(官方口径 kl_ctl=0;依据 RSO_open_risks.md §二)")
    assert "textcraft_synth" in str(config.env.env_name).lower(), (
        f"[RSO] Φ 依赖 textcraft_synth 的配方数据库,收到 env.env_name={config.env.env_name}")
    assert config.actor_rollout_ref.rollout.n == 1, (
        "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. In verl+env, we keep n=1, and achieve GRPO by env.rollout.n")
    rso_alg = dict(config.algorithm.get("rso", {}) or {})
    rso_env = dict(config.env.get("rso", {}) or {})
    print(f"[RSO] algorithm.rso={rso_alg}  env.rso={rso_env}  "
          "(缺省:progress_coef=0.1, progress_clip=3, baseline_loo=True, invalid_coef=0.1 常开, phi_gamma=1)")


@ray.remote(num_cpus=1)
class RSOTaskRunner:
    def run(self, config):
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        validate_rso_config(config)

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
