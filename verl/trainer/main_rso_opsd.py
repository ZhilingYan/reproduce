# -*- coding: utf-8 -*-
"""RSO + OPSD(节点局部 priv 蒸馏)训练入口。

从 verl/trainer/main_rso.py 克隆(2026-09-10),装配换三样、断言换一组:
  换 1:环境工厂 make_recursive_synth_envs → make_rso_opsd_synth_envs
        (OPSD 适配器带 build_priv,编排器据此打开 priv 行开局现算通道)
  换 2:trainer RayPPOTrainer → RSOOPSDRayTrainer(fit 克隆 + teacher/act_mask 插入块)
  换 3:开头用 open_dict 把 algorithm.rso_opsd.* 搬进 actor 配置树
        (照抄 main_rao_opsd.py:85-90 的手法;dp_actor 只读 actor 树下的键)
  断言:KL 反转——阶段 1 main_rso 的"永久 KL 关"在这里换成 §一点五 的
        use_kl_loss=True / kl_loss_coef=0.01 / low_var_kl;熵仍必须为 0。
        main_rso 自己的断言一字不动,两个入口各守各的(RSO_method_design §2a 入口条款)。
方法设计出处:Ideation/ideas/RSO_method_design.md §二 + §一点五;
实现对照表:docs/RSO_OPSD_DESIGN.md。
"""
import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.rso_opsd_ray_trainer import RSOOPSDRayTrainer
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_rso_opsd(config)


def run_rso_opsd(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = RSOOPSDTaskRunner.remote()
    ray.get(runner.run.remote(config))


def validate_rso_opsd_config(config) -> None:
    """RSO+OPSD 专属断言。与 main_rso.validate_rso_config 逐条对照:
    前四条与阶段 1 相同(estimator/filter_groups/无效动作/熵);KL 两条反转(§一点五);
    末尾加 OPSD 组。不直接复用 validate_rso_config——它断言 KL 关,这里要 KL 开。"""
    est = str(config.algorithm.adv_estimator)
    assert est == "rso", f"[RSO+OPSD] 优势仍是阶段 1 的 rso,收到 adv_estimator={est!r}"
    assert not bool(config.algorithm.filter_groups.enable), (
        "[RSO+OPSD] 不支持 filter_groups(动态采样的多批拼接会破坏收集器的位置回填)")
    assert not bool(config.actor_rollout_ref.actor.get("use_invalid_action_penalty", False)), (
        "[RSO+OPSD] 请关闭 actor.use_invalid_action_penalty;无效动作用 +algorithm.rso.invalid_coef 配置")
    assert float(config.actor_rollout_ref.actor.get("entropy_coeff", 0.001)) == 0.0, (
        "[RSO+OPSD] 必须 actor.entropy_coeff=0(熵只观测不进梯度;两次 RAO 崩塌的教训不豁免)")
    # ---- KL:与 main_rso 相反(RSO_method_design §一点五;例外条款只豁免 main_rso 自己)
    assert bool(config.actor_rollout_ref.actor.get("use_kl_loss", False)), (
        "[RSO+OPSD] 必须 actor.use_kl_loss=True(§一点五 底座;阶段 1 的 KL 关豁免只属于 main_rso)")
    kl_coef = float(config.actor_rollout_ref.actor.get("kl_loss_coef", 0.0))
    assert kl_coef == 0.01, (
        f"[RSO+OPSD] kl_loss_coef 必须是 0.01(§一点五 定值,收到 {kl_coef});"
        "要改值请先改 RSO_method_design.md §一点五 再改这里")
    kl_type = str(config.actor_rollout_ref.actor.get("kl_loss_type", ""))
    assert kl_type == "low_var_kl", (
        f"[RSO+OPSD] kl_loss_type 必须是 low_var_kl(§一点五),收到 {kl_type!r}")
    assert "textcraft_synth" in str(config.env.env_name).lower(), (
        f"[RSO+OPSD] Φ/priv 依赖 textcraft_synth 的配方数据库,收到 env.env_name={config.env.env_name}")
    assert config.actor_rollout_ref.rollout.n == 1, (
        "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. In verl+env, we keep n=1, and achieve GRPO by env.rollout.n")
    # ---- OPSD 组
    opsd = dict(config.algorithm.get("rso_opsd", {}) or {})
    lam = float(opsd.get("lambda_coef", 0.01))
    beta = float(opsd.get("gate_beta", 2.5))
    assert lam > 0, f"[RSO+OPSD] lambda_coef 必须大于 0,否则蒸馏等于没开,当前 {lam}"
    assert beta > 0, f"[RSO+OPSD] gate_beta 必须大于 0(居中门 §2c),当前 {beta}"
    vbs = int(config.data.val_batch_size)
    if vbs > 50:
        print(f"[RSO+OPSD] WARNING data.val_batch_size={vbs} > 50:ref + teacher 双前向的"
              "主机内存配方曾把 rao_opsd 顶到 203G/240G,建议 50(checklist #4)")
    rso_alg = dict(config.algorithm.get("rso", {}) or {})
    print(f"[RSO+OPSD] algorithm.rso={rso_alg}  rso_opsd={{lambda_coef:{lam}, gate_beta:{beta}}}  "
          "(§2c 初始值 λ=0.01 β=2.5;前置诊断后按实测重标,β 先 λ 后)")


# 接线守卫②(checklist #1)认可的全部配置键;多出的键 = 拼写错误,直接拒绝启动
_KNOWN_RSO_KEYS = {"progress_coef", "progress_clip", "progress_baseline_loo",
                   "invalid_coef", "invalid_gate_min_valid_ratio"}
_KNOWN_OPSD_KEYS = {"gate_beta", "lambda_coef", "prefix_template"}


def validate_rso_opsd_wiring(trainer, config) -> None:
    """守卫②:启动前断言 rso_* 全组与 OPSD 组均被 trainer 实际读取
    (config 值 == trainer 属性,而非仅检查 config 键存在),并反向拒绝未知键
    (往 +algorithm.rso.* 注入写错的键名 = 静默用默认值跑,这里把它变成启动失败)。"""
    rso_cfg = dict(config.algorithm.get("rso", {}) or {})
    unknown = set(rso_cfg) - _KNOWN_RSO_KEYS
    assert not unknown, f"[RSO+OPSD 接线] algorithm.rso 里有未知键 {sorted(unknown)}(拼写错误?)"
    for k in _KNOWN_RSO_KEYS:
        if k in rso_cfg:
            cfg_v = rso_cfg[k]
            tr_v = trainer.rso_params[k]
            assert type(tr_v)(cfg_v) == tr_v, (
                f"[RSO+OPSD 接线] algorithm.rso.{k}={cfg_v} 没有到达 trainer(trainer 读到 {tr_v})")
    opsd_cfg = dict(config.algorithm.get("rso_opsd", {}) or {})
    unknown = set(opsd_cfg) - _KNOWN_OPSD_KEYS
    assert not unknown, f"[RSO+OPSD 接线] algorithm.rso_opsd 里有未知键 {sorted(unknown)}(拼写错误?)"
    if "gate_beta" in opsd_cfg:
        assert float(opsd_cfg["gate_beta"]) == trainer.opsd_gate_beta, "[RSO+OPSD 接线] gate_beta 未到达 trainer"
    if "lambda_coef" in opsd_cfg:
        assert float(opsd_cfg["lambda_coef"]) == trainer.opsd_lambda, "[RSO+OPSD 接线] lambda_coef 未到达 trainer"
    # actor 树(dp_actor 只读这里;由本入口 open_dict 搬入)
    actor = config.actor_rollout_ref.actor
    assert bool(actor.get("use_rso_opsd_loss", False)) is True, "[RSO+OPSD 接线] actor.use_rso_opsd_loss 未打开"
    assert float(actor.get("rso_opsd_loss_coef", -1)) == trainer.opsd_lambda, "[RSO+OPSD 接线] actor.rso_opsd_loss_coef 与 trainer 不一致"
    assert float(actor.get("rso_opsd_gate_beta", -1)) == trainer.opsd_gate_beta, "[RSO+OPSD 接线] actor.rso_opsd_gate_beta 与 trainer 不一致"
    print("[RSO+OPSD 接线] 守卫②通过:rso 组 + rso_opsd 组 + actor 树全部对上")


@ray.remote(num_cpus=1)
class RSOOPSDTaskRunner:
    def run(self, config):
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        # 换 3:dp_actor 读的是 actor 配置树下的三个键,bash 往 algorithm.rso_opsd.* 注入,
        # 不搬过去 dp_actor 就读不到(照抄 main_rao_opsd.py:85-90 的教训与手法)。
        # use_rso_opsd_loss 写死 True:跑这个入口就必然有蒸馏损失,想关请改用 main_rso。
        from omegaconf import open_dict as _open_dict
        _opsd_cfg = config.algorithm.get("rso_opsd", {}) or {}
        with _open_dict(config):
            config.actor_rollout_ref.actor.use_rso_opsd_loss = True
            config.actor_rollout_ref.actor.rso_opsd_loss_coef = float(_opsd_cfg.get("lambda_coef", 0.01))
            config.actor_rollout_ref.actor.rso_opsd_gate_beta = float(_opsd_cfg.get("gate_beta", 2.5))

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        validate_rso_opsd_config(config)

        local_path = copy_to_local(config.actor_rollout_ref.model.path,
                                   use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        # ---- 换 1(main_rso.py:82-84):OPSD 环境工厂(适配器带 build_priv)----
        from agent_system.environments.env_package.textcraft_synth.opsd_factory import (
            make_rso_opsd_synth_envs)
        envs, val_envs = make_rso_opsd_synth_envs(config)

        from verl.utils import hf_processor, hf_tokenizer
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        # act_mask 走字节级定位(rso_opsd_core._token_byte_pieces),不依赖 fast tokenizer
        # 的 offset_mapping——re-tokenize 路线已废弃(采样切分≠规范切分,冒烟 21947735)。

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

        # ---- 递归收集器(与 main_rso 相同;priv 列由它按 (slot, step_idx) 回填)----
        from agent_system.multi_turn_rollout.recursive_rollout_loop import RecursiveTrajectoryCollector
        traj_collector = RecursiveTrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # ---- 换 2:RSO+OPSD trainer ----
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
        validate_rso_opsd_wiring(trainer, config)   # 守卫②:开进程之前把接线核死
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
