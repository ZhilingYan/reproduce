# -*- coding: utf-8 -*-
"""RSO+OPSD 的 trainer:阶段 1 RSO 的优势 + 节点局部 priv 的居中门蒸馏。

克隆底本(checklist #1,RSO_method_design §2a 入口与守卫条款):
  **以阶段 1 RSO 的 trainer 为底**——即 ray_trainer.py 的 RayPPOTrainer(main_rso 用的
  就是它裸的 fit),它的 fit 已经把 rso_* 五参数与 rso_stats 日志装好(ray_trainer.py:
  1512-1522),克隆后只需【加】OPSD 侧;不复用只带 rao_* kwargs 的 RAOOPSDRayTrainer
  (那条路要手工补搬 rso_*,漏一个= 静默用默认值跑,不可见故障)。

fit 与来源 ray_trainer.py:1221-1594 的差异 = 恰好一个插入块([OPSD 差异 1] 标注):
  teacher 前向(build_priv_teacher_batch 拼 node_priv 前缀,rso_opsd_teacher.py)
  + act_mask 数据侧构造(rso_opsd_core.build_action_token_mask)+ δ 的 driver 侧监控。
  插入点在 old_log_prob 之后、ref 之前(对照 rao_opsd_ray_trainer.py:179-181 的先例)。
  蒸馏损失本体在 dp_actor 的 use_rso_opsd_loss 独立分支(compute_rso_opsd_loss)。
守卫测试 tests/test_rso_opsd_guard.py:①fit 与来源逐字 diff 只允许该插入块;
②参数接线断言(config 值 == trainer 属性,见 main_rso_opsd.validate_rso_opsd_wiring)。
以后 ray_trainer 的 fit 改了,这里要跟着改(与 rao_opsd 的克隆债同款,记在案)。
"""

from copy import deepcopy
from pprint import pprint

import numpy as np
import ray
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    _timer,
    apply_invalid_action_penalty,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.rso_opsd_core import build_action_token_mask
from verl.trainer.ppo.rso_opsd_teacher import build_priv_teacher_batch
from verl.utils.metric import reduce_metrics
from verl.utils.torch_functional import masked_mean
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from gigpo import core_gigpo

from agent_system.multi_turn_rollout import adjust_batch


class RSOOPSDRayTrainer(RayPPOTrainer):
    """RSO 优势一个字不改;只加 teacher 前向 + act_mask + 蒸馏列的准备。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ---- 参数接线的"trainer 属性"侧(守卫② 的核对对象)。表达式与 fit 内联读法
        # 同源:rso_* 五条与 ray_trainer.py:1513-1517 逐字相同的取值路径。
        _rso = (self.config.algorithm.get('rso', None) or {})
        self.rso_params = {
            'progress_coef': float(_rso.get('progress_coef', 0.1)),
            'progress_clip': float(_rso.get('progress_clip', 3.0)),
            'progress_baseline_loo': bool(_rso.get('progress_baseline_loo', True)),
            'invalid_coef': float(_rso.get('invalid_coef', 0.1)),
            'invalid_gate_min_valid_ratio': float(_rso.get('invalid_gate_min_valid_ratio', 0.0)),
        }
        _opsd = (self.config.algorithm.get('rso_opsd', None) or {})
        self.opsd_gate_beta = float(_opsd.get('gate_beta', 2.5))     # §2c 初始值
        self.opsd_lambda = float(_opsd.get('lambda_coef', 0.01))     # §2c λ 初始值
        self.opsd_prefix_template = _opsd.get('prefix_template', None)
        print(f"[RSO+OPSD trainer] rso_params={self.rso_params} "
              f"gate_beta={self.opsd_gate_beta} lambda={self.opsd_lambda}")

    # ------------------------------------------------------------ teacher 前向
    def _compute_teacher_log_probs(self, batch: DataProto):
        """teacher = 开了小抄(node_priv)的同一份权重。做法对照 rlsd_ray_trainer.py:532-556:
        同一个 compute_log_prob worker 方法、同一份权重,只是 input_ids 换成
        "priv 前缀 + 节点 prompt + 原封不动的 response"。返回 (log_probs, 截断等指标)。"""
        teacher_batch, teacher_metrics = build_priv_teacher_batch(
            batch=batch,
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.data.max_prompt_length,
            prefix_template=self.opsd_prefix_template,
        )
        teacher_output = self.actor_rollout_wg.compute_log_prob(teacher_batch)
        return teacher_output.batch["old_log_probs"], teacher_metrics

    # ------------------------------------------------------------ fit(克隆)
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        ################ agent-environment loop ###############
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                                                                gen_batch=gen_batch,
                                                                actor_rollout_wg=self.actor_rollout_wg,
                                                                envs=self.envs,
                                                                is_train=True,
                                                                )
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    del batch
                    batch = gen_batch_output

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.GiGPO:
                        step_rewards_tensor = core_gigpo.compute_step_discounted_returns(
                            batch=batch,
                            gamma=self.config.algorithm.gamma
                        )
                        batch.batch['step_rewards'] = step_rewards_tensor

                    batch = adjust_batch(self.config, batch)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    # ---- [OPSD 差异 1] teacher 前向(node_priv 特权前缀)+ act_mask 构造 + δ 监控。
                    # 唯一的插入块。teacher = 开了小抄的同一份权重(rlsd_ray_trainer.py:549 注释);
                    # teacher_log_probs 与 act_mask 作为新列随 batch 流到 dp_actor 的
                    # use_rso_opsd_loss 独立分支(蒸馏损失本体在那里,rso_opsd_core.py);
                    # act_mask 在数据侧构造、与 advantages/responses 同通路(§2a 落地路径 6);
                    # 截断/priv 缺失指标由 build_priv_teacher_batch 一并返回(落地路径 4)。
                    with _timer("teacher_forward", timing_raw):
                        teacher_log_probs, teacher_metrics = self._compute_teacher_log_probs(batch)
                        batch.batch["teacher_log_probs"] = teacher_log_probs
                        metrics.update(teacher_metrics)
                    with _timer("act_mask", timing_raw):
                        act_mask, act_metrics = build_action_token_mask(
                            batch.batch["responses"], batch.batch["response_mask"], self.tokenizer)
                        batch.batch["act_mask"] = act_mask.to(batch.batch["responses"].device)
                        metrics.update(act_metrics)
                    _opsd_delta = (teacher_log_probs - batch.batch["old_log_probs"]) * batch.batch["response_mask"]
                    metrics["rso/teacher_gap_mean_driver"] = masked_mean(
                        _opsd_delta, batch.batch["response_mask"]).item()
                    _act_and_resp = batch.batch["act_mask"] * batch.batch["response_mask"]
                    metrics["rso/teacher_gap_act_driver"] = masked_mean(
                        _opsd_delta, _act_and_resp).item()
                    metrics["rso/gate_active_act_driver"] = masked_mean(
                        (_opsd_delta > 0).float(), _act_and_resp).item()
                    if "rso/useless_goal_rate" in batch.non_tensor_batch:
                        metrics["rso/useless_goal_rate"] = float(batch.non_tensor_batch["rso/useless_goal_rate"][0])
                    # ---- [OPSD 差异 1] 插入块到此为止,以下与来源逐字相同。

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_invalid_action_penalty if available
                        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                            batch, invalid_metrics = apply_invalid_action_penalty(batch,
                                                                                  invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                                                                                  )
                            metrics.update(invalid_metrics)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                            gigpo_mode=self.config.algorithm.gigpo.mode,
                            gigpo_enable_similarity= self.config.algorithm.gigpo.enable_similarity,
                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                            # [RAO 移植 2026-08-24] algorithm.rao 子树可缺省(非 rao 时用不到)。
                            # lam 默认 0 = 官方 TextCraft-Synth 口径(train_areal_synth.py:35 CAP=0.0)
                            rao_lam=float((self.config.algorithm.get('rao', None) or {}).get('lam', 0.0)),
                            rao_leave_one_out=bool((self.config.algorithm.get('rao', None) or {}).get('leave_one_out_baseline', True)),
                            rao_depth_level_weighting=bool((self.config.algorithm.get('rao', None) or {}).get('depth_level_weighting', True)),
                            rao_invalid_action_penalty_coef=(
                                float(self.config.actor_rollout_ref.actor.get('invalid_action_penalty_coef', 0.0))
                                if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', False) else 0.0),
                            rao_invalid_penalty_depth_weighted=bool((self.config.algorithm.get('rao', None) or {}).get('invalid_penalty_depth_weighted', False)),
                            # [RSO 2026-09-06] algorithm.rso 子树,可缺省(非 rso 时用不到)
                            rso_progress_coef=float((self.config.algorithm.get('rso', None) or {}).get('progress_coef', 0.1)),
                            rso_progress_clip=float((self.config.algorithm.get('rso', None) or {}).get('progress_clip', 3.0)),
                            rso_progress_baseline_loo=bool((self.config.algorithm.get('rso', None) or {}).get('progress_baseline_loo', True)),
                            rso_invalid_coef=float((self.config.algorithm.get('rso', None) or {}).get('invalid_coef', 0.1)),
                            rso_invalid_gate=float((self.config.algorithm.get('rso', None) or {}).get('invalid_gate_min_valid_ratio', 0.0)),
                        )
                        if 'rao_stats' in batch.meta_info:          # [RAO 移植] 诊断量进日志
                            metrics.update(batch.meta_info.pop('rao_stats'))
                        if 'rso_stats' in batch.meta_info:          # [RSO] 诊断量进日志
                            metrics.update(batch.meta_info.pop('rso_stats'))

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    test_start_step = self.config.trainer.get("test_start_step", 0)
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or (self.global_steps >= test_start_step and self.global_steps % self.config.trainer.test_freq == 0)):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
