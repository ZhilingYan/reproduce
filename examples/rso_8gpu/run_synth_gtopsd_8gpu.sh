set -x
# ============================================================================
# RSO / TextCraft-Synth — baseline 2/3: GRPO+OPSD(GT特权) (8 GPU 版, Qwen3.5-4B / 9B)
# ============================================================================
# 与 4 卡版(examples/grpo_opsd_trainer/run_textcraft_synth_full_gtopsd.sh)的唯一差异:
#   n_gpus_per_node 4 -> 8。批量/算法/环境参数全部保持一致,保证结果可比。
#
# 可调环境变量(不改脚本即可切换):
#   MODEL      模型名。默认 Qwen/Qwen3.5-4B;9B 用 Qwen/Qwen3.5-9B(建议 TP=4, MICRO_BSZ=1)
#   MICRO_BSZ  actor 更新的每卡微批(默认 2)。40GB 卡用 1-2;80GB 卡可试 4。
#              显存不够就调小,只影响梯度累积粒度,数学等价。
#   TP         张量并行度(默认 2)。4B 模型 2 足够;9B+ 可设 4。
#   OUT        输出根目录(ckpt/rollouts/tensorboard)
# ============================================================================
ENGINE=${ENGINE:-vllm}
MODEL=${MODEL:-Qwen/Qwen3.5-4B}          # 9B 用 Qwen/Qwen3.5-9B + TP=4, 见 README 第四节
MICRO_BSZ=${MICRO_BSZ:-2}
TP=${TP:-2}
OUT=${OUT:-$HOME/rso_runs/gtopsd}

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
mkdir -p $OUT/rollouts

python3 -m verl.trainer.main_sdar \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/synth_full/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/synth_full/text/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=100 \
    data.max_prompt_length=8192 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BSZ \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    +algorithm.sdar.sdar_coef=0.01 \
    +algorithm.sdar.gate_beta=0.0 \
    +algorithm.sdar.skills_dir=skills/textcraft_synth \
    +algorithm.sdar.skill_all=false \
    +algorithm.sdar.privileged_source=gt \
    env.env_name=textcraft_synth \
    env.seed=0 \
    env.max_steps=100 \
    env.history_length=2 \
    env.rollout.n=8 \
    "env.textcraft_synth.train_difficulties=[easy,medium]" \
    "env.textcraft_synth.val_difficulties=[easy,medium,hard,extreme]" \
    env.textcraft_synth.val_split=val100 \
    env.resources_per_worker.num_cpus=0.04 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='rso_textcraft_synth' \
    trainer.experiment_name='synth_gtopsd_8gpu' \
    trainer.default_local_dir=$OUT/ckpts \
    trainer.n_gpus_per_node=8 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    +trainer.max_actor_ckpt_to_keep=2 \
    trainer.rollout_data_dir=$OUT/rollouts \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $@
