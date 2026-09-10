set -x
# ============================================================================
# RSO / TextCraft-Synth — RSO+OPSD(RSO 优势 + 节点局部 priv 的居中门蒸馏)  (8 GPU 版)
# ============================================================================
# 与 run_synth_rso_8gpu.sh 的关系:同一环境、同一数据、同一批量、同一 RSO 优势;
# 只多一路"开小抄的 teacher":每步训练里同一份权重再前向一次,prompt 前拼上
# 该行【节点自己】的特权工作计划(priv,按当时库存与该节点私有配方笔记本现算),
# 蒸馏损失 = act_mask 内 token 的居中门加权 KL(g=max(tanh(βδ),0),β=2.5,λ=0.01),
# 加在 PPO 损失上。底座差异一处:KL 开(use_kl_loss=True, coef=0.01, low_var_kl)——
# 这是 OPSD 系的设定;纯 RSO 保持 KL 关。
#
# 可调环境变量(与 run_synth_rso_8gpu.sh 相同):
#   MODEL      模型名。Qwen 与 Gemma 均可(Gemma 需升级 transformers/vllm,见 README)。
#   MICRO_BSZ  actor 更新的每卡微批(默认 1)。
#   TP         张量并行度(默认 2)。
#   OUT        输出根目录(ckpt/rollouts/tensorboard/tree_trace)
#   TRACE      置 1 开启 tree_trace(约 0.5GB/训练步;默认关)
# ============================================================================
ENGINE=${ENGINE:-vllm}
MODEL=${MODEL:-Qwen/Qwen3.5-4B}          # 9B 用 Qwen/Qwen3.5-9B + TP=4, 见 README 第四节
MICRO_BSZ=${MICRO_BSZ:-1}
TP=${TP:-2}
OUT=${OUT:-$HOME/rso_runs/rso_opsd}
TRACE=${TRACE:-0}
TRACE_ARG=""
[ "$TRACE" = "1" ] && TRACE_ARG="+env.rao.trace_dir=$OUT/tree_trace" && mkdir -p $OUT/tree_trace

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
mkdir -p $OUT/rollouts

python3 -m verl.trainer.main_rso_opsd \
    algorithm.adv_estimator=rso \
    +algorithm.rso.progress_coef=0.1 \
    +algorithm.rso.progress_clip=3 \
    +algorithm.rso.progress_baseline_loo=True \
    +algorithm.rso.invalid_coef=0.1 \
    +algorithm.rso_opsd.gate_beta=2.5 \
    +algorithm.rso_opsd.lambda_coef=0.01 \
    +env.rso.phi_gamma=1.0 \
    data.train_files=$HOME/data/verl-agent/synth_full/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/synth_full/text/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=50 \
    data.max_prompt_length=8192 \
    data.max_response_length=1024 \
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
    actor_rollout_ref.actor.entropy_coeff=0 \
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
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    algorithm.use_kl_in_reward=False \
    env.env_name=textcraft_synth \
    env.seed=0 \
    env.max_steps=200 \
    +env.rao.per_agent_max_steps=25 \
    +env.rao.max_depth=6 \
    +env.rao.state_block_scope=node \
    env.history_length=2 \
    env.rollout.n=8 \
    "env.textcraft_synth.train_difficulties=[easy,medium]" \
    "env.textcraft_synth.val_difficulties=[easy,medium,hard,extreme]" \
    env.textcraft_synth.val_split=val100 \
    env.resources_per_worker.num_cpus=0.04 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='rso_textcraft_synth' \
    trainer.experiment_name='synth_rso_opsd_8gpu' \
    trainer.default_local_dir=$OUT/ckpts \
    trainer.n_gpus_per_node=8 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    +trainer.max_actor_ckpt_to_keep=2 \
    trainer.rollout_data_dir=$OUT/rollouts \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $TRACE_ARG $@
# [RSO+OPSD] 相对 run_synth_rso_8gpu.sh 的差异清单(其余逐字相同):
#   main_rso_opsd(替代 main_rso);+algorithm.rso_opsd.gate_beta=2.5 / lambda_coef=0.01 两行;
#   KL 三行:use_kl_loss=True + kl_loss_coef=0.01 + kl_loss_type=low_var_kl(替代 use_kl_loss=False 一行);
#   data.val_batch_size 100→50(ref + teacher 双前向的主机内存配方,4 卡 240G 节点曾顶到 203G;
#     8 卡大内存节点可自行调回 100);OUT 默认 rso_opsd;experiment_name。
# 监控要点:rso/gate_mean、rso/gate_active_ratio(δ>0 占比)、rso/opsd_loss(恒≥0,
#   >0 即 teacher 有信息差)、rso/teacher_gap_act_driver、rso/priv_truncation_rate_{hard,extreme}
#   (持续非零需压缩 priv 渲染)、rso/useless_goal_rate(应随训练下降)。
