set -x
# ============================================================================
# RSO / TextCraft-Synth — RAO(递归 agent,arXiv:2605.06639 复刻)   (8 GPU 版)
# ============================================================================
# 与三条 flat baseline 的关系:同一环境、同一数据、同一批量;差异只有两组:
#   [算法] main_rao + adv_estimator=rao + +algorithm.rao.*(见下)
#   [底座] 递归方法按 RAO 官方口径:entropy_coeff=0(熵只记日志不进梯度)、
#          use_kl_loss=False(官方 kl_ctl=0)、max_response_length=1024(官方给 8192,
#          1024 为成本折中)、无 invalid_action_penalty(官方无此惩罚)。
#   flat baseline 保持 SDAR 原口径不变,两套口径的差异在论文中作为方法自带设定声明。
# 递归内核参数(env.rao.*):每 agent 25 步预算、树深上限 6 —— 与 RAO 官方
# TextCraft-Synth 配置逐项一致(synth_rollout.py:83,89;yaml rollout_config.max_steps: 25)。
# env.max_steps=200 是 lockstep 实现的整树全局轮数上限(官方 asyncio 无此量,纯安全墙)。
#
# 可调环境变量(不改脚本即可切换):
#   MODEL      模型名。Qwen 与 Gemma 均可,如 Qwen/Qwen3.5-4B / google/gemma-4-E4B-it。
#              gemma-4-E4B-it 需要升级 transformers/vllm(实测 pin 版加载不了),见 README 四点七的 Gemma 小节。
#   MICRO_BSZ  actor 更新的每卡微批(默认 1)。响应长度 1024 比 flat 的 512 长一倍,
#              40GB 卡建议 1;80GB 卡可试 2-4。
#   TP         张量并行度(默认 2)。
#   OUT        输出根目录(ckpt/rollouts/tensorboard/tree_trace)
#   TRACE      置 1 开启 tree_trace(每棵树全部 prompt/输出/观测/委派关系落盘 jsonl,
#              约 0.5GB/训练步,调试与分析用;默认关)
# ============================================================================
ENGINE=${ENGINE:-vllm}
MODEL=${MODEL:-Qwen/Qwen3.5-4B}          # 9B 用 Qwen/Qwen3.5-9B + TP=4, 见 README 第四节
MICRO_BSZ=${MICRO_BSZ:-1}
TP=${TP:-2}
OUT=${OUT:-$HOME/rso_runs/rao}
TRACE=${TRACE:-0}
TRACE_ARG=""
[ "$TRACE" = "1" ] && TRACE_ARG="+env.rao.trace_dir=$OUT/tree_trace" && mkdir -p $OUT/tree_trace

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
mkdir -p $OUT/rollouts

python3 -m verl.trainer.main_rao \
    algorithm.adv_estimator=rao \
    +algorithm.rao.lam=0.0 \
    +algorithm.rao.leave_one_out_baseline=True \
    +algorithm.rao.depth_level_weighting=True \
    data.train_files=$HOME/data/verl-agent/synth_full/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/synth_full/text/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=100 \
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
    actor_rollout_ref.actor.use_kl_loss=False \
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
    trainer.experiment_name='synth_rao_8gpu' \
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
