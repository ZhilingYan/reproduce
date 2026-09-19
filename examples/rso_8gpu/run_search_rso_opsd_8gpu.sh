set -x
# ============================================================================
# RSO / Search-QA — RSO+OPSD(递归委托 + 节点局部特权蒸馏)  (8 GPU 版)
# ============================================================================
# 训练集 = MuSiQue + 2Wiki(examples/data_preprocess 三个脚本生成,见 README_SEARCH.md);
# 训练中验证 = val_sub.parquet(musique/2wiki/hotpotqa 三源各 15 条固定 case);检索走本地 e5 服务(SEARCH_URL)。
# 参数口径 = 2026-09-19 审核定稿的参数表(Ideation/flat参数对照_textcraft_vs_searchqa.md):
#   batch 128×组8 / mini 256 / micro 16 / logprob 32 / TP2 / KL 0.01 low_var / 熵 0 /
#   α=0.1 c=3 η=0.1 / λ=0.01 β=2.5 / per_agent 25 / depth 6 / prompt 4096 / response 1024 /
#   全局轮数 60 / 150 步 / test_freq 5 / save_freq 5 + 保留 2 个 ckpt / val_before_train。
# act_mask 用三标签 [search,answer,delegate];Φ/priv 需要 decomp_store.json(DECOMP)。
#
# 可调环境变量:
#   MODEL      模型名(默认 Qwen/Qwen3-4B-Instruct-2507)
#   MICRO_BSZ  actor 更新每卡微批(默认 16)
#   TP         张量并行度(默认 2)
#   OUT        输出根目录
#   DATA_DIR   数据目录(默认 $HOME/data/searchR1_musique_2wiki)
#   SEARCH_URL 检索服务(默认 http://0.0.0.0:8000/retrieve)
#   TRACE      置 1 开启 tree_trace(默认关)
# ============================================================================
ENGINE=${ENGINE:-vllm}
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
MICRO_BSZ=${MICRO_BSZ:-16}
TP=${TP:-2}
OUT=${OUT:-$HOME/rso_runs/search_rso_opsd}
DATA_DIR=${DATA_DIR:-$HOME/data/searchR1_musique_2wiki}
SEARCH_URL=${SEARCH_URL:-http://0.0.0.0:8000/retrieve}
TRACE=${TRACE:-0}
TRACE_ARG=""
[ "$TRACE" = "1" ] && TRACE_ARG="+env.rao.trace_dir=$OUT/tree_trace" && mkdir -p $OUT/tree_trace

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
mkdir -p $OUT/rollouts

python3 -m verl.trainer.main_rso_opsd_search \
    algorithm.adv_estimator=rso \
    +algorithm.rso.progress_coef=0.1 \
    +algorithm.rso.progress_clip=3 \
    +algorithm.rso.progress_baseline_loo=True \
    +algorithm.rso.invalid_coef=0.1 \
    +algorithm.rso_opsd.gate_beta=2.5 \
    +algorithm.rso_opsd.lambda_coef=0.001 \
    "+algorithm.rso_opsd.act_tags=[search,answer,delegate]" \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/val_sub.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=45 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BSZ \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    algorithm.use_kl_in_reward=False \
    env.env_name=search_rso \
    env.seed=0 \
    env.max_steps=60 \
    env.history_length=4 \
    env.rollout.n=8 \
    +env.rao.per_agent_max_steps=25 \
    +env.rao.max_depth=4 \
    +env.search_rso.decomp_path=$DATA_DIR/decomp_store.json \
    env.search.search_url=$SEARCH_URL \
    env.search.topk=3 \
    env.search.timeout=30 \
    trainer.critic_warmup=0 \
    "trainer.logger=[console,tensorboard]" \
    trainer.project_name='rso_search' \
    trainer.experiment_name=search_rso_opsd \
    trainer.n_gpus_per_node=8 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    +trainer.max_actor_ckpt_to_keep=2 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.rollout_data_dir=$OUT/rollouts \
    trainer.default_local_dir=$OUT/ckpts \
    trainer.val_before_train=True $TRACE_ARG $@
