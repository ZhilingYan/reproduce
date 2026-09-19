set -x
# ============================================================================
# RSO / Search-QA — flat GRPO+OPSD(特权 = gt answer)  (8 GPU 版)
# ============================================================================
# 参数 = SDAR 官方 search 脚本原口径(examples/grpo_opsd_trainer/run_search_3b.sh),
# 加三处 2026-09-19 审核定稿的差异(Ideation/flat参数对照_textcraft_vs_searchqa.md):
#   ①数据换 MuSiQue+2Wiki 训练集 + val_sub(三源×15)验证;
#   ②+algorithm.sdar.privileged_source=gt(teacher 的特权 = 本题标准答案,
#     经 search env 的 info['extra.gt_plan'] 通道逐行注入);
#   ③test_freq=5 / save_freq=10 + 保留 2 个 ckpt;推理标签 <thought>(prompts/search.py,
#     Qwen3 词表里 <think> 是特殊 token,模型会无视该推理指令,探针 22191832)。
#
# 可调环境变量:MODEL(默认 Qwen3-4B-Instruct-2507)/ MICRO_BSZ(默认 16)/ TP(默认 1)/
#   OUT / DATA_DIR / SEARCH_URL,同 run_search_rso_opsd_8gpu.sh。
# ============================================================================
ENGINE=${ENGINE:-vllm}
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
MICRO_BSZ=${MICRO_BSZ:-16}
TP=${TP:-1}
OUT=${OUT:-$HOME/rso_runs/search_gtopsd}
DATA_DIR=${DATA_DIR:-$HOME/data/searchR1_musique_2wiki}
SEARCH_URL=${SEARCH_URL:-http://0.0.0.0:8000/retrieve}

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
mkdir -p $OUT

python3 -m verl.trainer.main_sdar \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/val_sub.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=45 \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BSZ \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
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
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    +algorithm.sdar.sdar_coef=0.01 \
    +algorithm.sdar.gate_beta=0.0 \
    +algorithm.sdar.skills_dir=skills/search \
    +algorithm.sdar.skill_all=false \
    +algorithm.sdar.privileged_source=gt \
    env.env_name=search \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n=8 \
    env.history_length=4 \
    env.search.search_url=$SEARCH_URL \
    trainer.critic_warmup=0 \
    "trainer.logger=[console,tensorboard]" \
    trainer.project_name='rso_search' \
    trainer.experiment_name=search_gtopsd \
    trainer.n_gpus_per_node=8 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    +trainer.max_actor_ckpt_to_keep=2 \
    trainer.test_freq=5 \
    trainer.total_training_steps=150 \
    trainer.default_local_dir=$OUT/ckpts \
    trainer.val_before_train=True $@
