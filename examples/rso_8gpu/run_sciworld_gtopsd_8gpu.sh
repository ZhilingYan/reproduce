set -x
# ============================================================================
# RSO / ScienceWorld — flat GRPO+OPSD(特权 = subgoal 骨架 gt_plan)  (8 GPU 版)
# ============================================================================
# 参数 = Ideation/参数对照_textcraft_vs_sciworld.md(2026-09-19 拍板:方案 A、
# val 温度 0、固定 50 验证 case = 30 任务×dev[0] + 官方任务序前 20×dev[1])。
# 依赖(TextCraft 环境之外新增):
#   pip install scienceworld==1.3.0     # py4j + 内置 jar;需要 java 11+ 在 PATH
#   python examples/data_preprocess/make_sciworld_dummy_parquet.py   # 假 parquet(批次驱动)
# 任务由环境自采(官方 train 划分任务均匀采样),parquet 不含真实数据。
# 主机内存需求 ≥ 240G(实测:训练本体 ~130G@batch8 冒烟,正式 batch 128 + 178 JVM;
# JVM 每个 ~140MB,已限 -Xmx512m 并关 UsePerfData——perf 警告会污染 py4j 端口解析)。
#
# 可调环境变量:MODEL / MICRO_BSZ(默认 2)/ TP(默认 2)/ OUT / DATA_DIR。
# ============================================================================
ENGINE=${ENGINE:-vllm}
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
MICRO_BSZ=${MICRO_BSZ:-2}
TP=${TP:-2}
OUT=${OUT:-$HOME/rso_runs/sciworld_gtopsd}
DATA_DIR=${DATA_DIR:-$HOME/data/sciworld}

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export JAVA_TOOL_OPTIONS="-Xmx512m -XX:ActiveProcessorCount=2 -XX:-UsePerfData"
mkdir -p $OUT

python3 -m verl.trainer.main_sdar \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/val.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=50 \
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
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_batched_tokens=12288 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    "+actor_rollout_ref.rollout.stop='</action>'" \
    +actor_rollout_ref.rollout.include_stop_str_in_output=True \
    +actor_rollout_ref.rollout.detokenize=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    +algorithm.sdar.sdar_coef=0.01 \
    +algorithm.sdar.gate_beta=0.0 \
    +algorithm.sdar.skills_dir=skills/sciworld \
    +algorithm.sdar.skill_all=false \
    +algorithm.sdar.privileged_source=gt \
    env.env_name=sciworld \
    env.seed=0 \
    env.max_steps=60 \
    env.rollout.n=8 \
    env.history_length=2 \
    +env.sciworld.simplification=easy \
    +env.sciworld.env_step_limit=200 \
    +env.sciworld.max_valid_actions=300 \
    trainer.critic_warmup=0 \
    "trainer.logger=[console,tensorboard]" \
    trainer.project_name='sciworld_flat' \
    trainer.experiment_name=sciworld_gtopsd_qwen3_4b \
    trainer.n_gpus_per_node=8 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    +trainer.max_actor_ckpt_to_keep=2 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.default_local_dir=$OUT/ckpts \
    trainer.val_before_train=True $@
