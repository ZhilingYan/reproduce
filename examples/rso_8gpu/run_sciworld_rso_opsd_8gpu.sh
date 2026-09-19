set -x
# ============================================================================
# RSO / ScienceWorld — recursive RSO+OPSD(节点局部特权 = 运行时 goal_progress)  (8 GPU 版)
# ============================================================================
# 参数 = Ideation/参数对照_textcraft_vs_sciworld.md Recursive 一节(2026-09-19 拍板:
# 方案 A root 60 / child 20 / depth 2、val 温度 0、固定 50 验证 case)。
# 方法设计:Ideation/RSO_sciworld_data_mapping.md(Φ = 官方分峰值推进;
# priv = [目标]/[已达]/[待办]/[提示],零数据预处理;act_tags 缺省 [action],
# delegate 与子 agent 的 `answer: <report>` 都是 <action> 内动作)。
# 依赖与内存需求同 run_sciworld_gtopsd_8gpu.sh 头注释。
#
# 可调环境变量:MODEL / MICRO_BSZ(默认 2)/ TP(默认 2)/ OUT / DATA_DIR。
# ============================================================================
ENGINE=${ENGINE:-vllm}
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
MICRO_BSZ=${MICRO_BSZ:-2}
TP=${TP:-2}
OUT=${OUT:-$HOME/rso_runs/sciworld_rso_opsd}
DATA_DIR=${DATA_DIR:-$HOME/data/sciworld}

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export JAVA_TOOL_OPTIONS="-Xmx512m -XX:ActiveProcessorCount=2 -XX:-UsePerfData"
mkdir -p $OUT $OUT/rollouts $OUT/tree_trace

python3 -m verl.trainer.main_rso_opsd_sciworld \
    algorithm.adv_estimator=rso \
    +algorithm.rso.progress_coef=0.1 \
    +algorithm.rso.progress_clip=3 \
    +algorithm.rso.progress_baseline_loo=True \
    +algorithm.rso.invalid_coef=0.1 \
    +algorithm.rso_opsd.gate_beta=2.5 \
    +algorithm.rso_opsd.lambda_coef=0.01 \
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
    actor_rollout_ref.actor.entropy_coeff=0 \
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
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    algorithm.use_kl_in_reward=False \
    env.env_name=sciworld_rso \
    env.seed=0 \
    env.max_steps=150 \
    env.history_length=2 \
    env.rollout.n=8 \
    +env.rao.root_max_steps=60 \
    +env.rao.per_agent_max_steps=25 \
    +env.rao.max_depth=2 \
    +env.rao.trace_dir=$OUT/tree_trace \
    +env.sciworld.simplification=easy \
    +env.sciworld.env_step_limit=200 \
    +env.sciworld.max_valid_actions=300 \
    trainer.critic_warmup=0 \
    "trainer.logger=[console,tensorboard]" \
    trainer.project_name='sciworld_recursive' \
    trainer.experiment_name=sciworld_rso_opsd_qwen3_4b \
    trainer.n_gpus_per_node=8 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    +trainer.max_actor_ckpt_to_keep=2 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.rollout_data_dir=$OUT/rollouts \
    trainer.default_local_dir=$OUT/ckpts \
    trainer.val_before_train=True $@
