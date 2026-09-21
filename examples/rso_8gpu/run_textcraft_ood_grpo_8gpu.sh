# textcraft_ood:textcraft 训练出的 ckpt 在真 TextCraft(ADaPT 官方环境,Minecraft 1.16.5 配方)
# 上的 OOD 推理,只做 trainer.val_only 推理、不训练;与第 1 块的 textcraft(= textcraft_synth 训练)
# 是不同实验,实验名用 <model>_textcraft_ood_<method>,勿与 <model>_textcraft_<method> 混用。
# 任务集:官方式 test 划分 84 题(合成深度 >=2 的物品每 5 个抽 1,附 <=10 条干扰配方)。
# 前置:bash examples/data_preprocess/fetch_textcraft_recipes.sh(配方,sha256 校验)
#       python examples/data_preprocess/make_textcraft_ood_parquet.py(84 行 dummy parquet)
# prompt:env_name=textcraft_ood 自动启用 synth 同构派生的 OOD 模板(prompts/textcraft_ood.py)。
# flat 推理(服务所有 flat 方法的 ckpt 与基座:MODEL=<ckpt>/best/actor/huggingface)。
set -x
# [口径 2026-09-11] 训练 env.max_steps=100、只训 medium、训练中 val=val100 的 easy+medium 50 题;
#   训练完毕后的全量测评 max_steps=500(eval_full_val.py 默认)。其余参数不变。
# ============================================================================
# RSO / TextCraft-Synth — baseline 1/3: 纯 GRPO   (8 GPU 版, Qwen3.5-4B / 9B)
# ============================================================================
# 与 4 卡版(examples/grpo_trainer/run_textcraft_synth_full_grpo.sh)的唯一差异:
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
OUT=${OUT:-$HOME/rso_runs/textcraft_ood_grpo}

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
mkdir -p $OUT/rollouts

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/textcraft_ood/test.parquet \
    data.val_files=$HOME/data/textcraft_ood/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=84 \
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
    "actor_rollout_ref.actor.checkpoint.contents=[model,optimizer,extra,hf_model]" \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_batched_tokens=12288 \
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
    env.env_name=textcraft_ood \
    env.seed=0 \
    env.max_steps=100 \
    env.history_length=2 \
    env.rollout.n=8 \
    env.resources_per_worker.num_cpus=0.04 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='rso_textcraft_ood' \
    trainer.experiment_name='textcraft_ood_grpo_8gpu' \
    trainer.default_local_dir=$OUT/ckpts \
    trainer.n_gpus_per_node=8 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    +trainer.max_actor_ckpt_to_keep=1 \
    +trainer.best_ckpt_metric=val/success_rate \
    trainer.rollout_data_dir=$OUT/rollouts \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $@
