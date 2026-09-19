# TASK — 三 benchmark 实验任务书

**只需要跑没有标记的实验;标了 DONE / ONGOING 的一律不用管、不用跑。**

2026-09-20 版,取代此前全部需求。先 `git pull` 到最新;安装/数据/训练/全量测评命令见 [`README.md`](README.md)。

## 模型(3 个)

| 代号 | `MODEL=` | 说明 |
|---|---|---|
| qwen3-4b | `Qwen/Qwen3-4B-Instruct-2507` | 仓库 pin 的环境直接用 |
| gemma-2b | `google/gemma-4-E2B-it` | 需单独环境:transformers ≥ 5.x + 支持 gemma-4 的近期 vllm |
| gemma-26b | `google/gemma-4-26B-A4B-it` | 需单独环境:transformers ≥ 5.x + 支持 gemma-4 的近期 vllm;建议 TP=4、MICRO_BSZ=1 |

gemma-26b需要2节点*8 H100，共16 GPU训练

## 实验名 = bash 参数

每条 run 的实验名 `NAME=<模型代号>_<bench>_<方法>`,统一这样起(交付文件同名):

```bash
NAME=<实验名> ; MODEL=<模型> OUT=$HOME/rso_runs/$NAME \
  bash examples/rso_8gpu/run_<bench脚本>_<方法>_8gpu.sh trainer.experiment_name=$NAME
```

方法代号 ↔ 脚本:rso_opsd / rso / skill / gtopsd / grpo(README §三 矩阵;下表即优先级序:同时提交,rso_opsd 最优先、grpo 最后;bench 脚本名:textcraft=`synth`,searchqa=`search`,sciworld=`sciworld`)。

### 1. TextCraft(训练 + 测评;在跑的照旧跑完,缺的补齐——第 3 块要用这五个方法的 ckpt)

| 方法 | qwen3-4b | gemma-2b | gemma-26b |
|---|---|---|---|
| rso_opsd | qwen3-4b_textcraft_rso_opsd **(DONE)** | gemma-2b_textcraft_rso_opsd **(DONE)** | gemma-26b_textcraft_rso_opsd **(DONE)** |
| rso | qwen3-4b_textcraft_rso **(DONE)** | gemma-2b_textcraft_rso **(DONE)** | gemma-26b_textcraft_rso **(DONE)** |
| skill | qwen3-4b_textcraft_skill **(DONE)** | gemma-2b_textcraft_skill **(DONE)** | gemma-26b_textcraft_skill **(DONE)** |
| gtopsd | qwen3-4b_textcraft_gtopsd **(DONE)** | gemma-2b_textcraft_gtopsd **(DONE)** | gemma-26b_textcraft_gtopsd **(DONE)** |
| grpo | qwen3-4b_textcraft_grpo **(DONE)** | gemma-2b_textcraft_grpo **(DONE)** | gemma-26b_textcraft_grpo **(DONE)** |

### 2. Search-QA(训练 2wiki+musique;全量测评 2wiki+musique+hotpotqa(OOD),README §四 val_only 配方;先起检索服务)

| 方法 | qwen3-4b | gemma-2b | gemma-26b |
|---|---|---|---|
| rso_opsd | qwen3-4b_searchqa_rso_opsd | gemma-2b_searchqa_rso_opsd **(ONGOING)** | gemma-26b_searchqa_rso_opsd **(ONGOING)** |
| rso | qwen3-4b_searchqa_rso | gemma-2b_searchqa_rso | gemma-26b_searchqa_rso |
| skill | qwen3-4b_searchqa_skill | gemma-2b_searchqa_skill | gemma-26b_searchqa_skill |
| gtopsd | qwen3-4b_searchqa_gtopsd | gemma-2b_searchqa_gtopsd | gemma-26b_searchqa_gtopsd |
| grpo | qwen3-4b_searchqa_grpo | gemma-2b_searchqa_grpo | gemma-26b_searchqa_grpo |

### 3. ScienceWorld-OOD(仅 inference,无训练:第 1 块的 TextCraft ckpt 直接迁移到 sciworld 官方 test 1,819 变体)

```bash
NAME=<实验名> ; MODEL=<对应 textcraft ckpt>/global_step_150/actor/huggingface OUT=$HOME/rso_runs/$NAME \
  bash examples/rso_8gpu/run_sciworld_<方法>_8gpu.sh trainer.experiment_name=$NAME \
  data.val_files=$HOME/data/sciworld/test.parquet data.val_batch_size=128 trainer.val_only=True
```

| 方法 | qwen3-4b | gemma-2b | gemma-26b |
|---|---|---|---|
| rso_opsd | qwen3-4b_sciworld_ood_rso_opsd | gemma-2b_sciworld_ood_rso_opsd **(ONGOING)** | gemma-26b_sciworld_ood_rso_opsd **(ONGOING)** |
| rso | qwen3-4b_sciworld_ood_rso | gemma-2b_sciworld_ood_rso **(ONGOING)** | gemma-26b_sciworld_ood_rso **(ONGOING)** |
| skill | qwen3-4b_sciworld_ood_skill | gemma-2b_sciworld_ood_skill **(ONGOING)** | gemma-26b_sciworld_ood_skill **(ONGOING)** |
| gtopsd | qwen3-4b_sciworld_ood_gtopsd | gemma-2b_sciworld_ood_gtopsd **(ONGOING)** | gemma-26b_sciworld_ood_gtopsd **(ONGOING)** |
| grpo | qwen3-4b_sciworld_ood_grpo | gemma-2b_sciworld_ood_grpo **(ONGOING)** | gemma-26b_sciworld_ood_grpo **(ONGOING)** |

### 4. ScienceWorld(训练 + 测评;java 11+、主机内存 ≥240G;测评 = README §四 val_only,MODEL 换本块 ckpt)

| 方法 | qwen3-4b | gemma-2b | gemma-26b |
|---|---|---|---|
| rso_opsd | qwen3-4b_sciworld_rso_opsd | gemma-2b_sciworld_rso_opsd **(ONGOING)** | gemma-26b_sciworld_rso_opsd **(ONGOING)** |
| rso | qwen3-4b_sciworld_rso | gemma-2b_sciworld_rso | gemma-26b_sciworld_rso |
| skill | qwen3-4b_sciworld_skill | gemma-2b_sciworld_skill | gemma-26b_sciworld_skill |
| gtopsd | qwen3-4b_sciworld_gtopsd | gemma-2b_sciworld_gtopsd | gemma-26b_sciworld_gtopsd |
| grpo | qwen3-4b_sciworld_grpo | gemma-2b_sciworld_grpo | gemma-26b_sciworld_grpo |

前 5 步健康检查(所有训练 run):`episode/valid_action_ratio` > 0.9;递归另看 `rso/delegating_trees` > 0、`rso/G_negative_ratio` ≡ 0;异常即停排查。

## 交付(每条 run 两件,文件名以 $NAME 开头)

**交付 1:训练进展 csv**(第 3 块无训练,免交):

```bash
python scripts_rso/export_train_progress.py --logdir $OUT/tensorboard --out ${NAME}_train_progress.csv
```

**交付 2:全量测评结果**:

- textcraft:`eval_full_val.py` 的 `${NAME}_metrics.json` + 由 `*_cases.jsonl` 制作的 `${NAME}_summary.jsonl`(632 行,字段:task_id, difficulty, success, reward, turns_used, gold_plan_len, max_depth, fail_reason, n_turns_recorded, last_action, input_tokens, output_tokens, total_tokens;递归另带 n_nodes, max_depth, delegated, n_subagents, subagent_success, root_turns, sub_turns);
- searchqa / sciworld / sciworld_ood:val_only run 的完整控制台日志 `${NAME}_eval.log`(含 per-source / per-task 分桶指标)+ 该 run 的 tensorboard 导出 `${NAME}_eval.csv`(同交付 1 脚本)。

全部打成一个 tar.gz。
