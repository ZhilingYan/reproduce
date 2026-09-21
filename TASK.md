# TASK — 三 benchmark 实验任务书

**只需要跑没有标记的实验;标了 DONE / ONGOING 的一律不用管、不用跑。**

2026-09-20 版,取代此前全部需求。先 `git pull` 到最新;安装/数据/训练/全量测评命令见 [`README.md`](README.md)。

## 模型(3 个)

| 代号 | `MODEL=` | 说明 |
|---|---|---|
| qwen3-4b | `Qwen/Qwen3-4B-Instruct-2507` | 仓库 pin 的环境直接用 |
| gemma-2b | `google/gemma-4-E2B-it` | 需单独环境:transformers ≥ 5.x + 支持 gemma-4 的近期 vllm |
| gemma-26b | `google/gemma-4-26B-A4B-it` | 需单独环境:transformers ≥ 5.x + 支持 gemma-4 的近期 vllm;建议 TP=4、MICRO_BSZ=1 |

记得关掉thinking

gemma-26b需要2节点*8 H100，共16 GPU训练

## 实验名 = bash 参数

每条 run 的实验名 `NAME=<模型代号>_<bench>_<方法>`,统一这样起(交付文件同名):

```bash
NAME=<实验名> ; MODEL=<模型> OUT=$HOME/rso_runs/$NAME \
  bash examples/rso_8gpu/run_<bench脚本>_<方法>_8gpu.sh trainer.experiment_name=$NAME
```

方法代号 ↔ 脚本:rso_opsd / rso / rao / skill / gtopsd / grpo(README §三 矩阵;RAO 为官方递归 baseline,各参数与 RSO 对齐;下表即优先级序:同时提交,rso_opsd 最优先、grpo 最后;bench 脚本名:textcraft=`synth`,searchqa=`search`,sciworld=`sciworld`)。

### 1. TextCraft(训练 + 测评;在跑的照旧跑完,缺的补齐——第 3、4 块要用这五个方法的 ckpt)

| 方法 | qwen3-4b | gemma-2b | gemma-26b |
|---|---|---|---|
| rso | qwen3-4b_textcraft_rso **(DONE)** | gemma-2b_textcraft_rso **(DONE)** | gemma-26b_textcraft_rso **(DONE)** |
| rao | qwen3-4b_textcraft_rao | gemma-2b_textcraft_rao **(DONE)**  | gemma-26b_textcraft_rao **(DONE)**  |
| skill | qwen3-4b_textcraft_skill **(DONE)** | gemma-2b_textcraft_skill **(DONE)** | gemma-26b_textcraft_skill **(DONE)** |
| gtopsd | qwen3-4b_textcraft_gtopsd **(DONE)** | gemma-2b_textcraft_gtopsd **(DONE)** | gemma-26b_textcraft_gtopsd **(DONE)** |
| grpo | qwen3-4b_textcraft_grpo **(DONE)** | gemma-2b_textcraft_grpo **(DONE)** | gemma-26b_textcraft_grpo **(DONE)** |

### 2. Search-QA(训练 2wiki+musique;全量测评 2wiki+musique+hotpotqa(OOD),README §四 val_only 配方;先起检索服务)

| 方法 | qwen3-4b | gemma-2b | gemma-26b |
|---|---|---|---|
| rso | qwen3-4b_searchqa_rso | gemma-2b_searchqa_rso **(ONGOING)** | gemma-26b_searchqa_rso **(ONGOING)** |
| rao | qwen3-4b_searchqa_rao | gemma-2b_searchqa_rao | gemma-26b_searchqa_rao |
| skill | qwen3-4b_searchqa_skill | gemma-2b_searchqa_skill | gemma-26b_searchqa_skill |
| gtopsd | qwen3-4b_searchqa_gtopsd | gemma-2b_searchqa_gtopsd | gemma-26b_searchqa_gtopsd |
| grpo | qwen3-4b_searchqa_grpo | gemma-2b_searchqa_grpo | gemma-26b_searchqa_grpo |

### 3. ScienceWorld-OOD(仅 inference,无训练:第 1 块的 TextCraft ckpt 直接迁移到 sciworld 官方 test 1,819 变体)

```bash
# textcraft ckpt 一律用 val 最优:<ckpt>/best/actor/huggingface(旧策略产物无 best 软链时用 global_step_150)
NAME=<实验名> BENCH=sciworld MODEL=<对应 textcraft ckpt>/best/actor/huggingface OUT=$HOME/rso_runs/$NAME \
SCRIPT=examples/rso_8gpu/run_sciworld_<方法>_8gpu.sh \
  bash scripts_rso/run_full_eval.sh data.val_files=$HOME/data/sciworld/test.parquet data.val_batch_size=128
```

| 方法 | qwen3-4b | gemma-2b | gemma-26b |
|---|---|---|---|
| rso | qwen3-4b_sciworld_ood_rso | gemma-2b_sciworld_ood_rso **(ONGOING)** | gemma-26b_sciworld_ood_rso **(ONGOING)** |
| rao | qwen3-4b_sciworld_ood_rao | gemma-2b_sciworld_ood_rao **(ONGOING)**  | gemma-26b_sciworld_ood_rao **(ONGOING)**  |
| skill | qwen3-4b_sciworld_ood_skill | gemma-2b_sciworld_ood_skill **(ONGOING)** | gemma-26b_sciworld_ood_skill **(ONGOING)** |
| gtopsd | qwen3-4b_sciworld_ood_gtopsd | gemma-2b_sciworld_ood_gtopsd **(ONGOING)** | gemma-26b_sciworld_ood_gtopsd **(ONGOING)** |
| grpo | qwen3-4b_sciworld_ood_grpo | gemma-2b_sciworld_ood_grpo **(ONGOING)** | gemma-26b_sciworld_ood_grpo **(ONGOING)** |

### 4. TextCraft-OOD(仅 inference,无训练:第 1 块的 TextCraft ckpt 直接迁移到真 TextCraft 官方 test 84 题)

```bash
# textcraft ckpt 一律用 val 最优:<ckpt>/best/actor/huggingface(旧策略产物无 best 软链时用 global_step_150)
# 脚本按方法族选:递归 rso / rao → run_textcraft_ood_rso_8gpu.sh;flat skill / gtopsd / grpo → run_textcraft_ood_grpo_8gpu.sh
NAME=<实验名> BENCH=textcraft_ood MODEL=<对应 textcraft ckpt>/best/actor/huggingface OUT=$HOME/rso_runs/$NAME \
SCRIPT=examples/rso_8gpu/run_textcraft_ood_<rso|grpo>_8gpu.sh \
  bash scripts_rso/run_full_eval.sh data.val_files=$HOME/data/textcraft_ood/test.parquet data.val_batch_size=84
```

| 方法 | qwen3-4b | gemma-2b | gemma-26b |
|---|---|---|---|
| rso | qwen3-4b_textcraft_ood_rso | gemma-2b_textcraft_ood_rso **(ONGOING)** | gemma-26b_textcraft_ood_rso **(ONGOING)** |
| rao | qwen3-4b_textcraft_ood_rao | gemma-2b_textcraft_ood_rao **(ONGOING)**  | gemma-26b_textcraft_ood_rao **(ONGOING)**  |
| skill | qwen3-4b_textcraft_ood_skill | gemma-2b_textcraft_ood_skill **(ONGOING)** | gemma-26b_textcraft_ood_skill **(ONGOING)** |
| gtopsd | qwen3-4b_textcraft_ood_gtopsd | gemma-2b_textcraft_ood_gtopsd **(ONGOING)** | gemma-26b_textcraft_ood_gtopsd **(ONGOING)** |
| grpo | qwen3-4b_textcraft_ood_grpo | gemma-2b_textcraft_ood_grpo **(ONGOING)** | gemma-26b_textcraft_ood_grpo **(ONGOING)** |

### 5. (PENDING) ScienceWorld(训练 + 测评;java 11+、主机内存 ≥240G;测评 = README §四 val_only,MODEL 换本块 ckpt)

| 方法 | qwen3-4b | gemma-2b | gemma-26b |
|---|---|---|---|
| rso | qwen3-4b_sciworld_rso | gemma-2b_sciworld_rso **(ONGOING)** | gemma-26b_sciworld_rso **(ONGOING)** |
| rao | qwen3-4b_sciworld_rao | gemma-2b_sciworld_rao | gemma-26b_sciworld_rao |
| skill | qwen3-4b_sciworld_skill | gemma-2b_sciworld_skill | gemma-26b_sciworld_skill |
| gtopsd | qwen3-4b_sciworld_gtopsd | gemma-2b_sciworld_gtopsd | gemma-26b_sciworld_gtopsd |
| grpo | qwen3-4b_sciworld_grpo | gemma-2b_sciworld_grpo | gemma-26b_sciworld_grpo |

ckpt 策略(所有训练 run 统一,脚本已配好):每 5 步 val + 存最新 ckpt(保 1 份)+ 自动保留 val 最优一份(`$OUT/ckpts/best`);全量测评一律用 best。

前 5 步健康检查(所有训练 run):`episode/valid_action_ratio` > 0.9;递归另看 `rso/delegating_trees` > 0、`rso/G_negative_ratio` ≡ 0;异常即停排查。

## 全量测评(所有块统一用 `scripts_rso/run_full_eval.sh`)

训练完毕后一律用 val 最优 ckpt(`$OUT/ckpts/best`)跑全量测评。该脚本执行的就是 README 第四节的现有
全量测评命令(题目、解码设置都不变),另外逐题落盘,并产出交付用的 `${NAME}_cases.jsonl`(写在当前目录)。
第 3、4 块的命令见各块;训练块(1、2、5)如下,`<方法>` 与训练时相同:

```bash
MODEL=$HOME/rso_runs/$NAME/ckpts/best/actor/huggingface
# 1. TextCraft(eval_full_val.py 原参数;递归方法另加 --recursive --per-agent-steps 25 --max-depth 6 --max-steps 200)
NAME=<实验名> BENCH=textcraft_synth MODEL=$MODEL OUT=$HOME/rso_runs/$NAME \
  bash scripts_rso/run_full_eval.sh --split val --tp 2
# 2. Search-QA(先起检索服务)
NAME=<实验名> BENCH=searchqa MODEL=$MODEL OUT=$HOME/rso_runs/$NAME SCRIPT=examples/rso_8gpu/run_search_<方法>_8gpu.sh \
  bash scripts_rso/run_full_eval.sh data.val_files=$HOME/data/searchR1_processed_direct/test_3src.parquet data.val_batch_size=512
# 5. ScienceWorld
NAME=<实验名> BENCH=sciworld MODEL=$MODEL OUT=$HOME/rso_runs/$NAME SCRIPT=examples/rso_8gpu/run_sciworld_<方法>_8gpu.sh \
  bash scripts_rso/run_full_eval.sh data.val_files=$HOME/data/sciworld/test.parquet data.val_batch_size=128
```

已经用旧命令跑完全量测评、但还没有 `${NAME}_cases.jsonl` 的实验,用 best ckpt 按上面的命令重跑一次即可(只推理,不训练)。

## 交付(文件名以 $NAME 或 bench 名开头)

**交付 1:训练进展 csv**(第 3、4 块无训练,免交):

```bash
python scripts_rso/export_train_progress.py --logdir $OUT/tensorboard --out ${NAME}_train_progress.csv
```

**交付 2:全量测评结果**(指标定义见 `Ideation/ideas/RSO_evaluation_metrics.md`):

| 粒度 | 文件 | 内容 | 产出方式 |
|---|---|---|---|
| 每个实验 | `${NAME}_cases.jsonl` | 逐题结果:每题一行,含该 benchmark 的全部逐题指标(字段见下表) | `run_full_eval.sh` 自动产出 |
| 每个 benchmark | `<bench>_report.md` | 每个模型一组对照表(行 = 方法,列 = 汇总指标)+ 分桶表 + 事实摘要 + 总结分析 | `make_eval_report.py` 生成表格与摘要;末尾"分析"一节由执行者补写 |
| 全部 | `rso_deliverables.tar.gz` | 以上全部 + 交付 1 | 见下方命令 |

`${NAME}_cases.jsonl` 的逐题字段:

| benchmark | 字段 |
|---|---|
| 全部共有 | `name` `bench` `subset` `case_id` `success` `steps_total` `gen_tokens` `max_node_ctx` `is_recursive` `delegated` `n_delegations`;递归方法另有 `steps_root` `n_nodes` `tree_depth` |
| textcraft | `difficulty` |
| textcraft_ood | `goal_depth` `env_actions` `invalid_actions` `format_invalid_actions` |
| sciworld_ood / sciworld | `task` `variation` `score`(0-100,负分裁零) |
| searchqa | `data_source` `question_type` `hops` `em` `f1` `subem` `answer` `golds` `n_searches`(只含 musique / 2wikimultihopqa / hotpotqa) |

汇总与打包(收齐全部实验的逐题文件后执行一次):

```bash
mkdir -p deliver && cp *_cases.jsonl *_train_progress.csv deliver/
python scripts_rso/make_eval_report.py --cases-dir deliver --out-dir deliver   # 每个 bench 一份 <bench>_report.md
# 打开每份 deliver/<bench>_report.md,在末尾"分析(执行者补写)"一节写总结分析,每条结论引用表中数字
tar czf rso_deliverables.tar.gz deliver/
```
