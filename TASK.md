# TASK — 三 benchmark 实验任务书

**只需要跑标 TODO 的实验;标 DONE 的格子已附成绩,一律不用再跑。**

2026-10-10 版,取代此前全部需求(qwen 全线移除,只保留两个 gemma;TextCraft 已完成并回填成绩)。
先 `git pull` 到最新;安装/数据/训练/全量测评命令见 [`README.md`](README.md)。
成绩均为该实验 val 最优 checkpoint 的全量测评值(2026-09-24 汇入核验版);`runN / stepK` 为训练来源。

## 模型(2 个)

| 代号 | `MODEL=` | 说明 |
|---|---|---|
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

方法代号 ↔ 脚本:rso / rao / skill / gtopsd / grpo(README §三 矩阵;RAO 为官方递归 baseline,
各参数与 RSO 对齐;待跑方法按 rao → skill → gtopsd → grpo 的顺序优先;
bench 脚本名:textcraft=`synth`,searchqa=`search`,sciworld=`sciworld`)。

### 1. TextCraft(训练 + 测评)— 已完成(val 632 题,成功率 %)

| 方法 | gemma-2b | gemma-26b |
|---|---|---|
| rso | **DONE** 29.91(189/632,run178/step30) | **DONE** 42.88(271/632,run210/step10) |
| rao | **DONE** 16.14(102/632,run177/step30) | **DONE** 43.35(run226/step5) |
| skill | **DONE** 16.14(102/632,run135/step60) | **DONE** 37.66(238/632,run161/step20) |
| gtopsd | **DONE** 16.46(104/632,run134/step20) | **DONE** 38.13(241/632,run153/step20) |
| grpo | **DONE** 17.25(109/632,run133/step10) | **DONE** 37.97(240/632,run159/step20) |



#### 1a. (TODO) 26B rso 多 checkpoint 全量 val 选优

对 26B rso(run210)磁盘上尚存的每一个 checkpoint 跑 632 题全量 val(口径与上表 rso 成绩完全一致),
选最高者作为本块 rso 的正式成绩与第 3、4 块迁移用 ckpt:

```bash
NAME=gemma-26b_textcraft_rso_step<K> BENCH=textcraft_synth MODEL=<ckpt_stepK>/actor/huggingface \
OUT=$HOME/rso_runs/$NAME bash scripts_rso/run_full_eval.sh --split val --tp 2 \
  --recursive --per-agent-steps 25 --max-depth 6 --max-steps 200
```

交付:每 ckpt 一份 `${NAME}_cases.jsonl` + 一张 step × 成功率(含四档分档)汇总表。
若旧 ckpt 已被滚动删除,如实报告剩余步数,不要重训。

### 2. Search-QA(训练 2wiki+musique;全量测评 2wiki+musique+hotpotqa(OOD),README §四 val_only 配方;先起检索服务;test 22,398 题,成功率 %)
只跑50step即可

| 方法 | gemma-2b | gemma-26b |
|---|---|---|
| rso | **DONE** 34.05(7,626/22,398,run245/step30) | **DONE** 35.47(7,945/22,398,run247/step5) |
| rao | **TODO** gemma-2b_searchqa_rao | **TODO** gemma-26b_searchqa_rao |
| skill | **TODO** gemma-2b_searchqa_skill | **TODO** gemma-26b_searchqa_skill |
| gtopsd | **TODO** gemma-2b_searchqa_gtopsd | **TODO** gemma-26b_searchqa_gtopsd |
| grpo | **TODO** gemma-2b_searchqa_grpo | **TODO** gemma-26b_searchqa_grpo |

> rso 分项:E2B = 2wiki 36.17 / musique 17.05 / hotpotqa 35.99;26B = 2wiki 34.91 / musique 18.58 /
> hotpotqa 41.94。26B 另有 step50 追加评测未完成,不影响本表。
> 检索服务是本块历史上的共同瓶颈(曾因平台判低利用率被停、HTTP 400),长跑前先确认服务稳定。

### 3. ScienceWorld-OOD(仅 inference,无训练:第 1 块的 TextCraft ckpt 直接迁移到 sciworld 官方 test 1,819 变体)— 已完成(官方分 0-100 / 完全成功率 %)

```bash
# textcraft ckpt 一律用 val 最优:<ckpt>/best/actor/huggingface(旧策略产物无 best 软链时用 global_step_150)
NAME=<实验名> BENCH=sciworld MODEL=<对应 textcraft ckpt>/best/actor/huggingface OUT=$HOME/rso_runs/$NAME \
SCRIPT=examples/rso_8gpu/run_sciworld_<方法>_8gpu.sh \
  bash scripts_rso/run_full_eval.sh data.val_files=$HOME/data/sciworld/test.parquet data.val_batch_size=128
```

| 方法 | gemma-2b | gemma-26b |
|---|---|---|
| rso | **DONE** 16.79 / 8.41 | **DONE** 39.79 / 17.76 |
| rao | **DONE** 17.47 / 7.70 | **DONE** 40.04 / 15.94 |
| skill | **DONE** 15.90 / 8.47 | **DONE** 40.12 / 9.73 |
| gtopsd | **DONE** 16.27 / 8.41 | **DONE** 40.41 / 10.17 |
| grpo | **DONE** 17.65 / 9.68 | **DONE** 39.87 / 9.68 |

### 4. TextCraft-OOD(仅 inference,无训练:第 1 块的 TextCraft ckpt 直接迁移到真 TextCraft 官方 test 84 题)

```bash
# textcraft ckpt 一律用 val 最优:<ckpt>/best/actor/huggingface(旧策略产物无 best 软链时用 global_step_150)
# 脚本按方法族选:递归 rso / rao → run_textcraft_ood_rso_8gpu.sh;flat skill / gtopsd / grpo → run_textcraft_ood_grpo_8gpu.sh
NAME=<实验名> BENCH=textcraft_ood MODEL=<对应 textcraft ckpt>/best/actor/huggingface OUT=$HOME/rso_runs/$NAME \
SCRIPT=examples/rso_8gpu/run_textcraft_ood_<rso|grpo>_8gpu.sh \
  bash scripts_rso/run_full_eval.sh data.val_files=$HOME/data/textcraft_ood/test.parquet data.val_batch_size=84
```

| 方法 | gemma-2b | gemma-26b |
|---|---|---|
| rso | **TODO** gemma-2b_textcraft_ood_rso | **TODO** gemma-26b_textcraft_ood_rso |
| rao | **TODO** gemma-2b_textcraft_ood_rao | **TODO** gemma-26b_textcraft_ood_rao |
| skill | **TODO** gemma-2b_textcraft_ood_skill | **TODO** gemma-26b_textcraft_ood_skill |
| gtopsd | **TODO** gemma-2b_textcraft_ood_gtopsd | **TODO** gemma-26b_textcraft_ood_gtopsd |
| grpo | **TODO** gemma-2b_textcraft_ood_grpo | **TODO** gemma-26b_textcraft_ood_grpo |

### 5. ScienceWorld(训练 + 测评;java 11+、主机内存 ≥240G;测评 = README §四 val_only,MODEL 换本块 ckpt;官方分 0-100 / 完全成功率 %)

| 方法 | gemma-2b | gemma-26b |
|---|---|---|
| rso | **DONE** 36.70 / 21.72(run231/step105) | **DONE** 48.05 / 23.91(run230/step150) |
| rao | **TODO** gemma-2b_sciworld_rao | **TODO** gemma-26b_sciworld_rao |
| skill | **TODO** gemma-2b_sciworld_skill | **TODO** gemma-26b_sciworld_skill |
| gtopsd | **TODO** gemma-2b_sciworld_gtopsd | **TODO** gemma-26b_sciworld_gtopsd |
| grpo | **TODO** gemma-2b_sciworld_grpo | **TODO** gemma-26b_sciworld_grpo |

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
| textcraft_ood | `goal_depth` `env_actions` `invalid_actions`(被环境拒绝的动作数,含输出里抽不出动作的步) |
| sciworld_ood / sciworld | `task` `variation` `score`(0-100,负分裁零) |
| searchqa | `data_source` `question_type` `hops` `em` `f1` `subem` `answer` `golds` `n_searches`(只含 musique / 2wikimultihopqa / hotpotqa) |

汇总与打包(收齐全部实验的逐题文件后执行一次):

```bash
mkdir -p deliver && cp *_cases.jsonl *_train_progress.csv deliver/
python scripts_rso/make_eval_report.py --cases-dir deliver --out-dir deliver   # 每个 bench 一份 <bench>_report.md
# 打开每份 deliver/<bench>_report.md,在末尾"分析(执行者补写)"一节写总结分析,每条结论引用表中数字
tar czf rso_deliverables.tar.gz deliver/
```
