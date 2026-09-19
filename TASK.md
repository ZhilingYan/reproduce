# TASK — 三 benchmark 实验任务书

2026-09-20 版,取代此前全部需求。先 `git pull` 到最新;安装/数据/训练/全量测评命令一律见 [`README.md`](README.md)。

## 模型(3 个)

| 代号 | `MODEL=` | 说明 |
|---|---|---|
| qwen3-4b | `Qwen/Qwen3-4B-Instruct-2507` | 仓库 pin 的环境直接用 |
| gemma-4b | `google/gemma-4-E4B-it` | 需单独环境:transformers ≥ 5.x + 支持 gemma-4 的近期 vllm;仓库 pin 的版本加载不了它 |
| gemma-26b | gemma-4 26B 对应 HF id(以你环境为准) | 同上单独环境;建议 TP=4、MICRO_BSZ=1 |

## 实验(4 块;第 2/3/4 块 = 5 个方法 × 上表你负责的模型,每个 (方法, 模型) 为一条 run)

5 个方法与脚本对应关系见 README §三的矩阵(GRPO / GRPO+GT-OPSD / SDAR 技能库 / RSO / RSO+OPSD)。

### 1. TextCraft:继续监控在跑实验

在跑的 run 照旧跑完并交付;五方法中你的模型尚未启动的,按 README §三 textcraft 列补齐——第 3 块要用这五个方法的 ckpt。

### 2. Search-QA:五组训练 + 测评

- 训练集 = 2Wiki + MuSiQue(README §二 数据管线);训练中 val = 三源×15 固定 case(脚本已配)。
- 训完全量测评 = 2Wiki + MuSiQue(in-domain)+ HotpotQA(OOD),`test_3src.parquet` 22,398 题,README §四 的 `val_only` 配方。
- 命令:README §三 Search-QA 列的五个脚本,只改 `MODEL/TP/MICRO_BSZ/OUT`;先起检索服务。

### 3. ScienceWorld:仅 inference(TextCraft ckpt 直接迁移,当作 TextCraft 的 OOD bench)

不训练。用第 1 块产出的 TextCraft 五方法 ckpt,各自走对应方法的 sciworld 脚本 + README §四 的 `val_only` 配方,在官方 test 全量 1,819 变体上直接 inference:

```bash
MODEL=<textcraft_ckpt>/global_step_150/actor/huggingface \
  bash examples/rso_8gpu/run_sciworld_<方法>_8gpu.sh \
  data.val_files=$HOME/data/sciworld/test.parquet \
  data.val_batch_size=128 trainer.val_only=True
```

### 4. ScienceWorld:五组训练 + 正常测评

README §三 ScienceWorld 列的五个脚本训练(java 11+、主机内存 ≥240G),训完按 README §四 sciworld 配方全量测评(同上 `val_only`,MODEL 换本块自己的 ckpt)。

前 5 步健康检查(所有训练 run):`episode/valid_action_ratio` > 0.9;递归另看 `rso/delegating_trees` > 0、`rso/G_negative_ratio` ≡ 0;异常即停排查。

## 交付(要求不变,每条 run 两件)

命名前缀统一 `<模型代号>_<bench>_<方法>`(bench ∈ textcraft / searchqa / sciworld_ood / sciworld)。

**交付 1:训练进展 csv**(第 3 块无训练,免交):

```bash
python scripts_rso/export_train_progress.py --logdir $OUT/tensorboard \
  --out <模型>_<bench>_<方法>_train_progress.csv
```

**交付 2:全量测评结果**:

- textcraft:`eval_full_val.py` 的 `*_metrics.json` + 由 `*_cases.jsonl` 制作的 `*_summary.jsonl`(632 行,字段与上版任务书一致:task_id, difficulty, success, reward, turns_used, gold_plan_len, max_depth, fail_reason, n_turns_recorded, last_action, input_tokens, output_tokens, total_tokens;递归另带 tree 字段 n_nodes, max_depth, delegated, n_subagents, subagent_success, root_turns, sub_turns);
- searchqa / sciworld(含第 3 块):`val_only` run 的完整控制台日志(含 per-source / per-task 分桶指标)+ 该 run 的 tensorboard 目录导出 csv(同交付 1 脚本)。

全部打成一个 tar.gz。
