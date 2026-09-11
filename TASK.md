# TASK — TextCraft-Synth 实验任务书

2026-09-11 版,取代此前全部需求。先 `git pull` 到最新;安装/数据/评测细节见 [`README.md`](README.md)。
**优先跑递归方法(4-6:RAO / RSO / RSO+OPSD),flat(1-3)排后。**

## 跑什么:3 模型 × 6 方法 = 18 条训练 + 18 次全量评测

| 模型(`MODEL=`) | TP | MICRO_BSZ(flat/递归) | 备注 |
|---|---|---|---|
| `Qwen/Qwen3-4B-Instruct-2507` | 2 | 2 / 1 | |
| `Qwen/Qwen3-8B` | 2 | 1 / 1 | thinking 混合模型,须关思考模式;首跑前 5 步看 `episode/valid_action_ratio` > 0.9,低了先排查格式再排查能力 |
| `google/gemma-4-E4B-it` | 2 | 1 / 1 | 需单独环境(transformers ≥ 5.16 + 近期 vllm,见 README 安装节) |

| # | 方法 | 8 卡脚本 |
|---|---|---|
| 1 | GRPO | `examples/rso_8gpu/run_synth_grpo_8gpu.sh` |
| 2 | GRPO + GT-OPSD | `examples/rso_8gpu/run_synth_gtopsd_8gpu.sh` |
| 3 | SDAR(技能库特权) | `examples/rso_8gpu/run_synth_skill_8gpu.sh` |
| 4 | RAO(递归) | `examples/rso_8gpu/run_synth_rao_8gpu.sh` |
| 5 | RSO(递归) | `examples/rso_8gpu/run_synth_rso_8gpu.sh` |
| 6 | RSO+OPSD(递归) | `examples/rso_8gpu/run_synth_rso_opsd_8gpu.sh` |

```bash
# 训练命令模板(18 条各一次;口径已在脚本里配好:只训 medium、训练中 val=easy+medium 50 题;
# 除 MODEL/TP/MICRO_BSZ/OUT 外请勿改参数)
MODEL=<上表> TP=2 MICRO_BSZ=<上表> OUT=$HOME/rso_runs/<模型简称>_<方法> \
  bash examples/rso_8gpu/run_synth_<方法>_8gpu.sh
```

前 5 步健康检查:`episode/valid_action_ratio` > 0.9;递归另看 `rao|rso/delegating_trees` > 0;异常即停排查。

## 交付 1:训练进展文件(每条 run 一个 csv,共 18 个)

```bash
python scripts_rso/export_train_progress.py --logdir $OUT/tensorboard \
  --out <模型简称>_<方法>_train_progress.csv
```

每行一个训练 step:每步训练成功率(`episode/*_success_rate`)、每 5 步的 val 成功率(`val/*_success_rate`)、健康指标(合规率/熵)。

## 交付 2:全量 val 精简结果(每条 run 一个 jsonl,共 18 个)

```bash
# flat(1-3):max_steps 用脚本默认 500
python scripts_rso/eval_full_val.py \
  --model <ckpt>/global_step_150/actor/huggingface --out <dir>/eval_<名> --split val --tp 2
# 递归(4-6)
python scripts_rso/eval_full_val.py \
  --model <ckpt>/global_step_150/actor/huggingface --out <dir>/eval_<名> --split val \
  --recursive --per-agent-steps 25 --max-depth 6 --max-steps 200 --tp 2
```

从 `*_cases.jsonl` 制作 `<模型简称>_<方法>_summary.jsonl`(632 行,每题一行),字段固定:

```
task_id, difficulty, success, reward, turns_used, gold_plan_len, max_depth,
fail_reason, n_turns_recorded, last_action, input_tokens, output_tokens, total_tokens
```

- `fail_reason`:"" = 成功;loop_detected;cap_hit(耗尽步数预算)。
- 递归三条另附 tree 字段(cases 里已有,原样带上):
  `n_nodes, max_depth, delegated, n_subagents, subagent_success, root_turns, sub_turns`。
- `*_metrics.json` 原样入包。全部(交付 1 + 交付 2)打成一个 tar.gz。
