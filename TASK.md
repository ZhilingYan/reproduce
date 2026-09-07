# TASK — TextCraft-Synth 实验任务书

维护记录:
- 2026-08-24 首版:三条 flat baseline(GRPO / GT-OPSD / SDAR-skill),Qwen3-4B。
- 2026-09-07 收到三条 baseline 的训练与评测结果(评测为旧口径 max_steps=200 / temperature=0.4)。训练部分完成 ✅。
- 2026-09-08 新增需求 1(旧 ckpt 重评测)与需求 2(RAO / RSO 两条新方法);仓库已推送对应代码(commit 7fe7334)。

---

## 需求清单

| # | 内容 | 状态 |
|---|---|---|
| 0 | 三条 flat baseline 训练 150 步(Qwen3-4B) | ✅ 完成(2026-09-07) |
| **1** | **用已训好的三个 ckpt(grpo / gtopsd / skill,step150)+ 未训练基座,重跑全量 val 评测,新口径 max_steps=2000 / temperature=0(脚本默认,不传参即是);产出精简结果文件并打包** | ⬜ 待做 |
| **2** | **两条新方法 RAO、RSO:Qwen3-4B 各训练 150 步;训完后同样跑新口径全量 val;产出精简结果文件并打包** | ⬜ 待做 |

---

## 需求 1:旧 ckpt 重评测(先做,不占训练卡)

先 `git pull`(需要 commit 7fe7334 之后的 `eval_full_val.py`)。对四个模型各跑一次:

```bash
# 三个 ckpt + 基座,共 4 次;不传 --max-steps/--temperature,默认即 2000/贪心
python scripts_rso/eval_full_val.py \
  --model <ckpt>/global_step_150/actor/huggingface \
  --out   <dir>/eval2k_<name> --split val --tp 2
python scripts_rso/eval_full_val.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --out   <dir>/eval2k_before --split val --tp 2
```

支持断点续跑:同一 `--out` 重复执行会跳过已完成的题。

## 需求 2:RAO 与 RSO(训练 + 评测)

```bash
# 训练(8 卡;两条各一次;MICRO_BSZ 用 1)
MODEL=Qwen/Qwen3-4B-Instruct-2507 TP=2 MICRO_BSZ=1 OUT=$HOME/rso_runs/q3_4b_rao \
  bash examples/rso_8gpu/run_synth_rao_8gpu.sh
MODEL=Qwen/Qwen3-4B-Instruct-2507 TP=2 MICRO_BSZ=1 OUT=$HOME/rso_runs/q3_4b_rso \
  bash examples/rso_8gpu/run_synth_rso_8gpu.sh
# 脚本内的算法与底座参数已配好,除 MODEL/TP/MICRO_BSZ/OUT 外请勿改动。

# 训完后全量 val(注意递归要加 --recursive 及其参数)
python scripts_rso/eval_full_val.py \
  --model <ckpt>/global_step_150/actor/huggingface \
  --out   <dir>/eval2k_rao --split val \
  --recursive --per-agent-steps 25 --max-depth 6 --max-steps 200 --tp 2
```

训练期健康检查(tensorboard,前 5 步内确认):
`rso/delegating_trees > 0`;`rso/valid_action_ratio > 0.9` 且不持续下滑;`actor/entropy_loss` 不单调上行。
(`rao/` 前缀同理。)成功率曲线看 `rao/root_reward_mean` 或 `rso/root_reward_mean`。

---

## 精简结果文件规范(两个需求共用)

评测产物 `*_cases.jsonl` 很大(含逐轮 trajectory)。请按 2026-09-07 那批的同款格式制作摘要:
每个模型一对 `{name}_summary.jsonl` + `{name}_summary.csv`,632 行,每题一行,字段固定为:

```
task_id, difficulty, success, reward, turns_used, gold_plan_len, max_depth,
fail_reason, n_turns_recorded, last_action, input_tokens, output_tokens, total_tokens
```

- `fail_reason`:"" = 成功;loop_detected;cap_hit(耗尽步数预算)。
- 递归两条(RAO/RSO)另附每题的 tree 字段(cases 文件里已有,原样带上即可):
  `n_nodes, max_depth, delegated, n_subagents, subagent_success, root_turns, sub_turns`。
- 附一份 README.txt:写明评测口径(max_steps / temperature / split)与成功率汇总表(overall / 分难度)。
- 全部打包为一个 tar.gz。命名:
  - 需求 1:`flat3baseline_qwen3_4b_textcraft_eval2k_summaries.tar.gz`
  - 需求 2:`raorso_qwen3_4b_textcraft_eval2k_summaries.tar.gz`
- 同时保留 `*_metrics.json` 原样入包(脚本自动生成,不用改)。

---

## 环境与数据(与首版相同,已配好的可跳过)

```bash
pip install -r requirements_rso.txt
python scripts_rso/prepare_synth_parquet.py --out ~/data/verl-agent/synth_full/text
```
问题排查见 README(递归相关:第四点七节与"已知坑")。
