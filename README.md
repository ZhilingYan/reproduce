# RSO — TextCraft-Synth 实验仓库

在 SDAR(verl-agent 系)框架上移植 RAO 论文([platoon](https://github.com/ApGa/platoon/tree/apga/rao-snapshot/))的 TextCraft-Synth 环境,提供 6 条可直接跑的方法。要跑哪些实验、健康检查判据、交付物清单见 [`TASK.md`](TASK.md)。

## 一、安装

```bash
git clone <this-repo-url> && cd <repo>
conda create -n rso python=3.11 -y && conda activate rso
pip install -r requirements_rso.txt   # torch 2.6.0+cu124 / vllm 0.8.5 / transformers 4.51.1 / ray 2.50.0
```

- flash-attn 在 requirements 里是直链 wheel(cu12/torch2.6/cp311),CUDA/python 版本不同请换对应 wheel。
- 集群有 user-site 包污染(`~/.local/lib/python3.x`)时,先 `export PYTHONNOUSERSITE=1`。
- **Gemma(`google/gemma-4-E4B-it`)需单独建环境**:transformers ≥ 5.x(5.16.1 实测可用)+ 支持 gemma-4 的近期 vllm;本仓库 pin 的版本加载不了它。Qwen 系用原 pin,两套环境别混。

## 二、数据

```bash
python scripts_rso/prepare_synth_parquet.py --out ~/data/verl-agent/synth_full/text
```

任务数据随仓库自带(`agent_system/environments/env_package/textcraft_synth/data/`):
train 2522 题(easy 588 / medium 852 / hard 544 / extreme 538),val 632 题(147/213/136/136),val100 固定验证子集(每难度 25 题)。flat 三条只训 medium(2026-09-11 口径),递归三条训 easy+medium;难度 = 合成树深度(easy 2-3 层 … extreme 10-12 层)。

## 三、训练(8 卡单节点)

| # | 方法 | 脚本 |
|---|---|---|
| 1 | GRPO | `examples/rso_8gpu/run_synth_grpo_8gpu.sh` |
| 2 | GRPO + GT-OPSD | `examples/rso_8gpu/run_synth_gtopsd_8gpu.sh` |
| 3 | SDAR(技能库特权) | `examples/rso_8gpu/run_synth_skill_8gpu.sh` |
| 4 | RAO(递归) | `examples/rso_8gpu/run_synth_rao_8gpu.sh` |
| 5 | RSO(递归) | `examples/rso_8gpu/run_synth_rso_8gpu.sh` |
| 6 | RSO+OPSD(递归) | `examples/rso_8gpu/run_synth_rso_opsd_8gpu.sh` |

```bash
# flat(1-3):MICRO_BSZ 用 2
MODEL=Qwen/Qwen3.5-4B TP=2 MICRO_BSZ=2 OUT=$HOME/rso_runs/q35_4b_grpo \
  bash examples/rso_8gpu/run_synth_grpo_8gpu.sh
# 递归(4-6):MICRO_BSZ 用 1;可选 TRACE=1 落盘每棵树完整轨迹(约 0.5GB/步)
MODEL=Qwen/Qwen3.5-4B TP=2 MICRO_BSZ=1 OUT=$HOME/rso_runs/q35_4b_rso \
  bash examples/rso_8gpu/run_synth_rso_8gpu.sh
# 9B:TP=4、MICRO_BSZ=1;换模型只改 MODEL,不需要动代码
```

- 可调的只有 `MODEL / TP / MICRO_BSZ / OUT`(递归另有 `TRACE`);脚本内算法与底座参数已配好,**请勿改动**。`MICRO_BSZ` 只改梯度累积粒度,数学等价,OOM 时放心调小。
- 曲线:`tensorboard --logdir $OUT/tensorboard`;成功率看 `val/easymedium_success_rate`(flat)或 `rao/root_reward_mean`、`rso/root_reward_mean`(递归)。
- 训练中每 5 步 val 一次:flat 用 val100 的 easy+medium 子集(50 题,2026-09-11 口径);递归用 val100 全 4 难度 100 题(RSO+OPSD 因 ref+teacher 双前向内存取 50)。
- CPU 测试:`PYTHONPATH=$PWD python tests/test_rso_core.py` 等,`tests/` 下共 12 个。

## 四、全量评测(632 题)

```bash
# flat(1-3)与未训练基座
python scripts_rso/eval_full_val.py \
  --model <ckpt>/global_step_150/actor/huggingface --out <dir>/eval_full --split val --tp 2
# 递归(4-6)
python scripts_rso/eval_full_val.py \
  --model <ckpt>/global_step_150/actor/huggingface --out <dir>/eval_full --split val \
  --recursive --per-agent-steps 25 --max-depth 6 --max-steps 200 --tp 2
```

- flat 口径为脚本默认值,无需显式传:`max_steps=500`(2026-09-11 起,由 2000 调整)、
  `temperature=0`(贪心);递归口径不变:显式 `--max-steps 200`(lockstep 轮数)。
  难度覆盖均为 632 题全难度。
- 产出:`<out>_metrics.json`(总体/分难度成功率、平均轮数)+ `<out>_cases.jsonl`(每题一行,含逐轮完整轨迹)。同一 `--out` 重复执行会跳过已完成的题(断点续跑)。
- 请同时对**未训练基座**跑一次,作为 before/after 起点。RSO+OPSD 的 ckpt 与 RSO 同构,评测命令相同。

## 五、参考基线结果

Qwen3-4B-Instruct-2507 **未训练基座**,max_steps=2000(旧口径实测)/ temperature=0 / 全量 632 题。
注意:flat 现行口径 max_steps=500 下 medium/hard 的数值会低于此表(表中 medium 平均 303 轮、
hard 1018 轮,500 步会截断一部分),仅作环境正确性参照:

| 难度 | 题数 | 成功率 | 平均轮数 |
|---|---|---|---|
| easy | 147 | 0.898 | 62 |
| medium | 213 | 0.432 | 303 |
| hard | 136 | 0.007 | 1018 |
| extreme | 136 | 0.000 | 950 |
| easy+medium | 360 | 0.622 | |
| **全部** | **632** | **0.356** | |

换模型后数值会不同,但 medium 显著非零是环境正常的标志;flat 方法在 hard/extreme 接近 0 属预期(平铺方法的能力上限)。

## License

MIT (see `LICENSE`). Upstream verl / SDAR portions remain under Apache-2.0; attribution in `Notice.txt`.
