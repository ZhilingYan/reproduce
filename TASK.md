# TASK — 要做什么、怎么判断做完了

给合作者的任务说明书。README.md 讲"怎么装、怎么跑",本文讲"跑什么、为什么、
交付什么"。

---

## 一、一句话任务

在 TextCraft-Synth 环境上,用 **Qwen3.5-4B** 与 **Qwen3.5-9B** 两个模型,各跑
**三条 baseline**(纯 GRPO / GRPO+OPSD-GT / SDAR-skill)的 150 步强化学习训练,
训练完成后在**全量 632 道验证题**上做推理评测,交回指标与每题完整轨迹。

---

## 二、背景(为什么做这件事)

我们在研究"递归式 agent + 自蒸馏"。递归方法要证明有效,必须先有一组**扎实的
平铺(flat)基线**做参照:同样的环境、同样的数据、同样的训练预算下,不用递归
能做到什么程度。这三条 baseline 就是那个参照系:

| baseline | 特点 | 在对比中的角色 |
|---|---|---|
| 纯 GRPO | 只用环境奖励做策略梯度 | 最朴素的 RL 下限 |
| GRPO + OPSD(GT) | 训练时给 teacher 看本题标准答案做自蒸馏 | "有特权信息能好多少"的上限参照 |
| SDAR + skill | 训练时给 teacher 看通用攻略做自蒸馏 | 介于两者之间的实用方案 |

三者**除算法开关外所有参数完全一致**,所以曲线可以直接同图比较。

TextCraft-Synth 是合成的多步合成(crafting)任务:给定一批原料和一个目标物品,
agent 要自己查配方、规划合成链、按正确的数量顺序执行。难度 = 合成树深度,
easy(2-3 层)/ medium(4-6)/ hard(7-9)/ extreme(10-12)。

---

## 三、要跑的实验矩阵

**2 个模型 × 3 条 baseline = 6 个训练任务**,外加每个训练完成后的 1 次全量评测。

| # | 模型 | baseline | 脚本 |
|---|---|---|---|
| 1 | Qwen/Qwen3.5-4B | GRPO | `examples/rso_8gpu/run_synth_grpo_8gpu.sh` |
| 2 | Qwen/Qwen3.5-4B | GRPO+OPSD-GT | `examples/rso_8gpu/run_synth_gtopsd_8gpu.sh` |
| 3 | Qwen/Qwen3.5-4B | SDAR+skill | `examples/rso_8gpu/run_synth_skill_8gpu.sh` |
| 4 | Qwen/Qwen3.5-9B | GRPO | 同上,`MODEL` 换成 9B |
| 5 | Qwen/Qwen3.5-9B | GRPO+OPSD-GT | 同上 |
| 6 | Qwen/Qwen3.5-9B | SDAR+skill | 同上 |

如果算力有限,**优先级顺序**是:先把 4B 的三条跑完(1→2→3),再上 9B。
单模型内部三条同等重要,不要只跑一条。

---

## 四、成本预期(重要,请先读)

我们在 4×A100-40G 上实测过:**episode 步数上限直接决定成本**,因为框架是
lockstep(每轮所有环境一起生成一次)。

| env.max_steps | episode 平均长度 | 每训练步的数据量 | 每训练步耗时(4×A100-40G) |
|---|---|---|---|
| 50 | ~21 轮 | ~250 万 token | ~15 分钟 |
| **200** | ~112 轮 | ~4660 万 token | **~2.8 小时** |

200 步预算下 150 训练步 ≈ 420 小时/条,我们的配额撑不住,所以**脚本默认
`env.max_steps=100`**(折中:medium 的成功案例实测用 55-96 轮,100 覆盖得住)。

你们 8 卡应该比我们 4 卡快,但请**先跑 3-5 步看实测 `timing_s/step`**
(在 tensorboard 里),据此估算总时长再决定是否调整。可调的旋钮按性价比排序:

1. `env.max_steps`(默认 100):最有效,同时压缩 rollout 轮数与训练行数;
2. `MICRO_BSZ`(默认 2):影响 actor 更新耗时,显存够就调大到 4;
3. `data.train_batch_size`(默认 16):减半则每步数据减半,但梯度更噪。

---

## 五、执行步骤

### 步骤 1:环境与数据(一次性)

```bash
git clone <repo> && cd <repo>
conda create -n rso python=3.11 -y && conda activate rso
pip install -r requirements_rso.txt
python scripts_rso/prepare_synth_parquet.py --out ~/data/verl-agent/synth_full/text
```

任务数据已在仓库里(632+2522 题),**不需要下载**。只有模型权重会自动从
HuggingFace 拉取。

### 步骤 2:训练(每条 baseline 一次)

```bash
MODEL=Qwen/Qwen3.5-4B OUT=$HOME/rso_runs/qwen35_4b_grpo TP=2 MICRO_BSZ=2 \
  bash examples/rso_8gpu/run_synth_grpo_8gpu.sh
```

9B 建议 `TP=4`,并先用 `MICRO_BSZ=1` 试跑几步确认不 OOM。

**跑之前请先跑 3-5 步做健康检查**(见第六节的判据),确认没问题再让它跑满 150 步。

### 步骤 3:全量验证集评测(训练完成后)

```bash
python scripts_rso/eval_full_val.py \
  --model $HOME/rso_runs/qwen35_4b_grpo/ckpts/global_step_150/actor/huggingface \
  --out   $HOME/rso_runs/qwen35_4b_grpo/eval_full \
  --split val --tp 2
```

注意:
- `--split val` 是**全量 632 题**(训练中途的验证用的是 100 题子集,不能替代);
- 评测口径由脚本默认值给出:`max_steps=2000`、`temperature=0`(贪心解码,与 RAO 官方
  推理协议一致),无需显式传参。深题 gold 最多 209 步,2000 步确保失败反映能力而非预算;
- 9B 用 `--tp 4`;
- 也请对**未训练的基座模型**跑一次同样的评测作为"训练前基线"
  (`--model Qwen/Qwen3.5-4B`),这样才有 before/after 的对比。

---

## 六、健康检查判据(跑满之前先看这几个数)

在 tensorboard 里看前几步,四条都满足才值得继续跑:

| 指标 | 期望 | 不满足意味着 |
|---|---|---|
| `val/medium_success_rate`(step 0) | **> 0.2** | 环境或 prompt 有问题——medium 若为 0,后面 150 步也学不动 |
| `critic/advantages/max` | **> 0**(通常 1 左右) | 组内奖励零方差,GRPO 没有梯度 |
| `episode/valid_action_ratio` | **> 0.95** | 模型输出格式不合规,死在格式而非能力 |
| 无 OOM、`timing_s/step` 稳定 | — | 见第四节调参 |

我们在 Qwen3-4B-Instruct-2507 上的训练前基线(供对照,你们的模型数值会不同):
总体 0.29-0.32 / easy 0.76-0.80 / **medium 0.40-0.48** / hard 与 extreme 0.00。
hard/extreme 为 0 是**预期内的**(平铺方法的结构性上限,正是留给递归方法的空间)。

---

## 七、交付物

每条 baseline 请交回:

1. **`<OUT>/tensorboard/`** —— 完整训练曲线(最重要,别只截图);
2. **`<OUT>/eval_full_metrics.json`** —— 全量 632 题的汇总指标
   (总体 / 分难度 / easy+medium 合并 / 平均轮数);
3. **`<OUT>/eval_full_cases.jsonl`** —— **每题一行**的完整结果:
   task_id、难度、gold 计划长度、是否成功、reward、用了多少轮,以及
   **逐轮完整轨迹**(每轮的完整 prompt、模型原始输出、解析出的动作、
   环境反馈、reward)。这份是做失败模式分析用的,请务必保留;
4. **训练日志**(slurm 输出或 console 日志);
5. 一句话说明实际用的配置(模型、`max_steps`、`MICRO_BSZ`、卡数),
   以及跑了多久。

如果某条没跑满 150 步(超时、抢占等),把实际跑到的步数和最后的 ckpt 交回即可,
**不要因为没跑满就不交** —— 部分曲线也有价值。

---

## 八、常见问题

**Q: 训练中途的验证(val100)和最后的全量评测(632)是什么关系?**
A: 前者是训练过程中每 5 步跑一次的监控,用 100 题固定子集(每难度 25 题),
为了省时间;后者是训练结束后的正式评测,用全量 632 题。两者的题目 id 都是
固定的,跨模型跨方法可比。

**Q: 为什么训练只用 easy+medium,不用 hard/extreme?**
A: hard/extreme 的标准解中位就要 88/173 步,在可行预算内必然失败 → 组内奖励
零方差 → GRPO 拿不到梯度,纯烧算力。它们只保留在**验证集**里作为泛化探针。

**Q: 中途挂了怎么办?**
A: 脚本每 5 步存一次 ckpt,`resume_mode=auto` 会自动从最近的 ckpt 续跑。
直接重新提交同一条命令即可。

**Q: 显存不够?**
A: 按顺序试:`MICRO_BSZ=1` → `TP` 调大 → `data.train_batch_size` 减半。
前两个不改变实验语义(只是梯度累积粒度和模型切分方式),第三个会改变。

**Q: 三条 baseline 可以并行跑吗?**
A: 可以,它们完全独立,输出目录不同即可(`OUT` 记得区分模型和方法)。
