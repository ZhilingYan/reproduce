# RSO — TextCraft-Synth 上的 GRPO / OPSD baseline 实验

本分支在 [SDAR](https://github.com/ZJU-REAL/SDAR)(verl-agent 系)框架里移植了 RAO 论文
([platoon](https://github.com/ApGa/platoon/tree/apga/rao-snapshot/))的 **TextCraft-Synth**
环境,并提供三条可直接跑的 baseline。目标是在同一环境上比较:纯 RL、GT 特权蒸馏、
技能库特权蒸馏,为后续"递归 agent + 自蒸馏"实验建立参照系。

> **先读 [`TASK.md`](TASK.md)** —— 那里写明了要跑哪些实验、成本预期、健康检查判据
> 和交付物清单。本文只讲"怎么装、怎么跑"。

---

## 一、5 步跑起来

```bash
# 1) 克隆
git clone <this-repo-url> && cd SDAR && git checkout rso-textcraft-synth

# 2) 建环境(见第二节。已有 verl/vllm 环境的话直接复用)
conda create -n rso python=3.11 -y && conda activate rso
pip install -r requirements_rso.txt

# 3) 生成占位 parquet(决定训练步数与验证批量,内容不参与训练)
python scripts_rso/prepare_synth_parquet.py --out ~/data/verl-agent/synth_full/text

# 4) 跑一条 baseline(8 卡单节点)。MODEL 见第四节
MODEL=Qwen/Qwen3.5-4B TP=2 MICRO_BSZ=2 OUT=$HOME/rso_runs/q35_4b_grpo \
  bash examples/rso_8gpu/run_synth_grpo_8gpu.sh

# 5) 看曲线
tensorboard --logdir $HOME/rso_runs/q35_4b_grpo/tensorboard

# 6) 训练完成后:全量 632 题评测,产出指标 + 每题完整轨迹
python scripts_rso/eval_full_val.py \
  --model $HOME/rso_runs/q35_4b_grpo/ckpts/global_step_150/actor/huggingface \
  --out   $HOME/rso_runs/q35_4b_grpo/eval_full --split val --tp 2
```

三条 baseline 分别是:

| 脚本 | 方法 | 关键开关 |
|---|---|---|
| `examples/rso_8gpu/run_synth_grpo_8gpu.sh` | 纯 GRPO | — |
| `examples/rso_8gpu/run_synth_gtopsd_8gpu.sh` | GRPO + OPSD(GT 特权) | `privileged_source=gt, gate_beta=0` |
| `examples/rso_8gpu/run_synth_skill_8gpu.sh` | SDAR + 技能库特权 | `privileged_source=skill, gate_beta=5` |

三者除算法开关外**所有参数完全一致**,可直接同表比较。

---

## 二、环境

我们跑通的组合(`requirements_rso.txt` 是完整冻结):

| 包 | 版本 | 备注 |
|---|---|---|
| python | 3.11 | |
| torch | 2.6.0+cu124 | |
| vllm | 0.8.5 | rollout 引擎 |
| transformers | 4.51.1 | |
| ray | 2.50.0 | |
| flash-attn | 2.7.4.post1 | 对应 cu12/torch2.6/cp311 的 wheel |

注意事项:
- `requirements_rso.txt` 里 flash-attn 是直链 wheel,若 CUDA/python 版本不同请换对应 wheel。
- 若集群有 user-site 包污染(`~/.local/lib/python3.x`),跑之前 `export PYTHONNOUSERSITE=1`。
- TextCraft-Synth **不需要** ALFWorld/WebShop 的数据与依赖,那些 benchmark 的报错可忽略。

---

## 三、数据

任务数据已随仓库提供,无需下载:

```
agent_system/environments/env_package/textcraft_synth/data/
├── textcraft_synth_train.jsonl   # 2522 题 (easy 588 / medium 852 / hard 544 / extreme 538)
├── textcraft_synth_val.jsonl     # 632 题 (147 / 213 / 136 / 136)
├── textcraft_synth_val100.jsonl  # 固定验证子集: 每难度 25 题, 共 100
└── val100_ids.json               # val100 的任务 id 清单 + 抽样种子(保证跨实验一致)
```

数据由 RAO 官方生成器(`synth_recipe_generator.py`, seed=42)产出,与论文同源。难度 =
合成树深度:easy 2-3 层、medium 4-6、hard 7-9、extreme 10-12。

**当前实验口径**:训练用 easy+medium(1440 题);验证用 val100(100 题,含 hard/extreme
作为泛化探针)。理由见第六节。

---

## 四、模型:Qwen3.5-4B / Qwen3.5-9B

代码对模型是中立的 —— 换模型只需改 `MODEL` 环境变量,**不需要动任何代码**:

```bash
# 4B
MODEL=Qwen/Qwen3.5-4B  TP=2 MICRO_BSZ=2 OUT=$HOME/rso_runs/q35_4b_grpo \
  bash examples/rso_8gpu/run_synth_grpo_8gpu.sh

# 9B
MODEL=Qwen/Qwen3.5-9B  TP=4 MICRO_BSZ=1 OUT=$HOME/rso_runs/q35_9b_grpo \
  bash examples/rso_8gpu/run_synth_grpo_8gpu.sh
```

之所以能通用,是因为 prompt 层做了两件与模型无关的事:整个 prompt 作为**一条 user
消息**发送(不依赖 system 角色),推理与动作用 `<thought>` / `<action>` **纯文本标签**
(不是任何模型词表里的特殊 token)。

两个规格的建议起点:

| | Qwen3.5-4B | Qwen3.5-9B |
|---|---|---|
| `TP`(张量并行) | 2 | **4** |
| `MICRO_BSZ` | 2(80GB 卡可试 4) | **先用 1**,稳定后再试 2 |
| 评测时 `--tp` | 2 | 4 |

**换新模型时请先确认三件事**(我们只在 Qwen3-4B-Instruct-2507 上完整验证过):

1. **是否是思考(thinking)模型**。如果模型会自动产出 `<think>...</think>` 思考块,
   要么关掉思考模式(chat template 参数),要么给足 `max_response_length`——否则思考块
   吃满生成预算,正文动作产不出来,表现为"整局空转"。判断方法:跑几条推理看
   `episode/valid_action_ratio`,低于 0.9 就是格式出了问题而非能力问题。
2. **词表大小**。词表越大输出层 logits 显存越高(Qwen3 是 151k)。首次跑
   `MICRO_BSZ=1`,看 `perf/max_memory_allocated_gb` 有余量再往上调。
3. **vLLM 是否支持该架构**。`requirements_rso.txt` 冻结的是 vllm 0.8.5;
   若模型较新需要升级 vLLM,升级后请重跑一次第六节的健康检查。

- `MICRO_BSZ` 只改梯度累积粒度,**数学等价,不影响结果**,OOM 时放心调小。
- **同模型的三条 baseline 曲线可以直接对比**:除算法开关外所有参数完全一致。
  `OUT` 目录名请带上模型和方法标识,避免互相覆盖。

---

## 四点五、训练完成后的全量评测

训练中途的验证用的是 **val100 固定子集**(每难度 25 题,为了省时间);
正式结论要用 **全量 632 题**,由独立脚本完成:

```bash
python scripts_rso/eval_full_val.py \
  --model <ckpt 或 HF 模型名> --out <输出前缀> \
  --split val --max-steps 200 --tp 2
```

产出两个文件:

| 文件 | 内容 |
|---|---|
| `<out>_metrics.json` | 汇总指标:总体成功率、**easy+medium 合并口径**、分难度成功率、平均轮数、耗时 |
| `<out>_cases.jsonl` | **每题一行**:task_id / 难度 / gold 计划长度 / 是否成功 / reward / 用了多少轮 / **逐轮完整轨迹**(完整 prompt、模型原始输出、解析动作、环境反馈、reward) |

要点:
- 该脚本用「顺序单环境 + vLLM 批量生成」,内存恒定,可以安心跑全量 632 题
  (训练时的 lockstep 验证如果开到 632 会常驻 632 个 Ray worker 而 OOM);
- 它的 prompt 拼接与训练时**逐字节一致**(已用对拍测试验证),所以评测结果
  与训练指标同口径;
- 请同时对**未训练的基座模型**跑一次,作为 before/after 对比的起点;
- 调试时可用 `--limit 20 --difficulties easy medium` 快速试跑。

## 五、结果在哪里看

```
$OUT/
├── tensorboard/          # 全部曲线(最可靠的真相源, 见下方"已知坑")
├── ckpts/global_step_N/  # 滚动保留最近 2 个全量 ckpt(断点续跑用)
├── rollouts/N.jsonl      # 每个训练步采集到的全部 rollout(prompt/输出/得分)
├── eval_full_metrics.json  # 全量 632 题评测的汇总指标(训练后生成)
├── eval_full_cases.jsonl   # 全量 632 题的每题完整结果+轨迹(训练后生成)
└── slurm-*.out           # console 日志
```

关键指标:

| 指标 | 含义 |
|---|---|
| `val/success_rate` | val100 全部 100 题的成功率 |
| `val/easymedium_success_rate` | easy+medium 50 题的成功率(**主要观察对象**) |
| `val/{easy,medium,hard,extreme}_success_rate` | 分难度成功率(各 25 题) |
| `episode/success_rate` | 训练 rollout 的成功率 |
| `episode/valid_action_ratio` | 动作格式合规率(应 >0.98) |
| `critic/advantages/max` | GRPO 优势幅度(为 0 表示组内无方差 = 学不到东西) |

**参考基线**(Qwen3-4B-Instruct-2507,训练前零样本,4 卡 A100-40G 实测):
val 总 0.29-0.32 / easy 0.76-0.80 / medium 0.40-0.48 / easy+medium 0.58-0.64 /
hard 与 extreme 0.00。换模型后数值会不同,但 **medium 显著非零** 是环境正常的标志。

---

## 六、为什么是这些参数(设计决策)

- **`env.max_steps=100`**(脚本默认):RAO 官方 linear 训练用 200,但我们实测 200 下
  episode 平均长 112 轮 → 每训练步 1.4 万训练行 / 4660 万 token → 单步 2.8 小时
  (4×A100-40G),150 步跑不完。100 仍覆盖 medium 成功所需的 55-96 轮。
  成本与取舍详见 [`TASK.md`](TASK.md) 第四节。
- **训练只用 easy+medium**:hard/extreme 的标准解(gold)中位就要 88/173 步,在任何可行
  预算下都必然失败 → 组内奖励零方差 → GRPO 无梯度,纯烧算力。它们只保留在验证集里
  作为深度泛化探针(flat 方法预期 0,是留给递归方法的空间)。
- **验证用 val100 固定子集**:框架是 lockstep 同步批处理,验证环境进程数 = `val_batch_size`,
  用全量 632 会同时常驻 632 个 Ray worker 而 OOM。val100 每难度 25 题、id 固定,
  跨实验可比。
- **`max_prompt_length=8192` + `truncation='left'`**:对齐 RAO 的 ctx8192 配置。超长时
  另有 manager 层守卫把该 episode 判失败(RAO 哲学:超长是 episode 级失败,不是 job 崩溃),
  `left` 截断只是最后保险丝。
- **prompt 拼接 = 状态块**:每步观察 = 动作结果 + **配方笔记本**(本局 `get_info` 查过的
  配方,永久重放)+ **库存快照**(最近一次主动 `inventory` 的结果,带 "as of step N" 标注)
  + 最近 2 步滑窗。这是本项目最关键的修改:此前只有 2 步滑窗时,medium 需要跨 20+ 步
  记住 5-10 个配方,配方在用到前就被冲掉 → **medium 训练 150 步成功率恒为 0**。加状态块后
  零样本就有 0.40+。信息制度上它严格是 RAO"完整对话历史"的子集,只重放模型自己查询
  挣来的信息,不赠送任何新信息。

---

## 七、已知坑(踩过的,建议直接规避)

1. **Ray driver 日志懒刷新**:console 日志可能几十小时不刷新指标行,不要据此判断是否在跑
   —— 读 tensorboard 的 events 文件才是真相源。
2. **ckpt 轮换留空壳**:`max_actor_ckpt_to_keep` 删的是 `global_step_N/actor/` 里的内容,
   目录本身会留着。判断 ckpt 是否可用要检查 `actor/` 子目录存在。
3. **verl 无 best-ckpt 机制**:滚动窗口只留最近 2 个,历史最优权重会被覆盖。需要保留最优
   模型的话要另写脚本(读 tensorboard 曲线,val SR 创新高时把该步权重拷走)。
4. **改 `max_prompt_length` 必须同步复核微批**:显存峰值 ≈ 微批样本数 × 单样本 token 数。
   我们从 3072 提到 8192 时,原来的微批 4 直接 OOM。
5. **`optimizer_offload` 会把显存问题变成主机内存问题**:开启后约 32GB 优化器状态搬到
   CPU RAM,我们因此撞破 200G 的作业内存上限。除非显存实在不够,否则保持关闭。
6. **episode 步数上限是成本的主导因素**:lockstep 每轮对全 batch 生成一次,
   `env.max_steps` 从 50 提到 200 会让每步数据量涨约 18 倍。改它之前先估算总时长。
7. **动作文本解析的鲁棒性**:`get_info` 同时接受逗号和空格分隔(模型两种都会写)。若新增
   文本动作语法,务必测试模型的各种自然写法,否则"格式死"会伪装成"能力不行"。

---

## 八、代码地图(改了什么)

我们的改动集中在这些文件,其余保持 SDAR 上游原样:

```
agent_system/environments/env_package/textcraft_synth/   # 新增: 环境移植(核心)
  ├── synth_core.py        # 单环境逻辑: 动作/配方/成功判定/【状态块】
  ├── envs.py              # Ray 批量封装 + 难度过滤 + val_split 开关
  ├── projection.py        # 模型输出 -> 动作解析(<thought>/<action>)
  └── data/                # 任务 jsonl + val100 子集
agent_system/environments/prompts/textcraft_synth.py     # 新增: prompt 模板(对齐 RAO 措辞)
agent_system/environments/env_manager.py                 # 改: 新增 TextCraftSynthEnvironmentManager
                                                         #     + prompt 超长守卫 + easymedium 指标
skills/textcraft_synth/                                  # 新增: 技能库(skill baseline 用)
examples/rso_8gpu/                                       # 新增: 8 卡运行脚本 ×3
scripts_rso/prepare_synth_parquet.py                     # 新增: 占位 parquet 生成
scripts_rso/eval_full_val.py                             # 新增: 全量验证集评测(指标+每题轨迹)
TASK.md                                                  # 新增: 任务说明书(跑什么/交付什么)
```

成功判定与 RAO 官方一致:**净合成量**达标(当前库存 − 初始库存 ≥ 目标数),即开局已持有
目标物品不计入,必须新合成。gold 轨迹回放 3154/3154 全部可执行(环境正确性验证)。
