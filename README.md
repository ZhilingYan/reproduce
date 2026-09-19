# RSO — 三 benchmark 实验仓库(TextCraft-Synth / Search-QA / ScienceWorld)

任务书见 [`TASK.md`](TASK.md)。

## 一、安装

```bash
git clone <this-repo-url> && cd <repo>
conda create -n rso python=3.11 -y && conda activate rso
pip install -r requirements_rso.txt   # torch 2.6.0+cu124 / vllm 0.8.5 / transformers 4.51.1 / ray 2.50.0
pip install scienceworld==1.3.0       # 仅 ScienceWorld 需要;py4j + 内置 jar,另需 java 11+ 在 PATH
```

- flash-attn 在 requirements 里是直链 wheel(cu12/torch2.6/cp311),CUDA/python 版本不同请换对应 wheel。
- 集群有 user-site 包污染(`~/.local/lib/python3.x`)时,先 `export PYTHONNOUSERSITE=1`。

## 二、数据

### TextCraft-Synth

任务数据随仓库自带(`agent_system/environments/env_package/textcraft_synth/data/`):
train 2522 题(easy 588 / medium 852 / hard 544 / extreme 538),val 632 题(147/213/136/136),
val100 固定验证子集(每难度 25 题)。全部方法只训 medium。

```bash
python scripts_rso/prepare_synth_parquet.py --out ~/data/verl-agent/synth_full/text
```

### Search-QA(musique / 2wiki / hotpotqa)

检索服务(wiki-18 语料 + e5 索引,约 75GB,Search-R1 官方流程):

```bash
conda create -n retriever python=3.10 -y && conda activate retriever
conda install numpy==1.26.4 && pip install torch transformers datasets pyserini huggingface_hub uvicorn fastapi
conda install faiss-gpu==1.8.0 -c pytorch -c nvidia -y   # H200/sm90 需 faiss-cpu
local_dir=~/data/searchR1
python examples/search/searchr1_download.py --local_dir $local_dir
cat $local_dir/part_* > $local_dir/e5_Flat.index && gzip -d $local_dir/wiki-18.jsonl.gz
bash examples/search/retriever/retrieval_launch.sh > retrieval_server.log &   # http://0.0.0.0:8000/retrieve
```

数据(原料下载链接在各脚本头注释;`$FLASHRAG_RAW` = FlashRAG 两个 train.jsonl,
`$WIKI2_DATA_IDS` = 2wiki 官方 data_ids.zip 解压目录):

```bash
python examples/data_preprocess/preprocess_search_r1_dataset.py     # ① 官方测试集 → ~/data/searchR1_processed_direct/
python examples/data_preprocess/preprocess_musique_2wiki_train.py   # ② 训练集 34,938 行 → ~/data/searchR1_musique_2wiki/train.parquet
python examples/data_preprocess/convert_2wiki_evidences_to_decomp.py # ③ 2wiki evidences → question_decomposition 回填
python examples/data_preprocess/make_searchrso_data_products.py     # ④ val_sub(3源×15=45)+ val_smoke + test_3src(22,398)+ decomp_store.json
```

### ScienceWorld

任务数据随 `pip install scienceworld` 内置,官方 train/dev/test 划分走 API,零手动下载。

```bash
python examples/data_preprocess/make_sciworld_dummy_parquet.py   # train/val 假 parquet(批次驱动;任务由环境自采)
python examples/data_preprocess/make_sciworld_test_parquet.py    # 官方 test 全量 1,819 行 → ~/data/sciworld/test.parquet
```

## 三、训练(8 卡单节点,5 个方法)

| # | 方法 | TextCraft-Synth | Search-QA | ScienceWorld |
|---|---|---|---|---|
| 1 | GRPO | `run_synth_grpo_8gpu.sh` | `run_search_grpo_8gpu.sh` | `run_sciworld_grpo_8gpu.sh` |
| 2 | GRPO + GT-OPSD | `run_synth_gtopsd_8gpu.sh` | `run_search_gtopsd_8gpu.sh` | `run_sciworld_gtopsd_8gpu.sh` |
| 3 | SDAR(技能库特权) | `run_synth_skill_8gpu.sh` | `run_search_skill_8gpu.sh` | `run_sciworld_skill_8gpu.sh` |
| 4 | RSO(递归) | `run_synth_rso_8gpu.sh` | `run_search_rso_8gpu.sh` | `run_sciworld_rso_8gpu.sh` |
| 5 | RSO+OPSD(递归) | `run_synth_rso_opsd_8gpu.sh` | `run_search_rso_opsd_8gpu.sh` | `run_sciworld_rso_opsd_8gpu.sh` |

脚本都在 `examples/rso_8gpu/`;可调的只有 `MODEL / TP / MICRO_BSZ / OUT`(递归另有 `TRACE`),
脚本内算法与底座参数已配好,**请勿改动**。`MICRO_BSZ` 只改梯度累积粒度,数学等价,OOM 时放心调小。

```bash
# TextCraft(flat 1-3:MICRO_BSZ=2;递归 4-5:MICRO_BSZ=1)
MODEL=Qwen/Qwen3.5-4B TP=2 MICRO_BSZ=1 OUT=$HOME/rso_runs/q35_4b_rso \
  bash examples/rso_8gpu/run_synth_rso_8gpu.sh

# Search-QA(先起检索服务;SEARCH_URL 缺省 http://0.0.0.0:8000/retrieve)
MODEL=Qwen/Qwen3-4B-Instruct-2507 OUT=$HOME/rso_runs/search_gtopsd \
  bash examples/rso_8gpu/run_search_gtopsd_8gpu.sh
MODEL=Qwen/Qwen3-4B-Instruct-2507 OUT=$HOME/rso_runs/search_rso_opsd \
  bash examples/rso_8gpu/run_search_rso_opsd_8gpu.sh

# ScienceWorld(java 11+;主机内存 ≥240G:训练本体 + 178 个常驻 JVM)
MODEL=Qwen/Qwen3-4B-Instruct-2507 OUT=$HOME/rso_runs/sciworld_gtopsd \
  bash examples/rso_8gpu/run_sciworld_gtopsd_8gpu.sh
MODEL=Qwen/Qwen3-4B-Instruct-2507 OUT=$HOME/rso_runs/sciworld_rso_opsd \
  bash examples/rso_8gpu/run_sciworld_rso_opsd_8gpu.sh
```

- 曲线:`tensorboard --logdir $OUT/tensorboard`。训练中每 5 步 val 一次:textcraft = val100
  的 easy+medium 50 题;search = val_sub 45 题(per-source 三桶);sciworld = 固定 50 dev case
  (per-task 分桶,identify-life-stages 猝死族单列)。
- 健康检查(前 5 步):`episode/valid_action_ratio` > 0.9;递归另看
  `rso/delegating_trees` > 0、`rso/G_negative_ratio` ≡ 0。
- CPU 测试:`PYTHONPATH=$PWD python tests/test_rso_core.py` / `tests/test_search_rso.py` /
  `tests/test_sciworld_rso.py` 等。

## 四、全量测评

### TextCraft-Synth(632 题全难度)

```bash
# flat(1-3)与未训练基座
python scripts_rso/eval_full_val.py \
  --model <ckpt>/global_step_150/actor/huggingface --out <dir>/eval_full --split val --tp 2
# 递归(4-5)
python scripts_rso/eval_full_val.py \
  --model <ckpt>/global_step_150/actor/huggingface --out <dir>/eval_full --split val \
  --recursive --per-agent-steps 25 --max-depth 6 --max-steps 200 --tp 2
```

产出 `<out>_metrics.json`(总体/分难度成功率、平均轮数)+ `<out>_cases.jsonl`(每题逐轮轨迹);
同一 `--out` 断点续跑。请同时对未训练基座跑一次作 before/after 起点。

### Search-QA(test_3src.parquet,22,398 题 = musique 2,417 + 2wiki 12,576 + hotpotqa 7,405)

训练同款脚本 + 三项覆盖(ckpt 用 `MODEL=<ckpt>/global_step_*/actor/huggingface`):

```bash
MODEL=<ckpt_hf_dir> bash examples/rso_8gpu/run_search_gtopsd_8gpu.sh \
  data.val_files=$HOME/data/searchR1_processed_direct/test_3src.parquet \
  data.val_batch_size=512 trainer.val_only=True
# 递归同理换 run_search_rso_opsd_8gpu.sh
```

### ScienceWorld(test.parquet,官方 test 划分 1,819 变体)

```bash
MODEL=<ckpt_hf_dir> bash examples/rso_8gpu/run_sciworld_gtopsd_8gpu.sh \
  data.val_files=$HOME/data/sciworld/test.parquet \
  data.val_batch_size=128 trainer.val_only=True
# 递归同理换 run_sciworld_rso_opsd_8gpu.sh
```

指标:官方分(负分裁零)per-task 分桶 + main / sudden_death 宏平均;JVM 池 = val_batch_size,
分批复用,余数 batch 自动收缩。

## 五、参考基线结果(TextCraft-Synth)

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
