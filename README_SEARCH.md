# Search-QA 域:安装、数据、训练(2026-09-19 新增,不影响 TextCraft 一切现有内容)

TextCraft 的安装与任务书照旧(README.md / TASK.md);本文只讲 Search-QA 新增部分。
方法设计与参数出处:`RSO_searchqa_data_mapping.md` 与 `flat参数对照_textcraft_vs_searchqa.md`
(随主仓 Ideation 维护,冲突以参数表为准)。

## 1. 额外依赖与检索服务

Search 环境经 HTTP 调本地 e5 检索服务(wiki-18 语料)。索引与服务安装按 Search-R1 官方流程:

```bash
# 检索服务环境(faiss 建议独立 conda 环境;GPU faiss 在 H100 可用,H200/sm90 需 CPU faiss)
conda create -n retriever python=3.10 -y && conda activate retriever
conda install numpy==1.26.4 && pip install torch transformers datasets pyserini huggingface_hub uvicorn fastapi
conda install faiss-gpu==1.8.0 -c pytorch -c nvidia -y   # 或 faiss-cpu

# 下载 wiki-18 索引与语料(约 75GB)
local_dir=~/data/searchR1
python examples/search/searchr1_download.py --local_dir $local_dir
cat $local_dir/part_* > $local_dir/e5_Flat.index && gzip -d $local_dir/wiki-18.jsonl.gz

# 起服务(默认 http://0.0.0.0:8000/retrieve;训练脚本经 SEARCH_URL 指向它):
bash examples/search/retriever/retrieval_launch.sh > retrieval_server.log &
```

训练环境本体与 TextCraft 共用(无新增 python 依赖;skyrl_gym 以仓内子包形式引用,无需安装)。

## 2. 数据管线(四步,产物落 ~/data/)

```bash
# ① Search-R1 官方 7 子集测试集(train 部分本方案不用):
python examples/data_preprocess/preprocess_search_r1_dataset.py     # → ~/data/searchR1_processed_direct/

# ② MuSiQue+2Wiki 训练集(先按脚本头注释下载 FlashRAG 两个 train.jsonl 到 $FLASHRAG_RAW):
python examples/data_preprocess/preprocess_musique_2wiki_train.py   # → ~/data/searchR1_musique_2wiki/train.parquet

# ③ 2Wiki evidences → question_decomposition 回填(先按脚本头注释取官方 data_ids 到 $WIKI2_DATA_IDS):
python examples/data_preprocess/convert_2wiki_evidences_to_decomp.py

# ④ 验证子集(7源×15=105)与 decomp/别名库:
python examples/data_preprocess/make_searchrso_data_products.py     # → val_sub.parquet + decomp_store.json
```

## 3. 训练(与 TextCraft 同一套环境变量约定)

```bash
# flat GRPO+OPSD(特权 = gt answer;SDAR search 原口径 + 参数表三处定稿差异)
MODEL=Qwen/Qwen3-4B-Instruct-2507 TP=1 OUT=$HOME/rso_runs/search_gtopsd \
  bash examples/rso_8gpu/run_search_gtopsd_8gpu.sh

# recursive RSO+OPSD(递归委托 + Φ 进展优势 + 节点局部特权蒸馏)
MODEL=Qwen/Qwen3-4B-Instruct-2507 TP=2 OUT=$HOME/rso_runs/search_rso_opsd \
  bash examples/rso_8gpu/run_search_rso_opsd_8gpu.sh
```

健康检查(前 5 步):`rso/valid_action_ratio` > 0.9;递归另看 `rso/delegating_trees` > 0、
`rso/G_negative_ratio` ≡ 0;验证指标为 per-source(`val/musique_success_rate` 等 7 桶)。

## 4. 与 TextCraft 的隔离说明

- 新增:`env_package/search`(SDAR 原环境 + gt 通道)、`env_package/search_rso`(递归)、
  `prompts/search*.py`、`verl/trainer/main_rso_opsd_search.py`、`skills/search`、本文与两个脚本;
- 共享文件仅两个有改动:`rso_opsd_core.py` / `rso_opsd_ray_trainer.py` 的 act_mask 标签集
  参数化(缺省 `["action"]`,TextCraft 行为逐字节不变;`tests/` 既有守卫测试可复核);
- CPU 测试:`python tests/test_search_rso.py`(免 pytest,内置 runner)。

## 5. 已知口径备忘

- 推理标签是 `<thought>`(prompts/search.py 头注释有据:Qwen3 词表里 `<think>` 是特殊
  token,模型会无视该推理指令;SDAR 原配 Qwen2.5 无此问题);
- 训练/验证无泄漏:训练用两家官方 train split,test.parquet 里的 musique/2wiki 是官方 dev;
- Φ 只认"承诺"事件(子 agent `<answer>` / 任一节点 `<search>`、`<delegate>` 文本含某跳答案,
  别名扩展),检索返回内容与 `<thought>` 文本不产生进展。
