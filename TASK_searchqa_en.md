# TASK — Search-QA experiments

**Only run the experiments marked TODO. Cells marked DONE already carry their scores and must not be rerun.**

First `git pull` to the latest. Installation, data, training and full-evaluation commands are in [`README.md`](README.md). All reported scores are full evaluations of the best-val checkpoint of each run. `runN / stepK` denotes the training source.

## Models (2)

| Code | `MODEL=` | Notes |
|---|---|---|
| gemma-2b | `google/gemma-4-E2B-it` | Needs its own environment: transformers ≥ 5.x + a recent vllm that supports gemma-4 |
| gemma-26b | `google/gemma-4-26B-A4B-it` | Needs its own environment: transformers ≥ 5.x + a recent vllm that supports gemma-4. Recommended TP=4, MICRO_BSZ=1 |

Remember to disable thinking.

gemma-26b requires 2 nodes × 8 H100 (16 GPUs total) for training.

## Retrieval service (start before any training or evaluation)

Search-QA training and evaluation both call a local e5 retrieval server (wiki-18 corpus + e5 index, about 75GB). Set it up as described in README §2 "Search-QA":

```bash
bash data/searchqa/install.sh   # train 34,938 / val_sub 45 / val_smoke 9 / decomp_store.json / test_3src 22,398
conda activate retriever
bash examples/search/fetch_retrieval_from_release.sh      # → ~/data/searchR1/{e5_Flat.index, wiki-18.jsonl}
bash examples/search/retriever/retrieval_launch.sh > retrieval_server.log &   # http://0.0.0.0:8000/retrieve
```

The scripts read the service address from `SEARCH_URL` (default `http://0.0.0.0:8000/retrieve`).

The retrieval service has historically been the common bottleneck of this block (it was once stopped by the platform for low utilization, and has returned HTTP 400). Confirm the service is stable before any long run.

## Experiment name = bash argument

Name every run `NAME=<model-code>_<bench>_<method>` (deliverable files use the same name):

```bash
NAME=<experiment-name> ; MODEL=<model> OUT=$HOME/rso_runs/$NAME \
  bash examples/rso_8gpu/run_search_<method>_8gpu.sh trainer.experiment_name=$NAME trainer.total_training_steps=50
```

Train for 50 steps only. The scripts default to `trainer.total_training_steps=150`, so pass the override above.

Method codes and scripts: rso / rao / skill / gtopsd / grpo (see the matrix in README §3. RAO is the official recursive baseline with all parameters aligned to RSO). Run the pending methods in the priority order rao → skill → gtopsd → grpo.

### 2. Search-QA (train on 2wiki + musique. Full evaluation on 2wiki + musique + hotpotqa (OOD) following the README §4 val_only recipe. Start the retrieval service first. Test set 22,398 questions, success rate %)

| Method | gemma-2b | gemma-26b |
|---|---|---|
| rso | **DONE** 34.05 (7,626/22,398, run245/step30) | **DONE** 35.47 (7,945/22,398, run247/step5) |
| rao | **TODO** gemma-2b_searchqa_rao | **TODO** gemma-26b_searchqa_rao |
| skill | **TODO** gemma-2b_searchqa_skill | **TODO** gemma-26b_searchqa_skill |
| gtopsd | **TODO** gemma-2b_searchqa_gtopsd | **TODO** gemma-26b_searchqa_gtopsd |
| grpo | **TODO** gemma-2b_searchqa_grpo | **TODO** gemma-26b_searchqa_grpo |

> rso per-source breakdown: E2B = 2wiki 36.17 / musique 17.05 / hotpotqa 35.99. 26B = 2wiki 34.91 / musique 18.58 / hotpotqa 41.94. A step50 follow-up evaluation for 26B is unfinished and does not affect this table.

The test set `test_3src.parquet` contains 22,398 questions = musique 2,417 + 2wiki 12,576 + hotpotqa 7,405. In-training validation uses `val_sub.parquet` (15 fixed cases from each of the three sources, 45 total).

Checkpoint policy (same for all training runs, already configured in the scripts): validate every 5 steps + keep the latest checkpoint (1 copy) + automatically keep the best-val checkpoint (`$OUT/ckpts/best`). Full evaluation always uses best.

Health check over the first 5 steps (all training runs): `episode/valid_action_ratio` > 0.9. For recursive methods, additionally check `rso/delegating_trees` > 0 and `rso/G_negative_ratio` ≡ 0. Stop and investigate on any anomaly.

Full evaluation (after training, always on the best-val checkpoint, with the retrieval service running. The script runs the existing full-evaluation command from README §4 with identical questions and decoding settings, and additionally dumps per-case results into the deliverable `${NAME}_cases.jsonl`, written to the current directory. `<method>` is the same as in training):

```bash
MODEL=$HOME/rso_runs/$NAME/ckpts/best/actor/huggingface
NAME=<experiment-name> BENCH=searchqa MODEL=$MODEL OUT=$HOME/rso_runs/$NAME SCRIPT=examples/rso_8gpu/run_search_<method>_8gpu.sh \
  bash scripts_rso/run_full_eval.sh data.val_files=$HOME/data/searchR1_processed_direct/test_3src.parquet data.val_batch_size=512
```

For experiments whose full evaluation was already run with the old command but lacks `${NAME}_cases.jsonl`, rerun once with the best checkpoint using the command above (inference only, no retraining).

## Deliverables (file names start with $NAME or the bench name)

**Deliverable 1: training progress csv**:

```bash
python scripts_rso/export_train_progress.py --logdir $OUT/tensorboard --out ${NAME}_train_progress.csv
```

**Deliverable 2: full evaluation results** (metric definitions in `Ideation/ideas/RSO_evaluation_metrics.md`):

| Granularity | File | Content | How produced |
|---|---|---|---|
| Per experiment | `${NAME}_cases.jsonl` | Per-case results: one line per case, with all per-case metrics of the benchmark (fields below) | produced automatically by `run_full_eval.sh` |
| Per benchmark | `<bench>_report.md` | One comparison table per model (rows = methods, columns = aggregate metrics) + bucket tables + factual summary + closing analysis | tables and summary generated by `make_eval_report.py`. The final "Analysis" section is written by the executor |
| All | `rso_deliverables.tar.gz` | Everything above + Deliverable 1 | command below |

Per-case fields in `${NAME}_cases.jsonl`:

| benchmark | fields |
|---|---|
| common to all | `name` `bench` `subset` `case_id` `success` `steps_total` `gen_tokens` `max_node_ctx` `is_recursive` `delegated` `n_delegations`. Recursive methods additionally `steps_root` `n_nodes` `tree_depth` |
| searchqa | `data_source` `question_type` `hops` `em` `f1` `subem` `answer` `golds` `n_searches` (only musique / 2wikimultihopqa / hotpotqa) |

Aggregation and packaging (run once after all per-case files are collected):

```bash
mkdir -p deliver && cp *_cases.jsonl *_train_progress.csv deliver/
python scripts_rso/make_eval_report.py --cases-dir deliver --out-dir deliver   # one <bench>_report.md per bench
# Open deliver/<bench>_report.md and write the closing "Analysis" section, citing numbers from the tables for every conclusion
tar czf rso_deliverables.tar.gz deliver/
```
