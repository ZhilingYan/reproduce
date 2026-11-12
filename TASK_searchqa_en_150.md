# TASK: Search-QA continuation training (gemma-2b, step 50 → step 150)

**Goal.** Resume each of the four gemma-2b Search-QA runs from its **step 50** checkpoint and continue training until **step 150** with exactly the same settings. Then run full evaluation on the new best checkpoint and deliver the same files as before.

First `git pull` to the latest. Installation, retrieval service, training and full-evaluation commands are the same as in [`TASK_searchqa_en.md`](TASK_searchqa_en.md) and [`README.md`](README.md). Do not change any hyperparameter, prompt, decoding setting, data file or retrieval setting. The only differences from the original runs are the resume flags and the step count.

## Scope

| Method | Source run (DONE, 50 steps) | Resume from | New experiment name |
|---|---|---|---|
| rao | gemma-2b_searchqa_rao | `global_step_50` | **TODO** gemma-2b_searchqa_rao_s150 |
| skill | gemma-2b_searchqa_skill | `global_step_50` | **TODO** gemma-2b_searchqa_skill_s150 |
| gtopsd | gemma-2b_searchqa_gtopsd | `global_step_50` | **TODO** gemma-2b_searchqa_gtopsd_s150 |
| grpo | gemma-2b_searchqa_grpo | `global_step_50` | **TODO** gemma-2b_searchqa_grpo_s150 |

Priority order: rao → skill → gtopsd → grpo. gemma-26b is out of scope for this task.

Always resume from `global_step_50` (the latest checkpoint), **not** from `ckpts/best`. For skill, gtopsd and grpo the best checkpoint is an earlier step (30, 40 and 45), but the continuation must start from step 50 so that the curve from step 1 to step 150 is one continuous run.

## Step 0: Preflight checks (per method, before launching)

The 50-step runs live under the previous output root, for example `/workspace/xda161/reproduce-searchqa-e2b-r001/runs/<NAME>`. Set `PREV_OUT` to that directory.

```bash
NAME=gemma-2b_searchqa_rao            # source run
PREV_OUT=/workspace/xda161/reproduce-searchqa-e2b-r001/runs/$NAME
CK=$PREV_OUT/ckpts/global_step_50
ls $CK/actor                           # must contain model, optimizer and extra (lr scheduler, RNG) shards for every rank
ls $CK/data.pt                         # dataloader state, required so that steps 51+ see new batches instead of replaying steps 1..50
```

Stop and report if any of the following is true:

* `global_step_50/actor` has model weights only (no optimizer or extra state). Resuming without optimizer state is not an exact continuation.
* `data.pt` is missing. Without it the dataloader restarts from the first batch.
* For rao: the step 50 checkpoint was repaired after the original exit code 1. Confirm that the repaired folder still has complete optimizer, extra and `data.pt` files, not only the exported HuggingFace weights.

Notes for the executor (no action needed):

* Learning rate stays constant at 1e-6. The schedule is `warmup_style: constant` with `lr_warmup_steps_ratio=0.1`, so warmup ends at step 15 when `total_training_steps=150`, before the resume point.
* The training set has 34,938 prompts and each step uses 128, so 150 steps stay inside the first epoch. No prompt is repeated.

## Step 1: Launch the continuation

Write all new outputs to a **new** directory so that the 50-step run, its checkpoints and its deliverables stay untouched.

```bash
SRC=gemma-2b_searchqa_rao             # source run
METHOD=rao
NAME=${SRC}_s150
PREV_OUT=/workspace/xda161/reproduce-searchqa-e2b-r001/runs/$SRC
OUT=$HOME/rso_runs/$NAME
mkdir -p $OUT
MODEL=google/gemma-4-E2B-it OUT=$OUT \
  bash examples/rso_8gpu/run_search_${METHOD}_8gpu.sh \
    trainer.experiment_name=$NAME \
    trainer.total_training_steps=150 \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=$PREV_OUT/ckpts/global_step_50 \
    trainer.val_before_train=False
```

Requirements:

* `MODEL` stays the base model id. The actor, optimizer, scheduler and dataloader state come from `resume_from_path`.
* `resume_from_path` must be an absolute path that ends in `global_step_50`.
* Keep the same number of nodes and GPUs as the original run (2 nodes × 4 GPUs for these runs). The FSDP sharded optimizer state must be loaded with the same world size.
* `trainer.val_before_train=False` is required. With it set to True, the in-memory best tracker records the step 50 validation as pending and then labels the step 55 checkpoint as best even if step 55 validates lower. Validation still runs at steps 55, 60, ..., 150.
* Do not pass `trainer.total_training_steps=50`.

Confirm in the first lines of the log:

```
Load from checkpoint folder: .../global_step_50
Setting global step to 50
```

and that the first training step printed is `step:51`. If the log says `Training from scratch` or the first step is `step:1`, stop immediately.

## Step 2: Health check (steps 51 to 55)

Same gate as the original task, applied to steps 51..55:

* `episode/valid_action_ratio` > 0.9 for flat methods, `rao/valid_action_ratio` > 0.9 for rao
* `response/aborted_ratio` == 0
* exactly 1,024 trajectories per step
* rao only: `rao/delegating_trees` > 0
* No sudden jump at step 51 relative to step 50 in `actor/entropy`, `actor/ppo_kl`, `actor/grad_norm` or (flat methods) `actor/kl_loss`. A large jump usually means optimizer or reference state was not restored. Compare against step 50 in the original `job.log`.

Also monitor during the run and report if seen:

* `actor/grad_norm` above 50 (grpo logged 133.5 at step 50 in the original run)
* rao `rao/adv_abs_max` above 500

## Step 3: Full evaluation

Checkpoint policy is unchanged: validate every 5 steps, keep the latest checkpoint and the best-val checkpoint. In the new `$OUT/ckpts`, `best` points to the best validation step among **55..150** of the continuation.

Run full evaluation on two checkpoints per method:

1. `$OUT/ckpts/best` (main result, same rule as the original task)
2. `$OUT/ckpts/global_step_150` (final step, reported as a secondary row because the 45-case validation set is noisy)

```bash
# (1) best of 55..150
EVAL_MODEL=$OUT/ckpts/best/actor/huggingface
NAME=$NAME BENCH=searchqa MODEL=$EVAL_MODEL OUT=$OUT SCRIPT=examples/rso_8gpu/run_search_${METHOD}_8gpu.sh \
  bash scripts_rso/run_full_eval.sh data.val_files=$HOME/data/searchR1_processed_direct/test_3src.parquet data.val_batch_size=512

# (2) step 150
EVAL_MODEL=$OUT/ckpts/global_step_150/actor/huggingface
NAME=${NAME}_step150 BENCH=searchqa MODEL=$EVAL_MODEL OUT=$OUT SCRIPT=examples/rso_8gpu/run_search_${METHOD}_8gpu.sh \
  bash scripts_rso/run_full_eval.sh data.val_files=$HOME/data/searchR1_processed_direct/test_3src.parquet data.val_batch_size=512
```

If `best` resolves to `global_step_150`, run evaluation (1) only and note it in the report.

Use the same retrieval setup as the r002 evaluation (exhaustive CPU float32). Confirm the retrieval service is stable before each run.

## Deliverables

**Deliverable 1: training progress csv, steps 1..150**

```bash
python scripts_rso/export_train_progress.py --logdir $OUT/tensorboard --out ${NAME}_train_progress_51_150.csv
```

Then concatenate it with the original 50-step CSV (`gemma-2b_searchqa_<method>_train_progress.csv`) into `${NAME}_train_progress.csv` with steps 1..150, one row per step, same columns. Check that there are exactly 150 rows and no duplicate step. Produce the matching `_audit.json` the same way as for r002.

**Deliverable 2: full evaluation**

* `${NAME}_cases.jsonl` (best of 55..150) and `${NAME}_step150_cases.jsonl`
* `searchqa_report.md` regenerated with all rows: the four original 50-step methods, the four `_s150` best rows, the four `_s150_step150` rows, and the RSO reference row
* In the closing Analysis section, report for each method the difference between the 50-step result and the `_s150` result, the selected best step, and the rao delegation rate at evaluation

**Deliverable 3: package**

```bash
mkdir -p deliver && cp *_cases.jsonl *_train_progress.csv *_audit.json deliver/
python scripts_rso/make_eval_report.py --cases-dir deliver --out-dir deliver
tar czf searchqa_e2b_s150_deliverables.tar.gz deliver/
```

Include the master `job.log` and first-five-step gate JSON of each continuation run, as in the r002 log bundle.
