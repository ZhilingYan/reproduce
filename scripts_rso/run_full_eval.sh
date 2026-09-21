#!/usr/bin/env bash
# 训练完毕后的全量测评,产出交付用的逐题指标文件 <NAME>_cases.jsonl(每题一行、带齐该 benchmark 的全部逐题指标)。
#
# 与现有全量测评的关系:跑的就是 README 第四节的现有命令,温度、种子、题目都不改,
# val 路径只多加一个逐题落盘开关(+trainer.val_case_dump_dir),所以现有指标照旧产出、数值不变;
# 逐题指标由 scripts_rso/compute_eval_metrics.py 从同一次运行的逐题记录算出。
#
# 用法(val 路径:searchqa / sciworld / textcraft_ood。SCRIPT 与现有命令里的 run_*_8gpu.sh 相同,
# 其后的参数也与现有命令相同;trainer.val_only=True 由本脚本自动追加):
#   NAME=<实验名> BENCH=sciworld MODEL=<ckpt>/best/actor/huggingface OUT=$HOME/rso_runs/$NAME \
#   SCRIPT=examples/rso_8gpu/run_sciworld_rso_8gpu.sh \
#     bash scripts_rso/run_full_eval.sh data.val_files=$HOME/data/sciworld/test.parquet data.val_batch_size=128
#
# 用法(textcraft_synth:其后的参数与现有 eval_full_val.py 命令相同,不带 --out):
#   NAME=<实验名> BENCH=textcraft_synth MODEL=<ckpt>/best/actor/huggingface OUT=$HOME/rso_runs/$NAME \
#     bash scripts_rso/run_full_eval.sh --split val --tp 2 [--recursive --per-agent-steps 25 --max-depth 6 --max-steps 200]
#
# 产出:当前目录下 <NAME>_cases.jsonl(交付物);$OUT/full_eval/ 下是原始逐题记录与完整日志。
set -uo pipefail
: "${NAME:?需要 NAME}"; : "${BENCH:?需要 BENCH(searchqa|sciworld|textcraft_ood|textcraft_synth)}"
: "${MODEL:?需要 MODEL}"; : "${OUT:?需要 OUT}"
case "$BENCH" in searchqa|sciworld|textcraft_ood|textcraft_synth) ;; *) echo "未知 BENCH=$BENCH"; exit 2 ;; esac
HERE=$(cd "$(dirname "$0")/.." && pwd)
EV=$OUT/full_eval
mkdir -p "$EV"
USER_ARGS=("$@")

run_val() {   # $1=标签;其余=该次额外覆盖项
  local tag=$1; shift
  ray stop --force >/dev/null 2>&1 || true
  rm -f "$EV/$tag/cases.jsonl"                     # 逐题落盘是追加写,重跑前清掉旧文件
  echo "=== [full_eval] $NAME $tag $(date) ==="
  MODEL=$MODEL OUT=$EV/$tag TENSORBOARD_DIR=$EV/$tag/tensorboard \
    bash "$HERE/$SCRIPT" trainer.experiment_name=$NAME trainer.val_only=True \
    "${USER_ARGS[@]}" +trainer.val_case_dump_dir=$EV/$tag "$@" 2>&1 | tee "$EV/$tag.log"
  local rc=${PIPESTATUS[0]}
  echo "=== [full_eval] $tag rc=$rc ==="
  return $rc
}

if [ "$BENCH" = "textcraft_synth" ]; then
  echo "=== [full_eval] $NAME $(date) ==="
  python3 "$HERE/scripts_rso/eval_full_val.py" --model "$MODEL" --out "$EV/main" "${USER_ARGS[@]}" \
    2>&1 | tee "$EV/main.log"
  rc=${PIPESTATUS[0]}
  RUN="$EV/main_cases.jsonl"
else
  : "${SCRIPT:?需要 SCRIPT(现有全量测评命令里的 run_*_8gpu.sh)}"
  run_val main; rc=$?
  RUN="$EV/main"
fi
if [ $rc -ne 0 ]; then echo "[full_eval] 全量测评失败(rc=$rc),不生成逐题指标"; exit $rc; fi

python3 "$HERE/scripts_rso/compute_eval_metrics.py" --bench "$BENCH" --run "$RUN" \
  --name "$NAME" --out-dir . > "$EV/summary.json"
echo "=== [full_eval] 交付:$(pwd)/${NAME}_cases.jsonl;汇总预览 $EV/summary.json;完整日志 $EV/*.log ==="
