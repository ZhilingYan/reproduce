# -*- coding: utf-8 -*-
"""导出训练进展 csv(TASK 交付 1):从训练输出的 tensorboard 目录读 scalar,按 step 合并成宽表。

用法:
    python scripts_rso/export_train_progress.py --logdir $OUT/tensorboard --out train_progress.csv

每行一个训练 step;列 = step + 下列存在的指标(方法不同,存在的列不同,缺的留空):
    episode/*_success_rate      训练 rollout 的成功率(每步都有)
    val/*_success_rate          训练中 val 的成功率(每 test_freq=5 步一次,其余行空)
    rao|rso/root_reward_mean    递归方法的 root 成功率
    rao|rso/valid_action_ratio  动作合规率(健康检查)
    actor/entropy_loss          熵(健康检查:不应单调上行)
"""
from __future__ import annotations

import argparse
import csv

TAGS = [
    "episode/success_rate", "episode/easy_success_rate", "episode/medium_success_rate",
    "episode/easymedium_success_rate",
    "val/success_rate", "val/easy_success_rate", "val/medium_success_rate",
    "val/easymedium_success_rate",
    "rao/root_reward_mean", "rso/root_reward_mean",
    "rao/valid_action_ratio", "rso/valid_action_ratio",
    "actor/entropy_loss",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True, help="训练输出的 tensorboard 目录($OUT/tensorboard)")
    ap.add_argument("--out", required=True, help="输出 csv 路径")
    args = ap.parse_args()

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(args.logdir, size_guidance={"scalars": 0})
    ea.Reload()
    present = [t for t in TAGS if t in ea.Tags()["scalars"]]
    if not present:
        raise SystemExit(f"[export] {args.logdir} 里没有任何目标指标;确认路径是 $OUT/tensorboard")

    rows: dict[int, dict[str, float]] = {}
    for tag in present:
        for ev in ea.Scalars(tag):
            rows.setdefault(int(ev.step), {})[tag] = ev.value

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step"] + present)
        for step in sorted(rows):
            w.writerow([step] + [rows[step].get(t, "") for t in present])
    print(f"[export] {len(rows)} 步 × {len(present)} 指标 → {args.out}")


if __name__ == "__main__":
    main()
