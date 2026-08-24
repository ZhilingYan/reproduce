#!/usr/bin/env python3
"""生成 TextCraft-Synth 训练所需的占位 parquet。

为什么需要它:verl 的 dataloader 必须有一个数据文件来决定"每个 epoch 多少步、
每步多少样本",但 TextCraft-Synth 的真实任务是在 env.reset() 时由环境从
agent_system/environments/env_package/textcraft_synth/data/*.jsonl 里取的,
parquet 里的 prompt 内容【完全不被使用】。所以这里只造结构正确的空壳。

行数的含义:
  train.parquet 行数 = data.train_batch_size  ->  1 步/epoch
      配合 trainer.total_epochs=150 得到总共 150 个训练步。
  test.parquet  行数 = data.val_batch_size    ->  验证环境个数
      必须与验证任务池大小一致(val100 子集 = 100)。

用法:
  python scripts_rso/prepare_synth_parquet.py --out ~/data/verl-agent/synth_full/text
"""
import argparse
import os

import pandas as pd


def make_rows(n: int, split: str):
    return pd.DataFrame([
        {
            "answer": "3",
            "data_source": "text",
            "prompt": [{"role": "user", "content": ""}],  # 占位:真实 prompt 由 env 生成
            "ability": "agent",
            "extra_info": {"split": split, "index": i},
        }
        for i in range(n)
    ])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.expanduser("~/data/verl-agent/synth_full/text"))
    p.add_argument("--train-rows", type=int, default=16, help="= data.train_batch_size")
    p.add_argument("--test-rows", type=int, default=100, help="= data.val_batch_size (val100 子集)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    make_rows(args.train_rows, "train").to_parquet(os.path.join(args.out, "train.parquet"))
    make_rows(args.test_rows, "test").to_parquet(os.path.join(args.out, "test.parquet"))
    print(f"written: {args.out}/train.parquet ({args.train_rows} rows), "
          f"test.parquet ({args.test_rows} rows)")


if __name__ == "__main__":
    main()
