# -*- coding: utf-8 -*-
"""生成 ScienceWorld 训练用的假 parquet(批次驱动用,内容不被环境消费)。

textcraft 同款做法(~/data/verl-agent/text_tc_full/train.parquet,16 行,
schema: answer/data_source/prompt/ability/extra_info):任务由环境自采
(官方 train 划分任务均匀采样;val = 固定 50 case,见 env_package/sciworld/envs.py),
parquet 只决定 dataloader 的行数与 batch 节奏。

用法:python examples/data_preprocess/make_sciworld_dummy_parquet.py
产物:~/data/sciworld/train.parquet(16 行)、~/data/sciworld/val.parquet(50 行)。
"""
import os

import pandas as pd


def _rows(n: int, split: str):
    return [{
        "answer": "",
        "data_source": "sciworld",
        "prompt": [{"role": "user", "content": ""}],
        "ability": "agent",
        "extra_info": {"split": split, "index": i},
    } for i in range(n)]


def main():
    out_dir = os.path.expanduser("~/data/sciworld")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(_rows(16, "train")).to_parquet(os.path.join(out_dir, "train.parquet"))
    pd.DataFrame(_rows(50, "val")).to_parquet(os.path.join(out_dir, "val.parquet"))
    # 冒烟版 val:行数必须 = val_batch_size = val 环境槽数(单 batch 不变量;
    # 50 行 ÷ batch 8 的余数 2 曾撞 rollout_loop.py:431 断言,冒烟 22202209)
    pd.DataFrame(_rows(8, "val")).to_parquet(os.path.join(out_dir, "val_smoke.parquet"))
    print(f"written {out_dir}/train.parquet (16), val.parquet (50), val_smoke.parquet (8)")


if __name__ == "__main__":
    main()
