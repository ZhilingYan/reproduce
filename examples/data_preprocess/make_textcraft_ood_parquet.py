# -*- coding: utf-8 -*-
"""生成 textcraft_ood(真 TextCraft / ADaPT 官方)OOD 推理用的 val parquet(dummy 行)。

真 textcraft 的 val 任务不由 parquet 内容驱动:worker 池在 is_train=False 时
按官方式 test 划分持有全部 84 个目标(envs.py compute_item_split,depth>=2 每 5 抽 1),
批 reset 以不重复随机抽满槽位——env_num = 池大小时即全覆盖。parquet 只需要行数
与 data.val_batch_size 相等(verl val 循环按行数驱动),schema 照 synth test.parquet。

用法:
    python examples/data_preprocess/make_textcraft_ood_parquet.py [--out-dir ~/data/textcraft_ood]
产物:
    test.parquet(84 行 = 官方式 test 全量;val_batch_size=84 时全覆盖)
    test_head40.parquet(40 行冒烟子集;val_batch_size=40,随机确定性抽 40 题)
"""
import argparse
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.realpath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent_system.environments.env_package.textcraft.envs import (  # noqa: E402
    compute_item_split)

_TC_PKG = os.path.join(_ROOT, "agent_system", "environments", "env_package",
                       "textcraft", "textcraft_pkg")


def _dummy_rows(n, split_name):
    return pd.DataFrame({
        "answer": ["" for _ in range(n)],
        "data_source": ["textcraft_ood" for _ in range(n)],
        "prompt": [[{"role": "user", "content": ""}] for _ in range(n)],
        "ability": ["agent" for _ in range(n)],
        "extra_info": [{"split": split_name, "index": i} for i in range(n)],
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.expanduser("~/data/textcraft_ood"))
    args = ap.parse_args()

    _, test_items = compute_item_split(_TC_PKG)
    n = len(test_items)
    print(f"官方式 test 划分:{n} 个目标物品(depth>=2 每 5 抽 1)")

    os.makedirs(args.out_dir, exist_ok=True)
    full = os.path.join(args.out_dir, "test.parquet")
    head = os.path.join(args.out_dir, "test_head40.parquet")
    _dummy_rows(n, "test").to_parquet(full)
    _dummy_rows(min(40, n), "test_head40").to_parquet(head)
    print(f"写出 {full}({n} 行)与 {head}({min(40, n)} 行)")


if __name__ == "__main__":
    main()
