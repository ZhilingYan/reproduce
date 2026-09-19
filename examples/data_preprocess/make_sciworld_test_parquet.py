# -*- coding: utf-8 -*-
"""生成 ScienceWorld 官方 test 划分的全量评测 parquet(方案 b,2026-09-19 用户拍板)。

产物:~/data/sciworld/test.parquet —— 官方 `get_variations_test()` 全部变体
(30 任务共 1,819 行),每行 env_kwargs = {"task","variation"};环境按行加载
(env_package/sciworld/envs.py 的 case 指定模式),JVM 池大小 = val_batch_size,
分批复用,余数 batch 槽数自动收缩。

全量测试配方(与训练同口径,flat / recursive 各自的训练脚本加三项覆盖):
    data.val_files=$HOME/data/sciworld/test.parquet \\
    data.val_batch_size=128 \\
    trainer.val_only=True
指标沿训练中验证同款:官方分(负分裁零)per-task 分桶 + main/sudden_death 宏平均。

依赖:pip install scienceworld==1.3.0(java 11+);pandas。
"""
import os

import pandas as pd


def main():
    from scienceworld import ScienceWorldEnv
    env = ScienceWorldEnv("", envStepLimit=100)
    rows = []
    for t in sorted(env.get_task_names()):
        env.load(t, 0, "easy", generateGoldPath=False)
        for v in sorted(env.get_variations_test()):
            rows.append({
                "answer": "",
                "data_source": "sciworld",
                "prompt": [{"role": "user", "content": ""}],
                "ability": "agent",
                "extra_info": {"split": "test", "index": len(rows)},
                "env_kwargs": {"task": t, "variation": int(v)},
            })
    env.close()
    out_dir = os.path.expanduser("~/data/sciworld")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "test.parquet")
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"written {out}: {len(rows)} rows (官方 test 变体全量)")


if __name__ == "__main__":
    main()
