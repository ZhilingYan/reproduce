#!/usr/bin/env bash
# 把随仓的 textcraft_ood 任务 parquet 落位到脚本缺省路径(一次即可):
#   ~/data/textcraft_ood/test.parquet(84 行 dummy,任务由环境按官方式 test 划分生成)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p ~/data/textcraft_ood
cp "$HERE"/test.parquet ~/data/textcraft_ood/
echo "done: ~/data/textcraft_ood/test.parquet(84 行)"
