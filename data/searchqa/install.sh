#!/usr/bin/env bash
# 把随仓的 Search-QA 成品数据落位到训练/测评脚本的缺省路径(一次即可):
#   ~/data/searchR1_musique_2wiki/{train,val_sub,val_smoke}.parquet + decomp_store.json
#   ~/data/searchR1_processed_direct/test_3src.parquet
# train.parquet 已含 2wiki 的 question_decomposition 回填;musique 部分来自官方源
# (StonyBrookNLP/musique,与旧 FlashRAG 源全量对拍零差异)。
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p ~/data/searchR1_musique_2wiki ~/data/searchR1_processed_direct
cp "$HERE"/train.parquet "$HERE"/val_sub.parquet "$HERE"/val_smoke.parquet "$HERE"/decomp_store.json \
   ~/data/searchR1_musique_2wiki/
cp "$HERE"/test_3src.parquet ~/data/searchR1_processed_direct/
echo "done: ~/data/searchR1_musique_2wiki/{train(34,938 行),val_sub(45),val_smoke(9),decomp_store}"
echo "      ~/data/searchR1_processed_direct/test_3src.parquet(22,398 行)"
