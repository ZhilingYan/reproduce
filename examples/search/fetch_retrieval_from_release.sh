#!/usr/bin/env bash
# 从本仓 GitHub Release 下载检索资产(wiki-18 语料 + e5 索引,切片 ≤1.9G),
# 校验 sha256 后重组到 $SAVE_PATH(缺省 ~/data/searchR1)。合作者唯一需要的命令:
#   bash examples/search/fetch_retrieval_from_release.sh
# 可调:GH_REPO(缺省本仓)/ RELEASE_TAG(缺省 retrieval-v1)/ SAVE_PATH。
set -e
GH_REPO=${GH_REPO:-ZhilingYan/reproduce}
RELEASE_TAG=${RELEASE_TAG:-retrieval-v1}
SAVE_PATH=${SAVE_PATH:-$HOME/data/searchR1}
BASE="https://github.com/$GH_REPO/releases/download/$RELEASE_TAG"
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$SAVE_PATH" && cd "$SAVE_PATH"

echo "[fetch] manifest..."
curl -fL -o MANIFEST.sha256 "$BASE/MANIFEST.sha256"
# 与仓内提交的副本比对,防 release 被替换
if ! diff -q MANIFEST.sha256 "$HERE/RETRIEVAL_MANIFEST.sha256" >/dev/null; then
  echo "[fetch] WARNING: release manifest 与仓内副本不一致,以仓内为准继续校验"
  cp "$HERE/RETRIEVAL_MANIFEST.sha256" MANIFEST.sha256
fi

for f in $(awk '{print $2}' MANIFEST.sha256); do
  if [ -f "$f" ] && sha256sum -c <(grep " $f\$" MANIFEST.sha256) >/dev/null 2>&1; then
    echo "[fetch] 已有且校验通过: $f"; continue
  fi
  echo "[fetch] $f"
  curl -fL --retry 5 --retry-delay 10 -C - -o "$f" "$BASE/$f"
done
echo "[fetch] 全量校验..."
sha256sum -c MANIFEST.sha256

echo "[fetch] 重组..."
cat e5_Flat.index.part_* > e5_Flat.index
cat wiki-18.jsonl.gz.part_* | gunzip > wiki-18.jsonl
rm -f e5_Flat.index.part_* wiki-18.jsonl.gz.part_* MANIFEST.sha256
ls -lh e5_Flat.index wiki-18.jsonl
echo "[fetch] done → $SAVE_PATH;起服务:bash examples/search/retriever/retrieval_launch.sh"
