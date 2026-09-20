#!/usr/bin/env bash
# 维护者用:把切片好的检索资产上传成 GitHub Release(需对仓库有写权限的 token)。
#   GITHUB_TOKEN=<token> STAGE=<切片目录> bash examples/search/upload_retrieval_release.sh
# STAGE 目录须含 e5_Flat.index.part_* / wiki-18.jsonl.gz.part_* / MANIFEST.sha256
#(制作方式:gzip 语料;两文件各 `split -b 1900M -d -a 3`;`sha256sum *.part_* > MANIFEST.sha256`)。
# Release 附件单文件上限 2GB,下载免费不限量,不进 git 历史。
set -e
: "${GITHUB_TOKEN:?需要 GITHUB_TOKEN(repo 写权限)}"
: "${STAGE:?需要 STAGE=切片目录}"
GH_REPO=${GH_REPO:-ZhilingYan/reproduce}
RELEASE_TAG=${RELEASE_TAG:-retrieval-v1}
API="https://api.github.com/repos/$GH_REPO"
AUTH="Authorization: Bearer $GITHUB_TOKEN"

echo "[upload] 确认/创建 release $RELEASE_TAG ..."
RID=$(curl -fsS -H "$AUTH" "$API/releases/tags/$RELEASE_TAG" 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])" 2>/dev/null || true)
if [ -z "$RID" ]; then
  RID=$(curl -fsS -X POST -H "$AUTH" "$API/releases" \
    -d "{\"tag_name\":\"$RELEASE_TAG\",\"name\":\"Search-QA retrieval assets (wiki-18 + e5 index)\",\"body\":\"cat parts per examples/search/fetch_retrieval_from_release.sh\"}" \
    | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")
fi
echo "[upload] release id = $RID"

EXISTING=$(curl -fsS -H "$AUTH" "$API/releases/$RID/assets?per_page=100" | python3 -c "import sys,json;print(' '.join(a['name'] for a in json.load(sys.stdin)))")
cd "$STAGE"
for f in MANIFEST.sha256 e5_Flat.index.part_* wiki-18.jsonl.gz.part_*; do
  if echo " $EXISTING " | grep -q " $f "; then echo "[upload] 已存在,跳过: $f"; continue; fi
  echo "[upload] $f ($(du -h "$f" | cut -f1))"
  curl -fsS --retry 3 -X POST -H "$AUTH" -H "Content-Type: application/octet-stream" \
    --data-binary @"$f" \
    "https://uploads.github.com/repos/$GH_REPO/releases/$RID/assets?name=$f" > /dev/null
done
echo "[upload] done → https://github.com/$GH_REPO/releases/tag/$RELEASE_TAG"
