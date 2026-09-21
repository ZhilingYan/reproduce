#!/usr/bin/env bash
# 下载 textcraft_ood 所需的 860 个 Minecraft 1.16.5 合成配方(真 TextCraft 环境的数据),
# 校验 sha256 后放进 env_package/textcraft/textcraft_src/textcraft/recipes/。
#
# 为什么不直接放进本仓:配方是 Minecraft 游戏数据。TextCraft 官方仓(ADaPT,
# https://github.com/archiki/ADaPT,README 第 9 行)同样不分发,而是指定从
# InventivetalentDev/minecraft-assets 的 1.16.5 标签下载 data/minecraft/recipes——本脚本照此执行。
#
# 只拷贝清单(textcraft_src/RECIPES_MANIFEST.sha256)里列出的文件并逐个校验,
# 保证拿到的数据与本仓实测时逐字节一致(CraftingTree 会读 recipes/ 下全部 json,
# 多一个文件就可能改变任务集)。
#
# 用法(在仓根目录):bash examples/data_preprocess/fetch_textcraft_recipes.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")/../.." && pwd)
PKG="$HERE/agent_system/environments/env_package/textcraft/textcraft_src"
DEST="$PKG/textcraft/recipes"
MANIFEST="$PKG/RECIPES_MANIFEST.sha256"
REPO=https://github.com/InventivetalentDev/minecraft-assets.git
TAG=1.16.5
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

if [ -d "$DEST" ] && (cd "$DEST" && sha256sum -c "$MANIFEST" --quiet 2>/dev/null); then
  echo "[recipes] 已存在且校验通过,跳过:$DEST"
  exit 0
fi

echo "[recipes] 稀疏克隆 $REPO @ $TAG(只取 data/minecraft/recipes)..."
git clone -q --depth 1 --branch "$TAG" --filter=blob:none --sparse "$REPO" "$TMP/mc"
git -C "$TMP/mc" sparse-checkout set data/minecraft/recipes

mkdir -p "$DEST"
n=0
while read -r _ fname; do
  cp "$TMP/mc/data/minecraft/recipes/$fname" "$DEST/$fname"
  n=$((n + 1))
done < "$MANIFEST"
echo "[recipes] 拷贝 $n 个文件,校验中..."
(cd "$DEST" && sha256sum -c "$MANIFEST" --quiet)
echo "[recipes] 完成:$n 个配方全部校验通过 → $DEST"
