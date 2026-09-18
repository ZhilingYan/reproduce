# [2026-09-18 新增,SDAR 原文件未动] 把 2WikiMultihopQA 官方 evidences(三元组)转成
# MuSiQue question_decomposition 的同款结构,回填进两份 parquet 里 2wiki 行的
# metadata.question_decomposition(原为 null;musique 行原样不动,parquet schema 零变化):
#   ~/data/searchR1_musique_2wiki/train.parquet   15,000 行 2wiki(官方 train.json 回连,已实测全命中)
#   ~/data/searchR1_processed_direct/test.parquet 12,576 行 2wiki(官方 dev.json 回连,
#                                                  精确 12,430 + 归一化兜底,余量保持 null 并计数)
# 覆写前各留 .bak_nodecomp 备份。
#
# 转换规则(与 Ideation/ideas/RSO_searchqa_data_mapping.md 的 priv/Φ 口径配套):
#   每条三元组 [subj, rel, obj] → 一跳 {id, question, answer, paragraph_support_idx, support_paragraph}
#   - question = "subj >> rel";若 subj 等于某前跳的 answer(归一化比对),改写 "#k >> rel"
#     (k 为该前跳的 id),与 musique 的占位记法一致;
#   - answer = obj;
#   - support_paragraph:在本行自带 metadata.context 里按标题归一化找 subj 对应文档,
#     命中则填 {idx, title, paragraph_text(句子拼接), is_supporting=True},未命中填 null;
#   - 末尾虚拟比较跳:若 golden answer 不等于任何一跳的 obj(comparison / bridge_comparison
#     的最终答案是比较运算结果,不在三元组宾语里),补一跳
#     {question: "compare(#i, #j, ...)"(i,j 为终端跳,即 answer 未被后续跳当主语的跳),
#      answer: golden answer, support 两字段 null}。
#     下游识别口径:question 以 "compare(" 开头的跳不是检索跳,不进 Φ 分母。
import json
import os
import re
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_IDS = os.environ.get("WIKI2_DATA_IDS", os.path.expanduser("~/data/data_ids"))
# 2WikiMultihopQA 官方 data_ids.zip(含 train/dev/test.json 与 id_aliases.json):
#   官方仓库 https://github.com/Alab-NII/2wikimultihop 的 README 提供下载链接,解压得 data_ids/
TRAIN_PQ = os.path.expanduser("~/data/searchR1_musique_2wiki/train.parquet")
TEST_PQ = os.path.expanduser("~/data/searchR1_processed_direct/test.parquet")


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def to_plain(obj):
    if isinstance(obj, np.ndarray):
        return [to_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


STATS = {"hops": 0, "dep_sub": 0, "support_hit": 0, "compare_rows": 0, "rows": 0}


def evidences_to_decomp(evidences, golden_answers, context):
    ctx_titles = [norm(t) for t in (context or {}).get("title") or []]
    ctx_content = (context or {}).get("content") or []
    hops, prev_answers = [], []
    for i, (subj, rel, obj) in enumerate(evidences, 1):
        dep = next((j for j, a in enumerate(prev_answers, 1) if norm(a) == norm(subj)), None)
        if dep is not None:
            STATS["dep_sub"] += 1
        q = f"#{dep} >> {rel}" if dep is not None else f"{subj} >> {rel}"
        idx = ctx_titles.index(norm(subj)) if norm(subj) in ctx_titles else None
        support = None
        if idx is not None and idx < len(ctx_content):
            STATS["support_hit"] += 1
            support = {"idx": idx, "title": to_plain((context.get("title"))[idx]),
                       "paragraph_text": " ".join(to_plain(ctx_content[idx])), "is_supporting": True}
        hops.append({"id": i, "question": q, "answer": str(obj),
                     "paragraph_support_idx": idx, "support_paragraph": support})
        prev_answers.append(obj)
        STATS["hops"] += 1
    golds = {norm(a) for a in golden_answers}
    if not any(norm(h["answer"]) in golds for h in hops):
        used_as_subj = {norm(s) for s, _, _ in evidences}
        terminal = [h["id"] for h in hops if norm(h["answer"]) not in used_as_subj]
        hops.append({"id": len(hops) + 1,
                     "question": "compare(" + ", ".join(f"#{i}" for i in (terminal or [h['id'] for h in hops])) + ")",
                     "answer": str(golden_answers[0]), "paragraph_support_idx": None, "support_paragraph": None})
        STATS["compare_rows"] += 1
    STATS["rows"] += 1
    return hops


def load_official(fname):
    data = json.load(open(os.path.join(DATA_IDS, fname)))
    exact, normed = {}, {}
    for d in data:
        exact.setdefault(d["question"], []).append(d)
        normed.setdefault(norm(d["question"]), []).append(d)
    return exact, normed


def lookup(exact, normed, q):
    m = exact.get(q) or normed.get(norm(q))
    return m[0] if m and len(m) == 1 else None


def process(pq_path, official_file):
    exact, normed = load_official(official_file)
    ref_schema = pq.read_schema(pq_path)
    df = pd.read_parquet(pq_path)
    miss = filled = 0
    metas = df["metadata"].tolist()
    for i in range(len(df)):
        if df["data_source"].iat[i] != "2wikimultihopqa":
            continue
        q = df["extra_info"].iat[i]["question"]
        rec = lookup(exact, normed, q)
        if rec is None:
            miss += 1
            continue
        meta = to_plain(metas[i])
        golds = to_plain(df["reward_model"].iat[i]["ground_truth"]["target"])
        meta["question_decomposition"] = evidences_to_decomp(rec["evidences"], golds, meta.get("context"))
        metas[i] = meta
        filled += 1
    df["metadata"] = metas
    # 全列转纯 python 再重建,cast 回原 schema —— schema 零变化的硬校验
    for col in df.columns:
        df[col] = [to_plain(v) for v in df[col]]
    table = pa.Table.from_pandas(df, preserve_index=False).select(ref_schema.names).cast(ref_schema)
    bak = pq_path + ".bak_nodecomp"
    if not os.path.exists(bak):
        shutil.copy2(pq_path, bak)
    pq.write_table(table, pq_path)
    print(f"{os.path.basename(pq_path)}: 回填 {filled} 行,未回连保持 null {miss} 行,备份 {os.path.basename(bak)}")
    return filled, miss


def main():
    process(TRAIN_PQ, "train.json")
    process(TEST_PQ, "dev.json")
    print("累计统计:", STATS,
          f"| support 命中率 {STATS['support_hit']/max(STATS['hops'],1):.1%}",
          f"| 依赖占位改写率 {STATS['dep_sub']/max(STATS['hops'],1):.1%}")
    # 回读抽查:train 里 Move/Méditerranée 那条的转换结果
    back = pd.read_parquet(TRAIN_PQ)
    row = back[back["data_source"] == "2wikimultihopqa"].iloc[0]
    qd = row["metadata"]["question_decomposition"]
    assert qd is not None and len(qd) == 5, f"预期 4 三元组 + 1 比较跳,得到 {None if qd is None else len(qd)}"
    for h in qd:
        print(" ", h["id"], "|", h["question"], "=>", h["answer"],
              "| support:", None if h["support_paragraph"] is None else h["support_paragraph"]["title"])
    mu = back[back["data_source"] == "musique"].iloc[0]
    assert mu["metadata"]["question_decomposition"][0]["question"] == "The Collegian >> owned by", "musique 行被误改"
    print("musique 行未动,校验通过")


if __name__ == "__main__":
    main()
