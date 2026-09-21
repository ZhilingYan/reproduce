# -*- coding: utf-8 -*-
"""为 Search-QA 全量测评集生成分桶用的元数据对照表 data/searchqa/test_3src_meta.jsonl。

为什么需要:test_3src.parquet 来自 Search-R1 的处理版,每题只剩问题原文、金答案与编号,
跳数(MuSiQue)与问题类型(2WikiMultihopQA、HotpotQA)在处理时被丢弃了;这两个字段只存在于
各数据集的官方 dev 文件里。本脚本按问题原文把每道测试题回连到官方 dev,取出这两个字段。
测试题目与答案本身不变,对照表随仓提供,合作者无需下载任何东西;本脚本只为可复现而保留。

回连键:只保留字母与数字、全部小写后的问题原文(Search-R1 版有时在句尾多一个句点,
严格比对会漏掉少数题,宽松键下三个来源全部命中且无歧义,见运行输出)。

官方 dev 来源(维护者运行本脚本时需要,合作者不需要):
  - MuSiQue:官方仓 github.com/StonyBrookNLP/musique 的数据包(README 第二节同一个 Google Drive 链接),
    取 musique_ans_v1.0_dev.jsonl;跳数由官方 id 前缀给出(2hop__ / 3hop1__ / 4hop1__ …)。
  - 2WikiMultihopQA:官方 dev.json(README 第二节 $WIKI2_DATA_IDS 解压目录),字段 type。
  - HotpotQA:HuggingFace hotpotqa/hotpot_qa 的 fullwiki validation(与官方 dev 同一份 7,405 题),字段 type。

用法:
  python examples/data_preprocess/build_searchqa_eval_meta.py \
      --musique-dev <musique_ans_v1.0_dev.jsonl> --twowiki-dev <dev.json> \
      --hotpot-dev <hotpot_dev_fullwiki_meta.jsonl 或官方 dev json>
"""
import argparse
import collections
import json
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.realpath(os.path.join(HERE, "..", ".."))


def qkey(q: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(q).lower())


def load_musique(path):
    out = collections.defaultdict(set)
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        m = re.match(r"(\d)hop", d["id"])
        out[qkey(d["question"])].add(int(m.group(1)) if m else None)
    return {k: {"hops": next(iter(v))} for k, v in out.items() if len(v) == 1}, out


def load_twowiki(path):
    out = collections.defaultdict(set)
    for d in json.load(open(path, encoding="utf-8")):
        out[qkey(d["question"])].add(d["type"])
    return {k: {"type": next(iter(v))} for k, v in out.items() if len(v) == 1}, out


def load_hotpot(path):
    rows = ([json.loads(l) for l in open(path, encoding="utf-8")] if path.endswith(".jsonl")
            else json.load(open(path, encoding="utf-8")))
    out = collections.defaultdict(set)
    for d in rows:
        out[qkey(d["question"])].add(d["type"])
    return {k: {"type": next(iter(v))} for k, v in out.items() if len(v) == 1}, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=os.path.join(REPO, "data/searchqa/test_3src.parquet"))
    ap.add_argument("--musique-dev", required=True)
    ap.add_argument("--twowiki-dev", required=True)
    ap.add_argument("--hotpot-dev", required=True)
    ap.add_argument("--out", default=os.path.join(REPO, "data/searchqa/test_3src_meta.jsonl"))
    args = ap.parse_args()

    tables = {"musique": load_musique(args.musique_dev),
              "2wikimultihopqa": load_twowiki(args.twowiki_dev),
              "hotpotqa": load_hotpot(args.hotpot_dev)}
    df = pd.read_parquet(args.test)
    stat = collections.Counter()
    with open(args.out, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            src = r["data_source"]
            q = r["extra_info"]["question"]
            uniq, raw = tables[src]
            k = qkey(q)
            stat[(src, "total")] += 1
            if len(raw.get(k, ())) > 1:
                stat[(src, "ambiguous")] += 1
            meta = uniq.get(k)
            if meta is None:
                stat[(src, "missing")] += 1
                meta = {}
            # 不存问题原文(test_3src.parquet 里已有);回连键 qkey 足够对齐逐题记录
            f.write(json.dumps({"data_source": src, "qkey": k,
                                "index": int(r["extra_info"]["index"]), **meta},
                               ensure_ascii=False) + "\n")
    for src in tables:
        t, a, m = stat[(src, "total")], stat[(src, "ambiguous")], stat[(src, "missing")]
        print(f"{src:18s} 共 {t:5d} 题,命中 {t - m}({(t - m) / t:.1%}),歧义 {a},缺失 {m}")
    print("写出", args.out)


if __name__ == "__main__":
    main()
