# -*- coding: utf-8 -*-
# [2026-09-18 新增] Search-RSO 的两个数据产物:
#   1) ~/data/searchR1_musique_2wiki/val_sub.parquet
#      训练中验证用的固定子集:7 个 data_source 各抽 15 条(2026-09-19 用户拍板,原 25)(seed=42,排序后抽,跨机器可复现),
#      共 105 行;schema 与 test.parquet 完全一致(直接行子集,不重建)。
#   2) ~/data/searchR1_musique_2wiki/decomp_store.json
#      递归适配器的 decomp/别名库,键 = 归一化题面:
#        {norm_q: {"targets": [gold...],
#                  "hops": [{"q": 子问题, "ans": 答案, "aliases": [归一化别名...],
#                            "compare": bool}, ...]}}
#      范围:train.parquet 全部 34,938 行 + test.parquet 的 musique/2wiki 行(评测树也要 Φ)。
#      2wiki 的别名来自官方 data_ids:evidences_id 的宾语 Q-id → id_aliases.json 的
#      aliases + demonyms;musique 无官方别名表,别名集 = 答案原文一项。
#      归一化口径与运行时 decomp.py 的 _norm 完全一致(小写、非字母数字改空格、压空白)。
import json
import os
import re

import pandas as pd

DATA_DIR = os.path.expanduser("~/data/searchR1_musique_2wiki")
TEST_PQ = os.path.expanduser("~/data/searchR1_processed_direct/test.parquet")
DATA_IDS = os.environ.get("WIKI2_DATA_IDS", os.path.expanduser("~/data/data_ids"))


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", str(s).lower())).strip()


def build_val_sub():
    df = pd.read_parquet(TEST_PQ)
    parts = []
    for src, g in df.groupby("data_source"):
        g = g.sort_values(by="extra_info", key=lambda col: col.map(lambda e: int(e["index"])))
        parts.append(g.sample(n=min(15, len(g)), random_state=42))
    sub = pd.concat(parts).reset_index(drop=True)
    out = os.path.join(DATA_DIR, "val_sub.parquet")
    sub.to_parquet(out, index=False)
    print("val_sub:", len(sub), "行", dict(sub["data_source"].value_counts()), "→", out)


def load_2wiki_alias_maps():
    """官方档:题面 → {obj 原文(strip 后): 别名列表}。train.json + dev.json 都要。"""
    qid_alias = {}
    with open(os.path.join(DATA_IDS, "id_aliases.json")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            qid_alias[d["Q_id"]] = list(d.get("aliases") or []) + list(d.get("demonyms") or [])

    by_question = {}
    for fname in ("train.json", "dev.json"):
        for d in json.load(open(os.path.join(DATA_IDS, fname))):
            obj_aliases = {}
            ev = d.get("evidences") or []
            evid = d.get("evidences_id") or []
            for k, triple in enumerate(ev):
                obj = str(triple[2]).strip()
                ids = evid[k] if k < len(evid) else None
                obj_id = ids[2] if ids and len(ids) == 3 else None
                obj_aliases[obj] = qid_alias.get(obj_id, []) if obj_id else []
            by_question[d["question"]] = obj_aliases
            by_question.setdefault(norm(d["question"]), obj_aliases)
    return by_question


def hop_entry(h, alias_lookup):
    ans = str(h["answer"]).strip()
    q = str(h["question"]).strip()
    aliases = {norm(ans)}
    for a in alias_lookup.get(ans, []):
        aliases.add(norm(a))
    aliases = sorted(x for x in aliases if len(x) >= 2)   # 1 字符别名匹配噪声太大,弃
    return {"q": q, "ans": ans, "aliases": aliases, "compare": q.startswith("compare(")}


def build_store():
    wiki_alias = load_2wiki_alias_maps()
    store = {}
    stats = {"rows": 0, "no_decomp": 0, "hops": 0, "alias_total": 0}

    def add_row(row):
        md = row["metadata"]
        qd = md.get("question_decomposition") if isinstance(md, dict) else None
        question = row["extra_info"]["question"]
        if qd is None or len(qd) == 0:
            stats["no_decomp"] += 1
            return
        if row["data_source"] == "2wikimultihopqa":
            lookup = wiki_alias.get(question) or wiki_alias.get(norm(question)) or {}
        else:
            lookup = {}
        gt = row["reward_model"]["ground_truth"]
        targets = [str(t) for t in (gt.get("target") if isinstance(gt, dict) else gt)]
        hops = [hop_entry(h, lookup) for h in qd]
        store[norm(question)] = {"targets": targets, "hops": hops}
        stats["rows"] += 1
        stats["hops"] += len(hops)
        stats["alias_total"] += sum(len(h["aliases"]) for h in hops)

    for pq in (os.path.join(DATA_DIR, "train.parquet"), TEST_PQ):
        df = pd.read_parquet(pq)
        df = df[df["data_source"].isin(["musique", "2wikimultihopqa"])]
        for _, row in df.iterrows():
            add_row(row)

    out = os.path.join(DATA_DIR, "decomp_store.json")
    with open(out, "w") as f:
        json.dump(store, f, ensure_ascii=False)
    print(f"decomp_store: {len(store)} 题 | {stats} → {out} "
          f"({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    build_val_sub()
    build_store()
