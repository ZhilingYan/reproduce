# [2026-09-18 新增,SDAR 原文件未动] 用 MuSiQue + 2WikiMultihopQA 的官方 train split
# 制作新的 SearchQA 训练集,替换原 nq+hotpotqa 训练混合(测试集不变,仍用
# ~/data/searchR1_processed_direct/test.parquet 的 7 个子 benchmark)。
#
# 输入:FlashRAG_datasets 的原始 jsonl(已下载到项目盘):
#   $FLASHRAG_RAW/musique_train.jsonl          19,938 行
#   $FLASHRAG_RAW/2wikimultihopqa_train.jsonl  15,000 行(下载链接见 RAW_DIR 处注释)
#   每行:{"id", "question", "golden_answers", "metadata"{各子集自有字段}}
#
# 输出:~/data/searchR1_musique_2wiki/train.parquet,共 34,938 行。
# 行格式逐列复刻 examples/data_preprocess/preprocess_search_r1_dataset.py 的
# process_single_row(该脚本吃的是 HF parquet,吃不了 jsonl,所以这里另写,但字段
# 构造逻辑照抄):data_source / prompt / ability / reward_model / extra_info /
# metadata / env_kwargs 七列;metadata 用现有 test.parquet 的并集 schema 透传,
# 各子集缺的字段填 null,保证将来与 test.parquet 拼接时 arrow 类型不冲突。
# 最后用 pyarrow 把新表 cast 成 test.parquet 的 schema 做硬校验。
import json
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RAW_DIR = os.environ.get("FLASHRAG_RAW", os.path.expanduser("~/data/flashrag_raw"))
# FlashRAG 原始 jsonl 下载(两个文件放进 RAW_DIR,命名 musique_train.jsonl / 2wikimultihopqa_train.jsonl):
#   https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/musique/train.jsonl
#   https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/2wikimultihopqa/train.jsonl
REF_TEST = os.path.expanduser("~/data/searchR1_processed_direct/test.parquet")
OUT_DIR = os.path.expanduser("~/data/searchR1_musique_2wiki")

SYSTEM_CONTENT = "You are a helpful and harmless assistant."  # 与原脚本 DEFAULT_SYSTEM_CONTENT 一致
ABILITY = "fact-reasoning"  # 现有 parquet 里全部行都是这个值

# 现有 test.parquet 的 metadata 并集字段(顺序无所谓,cast 时按名对齐)
METADATA_KEYS = [
    "subj", "prop", "obj", "subj_id", "prop_id", "obj_id",
    "s_aliases", "o_aliases", "s_uri", "o_uri", "s_wiki_title", "o_wiki_title",
    "s_pop", "o_pop", "type", "level", "supporting_facts", "context",
    "answerable", "question_decomposition",
]


def build_row(raw, data_source, index):
    question = raw["question"]
    golden = list(raw.get("golden_answers") or [])
    assert question and golden, f"{data_source} 第 {index} 行缺 question 或答案"
    ground_truth = {"target": golden}
    triple = {"ground_truth": ground_truth, "question": question, "data_source": data_source}
    metadata = {k: raw.get("metadata", {}).get(k) for k in METADATA_KEYS}
    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": SYSTEM_CONTENT},
            {"role": "user", "content": question},  # 与原脚本一致:user 就是裸问题,无前缀
        ],
        "ability": ABILITY,
        "reward_model": {"ground_truth": ground_truth, "style": "rule"},
        "extra_info": {
            "index": index,
            "need_tools_kwargs": True,
            "question": question,
            "split": "train",
            "tools_kwargs": {"search": {"create_kwargs": triple}},
        },
        "metadata": metadata,
        "env_kwargs": triple,
    }


def main():
    rows = []
    for fname, source in [("musique_train.jsonl", "musique"),
                          ("2wikimultihopqa_train.jsonl", "2wikimultihopqa")]:
        with open(os.path.join(RAW_DIR, fname)) as f:
            for line in f:
                rows.append(build_row(json.loads(line), source, index=len(rows)))
        print(f"{source}: 累计 {len(rows)} 行")

    df = pd.DataFrame(rows)

    # 硬校验:新表必须能无损 cast 成现有 test.parquet 的 schema(列名、嵌套类型全同)
    ref_schema = pq.read_schema(REF_TEST)
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.select(ref_schema.names).cast(ref_schema)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "train.parquet")
    pq.write_table(table, out_path)

    # 回读复核:行数、子集分布、抽查第一行与原始 jsonl 一致
    back = pd.read_parquet(out_path)
    assert len(back) == 34938, len(back)
    print(back["data_source"].value_counts())
    first = back.iloc[0]
    assert first["prompt"][1]["content"] == "When was the institute that owned The Collegian founded?"
    assert list(first["reward_model"]["ground_truth"]["target"]) == ["1960"]
    assert first["extra_info"]["split"] == "train"
    print("写出:", out_path, "| schema 与 test.parquet 逐列一致,抽查通过")


if __name__ == "__main__":
    main()
