# -*- coding: utf-8 -*-
"""由各实验的 <NAME>_cases.jsonl 生成每个 benchmark 的对照报告 <bench>_report.md。

每份报告按模型分节;每个模型一组对照表(行 = 方法,列 = 汇总指标),外加分桶表、
自动生成的事实摘要,以及一节留给执行者补写的分析。汇总数由 compute_eval_metrics.summarize()
从逐题行算出,与 <NAME>_cases.jsonl 一一可复核。

实验名约定(TASK.md):<模型代号>_<bench>_<方法>,例如 qwen3-4b_sciworld_ood_rso_opsd。

用法:
  python scripts_rso/make_eval_report.py --cases-dir <放齐全部 *_cases.jsonl 的目录> --out-dir <输出目录>
"""
import argparse
import collections
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_eval_metrics import summarize  # noqa: E402

MODELS = ["qwen3-4b", "gemma-2b", "gemma-26b"]
METHODS = ["rso_opsd", "rso", "rao", "skill", "gtopsd", "grpo"]          # TASK 的优先级序
RECURSIVE = {"rso_opsd", "rso", "rao"}
BENCH_TITLE = {"textcraft": "TextCraft-Synth", "searchqa": "Search-QA", "sciworld_ood": "ScienceWorld-OOD",
               "textcraft_ood": "TextCraft-OOD", "sciworld": "ScienceWorld"}


def parse_name(name: str):
    model = name.split("_", 1)[0]
    for m in sorted(METHODS, key=len, reverse=True):
        if name.endswith("_" + m):
            return model, name[len(model) + 1: -(len(m) + 1)], m
    return model, "unknown", "unknown"


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}"


def num(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


def integer(x):
    return "—" if x is None else f"{x:,.0f}"


def _table(header: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _bucket_table(title, key_label, per_method: Dict[str, Dict[str, Any]], getter, fmt=pct) -> str:
    keys = sorted({k for s in per_method.values() for k in (getter(s) or {})}, key=str)
    if not keys:
        return ""
    methods = [m for m in METHODS if m in per_method]
    rows = [[k] + [fmt((getter(per_method[m]) or {}).get(k)) for m in methods] for k in keys]
    return f"#### {title}\n\n" + _table([key_label] + methods, rows) + "\n"


def _behavior_cols(s):
    return [num(s["avg_steps_all"]), num(s["avg_steps_success"]),
            num(s.get("avg_root_steps_all")), num(s.get("avg_root_steps_success")),
            pct(s["delegation_rate"]), num(s["avg_delegations"], 2),
            integer(s["avg_gen_tokens"]), integer(s["avg_max_node_ctx"]),
            num(s.get("avg_tree_nodes"), 2), num(s.get("avg_tree_depth"), 2)]


BEHAVIOR_HDR = ["平均步数(全部)", "平均步数(成功)", "root 步数(全部)", "root 步数(成功)", "委托率(%)",
                "平均委托次数", "平均生成 token", "平均最大单节点上下文", "平均树节点数", "平均树深度"]


def _headline(bench_type: str, s: Dict[str, Any]) -> Optional[float]:
    if bench_type == "sciworld":
        return s.get("avg_score_macro")
    if bench_type == "searchqa":
        return (s.get("overall") or {}).get("EM")
    return s.get("success_rate")


HEADLINE_NAME = {"sciworld": "平均分(宏平均)", "searchqa": "EM(三来源合计)",
                 "textcraft_ood": "成功率", "textcraft_synth": "成功率"}


def _facts(bench_type: str, per_method: Dict[str, Dict[str, Any]]) -> str:
    vals = {m: _headline(bench_type, s) for m, s in per_method.items() if _headline(bench_type, s) is not None}
    if not vals:
        return "- (无数据)\n"
    scale = 1.0 if bench_type == "sciworld" else 100.0
    unit = "分" if bench_type == "sciworld" else "个百分点"
    best = max(vals, key=vals.get)
    lines = [f"- {HEADLINE_NAME[bench_type]}最高:{best}({vals[best] * scale:.1f})。"]
    rec = {m: v for m, v in vals.items() if m in RECURSIVE}
    flat = {m: v for m, v in vals.items() if m not in RECURSIVE}
    if rec and flat:
        br, bf = max(rec, key=rec.get), max(flat, key=flat.get)
        lines.append(f"- 递归方法最好为 {br}({rec[br] * scale:.1f}),flat 方法最好为 {bf}({flat[bf] * scale:.1f}),"
                     f"差 {(rec[br] - flat[bf]) * scale:+.1f} {unit}。")
    if "grpo" in vals and "rso_opsd" in vals:
        lines.append(f"- rso_opsd 相对 grpo:{(vals['rso_opsd'] - vals['grpo']) * scale:+.1f} {unit}。")
    missing = [m for m in METHODS if m not in per_method]
    if missing:
        lines.append(f"- 缺失的方法:{', '.join(missing)}。")
    return "\n".join(lines) + "\n"


def model_section(bench_type: str, per_method: Dict[str, Dict[str, Any]]) -> str:
    methods = [m for m in METHODS if m in per_method]
    parts = []
    if bench_type in ("textcraft_synth", "textcraft_ood"):
        extra_hdr = ["无效动作率(%)", "格式无效率(%)"] if bench_type == "textcraft_ood" else []
        rows = []
        for m in methods:
            s = per_method[m]
            extra = [pct(s.get("invalid_action_rate")), pct(s.get("format_invalid_rate"))] \
                if bench_type == "textcraft_ood" else []
            rows.append([m, str(s["n_cases"]), pct(s["success_rate"])] + extra + _behavior_cols(s))
        parts.append("#### 主表\n\n" + _table(["方法", "题数", "成功率(%)"] + extra_hdr + BEHAVIOR_HDR, rows) + "\n")
        if bench_type == "textcraft_synth":
            parts.append(_bucket_table("分难度成功率(%)", "难度", per_method, lambda s: s.get("success_by_difficulty")))
        else:
            parts.append(_bucket_table("分合成深度成功率(%)", "深度", per_method, lambda s: s.get("success_by_depth")))
    elif bench_type == "sciworld":
        rows = [[m, str(per_method[m]["n_cases"]), num(per_method[m].get("avg_score_macro")),
                 num(per_method[m].get("avg_score_micro")), pct(per_method[m]["success_rate"])]
                + _behavior_cols(per_method[m]) for m in methods]
        parts.append("#### 主表\n\n" + _table(["方法", "变体数", "平均分·宏", "平均分·微", "成功率(%)"] + BEHAVIOR_HDR,
                                            rows) + "\n")
        parts.append(_bucket_table("按任务类型的平均分(0-100)", "任务", per_method,
                                   lambda s: s.get("score_by_task"), fmt=num))
    elif bench_type == "searchqa":
        srcs = ["musique", "2wikimultihopqa", "hotpotqa"]
        rows = []
        for m in methods:
            s = per_method[m]
            r = [m]
            for src in srcs:
                b = (s.get("by_source") or {}).get(src) or {}
                r += [pct(b.get("EM")), pct(b.get("F1")), pct(b.get("SubEM"))]
            r += [pct((s.get("overall") or {}).get("EM"))]
            rows.append(r)
        hdr = ["方法"] + [f"{src} {k}" for src in ("MuSiQue", "2Wiki", "HotpotQA") for k in ("EM", "F1", "SubEM")] \
            + ["合计 EM"]
        parts.append("#### 主表(%)\n\n" + _table(hdr, rows) + "\n")
        eff = []
        for m in methods:
            s = per_method[m]
            r = [m]
            for src in srcs:
                b = (s.get("by_source") or {}).get(src) or {}
                r += [num(b.get("avg_searches"), 2), num(b.get("avg_turns"), 2)]
            r += [pct(s["delegation_rate"]),
                  integer(s["avg_gen_tokens"]), integer(s["avg_max_node_ctx"]),
                  num(s.get("avg_tree_nodes"), 2), num(s.get("avg_tree_depth"), 2)]
            eff.append(r)
        ehdr = ["方法"] + [f"{src} {k}" for src in ("MuSiQue", "2Wiki", "HotpotQA") for k in ("检索次数", "轮数")] \
            + ["委托率(%)", "平均生成 token", "平均最大单节点上下文", "平均树节点数", "平均树深度"]
        parts.append("#### 效率与行为\n\n" + _table(ehdr, eff) + "\n")
        parts.append(_bucket_table("MuSiQue 按跳数 EM(%)", "跳数", per_method,
                                   lambda s: ((s.get("by_source") or {}).get("musique") or {}).get("EM_by_hops")))
        parts.append(_bucket_table("2WikiMultihopQA 按问题类型 EM(%)", "类型", per_method,
                                   lambda s: ((s.get("by_source") or {}).get("2wikimultihopqa") or {}).get("EM_by_type")))
        parts.append(_bucket_table("HotpotQA 按问题类型 EM(%)", "类型", per_method,
                                   lambda s: ((s.get("by_source") or {}).get("hotpotqa") or {}).get("EM_by_type")))
    parts.append("#### 事实摘要(自动生成)\n\n" + _facts(bench_type, per_method))
    return "\n".join(p for p in parts if p)


ANALYSIS_GUIDE = """## 分析(执行者补写)

<!-- 请用几段文字回答,每条结论都要引用上面表中的具体数字:
1. 各模型上哪个方法最好,递归方法与 flat 方法的差距多大,差距是否在三个模型上一致;
2. 分桶里差距集中在哪些难度 / 深度 / 任务 / 题型 / 跳数;
3. 效率与行为:步数、生成 token、上下文长度、委托率的差异,以及它们与成功率的关系;
4. 异常与可疑之处(某方法明显偏低、题数不齐、某些桶为空等)及可能原因。 -->
"""


def build_reports(cases_dir: str, out_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(cases_dir, "*_cases.jsonl")))
    groups = collections.defaultdict(dict)                 # task_bench -> {(model, method): (bench_type, rows)}
    for fp in files:
        rows = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
        if not rows:
            continue
        name = rows[0]["name"]
        model, task_bench, method = parse_name(name)
        groups[task_bench][(model, method)] = (rows[0]["bench"], rows)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for task_bench, exps in sorted(groups.items()):
        bench_type = next(iter(exps.values()))[0]
        title = BENCH_TITLE.get(task_bench, task_bench)
        md = [f"# {title} 全量测评对照\n",
              f"由 `scripts_rso/make_eval_report.py` 从 {len(exps)} 个实验的 `*_cases.jsonl` 生成;"
              "指标定义见 `Ideation/ideas/RSO_evaluation_metrics.md`,每个数都可由对应逐题文件复核。"
              "比率类为百分数,ScienceWorld 分数为 0-100。\n"]
        models = [m for m in MODELS if any(k[0] == m for k in exps)] + \
                 sorted({k[0] for k in exps} - set(MODELS))
        for model in models:
            per_method = {meth: summarize(bt, rows) for (mo, meth), (bt, rows) in exps.items() if mo == model}
            md.append(f"## {model}\n")
            md.append(model_section(bench_type, per_method))
        md.append(ANALYSIS_GUIDE)
        out = os.path.join(out_dir, f"{task_bench}_report.md")
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(md))
        written.append(out)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-dir", required=True)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args(argv)
    for p in build_reports(args.cases_dir, args.out_dir):
        print("写出", p)


if __name__ == "__main__":
    main()
