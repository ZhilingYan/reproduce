# -*- coding: utf-8 -*-
"""全量测评的逐题指标:把一次全量测评的逐题记录,整理成交付用的 <NAME>_cases.jsonl。

指标定义见 Ideation/ideas/RSO_evaluation_metrics.md。本脚本只读逐题记录、只写自己的输出,
不改动现有测评流程与其结果。

输入:
  --run        一次全量测评的逐题记录。val_only 路径(searchqa / sciworld / textcraft_ood)是训练脚本加
               `+trainer.val_case_dump_dir=<目录>` 后写出的 <目录>/cases.jsonl
               (agent_system/multi_turn_rollout/val_case_dump.py);textcraft_synth 是
               scripts_rso/eval_full_val.py 写出的 <前缀>_cases.jsonl。可以给文件,也可以给含 cases.jsonl 的目录。
  --meta       Search-QA 分桶元数据(缺省 data/searchqa/test_3src_meta.jsonl)。

输出 <out-dir>/<NAME>_cases.jsonl:每题一行(subset 恒为 "test",即全量测评集)。
各 benchmark 的逐题字段:
  共有      case_id, success, steps_total, steps_root, gen_tokens, max_node_ctx,
            is_recursive, delegated, n_delegations, n_nodes, tree_depth
  textcraft_ood   goal_depth, env_actions, invalid_actions, format_invalid_actions
  textcraft_synth difficulty
  sciworld        task, variation, score(0-100)
  searchqa        data_source, question_type, hops, em, f1, subem, answer, golds, n_searches
汇总(对照表里的数)由 summarize() 从这些逐题行算出,scripts_rso/make_eval_report.py 复用同一函数。

判分口径(与现有测评一致):
  - 成功取最后一个有效步的 info['won'],与现有 success_rate 同一口径(递归方法也是,
    见 agent_system/recursive/orchestrator.py 的 _process_batch)。
  - Search-QA 的 EM 即 won(训练奖励同口径);F1 与 SubEM 用同一个 normalize_answer 在抽出的
    <answer> 上计算。F1 按 HotpotQA 官方实现:词级重叠,预测或金答案为 yes/no/noanswer 且两者
    不同时记 0;多个金答案取最大值。
  - ScienceWorld 分数取 episode 奖励 × 100:flat 奖励是负分裁零后分数的增量之和,递归奖励在 root
    关闭时补成 root 官方分,两者都等于"负分裁零后的分数 / 100",与现有 test_score 严格一致。
  - 解码口径随现有全量测评命令:val_only 路径是脚本里的 val 设置(目前温度 0.4 采样一次),
    textcraft_synth 的 eval_full_val.py 是贪心。
"""
import argparse
import collections
import importlib.util
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.realpath(os.path.join(HERE, ".."))
BENCHES = ("searchqa", "sciworld", "textcraft_ood", "textcraft_synth")

# ---- 判分函数:直接加载仓内的 search/utils.py(训练奖励同一份实现),不经过 search 包的 __init__(避免拉起 ray)
_UTILS = os.path.join(REPO, "agent_system/environments/env_package/search/third_party/"
                            "skyrl_gym/envs/search/utils.py")
_spec = importlib.util.spec_from_file_location("_search_utils", _UTILS)
_su = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_su)
normalize_answer, em_check, subem_check, extract_solution = (
    _su.normalize_answer, _su.em_check, _su.subem_check, _su.extract_solution)


def f1_single(pred: str, gold: str) -> float:
    """HotpotQA 官方 f1_score 口径(词级重叠;yes/no/noanswer 不一致记 0)。"""
    p, g = normalize_answer(pred), normalize_answer(gold)
    special = ("yes", "no", "noanswer")
    if (p in special or g in special) and p != g:
        return 0.0
    pt, gt = p.split(), g.split()
    same = sum((collections.Counter(pt) & collections.Counter(gt)).values())
    if same == 0:
        return 0.0
    prec, rec = same / len(pt), same / len(gt)
    return 2 * prec * rec / (prec + rec)


def f1_max(pred: Optional[str], golds: List[str]) -> float:
    if pred is None or not golds:
        return 0.0
    return max(f1_single(pred, g) for g in golds)


def qkey(q: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(q).lower())


# ============================================================================ 读入与逐题整理
def _resolve(path: str) -> str:
    if os.path.isdir(path):
        path = os.path.join(path, "cases.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(_resolve(path), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass                                   # 作业被杀时可能写了半行
    return rows


def load_meta(path: Optional[str]) -> Dict[Any, Dict[str, Any]]:
    meta = {}
    if path and os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            meta[(d["data_source"], d["qkey"])] = d
    return meta


def _common_from_dump(r: Dict[str, Any]) -> Dict[str, Any]:
    rec = r.get("rec")
    return {
        "success": 1 if r.get("won") else 0,
        "steps_total": rec["total_turns"] if rec else r.get("n_steps"),
        "steps_root": rec["root_turns"] if rec else None,
        "gen_tokens": r.get("gen_tokens"),
        "max_node_ctx": r.get("max_prompt_tokens"),
        "is_recursive": rec is not None,
        "delegated": (rec["delegation_turns"] > 0) if rec else False,
        "n_delegations": rec["delegation_turns"] if rec else 0,
        "n_nodes": rec["n_nodes"] if rec else None,
        "tree_depth": rec["max_depth"] if rec else None,
    }


def case_row(bench: str, r: Dict[str, Any], meta: Dict[Any, Dict[str, Any]]) -> Dict[str, Any]:
    """一条原始逐题记录 → 交付用的逐题指标行。"""
    if bench == "textcraft_synth":                    # eval_full_val.py 的一行
        tree, traj = r.get("tree"), r.get("trajectory") or []
        ins = [t.get("input_tokens") for t in traj if t.get("input_tokens") is not None]
        row = {"case_id": r.get("task_id"), "difficulty": r.get("difficulty"),
               "success": 1 if r.get("success") else 0,
               "gen_tokens": r.get("output_tokens"), "max_node_ctx": max(ins) if ins else None,
               "is_recursive": tree is not None}
        if tree is not None:
            n_deleg = sum(1 for t in traj if t.get("is_delegation")) if traj else tree.get("n_subagents", 0)
            row.update(steps_total=int(tree.get("root_turns", 0)) + int(tree.get("sub_turns", 0)),
                       steps_root=tree.get("root_turns"), delegated=n_deleg > 0, n_delegations=n_deleg,
                       n_nodes=tree.get("n_nodes"), tree_depth=tree.get("max_depth"))
        else:
            row.update(steps_total=r.get("turns_used"), steps_root=None, delegated=False, n_delegations=0,
                       n_nodes=None, tree_depth=None)
        return row

    ex, ek = r.get("extra") or {}, r.get("env_kwargs") or {}
    row = _common_from_dump(r)
    if bench == "searchqa":
        src = r.get("data_source") or ek.get("data_source")
        k = qkey(ek.get("question", ""))
        m = meta.get((src, k), {})
        golds = [str(x) for x in ((ek.get("ground_truth") or {}).get("target") or [])]
        ans = extract_solution(r.get("last_response_tail") or "")
        row.update(case_id=f"{src}:{m['index']}" if "index" in m else f"{src}:{k}",
                   data_source=src, question_type=m.get("type"), hops=m.get("hops"),
                   em=row["success"], f1=f1_max(ans, golds),
                   subem=int(subem_check(ans, golds)) if ans is not None else 0,
                   answer=ans, golds=golds, n_searches=r.get("n_tool_calls"))
    elif bench == "sciworld":
        task = ex.get("extra.task") or ek.get("task")
        var = ex.get("extra.variation", ek.get("variation"))
        row.update(case_id=f"{task}:{var}", task=task, variation=var,
                   score=max(float(r.get("reward", 0.0) or 0.0), 0.0) * 100.0)
    elif bench == "textcraft_ood":
        rec = r.get("rec")
        row.update(case_id=ex.get("extra.goal_item"), goal_depth=ex.get("extra.goal_depth"),
                   env_actions=rec["env_turns"] if rec else r.get("n_steps"),
                   invalid_actions=r.get("n_action_failed"),
                   format_invalid_actions=r.get("n_format_invalid"))
    return row


def build_rows(bench: str, run: str, meta_path: Optional[str], name: str):
    meta = load_meta(meta_path) if bench == "searchqa" else {}
    by_id: Dict[Any, Dict[str, Any]] = {}
    dup = 0
    for raw in read_jsonl(run):
        row = case_row(bench, raw, meta)
        if row["case_id"] in by_id:                   # 作业重启后同一题可能写了两次:保留最后一次
            dup += 1
        by_id[row["case_id"]] = {"name": name, "bench": bench, "subset": "test", **row}
    return list(by_id.values()), dup


# ============================================================================ 汇总(对照表用)
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def _by(rows, field):
    g = collections.defaultdict(list)
    for r in rows:
        g[r.get(field)].append(r)
    return dict(sorted(g.items(), key=lambda kv: str(kv[0])))


def summarize(bench: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """一个实验的汇总指标。"""
    t = [r for r in rows if r["subset"] == "test"]
    rec = any(r.get("is_recursive") for r in t)
    succ = [r for r in t if r["success"]]
    s: Dict[str, Any] = {"n_cases": len(t), "is_recursive": rec,
                         "success_rate": _mean(r["success"] for r in t),
                         "avg_steps_all": _mean(r.get("steps_total") for r in t),
                         "avg_steps_success": _mean(r.get("steps_total") for r in succ),
                         "avg_gen_tokens": _mean(r.get("gen_tokens") for r in t),
                         "avg_max_node_ctx": _mean(r.get("max_node_ctx") for r in t),
                         "delegation_rate": _mean(1.0 if r.get("delegated") else 0.0 for r in t),
                         "avg_delegations": _mean(r.get("n_delegations") for r in t)}
    if rec:
        s.update(avg_root_steps_all=_mean(r.get("steps_root") for r in t),
                 avg_root_steps_success=_mean(r.get("steps_root") for r in succ),
                 avg_tree_nodes=_mean(r.get("n_nodes") for r in t),
                 avg_tree_depth=_mean(r.get("tree_depth") for r in t))
    if bench == "textcraft_synth":
        s["success_by_difficulty"] = {str(k): _mean(r["success"] for r in v)
                                      for k, v in _by(t, "difficulty").items()}
    elif bench == "textcraft_ood":
        act = sum(r.get("env_actions") or 0 for r in t)
        s["invalid_action_rate"] = (sum(r.get("invalid_actions") or 0 for r in t) / act) if act else None
        s["format_invalid_rate"] = (sum(r.get("format_invalid_actions") or 0 for r in t) / act) if act else None
        s["success_by_depth"] = {str(k): _mean(r["success"] for r in v) for k, v in _by(t, "goal_depth").items()}
    elif bench == "sciworld":
        per_task = {str(k): _mean(r["score"] for r in v) for k, v in _by(t, "task").items()}
        s.update(avg_score_macro=_mean(per_task.values()), avg_score_micro=_mean(r["score"] for r in t),
                 score_by_task=per_task)
    elif bench == "searchqa":
        def qa(sub):
            return {"n": len(sub), "EM": _mean(r["em"] for r in sub), "F1": _mean(r["f1"] for r in sub),
                    "SubEM": _mean(r["subem"] for r in sub), "avg_searches": _mean(r.get("n_searches") for r in sub),
                    "avg_turns": _mean(r.get("steps_total") for r in sub)}
        s["overall"] = qa(t)
        s["by_source"] = {}
        for src, sub in _by(t, "data_source").items():
            blk = qa(sub)
            if src == "musique":
                blk["EM_by_hops"] = {str(k): _mean(r["em"] for r in v) for k, v in _by(sub, "hops").items()}
            else:
                blk["EM_by_type"] = {str(k): _mean(r["em"] for r in v) for k, v in _by(sub, "question_type").items()}
            s["by_source"][str(src)] = blk
        s["sanity_missing_meta"] = sum(1 for r in t if r.get("question_type") is None and r.get("hops") is None)
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=BENCHES)
    ap.add_argument("--run", required=True)
    ap.add_argument("--meta", default=os.path.join(REPO, "data/searchqa/test_3src_meta.jsonl"))
    ap.add_argument("--name", required=True, help="实验名,即交付文件名前缀")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args(argv)
    rows, dup = build_rows(args.bench, args.run, args.meta, args.name)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.name}_cases.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(summarize(args.bench, rows), ensure_ascii=False, indent=2))
    print(f"写出 {out}({len(rows)} 行;重复记录去掉 {dup} 条)", file=sys.stderr)


if __name__ == "__main__":
    main()
