# -*- coding: utf-8 -*-
"""scripts_rso/eval_full_val.py 递归分支(--recursive)的 CPU 测试:用脚本化假策略替代 vLLM。

验证:
  R1  run_recursive_eval 跑通 3 道 easy 题(委托成功 / 不委托直接做 / 故意失败),不经 Ray、不经 GPU
  R2  _cases.jsonl 每题含 success / turns / input|output|total_tokens / tree(节点列表) / tokens_by_depth
  R3  _metrics.json 含 tokens 段与 tree 段(delegating_rate / preexisting / stuck / steps root|sub / 深度分布)
  R4  tree_trace 落盘(eval 标签),节点数与 cases 里一致
  R5  --resume 读回带 tree 的旧 cases 不报错、树统计仍能汇总
运行:
    source env_rao.sh && cd "$SDAR_REPO" && HF_HUB_OFFLINE=1 python tests/test_eval_recursive.py
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agent_system.environments.env_package.textcraft_synth.synth_core import (  # noqa: E402
    gold_to_actions, get_shared_recipe_db, load_tasks)

spec = importlib.util.spec_from_file_location("ev", os.path.join(ROOT, "scripts_rso", "eval_full_val.py"))
ev = importlib.util.module_from_spec(spec)
sys.argv = ["x"]
spec.loader.exec_module(ev)

passed = 0
def check(cond, msg):
    global passed
    assert cond, msg
    passed += 1
    print(f"  ✓ {msg}")


def wrap(a):
    return f"<thought>ok</thought><action>{a}</action>"


def pick_two_step_tasks(k=3):
    out = []
    for t in load_tasks("train", ["easy"]):
        g = t["misc"]["gold_trajectory"]
        if len(g) == 2 and g[0]["target"][0] != list(t["misc"]["target_items"])[0]:
            out.append(t)
        if len(out) == k:
            break
    return out


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    tasks = pick_two_step_tasks(3)
    gold = [gold_to_actions(t["misc"]["gold_trajectory"]) for t in tasks]
    mids = [(t["misc"]["gold_trajectory"][0]["target"][0], t["misc"]["gold_trajectory"][0]["result_count"]) for t in tasks]

    # 脚本化策略:按 prompt 内容判断自己是 root 还是子、走到第几步(与 collector 测试同思路)
    turn_count = {}                       # (task_idx, is_child) -> 已出招次数
    def policy(prompt, ti):
        is_child = "Context provided from parent agent" in prompt or "Make sure your inventory" in prompt
        key = (ti, is_child); n = turn_count.get(key, 0); turn_count[key] = n + 1
        if ti == 0:                                   # 委托:root get_info → delegate 中间品 → craft 目标
            if is_child: return wrap(gold[0][0])
            return wrap([f"get_info {list(tasks[0]['misc']['target_items'])[0]}",
                         f"delegate: craft {mids[0][1]} {mids[0][0]} | for {list(tasks[0]['misc']['target_items'])[0]}",
                         gold[0][1]][min(n, 2)])
        if ti == 1:                                   # 不委托:直接按 gold 做
            return wrap(gold[1][min(n, 1)])
        return wrap(["inventory", "get_info x"][n % 2])   # 失败:交替动作烧预算(不触发按节点循环检测)

    def generate(prompts):
        out = []
        for p in prompts:
            # 从 prompt 里识别是哪道题(目标物品唯一)
            ti = next(i for i, t in enumerate(tasks) if list(t["misc"]["target_items"])[0] in p)
            text = policy(p, ti)
            out.append((text, len(tok.encode(p, add_special_tokens=False)), len(tok.encode(text, add_special_tokens=False))))
        return out

    tmp = tempfile.mkdtemp(prefix="rao_eval_")
    out = os.path.join(tmp, "eval")
    args = SimpleNamespace(model="fake", out=out, split="train", difficulties=["easy"], max_steps=40,
                           history_length=2, temperature=0.0, batch_size=8, save_traj=True,
                           recursive=True, per_agent_steps=6, max_depth=6, trace_dir=None, resume=False)
    print("R1 递归评测主循环(假策略)")
    results = []
    with open(out + "_cases.jsonl", "w") as f:
        ev.run_recursive_eval(args, tasks, tok, generate, get_shared_recipe_db(), f, results, time.time())
    ev.write_metrics(args, results, time.time(), final=False)
    check(len(results) == 3, "3 道题全部落盘")
    by = {r["task_id"]: r for r in results}
    check(by[tasks[0]["id"]]["success"] and by[tasks[1]["id"]]["success"] and not by[tasks[2]["id"]]["success"],
          "委托题成功 / 直做题成功 / 烧预算题失败")

    print("R2 cases.jsonl 字段")
    recs = [json.loads(l) for l in open(out + "_cases.jsonl")]
    r0 = next(r for r in recs if r["task_id"] == tasks[0]["id"])
    check(r0["total_tokens"] == r0["input_tokens"] + r0["output_tokens"] > 0, f"token 合计正确 ({r0['total_tokens']})")
    check(r0["tree"]["n_nodes"] == 2 and r0["tree"]["delegated"] and r0["tree"]["n_subagents"] == 1
          and r0["tree"]["subagent_success"] == 1.0, "委托题:2 节点、1 个成功的子代理")
    check(set(r0["tokens_by_depth"]) == {"0", "1"} and r0["tokens_by_depth"]["1"][0] > 0, "按深度的 token 分到 d0/d1")
    check(len(r0["trajectory"]) == r0["turns_used"] and any(t["is_delegation"] for t in r0["trajectory"]),
          "逐轮记录含委托轮")
    r2 = next(r for r in recs if r["task_id"] == tasks[2]["id"])
    check(r2["tree"]["root_close_reason"] == "budget_exhausted" and r2["turns_used"] == 6,
          "失败题 root 预算 6 步耗尽")
    r1 = next(r for r in recs if r["task_id"] == tasks[1]["id"])
    check(r1["tree"]["n_nodes"] == 1 and not r1["tree"]["delegated"], "直做题单节点")

    print("R3 metrics.json")
    m = json.load(open(out + "_metrics.json"))
    check(m["recursive"] and m["per_agent_steps"] == 6, "标记为递归评测")
    t = m["tree"]
    check(t["n"] == 3 and abs(t["delegating_rate"] - 1/3) < 1e-9 and t["n_subagents_total"] == 1
          and t["subagent_success_rate"] == 1.0 and t["stuck_nodes_total"] == 0, "tree 段:委托率 1/3、子代理 1 个成功、无打转")
    check(t["mean_steps_root"] > 0 and t["mean_steps_subtrajectories"] > 0 and t["max_depth_distribution"] == {"0": 2, "1": 1}
          or t["max_depth_distribution"] == {0: 2, 1: 1}, "steps root/sub 与深度分布")
    check(m["tokens"]["overall"]["n"] == 3 and m["tokens"]["on_failure"]["n"] == 1, "tokens 段按成败分组")
    check(m["tree_per_difficulty"]["easy"]["n"] == 3, "tree 按难度")

    print("R4 tree_trace 落盘")
    files = glob.glob(os.path.join(out + "_tree_trace", "eval", "reset_*.jsonl"))
    check(len(files) == 1, "1 个 trace 文件(1 批)")
    ev_ = [json.loads(l) for l in open(files[0])]
    check(sum(1 for e in ev_ if e["type"] == "node_open") == sum(r["tree"]["n_nodes"] for r in recs),
          "trace 节点数 == cases 里节点数之和")

    print("R5 --resume 读回带 tree 的旧 cases")
    prior = []
    for line in open(out + "_cases.jsonl"):
        r = json.loads(line)
        row = {k: r.get(k, 0) for k in ev.RESULT_KEYS}
        if r.get("tree"): row.update({k: r["tree"].get(k) for k in ev.TREE_KEYS})
        prior.append(row)
    ev.write_metrics(args, prior, time.time(), final=False)
    m2 = json.load(open(out + "_metrics.json"))
    check(m2["tree"]["n"] == 3 and m2["tokens"]["overall"]["sum_total_tokens"] == m["tokens"]["overall"]["sum_total_tokens"],
          "续跑汇总与原始一致")
    print(f"\n=== 递归评测分支测试全部通过({passed} 项断言)===")


if __name__ == "__main__":
    main()
