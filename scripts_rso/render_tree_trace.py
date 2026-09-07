# -*- coding: utf-8 -*-
"""把 tree_trace 的 jsonl 渲染成人读文本:每棵树的结构 + 每个节点逐轮的 prompt / 模型输出 / 结果。

用法:
    python scripts_rao/render_tree_trace.py <reset_0000.jsonl> [--slot N] [--no-prompt] [--out 文件]
    --slot N      只看第 N 个 batch 槽(一棵树)
    --no-prompt   不打印完整 prompt(只留模型输出与结果,便于速览)
    --out         写到文件而不是 stdout

输入文件由 agent_system/recursive/tree_trace.py 产生(env.rao.trace_dir 开启时)。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def render(events, slot_filter=None, show_prompt=True, out=sys.stdout):
    by_slot = defaultdict(list)
    for e in events:
        if "slot" in e:
            by_slot[e["slot"]].append(e)
    for slot in sorted(by_slot):
        if slot_filter is not None and slot != slot_filter:
            continue
        ev = by_slot[slot]
        nodes = {e["node_uid"]: e for e in ev if e["type"] == "node_open"}
        closes = {e["node_uid"]: e for e in ev if e["type"] == "node_close"}
        turns = defaultdict(list)
        for e in ev:
            if e["type"] == "turn":
                turns[e["node_uid"]].append(e)
        children = defaultdict(list)
        for uid, n in nodes.items():
            if n["parent_uid"]:
                children[n["parent_uid"]].append(uid)
        # 委托轮的 turn 事件写在子返回之前,env_result 是占位文本;真正的结果是之后的 child_report。
        # 按"父 uid + 事件顺序"把战报配回对应的委托轮,渲染时接在那一轮下面。
        reports_by_parent = defaultdict(list)
        for e in ev:
            if e["type"] == "child_report":
                reports_by_parent[e["parent_uid"]].append(e)
        report_for_turn = {}
        for uid in nodes:
            q = list(reports_by_parent.get(uid, []))
            for t in turns[uid]:
                if t["is_delegation"] and "(delegated" in str(t["env_result"]) and q:
                    report_for_turn[(uid, t["turn_index"])] = q.pop(0)
        reset = next((e for e in ev if e["type"] == "reset"), {})
        roots = [uid for uid, n in nodes.items() if n["parent_uid"] is None]

        print("=" * 100, file=out)
        print(f"SLOT {slot}  tag={reset.get('tag')}  task={reset.get('task_id')}  "
              f"difficulty={reset.get('difficulty')}  targets={reset.get('target_items')}", file=out)
        print(f"initial_inventory={reset.get('initial_inventory')}", file=out)

        # ---- 树结构
        print("-" * 100, file=out)
        print("TREE:", file=out)

        def tree_line(uid, indent):
            n, c = nodes[uid], closes.get(uid, {})
            mark = "✓" if c.get("success", 0) >= 1 else "✗"
            print(f"{'    ' * indent}{mark} d{n['depth']} [{uid[:8]}] {n['goal_text']}  "
                  f"(turns={c.get('turns', '?')}, close={c.get('reason', 'OPEN')}, "
                  f"opened@r{n['round']}, budget={n['budget_total']})"
                  + (f"  ctx='{n['context']}'" if n.get("context") else "")
                  + (f"  ERROR" if c.get("error") else ""), file=out)
            for ch in children.get(uid, []):
                tree_line(ch, indent + 1)

        for r in roots:
            tree_line(r, 0)

        # ---- 每个节点的对话(按开张顺序)
        for uid in sorted(nodes, key=lambda u: (nodes[u]["round"], nodes[u]["depth"])):
            n, c = nodes[uid], closes.get(uid, {})
            print("-" * 100, file=out)
            print(f"NODE d{n['depth']} [{uid[:8]}] parent={str(n['parent_uid'])[:8]}  goal: {n['goal_text']}", file=out)
            if n.get("context"):
                print(f"  context from parent: {n['context']}", file=out)
            for t in turns[uid]:
                print(f"  --- turn {t['turn_index']} (round {t['round']}, budget_left_after={t['budget_left_after']}"
                      f"{', DELEGATION' if t['is_delegation'] else ''}) ---", file=out)
                if show_prompt:
                    print("  [PROMPT]", file=out)
                    for line in t["prompt"].splitlines():
                        print("    " + line, file=out)
                print("  [MODEL OUTPUT]", file=out)
                for line in t["model_output"].splitlines():
                    print("    " + line, file=out)
                print(f"  [PARSED ACTION] {t['parsed_action']}", file=out)
                rep = report_for_turn.get((uid, t["turn_index"]))
                if rep is not None:
                    print(f"  [RESULT = sub-agent {rep['child_uid'][:8]} report, arrived round {rep['round']}]", file=out)
                    for line in str(rep["report_text"]).splitlines():
                        print("    " + line, file=out)
                else:
                    print("  [RESULT]", file=out)
                    for line in str(t["env_result"]).splitlines():
                        print("    " + line, file=out)
            print(f"  => close: success={c.get('success')} reason={c.get('reason')} turns={c.get('turns')}", file=out)
            if c.get("error"):
                print("  [ERROR]\n" + "\n".join("    " + l for l in c["error"].splitlines()), file=out)

        # ---- 节点间交互
        rel = [e for e in ev if e["type"] in ("delegate", "child_report", "refusal")]
        if rel:
            print("-" * 100, file=out)
            print("INTERACTIONS:", file=out)
            for e in rel:
                if e["type"] == "delegate":
                    print(f"  r{e['round']}: {e['parent_uid'][:8]} --delegate--> {e['child_uid'][:8]}  "
                          f"{e['goal_text']}" + (f"  ctx='{e['context']}'" if e.get("context") else ""), file=out)
                elif e["type"] == "child_report":
                    print(f"  r{e['round']}: {e['child_uid'][:8]} --report({e['child_reason']}, "
                          f"success={e['child_success']})--> {e['parent_uid'][:8]}", file=out)
                else:
                    print(f"  r{e['round']}: {e['node_uid'][:8]} delegation REFUSED", file=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--slot", type=int, default=None)
    ap.add_argument("--no-prompt", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    events = load(a.path)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            render(events, a.slot, not a.no_prompt, out=f)
        print(f"written: {a.out}")
    else:
        render(events, a.slot, not a.no_prompt)


if __name__ == "__main__":
    main()
