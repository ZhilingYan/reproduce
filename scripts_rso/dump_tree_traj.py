# -*- coding: utf-8 -*-
"""把 tree_trace 里的一棵 agent 树导成一份人能读的 markdown。

和 scripts_rao/render_tree_trace.py 的区别:那个是速览用的纯文本,这个是给人精读用的,
按"节点为单位"组织,每一轮都完整给出四样东西——喂给模型的整段 prompt、模型的原始输出、
解析出来的动作、环境返回的观测。父节点发布子任务的那一轮会被单独标出来,并把子节点
收到的开场 prompt 贴在紧邻的位置,方便对照"父亲说了什么"和"儿子看到了什么"。

用法:
    python scripts_rso/dump_tree_traj.py <reset_XXXX.jsonl> --slot N --out <输出.md>

不带 --slot 时会把该文件里所有树的规模列出来供挑选。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict


def load_slot(path, slot):
    """只读出指定 slot 的事件,避免把几十兆的整个文件读进内存。"""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("slot") == slot or e.get("type") == "reset" and e.get("slot") == slot:
                out.append(e)
    return out


def survey(path):
    per = defaultdict(lambda: {"nodes": 0, "maxd": 0, "turns": 0, "deleg": 0,
                               "root_ok": None, "goal": ""})
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            s = e.get("slot")
            if s is None:
                continue
            t = e.get("type")
            if t == "node_open":
                per[s]["nodes"] += 1
                per[s]["maxd"] = max(per[s]["maxd"], e.get("depth", 0))
                if e.get("depth") == 0:
                    per[s]["goal"] = str(e.get("goal_text", ""))[:60]
            elif t == "turn":
                per[s]["turns"] += 1
            elif t == "delegate":
                per[s]["deleg"] += 1
            elif t == "node_close" and e.get("depth") == 0:
                per[s]["root_ok"] = e.get("success")
    print(f"{'槽':>5} {'深度':>4} {'节点':>4} {'委派':>4} {'轮次':>4} {'根成功':>6}  根目标")
    for s in sorted(per):
        d = per[s]
        print(f"{s:>5} {d['maxd']:>4} {d['nodes']:>4} {d['deleg']:>4} {d['turns']:>4} "
              f"{str(d['root_ok']):>6}  {d['goal']}")


def fence(text, lang=""):
    """套代码块。内容里若已有 ``` 就改用更长的围栏,免得把 markdown 撑破。"""
    t = str(text)
    f = "```"
    while f in t:
        f += "`"
    return f"{f}{lang}\n{t}\n{f}\n"


def obs_to_text(env_result):
    """环境返回的观测。可能是字符串,也可能是配方查询返回的结构化列表,后者原样美化输出。"""
    if isinstance(env_result, str):
        return env_result
    return json.dumps(env_result, ensure_ascii=False, indent=2)


def render(events, out, path, slot):
    reset = next((e for e in events if e["type"] == "reset"), {})
    opens = {e["node_uid"]: e for e in events if e["type"] == "node_open"}
    closes = {e["node_uid"]: e for e in events if e["type"] == "node_close"}
    turns = defaultdict(list)
    for e in events:
        if e["type"] == "turn":
            turns[e["node_uid"]].append(e)
    delegs = [e for e in events if e["type"] == "delegate"]
    deleg_by_parent_round = {(e["parent_uid"], e["round"]): e for e in delegs}
    reports = {e["child_uid"]: e for e in events if e["type"] == "child_report"}
    refusals = [e for e in events if e["type"] == "refusal"]

    children = defaultdict(list)
    for uid, o in opens.items():
        if o.get("parent_uid"):
            children[o["parent_uid"]].append(uid)
    root = next((u for u, o in opens.items() if o.get("depth") == 0), None)

    w = out.write
    w(f"# 一棵完整的 RAO agent 树:从根节点到第 {max(o.get('depth',0) for o in opens.values())} 层\n\n")
    w("> 本文由 `scripts_rso/dump_tree_traj.py` 从训练过程落盘的 `tree_trace` 自动导出,\n")
    w("> 没有任何人工编辑或删减。每一轮都给出四样东西:喂给模型的**整段 prompt**、\n")
    w("> 模型的**原始输出**、解析出来的**动作**、环境返回的**观测**。\n\n")

    w("## 这棵树是哪来的\n\n")
    w(f"| 项 | 值 |\n|---|---|\n")
    w(f"| 来源文件 | `{path}` |\n")
    w(f"| batch 槽 | {slot}(一个槽就是一棵树) |\n")
    w(f"| 任务 id | `{reset.get('task_id')}` |\n")
    w(f"| 难度 | {reset.get('difficulty')} |\n")
    w(f"| 目标物品 | `{reset.get('target_items')}` |\n")
    w(f"| 初始背包 | `{reset.get('initial_inventory')}` |\n")
    w(f"| 节点数 | {len(opens)} |\n")
    w(f"| 委派次数 | {len(delegs)} |\n")
    w(f"| 总轮次 | {sum(len(v) for v in turns.values())} |\n")
    rc = closes.get(root, {})
    w(f"| 根节点结果 | success={rc.get('success')},关闭原因 `{rc.get('reason')}` |\n\n")

    # ---- 树形结构 ----
    w("## 树的形状\n\n")
    w("```\n")

    def draw(uid, prefix="", is_last=True):
        o = opens[uid]
        c = closes.get(uid, {})
        mark = "└─ " if is_last else "├─ "
        head = "" if o.get("depth") == 0 else mark
        pre = "(目标开局就已满足)" if o.get("preexisting") else ""
        w(f"{prefix}{head}[深度{o['depth']}] {o['goal_text']}  "
          f"→ success={c.get('success')} {c.get('reason','')} "
          f"{len(turns.get(uid,[]))}轮 {pre}\n")
        kids = sorted(children.get(uid, []), key=lambda k: opens[k]["round"])
        for i, k in enumerate(kids):
            nxt = prefix + ("" if o.get("depth") == 0 else ("   " if is_last else "│  "))
            draw(k, nxt, i == len(kids) - 1)

    if root:
        draw(root)
    w("```\n\n")

    # ---- 父亲怎么发布子任务:先给一张总表 ----
    w("## 父节点是怎么发布子任务的(总表)\n\n")
    w("委派动作的文本文法是 `delegate: <子目标> | <给子节点的补充说明>`,竖线后面是可选的。\n")
    w("解析器把竖线前面的部分变成子节点的目标,后面的部分原样作为 context 塞进子节点的开场 prompt。\n\n")
    w("| 轮 | 父节点(深度) | 发给子节点的目标 | context(竖线后面那段) | 子节点结果 |\n")
    w("|---|---|---|---|---|\n")
    for e in sorted(delegs, key=lambda x: x["round"]):
        po = opens.get(e["parent_uid"], {})
        cc = closes.get(e["child_uid"], {})
        ctx = str(e.get("context", "")) or "(空)"
        w(f"| {e['round']} | 深度{po.get('depth')} | {e['goal_text']} | {ctx} | "
          f"success={cc.get('success')} `{cc.get('reason','')}` |\n")
    w("\n")
    if refusals:
        w("被拒绝的委派:\n\n")
        for e in refusals:
            w(f"- 第 {e['round']} 轮,节点 `{e['node_uid'][:8]}`:{e['text']}\n")
        w("\n")

    # ---- 逐节点、逐轮的完整记录 ----
    w("---\n\n## 逐节点的完整记录\n\n")
    order = []

    def walk(uid):
        order.append(uid)
        for k in sorted(children.get(uid, []), key=lambda x: opens[x]["round"]):
            walk(k)

    if root:
        walk(root)

    for uid in order:
        o = opens[uid]
        c = closes.get(uid, {})
        d = o["depth"]
        w(f"\n### {'#' * min(d,3)} 节点 `{uid[:8]}` — 深度 {d}\n\n")
        w(f"- **目标**:{o['goal_text']}\n")
        if o.get("context"):
            w(f"- **父亲给的补充说明(context)**:{o['context']}\n")
        w(f"- **父节点**:{('根节点,没有父亲' if not o.get('parent_uid') else '`'+o['parent_uid'][:8]+'`')}\n")
        w(f"- **在第 {o['round']} 轮被创建**,预算 {o.get('budget_total')} 步\n")
        if o.get("preexisting"):
            w("- **注意**:这个子目标在节点开局时就已经满足了(背包里已经有足够的物品),"
              "所以它会立刻关闭、一轮都不跑。\n")
        w(f"- **结束**:success={c.get('success')},原因 `{c.get('reason')}`,"
          f"共 {len(turns.get(uid, []))} 轮\n")
        if reports.get(uid):
            w(f"- **回给父亲的战报**:\n\n{fence(reports[uid]['report_text'])}")
        w("\n")

        for t in sorted(turns.get(uid, []), key=lambda x: x["turn_index"]):
            dg = deleg_by_parent_round.get((uid, t["round"]))
            flag = "  ← **这一轮发布了子任务**" if dg else ""
            w(f"#### 第 {t['turn_index']} 轮(全局第 {t['round']} 轮){flag}\n\n")
            w("**喂给模型的完整 prompt:**\n\n")
            w(fence(t.get("prompt", "")))
            w("**模型的原始输出:**\n\n")
            w(fence(t.get("model_output", "")))
            w(f"**解析出来的动作:** `{t.get('parsed_action')}`"
              f"{'  (这是一个委派动作)' if t.get('is_delegation') else ''}\n\n")
            if dg:
                w("**这次委派被解析成:**\n\n")
                w(f"- 子目标:`{dg['goal_text']}`\n")
                w(f"- context:`{dg.get('context','') or '(空)'}`\n")
                w(f"- 新建的子节点:`{dg['child_uid'][:8]}`\n\n")
            w("**环境返回的观测:**\n\n")
            w(fence(obs_to_text(t.get("env_result", ""))))
            w(f"**这一轮的奖励:** {t.get('reward')}    **剩余预算:** {t.get('budget_left_after')}\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--slot", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.slot is None:
        survey(a.path)
        return
    ev = load_slot(a.path, a.slot)
    if not ev:
        print(f"槽 {a.slot} 没有事件", file=sys.stderr)
        sys.exit(1)
    f = open(a.out, "w", encoding="utf-8") if a.out else sys.stdout
    try:
        render(ev, f, a.path, a.slot)
    finally:
        if a.out:
            f.close()
            print(f"写到 {a.out}")


if __name__ == "__main__":
    main()
