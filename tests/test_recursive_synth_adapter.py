# -*- coding: utf-8 -*-
"""步骤 3 端到端 CPU 测试:真实 SynthTextCraftEnv(不经 Ray)+ 真实适配器 + 真实编排器 + 真实 projection。

  E1  prompt:flat 的每一行都保留、无 <think>、新增内容只有委托相关
  E2  parse_delegation 文法:冒号可省 / 数量带 x / 逗号或 and 多目标 / 竖线 context / 畸形 → None
  E3  gold 轨迹回放(无委托):经编排器跑通,root won,记录 depth=0 success=1 —— 与 flat 等价
  E4  人造委托序列:root 查配方 → 委托中间品 → 子用 gold 动作做出 → goal_met 弹栈 → root 合成目标 → won
  E5  上下文隔离:子的观测含自己的 goal/context,不含 root 查到的配方;子查过之后只进子的笔记本
  E6  连续四层委托不触发环境循环检测(NOOP filler)
  E7  节点私有库存快照互不影响

运行:source SDAR_RAO/env_rao.sh && cd "$SDAR_REPO" && python tests/test_recursive_synth_adapter.py
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_system.environments.env_package.textcraft_synth.projection import textcraft_synth_projection  # noqa: E402
from agent_system.environments.env_package.textcraft_synth.recursive_adapter import (              # noqa: E402
    TextCraftSynthRecursiveAdapter)
from agent_system.environments.env_package.textcraft_synth.synth_core import (                     # noqa: E402
    NOOP_ACTION, SynthTextCraftEnv, get_shared_recipe_db, gold_to_actions, load_tasks)
from agent_system.environments.prompts.textcraft_synth import TEXTCRAFT_SYNTH_TEMPLATE_NO_HIS       # noqa: E402
from agent_system.environments.prompts.textcraft_synth_recursive import (                          # noqa: E402
    TEXTCRAFT_SYNTH_RECURSIVE_TEMPLATE_NO_HIS)
from agent_system.recursive.orchestrator import RecursiveEnvironmentManager                        # noqa: E402
from agent_system.recursive.protocol import CLOSE_GOAL_MET                                         # noqa: E402


class LocalSynthEnvs:
    """TextCraftSynthEnvs 的无 Ray 替身,接口相同。每个槽跑指定的任务。"""

    def __init__(self, tasks, max_steps=60):
        self.tasks = tasks
        self.envs = [SynthTextCraftEnv(get_shared_recipe_db()) for _ in tasks]
        self.max_steps = max_steps

    def reset(self):
        outs = [e.reset(t, max_steps_override=self.max_steps, append_state_block=False, loop_detection=False)
                for e, t in zip(self.envs, self.tasks)]
        return [o for o, _ in outs], [i for _, i in outs]

    def step(self, actions):
        outs = [e.step(a) for e, a in zip(self.envs, actions)]
        return ([o for o, _, _, _ in outs], np.array([r for _, r, _, _ in outs]),
                np.array([d for _, _, d, _ in outs]), [i for _, _, _, i in outs])

    def close(self):
        pass


def make_cfg(history_length=2, per_agent=25, max_depth=6, trace_dir=None):
    cfg = SimpleNamespace(env=SimpleNamespace(
        max_steps=60, history_length=history_length,
        rao={"per_agent_max_steps": per_agent, "max_depth": max_depth, "trace_dir": trace_dir}))
    cfg.env.get = lambda k, d=None: getattr(cfg.env, k, d)
    return cfg


def wrap(a):
    """把裸动作包成模型输出格式,经真实 projection。"""
    return f"<thought>ok</thought><action>{a}</action>"


def pick_two_step_task():
    """找一个 gold 恰好两步的 easy 任务:第一步做中间品,第二步做目标。"""
    for t in load_tasks("train", ["easy"]):
        g = t["misc"]["gold_trajectory"]
        if len(g) == 2 and g[0]["target"][0] != g[1]["target"][0]:
            return t
    raise RuntimeError("no 2-step easy task")


passed = 0
def check(cond, msg):
    global passed
    assert cond, msg
    passed += 1
    print(f"  ✓ {msg}")


# ----------------------------------------------------------------------------- E1
def test_E1_prompt():
    print("E1 prompt 只增不删、无 <think>")
    task = "Craft the following items: 3x a1_i5\nBudget: you have 25 steps in total."
    flat = TEXTCRAFT_SYNTH_TEMPLATE_NO_HIS.format(current_observation=task).splitlines()
    rec = TEXTCRAFT_SYNTH_RECURSIVE_TEMPLATE_NO_HIS.format(current_observation=task).splitlines()
    missing = [l for l in flat if l not in rec]
    check(not missing, f"flat 的 {len(flat)} 行全部保留")
    added = [l for l in rec if l not in flat]
    check(all(("delegat" in l.lower() or "sub-agent" in l.lower() or "subtask" in l.lower()
               or l.strip() == "" or "inventory" in l.lower() or "fresh context" in l.lower()
               or "Example:" in l or "Launch a" in l) for l in added),
          f"新增 {len(added)} 行全部与委托相关")
    check("<think>" not in "\n".join(rec) and "<thought>" in "\n".join(rec) and "<action>" in "\n".join(rec),
          "标签为 <thought>/<action>,无 <think>")


# ----------------------------------------------------------------------------- E2
def test_E2_grammar():
    print("E2 委托文法")
    ad = TextCraftSynthRecursiveAdapter(make_cfg())
    cases = {
        "delegate: craft 4 a0_i1": ({"a0_i1": 4}, ""),
        "delegate craft 4 a0_i1": ({"a0_i1": 4}, ""),
        "delegate: craft 4x a0_i1, 2x a0_i3": ({"a0_i1": 4, "a0_i3": 2}, ""),
        "delegate: craft 4 a0_i1 and 2 a0_i3 | for a1_i2": ({"a0_i1": 4, "a0_i3": 2}, "for a1_i2"),
        "delegate: craft 3 a0_i1, 1 a0_i1": ({"a0_i1": 4}, ""),
    }
    for text, (goal, ctx) in cases.items():
        r = ad.parse_delegation(text)
        check(r is not None and r.goal == goal and r.context == ctx, f"{text!r} → {goal} ctx={ctx!r}")
    # 2026-08-25 冒烟 21459012 暴露的真实写法:模型把笔记本里的配方行(craft N item using ...)整行抄进委托
    real = [
        ("delegate: craft 2 t0_i2_10 using 2 t2_i1_12, 2 t1_i1, craft 2 t7_i1 using 2 raw_t0 | needed for t6_i3_16",
         {"t0_i2_10": 2, "t7_i1": 2}, "needed for t6_i3_16"),          # using 配料表里含逗号,不能误当目标
        ("delegate: craft 2 m5_i1 using 1 m4_ore | these are required for m5_i2, craft 4 m5_i1_15 using 1 m7_ore | these too",
         {"m5_i1": 2, "m5_i1_15": 4},
         "these are required for m5_i2, craft 4 m5_i1_15 using 1 m7_ore | these too"),   # 竖线后的 craft 补为目标,context 原样
        ("delegate: craft 3 m5_i2 using 1 m5_i1, 2 m5_i1_15 | needed for making m5_i2",
         {"m5_i2": 3}, "needed for making m5_i2"),
        ("delegate: craft 4 a0_i1.", {"a0_i1": 4}, ""),                 # 尾部标点
        # 父经 context 通道把配方传给子(官方 launch_subagent 的 context 参数):
        # 目标不重复计数,context 原样保留(含 using)
        ("delegate: craft 2 m5_i1 | recipe: craft 2 m5_i1 using 1 m4_ore, then you are done",
         {"m5_i1": 2}, "recipe: craft 2 m5_i1 using 1 m4_ore, then you are done"),
    ]
    for text, want_goal, want_ctx in real:
        req = ad.parse_delegation(text)
        check(req is not None and req.goal == want_goal and req.context == want_ctx,
              f"{text[:60]!r}… → {req.goal if req else None} ctx={req.context if req else None!r}")
    # 目标段有解析不了的内容 → 整条拒绝,绝不静默取子集
    for bad in ["delegate: craft 2 a0_i1, some words here", "delegate: craft 2 a0_i1 and then rest"]:
        check(ad.parse_delegation(bad) is None, f"{bad!r} → None(不静默取子集)")
    for bad in ["craft 4 a0_i1 using 2 raw_a4", "get_info a0_i1", "delegate: craft", "delegate: craft zero a0_i1"]:
        check(ad.parse_delegation(bad) is None, f"{bad!r} → None")
    # 经真实 projection(会小写)后仍可解析
    acts, valids = textcraft_synth_projection([wrap("Delegate: Craft 2 A0_I1 | Needed For A1_I2")])
    r = ad.parse_delegation(acts[0])
    check(valids[0] == 1 and r is not None and r.goal == {"a0_i1": 2}, "经 projection 小写后可解析")


# ----------------------------------------------------------------------------- E3
def test_E3_gold_replay_no_delegation():
    print("E3 gold 回放(无委托)= flat 等价")
    task = pick_two_step_task()
    envs = LocalSynthEnvs([task])
    mgr = RecursiveEnvironmentManager(envs, textcraft_synth_projection, make_cfg(),
                                      adapter=TextCraftSynthRecursiveAdapter(make_cfg()))
    obs, infos = mgr.reset()
    check(obs["text"][0].startswith("You are an agent in a crafting game"), "root 首轮观测是完整 prompt")
    check(task["goal"] in obs["text"][0] and "Budget: you have 25 steps" in obs["text"][0], "含任务目标与 25 步预算行")
    done = False
    for a in gold_to_actions(task["misc"]["gold_trajectory"]):
        o, r, d, inf = mgr.step([wrap(a)])
        done = bool(d[0])
    check(done and inf[0]["won"] is True, "gold 两步跑完,环境 won")
    recs = mgr.collect_node_records()[0]
    check(len(recs) == 1 and recs[0].depth == 0 and recs[0].success == 1.0, "单节点树,root success=1")


# ----------------------------------------------------------------------------- E4
def test_E4_delegation_sequence():
    print("E4 人造委托序列")
    task = pick_two_step_task()
    g = task["misc"]["gold_trajectory"]
    mid_item, mid_count = g[0]["target"][0], g[0]["result_count"]
    envs = LocalSynthEnvs([task])
    mgr = RecursiveEnvironmentManager(envs, textcraft_synth_projection, make_cfg(),
                                      adapter=TextCraftSynthRecursiveAdapter(make_cfg()))
    mgr.reset()
    root = mgr.stacks[0][0]
    tgt = list(task["misc"]["target_items"])[0]
    mgr.step([wrap(f"get_info {tgt}")])                                  # root 第 1 步
    o, r, d, inf = mgr.step([wrap(f"delegate: craft {mid_count} {mid_item} | ingredient for {tgt}")])
    check(len(mgr.stacks[0]) == 2 and mgr.stacks[0][-1].goal == {mid_item: mid_count}, "子压栈,目标正确")
    child = mgr.stacks[0][-1]
    child_obs = o["text"][0]
    check(f"Make sure your inventory contains at least: {mid_count}x {mid_item}" in child_obs   # 子目标措辞(2026-08-25 决定)
          and f"Context provided from parent agent: ingredient for {tgt}" in child_obs,
          "子首轮观测含自己的目标行与父给的 context")
    check(envs.envs[0].inventory == task["misc"]["initial_inventory"], "委托那轮库存未变(NOOP)")
    o, r, d, inf = mgr.step([wrap(gold_to_actions([g[0]])[0])])         # 子做中间品
    check(len(mgr.stacks[0]) == 1, "子达标弹栈")
    rec = mgr.finished_records[0][0]
    check(rec.close_reason == CLOSE_GOAL_MET and rec.success == 1.0 and rec.turns == 1, "子记录 goal_met/1/1 步")
    check(root.children_success == [1.0] and "SUCCEEDED" in root.history[-1].result
          and f"Budget used by subagent: 1/25" in root.history[-1].result, "父收到战报(含官方措辞的预算行)")
    check("Sub-agent report" in o["text"][0] and o["text"][0].count("Step 1:") == 1, "root 观测切回自己视角,含战报")
    o, r, d, inf = mgr.step([wrap(gold_to_actions([g[1]])[0])])         # root 合成目标
    check(bool(d[0]) and inf[0]["won"] is True and float(r[0]) == 1.0, "root 合成目标,环境 won,奖励 1")
    recs = {x.depth: x for x in mgr.collect_node_records()[0]}
    check(recs[0].success == 1.0 and recs[0].children_success == [1.0] and recs[1].success == 1.0,
          "最终记录:root 1 / 子 1 / root 子代账本 [1.0]")


# ----------------------------------------------------------------------------- E5
def test_E5_context_isolation():
    print("E5 上下文隔离(状态块按节点私有)")
    task = pick_two_step_task()
    g = task["misc"]["gold_trajectory"]
    mid_item, mid_count = g[0]["target"][0], g[0]["result_count"]
    tgt = list(task["misc"]["target_items"])[0]
    envs = LocalSynthEnvs([task])
    mgr = RecursiveEnvironmentManager(envs, textcraft_synth_projection, make_cfg(),
                                      adapter=TextCraftSynthRecursiveAdapter(make_cfg()))
    mgr.reset()
    root = mgr.stacks[0][0]
    mgr.step([wrap(f"get_info {tgt}")])
    check(tgt in root.scratch["known_recipes"], "root 查配方后进 root 的笔记本")
    o, *_ = mgr.step([wrap(f"delegate: craft {mid_count} {mid_item}")])
    child = mgr.stacks[0][-1]
    check("Recipes you have learned" not in o["text"][0] and tgt not in o["text"][0].split("Budget:")[0],
          "子首轮观测不含 root 的配方笔记本")
    o, *_ = mgr.step([wrap(f"get_info {mid_item}")])
    check(mid_item in child.scratch["known_recipes"] and mid_item not in root.scratch["known_recipes"],
          "子查的配方只进子的笔记本,root 的不变")
    check("Recipes you have learned so far" in o["text"][0] and f"craft {mid_count} {mid_item}" in o["text"][0]
          and f"craft " + str(g[1]["result_count"]) + f" {tgt}" not in o["text"][0],
          "子第二轮观测的状态块只有子自己查的配方")


# ----------------------------------------------------------------------------- E6
def test_E6_no_loop_detection_on_chained_delegation():
    print("E6 连续四层委托不触发环境循环检测")
    task = pick_two_step_task()
    envs = LocalSynthEnvs([task])
    mgr = RecursiveEnvironmentManager(envs, textcraft_synth_projection, make_cfg(max_depth=6),
                                      adapter=TextCraftSynthRecursiveAdapter(make_cfg()))
    mgr.reset()
    for k in range(5):
        o, r, d, inf = mgr.step([wrap(f"delegate: craft 1 x{k}")])
        check(not bool(d[0]) and "Loop detected" not in o["text"][0], f"第 {k+1} 层委托后整局未被误杀")
    check(len(mgr.stacks[0]) == 6 and mgr.stacks[0][-1].depth == 5, "栈深 6(root + 5 层)")
    check(envs.envs[0]._repeat_count <= 1, "环境循环计数未累积")


# ----------------------------------------------------------------------------- E7
def test_E7_private_inventory_snapshot():
    print("E7 节点私有库存快照")
    task = pick_two_step_task()
    envs = LocalSynthEnvs([task])
    mgr = RecursiveEnvironmentManager(envs, textcraft_synth_projection, make_cfg(),
                                      adapter=TextCraftSynthRecursiveAdapter(make_cfg()))
    mgr.reset()
    root = mgr.stacks[0][0]
    mgr.step([wrap("inventory")])
    check(root.scratch["inv_step"] == 1 and root.scratch["inv_snapshot"], "root 快照 as of step 1")
    mgr.step([wrap("delegate: craft 1 whatever")])
    child = mgr.stacks[0][-1]
    check(child.scratch["inv_snapshot"] is None, "子开张时没有快照")
    o, *_ = mgr.step([wrap("inventory")])
    check(child.scratch["inv_step"] == 1 and "as of step 1" in o["text"][0], "子自己查后 as of step 1(子的步数)")
    check(root.scratch["inv_step"] == 1, "root 的快照步数不受子影响")


# ----------------------------------------------------------------------------- E8
def test_E8_tree_trace():
    print("E8 完整轨迹记录(tree_trace)")
    import glob
    import json
    import subprocess
    import tempfile
    tmp = tempfile.mkdtemp(prefix="rao_trace_")
    task = pick_two_step_task()
    g = task["misc"]["gold_trajectory"]
    mid_item, mid_count = g[0]["target"][0], g[0]["result_count"]
    tgt = list(task["misc"]["target_items"])[0]
    cfg = make_cfg(trace_dir=tmp)
    envs = LocalSynthEnvs([task])
    mgr = RecursiveEnvironmentManager(envs, textcraft_synth_projection, cfg,
                                      adapter=TextCraftSynthRecursiveAdapter(cfg), trace_tag="train")
    mgr.reset()
    mgr.step([wrap(f"get_info {tgt}")])
    mgr.step([wrap(f"delegate: craft {mid_count} {mid_item} | for {tgt}")])
    mgr.step([wrap(gold_to_actions([g[0]])[0])])
    mgr.step([wrap(gold_to_actions([g[1]])[0])])
    mgr.collect_node_records()
    files = glob.glob(os.path.join(tmp, "train", "reset_*.jsonl"))
    check(len(files) == 1, f"生成 1 个轨迹文件 {os.path.basename(files[0])}")
    ev = [json.loads(l) for l in open(files[0], encoding="utf-8") if l.strip()]
    types = [e["type"] for e in ev]
    check(types.count("reset") == 1 and types.count("node_open") == 2 and types.count("node_close") == 2,
          f"reset×1 / node_open×2 / node_close×2")
    check(types.count("turn") == 4 and types.count("delegate") == 1 and types.count("child_report") == 1,
          "turn×4 / delegate×1 / child_report×1")
    turns = [e for e in ev if e["type"] == "turn"]
    t_del = [t for t in turns if t["is_delegation"]][0]
    check(t_del["model_output"].startswith("<thought>") and "delegate:" in t_del["parsed_action"]
          and t_del["prompt"].startswith("You are an agent in a crafting game"),
          "委托轮记录了原始模型输出、解析后动作、完整 prompt")
    child_open = [e for e in ev if e["type"] == "node_open" and e["depth"] == 1][0]
    child_turn = [t for t in turns if t["node_uid"] == child_open["node_uid"]][0]
    check(child_open["parent_uid"] == t_del["node_uid"] and child_open["context"] == f"for {tgt}"
          and f"Context provided from parent agent: for {tgt}" in child_turn["prompt"],
          "子节点 node_open 含父指针与 context;子的 prompt 里能看到 context")
    rep = [e for e in ev if e["type"] == "child_report"][0]
    check(rep["child_uid"] == child_open["node_uid"] and rep["parent_uid"] == t_del["node_uid"]
          and rep["child_success"] == 1.0 and "SUCCEEDED" in rep["report_text"], "child_report 边正确")
    last_root_turn = [t for t in turns if t["node_uid"] == t_del["node_uid"]][-1]
    check("Sub-agent report" in last_root_turn["prompt"], "父在收到战报后的下一轮 prompt 含战报")
    # 阅读脚本能跑
    r = subprocess.run([sys.executable, os.path.join(REPO_ROOT, "scripts_rso", "render_tree_trace.py"),
                        files[0], "--no-prompt"], capture_output=True, text=True)
    check(r.returncode == 0 and "TREE:" in r.stdout and "--delegate-->" in r.stdout and "--report(" in r.stdout,
          "render_tree_trace.py 渲染出树结构与交互边")


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------------------- E9
def test_E9_child_absolute_criterion():
    print("E9 子目标 = 绝对可用量 + 开张即判(2026-08-25 决定)")
    from agent_system.recursive.protocol import CLOSE_GOAL_MET
    task = pick_two_step_task()
    g = task["misc"]["gold_trajectory"]
    mid_item, mid_count = g[0]["target"][0], g[0]["result_count"]
    tgt = list(task["misc"]["target_items"])[0]
    cfg = make_cfg()
    mgr = RecursiveEnvironmentManager(LocalSynthEnvs([task]), textcraft_synth_projection, cfg,
                                      adapter=TextCraftSynthRecursiveAdapter(cfg))
    mgr.reset()
    root = mgr.stacks[0][0]
    check("Craft the following items" in mgr.last_prompt[0] and "on top of what you already have" in mgr.last_prompt[0],
          "root 措辞与 Note 句与 flat/官方一致")
    mgr.step([wrap(gold_to_actions([g[0]])[0])])     # root 自己先把中间产物做出来
    out = mgr.step([wrap(f"delegate: craft {mid_count} {mid_item}")])   # 再委托同样的东西
    check(len(mgr.stacks[0]) == 1, "子开张即判:目标已满足,0 步关闭")
    rec = mgr.finished_records[0][0]
    check(rec.depth == 1 and rec.close_reason == CLOSE_GOAL_MET and rec.success == 1.0 and rec.turns == 0
          and rec.preexisting, "记录:goal_met / 0 步 / preexisting=True")
    check("already available" in root.history[-1].result, "父收到 'already available' 战报")
    check(out[0]["text"][0].startswith("You are an agent"), "root 继续行动")
    # 部分已有:委托 (已有数量 + 1),子只需做出缺口
    mgr2 = RecursiveEnvironmentManager(LocalSynthEnvs([task]), textcraft_synth_projection, cfg,
                                       adapter=TextCraftSynthRecursiveAdapter(cfg))
    mgr2.reset()
    mgr2.step([wrap(gold_to_actions([g[0]])[0])])
    out = mgr2.step([wrap(f"delegate: craft {mid_count + 1} {mid_item}")])
    child = mgr2.stacks[0][-1]
    check(len(mgr2.stacks[0]) == 2 and child.goal_text.startswith("Make sure your inventory contains at least"),
          "缺口未满足 → 子压栈;子目标行用'库存里有 N 个'措辞")
    check("Items already in the inventory count toward the target" in out[0]["text"][0]
          and "on top of what you already have" not in out[0]["text"][0], "子 prompt 的 Note 句换成绝对可用量口径")
    mgr2.step([wrap(gold_to_actions([g[0]])[0])])    # 子再做一份 → 绝对数量 ≥ 目标
    check(len(mgr2.stacks[0]) == 1 and mgr2.finished_records[0][0].success == 1.0
          and mgr2.finished_records[0][0].turns == 1, "子做出缺口后 1 步成功(不必再做满 N 个)")


# ----------------------------------------------------------------------------- E10
def test_E10_stuck_per_node():
    print("E10 节点级循环检测(镜像官方 agent.py:90-109)")
    from agent_system.recursive.protocol import CLOSE_STUCK
    from agent_system.environments.env_package.textcraft_synth.recursive_adapter import stuck_in_loop
    check(stuck_in_loop(["a", "a", "a", "a"]) and not stuck_in_loop(["a", "a", "a"]), "周期 1:4 次触发,3 次不触发")
    check(stuck_in_loop(["a", "b"] * 4) and not stuck_in_loop(["a", "b"] * 3 + ["a"]), "周期 2:A B ×4 触发")
    check(stuck_in_loop(["a", "b", "c"] * 4) and not stuck_in_loop(["a", "b", "c", "d"] * 4), "周期 3 触发;周期 4 超出 window 不触发")
    task = pick_two_step_task()
    tgt = list(task["misc"]["target_items"])[0]
    cfg = make_cfg()
    mgr = RecursiveEnvironmentManager(LocalSynthEnvs([task]), textcraft_synth_projection, cfg,
                                      adapter=TextCraftSynthRecursiveAdapter(cfg))
    mgr.reset()
    root = mgr.stacks[0][0]
    mgr.step([wrap("delegate: craft 99 zz_i9")])     # 子目标做不出来
    child = mgr.stacks[0][-1]
    for _ in range(3):
        mgr.step([wrap("inventory")])
    check(len(mgr.stacks[0]) == 2, "子 3 次 inventory 未触发")
    out = mgr.step([wrap("inventory")])              # 第 4 次 → 子打转
    check(len(mgr.stacks[0]) == 1 and not mgr.episode_done[0], "子按 stuck 关闭,root 继续(整局未结束)")
    rec = [r for r in mgr.finished_records[0] if r.uid == child.uid][0]
    check(rec.close_reason == CLOSE_STUCK and rec.success == 0.0 and "FAILED (stuck_in_loop)" in root.history[-1].result,
          "子记录 stuck_in_loop,父收到失败战报")
    # 环境级检测已关:root 接着再连发 inventory,不会被环境误杀;累计到 root 自己 4 次才按 stuck 结束整局
    for k in range(3):
        out = mgr.step([wrap("inventory")])
        check(not mgr.episode_done[0], f"root 第 {k+1} 次 inventory 未结束")
    out = mgr.step([wrap("inventory")])
    recs = {r.depth: r.close_reason for r in mgr.finished_records[0]}
    check(mgr.episode_done[0] and recs[0] == CLOSE_STUCK, "root 自己 4 次 → 整局按 stuck_in_loop 结束")


# ----------------------------------------------------------------------------- E11
def test_E11_factory_kwargs_with_real_yaml():
    print("E11 工厂 env_kwargs(真实 yaml + 冒烟脚本同款覆盖键,不碰 Ray)")
    from omegaconf import OmegaConf
    from agent_system.environments.env_package.textcraft_synth.recursive_factory import (
        build_recursive_env_kwargs, DEFAULT_ENV_MAX_STEPS)
    from agent_system.recursive.budget import PerAgentBudget
    base = OmegaConf.load(os.path.join(REPO_ROOT, "verl", "trainer", "config", "ppo_trainer.yaml"))
    # 注意:OmegaConf.from_dotlist 不认 hydra 的 "+" 前缀(会变成叫 "+env" 的顶层键),这里不带 "+";
    # 数值故意与默认值不同(默认 25/6/4/3),否则"读到默认值"和"真的读到 yaml"分不清。
    overrides = OmegaConf.from_dotlist([
        "env.env_name=textcraft_synth", "env.max_steps=100", "env.history_length=3", "env.rollout.n=8",
        "env.rao.per_agent_max_steps=30", "env.rao.max_depth=5", "env.rao.state_block_scope=node",
        "env.rao.stuck_threshold=5", "env.rao.stuck_window=2",
        "env.rao.trace_dir=/tmp/x", "env.textcraft_synth.train_difficulties=[easy]",
        "env.textcraft_synth.val_difficulties=[easy,medium]", "env.textcraft_synth.val_split=val100",
    ])
    cfg = OmegaConf.merge(base, overrides)
    check("rao" in cfg.env and "+env" not in cfg, "覆盖键落在 env.rao 下(没有产生 '+env' 假键)")
    kw = build_recursive_env_kwargs(cfg)
    check(kw["append_state_block"] is False and kw["loop_detection"] is False, "递归三键:状态块关、槽级循环检测关")
    check(kw["max_steps"] == DEFAULT_ENV_MAX_STEPS, f"底层 env 上限 = {DEFAULT_ENV_MAX_STEPS}(不用 env.max_steps)")
    check(kw["train_difficulties"] == ["easy"] and kw["val_difficulties"] == ["easy", "medium"] and kw["val_split"] == "val100",
          "难度/val_split 与 flat 装配同口径")
    b = PerAgentBudget.from_config(dict(cfg.env.get("rao", {})))
    check(b.per_agent_steps == 30 and b.max_depth == 5, "预算真的从 yaml 读到(30/5,非默认值)")
    ad = TextCraftSynthRecursiveAdapter(cfg)
    check(ad.history_length == 3 and ad.stuck_threshold == 5 and ad.stuck_window == 2, "适配器真的从 yaml 读到(3/5/2,非默认值)")
    cfg2 = OmegaConf.merge(cfg, OmegaConf.from_dotlist(["env.rao.env_max_steps=500"]))
    check(build_recursive_env_kwargs(cfg2)["max_steps"] == 500, "env.rao.env_max_steps 可覆盖")


if __name__ == "__main__":
    for fn in [test_E1_prompt, test_E2_grammar, test_E3_gold_replay_no_delegation,
               test_E4_delegation_sequence, test_E5_context_isolation,
               test_E6_no_loop_detection_on_chained_delegation, test_E7_private_inventory_snapshot,
               test_E8_tree_trace, test_E9_child_absolute_criterion, test_E10_stuck_per_node,
               test_E11_factory_kwargs_with_real_yaml]:
        fn()
    print(f"\n=== 步骤 3 端到端测试全部通过({passed} 项断言)===")
