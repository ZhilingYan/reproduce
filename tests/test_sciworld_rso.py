# -*- coding: utf-8 -*-
"""ScienceWorld 训练接入的 CPU 测试(无 JVM:envs 经 env_factory 注入假环境)。

骨架照 tests/test_search_rso.py:pytest 可选,尾部带手动 runner。
覆盖:projection / goal_progress 解析与双渲染 / RootChildBudget /
底层环境语义(组一致性、noop、answer、clip 增量奖励、焦死、固定 val 50)/
适配器(delegate 解析、budget 尾巴剥离、Φ 峰值推进、child answer 关闭)/
编排器端到端(委托→子汇报→父收报告→root 完成,G≥0 与 reward 补丁)。
"""
import sys

import numpy as np
from omegaconf import OmegaConf

try:
    import pytest
except Exception:
    pytest = None

from agent_system.environments.env_package.sciworld.budget import RootChildBudget
from agent_system.environments.env_package.sciworld.envs import (
    OFFICIAL_TASK_NAMES, build_sciworld_envs)
from agent_system.environments.env_package.sciworld.goal_progress import (
    parse_goal_progress, render_gt_plan, render_priv)
from agent_system.environments.env_package.sciworld.projection import sciworld_projection

GOAL_PROGRESS_SAMPLE = """Completed keys:\x20
----------------------------------------------------------------------------------------------------
Sequential Subgoals:
----------------------------------------------------------------------------------------------------
0\tfalse\tGoalFind\tfocus on substance
1\ttrue\tGoalChangeStateOfMatter\tsubstance is in a solid state
2\tfalse\tGoalChangeStateOfMatter\tsubstance is in a liquid state
----------------------------------------------------------------------------------------------------
Unordered and Optional Subgoals:
----------------------------------------------------------------------------------------------------
0\tfalse\tGoalInRoomWithObject\tbe in same location as lead
1\ttrue\tGoalActivateDeviceWithName\tactivate heater (stove)
2\tfalse\tGoalActivateDeviceWithName\tactivate heater (oven)
3\tfalse\tGoalActivateDeviceWithName\tactivate heater (hot plate)
"""


# ============================================================ 假环境
class FakeSciWorldEnv:
    """脚本化假环境:activate stove +20 → focus on lead +40 → wait1 → 100 完成;
    focus on oven 即焦死(终分定值 -100,与 Goal.scala:100 实测一致)。"""

    def __init__(self):
        self.score = 0
        self.done = False
        self.task = ""
        self.var = 0

    # ---- 官方 API 子集(envs.py 消费的全部)
    def get_task_names(self):
        return list(OFFICIAL_TASK_NAMES)

    def get_variations_train(self):
        return [0, 1, 2, 3]

    def get_variations_dev(self):
        return [10, 11]

    def load(self, task, variation, simplification, generateGoldPath=False):
        self.task, self.var = task, int(variation)
        self.score, self.done = 0, False

    def reset(self):
        return f"This room is called the kitchen. ({self.task} v{self.var})", {}

    def get_task_description(self):
        return f"Your task is to melt lead. ({self.task})"

    def get_valid_action_object_combinations(self):
        return ["look around", "activate stove", "focus on lead", "focus on oven", "wait1"]

    def get_goal_progress(self):
        return GOAL_PROGRESS_SAMPLE

    def step(self, act):
        assert not self.done, "step after done 应被 envs.py 挡在外面"
        obs = f"You did: {act}."
        if act == "activate stove":
            self.score = 20
        elif act == "focus on lead":
            self.score = 60
        elif act == "wait1" and self.score >= 60:
            self.score = 100
            self.done = True
        elif act == "focus on oven":
            self.score = -100
            self.done = True
            obs = "You focus on the oven."
        elif act.startswith(("look", "wait")):
            pass
        else:
            obs = "No known action matches that input."
        return obs, 0, self.done, {"score": self.score, "reward": 0, "moves": 0}

    def close(self):
        pass


def _env_config():
    return OmegaConf.create({
        "sciworld": {"simplification": "easy", "env_step_limit": 200,
                     "max_valid_actions": 300, "fetch_goal_progress": True},
    })


def _mk_config():
    return OmegaConf.create({
        "env": {
            "env_name": "sciworld_rso",
            "seed": 0,
            "history_length": 2,
            "rollout": {"n": 1},
            "max_steps": 50,
            "rao": {"root_max_steps": 8, "per_agent_max_steps": 4, "max_depth": 2},
            "sciworld": {"simplification": "easy", "env_step_limit": 200,
                         "max_valid_actions": 300, "fetch_goal_progress": True},
        },
        "data": {"train_batch_size": 1, "val_batch_size": 1},
    })


# ============================================================ 单元
def test_projection():
    acts, valids = sciworld_projection([
        "<thought>go.</thought><action>Activate Stove</action>",
        "<action>look around</action>",                      # 缺 thought → invalid
        "no tags at all",
        "<thought>d.</thought><action>delegate: find a pot budget 8</action>",
    ])
    assert acts[0] == "activate stove" and valids[0] == 1    # 小写化 = 官方 sanitize
    assert acts[1] == "look around" and valids[1] == 0
    assert valids[2] == 0
    assert acts[3].startswith("delegate:") and valids[3] == 1


def test_goal_progress_parse_and_render():
    gp = parse_goal_progress(GOAL_PROGRESS_SAMPLE)
    assert [g.done for g in gp.sequential] == [False, True, False]
    assert len(gp.optional) == 4
    plan = render_gt_plan(gp)
    assert "1. focus on substance" in plan
    # 同模板合并:三个 heater 变体合一行
    assert "activate heater (stove/oven/hot plate)" in plan
    priv = render_priv("melt lead", gp, score=20, peak=20)
    assert "Goal: melt lead" in priv
    assert "focus on substance" in priv and "liquid state" in priv
    assert "substance is in a solid state" in priv           # 已达段
    assert "activate heater (oven/hot plate)" in priv        # 提示段只列未完成
    assert "gold" not in priv.lower() or "subgoal" in priv.lower()  # 金动作序列禁入


def test_budget():
    b = RootChildBudget.from_config({"root_max_steps": 60, "per_agent_max_steps": 20,
                                     "max_depth": 2})
    assert b.allocate(0) == 60 and b.allocate(1) == 20 and b.allocate(2) == 20
    assert b.can_delegate(0) and b.can_delegate(1) and not b.can_delegate(2)


def test_envs_semantics():
    envs = build_sciworld_envs(seed=0, env_num=2, group_n=2, is_train=True,
                               env_config=_env_config(), zero_reward=False,
                               env_factory=FakeSciWorldEnv)
    obs, infos = envs.reset()
    assert len(obs) == 4
    # 组一致性:同组两个槽同 (task, variation)
    assert (infos[0]["extra.task"], infos[0]["extra.variation"]) == \
           (infos[1]["extra.task"], infos[1]["extra.variation"])
    assert infos[0]["extra.gt_plan"].startswith("Gold subgoal plan")
    assert infos[0]["extra.goal_progress"]

    o, r, d, i2 = envs.step(["activate stove", "", "answer: found it", "bogus action xyz"])
    assert abs(r[0] - 0.20) < 1e-9 and i2[0]["extra.score"] == 20      # clip 增量/100
    assert i2[1]["extra.last_action_kind"] == "noop" and o[1] == obs[1]
    assert i2[2]["extra.last_action_kind"] == "answer"
    assert i2[3]["extra.env_unmatched"] is True

    o, r, d, i3 = envs.step(["focus on oven", "", "", ""])
    assert d[0] and i3[0]["extra.focus_death"] is True
    assert i3[0]["extra.score"] == -100 and abs(r[0] - (-0.20)) < 1e-9  # clip 20→0
    assert i3[0]["extra.peak"] == 20                                    # 峰值不回退
    o, r, d, i4 = envs.step(["look around", "", "", ""])
    assert d[0] and i4[0]["extra.last_action_kind"] == "post_done"      # 死后不进 env

    # val:固定 50 case,前 30 槽 = 30 任务的 dev[0]
    val = build_sciworld_envs(seed=1, env_num=50, group_n=1, is_train=False,
                              env_config=_env_config(), env_factory=FakeSciWorldEnv)
    vobs, vinfos = val.reset()
    tasks = [i["extra.task"] for i in vinfos]
    assert tasks[:30] == OFFICIAL_TASK_NAMES
    assert all(i["extra.variation"] == 10 for i in vinfos[:30])
    assert tasks[30:] == OFFICIAL_TASK_NAMES[:20]
    assert all(i["extra.variation"] == 11 for i in vinfos[30:])
    envs.close(), val.close()


def test_adapter_units():
    from agent_system.environments.env_package.sciworld.recursive_adapter import (
        SciWorldRecursiveAdapter)
    ad = SciWorldRecursiveAdapter(_mk_config())
    req = ad.parse_delegation("delegate: find a pot or greenhouse budget 8")
    assert req is not None and req.goal["task"] == "find a pot or greenhouse"  # 预算尾巴剥离
    assert ad.parse_delegation("delegate:") is None
    assert ad.parse_delegation("activate stove") is None
    assert ad.filler_action() == ""


def test_orchestrator_end_to_end():
    """委托→子 agent 干活拿分→answer 汇报→父收报告→root 走完:Φ/G、成功判定、reward 补丁。"""
    from agent_system.environments.env_package.sciworld.opsd_factory import (
        SciWorldRecursiveEnvironmentManager)
    from agent_system.environments.env_package.sciworld.opsd_priv import SciWorldOPSDAdapter

    config = _mk_config()
    _envs = build_sciworld_envs(seed=0, env_num=1, group_n=1, is_train=True,
                                env_config=config.env, zero_reward=True,
                                env_factory=FakeSciWorldEnv)
    budget = RootChildBudget.from_config(dict(config.env.rao))
    adapter = SciWorldOPSDAdapter(config)
    mgr = SciWorldRecursiveEnvironmentManager(
        _envs, sciworld_projection, config, adapter=adapter, budget=budget,
        trace_tag="test")
    obs, infos = mgr.reset()
    assert "Task:" in obs["text"][0] and "delegate: <sub-task>" in obs["text"][0]

    def act(s):
        return [f"<thought>t.</thought><action>{s}</action>"]

    # root 委托
    o, r, d, i = mgr.step(act("delegate: activate the stove for me"))
    assert not d[0]
    assert "sub-agent" in o["text"][0].lower() or "sub-task" in o["text"][0].lower()
    # 子 agent 干活(+20 分 → delta_phi 归子行)
    o, r, d, i = mgr.step(act("activate stove"))
    assert float(i[0].get("delta_phi", 0.0)) == 20.0
    # 子 agent 汇报 → 关闭,父收报告
    o, r, d, i = mgr.step(act("answer: the stove is on now"))
    assert "the stove is on now" in o["text"][0]
    assert "Score gained during the sub-task: 20" in o["text"][0]
    # root 亲自 focus(+40)→ wait1 完成(100)
    o, r, d, i = mgr.step(act("focus on lead"))
    assert float(i[0].get("delta_phi", 0.0)) == 40.0
    o, r, d, i = mgr.step(act("wait1"))
    assert d[0] and float(r[0]) == 1.0                       # reward 补丁 = root 官方分
    recs = mgr.finished_records[0]
    roots = [x for x in recs if x.depth == 0]
    subs = [x for x in recs if x.depth > 0]
    assert roots and roots[-1].success == 1.0
    assert subs and subs[-1].success == 1.0 and subs[-1].close_reason == "goal_met"
    # priv 通道开着(OPSD 适配器带 build_priv)
    assert mgr.turn_meta and "node_priv" in mgr.turn_meta[0][0]
    priv0 = mgr.turn_meta[0][0]["node_priv"]
    assert priv0.startswith("[Privileged subgoal plan")
    # G 守恒:全树 delta_phi 之和 = 终局峰值
    total_g = sum(row["delta_phi"] for rows in mgr.turn_meta for row in rows)
    assert total_g == 100.0
    mgr.envs.close()


def test_orchestrator_focus_death():
    """焦死:整树收摊,root success = 0(负分裁零),delta_phi 无负值。"""
    from agent_system.environments.env_package.sciworld.opsd_factory import (
        SciWorldRecursiveEnvironmentManager)
    from agent_system.environments.env_package.sciworld.recursive_adapter import (
        SciWorldRecursiveAdapter)

    config = _mk_config()
    _envs = build_sciworld_envs(seed=0, env_num=1, group_n=1, is_train=True,
                                env_config=config.env, zero_reward=True,
                                env_factory=FakeSciWorldEnv)
    mgr = SciWorldRecursiveEnvironmentManager(
        _envs, sciworld_projection, config, adapter=SciWorldRecursiveAdapter(config),
        budget=RootChildBudget.from_config(dict(config.env.rao)), trace_tag="test")
    mgr.reset()

    def act(s):
        return [f"<thought>t.</thought><action>{s}</action>"]

    mgr.step(act("activate stove"))
    o, r, d, i = mgr.step(act("focus on oven"))
    assert d[0] and float(r[0]) == 0.0
    roots = [x for x in mgr.finished_records[0] if x.depth == 0]
    assert roots and roots[-1].success == 0.0
    for rows in mgr.turn_meta:
        for row in rows:
            assert row["delta_phi"] >= 0.0                   # G≥0 构造性成立
    mgr.envs.close()


_ALL = [test_projection, test_goal_progress_parse_and_render, test_budget,
        test_envs_semantics, test_adapter_units, test_orchestrator_end_to_end,
        test_orchestrator_focus_death]

if __name__ == "__main__":
    failed = 0
    for fn in _ALL:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"FAIL {fn.__name__}: {e}")
            traceback.print_exc()
    print(f"{len(_ALL) - failed}/{len(_ALL)} passed")
    sys.exit(1 if failed else 0)
