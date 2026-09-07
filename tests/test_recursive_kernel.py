# -*- coding: utf-8 -*-
"""递归内核(agent_system/recursive/)的 CPU 单元测试。不依赖 torch/vLLM/Ray/TextCraft。

用一个"计数器世界"假环境和一个最小假适配器驱动状态机,逐条验证与 RAO 官方语义的对应:
  T1  委托压栈:子节点 depth+1、parent_uid、fork_step 正确;父那一步照扣预算(P5 正解)
  T2  子成功弹栈:goal_met;父收到战报覆盖占位文本;父 children_success=[1.0]
  T3  子预算耗尽弹栈:budget_exhausted;success 由 evaluate_node 打分(此处 0)
  T4  深度上限:max_depth 到顶时委托被拒,拒绝文本进父的 history,不压栈,仍扣一步
  T5  级联弹栈:子关闭时父恰好预算耗尽 → 父也关闭;root 预算耗尽 → 整局结束
  T6  整局结束(环境 done)→ 栈内全部节点按 episode_end 归档,root 成败来自 evaluate_node
  T7  全局轮数到 → collect_node_records 按 rollout_end 强制收摊
  T8  上下文隔离:子节点的观测只含自己的历史
  T9  内核异常:适配器抛错 → 该槽按 error 收摊、记录带 traceback、其他槽不受影响
  T10 lockstep 不变量:已结束的槽仍收到 filler、行元数据为空

运行:
    source SDAR_RAO/env_rao.sh && cd "$SDAR_REPO" && python tests/test_recursive_kernel.py
    (或 pytest tests/test_recursive_kernel.py)
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_system.recursive.budget import PerAgentBudget                       # noqa: E402
from agent_system.recursive.node import Node, NodeRecord                       # noqa: E402
from agent_system.recursive.orchestrator import RecursiveEnvironmentManager    # noqa: E402
from agent_system.recursive.protocol import (                                  # noqa: E402
    CLOSE_BUDGET_EXHAUSTED, CLOSE_EPISODE_END, CLOSE_ERROR, CLOSE_GOAL_MET, CLOSE_OVERFLOW,
    CLOSE_ROLLOUT_END, CLOSE_STUCK, DelegationRequest, NodeOutcome,
)


# =============================================================================
# 假环境:每个槽一个 inventory 计数器。动作文法:
#   "make X"   inventory[X] += 1
#   "noop"     什么都不做
#   "inv"      filler,什么都不做
#   "finish"   整局结束(done=True),won = inventory[root目标] >= 需求
# =============================================================================
class FakeEnvs:
    def __init__(self, n: int, root_goal: Dict[str, int]):
        self.n = n
        self.root_goal = dict(root_goal)
        self.inv: List[Dict[str, int]] = []
        self.done: List[bool] = []

    def _info(self, i):
        return {"won": all(self.inv[i].get(k, 0) >= v for k, v in self.root_goal.items()),
                "extra.inventory": dict(self.inv[i]),
                "extra.target_items": dict(self.root_goal),
                "extra.difficulty": "easy"}

    def reset(self):
        self.inv = [dict() for _ in range(self.n)]
        self.done = [False] * self.n
        return [f"Goal: {self.root_goal}" for _ in range(self.n)], [self._info(i) for i in range(self.n)]

    def step(self, actions):
        obs, rew, dones, infos = [], [], [], []
        for i, a in enumerate(actions):
            if self.done[i]:
                obs.append(""); rew.append(0.0); dones.append(True); infos.append(self._info(i)); continue
            if a.startswith("make "):
                item = a.split(" ", 1)[1]
                self.inv[i][item] = self.inv[i].get(item, 0) + 1
                obs.append(f"made 1 {item}")
            elif a == "finish":
                self.done[i] = True
                obs.append("finished")
            else:
                obs.append("ok")
            info = self._info(i)
            won = info["won"] and a == "finish"
            rew.append(1.0 if won else 0.0)
            dones.append(self.done[i])
            infos.append(info)
        return obs, np.array(rew), np.array(dones), infos

    def close(self):
        pass


# =============================================================================
# 假适配器:委托文法 "delegate X N"(可带 "| ctx");节点目标 = {X: N};
# 成功判定 = 库存净增量 >= N(与官方 evaluate 同款);观测 = 目标 + 自己的历史。
# =============================================================================
class FakeAdapter:
    def __init__(self, raise_in_outcome_for_goal: Optional[str] = None,
                 stuck_after: Optional[int] = None, long_obs_for_depth: Optional[int] = None):
        self.raise_for = raise_in_outcome_for_goal
        self.stuck_after = stuck_after                  # 节点走满这么多轮就判"打转"(模拟官方每-agent 循环检测)
        self.long_obs_for_depth = long_obs_for_depth    # 该深度的观测渲染成超长文本(模拟 prompt 超限)

    def root_goal(self, env_info, obs_text):
        return dict(env_info["extra.target_items"])

    def parse_delegation(self, text):
        if not text.startswith("delegate "):
            return None
        body, _, ctx = text[len("delegate "):].partition("|")
        item, n = body.split()
        return DelegationRequest(goal={item: int(n)}, context=ctx.strip(), raw_action=text)

    def filler_action(self):
        return "inv"

    def goal_text(self, goal, is_root=True):
        return "craft " + ", ".join(f"{v} {k}" for k, v in goal.items())

    def child_report_text(self, rec: NodeRecord):
        verdict = "SUCCEEDED" if rec.success >= 1 else f"FAILED ({rec.close_reason})"
        return f"Sub-agent report: '{rec.goal_text}' {verdict}. Steps used {rec.turns}."

    def delegation_refused_text(self, reason, detail):
        return f"Delegation refused ({reason}): {detail}"

    def on_node_open(self, node: Node, env_info):
        node.open_snapshot = dict(env_info["extra.inventory"])

    def on_turn_result(self, node: Node, env_info):
        node.scratch["turns_seen"] = node.scratch.get("turns_seen", 0) + 1

    def _net_ok(self, node: Node, env_info):
        inv = env_info["extra.inventory"]
        return all(inv.get(k, 0) - node.open_snapshot.get(k, 0) >= v for k, v in node.goal.items())

    def node_outcome(self, node: Node, env_info):
        if self.raise_for and self.raise_for in node.goal:
            raise RuntimeError("adapter exploded on purpose")
        if self._net_ok(node, env_info):
            return NodeOutcome(done=True, success=1.0, reason=CLOSE_GOAL_MET)
        if self.stuck_after is not None and node.turns_used >= self.stuck_after:
            return NodeOutcome(done=True, success=self.evaluate_node(node, env_info), reason=CLOSE_STUCK)
        return NodeOutcome(done=False)

    def evaluate_node(self, node: Node, env_info):
        return 1.0 if self._net_ok(node, env_info) else 0.0

    def build_observation(self, node: Node, is_first_turn):
        if self.long_obs_for_depth is not None and node.depth == self.long_obs_for_depth and node.history:
            return "X" * 5000
        hist = " | ".join(f"{t.action}->{t.result}" for t in node.history)
        return f"[d{node.depth}] GOAL={node.goal_text} HIST={hist}"


def make_mgr(n=2, root_goal=None, steps=5, max_depth=6, adapter=None, max_obs_chars=None):
    rao = {"per_agent_max_steps": steps, "max_depth": max_depth}
    if max_obs_chars is not None:
        rao["max_obs_chars"] = max_obs_chars
    cfg = SimpleNamespace(env=SimpleNamespace(max_steps=50, rao=rao, history_length=2))
    # OmegaConf 风格的 .get 支持
    cfg.env.get = lambda k, d=None: getattr(cfg.env, k, d)
    envs = FakeEnvs(n, root_goal or {"axe": 1})
    proj = lambda acts: (list(acts), [1] * len(acts))   # 假投影:原样通过,全部合法
    mgr = RecursiveEnvironmentManager(envs, proj, cfg, adapter=adapter or FakeAdapter())
    return mgr, envs


def run(mgr, actions_per_round: List[List[str]]):
    """逐轮喂动作,返回每轮的 (obs_text, rewards, dones, infos)。"""
    outs = []
    for acts in actions_per_round:
        outs.append(mgr.step(acts))
    return outs


passed = 0
def check(cond, msg):
    global passed
    assert cond, msg
    passed += 1
    print(f"  ✓ {msg}")


# ----------------------------------------------------------------------------- T1
def test_T1_delegation_push():
    print("T1 委托压栈")
    mgr, envs = make_mgr(n=1, steps=5)
    mgr.reset()
    root = mgr.stacks[0][0]
    mgr.step(["noop"])                               # root 第 1 步
    mgr.step(["delegate wood 2 | for the axe"])      # root 第 2 步:委托
    st = mgr.stacks[0]
    check(len(st) == 2 and st[-1].depth == 1, "压栈后栈深 2,栈顶 depth=1")
    child = st[-1]
    check(child.parent_uid == root.uid, "子的 parent_uid 指向 root")
    check(child.fork_step == 1, "fork_step = 父委托前的历史长度 1(trajectory.py:131)")
    check(child.goal == {"wood": 2} and child.budget_left == 5, "子目标与独立预算(25 步语义)正确")
    check(root.budget_left == 3 and root.history[-1].is_delegation, "父委托那步照扣预算(P5 正解)")
    check(root.history[-1].result.startswith("(delegated"), "父那步暂记占位结果")
    check(root.children_uids == [child.uid], "父登记了子代 uid")
    check(envs.inv[0] == {}, "委托那轮底层环境只收到 filler,状态未变")


# ----------------------------------------------------------------------------- T2
def test_T2_child_success_pop():
    print("T2 子成功弹栈 + 父收战报")
    mgr, envs = make_mgr(n=1, steps=5)
    mgr.reset()
    root = mgr.stacks[0][0]
    mgr.step(["delegate wood 2"])
    child = mgr.stacks[0][-1]
    mgr.step(["make wood"])                          # 子第 1 步:净增 1,未达标
    check(len(mgr.stacks[0]) == 2, "净增 1 < 2,子未关闭")
    out = mgr.step(["make wood"])                    # 子第 2 步:净增 2,达标
    check(len(mgr.stacks[0]) == 1, "达标后弹栈,栈只剩 root")
    rec = mgr.finished_records[0][0]
    check(rec.uid == child.uid and rec.close_reason == CLOSE_GOAL_MET and rec.success == 1.0,
          "记录:goal_met / success=1")
    check(rec.turns == 2 and rec.depth == 1, "记录:turns=2, depth=1")
    check(root.children_success == [1.0], "父的 children_success=[1.0](Eq.1 原料)")
    check("SUCCEEDED" in root.history[-1].result, "父委托那步的结果被战报覆盖(subagent.py:98 语义)")
    check(root.last_result == root.history[-1].result, "父 last_result 同步")
    obs = out[0]["text"][0]
    check(obs.startswith("[d0]") and "SUCCEEDED" in obs, "下一轮观测已切回 root 视角且含战报")


# ----------------------------------------------------------------------------- T3
def test_T3_child_budget_exhausted():
    print("T3 子预算耗尽弹栈")
    mgr, envs = make_mgr(n=1, steps=2)               # 每节点只有 2 步
    mgr.reset()
    root = mgr.stacks[0][0]
    mgr.step(["delegate wood 5"])                    # root 用 1 步
    mgr.step(["noop"])                               # 子用 1 步
    check(len(mgr.stacks[0]) == 2, "子剩 1 步,仍在栈上")
    mgr.step(["noop"])                               # 子用完 2 步
    rec = mgr.finished_records[0][0]
    check(rec.close_reason == CLOSE_BUDGET_EXHAUSTED and rec.success == 0.0,
          "子按 budget_exhausted 关闭,evaluate_node 打 0 分")
    check(root.children_success == [0.0] and "FAILED" in root.history[-1].result,
          "父记 0 并收到失败战报")


# ----------------------------------------------------------------------------- T4
def test_T4_depth_limit_refusal():
    print("T4 深度上限拒绝委托")
    mgr, envs = make_mgr(n=1, steps=5, max_depth=1)  # 只允许一层子代
    mgr.reset()
    mgr.step(["delegate wood 1"])                    # root(d0) 派子 → 允许(0+1<=1)
    child = mgr.stacks[0][-1]
    mgr.step(["delegate stone 1"])                   # 子(d1) 再派 → 拒绝(1+1>1)
    check(len(mgr.stacks[0]) == 2, "被拒后不压栈")
    check(child.history[-1].is_delegation and "refused" in child.history[-1].result
          and "perform the task yourself" in child.history[-1].result,
          "拒绝文本(含官方 guidance 原文)进子的 history")
    check(child.budget_left == 4 and child.children_uids == [], "被拒仍扣一步,且无子代登记")


# ----------------------------------------------------------------------------- T5
def test_T5_cascade_and_root_exhaustion():
    print("T5 级联弹栈 + root 预算耗尽结束整局")
    mgr, envs = make_mgr(n=1, steps=1)               # 每节点仅 1 步:root 委托即用尽
    mgr.reset()
    root = mgr.stacks[0][0]
    out = mgr.step(["delegate wood 1"])              # root 用掉唯一一步;子压栈(子还有 1 步)
    check(len(mgr.stacks[0]) == 2 and root.budget_exhausted, "root 预算已尽但子在跑,整局不结束")
    out = mgr.step(["make wood"])                    # 子达标 → 弹栈 → root 也预算尽 → 整局结束
    check(mgr.stacks[0] == [] and mgr.episode_done[0], "级联:子关→root 关→整局结束")
    check(bool(out[2][0]), "dones 对采集循环置 True")
    reasons = {r.depth: r.close_reason for r in mgr.finished_records[0]}
    check(reasons == {1: CLOSE_GOAL_MET, 0: CLOSE_BUDGET_EXHAUSTED}, f"关闭原因 {reasons}")


# ----------------------------------------------------------------------------- T6
def test_T6_episode_end_closes_all():
    print("T6 整局结束归档全树")
    mgr, envs = make_mgr(n=1, steps=10, root_goal={"axe": 1})
    mgr.reset()
    mgr.step(["delegate wood 1"])                    # d1
    mgr.step(["delegate stone 1"])                   # d2(子再派孙)
    check(len(mgr.stacks[0]) == 3, "三层栈")
    mgr.step(["make axe"])                           # 孙做出 root 的目标物(但孙自己目标是 stone)
    mgr.step(["finish"])                             # 孙调 finish → 环境 done
    check(mgr.stacks[0] == [] and mgr.episode_done[0], "整局结束,栈清空")
    recs = {r.depth: r for r in mgr.finished_records[0]}
    check(all(r.close_reason == CLOSE_EPISODE_END for r in recs.values()), "全部 episode_end")
    check(recs[0].success == 1.0, "root 由 evaluate_node 判成功(axe 净增 1)")
    check(recs[2].success == 0.0 and recs[1].success == 0.0, "孙/子各自目标未达成 → 0")
    check(recs[0].children_success == [0.0] and recs[1].children_success == [0.0],
          "父层 children_success 在收摊时也被回填")


# ----------------------------------------------------------------------------- T7
def test_T7_rollout_end_force_close():
    print("T7 全局轮数到,collect_node_records 强制收摊")
    mgr, envs = make_mgr(n=1, steps=10)
    mgr.reset()
    mgr.step(["delegate wood 3"])
    mgr.step(["make wood"])
    recs = mgr.collect_node_records()[0]
    check(len(recs) == 2 and all(r.close_reason == CLOSE_ROLLOUT_END for r in recs), "两节点 rollout_end")
    check({r.depth: r.success for r in recs} == {1: 0.0, 0: 0.0}, "按 evaluate_node 打分,不是一律判负")
    check(mgr.stacks[0] == [] and mgr.episode_done[0], "收摊后栈空、槽标记结束")


# ----------------------------------------------------------------------------- T8
def test_T8_context_isolation():
    print("T8 上下文隔离")
    mgr, envs = make_mgr(n=1, steps=10)
    mgr.reset()
    mgr.step(["noop"]); mgr.step(["noop"])           # root 两步历史
    out = mgr.step(["delegate wood 1"])
    obs = out[0]["text"][0]
    check(obs.startswith("[d1]") and "HIST=" in obs and obs.endswith("HIST="),
          "子的首轮观测只有自己的(空)历史,看不到 root 的两步")
    out = mgr.step(["noop"])
    obs = out[0]["text"][0]
    check("noop->ok" in obs and obs.count("noop->ok") == 1, "子第二轮只见自己的 1 步,不见 root 的 2 步")


# ----------------------------------------------------------------------------- T9
def test_T9_error_isolation():
    print("T9 内核异常留档且不炸 batch")
    mgr, envs = make_mgr(n=2, steps=10, adapter=FakeAdapter(raise_in_outcome_for_goal="bomb"))
    mgr.reset()
    root0 = mgr.stacks[0][0]
    out = mgr.step(["delegate bomb 1", "noop"])      # 槽 0 的子目标含 bomb,子的判定时会炸
    # 2026-08-25 生命周期下沉:异常只结束出错的那个 agent(loop.py:79-97),父继续
    check(not out[2][0] and not mgr.episode_done[0], "子出错不结束整局")
    recs = mgr.finished_records[0]
    child_rec = [r for r in recs if r.close_reason == CLOSE_ERROR]
    check(len(child_rec) == 1 and child_rec[0].depth == 1, "只有出错的子节点按 error 关闭")
    check("adapter exploded on purpose" in child_rec[0].error and "Traceback" in child_rec[0].error,
          "记录含 traceback 文本(P11)")
    check(len(mgr.stacks[0]) == 1 and mgr.stacks[0][0] is root0 and root0.error is None,
          "root 仍在栈上、未被标记出错")
    check("FAILED" in root0.history[-1].result and root0.children_success == [0.0],
          "父收到失败战报,children_success=[0]")
    check(not mgr.episode_done[1] and len(mgr.stacks[1]) == 1, "槽 1 完全不受影响")
    out = mgr.step(["noop", "noop"])
    check(out[3][0]["node_uid"] == root0.uid and out[3][1]["node_uid"] != "", "两槽下一轮都由各自 root 行动")
    # 已结束槽的行元数据为空:让槽 1 的 root 预算耗尽后再看
    mgr2, _ = make_mgr(n=1, steps=1)
    mgr2.reset(); mgr2.step(["noop"])
    out = mgr2.step(["noop"])
    check(mgr2.episode_done[0] and out[3][0]["node_uid"] == "", "已结束槽的行元数据为空(T10 部分)")


# ----------------------------------------------------------------------------- T11
def test_T11_stuck_per_node():
    print("T11 循环检测按节点:子打转只关子,root 打转结束整局(官方 agent.py:90-118 语义)")
    mgr, envs = make_mgr(n=1, steps=10, adapter=FakeAdapter(stuck_after=3))
    mgr.reset()
    root = mgr.stacks[0][0]
    mgr.step(["delegate wood 5"])                    # root 1 步;子压栈
    child = mgr.stacks[0][-1]
    mgr.step(["noop"]); mgr.step(["noop"])           # 子 2 步
    check(len(mgr.stacks[0]) == 2, "子 2 步未触发")
    out = mgr.step(["noop"])                         # 子第 3 步 → 判打转
    check(len(mgr.stacks[0]) == 1 and not mgr.episode_done[0], "子按 stuck 关闭,整局未结束")
    rec = [r for r in mgr.finished_records[0] if r.uid == child.uid][0]
    check(rec.close_reason == CLOSE_STUCK and rec.success == 0.0, "子记录 stuck_in_loop / 0 分")
    check("FAILED" in root.history[-1].result and out[0]["text"][0].startswith("[d0]"), "父收到失败战报并继续行动")
    mgr.step(["noop"]); out = mgr.step(["noop"])     # root 累计 3 步 → 自己打转
    check(mgr.episode_done[0] and bool(out[2][0]), "root 打转 → 整局结束")
    reasons = {r.depth: r.close_reason for r in mgr.finished_records[0]}
    check(reasons[0] == CLOSE_STUCK, f"root 关闭原因 {reasons[0]}")


# ----------------------------------------------------------------------------- T12
def test_T12_overflow_per_node():
    print("T12 prompt 超长按节点:子超长只关子,root 超长结束整局")
    mgr, envs = make_mgr(n=1, steps=10, adapter=FakeAdapter(long_obs_for_depth=1), max_obs_chars=1000)
    mgr.reset()
    root = mgr.stacks[0][0]
    mgr.step(["delegate wood 5"])                    # 子压栈,子首轮观测正常(history 为空不触发)
    check(len(mgr.stacks[0]) == 2, "子在栈上")
    out = mgr.step(["noop"])                         # 子行动一次后,它的观测超长
    child_recs = [r for r in mgr.finished_records[0] if r.depth == 1]
    check(len(child_recs) == 1 and child_recs[0].close_reason == CLOSE_OVERFLOW, "子按 prompt_overflow 关闭")
    check(len(mgr.stacks[0]) == 1 and not mgr.episode_done[0] and out[0]["text"][0].startswith("[d0]"),
          "root 继续,观测重新渲染为 root 视角")
    mgr2, _ = make_mgr(n=1, steps=10, adapter=FakeAdapter(long_obs_for_depth=0), max_obs_chars=1000)
    mgr2.reset(); out = mgr2.step(["noop"])
    check(mgr2.episode_done[0] and bool(out[2][0]) and mgr2.finished_records[0][0].close_reason == CLOSE_OVERFLOW,
          "root 超长 → 整局结束,原因 prompt_overflow")


# ----------------------------------------------------------------------------- T10
def test_T10_metadata_and_records():
    print("T10 行元数据与记录结构")
    mgr, envs = make_mgr(n=1, steps=10)
    _, infos = mgr.reset()
    root_uid = mgr.stacks[0][0].uid
    check(infos[0]["node_uid"] == root_uid and infos[0]["node_depth"] == 0
          and infos[0]["root_node_uid"] == root_uid, "reset 的 info 带 root 元数据")
    out = mgr.step(["delegate wood 1"])
    info = out[3][0]
    check(info["is_delegation_turn"] is True and info["node_uid"] == root_uid, "委托那轮元数据归父")
    out = mgr.step(["noop"])
    info = out[3][0]
    child_uid = mgr.stacks[0][-1].uid
    check(info["node_uid"] == child_uid and info["node_depth"] == 1
          and info["root_node_uid"] == root_uid, "子出招那轮元数据归子,root_node_uid 仍指 root")
    rec = mgr.collect_node_records()[0][0]
    check(isinstance(rec, NodeRecord) and set(vars(rec)) >= {
        "uid", "parent_uid", "depth", "success", "children_uids", "children_success"},
          "NodeRecord 字段齐全(Eq.1/3/4 所需)")


if __name__ == "__main__":
    for fn in [test_T1_delegation_push, test_T2_child_success_pop, test_T3_child_budget_exhausted,
               test_T4_depth_limit_refusal, test_T5_cascade_and_root_exhaustion,
               test_T6_episode_end_closes_all, test_T7_rollout_end_force_close,
               test_T8_context_isolation, test_T9_error_isolation, test_T10_metadata_and_records,
               test_T11_stuck_per_node, test_T12_overflow_per_node]:
        fn()
    print(f"\n=== 递归内核单元测试全部通过({passed} 项断言)===")
