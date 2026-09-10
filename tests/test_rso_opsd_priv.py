# -*- coding: utf-8 -*-
"""RSO+OPSD priv 通道的 CPU 单元测试:渲染 / 适配器 / 编排器"行开局现算"判序。

守的三件事(设计出处 RSO_method_design §2a + docs/RSO_OPSD_DESIGN.md §三/§四):
  一、render_priv 四段渲染正确(工作清单数量/已知待查标注/子树规模;已满足与不可解
      两种边界各有显式措辞);K_X 只取节点私有笔记本;
  二、useless_goal 标记:子 goal 有物品不在 root 剩余闭包内 → True(指标原料);
  三、编排器判序——这是 2026-09-10 从"滞后一拍缓存"改成"行开局现算"要修的那一行:
      委托间隙后父节点首行的 priv 必须反映【子树运行后】的库存;
      同时,不带 build_priv 的适配器(阶段 1 RSO)turn_meta 一个键都不多(非干扰底线)。

运行(不需要 GPU):PYTHONPATH=$PWD python tests/test_rso_opsd_priv.py
"""
from __future__ import annotations

import os
import sys
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_system.environments.env_package.textcraft_synth.opsd_priv import (   # noqa: E402
    TextCraftSynthOPSDAdapter, needed_closure_counts, render_priv, static_subtree_size)
from agent_system.environments.env_package.textcraft_synth.phi import (         # noqa: E402
    PhiTracker, needed_closure)
from agent_system.recursive.node import Node                                    # noqa: E402

# 复用内核测试的计数器世界(假环境 / 假适配器 / 管理器装配)
from tests.test_recursive_kernel import FakeAdapter, make_mgr                   # noqa: E402

passed = 0


def check(cond, msg):
    global passed
    assert cond, f"✗ {msg}"
    passed += 1
    print(f"  ✓ {msg}")


# ---------------------------------------------------------------------------
# 假配方库:axe = 2 stick + 1 rock;stick(每批 2)= 1 wood;wood/rock 是叶子。
# ---------------------------------------------------------------------------
class _R:
    def __init__(self, result_count, ingredients):
        self.result_count = result_count
        self.ingredients = ingredients


class FakeDB:
    def __init__(self, recipes: Dict[str, _R]):
        self.recipes = recipes

    def get_recipes_for_item(self, item):
        r = self.recipes.get(item)
        return [r] if r is not None else []


DB = FakeDB({
    "axe": _R(1, {"stick": 2, "rock": 1}),
    "stick": _R(2, {"wood": 1}),
})


# ---------------------------------------------------------------------------- P1
def test_P1_closure_counts_match_phi():
    print("P1 needed_closure_counts 与 phi.needed_closure 同源(集合一致 + 数量正确)")
    inv = {"wood": 5, "rock": 5}                 # 叶子材料在库存里(真实任务的数据性质)
    counts = needed_closure_counts(DB, {"axe": 1}, inv)
    base = needed_closure(DB, {"axe": 1}, inv)
    check(set(counts) == base == {"axe", "stick"}, "闭包集合与 phi 版逐项一致")
    check(counts["axe"] == 1 and counts["stick"] == 2, "产量按 result_count 向上取整(2 根 stick 一批)")
    check(needed_closure_counts(DB, {"axe": 1}, {"stick": 2, "rock": 1}) == {"axe": 1},
          "库存抵扣后 stick 出清单")
    check(needed_closure_counts(DB, {"gold": 1}, {}) is None, "叶子缺口(无配方无库存)→ None 不可解")


# ---------------------------------------------------------------------------- P2
def test_P2_render_worklist_known_unknown():
    print("P2 四段渲染:数量 / 已知与待查标注 / 子树规模 / K_X 完整原文")
    known = {"stick": "craft 2 stick using 1 wood   (depth 1)"}
    text = render_priv(DB, {"axe": 1}, {"wood": 5, "rock": 5}, known)
    check("Goal: 1x axe" in text, "Goal 段")
    check("1x axe  (recipe unknown — look it up first; subtree size 2)" in text,
          "axe:待查标注 + 子树规模 2(axe+stick)")
    check("2x stick  (recipe known; subtree size 1)" in text, "stick:已知标注 + 数量 2 + 规模 1")
    check("craft 2 stick using 1 wood" in text, "已知配方给【完整内容】")
    check("Recipes still unknown" in text and "axe" in text.split("Recipes still unknown")[1],
          "待查段只点名 axe")
    check(static_subtree_size(DB, "wood") == 0, "叶子的子树规模 0")

    text_empty_kx = render_priv(DB, {"axe": 1}, {"wood": 5, "rock": 5}, {})
    check("Known recipes" not in text_empty_kx, "K_X 为空 → 没有已知配方段(未挣到的内容不进 priv)")
    check("axe, stick" in text_empty_kx.split("Recipes still unknown")[1], "两项全在待查名单")


# ---------------------------------------------------------------------------- P3
def test_P3_satisfied_and_infeasible():
    print("P3 边界:已满足 → empty 措辞(第 6 行判据);不可解 → unavailable 措辞")
    t1 = render_priv(DB, {"axe": 1}, {"axe": 1}, {})
    check("empty — the goal is already satisfied" in t1, "已满足:显式 empty(teacher 沉默的判据)")
    check("Work list (what still" not in t1, "已满足时没有清单条目")
    t2 = render_priv(DB, {"gold": 1}, {}, {})
    check("unavailable — this goal cannot be completed" in t2, "不可解:显式 unavailable(Φ 冻结对偶)")


# ---------------------------------------------------------------------------- P4
def test_P4_adapter_build_priv_uses_private_kx():
    print("P4 适配器 build_priv:读当时库存 + 节点【私有】K_X")
    ad = TextCraftSynthOPSDAdapter(config=None)
    ad._db_cache = DB                                     # 不碰真实配方库
    node = Node(depth=1, goal={"axe": 1}, goal_text="craft 1 axe", budget=5)
    node.scratch["known_recipes"] = {"stick": "craft 2 stick using 1 wood   (depth 1)"}
    info = {"extra.inventory": {"stick": 2, "rock": 1}}
    text = ad.build_priv(node, info)
    check("1x axe" in text and "2x stick" not in text, "库存里的 2 根 stick 已抵扣出清单")
    check("craft 2 stick using 1 wood" in text, "私有笔记本里的配方进已知段(静态子 DAG 内)")
    node2 = Node(depth=1, goal={"axe": 1}, goal_text="craft 1 axe", budget=5)
    text2 = ad.build_priv(node2, info)                    # 同一棵树上另一个节点,笔记本为空
    check("craft 2 stick using 1 wood" not in text2, "K_X 按节点隔离:别的节点挣的配方不进本节点 priv")


# ---------------------------------------------------------------------------- P5
def test_P5_useless_goal_flag():
    print("P5 useless_goal:goal 有物品不在 root 剩余闭包内 → True(rso/useless_goal_rate 原料)")
    ad = TextCraftSynthOPSDAdapter(config=None)
    ad._db_cache = DB
    tracker = PhiTracker(DB)
    tracker.reset({"axe": 1}, {"wood": 5, "rock": 5})      # root 闭包 = {axe, stick}
    root_uid = "root-1"
    ad._phi_trackers[root_uid] = tracker
    info = {"root_node_uid": root_uid, "extra.inventory": {"wood": 5, "rock": 5}}

    useful = Node(depth=1, goal={"stick": 2}, goal_text="", budget=5)
    ad.on_node_open(useful, info)
    check(useful.scratch.get("useless_goal") is False, "stick 在闭包内 → False")

    useless = Node(depth=1, goal={"wood": 3}, goal_text="", budget=5)
    ad.on_node_open(useless, info)
    check(useless.scratch.get("useless_goal") is True, "wood 是叶子、不在闭包 → True")

    rec = useless.to_record()
    check(rec.useless_goal is True, "标记随 to_record 导出(NodeRecord.useless_goal)")
    plain = Node(depth=1, goal={"stick": 1}, goal_text="", budget=5)
    check(plain.to_record().useless_goal is False, "不写 scratch 时记录缺省 False(阶段 1 兼容)")


# ---------------------------------------------------------------------------- P6
class PrivFakeAdapter(FakeAdapter):
    """计数器世界的 build_priv:把"当时库存"渲染进文本,便于逐字核对判序。"""

    def build_priv(self, node, env_info):
        inv = env_info.get("extra.inventory") or {}
        inv_s = ",".join(f"{k}:{v}" for k, v in sorted(inv.items())) or "-"
        return f"PRIV[{node.goal_text}|inv={inv_s}]"


def test_P6_orchestrator_row_start_priv():
    print("P6 编排器判序:行开局现算——委托间隙后父首行的 priv 反映子树后的库存")
    mgr, envs = make_mgr(n=1, steps=5, adapter=PrivFakeAdapter())
    mgr.reset()
    check(mgr._priv_enabled, "带 build_priv 的适配器 → priv 通道开")
    mgr.step(["delegate wood 2"])            # 轮 1:父发委托(行的 priv = 开局现算,inv 空)
    mgr.step(["make wood"])                  # 轮 2:子第 1 步
    mgr.step(["make wood"])                  # 轮 3:子第 2 步 → 达标弹栈,父恢复
    mgr.step(["noop"])                       # 轮 4:父回归后的第一行
    tm = mgr.turn_meta
    check(tm[0][0]["node_priv"] == "PRIV[craft 1 axe|inv=-]",
          "轮 1(委托行)的 priv = 委托【前】快照(委托动作本身由它监督——用户 2026-09-10 的论点)")
    check("inv=-" not in tm[1][0]["node_priv"] or tm[1][0]["node_priv"].startswith("PRIV[craft 2 wood"),
          "轮 2 是子节点自己的行,priv 换成子的 goal")
    check(tm[3][0]["node_priv"] == "PRIV[craft 1 axe|inv=wood:2]",
          "轮 4(父回归首行)的 priv 含 wood:2——子树的产出已生效,陈旧漏洞关闭")
    check(tm[3][0]["task_difficulty"] == "easy", "task_difficulty 随行落表(截断分桶的原料)")


# ---------------------------------------------------------------------------- P7
def test_P7_stage1_adapter_untouched():
    print("P7 非干扰底线:阶段 1 适配器(无 build_priv)的 turn_meta 一个键都不多")
    mgr, envs = make_mgr(n=1, steps=5, adapter=FakeAdapter())
    mgr.reset()
    check(not mgr._priv_enabled, "无 build_priv → priv 通道关")
    mgr.step(["noop"])
    mgr.step(["delegate wood 2"])
    for r, row_metas in enumerate(mgr.turn_meta):
        for m in row_metas:
            check2 = ("node_priv" not in m) and ("task_difficulty" not in m)
            assert check2, f"✗ 轮 {r} 的 turn_meta 混入了 OPSD 键: {sorted(m)}"
    check(True, "全部 turn_meta 行都只有阶段 1 的 6 个键")
    keys = sorted(mgr.turn_meta[0][0].keys())
    check(keys == ["delta_phi", "is_delegation_turn", "node_depth", "node_uid",
                   "phi_frozen", "root_node_uid"], f"键集合逐字符合阶段 1 契约: {keys}")


if __name__ == "__main__":
    for fn in [test_P1_closure_counts_match_phi,
               test_P2_render_worklist_known_unknown,
               test_P3_satisfied_and_infeasible,
               test_P4_adapter_build_priv_uses_private_kx,
               test_P5_useless_goal_flag,
               test_P6_orchestrator_row_start_priv,
               test_P7_stage1_adapter_untouched]:
        fn()
    print(f"\n全部通过: {passed} 条断言")
