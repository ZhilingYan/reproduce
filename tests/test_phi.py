# -*- coding: utf-8 -*-
"""Φ 势函数的测试。对应设计文档 RSO_advantage_design.md 第三节。

三层:①632/632 复验(与 probe_costtogo 同口径,证明移植没走样);
     ②单调性与冻结口径的脚本化推演;③K(全树配方并集)的记账。
运行:PYTHONPATH=$PWD python tests/test_phi.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_system.environments.env_package.textcraft_synth.synth_core import get_shared_recipe_db  # noqa: E402
from agent_system.environments.env_package.textcraft_synth.phi import (  # noqa: E402
    PhiTracker, assert_initial_inventory_all_leaves, compute_phi, needed_closure)

passed = 0


def check(cond, msg):
    global passed
    assert cond, f"✗ {msg}"
    passed += 1
    print(f"  ✓ {msg}")


DB = get_shared_recipe_db()
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "agent_system/environments/env_package/textcraft_synth/data/textcraft_synth_val.jsonl")


def test_632_replication():
    """C 与 gold 步数逐题相等 —— 复现 probe_costtogo 的 632/632。"""
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    eq = 0
    for t in rows:
        m = t["misc"]
        closure = needed_closure(DB, m["target_items"], m["initial_inventory"])
        assert closure is not None, t["id"]
        if len(closure) == len(m["gold_trajectory"]):
            eq += 1
        assert_initial_inventory_all_leaves(DB, m["initial_inventory"])
    check(eq == len(rows) == 632, f"C 与 gold 步数逐题相等:{eq}/{len(rows)}(初始库存全叶子断言同过)")


def test_phi_initial_is_2c():
    """K 为空时 Φ = 2C(每个待造物品各摊一次查询、一次制造)。"""
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")][:50]
    for t in rows:
        m = t["misc"]
        c = len(needed_closure(DB, m["target_items"], m["initial_inventory"]))
        phi = compute_phi(DB, m["target_items"], m["initial_inventory"], set(), gamma=1.0)
        assert phi == 2 * c, t["id"]
    check(True, "前 50 题上 Φ(s0, ∅) = 2C 成立")


def _script_task():
    """挑一道浅任务,手工推演动作序列。返回 (task, tracker)。"""
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    # 找一道 C=2 的题:目标 1 个物品,其配方吃 1 种可 craft 的中间物
    for t in rows:
        m = t["misc"]
        cl = needed_closure(DB, m["target_items"], m["initial_inventory"])
        if len(cl) == 2 and len(m["target_items"]) == 1:
            return t
    raise RuntimeError("没找到 C=2 的题")


def test_monotone_and_get_info_credit():
    """查一条需要的配方 → Φ 减 1;查不需要的/重复查 → Φ 不变;craft 需要的 → 减 1。"""
    t = _script_task()
    m = t["misc"]
    tr = PhiTracker(DB)
    phi0 = tr.reset(m["target_items"], m["initial_inventory"])
    cl = sorted(needed_closure(DB, m["target_items"], m["initial_inventory"]))
    target = list(m["target_items"])[0]
    mid = [x for x in cl if x != target][0]

    # 查一条需要的配方(模拟 get_info 返回)
    fake_info = [{"item": target, "recipes": [{"ingredients": {}, "result_count": 1}]}]
    d, phi, fr = tr.update(m["initial_inventory"], fake_info)
    check(d == 1.0 and phi == phi0 - 1 and not fr, "查到一条需要且未知的配方:ΔΦ=1")

    # 重复查同一条:0
    d, phi, _ = tr.update(m["initial_inventory"], fake_info)
    check(d == 0.0, "重复查已知配方:ΔΦ=0")

    # 查一条【不需要】的配方:0(拿一个不在闭包里的可 craft 物品)
    other = next(it for it in DB.get_all_items() if DB.get_recipes_for_item(it) and it not in cl) \
        if hasattr(DB, "get_all_items") else None
    if other:
        d, _, _ = tr.update(m["initial_inventory"], [{"item": other, "recipes": [{"x": 1}]}])
        check(d == 0.0, "查不需要的配方:ΔΦ=0")

    # craft 中间物到位(直接构造 craft 后的库存:中间物数量拉满)
    inv2 = dict(m["initial_inventory"]); inv2[mid] = inv2.get(mid, 0) + 999
    d, phi2, _ = tr.update(inv2, None)
    check(d >= 1.0, f"造出需要的中间物:ΔΦ={d} ≥ 1(闭包缩小)")
    check(phi2 < phi0, "Φ 相对初值净下降")


def test_freeze_on_infeasible():
    """把叶子材料掏空 → 不可解 → 冻结:Φ 停住,之后 ΔΦ 恒 0。"""
    t = _script_task()
    m = t["misc"]
    tr = PhiTracker(DB)
    tr.reset(m["target_items"], m["initial_inventory"])
    phi_before = tr.phi
    d, phi, fr = tr.update({}, None)          # 库存清空:叶子缺口补不回
    check(fr and d == 0.0 and phi == phi_before, "不可解:frozen=True,Φ 冻结,ΔΦ=0")
    d, _, fr = tr.update(dict(m["initial_inventory"]), None)
    check(fr and d == 0.0, "冻结不可逆:即使库存恢复,ΔΦ 仍为 0(吸收态口径)")


def test_conservation_along_gold():
    """沿 gold 轨迹逐步执行:①每步 ΔΦ 非负;②总和 = Φ(s0)−Φ(sT) = 2C(守恒律)。"""
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    done = 0
    for t in rows[:30]:
        m = t["misc"]
        tr = PhiTracker(DB)
        phi0 = tr.reset(m["target_items"], m["initial_inventory"])
        inv = dict(m["initial_inventory"])
        total = 0.0
        for step in m["gold_trajectory"]:
            item, cnt = step["target"][0], step["target"][1]
            # 先"查配方"(gold 不含查询;这里显式补上,验证查询学分)
            d1, _, _ = tr.update(inv, [{"item": item, "recipes": [{"r": 1}]}])
            # 再执行 craft:按 gold 的原料消耗与产出更新库存
            for ing, q in step["ingredients"].items():
                inv[ing] = inv.get(ing, 0) - q
                assert inv[ing] >= 0, (t["id"], ing)
            inv[item] = inv.get(item, 0) + step.get("result_count", cnt)
            d2, _, fr = tr.update(inv, None)
            assert not fr, t["id"]
            assert d1 >= 0 and d2 >= 0, (t["id"], d1, d2)
            total += d1 + d2
        assert abs(total - phi0) < 1e-9, (t["id"], total, phi0)
        assert tr.phi == 0.0, (t["id"], tr.phi)
        done += 1
    check(done == 30, "30 道题沿 gold 逐步推演:ΔΦ 全非负、ΣΔΦ = Φ(s0)、终点 Φ=0")


def _find_A_with_nonleaf_ingredient():
    """找一个配方含非叶子配料 B 的物品 A(与 probe_phi_wash.py 同款构造)。"""
    for it in list(DB._db.recipes.keys()):
        r = DB.get_recipes_for_item(it)[0]
        nonleaf = [g for g in r.ingredients if DB.get_recipes_for_item(g)]
        if nonleaf:
            return it, nonleaf[0], r
    raise RuntimeError("没找到")


def _wash_scene():
    """probe_phi_wash.py 的场景:目标 {A:1批, B:2},库存 = 叶子 + 恰好 (pb+2) 个 B。"""
    A, B, rA = _find_A_with_nonleaf_ingredient()
    rB = DB.get_recipes_for_item(B)[0]
    leaves = {}
    for r in (rA, rB):
        for g in r.ingredients:
            if not DB.get_recipes_for_item(g):
                leaves[g] = 999
    pb = rA.ingredients[B]
    targets = {A: rA.result_count, B: 2}
    inv0 = dict(leaves); inv0[B] = pb + 2
    return A, B, rA, targets, inv0


def _craft(inv, item, batches):
    r = DB.get_recipes_for_item(item)[0]
    iv = dict(inv)
    for g, per in r.ingredients.items():
        iv[g] = iv.get(g, 0) - per * batches
    iv[item] = iv.get(item, 0) + r.result_count * batches
    return iv


def test_negative_delta_is_real(*, _init=None):
    """甲案:挥霍被需要的自造中间品 → ΔΦ 如实为负;挥霍-重造合计净零;
    单轮超量 = wash 记 0;两条路径同记零(套利关闭)。注意本场景初始库存含非叶子 B,
    是刻意构造(跳过全叶子断言,直接操作 tracker 内部口径)。"""
    A, B, rA, targets, inv0 = _wash_scene()
    tr = PhiTracker(DB)
    # 绕过 reset 的全叶子断言,手动装配(测试的是 update 的负值口径,不是断言)
    tr.targets = dict(targets); tr.known = set(); tr.frozen = False
    tr.negative_delta_rounds = 0
    tr.phi = compute_phi(DB, targets, inv0, set())
    tr.phi_initial = tr.phi
    phi0 = tr.phi

    # 路径一:单轮超量(2 批 A,多吃的 B 令直接需求出缺口)→ wash,ΔΦ=0
    inv_over = _craft(inv0, A, 2)
    d, _, fr = tr.update(inv_over, None)
    check(d == 0.0 and not fr, f"单轮超量生产:wash,ΔΦ=0(实际 {d})")

    # 路径二:拆两轮。重置状态,先"挥霍"(直接构造 B 缺口:拿走 pb 个 B 换成无用物)
    tr.phi = phi0; tr.negative_delta_rounds = 0
    inv_waste = dict(inv0); inv_waste[B] = 2 - rA.ingredients[B] if False else inv0[B] - rA.ingredients[B]
    d1, _, _ = tr.update(inv_waste, None)          # B 只剩 2 但仍需造 A 用 pb 个 → B 入闭包
    check(d1 < 0, f"挥霍轮:ΔΦ={d1} < 0(净破坏如实计负)")
    check(tr.negative_delta_rounds == 1, "负值轮计数 = 1(打点原料)")
    inv_rebuild = dict(inv_waste); inv_rebuild[B] = inv0[B]
    d2, _, _ = tr.update(inv_rebuild, None)        # 重造 B
    check(abs(d1 + d2) < 1e-9, f"挥霍({d1}) + 重造({d2})合计净零——与单轮 wash 同记 0")


def test_consume_ingredient_for_needed_target_not_wash():
    """probe_phi_wash 补充演示:正常链条里"消耗 A 造 B"是干净的 +1,不 wash
    (闭包是净额账:B 的缺口与 B 对 A 的需求同时注销)。"""
    A, B, rA = _find_A_with_nonleaf_ingredient()
    rB = DB.get_recipes_for_item(B)[0]
    leaves = {}
    for r in (rA, rB):
        for g in r.ingredients:
            if not DB.get_recipes_for_item(g):
                leaves[g] = 999
    targets = {A: rA.result_count}
    inv0 = dict(leaves)                              # 只有叶子:B 要自己造
    tr = PhiTracker(DB)
    phi0 = tr.reset(targets, inv0)                   # 全叶子,断言可过
    check(phi0 == 4.0, f"闭包 {{A,B}},Φ(s0)=2C=4(实际 {phi0})")
    inv1 = _craft(inv0, B, (rA.ingredients[B] + rB.result_count - 1) // rB.result_count)
    d1, _, _ = tr.update(inv1, None)
    check(d1 == 2.0, f"造中间品 B:ΔΦ=+2(实际 {d1})")
    inv2 = _craft(inv1, A, 1)
    d2, _, fr = tr.update(inv2, None)
    check(d2 == 2.0 and not fr, f"消耗 B 造 A:ΔΦ=+2,不因消耗被抵扣(实际 {d2})")
    check(tr.phi == 0.0 and tr.negative_delta_rounds == 0, "终点 Φ=0,全程无负值轮")


if __name__ == "__main__":
    for fn in [test_632_replication, test_phi_initial_is_2c,
               test_monotone_and_get_info_credit, test_freeze_on_infeasible,
               test_conservation_along_gold, test_negative_delta_is_real,
               test_consume_ingredient_for_needed_target_not_wash]:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n=== Φ 测试全部通过({passed} 项断言)===")
