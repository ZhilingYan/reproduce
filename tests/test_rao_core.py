# -*- coding: utf-8 -*-
"""rao_core.py 的数值测试。手算一组小例子,逐项对拍 Eq.1 / Eq.3 / Eq.4 与官方口径。

  M1  Eq.1:λ=0 时 R = success;λ=0.4 时加子代成功【率】;无子代不加
  M2  Eq.3:LOO 用各树 root 的奖励;树内所有行(含子节点)共用同一基线
  M3  Eq.3:单树组 → 基线 = 含自己的组均值 → 优势 0(P7 修复,对应 step_wise.py:130)
  M4  Eq.4:分母 = 各深度节点数;Σ 行数_d × w_d = 总行数(rl.py:381-389 口径)
  M5  顺序:A = w × (R − b),先减后乘
  M6  不做优势归一化:同组不同树的优势不被除以标准差
  M7  广播:(bs, L) 形状,pad 位置为 0;无元数据的行优势 0
  M8  与独立朴素实现全量对拍(随机批次)

运行:source SDAR_RAO/env_rao.sh && cd "$SDAR_REPO" && python tests/test_rao_core.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verl.trainer.ppo import rao_core  # noqa: E402  (惩罚相关的用例直接用模块名调用)
from verl.trainer.ppo.rao_core import (  # noqa: E402
    compute_rao_outcome_advantage, depth_weights, node_rewards, tree_baselines)

passed = 0
def check(cond, msg):
    global passed
    assert cond, msg
    passed += 1
    print(f"  ✓ {msg}")


def build_batch():
    """一个 GRPO 组 "A" 里 3 棵树,外加一个单树组 "B"。每行 = 一轮。
    树 T1: root r1(s=1, 有子) ── c1(s=1, 3 行) ── c2(s=0, 1 行);r1 自己 2 行
    树 T2: root r2(s=0, 无子, 2 行)
    树 T3: root r3(s=0, 有子, 1 行) ── c3(s=1, 2 行)
    树 T4(组 B): root r4(s=1, 无子, 2 行)
    """
    rows = []   # (uid, tree, node, depth, success, child_mean, has_children, root)
    rows += [("A", "T1", "r1", 0, 1.0, 0.5, True, "r1")] * 2
    rows += [("A", "T1", "c1", 1, 1.0, 0.0, False, "r1")] * 3
    rows += [("A", "T1", "c2", 1, 0.0, 0.0, False, "r1")] * 1
    rows += [("A", "T2", "r2", 0, 0.0, 0.0, False, "r2")] * 2
    rows += [("A", "T3", "r3", 0, 0.0, 1.0, True, "r3")] * 1
    rows += [("A", "T3", "c3", 1, 1.0, 0.0, False, "r3")] * 2
    rows += [("B", "T4", "r4", 0, 1.0, 0.0, False, "r4")] * 2
    cols = list(zip(*rows))
    d = dict(index=np.array(cols[0], dtype=object), traj_index=np.array(cols[1], dtype=object),
             node_uid=np.array(cols[2], dtype=object), node_depth=np.array(cols[3]),
             node_success=np.array(cols[4], dtype=np.float32),
             node_children_success_mean=np.array(cols[5], dtype=np.float32),
             node_has_children=np.array(cols[6], dtype=bool),
             root_node_uid=np.array(cols[7], dtype=object))
    L = 4
    mask = torch.ones(len(rows), L)
    mask[0, 2:] = 0                      # 第 0 行只有 2 个真 token
    return d, mask


def test_M1():
    print("M1 Eq.1")
    d, _ = build_batch()
    R0 = node_rewards(d["node_uid"], d["node_success"], d["node_children_success_mean"], d["node_has_children"], lam=0.0)
    check(R0 == {"r1": 1.0, "c1": 1.0, "c2": 0.0, "r2": 0.0, "r3": 0.0, "c3": 1.0, "r4": 1.0}, "λ=0:R = 自身成败")
    R4 = node_rewards(d["node_uid"], d["node_success"], d["node_children_success_mean"], d["node_has_children"], lam=0.4)
    check(abs(R4["r1"] - 1.2) < 1e-9 and abs(R4["r3"] - 0.4) < 1e-9, "λ=0.4:r1=1+0.4×0.5, r3=0+0.4×1.0")
    check(R4["c1"] == 1.0 and R4["r2"] == 0.0 and R4["r4"] == 1.0, "无子代节点不加委托项")


def test_M2_M3():
    print("M2/M3 Eq.3")
    b = tree_baselines(np.array(["T1", "T2", "T3"]), np.array(["A", "A", "A"]), np.array([1.0, 0.0, 0.0]))
    check(abs(b["T1"] - 0.0) < 1e-9 and abs(b["T2"] - 0.5) < 1e-9 and abs(b["T3"] - 0.5) < 1e-9,
          "组 A 的 LOO:T1→(0+0)/2=0, T2→(1+0)/2=0.5, T3→0.5")
    b1 = tree_baselines(np.array(["T4"]), np.array(["B"]), np.array([1.0]))
    check(abs(b1["T4"] - 1.0) < 1e-9, "单树组:基线 = 含自己的均值 = 自己 → 优势将为 0(P7)")
    b_off = tree_baselines(np.array(["T1", "T2", "T3"]), np.array(["A"] * 3), np.array([1.0, 0.0, 0.0]), leave_one_out=False)
    check(all(abs(v - 1 / 3) < 1e-9 for v in b_off.values()), "关 LOO:全组减含自己的均值 1/3(step_wise.py:130)")


def test_M4():
    print("M4 Eq.4")
    d, _ = build_batch()
    w = depth_weights(d["node_depth"], d["node_uid"])
    # depth0: 节点 r1,r2,r3,r4 = 4 个,行数 2+2+1+2 = 7;depth1: 节点 c1,c2,c3 = 3 个,行数 3+1+2 = 6
    # raw = {0: 1/4, 1: 1/3};unnorm = 7/4 + 6/3 = 3.75;norm = 13/3.75
    norm = 13 / 3.75
    check(abs(w[0] - norm / 4) < 1e-9 and abs(w[1] - norm / 3) < 1e-9, f"w0={w[0]:.4f} w1={w[1]:.4f}(分母=节点数)")
    check(abs(7 * w[0] + 6 * w[1] - 13) < 1e-9, "Σ 行数_d × w_d = 总行数 13(守恒量=行数)")
    check(w[0] < w[1], "深度 0 节点数多 → 权重反而小?否:4 节点 vs 3 节点,w0<w1 ✓ 反频率")


def test_M5_M6_M7():
    print("M5/M6/M7 组合、无归一化、广播")
    d, mask = build_batch()
    stats = {}
    adv, ret = compute_rao_outcome_advantage(response_mask=mask, lam=0.0, stats=stats, **d)
    check(adv.shape == mask.shape and torch.equal(adv, ret), "形状 (bs,L),returns==advantages")
    w = depth_weights(d["node_depth"], d["node_uid"])
    row = adv[:, 0].numpy()
    # 树 T1 基线 0:r1 行 = w0×(1−0);c1 行 = w1×(1−0);c2 行 = w1×(0−0)=0
    check(abs(row[0] - w[0] * 1.0) < 1e-6 and abs(row[2] - w[1] * 1.0) < 1e-6 and abs(row[5]) < 1e-9,
          "T1:root=w0×1, c1=w1×1, c2=0(先减基线再乘权重)")
    # 树 T2 基线 0.5:r2 行 = w0×(0−0.5)
    check(abs(row[6] - w[0] * (-0.5)) < 1e-6, "T2:root = w0×(0−0.5)")
    # 树 T3 基线 0.5:r3 = w0×(0−0.5);c3 = w1×(1−0.5)
    check(abs(row[8] - w[0] * (-0.5)) < 1e-6 and abs(row[9] - w[1] * 0.5) < 1e-6, "T3:root=w0×(−0.5), c3=w1×(+0.5)(子共用 root 基线)")
    # 单树组 B:优势 0
    check(abs(row[11]) < 1e-9 and abs(row[12]) < 1e-9, "组 B 单树:优势 0")
    # 无归一化:同组内优势的 std 不等于 1
    a_grp = row[:11]
    check(abs(a_grp.std() - 1.0) > 1e-3, f"未做 std 归一化(组内 std={a_grp.std():.3f}≠1)")
    check(adv[0, 2].item() == 0.0 and adv[0, 3].item() == 0.0 and adv[0, 1].item() != 0.0, "pad 位置为 0")
    check(stats["rao/nodes"] == 7 and stats["rao/trees"] == 4 and stats["rao/delegating_trees"] == 2
          and stats["rao/max_depth"] == 1, f"诊断量 {{nodes 7, trees 4, delegating 2, max_depth 1}}")
    # 无元数据的行
    d2 = {k: v.copy() for k, v in d.items()}
    d2["node_uid"][4] = ""
    adv2, _ = compute_rao_outcome_advantage(response_mask=mask, lam=0.0, **d2)
    check(adv2[4].abs().sum().item() == 0.0, "无节点元数据的行优势为 0")


def naive_reference(d, lam, loo, dw):
    """逐行朴素实现(与 rao_core 完全独立写法),用于对拍。"""
    n = len(d["node_uid"])
    R = {}
    for i in range(n):
        u = d["node_uid"][i]
        R[u] = float(d["node_success"][i]) + (lam * float(d["node_children_success_mean"][i]) if d["node_has_children"][i] else 0.0)
    trees = {}
    for i in range(n):
        trees.setdefault(d["traj_index"][i], (d["index"][i], d["root_node_uid"][i]))
    groups = {}
    for t, (g, r) in trees.items():
        groups.setdefault(g, []).append((t, R[r]))
    b = {}
    for g, lst in groups.items():
        for t, rr in lst:
            others = [x for tt, x in lst if tt != t]
            b[t] = (sum(others) / len(others)) if (loo and others) else (sum(x for _, x in lst) / len(lst))
    if dw:
        depths = sorted(set(int(x) for x in d["node_depth"]))
        rows_d = {dd: sum(1 for x in d["node_depth"] if int(x) == dd) for dd in depths}
        nodes_d = {dd: len({d["node_uid"][i] for i in range(n) if int(d["node_depth"][i]) == dd}) for dd in depths}
        raw = {dd: 1.0 / nodes_d[dd] for dd in depths}
        norm = n / sum(rows_d[dd] * raw[dd] for dd in depths)
        w = {dd: norm * raw[dd] for dd in depths}
    else:
        w = None
    out = np.zeros(n)
    for i in range(n):
        out[i] = (R[d["node_uid"][i]] - b[d["traj_index"][i]]) * (w[int(d["node_depth"][i])] if w else 1.0)
    return out


def test_M8():
    print("M8 随机批次与朴素实现对拍")
    rng = np.random.RandomState(0)
    for trial in range(20):
        rows = []
        for g in range(rng.randint(1, 4)):                      # 1-3 个组
            for t in range(rng.randint(1, 5)):                  # 每组 1-4 棵树
                tree = f"g{g}t{t}"
                root = f"{tree}_r"
                n_children = rng.randint(0, 4)
                cs = [float(rng.randint(0, 2)) for _ in range(n_children)]
                rs = float(rng.randint(0, 2))
                for _ in range(rng.randint(1, 4)):
                    rows.append((f"g{g}", tree, root, 0, rs, (np.mean(cs) if cs else 0.0), n_children > 0, root))
                for k, s in enumerate(cs):
                    depth = rng.randint(1, 4)
                    for _ in range(rng.randint(1, 4)):
                        rows.append((f"g{g}", tree, f"{tree}_c{k}", depth, s, 0.0, False, root))
        cols = list(zip(*rows))
        d = dict(index=np.array(cols[0], dtype=object), traj_index=np.array(cols[1], dtype=object),
                 node_uid=np.array(cols[2], dtype=object), node_depth=np.array(cols[3]),
                 node_success=np.array(cols[4], dtype=np.float32),
                 node_children_success_mean=np.array(cols[5], dtype=np.float32),
                 node_has_children=np.array(cols[6], dtype=bool),
                 root_node_uid=np.array(cols[7], dtype=object))
        mask = torch.ones(len(rows), 3)
        for lam in (0.0, 0.4):
            for loo in (True, False):
                for dw in (True, False):
                    adv, _ = compute_rao_outcome_advantage(response_mask=mask, lam=lam,
                                                           leave_one_out=loo, depth_level_weighting=dw, **d)
                    ref = naive_reference(d, lam, loo, dw)
                    assert np.allclose(adv[:, 0].numpy(), ref, atol=1e-5), (trial, lam, loo, dw)
    check(True, "20 个随机批次 × {λ∈{0,0.4}} × {LOO 开/关} × {深度加权 开/关} 全部与朴素实现一致")




# ---------------------------------------------------------------------------
# 无效动作惩罚(2026-08-30 加)。规则参照 ray_trainer.apply_invalid_action_penalty:236-260:
# 每一个动作不合格的回合,固定扣 invalid_action_penalty_coef 分。区别只在落点——flat 减在
# token_level_scores 上,RAO 减在行优势上,因为 RAO 根本不读 token_level_rewards。
# ---------------------------------------------------------------------------
def _two_node_batch(unbalanced: bool = False):
    """两棵树,每棵一个根一个子节点,每个节点两行。返回 compute_rao_outcome_advantage 的入参。

    unbalanced=True 时把第二棵树的子节点也算成根节点的行,于是深度 0 有两个节点、深度 1 只有一个,
    深度权重就不再全是 1,能把"扣分乘不乘权重"这两种模式区分开。
    """
    import numpy as np, torch
    rows = [("t0", "t0r", 0, 1.0), ("t0", "t0r", 0, 1.0),
            ("t0", "t0c", 1, 1.0), ("t0", "t0c", 1, 1.0),
            ("t1", "t1r", 0, 0.0), ("t1", "t1r", 0, 0.0),
            ("t1", "t1c", 1, 0.0), ("t1", "t1c", 1, 0.0)]
    if unbalanced:
        rows[6] = ("t1", "t1r", 0, 0.0)
        rows[7] = ("t1", "t1r", 0, 0.0)
    n = len(rows)
    return dict(
        response_mask=torch.ones(n, 3),
        index=np.array(["g0"] * n, dtype=object),
        traj_index=np.array([r[0] for r in rows], dtype=object),
        node_uid=np.array([r[1] for r in rows], dtype=object),
        node_depth=np.array([r[2] for r in rows]),
        node_success=np.array([r[3] for r in rows]),
        node_children_success_mean=np.zeros(n),
        node_has_children=np.array([r[2] == 0 for r in rows]),
        root_node_uid=np.array([r[0] + "r" for r in rows], dtype=object),
    )


def test_penalty_off_by_default():
    """不传系数时,行优势和加惩罚之前一模一样。"""
    import torch
    kw = _two_node_batch()
    a0, _ = rao_core.compute_rao_outcome_advantage(**kw)
    a1, _ = rao_core.compute_rao_outcome_advantage(**kw, invalid_action_penalty_coef=0.0,
                                                   is_action_valid=np.zeros(8, dtype=bool))
    assert torch.allclose(a0, a1)


def test_penalty_subtracts_fixed_amount_on_invalid_rows():
    """只有不合格的那一行被扣,扣的数额正好是系数,合格的行分毫不动。"""
    import torch
    kw = _two_node_batch()
    valid = np.ones(8, dtype=bool)
    valid[2] = False                      # 第 2 行(t0 的子节点第一轮)动作不合格
    base, _ = rao_core.compute_rao_outcome_advantage(**kw)
    pen, _ = rao_core.compute_rao_outcome_advantage(**kw, is_action_valid=valid,
                                                    invalid_action_penalty_coef=0.1)
    diff = (base - pen)[:, 0]
    assert abs(float(diff[2]) - 0.1) < 1e-6
    for i in range(8):
        if i != 2:
            assert abs(float(diff[i])) < 1e-6


def test_penalty_not_depth_weighted_by_default_but_can_be():
    """默认扣分不乘深度权重;打开开关后,扣的数额等于系数乘上该深度的权重。"""
    kw = _two_node_batch(unbalanced=True)
    valid = np.ones(8, dtype=bool); valid[2] = False
    base, _ = rao_core.compute_rao_outcome_advantage(**kw)
    flat_pen, _ = rao_core.compute_rao_outcome_advantage(
        **kw, is_action_valid=valid, invalid_action_penalty_coef=0.1)
    dw_pen, _ = rao_core.compute_rao_outcome_advantage(
        **kw, is_action_valid=valid, invalid_action_penalty_coef=0.1,
        invalid_penalty_depth_weighted=True)
    stats = {}
    rao_core.compute_rao_outcome_advantage(**kw, stats=stats)
    w1 = stats["rao/depth_weight_d1"]
    assert abs(float((base - flat_pen)[2, 0]) - 0.1) < 1e-6
    assert abs(float((base - dw_pen)[2, 0]) - 0.1 * w1) < 1e-6
    assert abs(w1 - 1.0) > 1e-9, "本用例要求深度权重不等于 1,否则两种模式区分不开"


def test_penalty_skips_rows_without_node_metadata():
    """没有节点元数据的行(某一轮这个槽没有节点在行动)不该被扣分,优势仍然是 0。"""
    kw = _two_node_batch()
    kw["node_uid"] = np.array(["", "t0r", "t0c", "t0c", "t1r", "t1r", "t1c", "t1c"], dtype=object)
    valid = np.zeros(8, dtype=bool)       # 全部标成不合格
    out, _ = rao_core.compute_rao_outcome_advantage(
        **kw, is_action_valid=valid, invalid_action_penalty_coef=0.1)
    assert abs(float(out[0, 0])) < 1e-9


def test_penalty_stats_report_valid_ratio():
    """统计量里要能看到这一步有多少行动作不合格,方便训练时盯着它。"""
    kw = _two_node_batch()
    valid = np.ones(8, dtype=bool); valid[2] = False; valid[5] = False
    stats = {}
    rao_core.compute_rao_outcome_advantage(**kw, is_action_valid=valid,
                                           invalid_action_penalty_coef=0.1, stats=stats)
    assert stats["rao/invalid_action_rows"] == 2.0
    assert abs(stats["rao/valid_action_ratio"] - 6 / 8) < 1e-9


if __name__ == "__main__":
    for fn in [test_M1, test_M2_M3, test_M4, test_M5_M6_M7, test_M8,
               test_penalty_off_by_default,
               test_penalty_subtracts_fixed_amount_on_invalid_rows,
               test_penalty_not_depth_weighted_by_default_but_can_be,
               test_penalty_skips_rows_without_node_metadata,
               test_penalty_stats_report_valid_ratio]:
        fn()
        print(f"  通过: {fn.__name__}")
    print(f"\n=== rao_core 数学测试与无效动作惩罚测试全部通过({passed} 项断言)===")
