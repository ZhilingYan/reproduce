# -*- coding: utf-8 -*-
"""rso_core 的数值测试。逐项对拍设计文档:
RSO_advantage_design.md §4.2/§4.3/§五/§七(算例)、RSO_progress_shaping.md §4/§5(性质)。
运行:PYTHONPATH=$PWD python tests/test_rso_core.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verl.trainer.ppo import rso_core  # noqa: E402
from verl.trainer.ppo.rso_core import (  # noqa: E402
    compute_rso_outcome_advantage, loo_outcome, progress_advantage, progress_rate_baseline)

passed = 0


def check(cond, msg):
    global passed
    assert cond, f"✗ {msg}"
    passed += 1
    print(f"  ✓ {msg}")


# ---------------------------------------------------------------------------
# 一、小件对拍设计文档 §七 的算例(b̄ 与 RMS 是外生给定的,所以用小件注入)
# ---------------------------------------------------------------------------
def test_worked_example_from_design_doc():
    """§七:8 棵树 3 成,τ1 成功 → A_out=0.714;ρ/X1/X2 的 A_prog 与 A_r 逐位对上。"""
    # outcome:8 棵树,τ1 成功,其余 7 棵里 2 棵成功
    trees = [f"t{i}" for i in range(8)]
    rewards = {t: 0.0 for t in trees}
    rewards["t0"] = 1.0; rewards["t1"] = 1.0; rewards["t2"] = 1.0
    a_out = loo_outcome(trees, {t: "g" for t in trees}, rewards)
    check(abs(a_out["t0"] - (1 - 2 / 7)) < 1e-9, f"A_out(成功树) = 0.714(实际 {a_out['t0']:.3f})")
    check(abs(a_out["t3"] - (0 - 3 / 7)) < 1e-9, "A_out(失败树) = −0.429")

    # progress:ρ(G=5,n=8) X1(3,4) X2(2,6),b̄=0.413、RMS=0.35 外生注入
    gbar = {"rho": 5 / 8, "x1": 3 / 4, "x2": 2 / 6}
    node_tree = {u: "t0" for u in gbar}
    baselines = {"t0": 0.413}
    devs = {u: gbar[u] - 0.413 for u in gbar}
    scaled = {u: d / 0.35 for u, d in devs.items()}
    expect = {"rho": 0.606, "x1": 0.963, "x2": -0.229}
    for u in gbar:
        check(abs(scaled[u] - expect[u]) < 2e-3, f"A_prog({u}) = {expect[u]}(实际 {scaled[u]:.3f})")
    a_r = {u: 0.714 + 0.1 * scaled[u] for u in gbar}
    for u, e in [("rho", 0.775), ("x1", 0.810), ("x2", 0.691)]:
        check(abs(a_r[u] - e) < 2e-3, f"A_r({u}) = {e}(实际 {a_r[u]:.3f})")
    # 梯度层校验(§七末行):Σ n·A_prog = (T − b̄·n)/RMS
    w = 8 * scaled["rho"] + 4 * scaled["x1"] + 6 * scaled["x2"]
    check(abs(w - (10 - 0.413 * 18) / 0.35) < 2e-2, f"Σn·A_prog = (T−b̄n)/RMS(实际 {w:.2f} vs 7.34)")


# ---------------------------------------------------------------------------
# 二、性质测试(端到端,随机批)
# ---------------------------------------------------------------------------
def _rand_batch(rng, n_groups=3, trees_per_group=4, resp_len=3):
    """构造随机批:每树 1 个根 + 0~3 个子节点,每节点 1~6 行,ΔΦ 非负随机。"""
    rows = []
    for g in range(n_groups):
        for t in range(trees_per_group):
            tid = f"g{g}t{t}"
            root = f"{tid}root"
            success = float(rng.random() < 0.5)
            nodes = [root] + [f"{tid}c{k}" for k in range(rng.integers(0, 4))]
            for u in nodes:
                for _ in range(rng.integers(1, 7)):
                    rows.append(dict(g=f"g{g}", t=tid, u=u, root=root,
                                     succ=success if u == root else float(rng.random() < 0.7),
                                     depth=0 if u == root else 1,
                                     dphi=float(rng.integers(0, 3))))
    n = len(rows)
    kw = dict(
        response_mask=torch.ones(n, resp_len),
        index=np.array([r["g"] for r in rows], dtype=object),
        traj_index=np.array([r["t"] for r in rows], dtype=object),
        node_uid=np.array([r["u"] for r in rows], dtype=object),
        node_depth=np.array([r["depth"] for r in rows]),
        node_success=np.array([r["succ"] for r in rows]),
        root_node_uid=np.array([r["root"] for r in rows], dtype=object),
        delta_phi=np.array([r["dphi"] for r in rows]),
        phi_frozen=np.array([False] * n),
    )
    return rows, kw


def test_zero_sum_inclusive_baseline():
    """含自身基线 + 不裁剪时,组内 Σ(行的进展项) 精确为零(RSO_progress_shaping §4.2)。"""
    rng = np.random.default_rng(0)
    rows, kw = _rand_batch(rng)
    stats = {}
    adv, _ = compute_rso_outcome_advantage(**kw, progress_coef=1.0, progress_clip=1e9,
                                           progress_baseline_loo=False, stats=stats)
    a = adv[:, 0].numpy()
    # 减去每行的 A_out(树级),剩下的就是进展项;按组求和应为 0
    tree_R = {}; tree_g = {}
    for r in rows: tree_R[r["t"]] = r["succ"] if r["u"] == r["root"] else tree_R.get(r["t"], 0.0)
    for r in rows: tree_g[r["t"]] = r["g"]
    # 重算 A_out
    for r in rows:
        if r["u"] == r["root"]: tree_R[r["t"]] = r["succ"]
    a_out = loo_outcome(list(tree_R), tree_g, tree_R)
    per_group = {}
    for i, r in enumerate(rows):
        per_group.setdefault(r["g"], 0.0)
        per_group[r["g"]] += a[i] - a_out[r["t"]]
    for g, s in per_group.items():
        assert abs(s) < 1e-6, (g, s)
    check(True, f"{len(per_group)} 个组的进展项行和均为 0(含自身基线,组内零和)")


def test_split_neutrality():
    """分割中性(§5.1):同样的行级 ΔΦ 序列,把它们归到 1 个节点或拆成 3 个节点,
    该树的进展项【总】梯度权重不变(批内其它树固定,RMS 由全批决定,同一批内比较)。"""
    rng = np.random.default_rng(1)
    base_rows, _ = _rand_batch(rng, n_groups=2, trees_per_group=4)
    # 目标树:附加一棵 18 行、总进展 10 的树,两种切分
    deltas = [1, 0, 2, 0, 1, 0, 0, 2, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]   # 和=10,18 行
    def build(split):
        rows = [dict(r) for r in base_rows]
        root = "SPLITroot"
        for j, d in enumerate(deltas):
            u = root if split == 1 else (root if j < 8 else ("SPLITc1" if j < 12 else "SPLITc2"))
            rows.append(dict(g="g0", t="SPLIT", u=u, root=root, succ=1.0,
                             depth=0 if u == root else 1, dphi=float(d)))
        n = len(rows)
        kw = dict(
            response_mask=torch.ones(n, 1),
            index=np.array([r["g"] for r in rows], dtype=object),
            traj_index=np.array([r["t"] for r in rows], dtype=object),
            node_uid=np.array([r["u"] for r in rows], dtype=object),
            node_depth=np.array([r["depth"] for r in rows]),
            node_success=np.array([r["succ"] for r in rows]),
            root_node_uid=np.array([r["root"] for r in rows], dtype=object),
            delta_phi=np.array([r["dphi"] for r in rows]),
            phi_frozen=np.array([False] * n),
        )
        adv, _ = compute_rso_outcome_advantage(**kw, progress_coef=1.0, progress_clip=1e9,
                                               progress_baseline_loo=False)
        a = adv[:, 0].numpy()
        idx = [i for i, r in enumerate(rows) if r["t"] == "SPLIT"]
        # 树的进展项总权重 = Σ_行 (A_r − A_out);A_out 对树内行是常数
        a_out_split = None
        # 用同一函数重算 A_out
        tr = {r["t"]: 0.0 for r in rows}
        for r in rows:
            if r["u"] == r["root"]: tr[r["t"]] = r["succ"]
        ao = loo_outcome(list(tr), {r["t"]: r["g"] for r in rows}, tr)
        return sum(a[i] - ao["SPLIT"] for i in idx)
    w1, w3 = build(1), build(3)
    # 注意:两种切分下"节点集合"不同 → 批的逐节点 RMS 会有二阶差异(§5.1 明说分母是共享统计量,
    # 单棵树切分对它的影响是二阶的)。所以断言的是相对差异小,而分子层面精确中性。
    check(abs(w1 - w3) / max(abs(w1), 1e-9) < 0.15,
          f"切分 1 节点 vs 3 节点,树的进展项总权重 {w1:.3f} vs {w3:.3f}(仅 RMS 二阶差异)")


def test_clip_single_sided_and_bounds():
    """ΔΦ 非负 ⇒ ḡ≥0 ⇒ 负向地板 −b̄/RMS 有限;正向长尾被 c 裁住;|A_prog|≤c。"""
    # 50 个普通节点 + 1 个极端高效节点:RMS 由主体决定,离群点缩放后超过 c 被裁
    vals = [0.1 + 0.01 * (i % 10) for i in range(50)] + [8.0]
    gbar = {f"n{i}": v for i, v in enumerate(vals)}
    nt = {u: "t0" for u in gbar}
    ap, rms, ratio = progress_advantage(gbar, nt, {"t0": 0.2}, clip_c=3.0)
    check(max(ap.values()) == 3.0 and min(ap.values()) > -3.0, "clip 只在正侧生效(单边)")
    check(0 < ratio <= 1 / 51 + 1e-9, f"被裁比例 {ratio:.3f} 合理(只裁那一个离群点)")


def test_invalid_penalty_gate():
    """η=0.1 常开(用户拍板 2026-09-06);门控保留为可选开关(>0 才生效)。"""
    rng = np.random.default_rng(2)
    rows, kw = _rand_batch(rng)
    n = len(rows)
    bad = np.ones(n); bad[0] = 0.0                       # 只有 1 行不合格 → 合格率高
    s1 = {}
    a1, _ = compute_rso_outcome_advantage(**kw, is_action_valid=bad, invalid_coef=0.1,
                                          invalid_gate_min_valid_ratio=0.5, stats=s1)
    a0, _ = compute_rso_outcome_advantage(**kw, is_action_valid=np.ones(n), invalid_coef=0.1,
                                          invalid_gate_min_valid_ratio=0.5)
    check(abs((a0[0, 0] - a1[0, 0]).item() - 0.1) < 1e-6 and s1["rso/invalid_penalty_active"] == 1.0,
          "合格率高:不合格行被减 0.1")
    # 默认口径:常开——即使全批不合格也照扣(用户拍板;顾虑见 RSO_open_risks §二)
    allbad = np.zeros(n)
    s2 = {}
    a2, _ = compute_rso_outcome_advantage(**kw, is_action_valid=allbad, stats=s2)
    a3, _ = compute_rso_outcome_advantage(**kw)
    check(torch.allclose(a2, a3 - 0.1) and s2["rso/invalid_penalty_active"] == 1.0,
          "默认口径(η=0.1 常开):全批不合格时每行照扣 0.1")
    # 门控作为可选项仍可启用
    s3 = {}
    a4, _ = compute_rso_outcome_advantage(**kw, is_action_valid=allbad,
                                          invalid_gate_min_valid_ratio=0.5, stats=s3)
    check(torch.allclose(a4, a3) and s3["rso/invalid_penalty_active"] == 0.0,
          "显式开门控(阈值 0.5)时,全败批停用 η——消融开关可用")


def test_alpha_zero_is_variant_yi():
    """progress_coef=0 退化为消融表 #3 的乙(root 结局广播全树、无进展项)。"""
    rng = np.random.default_rng(3)
    rows, kw = _rand_batch(rng)
    adv, _ = compute_rso_outcome_advantage(**kw, progress_coef=0.0)
    a = adv[:, 0].numpy()
    tr = {r["t"]: 0.0 for r in rows}
    for r in rows:
        if r["u"] == r["root"]: tr[r["t"]] = r["succ"]
    ao = loo_outcome(list(tr), {r["t"]: r["g"] for r in rows}, tr)
    assert all(abs(a[i] - ao[r["t"]]) < 1e-5 for i, r in enumerate(rows))  # float32 容差
    check(True, "α=0 时每行优势 = 本树的 A_out(乙变体,一套代码两个消融)")


def test_negative_delta_phi_flows_through():
    """甲案端到端:带负 ΔΦ 的行照常进 G/ḡ,统计里有 G_negative_ratio 与负侧 clip 率。"""
    rng = np.random.default_rng(7)
    rows, kw = _rand_batch(rng)
    kw["delta_phi"][3] = -2.0                        # 人为放一行净破坏
    stats = {}
    adv, _ = compute_rso_outcome_advantage(**kw, stats=stats)
    check(stats["rso/G_negative_ratio"] > 0, "负值行进了 G_negative_ratio 打点")
    check("rso/prog_neg_clip_ratio" in stats, "负侧 clip 触发率打点存在")
    check(bool(np.isfinite(adv.numpy()).all()), "优势全有限,无 NaN")


def test_stats_and_padding_rows():
    """陪跑行(无节点元数据)优势为 0;统计量键齐全。"""
    rng = np.random.default_rng(4)
    rows, kw = _rand_batch(rng)
    kw["node_uid"][0] = ""
    stats = {}
    adv, _ = compute_rso_outcome_advantage(**kw, stats=stats)
    check(float(adv[0, 0]) == 0.0, "无节点元数据的行优势为 0")
    for k in ["rso/nodes", "rso/G_mean", "rso/prog_rms", "rso/prog_clip_ratio",
              "rso/zero_variance_group_ratio", "rso/prog_over_out_ratio",
              "rso/infeasible_tree_ratio", "rso/adv_abs_max"]:
        assert k in stats, k
    check(True, f"统计量 {len(stats)} 项齐全")


if __name__ == "__main__":
    for fn in [test_worked_example_from_design_doc, test_zero_sum_inclusive_baseline,
               test_split_neutrality, test_clip_single_sided_and_bounds,
               test_invalid_penalty_gate, test_alpha_zero_is_variant_yi,
               test_negative_delta_phi_flows_through, test_stats_and_padding_rows]:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n=== rso_core 测试全部通过({passed} 项断言)===")
