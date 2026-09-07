# -*- coding: utf-8 -*-
"""RSO 优势估计:A_r = A_out(树) + α·A_prog(节点) − η·1[无效](η 默认 0)。

设计出处(逐条对应):
  A_out:RSO_advantage_design.md §4.2 —— 组内留一法,只用各树 root 结局,同树所有行共用;
  A_prog:RSO_progress_shaping.md §四 —— 节点每轮平均进展 ḡ 减组内平均进展速率 b̄,
         除整批(逐节点)RMS,clip(−c, c),c=3;同节点所有行共用;
  组合:RSO_advantage_design.md §五 —— 行优势 = 树级项 + α·节点级项;根节点不是特例;
  b̄ 口径:RSO_progress_shaping.md §十三 —— 倾向留一法(与 A_out 口径一致),留开关;
  η:【用户拍板 2026-09-06】η=0.1 常开(门控默认关闭,阈值 0)。RSO_open_risks.md §二
     记录的顾虑(全退化批里 η 会成为唯一的均匀负梯度,v2 实测教训)作为已知风险挂账,
     靠底座修正(熵不进梯度)防止走到那一步;门控代码保留,`invalid_gate_min_valid_ratio>0`
     可重新启用,供消融。
  已去掉(相对 rao_core):节点局部成败 s(X)、深度权重 w_d、λ、优势不做任何全局非线性
  (clip 之外),树大小归一(设计文档 §十 已废弃)。

与 rao_core 相同的工程约定:纯 numpy/torch 函数、行级输入、(bs, resp_len) 输出、
stats 字典选填;不读 token_level_rewards。
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch


# =============================================================================
# 小件:每个都可单测,也便于用设计文档 §七 的算例逐项对拍
# =============================================================================

def loo_outcome(tree_ids, tree_groups: Dict[str, str], tree_rewards) -> Dict[str, float]:
    """A_out(τ) = R(τ) − 组内其余树 root 结局的均值(留一法)。
    tree_groups 是 {树id: 组id} 的字典(与 progress_rate_baseline 同口径)。
    组里只有一棵树时没有"其余树",退化为减自身均值 → 0(与 rao_core 的处置一致)。"""
    by_group = defaultdict(list)
    for t in tree_ids:
        by_group[tree_groups[t]].append(t)
    out = {}
    for g, ts in by_group.items():
        total = sum(tree_rewards[t] for t in ts)
        k = len(ts)
        for t in ts:
            if k > 1:
                out[t] = tree_rewards[t] - (total - tree_rewards[t]) / (k - 1)
            else:
                out[t] = 0.0
    return out


def progress_rate_baseline(tree_T: Dict[str, float], tree_n: Dict[str, int],
                           tree_groups: Dict[str, str], leave_one_out: bool) -> Dict[str, float]:
    """b̄:组内平均进展速率 = 组内总进展 ÷ 组内总行数(RSO_progress_shaping.md §4.1)。
    留一法口径下,每棵树的基线用组内【其余】树的总进展/总行数;组里只有一棵树时
    退回含自身口径。返回 {tree_id: b̄}。"""
    by_group = defaultdict(list)
    for t, g in tree_groups.items():
        by_group[g].append(t)
    out = {}
    for g, ts in by_group.items():
        sum_T = sum(tree_T[t] for t in ts)
        sum_n = sum(tree_n[t] for t in ts)
        for t in ts:
            if leave_one_out and len(ts) > 1 and (sum_n - tree_n[t]) > 0:
                out[t] = (sum_T - tree_T[t]) / (sum_n - tree_n[t])
            else:
                out[t] = sum_T / max(sum_n, 1)
    return out


def progress_advantage(node_gbar: Dict[str, float], node_tree: Dict[str, str],
                       baselines: Dict[str, float], clip_c: float,
                       eps: float = 1e-8) -> Tuple[Dict[str, float], float, float]:
    """A_prog(X) = clip((ḡ − b̄)/RMS, −c, +c)。
    RMS 在整批、【逐节点】上取(每节点一票,不按行数加权——否则大节点主导尺度,
    RSO_progress_shaping.md §八),作用于已减基线的值。
    返回 ({node: A_prog}, 裁剪前 RMS, 被裁比例)。"""
    devs = {u: node_gbar[u] - baselines[node_tree[u]] for u in node_gbar}
    if not devs:
        return {}, 0.0, 0.0
    arr = np.array(list(devs.values()), dtype=np.float64)
    rms = float(np.sqrt(np.mean(arr ** 2)))
    scaled = {u: d / (rms + eps) for u, d in devs.items()}
    clipped = {u: float(np.clip(v, -clip_c, clip_c)) for u, v in scaled.items()}
    clip_ratio = float(np.mean([abs(v) > clip_c for v in scaled.values()]))
    return clipped, rms, clip_ratio


# =============================================================================
# 组合:DataProto 行级入口(签名风格与 rao_core.compute_rao_outcome_advantage 一致)
# =============================================================================

def compute_rso_outcome_advantage(
    response_mask: torch.Tensor,
    index: np.ndarray,            # uid:GRPO 组 = 同一道题的 8 棵树
    traj_index: np.ndarray,       # traj_uid:树 id
    node_uid: np.ndarray,         # 本行属于哪个节点
    node_depth: np.ndarray,       # 深度(只进统计,不进公式——w_d 已废)
    node_success: np.ndarray,     # 节点自身成败(只用 root 的,取树结局)
    root_node_uid: np.ndarray,    # 本树 root 的 node_uid
    delta_phi: np.ndarray,        # 本行的 ΔΦ(适配器算好回填;委派/陪跑行为 0)
    phi_frozen: np.ndarray,       # 本行时刻 Φ 是否已冻结(只进统计)
    is_action_valid: Optional[np.ndarray] = None,
    progress_coef: float = 0.1,           # α
    progress_clip: float = 3.0,           # c
    progress_baseline_loo: bool = True,   # b̄ 的留一法口径(§十三 倾向)
    invalid_coef: float = 0.1,            # η,默认 0.1 常开(用户拍板 2026-09-06)
    invalid_gate_min_valid_ratio: float = 0.0,   # 门控阈值,0=常开;>0 时批合格率低于它则不施加
    eps: float = 1e-8,
    stats: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bs = response_mask.shape[0]
    node_uid = np.asarray(node_uid, dtype=object)
    index = np.asarray(index, dtype=object)
    traj_index = np.asarray(traj_index, dtype=object)
    root_node_uid = np.asarray(root_node_uid, dtype=object)
    delta_phi = np.asarray(delta_phi, dtype=np.float64)
    node_success = np.asarray(node_success, dtype=np.float64)
    node_depth = np.asarray(node_depth)

    valid = np.array([bool(str(u)) for u in node_uid])   # 无节点元数据的行(陪跑)优势为 0
    adv = np.zeros(bs, dtype=np.float64)
    if valid.any():
        v = np.where(valid)[0]
        # ΔΦ 允许为负(2026-09-07 甲案):负值 = 净破坏,如实进 G 与 ḡ。
        # 原来的非负断言已删;负值率作为退化预警打点(rso/G_negative_ratio)。

        # ---- 聚合:树级 R/T/n,节点级 G/n
        tree_R: Dict[str, float] = {}
        tree_T: Dict[str, float] = defaultdict(float)
        tree_n: Dict[str, int] = defaultdict(int)
        tree_group: Dict[str, str] = {}
        node_G: Dict[str, float] = defaultdict(float)
        node_n: Dict[str, int] = defaultdict(int)
        node_tree: Dict[str, str] = {}
        for i in v:
            t = str(traj_index[i]); u = str(node_uid[i])
            tree_T[t] += delta_phi[i]
            tree_n[t] += 1
            tree_group[t] = str(index[i])
            node_G[u] += delta_phi[i]
            node_n[u] += 1
            node_tree[u] = t
            if u == str(root_node_uid[i]):
                tree_R[t] = float(node_success[i])   # 树结局 = root 节点的成败
        for t in tree_T:
            tree_R.setdefault(t, 0.0)    # 理论上每树必有 root 行;防御性缺省为失败

        # ---- 三个量
        a_out = loo_outcome(list(tree_T), tree_group, tree_R)
        baselines = progress_rate_baseline(tree_T, tree_n, tree_group,
                                           leave_one_out=progress_baseline_loo)
        node_gbar = {u: node_G[u] / node_n[u] for u in node_G}
        a_prog, prog_rms, clip_ratio = progress_advantage(
            node_gbar, node_tree, baselines, clip_c=progress_clip, eps=eps)
        neg_clip_ratio = float(np.mean([v_ <= -progress_clip for v_ in a_prog.values()])) \
            if a_prog else 0.0   # 负侧 clip 触发率(甲案后 −c 成为真约束,原恒 0)

        # ---- η 的门控(RSO_open_risks §二:全退化批里绝对惩罚会成为唯一梯度,禁用)
        eta = 0.0
        batch_valid_ratio = 1.0
        if is_action_valid is not None:
            row_ok = np.asarray(is_action_valid).astype(np.float64).reshape(-1)
            batch_valid_ratio = float(row_ok[v].mean()) if len(v) else 1.0
            if invalid_coef > 0 and batch_valid_ratio >= invalid_gate_min_valid_ratio:
                eta = float(invalid_coef)

        # ---- 行优势(设计文档 §五:同一个式子,根不是特例)
        for i in v:
            t = str(traj_index[i]); u = str(node_uid[i])
            adv[i] = a_out[t] + progress_coef * a_prog[u]
            if eta and row_ok[i] < 0.5:
                adv[i] -= eta

        if stats is not None:
            trees = list(tree_T)
            groups = defaultdict(list)
            for t in trees:
                groups[tree_group[t]].append(tree_R[t])
            depths_of_node = {}
            frozen_trees = set()
            for i in v:
                depths_of_node[str(node_uid[i])] = int(node_depth[i])
                if bool(np.asarray(phi_frozen)[i]):
                    frozen_trees.add(str(traj_index[i]))
            delegating = sum(1 for t in trees
                             if any(node_tree[u] == t and depths_of_node.get(u, 0) > 0
                                    for u in node_G))
            gbars = np.array(list(node_gbar.values()))
            aouts = np.array([a_out[t] for t in trees])
            aprogs = np.array([a_prog[u] for u in node_G])
            mean_abs_out = float(np.abs(aouts).mean())
            mean_abs_prog = float(np.abs(aprogs).mean()) * progress_coef
            stats.update({
                "rso/nodes": float(len(node_G)),
                "rso/trees": float(len(trees)),
                "rso/delegating_trees": float(delegating),
                "rso/root_reward_mean": float(np.mean([tree_R[t] for t in trees])),
                "rso/G_mean": float(np.mean(list(node_G.values()))),
                "rso/gbar_mean": float(gbars.mean()),
                "rso/gbar_p99": float(np.percentile(gbars, 99)),
                "rso/baseline_rate_mean": float(np.mean(list(baselines.values()))),
                "rso/prog_rms": prog_rms,
                "rso/prog_clip_ratio": clip_ratio,
                "rso/prog_neg_clip_ratio": neg_clip_ratio,
                "rso/G_negative_ratio": float(np.mean(delta_phi[v] < -1e-9)),
                "rso/infeasible_tree_ratio": float(len(frozen_trees) / max(1, len(trees))),
                "rso/zero_variance_group_ratio": float(np.mean(
                    [len(set(rs)) == 1 for rs in groups.values()])),
                "rso/a_out_abs_mean": mean_abs_out,
                "rso/a_prog_scaled_abs_mean": mean_abs_prog,
                "rso/prog_over_out_ratio": float(mean_abs_prog / (mean_abs_out + 1e-12)),
                "rso/valid_action_ratio": batch_valid_ratio,
                "rso/invalid_penalty_active": float(bool(eta)),
                "rso/adv_mean": float(adv[v].mean()),
                "rso/adv_std": float(adv[v].std()),
                "rso/adv_abs_max": float(np.abs(adv[v]).max()),
            })

    adv_t = torch.as_tensor(adv, dtype=torch.float32, device=response_mask.device)
    advantages = adv_t.unsqueeze(-1) * response_mask.to(torch.float32)
    return advantages, advantages
