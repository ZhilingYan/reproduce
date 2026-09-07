# -*- coding: utf-8 -*-
"""RAO(Recursive Agent Optimization, arXiv:2605.06639)的训练数学:Eq.1 / Eq.3 / Eq.4。

这是一个 DataProto 纯函数,接法与 GiGPO 的先例相同(ray_trainer.compute_advantage 里一个 elif
分支 + 独立模块),只读 batch 里的字段,不依赖 trainer 状态。每行 = 一个节点的一轮(turn),
节点级字段由采集器在 gather_rollout_data 之前回填(见 recursive_rollout_loop.py)。

移植来源(路径相对 Experiment/code_references/platoon/,行号为**本地副本**的行号):

  Eq.1 节点奖励
    plugins/textcraft/platoon/textcraft/train_scripts/areal/train_areal_synth.py:44-63  reward_processor
      :35   _TEXTCRAFT_SYNTH_DELEGATION_REWARD_CAP = 0.0        ← λ,官方 TextCraft-Synth 恒为 0
      :56   score = success_reward                              ← 自己的成败
      :59   subagent_success_rate = succeeded / launched        ← 直接子代的成功【率】(不是个数)
      :60   score += CAP * subagent_success_rate                ← λ·mean_c s(c)
    论文 §2.2.1 Eq.1:R(X) = s(X) + λ · (1/|C(X)|) Σ_c s(c),|C(X)|=0 时第二项为 0;
    论文 §3.1:"We do not include a delegation bonus (λ = 0)"。

  Eq.3 LOO 基线
    platoon/train/areal/workflows/step_wise.py:120-130
      :124  loo_baselines = (total_reward − task_rewards) / (N − 1)   ← 用【各树 root 的奖励】
      :127  per_datum_baselines = repeat_interleave(loo_baselines, datum_counts)
      :128  rewards −= per_datum_baselines                            ← 减在本树【全部】行上,含子节点
      :130  else 分支:关 LOO 或组内只有 1 棵树时,减【含自己】的组均值 ← 单树组优势为 0(审查报告 P7)
    论文 §2.2.2 Eq.3:A(τ) = R(τ) − b_{−g},b_{−g} = (1/(G−1)) Σ_{g'≠g} R_root^(g');
    "We use the same root-group baseline for all trajectories within a rollout tree, including child trajectories."

  Eq.4 深度反频率权重
    platoon/train/areal/rl.py:355-392
      :372  traj_counts[d] = Σ traj_start                       ← 每个深度的【轨迹(节点)数】
      :371  datum_counts[d]                                     ← 每个深度的【行数】
      :381  raw_weights = 1 / traj_counts
      :387  unnorm_total = Σ_d datum_counts_d × raw_weights_d
      :388  normalization = total_datums / unnorm_total         ← 守恒量是【行数】
      :389  per_depth_weights = normalization × raw_weights
      :392  batch["rewards"] = batch["rewards"] × per_datum_weights   ← 乘在【已减基线】的 reward 上
    论文 Eq.4:w_d = α / N_d,α 使总权重守恒。
    注意口径:权重分母用节点数,归一化守恒量用行数——这是官方代码的实际做法(论文 Eq.4 只写了
    按轨迹数守恒);本实现镜像代码(审查报告 P9)。

  不做优势归一化:官方 yaml `adv_norm: {mean_level: null, std_level: null}`。
  归一化会抹掉 Eq.4 的权重尺度,所以本函数【没有】GRPO 那种除以组标准差的步骤。

  【不移植】rl.py:325-349 的 γ^d 深度折扣(官方两份配置均为 null)。

顺序:先 Eq.1 得每行的节点奖励 → Eq.3 减本树的 LOO 基线 → Eq.4 乘所在深度的权重 → 广播到 token。
即 A(row) = w_{depth} × (R(node) − b_{tree})。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch


# =============================================================================
# Eq.1
# =============================================================================
def node_rewards(node_uid: np.ndarray, node_success: np.ndarray,
                 node_children_success_mean: np.ndarray, node_has_children: np.ndarray,
                 lam: float) -> Dict[str, float]:
    """每个节点的 Eq.1 奖励。输入按行给(同一节点的多行值相同),输出 {node_uid: R}。

    对应 train_areal_synth.py:56-60。λ=0 时 R = 节点自己的成败。
    has_children=False 时第二项为 0(论文:|C(X)|=0 时该项为零;官方代码 launched>0 才加)。
    """
    out: Dict[str, float] = {}
    for i in range(len(node_uid)):
        u = str(node_uid[i])
        if not u or u in out:
            continue
        r = float(node_success[i])
        if lam != 0.0 and bool(node_has_children[i]):
            r += float(lam) * float(node_children_success_mean[i])
        out[u] = r
    return out


# =============================================================================
# Eq.3
# =============================================================================
def tree_baselines(tree_ids: np.ndarray, group_ids: np.ndarray, root_rewards: np.ndarray,
                   leave_one_out: bool = True) -> Dict[str, float]:
    """每棵树的基线。三个数组按【树】给(每棵树一项),输出 {tree_id: b}。

    对应 step_wise.py:120-130:
      LOO 且组内树数 > 1:b_g = (Σ_group R_root − R_root^(g)) / (N − 1)
      否则(关 LOO,或组内只有 1 棵树):b_g = 组内 R_root 的均值(含自己)→ 单树组优势为 0。
    基线只用 root 的奖励,不用子节点的——树内所有节点共用这一个数。
    """
    tree_ids = np.asarray(tree_ids, dtype=object)
    group_ids = np.asarray(group_ids, dtype=object)
    root_rewards = np.asarray(root_rewards, dtype=np.float64)
    out: Dict[str, float] = {}
    for g in np.unique(group_ids):
        idx = np.where(group_ids == g)[0]
        rr = root_rewards[idx]
        n = len(idx)
        if leave_one_out and n > 1:
            b = (rr.sum() - rr) / (n - 1)
        else:
            b = np.full(n, rr.mean(), dtype=np.float64)
        for k, i in enumerate(idx):
            out[str(tree_ids[i])] = float(b[k])
    return out


# =============================================================================
# Eq.4
# =============================================================================
def depth_weights(row_depth: np.ndarray, row_node_uid: np.ndarray) -> Dict[int, float]:
    """每个深度的权重 {depth: w_d}。

    对应 rl.py:371-389:分母是该深度的【节点数】(官方 traj_counts,用 traj_start 数出来的
    轨迹数),归一化使 Σ_d 行数_d × w_d = 总行数(官方 total_datums / unnorm_total)。
    """
    row_depth = np.asarray(row_depth).astype(int)
    row_node_uid = np.asarray(row_node_uid, dtype=object)
    depths = np.unique(row_depth)
    datum_counts = {int(d): int((row_depth == d).sum()) for d in depths}
    node_counts = {int(d): len(set(str(u) for u in row_node_uid[row_depth == d])) for d in depths}
    raw = {d: (1.0 / node_counts[d] if node_counts[d] > 0 else 0.0) for d in datum_counts}
    total = float(sum(datum_counts.values()))
    unnorm = float(sum(datum_counts[d] * raw[d] for d in datum_counts))
    if unnorm <= 0:
        return {d: 1.0 for d in datum_counts}
    norm = total / unnorm
    return {d: norm * raw[d] for d in datum_counts}


# =============================================================================
# 组合:DataProto 纯函数
# =============================================================================
def compute_rao_outcome_advantage(
    response_mask: torch.Tensor,
    index: np.ndarray,                      # uid:GRPO 组 = 同一 root 任务的 G 棵树
    traj_index: np.ndarray,                 # traj_uid:树 id(一个 batch 槽一棵树)
    node_uid: np.ndarray,                   # 本行由哪个节点产生
    node_depth: np.ndarray,                 # 节点深度,root=0
    node_success: np.ndarray,               # 节点自身成败(回填)
    node_children_success_mean: np.ndarray, # 直接子代成功率(回填,无子代 0)
    node_has_children: np.ndarray,          # 有无子代
    root_node_uid: np.ndarray,              # 本树 root 的 node_uid
    lam: float = 0.0,
    leave_one_out: bool = True,
    depth_level_weighting: bool = True,
    is_action_valid: Optional[np.ndarray] = None,      # 每行的动作是否合格(有 <thought> 且有 <action>)
    invalid_action_penalty_coef: float = 0.0,          # 每个不合格回合要扣多少分,0 表示不扣
    invalid_penalty_depth_weighted: bool = False,      # 扣分要不要也乘深度权重,默认不乘
    stats: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A(row) = w_depth × (R(node) − b_tree),广播到 response 的每个 token。

    签名镜像 core_algos.compute_grpo_outcome_advantage(:133-141):返回 (advantages, returns),
    形状均为 (bs, response_length),returns 与 advantages 相同(无 critic,占位)。
    不做优势归一化(官方 adv_norm: null)。

    无效动作惩罚(2026-08-30 加):如果这一行的动作不合格,就再从这一行的优势里减去
    invalid_action_penalty_coef。为什么要在优势上减、而不是像 flat 那样在 token_level_scores
    上减:RAO 的奖励来自节点记录,压根不读 token_level_rewards,在那上面减会被静默丢弃
    (审查报告 P6 记过这个坑)。所以这里是同一条规则的等价落点,系数也用 flat 的同一个
    配置项 actor_rollout_ref.actor.invalid_action_penalty_coef。
    默认不给这个扣分乘深度权重,因为 flat 的规则是"每个不合格回合固定扣同样多",而深度权重
    实测在 0.4 到 15 之间浮动,乘上去会让扣分忽大忽小,那就不是同一条规则了。
    只有带节点元数据的行才会被扣分;没有节点元数据的行(某一轮里这个槽没有节点在行动)
    本来优势就是 0,不是模型写出来的回合,不该罚。
    stats 若传入字典,会被填入诊断量(节点数/树数/委托树数/优势均值等)供日志。
    """
    bs = response_mask.shape[0]
    node_uid = np.asarray(node_uid, dtype=object)
    node_depth = np.asarray(node_depth).astype(int)
    index = np.asarray(index, dtype=object)
    traj_index = np.asarray(traj_index, dtype=object)
    root_node_uid = np.asarray(root_node_uid, dtype=object)

    valid = np.array([bool(str(u)) for u in node_uid])          # 无节点元数据的行优势为 0
    adv = np.zeros(bs, dtype=np.float64)
    if valid.any():
        v = np.where(valid)[0]

        # ---- Eq.1
        R = node_rewards(node_uid[v], np.asarray(node_success)[v],
                         np.asarray(node_children_success_mean)[v],
                         np.asarray(node_has_children)[v], lam)

        # ---- Eq.3:每棵树一个 root 奖励 → 组内 LOO
        trees, first = np.unique(traj_index[v], return_index=True)
        tree_group = index[v][first]
        tree_root = root_node_uid[v][first]
        tree_root_R = np.array([R.get(str(r), 0.0) for r in tree_root], dtype=np.float64)
        b = tree_baselines(trees, tree_group, tree_root_R, leave_one_out=leave_one_out)

        # ---- Eq.4
        w = depth_weights(node_depth[v], node_uid[v]) if depth_level_weighting else None

        # ---- 组合(顺序:先减基线,再乘权重;rl.py:392 乘在已减基线的 reward 上)
        for i in v:
            u = str(node_uid[i])
            centered = R[u] - b[str(traj_index[i])]
            adv[i] = centered * (w[int(node_depth[i])] if w is not None else 1.0)

        # ---- 无效动作惩罚(对应 ray_trainer.apply_invalid_action_penalty:236-260 的规则)
        n_invalid = 0
        if is_action_valid is not None and invalid_action_penalty_coef:
            row_valid = np.asarray(is_action_valid).astype(np.float32).reshape(-1)
            for i in v:
                if row_valid[i] >= 0.5:
                    continue
                n_invalid += 1
                pen = float(invalid_action_penalty_coef)
                if invalid_penalty_depth_weighted and w is not None:
                    pen *= w[int(node_depth[i])]
                adv[i] -= pen

        if stats is not None:
            depths_of_nodes = {u: int(d) for u, d in zip(node_uid[v], node_depth[v])}
            n_trees = len(trees)
            delegating = sum(1 for t in trees
                             if any(depths_of_nodes[str(u)] > 0
                                    for u in node_uid[v][traj_index[v] == t]))
            stats.update({
                "rao/nodes": float(len(R)),
                "rao/trees": float(n_trees),
                "rao/delegating_trees": float(delegating),
                "rao/max_depth": float(max(depths_of_nodes.values())),
                "rao/root_reward_mean": float(tree_root_R.mean()),
                "rao/adv_mean": float(adv[v].mean()),
                "rao/adv_std": float(adv[v].std()),
                "rao/adv_abs_max": float(np.abs(adv[v]).max()),
                "rao/invalid_action_rows": float(n_invalid),
                "rao/valid_action_ratio": float(1.0 - n_invalid / max(1, len(v))),
            })
            if w is not None:
                for d, wd in sorted(w.items()):
                    stats[f"rao/depth_weight_d{d}"] = float(wd)

    adv_t = torch.as_tensor(adv, dtype=torch.float32, device=response_mask.device)
    advantages = adv_t.unsqueeze(-1) * response_mask.to(torch.float32)
    return advantages, advantages
