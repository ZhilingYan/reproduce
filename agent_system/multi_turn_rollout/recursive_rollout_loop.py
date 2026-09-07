# -*- coding: utf-8 -*-
"""RAO 的轨迹收集器:在 flat 的 TrajectoryCollector 上加"节点字段回填",不复制采集循环。

移植来源(路径相对 Experiment/code_references/platoon/,行号为**本地副本**的行号):
  platoon/utils/areal_data_processing.py:401-461  get_train_data_for_trajectory_collection
      :434-450  逐条 trajectory 产训练行,每行贴 traj_depth(:441)与 traj_start(:447-449)
      :456-460  root 单独再算一次 reward 存成 task_reward(LOO 基线的原料)
  platoon/train/areal/workflows/step_wise.py:205-258  _process_trajectory_result
  官方是"一条 trajectory → 若干 datum,每个 datum 继承该 trajectory 的 reward";
  我们是"一个节点 → 若干行(每轮一行),每行贴该节点的 6 个字段",训练侧 rao_core 再按行算。

框架先例:
  agent_system/multi_turn_rollout/rollout_loop.py:325-395  gather_rollout_data(本文件覆写它)
  agent_system/multi_turn_rollout/rollout_loop.py:397-613  vanilla_multi_turn_loop(本文件【不复制】,只薄包装)
  agent_system/multi_turn_rollout/rollout_loop.py:693-762  multi_turn_loop(先 vanilla 再 gather 的调用顺序)

为什么不像旧实现那样整段拷贝 vanilla loop:旧 recursive_rollout_loop.py 拷了 168 行只为往行里塞
三个字段,任何上游修复都要手工同步。采集循环里 total_batch_list[slot][step_idx] 每轮每槽恰好一条
(rollout_loop.py:583-586,含已结束的陪跑槽,靠 active_masks 区分),而编排器每轮也记一份
turn_meta[round][slot](orchestrator.step 末尾),二者按位置一一对应——所以只要在 gather 阶段
按 (slot, step_idx) 查表就能回填,循环本身一行不用动。

回填的 6 个字段(rao_core.compute_rao_outcome_advantage 的输入契约,见 docs §步骤 4):
  node_uid / node_depth / root_node_uid            —— 直接来自 turn_meta
  node_success / node_children_success_mean / node_has_children
                                                    —— 来自 collect_node_records() 的 NodeRecord,
                                                       节点关闭时才知道,所以必须在 gather 阶段回填
另附 is_delegation_turn(诊断用,不参与优势计算)。

不支持 dynamic_multi_turn_loop(algorithm.filter_groups.enable=True 的 DAPO 动态采样):
官方 dynamic_sampling: false,且动态采样会多次调 vanilla loop、拼接多批,turn_meta 的位置对应会失效。
main_rao.py 里断言关闭。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from verl import DataProto
from agent_system.environments import EnvironmentManagerBase
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from agent_system.recursive.node import NodeRecord


class RecursiveTrajectoryCollector(TrajectoryCollector):
    """flat 的 TrajectoryCollector + 节点字段回填。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_envs = None
        self.last_node_records: List[List[NodeRecord]] = []

    # ------------------------------------------------------------ 薄包装
    def vanilla_multi_turn_loop(self, gen_batch: DataProto, actor_rollout_wg,
                                envs: EnvironmentManagerBase):
        """调 flat 的循环(一行不改),结束后收执行树记录。"""
        out = super().vanilla_multi_turn_loop(gen_batch, actor_rollout_wg, envs)
        self._last_envs = envs
        if hasattr(envs, "collect_node_records"):
            # 全局轮数到了还开着的节点在这里按 CLOSE_ROLLOUT_END 收摊(orchestrator.collect_node_records)
            self.last_node_records = envs.collect_node_records()
        else:
            self.last_node_records = []
        return out

    def dynamic_multi_turn_loop(self, *args, **kwargs):
        raise NotImplementedError(
            "[RAO] 递归收集器不支持 algorithm.filter_groups.enable=True 的动态采样"
            "(官方 dynamic_sampling: false;且多批拼接会破坏 turn_meta 的位置对应)。")

    # ------------------------------------------------------------ 回填
    def gather_rollout_data(self, total_batch_list: List[List[Dict]], episode_rewards: np.ndarray,
                            episode_lengths: np.ndarray, success: Dict[str, np.ndarray],
                            traj_uid: np.ndarray, tool_callings: np.ndarray) -> DataProto:
        envs = self._last_envs
        turn_meta = getattr(envs, "turn_meta", None)
        if turn_meta is None:
            raise RuntimeError("[RAO] envs 不是 RecursiveEnvironmentManager(没有 turn_meta),"
                               "RecursiveTrajectoryCollector 只能配递归环境管理器使用。")

        # 节点记录索引:uid → NodeRecord(全部槽合并;uid 是 uuid4,不会跨槽撞)
        rec_by_uid: Dict[str, NodeRecord] = {}
        for slot_recs in self.last_node_records:
            for r in slot_recs:
                rec_by_uid[r.uid] = r

        n_slots = len(total_batch_list)
        n_rounds_meta = len(turn_meta)
        missing = 0
        for bs in range(n_slots):
            rows = total_batch_list[bs]
            # 位置对应的硬断言:每轮每槽恰好一行,否则 step_idx ↔ round 的对应就错了
            assert len(rows) == n_rounds_meta, (
                f"[RAO] slot {bs} 有 {len(rows)} 行但编排器记了 {n_rounds_meta} 轮 turn_meta,位置对应失效")
            for step_idx, data in enumerate(rows):
                meta = turn_meta[step_idx][bs]
                uid = meta["node_uid"]
                data["node_uid"] = uid
                data["node_depth"] = int(meta["node_depth"])
                data["root_node_uid"] = meta["root_node_uid"]
                data["is_delegation_turn"] = bool(meta["is_delegation_turn"])
                # [RSO] 第 7/8 列:该行的 ΔΦ 与冻结标记(rso_core 的进展项原料)
                data["delta_phi"] = float(meta.get("delta_phi", 0.0))
                data["phi_frozen"] = bool(meta.get("phi_frozen", False))
                rec = rec_by_uid.get(uid)
                if rec is None:
                    # 陪跑行(槽已结束,uid 为空)在 flat 的 gather 里会被 active_masks 过滤掉;
                    # 活跃行却查不到记录说明节点没被关闭,是编排器的 bug,计数后报错
                    if data.get("active_masks"):
                        missing += 1
                    data["node_success"] = 0.0
                    data["node_children_success_mean"] = 0.0
                    data["node_has_children"] = False
                    continue
                cs = rec.children_success
                data["node_success"] = float(rec.success)
                data["node_children_success_mean"] = float(np.mean(cs)) if cs else 0.0
                data["node_has_children"] = bool(rec.children_uids)
        if missing:
            raise RuntimeError(f"[RAO] {missing} 个活跃行找不到所属节点的记录,节点未被正确关闭")

        # 之后交给 flat 的 gather:压平、过滤陪跑行、贴 episode 级字段、collate 成 DataProto。
        # 上面新贴的键会随每行 dict 一起进 collate,成为 non_tensor_batch 的列。
        return super().gather_rollout_data(
            total_batch_list=total_batch_list, episode_rewards=episode_rewards,
            episode_lengths=episode_lengths, success=success, traj_uid=traj_uid,
            tool_callings=tool_callings)
