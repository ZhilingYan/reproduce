# -*- coding: utf-8 -*-
"""步骤 5 CPU dry-run:不跑模型、不用 GPU、不起 Ray,验证从采集到优势的完整数据流。

链路:RecursiveTrajectoryCollector.multi_turn_loop(真实,含 flat 的 vanilla loop 与 gather)
      ← 假 actor(按剧本产生动作文本,用真 tokenizer 变 token)
      ← RecursiveEnvironmentManager + TextCraftSynthRecursiveAdapter(真实)
      ← LocalSynthEnvs(真实 SynthTextCraftEnv,不经 Ray)
      → DataProto → compute_advantage(adv_estimator='rao')(真实 ray_trainer 分支)

  C1  gather 输出含 6 个节点字段 + is_delegation_turn,且活跃行 node_uid 非空
  C2  委托树:root 3 行(get_info/delegate/craft)+ 子 1 行;回填值与节点记录一致
  C3  失败树(只查 inventory 直到 root 预算耗尽):root 行数 = 预算,success 0,无子代
  C4  compute_advantage(rao) 跑通:形状、同节点各行优势相同、成功树为正、失败树为负、子行为正
  C5  rao_stats 诊断量(trees=6, delegating_trees=2)进 meta_info

剧本(两组,每组 3 棵树,同组同任务):
  组 1:A=委托后合成(成功) B=直接按 gold 合成(成功) C=只查 inventory(失败)
  组 2:A、C、C

运行:source SDAR_RAO/env_rao.sh && cd "$SDAR_REPO" && HF_HUB_OFFLINE=1 python tests/test_recursive_collector.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from verl import DataProto                                                                          # noqa: E402
from verl.trainer.ppo.ray_trainer import compute_advantage                                          # noqa: E402
from agent_system.environments.env_package.textcraft_synth.projection import textcraft_synth_projection  # noqa: E402
from agent_system.environments.env_package.textcraft_synth.recursive_adapter import TextCraftSynthRecursiveAdapter  # noqa: E402
from agent_system.environments.env_package.textcraft_synth.synth_core import gold_to_actions, load_tasks  # noqa: E402
from agent_system.multi_turn_rollout.recursive_rollout_loop import RecursiveTrajectoryCollector    # noqa: E402
from agent_system.recursive.budget import PerAgentBudget                                            # noqa: E402
from agent_system.recursive.orchestrator import RecursiveEnvironmentManager                         # noqa: E402
from tests.test_recursive_synth_adapter import LocalSynthEnvs, pick_two_step_task                   # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PER_AGENT_STEPS = 6
GROUP = 3


def make_config():
    cfg = OmegaConf.load(os.path.join(REPO, "verl/trainer/config/ppo_trainer.yaml"))
    OmegaConf.set_struct(cfg, False)
    cfg.data.max_prompt_length = 4096
    cfg.data.max_response_length = 128
    cfg.data.truncation = "left"
    cfg.data.return_raw_chat = True
    cfg.env.env_name = "textcraft_synth"
    cfg.env.max_steps = 20                       # 全局轮数上限
    cfg.env.history_length = 2
    cfg.env.rollout.n = GROUP
    cfg.env.rao = {"per_agent_max_steps": PER_AGENT_STEPS, "max_depth": 6, "state_block_scope": "node"}
    cfg.algorithm.adv_estimator = "rao"
    cfg.algorithm.filter_groups.enable = False
    cfg.algorithm.rao = {"lam": 0.0}
    return cfg


class ScriptedActor:
    """假 actor_rollout_wg:看编排器的栈顶节点,按剧本给动作文本,再用真 tokenizer 变 token。"""
    world_size = 1

    def __init__(self, tokenizer, mgr, scripts, tasks, resp_len):
        self.tok, self.mgr, self.scripts, self.tasks, self.R = tokenizer, mgr, scripts, tasks, resp_len

    def _action_for_slot(self, i):
        if self.mgr.episode_done[i] or not self.mgr.stacks[i]:
            return "inventory"
        top = self.mgr.stacks[i][-1]
        task = self.tasks[i]
        g = task["misc"]["gold_trajectory"]
        gold = gold_to_actions(g)
        tgt = list(task["misc"]["target_items"])[0]
        mid_item, mid_count = g[0]["target"][0], g[0]["result_count"]
        kind = self.scripts[i]
        if kind == "A":                                   # 委托
            if top.depth == 1:
                return gold[0]                            # 子:做中间品
            return {0: f"get_info {tgt}", 1: f"delegate: craft {mid_count} {mid_item} | for {tgt}",
                    2: gold[1]}.get(top.turns_used, "inventory")
        if kind == "B":                                   # 直接 gold
            return gold[top.turns_used] if top.turns_used < len(gold) else "inventory"
        # C:摆烂直到 root 预算耗尽。必须交替动作——连续 4 次相同动作会触发环境的循环检测
        # (synth_core.LOOP_REPEAT_LIMIT=4)提前终局,那是 flat 也有的正确行为,不是这里要测的。
        return "inventory" if top.turns_used % 2 == 0 else f"get_info {tgt}"

    def generate_sequences(self, batch_input: DataProto) -> DataProto:
        n = batch_input.batch["input_ids"].shape[0]
        prompts = batch_input.batch["input_ids"]
        p_mask = batch_input.batch["attention_mask"]
        pad = self.tok.pad_token_id
        resp = torch.full((n, self.R), pad, dtype=torch.long)
        r_mask = torch.zeros((n, self.R), dtype=torch.long)
        for i in range(n):
            text = f"<thought>ok</thought><action>{self._action_for_slot(i)}</action>"
            ids = self.tok.encode(text, add_special_tokens=False)[: self.R]
            resp[i, :len(ids)] = torch.tensor(ids)
            r_mask[i, :len(ids)] = 1
        input_ids = torch.cat([prompts, resp], dim=1)
        attn = torch.cat([p_mask, r_mask], dim=1)
        pos = (torch.cumsum(attn, dim=1) - 1).clamp(min=0)
        return DataProto.from_single_dict({
            "prompts": prompts, "responses": resp, "input_ids": input_ids,
            "attention_mask": attn, "position_ids": pos,
        })


passed = 0
def check(cond, msg):
    global passed
    assert cond, msg
    passed += 1
    print(f"  ✓ {msg}")


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    cfg = make_config()

    tasks_pool = [t for t in load_tasks("train", ["easy"])
                  if len(t["misc"]["gold_trajectory"]) == 2
                  and t["misc"]["gold_trajectory"][0]["target"][0] != t["misc"]["gold_trajectory"][1]["target"][0]]
    T1, T2 = tasks_pool[0], tasks_pool[1]
    slot_tasks = [T1] * GROUP + [T2] * GROUP
    scripts = ["A", "B", "C", "A", "C", "C"]
    n_slots = len(slot_tasks)

    envs = LocalSynthEnvs(slot_tasks, max_steps=60)
    mgr = RecursiveEnvironmentManager(envs, textcraft_synth_projection, cfg,
                                      adapter=TextCraftSynthRecursiveAdapter(cfg),
                                      budget=PerAgentBudget.from_config(cfg.env.rao))
    collector = RecursiveTrajectoryCollector(config=cfg, tokenizer=tok, processor=None)
    actor = ScriptedActor(tok, mgr, scripts, slot_tasks, cfg.data.max_response_length)

    # gen_batch:两个任务(is_train 时 multi_turn_loop 会按 env.rollout.n 复制成 6 行)
    n_tasks = n_slots // GROUP
    raw_prompt = np.empty(n_tasks, dtype=object)
    for k in range(n_tasks):
        raw_prompt[k] = [{"role": "user", "content": ""}]
    gen_batch = DataProto.from_single_dict({
        "input_ids": torch.zeros((n_tasks, 1), dtype=torch.long),
        "raw_prompt": raw_prompt,
        "data_source": np.array(["textcraft_synth"] * n_tasks, dtype=object),
    })

    print("运行 multi_turn_loop(真实采集循环 + 真实 gather)...")
    data = collector.multi_turn_loop(gen_batch=gen_batch, actor_rollout_wg=actor, envs=mgr, is_train=True)
    ntb = data.non_tensor_batch
    n_rows = len(data)
    print(f"  产出 {n_rows} 行")

    # ---------------------------------------------------------------- C1
    print("C1 字段齐全")
    for k in ["node_uid", "node_depth", "node_success", "node_children_success_mean",
              "node_has_children", "root_node_uid", "is_delegation_turn", "uid", "traj_uid"]:
        check(k in ntb, f"non_tensor_batch 含 {k}")
    check(all(str(u) for u in ntb["node_uid"]), "活跃行 node_uid 全非空(陪跑行已被 flat gather 过滤)")

    # 按树分组
    recs = {r.uid: r for slot in collector.last_node_records for r in slot}
    trees = {}
    for i in range(n_rows):
        trees.setdefault(str(ntb["traj_uid"][i]), []).append(i)
    check(len(trees) == n_slots, f"{n_slots} 棵树")
    # 找出各剧本的树:按 root 节点的 children 与 success 识别
    def tree_kind(rows):
        root_rows = [i for i in rows if int(ntb["node_depth"][i]) == 0]
        r0 = root_rows[0]
        if bool(ntb["node_has_children"][r0]):
            return "A"
        return "B" if float(ntb["node_success"][r0]) == 1.0 else "C"
    kinds = {t: tree_kind(rows) for t, rows in trees.items()}
    check(sorted(kinds.values()) == sorted(scripts), f"树的剧本识别 {sorted(kinds.values())}")

    # ---------------------------------------------------------------- C2
    print("C2 委托树回填")
    for t, rows in trees.items():
        if kinds[t] != "A":
            continue
        root_rows = [i for i in rows if int(ntb["node_depth"][i]) == 0]
        child_rows = [i for i in rows if int(ntb["node_depth"][i]) == 1]
        check(len(root_rows) == 3 and len(child_rows) == 1, f"树 {t[:6]}: root 3 行, 子 1 行")
        r0, c0 = root_rows[0], child_rows[0]
        check(float(ntb["node_success"][r0]) == 1.0 and float(ntb["node_children_success_mean"][r0]) == 1.0,
              "root success=1, children_mean=1")
        check(float(ntb["node_success"][c0]) == 1.0 and not bool(ntb["node_has_children"][c0]), "子 success=1, 无子代")
        check(str(ntb["root_node_uid"][c0]) == str(ntb["node_uid"][r0]), "子行的 root_node_uid 指向 root")
        check(sum(bool(ntb["is_delegation_turn"][i]) for i in root_rows) == 1, "root 恰有 1 个委托轮")
        rec = recs[str(ntb["node_uid"][c0])]
        check(rec.depth == 1 and rec.success == 1.0 and rec.close_reason == "goal_met", "子节点记录一致")

    # ---------------------------------------------------------------- C3
    print("C3 失败树")
    for t, rows in trees.items():
        if kinds[t] != "C":
            continue
        check(len(rows) == PER_AGENT_STEPS and all(int(ntb["node_depth"][i]) == 0 for i in rows),
              f"树 {t[:6]}: root 预算 {PER_AGENT_STEPS} 步用尽,{len(rows)} 行")
        check(all(float(ntb["node_success"][i]) == 0.0 for i in rows), "success 全 0")
        rec = recs[str(ntb["node_uid"][rows[0]])]
        check(rec.close_reason == "budget_exhausted", "root 关闭原因 budget_exhausted")

    # ---------------------------------------------------------------- C4
    print("C4 compute_advantage(rao)")
    data = compute_advantage(data, adv_estimator="rao", gamma=1.0, lam=1.0, num_repeat=1,
                             multi_turn=False, norm_adv_by_std_in_grpo=True,
                             rao_lam=0.0, rao_leave_one_out=True, rao_depth_level_weighting=True)
    adv = data.batch["advantages"]
    mask = data.batch["response_mask"]
    check(adv.shape == mask.shape and torch.isfinite(adv).all(), f"advantages 形状 {tuple(adv.shape)},有限")
    row_adv = (adv.sum(-1) / mask.sum(-1).clamp(min=1)).numpy()
    for t, rows in trees.items():
        by_node = {}
        for i in rows:
            by_node.setdefault(str(ntb["node_uid"][i]), []).append(row_adv[i])
        for u, vals in by_node.items():
            check(np.allclose(vals, vals[0]), f"节点 {u[:6]} 各行优势相同 ({vals[0]:+.3f})")
        root_v = [row_adv[i] for i in rows if int(ntb["node_depth"][i]) == 0][0]
        if kinds[t] == "C":
            check(root_v < 0, f"失败树 root 优势为负 ({root_v:+.3f})")
        else:
            check(root_v > 0, f"成功树({kinds[t]})root 优势为正 ({root_v:+.3f})")
        if kinds[t] == "A":
            child_v = [row_adv[i] for i in rows if int(ntb["node_depth"][i]) == 1][0]
            check(child_v > 0, f"委托树子节点优势为正 ({child_v:+.3f})")

    # ---------------------------------------------------------------- C5
    print("C5 诊断量")
    st = data.meta_info.get("rao_stats", {})
    check(st.get("rao/trees") == n_slots and st.get("rao/delegating_trees") == 2 and st.get("rao/max_depth") == 1,
          f"rao_stats: trees={st.get('rao/trees')} delegating={st.get('rao/delegating_trees')} max_depth={st.get('rao/max_depth')}")
    check("rao/depth_weight_d0" in st and "rao/depth_weight_d1" in st, "深度权重进诊断量")

    print(f"\n=== 步骤 5 CPU dry-run 全部通过({passed} 项断言)===")


if __name__ == "__main__":
    main()
