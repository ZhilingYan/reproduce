# -*- coding: utf-8 -*-
"""全量测评逐题指标的 CPU 测试。

覆盖:
  E1 flat 轨迹 → 原始逐题记录(有效步、token、被拒动作、余数批里未激活的槽被跳过)
  E2 递归轨迹 → 原始逐题记录(root 步数、委托轮、节点数与深度)
  E3 F1 口径(HotpotQA 官方:yes/no 特判、多金答案取最大)
  E4 四个 benchmark:原始记录 → 交付逐题行 → 汇总
  E5 成功取 won(与现有测评同口径),不取 root 成败
  E6 逐题落盘分批追加;重复记录只保留最后一条
  E7 报表:按模型分节、方法对照表、事实摘要
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts_rso"))

from agent_system.multi_turn_rollout.val_case_dump import build_case_records, dump_val_cases  # noqa: E402
import compute_eval_metrics as M  # noqa: E402
import make_eval_report as R  # noqa: E402


class FakeTok:
    def decode(self, ids, skip_special_tokens=True):
        return "reasoning ... <answer>Paris</answer>"


def _row(active, prompt_len, gen_len, resp_width=8, prompt_width=16, **kw):
    am = torch.zeros(prompt_width + resp_width, dtype=torch.long)
    am[prompt_width - prompt_len:prompt_width] = 1            # 左 pad 的 prompt
    am[prompt_width:prompt_width + gen_len] = 1               # 右 pad 的 response
    r = {"active_masks": active, "attention_mask": am,
         "responses": torch.zeros(resp_width, dtype=torch.long), "is_action_valid": True}
    r.update(kw)
    return r


def test_E1_flat_records():
    rows = [[_row(True, 10, 5), _row(True, 12, 3, is_action_valid=False), _row(False, 0, 0)],
            [_row(False, 0, 0)]]                               # 第 2 个槽从未激活
    infos = [[{"won": False}, {"won": True, "extra.action_failed": True, "extra.goal_item": "stick",
                               "extra.observation": "x" * 500}, {}], [{}]]
    recs = build_case_records(rows, infos, np.array([0.0, 0.0]), np.array([2, 0]), np.array([0.0, 0.0]),
                              env_kwargs=np.array([{"question": "q"}, {}], dtype=object),
                              data_sources=np.array(["nq", "nq"], dtype=object), tokenizer=FakeTok())
    assert len(recs) == 1
    r = recs[0]
    assert r["won"] is True and r["n_steps"] == 2
    assert r["gen_tokens"] == 8 and r["max_prompt_tokens"] == 12
    assert r["n_format_invalid"] == 1 and r["n_action_failed"] == 1
    assert r["extra"] == {"extra.action_failed": True, "extra.goal_item": "stick"}   # 长字段被丢弃
    assert "<answer>Paris</answer>" in r["last_response_tail"]
    assert "rec" not in r


def test_E2_recursive_records():
    rows = [[_row(True, 10, 4, node_depth=0, is_delegation_turn=True),
             _row(True, 20, 4, node_depth=1, is_delegation_turn=False),
             _row(True, 11, 4, node_depth=0, is_delegation_turn=False)]]
    infos = [[{}, {}, {"won": True}]]
    nodes = [[SimpleNamespace(depth=1, success=1.0, close_reason="goal_met"),
              SimpleNamespace(depth=0, success=1.0, close_reason="episode_end")]]
    r = build_case_records(rows, infos, np.array([0.0]), np.array([3]), np.array([0.0]),
                           node_records=nodes)[0]["rec"]
    assert r == {"total_turns": 3, "root_turns": 2, "delegation_turns": 1, "env_turns": 2,
                 "n_nodes": 2, "max_depth": 1, "root_success": 1.0, "root_close_reason": "episode_end"}


def test_E3_f1():
    assert M.f1_single("the Eiffel Tower", "Eiffel Tower") == 1.0
    assert M.f1_single("yes", "no") == 0.0
    assert M.f1_single("yes indeed", "yes") == 0.0              # yes/no 特判:不一致即 0
    assert abs(M.f1_single("Barack Obama Jr", "Barack Obama") - 0.8) < 1e-9
    assert M.f1_max("Obama", ["Barack Obama", "Obama"]) == 1.0
    assert M.f1_max(None, ["x"]) == 0.0


def _write(tmp_path, name, recs):
    d = tmp_path / name
    d.mkdir()
    with open(d / "cases.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return str(d)


def test_E4_sciworld(tmp_path):
    def rec(task, var, score):
        return {"extra": {"extra.task": task, "extra.variation": var, "extra.score": score},
                "reward": max(score, 0) / 100.0,               # 与环境一致:奖励 = 负分裁零后的分数 / 100
                "won": score >= 100, "n_steps": 5, "gen_tokens": 50, "max_prompt_tokens": 900}
    g = _write(tmp_path, "g", [rec("boil", 0, 100), rec("boil", 1, -100), rec("melt", 0, 50)])
    rows, _ = M.build_rows("sciworld", g, None, "qwen3-4b_sciworld_ood_grpo")
    assert [r["score"] for r in rows] == [100.0, 0.0, 50.0] and rows[0]["case_id"] == "boil:0"
    s = M.summarize("sciworld", rows)
    assert s["avg_score_micro"] == pytest.approx(50.0) and s["avg_score_macro"] == pytest.approx(50.0)
    assert s["success_rate"] == pytest.approx(1 / 3) and s["score_by_task"] == {"boil": 50.0, "melt": 50.0}


def test_E4_searchqa(tmp_path):
    def rec(src, q, gold, ans, won, calls, deleg=None):
        r = {"data_source": src, "env_kwargs": {"question": q, "ground_truth": {"target": [gold]}},
             "won": won, "n_steps": 3, "n_tool_calls": calls, "last_response_tail": f"... <answer>{ans}</answer>"}
        if deleg is not None:
            r["rec"] = {"total_turns": 3, "root_turns": 2, "delegation_turns": deleg, "env_turns": 3 - deleg,
                        "n_nodes": 1 + deleg, "max_depth": 1 if deleg else 0, "root_success": 1.0,
                        "root_close_reason": "x"}
        return r
    g = _write(tmp_path, "g", [rec("musique", "Q1?", "Paris", "Paris", True, 2, 1),
                               rec("musique", "Q2?", "Barack Obama", "Obama", False, 4, 0),
                               rec("hotpotqa", "Q3?", "yes", "no", False, 1, 1)])
    meta = tmp_path / "meta.jsonl"
    meta.write_text("\n".join(json.dumps(d) for d in [
        {"data_source": "musique", "qkey": M.qkey("Q1?"), "index": 7, "hops": 2},
        {"data_source": "musique", "qkey": M.qkey("Q2?"), "index": 8, "hops": 3},
        {"data_source": "hotpotqa", "qkey": M.qkey("Q3?"), "index": 9, "type": "comparison"}]))
    rows, _ = M.build_rows("searchqa", g, str(meta), "qwen3-4b_searchqa_rso")
    assert rows[0]["case_id"] == "musique:7" and rows[0]["answer"] == "Paris"
    assert [r["subset"] for r in rows] == ["test"] * 3
    s = M.summarize("searchqa", rows)
    mu = s["by_source"]["musique"]
    assert mu["EM"] == 0.5 and mu["SubEM"] == 0.5 and mu["F1"] == pytest.approx((1 + 2 / 3) / 2)
    assert mu["avg_searches"] == 3.0 and mu["EM_by_hops"] == {"2": 1.0, "3": 0.0}
    assert s["by_source"]["hotpotqa"]["EM_by_type"] == {"comparison": 0.0}
    assert s["delegation_rate"] == pytest.approx(2 / 3) and "singlehop_delegation_rate" not in s
    assert s["sanity_missing_meta"] == 0


def test_E4_textcraft_ood(tmp_path):
    def rec(item, depth, ok, root_t, tot_t, deleg, failed):
        return {"extra": {"extra.goal_item": item, "extra.goal_depth": depth}, "won": ok,
                "n_action_failed": failed, "n_format_invalid": 0,
                "rec": {"total_turns": tot_t, "root_turns": root_t, "delegation_turns": deleg,
                        "env_turns": tot_t - deleg, "n_nodes": 1 + deleg, "max_depth": 1 if deleg else 0,
                        "root_success": 1.0 if ok else 0.0, "root_close_reason": "x"}}
    g = _write(tmp_path, "g", [rec("stick", 2, True, 4, 6, 1, 1), rec("sign", 3, False, 10, 10, 0, 5)])
    s = M.summarize("textcraft_ood", M.build_rows("textcraft_ood", g, None, "x_textcraft_ood_rso")[0])
    assert s["success_rate"] == 0.5
    assert s["avg_steps_all"] == 8.0 and s["avg_steps_success"] == 6.0
    assert s["avg_root_steps_all"] == 7.0 and s["avg_root_steps_success"] == 4.0
    assert s["invalid_action_rate"] == pytest.approx(6 / 15)    # (1+5)/((6-1)+10)
    assert s["delegation_rate"] == 0.5 and s["avg_delegations"] == 0.5
    assert s["success_by_depth"] == {"2": 1.0, "3": 0.0}


def test_E4_textcraft_synth(tmp_path):
    p = tmp_path / "x_cases.jsonl"
    rows = [{"task_id": "t1", "difficulty": "easy", "success": True, "turns_used": 5, "output_tokens": 100,
             "trajectory": [{"input_tokens": 300}, {"input_tokens": 700}]},
            {"task_id": "t2", "difficulty": "hard", "success": False, "turns_used": 9, "output_tokens": 200,
             "trajectory": [{"input_tokens": 400}]}]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    out = M.build_rows("textcraft_synth", str(p), None, "qwen3-4b_textcraft_grpo")[0]
    s = M.summarize("textcraft_synth", out)
    assert s["success_by_difficulty"] == {"easy": 1.0, "hard": 0.0}
    assert s["avg_steps_all"] == 7.0 and s["avg_max_node_ctx"] == 550.0


def test_E5_success_uses_won_like_existing_eval(tmp_path):
    r = {"extra": {"extra.goal_item": "stick"}, "won": True, "n_action_failed": 0, "n_format_invalid": 0,
         "rec": {"total_turns": 3, "root_turns": 3, "delegation_turns": 0, "env_turns": 3, "n_nodes": 1,
                 "max_depth": 0, "root_success": 0.0, "root_close_reason": "x"}}
    rows, _ = M.build_rows("textcraft_ood", _write(tmp_path, "g", [r]), None, "x_textcraft_ood_rso")
    assert M.summarize("textcraft_ood", rows)["success_rate"] == 1.0


def test_E6_dump_appends_and_dedup(tmp_path):
    rows = [[_row(True, 10, 5)]]
    kw = dict(total_batch_list=rows, total_infos=[[{"won": True, "extra.goal_item": "stick"}]],
              episode_rewards=np.array([1.0]), episode_lengths=np.array([1]), tool_callings=np.array([0.0]))
    dump_val_cases(str(tmp_path), **kw)
    dump_val_cases(str(tmp_path), **kw)                          # 验证分多批调用 → 追加
    assert len((tmp_path / "cases.jsonl").read_text().strip().splitlines()) == 2
    out, dup = M.build_rows("textcraft_ood", str(tmp_path), None, "x_textcraft_ood_grpo")
    assert len(out) == 1 and dup == 1


def test_E7_report(tmp_path):
    cases = tmp_path / "cases"
    cases.mkdir()
    for method, ok in (("rso", [1, 1, 0]), ("grpo", [1, 0, 0])):
        name = f"qwen3-4b_textcraft_ood_{method}"
        with open(cases / f"{name}_cases.jsonl", "w") as f:
            for i, s in enumerate(ok):
                f.write(json.dumps({"name": name, "bench": "textcraft_ood", "subset": "test", "case_id": f"i{i}",
                                    "success": s, "steps_total": 5, "steps_root": None, "gen_tokens": 10,
                                    "max_node_ctx": 100, "is_recursive": method == "rso",
                                    "delegated": method == "rso", "n_delegations": 1 if method == "rso" else 0,
                                    "n_nodes": 2 if method == "rso" else None, "tree_depth": 1 if method == "rso" else None,
                                    "goal_depth": 2, "env_actions": 5, "invalid_actions": 1,
                                    "format_invalid_actions": 0}) + "\n")
    assert R.parse_name("gemma-26b_sciworld_ood_rso_opsd") == ("gemma-26b", "sciworld_ood", "rso_opsd")
    out = R.build_reports(str(cases), str(tmp_path / "rep"))
    md = open(out[0]).read()
    assert out[0].endswith("textcraft_ood_report.md")
    assert "## qwen3-4b" in md and "| rso | 3 | 66.7 |" in md and "| grpo | 3 | 33.3 |" in md
    assert "差 +33.3 个百分点" in md and "缺失的方法" in md and "分析(执行者补写)" in md
