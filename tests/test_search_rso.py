# -*- coding: utf-8 -*-
"""search_rso 包的 CPU 测试:投影、tracker、适配器、编排器整链、act_mask 多标签。

运行:cd RSO_search && python -m pytest tests/test_search_rso.py -q
不依赖 GPU / Ray / 检索服务(检索用 monkeypatch 假返回)。
"""
import json
import os
import sys
import tempfile

import numpy as np
try:
    import pytest
except ImportError:                                   # sdar 环境无 pytest:走文件底部的手动 runner
    pytest = None
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_system.environments.env_package.search_rso.projection import search_rso_projection
from agent_system.environments.env_package.search_rso.decomp import (
    SearchDecompTracker, norm_text)


# ---------------------------------------------------------------- projection
def test_projection_basic():
    acts = [
        "<think>x</think><search> spouse of Steve Hillage </search>",
        "<think>x</think><answer>no</answer>",
        "<delegate> Who directed Move (1970 film)? </delegate>",
        "<search>a</search><answer>b</answer>",          # 混两种 → invalid
        "<search>a</search><search>b</search>",           # 同种两个 → invalid
        "no tags at all",
    ]
    res, val = search_rso_projection(acts)
    assert res[0] == "<search>spouse of Steve Hillage</search>" and val[0] == 1
    assert res[1] == "<answer>no</answer>" and val[1] == 1
    assert res[2] == "<delegate>Who directed Move (1970 film)?</delegate>" and val[2] == 1
    assert val[3] == 0 and val[4] == 0
    assert res[5] == "" and val[5] == 0


# ---------------------------------------------------------------- tracker
HOPS = [
    {"q": "Move (1970 film) >> director", "ans": "Stuart Rosenberg",
     "aliases": [norm_text("Stuart Rosenberg")], "compare": False},
    {"q": "#1 >> country of citizenship", "ans": "American",
     "aliases": [norm_text("American"), norm_text("United States of America")], "compare": False},
    {"q": "compare(#1, #2)", "ans": "no", "aliases": [norm_text("no")], "compare": True},
]


def test_tracker_channels_and_phi():
    tr = SearchDecompTracker(HOPS)
    assert tr.n_hops == 2 and tr.phi == 2.0            # compare 尾步不进分母
    # (d) 查询承诺:query 含 hop1 答案
    d1 = tr.resolve_events("Stuart Rosenberg nationality", via="search", node_uid="n1")
    assert d1 == 1.0 and tr.phi == 1.0
    # 重复不再记
    assert tr.resolve_events("stuart rosenberg", via="search", node_uid="n1") == 0.0
    # 词边界:"americana" 不命中 "american"
    assert tr.resolve_events("americana music", via="search", node_uid="n1") == 0.0
    # (c) 别名命中
    d2 = tr.resolve_events("He was American.", via="answer", node_uid="n2")
    assert d2 == 1.0 and tr.phi == 0.0
    assert not tr.unresolved_hops()
    assert {r["node_uid"] for r in tr.resolved_by_node("n1")} == {"n1"}


def test_tracker_empty_for_ood():
    tr = SearchDecompTracker(None)
    assert tr.n_hops == 0 and tr.phi == 0.0
    assert tr.resolve_events("anything", via="search", node_uid="x") == 0.0


# ---------------------------------------------------------------- 整链(编排器)
def _mk_config(decomp_path):
    return OmegaConf.create({
        "env": {
            "env_name": "search_rso",
            "seed": 0,
            "history_length": 4,
            "rollout": {"n": 1},
            "max_steps": 50,
            "rao": {"per_agent_max_steps": 25, "max_depth": 3},
            "search_rso": {"decomp_path": decomp_path},
            "search": {"search_url": "http://127.0.0.1:9", "topk": 3, "timeout": 1},
        },
        "data": {"train_batch_size": 1, "val_batch_size": 1},
    })


QUESTION = ("Are director of film Move (1970 Film) and director of film "
            "Méditerranée (1963 Film) from the same country?")


def _make_store_path(dirname):
    p = os.path.join(dirname, "store.json")
    with open(p, "w") as f:
        json.dump({norm_text(QUESTION): {"targets": ["no"], "hops": HOPS}}, f)
    return p


if pytest is not None:
    @pytest.fixture()
    def store_path(tmp_path):
        return _make_store_path(str(tmp_path))


def _fake_retrieval(monkeypatch, text):
    import agent_system.environments.env_package.search_rso.envs as envs_mod
    monkeypatch.setattr(envs_mod, "call_search_api",
                        lambda *a, **k: ({"result": [[{"document": {"contents": text}}]]}, None))


def test_orchestrator_end_to_end(monkeypatch, store_path):
    from agent_system.environments.env_package.search_rso.envs import SearchRsoEnvs
    from agent_system.environments.env_package.search_rso.opsd_factory import (
        SearchRecursiveEnvironmentManager)
    from agent_system.environments.env_package.search_rso.opsd_priv import SearchOPSDAdapter
    from agent_system.recursive.budget import PerAgentBudget

    config = _mk_config(store_path)
    _fake_retrieval(monkeypatch, '"Move (1970 film)"\nMove is a 1970 American comedy film ...')
    envs = SearchRsoEnvs(seed=0, env_num=1, group_n=1, is_train=True, env_config=config.env)
    adapter = SearchOPSDAdapter(config)
    mgr = SearchRecursiveEnvironmentManager(
        envs, search_rso_projection, config, adapter=adapter,
        budget=PerAgentBudget.from_config(dict(config.env.rao)), trace_tag="test")

    kwargs = [{"question": QUESTION, "ground_truth": {"target": ["no"]}, "data_source": "2wikimultihopqa"}]
    obs, infos = mgr.reset(kwargs)
    assert "Your question:" in obs["text"][0]
    assert infos[0]["phi_after"] == 2.0

    # 轮 1:root 查询,含 hop1 答案 →(d)ΔΦ=1
    obs, r, d, infos = mgr.step(["<think>t</think><search>Stuart Rosenberg country</search>"])
    assert mgr.turn_meta[-1][0]["delta_phi"] == 1.0
    assert "<information>" in obs["text"][0]

    # 轮 2:root 委托,子任务文本含 hop2 答案的别名 →(d)ΔΦ=1,useless_goal=False
    obs, r, d, infos = mgr.step(["<delegate>Is Stuart Rosenberg American?</delegate>"])
    assert mgr.turn_meta[-1][0]["delta_phi"] == 1.0
    assert mgr.turn_meta[-1][0]["is_delegation_turn"] is True
    assert "sub-question" in obs["text"][0]            # 子节点的 prompt
    assert len(mgr.stacks[0]) == 2

    # 轮 3:子作答(命中已解决跳,ΔΦ=0 但子 success=1,弹栈回父,战报含答案)
    obs, r, d, infos = mgr.step(["<answer>yes, American</answer>"])
    assert mgr.turn_meta[-1][0]["delta_phi"] == 0.0
    assert len(mgr.stacks[0]) == 1
    assert "Sub-agent report" in obs["text"][0] or "American" in obs["text"][0]
    child_rec = mgr.finished_records[0][-1]
    assert child_rec.depth == 1 and child_rec.success == 1.0
    assert child_rec.useless_goal is False

    # priv:行开局现算,含未解决清单为空(两个 retrieval 跳已解决)
    priv = mgr._acting_priv[0]
    assert "Privileged decomposition" in priv and "all hops are resolved" in priv

    # 轮 4:root 终答 → 整局结束,EM 判对,won=True,episode reward 补 1
    obs, r, d, infos = mgr.step(["<answer>no</answer>"])
    assert bool(d[0]) is True
    assert infos[0]["won"] is True
    assert float(np.asarray(r)[0]) == 1.0
    root_rec = [x for x in mgr.finished_records[0] if x.depth == 0][-1]
    assert root_rec.success == 1.0

    recs = mgr.collect_node_records()
    assert len(recs[0]) == 2                            # 子 + root


def test_useless_delegate_and_child_fail(monkeypatch, store_path):
    from agent_system.environments.env_package.search_rso.envs import SearchRsoEnvs
    from agent_system.environments.env_package.search_rso.opsd_factory import (
        SearchRecursiveEnvironmentManager)
    from agent_system.environments.env_package.search_rso.opsd_priv import SearchOPSDAdapter
    from agent_system.recursive.budget import PerAgentBudget

    config = _mk_config(store_path)
    _fake_retrieval(monkeypatch, "irrelevant text")
    envs = SearchRsoEnvs(seed=0, env_num=1, group_n=1, is_train=True, env_config=config.env)
    adapter = SearchOPSDAdapter(config)
    mgr = SearchRecursiveEnvironmentManager(
        envs, search_rso_projection, config, adapter=adapter,
        budget=PerAgentBudget.from_config(dict(config.env.rao)), trace_tag="test")
    mgr.reset([{"question": QUESTION, "ground_truth": {"target": ["no"]},
                "data_source": "2wikimultihopqa"}])

    # 清单外委托:子任务文本不含任何跳答案 → useless_goal=True,ΔΦ=0
    mgr.step(["<delegate>What is the capital of France?</delegate>"])
    assert mgr.turn_meta[-1][0]["delta_phi"] == 0.0
    # 子答非所问 → success=0
    mgr.step(["<answer>Paris</answer>"])
    child = mgr.finished_records[0][-1]
    assert child.depth == 1 and child.success == 0.0 and child.useless_goal is True

    # root 答错 → EM=0,won=False
    obs, r, d, infos = mgr.step(["<answer>yes</answer>"])
    assert infos[0]["won"] is False
    root_rec = [x for x in mgr.finished_records[0] if x.depth == 0][-1]
    assert root_rec.success == 0.0


# ---------------------------------------------------------------- act_mask 多标签
def test_act_mask_multi_tags():
    import torch
    from verl.trainer.ppo.rso_opsd_core import build_action_token_mask

    class ByteTok:
        """一 token = 一字节的假 tokenizer,走 convert_ids_to_tokens 字面路径。"""
        def convert_ids_to_tokens(self, ids):
            return [chr(i) for i in ids]

    text = "<think>x</think><search>abc</search> tail"
    ids = [ord(c) for c in text]
    responses = torch.tensor([ids])
    mask = torch.ones_like(responses)
    am, metrics = build_action_token_mask(responses, mask, ByteTok(),
                                          act_tags=["search", "answer", "delegate"])
    covered = "".join(chr(i) for i, m in zip(ids, am[0].tolist()) if m > 0)
    assert covered == "abc"
    assert metrics["rso/act_rows_no_action_ratio"] == 0.0

    # 缺省标签集回退 <action>(textcraft 行为不变)
    text2 = "<action>craft 1 x</action>"
    ids2 = [ord(c) for c in text2]
    am2, _ = build_action_token_mask(torch.tensor([ids2]), torch.ones(1, len(ids2), dtype=torch.long),
                                     ByteTok())
    covered2 = "".join(chr(i) for i, m in zip(ids2, am2[0].tolist()) if m > 0)
    assert covered2 == "craft 1 x"


# ---------------------------------------------------------------- 免 pytest 的手动 runner
class _Monkey:
    def __init__(self):
        self._undo = []

    def setattr(self, obj, name, value):
        self._undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def restore(self):
        for obj, name, old in reversed(self._undo):
            setattr(obj, name, old)
        self._undo = []


if __name__ == "__main__":
    failures = 0
    with tempfile.TemporaryDirectory() as td:
        sp = _make_store_path(td)
        for fn, needs in [(test_projection_basic, ()),
                          (test_tracker_channels_and_phi, ()),
                          (test_tracker_empty_for_ood, ()),
                          (test_orchestrator_end_to_end, ("monkey", "store")),
                          (test_useless_delegate_and_child_fail, ("monkey", "store")),
                          (test_act_mask_multi_tags, ())]:
            mk = _Monkey()
            try:
                if needs:
                    fn(mk, sp)
                else:
                    fn()
                print(f"PASS {fn.__name__}")
            except Exception as e:
                failures += 1
                import traceback
                print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
                traceback.print_exc()
            finally:
                mk.restore()
    print("failures:", failures)
    sys.exit(1 if failures else 0)
