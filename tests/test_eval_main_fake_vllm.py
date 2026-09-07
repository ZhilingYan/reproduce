# -*- coding: utf-8 -*-
"""用假 vllm 模块把 scripts_rso/eval_full_val.py 的 main() 端到端跑一遍(flat 与 --recursive 两条分支),
验证命令行解析、vLLM 返回值的 token 计数取法(prompt_token_ids / outputs[0].token_ids)、落盘与汇总。
不碰 GPU。运行:
    source env_rao.sh && cd "$SDAR_REPO" && HF_HUB_OFFLINE=1 python tests/test_eval_main_fake_vllm.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class _Out:
    def __init__(self, text, n_out):
        self.text = text
        self.token_ids = list(range(n_out))


class _Req:
    def __init__(self, text, n_in, n_out):
        self.prompt_token_ids = list(range(n_in))
        self.outputs = [_Out(text, n_out)]


class FakeLLM:
    """每次生成都回一个固定动作;token 数用 prompt 长度/50 与固定 7 造出来,只为验证计数管道。"""
    calls = 0

    def __init__(self, *a, **k):
        pass

    def generate(self, prompts, sp, use_tqdm=False):
        FakeLLM.calls += 1
        return [_Req("<thought>ok</thought><action>inventory</action>", max(1, len(p) // 50), 7) for p in prompts]


class FakeSP:
    def __init__(self, *a, **k):
        pass


fake_vllm = types.ModuleType("vllm")
fake_vllm.LLM = FakeLLM
fake_vllm.SamplingParams = FakeSP
sys.modules["vllm"] = fake_vllm

spec = importlib.util.spec_from_file_location("ev", os.path.join(ROOT, "scripts_rso", "eval_full_val.py"))
ev = importlib.util.module_from_spec(spec)
sys.argv = ["x"]
spec.loader.exec_module(ev)

passed = 0
def check(cond, msg):
    global passed
    assert cond, msg
    passed += 1
    print(f"  ✓ {msg}")


def run(extra):
    tmp = tempfile.mkdtemp(prefix="rao_evalmain_")
    out = os.path.join(tmp, "e")
    sys.argv = ["eval_full_val.py", "--model", "Qwen/Qwen3-4B-Instruct-2507", "--out", out,
                "--split", "train", "--difficulties", "easy", "--limit", "3", "--max-steps", "4",
                "--batch-size", "2", "--temperature", "0.0"] + extra
    ev.main()
    recs = [json.loads(l) for l in open(out + "_cases.jsonl")]
    m = json.load(open(out + "_metrics.json"))
    return recs, m, out


print("F1 flat 分支(main 端到端,假 vLLM)")
recs, m, out = run([])
check(len(recs) == 3 and m["completed"] and m["n_tasks"] == 3, "3 题落盘,metrics 标记完成")
r = recs[0]
check(r["turns_used"] == 4 and r["input_tokens"] == sum(t["input_tokens"] for t in r["trajectory"])
      and r["output_tokens"] == 4 * 7 and r["total_tokens"] == r["input_tokens"] + r["output_tokens"],
      f"token 从 vLLM 返回值逐轮累加(in={r['input_tokens']}, out={r['output_tokens']})")
check(m["tokens"]["overall"]["n"] == 3 and m["tree"] is None and m["recursive"] is False,
      "flat:tokens 段有、tree 段为 None、recursive=False")

print("F2 --recursive 分支(main 端到端,假 vLLM)")
recs, m, out = run(["--recursive", "--per-agent-steps", "3", "--max-depth", "2"])
check(len(recs) == 3 and all(r["tree"]["n_nodes"] == 1 for r in recs), "3 题落盘,假策略不委托 → 单节点树")
check(all(r["tree"]["root_close_reason"] == "budget_exhausted" and r["turns_used"] == 3 for r in recs),
      "每 agent 3 步预算耗尽(全局上限 4 未触发)")
check(m["recursive"] and m["per_agent_steps"] == 3 and m["max_depth"] == 2 and m["tree"]["n"] == 3
      and m["tree"]["delegating_rate"] == 0.0, "metrics:递归标记与树统计")
check(all(r["tokens_by_depth"] == {"0": [r["input_tokens"], r["output_tokens"]]} for r in recs),
      "tokens_by_depth 只有 d0 且等于总量")
check(os.path.isdir(out + "_tree_trace") and len(os.listdir(os.path.join(out + "_tree_trace", "eval"))) == 2,
      "tree_trace 默认目录 <out>_tree_trace,2 批 → 2 个文件")

print("F3 --recursive --resume 续跑")
prev = out
sys.argv = ["eval_full_val.py", "--model", "Qwen/Qwen3-4B-Instruct-2507", "--out", prev,
            "--split", "train", "--difficulties", "easy", "--limit", "3", "--max-steps", "4",
            "--batch-size", "2", "--recursive", "--per-agent-steps", "3", "--resume"]
calls_before = FakeLLM.calls
ev.main()
m2 = json.load(open(prev + "_metrics.json"))
check(FakeLLM.calls == calls_before and m2["n_tasks"] == 3 and m2["tree"]["n"] == 3,
      "全部已完成 → 不再生成;汇总仍含 3 题的树统计")
print(f"\n=== eval main 假 vLLM 端到端测试全部通过({passed} 项断言)===")
