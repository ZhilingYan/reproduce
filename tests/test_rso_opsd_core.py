# -*- coding: utf-8 -*-
"""RSO+OPSD 损失内核的 CPU 单元测试:居中门(§2c)/ 梯度 / act_mask(§2b)/ teacher 批。

运行(不需要 GPU):PYTHONPATH=$PWD python tests/test_rso_opsd_core.py
act_mask 的 tokenizer 用例用 HF 缓存里的 Qwen3-4B fast tokenizer(离线);缓存缺失时
跳过并显式打印(不算通过)。
"""
from __future__ import annotations

import math
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verl.trainer.ppo.rso_opsd_core import (                                   # noqa: E402
    build_action_token_mask, compute_rso_opsd_loss)

passed = 0


def check(cond, msg):
    global passed
    assert cond, f"✗ {msg}"
    passed += 1
    print(f"  ✓ {msg}")


def _load_qwen_tokenizer():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
        return tok if getattr(tok, "is_fast", False) else None
    except Exception as e:
        print(f"  [skip] Qwen tokenizer 不可用({type(e).__name__}: {e}),act_mask 分词用例跳过")
        return None


# ---------------------------------------------------------------------------- C1
def test_C1_gate_properties():
    print("C1 居中门四条性质(§2c:g≥0 / g(0)=0 / 单调 / 有界;与 sigmoid 门的换算式)")
    beta = 2.5
    deltas = torch.tensor([[-1.0, -0.1, 0.0, 0.1, 0.5, 3.0]])
    student = torch.zeros_like(deltas, requires_grad=True)
    teacher = deltas.clone()                       # δ = teacher − student = deltas
    mask = torch.ones_like(deltas)
    _, m = compute_rso_opsd_loss(student, teacher, mask, mask, gate_beta=beta)
    g = torch.clamp(torch.tanh(beta * deltas), min=0.0)
    check(float(g[0, 0]) == 0.0 and float(g[0, 1]) == 0.0, "δ<0 → 门全关(只推不拉,负侧静音)")
    check(float(g[0, 2]) == 0.0, "g(0)=0:δ 收敛即静默,无 0.5 静息电平(与 sigmoid 门的唯一差别)")
    check(float(g[0, 3]) < float(g[0, 4]) < float(g[0, 5]) <= 1.0, "正侧单调、有界 ≤1")
    # 换算式 max(tanh(βδ),0) = max(2σ(2βδ)−1, 0):tanh 的 β=2.5 ↔ sigmoid 的 β=5
    sig = torch.clamp(2 * torch.sigmoid(2 * beta * deltas) - 1, min=0.0)
    check(torch.allclose(g, sig, atol=1e-6), "tanh(βδ)=2σ(2βδ)−1 换算式数值成立(β 2.5↔5)")
    check(0 < m["rso/gate_active_ratio"] < 1, "gate_active_ratio = δ>0 占比(此处 3/6)")
    check(abs(m["rso/gate_active_ratio"] - 0.5) < 1e-6, "δ>0 恰好一半 → 0.5(重定义口径,非 g>0.5)")


# ---------------------------------------------------------------------------- C2
def test_C2_gradient_is_minus_gate():
    print("C2 梯度分析:∂L/∂student = −g·m/Σm(teacher/门都 detach,student 是唯一梯度通道)")
    torch.manual_seed(0)
    student = torch.randn(2, 5, requires_grad=True)
    teacher = torch.randn(2, 5)
    resp_mask = torch.ones(2, 5)
    act_mask = torch.tensor([[1., 1., 0., 0., 1.], [0., 1., 1., 0., 0.]])
    loss, _ = compute_rso_opsd_loss(student, teacher, resp_mask, act_mask, gate_beta=2.5,
                                    loss_agg_mode="token-mean")
    loss.backward()
    with torch.no_grad():
        delta = teacher - student.detach()
        gate = torch.clamp(torch.tanh(2.5 * delta), min=0.0)
        mask = resp_mask * act_mask
        expected = -gate * mask / (mask.sum() + 1e-8)     # token-mean = masked_mean(含 1e-8)
    check(torch.allclose(student.grad, expected, atol=1e-6),
          "逐 token 梯度 = −门·掩码/掩码和(设计 §二:−λ·m·g,λ 在 dp_actor 外乘)")
    check(float(student.grad[0, 2]) == 0.0 and float(student.grad[1, 0]) == 0.0,
          "act_mask 外的 token 零梯度(蒸馏只作用在 <action> 内容上)")


# ---------------------------------------------------------------------------- C3
def test_C3_loss_value_and_metrics():
    print("C3 数值:loss = Σ g·δ·m / Σm;指标在 act token 上统计")
    student = torch.zeros(1, 4, requires_grad=True)
    teacher = torch.tensor([[0.4, -0.2, 0.8, 0.6]])
    resp_mask = torch.ones(1, 4)
    act_mask = torch.tensor([[1., 1., 1., 0.]])           # 最后一个 token 不在动作块
    loss, m = compute_rso_opsd_loss(student, teacher, resp_mask, act_mask, gate_beta=2.5)
    g = [max(math.tanh(2.5 * d), 0.0) for d in [0.4, -0.2, 0.8]]
    expect = (g[0] * 0.4 + g[1] * (-0.2) + g[2] * 0.8) / (3 + 1e-8)
    check(abs(float(loss) - expect) < 1e-6, "loss 数值逐项对上(δ<0 项被门归零)")
    check(abs(m["rso/gate_active_ratio"] - 2 / 3) < 1e-6, "active_ratio=2/3(act token 里 δ>0 的)")
    check(abs(m["rso/act_token_ratio"] - 3 / 4) < 1e-6, "act_token_ratio=3/4")
    check(abs(m["rso/teacher_gap_mean"] - (0.4 - 0.2 + 0.8) / 3) < 1e-5, "gap_mean 在 act token 上")


# ---------------------------------------------------------------------------- C4
def test_C4_act_mask_with_real_tokenizer():
    tok = _load_qwen_tokenizer()
    if tok is None:
        return
    print("C4 act_mask(真 tokenizer):内容进、标签不进、截断行全空、多块只取首段")
    texts = [
        "<thought>need sticks first</thought>\n<action>craft 2 stick using 1 wood</action>",
        "<thought>hmm</thought> no action tag here at all",
        "<thought>x</thought>\n<action>get_info axe",                       # 截断:无闭合标签
        "<action>get_info axe</action> ok <action>craft 1 axe</action>",    # 多块
    ]
    enc = [tok.encode(t, add_special_tokens=False) for t in texts]
    L = max(len(e) for e in enc)
    responses = torch.zeros(len(enc), L, dtype=torch.long)
    resp_mask = torch.zeros(len(enc), L, dtype=torch.long)
    for i, e in enumerate(enc):
        responses[i, :len(e)] = torch.tensor(e)
        resp_mask[i, :len(e)] = 1
    mask, m = build_action_token_mask(responses, resp_mask, tok)

    row0 = tok.decode(responses[0][mask[0] > 0].tolist())
    check("craft 2 stick using 1 wood" in row0, "行 0:动作内容全部被掩码覆盖")
    check("<thought>" not in row0 and "thought" not in row0.replace("craft", ""),
          "行 0:<thought> 段一个 token 都不进")
    check(mask[1].sum() == 0 and m["rso/act_rows_no_action_ratio"] == 0.25, "行 1:无动作块 → 全空 + 计数")
    check(mask[2].sum() == 0 and m["rso/act_rows_truncated_ratio"] == 0.25, "行 2:截断 → 全空(安全性质)")
    row3 = tok.decode(responses[3][mask[3] > 0].tolist())
    check("get_info axe" in row3 and "craft 1 axe" not in row3,
          "行 3:多块只取首段(环境只执行首个动作)")
    check(m["rso/act_multi_block_ratio"] == 0.25, "多块计数 1/4")


def test_C4b_non_canonical_segmentation():
    tok = _load_qwen_tokenizer()
    if tok is None:
        return
    print("C4b 非规范切分(冒烟 21947735 的 85-93% 弃行事故):字节级定位与切分无关")
    text = "<thought>plan</thought>\n<action>craft 2 t1_i1_21 using 1 raw_t0</action>"
    canonical = tok.encode(text, add_special_tokens=False)
    # 人为造非规范切分:在动作内容中段逐个位置断开、分别编码再拼接(文本相同、token
    # 边界不同),扫到第一个与规范序列不同的切点。这正是采样序列的真实形态——
    # 模型逐 token 生成,不保证落在规范 BPE 边界上。
    alt = None
    for cut in range(text.find("craft"), len(text) - len("</action>")):
        cand = (tok.encode(text[:cut], add_special_tokens=False)
                + tok.encode(text[cut:], add_special_tokens=False))
        if cand != canonical and tok.decode(cand) == text:
            alt = cand
            break
    check(alt is not None, "构造出的切分确实非规范(token 序列不同)")
    check(tok.decode(alt) == tok.decode(canonical) == text, "两种切分 decode 出同一文本")
    L = max(len(canonical), len(alt))
    responses = torch.zeros(2, L, dtype=torch.long)
    resp_mask = torch.zeros(2, L, dtype=torch.long)
    for i, e in enumerate([canonical, alt]):
        responses[i, :len(e)] = torch.tensor(e)
        resp_mask[i, :len(e)] = 1
    mask, m = build_action_token_mask(responses, resp_mask, tok)
    for i, e in enumerate([canonical, alt]):
        got = tok.decode(responses[i][mask[i] > 0].tolist())
        check("craft 2 t1_i1_21 using 1 raw_t0" in got,
              f"行 {i}({'规范' if i == 0 else '非规范'}切分)动作内容均被完整覆盖,零弃行")
    check(mask[1].sum() > 0 and m["rso/act_rows_no_action_ratio"] == 0.0,
          "非规范切分不再触发任何弃行(旧 re-tokenize 路线在这里必然弃行)")
    # 附带核对特殊 token 不搅局:结尾挂 eos 也不影响定位
    with_eos = canonical + [tok.eos_token_id]
    r2 = torch.tensor([with_eos]); rm2 = torch.ones_like(r2)
    mask2, _ = build_action_token_mask(r2, rm2, tok)
    check("craft 2 t1_i1_21" in tok.decode(r2[0][mask2[0] > 0].tolist()),
          "带 eos 的行定位不受影响(特殊 token 走字面 utf-8 路径)")


# ---------------------------------------------------------------------------- C5
def test_C5_teacher_batch_priv_prefix_and_truncation():
    tok = _load_qwen_tokenizer()
    if tok is None:
        return
    print("C5 teacher 批:node_priv 逐行拼前缀、response token 原样、截断按难度分桶计数")
    from verl import DataProto
    from verl.trainer.ppo.rso_opsd_teacher import build_priv_teacher_batch

    from verl.trainer.ppo.rso_opsd_teacher import DEFAULT_PRIV_PREFIX
    prompts = ["ROOT PROMPT ALPHA", "CHILD PROMPT BETA"]
    privs = ["PLAN-ROOT " + "filler " * 80, "PLAN-CHILD"]   # 行 0 的 priv 很长,行 1 很短
    resp_ids = tok.encode("<action>inventory</action>", add_special_tokens=False)
    resp_len = len(resp_ids)
    # 上限取"行 1 的 teacher prompt 全长 + 8":行 1 恰好装得下,行 0(长 priv)必然截断
    _row1_len = len(tok.encode(DEFAULT_PRIV_PREFIX.format(priv=privs[1]) + prompts[1],
                               add_special_tokens=False))
    max_prompt_length = _row1_len + 8
    rows_ids, rows_mask = [], []
    for p in prompts:
        pids = tok.encode(p, add_special_tokens=False)
        pad = max_prompt_length - len(pids)
        rows_ids.append([tok.pad_token_id] * pad + pids + resp_ids)
        rows_mask.append([0] * pad + [1] * (len(pids) + resp_len))
    batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor(rows_ids),
            "attention_mask": torch.tensor(rows_mask),
            "position_ids": torch.arange(max_prompt_length + resp_len).unsqueeze(0).repeat(2, 1),
            "responses": torch.tensor(rows_ids)[:, -resp_len:],
        },
        non_tensors={
            "node_priv": np.array(privs, dtype=object),
            "task_difficulty": np.array(["hard", "easy"], dtype=object),
        },
    )
    tb, m = build_priv_teacher_batch(batch, tok, max_prompt_length=max_prompt_length)
    check(torch.equal(tb.batch["responses"], batch.batch["responses"]),
          "response token 原封不动(δ 打分同一对象的前提)")
    t1 = tok.decode(tb.batch["input_ids"][1][tb.batch["attention_mask"][1] > 0])
    check("PLAN-CHILD" in t1 and "CHILD PROMPT BETA" in t1 and
          t1.find("PLAN-CHILD") < t1.find("CHILD PROMPT BETA"),
          "行 1:priv 前缀拼在该行【自己的】prompt 之前")
    check(m["rso/priv_truncation_rate"] == 0.5, "行 0 超长被截断 → 总截断率 1/2(显式计数,不再静默)")
    check(m.get("rso/priv_truncation_rate_hard") == 1.0 and m.get("rso/priv_truncation_rate_easy") == 0.0,
          "截断按难度分桶:hard 1/1,easy 0/1(总体率会稀释,分桶才看得见)")
    check(m["rso/priv_miss_ratio"] == 0.0, "priv 无缺失")

    # 缺列 = 接线断了,必须响亮地炸,不许静默退化
    bad = DataProto.from_dict(tensors={k: v for k, v in batch.batch.items()})
    try:
        build_priv_teacher_batch(bad, tok, max_prompt_length=max_prompt_length)
        check(False, "缺 node_priv 列应当报错")
    except AssertionError as e:
        check("node_priv" in str(e), "缺 node_priv 列 → 显式 AssertionError(接线守卫)")


if __name__ == "__main__":
    for fn in [test_C1_gate_properties, test_C2_gradient_is_minus_gate,
               test_C3_loss_value_and_metrics, test_C4_act_mask_with_real_tokenizer,
               test_C4b_non_canonical_segmentation,
               test_C5_teacher_batch_priv_prefix_and_truncation]:
        fn()
    print(f"\n全部通过: {passed} 条断言")
