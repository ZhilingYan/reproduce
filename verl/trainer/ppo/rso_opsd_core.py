# -*- coding: utf-8 -*-
"""RSO+OPSD 的两个核心部件:居中门蒸馏损失(设计 §2c)与 act_mask 构造(设计 §2b)。

设计出处:Ideation/ideas/RSO_method_design.md §二;实现对照表:docs/RSO_OPSD_DESIGN.md §五/§六。
损失结构对照 sdar_utils.py:14-86 的 compute_sdar_loss——同样的"δ→门→门控 KL→聚合"骨架,
三处不同:门从 sigmoid(βδ) 换成居中门 max(tanh(βδ),0)(去掉 0.5 静息电平,零证据零推力);
掩码多乘一个 act_mask(蒸馏只作用在 <action> 内容 token 上);指标前缀 rso/*、
gate_active_ratio 重定义为 δ>0 占比(§2c 打点条款)。
独立函数、独立开关(dp_actor 的 use_rso_opsd_loss),不动 flat 基线在用的 compute_sdar_loss。
"""
from __future__ import annotations

from typing import Tuple

import torch

from verl.trainer.ppo.core_algos import agg_loss

# 环境动作块的定界标签(projection.py 抠取 <action> 的同一对文法)
_ACTION_OPEN = "<action>"
_ACTION_CLOSE = "</action>"


def compute_rso_opsd_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    act_mask: torch.Tensor,
    gate_beta: float = 2.5,
    loss_agg_mode: str = "token-mean",
) -> Tuple[torch.Tensor, dict]:
    """居中门蒸馏损失(RSO_method_design §2c,2026-09-10 定稿)。

    L_distill = agg[ m^act · g · (log π_T − log π_θ) ],
    g = max(tanh(β·δ), 0),δ = sg[log π_T − log π_θ]。
    门的四条性质(即设计要求):g≥0 只推不拉;g(0)=0 δ 收敛即静默(无残余恒推,
    与 sigmoid 门的 0.5 静息电平的唯一差别);δ>0 上单调平滑;有界 ≤1。
    梯度:g 与 teacher 均 detach 后 ∂L/∂(student_lp) = −g·m,teacher 信息只经门的
    逐 token 变化进入(设计 §二的梯度分析)。

    Args:
        student_log_probs: (bs, resp_len) 本次前向、带梯度。
        teacher_log_probs: (bs, resp_len) 特权前缀条件下的同 token log 概率,冻结。
        response_mask:     (bs, resp_len) 有效响应 token。
        act_mask:          (bs, resp_len) <action> 内容 token(build_action_token_mask 产出)。
        gate_beta:         门温度,§2c 初始值 2.5(sigmoid β=5 经 tanh(βδ)=2σ(2βδ)−1 换算)。
    Returns:
        (loss, metrics)。loss 数值可为负(δ 均值为负时),优化只看梯度方向,正常现象。
    """
    teacher_log_probs = teacher_log_probs.detach()

    delta_t = teacher_log_probs - student_log_probs.detach()

    gate = torch.clamp(torch.tanh(gate_beta * delta_t), min=0.0).detach()   # 居中门

    kl_per_token = teacher_log_probs - student_log_probs

    mask = (response_mask * act_mask).to(student_log_probs.dtype)

    gated_kl = gate * kl_per_token

    loss = agg_loss(loss_mat=gated_kl, loss_mask=mask, loss_agg_mode=loss_agg_mode)

    with torch.no_grad():
        mask_sum = mask.sum().clamp(min=1)
        resp_sum = response_mask.sum().clamp(min=1)
        gate_mean = (gate * mask).sum() / mask_sum
        # §2c 打点条款:gate_active_ratio 定义【改为】δ>0 占比(act token 上)
        gate_active = ((delta_t > 0).to(mask.dtype) * mask).sum() / mask_sum
        gap_mean = (delta_t * mask).sum() / mask_sum
        act_ratio = mask.sum() / resp_sum

    metrics = {
        "rso/gate_mean": gate_mean.item(),
        "rso/gate_active_ratio": gate_active.item(),
        "rso/teacher_gap_mean": gap_mean.item(),
        "rso/opsd_loss": loss.detach().item(),
        "rso/act_token_ratio": act_ratio.item(),
    }
    return loss, metrics


def _token_byte_pieces(tokenizer, ids):
    """把每个 token id 还原成它的字节串,拼接即该序列的响应字节流。

    为什么不 re-tokenize + offset_mapping(2026-09-10 冒烟 21947735 的教训):
    采样出的 token 序列【不必】是规范切分——合成物品名(t1_i1_21)、连续换行等有多种
    合法切分,decode→re-encode 的规范切分与原序列对不上的行占 85-93%,按往返一致性
    弃行等于把蒸馏整个关掉。本做法与切分无关:byte-level BPE 的每个词元字符都对应
    一个确定的字节(GPT-2 bytes_to_unicode 的逆映射),逐 token 还原、逐 token 累计
    偏移,天然与"整段 decode"的文本一致,不存在对不上的可能。
    特殊 token(<|im_end|> 等)与 SentencePiece 词元(含 ▁,给 Gemma 类模型留的路)
    的字符不全在字节表里,按字面 utf-8(▁→空格)处理。"""
    global _BYTE_DECODER
    if _BYTE_DECODER is None:
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
        _BYTE_DECODER = {ch: b for b, ch in bytes_to_unicode().items()}
    pieces = tokenizer.convert_ids_to_tokens(ids)
    out = []
    for p in pieces:
        if p is None:
            out.append(b"")
            continue
        try:
            out.append(bytes(_BYTE_DECODER[c] for c in p))
        except KeyError:
            out.append(p.replace("▁", " ").encode("utf-8"))
    return out


_BYTE_DECODER = None

_ACTION_OPEN_B = _ACTION_OPEN.encode()
_ACTION_CLOSE_B = _ACTION_CLOSE.encode()


def build_action_token_mask(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer,
) -> Tuple[torch.Tensor, dict]:
    """按设计 §2b 构造 act_mask:蒸馏只作用在【首个】<action>…</action> 的内容 token 上。

    定位走字节级(见 _token_byte_pieces 头注释;标签与动作文法都是 ASCII,
    字节偏移即字符偏移),与采样切分无关,不做 re-tokenize。口径(2026-09-10 边角):
      * 标签本身不进掩码,只盖标签之间的内容;与内容区间【相交】的 token 计入
        (跨标签边界的 token 携带内容字符,计入);
      * 无 </action>(截断行)→ 全空掩码,该行不参与蒸馏(安全性质,非 bug);
      * 多 <action> 块(实测 ~1.5%)只取首段:环境只解析执行首个动作,后续未被执行。

    Args:
        responses:     (bs, resp_len) 响应 token id。
        response_mask: (bs, resp_len) 有效 token 掩码。
    Returns:
        (act_mask float32 (bs, resp_len), metrics)。
    """
    bs, resp_len = responses.shape
    act_mask = torch.zeros((bs, resp_len), dtype=torch.float32)
    n_no_action = 0        # 整行没有 <action>(格式坏行)
    n_truncated = 0        # 有 <action> 无 </action>(响应被截断)
    n_multi = 0            # 多个 <action> 块(只取首段)
    n_rows = 0

    for i in range(bs):
        valid_idx = response_mask[i].nonzero(as_tuple=True)[0]
        if valid_idx.numel() == 0:
            continue
        n_rows += 1
        ids = responses[i, valid_idx].tolist()
        pieces = _token_byte_pieces(tokenizer, ids)
        text = b"".join(pieces)
        lo = text.find(_ACTION_OPEN_B)
        if lo < 0:
            n_no_action += 1
            continue
        hi = text.find(_ACTION_CLOSE_B, lo)
        if hi < 0:
            n_truncated += 1
            continue
        if text.find(_ACTION_OPEN_B, hi) >= 0:
            n_multi += 1
        content_lo = lo + len(_ACTION_OPEN_B)
        content_hi = hi
        if content_hi <= content_lo:
            continue                               # 空动作块:没有内容 token 可盖
        pos = 0
        for k, piece in enumerate(pieces):
            s, e = pos, pos + len(piece)
            pos = e
            if s >= content_hi:
                break
            if e > content_lo and s < content_hi:
                act_mask[i, valid_idx[k]] = 1.0

    denom = max(n_rows, 1)
    metrics = {
        "rso/act_rows_no_action_ratio": n_no_action / denom,
        "rso/act_rows_truncated_ratio": n_truncated / denom,
        "rso/act_multi_block_ratio": n_multi / denom,
    }
    return act_mask, metrics
