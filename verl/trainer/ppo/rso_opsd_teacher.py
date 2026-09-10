# -*- coding: utf-8 -*-
"""RSO+OPSD 的 teacher 批构造:每行的特权前缀 = 该行【节点局部】的 priv(node_priv 列)。

逐行克隆 rlsd_ray_trainer.py:96-257 的 build_teacher_batch(那份"手术"的一图流注释
仍然适用:切 prompt→剥 pad→decode→拼前缀→re-tokenize→左截断→左 pad→拼回原封不动的
response token→重算 position_ids;最重要的保证是 response token 一个不变)。差异三处:
  1. 特权源:不再走 skill_provider / gt_plan,改读每行 non_tensor 列 node_priv
     (收集器按 (slot, step_idx) 从 turn_meta 回填,行开局现算,见 orchestrator);
  2. 左截断从"静默"改为【显式计数 + 告警】,且按任务难度分桶(task_difficulty 列)——
     extreme 闭包 ~170 项时 priv 全文 2-3K token,总体率会被 easy/medium 稀释,
     不分桶看不见(RSO_method_design §2a 落地路径 4,必须项;checklist #5);
  3. 返回 (teacher_batch, metrics),截断/缺失指标由 trainer 并进日志。
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask

# 前缀措辞对照 rlsd_ray_trainer.py:61-67 的 DEFAULT_GT_PREFIX:同样明示"仅训练期
# teacher 可见",但内容是【这个节点当前子任务】的工作计划,不是整局的标准答案。
DEFAULT_PRIV_PREFIX = (
    "[Privileged Plan]\n"
    "The following is a privileged work plan for YOUR current task, computed from the "
    "task's recipe database and your own notebook. It is available only as teacher-side "
    "supervision during training; the student does NOT see it at test time.\n"
    "{priv}\n\n"
)

_DIFF_BUCKETS = ("easy", "medium", "hard", "extreme")


def build_priv_teacher_batch(
    batch: DataProto,
    tokenizer,
    max_prompt_length: int,
    prefix_template: Optional[str] = None,
) -> Tuple[DataProto, dict]:
    """为每行拼 node_priv 前缀,返回 (teacher_batch, metrics)。

    node_priv 列不存在 → 直接报错:main_rso_opsd 的通路里它必然存在
    (编排器 _priv_enabled → turn_meta → 收集器回填),缺失=接线断了,不能静默跑。
    """
    ntb = batch.non_tensor_batch
    assert "node_priv" in ntb, (
        "[RSO+OPSD] batch 里没有 node_priv 列——priv 接线断了(编排器 _priv_enabled/"
        "收集器回填任一环节失效)。拒绝静默退化,请排查。")
    priv_col = ntb["node_priv"]
    diff_col = ntb.get("task_difficulty", None)

    prefix_template = prefix_template or DEFAULT_PRIV_PREFIX

    bs = batch.batch["input_ids"].size(0)
    response_length = batch.batch["responses"].size(1)

    n_miss = 0                                        # priv 为空串的行(应≈0,只在防御路径出现)
    n_trunc_total = 0
    n_trunc_by_diff = {d: 0 for d in _DIFF_BUCKETS}
    n_rows_by_diff = {d: 0 for d in _DIFF_BUCKETS}

    teacher_input_ids_list = []
    teacher_attention_mask_list = []
    teacher_position_ids_list = []

    for i in range(bs):
        original_input_ids = batch.batch["input_ids"][i]
        original_attention_mask = batch.batch["attention_mask"][i]
        prompt_length = original_input_ids.size(0) - response_length

        prompt_ids = original_input_ids[:prompt_length]
        prompt_mask = original_attention_mask[:prompt_length]

        valid_start = prompt_mask.nonzero(as_tuple=True)[0]
        valid_start = valid_start[0].item() if len(valid_start) > 0 else 0

        valid_prompt_ids = prompt_ids[valid_start:]
        prompt_text = tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)

        priv_text = priv_col[i]
        priv_text = priv_text if isinstance(priv_text, str) else ("" if priv_text is None else str(priv_text))
        if priv_text == "":
            n_miss += 1
        priv_prefix = prefix_template.format(priv=priv_text)

        teacher_prompt_text = priv_prefix + prompt_text
        teacher_prompt_ids = tokenizer.encode(teacher_prompt_text, add_special_tokens=False)

        # 左截断:显式计数(差异 2)。被砍的是序列开头 = 特权前缀的头部,砍多了 teacher
        # 等于没拿到小抄——这正是必须打点分桶的原因。
        diff = ""
        if diff_col is not None and i < len(diff_col):
            diff = str(diff_col[i] or "")
        if diff in n_rows_by_diff:
            n_rows_by_diff[diff] += 1
        if len(teacher_prompt_ids) > max_prompt_length:
            n_trunc_total += 1
            if diff in n_trunc_by_diff:
                n_trunc_by_diff[diff] += 1
            teacher_prompt_ids = teacher_prompt_ids[-max_prompt_length:]

        teacher_prompt_ids = torch.tensor(teacher_prompt_ids, dtype=torch.long)
        actual_prompt_len = len(teacher_prompt_ids)

        pad_length = max_prompt_length - actual_prompt_len
        if pad_length > 0:
            pad_ids = torch.full((pad_length,), tokenizer.pad_token_id, dtype=torch.long)
            teacher_prompt_ids = torch.cat([pad_ids, teacher_prompt_ids])
            t_prompt_mask = torch.cat([
                torch.zeros(pad_length, dtype=torch.long),
                torch.ones(actual_prompt_len, dtype=torch.long),
            ])
        else:
            t_prompt_mask = torch.ones(actual_prompt_len, dtype=torch.long)

        response_ids = batch.batch["responses"][i]
        response_mask = original_attention_mask[-response_length:]

        teacher_full_ids = torch.cat([teacher_prompt_ids, response_ids])
        teacher_full_mask = torch.cat([t_prompt_mask, response_mask])
        teacher_position_ids = compute_position_id_with_mask(teacher_full_mask.unsqueeze(0))[0]

        teacher_input_ids_list.append(teacher_full_ids)
        teacher_attention_mask_list.append(teacher_full_mask)
        teacher_position_ids_list.append(teacher_position_ids)

    teacher_batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.stack(teacher_input_ids_list),
            "attention_mask": torch.stack(teacher_attention_mask_list),
            "position_ids": torch.stack(teacher_position_ids_list),
            "responses": batch.batch["responses"],
        },
    )

    metrics = {
        "rso/priv_truncation_rate": n_trunc_total / max(bs, 1),
        "rso/priv_miss_ratio": n_miss / max(bs, 1),
    }
    for d in _DIFF_BUCKETS:
        if n_rows_by_diff[d] > 0:
            metrics[f"rso/priv_truncation_rate_{d}"] = n_trunc_by_diff[d] / n_rows_by_diff[d]
    if n_trunc_total > 0:
        detail = ", ".join(f"{d}:{n_trunc_by_diff[d]}/{n_rows_by_diff[d]}"
                           for d in _DIFF_BUCKETS if n_rows_by_diff[d] > 0)
        print(f"[RSO+OPSD] WARNING teacher prompt 左截断 {n_trunc_total}/{bs} 行"
              f"(按难度 {detail})——截掉的是特权前缀头部;hard/extreme 持续非零"
              "需压缩 priv 渲染(RSO_method_design §四监控条款)")
    if n_miss > 0:
        print(f"[RSO+OPSD] WARNING {n_miss}/{bs} 行的 node_priv 为空串(防御路径,应≈0)")
    return teacher_batch, metrics
