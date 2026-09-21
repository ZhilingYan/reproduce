# -*- coding: utf-8 -*-
"""全量测评的逐题落盘:每条轨迹(每道题)写一行 JSON,供 scripts_rso/compute_eval_metrics.py 离线算指标。

只在配置了 `+trainer.val_case_dump_dir=<目录>` 且处于验证(is_train=False)时由
TrajectoryCollector.multi_turn_loop 调用;训练与训练中测试不传这个键,本模块不会被执行。

一行记录的内容(不存完整轨迹,只存算指标要用的量):
  - 题目标识:data_source、env_kwargs(问题原文 / 金答案 / sciworld 的 task 与 variation 等)、
    最后一个有效步 info 里的短小 extra.* 字段(如 extra.task、extra.goal_item、extra.score)。
  - 结果:won、episode 累计奖励。
  - 过程:有效步数、工具调用次数(检索次数)、格式无效的步数、被环境拒绝的步数。
  - 成本:全部有效步生成的 token 总数、单步 prompt 长度的最大值。
  - 最后一步回答的末尾文本(问答任务从中抽取 <answer>,用于 F1 / SubEM)。
  - 递归方法另记:节点数、最大深度、root 自身步数、委托轮数、root 的成败与关闭原因。
"""
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

# 最后一步回答只留末尾这么多字符:<answer> 标签在回答末尾,留尾巴足够抽答案,又不让文件膨胀
LAST_RESPONSE_TAIL_CHARS = 1500
# info 里只收短小的 extra.* 字段;观测全文、合法动作清单、特权计划之类的大字段一律不存
MAX_EXTRA_STR_LEN = 200


def _jsonable(x: Any) -> Any:
    """numpy / torch 标量与数组转成 JSON 能写的 Python 类型。"""
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "item") and callable(x.item):          # torch 标量
        try:
            return x.item()
        except Exception:
            pass
    return x


def _small_extras(info: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in (info or {}).items():
        if not str(k).startswith("extra."):
            continue
        v = _jsonable(v)
        if isinstance(v, bool) or isinstance(v, (int, float)):
            out[k] = v
        elif isinstance(v, str) and len(v) <= MAX_EXTRA_STR_LEN:
            out[k] = v
    return out


def _token_counts(row: Dict[str, Any]):
    """由一步的 attention_mask 与 responses 算(生成 token 数, prompt token 数)。
    attention_mask 是 [prompt | response] 两段拼接,response 段长度等于 responses 的宽度。"""
    am, resp = row.get("attention_mask"), row.get("responses")
    if am is None or resp is None:
        return None, None
    r = int(resp.shape[-1])
    gen = int(am[-r:].sum())
    prompt = int(am[:-r].sum())
    return gen, prompt


def build_case_records(total_batch_list: List[List[Dict]], total_infos: Optional[List[List[Dict]]],
                       episode_rewards, episode_lengths, tool_callings,
                       env_kwargs=None, data_sources=None, tokenizer=None,
                       node_records: Optional[List[list]] = None) -> List[Dict[str, Any]]:
    """把一批轨迹整理成逐题记录(纯函数,便于单测)。没有任何有效步的槽(余数批里未激活的槽)跳过。"""
    records = []
    for i, rows in enumerate(total_batch_list):
        active = [k for k, r in enumerate(rows) if bool(r.get("active_masks"))]
        if not active:
            continue
        infos = total_infos[i] if total_infos is not None and i < len(total_infos) else []
        last = active[-1]
        last_info = infos[last] if last < len(infos) else {}

        n_format_invalid = sum(1 for k in active if not bool(rows[k].get("is_action_valid", True)))
        n_action_failed = sum(1 for k in active
                              if k < len(infos) and bool((infos[k] or {}).get("extra.action_failed", False)))
        gens, prompts = [], []
        for k in active:
            g, p = _token_counts(rows[k])
            if g is not None:
                gens.append(g)
                prompts.append(p)

        rec: Dict[str, Any] = {
            "slot": i,
            "data_source": _jsonable(data_sources[i]) if data_sources is not None and i < len(data_sources) else None,
            "env_kwargs": _jsonable(env_kwargs[i]) if env_kwargs is not None and i < len(env_kwargs) else None,
            "extra": _small_extras(last_info),
            "won": bool(last_info.get("won", False)),
            "reward": float(episode_rewards[i]),
            "n_steps": int(episode_lengths[i]),
            "n_tool_calls": float(tool_callings[i]),
            "n_format_invalid": int(n_format_invalid),
            "n_action_failed": int(n_action_failed),
            "gen_tokens": int(sum(gens)) if gens else None,
            "max_prompt_tokens": int(max(prompts)) if prompts else None,
        }
        if tokenizer is not None and rows[last].get("responses") is not None:
            text = tokenizer.decode(rows[last]["responses"], skip_special_tokens=True)
            rec["last_response_tail"] = text[-LAST_RESPONSE_TAIL_CHARS:]

        # 递归方法:行上带有 node_depth / is_delegation_turn(RecursiveTrajectoryCollector.gather_rollout_data 贴上)
        if any("node_depth" in rows[k] for k in active):
            n_deleg = sum(1 for k in active if bool(rows[k].get("is_delegation_turn", False)))
            recs = node_records[i] if node_records is not None and i < len(node_records) else []
            roots = [r for r in recs if int(getattr(r, "depth", -1)) == 0]
            root = roots[-1] if roots else None
            rec["rec"] = {
                "total_turns": len(active),
                "root_turns": sum(1 for k in active if int(rows[k].get("node_depth", -1)) == 0),
                "delegation_turns": int(n_deleg),
                "env_turns": len(active) - int(n_deleg),
                "n_nodes": len(recs),
                "max_depth": max((int(r.depth) for r in recs), default=0),
                "root_success": float(root.success) if root is not None else None,
                "root_close_reason": str(root.close_reason) if root is not None else None,
            }
        records.append(rec)
    return records


def dump_val_cases(dump_dir: str, **kwargs) -> str:
    """整理并追加写入 <dump_dir>/cases.jsonl。验证可能分多批调用,所以用追加模式。"""
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, "cases.jsonl")
    records = build_case_records(**kwargs)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[val_case_dump] 追加 {len(records)} 题 → {path}", flush=True)
    return path
