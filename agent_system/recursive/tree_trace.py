# -*- coding: utf-8 -*-
"""执行树的完整轨迹记录器:每个 agent(节点)每一轮的 prompt / 模型输出 / 环境结果,以及节点间的
委托与战报关系,按事件逐行写 jsonl。

移植来源(路径相对 Experiment/code_references/platoon/,行号为本地副本行号):
  platoon/visualization/event_sinks.py:31-105   JsonlFileSink —— 每个事件一行、带 "type" 字段、
                                                  按 trajectory_created / step_added / finished 落盘
  platoon/episode/trajectory.py:71-77           TrajectoryEventHandler 的四个钩子(事件的触发点)
  platoon/episode/loop.py:47-57                 每集的元数据:trajectory_id / parent_trajectory_id / task_id

与官方的差异:官方 sink 挂在 TrajectoryCollection 上被动接收事件;我们没有那个账本,
由编排器在对应位置主动调用本类的 emit。事件类型与字段设计见 docs/RAO_PORT_DESIGN.md。

默认关闭(env.rao.trace_dir 为空),关闭时所有方法都是空操作、零开销。
每次 envs.reset() 开一个新文件:{trace_dir}/{tag}/reset_{序号:04d}.jsonl
(tag 区分 train / val,由工厂传入)。配套阅读脚本:scripts_rao/render_tree_trace.py。

事件类型与字段:
  reset         slot, tag, task_id, difficulty, target_items, initial_inventory
  node_open     slot, round, node_uid, parent_uid, tree_id, depth, goal_text, context, fork_step, budget_total
  turn          slot, round, node_uid, depth, turn_index, prompt, model_output, parsed_action,
                is_delegation, env_result, reward, budget_left_after
  delegate      slot, round, parent_uid, child_uid, goal_text, context
  refusal       slot, round, node_uid, text
  child_report  slot, round, child_uid, parent_uid, report_text, child_success, child_reason
  node_close    slot, round, node_uid, depth, success, reason, turns, error
关于 prompt 字段:记的是模型这一轮看到的用户消息原文(build_observation 的输出)。模型实际输入是
tokenizer.apply_chat_template([{"role":"user","content": prompt}])(rollout_loop.py:144-155),
那是一个不随轮次变化的确定性包裹,不逐行重复。
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Optional


class TreeTraceWriter:
    def __init__(self, trace_dir: Optional[str], tag: str = "train"):
        self.enabled = bool(trace_dir)
        self.trace_dir = os.path.join(trace_dir, tag) if self.enabled else None
        self.tag = tag
        self._f = None
        self._reset_idx = 0
        self._lock = threading.Lock()
        if self.enabled:
            os.makedirs(self.trace_dir, exist_ok=True)

    # ------------------------------------------------------------ 生命周期
    def begin_reset(self) -> None:
        """每次 envs.reset() 调一次:关掉上一个文件(若还开着),开新文件。"""
        if not self.enabled:
            return
        self.end_reset()
        path = os.path.join(self.trace_dir, f"reset_{self._reset_idx:04d}.jsonl")
        self._reset_idx += 1
        self._f = open(path, "a", encoding="utf-8")
        self.path = path

    def end_reset(self) -> None:
        if self._f is not None:
            try:
                self._f.flush()
                self._f.close()
            finally:
                self._f = None

    # ------------------------------------------------------------ 写事件
    def emit(self, event_type: str, **fields: Any) -> None:
        if not self.enabled or self._f is None:
            return
        rec: Dict[str, Any] = {"type": event_type}
        rec.update(fields)
        line = json.dumps(rec, ensure_ascii=False, default=_jsonable)
        with self._lock:
            self._f.write(line + "\n")
            self._f.flush()          # 每行即刷:作业被杀时也能保住已写的轮次

    def __del__(self):
        try:
            self.end_reset()
        except Exception:
            pass


def _jsonable(o: Any):
    """numpy 标量 / 其他非 JSON 类型的兜底转换。"""
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)
