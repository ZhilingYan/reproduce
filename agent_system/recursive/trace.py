# -*- coding: utf-8 -*-
"""异常留档与分层计时。补审查报告的 P11(无 traceback 留档)和 P12(无 episode 计时器)。

移植来源(路径相对 Experiment/code_references/platoon/,行号为**本地副本**的行号):

  platoon/episode/loop.py:79-97      异常 → 可读文本,不外抛
      :79   except (Exception, asyncio.CancelledError) as e:
      :84   traceback.extract_tb(e.__traceback__)   取栈帧,用最后一帧拼"死在哪个文件哪行哪个函数"
      :93   traceback.format_exc()                   留完整调用栈
      :97   error_message.set(detailed_msg)          存进上下文变量,finally 里落到轨迹上
  platoon/utils/span_profile.py:22-89   环境变量开关的分层计时器
      :22-24  enabled()      靠环境变量 PLATOON_PROFILE_SPANS 开关
      :26-27  output_path()  靠环境变量指定 jsonl 路径
      :49-52  未开启时直接 yield None,零开销
      :60     started_at = time.perf_counter()
      :61/:83 parent_span_id  用 ContextVar 维护 span 栈,记父 span
      :70-74  self_ms = inclusive_ms - child_time_ms   自身耗时 = 含子耗时 减去 子耗时
      :40/:77 _append_record 加线程锁后追加写 jsonl

与官方的实现差异(不影响用途):
  官方 profile_span 是 async 上下文管理器,套在协程上,靠 ContextVar 在协程间维护 span 栈。
  我们是同步、逐槽推进的,一个 span 对应"一个节点从开张到关闭",父子关系直接用节点的
  parent_uid 表达,不需要 ContextVar。计时的粒度是轮数而非墙钟——lockstep 下所有槽同步走,
  单个节点的墙钟耗时没有意义,有意义的是"它占了几轮、是第几轮开张第几轮关闭"。
  墙钟只在整个 rollout 层面记一次(由采集器调 rollout_span)。
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# 异常留档(对应 loop.py:79-97)
# ---------------------------------------------------------------------------

def format_exception(exc: BaseException, where: str = "") -> str:
    """把异常变成一段可读文本,格式与官方 loop.py:84-96 拼出来的 detailed_msg 一致:

        Error in <where> (<文件>:<行号> in <函数>)
        <异常类型>: <异常消息>
        <完整调用栈>

    官方的设计原则:episode 怎么死都不能让异常炸穿整个 rollout,但死因必须留档。
    我们把这段文本写进 NodeRecord.error,而不是像官方那样存进 contextvar——
    因为我们没有协程上下文,节点记录就是唯一的落点。
    """
    tb_summary = traceback.extract_tb(exc.__traceback__)
    origin = ""
    if tb_summary:
        last = tb_summary[-1]
        origin = f"{last.filename}:{last.lineno} in {last.name}"
    head = f"Error in {where or 'recursive orchestrator'}"
    if origin:
        head += f" ({origin})"
    full_tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return f"{head}\n{type(exc).__name__}: {exc}\n{full_tb}"


# ---------------------------------------------------------------------------
# 分层计时(对应 span_profile.py)
# ---------------------------------------------------------------------------

_ENV_SWITCH = "RAO_PROFILE_SPANS"                   # 对应官方 PLATOON_PROFILE_SPANS
_ENV_PATH = "RAO_PROFILE_SPANS_PATH"                # 对应官方 PLATOON_PROFILE_SPANS_PATH
_DEFAULT_PATH = "/tmp/rao_span_profile.jsonl"
_WRITE_LOCK = threading.Lock()                      # 对应官方 span_profile.py:15


def enabled() -> bool:
    """是否开启计时。对应 span_profile.py:22-24,默认关闭、零开销。"""
    return os.getenv(_ENV_SWITCH, "").lower() in {"1", "true", "yes", "on"}


def output_path() -> str:
    """jsonl 落盘路径。对应 span_profile.py:26-27。"""
    return os.getenv(_ENV_PATH, _DEFAULT_PATH)


def _append_record(record: Dict[str, Any]) -> None:
    """加锁追加写一行 jsonl。对应 span_profile.py:40-46。"""
    path = output_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with _WRITE_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def record_node_span(node, slot: int, opened_round: int, closed_round: int,
                     extra: Optional[Dict[str, Any]] = None) -> None:
    """节点关闭时记一条 span。未开启计时则什么都不做(对应 :51-52 的零开销分支)。

    字段设计对应官方 span_profile.py:77-88 那条 record,但把墙钟换成轮数:
        name / span_id / parent_span_id  -> "node" / node.uid / node.parent_uid
        started_at / inclusive_ms        -> opened_round / rounds(占了几轮)
        metadata                         -> depth / goal / close_reason / success / slot
    有了这些,排查"委托为什么总以预算耗尽收场""时间(轮数)花在哪一层"时可以直接定位。
    """
    if not enabled():
        return
    rec = {
        "name": "node",
        "span_id": node.uid,
        "parent_span_id": node.parent_uid,
        "slot": slot,
        "depth": node.depth,
        "goal": node.goal_text,
        "opened_round": opened_round,
        "closed_round": closed_round,
        "rounds": closed_round - opened_round,
        "turns": node.turns_used,
        "budget_total": node.budget_total,
        "close_reason": node.close_reason,
        "success": node.success,
        "n_children": len(node.children_uids),
        "error": bool(node.error),
    }
    if extra:
        rec.update(extra)
    _append_record(rec)


class rollout_span:
    """整个 rollout(一次采集,所有槽从 reset 到收摊)的墙钟计时,同步上下文管理器。

    对应官方 loop.py:47-57 用 profile_span("run_episode", ...) 包住整集的做法,
    只是我们的"一集"是整批 128 个槽同步走完的一次采集。用法:
        with rollout_span(metadata={"n_slots": 128}):
            ... 采集循环 ...
    未开启计时时 __enter__/__exit__ 都是空操作。
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        self.metadata = dict(metadata or {})
        self._t0 = 0.0

    def __enter__(self):
        if enabled():
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not enabled():
            return False
        elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        rec = {
            "name": "rollout",
            "inclusive_ms": elapsed_ms,
            "ok": exc_type is None,
        }
        rec.update(self.metadata)
        _append_record(rec)
        return False    # 不吞异常
