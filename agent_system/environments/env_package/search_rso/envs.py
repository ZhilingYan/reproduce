# -*- coding: utf-8 -*-
"""Search-QA 递归模式的底层批量环境。

与 flat 的 SearchMultiProcessEnv(env_package/search/envs.py)的关系:同样的
线程池 + 每槽一个逻辑环境 + 检索 HTTP 协议(复用 skyrl 的 call_search_api,
连接池/重试原样);差异按递归框架的需要:
  1. reset 的题目来自数据集 env_kwargs,但 RecursiveEnvironmentManager.reset()
     不传参(textcraft 任务是环境内部自采的)——所以本环境提供 stage_reset_kwargs(),
     由 SearchRecursiveEnvironmentManager 在调 super().reset() 之前把 kwargs 暂存进来;
  2. <answer> 不结束 episode、不算奖励(只回一句确认)。root 的终止与 EM 判分全部
     由适配器负责(rso_core 的 R(τ) 取 root 行 node_success,不读环境奖励);
  3. 空动作 "" 是合法 noop(委托轮的 filler;不打检索、不改状态);
  4. 步数无上限(全局轮数上限由采集循环负责,与 recursive_factory 的口径一致)。
本环境自身永远 done=False、reward=0。
"""
from __future__ import annotations

import concurrent.futures
import re
from typing import Any, Dict, List

from agent_system.environments.env_package.search.third_party.skyrl_gym.tools.search import (
    SearchToolGroup, call_search_api, _passages2string)

_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)

INVALID_OBS = ("Invalid action. Use exactly one of <search> query </search>, "
               "<answer> answer </answer>, or <delegate> sub-question </delegate>.")


class SearchRsoEnvs:
    def __init__(self, seed: int, env_num: int, group_n: int, is_train: bool, env_config):
        self.batch_size = env_num * group_n
        search_cfg = env_config.search
        self.search_url = str(search_cfg.search_url if not isinstance(search_cfg.search_url, (list, tuple))
                              else search_cfg.search_url[0])
        self.topk = int(getattr(search_cfg, "topk", 3))
        self.timeout = int(getattr(search_cfg, "timeout", 30))
        self._session = SearchToolGroup._get_shared_session(self.search_url)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self.batch_size, 256))
        self._staged_kwargs: List[Dict[str, Any]] = []
        self._slots: List[Dict[str, Any]] = [{} for _ in range(self.batch_size)]

    # ------------------------------------------------------------ 桥接
    def stage_reset_kwargs(self, kwargs: List[Dict[str, Any]]):
        self._staged_kwargs = list(kwargs or [])

    # ------------------------------------------------------------ reset
    @staticmethod
    def _targets(gt) -> List[str]:
        if isinstance(gt, dict):
            gt = gt.get("target")
        if gt is None:
            return []
        if isinstance(gt, str):
            return [gt]
        return [str(x) for x in list(gt)]

    def reset(self):
        kwargs = self._staged_kwargs
        assert kwargs, ("[search_rso] reset 前没有 stage_reset_kwargs——"
                        "必须经 SearchRecursiveEnvironmentManager 使用本环境")
        assert len(kwargs) <= self.batch_size, (
            f"[search_rso] 收到 {len(kwargs)} 条 env_kwargs,环境只有 {self.batch_size} 槽")
        obs, infos = [], []
        self._slots = []
        for kw in kwargs:
            q = str(kw.get("question", ""))
            slot = {"question": q, "targets": self._targets(kw.get("ground_truth")),
                    "data_source": str(kw.get("data_source", "unknown"))}
            self._slots.append(slot)
            obs.append(q)
            infos.append(self._base_info(slot))
        return obs, infos

    def _base_info(self, slot) -> Dict[str, Any]:
        return {
            "extra.question": slot["question"],
            "extra.ground_truth_targets": list(slot["targets"]),
            "data_source": slot["data_source"],
            "won": False,
            "extra.last_action_kind": "none",
        }

    # ------------------------------------------------------------ step
    def _one(self, slot: Dict[str, Any], action: str):
        info = self._base_info(slot)
        act = (action or "").strip()
        if not act:
            info["extra.last_action_kind"] = "noop"
            return "", 0.0, False, info
        m = _SEARCH_RE.search(act)
        if m:
            query = m.group(1).strip()
            info["extra.last_action_kind"] = "search"
            info["extra.last_query"] = query
            resp, err = call_search_api(self.search_url, query, topk=self.topk,
                                        timeout=self.timeout, log_requests=False,
                                        session=self._session)
            if err or not resp:
                obs = f"<information>Search failed: {str(err)[:200]}</information>"
            else:
                raw = resp.get("result", [])
                text = _passages2string(raw[0]) if raw else "No results found."
                obs = f"<information>{text.strip()}</information>"
            return obs, 0.0, False, info
        m = _ANSWER_RE.search(act)
        if m:
            info["extra.last_action_kind"] = "answer"
            info["extra.last_answer"] = m.group(1).strip()
            return "Your answer has been recorded.", 0.0, False, info
        info["extra.last_action_kind"] = "invalid"
        return INVALID_OBS, 0.0, False, info

    def step(self, actions: List[str]):
        assert len(actions) == len(self._slots), \
            f"[search_rso] step 收到 {len(actions)} 条动作,槽数 {len(self._slots)}"
        futures = [self._executor.submit(self._one, s, a)
                   for s, a in zip(self._slots, actions)]
        results = [f.result() for f in futures]
        obs, rewards, dones, infos = map(list, zip(*results))
        return obs, rewards, dones, infos

    def close(self):
        self._executor.shutdown(wait=False)


def build_search_rso_envs(seed, env_num, group_n, is_train, env_config):
    return SearchRsoEnvs(seed=seed, env_num=env_num, group_n=group_n,
                         is_train=is_train, env_config=env_config)
