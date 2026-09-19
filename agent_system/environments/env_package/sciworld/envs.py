# -*- coding: utf-8 -*-
"""ScienceWorld 的底层批量环境(flat 与递归共用)。

底本:search_rso/envs.py 的线程池骨架。sciworld 特有(移植对照表 2026-09-19):
  1. 每槽一个常驻 JVM(py4j;ScienceWorldEnv 懒创建,reset 走 load(task,var,"easy"));
  2. train 组语义:同组(连续 group_n 个槽)加载同一 (task, variation)——GRPO 组内可比
     的前提(env_manager.py 头注释);任务均匀采样(D2:防斜面家族 692 变体统治 batch);
  3. val:固定 50 case(用户拍板)= 30 任务 × dev[0] + 官方任务表前 20 任务 × dev[1],
     全部 sorted(get_variations_dev()) 取,slot i = case i;
  4. **累计分一律读 info['score']**:官方 python 包装 step() 的第二返回值是增量 reward
     (scienceworld.py:426-427),2026-09-19 实测踩坑;
  5. flat 奖励 = clip 分数增量 / 100(telescope 和 = clip(final,0,100)/100 = mapping §一
     的 R(τ));递归模式(zero_reward=True)奖励恒 0,R(τ) 由 rso_core 取 root 行
     node_success(search_rso/envs.py:14 同款口径);
  6. 空动作 "" 是合法 noop(递归委托轮 filler);`answer:` 前缀是子 agent 汇报,
     同样不进 env(两者都返回缓存观测);
  7. 动作送 env 前 .lower().strip()(官方 sanitize;projection 已小写,这里兜底);
  8. unmatched("No known action matches"/"Unknown action")、focus_death、won 打标;
  9. envStepLimit 只计费非 free 动作(官方口径);轮数上限由外层负责
     (flat = rollout loop 的 env.max_steps,递归 = root/child 预算 + 全局轮上限)。
"""
from __future__ import annotations

import concurrent.futures
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from agent_system.environments.env_package.sciworld.goal_progress import (
    parse_goal_progress, render_gt_plan)

_UNMATCHED_PREFIXES = ("no known action matches", "unknown action")

# 官方任务表顺序(env.get_task_names() 的顺序,1.3.0 实测;val 固定 case 的
# "前 20 任务拿第二条 dev"就按这个顺序取,避免运行时顺序漂移)
OFFICIAL_TASK_NAMES = [
    "boil", "change-the-state-of-matter-of", "chemistry-mix",
    "chemistry-mix-paint-secondary-color", "chemistry-mix-paint-tertiary-color",
    "find-animal", "find-living-thing", "find-non-living-thing", "find-plant",
    "freeze", "grow-fruit", "grow-plant", "identify-life-stages-1",
    "identify-life-stages-2", "inclined-plane-determine-angle",
    "inclined-plane-friction-named-surfaces", "inclined-plane-friction-unnamed-surfaces",
    "lifespan-longest-lived", "lifespan-longest-lived-then-shortest-lived",
    "lifespan-shortest-lived", "measure-melting-point-known-substance",
    "measure-melting-point-unknown-substance", "melt", "mendelian-genetics-known-plant",
    "mendelian-genetics-unknown-plant", "power-component",
    "power-component-renewable-vs-nonrenewable-energy", "test-conductivity",
    "test-conductivity-of-unknown-substances", "use-thermometer",
]
SUDDEN_DEATH_TASKS = {"identify-life-stages-1", "identify-life-stages-2"}


def _default_env_factory(env_step_limit: int):
    from scienceworld import ScienceWorldEnv
    return ScienceWorldEnv("", envStepLimit=env_step_limit)


class _Slot:
    """单槽状态:一个常驻 JVM + 当前 episode 的记账。"""

    def __init__(self):
        self.env = None
        self.task = ""
        self.variation = -1
        self.done = True
        self.score = 0          # info['score'](可为 -100)
        self.peak = 0           # clip 后的分数峰值(只增不减,Φ 的 P)
        self.last_obs = ""
        self.admissible = ""
        self.task_desc = ""
        self.gt_plan = ""
        self.goal_progress = ""


class SciWorldEnvs:
    def __init__(self, seed: int, env_num: int, group_n: int, is_train: bool, env_config,
                 zero_reward: bool = False, env_factory=None):
        self.env_num = int(env_num)
        self.group_n = max(int(group_n), 1)
        self.batch_size = self.env_num * self.group_n
        self.is_train = bool(is_train)
        self.zero_reward = bool(zero_reward)
        self.rng = np.random.RandomState(seed)

        sw_cfg = {}
        try:
            sw_cfg = dict(env_config.get("sciworld", {}) or {})
        except Exception:
            pass
        self.simplification = str(sw_cfg.get("simplification", "easy"))
        self.env_step_limit = int(sw_cfg.get("env_step_limit", 200))
        self.max_valid_actions = int(sw_cfg.get("max_valid_actions", 300))
        # 递归 OPSD 需要每步的 goal_progress(priv 现算);flat gt 特权只要 reset 时的骨架
        self.fetch_goal_progress = bool(sw_cfg.get("fetch_goal_progress", False))

        self._env_factory = env_factory or (lambda: _default_env_factory(self.env_step_limit))
        self._slots = [_Slot() for _ in range(self.batch_size)]
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self.batch_size, 128))
        self._split_lock = threading.Lock()
        self._splits: Optional[Dict[str, Dict[str, List[int]]]] = None
        self._val_cases: Optional[List[tuple]] = None

    # ------------------------------------------------------------ 官方划分
    def _load_splits(self) -> Dict[str, Dict[str, List[int]]]:
        """探针 env 逐任务查官方 train/dev 划分(30 次 load,秒级,只做一次)。"""
        with self._split_lock:
            if self._splits is not None:
                return self._splits
            probe = self._env_factory()
            splits: Dict[str, Dict[str, List[int]]] = {}
            names = list(probe.get_task_names())
            assert set(names) == set(OFFICIAL_TASK_NAMES), (
                f"[sciworld] 官方任务表与硬编码顺序表不一致:{sorted(set(names) ^ set(OFFICIAL_TASK_NAMES))}")
            for t in OFFICIAL_TASK_NAMES:
                probe.load(t, 0, self.simplification, generateGoldPath=False)
                splits[t] = {
                    "train": sorted(probe.get_variations_train()),
                    "dev": sorted(probe.get_variations_dev()),
                }
            probe.close()
            self._splits = splits
            return splits

    def _fixed_val_cases(self) -> List[tuple]:
        """50 个固定验证 case(用户拍板 2026-09-19):
        30 任务各取 sorted(dev)[0],官方任务表前 20 个任务再取 sorted(dev)[1]。"""
        if self._val_cases is not None:
            return self._val_cases
        splits = self._load_splits()
        cases = [(t, splits[t]["dev"][0]) for t in OFFICIAL_TASK_NAMES]
        cases += [(t, splits[t]["dev"][1]) for t in OFFICIAL_TASK_NAMES[:20]]
        assert len(cases) == 50
        self._val_cases = cases
        return cases

    # ------------------------------------------------------------ reset
    def _sample_train_cases(self) -> List[tuple]:
        """每组一个 (task, variation):任务均匀 × 任务内变体均匀(D2)。"""
        splits = self._load_splits()
        cases = []
        for _ in range(self.env_num):
            t = OFFICIAL_TASK_NAMES[self.rng.randint(len(OFFICIAL_TASK_NAMES))]
            pool = splits[t]["train"]
            cases.append((t, int(pool[self.rng.randint(len(pool))])))
        return cases

    def _reset_one(self, slot: _Slot, task: str, variation: int):
        if slot.env is None:
            slot.env = self._env_factory()
        env = slot.env
        env.load(task, int(variation), self.simplification, generateGoldPath=False)
        obs, _ = env.reset()
        slot.task, slot.variation = task, int(variation)
        slot.done = False
        slot.score, slot.peak = 0, 0
        slot.task_desc = str(env.get_task_description()).strip()
        slot.last_obs = str(obs).strip()
        slot.admissible = self._admissible_str(env)
        gp = str(env.get_goal_progress())
        slot.goal_progress = gp
        # flat OPSD 的 gt 特权:整题 subgoal 骨架(静态,reset 时渲染一次;
        # 经 info['extra.gt_plan'] reset+step 双注入 —— search/envs.py:84-120 的补丁口径)
        slot.gt_plan = render_gt_plan(parse_goal_progress(gp))
        return slot.last_obs, self._info(slot, kind="reset")

    # ------------------------------------------------------------ 桥接(递归管理器用,search_rso 同款)
    def stage_reset_kwargs(self, kwargs):
        # 注意 kwargs 可能是 numpy 对象数组——严禁 or/if 真值判断(已修坑②),显式判 None
        self._staged_kwargs = list(kwargs) if kwargs is not None else []

    def reset(self, kwargs=None):
        """三种模式(2026-09-19 加 case 指定模式,供官方 test 全量评测走训练框架):
        ① kwargs 逐行含 {"task","variation"} → 按行加载,只激活前 len(kwargs) 个槽
           (env_kwargs 经数据集 parquet 下发,search 域同款通道;分批评测时
           余数 batch 槽数自动缩,JVM 池按 val_batch_size 复用);
        ② is_train → 官方 train 划分任务均匀采样(组内同 case);
        ③ 否则 → 固定 50 dev case(训练中验证)。"""
        if kwargs is None:
            kwargs = getattr(self, "_staged_kwargs", None)
            self._staged_kwargs = None
        kw_list = list(kwargs) if kwargs is not None else []
        kw_list = [k for k in kw_list if isinstance(k, dict) and "task" in k]
        if kw_list:
            assert len(kw_list) <= self.batch_size, (
                f"[sciworld] env_kwargs {len(kw_list)} 条超过槽数 {self.batch_size}")
            cases = [(str(k["task"]), int(k["variation"])) for k in kw_list]
        elif self.is_train:
            group_cases = self._sample_train_cases()
            cases = [group_cases[i // self.group_n] for i in range(self.batch_size)]
        else:
            fixed = self._fixed_val_cases()
            assert self.batch_size <= len(fixed) or self.group_n == 1, \
                f"[sciworld] val 槽数 {self.batch_size} 超过固定 case 数 {len(fixed)}"
            cases = [fixed[i % len(fixed)] for i in range(self.batch_size)]
        self._active = len(cases)
        futures = [self._executor.submit(self._reset_one, s, t, v)
                   for s, (t, v) in zip(self._slots[: self._active], cases)]
        results = [f.result() for f in futures]
        obs, infos = map(list, zip(*results))
        return obs, infos

    # ------------------------------------------------------------ info
    def _admissible_str(self, env) -> str:
        combos = list(env.get_valid_action_object_combinations())
        s = ", ".join(combos[: self.max_valid_actions])
        if len(combos) > self.max_valid_actions:
            s += f", ... ({len(combos) - self.max_valid_actions} more omitted)"
        return s

    def _info(self, slot: _Slot, kind: str, unmatched: bool = False,
              focus_death: bool = False) -> Dict[str, Any]:
        info = {
            "extra.task": slot.task,
            # skill 特权分发通道:SkillProvider.get_privileged_info 按 task_to_skill 键
            # 对 gamefile 做子串匹配(alfworld 同款;rlsd_ray_trainer.py:154-160)
            "extra.gamefile": slot.task,
            "extra.variation": slot.variation,
            "extra.task_desc": slot.task_desc,
            "extra.observation": slot.last_obs,
            "extra.admissible": slot.admissible,
            "extra.score": int(slot.score),
            "extra.peak": int(slot.peak),
            "extra.sudden_death_task": slot.task in SUDDEN_DEATH_TASKS,
            "extra.last_action_kind": kind,
            "extra.env_unmatched": bool(unmatched),
            "extra.focus_death": bool(focus_death),
            "extra.gt_plan": slot.gt_plan,
            "won": bool(slot.score >= 100),
        }
        if self.fetch_goal_progress:
            info["extra.goal_progress"] = slot.goal_progress
        return info

    # ------------------------------------------------------------ step
    def _one(self, slot: _Slot, action: str):
        act = (action or "").strip()
        # 终局后的陪跑轮(lockstep 可能继续喂):不进 env
        if slot.done:
            return "Episode ended.", 0.0, True, self._info(slot, kind="post_done")
        # 递归 filler / 子 agent 汇报:不进 env,返回缓存观测
        if not act:
            return slot.last_obs, 0.0, False, self._info(slot, kind="noop")
        if act.lower().startswith("answer"):
            return slot.last_obs, 0.0, False, self._info(slot, kind="answer")

        env = slot.env
        obs, _delta, done, step_info = env.step(act.lower().strip())
        score = int(step_info["score"])          # 累计分,不是增量(移植对照表第 4 条)
        prev_clip = max(slot.score, 0)
        slot.score = score
        clip = max(score, 0)
        reward = 0.0 if self.zero_reward else (clip - prev_clip) / 100.0
        slot.peak = max(slot.peak, clip)
        slot.last_obs = str(obs).strip()
        unmatched = slot.last_obs.lower().startswith(_UNMATCHED_PREFIXES)
        focus_death = bool(done and score < 0 and act.lower().startswith("focus"))
        if not done:
            slot.admissible = self._admissible_str(env)
            if self.fetch_goal_progress:
                slot.goal_progress = str(env.get_goal_progress())
        slot.done = bool(done)
        return slot.last_obs, reward, bool(done), self._info(
            slot, kind="env", unmatched=unmatched, focus_death=focus_death)

    def step(self, actions: List[str]):
        active = getattr(self, "_active", self.batch_size)
        assert len(actions) == active, \
            f"[sciworld] step 收到 {len(actions)} 条动作,活跃槽数 {active}"
        futures = [self._executor.submit(self._one, s, a)
                   for s, a in zip(self._slots[:active], actions)]
        results = [f.result() for f in futures]
        obs, rewards, dones, infos = map(list, zip(*results))
        return obs, rewards, dones, infos

    def close(self):
        for s in self._slots:
            try:
                if s.env is not None:
                    s.env.close()
            except Exception:
                pass
        self._executor.shutdown(wait=False)


def build_sciworld_envs(seed, env_num, group_n, is_train, env_config,
                        zero_reward=False, env_factory=None):
    return SciWorldEnvs(seed=seed, env_num=env_num, group_n=group_n, is_train=is_train,
                        env_config=env_config, zero_reward=zero_reward,
                        env_factory=env_factory)
