# -*- coding: utf-8 -*-
"""Φ(cost-to-go)势函数:RSO 进展奖励的核心部件。

设计出处:Ideation/ideas/RSO_advantage_design.md 第三节(定义与三项验证)。
算法出处:代码理解/probe_costtogo.py 的 needed_closure —— 那份代码已对全部 632 道
验证集任务与 gold 步数逐题对拍相等(632/632),本文件把它原样搬来,一行未改逻辑。

Φ(inv, K) = C(inv) + γ·|M(inv) \\ K|
  C(inv)   还需 craft 的不同物品个数(每物品只有一条配方、一次动作可做任意整数倍,
           所以"最少 craft 动作数"就是必须生产的不同物品数)
  M(inv)   完成这些 craft 需要用到的配方集合(即上面那些物品各自的唯一配方)
  K        全树已经 get_info 查到过的配方并集(槽级,跨节点共享)
  γ        配方项权重,默认 1;γ=1 时 Φ(s0, ∅) = 2C,每个待造物品各摊一次查询一次制造

不可解状态(叶子材料的缺口补不回来)时闭包无解,Φ 无定义 —— 按设计口径【冻结】:
停在最后一个有定义的值,之后每轮 ΔΦ = 0(RSO_progress_shaping.md §八)。
边界判序:先查可解性——把状态毁到不可解的那一轮按冻结记 0,不记负(先冻结后计数)。

ΔΦ 允许为负(2026-09-07 甲案,RSO_phi_monotonicity_hole.md + RSO_progress_shaping.md §二修订):
挥霍/超量消耗"库存正覆盖需求的自造中间品"会让该物品重新入闭包、Φ 上升,如实计负,
负值语义 = 净破坏。守恒律 ΣΔΦ = Φ(s0) − Φ(s_T) 因此精确成立(冻结段内),
挥霍-重造与单轮 wash 两条路径同记 0,套利通道关闭。单调性只是条件性质
(沿不消耗正覆盖需求中间品的轨迹成立;gold 满足),初始库存全叶子断言保留——
它支撑的是"上升只能源于消耗自造中间品"的分类学,不再支撑单调性。

Φ 是特权量:只在训练侧算优势用,agent 从不观测它,不构成信息泄漏。
"""
from __future__ import annotations

import collections
import math
from typing import Dict, Optional, Set, Tuple


def needed_closure(db, targets: Dict[str, int], inventory: Dict[str, int]):
    """返回"必须生产的物品集合";状态不可解时返回 None。

    逐行移植自 代码理解/probe_costtogo.py(632/632 已验证)。做法:按需求自顶向下展开,
    用待处理队列累积需求;每物品只有一条配方且图无环,展开顺序不影响结果集合。
    产量按 result_count 向上取整,多做出来的存回库存供别处使用。
    """
    inv = dict(inventory)
    demand = collections.Counter(targets)
    produce = collections.Counter()
    queue = list(targets)
    seen_guard = 0
    while queue:
        seen_guard += 1
        if seen_guard > 200000:
            raise RuntimeError("closure expansion exceeded guard; recipe graph may have a cycle")
        item = queue.pop()
        need = demand[item]
        if need <= 0:
            continue
        have = inv.get(item, 0)
        use = min(have, need)
        inv[item] = have - use
        deficit = need - use
        demand[item] = 0
        if deficit <= 0:
            continue
        rs = db.get_recipes_for_item(item)
        if not rs:                      # 叶子却还缺 → 不可解
            return None
        r = rs[0]
        n_batches = math.ceil(deficit / r.result_count)
        made = n_batches * r.result_count
        produce[item] += made
        inv[item] = inv.get(item, 0) + (made - deficit)
        for ing, per in r.ingredients.items():
            demand[ing] += per * n_batches
            queue.append(ing)
    return set(produce)


def compute_phi(db, targets: Dict[str, int], inventory: Dict[str, int],
                known_recipes: Set[str], gamma: float = 1.0) -> Optional[float]:
    """Φ = C + γ·|M\\K|。不可解返回 None(调用方负责冻结口径)。

    M 与"必须生产的物品集合"一一对应(每物品恰一条配方),所以 |M\\K| 就是
    "闭包里还没被 get_info 查到过的物品数"。
    """
    closure = needed_closure(db, targets, inventory)
    if closure is None:
        return None
    c = len(closure)
    unknown = len(closure - known_recipes)
    return float(c) + gamma * float(unknown)


def assert_initial_inventory_all_leaves(db, inventory: Dict[str, int]) -> None:
    """启动断言(RSO_progress_shaping.md §二 ⚠):Φ 单调不增的证明依赖
    "初始库存只含叶子物品"这条【数据性质】。632 道题、5510 个条目实测全部满足;
    数据集换代时这里立即暴露,而不是让 G≥0 悄悄失效。"""
    craftable = [k for k in inventory if db.get_recipes_for_item(k)]
    if craftable:
        raise ValueError(
            f"[RSO/phi] 初始库存里出现可 craft 物品 {craftable[:5]}(共 {len(craftable)} 个)。"
            "Φ 的单调性证明依赖初始库存全为叶子,该数据性质被破坏,进展奖励不再保证非负。"
            "请核查数据集或修订 RSO_progress_shaping.md §二 的口径。")


class PhiTracker:
    """一个槽(=一棵树)的 Φ 状态机。适配器每轮喂它环境快照,它吐出 ΔΦ。

    生命周期:reset(每局开始)→ update(每个真实环境轮)→ 读 delta / phi / frozen。
    委派占位轮不经过 update(环境收到的是 filler),ΔΦ 语义上就是 0,由调用方缺省处理。
    """

    __slots__ = ("db", "gamma", "targets", "known", "phi", "frozen",
                 "phi_initial", "negative_delta_rounds")

    def __init__(self, db, gamma: float = 1.0):
        self.db = db
        self.gamma = float(gamma)
        self.targets: Dict[str, int] = {}
        self.known: Set[str] = set()
        self.phi: float = 0.0
        self.frozen: bool = False
        self.phi_initial: float = 0.0
        self.negative_delta_rounds: int = 0   # 观测量:发生净破坏(ΔΦ<0)的轮数

    def reset(self, targets: Dict[str, int], inventory: Dict[str, int]) -> float:
        assert_initial_inventory_all_leaves(self.db, inventory)
        self.targets = dict(targets)
        self.known = set()
        self.frozen = False
        self.negative_delta_rounds = 0
        phi = compute_phi(self.db, self.targets, inventory, self.known, self.gamma)
        if phi is None:
            # 数据集 632/632 可解,初始态不可解只能是数据坏了
            raise ValueError("[RSO/phi] 初始状态即不可解,任务数据异常")
        self.phi = phi
        self.phi_initial = phi
        return phi

    def note_get_info(self, last_get_info) -> None:
        """把这一轮 get_info 揭示的配方并进全树 K。只收"可 craft"的条目——
        叶子物品没有配方,永远不会出现在 M 里,收不收无影响,不收保持口径干净。"""
        if not last_get_info:
            return
        for entry in last_get_info:
            try:
                if entry.get("recipes"):
                    self.known.add(str(entry.get("item")))
            except AttributeError:
                continue

    def update(self, inventory: Dict[str, int], last_get_info) -> Tuple[float, float, bool]:
        """一个真实环境轮之后调用。返回 (delta_phi, phi_after, frozen)。

        判序(边界口径,2026-09-07 拍板):
          ① 先查可解性——不可解 ⇒ 冻结,记 0(先冻结后计数,毁到不可解的那轮不记负);
          ② 可解 ⇒ delta = Φ_前 − Φ_后 如实返回,允许为负(净破坏,甲案)。
        负值只计数(negative_delta_rounds,进 rso/G_negative_ratio 打点,
        基线 0.13% 轮级;该曲线上升 = 策略在学破坏,退化预警),不告警不钳位。
        """
        if self.frozen:
            return 0.0, self.phi, True
        self.note_get_info(last_get_info)
        new_phi = compute_phi(self.db, self.targets, inventory, self.known, self.gamma)
        if new_phi is None:
            self.frozen = True
            return 0.0, self.phi, True
        delta = self.phi - new_phi
        if delta < -1e-9:
            self.negative_delta_rounds += 1
        self.phi = new_phi
        return float(delta), self.phi, False
